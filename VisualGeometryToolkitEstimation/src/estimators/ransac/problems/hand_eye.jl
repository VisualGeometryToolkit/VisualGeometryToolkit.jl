# =============================================================================
# RANSAC Hand-Eye / Robot-World Calibration Problem  (AX = XB)
# =============================================================================
#
# Given synchronized pose pairs A_i = T_WR (rig in world, mocap) and
# B_i = T_CP (pattern in camera, PnP), recover the constant rigid transform
# X = T_RP (pattern in rig).
#
# The classical robot-world form A_i X = Y B_i contains a nuisance pose
# Y = T_WC. Taking ratios of two observations eliminates Y:
#
#     (A_j^{-1} A_i) X = X (B_j^{-1} B_i)        ⇔   A_ij X = X B_ij
#
# This is the AX = XB hand-eye problem (Tsai-Lenz 1989). It plugs into
# the standard RVG RANSAC framework:
#
#   - Minimal sample: 2 motion pairs
#   - Minimal solver: Tsai-Lenz (closed-form modified-Rodrigues LLS)
#   - Residual: 6-vector r_ij = [L · Log(R_err); t_err] under a
#     characteristic length L that balances rotation (rad) and
#     translation (length) units; Mahalanobis distance via the
#     default residuals! from constraint_with_jacobians.
#   - Local optimization (`refine!`): the chart LM on VGC's
#     RigidExpManifold6 (ForwardDiff Jacobian); default strategy Irls{TLSShape}.
# =============================================================================

# Hand-eye currency is generic rigid motion (relative motions A_ij, B_ij and
# the calibration X = T_RP) — NOT camera extrinsics, so the labeled
# `Extrinsics3` (WorldFrame→CameraFrame) would be semantically wrong and its
# frame checks would reject the `inv(A[j]) ∘ A[i]` motion algebra. Local
# UNLABELED concrete alias (same layout as the retired VGC Pose3 alias);
# honest labels await VGC 0.8 view-identity frames.
const Rigid3 = EuclideanMap{
    3, Float64, RotMatrix{3, Float64, 9}, Vec3{Float64},
    UnlabeledFrame{3}, UnlabeledFrame{3},
}

# =============================================================================
# HandEyeProblem
# =============================================================================

"""
    HandEyeProblem{S} <: AbstractRansacProblem{2}

RANSAC problem for rigid hand-eye / robot-world calibration. Solves
`A_ij X = X B_ij` for the constant `X = T_RP` ∈ SE(3) from a stream of
relative motion pairs. Minimal sample is 2 motion pairs.

Type parameters:
- `S <: AbstractSampler`: sampling strategy

# Constructor
```julia
# From synchronized absolute-pose observations, one A_i => B_i per frame
cs = A .=> B        # A_i = T_WR (mocap), B_i = T_CP (PnP), in time order
problem = HandEyeProblem(cs)
```

# Fields
- `motions::Vector{Tuple{Rigid3, Rigid3}}`: (A_ij, B_ij) relative-motion pairs
- `axes::Vector{SVector{3, Float64}}`: the rotation vector `log_so3(R_A)` of
  each rig motion, cached for `test_sample`'s parallel-axis test
- `char_length::Float64`: characteristic length used to balance the rotation
  and translation residual blocks (set automatically from data, override via
  the constructor)
- `min_rel_angle::Float64`: the smallest rotation (rad) every motion has —
  checked once, here
- `_sampler::S`: minimal-sample sampler

The inner constructor `HandEyeProblem{S}(motions, char_length, min_rel_angle,
sampler)` is THE rotation gate: it throws an `ArgumentError` when a motion's
rig or pattern rotation is below `min_rel_angle` (throws rather than drops, so
the sampler built for `motions` stays consistent). The keyword constructor
filters its relative motions by the same predicate first, so it never throws
there; a problem built from raw motions cannot bypass the gate. A near-identity
motion's axis is noise-dominated, and its hypotheses scatter far outside the
first-order covariance they are scored with: the predictive NEES tail (228/6 at
1 mrad, median ratio still 1.03) is exactly these motions, and admitting only
motions of ≥ 5° calibrates it (Monte Carlo, `ransac_mc_calibration` (b)).
"""
struct HandEyeProblem{S <: AbstractSampler} <: AbstractRansacProblem{2}
    motions::Vector{Tuple{Rigid3, Rigid3}}
    axes::Vector{SVector{3, Float64}}
    char_length::Float64
    min_rel_angle::Float64
    _sampler::S
    function HandEyeProblem{S}(
            motions::AbstractVector{Tuple{Rigid3, Rigid3}}, char_length::Real,
            min_rel_angle::Real, smplr::S
        ) where {S <: AbstractSampler}
        axes = Vector{SVector{3, Float64}}(undef, length(motions))
        for (k, (A_ij, B_ij)) in enumerate(motions)
            ωA = log_so3(rotation(A_ij))
            (_rotates_enough(ωA, min_rel_angle) && _rotates_enough(log_so3(rotation(B_ij)), min_rel_angle)) ||
                throw(
                ArgumentError(
                    "relative motion $k rotates less than min_rel_angle = " *
                        "$(round(rad2deg(min_rel_angle); digits = 2))°"
                )
            )
            axes[k] = ωA
        end
        return new{S}(Vector{Tuple{Rigid3, Rigid3}}(motions), axes, Float64(char_length), Float64(min_rel_angle), smplr)
    end
end

"""
    HandEyeProblem(cs; motion_pairs, char_length=nothing, min_rel_angle)

Build a `HandEyeProblem` from synchronized absolute-pose observations.

# Arguments
- `cs` — one correspondence between rigid transforms per frame, in TIME order
  (`motion_pairs = :consecutive` pairs neighbours): `A_i => B_i`, the
  rig-in-world pose (e.g. mocap) and the pattern-in-camera pose (e.g. PnP;
  labeled `Extrinsics3` accepted — unlabeled at the boundary). Any
  `sources`/`targets` container of `Cspond{<:PlainEuclideanMap{3}, <:PlainEuclideanMap{3}}`:
  plain translations, in one length unit — a problem is the inner loop, and a
  physical (unitful) pose is a `MethodError` here; [`find_hand_eye`](@ref)
  takes physical poses and strips them in metres once.

# Keywords
- `motion_pairs::Symbol` — `:consecutive` uses pairs `(i, i+1)` yielding
  `N-1` motion pairs; `:all` uses every `(i, j)` with `i < j`. Defaults to
  `configs/default.toml` `[pipelines.hand_eye]` (`"consecutive"`), the same
  authority `find_hand_eye` reads.
- `char_length` — characteristic length to balance rotation (rad) and
  translation residuals. Default: median translation magnitude of the
  relative motions A_ij.
- `min_rel_angle` — the smallest rotation (rad) a relative motion may have:
  motions below it are dropped here, before the inner constructor's gate.
  Defaults to `configs/default.toml` `[robot_world_hand_eye]`
  `min_rel_angle_deg`, the SAME authority `fit_robot_world_hand_eye`'s init reads:
  the rotation below which a relative motion is degenerate for recovering X.

Throws when there are fewer than 3 frames, or when fewer than two relative
motions rotate at least `min_rel_angle` (no minimal sample could satisfy the gate —
refused here rather than as a `:no_model` after every RANSAC trial).
"""
function HandEyeProblem(
        cs::AbstractVector{<:Cspond{<:PlainEuclideanMap{3}, <:PlainEuclideanMap{3}}};
        motion_pairs::Symbol = Symbol(_dflt("pipelines.hand_eye", "motion_pairs")),
        char_length::Union{Nothing, Real} = nothing,
        min_rel_angle::Real = deg2rad(_dflt("robot_world_hand_eye", "min_rel_angle_deg"))
    )
    A = sources(cs); B = targets(cs)
    length(A) >= 3 ||
        throw(ArgumentError("need at least 3 pose observations (got $(length(A)))"))

    motions = _form_relative_motions(A, B, motion_pairs, Float64(min_rel_angle))
    L = isnothing(char_length) ? _default_char_length(motions) : Float64(char_length)
    n = length(motions)
    smplr = _build_sampler(n, 2, nothing)
    return HandEyeProblem{typeof(smplr)}(motions, L, Float64(min_rel_angle), smplr)
end

# =============================================================================
# AbstractRansacProblem Interface
# =============================================================================

codimension(::Type{<:HandEyeProblem}) = 6                          # d_g = 6 (3 rot + 3 trans)
data_size(p::HandEyeProblem) = length(p.motions)
model_type(::Type{<:HandEyeProblem}) = Rigid3
sampler(p::HandEyeProblem) = p._sampler

"""
    solve(p::HandEyeProblem, idx) -> FixedModels{1, Rigid3}

Closed-form Tsai-Lenz solver on the two minimal-sample relative motions.
Returns `FixedModels{1, Rigid3}` with `count = 0` (no valid solution) or
`count = 1`.
"""
function solve(p::HandEyeProblem, idx::AbstractVector{Int})
    @inbounds m1 = p.motions[idx[1]]
    @inbounds m2 = p.motions[idx[2]]
    X = _tsai_lenz_minimal(m1, m2)
    isnothing(X) && return FixedModels{1, Rigid3}(0, (zero(Rigid3),))
    return FixedModels{1, Rigid3}(1, (X,))
end

"""
    test_sample(p, idx) -> Bool

Reject the minimal sample when its two rig motions have nearly parallel
rotation axes, which ill-conditions X. (Every motion already rotates at least
`p.min_rel_angle` — the constructor's gate.)
"""
function test_sample(p::HandEyeProblem, idx::AbstractVector{Int})
    @inbounds ωA = p.axes[idx[1]]
    @inbounds ωB = p.axes[idx[2]]
    c = abs(dot(ωA, ωB) / (norm(ωA) * norm(ωB)))
    return c < 0.999                                                # ≳ 2.5° between axes
end

test_model(::HandEyeProblem, X::Rigid3, ::AbstractVector{Int}) =
    all(isfinite, rotation(X)) && all(isfinite, translation(X))

# =============================================================================
# constraint_with_jacobians — single source of truth for residual + Jacobians
# =============================================================================

# The hand-eye constraint g of one motion pair at X — the ONE definition, shared
# by `constraint_with_jacobians` and the LO rows (`residual_rows!`): the
# coordinates of A∘X in VGC's decoupled chart at X∘B (`local_coordinates`:
# Log(R_A R_X R_Bᵀ R_Xᵀ) and R_A t_X + t_A − R_X t_B − t_X), the rotation block
# scaled by L. Eltype-generic, so ForwardDiff Duals flow through X.
@inline function _hand_eye_constraint(L, X::PlainEuclideanMap{3}, A_ij, B_ij)
    δ = local_coordinates(RigidExpManifold6(X ∘ B_ij), A_ij ∘ X)
    return SVector(L * δ[1], L * δ[2], L * δ[3], δ[4], δ[5], δ[6])
end

"""
    constraint_with_jacobians(p::HandEyeProblem, X::Rigid3, i::Int)
        -> (g::SVector{6}, ∂ₓg::SMatrix{6,6}, ∂_θg::SMatrix{6,6})

Decoupled SE(3) hand-eye constraint and its derivatives.

For motion pair `(A_ij, B_ij) = p.motions[i]`:
```
g = [ L · Log(R_A R_X R_B^T R_X^T)
      R_A t_X + t_A − R_X t_B − t_X ]   ∈ ℝ⁶
```
(L = `p.char_length`, balancing rotation rad and translation length units.)

The model Jacobian `∂_θg` uses the decoupled left-perturbation
parameterization `(δω, δt) ∈ ℝ⁶` matching `AbsolutePoseProblem`:
  R_X ← R(δω) · R_X,    t_X ← t_X + δt.
Under that convention, near the optimum:
```
∂_θg = [  L · (R_A − I)       0       ]   ← rotation block
       [   [R_X t_B]_×    (R_A − I)   ]   ← translation block
```

The measurement Jacobian `∂ₓg = I_6` reports the noise model directly on
the constraint (the per-pair 6-vector is already in unit-balanced form),
so the default `residuals!` produces ‖g‖.
"""
@inline function constraint_with_jacobians(p::HandEyeProblem, X::Rigid3, i::Int)
    L = p.char_length
    R_X = rotation(X)
    A_ij, B_ij = p.motions[i]
    R_A = rotation(A_ij)

    g = _hand_eye_constraint(L, X, A_ij, B_ij)

    I3 = SMatrix{3, 3, Float64}(LinearAlgebra.I)
    RA_minus_I = SMatrix{3, 3, Float64}(R_A) - I3
    skew_RXtB = skew(R_X * translation(B_ij))                                 # VGC's skew
    Z3 = zero(SMatrix{3, 3, Float64})

    Gθ = vcat(
        hcat(L * RA_minus_I, Z3),
        hcat(skew_RXtB, RA_minus_I)
    )

    Gx = SMatrix{6, 6, Float64}(LinearAlgebra.I)
    return (g, Gx, Gθ)
end
provides_solver_covariance(::Type{<:HandEyeProblem}) = true   # predictive scoring (covariance.jl)

# =============================================================================
# The chart — VGC's RigidExpManifold6 (exp(δω)·R ⊕ t + δt; 6 dof, codim 6)
# =============================================================================
#
# The chart of `constraint_with_jacobians`' `Gθ` and so of the solver
# covariance: the chart-LM refit, the hypothesis fit and the covariance share
# it. Unlike P3P, that `Gθ` is only the NEAR-OPTIMUM (GN) approximation of
# ∂g/∂δ, so it cannot serve as an exact Jacobian: the rows are differentiated
# with ForwardDiff (exact; the log is autodiff-safe away from R_err = I, where
# the LM iterates). No lift: the lift core supports codimension 1 and 2.
# =============================================================================

chart_dof(::Type{<:HandEyeProblem}) = 6
_chart(::HandEyeProblem, X) = RigidExpManifold6(rotation(X), translation(X))
_model_at(::HandEyeProblem, ch) = convert(Rigid3, model(ch))

# √w-weighted, unit-balanced hand-eye rows in the chart at `δ` (VGC's
# `residual_rows!`), 6 per motion with the motion's √w on each; eltype-generic
# so ForwardDiff Duals flow through.
function residual_rows!(r::AbstractVector, lm::ChartLm{<:RigidExpManifold6, <:HandEyeProblem}, δ)
    problem = lm.problem; sqrtw = lm.sqrtw
    X = retract(lm.ch, δ)
    L = problem.char_length
    @inbounds for i in 1:data_size(problem)
        A_ij, B_ij = problem.motions[i]
        g = _hand_eye_constraint(L, X, A_ij, B_ij)
        b = 6 * (i - 1)
        for j in 1:6
            r[b + j] = sqrtw[b + j] * g[j]
        end
    end
    return r
end

# The Jacobian is VGC's default: ForwardDiff through these rows (see the
# section header).

# Two motions are 12 constraint rows over 6 dof: already over-determined (the
# generic rule, minimal sample + 1 points, is for exactly-determined samples).
_underdetermined(::HandEyeProblem, weights::AbstractVector) = count(>(zero(eltype(weights))), weights) < 2

# The hypothesis of a two-motion sample: Tsai–Lenz (`solve`) is a decoupled
# algebraic solve, not the least-squares estimator the solver covariance (the
# sandwich over the sample's 12 rows) describes — its hypotheses scattered
# ~2.4× wider than that covariance (Monte Carlo, median). The sample's own
# chart-LM fit, seeded by Tsai–Lenz, is that estimator.
function fit_hypothesis(p::HandEyeProblem, X::Rigid3, idx::AbstractVector{Int})
    sub = HandEyeProblem{typeof(p._sampler)}(p.motions[idx], p.char_length, p.min_rel_angle, p._sampler)   # validated subset
    return _refine!(NonlinLsq(), nothing, sub, X, ones(length(idx)))
end

# =============================================================================
# Closed-form Tsai-Lenz minimal solver
# =============================================================================

"""
    _tsai_lenz_minimal(m1, m2) -> Union{Rigid3, Nothing}

Tsai-Lenz hand-eye solver specialized for the minimal sample of 2 motion
pairs. Stacks the 6×3 rotation system

    skew(αᴬ + αᴮ) · p′  =  αᴮ − αᴬ

where α = 2 sin(θ/2) · n̂ is the scaled-axis Rodrigues vector. Recovers
R_X from p′, then solves the 6×3 translation system

    (R_A − I) · t_X  =  R_X · t_B − t_A    (per motion)

via least squares. Returns `nothing` when the rotation axes are parallel
(rank-deficient system).
"""
function _tsai_lenz_minimal(m1::Tuple{Rigid3, Rigid3}, m2::Tuple{Rigid3, Rigid3})
    αA1 = _scaled_axis(rotation(m1[1])); αB1 = _scaled_axis(rotation(m1[2]))
    αA2 = _scaled_axis(rotation(m2[1])); αB2 = _scaled_axis(rotation(m2[2]))

    # 6×3 rotation system, stacked from two pairs.
    S = vcat(skew(αA1 + αB1), skew(αA2 + αB2))
    d = vcat(αB1 - αA1, αB2 - αA2)
    p_half = try
        S \ d
    catch
        return nothing
    end
    # p_half ≡ tan(θ_X/2) · n̂_X: the Rodrigues (Gibbs) vector of R_X.
    all(isfinite, p_half) || return nothing
    R_X = RotMatrix{3, Float64}(Rotations.RodriguesParam(p_half[1], p_half[2], p_half[3]))

    I3 = SMatrix{3, 3, Float64}(LinearAlgebra.I)
    C = vcat(
        SMatrix{3, 3, Float64}(rotation(m1[1])) - I3,
        SMatrix{3, 3, Float64}(rotation(m2[1])) - I3
    )
    e = vcat(
        R_X * translation(m1[2]) - translation(m1[1]),
        R_X * translation(m2[2]) - translation(m2[1])
    )
    t_X = try
        C \ e
    catch
        return nothing
    end
    all(isfinite, t_X) || return nothing

    return Rigid3(R_X, SVector{3, Float64}(t_X))
end

# =============================================================================
# Helpers
# =============================================================================

"""
    _form_relative_motions(A, B, mode, min_angle) -> Vector{Tuple{Rigid3, Rigid3}}

Form `(A_ij, B_ij) = (A_j^{-1} A_i, B_j^{-1} B_i)` from consecutive or
all-pair frames. Discards pairs whose rig or pattern motion rotates less than
`min_angle` (rad) — the predicate the inner constructor's gate applies, so
every motion kept succeeds at it — and throws when fewer than two remain.
"""
function _form_relative_motions(
        A::AbstractVector{<:PlainEuclideanMap{3}},
        B::AbstractVector{<:PlainEuclideanMap{3}},
        mode::Symbol,
        min_angle::Float64
    )
    N = length(A)
    motions = Tuple{Rigid3, Rigid3}[]

    pairs = mode == :consecutive ? ((i, i + 1) for i in 1:(N - 1)) :
        mode == :all ? ((i, j) for i in 1:(N - 1) for j in (i + 1):N) :
        throw(ArgumentError("motion_pairs must be :consecutive or :all, got $mode"))

    for (i, j) in pairs
        # `unlabel` at the boundary: p3p/PnP hand in labeled `Extrinsics3`
        # (world→camera), and `inv(A[j]) ∘ A[i]` over labeled maps yields a
        # {WorldFrame, WorldFrame} map that will not convert into the
        # unlabeled `Rigid3` motion tuples. Hand-eye algebra is frame-free.
        A_ij = inv(unlabel(A[j])) ∘ unlabel(A[i])
        B_ij = inv(unlabel(B[j])) ∘ unlabel(B[i])
        if _rotates_enough(log_so3(rotation(A_ij)), min_angle) &&
                _rotates_enough(log_so3(rotation(B_ij)), min_angle)
            push!(motions, (A_ij, B_ij))
        end
    end

    length(motions) >= 2 ||
        throw(
        ArgumentError(
            "only $(length(motions)) relative motion pair(s) rotate at least " *
                "min_rel_angle = $(round(rad2deg(min_angle); digits = 2))°; need ≥ 2 " *
                "(lower `min_rel_angle` or supply more rotation between frames)"
        )
    )
    return motions
end

"""
    _default_char_length(motions) -> Float64

Pick `L` = median of ‖t_A_ij‖ (the typical rig translation per motion) so
that the rotation block of `g` (in radians) and translation block (in
length units) are comparable. Falls back to 1.0 when all translations
are zero (e.g., pure-rotation calibration).
"""
function _default_char_length(motions)
    tnorms = [norm(translation(A_ij)) for (A_ij, _) in motions]
    med = isempty(tnorms) ? 1.0 : Statistics.median(tnorms)
    return med > 1.0e-9 ? med : 1.0
end

# THE minimum-rotation predicate on a rotation vector ω = log_so3(R) (the angle
# is ‖ω‖): a relative motion is admitted when it rotates at least `min_angle` —
# and not at all when it does not rotate (no axis), even at `min_angle = 0`.
# Shared by the construction filter (`_form_relative_motions`), the inner
# constructor's gate, and `fit_robot_world_hand_eye`'s init.
@inline _rotates_enough(ω::StaticVector{3}, min_angle) = (θ = norm(ω); θ >= min_angle && θ > 0)

# Tsai's scaled-axis vector α = 2 sin(θ/2)·n̂ of a rotation — twice the
# vector part of its unit quaternion (Rotations.jl), on the w ≥ 0 hemisphere
# (θ ∈ [0, π]).
@inline function _scaled_axis(R)
    w, x, y, z = Rotations.params(Rotations.QuatRotation(R))
    s = w < 0 ? -2.0 : 2.0
    return SVector{3, Float64}(s * x, s * y, s * z)
end
