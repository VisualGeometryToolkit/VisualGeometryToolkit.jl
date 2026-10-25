# =============================================================================
# division_camera_step — one iteration of the stacked linear focal solve, its state
# and its failure (moved from VisualGeometryToolkitCore, estimator taxonomy v2: a
# `*_step` is one iteration of an iterative fit and lives with it — beside
# `fit_division_camera`, the IRLS that iterates it)
# =============================================================================

# The (up to four) partial poses (R, t₁₂) consistent with a radial camera.
# Frontal boards (A = B = 0) give r₁₃ = r₂₃ = 0.
function _radial_pose_candidates(P::RadialCamera)
    M = radial_rows(P)
    r11, r12, t1 = M[1, 1], M[1, 2], M[1, 3]
    r21, r22, t2 = M[2, 1], M[2, 2], M[2, 3]
    A = r11 * r21 + r12 * r22
    B = r11^2 - r21^2 + r12^2 - r22^2
    r23sq = (B + sqrt(B^2 + 4A^2)) / 2
    r23 = sqrt(max(r23sq, 0.0))
    pairs = if r23 > 1.0e-12 * max(abs(r11), abs(r12), abs(r21), abs(r22), 1.0e-300)
        ((-A / r23, r23), (A / r23, -r23))
    else                                   # r₂₃ ≈ 0: r₁₃² = −B, sign free
        r13 = sqrt(max(-B, 0.0))
        ((r13, 0.0), (-r13, 0.0))
    end
    out = Tuple{SMatrix{3, 3, Float64, 9}, SVector{2, Float64}}[]
    for (r13, r23) in pairs, sgn in (1.0, -1.0)
        s = sgn / norm(SVector(r11, r12, r13))
        row1 = SVector(r11, r12, r13) * s
        row2 = SVector(r21, r22, r23) * s
        row3 = cross(row1, row2)
        R = SMatrix{3, 3, Float64}(
            row1[1], row2[1], row3[1],
            row1[2], row2[2], row3[2],
            row1[3], row2[3], row3[3]
        )
        push!(out, (R, SVector(t1, t2) * s))
    end
    return out
end

# Camera-frame (X, Y, Z) of plane point p under the partial pose (t₃ = 0): the
# rigid map (R, [t₁₂; 0]) applied to [p; 0]. Dual-safe (the element types promote).
@inline function _partial_cam(R, t12, p)
    z = zero(promote_type(eltype(t12), eltype(p)))
    return EuclideanMap(R, SVector(t12[1], t12[2], z); check = false)(SVector(p[1], p[2], z))
end

# ------------------------------------------------------------------------------
# RadialLift — the division-lift constraint of one board, with its derivatives
# ------------------------------------------------------------------------------

"""
    RadialLift{N,T}

The division-lift constraint model of ONE board: the partial pose `(R, t12)`
from its radial camera and the linear unknowns θ = (f, λ₁..λ_N, t₃) — `f`,
the lift coefficients `λₚ = kₚ·f` of r²ᵖ, and the board's depth. For a plane
point `X` and its RAW centered pixel `x` the constraint is

    g = x·(Z + t₃) − X_c·(f + Σₚ λₚ ρ²ᵖ)   (two rows; X_c = camera-frame point at t₃ = 0)

which is LINEAR in θ: `g = G_θ·θ + c₀`, `c₀ = x·Z`. [`constraint_with_jacobians`](@ref)
returns `g`, the measurement Jacobian `G_x = ∂g/∂x` (2×2) and `G_θ` (2×(N+2)),
and `whiten` turns them into pixel-currency rows. A KNOWN k folds `λ₁ = k·f`
into the f column (chain rule) — see `_lift_columns`.
"""
struct RadialLift{N, T <: Real}
    R::SMatrix{3, 3, T, 9}
    t12::SVector{2, T}
    f::T
    λ::SVector{N, T}
    t3::T
end

@inline function VgtkCore.constraint_with_jacobians(m::RadialLift{N}, X::StaticVector{2}, x::StaticVector{2}) where {N}
    # Element types promote (Dual-safe: ForwardDiff differentiates these bodies).
    Xc = _partial_cam(m.R, m.t12, X)
    x1, x2 = x[1], x[2]
    ρ² = x1 * x1 + x2 * x2
    pows = ntuple(p -> ρ²^p, Val(N))                                   # ρ², ρ⁴, …
    Λ = _lift_factor(m.f, m.λ, ρ², pows)                               # f + Σ λₚ ρ²ᵖ
    dΛ = sum(p -> p * m.λ[p] * (p == 1 ? one(ρ²) : pows[p - 1]), 1:N; init = zero(ρ²))   # dΛ/dρ²
    z = Xc[3] + m.t3
    g = SVector(x1 * z - Xc[1] * Λ, x2 * z - Xc[2] * Λ)
    G_x = SMatrix{2, 2}(
        z - 2 * Xc[1] * dΛ * x1, -2 * Xc[2] * dΛ * x1,     # ∂(g₁, g₂)/∂x
        -2 * Xc[1] * dΛ * x2, z - 2 * Xc[2] * dΛ * x2      # ∂(g₁, g₂)/∂y
    )
    # θ = (f, λ₁..λ_N, t₃): ∂g/∂f = −X_c, ∂g/∂λₚ = −X_c·ρ²ᵖ, ∂g/∂t₃ = x
    cols = ntuple(N + 2) do j
        j == 1 && return SVector(-Xc[1], -Xc[2])
        j == N + 2 && return SVector(x1, x2)
        SVector(-Xc[1] * pows[j - 1], -Xc[2] * pows[j - 1])
    end
    G_θ = hcat(cols...)
    return g, G_x, G_θ
end

# The lift's factor Λ = f + Σₚ λₚ ρ²ᵖ at the squared centred radius ρ² — the one
# spelling of it (the constraint above, the board depth and the branch audit
# below), from the powers `pows = (ρ², ρ⁴, …)` when the caller holds them.
# Dual-safe.
@inline _lift_factor(f, λ, ρ², pows = ntuple(p -> ρ²^p, Val(length(λ)))) =
    f + sum(p -> λ[p] * pows[p], eachindex(λ); init = zero(ρ²))

# Local row block for the stack: KNOWN k folds λ₁ = k·f into the f column
# (∂g/∂f_total = ∂g/∂f + k·∂g/∂λ₁), leaving (f, t₃); estimated k keeps all.
@inline _lift_columns(G_θ::SMatrix{2, 3, T}, k::T) where {T <: AbstractFloat} =
    SMatrix{2, 2, T}(G_θ[:, 1] .+ k .* G_θ[:, 2]..., G_θ[:, 3]...)
@inline _lift_columns(G_θ::SMatrix{2, M, T}, ::Int) where {M, T} = G_θ

# The coefficient MODE of a stack, dispatched: an `Int` is the number of
# division coefficients ESTIMATED (λ₁..λ_N shared columns); a `Float64` is the
# one KNOWN coefficient k (folded into the f column).
_nshared(k::Float64) = 1        # f
_nshared(n::Int) = 1 + n        # f, λ₁..λ_N
_ncoeffs(::Float64) = 1         # the lift model's λ count
_ncoeffs(n::Int) = n
# θ = the model's unknowns in the row block's column order (`_lift_columns`).
_lift_unknowns(m::RadialLift, ::Int) = vcat(SVector(m.f), m.λ, SVector(m.t3))
_lift_unknowns(m::RadialLift, ::Float64) = SVector(m.f, m.t3)
# The mode in the normalized frame (x' = σx): a known k enters as k/σ².
_normalized_kmode(n::Int, _σ) = n
_normalized_kmode(k::Float64, σ) = _rescaled_coeffs((k,), inv(σ))[1]
# The whitening model's λ at focal `fw` (normalized): a known k is λ₁ = k·f
# (the fold's manifold); an estimated set carries a previous iteration's
# coefficients, else the pinhole k₀ = 0.
_whitening_lambdas(kn::Float64, _whiten_at, fw, _σ) = (kn * fw,)
_whitening_lambdas(n::Int, _whiten_at, _fw, _σ) = ntuple(_ -> 0.0, n)
# The solved coefficients (normalized kₚ', pixel kₚ) from the shared unknowns `x`.
function _solved_coeffs(n::Int, _kn, x, σ)
    kn′ = [x[1 + p] / x[1] for p in 1:n]
    return kn′, _rescaled_coeffs(kn′, σ)
end
_solved_coeffs(k::Float64, kn, _x, _σ) = ([Float64(kn)], [k])

# Whitened (or plain) constraint rows of one correspondence at the model `m`:
# returns (row block, rhs). `g = G·θ + c₀` with c₀ = g − G·θ(m) — so the LS is
# G θ = −c₀; whitening multiplies both by L⁻¹. `m` carries the iteration's
# whitening point; iteration 1 supplies `nothing` and gets plain rows.
@inline function _lift_rows(m::RadialLift{N, T}, kmode, X, x, policy::WhiteningPolicy) where {N, T}
    g, G_x, G_θfull = constraint_with_jacobians(m, X, x)
    G_θ = _lift_columns(G_θfull, kmode)
    θ = _lift_unknowns(m, kmode)
    r, J = whitened_rows(policy, g, G_x, G_θ)
    return J, J * θ - r                          # rows and rhs: L⁻¹G θ = −L⁻¹c₀ = L⁻¹(Gθ − g)
end

# Build the lift model of one board for a given (f, λ, t₃) in the frame the stack works in.
@inline function _lift_model(R, t12, f, λs::Tuple, t3)
    T = float(promote_type(typeof(f), typeof(t3), map(typeof, λs)...))   # untyped tuple: `T` stays bound for N = 0
    N = length(λs)
    return RadialLift{N, T}(R, t12, T(f), SVector{N, T}(λs), T(t3))
end

# RADIAL CHEIRALITY — the test that actually decides the branch, and the whole
# reason it is a predicate of its own: it depends on (R, t12) ALONE, so it is
# settled before any least squares is paid for.
#
# At g = 0 the constraint reads x = X_c[1:2]·Λ/(Z + t₃), so a physically imaged
# point has its centered pixel on the SAME SIDE of the centre as its
# partial-pose point. Cheirality in z — `_focal_board_solve`'s `front` — is the
# gnomonic frame's domain, not the camera's, and is BLIND to the 180°
# optical-axis flip
#
#     R → diag(-1, -1, 1)·R,   (t₁, t₂) → (-t₁, -t₂)
#
# which leaves z, det R and the collinearity residual untouched while sending
# every point to the antipode: collinearity holds on a LINE, and a line contains
# both a point and its reflection. The flip is therefore an EXACT symmetry of
# that objective, so the `rms` cannot separate the branches either — measured on
# a fisheye set, the radial residual of a flipped pose is bit-identical while the
# pixel reprojection goes from 0.5 px to 2400 px. The sign of this dot product is
# the only cheap discriminator, and it needs no camera (X_c[1:2] does not involve
# t₃), so it works on iteration 0 where the branch is actually decided.
function _same_side_fraction(R, t12, pts)
    n = 0
    @inbounds for (X, x) in pts
        Xc = _partial_cam(R, t12, X)
        (Xc[1] * x[1] + Xc[2] * x[2] > 0) && (n += 1)
    end
    return n / length(pts)
end

# Per-board least squares for one pose candidate (plain rows); returns
# (rms residual, fraction of points in front of the camera).
function _focal_board_solve(R, t12, pts, kmode)
    n = length(pts)
    ns = _nshared(kmode)
    m0 = _lift_model(R, t12, 0.0, ntuple(_ -> 0.0, _ncoeffs(kmode)), 0.0)
    A = zeros(Float64, 2n, ns + 1)
    b = Vector{Float64}(undef, 2n)
    for (j, (pt, d)) in enumerate(pts)
        G, rhs = _lift_rows(m0, kmode, pt, d, NoWhitening())
        A[(2j - 1):2j, :] .= G; b[(2j - 1):2j] .= rhs
    end
    x = A \ b
    return sqrt(sum(abs2, A * x - b) / (2n)), _front_fraction(R, t12, pts, x[ns + 1])
end

# The fraction of a board's points in front of the camera at depth t₃.
_front_fraction(R, t12, pts, t3) = count(pd -> _partial_cam(R, t12, pd[1])[3] + t3 > 0, pts) / length(pts)

# The depth of one board at a camera (f, λ) — the t₃ solving
# x·(Z + t₃) = Λ·X_c[1:2], Λ = f + Σₚ λₚ ρ²ᵖ, in (weighted) least squares, in the
# stack's frame; points of weight zero (`w` a weight vector, a mask, or
# `nothing` for all) are left out. It places every board on the SEEDED iteration
# (the seed camera: λ = 0, or λ₁ = k·f for a known k), and the branch audit's
# trimmed depths. `nothing` when no weighted point is off the centre: the board
# then enters unwhitened (its rows carry no weight anyway).
function _whitening_depth(R, t12, pts, f, λs::Tuple, w)
    num = 0.0; den = 0.0
    @inbounds for (j, (X, x)) in enumerate(pts)
        wj = w === nothing ? 1.0 : Float64(w[j])
        wj > 0 || continue
        Xc = _partial_cam(R, t12, X)
        ρ² = x[1]^2 + x[2]^2
        Λ = _lift_factor(f, λs, ρ²)
        num += wj * (x[1] * (Λ * Xc[1] - x[1] * Xc[3]) + x[2] * (Λ * Xc[2] - x[2] * Xc[3]))
        den += wj * ρ²
    end
    return den > 0 ? num / den : nothing
end

# The candidate carrying a previous iteration's rotation `Rp`. The candidates
# are recomputed from the radial camera under THIS call's normalization, and
# the rotation is invariant to the normalization only up to rounding — so a
# previous state whitened under another one (a caller's, or one built from
# other weights) matches by tolerance, not bit for bit. An exact match dropped
# every view's whitening there, silently (measured: f 5 % off). The
# candidates of one camera differ by sign changes of whole entries, far above
# the tolerance; when two nearly coincide either is the same pose.
function _matching_candidate(cands, Rp)
    best = nothing; dmin = Inf
    for (c, (R, _)) in enumerate(cands)
        d = maximum(abs, R - Rp)
        d < dmin && (best = c; dmin = d)
    end
    return dmin <= 1.0e-6 ? best : nothing
end

# The views' `weights` columns' contract: each column VGC's (finite and
# non-negative, ArgumentError), and not all zero — the DATA condition every row
# weighted to zero, a `DivisionCameraError`. Checked where the columns are read
# (`_point_weights`), so before the normalization is built, which would
# otherwise fall back to the identity and be refused as not about the centre,
# blaming the caller's normalization for the weights.
function _check_row_weights(w)
    w === nothing && return nothing
    foreach(VgtkCore.check_weights, w)
    sum(sum, w; init = 0.0) >= eps(Float64) ||
        throw(DivisionCameraError("no board carries depth information (every row weighted to zero)"))
    return w
end

# THE BRANCH AUDIT. A view's branch — the tilt of its radial camera's pose — is
# chosen once, on the iteration that first places the view, from its own solve with
# f free (`_focal_board_solve`), and kept from then on. That solve cannot tell
# the tilt pair apart by residual — (f, λ, t₃) → (−f, −λ, −t₃) fits the partner
# exactly — so the choice rests on cheirality at its depth t₃, which a view's
# own f leaves weakly determined: one mismatch pushed along its ray through the
# centre, which the radial camera's line test cannot see, can carry t₃ across
# zero (measured 2026-09-22: 104 of 642 views carrying a mismatch at 3 %
# ray-aligned took a branch 48–85° off; 0 of 558 clean ones). The SHARED
# solution breaks that symmetry (f > 0 is held), so the audit, at convergence,
# re-judges every admissible candidate there: f and λ held, the view's depth by
# least squares trimmed twice (rows beyond 9 × the median residual² dropped), the
# candidate in front with the smallest TRIMMED MEAN residual² kept — a switch
# only when the alternative's is below `_AUDIT_MARGIN` × the current one's. On
# ALL the view's points, unweighted: a wrong branch can have driven the view's
# own inliers' posterior weights to zero (exactly, past the exponent's range),
# leaving only the mismatch that caused it weighted. The score and the margin,
# measured at convergence over 2182 correctly placed views (0/3/5 % ray-aligned):
# the best alternative never scored below 0.75 × the current (1.06 on clean
# stacks), while a wrong view's median ratio is 0.23–0.27 (q90 0.55–0.57). The
# plain median separates worse (a correct view down to 0.63); the
# posterior-weighted residual separates better but is blind exactly when the
# posterior has rejected the view's inliers.
const _AUDIT_MARGIN = 0.7
function _branch_residual(R, t12, pts, f, λs::Tuple)
    n = length(pts)
    use = trues(n)
    r² = Vector{Float64}(undef, n)
    t3 = 0.0
    for refit in 1:3                                  # least squares, then two trimmed refits
        d = _whitening_depth(R, t12, pts, f, λs, use)
        d === nothing && return Inf, 0.0
        t3 = d
        m = _lift_model(R, t12, f, λs, t3)
        @inbounds for (j, (X, x)) in enumerate(pts)
            r²[j] = sum(abs2, first(constraint_with_jacobians(m, X, x)))   # ‖g‖² of the lift
        end
        refit == 3 && break
        cut = 9 * Statistics.median(r²)
        use .= r² .<= cut
    end
    return Statistics.mean(view(r², use)), _front_fraction(R, t12, pts, t3)   # the TRIMMED mean residual²
end

function _audit_branch(cands, kept, pts, f, λs)
    best = kept
    score, front = _branch_residual(cands[kept][1], cands[kept][2], pts, f, λs)
    bestscore = front > 0.5 ? _AUDIT_MARGIN * score : Inf        # a switch must win by the margin
    for (c, (R, t12)) in enumerate(cands)
        c == kept && continue
        _same_side_fraction(R, t12, pts) > 0.5 || continue
        score, front = _branch_residual(R, t12, pts, f, λs)
        front > 0.5 && score < bestscore && (best = c; bestscore = score)
    end
    return best
end

"""
    DivisionCameraError(msg)

The stacked division-camera solve (`division_camera_step`) has no
system to solve for the boards it was given: no board admits a pose with its
points in front of the camera and imaged on the same side of the centre, no
board carries depth information (every row weighted to zero or at the centre),
the shared unknowns are not determined or not finite, or the focal length comes
out non-positive. A condition of the DATA, not of the arguments — the argument
contract (one correspondence set per camera, one shared centre, …) still
raises `ArgumentError` — so a caller that can go on without the solve catches
this type and nothing else.
"""
struct DivisionCameraError <: Exception
    msg::String
end
Base.showerror(io::IO, e::DivisionCameraError) = print(io, "DivisionCameraError: ", e.msg)

# A board's pose: the rigid map from the board plane (z = 0) into the camera —
# rotation from the radial camera's branch, translation (t₁₂, t₃).
const _BoardPose = EuclideanMap{3, Float64, RotMatrix{3, Float64, 9}, SVector{3, Float64}, UnlabeledFrame{3}, UnlabeledFrame{3}}

# One iteration's state in POSITIONAL form — the views in a fixed order (`ids`): the state the
# stacked solve and the IRLS iterate on, before it is keyed. `poses[i] === nothing`
# for a view that admitted no pose; `sq_residuals[i]` mirrors view i.
struct _PositionalDivisionState
    f::Float64
    k::Vector{Float64}
    center::Point2{Float64}
    poses::Vector{Union{Nothing, _BoardPose}}
    sq_residuals::Vector{Vector{Float64}}
end

"""
    DivisionCameraState{K}

Result of one iteration of the stacked linear focal solve, in DATA (pixel) units,
keyed by the views' identity `K`: `f`; the division coefficients `k` (of r², r⁴,
… in pixels centred on `centre`); the pencil's `centre`, the principal point
every input radial camera shares; `poses[id]`, the board pose — an
`EuclideanMap{3}` from the board plane (z = 0) into the camera, rotation from
the radial camera, translation (t₁₂, t₃) — in the plane's units, of each view that admitted one (a view
without a pose this iteration is absent); and `sq_residuals[id]`, the whitened
residual norms² of the view's correspondences (pixel², in its order; NaN for a
view without a pose) — the currency the robust re-weighting reads. `ids` is the
sorted key order. Feed it back as `whiten_at` for the next iteration;
`CameraModel(state)` is the camera it determines.
"""
struct DivisionCameraState{K}
    f::Float64
    k::Vector{Float64}
    center::Point2{Float64}
    poses::Dict{K, _BoardPose}
    sq_residuals::Dict{K, Vector{Float64}}
    ids::Vector{K}
end

# positional ⇄ keyed at the boundary
function _keyed(p::_PositionalDivisionState, ids::Vector{K}) where {K}
    poses = Dict{K, _BoardPose}(ids[i] => p.poses[i] for i in eachindex(ids) if p.poses[i] !== nothing)
    sq = Dict{K, Vector{Float64}}(ids[i] => p.sq_residuals[i] for i in eachindex(ids))
    return DivisionCameraState{K}(p.f, p.k, p.center, poses, sq, ids)
end
_positional(w::Union{Nothing, Real}, _ids) = w      # a seed focal, or none: no views to order
_positional(st::DivisionCameraState, ids) = _PositionalDivisionState(
    st.f, st.k, st.center,
    Union{Nothing, _BoardPose}[get(st.poses, id, nothing) for id in ids],
    [get(st.sq_residuals, id, Float64[]) for id in ids]
)

"""
    focal_length(state::DivisionCameraState) -> (f, f)

The iteration's focal length in pixels, as `(fx, fy)` like every `focal_length`
(the stack solves one: orthogonal raster, unit aspect ratio).
"""
VgtkCore.focal_length(state::DivisionCameraState) = (state.f, state.f)

"The iteration's division coefficients `k` (of r², r⁴, … in pixels about the centre)."
division_coeffs(state::DivisionCameraState) = state.k

"""
    board_sq_residuals(state::DivisionCameraState{K}) -> Dict{K, Vector{Float64}}

The whitened residual norms² of every correspondence, per view in its order
(pixel²) — the currency the robust re-weighting reads.
"""
board_sq_residuals(state::DivisionCameraState) = state.sq_residuals

# Shared driver: normalize through `normalization`, pick a pose candidate per board,
# stack the (whitened, weighted) constraint rows, solve. Returns the iteration's
# `DivisionCameraState` in DATA (pixel) units, carrying the stack's `center`.
#
# Normalization is essential, not cosmetic: in raw pixels the shared columns
# X, ρ²X, ρ⁴X span ~13 orders of magnitude and the two-coefficient stack is
# numerically destroyed (measured 2026-08-31: cond 1.3e12 → f < 0; cond 1.9
# after scaling). With σ the target-side scale, f' = f·σ, λₚ' = λₚ·σ^(1−2p),
# and a KNOWN k enters as k/σ²; the plane side (X' = S(X − X̄)) reaches the
# radial camera by composition (`D ∘ P ∘ inv(S)`, the normalization's two
# maps) and the poses by
# composing its origin shift (`pose ∘ Translation(X̄0)`, t scaled by S).
# The whitening point, dispatched on what `whiten_at` is: nothing (plain rows),
# a SEED focal (`Float64`: each board at the depth that focal implies), or a
# previous iteration (its focal, and each board's pose in its branch).
_whitening_focal(::Nothing) = 0.0
_whitening_focal(f::Float64) = f
_whitening_focal(w::_PositionalDivisionState) = w.f
_previous_pose(::Union{Nothing, Float64}, _i) = nothing
_previous_pose(w::_PositionalDivisionState, i) = w.poses[i]
_seed_depth(::Union{Nothing, _PositionalDivisionState}, _R, _t12, _pts, _fw, _λw, _w) = nothing
_seed_depth(::Float64, R, t12, pts, fw, λw, w) = _whitening_depth(R, t12, pts, fw, λw, w)
function _whitening_lambdas(n::Int, w::_PositionalDivisionState, fw, σ)
    kw = _rescaled_coeffs(w.k, inv(σ))
    return ntuple(p -> kw[p] * fw, n)
end

function _division_camera_solve(
        Hs, boards, k, normalization::CspondNormalization;
        # No defaults: the one caller (`_division_camera_step`) supplies every one, and
        # a default here would be a second authority for its keywords'.
        center::Point2{Float64}, point_weights, whiten_at::Union{Nothing, _PositionalDivisionState, Float64},
        whitening::WhiteningPolicy, audit::Bool
    )
    nb = length(Hs)
    nb == length(boards) || throw(ArgumentError("one correspondence set per homography"))
    point_weights === nothing ||
        (length(point_weights) == nb && all(length(point_weights[i]) == length(boards[i]) for i in 1:nb)) ||
        throw(ArgumentError("point_weights must mirror boards: one weight per correspondence"))
    σ, S = _check_division_normalization(normalization, center)
    X̄ = -source_map(normalization).translation / S                       # the plane side's centre
    # A pose between plane units and the normalized plane (X' = S(X − X̄)):
    # X_c = R·[X; 0] + t is X_c = (1/S)·(R·[X'; 0] + S·(R·[X̄; 0] + t)), the
    # overall 1/S a projective scale the camera absorbs — so the origin shift
    # [X̄; 0] COMPOSES onto the pose (`pose ∘ Translation(X̄0)`) and S scales
    # only its translation: t' = S·(t + R·[X̄; 0]), and back t = t'/S − R·[X̄; 0].
    X̄0 = SVector{3, Float64}(X̄[1], X̄[2], 0.0)
    # The radial cameras in the normalized frames: the frame change is plain
    # composition with the normalization's two maps, `D ∘ P ∘ inv(S)`.
    Sinv = inv(source_map(normalization)); Dm = target_map(normalization)
    kn = _normalized_kmode(k, σ)
    N = _ncoeffs(k)
    nboards = [normalize(normalization, pts) for pts in boards]
    # The whitening point in NORMALIZED units: the previous iteration's solution, or
    # the SEED (focal `whiten_at`, each board at the depth that focal implies —
    # weights only: f and k are still solved below). The audit judges the
    # branches there too.
    # Frame law (x' = σx, X' = SX): f' = σf, kₚ' = kₚ σ^(−2p), λₚ' = kₚ' f' = kₚ f σ^(1−2p), t' = S t.
    #
    # A KNOWN k is folded into the f column (`_lift_columns`), and the row's
    # right-hand side J·θ − r (`_lift_rows`) is exact only on that fold's
    # manifold, λ₁ = k·f: so every whitening model carries λ₁ = k·f — the seed
    # included, which is the pinhole k₀ = 0 only when k is estimated. (A seed
    # at λ = 0 under a known k biased iteration 0: f +4.3 % at the true focal,
    # and an extreme seed drove the solve to f ≤ 0.)
    fw = _whitening_focal(whiten_at) * σ
    λw = _whitening_lambdas(kn, whiten_at, fw, σ)
    # (board, rotation, t₁₂, whether it carries the previous iteration's pose)
    chosen = Tuple{Int, SMatrix{3, 3, Float64, 9}, SVector{2, Float64}, Bool}[]
    for i in 1:nb
        Pn = Dm ∘ _radial_camera(Hs[i]) ∘ Sinv                      # the radial camera, normalized
        cands = _radial_pose_candidates(Pn)
        # The branch is a property of the radial camera alone: the candidate
        # solves are plain, unweighted rows, so a board the previous iteration placed
        # keeps its branch — the candidate whose rotation it carries — and the
        # four solves are not repeated. Measured 2026-09-15 on 5 frames / 48 000
        # pairs: the re-selection was 45 % of a 30-iteration fit. A board the previous
        # iteration left without a pose (no admissible branch, or every row weighted
        # to zero) is judged afresh, as before.
        prev = _previous_pose(whiten_at, i)
        carried = prev === nothing ? nothing : _matching_candidate(cands, rotation(prev))
        if carried !== nothing
            kept = audit ? _audit_branch(cands, carried, nboards[i], fw, λw) : carried
            push!(chosen, (i, cands[kept][1], cands[kept][2], kept == carried))
            continue
        end
        best = nothing
        for (R, t12) in cands
            # The four candidates are TWO FLIP-PAIRS: `_radial_pose_candidates`
            # generates the tilt ambiguity (the sign of r₂₃) in its `pairs`
            # loop and the OPTICAL-AXIS FLIP in its `sgn` loop. `rms` resolves
            # the first — a genuine geometric difference — and provably cannot
            # resolve the second, which is a symmetry of this very residual.
            # `_same_side_fraction` kills one member of each pair exactly, and
            # it depends only on (R, t12), so it runs FIRST and halves the
            # per-board solves rather than discarding two of them afterwards.
            #
            # The focal length is deliberately NOT gated here. It used to be
            # (`f > 0`), as a proxy for the sign: the flip sends f → −f exactly,
            # so for clean data the pair does split on the sign of f. But `f`
            # comes from a per-board solve, and a small or distant target
            # cannot condition its own focal length (the single-view focal
            # degeneracy) — measured on a fisheye set, that solve returned a
            # NEGATIVE focal for the TRUE branch on 26 of 217 boards, so
            # `f > 0` actively selected the flip on exactly the boards it was
            # meant to protect. The same-side test checks the same sign without
            # estimating anything, and `f` is a SHARED unknown the stacked
            # solve below determines properly.
            _same_side_fraction(R, t12, nboards[i]) > 0.5 || continue
            rms, front = _focal_board_solve(R, t12, nboards[i], kn)
            front > 0.5 || continue
            if best === nothing || rms < best[1]
                best = (rms, R, t12)
            end
        end
        best === nothing || push!(chosen, (i, best[2], best[3], false))
    end
    isempty(chosen) && throw(
        DivisionCameraError(
            "no board admits a pose with its points in front of the camera and imaged on " *
                "the same side of the centre — check the centered-frame contract and k"
        )
    )
    ns = _nshared(kn)
    nchosen = length(chosen)
    # THE SOLVE IS ARROW-SHAPED, AND SOLVED AS SUCH. The unknowns are the
    # ns shared columns (f, λ₁..λ_N) plus ONE depth t₃ per board, and a board's
    # depth column is nonzero only on that board's own rows. So the normal
    # equations are
    #     [Sₙ Cₙ; Cₙᵀ D]·[xs; t] = [bₛ; bₜ],   D diagonal,
    # and the depths eliminate per board (Schur complement):
    #     (S − Σᵢ cᵢcᵢᵀ/dᵢ)·xs = bs − Σᵢ cᵢ·btᵢ/dᵢ,     tᵢ = (btᵢ − cᵢᵀxs)/dᵢ.
    # One sweep over the rows accumulates everything; the solve is ns × ns.
    # This replaced a dense `A \ b` over 2·nobs × (ns + boards) — a pivoted QR on
    # 148 000 × 1232 for a 41-frame session, 1.4 GB, ~20 s per iteration, ~30
    # iterations: the calibration's dominant cost (measured 2026-09-08, 16 BLAS
    # threads pinned for a quarter of an hour). The rows are Hartley-normalized
    # (σ, S below), so squaring the conditioning is benign — cond ≈ 2 measured.
    Snn = zeros(Float64, ns, ns); bsh = zeros(Float64, ns)
    Cnb = zeros(Float64, ns, nchosen); Dbb = zeros(Float64, nchosen); bdep = zeros(Float64, nchosen)
    for (col, (i, R, t12, carried)) in enumerate(chosen)
        # the board's whitening model: the previous iteration's pose (a board it
        # placed, in the branch it chose), the seed's (a board with a seed
        # depth), else plain rows. A board placed afresh on a later iteration —
        # the audit moved its branch, or the previous iteration left it without
        # a pose — enters that ONE iteration plain, and whitened from the next
        # (it then carries a pose). Plain rows weigh it by its depth, well above
        # a whitened view; whitening it at the shared solution instead (at its
        # weighted or unweighted least-squares depth) was measured 2026-09-30
        # and is WORSE: the MC calibration gate at 3 % ray-aligned
        # (`test/ransac_mc_calibration_tests.jl` (e), `is_calibrated(r3)`)
        # fails with it and holds without. The audit iteration's weights are
        # the pre-audit posterior, which has zeroed a wrongly placed view's
        # inliers; its plain rows are what still lets that view pull.
        prev = carried ? whiten_at.poses[i] : nothing
        pw_i = point_weights === nothing ? nothing : point_weights[i]
        t3w = prev === nothing ? _seed_depth(whiten_at, R, t12, nboards[i], fw, λw, pw_i) : nothing
        m = if prev !== nothing
            tn = S * translation(prev ∘ Translation(X̄0))            # the pose, normalized
            _lift_model(R, SVector(tn[1], tn[2]), fw, λw, tn[3])
        elseif t3w !== nothing
            _lift_model(R, t12, fw, λw, t3w)
        else
            _lift_model(R, t12, 0.0, ntuple(_ -> 0.0, N), 0.0)
        end
        whitened = prev !== nothing || t3w !== nothing
        for (j, (pt, d)) in enumerate(nboards[i])
            wj = pw_i === nothing ? 1.0 : sqrt(Float64(pw_i[j]))
            wj > 0 || continue                              # a zero-weight row adds nothing — and 0 · Inf is NaN
            G, rhs = _lift_rows(m, kn, pt, d, whitened ? whitening : NoWhitening())
            # A degenerate whitening (the measurement Jacobian vanishes at the
            # whitening point, |z| → 0) has no finite row: it carries no usable
            # constraint, and a non-finite entry would poison the whole solve.
            (all(isfinite, G) && all(isfinite, rhs)) || continue
            @inbounds for r in 1:2
                gt = wj * G[r, ns + 1]                      # the depth column entry
                rb = wj * rhs[r]
                for p in 1:ns
                    gp = wj * G[r, p]
                    for q in 1:ns
                        Snn[p, q] += gp * (wj * G[r, q])
                    end
                    Cnb[p, col] += gp * gt
                    bsh[p] += gp * rb
                end
                Dbb[col] += gt * gt
                bdep[col] += gt * rb
            end
        end
    end
    # A board whose depth column carries no information — every row's weight
    # driven to zero by the robust re-weighting, or every row at the centre —
    # leaves the iteration without a pose (the record's `nothing`), the way a board
    # with no admissible metric upgrade does; it contributed nothing to the
    # shared rows either. Only an iteration with no board left is an error.
    live = [Dbb[col] > 0 for col in 1:nchosen]
    any(live) || throw(DivisionCameraError("no board carries depth information (every row weighted to zero or at the centre)"))
    Sred = copy(Snn); bred = copy(bsh)
    @inbounds for col in 1:nchosen
        live[col] || continue
        dcol = Dbb[col]
        for p in 1:ns
            cp = Cnb[p, col] / dcol
            for q in 1:ns
                Sred[p, q] -= cp * Cnb[q, col]
            end
            bred[p] -= cp * bdep[col]
        end
    end
    xs = try
        Sred \ bred
    catch e
        e isa LinearAlgebra.SingularException || rethrow()
        throw(DivisionCameraError("the solve is singular — the boards do not determine the shared unknowns"))
    end
    all(isfinite, xs) || throw(DivisionCameraError("the solve did not produce finite shared unknowns"))
    x = Vector{Float64}(undef, ns + nchosen)
    x[1:ns] .= xs
    @inbounds for col in 1:nchosen
        if !live[col]
            x[ns + col] = NaN
            continue
        end
        acc = bdep[col]
        for p in 1:ns
            acc -= Cnb[p, col] * xs[p]
        end
        x[ns + col] = acc / Dbb[col]
    end
    x[1] > 0 || throw(DivisionCameraError("the solve gave a non-positive focal length ($(x[1] / σ))"))
    # back to data units: f = f'/σ, and the normalized-frame coefficients kₚ' = λₚ'/f'
    # re-expressed in pixels by the coefficient law (pixels are the normalized
    # coordinates divided by σ). The iteration's DivisionCameraState: poses/depths in
    # data units, whitened residuals at the solution.
    fpx = x[1] / σ
    kn′, kpx = _solved_coeffs(k, kn, x, σ)
    λn = ntuple(p -> kn′[p] * x[1], N)                            # λₚ' = kₚ'·f' in normalized units
    poses = Vector{Union{Nothing, _BoardPose}}(nothing, nb)
    sq = [fill(NaN, length(boards[i])) for i in 1:nb]
    for (col, (i, R, t12)) in enumerate(chosen)
        live[col] || continue                                         # no pose this iteration
        t3n = x[ns + col]
        poses[i] = EuclideanMap(R, SVector(t12[1], t12[2], t3n) / S; check = false) ∘ Translation(-X̄0)   # plane units
        m = _lift_model(R, t12, x[1], λn, t3n)
        for (j, (pt, d)) in enumerate(nboards[i])
            g, G_x, G_θ = constraint_with_jacobians(m, pt, d)
            gw, _ = whitened_rows(whitening, g, G_x, G_θ)
            sq[i][j] = (gw[1]^2 + gw[2]^2) / σ^2                     # pixel² (policy currency)
        end
    end
    return _PositionalDivisionState(fpx, collect(Float64, kpx), center, poses, sq)
end

const _RadialCameras = AbstractVector{<:Union{HomographyMat, RadialCamera}}

# --- The pencil's centre --------------------------------------------------
#
# The lift and the pose candidates are written about the ORIGIN of the
# normalized coordinates the stack is solved in. Each `RadialCamera` carries the centre; the stack's
# normalization is Hartley ABOUT it on the image side
# (`CspondNormalization(DivisionCameraState, views; center)`), so the
# normalized pixels and the composed cameras are centred by the one map, and
# `_check_division_normalization` holds the normalization to that. A bare
# `HomographyMat` means the origin, which is what it has always meant.
_model_center(m::AbstractRadialModel) = (c = center(m); Point2{Float64}(c[1], c[2]))
_model_center(::HomographyMat) = Point2(0.0, 0.0)
# The ONE lift of an input to its radial camera (rows 1–2 of a homography about
# its origin centre, VGC's `RadialCamera(H)`), so every map the solve composes
# is a camera, never a bare matrix.
_radial_camera(H::HomographyMat) = RadialCamera(H)
_radial_camera(P::RadialCamera) = P

# The stack's shared centre. One focal length implies one pencil, so every
# model must agree on the centre.
function _shared_center(Hs::_RadialCameras)
    isempty(Hs) && throw(ArgumentError("division_camera_step: no radial cameras"))
    c = _model_center(first(Hs))
    for H in Hs
        ci = _model_center(H)
        isapprox(ci, c; atol = 1.0e-9) || throw(
            ArgumentError(
                "division_camera_step: every radial camera must share one centre " *
                    "(the principal point); got $(Tuple(c)) and $(Tuple(ci))"
            )
        )
    end
    return c
end

# THE division-camera stack's normalization — which both the step's callers and
# `fit_division_camera` build — is the RADIAL CAMERA's strategy
# (`CspondNormalization(RadialCamera, cs; center)`, VisualGeometryToolkitCore) over the
# views pooled: the stack shares one pencil, so one camera's strategy serves
# it. Hartley at the plane points' centroid, Hartley ABOUT `c` on the image
# side, each point weighed by its `weights` column when the views carry one.
function _division_normalization(boards::AbstractVector, c, w = _point_weights(boards))
    plane = [SVector{2, Float64}(pd[1]) for pts in boards for pd in pts]
    image = [SVector{2, Float64}(pd[2]) for pts in boards for pd in pts]
    pooled = w === nothing ? VgtkCore.Csponds(plane, image) :
        VgtkCore.Csponds(plane, image; weights = reduce(vcat, w))
    return CspondNormalization(RadialCamera, pooled; center = SVector{2, Float64}(c[1], c[2]))
end

"""
    CspondNormalization(::Type{DivisionCameraState}, views::AbstractDict; center)

The normalization the division-camera stack is solved in — the `N` that
`division_camera_step` takes — pooled over every view's
`plane_point => raw_pixel` correspondences — the radial camera's strategy,
`CspondNormalization(RadialCamera, cs; center)`, over the pooled views: Hartley
at the plane points' centroid, and Hartley about `center` (pixels; the radial
cameras' shared centre, the principal point) on the image side. A view's
`weights` column weighs its points. `fit_division_camera` builds the same.
"""
VgtkCore.CspondNormalization(
    ::Type{<:DivisionCameraState}, views::AbstractDict{K, <:AbstractVector{<:Cspond}}; center
) where {K} = _division_normalization([views[id] for id in block_order(views)], center)

# The positional form — views in a fixed order, the row weights as
# `point_weights` (one vector per view), the estimator's normalization `N` —
# that the keyed `division_camera_step` and the IRLS of `fit_division_camera`
# run on. Pixels in, pixels out; `N` is applied inside, never built.
function _division_camera_step(
        N::CspondNormalization, Hs::_RadialCameras, boards::AbstractVector;
        ncoef::Int = 1, k::Union{Nothing, Real} = nothing,
        point_weights = nothing, whiten_at::Union{Nothing, _PositionalDivisionState, Real} = nothing,
        whitening::WhiteningPolicy = SampsonWhitening(), audit::Bool = false
    )
    kk = _coeff_mode(k, ncoef)
    seed = _check_whiten_at(whiten_at, kk, length(boards), audit)
    length(Hs) == length(boards) || throw(ArgumentError("division_camera_step: one correspondence set per radial camera"))
    center = _shared_center(Hs)
    return _division_camera_solve(Hs, boards, kk, N; center, point_weights, whiten_at = seed, whitening, audit)
end

# The coefficient mode of the stack (see `_nshared`), checked: without `k`,
# `ncoef ≥ 1` coefficients are estimated; with it the one coefficient is fixed
# and only f and the t₃'s are estimated, so a larger `ncoef` would be silently
# ignored — refused.
function _coeff_mode(::Nothing, ncoef::Int)
    ncoef < 1 && throw(ArgumentError("division_camera_step: ncoef must be ≥ 1"))
    return ncoef
end
function _coeff_mode(k::Real, ncoef::Int)
    ncoef == 1 ||
        throw(ArgumentError("division_camera_step: `k` fixes the division coefficient, so ncoef must stay 1 (got $ncoef)"))
    return Float64(k)
end

# The whitening point, checked and in the solve's form (a seed focal as `Float64`).
function _check_whiten_at(w::Union{Nothing, Real}, _kmode, _nviews, audit::Bool)
    audit && throw(ArgumentError("division_camera_step: the branch audit needs a previous iteration to judge at"))
    return _checked_seed(w)
end
_checked_seed(::Nothing) = nothing
function _checked_seed(f::Real)
    (f > 0 && isfinite(f)) ||
        throw(ArgumentError("division_camera_step: a seed focal `whiten_at` must be positive and finite, got $f"))
    return Float64(f)
end
function _check_whiten_at(w::_PositionalDivisionState, kmode, nviews, _audit::Bool)
    length(w.poses) == nviews ||
        throw(ArgumentError("division_camera_step: `whiten_at` carries $(length(w.poses)) views, the stack has $nviews"))
    nk = _ncoeffs(kmode)
    length(w.k) == nk || throw(
        ArgumentError(
            "division_camera_step: `whiten_at` carries $(length(w.k)) division coefficient(s), " *
                "this iteration solves $nk — whiten at a state of the same model"
        )
    )
    return w
end

function _keyed_views(radial_cameras::AbstractDict, views::AbstractDict)
    ids = block_order(views, radial_cameras)          # one radial camera per view: the key sets must match
    foreach(id -> _refuse_physical_plane(views[id], "the division camera"), ids)   # board points as the radial cameras took them
    Hs = [radial_cameras[id] for id in ids]
    boards = [views[id] for id in ids]
    return ids, Hs, boards
end

# The views' `weights` columns (VGC's one reader) as per-row priors (a view
# without one weighs 1), checked (`_check_row_weights`), or `nothing` when no
# view has one — the unweighted solve, bit for bit.
function _point_weights(boards::AbstractVector)
    any(b -> VgtkCore.weights_column(b) !== nothing, boards) || return nothing
    return _check_row_weights([_row_weights(b) for b in boards])
end
function _row_weights(b)
    w = VgtkCore.weights_column(b)
    return w === nothing ? ones(length(b)) : collect(Float64, w)
end

"""
    division_camera_step(N, radial_cameras::AbstractDict{K}, views::AbstractDict{K};
                         ncoef = 1, k = nothing, whiten_at, whitening)
        -> DivisionCameraState{K}

A declared INNER-LOOP piece (public, not exported): it takes the estimator's
normalization `N` and pixel data, and returns pixel state. `N` is REQUIRED —
`CspondNormalization(DivisionCameraState, views; center)`, the one strategy
[`fit_division_camera`](@ref) builds — applied to the data inside, never built
here; there is no fallback.

One iteration of the stacked linear focal solve via the division-model lift
(BabelCalib's back-end back-substitution, [Lochman2021](@citet)), as a [`DivisionCameraState`](@ref): the shared
unknowns plus the per-board poses and the whitened per-point residuals the next
iteration needs. Square pixels are assumed. Both inputs are keyed by the view's
identity `K` (the key sets must be equal — `ArgumentError`): `radial_cameras[id]`
is view `id`'s radial camera (a `RadialCamera`, or a `HomographyMat` whose rows
1–2 are read) in the SAME plane frame as `views[id]`, that view's
`plane_point => image_point` correspondences with the RAW (distorted) image
point in pixels. The pencil's centre rides on the cameras, must be shared across
the stack, and is carried by the result; a bare `HomographyMat` means the origin.

Per board, the radial camera fixes the rotation and (t₁, t₂) up to four sign
candidates; the remaining unknowns are LINEAR in the division-model lift
`(x, y, 1 + Σₚ kₚ r²ᵖ) ∝ (X, Y, Z + t₃)` ([`RadialLift`](@ref)). The candidate
is chosen by cheirality and residual, and every board is stacked into one
least-squares system with the shared unknowns and one `t₃` per board.
A view's `weights` column (a `Csponds` attribute, e.g. `withattrs(cs; weights)`)
weighs its rows — a board-level vote folds into it losslessly, the row weight
being √w; views with no admissible candidate are
dropped, and a [`DivisionCameraError`](@ref) is thrown if none survive or the
stack yields `f ≤ 0` (the argument contract still raises `ArgumentError`). With `k` given (the one-parameter division coefficient in the
centered-pixel frame, e.g. `RadialHomography.k`), only `f` and the `t₃`'s are
estimated (`ncoef` must stay 1); otherwise the stack carries `ncoef ≥ 1` extra shared columns
`λₚ = kₚ·f` and estimates the coefficients of r², r⁴, … too.

`N` is the normalization the stack is solved in: its image side must be an
isotropic scale about the radial cameras' shared centre, its plane side an
isotropic similarity (`ArgumentError` otherwise). The state is returned in
pixels regardless.
`whiten_at` whitens every row under `whitening` (the residual-currency
policy, `SampsonWhitening()` by default) at
- a previous iteration's `DivisionCameraState` — its solution, each view in the
  branch that iteration chose; or
- a SEED focal length (a positive `Real`, data units): the pinhole `f`, k₀ = 0
  (the given `k` when `k` fixes it), each view in its freshly chosen branch at
  the depth that camera implies. The seed
  sets the WEIGHTS only — `f` and `k` are still solved by this iteration. It puts
  iteration 0 in pixel currency, where the plain rows weigh each view by its depth
  (a far board's rows ~10× a near one's); see [`fit_division_camera`](@ref).

The camera an iteration determines is `CameraModel(state)`:
`K = diag(f, f, 1)` with the principal point at the cameras' shared centre, and
a `Division{ncoef}` lens re-expressed in that K's normalized frame
(`kₚ,norm = kₚ,px · f^(2p)`, VGC's radial-coefficient law — the lens law of
the `CameraModel(model, K′)` constructor).

ONE ITERATION, not an estimator. Without `whiten_at` there is nothing to linearize
at, so the SOLVE is plain algebraic least squares (`whitening` still sets the
currency of the returned `sq_residuals`, which the robust driver reweights by).
The robust driver's iteration 0 is this step at its seed,

    N = CspondNormalization(DivisionCameraState, views; center)
    CameraModel(division_camera_step(N, radial_cameras, views; ncoef, whiten_at = focal_seed))
    estimate(fit_division_camera(radial_cameras, views; ncoef, focal_seed, max_iter = 0))

the same camera, bit for bit. For a converged estimate call the robust driver.

Units: the board points are plain numbers. The radial cameras and the board
poses carry the board plane's unit, which neither a matrix nor this result
records, so physical (unitful) board points are an `ArgumentError`: strip
them in the unit the radial cameras were fit in (`ustrip.(mm, X)`).
"""
function division_camera_step(
        N::CspondNormalization,
        radial_cameras::AbstractDict{K, <:Union{HomographyMat, RadialCamera}},
        views::AbstractDict{K, <:AbstractVector{<:Cspond}};
        ncoef::Int = 1, k::Union{Nothing, Real} = nothing,
        whiten_at::Union{Nothing, DivisionCameraState{K}, Real} = nothing,
        whitening::WhiteningPolicy = SampsonWhitening()
    ) where {K}
    ids, Hs, boards = _keyed_views(radial_cameras, views)
    prev = _positional(whiten_at, ids)
    p = _division_camera_step(N, Hs, boards; ncoef, k, point_weights = _point_weights(boards), whiten_at = prev, whitening)
    return _keyed(p, ids)
end

# CameraModel(state::DivisionCameraState) -> CameraModel
#
# The camera a solve determines: `K = diag(f, f, 1)` with its principal
# point at the iteration's `centre` (the radial cameras' shared pencil centre,
# VGC's pixel intrinsics `pixel_intrinsics`), and the `Division{ncoef}` lens
# re-expressed in that K's normalized frame by the coefficient law (`st.k` is in
# centred pixels; normalized = pixels / f).
#
# THE one place this construction lives — it used to exist twice, here and
# hand-rolled in VisualGeometryToolkitEstimation's IRLS driver, which is how the two could
# drift apart.
#
# A COMMENT, not a docstring: `@doc` on a method extension of an imported
# constructor fails precompilation ("cannot document the following expression").
VgtkCore.CameraModel(st::DivisionCameraState) =
    CameraModel(pixel_intrinsics(st.f, st.f, st.center), Division(_rescaled_coeffs(st.k, st.f)...))

# The division coefficients in a frame whose coordinates are the old ones divided
# by `s`: kₚ → kₚ·s^(2p), VGC's one radial-coefficient law (`rescale_radial_coeff`,
# the law the `CameraModel(model, K′)` constructor applies to a lens).
_rescaled_coeffs(k, s) = [rescale_radial_coeff(k[p], s, p) for p in eachindex(k)]

# The image side must be an isotropic scale ABOUT THE CENTRE `c`: the lift is
# written about the normalized origin, so a map that does not send the principal
# point there would silently move it (a centroid Hartley map, say). The plane
# side is an isotropic scale too (with any translation). Returns the image- and
# plane-side scales (σ, S).
function _check_division_normalization(N::CspondNormalization, c)
    Dm, Sm = target_map(N), source_map(N)
    σ = _isotropic_scale(Dm, "image")
    oc = Dm(SVector{2, Float64}(c[1], c[2]))
    abs(oc[1]) + abs(oc[2]) <= 1.0e-9 || throw(
        ArgumentError(
            "division_camera_step: the normalization's image side must be about the radial cameras' centre " *
                "$(Tuple(c)) — build it with CspondNormalization(DivisionCameraState, views; center)"
        )
    )
    return σ, _isotropic_scale(Sm, "plane")
end

# A normalization side's scale: VGC's `similarity_scale` (the one isotropy test,
# to rounding), and no rotation or reflection — the poses (the origin shift above)
# and the lift are read off the normalized axes, which must be the data's.
function _isotropic_scale(m, side::AbstractString)
    σ = try
        similarity_scale(m)
    catch e
        e isa ArgumentError || rethrow()
        throw(ArgumentError("division_camera_step: the normalization's $side side must be an isotropic scale"))
    end
    L = m.linear
    (L[1, 1] > 0 && L[2, 2] > 0 && abs(L[1, 2]) + abs(L[2, 1]) <= 64 * eps(σ)) ||
        throw(ArgumentError("division_camera_step: the normalization's $side side must be an isotropic scale (no rotation or reflection)"))
    return σ
end
