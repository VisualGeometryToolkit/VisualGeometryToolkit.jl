# =============================================================================
# Implicit-function hypothesis covariance in minimal model manifolds
# (manifold promotion, step 2 of the uncertainty thread)
#
# THE DEFINITION. The hypothesis covariance attached to a model θ₀ is the
# local sensitivity of the constrained solution at θ₀: the solution θ(x) of
# the stacked algebraic constraints g(θ, x) = 0 over the minimal sample x
# (k points, d_g constraints each), through the point θ₀. Parameterizing θ
# by a Manifold anchored AT θ₀ (v = 0 ⇔ θ₀), the implicit function theorem
# gives the sensitivity in local coordinates:
#
#     ∂v/∂x = −(∂_v g)⁻¹ · ∂_x g,   ∂_v g = G_θ · T,   T = retract_jacobian(ch)
#
# where G_θ = ∂g/∂vec(θ) and G_x = ∂g/∂(x₁,y₁,x₂,y₂) come from
# `constraint_with_jacobians` (both factors already exist for H / F / E). For a
# MINIMAL sample, ∂_v g is square (k·d_g = dof) and generically invertible —
# the minimal manifolds have no gauge nullity, so no pseudo-inverse is needed.
#
# The definition involves the manifold and the constraint system only — it is
# ALGORITHM-INDEPENDENT (no solver enters), a property of the problem's
# constraint variety, exact for the smooth local solution through θ₀. In the
# pipeline it is evaluated at a DELIVERED hypothesis θ₀ (whatever produced
# it); ill-conditioned geometry announces itself through the size of Σ̃_v.
#
# The tangent-domain, UNIT-SCALE hypothesis covariance is then
#
#     Σ̃_v = J · Σ̃_x · Jᵀ,    J = ∂v/∂x  (D × 4k)
#
# CONVENTIONS
# • UNIT-SCALE (multiplicative): Σ̃_x defaults to I (unit isotropic measurement
#   noise); the physical covariance is Σ_v = s²·Σ̃_v with s profiled by the
#   consumer (generalized sweep lemma convention — never bake a frozen s² in).
#   Per-point (possibly anisotropic) 4×4 measurement covariances may be supplied
#   explicitly; they are then also interpreted as unit-scale SHAPES.
# • GAUGE/SCALE INVARIANCE: g is homogeneous of degree 1 in vec(θ), so J is
#   invariant to the representative scale/sign of the anchor model — manifold
#   anchors built from any solver normalization (‖F‖ = 1, F₃₃ ≥ 0, ‖t‖ = 1
#   with either sign of E, …) give the same Σ̃_v, and the pushed ambient
#   covariance T·Σ̃_v·Tᵀ depends only on the model-manifold tangent space,
#   not on the parameterization chosen through the anchor.
#
# VALIDATION: test/manifold_covariance_tests.jl (FD pins of the implicit J
# against the tracked/unique local solution) and the MC calibration harness
# (VisualGeometryToolkitCore.Synthetic: `run_calibration`, `run_containment` and the
# trial builders — output-domain containment per the Criminisi Table 4.1
# protocol, plus tangent-domain NEES / χ² coverage at the ground-truth anchor).
#
# PERFORMANCE NOTE: the core uses small heap matrices (D ≤ 8, 4k ≤ 28) —
# correctness-first. If predictive scoring makes this a per-hypothesis hot
# path, specialize per manifold type with static sizes (deliberately left to the
# performance work; do not fold into the LO-engine files).
# =============================================================================

"""
    constraint_dim(model) -> Int

Number of scalar algebraic constraints per correspondence for `model`'s
class: 1 for the epipolar models (`FundMat`, `EssentialMat`), 2 for
`HomographyMat` and for a pose (`EuclideanMap{3}`, the reprojection constraint).
Matches the row count of `constraint_with_jacobians`.
"""
constraint_dim(::Type{<:AbstractFundMat}) = 1
constraint_dim(::Type{<:HomographyMat}) = 2
constraint_dim(::Type{<:EuclideanMap{3}}) = 2
constraint_dim(::Type{<:Line2D}) = 1
constraint_dim(m) = constraint_dim(typeof(m))

# The measurement count of one correspondence: both endpoints of a two-view pair
# (x₁, y₁, x₂, y₂); for a pose, the calibrated image point — its 3D point is
# exact, the PnP measurement model.
_measurement_dim(::Type{<:EuclideanMap{3}}) = 2
_measurement_dim(::Type{<:Line2D}) = 2          # single-point data: the point is the measurement
_measurement_dim(::Type) = 4
_measurement_dim(m) = _measurement_dim(typeof(m))

"""
    minimal_manifold(model) -> Manifold

The minimal (gauge-free) manifold anchored at `model`:
`FundMat → FundMatManifold7` (via the anchor SVD), `HomographyMat →
HomographyManifold8`, `EssentialMat → EssentialManifold5` (via the first
`motion(E)` candidate — the tangent-domain covariance and its ambient push are
candidate-invariant, see the header gauge note), `EuclideanMap{3} →
RigidManifold6`, `RadialHomography →` sl(3) ⊕ k (a `ProductManifold`),
`RadialCamera →` S⁵ (a `SphereManifold`, cameras/calibration.jl), a lens → its
`LensManifold` (cameras/lens_chart.jl).
"""
minimal_manifold(T::PlainEuclideanMap{3}) = RigidManifold6(T.R, T.t)
minimal_manifold(R::Rotation{3}) = SO3Manifold(R)
minimal_manifold(F::FundMat) = FundMatManifold7(SMatrix{3, 3, Float64, 9}(F))
minimal_manifold(H::HomographyMat) = HomographyManifold8(SMatrix{3, 3, Float64, 9}(H))
function minimal_manifold(E::EssentialMat)
    R, t = first(motion(E))
    return EssentialManifold5(RotMatrix{3, Float64}(R), Vec3{Float64}(t))
end

# RadialHomography: the plane homography's sl(3) chart ⊕ the division coefficient,
# the centre riding along — the chart whose directions `radial_transfer_jacobian_rows!`
# takes (dH from the first part, k last).
struct _RadialHomographyParts{T}
    center::Point2{T}
end
function (e::_RadialHomographyParts)(parts)
    H, k = parts
    T = promote_type(eltype(H), eltype(k))
    return RadialHomography{T}(HomographyMat{T}(Tuple(H)), T(k[1]), Point2{T}(e.center[1], e.center[2]))
end
"""
    constraint_with_jacobians(l::Line2D, x::StaticVector{2}) -> (g, G_x::SVector{2}, G_θ::SVector{3})

The point-on-line constraint `g = n·x + d` — the point's signed distance (the
normal is unit) — with `G_x = n` and `G_θ = ∂g/∂(n₁, n₂, d) = (x₁, x₂, 1)`, the
line's coefficients ([`Line2DManifold`](@ref)'s ambient space). Single-point
data: the covariance machinery takes a vector of points for a line chart.
"""
@inline function constraint_with_jacobians(l::Line2D, x::StaticVector{2})
    n = normal(l)
    g = n[1] * x[1] + n[2] * x[2] + coeffs(l)[3]
    return g, n, SVector(x[1], x[2], one(eltype(x)))
end
minimal_manifold(l::Line2D) = Line2DManifold(l)

ambient_coordinates(m::RadialHomography) = vcat(SVector{9}(Tuple(SMatrix{3, 3}(m.H))), SVector(m.k))
minimal_manifold(m::RadialHomography) = ProductManifold(
    HomographyManifold8(SMatrix{3, 3, Float64, 9}(m.H)), EuclideanManifold(SVector(m.k));
    embed = _RadialHomographyParts(m.center)
)
local_coordinates(p::ProductManifold{<:Tuple, <:_RadialHomographyParts}, m::RadialHomography) =
    local_coordinates(p, (m.H, SVector(m.k)))

# One correspondence's constraint value, its measurement gradient and its AMBIENT
# model gradient — over the parameters `retract_jacobian` differentiates: vec(M)
# for the matrix models, [vec(R); t] for a pose (`X => u`, `u` the calibrated
# image point), where the model gradient is `Jπ·∂(R·X + t)/∂[vec(R); t]`.
@inline cd_pair(M, c::Pair) = constraint_with_jacobians(M, c.first, c.second)
@inline cd_pair(M, x::StaticVector) = constraint_with_jacobians(M, x)     # single-point data (a line's points)
@inline function cd_pair(pose::PlainEuclideanMap{3}, c::Pair{<:Any, <:StaticVector{2}})
    X = SVector{3, Float64}(c.first[1], c.first[2], c.first[3])
    g, _, _ = constraint_with_jacobians(pose, X, SVector{3, Float64}(c.second[1], c.second[2], 1.0))
    Jπ = transform_deriv(PerspectiveMap(), SMatrix{3, 3, Float64, 9}(pose.R) * X + SVector{3, Float64}(pose.t))
    return g, -SMatrix{2, 2, Float64, 4}(I), Jπ * transform_deriv_params(pose, X)
end
# The pose chart's image point is the calibrated z = 1 point; anything else (a
# unit bearing) would be read as its first two components. Refuse it.
cd_pair(::PlainEuclideanMap{3}, c::Pair) = throw(ArgumentError(
    "the pose chart takes calibrated 2-vector image points `X => u`, got a $(length(c.second))-vector"
))
# The entry every per-pair loop calls: correspondence `i` of the SET `cs`, so a
# constraint that reads per-point attribute columns can reach them (`cs[i]` is a
# bare Pair).
@inline cd_pair(M, cs::AbstractVector, i::Integer) = cd_pair(M, cs[i])
# An essential matrix on BEARINGS: the tangent form, its G_x in pixels through the
# backprojection Jacobians the set carries as its `M₁`/`M₂` columns
# (`withattrs(cs; M₁, M₂)`). A bare bearing Pair has no Jacobians and the
# keyword-only tangent form refuses it.
@inline function cd_pair(
        E::EssentialMat, cs::Csponds{<:StaticVector{3}, <:StaticVector{3}}, i::Integer
    )
    c = cs[i]
    return constraint_with_jacobians(E, c.first, c.second; M₁ = cs.M₁[i], M₂ = cs.M₂[i])
end

"""
    manifold_constraints(ch::Manifold, v, cs) -> Vector

The stacked algebraic constraints `g(v, x)` of the correspondences `cs`
(`u₁ => u₂` pairs; `X => u` for a pose chart, `u` calibrated; single points for
a line) at chart
coordinates `v`: `constraint_with_jacobians`' value for `retract(ch, v)`,
`constraint_dim` rows per pair. Zero on the constraint variety; the residual a
chart-Newton solve drives to zero.
"""
function manifold_constraints(ch::Manifold, v::StaticVector, cs::AbstractVector)
    M = retract(ch, v)
    dg = constraint_dim(M)
    g = Vector{Float64}(undef, dg * length(cs))
    @inbounds for i in eachindex(cs)
        gi, _, _ = cd_pair(M, cs, i)
        if dg == 1
            g[i] = gi
        else
            for rr in 1:dg
                g[dg * (i - 1) + rr] = gi[rr]
            end
        end
    end
    return g
end

"""
    constraint_jacobians(ch::Manifold, v, cs) -> (∂_v g, ∂_x g)

The Jacobians of [`manifold_constraints`](@ref) at chart coordinates `v`: with
respect to `v` (`G_θ·retract_jacobian(ch, v)`, `dg·k × D`) and to the measurements
(block diagonal, `dg·k × nₓk`, each pair's `G_x` over `(x₁, y₁, x₂, y₂)`, or over
the image point of a pose correspondence, nₓ = 2). At
`v = 0` these are the two factors of the implicit-function covariance
([`implicit_covariance`](@ref)); at `v ≠ 0`, the Newton matrix of a
chart-Newton solve that iterates in the manifold.

NO Ceres or GTSAM counterpart: both frame a problem as residuals of an
EXPLICIT cost, whereas this is the IMPLICIT constraint `g(θ, x) = 0` in
Förstner's form — `A = ∂g/∂θ`, `B = ∂g/∂x` — which is what an
implicit-function covariance and a Gauss-Helmert step consume. The
value-and-Jacobians sibling is `constraint_with_jacobians`.
"""
function constraint_jacobians(ch::Manifold, v::StaticVector, cs::AbstractVector)
    M = retract(ch, v)
    dg = constraint_dim(M)
    nx = _measurement_dim(M)
    D = tangent_size(ch)
    k = length(cs)
    T = retract_jacobian(ch, v)                    # ambient × D
    A = Matrix{Float64}(undef, dg * k, D)          # ∂_v g
    B = zeros(Float64, dg * k, nx * k)             # ∂_x g (block diagonal)
    @inbounds for i in 1:k
        _, Gx, Gθ = cd_pair(M, cs, i)
        rows = (dg * (i - 1) + 1):(dg * i)
        A[rows, :] = _chart_rows(T, Gθ, Val(dg))
        if dg == 1                                  # Gx :: SVector{nx}
            for j in 1:nx
                B[i, nx * (i - 1) + j] = Gx[j]
            end
        else
            for rr in 1:dg, j in 1:nx
                B[dg * (i - 1) + rr, nx * (i - 1) + j] = Gx[rr, j]
            end
        end
    end
    return A, B
end

# One correspondence's rows of ∂_v g = G_θ·T: a codimension-1 constraint's G_θ is
# the ambient gradient (a vector), so its one row is Tᵀ·G_θ laid flat.
@inline _chart_rows(T, Gθ, ::Val{1}) = _one_row(transpose(T) * Gθ)
@inline _chart_rows(T, Gθ, ::Val) = Gθ * T

"""
    implicit_covariance(ch::Manifold, cs; Σx = nothing) -> Matrix (D×D) or nothing
    implicit_covariance(block, indices; noise_weighted = false) -> Matrix or nothing
    implicit_covariance(A::AbstractMatrix, B::AbstractMatrix; Σx = nothing) -> Matrix or nothing

The first-order covariance of the solution `v(x)` of an implicitly defined
estimation problem (Haralick's framing): `g(v, x) = 0` for a square system (a
minimal solve), `∂_v(½‖W₀^{½}·g‖²) = 0` — least-squares stationarity of the
whitened constraints — for an overdetermined one. With `A = ∂_v g`, `B = ∂_x g`
and the estimator's own isotropic whitening `W₀ = blockdiag((BᵢBᵢᵀ)⁻¹)` over the
correspondences' row blocks,

    Σ̃_v = (AᵀW₀A)⁻¹ · AᵀW₀·(B·Σx·Bᵀ)·W₀A · (AᵀW₀A)⁻¹,

which is `A⁻¹·B·Σx·Bᵀ·A⁻ᵀ` when `A` is square (the whitening cancels — the
implicit-function theorem proper) and `(AᵀW₀A)⁻¹` for unit isotropic `Σx`. Unit
scale: `Σx` defaults to unit isotropic measurement noise; the physical covariance
is `s²·Σ̃_v` with `s` the consumer's. Invariant to any invertible
per-correspondence recombination of the constraint rows (the algebraic and the
transfer forms of a homography give the same `Σ̃_v`).

- **Chart first**: the constraints of the correspondences `cs` (`u₁ => u₂` pairs;
  `X => u` for a pose chart; single points for a line) differentiated at the
  anchor of `ch` ([`constraint_jacobians`](@ref)), in the coordinates of
  `ch`, which must be anchored at the estimate. `Σx`: one covariance shape per
  correspondence (4×4 over `(x₁, y₁, x₂, y₂)`; 2×2 for a pose's or a line's
  point), in the coordinates of `cs` — data mapped to another frame needs its
  noise transported there (the map's `transform_deriv`) before a pixel `s²` means
  anything. A minimal sample (`k·d_g = D`) takes the square solve.
- **Blocks**, never materializing `B` (`O(D²)` memory, for large inlier sets):
  `block(i)` returns `(Aᵢ, Bᵢ)` or `(Aᵢ, Bᵢ, Σxᵢ)` for each `i` in `indices` — the
  correspondence's rows of `∂_v g` and `∂_x g` (a `StaticVector` is one row) and
  its covariance shape (unit isotropic when absent). Function first, as in
  `map(f, xs)`. With `noise_weighted = true` the estimator is instead the one
  weighted by each block's OWN noise, `W₀ᵢ = (BᵢΣxᵢBᵢᵀ)⁻¹` (the Σxᵢ-whitened
  residuals a scorer uses), whose covariance is `(AᵀW₀A)⁻¹` — `Σxᵢ` folded into
  `Bᵢ` as `Bᵢ·chol(Σxᵢ).L`. The default sandwiches `Σxᵢ` around the
  isotropically weighted estimator: a different estimator, a different
  covariance whenever the `Σxᵢ` are not isotropic.
- **Dense square factors**: `A = ∂_v g` (D×D) and `B = ∂_x g` (D×nₓk) of a
  square system `g(v, x) = 0` whose rows are not per-correspondence constraints
  — e.g. the stationarity conditions of a staged estimator, one row block per
  stage — gives `A⁻¹·B·Σx·Bᵀ·A⁻ᵀ` directly. `Σx`: one covariance shape per
  column block of `B` (`nₓ` columns each, `k` blocks); unit when absent. The
  minimal-sample path of the chart form above is this method.

`nothing` when the system is singular (`A` for a square one, `AᵀW₀A` otherwise —
a degenerate sample, the same the solvers' own gates reject, or ill-posed
geometry) or when there are no blocks; a block whose `Bᵢ` is degenerate
([`whitening_factor`](@ref)'s rule) drops out, as it does from the estimator.

GTSAM equivalent: `Marginals::jointMarginalCovariance` at the optimum, when the
factor rows are correctly whitened. This one also covers the square minimal
solve and the sandwich for a mismatched weighting.
"""
function implicit_covariance(ch::Manifold, cs::AbstractVector; Σx = nothing)
    M = model(ch)
    dg = constraint_dim(M)
    nx = _measurement_dim(M)
    D = tangent_size(ch)
    k = length(cs)
    k * dg >= D || throw(
        ArgumentError("implicit_covariance: $(k * dg) constraints for $D degrees of freedom")
    )
    Σx === nothing || length(Σx) == k || throw(
        ArgumentError("implicit_covariance: need one $(nx)×$(nx) covariance per correspondence")
    )
    v₀ = zero(SVector{D, Float64})
    # a minimal sample: the square solve on the dense factors
    k * dg == D && return implicit_covariance(constraint_jacobians(ch, v₀, cs)...; Σx)
    # overdetermined: each correspondence's rows fed as a block, B never assembled
    return _chart_sandwich(retract(ch, v₀), retract_jacobian(ch, v₀), cs, Σx, Val(dg), Val(nx))
end

# One correspondence's block of the chart form: its rows of ∂_v g = G_θ·T (the
# rows `constraint_jacobians` writes), its G_x and its covariance shape.
function _chart_sandwich(M, T, cs::AbstractVector, Σx, ::Val{dg}, ::Val{nx}) where {dg, nx}
    function rows(i)
        _, Gx, Gθ = cd_pair(M, cs, i)
        return _chart_rows(T, Gθ, Val(dg)), SMatrix{dg, nx, Float64}(Tuple(Gx))
    end
    Σx === nothing && return implicit_covariance(rows, eachindex(cs))
    return implicit_covariance(eachindex(cs)) do i
        A, Bi = rows(i)
        (A, Bi, SMatrix{nx, nx, Float64}(Σx[i]))
    end
end

# The square system: J = −A⁻¹·B, Σ̃ = J·Σx·Jᵀ (the whitening cancels).
function implicit_covariance(A::AbstractMatrix, B::AbstractMatrix; Σx = nothing)
    D = size(A, 2)
    size(A, 1) == D == size(B, 1) ||
        throw(DimensionMismatch("implicit_covariance: A is $(size(A)), B has $(size(B, 1)) rows — a square system needs D × D and D rows"))
    F = lu(A; check = false)
    issuccess(F) || return nothing
    J = -(F \ B)                                   # D × nₓk solver sensitivity
    Σx === nothing && return Matrix(Symmetric(J * J'))
    nx, rem = divrem(size(B, 2), length(Σx))
    rem == 0 || throw(ArgumentError("implicit_covariance: $(size(B, 2)) columns of B for $(length(Σx)) blocks"))
    Σ = zeros(Float64, D, D)
    @inbounds for i in eachindex(Σx)
        Ji = view(J, :, (nx * (i - 1) + 1):(nx * i))
        Σ .+= Ji * Matrix{Float64}(Σx[i]) * Ji'
    end
    return Matrix(Symmetric(Σ))
end

function implicit_covariance(block, indices::AbstractVector; noise_weighted::Bool = false)
    noise_weighted && return implicit_covariance(i -> _noise_weighted_block(block(i)), indices)
    isempty(indices) && return nothing
    D = size(block_rows(first(block(first(indices)))), 2)
    N = zeros(Float64, D, D)                        # AᵀW₀A
    S = zeros(Float64, D, D)                        # AᵀW₀·BΣxBᵀ·W₀A
    anyΣ = false
    for i in indices
        t = block(i)
        A = block_rows(t[1])
        Bi = block_rows(t[2])
        L⁻¹ = whitening_factor(Bi)                  # W₀ᵢ = L⁻ᵀL⁻¹
        WA = L⁻¹ * A
        P = WA' * WA
        N .+= P
        if length(t) >= 3
            anyΣ = true
            Σg = L⁻¹ * Bi * t[3] * Bi' * L⁻¹'
            S .+= WA' * Σg * WA
        else
            S .+= P                                 # unit isotropic: L⁻¹·BBᵀ·L⁻ᵀ = I (or 0, degenerate)
        end
    end
    C = cholesky(Symmetric(N); check = false)
    issuccess(C) || return nothing
    Ninv = inv(C)
    return Matrix(Symmetric(anyΣ ? Ninv * S * Ninv : Ninv))
end
# `noise_weighted`: the block's Σxᵢ is folded into its measurement rows,
# Bᵢ·chol(Σxᵢ).L, so W₀ᵢ = (BᵢΣxᵢBᵢᵀ)⁻¹ and the sandwich collapses to (AᵀW₀A)⁻¹.
# A Σxᵢ with an exactly known coordinate — an `exact` endpoint's zero block (the
# measurement-noise rule) — has no Cholesky factor; its zero rows and columns
# take a zero column of the factor (`_psd_factor`), still L·Lᵀ = Σxᵢ.
@inline _noise_weighted_block(t::Tuple) =
    length(t) >= 3 ? (t[1], block_rows(t[2]) * _noise_factor(t[3])) : t
_noise_factor(Σ) = any(i -> iszero(Σ[i, i]), axes(Σ, 1)) ? _psd_factor(Σ) : cholesky(Symmetric(Σ)).L
# The lower Cholesky factor of a PSD matrix whose singular directions are whole
# zero rows/columns: a zero pivot gives a zero column.
function _psd_factor(Σ::StaticMatrix{n, n, T}) where {n, T}
    L = zero(MMatrix{n, n, float(T)})
    for j in 1:n
        d = Σ[j, j] - sum(L[j, k]^2 for k in 1:(j - 1); init = zero(float(T)))
        d > 0 || continue
        L[j, j] = sqrt(d)
        for i in (j + 1):n
            L[i, j] = (Σ[i, j] - sum(L[i, k] * L[j, k] for k in 1:(j - 1); init = zero(float(T)))) / L[j, j]
        end
    end
    return SMatrix{n, n, float(T)}(L)
end
_psd_factor(Σ::AbstractMatrix) = Matrix(_psd_factor(SMatrix{size(Σ, 1), size(Σ, 1)}(Σ)))
block_rows(r::StaticVector{n, T}) where {n, T} = SMatrix{1, n, T}(Tuple(r))
block_rows(M::AbstractMatrix) = M
# A constraint's gradient as its one row: static for a static differential.
_one_row(r::StaticVector) = block_rows(r)
_one_row(r::AbstractVector) = permutedims(r)

"""
    local_coordinates(ch::Manifold, Mtarget; iters=25, tol=1e-13)
        -> SVector{tangent_size(ch)}

Inverse retraction ("manifold log"): the local coordinates `v` such that
`retract(ch, v) ∝ Mtarget` (projective alignment — scale AND sign of the
target are matched to the manifold family at each Gauss–Newton step, so any
solver normalization convention may be supplied). This is the harness
primitive for expressing an estimate IN THE MANIFOLD RECENTERED AT TRUTH.

Gauge-deficient manifolds (e.g. `EssentialManifold6`, near-essential
`FundMatManifold7` anchors) resolve the step through `pinv`, projecting the
gauge direction out — the returned coordinates are the minimum-norm
representative. Converges quadratically for targets in the manifold's injectivity
neighborhood (MC-perturbation scale); for far targets check the residual.
"""
local_coordinates(
    ch::Manifold, Mtarget::AbstractMatrix;
    iters::Int = 25, tol::Float64 = 1.0e-13
) =
    _manifold_coordinates(ch, Mtarget, Val(tangent_size(ch)); iters = iters, tol = tol)

function _manifold_coordinates(
        ch::Manifold, Mtarget::AbstractMatrix, ::Val{D};
        iters::Int, tol::Float64
    ) where {D}
    mt = SVector{9, Float64}(Tuple(SMatrix{3, 3, Float64, 9}(Mtarget)))
    mt = mt / norm(mt)
    v = zero(SVector{D, Float64})
    s = dot(SVector{9, Float64}(Tuple(retract(ch, v))), mt)  # scale/sign init
    # JOINT Gauss–Newton over (v, s) on r(v, s) = s·mt − retract(v):
    # solving for the projective scale jointly is what drives r to 0 exactly —
    # aligning s by projection alone stalls with a residual along the manifold
    # complement (e.g. the trace direction of the sl(3) manifold) and the
    # returned v would not satisfy the model constraints.
    for _ in 1:iters
        Mv = SVector{9, Float64}(Tuple(retract(ch, v)))
        r = s * mt - Mv
        Tv = retract_jacobian(ch, v)
        Δ = pinv(hcat(-Tv, mt)) * (-r)   # ∂r/∂(v,s) = [−Tv  mt]
        v = v + Δ[SOneTo(D)]
        s += Δ[D + 1]
        dot(r, r) <= tol^2 * (s^2 + 1) && break
    end
    return v
end
