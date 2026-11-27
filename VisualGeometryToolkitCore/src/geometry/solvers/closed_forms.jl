# =============================================================================
# Core's closed-form algorithms — one deterministic step each (ClosedFormKind)
# =============================================================================
#
#     solve(EstimationProblem(M, data[, p]), Algorithm())  →  FitResult
#
# Each algorithm is a typed value; its numerics are the private kernel beside
# its model (homography_solvers.jl, fundmat.jl, rotation_fits.jl,
# total_least_squares.jl, algebraic_ellipse.jl, point_from_lines.jl), unchanged. A closed form's result:
# `u` the model, `retcode` Success and `stop_rule` ClosedForm, no iterations,
# `rms` NaN (an algebraic step reports no geometric residual), `n` the data it
# used. A configuration with no solution (a degenerate DLT, a point set that
# admits no ellipse) gives `u = nothing` with `(InitialFailure, NoRule)` and
# `n = 0`; malformed input (too few data, a negative weight) is an
# `ArgumentError`, as before. The one exception to `u = nothing`: Triangulation
# of parallel rays keeps its NaN point as `u` (a concrete `Point3`, the type and
# the golden mapping of `u` unchanged), recorded `(InitialFailure, NoRule)`, `n = 0`.
# =============================================================================

"""
    closed_form_result(u, n; metadata = nothing) -> FitResult
    closed_form_result(U::Type, u, n; metadata = nothing) -> FitResult{U}

THE result of a closed form ([`ClosedFormKind`](@ref)), what its `__solve` returns: `u` over `n` data,
`(ReturnCode.Success, ClosedForm)`, no iterations, `rms` NaN. The second form is an algorithm's that can be
degenerate, `U` the result's model type `Union{M, Nothing}`: `u === nothing` gives `(ReturnCode.InitialFailure,
NoRule)` and `n = 0`.
"""
@inline closed_form_result(u, n::Integer; metadata = nothing) =
    FitResult(u, ReturnCode.Success, ClosedForm, SolveStats(), NaN, n, metadata)
@inline closed_form_result(::Type{U}, u, n::Integer; metadata = nothing) where {U} =
    u === nothing ? FitResult{U}(nothing, ReturnCode.InitialFailure, NoRule, SolveStats(), NaN, 0, metadata) :
    FitResult{U}(u, ReturnCode.Success, ClosedForm, SolveStats(), NaN, n, metadata)

const _PlanePairs{T} = AbstractVector{<:Pair{<:StaticVector{2, T}, <:StaticVector{2, T}}}

# -----------------------------------------------------------------------------
# DLT — the Direct Linear Transform with Hartley normalization
# -----------------------------------------------------------------------------

"""
    DLT(backend = NativeSolver())

The Direct Linear Transform, a [`ClosedFormKind`](@ref) — the default algorithm of a homography's and a
fundamental matrix's [`EstimationProblem`](@ref) from plane correspondences:

- `solve(EstimationProblem(HomographyMat, cs), DLT())`: N ≥ 4 plane correspondences, Hartley normalization;
- `solve(EstimationProblem(FundMat, cs), DLT())`: N ≥ 8, Hartley normalization and the rank-2 constraint;
- `solve(EstimationProblem(EssentialMat, bearings), DLT(PoseLibSolver()))`: N ≥ 8 bearing pairs, the 8-point
  linear method ([`essential_8pt`](@ref)) — only PoseLib's backend exists (`using PoseLib`), so an essential
  matrix declares no default and any other backend (`DLT()`, the native one) is refused (`ArgumentError`).

Per-correspondence weights are the container's `weights` column ([`withattrs`](@ref)): each multiplies its rows'
squared algebraic residual and the normalization. Use a subset of the data (`cs[mask]`) to solve on a selection.
The result's `u` is `nothing`, record `(InitialFailure, NoRule)`, on a degenerate or non-finite configuration; `n`
is the number of correspondences, `rms` NaN. The inner-loop forms on a caller's buffer and normalization are
[`homography_dlt!`](@ref) and [`fundamental_dlt!`](@ref).
"""
struct DLT{B <: AbstractSolverBackend} <: AbstractEstimationAlgorithm
    backend::B
end
DLT() = DLT(NativeSolver())
estimator_kind(::DLT) = ClosedFormKind()
default_algorithm(::EstimationProblem{<:Union{HomographyMat, FundMat}, <:_PlanePairs}) = DLT()

function __solve(prob::EstimationProblem{<:HomographyMat, <:_PlanePairs{T}, NullParameters}, ::DLT{NativeSolver}) where {T}
    cs = prob.f
    n = length(cs)
    H = homography_dlt!(Matrix{T}(undef, 2n, 9), CorrespondenceNormalization(HomographyMat, cs), cs)
    return closed_form_result(Union{HomographyMat{T}, Nothing}, H, n)
end

function __solve(prob::EstimationProblem{<:FundMat, <:_PlanePairs{T}, NullParameters}, ::DLT{NativeSolver}) where {T}
    cs = prob.f
    n = length(cs)
    F = fundamental_dlt!(Matrix{T}(undef, n, 9), CorrespondenceNormalization(FundMat, cs), cs)
    return closed_form_result(Union{FundMat{T}, Nothing}, F, n)
end

# The essential matrix: the backend's 8-point kernel (`essential_8pt`, a minimal-solver backend contract
# generic: PoseLib implements it). There is no native one.
function __solve(prob::EstimationProblem{<:EssentialMat, <:Any, NullParameters}, alg::DLT)
    E = essential_8pt(alg.backend, prob.f)
    return closed_form_result(Union{EssentialMat{Float64}, Nothing}, E, length(prob.f))
end

# -----------------------------------------------------------------------------
# Rotations — Kabsch/Wahba, the chordal mean, spectral rotation averaging
# -----------------------------------------------------------------------------

"""
    Kabsch()

A [`ClosedFormKind`](@ref): `solve(EstimationProblem(RotMatrix{3}, cs), Kabsch())`, `u` the rotation, `n` the pairs.

Kabsch/Wahba is the rotation minimizing `Σᵢ ‖bᵢ − R·aᵢ‖²` over the vector
pairs `cs` (`aᵢ => bᵢ`, `StaticVector{3}` each side; any
[`sources`](@ref)/[`targets`](@ref) container, a [`Correspondences`](@ref)
preferred): one closed-form step, `Rotations.nearest_rotation` of the
cross-covariance `Σ bᵢaᵢᵀ` (an SVD with the determinant-sign correction).
A `weights` attribute column (`withattrs(cs; weights = w)`, `data-arguments.md`)
weighs the pairs, `Σᵢ wᵢ‖bᵢ − R·aᵢ‖²` and the cross-covariance `Σ wᵢbᵢaᵢᵀ`
(Wahba's problem): fixed, finite, non-negative weights, one per pair (an
`ArgumentError` otherwise); a zero weight drops its pair. The vectors are
used as given, NOT centred: centre them at the call site when aligning point
clouds rather than directions.

Physical pairs (lengths on both sides) are stripped in metres at entry; the
rotation is unit-free, so it comes back the same either way.

Throws `ArgumentError` on fewer than 2 pairs, or fewer than 2 of positive
weight (a single pair leaves a one-parameter family of solutions).

The default algorithm of a rotation from vector pairs.
"""
struct Kabsch <: AbstractEstimationAlgorithm end
estimator_kind(::Kabsch) = ClosedFormKind()
const _VectorPairs3 = AbstractVector{<:Pair{<:StaticVector{3}, <:StaticVector{3}}}
default_algorithm(::EstimationProblem{<:RotMatrix{3}, <:_VectorPairs3}) = Kabsch()
__solve(prob::EstimationProblem{<:RotMatrix{3}, <:_VectorPairs3, NullParameters}, ::Kabsch) = closed_form_result(_kabsch(prob.f), length(prob.f))

"""
    ChordalMean()

A [`ClosedFormKind`](@ref): `solve(EstimationProblem(RotMatrix{3}, Rs), ChordalMean())`, `u` the mean rotation, `n` the rotations.

Chordal L2 mean of a set of rotations: the rotation minimizing
`Σᵢ ‖R − Rᵢ‖²_F`, i.e. `Rotations.nearest_rotation` of the elementwise sum.
Suited to averaging rotations within a moderate spread (the chordal mean and
the geodesic mean agree to second order); it is NOT a robust estimator.

Throws `ArgumentError` on an empty collection.

The default algorithm of a rotation from rotations.
"""
struct ChordalMean <: AbstractEstimationAlgorithm end
estimator_kind(::ChordalMean) = ClosedFormKind()
const _Rotations3 = AbstractVector{<:Union{Rotation{3}, StaticMatrix{3, 3}}}
default_algorithm(::EstimationProblem{<:RotMatrix{3}, <:_Rotations3}) = ChordalMean()
__solve(prob::EstimationProblem{<:RotMatrix{3}, <:_Rotations3, NullParameters}, ::ChordalMean) = closed_form_result(_chordal_mean(prob.f), length(prob.f))

"""
    AbsoluteRotations

The model TAG of rotation averaging: the absolute rotations of a set of frames, a `Dict{K, RotMatrix{3, Float64}}`
keyed by frame, estimated from their relative rotations — `EstimationProblem(AbsoluteRotations, relative)`. A tag
names what is estimated where no single model type holds it; it is never instantiated.
"""
abstract type AbsoluteRotations end

"""
    SpectralAveraging()

A [`ClosedFormKind`](@ref): `solve(EstimationProblem(AbsoluteRotations, relative), SpectralAveraging())`, `u` the `Dict` of absolute rotations keyed by frame, `n` the pairs.

Spectral chordal rotation averaging (Singer 2011; Arie-Nachimson et al. 2012):
the absolute rotations of a set of frames from their relative rotations,
`relative` an `AbstractDict` keyed by frame pairs `(i, j)` whose value `R_ij`
takes frame `i`'s directions into frame `j`'s (`f_j = R_ij·f_i`; with
`d = R_k·f_k` a direction in the common frame, `R_i = R_j·R_ij`). The `3n × 3n`
symmetric block matrix with blocks `w_ij·R_ij` at `(j, i)` and its transpose at
`(i, j)`, identity diagonal blocks, normalized by the frames' weighted degrees
(`D^{-1/2} A D^{-1/2}`), has the stacked `R_kᵀ` in its three leading
eigenvectors; each `3 × 3` block of them is projected onto SO(3)
(`Rotations.nearest_rotation`, the eigenvectors' sign chosen so most blocks are
proper), and the result is gauged so the first of the sorted frames is the
identity (the common frame is that frame's). Symmetric in the frames, exact on
consistent relative rotations, one eigendecomposition — the seed of a robust
rotation averaging, not a robust estimator. A pair carries its non-negative
weight as `Attributed(R_ij, (; weight))`, read with [`weight_of`](@ref); a bare
`R_ij` weighs 1. A zero-weight pair contributes nothing. The pairs of positive weight must
connect every frame: the caller checks it on its graph (VisualGeometryToolkitEstimation's
`RotationAveraging` refuses a split one through its `ViewGraph`); on a split graph the
leading eigenspace is degenerate and the rotations returned are meaningless (a chain with one
zero-weight bridge came back 180° off).

The default algorithm of [`AbsoluteRotations`](@ref).
"""
struct SpectralAveraging <: AbstractEstimationAlgorithm end
estimator_kind(::SpectralAveraging) = ClosedFormKind()
default_algorithm(::EstimationProblem{AbsoluteRotations, <:AbstractDict}) = SpectralAveraging()
__solve(prob::EstimationProblem{AbsoluteRotations, <:AbstractDict, NullParameters}, ::SpectralAveraging) =
    closed_form_result(_spectral_rotation_averaging(prob.f), length(prob.f))

# -----------------------------------------------------------------------------
# Lines, ellipses, points
# -----------------------------------------------------------------------------

"""
    TotalLeastSquares()

A [`ClosedFormKind`](@ref): `solve(EstimationProblem(Line2D, points), TotalLeastSquares())`, FIXED per-point weights the points' `weights`
attribute column (`withattrs(points; weights = w)`, as for [`Kabsch`](@ref), [`NearestPoint`](@ref) and [`DLT`](@ref));
`u` the `Uncertain{Line2D}`, `n` the points.

Total least squares (orthogonal distance regression) line through `points`
(2-vectors): the centroid, and the scatter matrix's smallest-eigenvalue
eigenvector as the normal — one closed-form step. The line is in Hesse form
`n⊤p + d = 0`, `n = (cos φ, sin φ)`, and carries the covariance of
`θ = (φ, d)`: `s² (JᵀWJ)⁻¹`, with `s² = Σwᵢrᵢ²/(n₊−2)` the residual variance,
`n₊` the number of points with positive weight and `J` the residual Jacobian —
calibrated against Monte Carlo.

`weights` are FIXED non-negative per-point weights (`wᵢ = 1/σᵢ²` for known
isotropic variances): the weighted centroid, the weighted scatter, and a zero
weight drops the point. Without them every point weighs 1. Throws
`ArgumentError` for fewer than 3 points of positive weight, a negative or
non-finite weight, or a `weights` whose length differs from `points`'.

For points carrying their own covariances, VisualGeometryToolkitEstimation's
heteroscedastic line fit (an iterative weighted fit) applies.

The default algorithm of a line from points.
"""
struct TotalLeastSquares <: AbstractEstimationAlgorithm end
estimator_kind(::TotalLeastSquares) = ClosedFormKind()
default_algorithm(::EstimationProblem{<:Line2D, <:AbstractVector{<:StaticVector{2}}}) = TotalLeastSquares()
__solve(prob::EstimationProblem{<:Line2D, <:AbstractVector{<:StaticVector{2}}, NullParameters}, ::TotalLeastSquares) =
    closed_form_result(_tls_points(prob.f, weights_column(prob.f)), length(prob.f))
# the points' `weights` column, when they carry one (the one reader, `weights_column`), as the kernel's weights
_tls_points(points, ::Nothing) = _tls_line(points)
_tls_points(points, w) = _tls_line(points, w)

"""
    DirectEllipse()

A [`ClosedFormKind`](@ref): `solve(EstimationProblem(Ellipse, points), DirectEllipse())`, `u` the ellipse — `nothing`, recorded `(InitialFailure, NoRule)` with `n = 0`, when the points admit none — `n` the points.

The ellipse through the 2-D `points` minimizing the summed squared algebraic
residual `Σᵢ (xᵢᵀ C xᵢ)²` under the ellipse constraint `4AC − B² = 1`
(Fitzgibbon, Pilu & Fisher 1999, in the numerically stable form of Halir &
Flusser 1998), computed in Hartley-normalized coordinates: one closed-form step
whose solution is always an ellipse. No ellipse when the points admit none (too
few in general position, e.g. collinear). Throws `ArgumentError` for fewer than
5 points.

A start value: the algebraic residual weighs points by where they sit on the
curve. VisualGeometryToolkitEstimation's ellipse refine (`LevenbergMarquardt` on an `EllipseProblem`) minimizes the
orthogonal distance from it.

The default algorithm of an ellipse from points.
"""
struct DirectEllipse <: AbstractEstimationAlgorithm end
estimator_kind(::DirectEllipse) = ClosedFormKind()
default_algorithm(::EstimationProblem{<:Ellipse, <:AbstractVector{<:StaticVector{2}}}) = DirectEllipse()
function __solve(prob::EstimationProblem{<:Ellipse, <:AbstractVector{<:StaticVector{2}}, NullParameters}, ::DirectEllipse)
    pts = prob.f
    E = _direct_ellipse(pts)
    return closed_form_result(Union{Ellipse{float(eltype(eltype(pts)))}, Nothing}, E, length(pts))
end

"""
    NearestPoint()

A [`ClosedFormKind`](@ref): `solve(EstimationProblem(Point, lines), NearestPoint())`, `u` the point, `n` the lines.

The point `x` nearest to the `lines` in the least-squares sense: the minimizer
of `Σᵢ wᵢ ‖(I − dᵢdᵢᵀ)(x − oᵢ)‖²`, the weighted sum of squared distances from
`x` to each line, by one closed-form solve of the normal equations
`(Σ wᵢPᵢ) x = Σ wᵢPᵢoᵢ`, `Pᵢ = I − dᵢdᵢᵀ` (the projector onto the line's normal
space).

`lines` is a vector of `origin => direction` pairs (a [`Correspondences`](@ref) by
`correspondences(origins, directions)`, or a plain vector of `Pair`s): `oᵢ` any point
on the line, `dᵢ` its direction, normalized inside (any nonzero length).
`N`-dimensional: 3-D lines and 2-D lines alike. Per-line weights are the
`weights` attribute column (`withattrs(lines; weights = w)`): FIXED,
non-negative, a zero weight drops the line; without it every line weighs 1.

Units: origins that are lengths (`SVector(1.0, 2.0, 3.0) .* mm`) are stripped
in metres at entry and the point comes back in metres; plain origins give a
plain point in the caller's own unit. The solve runs about the weighted
centroid of the origins, so translating every line by `t` translates the point
by `t` to rounding.

Throws `ArgumentError` for fewer than two lines of positive weight, a zero or
non-finite direction, or lines that do not determine a point — all parallel,
the normal matrix singular to `1e-12` of its largest eigenvalue.

```julia
lines = correspondences([SVector(0.0, 0.0, 0.0), SVector(1.0, 0.0, 0.0)], [SVector(0.0, 0.0, 1.0), SVector(-1.0, 0.0, 1.0)])
solve(EstimationProblem(Point, lines), NearestPoint()).u   # [0.0, 0.0, 1.0]: where the two lines meet
```

The default algorithm of a point from lines.
"""
struct NearestPoint <: AbstractEstimationAlgorithm end
estimator_kind(::NearestPoint) = ClosedFormKind()
const _Lines = AbstractVector{<:Pair{<:StaticVector{N}, <:StaticVector{N}}} where {N}
default_algorithm(::EstimationProblem{<:Point, <:_Lines}) = NearestPoint()
__solve(prob::EstimationProblem{<:Point, <:_Lines, NullParameters}, ::NearestPoint) = closed_form_result(_nearest_point(prob.f), length(prob.f))

"""
    Triangulation()

Two-view triangulation of a pair of UNIT BEARINGS, a [`ClosedFormKind`](@ref):
`solve(EstimationProblem(Point3, f₁ => f₂, pose), Triangulation())`, the context `pose` the plain (unlabelled, unitless) `EuclideanMap{3}` from
view 1 to view 2 (`X₂ = R·X₁ + t`). `u` is the midpoint of the rays' mutual perpendicular in view 1's frame, and
`metadata.depths` the two ray depths `(λ₁, λ₂)` — cheirality on rays is `λ₁ > 0 && λ₂ > 0`. Parallel rays (no
parallax) are degenerate, not exceptional: the point and the depths are NaN and the record is `(InitialFailure,
NoRule)` with `n = 0` — the one closed form whose degenerate `u` is not `nothing` but its NaN point, so `u` stays a
concrete `Point3`. The default algorithm of a point from a
bearing pair and a pose. The inner loop's form is the private kernel beside it.
"""
struct Triangulation <: AbstractEstimationAlgorithm end
estimator_kind(::Triangulation) = ClosedFormKind()
const _BearingPair = Pair{<:StaticVector{3}, <:StaticVector{3}}
default_algorithm(::EstimationProblem{<:Point3, <:_BearingPair, <:PlainEuclideanMap{3}}) = Triangulation()
# The kernel in a compiled copy of its own: inlined into a caller, LLVM fuses its StaticArrays `muladd`s by the caller's
# context, so the same rays triangulated last-bit differently by call site (the golden replay, 2026-11-23); one
# non-inlined copy gives every solve the bits of the retired public triangulation function (a plain call of the kernel).
@noinline _triangulation_kernel(pose, f₁, f₂) = triangulate_bearings(pose, f₁, f₂)
function __solve(prob::EstimationProblem{<:Point3, <:_BearingPair, <:PlainEuclideanMap{3}}, ::Triangulation)
    X, λ₁, λ₂ = _triangulation_kernel(prob.p, first(prob.f), last(prob.f))
    metadata = (; depths = (λ₁, λ₂))
    # parallel rays: the kernel's NaN point, recorded degenerate (the kernel's own test: NaN depths)
    isnan(λ₁) && return FitResult(X, ReturnCode.InitialFailure, NoRule, SolveStats(), NaN, 0, metadata)
    return closed_form_result(X, 1; metadata)
end
