# =============================================================================
# RANSAC Rotation Problem — a pure rotation between two calibrated views
# =============================================================================
#
# `RotationProblem` is the `MinimalSolveProblem{RotMatrix}` alias: the rotation
# `R` with `f₂ ∥ R f₁` between the unit bearings of two views of a camera that
# only rotated (a panorama sweep, a rotating rig) — the INFINITE HOMOGRAPHY of
# a calibrated camera, on rays rather than on the z = 1 chart, so the whole
# field of a fisheye takes part (past 90°). Its construction, minimal solve
# (VGC `Rotation2pt`: `kabsch` on two pairs), sampler and RANSAC wiring are the
# family's (cspond.jl), patterned on the essential problem: the data are unit
# bearings from the cameras' own back-projection carrying their per-view
# back-projection Jacobians as the `M₁`/`M₂` attribute columns, which put every
# residual and threshold in PIXELS.
#
# ONE Sampson machinery. The constraint is the homography's transfer form on
# bearings — VGC's `constraint_with_jacobians(TransferForm(H), f₁, f₂; M₁, M₂)`
# with `H = R`: the transferred ray in `f₂`'s tangent plane, `G_x` the 2×4
# Jacobian of the joint observation (both images' pixels): the residual is the
# Sampson distance of that observation to the constraint variety at the pixels'
# noise Σ_x = blockdiag(Σ₁, Σ₂) — `I` for plain pixels (Σ_g = P(R M₁M₁ᵀRᵀ + M₂M₂ᵀ)Pᵀ),
# the detections' own covariances for `Uncertain` pixels (the `Σ₁`/`Σ₂` columns,
# `measurement_covariance`), so every route is Mahalanobis. Scoring
# whitens it through VGC's Sampson layer (the default `whitened_residual`, behind a
# hemisphere gate), and the chart LM and the lifts descend the same residual
# through the Sampson-constraint problems' one chart-rows method (sampson_lm.jl:
# VGC's batch rows, the whitening re-evaluated at every iterate and its
# derivative carried). Nothing here restates g/‖∇g‖.
#
# The chart is VGC's `SO3Manifold` (the rotation member of the rigid chart
# family), the left exp perturbation R(ω) = exp(ω)·R.
# The refit is the family's chart LM (`refine!` → `_refine!(::NonlinLsq, …)`);
# the HQ/GNC lifts (LocalOptimizationDoneRight) read the same rows through the
# generic provider (manifold_providers.jl).
# =============================================================================


codimension(::Type{<:MinimalSolveProblem{<:RotMatrix}}) = 2    # f₂ × R f₁ = 0: two equations in f₂'s tangent plane
chart_dof(::Type{<:RotationProblem}) = 3

"""
    RotationProblem(camera1, camera2, pixel_cs; kw...)

A `RotationProblem` from PIXEL correspondences and the cameras of the two views
(the same camera twice for a rotating camera, `RotationProblem(cam, cam, cs)`):
each view back-projected with its own camera through
`backproject_with_jacobian`, the Jacobians landing with their endpoints by
construction (as [`EssentialProblem`](@ref)`(camera1, camera2, pixel_cs)`). The
attribute columns of `pixel_cs` (the `:score` prior) travel along.

The pixels may be VisualGeometryToolkitCore `Uncertain` (or `exact` on one side):
each detection's 2×2 covariance becomes the problem's `Σ₁`/`Σ₂` column (Core's
`noise_columns`), and the residual of scoring, the local optimization's chart LM
and `solver_covariance` is whitened by `blockdiag(Σ₁ᵢ, Σ₂ᵢ)` through Core's
Sampson layer: a Mahalanobis residual, so a scorer's threshold and scale are in
whitened units. Plain pixels are unit isotropic noise, as before.
"""
function RotationProblem(camera1, camera2, pixel_cs; kw...)
    s, t = sources(pixel_cs), targets(pixel_cs)
    fm₁ = map(u -> VgtkCore.backproject_with_jacobian(camera1, VgtkCore.geom(u)), s)
    fm₂ = map(u -> VgtkCore.backproject_with_jacobian(camera2, VgtkCore.geom(u)), t)
    cols = merge(VgtkCore.attrs_of(pixel_cs), VgtkCore.noise_columns(s, t), (M₁ = last.(fm₁), M₂ = last.(fm₂)))
    return RotationProblem(VgtkCore.Csponds(first.(fm₁), first.(fm₂), cols); kw...)
end

"""
    RotationProblem(bearings::Csponds; sampler = nothing, coincident_tol = COINCIDENT_PX)

A `RotationProblem` from UNIT-BEARING correspondences `f₁ => f₂` (3-vectors from
the cameras' back-projection) carrying their 3×2 back-projection Jacobians as
the attribute columns `M₁`/`M₂` (`withattrs(cs; M₁, M₂)`), which put the
residual, the threshold and the sample gate's `coincident_tol` in PIXELS — so,
as for the essential problem, there is no default for them. A `:score` column
is the PROSAC prior. The detections' observation noise, when not unit isotropic,
is two more columns, `Σ₁`/`Σ₂` (2×2 covariances in PIXELS, the space `M₁`/`M₂`
map from: VisualGeometryToolkitCore's `noise_columns` of the `Uncertain` pixels,
as the pixel constructor builds them). Unit norm and finiteness are checked once, here.
"""
function RotationProblem(bearings::VgtkCore.Csponds; sampler::Union{Nothing, AbstractSampler} = nothing,
                         coincident_tol::Real = VgtkCore.COINCIDENT_PX)
    (eltype(sources(bearings)) <: StaticArrays.StaticVector{3} && eltype(targets(bearings)) <: StaticArrays.StaticVector{3}) ||
        throw(ArgumentError("RotationProblem takes unit BEARINGS (3-vectors), got $(eltype(sources(bearings))) => " *
                            "$(eltype(targets(bearings))); use `backproject_with_jacobian(camera, u)`"))
    _assert_unit_bearings(sources(bearings), "source", "RotationProblem")
    _assert_unit_bearings(targets(bearings), "target", "RotationProblem")
    (VgtkCore.hasattr(bearings, :M₁) && VgtkCore.hasattr(bearings, :M₂)) || throw(ArgumentError(
        "RotationProblem requires the back-projection Jacobians as attribute columns: `withattrs(cs; M₁ = ..., M₂ = ...)`, " *
            "from `backproject_with_jacobian(camera, u)` — they put the residual and the threshold in PIXELS"))
    _assert_finite_jacobians(bearings.M₁, "M₁", "RotationProblem")
    _assert_finite_jacobians(bearings.M₂, "M₂", "RotationProblem")
    smplr = _problem_sampler(bearings, 2, sampler)
    cs = _check_noise_support(RotMatrix, _canonicalize_pairs(Float64, bearings))
    return MinimalSolveProblem{RotMatrix{3, Float64, 9}, 2}(cs, nothing, cs, nothing, VgtkCore.Rotation2pt(), nothing, smplr;
                                                            coincident_tol)
end

# A Pair vector cannot carry the Jacobian columns: refused as the essential problem refuses it.
RotationProblem(::AbstractVector{<:Pair{<:StaticArrays.StaticVector{3}, <:StaticArrays.StaticVector{3}}}; kw...) =
    throw(ArgumentError("RotationProblem needs the back-projection Jacobians, which a vector of Pairs cannot carry: " *
                        "`withattrs(Csponds(f₁, f₂); M₁, M₂)`, from `backproject_with_jacobian(camera, u)`"))

# The sample gate: the bearing coincidence gate of VGC (a sample's two bearings
# the same detection in either view, the angle converted to pixels through M —
# keyed on the essential matrix's type, whose bearing form it is), at the
# problem's pixel tolerance. A parallel or antiparallel pair is the minimal
# solver's own refusal (`Rotation2pt`: count 0).
test_sample(p::RotationProblem, idx::AbstractVector{Int}) =
    test_sample(EssentialMat, p.cs, idx; tol_px = p.coincident_tol)

# The infinite homography of a rotation: the transfer form's matrix.
@inline _transfer_homography(R) = HomographyMat(Tuple(SMatrix{3, 3}(R)))   # eltype-generic (ForwardDiff through the chart)

# The scoring constraint — VGC's bearing transfer form at H = R (the transfer
# form's degeneracy is its Cholesky rank verdict: the exact-zero rounding default).
@inline function constraint_with_jacobians(p::RotationProblem, R::RotMatrix, i::Int)
    @inbounds return VgtkCore.constraint_with_jacobians(
        VgtkCore.TransferForm(_transfer_homography(R)), sources(p.cs)[i], targets(p.cs)[i]; M₁ = p.cs.M₁[i], M₂ = p.cs.M₂[i])
end

# The whitened residual: VGC's default (the Sampson residual of the constraint
# above, whitened, in pixels) behind the hemisphere gate — the transferred ray
# must be in f₂'s hemisphere, since an antipodal pair also has a small tangent
# residual. A gated point's residual is infinite in both components, so its
# norm, the Mahalanobis distance (VGC's default), is the gate's `Inf`.
@inline function whitened_residual(p::RotationProblem, R::RotMatrix, i::Int)
    @inbounds dot(R * sources(p.cs)[i], targets(p.cs)[i]) > 0 || return SVector(Inf, Inf)
    return @invoke whitened_residual(p::AbstractConstraintProblem, R, i::Int)
end

# The model covariance: the one constraint sandwich (covariance.jl), on the rows
# above — the implicit-function covariance of the whitened-constraint estimator
# the problem's local optimization runs (its Sampson chart LM over the inliers),
# in pixels of unit noise. In G_θ's coordinates, vec(R) (9×9, rank 3), as every
# MinimalSolveProblem's, so predictive scoring contracts it with the same rows;
# its so(3) chart form (left exp, `SO3Manifold`) is `Jᵀ·Σ·J/4`, `J` the chart's
# `retract_jacobian` at R (orthogonal columns of norm √2) — what
# `fit_rotation_averaging` reads from an `Uncertain` edge.
provides_solver_covariance(::Type{<:RotationProblem}) = true

# A rotation covariance in the so(3) chart of `SO3Manifold(R)` (left exp): a 3×3
# is already there; a 9×9 over vec(R) (solver_covariance's, the chart's ambient
# coordinates) is pulled into the chart by Core's `transform_covariance` from the
# ambient chart: `J⁺·Σ·J⁺ᵀ`, `J` the chart's `retract_jacobian` (orthogonal
# columns of norm √2, so `J⁺ = Jᵀ/2` up to rounding).
_rotation_chart_covariance(Σ::SMatrix{3, 3, Float64}, _R) = Σ
function _rotation_chart_covariance(Σ::SMatrix{9, 9, Float64}, R)
    ch = VgtkCore.SO3Manifold(R)
    return SMatrix{3, 3, Float64, 9}(VgtkCore.transform_covariance(EuclideanManifold(ch), ch, Σ))
end
_rotation_chart_covariance(Σ, _R) = throw(ArgumentError(
    "a rotation covariance is 3×3 (the so(3) chart) or 9×9 (vec(R), solver_covariance's), got $(size(Σ))"))

# The relative rotation of a pairwise fit as an averaging edge: its rotation with
# the covariance of the problem's own estimator, the one constraint sandwich
# (`solver_covariance`, covariance.jl: the implicit-function covariance of the
# whitened-constraint LM the problem's local optimization runs over the inliers)
# at unit noise in whitened units, read into the so(3) chart of `SO3Manifold`
# (left exp, R → exp(δ)·R) and scaled by the fit's noise scale squared: an
# absolute covariance, rad². The chart is fit_rotation_averaging's edge chart.

"""
    uncertain_rotation(problem::RotationProblem, result::RansacEstimate; scale = result) -> Uncertain{RotMatrix{3, Float64}}

The relative rotation of a pairwise rotation fit (`find_rotation`, `refine_rotation` on `problem`) with its covariance, the edge
[`fit_rotation_averaging`](@ref) reads: `Uncertain(R, CovMat(Σ_R))` with
`R = estimate(result)` and

    Σ_R = noise_scale² · Jᵀ·solver_covariance(problem, R, inliers)·J / 4

in the so(3) chart of `R` (left exp, `R → exp(δ)·R`; `J` the chart's
`retract_jacobian`, whose columns are orthogonal of norm √2), radians². The
covariance is the implicit-function covariance of the problem's own
whitened-constraint estimator over the result's inliers (`solver_covariance`),
so it carries every detection's covariance when the pixels are `Uncertain`
(`RotationProblem(camera1, camera2, pixel_cs)`), and is in whitened units:
`scale` is the variance factor of those units — the scale input of the
robust-estimation contract: by default the result's own `noise_scale`, the
scale its scorer estimated on the consensus set (`ScaleEstimated`, in whitened
units: ≈ 1 when the detections' covariances are right); or `scale = σ`, or
another upstream result. A result that recorded no scale (`ScaleNone`) is
refused. An edge covariance is absolute, so the averaging takes no further scale.

Pairs that share a detection across edges make their edges' errors correlated;
the averaging treats the edges as independent, the first-order approximation.

An `ArgumentError` for a result with no estimate, or a singular sandwich (too
few inliers).
"""
function uncertain_rotation(problem::RotationProblem, result::RansacEstimate; scale = result)
    R = estimate(result)
    R === nothing && throw(ArgumentError("uncertain_rotation: the result has no estimate"))
    σ = VgtkCore.scale_input(scale, nothing).scale
    Rm = RotMatrix{3, Float64, 9}(R)
    Σ9 = solver_covariance(problem, Rm, findall(VgtkCore.inlier_mask(result)))
    Σ9 === nothing && throw(ArgumentError("uncertain_rotation: the inliers do not determine the rotation (a singular covariance)"))
    Σ = _rotation_chart_covariance(SMatrix{9, 9, Float64, 81}(Σ9), SMatrix{3, 3, Float64, 9}(Rm))
    return VgtkCore.Uncertain(Rm, VgtkCore.CovMat(Float64(σ)^2 * Σ))::_UncertainRotation
end
# The edge's concrete type (Core's CovMat constructor is not inferred through): the assertion makes the call inferable.
const _UncertainRotation = typeof(VgtkCore.Uncertain(one(RotMatrix{3, Float64, 9}), VgtkCore.CovMat(SMatrix{3, 3, Float64, 9}(LinearAlgebra.I))))

# The sub-detector half-width refusal is the bearing problems' (essential.jl
# `_check_scoring` on `_BearingProblem`), in the same pixel unit.

# =============================================================================
# The chart — VGC's SO3Manifold — and its rows
# =============================================================================

# The chart is the family default, `minimal_manifold(R)` — VGC's `SO3Manifold`,
# R(ω) = exp(ω)·R — and its model read-out `model(ch)` (`_chart`, `_model_at`).
# The rows are the Sampson-constraint problems' one method (sampson_lm.jl): the
# constraint's model at chart coordinates `a` is the bearing transfer form of
# the rotation, whose `G_θ` is over vec(H) = vec(R), so the chart's
# `retract_jacobian` (∂vec(R)/∂ω) is its directions; the transfer whitening is
# re-evaluated at every iterate and its derivative carried, so the LM descends
# exactly the pixel residual scoring computes.
@inline _rotation_at(ch, a) = _transfer_homography(retract(ch, a))
_constraint_model(::RotationProblem, ch, a) = VgtkCore.TransferForm(_rotation_at(ch, a))
