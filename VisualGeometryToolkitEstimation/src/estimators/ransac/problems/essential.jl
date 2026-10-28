# =============================================================================
# RANSAC Essential Problem — the chart rows, the unit guard, the research arms
# =============================================================================
#
# `EssentialProblem` is the `MinimalSolveProblem{EssentialMat}` alias; its
# construction, minimal solve (VGC `Essential5pt` on the unit bearings) and
# sample gate are the family's (cspond.jl). `cs` holds genuine rays — 3-vector
# unit bearings from the camera's own back-projection — carrying their
# per-view back-projection Jacobians as the `M₁`/`M₂` attribute columns.
#
# There is no 2D-gnomonic construction and no mapped one: a map is a MATRIX, so
# the best it can produce is a point on the z = 1 plane, and that plane does not
# exist for a ray at 90°. The frame-specificity that motivated the maps is
# satisfied by construction rather than by conjugation, so the 2026-07-19 bug
# (scoring, residuals and LO conjugating through the maps while `solve` alone
# did not) cannot recur.
#
# The sample gate reads the BEARINGS with no normalization: coincidence is each
# pair's angular separation converted to pixels through `M₁`/`M₂`
# (`angle / σ_max(M)`), tested against `COINCIDENT_PX` in the caller's frame.
# =============================================================================

# =============================================================================
# The chart — VGC's minimal EssentialManifold5 (so(3) ⊕ T_tS², 5 dof, codim 1)
# =============================================================================
#
# E = skew(t)·R, anchored at the first of motion(E)'s candidates with |t| = 1
# (the Sampson rows are blind to the candidate, the scale and the sign): R
# steps by MRP(ω)·R, t along the S² tangent basis at t and renormalized — so E
# stays exactly on the essential manifold with no gauge direction. The rows
# are the TANGENT Sampson of E on bearings, whitened by the per-view
# back-projection Jacobians `M₁`/`M₂` — the SAME implementation `residuals!`
# scores through (VGC's `sampson_rows!` / `sampson_jacobian_rows!` with
# M₁/M₂), with the chart's directions ∂E/∂v from VGC's `retract_directions`.
# The chart-LM refit and the HQ/GNC lifts (LocalOptimizationDoneRight) share
# the chart and the rows (interface.jl `ChartLm`). Until 2026-09-30 the refit
# ran on the 6-dof EssentialManifold6 (a gauge direction τ ∥ t absorbed by the
# LM damping) while the lift ran on this chart; one chart now serves both.
# =============================================================================

chart_dof(::Type{<:EssentialProblem}) = 5
_chart(::EssentialProblem, E::EssentialMat) = EssentialManifold5(E)

# The rows read the problem's columns — sources with `M₁`, targets with `M₂` —
# through the epipolar family's one method (epipolar_lm.jl) with these per-view
# Jacobians, so a swap here is a swap everywhere, and the assembly canary in
# test/ransac_essential_tests.jl builds its state through the same `_lm_state`.
_sampson_columns(p::EssentialProblem) = (; M₁ = p.cs.M₁, M₂ = p.cs.M₂)

# =============================================================================
# The half-width unit guard — one check, every entry
# =============================================================================
#
# E's outlier half-width is in PIXELS since the bearing port (the residual
# is whitened by the back-projection Jacobian); it was in calibrated units
# before, where the default was 0.02. A calibrated value carried across is a
# sub-pixel half-width that scores almost nothing an inlier: measured 7 inliers where
# 112 are true, reported `:converged`. `find_essential`'s keyword alone was
# guarded, so `ransac(p, StudentTUniformScoring(p, 0.02))`, `locally_optimize` and
# `find_essential(...; scoring)` all went through unchecked. The check now
# lives here and the loop's entries all call it (`_check_scoring`, loop.jl).
#
# VisualGeometryToolkitFeatures' pinned worst-case sub-pixel refinement bound: no
# threshold below the detector's own noise floor can mean anything.
const _ESSENTIAL_HALFWIDTH_FLOOR_PX = 0.06

# A "pixel" subtending more than this many radians is not a detector pixel: it
# is a calibrated unit (a unit-focal camera, f = 1), where the half-width was
# never in any other unit and a calibrated value is exactly right. No real
# camera comes near it — a 360° field across 100 px is 0.06 rad/px.
const _CALIBRATED_PIXEL_RAD = 0.1

# The largest angle one pixel subtends anywhere in either view: the largest
# singular value of the back-projection Jacobian (VGC's `sigma_max`, the same
# conversion the E sample gate makes), ≈ 1/f at a pinhole's principal point.
_pixel_angle(p::_BearingProblem) =
    maximum(VgtkCore.sigma_max, Iterators.flatten((p.cs.M₁, p.cs.M₂)); init = 0.0)

_check_scoring(p::_BearingProblem, scoring::AbstractScoring) = _check_halfwidth(p, outlier_halfwidth(scoring))
# A threshold scorer's threshold is in pixels too, under the same floor.
_check_scoring(p::_BearingProblem, scoring::Union{MsacScoring{<:Any, Float64}, RansacScoring{<:Any, Float64}}) =
    _check_halfwidth(p, inlier_threshold(scoring, nothing), "threshold")
_check_halfwidth(::_BearingProblem, ::Nothing, _what = "") = nothing
function _check_halfwidth(p::_BearingProblem, halfwidth::Float64, what::AbstractString = "outlier half-width")
    (halfwidth >= _ESSENTIAL_HALFWIDTH_FLOOR_PX || _pixel_angle(p) > _CALIBRATED_PIXEL_RAD) ||
        throw(ArgumentError(
        "$what $(round(halfwidth; sigdigits = 4)) is below " *
            "$(_ESSENTIAL_HALFWIDTH_FLOOR_PX) px, VisualGeometryToolkitFeatures' worst-case " *
            "sub-pixel refinement bound — no threshold below the detector's own " *
            "noise floor is meaningful. NOTE THE UNIT CHANGED: the essential " *
            "half-width was in CALIBRATED coordinates and is now in PIXELS. The old " *
            "default 0.02 calibrated is ~24 px at f = 1200 and ~16 px at f = 800; " *
            "0.02 px is nothing. If you carried a calibrated value across, multiply " *
            "it by your focal length."
    ))
    return nothing
end


# =============================================================================
# Scoring residual — TANGENT Sampson of E on bearings, whitened by M₁/M₂
# =============================================================================
#
# TANGENT Sampson, whitened by the per-view back-projection Jacobians, which is
# what keeps this residual IN PIXELS.
#
# `M` picks the unit and the call cannot: the denominator is ‖∇C‖ with respect
# to the coordinate `M` differentiates against, so `M = ∂f/∂u` gives pixels
# while an orthonormal tangent basis would give radians. Bearings are unitless,
# so nothing else could set it.
#
# That is not a refinement, it is the difference between right and wrong here.
# The image-plane Sampson this used to call, fed bearings instead of calibrated
# 2-vectors, silently rescales: measured against the pre-port values, 10.93× at
# one point and a median of 5.01× across the set. The tangent form reproduces
# them to 0.999994. And nothing downstream is shape-checkable — the outlier
# half-width, the loss width, the IG
# prior, `POOL_FRAME_R_MAX` and SimFitPP's bounds are all expressed in this
# residual's unit, across a repo boundary, so a silent rescale would have
# invalidated every one of them with no test able to see it.
#
# The measurement-map conjugation that stood here is gone with E's maps: a
# matrix can only reach the z = 1 chart, so the camera's own back-projection
# replaces it.
#
# SCORED IN FLOAT64 whatever the problem's eltype, as the pre-port residual was
# (it formed a Float64 fundamental). The whitened gradient is ~1/f per pixel,
# so at Float32 the constraint and its gradient sit within a few ulps of the
# degeneracy test, and a Float32 camera at f = 2000 px scored every true inlier
# `Inf` (373 of 373) — `find_essential` returned `:no_model` where Float64
# converged on all 298. The data stay in the caller's eltype; only the
# arithmetic is widened, per point, with no allocation. The model is widened
# ONCE per model — RSDR's `residual_context`, which `residuals!` takes before
# its per-point loop — and the per-point widening is then a no-op.
@inline _bearing64(f) = SVector{3, Float64}(f[1], f[2], f[3])
@inline _jacobian64(M) = SMatrix{3, 2, Float64, 6}(M)

residual_context(::EssentialProblem, θ::EssentialMat) = EssentialMat{Float64}(Tuple(θ))

@inline whitened_residual(p::EssentialProblem, θ::EssentialMat, i::Int) =
    @inbounds VgtkCore.signed_sampson_distance(
    EssentialMat{Float64}(Tuple(θ)), _bearing64(sources(p.cs)[i]), _bearing64(targets(p.cs)[i]);
    M₁ = _jacobian64(p.cs.M₁[i]), M₂ = _jacobian64(p.cs.M₂[i])
)
