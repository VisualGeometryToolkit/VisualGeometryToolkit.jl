# =============================================================================
# RANSAC P3P Problem — absolute pose from 3D-2D correspondences
# =============================================================================
#
# `AbsolutePoseProblem` is the `MinimalSolveProblem{Extrinsics3}` alias (its
# construction and the minimal solve — VGC's `P3P` verb over the world→ray
# `cs_solve` view — are the family's, cspond.jl; rays + imaging transform travel
# in `PoseContext`). This file keeps the pose-specific residual, gates, the
# predictive constraint, and the chart rows the refit and the lift share.
#
# Residual: raw-pixel reprojection error through the full camera.
# Sampling: PROSAC (scored correspondences) or uniform (unscored).
# =============================================================================

# Pose arguments below are annotated `EuclideanMap{3, Float64}`, not the
# labeled `Extrinsics3` model alias: the LO/IRLS loop's in-flight iterates are
# rebuilt as UNLABELED maps (left-retraction recentering), while workspace
# models carry the Extrinsics3 labels — both must dispatch here. Labels
# attach at the storage boundary (`convert` into the workspace slot).
# The scoring residual, Jacobian-free: the RAW-pixel reprojection distance
# ‖project(model, R·X + t) − u‖ through the full camera — `project` answers on
# the ray domain, so a direction the lens cannot image is NaN, and the one
# participation rule (`_pnp_participates`) turns it into VGC's out-of-field
# residual: an outlier, not a NaN in the sort. `residuals!` is
# VisualGeometryCore's loop over this.
@inline function whitened_residual(p::AbsolutePoseProblem, pose::EuclideanMap{3, Float64}, i::Int)
    e = @inbounds VisualGeometryCore.reprojection_error(p.context.model, pose, sources(p.cs)[i], targets(p.cs)[i])
    _pnp_participates(e) || return VisualGeometryCore.OUT_OF_FIELD_RESIDUAL
    return norm(e)
end

# THE participation rule of a PnP correspondence — scoring, the predictive
# constraint, the chart LM and the loss lift: its raw-pixel reprojection
# residual is finite (the lens can place the point).
@inline _pnp_participates(e) = all(isfinite, e)

# Scale-invariant triangle "thinness" is VGC's `thin_ratio`
# (geometry/model_validity.jl), dispatching on 2-D and 3-D points alike.

# Minimum triangle thinness for a P3P minimal sample. Stricter than the
# homography sample test (1e-3): P3P is far more sensitive to near-degenerate
# samples. 3 control points that are nearly collinear — or whose image
# projections nearly coincide (two distinct world points landing at ~the same
# pixel) — admit a grazing pose that flings the board quad off to infinity yet
# still clears a bare `min_inliers` gate.
const _P3P_DEGENERACY_TOL = 1.0e-2

# Coincidence tolerances, supplied EXPLICITLY at both call sites below.
#
# VGC's `_are_points_distinct` defaults to `tol_px = COINCIDENT_PX` (0.1 px)
# since B7 made the epipolar/homography gate a pixel deduplication backstop.
# P3P shares the helper but not the domain: `X` is world control points in the
# CALLER's units (m, mm, …), where a tolerance named and valued in pixels is a
# category error — 0.1 "px" read as metres is a 10 cm coincidence radius. The
# image side is pixels and the name fits, but 0.1 px is a deduplication
# threshold calibrated for feature re-detections, not a P3P degeneracy bound.
#
# Both were previously riding the helper's old 1e-3 default. Both keep that
# value here, so B7 changes NO P3P behaviour — the point of naming them is to
# stop this gate silently inheriting a constant calibrated for another domain.
#
# NEITHER is calibrated, and the image one is known to be ineffective: 1e-3 px
# is 60x below a sub-pixel re-detection, so the image check never fires. That
# is the same dead-gate defect B7 fixed for F/E/H, still present here. Left as
# a separate decision rather than changed inside this window.
const _P3P_COINCIDENT_WORLD = 1.0e-3   # caller's world units — NOT pixels
const _P3P_COINCIDENT_IMAGE = 1.0e-3   # pixels — uncalibrated; see note above

"""
    test_sample(p::AbsolutePoseProblem, idx) -> Bool

Reject degenerate P3P minimal samples. Four checks, distinctness and
collinearity in each of the two spaces:
- **world**: the 3 control points distinct (`_P3P_COINCIDENT_WORLD`, in the
  caller's units) and non-collinear (`thin_ratio` > `_P3P_DEGENERACY_TOL`).
  Collinear control points ⇒ no unique P3P solution; this one earns its place.
- **image**: the 3 image points distinct (`_P3P_COINCIDENT_IMAGE`, pixels) and
  their RAYS not coplanar (thinness > `_P3P_DEGENERACY_TOL`). P3P degenerates
  when the three rays lie in one plane through the centre. For a `Pinhole`
  that is exactly collinear pixels (the pixel plane is an affine image of the
  z = 1 chart), so the test is `thin_ratio` on the pixels, as it always was.
  For a DISTORTING lens it is not: a line of rays images as a curve, so pixel
  collinearity admitted coplanar rays and rejected well-spread rim rays that
  the lens had pulled into a line. There the test is `_ray_thin_ratio` on the
  context's back-projected rays — `thin_ratio` in the gnomonic chart tangent
  at their mean direction.

# The image DISTINCTNESS check guards a premise that does not reproduce

It used to be justified here as "two detections at ~the same pixel (distinct
world points projecting together) force a grazing, near-degenerate pose". That
was constructed, not measured, and measurement does not support it. Sweeping
image separation for two genuinely distinct world points (common ray, ≥0.5
world units apart): noise-free, median error stays at 0° and q95 under 6e-5°
from 0 px to 40 px; at σ = 0.5 px, the median is 0.181° at EVERY separation
from 0 to 150 px, q95 0.655–0.702°, with no failure to return a solution.

Flat at both noise levels *including at separation zero*, which is
geometrically sensible: two world points at different depths along one ray is
a well-posed P3P configuration. The second point adds no independent
constraint, but the remaining correspondences still determine the pose. Two
points projecting together is not by itself a degeneracy.

It is also UNREACHABLE. A near-duplicate pair makes every triangle through it
thin, so the image collinearity check rejects first: measured, at any
separation small enough for the distinctness check to fire (<1e-3 px) the
image `thin_ratio` is already 3.1e-5, far under its 1e-2 bound; at 0.2 px,
where `thin_ratio` is healthy at 0.0125, distinctness admits. There is no
separation at which the image distinctness verdict decides anything.

So it is redundant against a premise that does not reproduce — vestigial, or
guarding a configuration nobody has constructed. Note that "a nearly-collinear
triple produces a grazing pose" is NOT the missing case: that is exactly what
`thin_ratio(u…)` tests, one line below.

The tolerance was deliberately NOT raised. At 1e-3 px — sixty times below
VisualGeometryFeatures' pinned 0.06 px worst-case refinement bound — its
measured duplicate capture is 0%, flat across spreads. That is the same
symptom B7 fixed for F, E and H but a different disease: those were
frame-COUPLED, their threshold moving with the data's spread, while this one is
correctly framed in pixels and simply too small. Raising it would make a dead
gate live across six repos on an unreproduced premise. If it is ever revisited,
do NOT borrow `COINCIDENT_PX`: 0.1 px answers "is this the same detection
twice?", anchored on a refinement bound, and this check asks a different
question — the constant does not transfer even though the units match.

What would settle it is a construction that produces the failure, or a
decision that it is vestigial. Not a threshold.

Mirrors `test_sample(::HomographyProblem, …)`, which had the same structure:
its coincidence check sat inoperative while `_not_collinear` caught duplicates
incidentally.
"""
function test_sample(p::AbsolutePoseProblem, idx::AbstractVector{Int})
    @inbounds begin
        X = (sources(p.cs)[idx[1]], sources(p.cs)[idx[2]], sources(p.cs)[idx[3]])
        u = (targets(p.cs)[idx[1]], targets(p.cs)[idx[2]], targets(p.cs)[idx[3]])
    end
    _are_points_distinct(X; tol_px = _P3P_COINCIDENT_WORLD) || return false
    _are_points_distinct(u; tol_px = _P3P_COINCIDENT_IMAGE) || return false
    thin_ratio(X[1], X[2], X[3]) > _P3P_DEGENERACY_TOL || return false
    return _image_thin_ratio(lens(p.context.model), p, u, idx) > _P3P_DEGENERACY_TOL
end

# The image triple's thinness, by the lens: a pinhole's pixels are a projective
# image of the rays, so their triangle is the test; any other lens bends lines,
# and the RAYS are tested instead (`_ray_thin_ratio`).
_image_thin_ratio(::VisualGeometryCore.Pinhole, ::AbsolutePoseProblem, u, _idx) = thin_ratio(u[1], u[2], u[3])
function _image_thin_ratio(_lens, p::AbsolutePoseProblem, _u, idx)
    rays = p.context.rays
    return @inbounds _ray_thin_ratio(rays[idx[1]], rays[idx[2]], rays[idx[3]])
end

# Coplanarity of three rays through the centre, as a triangle thinness: the
# rays' points on the gnomonic chart TANGENT at their mean direction, where a
# plane of rays is a line, measured by `thin_ratio` (0 ⟺ coplanar). The
# tangent chart keeps the measure what the pixel test is for a narrow field.
# A ray at or past 90° from the mean has no point on that chart; the triple
# is then judged by |det(a, b, c)| (0 ⟺ coplanar, 1 for orthogonal rays). A
# ray the lens could not place (non-finite) makes the sample degenerate.
@inline function _ray_thin_ratio(a::StaticVector{3}, b::StaticVector{3}, c::StaticVector{3})
    (all(isfinite, a) && all(isfinite, b) && all(isfinite, c)) || return 0.0
    ua, ub, uc = normalize(a), normalize(b), normalize(c)
    s = ua + ub + uc
    ns = norm(s)
    ns > 0 || return 0.0
    m = s / ns
    da, db, dc = dot(ua, m), dot(ub, m), dot(uc, m)
    min(da, db, dc) > 1.0e-6 || return abs(dot(ua, cross(ub, uc)))
    e1, e2 = s2_tangent_basis(m)                   # thin_ratio is invariant to the basis choice
    chart(v, d) = SVector(dot(v, e1) / d, dot(v, e2) / d)
    return thin_ratio(chart(ua, da), chart(ub, db), chart(uc, dc))
end

function test_model(::AbsolutePoseProblem, pose::EuclideanMap{3, Float64}, ::AbstractVector{Int})
    return all(isfinite, translation(pose))
end

# =============================================================================
# Predictive scoring — the pose constraint and its minimal-sample covariance
# =============================================================================

# One correspondence's constraint in the `residuals!` currency: VGC's lens form
# `constraint_with_jacobians(model, pose, X, u)` — e = project(model, R·X + t) − u
# in RAW pixels, G_x = −I₂ (the observed pixel is the measurement, X exact),
# G_θ its 2×6 Jacobian along LeftRigidManifold6 at the pose. A point the lens
# cannot place (`_pnp_participates`) is the out-of-field sentinel with a zero
# model row: an outlier, never a NaN in the sweep.
@inline function constraint_with_jacobians(p::AbsolutePoseProblem, pose::EuclideanMap{3, Float64}, i::Int)
    e, G_x, G_θ, valid = _pnp_pixel_row(p, pose, i)
    valid && return (e, G_x, G_θ)
    return (SVector{2, Float64}(VisualGeometryCore.OUT_OF_FIELD_RESIDUAL, 0.0), G_x, zero(G_θ))
end
provides_solver_covariance(::Type{<:AbsolutePoseProblem}) = true   # predictive scoring (covariance.jl)

# =============================================================================
# The PnP row — ONE definition of a correspondence's residual, its Jacobian and
# whether it participates, for scoring, the chart LM and the loss lift
# =============================================================================
#
# RAW-PIXEL row (every lens; the scoring currency): VGC's full-camera
# constraint, participating by `_pnp_participates` — `constraint_with_jacobians`
# projects on the ray domain and answers non-finite exactly when the lens cannot
# image the direction (no `Xc[3] > 0` test: that is the gnomonic chart's domain,
# and it rejected directions a fisheye images perfectly well). THE LM's
# zero-row guard lives here and only here: a participating point whose Jacobian
# is not finite (exactly at a lens fold, measure zero) contributes a zero model
# row, never a NaN in the normal equations.
@inline function _pnp_pixel_row(p::AbsolutePoseProblem, pose, i::Int)
    e, G_x, G_θ = @inbounds VisualGeometryCore.constraint_with_jacobians(p.context.model, pose, sources(p.cs)[i], targets(p.cs)[i])
    all(isfinite, G_θ) || (G_θ = zero(G_θ))
    return (e, G_x, G_θ, _pnp_participates(e))
end

# The row a local optimization runs on, by lens. A distorting lens: the raw-
# pixel row. A Pinhole: the CALIBRATED constraint against the precomputed ray,
# g = π(R·X + t) − π(ray), whitened by `A` (the lift's frozen pixel block
# K[1:2,1:2], which makes A·g EXACTLY the pixel residual; `nothing` for the
# chart LM's calibrated rows), participating iff the point lies along its own
# bearing — ORIENTED cheirality, not a coordinate sign: `Xc[3] > 0` asks about
# the Z = 0 plane, which is "in front" only when every bearing has z > 0.
@inline function _pnp_row(::VisualGeometryCore.AbstractLensModel, p::AbsolutePoseProblem, pose, i::Int, _A)
    e, _, G_θ, valid = _pnp_pixel_row(p, pose, i)
    return (e, G_θ, valid)
end
@inline function _pnp_row(::VisualGeometryCore.Pinhole, p::AbsolutePoseProblem, pose, i::Int, A)
    X = @inbounds sources(p.cs)[i]; ray = @inbounds p.context.rays[i]
    g, G_x, G_θ = VisualGeometryCore.constraint_with_jacobians(pose, X, ray)
    valid = dot(rotation(pose) * X + translation(pose), ray) > 0.0
    return (_pixel_whiten(A, g), _pixel_whiten(A, G_θ), valid)
end
@inline _pixel_whiten(::Nothing, x) = x
@inline _pixel_whiten(A::SMatrix{2, 2, Float64, 4}, x) = A * x

# =============================================================================
# The chart — VGC's LeftRigidManifold6 (6 dof, codim 2), refit and lift alike
# =============================================================================
#
# The pose steps in VGC's LEFT-perturbation chart, (R, t) ← (exp(δω)·R,
# exp(δω)·t + δt): the perturbation `constraint_with_jacobians(pose, …)`
# differentiates, so at the anchor the rows' Jacobian IS the constraint's G_θ
# and away from it the chain rule of its translation block ∂e/∂X_cam with the
# chart's action Jacobian (VGC `action_jacobian`, the SO(3) left Jacobian in
# closed form — never written by hand here). The chart-LM refit and the HQ/GNC
# lift (problems/manifold_providers.jl) share the chart, the rows and the
# re-anchor (`recenter`, which re-projects R onto SO(3), so chained refits and
# long lift runs accumulate no rounding drift). Until 2026-09-30 the refit ran
# on RigidManifold6 (MRP about R, free t) while the lift ran on this chart with
# its own row loops.
#
# STRUCTURAL REFERENCE: PoseLib's `AbsolutePoseRefiner`
# (PoseLib/robust/optim/absolute.h) — the same 6-dof tangent update about the
# current pose, per-point 2-vector reprojection residual, per-point weight and
# per-iteration participation guard. The MECHANISM of the lift differs: PoseLib
# applies a STATIC robust loss on the accumulator side, whereas the lift is the
# genuine joint-(θ,z) loss lift. POSE ONLY (the camera is given).
#
# RESIDUAL CURRENCY — pixels: for a Pinhole, the calibrated row against the
# precomputed ray (`_pnp_row`) is whitened by the lift's frozen pixel block
# A = K[1:2,1:2] (A·g is EXACTLY the pixel reprojection residual); a distorting
# lens's raw-pixel row already is pixels. So ‖rᵢ‖ is the scoring residual and
# `lift_cost`/ŝ/c live in one currency. (The refit's Pinhole rows are the
# calibrated ones, `A = nothing`: its weights come from the scoring.)
#
# PARTICIPATION — a point the lens cannot place (or, for a Pinhole, one not
# along its own bearing) is OUT OF FIELD: both of its rows carry VGC's one
# out-of-field residual `OUT_OF_FIELD_RESIDUAL` (the rule VGC's radial rows
# follow), weighted, with zero Jacobian rows — so a step that pushes a weighted
# point out is charged, never rewarded. The refit zeroes the weight of every
# point already out at its anchor (`_refit_state`), so the constant never enters
# an accepted iterate's cost; the lift masks such points out of its θ/z systems
# for the iteration (`lift_valid`: the rows write the mask), their z frozen,
# their cost saturated at the sentinel's norm. A point that returns re-enters
# with its z intact. (PoseLib skips such points entirely — the same θ-side rule.)
# =============================================================================

chart_dof(::Type{<:AbsolutePoseProblem}) = 6

# The chart at a pose: LeftRigidManifold6 re-anchored at zero coordinates —
# the nearest proper rotation, since minimal-solver seeds (VGC P3P backends)
# are unvalidated and can carry ~1e-7 of orthonormality drift.
_chart(::AbsolutePoseProblem, pose) = _reanchored(LeftRigidManifold6(rotation(pose), translation(pose)))

# The rows' extra: the pixel block of `_pnp_row` (`nothing` in the refit: its
# calibrated Pinhole rows) and the participation mask the rows write (the lift's
# `lws.valid`; `nothing` in the refit).
struct _PnPAux{TA, TV}
    A::TA
    valid::TV
end
_lm_state(p::AbsolutePoseProblem, ch, sqrtw) = ChartLm(ch, p, sqrtw, _PnPAux(nothing, nothing))

# A point out of the camera's reach at the ANCHOR is an outlier of the refit
# whatever its weight: it contributes nothing, and the out-of-field rows are
# left to charge only the points the LM itself would push out.
function _refit_state(p::AbsolutePoseProblem, ch, sqrtw)
    anchor = model(ch); L = lens(p.context.model)
    @inbounds for i in 1:data_size(p)
        _pnp_row(L, p, anchor, i, nothing)[3] || (sqrtw[2i - 1] = 0.0; sqrtw[2i] = 0.0)
    end
    return _lm_state(p, ch, sqrtw)
end

@inline _record_participation!(::Nothing, _i, _valid) = nothing
@inline _record_participation!(v::AbstractVector{Bool}, i, valid) = (@inbounds v[i] = valid; nothing)

# √w-weighted rows in the chart at `δ` (VGC's `residual_rows!`).
function residual_rows!(r, lm::ChartLm{<:LeftRigidManifold6, <:AbsolutePoseProblem}, δ)
    p = lm.problem; sqrtw = lm.sqrtw; L = lens(p.context.model)
    pose = retract(lm.ch, δ)
    @inbounds for i in 1:data_size(p)
        row = 2i - 1
        e, _, valid = _pnp_row(L, p, pose, i, lm.aux.A)
        _record_participation!(lm.aux.valid, i, valid)
        e₁, e₂ = valid ? (e[1], e[2]) : (VisualGeometryCore.OUT_OF_FIELD_RESIDUAL, VisualGeometryCore.OUT_OF_FIELD_RESIDUAL)
        r[row] = sqrtw[row] * e₁; r[row + 1] = sqrtw[row + 1] * e₂
    end
    return r
end

# Analytic Jacobian (2n×6): ∂e/∂X_cam (the translation block of the row's
# left-perturbation G_θ) times the chart's action Jacobian at δ. An
# out-of-field point gets zero rows. Cross-checked vs ForwardDiff in the tests.
function residual_jacobian_rows!(J, lm::ChartLm{<:LeftRigidManifold6, <:AbsolutePoseProblem}, δ)
    p = lm.problem; sqrtw = lm.sqrtw; L = lens(p.context.model); Xs = sources(p.cs)
    pose = retract(lm.ch, δ); v = SVector{6, Float64}(δ)
    @inbounds for i in 1:data_size(p)
        row = 2i - 1
        _, Gθ, valid = _pnp_row(L, p, pose, i, lm.aux.A)
        if !valid
            for k in 1:6
                J[row, k] = 0.0; J[row + 1, k] = 0.0
            end
            continue
        end
        Jᵢ = Gθ[:, SVector(4, 5, 6)] * action_jacobian(lm.ch, Xs[i], v)   # 2×6
        sw₁ = sqrtw[row]; sw₂ = sqrtw[row + 1]
        for k in 1:6
            J[row, k] = sw₁ * Jᵢ[1, k]; J[row + 1, k] = sw₂ * Jᵢ[2, k]
        end
    end
    return J
end

# The lift's pixel block `A` of `_pnp_row`: for a Pinhole, VGC's frozen pixel
# whitening of the imaging transform (`A = K[1:2,1:2]`, so A·g is the pixel
# residual); `nothing` for a distorting lens, whose raw-pixel row already IS
# pixels.
_lift_pixel_block(::VisualGeometryCore.Pinhole, p::AbsolutePoseProblem) = pnp_pixel_whitening(p.context.proj)
_lift_pixel_block(::VisualGeometryCore.AbstractLensModel, ::AbsolutePoseProblem) = nothing
