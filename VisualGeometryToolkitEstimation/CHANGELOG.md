# VisualGeometryToolkitEstimation changelog

## 0.6.0 (unreleased)

- **`fit_camera(T, source, region; …)`: one model-to-model regression** —
  the camera model of type `T` regressed onto `source` over `region` (Core's
  `imaged_region`, an `Ellipse`, a `Rect`, their `∩`), minimizing the
  integrated squared pixel error of the round trip; `FitResult{…,
  CameraFitMetadata}` (`max_px`, `max_px_edge`, the region, and the family's
  own record; the round trip's rms is the result's `rms`). Only the lens
  family takes a `weight` (the spline and grid targets' quadrature is on the
  image). The target type selects the family:
  - a lens family `L` (`fit_camera(L, model, region; weight)`): what
    `fit_lens` did, its radial profile's exact 1-D reduction with the density
    weighted by the region's share of each radius; on `fit_lens`'s cases it
    reproduces its parameters to 1.4e-9;
  - `SplineProjection` (`fit_camera(SplineProjection, source, region;
    tolerance)`): the forward map ray → pixel as Core's `ChartSpline` on the
    equidistant direction chart, penalized linear least squares (a P-spline,
    `λ = 1e-8·mean(diag(AᵀA))`) with its knot spacing searched — doubled, then
    bisected — to the largest whose held-out maximum meets `tolerance`
    (`:converged`; `:max_iterations` with what the finest spacing reached when
    it cannot). The search's fits are outer iterations, the metadata's
    `outer_iterations`; `iterations(result)` is 0;
  - `CentralGenericModel` (`fit_camera(CentralGenericModel, source, region;
    tolerance)`): a grid on Core's full-image lattice (a `BoundedCamera`'s
    extent or a grid source's image), the region restricting the quadrature
    only; the lattice spacing chosen by the same search (`grid_size` is not an
    input; the metadata records the chosen one), `:converged` or
    `:max_iterations`. The tolerance bounds the held-out direction error in
    pixels, the non-central target's metric (the fitted grid's direction
    against the source's, through the source's back-projection Jacobian): no
    seeded projection, so no solver floor reads an error as 0;
  - `NoncentralGenericModel` (`fit_camera(NoncentralGenericModel, source,
    region; tolerance)`): the same lattice search and penalized solve on
    the source's observation lines (a non-central grid, or any central source
    the central target takes), directions and points from one solve. Each
    line's point is its point closest to c*, Core's `nearest_point` of all
    the quadrature lines, so the points stay bounded for rays past 90°; they
    are absolute, in the source's frame, and c* is `metadata(r).family.center`.
    The tolerance bounds the held-out direction error in pixels (through the
    source's back-projection Jacobian); the line offset is reported as
    `max_offset`, `rms_offset`, `max_offset_edge` (metres). On a synthetic
    non-central fisheye to 97°, on a 40 × 40 lattice: 7.1e-4 px and 7.9e-9 m at
    most, against 0.39 px and 5.9e-6 m for sampling the source's lines at the
    nodes; `tolerance = 1e-3` chooses 26 × 26 (9.2e-4 px, 5.1e-8 m).
  The grid targets' held-out direction error skips a held-out pixel where
  the source's back-projection Jacobian is not finite (counted as
  `n_held_out_dropped`), as the quadrature skips non-finite samples. The
  `:image` density differentiates the source's radius exactly (ForwardDiff).
- **Breaking: removed `fit_lens` and `LensFitMetadata`** (retired in
  `check_naming`). `fit_lens(L, model; max_radius_px = R)` is
  `fit_camera(L, model, Ellipse(pp, R, R, 0.0))` (a disc about the principal
  point), `max_view_angle = θ` the disc of radius `f·r(θ)`; the metadata's `s`,
  `θ_max`, `θ_source`, `clipped` are `metadata(r).family`'s `focal_scale`,
  `θ_max`, `θ_source`, `clipped`. `fix_focal` and `n` are gone (the focal
  rescale is always fitted; the quadrature has 129 nodes), and the `:inverse`
  direction has no successor (`SplineProjection` is the forward map). The
  `[fit_lens]` table of `configs/default.toml` is `[fit_camera]`.
- CoordinateTransformations joins the dependencies (the chart's `LinearMap`).

## 0.5.0 (2026-11-03)

- **Breaking: a failed RANSAC has no estimate.** `RansacEstimate{M, T}` is
  `FitResult{Union{M, Nothing}, RansacMetadata{T}}`: a search in which no
  hypothesis survived (`retcode === :no_model`, `converged` false) returns
  `estimate(result) === nothing`, not a `zero(M)` placeholder that a forgotten
  `converged` test read as a model (a zero homography, a point ellipse at the
  origin). `estimate(result)` of a RANSAC result is `Union{M, Nothing}` —
  Base's `findfirst` idiom; test `converged(result)` before reading it. A
  converged result has the same type. The other diagnostics of a failed search
  are kept (its trials, sample-acceptance rate, `quality = -Inf`, an all-false
  `inlier_mask`, `rms`/`sigma` NaN).
- **Breaking: too few points is the same failure.** Every `find_*` and
  `refine_*` given fewer data than a minimal sample returns that `:no_model`
  result with `iterations == 0` (sample-acceptance rate NaN, the mask as long
  as the data, in the supplied workspace's storage eltype), not a bare
  `nothing`: one failure shape, tested with `converged`. A caller's
  `find_*(…) === nothing` becomes `!converged(r)`.
- The metre and frame re-wrappers of the physical (unitful) and labelled
  `find_absolute_pose`, `refine_absolute_pose`, `find_hand_eye` and
  `refine_hand_eye` forms return a failed result unchanged (no estimate to
  read back in metres or relabel).

## 0.4.0 (2026-11-02)

- NLLSsolver ≥ 5.3.1: the relative-pose bundle always uses NLLSsolver's typed
  `OptimizeCache` (its internal `cache` keyword is gone; `schur = false` still
  forces the sparse LDL while upstream #60 is open).

- Follows VisualGeometryToolkitCore's Sampson layers: the essential problem's
  residual is `only(sampson_residual(E, f₁, f₂; M₁, M₂))` and the division
  lift's `‖g‖²` is `constraint(m, X, x)`.
- One Sampson path, ONE chart-rows method: every chart problem whose residual
  is its constraint's whitened Sampson residual — F, E, H, the circle, the
  radial camera and the rotation — descends VisualGeometryToolkitCore's
  `whitened_residuals!`/`whitened_jacobian_rows!` over its `cs` (its noise and
  keyword columns, the LM's √w as `sqrt_weights`), at the chart's model, along
  its `retract_jacobian`; the per-problem row methods and the epipolar stamping
  are gone (`_constraint_model(problem, chart, a)` is the one per-problem fact,
  the rotation's bearing transfer form).
- Breaking in results: **the homography's and the rotation's LM descend the
  exact rows** of the residual they are scored with — the algebraic constraint
  whitened by the endpoints' noise for H (an exact source: the transfer error),
  the bearing transfer form for the rotation — with the whitening re-evaluated
  at every iterate and its derivative carried, instead of a transfer whitening
  frozen at the chart's anchor (a different function away from the anchor,
  whose fixed point is not the Sampson cost's minimum). Measured: H's refined
  models move ≤ 1.3e-5 relative (Frobenius, unit-normalized), the transfer RMS
  on the inliers by ≤ 3e-7 px, the inlier sets unchanged, the Sampson cost at
  the new fixed point lower (10.1491997 vs 10.1492046 on the test scene); the
  rotation ≤ 4.1e-7; an exact-source board homography unchanged (5e-16).
- The pool record's residual matrix is `sampson_distances!` and its RSS a
  composition of `sampson_distance`.
- Every problem's constraint is `(g, G_x, G_θ)`: a correspondence's noise is its
  `measurement_covariance(problem, i)` (the `Σ₁`/`Σ₂` columns; a line fit's
  per-point shapes) and its rounding scale `constraint_rounding_scale(problem,
  model, i)` (H, F, E, circle, ellipse, radial camera). The homography's
  data-frame model is its `residual_context`, formed once per model.
- Breaking: the whitening-policy keyword of `fit_radial_camera`,
  `fit_division_camera` and `division_camera_step` is removed with no
  replacement: the fits always whiten (the rows' frozen `whitening_factor`)
  from their second iteration; `fit_radial_camera(…; iterations = 1)` is the
  algebraic solve.
- Results: every `find_*`/`fit_*` agrees with 0.2 to ≤ 7e-14 relative except
  where a chart LM stops at its tolerance on rows now rounded in another order
  (`find_fundmat`: ≤ 7e-8 px on any residual, the inlier set unchanged).

## 0.2.0 (2026-10-29)

### Added

- **`fit_rotation_averaging(relative[, prior])`**, robust rotation averaging on
  a view graph: VisualGeometryToolkitCore's `spectral_rotation_averaging` seed,
  then the Lie-algebra IRLS of Chatterjee & Govindu over every edge (a sparse
  linear solve per iteration, a stack loss shape at the χ²₃-median scale);
  `[fit_rotation_averaging]`.
- **`RotationProblem` and `find_rotation`**: the pure rotation between two
  calibrated views (the infinite homography) by RANSAC on unit bearings with
  their back-projection Jacobians — VisualGeometryToolkitCore's `Rotation2pt`,
  the Sampson distance of the joint measurement (`Σ = I`) to the bearing
  transfer constraint, in pixels, the chart-LM refit
  on VisualGeometryToolkitCore's `SO3Manifold` through the transfer rows;
  `[pipelines.rotation]` (1 px, the essential problem's convention). The
  bearing checks and the sub-detector half-width refusal of the essential
  problem are shared (`_BearingProblem`).
- **`fit_linear(A, y[, prior])`**, the robust coefficients of a linear model:
  a reweighted least squares under a loss shape (default Huber at its
  efficiency constant 1.345) at the residuals' MAD scale, with fixed prior row
  weights; `configs/default.toml` `[fit_linear]`. Hoisted from
  RadiometricCamera's rooftop vignetting fit.

- **A `refine_*` for every `find_*`** — `refine_homography`, `refine_fundmat`,
  `refine_essential`, `refine_absolute_pose`, `refine_rotation`,
  `refine_radial_homography`, `refine_radial_camera`, `refine_hand_eye`,
  `refine_circle`, `refine_ellipse`: the `find_*`'s local optimization alone,
  run once from a given model (`locally_optimize` with the class's
  `[pipelines.<class>]` defaults). The model leads, then the `find_*`'s data;
  the keywords are the `find_*`'s strategy keywords (`threshold`, `scoring`,
  `local_optimization`, `workspace`, and its model-specific ones) without the
  sampling ones, so any local-optimization strategy plugs in.
- `EllipseProblem` provides the one model covariance (`solver_covariance`),
  so predictive scoring and an ellipse's covariance on its inliers.

### Changed

- **`Triggs` defaults to Welsch** in `[pipelines.circle]` and
  `[pipelines.ellipse]` (`Triggs{WelschShape}(scale_multiple = 2.5)`, was
  Cauchy; the multiple in Welsch's own currency): on a fisheye image circle's
  contour the Cauchy ellipse sat ~0.9 px from the inliers' least squares, the
  Welsch one on it. `Triggs{K}` admits exactly the shapes `Irls{K}` does, and
  `Triggs{GaussianShape}` is the plain chart LM, with no width.
- **`refine_ellipse(ellipse, points; kwargs...)`** (was `refine_ellipse(points,
  init = direct_ellipse(points); max_iter)`): the guess is required (a
  self-seeding refine is a fit) and the result is a `RansacEstimate`. The
  orthogonal (foot-point) distance it minimized is now the `EllipseProblem`'s
  refit rows, so `find_ellipse` and every strategy descend it; the Sampson
  distance still scores. Plain least squares is `local_optimization =
  Irls{GaussianShape}()` at a threshold above every residual; the covariance
  is `s²·solver_covariance(EllipseProblem(points), e, inliers)`.
- **`refine_absolute_pose(pose, camera, pairs; kwargs...)`** (was
  `refine_absolute_pose(pose, bearings; max_iter)`): `find_absolute_pose`'s
  step on the `AbsolutePoseProblem` rows, `X => u` pixel pairs through the
  camera (any lens; rays past 90°). On clean data it returns the same pose as
  the bearing form did (both to 1e-15); on 0.3 px noise the two differ by
  ≤ 6e-7. The refined pose carries the guess's frame labels (frames.md rule 1),
  in both forms.
- **`refine_absolute_pose(pose, bearings; kwargs...)`** keeps its bearing and
  generalized (ray-origin) forms, now inside the refine interface: `X => f`
  unit bearings with optional `origins`, `M` (pixel whitening) and `weights`
  columns, on the new **`BearingPoseProblem`** (the `MinimalSolveProblem` of
  `Extrinsics3` with a `BearingPoseContext`; `AbsolutePoseProblem` is now the
  one with a `PoseContext`) whose rows are the old whitened tangent-plane
  residual. Defaults `[pipelines.bearing_pose]`: least squares over every
  correspondence (`Irls{GaussianShape}()`, the threshold a fraction of the
  residual's bound), the same pose as the previous bearing form to ≤ 8e-14 on
  pinhole, 110° fisheye and non-central scenes, clean and at 0.3 px; any
  strategy plugs in (`Triggs`, `Irls{K}`). Refine-only for a non-central
  camera (no generalized minimal solver: `solve` refuses it). `max_iter` is
  gone (the cap is the strategy's): drop it from the call.

### Removed

- `refine_absolute_pose`'s `max_iter` and `[refine_absolute_pose]`, and
  `AbsolutePoseRefineMetadata` (its result is a `RansacEstimate`);
  `EllipseFitMetadata`.
