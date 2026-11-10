# VisualGeometryToolkitEstimation changelog

## 0.9.0 (unreleased)

On VisualGeometryToolkitCore 0.9 (the central grid camera folded into `CameraModel` as a `SplineLens`; the
non-central one is a `NoncentralSplineModel`).

- **Breaking: `fit_camera(SplineLens, source, region; tolerance, grid_size = nothing, forward = true)`** fits ONE
  camera: a `CameraModel` with a `SplineLens`, its K the source's own (skew 0). Its backward spline is the P-spline
  of Core's full-image lattice (`spline_lattice`) on the source's rays at x = K⁻¹u — at `grid_size` when given, else
  at the lattice the spacing search chooses to `tolerance` (the held-out direction error in pixels, as before). A
  SplineLens source's backward is kept exactly and only the forward is fitted. Its forward spline (`forward = true`)
  is a `ChartSpline{2}` on the equidistant chart about the lens axis, fitted to the camera's own rays by the same
  solve and the same spacing search, to `tolerance` in pixels of the held-out round trip; the camera then projects
  through it. The metadata `family` is the union of the old central grid's and the old forward map's records
  (`penalty, grid_size, stride, outer_iterations, n_held_out_dropped, tolerance`, then `spacing, counts,
  forward_penalty, forward_outer_iterations, forward_max_px, forward_max_px_edge, forward_rms`, `nothing` without a
  forward). It replaces `fit_camera(CentralGenericModel, …)` and `fit_camera(SplineProjection, …)`, both deleted.
- **Breaking: `fit_camera(NoncentralSplineModel, source, region; tolerance, grid_size = nothing)`** replaces
  `fit_camera(NoncentralGenericModel, …)`: the same lines, anchor and one solve, returning a `NoncentralSplineModel`
  (its directions and points `ChartSpline`s over the K-normalized plane).
- **Breaking: `pspline_grid(SplineLens | NoncentralSplineModel, source, region, grid_size)`** replaces the
  `CentralGenericModel`/`NoncentralGenericModel` forms (a SplineLens camera without a forward, or the non-central
  model).
- **Breaking: the spline targets take a `BoundedCamera` source only** (the lattice is laid over its image): a
  non-central source is `BoundedCamera(model, Size2(w, h))`, and a function of the pixel is no longer a source — build
  its SplineLens camera with Core's `CameraModel(SplineLens, f, size)` (which gauges its K) and fit that, which keeps
  its backward and fits its forward.
- **Breaking: a function of the pixel goes through `CameraModel(SplineLens, f, size)`**: its backward is Core's
  quasi-interpolant at the default lattice (`init = :quasi_interpolate`), and `fit_camera(SplineLens, …)` keeps that
  backward exactly and fits only its forward. The P-spline backward fitted to `tolerance` from a bare function of
  the pixel is gone (a camera source still gets it).
- **`fit_camera(L, source, region)` takes a SplineLens camera** (a `CameraModel` or a `BoundedCamera`): a gauge K has
  no radial profile of its own lens, so the regression samples the camera's rays on the image over the region (each
  pixel's view angle and normalized radius, the `:image` density; another `weight` is an `ArgumentError`) — the route
  VisualGeometryToolkitIO's exporters name for a SplineLens camera, which has no OpenCV/COLMAP form.
- **`fit_camera(SplineLens | NoncentralSplineModel, …)` and `pspline_grid` take `device = default_device()`**: the
  KernelAbstractions backend of a SplineLens source's batched back-projection (the rest of the fit runs on the host).
- **Breaking: a spline fit's `grid_size` is `(columns, rows)`** everywhere — the keyword, `pspline_grid`'s argument and
  the metadata's `family.grid_size` — Core's `grid_size(m)` order (a fit once reported `(rows, columns)`).
- **`refine_camera_and_poses` refines a SplineLens camera's controls with its poses, K frozen** (bit-identical): the
  present nodes' backward controls, any norms (a `fit_camera` lattice, a quasi-interpolant), each `c = e^ℓ·D` its
  direction on its sphere (Core's `ControlPatchManifold`) and its log-scale, grouped into 2 × 2 patches (NLLSsolver's
  10-variable block cap); one block per correspondence on its window's 3 × 3 patches and its pose, its Jacobian
  analytic by implicit differentiation of the projection solve at x* (Core's `_solved_projection` and
  `_projection_jacobians`: no ForwardDiff reaches the solve); a third-difference penalty over the present nodes
  (Core's `difference_rows`; `penalty`, default the P-spline fit's 1e-8 relative weight, read against the direction
  coordinates); the scale of the node nearest x = 0 held (a normalization constraint: the penalty is not scale-free); the poses refined alone first, then the directions with the scales held, then everything; the blocks rebuilt at the estimate until
  every window is its own; and the rotation gauge fixed after the solve by one rotation (the lens axis onto ẑ, the
  roll putting the x-derivative at x = 0 in the xz plane; the cost unchanged). `free` is `(:controls,)` or `()` (the
  poses alone). A forward spline is dropped and, with `refit_forward = true` (the default), refitted by
  `fit_camera(SplineLens, …; tolerance = forward_tolerance)` over the camera's imaged region (a `BoundedCamera` then
  required). Correspondences detected on an absent window are dropped and counted, and those not projected within a
  cell of one counted apart. `CameraPosesRefineMetadata` gains a fifth field, `spline` (`nothing` for a parametric
  camera; a positional construction now takes it). Every camera argument may be a `BoundedCamera` of a `CameraModel`
  (the estimate comes back bounded alike). CPU (NLLSsolver); a correspondence block hands NLLSsolver its rank-2 normal
  matrix lazily (`computecostgradhess`), not as a dense 114 × 114 matrix.
- `fit_camera(L, …)` warns about non-radial terms only when Core's type-level radial-symmetry test flags the lens; the
  decentering norm is the warning's magnitude (a symmetric `∘` composition is no longer measured as decentered).

## 0.8.0 (2026-11-09)

Needs the VisualGeometryToolkitCore release with `RadialHomography{T, N}` (its
`RadialHomography{T, 1}` is a TypeError on VisualGeometryToolkitCore 0.5.x).

- **Breaking: `RadialHomographyProblem` is over `RadialHomography{T, 1}`**
  (`RadialHomography{T}` is a UnionAll in VisualGeometryToolkitCore now; the
  RANSAC workspace's model placeholder needs a concrete type). The problem and
  its results are unchanged bit for bit.
- **Breaking: `division_coeffs` is VisualGeometryToolkitCore's function.**
  Estimation adds its `DivisionCameraState` method to it and no longer declares
  a public `division_coeffs` of its own, so a module that loads both packages
  sees one generic (`VisualGeometryToolkitCore.division_coeffs(state)` works;
  `using` both no longer conflicts).
- **`refine_radial_homography` takes `RadialHomography{<:Any, 1}` only**: an
  `N > 1` model is a `MethodError` at dispatch.
- The division-camera solve eliminates the per-board pose blocks by an arrow
  (Schur) solve over blocks of any static size, each factored once by
  Cholesky; the camera's one-coefficient solve is unchanged bit for bit (a
  golden fixture pins it). A block whose `D` has a non-finite entry is left out
  like one that is not positive definite (its pose NaN), and the blocks may be any
  `AbstractVector`.
- The radial-homography extent gate calls VisualGeometryToolkitCore's public
  `radial_in_front` (no private names of another stack package).

## 0.7.0 (2026-11-07)

### Breaking

- **`fit_residual_mixture(distances, scoring; …)`**: the input is the held model's DISTANCES (`≥ 0`, what
  `mahalanobis_distances!` fills); a negative entry is refused with an `ArgumentError` naming `abs` (it is a signed
  residual). The positional was named `residuals`.
- `RelposeBundleProblem` and `BearingPoseProblem` answer `whitened_residual` (the bundle's: the reprojection
  distance signed by the epipolar constraint's side; the bearing pose's: the whitened tangent 2-vector), and the
  conformance suite runs `test_constraint_problem` on both. `RotationProblem`'s hemisphere gate is on the residual
  (`(Inf, Inf)` for an antipodal pair; its distance is Core's default norm). `RadialHomographyProblem`'s residual
  is the reprojection offset in target pixels — the whitened residual, since the problem admits only exact (or
  plain) sources and plain targets — and, out of field, points `OUT_OF_FIELD_RESIDUAL` along the target's radial
  direction (its norm the distance's constant), no longer along x.
- The instance fallback `codimension(p::AbstractRansacProblem)` is deleted: VisualGeometryToolkitCore's
  `codimension(p::AbstractConstraintProblem) = codimension(typeof(p))` answers every problem.
- The chart LM's Sampson rows are VisualGeometryToolkitCore's `weighted_whitened_residuals!` /
  `weighted_whitened_jacobian_rows!` (Core's `whitened_*` are now unweighted); the rows are bit for bit unchanged.
- `fit_camera`'s P-spline penalty is Core's `difference_rows((ny, nx), 3)` (`DᵀD` identical, entry for entry), its
  direction chart Core's `equidistant_chart` (the same basis rule), and the relative-pose bundle's points Core's
  `Eliminable`; the private `_third_difference` and the bundle's own point wrapper are deleted.

- **`RansacWorkspace` takes its storage eltype as an optional LEADING
  positional** on every form, as the stack's signature rule places a type
  argument in a non-mutating call:
  `RansacWorkspace([T = Float32,] problem, scoring[, local_optimization])`,
  `RansacWorkspace([T = Float32,] P, n_max, scoring[, local_optimization])` and
  `RansacWorkspace([T = Float32,] P, n_data)`. The keyword `T` of the
  `(P, n_max, scoring[, lo])` form and the trailing positional `T` of the
  `(P, n_data)` and `(problem, scoring[, lo])` forms are removed, with no
  deprecated path: an old call is a `MethodError`. Migrate
  `RansacWorkspace(P, n, sc, lo; T = Float64)` to
  `RansacWorkspace(Float64, P, n, sc, lo)`, `RansacWorkspace(P, n, Float64)` to
  `RansacWorkspace(Float64, P, n)`, `RansacWorkspace(problem, sc, Float64)`
  to `RansacWorkspace(Float64, problem, sc)` and
  `RansacWorkspace(problem, sc, lo, Float64)` to
  `RansacWorkspace(Float64, problem, sc, lo)`. Calls without an eltype are
  unchanged.
- **The RANSAC protocol rename of VisualGeometryToolkitCore 0.7** (its
  changelog states it): every problem's per-point scalar is the Mahalanobis
  DISTANCE (`≥ 0`), computed by `mahalanobis_distances!(d, problem, model)`;
  scores are bit for bit unchanged. The problems' Jacobian-free overrides of
  the old `whitened_residual` hook — `AbsolutePoseProblem`,
  `BearingPoseProblem`, `RadialHomographyProblem`, `RotationProblem` (behind
  its hemisphere gate) and `RelposeBundleProblem` — are now
  `mahalanobis_distance` methods. `EssentialProblem`'s signed Sampson
  override stays `whitened_residual`, the signed residual, and
  `RadialHomographyProblem` gains a `whitened_residual` (the 2-component
  reprojection offset; out of field `(OUT_OF_FIELD_RESIDUAL, 0)`, whose norm
  is the distance's). `BoundedProblem` forwards `mahalanobis_distances!`,
  `residual_context`, `mahalanobis_distance` and `whitened_residual` to its
  inner problem.
  `DivisionCameraState`'s field `sq_residuals` is `sq_distances`
  (`board_sq_residuals(state)` is unchanged). `fit_residual_mixture`'s held
  residuals are scored as given; their square is what the mixture reads.

## 0.6.1 (2026-11-06)

VisualGeometryToolkitCore 0.6 (compat `"0.6"`).

- A pose is converted to the problem's model type once, where refinement and
  local optimization start; any other model type is refused.
- `fit_camera` reports a non-finite focal scale as `:solve_failed`.
- The spelled-out `CovMat` types follow Core 0.6's `CovMat{N, T, M}` (a
  static matrix held, full, diagonal or isotropic).

## 0.6.0 (2026-11-04)

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
  The grid targets' held-out errors — the direction error and the
  non-central grid's line offset alike — skip a held-out pixel where the
  source's back-projection Jacobian is not finite (counted as
  `n_held_out_dropped`), as the quadrature skips non-finite samples. The
  `:image` density differentiates the source's radius exactly (ForwardDiff,
  one Dual evaluation per node, for `:image` only); the regression's target
  radii are the source's plain `projected_radius`, so a lens refitted onto its
  own family at its anchor has exactly zero residuals. A source that is
  neither a camera nor a function — any value with a batched `backproject` —
  is sampled as it is, and a function source is evaluated serially (a user
  function is not known to be thread-safe).
- **`pspline_grid(G, source, region, grid_size)`: the grid targets' one
  P-spline solve on a GIVEN lattice** (`G` a `CentralGenericModel` or
  `NoncentralGenericModel`, `grid_size = (columns, rows)` of Core's
  full-image lattice) — an algorithm name (one penalized least squares, fixed
  weights): no spacing search, no tolerance, `retcode = :solved`, the same
  `CameraFitMetadata` and held-out errors. `fit_camera`'s grid targets run it
  at every spacing they try, so at the lattice `fit_camera` chose it is
  `fit_camera`'s result.
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
