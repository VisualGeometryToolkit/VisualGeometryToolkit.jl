# VisualGeometryToolkitEstimation changelog

## 0.9.0 (unreleased)

On VisualGeometryToolkitCore 0.9 (the central grid camera folded into `CameraModel` as a `SplineLens`; the
non-central one is a `NoncentralSplineModel`).

- **Breaking (behaviour): the reweighted fits take one FIXED scale** (the robust-estimation contract, B2).
  `fit_linear`, `fit_rotation` and `fit_rotation_averaging` keep their objective; their only robustness is
  `optimization_strategy = Irls{K}(…)` (was `local_optimization`; `nothing` is the plain least squares, a refine's
  `Triggs` is refused), at ONE loss width set from exactly one of `scale = σ`, `scale = <upstream result>` (its
  `noise_scale`) or `width = w`. The width is `w` or `efficiency_constant × σ`, the constant the strategy's (the plain
  multiple for every shape) or `efficiency_constant(K, d)` at the fit's codimension (1, 2, 3). Neither, both, a
  non-positive value, a `ScaleNone` upstream result, or a scale given to `optimization_strategy = nothing` is an
  `ArgumentError`, and so is `Irls{K}(loss_width = w)` (the width is the call's `width`). They no longer re-estimate
  the MAD scale σ̂ every iteration (≈ 1.3σ at 20 % contamination, a ~30 % wider width than the constant intends), so
  their numbers move. At the true σ, 20 seeds each, median / max error before → after: fit_linear 1.67e-4 / 2.44e-4 →
  1.60e-4 / 2.28e-4; fit_rotation 1.83e-4 / 3.80e-4 → 1.90e-4 / 3.63e-4 rad; fit_rotation_averaging (max frame error)
  0.945° / 2.394° → 0.961° / 2.393°. Accuracy is unchanged within noise; the iteration counts move by a few. A hard
  cutoff (TLS) at a fixed width must contain the least-squares seed's residuals, or every datum is rejected
  (`:degenerate`). The defaults stay Huber
  (`[fit_*] optimization_strategy = { type = "irls", loss = "huber", … }`, was `local_optimization`), so a bare call
  now needs its `scale` or `width`.
- **Breaking: the fits' metadata follow the contract's record.** `LinearFitMetadata{L, W}`, `RotationFitMetadata{L, W}`,
  `RotationAveragingMetadata{K, L, W}` (`L` the strategy type) carry `weights`, `scale`, `scale_source`, `loss_width`;
  `sigma` is retired (`noise_scale(result)`), and `RotationAveragingMetadata`'s `irls_weights` dictionary is replaced by
  `edges` and the index-aligned `weights`. `weights(result)` is the loss weight `ρ′` normalized to 1 (`IrlsWeights`:
  `inlier_mask(result)` needs a cut), the caller's prior NOT folded in (the solve's weight is prior × weight); least
  squares carries lazy unit weights (`BinaryWeights`). `n` is the support (data of positive solve weight) and `rms`
  the weight-weighted root mean square, `√(Σwᵢrᵢ²/Σwᵢ)` (`fit_rotation`'s was over every pair).
- **New: `fit_robot_world_hand_eye(…; optimization_strategy = Irls{K}(), σ_rot, σ_trans, scale | width, tol)`**: the
  staged solve reweighted per observation on the whitened 6-residual (codimension 6): weighted Kabsch pairs, weighted
  chordal means, the weighted translation least squares; the covariance and diagnostics at the final weights. A robust
  fit requires `σ_rot`/`σ_trans` (its scale's units). The default `optimization_strategy = nothing` is the former
  staged solve, bit for bit. `RobotWorldHandEyeMetadata{K, S, L, W}` gains `weights`, `scale`, `scale_source`,
  `loss_width` (shared with `refine_robot_world_hand_eye`). This is icra2026's "failure B" fix: five sparse,
  mis-registered frames no longer poison the mount. Synthetic (30 frames × 2 markers, 2 mrad / 2 mm, five gross poses,
  20 seeds): `Y` 1.84° / 43.5 mm (least squares) → 0.025° / 0.55 mm (Huber). A redescending loss from the poisoned seed
  may reject a whole marker: an `ArgumentError` naming Huber.
- **New: per-edge rotation covariances in `fit_rotation_averaging`** (the owner's ruling). An edge
  `Attributed(Uncertain(R_ij, Σ), (; weight))` carries its rotation's unit-scale covariance (3×3 in the so(3) chart,
  or 9×9 over vec(R) as `solver_covariance` returns it); each edge residual is whitened by Core's
  `whitening_factor(R_j, Σ)` before the loss and in the update, and `scale` is then the variance factor σ₀ (noise
  σ₀²Σ). Bare edges are solved exactly as before. `RotationProblem` declares `provides_solver_covariance`: its
  `solver_covariance(problem, R, inliers)` is the existing constraint sandwich (Core's `implicit_covariance` on the
  transfer-form rows, unit pixel noise), the implicit-function covariance of the Sampson LM its local optimization runs;
  a Monte Carlo of that estimator matches it (300 trials, within 15 %).
- **New: `fit_rigid_alignment(cs; optimization_strategy, scale | width, tol)`**, the rigid `b ≈ R·a + t` from point
  pairs (the ATE alignment): the weighted Kabsch on weighted centroids, least squares by default, a fixed-scale IRLS
  with an `Irls` strategy. `RigidAlignmentMetadata{L, W}`. A 150-frame walk (5 mm noise) with five centres 0.9–2.4 m
  wrong: tilt 1.20° / ATE 36.3 mm (least squares, the former Kabsch ATE) → 0.027° / 8.6 mm (Huber; the noise floor). `configs/default.toml` gains `[fit_rigid_alignment] tol` and
  `[robot_world_hand_eye] tol`.

- **Breaking: `refine_camera_and_poses`'s robust strategy is `local_optimization`** (was `robust`; the keyword `find_*`
  takes it by): the same `Irls{Shape}`/`Triggs{Shape}`, read by `local_optimization_from_toml`; `nothing`, the default,
  is plain least squares, bit for bit. The keyword `robust`, whatever its value, is an `ArgumentError` naming
  `local_optimization`.

- **Breaking (behaviour): a robust `refine_camera_and_poses` runs no least-squares stage and estimates no scale** (the
  owner's rulings, 2026-11-11). The Triggs or Irls solve starts directly from the caller's start (camera and poses), at
  the strategy's `scale_multiple` times the caller's `scale` (the detections' σ), or at its absolute `loss_width`; a
  `scale_multiple` strategy without a `scale` is an `ArgumentError`. Gone: the frozen Student-t scale of the
  least-squares rows, the scale anchored on a scorer's consensus partition (the `scoring` keyword and
  `configs/default.toml`'s `[refine_camera_and_poses] scoring`) and its thin-partition refusal. On the 8 %-outlier
  fixture (5 views × 60, σ 0.2 px, 8 % at 20–60 px, a start 20–80 px off), at σ: camera error 0.175 px (Irls{Huber},
  as with the stage) and 0.30 px (Triggs{Welsch}, 0.20 with the stage: one Welsch LM from 25 px off); from a start
  near the data, 0.175 / 0.202 px.

- **New: a SplineLens camera's principal point is its OPTICAL AXIS.** Breaking for every SplineLens fit's frame and K.
  - **`fit_camera(L, source, region)` dispatches on the source's K** (Core's `IntrinsicsRole`). A physical K: today's
    fit, its principal point and ẑ the source's own. A gauge K (a SplineLens camera): the principal point, the aspect
    and the axis are free, the axis on a 2-coordinate sphere chart at ẑ embedded as the minimal rotation onto ẑ, so its
    roll is fixed exactly. They live in the one chart builder, `_regression_manifold`.
  - **One 2-D residual replaces the radial one**, for every pair. A forward target's residual is in pixels (its
    `projected_radius` along the ray's azimuth, finite past a fold). A lift target's is angle-style, in the
    equidistant chart about ẑ, scaled by the radius.
    - At a physical K's sample (azimuth 0) both are exactly the former rows, so every recorded profile regression
      keeps its minimizer: the golden test is unchanged (OPENCV3, the decentered OPENCV target and the double sphere
      past its image circle included), and so is the own-family test's exact zero.
  - **Every `fit_camera` method's `CameraFitMetadata` gains `rotation`** (a frame-labelled `RotationMap`,
    `ViewFrame{:source}` → `ViewFrame{:fitted}`: exactly the identity where nothing turns) **and `symmetry_residual`**
    (px: the azimuthal misfit about the fitted principal point; `NaN` where not measured). The lens families' record
    gains `aspect`.
  - **A non-central source** of a lens-family fit is an `ArgumentError` (not supported yet).
- **NLLSsolver ≥ 5.3.4** (`[compat]` floor, the package and its tests): 5.3.4 adds our pull request #69, the run-time
  block views (a residual block of a run-time size assembled without a per-block allocation).

- **The lens-family regression runs on the device** (`fit_camera(L, …; device = default_device())`).
  - Each sample's 2-D residual and its Jacobian rows (ForwardDiff duals of the chart vector inside the kernel), and
    their reduction to ½‖r‖², Jᵀr and JᵀJ, are one KernelAbstractions kernel on every backend, through Core's
    `_launch_rows!`.
  - Only the N × N step is the host's: Core's `lmsolve_chart!` through its new reduction protocol, the same chart
    and stops.
  - Measured on the 5° synthetic: 0.20 s on the 4090, 0.27 s on 32 CPU threads (the former host ForwardDiff over
    every row: 3.9 s). The CPU and CUDA fits agree to 1e-13.
- **The re-gauge takes the fitted K**: principal point, focal lengths and aspect (`axis_gauged(camera, R, K′)`, no ray
  changes).

- **New: `optical_axis(camera::BoundedCamera)`** (exported), the one `fit_camera` call of the Fisheye624 family
  (`Additive(Tangential, ThinPrism{2}) ∘ Radial{Even, 6}`, its decentering held at zero) onto a camera, over the
  camera's own bound (`imaged_region`, never a default): its axis, principal point, rotation and symmetry residual.
  A bare model is an `ArgumentError` naming the wrap. The family's manifold frees a composition's first radial stage
  (Core's `_bounding_radial`), so a composed target fits at all.
- **`fit_camera(SplineLens, …)` gauges a gauge-K source onto its optical axis first** (`optical_axis`, then Core's
  `axis_gauged`: every ray turned by R, exactly), the axis fitted over the source's bound. Its forward is fitted there, and the metadata's `rotation` says by how
  much. A physical source is unchanged.
- **The SplineLens `refine_camera_and_poses` starts on the guess's optical axis.** The camera is turned and its
  principal point moved to the axis's pixel, the poses turned with it. Breaking: the camera must be a `BoundedCamera`
  (its bound is the axis fit's region); a bare SplineLens model is an `ArgumentError`.
  - The refine holds that frame and K exactly. Its end step (the ray at x = 0 onto ẑ, the roll, the mean-length
    rescale: `_gauged`, `_gauge_rotation`, `_turned_variables`) is deleted; with it, `spline.gauge_angle` and
    `gauge_roll`.
  - `spline` gains `axis_rotation`, `symmetry_residual` and the diagnostic `refined_axis_offset`,
    `refined_symmetry_residual`: one `optical_axis` fit of the refined camera, from which nothing is re-gauged.
  - Breaking: the refine returns the entry-gauged K, not the caller's.
  - Breaking: `spline.forward === :kept` is gone. A forward spline is now refitted or dropped, never kept (the turn
    leaves it stale). It is refitted by
    `fit_camera(SplineLens)`'s own forward fit, which does not re-gauge.

- **Breaking: the RANSAC entries take a scorer, never a bare width** (the owner's ruling: the `find_*` pipelines follow
  the one robust-estimation schema). Every `find_*` and `refine_*` (homography, fundmat, essential, rotation, circle,
  ellipse, absolute_pose — the bearing form too — radial_homography, radial_camera, hand_eye) and both
  `locally_optimize` methods take `scoring = <scorer>` beside `local_optimization`: `MsacScoring(problem; threshold)`,
  `RansacScoring(problem; threshold)`, `StudentTUniformScoring(problem, outlier_halfwidth; scale_prior, label_prior)`,
  a downstream family. The `threshold` keyword is removed, and so is the function-level `outlier_halfwidth`: each is
  refused with an `ArgumentError` naming `scoring` (not a retired name: `threshold` is a threshold scorer's field and
  keyword, and stays one). Write `find_homography(cs; scoring = MsacScoring(HomographyProblem; threshold = 2.0))` for
  `find_homography(cs; threshold = 2.0)`. `locally_optimize` has no class to default from: its `scoring` is required
  (`UndefKeywordError`). The physical `find_hand_eye`/`refine_hand_eye` forms no longer strip a `threshold` length: a
  scorer's width is a plain number in metres.
- **Breaking (TOML): `[pipelines.<class>]` states its scorer as a table**, `scoring = { type = "msac", threshold = … }`,
  read by `scoring_from_toml` for the problem's type, in place of the bare `threshold` key (which a pipeline table now
  never holds). The defaults are today's — MSAC at every class's convention, unchanged — and every `find_*`/`refine_*`
  result is bit-identical (the 473 pipeline calls of the suite's 55 items that call one, replayed against the parent and compared field by field with `isequal`). Hand-eye's and
  the bearing pose's file widths stay unit-free (a fraction of `L`, of the residual's bound), scaled by the problem
  (`_width_unit`); a caller's scorer is never rescaled.
- **Each scorer takes its own width key, and only it**: `scoring_from_toml` refuses `outlier_halfwidth` in a threshold
  scorer's table and `threshold` in the Student-t + uniform scorer's (it has no threshold), naming the key the scorer
  takes; a missing key is refused, the Student-t priors inheriting `[scoring.student_t_uniform]` as before.
- **Internal:** `_default_scoring`/`_default_scoring_at` and `_threshold_unit` are gone (`_class_scoring`,
  `_refuse_width_keywords`, `_width_unit`).

- **Breaking: `fit_division_camera` runs on the stack's shared robust machinery** (the owner's ruling (4), one
  schema). Its E-step is the posterior of the shared `StudentTUniformScoring` (keyword `scoring`, default
  `[fit_division_camera] scoring` read by `scoring_from_toml` with `problem = DivisionCameraProblem`, absent keys
  inheriting `[scoring.student_t_uniform]`); its loop is the posterior-weight iteration, `PosteriorIrlsMethod` /
  `PosteriorIrlsState`, of a `MixtureEmConfig` (keyword `mixture_em`, default `[fit_division_camera] mixture_em`
  read by `mixture_em_from_toml`, absent keys inheriting `[mixture_em]`), over the stack as the new public
  `DivisionCameraProblem` (its `refine!` the weighted, whitened `division_camera_step`). The private IRLS
  (`DivisionCameraIrls`, its state and its focal-change criterion) is removed. Removed keys and keywords: the
  `[fit_division_camera]` keys `outlier_halfwidth` (now the scorer table's) and `max_iter` (now the EM table's);
  the keywords `outlier_halfwidth` and `max_iter` are an `ArgumentError` naming the new form. The table keeps
  `ncoef` and `branch_audit_scale_multiple`. The `max_iter = 0` form (iteration 0 alone) is gone: the EM runs at
  least one iteration; the seeded step alone is `division_camera_step(N, radial_cameras, views; whiten_at =
  focal_seed)`. Results: every test and golden fit, and every stack of the MC calibration item (e), within 5e-13
  relative of before (4e-12 in the MC), the same iteration counts, branch decisions and gate statistics; the
  difference is rounding (the shared residual block squares the whitened norms the solve returns squared).
  Downstream: a caller's `outlier_halfwidth = h` becomes `scoring = scoring_from_toml(Dict("type" =>
  "student_t_uniform", "outlier_halfwidth" => h); problem = DivisionCameraProblem)`.
- **New: a selectable stop rule of the posterior-weight EM, `stop`** (`STOP_TAGS`, a required key of the EM table,
  defaulted in configs/default.toml only): `"no_improvement"` (`NoImprovementStop`, the incumbent's cost —
  under `keep_model` its NLL at `tol` — returning the incumbent; `[mixture_em]`'s and the RANSAC local
  optimization's rule, unchanged bit for bit) or `{ type = "parameter_change", rtol = … }`
  (`ParameterChangeStop`: the relative change of the problem's `change_parameters`, a contract generic,
  returning the LAST iterate — for a fixed-point iteration that re-linearizes its residuals each iteration, where
  no cost is comparable between iterations; `StopWhenParameterChangeLess`). `MixtureEmConfig` gains the field
  (positional order `model_update, nuisance, stop, max_iter`) and `PosteriorIrlsMethod` a third, `stop`
  (`PosteriorIrlsMethod(ws, config::MixtureEmConfig)` builds the EM's). The incumbent rule was measured on the
  division camera and rejected: it stopped the contaminated stacks at their second iteration, f 2 % off, and
  diverged 79 of 100 stacks at 8 % gross outliers (0 before).

- **New: the Student-t scorer's table, its priors required keys** (the owner's ruling (c)). `scoring_from_toml` reads
  VisualGeometryToolkitCore's `StudentTUniformScoring` (moved there from RANSACScoringDoneRight):
  `{ type = "student_t_uniform", outlier_halfwidth = h, scale_prior = "jeffreys" | "ig" (alpha0, beta0),
  label_prior = "flat" | "profiled" | "beta" (alpha, beta) }`. The stack's default priors (Jeffreys, flat) live in
  `configs/default.toml`'s new `[scoring.student_t_uniform]`, the one defaults place: a key the table leaves out
  inherits it there, as a strategy's keys inherit `[local_optimization.<type>]`; nothing is defaulted in code. A
  prior's parameters are required with it and refused with another. `config_dict` writes both priors, and reads back.

- **Faster: `refine_camera_and_poses`' residual blocks read their variables as concrete types** (the problem's
  variable vector holds a Union; each block asserts its camera chart's and pose charts' types), so the block's
  residual and ForwardDiff Jacobian no longer box every variable: a warm 3-board, 6-view refine (756
  correspondences) allocated 22 MB, now 17 MB. Core's restricted camera chart (`model` of a `SubsetManifold` of a
  `ParameterManifold`, 128 B a call) and `reprojection_error` (32 B) still allocate per evaluation.
- **New: `scoring_from_toml(table; problem, section, tags)`**, the scorers' TOML reader on the local optimization's
  schema (the owner's one-schema policy): `{ type = "msac", threshold = 3.0 }` through a scoring registry
  (VisualGeometryToolkitCore's `SCORING_TAGS`, or a downstream merge), every key required, unknown keys refused;
  `config_dict` writes a scorer at its threshold back. A downstream scorer adds its method of
  `scoring_from_toml(T, table; problem, section)` (a contract generic).
- **New: a robust refine, `refine_camera_and_poses(…; robust, scale)`** (the owner's one-schema policy): `robust` takes
  the RANSAC local optimization's own strategies, `Irls{Shape}` and `Triggs{Shape}` with Core's loss shapes and a
  `scale_multiple` (or an absolute `loss_width`), read by the same TOML reader (`local_optimization_from_toml`). From the
  least-squares estimate, Triggs runs one robustified LM and Irls weighted solves, at the loss width the strategy's
  multiple of a FROZEN scale gives: the caller's `scale`, else Core's Student-t scale of the least-squares rows
  (`partition_student_t`). No new loss, strategy or scale rule. A strategy without a width of its own (it would follow a
  scorer's threshold), or one that needs a RANSAC consensus set, is an `ArgumentError`. The metadata adds
  `residual_norms`, `weights` (ρ′), `scale` and `loss_width`. `robust = nothing` (the default) is the least-squares
  refine bit for bit. Both forms (one board, several boards); parametric cameras.

- **`SimFitPP`'s defaults are the TOML's** (robust-schema audit): its keyword defaults (`alpha`, `max_iter`, `ftol`,
  `train_fraction`, `min_loss_width`, `max_loss_width`, `dg`) read the new `[simfitpp]` table of
  configs/default.toml through `_dflt` (the reference's values, unchanged); the completeness test holds the table to
  the struct's fields.

- **The division camera's branch audit trims through the stack's loss** (robust-schema audit): `_branch_residual`'s
  two trimmed depth refits keep the rows inside a TLS loss at `scale_multiple` × σ̂ (σ̂ Core's `noise_sigma` of the
  lift's residual norms, codimension 2), the multiple the new `[fit_division_camera] branch_audit_scale_multiple`
  (3·√chisq_median(2), three medians), in place of the private cut `9 × median(r²)`. The held posterior was not
  taken: the audit exists for the views whose inliers the posterior has rejected. Every branch decision of the suite
  is unchanged (91 audits replayed, 14 of them switches; 356 of 364 candidate scores bit-identical, the rest within
  4e-15 relative). fit_division_camera's posterior EM is untouched.

- **The radial homography's lifted seed reads its gate from the TOML** (robust-schema audit): `lift_init` of a
  plain-homography hypothesis seeds k on the correspondences inside a TLS loss at `scale_multiple` × σ̂ (σ̂ Core's
  `noise_sigma` of the plain-H reprojection norms, codimension 2; the multiple the new table
  `[pipelines.radial_homography.lift_seed]`, 3·√chisq_median(2) — three medians, the former private `3 * median`).
  The scorer's decision point was not taken: the gate is relative to the hypothesis' own residuals, not an absolute
  width. The partition equals the former one on the test scenes (to rounding of the multiple).

- **Breaking: `fit_linear`, `fit_rotation` and `fit_rotation_averaging` take the stack's one robust-estimation schema,
  an `Irls{K}` strategy** (`local_optimization = Irls{HuberShape}(…)`; owner policy, robust-schema audit). Their
  private keywords `loss`, `scale_multiple` and `max_iter` are gone (a `MethodError`): the loss is the strategy's `K`,
  the iteration cap its `max_iter`, the width its width mode — `scale_multiple = m` (m·σ̂ in the loss's currency,
  `plain_scale_multiple`: a Geman–McClure or Cauchy multiple is now a half-weight depth, `m_old =
  decision_width(shape, m_new)`), `loss_width = w` (new: an absolute width) or none (the shape's efficiency
  constant at the fit's codimension, as before; a fit has no scorer to follow). The width resolves through
  `lo_loss` (Core's `loss_at`) at the fit's σ̂. TOML: `[fit_linear]`, `[fit_rotation]` and
  `[fit_rotation_averaging]` replace `loss = "huber"` and `max_iter` with `local_optimization = { type = "irls",
  loss = "huber", max_iter = … }`, read by `local_optimization_from_toml`; `tol` stays. The private weight step
  `_irls_weights!`/`_irls_loss` is replaced by one that resolves the strategy (`_fit_weights!`). Results are
  bit-identical to the previous release (15 reference calls: every loss, width and iteration cap, the estimate,
  the counters and the metadata compared with `isequal`).

- **New: board poses in `refine_camera_and_poses(camera, poses, board_poses, views; held_boards, held_views)`** (owner
  decision 20): several rigid boards, each board → world pose a variable of the same bundle (one more pose chart of the
  same NLLSsolver problem and assembler as the camera and the view poses), the correspondences keyed by
  `(view, board)`. The gauge of each connected component of the view–board graph is fixed exactly by a held pose (by
  default its first board; `held_views` holds a view instead), no pin and no prior; a component without one is an
  `ArgumentError`. The single-board refine is the same bundle with one held identity board. CaliBlobs' own refine
  moves onto it.
- **Breaking: `fit_camera(SplineLens, …)`'s forward fields are one nested `forward`** in its metadata family:
  `family.forward` is `(; spacing, counts, λ, outer_iterations, max_px, max_px_edge, rms)`, or `nothing` with
  `forward = false` (seven `nothing` fields before: `spacing`, `counts`, `forward_λ`, `forward_outer_iterations`,
  `forward_max_px`, `forward_max_px_edge`, `forward_rms`). The refine's own `forward_max_px` metadata is unchanged.
- **Changed: `fit_camera` and `pspline_grid` read every camera source's quadrature rays and held-out Jacobians
  through Core's batched `backproject` / `backproject_with_jacobian` on `device`** (a parametric source ran as a host
  map, and its Jacobians always did). No per-lens-type branch; values unchanged on the CPU.
- **New in `synthetic_scenes()`: `EpipolarScenes.box_point_far(rng)`**, the wider, farther box of the stereo pixel
  scenes, which this package's F tests and LocalOptimizationDoneRight's problem arms each spelled as `_box`.

- **Tests: the call set's timed runs are call_set.jl's too** (`call_set_size`, `run_call_set(call, pairs)`,
  `time_call_set(call, pairs)` and `call_set_arm(make, pairs, reps)`, an arm as the bench record measures it): the
  timing benches of this package and LocalOptimizationDoneRight (whose accv2026 bench restated `run_set`/`time_set`)
  build their arms on them.

- **Tests: the recorded LO call set has one reader, test/lo_timing/call_set.jl** (`CALL_SET_FIXTURE`,
  `read_fixture`, `pool_csponds`, `problem_of`), which this package's timing bench and LocalOptimizationDoneRight's
  include from `pkgdir(VisualGeometryToolkitEstimation)` (the latter carried its own copy of the reader and of the
  fixture file).

- **New in `synthetic_scenes()`: `EpipolarScenes.contaminated_pixel_pairs(rng, n; planar)` and `PIXEL_PAIRS_K`**: the
  contaminated two-view pixel scene of the block-coordinate-descent `Recorded` tests (a homography scene with
  `planar = true`, else F/E; 40 % gross outliers), which LocalOptimizationDoneRight's recorded-BCD test restated as
  `two_view`.

- **New in `synthetic_scenes()`: the scene builders downstream tests restated** (LocalOptimizationDoneRight's
  accv2026 test_problem_arms.jl copied them from this package's tests): `PoseScenes.lens_board_scene(rng, camera, R,
  t; σ, n_out)` (the absolute-pose board through a distorting lens), `DivisionBoards.radial_board(rng; f, k, n_out, σ)`
  with `RADIAL_BOARD_FRAME`/`RADIAL_BOARD_CENTER` (one board for the radial camera), a new family `RadialScenes`
  (`RADIAL_HOMOGRAPHY_FRAME`, `radial_homography_truth(k_norm)`, `radial_homography_pairs(rng, model; n_in, n_out, σ)`),
  `Contamination.contaminated_line(rng)` (the contaminated line problem and its seed), and `EpipolarScenes.box_point`
  takes its box as keywords (`width`, `height`, `near`, `far`; the default box unchanged). Every draw is in the order
  the private copies drew it, so the pinned tests keep their values; Contamination now loads first (the other
  families build on it).

- **Text: LocalOptimizationDoneRight's strategy registry is `STRATEGY_TAGS`** (renamed from `LO_TAGS` in its 0.9): the
  `scale_multiple = "cross_validated"` refusal of `local_optimization_from_toml`, its docstring, the guides and
  CLAUDE.md name the merge `merge(VisualGeometryToolkitEstimation.LO_TAGS, LocalOptimizationDoneRight.STRATEGY_TAGS)`.
- **Breaking: a SplineLens refine's gauge is fixed by its parametrization, and only the live controls are variables.**
  `refine_camera_and_poses` on a SplineLens camera held its rotation and scale gauges with zero-cost pin blocks and its
  unread coordinates (pad slots, absent nodes, nodes no row reads) with anchor rows `coordinate − guess`. Now each patch
  variable is Core's `restrict` of its chart to its live controls' coordinates (a static index: NLLSsolver assembles its
  system from statically sized blocks; a block's patches are ordered and typed by their restriction, so the compiled
  block types stay few; a patch with none is no variable), the
  view holding the rotation gauge has a translation-only pose chart and the scale node no log-scale coordinate; a block
  reads only the 2 × 2 to 3 × 3 patches its window lies in. The patch `offset` layout knob is gone. The spline metadata
  loses `cost_before_gauge`, `cost_after_gauge` and `anchor_cost` (each an invariant the parametrization now guarantees);
  `cost_start` is NLLSsolver's start cost of the joint solve. The estimates agree with the pinned form to the solver's
  tolerance (3-view fixture: controls within 3e-7 under GCV, 1e-8 at β = 1e-8; GCV's β to 1e-5 relative; the noisy GCV
  case to 2e-11), and the point-does-not-project condition travels as a status, stated once at the public boundary.
- **Fixed: GCV's start solve cut short by `max_time`** returned the β = 1 estimate with the searched β and tr H in its
  metadata; it now reports β = 1, no curve and no tr H. **Fixed: `windows_settled`** was `true` for a solve cut short by
  `max_time`, whose windows were never rebuilt; it is `false`.
- **Faster: a SplineLens refine's rebuilds.** A rebuild with its weight given forms no unit weight (the data's
  Gauss–Newton diagonal and the Δ³ operator's mean eigenvalue, a sparse QR); the correspondences' projections that
  build each problem run batched on the device (Core's projection kernel), a rebuild's warm-started at the last build's;
  a block's projection solve is reused between NLLSsolver's trial cost and its linearization at the same point; its
  gradient is lazy like its normal matrix, and its tangents are formed only for the controls it reads.
- **Breaking: `gcv_curve` loses its `hold` keyword** (both forms; the refine's anchor rows were its only user). The rows
  form past `exact_max` now drops the parameters nothing reads, as the pencil already did (it threw "not positive
  definite" on one), and factors its sparse matrix with one symbolic analysis for every λ. Both forms share one pencil
  curve.
- **Internal: dispatch and the graph's own refusal.** `fit_camera`'s decentering reads Core's `RadialSymmetry` trait by
  dispatch (was an `isa` branch); `fit_rotation_averaging` refuses a disconnected graph through `require_connected`,
  which names the components.
- **Internal: Core's primitives.** `fit_camera`'s spacing search counts its nodes with Core's `_node_counts`, a
  source ray's off-axis angle is Core's `angle_between`, SIMFIT++'s naive σ² divides by Core's `chisq_median`; the
  full-lattice Δ³ operator's mean eigenvalue (a sparse QR) is formed once per lattice size, not at every spacing tried.
- **Internal: one rows-and-Jacobian assembler.** The three refines' stall postconditions (the parametric and spline
  camera refines, `refine_robot_world_hand_eye`) read Core's `nlls_rows_and_jacobian`; the refines share one input
  contract (`_pose_charts`, `_correspondence`, `_plain_pose`).

- **New: `synthetic_scenes()`** (public, not exported; a Test extension, `using Test` loads it): the stack's shared synthetic
  test scenes, the module whose families `EpipolarScenes`, `HomographyScenes`, `PoseScenes`, `DivisionBoards`,
  `P3PScenes`, `Cameras` and `Contamination` export their seeded builders (`calibrated_pairs`, `stereo_pixel_pairs`,
  `homography_pairs`, `frustum_point`, `board_camera`, `noisy_board_views`, `p3p_scene`, `contaminated_pairs`, …). They
  were this package's private test setups, which LocalOptimizationDoneRight's paper tests copied; every stack package's
  tests now draw from the one definition (the package's own setups re-export them). The draws are unchanged.
- **Internal: one guarded entry for every `find_*` / `refine_*`.** The minimal-sample guard each entry point restated
  (`_holds_minimal_sample(P, n) || return _too_few(P, n)` before its pipeline) is `_guarded_ransac` /
  `_guarded_refine`; the two pose refines, whose result is relabelled, keep the guard at their call. A view graph's
  loop and orientation angles are `Rotations.rotation_angle` (was `‖log R‖` by hand).
- **Internal: one IRLS weight step.** `fit_linear`, `fit_rotation` and `fit_rotation_averaging` each restated the same
  reweighting (σ̂ = median/√χ²_d-median, the loss deployed at σ̂², its weights times the prior) with only the
  codimension differing; they share `_irls_weights!`, whose σ̂ is Core's `noise_sigma`. The results are unchanged bit for
  bit.
- **Docs: `Triggs`'s Huber second derivative.** ρ″ is zero inside the width only; in the tail it is `−w/(2s^{3/2})`,
  as Core's `loss_robustifier` gives it, so the Triggs term corrects the tail (the docstring said zero everywhere).
- **New: SIMFIT++ loss-width estimation, moved from RANSACScoringDoneRight** (Edstedt, "Less Biased Noise Scale
  Estimation for Threshold-Robust RANSAC", CVPRW 2025; a step-for-step port of github.com/Parskatt/simfitpp):
  `SimFitPP` (the reference's defaults), `simfitpp_loss_width(fit, residuals_sq, n, initial_loss_width, cfg; ...)`
  over a caller's estimator callbacks and `simfitpp_select` on a pool's residual matrix or a `PoolPair` (pixels in,
  pixels out) are exported; `unbiased_median_variance` (the reference's `UnbiasedMed`, σ²) and `with_residual_unit`
  are public. The train split's MSAC-best pool model is scored by Core's threshold `score!`, bit for bit the sum
  RANSACScoringDoneRight's copy computed. New dependency: Distributions (the χ² noise model). The stack's dependency
  allow-list gives RANSACScoringDoneRight Estimation, its consumer.
- **Fixed: a SplineLens `refine_camera_and_poses` always returns.** Its blocks hand NLLSsolver their linearizations
  through Core's `finite_linearization` (a non-finite one made NLLSsolver 5.3.2's damping loop spin: a fixed
  `penalty = 1e-9` on noisy detections ran > 60 min on dgx01), its rebuild rounds are a bounded loop, and a lattice
  that folds where a point projects (the backward's Jacobian exactly singular) raises the not-projected error, naming
  the fold, instead of being linearized.
- **Compat: NLLSsolver ≥ 5.3.3** (NLLSsolver#68: the damping loop stops at a non-finite step).
- **Breaking: a view-graph edge's weight and transform are Core's `weight_of` and `value_of`** (the private
  `_edge_weight`/`_relative` are gone; an edge without `:weight` still weighs 1). `fit_rotation_averaging`'s spectral
  seed hands Core's `spectral_rotation_averaging` the live (weight > 0) `Attributed` edges directly.
- **Breaking: `PosteriorIrlsState.best` is Core's `Attributed(θ, (; cost))`**, formerly an `Incumbent`.

- **New: view graphs.** `ViewGraph` (frames and their verified pairs, Graphs.jl underneath; an edge is the pair's
  relative transform or Core's `Attributed(transform, (; weight))`, a bare transform weighing 1) with
  `spanning_tree`, `chain`, `filter_cycles` (Theia's rule: an edge stays when one of its triangles closes to `tol`),
  `rotation_loop_angle` and `filter_by_orientation`; `ObservationGraph` (frames × what they observe) with
  `components` (largest first) and `covisibility(og; side)`, whose edges are `Attributed(shared, (; weight))`.
- **New: `filtered_rotation_averaging(vg; cycle_tol, orientation_tol, kwargs...) -> (; fit, graph)`**: the cycle
  filter, the averaging, the orientation filter and a refit, tolerances in radians.
- **Breaking: `fit_rotation_averaging(relative; loss, scale_multiple, max_iter, tol, seed)`** takes a pair's weight
  from its edge (`Attributed(R, (; weight))`) and no positional prior or `j_from_i` keyword; it also takes a
  `ViewGraph`. `seed = :spectral | :spanning_tree | Dict`, and every seed is re-gauged so the first sorted frame is the
  identity. The per-edge IRLS weights are the metadata's `irls_weights`.
- **`fit_camera`'s warning of a source's non-radial terms reads Core's `RadialSymmetry` trait** (a lens type
  declared `RadiallySymmetric()` is not measured for decentering); `_decentering` still measures the magnitude.
- **Breaking: `fit_camera(SplineLens, source, region; tolerance, grid_size = nothing, forward = true)`** fits ONE
  camera: a `CameraModel` with a `SplineLens`, its K the source's own (skew 0). Its backward spline is the P-spline
  of Core's full-image lattice (`spline_lattice`) on the source's rays at x = K⁻¹u — at `grid_size` when given, else
  at the lattice the spacing search chooses to `tolerance` (the held-out direction error in pixels, as before). A
  SplineLens source's backward is kept exactly and only the forward is fitted. Its forward spline (`forward = true`)
  is a `ChartSpline{2}` on the equidistant chart about ẑ (Core's one derivation, `_forward_chart`), fitted to the camera's own rays by the same
  solve and the same spacing search, to `tolerance` in pixels of the held-out round trip; the camera then projects
  through it. The metadata `family` is the union of the old central grid's and the old forward map's records
  (`λ, grid_size, stride, outer_iterations, n_held_out_dropped, tolerance`, then `spacing, counts,
  forward_λ, forward_outer_iterations, forward_max_px, forward_max_px_edge, forward_rms`, `nothing` without a
  forward). It replaces `fit_camera(CentralGenericModel, …)` and `fit_camera(SplineProjection, …)`, both deleted. A
  0.8 `SplineProjection`'s spline is not a valid forward (its chart is about the region-centroid ray, the forward's
  about ẑ): refit with `fit_camera(SplineLens, …)`, never reuse it.
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
- **`fit_camera`'s P-spline weight is in the stack's one convention**: `λ = β·penalty_scale(tr AᵀA, nodes, m̄)` with
  β = 3e-7 (`_PSPLINE_BETA`) and m̄ the Δ³ operator's mean positive eigenvalue (≈ 31 on a 12 × 10 lattice, → 40 on
  large ones), in place of `1e-8·mean(diag(AᵀA))`: λ is 3e-7/m̄ ≈ 0.75–1 × the former one, so the fitted lattices
  move within the fit's tolerance (the metadata's `family.λ`, formerly `penalty`, and `forward_λ`, formerly `forward_penalty`).
- **New (public, not exported): the stack's one generalized cross-validation of a penalty weight** — `gcv_score(n, rss,
  dof)`, `gcv_curve(G, h, rr, Q, q, n; hold)` and `gcv_curve(J, r, D, c; hold, exact_max = 3000, probes = 64, seed)`
  (the closed-form GCV curve of one penalized least squares, and its solution `y` at each λ: ONE generalized eigendecomposition of the pencil
  `(DᵀD, JᵀJ + RᵀR + DᵀD)` up to `exact_max` parameters, else per λ a sparse Cholesky and a Hutchinson trace from a fixed
  seed; `hold = R` rows that are in the system but not data; no ridge: a gauge is refused, `ArgumentError`),
  `gcv_select(score, scale; log10_bracket, points)` → `(; β, λ, gcv, rss, dof, log10_β_bracket, at_bracket_end)` (a
  coarse scan of log₁₀ β from the top down, then Brent's method through Core's `grid_brent_minimum`, a least value at
  a scan end refined on its one interval; `at_bracket_end` when the result lies at the end) and
  `check_penalty_weight(w, what; allow_nothing = false)`. The weight's normalization, `penalty_scale` and
  `mean_positive_eigenvalue`, is Core's (VisualGeometryToolkitCore 0.9; not re-exported here). Moved from
  RadiometricCamera's linear solves (its `_gcv_curve`, `gcv_select`, `_brent_minimize`, `_mean_positive_eigenvalue`,
  `_check_penalty_weight`: its copies go when it adopts these); `refine_camera_and_poses` on a SplineLens camera uses
  them too.
- **`refine_camera_and_poses` refines a SplineLens camera's controls with its poses, K frozen** (bit-identical): the
  present nodes' backward controls, any norms (a `fit_camera` lattice, a quasi-interpolant), each `c = e^ℓ·D` its
  direction on its sphere (Core's `ControlPatchManifold`) and its log-scale, grouped into 2 × 2 patches (NLLSsolver's
  10-variable block cap); one block per correspondence on its window's 3 × 3 patches and its pose, its Jacobian
  analytic by implicit differentiation of the projection solve at x* (Core's `_solved_projection` and
  `_projection_jacobians`: no ForwardDiff reaches the solve); a scale-free third-difference penalty over the present
  nodes (Core's `difference_rows`, each row divided by the mean control length of its own 4 nodes, its denominator in
  the analytic Jacobian), weighted by the stack's normalized β (`λ = β·energy/N/m̄`, the data's Gauss–Newton energy per
  direction coordinate and the Δ³ operator's mean positive eigenvalue; Core's `penalty_scale`) set by `penalty`:
  `:gcv` (the default) chooses β by generalized cross-validation, `GCV = n‖r‖²/(n − tr H)²`, of the refine's
  linearization: ONE nonlinear solve at β = 1 (the bracket's top, where no weakly held node swings), the GCV curve of
  its Gauss–Newton step for every λ from one decomposition (`gcv_curve`: the pencil up to 3000 coordinates, else a
  sparse Cholesky per λ and a 64-probe Hutchinson trace with a fixed seed; the gauge's held coordinates removed, the
  anchors a hold), `gcv_select` over 1e-12…1 (one β a decade, then Brent's method), then ONE final solve at the chosen
  β from the linearization's Gauss–Newton step there (two full solves in place of the per-weight search's 13–17), a number fixes β (GCV off; a small one that lets noisy data swing a
  node until a correspondence no longer projects is an `ArgumentError` naming `:gcv`), and `0` or `nothing` turns the smoothness off (the nodes no correspondence's window reads then held at their guesses); the
  poses refined alone first, then the directions with the scales held exactly (a directions-only patch variable, no
  weighted rows), then everything; the blocks rebuilt at the
  estimate until every window is its own; the gauge (the rotation and the common scale, both exact symmetries)
  held exactly during every solve (the first view's rotation and one node's log-scale get the step 0, no cost row,
  so the normal matrix is full rank), and the estimate moved after the solve, in one step, to the canonical
  representative (the lens axis onto ẑ, the roll putting the x-derivative at x = 0 in the xz plane, the live
  controls' mean length the guess's; the cost unchanged). `result.spline` carries the weight `β` and `λ`,
  the GCV curve `[(; β, λ, gcv)]` (every weight evaluated), whether its minimum is a bracket end (`gcv_at_bound`),
  tr H (`effective_dof`) and the present nodes no row reads
  (`n_nodes_unrefined`). `free` is `(:controls,)` or `()` (the
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
