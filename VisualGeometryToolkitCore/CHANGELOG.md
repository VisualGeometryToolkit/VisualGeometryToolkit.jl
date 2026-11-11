# VisualGeometryToolkitCore changelog

## 0.9.0 (unreleased)

Breaking (the next release is a minor bump). A bounded camera carries its LENS COVERAGE — where the lens covers its
sensor — so every consumer asks the camera what it images instead of supplying an image circle of its own.

- **Changed: `pool_frame_calibrations` moves K through `reframed_calibration`**, the one change of image frame of a
  calibration, instead of multiplying by the stored map by hand. The values are the same bit for bit; a stored map
  that is not an isotropic scale and a translation (the pool frame always is) is now an `ArgumentError`.
- **Changed: `spectral_rotation_averaging` no longer checks that its pairs connect every frame.** The caller checks it on
  its graph — VisualGeometryToolkitEstimation's `fit_rotation_averaging`, its only stack caller, already refuses a split
  `ViewGraph` before seeding — so the hand-written breadth-first search that ran the check a second time is deleted. A
  direct call on a split graph (two components, or a zero-weight bridge) now returns meaningless rotations instead of
  an `ArgumentError`; check connectivity first.
- **New: `test_docs(M; exceptions, backlog, signature_exceptions, dirs)`** (public, not exported; Test extension): the
  docstring gates every stack package's docs item runs — every public name documented, no docstring detached from its
  definition, every docstring signature accepted by a method — once. They were copied into each package's
  `test/docs_gates.jl` (seven drifted versions across the stack); the copies are deleted. Core's version is the one
  kept: it ignores methods a test item defined under `Main` (the result no longer depends on which items shared the
  docs item's worker), scans a documented module's body, and keeps a signature block across a blank line.
- **New: `check_duplication(M)` / `check_duplication(dir)`** (public, not exported; Test extension, beside
  `check_naming` and `check_raw_decoding`): the no-duplication rule, checked. (1) Name ownership: a binding the package
  owns (not an import, not a method extension) whose name an upstream stack package owns, private names included;
  upstream is every dependency that is Core or depends on it. (2) Function-body similarity: JuliaSyntax tokens, comments
  dropped, identifiers and strings normalized, 8-token shingles, Jaccard ≥ 0.6 with ≥ 60 tokens or ≥ 0.85 with ≥ 25
  (keyword `similarity`), within the repository (a monorepo's other packages included) and against the stack
  repositories in `VGTK_STACK_PATHS` (keyword `stack`; absent ones skipped with a note). (3) Whole files whose code
  tokens are equal. (4) Retired patterns, each naming its owner: `NLLSOptions(` (`nlls_options`), `acos(clamp((tr(`
  (Rotations' `rotation_angle`), the sRGB transfer constants (Features' `linear_image.jl`), a `\newcommand` writer
  (Eval's number macros). Exceptions, each with a reason, live in the repository root's `DUPLICATION.toml`; an entry
  that matches nothing is itself a finding. The whole monorepo scans in about 1.5 s.
- **Fixed: `check_raw_decoding`'s display-mode exemption is the call's own.** `mode = :demosaic` is taken out only
  inside a `load_raw(`/`LinearRaw(` call's parentheses, over every line the call spans: a call written over several lines
  is no longer flagged, and a demosaic by another route on the same line as such a call no longer goes unflagged. `allow`
  takes one exception given bare (a path or a Regex) as well as a collection.
- **Fixed: a SplineLens camera images its far edge.** The backward box is closed on both sides, `[min, max + 1]` per axis
  with the same rounding slack at both edges, so `backproject` is finite on the image's right and bottom edges (`x = W`,
  `y = H`) as on its left and top, where `valid_domain` already said it was; the non-central model shares the test.
  Before, the upper bound was strict and carried no slack, so a consumer that needs a pixel's four footprint corners lost
  the last column and row. The projection LM's box is the same closed rectangle (`ub = max + 1`, was `max + 0.999`), so
  a ray imaged on the far edge projects onto it.
- **New: `normal_cg(N, g; diagonal, x0, rtol, max_iter)`** (public, not exported, beside `cgls`): conjugate gradients
  on a penalized least squares' normal operator `AᵀA + DᵀWD` given applied (a function `N(y, x)` or an operator with
  `mul!`), never its stored rows, on `g`'s KernelAbstractions backend, with a warm start and cgls's stopping rule
  relative to `‖Aᵀb + DᵀWh‖`. `cgls` and `normal_cg` are one iteration (one loop, one stopping rule and iteration
  count), and both take `diagonal`, the normal matrix's diagonal, as a Jacobi preconditioner. The solve a panorama's
  vignetting fit makes on its accumulated stencil and Schur block (PanoramaStitching's `_pcg`).
- **New: a bounded camera and a posed camera move to a device** (`Adapt.adapt(backend, bc)`): the wrapped camera's
  arrays move (a spline lens's controls and presence mask), the bound, the lens coverage and the pose travel as they are,
  so a kernel reads where the camera images (its lens coverage and its backprojection) without the bound being stripped.
- **New: `with_directions` of a SplineLens camera**, the backward controls replaced (K, the lattice and the presence mask
  kept; the forward spline, which inverts the old directions, dropped), and **`with_present`/`with_directions` of a
  `BoundedCamera`**, which rebuild its spline model and keep the bound, the lens coverage and a pose. The non-central
  model's rebuilders and these share one control replacement.
- **Fixed: a chart-map surface's ray root resolves a nearly flat surface.** The generic root of a `MappedSurface`
  (`intersection`, `chart`) stopped at a residual below √eps, so a plane-hit seed with heights of 1e-9 to 1e-7 of the
  chart's size was returned unmoved (the hit moved 0.89 of the true displacement at 1e-9) and finite differences of the
  hit by the heights failed. The root is now resolved to rounding of the chart's scale; √eps of its diagonal stays the
  acceptance only. It agrees with the height field's 1-D fast path to rounding at every height.
- **Fixed: a thin-plate kind's parameter derivative is the derivative of its projection.** `with_parameters` of a
  `HeightField` or a `ThinPlateSpline` projects its values onto pure bending, so every perturbation off that subspace
  is removed; but at pure-bending values `tps_pure_bending` kept a ForwardDiff Dual's partials unprojected, and
  `transform_deriv_params` was the bare nodal weights, so the Jacobian along a node's coordinate disagreed with central
  differences of the function (75 % at a field's own heights). The values are still kept bit for bit and the partials
  are projected; `transform_deriv_params` is the nodal weights through the projection, `Pᵀn`. On every pure-bending
  direction (the target manifold's) nothing changes; in the stored coordinates each spline's affine directions are now
  null (a height field's 3, an in-plane spline's 6), as the function they differentiate has them.
- **The stack's dependency allow-list (`configs/stack_dependencies.toml`) gives RANSACScoringDoneRight
  VisualGeometryToolkitEstimation**: SIMFIT++ moved from RSDR into Estimation, and RSDR's grid family of it consumes
  Estimation's `SimFitPP`/`simfitpp_select`.
- **New: `finite_linearization(cost, g, H)`** (public, not exported, beside `finite_cost`): a residual block's
  linearization as the stack's NLLSsolver fits read it, `(Inf, 0, 0)` when any part is not finite. NLLSsolver 5.3.2's
  Levenberg–Marquardt damping loop spins on a non-finite gradient or Hessian at an accepted point (NLLSsolver#68,
  released in 5.3.3); the chart LM's rows return through it.
- **Compat: NLLSsolver ≥ 5.3.3**, the release with #68: its damping loop stops at a non-finite step, so every stack LM
  is covered, not only the rows that return through `finite_linearization`.
- **New: `check_raw_decoding(dir)` and `RawDecodingViolation`** (public, the Test extension, beside `check_naming`):
  the stack's one RAW path. Flags, in code outside VisualGeometryToolkitFeatures' raw module, any use of rawpy, a
  LibRaw C call, dcraw, rawpy's `postprocess(`, a demosaic, a Bayer/CFA interpolation, or a decoded RAW output re-read
  with `PNGFiles.load`/`Gray.(…)`, in Julia, Python, shell, TOML and YAML code. The Features API's display mode,
  `load_raw(…; mode = :demosaic)` or `LinearRaw(…; mode = :demosaic)`, is the approved path and is not flagged;
  `allow` takes a repository's rare genuine exceptions (paths or `Regex`es).
- **New: `reframed_calibration(S::SimilarityMap{2}, K)`**, a calibration matrix carried by a similarity of the image
  plane (`to_homogeneous(S)·K`; a rotated `S` is refused by K's structure check). The pixel-frame change
  `reframed_calibration(from, to, K)` is this map form with `SimilarityMap(EuclideanMap(from, to))`: one formula.

- **New: `weight_of(x)` and `value_of(x)`** (public, not exported): a datum's weight (the `weight` attribute of an
  `Attributed(x, (; weight))`, as `Float64`; `1.0` for a bare value or one without `:weight`) and its value. Weights
  travel with their data as `Attributed` values; `conventions/data-arguments.md` states the rule (a fit returns a
  `FitResult`, several fits a `Dict` keyed by what was fitted, per-datum outputs index-aligned in the metadata and
  attached by the caller with `withattrs`).
- **Breaking: `zhang_intrinsics(Hs[, pitch]; center)` reads each homography's weight off the data**: a homography is
  bare (weight 1) or `Attributed(H, (; weight))`. The `weights` keyword is gone.
- **Breaking: `spectral_rotation_averaging(relative)` reads each pair's weight off the data**: a relative rotation is
  bare (weight 1) or `Attributed(R_ij, (; weight))`. The positional `weights` dictionary is gone.
- **Breaking: `Incumbent` is retired**: the best iterate of a descent is `Attributed(θ, (; cost))`, and
  `keep_best(best, θ, cost)` takes and returns one (`best.value`, `best.cost`).
- **Breaking: the `matches` recipe has no `weights` attribute**: in its fixed-color mode a match's alpha is its target
  point's `weight_of` (`Attributed(x, (; weight = p))`; a bare point is opaque), and points are drawn at their
  `value_of`.

- **`spectral_rotation_averaging` computes only its three leading eigenvectors** (the symmetric eigensolver's range
  form): the same rotations, about 4× faster at 300 frames.
- **Breaking: a placed target's placement is Core's `SimilarityMap{3}`** (scale the chart's side, the identity
  rotation, the translation `(origin, 0)`): `PlanarTarget(…; placement = S₃)`, plain numbers in the target's own form
  (metres for a physical target), a unit move rescaling it; a rotation or an offset along the normal is an
  `ArgumentError`. A placed target's point is the composition `S₃ ∘ lift ∘ C ∘ S₂⁻¹` of the design point (its plane
  point the design point to rounding, no longer bit for bit). It retires `ChartPlacement` (no alias); an unresolved
  placed target takes no unit form from its placement. VisualGeometryToolkitIO's record keeps its
  `"placement": {"origin", "side"}` keys.
- **Breaking: `MappedSurface` replaces `HeightSurface`** (and `ImplicitRoot` / `implicit_root`): a deformed posed
  target's surface is the surface of ONE composed chart map, `pose ∘ S₃ ∘ lift ∘ C ∘ S₂⁻¹`, every member a
  CoordinateTransformations map with its `transform_deriv`. `surface(pt)` gives a `Quad` for a planar affine
  calibration and a `MappedSurface` for any other — **an in-plane `ThinPlateSpline`, alone or under a `HeightField`,
  is now supported** (formerly an `ArgumentError`). `MappedSurface(pose, calibration, placement, rect)` is the one
  builder (the device form of its parts too). Its intersection is a 2 × 2 root in the design chart behind one
  internal function, solved by Core's one static LM (`_spline_lm`, now generic in the float type: Float32 in a kernel,
  the SplineLens projection's Float64 operations unchanged) and differentiated by the implicit function theorem one
  Dual layer at a time (nested Duals exact). A height field alone or over the affine factors keeps a 1-D Newton as
  the root's specialization for its map type (plain numbers): 1.1–1.8 µs a ray against the 2 × 2 root's 17 µs, the
  hits within 8.9e-16 and every hit and miss the same.
- **New: the thin-plate kinds on the device.** `ThinPlateSplineSystem{N, T, M}`, `HeightField` and `ThinPlateSpline`
  take `Adapt.adapt_structure` (their matrices become device arrays, nodes and values stay isbits) and
  `convert_eltype` (a Float32 device form). `tps_coefficients` is a loop (no BLAS: allocation-free, kernel-safe, any
  element type), and the pure-bending projection is the new public `tps_pure_bending`: the host constructors and a
  kernel (`with_parameters` on an adapted kind) call the same two functions.
- **New: `HeightField(nodes, points .=> heights; smoothing = 0)`**, a one-step algorithm: the pure-bending
  `HeightField` fitted to scattered heights in the least-squares sense, the plane (the pose's piston and tilts)
  fitted and dropped.
- **Breaking: `CameraModel(K; lens, pixel_frame = :colmap)`**: the keyword naming the pixel frame K is stated in
  (`:colmap`, `:opencv`, `:julia`) is `pixel_frame`, formerly `image_origin` (no alias).
- **New: the `RadialSymmetry` Holy trait** (`RadiallySymmetric()` / `RadiallyAsymmetric()`, public, not exported:
  `VisualGeometryToolkitCore.Cameras.RadialSymmetry`) says whether a lens is radially symmetric about the principal
  point, by its type: every projection, no distortion, `Radial`, `FieldOfView` and `LiftLens` are; an `Additive`, a
  `∘` composition and a `Lens` are when every piece is; the default is `RadiallyAsymmetric()` (fail closed), so
  `SplineLens`, `Tangential`, `ThinPrism`, `Tilt` and any undeclared type are not. `RadialCamera(::PosedCamera)`
  dispatches on it, and a downstream lens type is admitted by declaring
  `RadialSymmetry(::Type{MyLens}) = RadiallySymmetric()`. It replaces the internal `_radially_symmetric`.
- **A SplineLens camera's K is checked wherever the camera is built** — `CameraModel(K, ::SplineLens)` itself, so
  every constructor, a stored camera store or rig sensor read back, a re-gauge and a device copy: skew 0 and positive
  focal lengths, else an `ArgumentError` (ONE check, the non-central model's too). A skewed K once built silently.
- **`CameraModel(camera, K′)` re-gauges a SplineLens camera to any skew-0 K′**, a moved principal point included: the
  backward area moves by the change of the normalized plane `x′ = x / s + (c − c′)/f′`, a forward's output with it;
  every pixel's ray and every projection are unchanged, a lens with a forward included. A parametric lens still
  refuses a moved principal point.
- **A SplineLens's forward spline is on the equidistant chart about ẑ** (fixed in the camera frame), not about the
  ray of the principal point, `normalize(backward(0, 0))`: no choice of K moves it, so a re-gauge keeps the forward
  exactly and a Float32 copy (`convert_eltype`) keeps its chart. On a gauged camera the two axes agree to rounding
  (measured: the fitted forwards' projections move by ≤ 9e-12 px, a forward on its own gauge by ≤ 1.4e-12 px). A 0.8
  `SplineProjection`'s spline is NOT a valid forward (its chart is about the region-centroid ray; reused, it projects
  wrong by pixels): refit with Estimation's `fit_camera(SplineLens, source, region; tolerance)`.
- **The off-axis gauge is refused where the principal point is read as the optical axis**: when a SplineLens camera's
  lattice does not image ẑ inside its calibrated area — the gauge's own fallback condition, one predicate shared with
  the gauge, so the principal point is the area's centre — `undistort`/`undistort!`, `distortion_map`, `valid_rect`,
  `undistorted_rect`, `angle_of_view`, `horizon_radius`/`monotone_radius`, `image_circle(camera, sensor, d)` and
  `view_overlap` throw an `ArgumentError` naming it, instead of centring on a ray the lens does not image.
- **A non-finite seed of the batched SplineLens `project`** (a previous frame's failed point) starts the solve from
  the backward area's centre, the scalar form's fallback, instead of returning NaN for a ray the lens images.
- **Owner decision (2026-11-10): the CPU timing gate "≤ the old grid" is waived for the SplineLens backprojection** —
  batched +2.8 %, scalar +0.3 % against the old central grid, within the re-measurement margin of the baseline host;
  recorded in test/spline_timing/baselines.toml, whose baselines the gate holds the code to.
- **Breaking: `BoundedCamera{C, B, R}`**: the fields `extent` and `sensor` (the extent stored twice when a sensor was
  given) become ONE `bound` (a `Size2{Int}` extent, or the `Sensor` whose resolution IS the extent), and a new
  `lens_coverage`: where the lens covers the sensor, a `Rect2`, `Circle` or `Ellipse` in `Float64` pixels, kept as
  the kind given, by default the sensor's own `Rect` (never "none"; a degenerate or non-finite shape is an
  `ArgumentError`). Both are type parameters, so a bounded camera is concrete and `isbits` with its camera. Construct
  with `BoundedCamera(camera, bound; lens_coverage)`; read through `extent`, `sensor` and the new `lens_coverage`;
  replace with the new `with_lens_coverage`. Every rebuild (re-posing, `with_model`, units, `convert_eltype`) keeps
  both. Code reading `bc.extent` / `bc.sensor` reads `extent(bc)` / `sensor(bc)`. (A sensor CROP is the bound and K,
  not the coverage.)
- **Breaking: `imaged_region(camera)`** is the sensor ∩ the camera's lens coverage ∩ the model's `valid_domain`; the
  two-argument `imaged_region(camera, circle)` is removed (put the circle on the camera: `with_lens_coverage`). It is
  a small two-type Union, kept on purpose: `valid_domain(::CameraModel)` is an `Ellipse` or `WholePlane` by the
  lens's coefficients.
- **Breaking: a SplineLens camera's `imaged_region(camera)` reads its lens's `valid_domain`**: where the lattice is
  defined (its calibrated area, the bounding rect of the fully present windows under a mask), the counterpart of a
  parametric lens's `valid_domain`.
- **Breaking: `view_overlap(cameras)` and `view_overlap(a, b)`** drop their circle arguments: each frame's footprint
  is its own `imaged_region`. The results for the same circles are unchanged bit for bit.
- **Breaking: `fov` is renamed `angle_of_view`** (it always returned an angle, the angle of view; "field of view"
  is the size it spans) and bounds the extent by the camera's lens coverage: each edge sample is read where the
  coverage ends on the way to it, so a circular fisheye's dark corners or a disc cut by a wide sensor give the angle the
  camera images. A camera whose whole sensor is covered reads as before, bit for bit. A principal point outside the
  coverage is an `ArgumentError`. No alias: `fov` is gone.
- **Breaking: `RadialCamera(::PosedCamera{<:CameraModel})` refuses a lens that is not radially symmetric** — a
  `Tangential`, `ThinPrism` or `Tilt` piece (alone, inside an `Additive` or in a `∘` composition) or a `SplineLens` —
  with an `ArgumentError`: such a lens moves points off their rays from the centre, so the radial camera is not exact.
  These were accepted silently before. The check fails closed: a lens or a distortion piece not known to be radially
  symmetric is refused; a `LiftLens` and a `Lens` of `Radial`, `FieldOfView` and `Additive`s of them are accepted.
- **Breaking: the grid camera is a `SplineLens`; the central and non-central grid models are retired** (plan 1b, the
  SplineLens fold-in). The central grid camera is an ordinary `CameraModel(K, SplineLens(…))`: a backward lens (a
  `ChartSpline` of rays over the K-normalized plane, `x = K⁻¹u`, K a skew-0 gauge) with an optional forward spline
  on the equidistant chart about ẑ. The non-central one is `NoncentralSplineModel`, on the same lattice.
  Retired, with no alias (replacement after the arrow):
  `CentralGenericModel` → `CameraModel(K, SplineLens(…))`, built from a source by `CameraModel(SplineLens, source,
  size; grid_size)`; `NoncentralGenericModel` → `NoncentralSplineModel`; `SplineProjection` → the `forward` of a
  `SplineLens`, REFITTED by Estimation's `fit_camera(SplineLens, …)` (an old spline is never reused as a forward); `GenericGridSpec` → `SplineLensSpec`, its TOML table type `central_generic` → `spline`;
  `GenericModelWindow` and `window(m, rows, cols)` of a grid → `window(cam, rows, cols)`, a SplineLens camera (or a
  `NoncentralSplineModel`) of the block; `default_generic_grid` → `default_spline_grid`; `generic_lattice` →
  `spline_lattice(SplineLens, source, size)`; `nominal_intrinsics`/`with_nominal` → `calibration_matrix(cam)`,
  `intrinsics(cam)` and the re-gauge `CameraModel(cam, K′)`; `with_domain` → `with_lens_coverage` (where light falls is
  the bound's); `interpolated_direction`/`interpolated_point` → the splines themselves, `cam.lens.backward(x)`,
  `m.points(x)`; `in_present_domain` → `all(isfinite, backproject(cam, u))`; `spline_support`, `window_controls`,
  `window_indices` → `spline_weights(cam.lens.backward, x)`, `directions(cam)` and `window_offset`. The accessors
  `directions`, `grid_size`, `window_offset`, `spline_weights` and `direction_tangents` live on the new types;
  `grid_size(m)` is `(columns, rows)`, the keyword's order. A SplineLens camera backprojects every pixel as the old grid
  model did (the golden values of `test/fixtures/spline_lens_golden.jl`), whatever its K.
- **Breaking: a SplineLens camera needs skew 0** (an `ArgumentError`), and K is frozen in every fit.
- **New: the lens's one Newton/LM** `_spline_lm` replaces the old grid's pixel LM (`_pixel_lm`,
  `_pixel_lm_increment`): scale-free constants, so `imaging_transform` and `project(cam, …)` agree, and the
  analytic direction Jacobian, from one sweep of the basis, serves the solve and `backproject_with_jacobian` (the lens's
  `LensDirection` trait: a backward lens differentiates its `backproject`, a forward lens its `project`).
- **Breaking: `undistort`/`rectify` of a SplineLens camera run through Core's one KernelAbstractions warp**
  (`image_warp`) on the destination's backend: exact on every backend (no sampled warp), the old grid's own kernels
  deleted.
- **New: `ka_upload` and `on_device`** (`base/device.jl`, beside `default_device`): the one upload to a device,
  `ka_upload(T, dev, v)`, and `on_device(dev, A)` (the array itself when it is already there). They replace
  VisualGeometryToolkitFeatures' `_ka_upload`, `_to_host`, `_upload_i32`, `_on_backend` and `in_kind_of`.
- **New: a `SplineLens` camera's K is a frozen gauge.** An internal trait (`IntrinsicsRole`: free by default, a gauge
  for a `SplineLens`) is the one switch of a camera's parameter protocol: for a gauge camera `parameter_names`,
  `parameter_vector`, `parameter_bounds`, `parameters` and `with_parameters` (K kept identical) read the lens's
  controls only. `monotone_radius(::SplineLens)` is the largest normalized disc about the principal point inside the
  lens's support, so `horizon_radius` reads a grid.
- **New: a `SplineLens`'s parameter protocol.** Its parameters are its backward controls at the present nodes (an
  absent node holds none), column-major over the grid, each node's x, y, z in turn, named `:c_<row>_<col>_<x|y|z>`,
  unbounded, in a heap `Vector`. `with_parameters` keeps the area and the mask, and refuses a lens holding a forward
  spline (`ArgumentError`: the forward is a fit of the backward and would be stale). `ParameterManifold`'s chart is
  static, so it refuses a model with a heap parameter vector (`ArgumentError`) — a SplineLens camera's
  `minimal_manifold` is not served yet; its controls' chart is a `ControlPatchManifold`.
- **New: `CameraModel(SplineLens, controls, size; calib_area, intrinsics, present, unit_controls)`**: the SplineLens
  camera of a given pixel lattice, as Schöps et al.'s `intrinsics0.yaml` and the stored grid records state one — its
  direction controls over a calibrated pixel area, K the given one (skew 0) or the lattice's gauge; each pixel
  backprojects as the pixel grid does, whatever the K. A lattice with no complete (fully present) 4 × 4 window at
  the gauge's pixel — ẑ's pixel, else the calibrated area's centre — has no local scale to derive K from: without a
  given K it is an `ArgumentError` naming that pixel and saying to supply K.
- **Internal: the implicit derivatives of a SplineLens projection** (`_projection_jacobians`, beside the direction
  Jacobian): `∂x*/∂t` and `∂x*/∂c` of a converged projection solve x* of the ray t, by the implicit function theorem
  on the direction residual — the analytic Jacobian of Estimation's `refine_camera_and_poses` on a SplineLens camera,
  so no ForwardDiff dual reaches the solve; and `_solved_projection`, THE solve it differentiates (from the area's
  centre: the LM returns a start within √eps unmoved, which would make x*(c) a step function). The direction
  differential also returns `‖S‖`; `_imaged_z` (the gauge's solve of ẑ) is `_solved_projection` too.
- **New (public, not exported): `grid_brent_minimum(f, xs; periodic = false)`**, the stack's one 1-D minimizer
  (numerics/minimize.jl): the least value of `f` on an evenly spaced grid, refined by Brent's method (golden-section
  steps with parabolic interpolation) between its neighbours to Brent's own floor (`√eps` relative: no tolerance to
  supply); on a linear grid a least value at an end is refined on its one neighbouring interval, and `at_end` says the
  result lies at the end; `Inf`/`NaN` values are never the minimum. `_azimuth_minimum` (the view-angle limit, the
  inscribed radius) reads it (its values move at rounding level from the former 40 golden-section steps), and so does
  Estimation's `gcv_select` (RadiometricCamera's scan-and-Brent search, whose `_brent_minimize` goes when it adopts it).
- **New (public, not exported): the stack's one penalized least squares**, `min ‖Ax − b‖² + λ‖D(x − c)‖²`, each piece
  with one owner (numerics/penalized_least_squares.jl):
  - THE OPERATOR. `difference_rows(mask | dims, order; axes = (1, 2))` takes any order ≥ 1 along chosen lattice axes
    (a full texel page `(TW, TH)` at order 1 is its gradient); `gradient_pairs(mask | dims)` is each node's isotropic
    pair of order-1 rows (an L0 or total-variation step reads both); `bending_rows` also takes a
    `ThinPlateSplineSystem` or a `SphericalSplineSystem` (the bending energy's rows, its null space without rows: the
    second operator kind, the form RadiometricCamera's focus charts build); `row_stencils(D)` is a sparse operator's
    rows as (nodes, coefficients), the form a nonlinear solve's residual blocks take (Estimation's SplineLens refine
    builds its Δ³ blocks from it).
  - THE NORMALIZATION. `mean_positive_eigenvalue(D)` / `mean_positive_eigenvalue(Symmetric(DᵀD))` (m̄) and
    `penalty_scale(energy, N, m̄)` (`λ = β·energy/N/m̄`), moved from VisualGeometryToolkitEstimation (unreleased there), so
    a linear solve calls them without Estimation.
  - THE LINEAR SOLVE. `cgls(op, b; w, h, x0, rtol, max_iter)` → `(; x, iterations, converged, relres)`: CGLS for
    `min ‖Ax − b‖² + ‖√w∘(Dx − h)‖²` on `PenalizedOperator(A, D; device)` (the stacked operators and their transposes
    as `CSRMatrix`es, one KernelAbstractions work item per row, deterministic, no atomics; CPU and CUDA), a warm start
    `x0`, a target `h`, and the stopping rule `‖Aᵀ(b − Ax) + DᵀW(h − Dx)‖ ≤ rtol·‖Aᵀb + DᵀWh‖` (relative to the
    problem, not to the start: a warm start that is already close stops at once). The solver RadiometricCamera's
    texture recovery and PanoramaStitching's vignetting call in place of their own CSR/CGLS and `_pcg`.
  - No penalty fixes a gauge: a direction neither the data nor the penalty holds is held exactly by its caller.
- **New: `field_of_view(camera, distance; dimension)`**: the size of the scene imaged at `distance`,
  `2·distance·tan(θ/2)` for the angle of view `θ`, in the distance's unit (180° or more: an `ArgumentError`).

## 0.8.0 (2026-11-09)

Breaking (the next release is a minor bump). The `N = 1` `RadialHomography`
is unchanged bit for bit: its solver, chart, rows and Jacobians.

- **Breaking: `RadialHomography{T, N}`**, `N` division coefficients. The
  field `k` is now `λ::NTuple{N, T}`; `m.k` still reads `λ₁` through
  `getproperty` (an `ArgumentError` for `N > 1`), but Accessors' `@set m.k`
  no longer works. `RadialHomography{T}` is now a UnionAll:
  `Vector{RadialHomography{Float64}}`, a `Ref{RadialHomography{Float64}}()[]`
  placeholder and a field annotated `RadialHomography{T}` become abstract (write
  `RadialHomography{T, 1}` for today's model). `division_coeffs(m)` (public)
  reads every coefficient; `division_coeff(m)` throws for `N > 1`.
- **Breaking: `CameraModel(::RadialHomography)` is projective** —
  `CameraModel(ProjectiveIntrinsics(center(m)), Division(division_coeffs(m)...))`.
  `calibration_matrix`, `unproject`, `image_circle` and the default nominal
  camera on it are MethodErrors; `fov` is an `ArgumentError`, as are `fov_radius`
  and a finite `max_fov` (`undistorted_rect`).
  `RadialHomography(H, camera)` and `radial_homography(camera, H)` build the
  model from such a camera.
- **`radial_homography(camera::CameraModel{<:LogicalIntrinsics, <:Division}, H)`**
  pairs a stored plane homography with a METRIC division camera: `λₚ = kₚ / f^(2p)`
  about the principal point, so `radial_transfer` of it is `project(camera, K⁻¹·H·X̃)`.
  An `ArgumentError` for `fx ≠ fy` or a skewed `K`.
- **`CameraModelSpec{I, S}(…)` normalizes a parameterized intrinsics kind**
  (`LogicalIntrinsics{Float64}`) to its wrapper in the inner constructor too, as
  the keyword path does, so the specification round trips equal.
- **`ProjectiveIntrinsics{T}` refuses an integer or rational `T`** (an
  `ArgumentError`; floats and AD numbers stay, `ProjectiveIntrinsics(center)`
  still takes `float(T)`).
- **`LensSpec`**: `LensSpec(lens)` names a lens at the reference values whatever
  its element type (a `Float32` lens is the named form), keeps a `-0.0`
  coefficient in the record form, and `lens(spec)` is `Float64`; the 4-argument
  inner constructor validates what the public ones do.
- **Breaking: the metric operations are narrowed to `LogicalIntrinsics`**:
  `calibration_matrix(::CameraModel)` and `unproject(::AbstractIntrinsics, …)`
  have no method for another intrinsics kind.
- **Breaking: `plot_rectification`'s `max_fov` defaults to `nothing`**, resolved
  per camera: `DEFAULT_MAX_FOV` for a metric camera (today's value), no cap for
  a projective one (a strong fisheye rectifies to the horizon clip).
- **`ProjectiveIntrinsics(center)`**: the principal point alone, a projective
  ray frame with no focal length; `principal_point` for both intrinsics kinds.
  Project, backproject, `distortion_map`/`undistort` and the frame walk serve
  both kinds through one pixel-matrix seam.
- **`LensSpec`** (a lens family by name and term counts, or a record form for
  the rest; `free_names`, public) and **`CameraModelSpec`** (what a calibration
  is for: an intrinsics kind and a `LensSpec`, or the metric
  `GenericGridSpec`; `intrinsics_kind`, public), with TOML tables through
  `toml_value` and `CameraModelSpec(table)`. A `central_generic` table may name
  `intrinsics = "metric"` (redundant but true); any other kind beside it is an
  `ArgumentError`.
- **`TargetHomographies{F, H}`**: per frame, each target's plane → pixel `HomographyMap`,
  a capture like `TargetPoses`: the pixel frame `F` is part of the type
  (`TargetHomographies{F}(frames; …)`; no frame-less form), the maps are stored
  unlabeled (plain or `Uncertain`, an 8×8 covariance in `HomographyManifold8`) and
  `homography(th, t, id)` mints `TargetPlaneFrame → F`; `targets` names the target
  definitions by content identity. `radial_homography(camera, h::HomographyMap)`.
- **`HomographyMap{3}` has the parameter protocol** (`parameter_names` `h11 … h33`,
  `parameter_vector`, `with_parameters`, `length_mask`): its nine entries row-major,
  canonical (unit Frobenius norm, `H[3,3] ≥ 0`, `sign_normalize`; a canonical matrix
  as held). Nine, not eight: a homography has no global 8-number chart (`h33 = 1`
  fails at `h33 = 0`), so the normalization is declared. Its tag is
  `PLANAR_MAP_TAGS`' `"homography_map"` (a new `TagRegistry`, public). `relabel` and
  `unlabel` take an `Uncertain` homography.
- `division_coeffs` is in `CONTRACT_GENERICS`: another stack package may add a
  method for its own model or state (VisualGeometryToolkitEstimation's
  `DivisionCameraState` does).

## 0.7.0 (2026-11-07)

- **SILENT CHANGE — breaking: a generic model's `backproject` is NaN where it images nothing, never `nothing`.
  A caller that tests `backproject(m, u) === nothing` (or `isnothing`) still compiles, and the test never fires:
  the NaN ray flows on where the point used to be skipped.** It is the lens contract's form, one concrete return
  type, kernel-callable: a `CentralGenericModel` returns an `SVector{3}` of NaN outside the calibrated area and
  where the pixel's window holds an absent node, a `NoncentralGenericModel` a `Ray3` of NaN. Migration: replace
  every `d === nothing` on a generic model's `backproject` with `!all(isfinite, d)` (or `all(isnan, d)`; for a
  non-central model, of `d.direction`), and every `d === nothing ? nothing : f(d)` with
  `all(isfinite, d) ? f(d) : nothing`. `unproject` still returns `nothing` there.
- **Breaking: `moment_matrix` is `shape_matrix` (M); `shape_matrix` returns M again** (0.6 removed it so old factor
  calls failed loudly). `shape_matrix(f)` is the shape matrix `M` of an elliptical region, `{x : (x-c)ᵀ M⁻¹ (x-c) ≤ 1}`,
  for an `EllipticFrame`, an `AffineFrame`, an `Ellipse` and a `Paraboloid3D`; the factor stays `shape_factor(f)`, with
  `M = shape_factor(f) * shape_factor(f)'`. `moment_matrix` has no method and no alias (retired, `check_naming`), and
  `EllipticFrame(center, M, Val(:shape_matrix))` replaces `Val(:moment_matrix)`. Migration: `moment_matrix(x)` →
  `shape_matrix(x)`.
- **SILENT CHANGE for code written against 0.5 (a 0.5 → 0.7 jump): `shape_matrix` returned the FACTOR in 0.5 and
  returns `M` in 0.7, so such a call compiles and gets the wrong matrix.** Replace every 0.5 `shape_matrix(x)` with
  `shape_factor(x)` before upgrading; 0.6 is the release where the old call fails loudly, so upgrading through 0.6
  first finds them all.
- **Breaking: the deprecated `distance(hyperplane, p)` and `field_center(v)` are removed.** Use
  `signed_distance(hyperplane, p)` (`abs` of it for the unsigned value) and `center(v)`. The `distance` generic stays,
  for a non-negative distance, with no method in Core.
- **`AbstractThresholdScoring{T} <: AbstractScoring` (public): one threshold loop for every loss-width family.**
  `RansacScoring{DG, T}` and `MsacScoring{DG, T}` subtype it (`T` the threshold's type: `Float64` at a threshold,
  `Nothing` width-free). A family at a threshold implements `residual_quality(sc, d², τ)` and holds `threshold`, and
  inherits `score!` on a `ThresholdScoringWorkspace` (the consensus set `d² < τ²`), `scoring_workspace`,
  `inlier_count`, `consensus_order`, `inlier_threshold` and `has_inlier_threshold`; its `score_bound` is `nothing`
  (no early exit) unless it states one, which it may only where every quality is in `[0, b]`. The width-free form
  still implements no scoring contract. RANSAC's and MSAC's scores are unchanged bit for bit; the loop now treats a
  minimal-sample point whose finite residual overflows when squared as the main sum does (quality 0), which for
  RANSAC and MSAC was already the value. The internal `_ScoringAtThreshold` union is gone.
- **`VIGNETTING_TAGS` (public):** the vignetting fields' `TagRegistry`, `"tensor_product"` (`TensorProductVignetting`),
  `"radial"` (`RadialVignetting`) and `"log"` (`LogVignetting`) — the `type` VisualGeometryToolkitIO's
  `vgio/vignetting/v1` body writes; `"tensor"` is retired for `"tensor_product"`.
- **`ThinPlateSplineSystem` solves by factorization.** The conditioned symmetric system is solved for its `N`
  right-hand sides by one LU factorization in place of forming `inv`; `coeff` moves by rounding only (9.5e-16, 6.0e-15
  and 2.2e-14 relative to 0.6 on `grid_nodes(3)`, `grid_nodes(5)` and a 5 × 5 pixel grid), and on a 5 × 5 grid over a
  5208 × 3476 sensor `coeff` and `Q` are within 2.3e-14 and 1.7e-14 of a 256-bit BigFloat solve of the unconditioned
  system (a new test). Two exactness tests that were pinned at 1e-15 to the old solve's rounding now allow 16 eps:
  RadiometricCamera's warped-board ray intersections (Δt 1.1e-15) and the pure-bending projection's affine part
  (1.0e-15 relative).
- `configs/stack_dependencies.toml`: the `neurips2026` row names VisualGeometryToolkitIO, which it uses.
- **Breaking: the RANSAC protocol's per-point scalar is a distance, under a distance's name**
  (conventions/residuals-and-distances.md). Scores are bit for bit unchanged: every scorer reads the square, which the
  sign does not move.
  - `residuals!(r, problem, model)` → `mahalanobis_distances!(d, problem, model)`, `d[i] ≥ 0` (was signed at
    `d_g = 1`).
  - The per-point hook `whitened_residual(problem, ctx, i)` (a signed distance at `d_g = 1`, a distance otherwise) →
    `mahalanobis_distance(problem, ctx, i)`, `≥ 0`: a new method of the exported model-level `mahalanobis_distance`.
    Its default is the norm of the new `whitened_residual`.
  - `whitened_constraint_residual(problem, model, i)` → `whitened_residual(problem, ctx, i)`: the SIGNED whitened
    residual in the constraint's shape (a number at `d_g = 1`, an `SVector` of its `d_g` components otherwise).
    **The name is reused**: an override of the old hook keeps compiling and now defines the signed residual; at
    `d_g ≥ 2` it returns a scalar, and `test_constraint_problem` refuses that shape. Migrate such an override to
    `mahalanobis_distance`.
  - New: the problem-level `whitened_residuals!(r, problem, model)` stacks every point's `whitened_residual`
    (rows `d_g(i−1)+1 … d_g·i`), beside the model-level `whitened_residuals!(r, M, xs)`.
  - `whitened_distance(r)` is removed (the hook's default takes the norm).
  - `sq_residuals` → `sq_distances` (the `PosteriorWeightState` accessor and field, and the argument names of
    `PosteriorWeightState`, `seed_posterior!`, `posterior_weights!`, `posterior_inlier_probs!`,
    `soft_posterior_weights!`, `next_nu` and `weighted_rss`); `square_residuals!` → `square_distances!`.
  - `test_constraint_problem` checks `mahalanobis_distances!` (inferred, allocation-free, `≥ 0`, equal to
    `mahalanobis_distance` to rounding) and the shape of `whitened_residual` at every point. `CONTRACT_GENERICS` lists
    `mahalanobis_distances!`, `mahalanobis_distance` and `whitened_residual` in place of `residuals!`.
  - The old names are retired in `check_naming`; no alias remains, so an old call is an `UndefVarError`.
    Migration: `residuals!(r, p, θ)` → `mahalanobis_distances!(r, p, θ)` (`abs` is already applied);
    an old `whitened_residual` override → `mahalanobis_distance`;
    `whitened_constraint_residual(p, θ, i)` → `whitened_residual(p, θ, i)`; `whitened_distance(r)` → `norm(r)`;
    `sq_residuals(pws)` → `sq_distances(pws)`; `square_residuals!` → `square_distances!`.

- **The upstream pieces the 2026-11-06 downstream duplication audit found missing** (its "(b)" items; the copies
  in RadiometricCamera, BlobBoards, PanoramaStitching, LocalOptimizationDoneRight and RANSACScoringDoneRight are
  deleted against these in the next wave):
  - `SCORING_TAGS` (public): the scoring families' `TagRegistry`, `"ransac"`, `"msac"` (suffix `Scoring`); a
    package that adds families declares its own of the same family and merges.
  - `difference_rows(mask, order)` / `difference_rows(dims, order)` (public): the sparse forward differences of a
    (masked) 2-D lattice along both axes, `DᵀD` the P-spline penalty. VisualGeometryToolkitCore now depends on the
    SparseArrays stdlib.
  - `FrozenNuisance()` (exported): the weight policy that holds nothing (no-op `update_nuisance!` /
    `update_weights!`, returning the policy).
  - `decision_threshold(shape, loss_width)`: the inverse of `decision_width` for every loss shape (Welsch √ln2·w,
    Geman–McClure, Cauchy, smooth-TLS, TLS, Barron).
  - `Pools.check_residual_matrix(pair, R)` and `Pools.best_model_index(score, M)` (public): the pool record's
    residual-matrix contract and the one pool argmax (first strict best; a non-finite score, `±Inf` included, never
    wins; `0` none); `best_pool_model` calls them.
  - The generic models' accessors `directions(m)`, `grid_size(m)` (`(gx, gy)`, the `grid_size` keyword's order),
    `window_offset(m)`, `lattice_spacing(m)` (were private `_directions`, `_grid_size`, `_offset`,
    `_lattice_spacing`), and their set accessors `with_domain`, `with_present` (absent slots NaN), `with_nominal`,
    `with_directions`, `with_points`, and `central_model(::NoncentralGenericModel)`; `primal_float(T)` (was
    `_primal_float`).
  - The generic models' NaN `backproject`: see the SILENT CHANGE entry below.
  - `convert_eltype(T, x)` takes any `T <: Real` (a ForwardDiff `Dual`) for a lens, `LogicalIntrinsics`,
    `CameraModel`, `BoundedCamera`, `RotationMap` and an `OrientedCamera`; new: a plain `EuclideanMap` (rotation
    representation and frames kept, translation converted).
  - `StudentTMixture{T}(ν, s_hat_sq, log_two_outlier_halfwidth, d_g[, inlier_prior]; lgamma = loggamma)`: the
    mixture built in `T` with the log-gamma supplied — kernel-lawful (isbits, allocation-free); the untyped
    constructor is its Float64 form, bit for bit as before. `nu_point_terms(z, ν, ψ_half_νdg, ψ_half_ν; d_g)` and
    `next_nu_from_sums(ν, dQ, d2Q_data, sum_w, ψ1_half_νdg, ψ1_half_ν; d_g, ν_min, ν_max)` (public): one point's
    Q-derivative terms and the safeguarded Newton update from the reduced sums, which `next_nu` itself now runs (its
    values unchanged, bit for bit).
  - `is_peak(C::StaticMatrix{2,2})`; `is_valid_shape` is closed form (`‖S‖²_F ≤ (k² + k⁻²)|det S|`, the same
    decision as the frame's `aspect_ratio ≤ k`, checked on 2·10⁵ random matrices), allocation-free and
    kernel-callable; its `max_aspect_ratio` takes any `Real`.
  - `equidistant_chart(axis)` (public): `Lens(Equidistant()) ∘ LinearMap(R)`, `R`'s rows `(e₁, e₂, axis)` —
    VisualGeometryToolkitEstimation's `fit_camera` basis rule, so its fits are bit for bit unchanged.
  - `Eliminable{E}(value)` (public): an NLLSsolver variable the Schur system may eliminate (`iseliminable` where
    the installed NLLSsolver has it); VisualGeometryToolkitEstimation's bundle points use it.
  - `pixel_centre([T,] I::CartesianIndex{2})` (public): the COLMAP centre `(col − ½, row − ½)` of an array index,
    in `T`, kernel-lawful; `pixel_mask` and `pixel_centres` read it.
  - `distortion_map(::RadialHomography)`: the radial model's undistorted → distorted pixel map,
    `distortion_map(CameraModel(m))`.
  - `convert_field(T, v, key)`: the keyed TOML conversion (a table refused, a tuple's length checked, a vector
    converted element-wise, a string refused for an array, the key in every error); `read_table`'s default
    converter is now the keyed form (stricter: a scalar or a string for a tuple or vector field is an
    `ArgumentError` naming the key). Any other array field (an `SVector`) converts by its constructor as before
    (`SVector{2, Float64}([1, 2])`), its failure named by the key; a `Vararg` tuple field
    (`Tuple{Vararg{Float64}}`) takes an array of any length, element by element.
  - `lens_label(lens)` (public; `lens_tag` is a retired name): a lens's display name by the tag rule — a named family's (`"pinhole"`,
    `"kannala_brandt"`, `"division"`, `"scaramuzza"`), otherwise its record's leaf tags joined by `_`.
  - `intersects(::Circle, ::Rect{2})` (either order; squared, strict: BlobBoards' placement test bit for bit).
    `intersects(::Circle, ::Circle)` is unchanged, `norm(c₁ − c₂) < r₁ + r₂`; BlobBoards' squared circle test
    differs from it within an ulp of tangency.
  - The parameter protocol (`parameter_vector`, `parameter_names`, `with_parameters`) for the vignettings (their
    control points, `c1…`), `EnvironmentLight` (its SH coefficients, `a<channel>_<index>`) and `GaussianBlur` (its
    storage's free entries, `variance…` bounded below by 0, `covariance_xy` free; a print's blur is stripped
    first), so each is a `ParameterManifold`; and `restrict(ch, Z::AbstractMatrix)`, the `LinearSubspaceManifold`
    of a basis (`v = Z y`; `Z` held as a `Matrix{Float64}` with the Cholesky factor of `ZᵀZ`, computed once;
    dependent columns are an `ArgumentError`).
  - `shape_matrix(::Disc)`, `shape_matrix(::OrientedDisc)`: `σ² I`.
  - `is_positive_definite(M)` is public (the 2×2 Sylvester test `is_valid_shape` reads), and a `SimilarityMap{2}`
    acts on `WholePlane` as the identity, beside its `Circle` and `Ellipse` actions (PanoramaStitching's copies).
  - `sha256_hex(s | bytes | io)` (public; moved from VisualGeometryToolkitIO): the one hex digest of the stack; a
    file's is `open(sha256_hex, path)`.
- **Breaking: the model-level row families are unweighted — `sampson_residuals!`, `sampson_jacobian_rows!`,
  `whitened_residuals!(r, M, xs)` and `whitened_jacobian_rows!`**, the currency of the problem-level
  `whitened_residuals!(r, problem, model)`: a set's `sqrt_weights` column is read only by the new
  `weighted_whitened_residuals!` / `weighted_whitened_jacobian_rows!` (a weighted fit's, the chart LM's, rows: the
  former whitened bodies, bit for bit; on a set without noise columns, the geometric rows times `√w`). The chart
  LMs call only the weighted forms, so every LM result is bit for bit unchanged. Migration: a caller relying on
  the weighting calls the `weighted_whitened_` form (for the geometric rows, on a set without noise columns).
- `codimension(problem)` on an instance of any `AbstractConstraintProblem` falls back to `codimension(typeof(problem))`
  (moved from VisualGeometryToolkitEstimation's `AbstractRansacProblem` method): a problem defines it on its type.
- The problem-level `whitened_residuals!(r, problem, model)` reads `codimension(problem)` components per point
  and requires `length(r) == codimension(problem)·data_size(problem)` (`DimensionMismatch` otherwise, and for a
  point whose residual has another number of components): a scalar override at `d_g ≥ 2` fails loudly.
- `test_constraint_problem` also checks `norm(whitened_residual(p, ctx, i)) ≈ mahalanobis_distance(p, ctx, i)` at
  every point (a non-finite distance matched by a non-finite residual).
- `check_naming` retires `field_center`.

## 0.6.0 (2026-11-06)

- **`ThinPlateSplineSystem` solves a conditioned system.** It centred nothing and scaled nothing, so on pixel nodes (a
  5208 × 3476 sensor's grid) `[K P; Pᵀ 0]` had a condition number of 2e18; the nodes are now centred on their centroid
  and scaled by one isotropic factor (`HartleyNormalization`, κ ≈ 1.2e3 on any 5 × 5 grid) and the solution mapped back
  exactly (`w = σ² w′`, the linear part `σ a′`, the constant `a′₁ + a′·𝐭 + ln σ Σ w′ᵢ|𝐩′ᵢ|²`, `Q = σ² Q′`). An
  interpolating thin-plate spline is the same surface under a similarity of its domain, so only rounding moves:
  `coeff` and `Q` by 1e-15–1e-13 relative on grids, 2e-11 on 100 random nodes. Everything it holds stays in the caller's
  frame. The thin-plate calibration kinds' docstrings now say their chart must be ONE scaling of the board (`[0, 1] ×
  [0, H/W]`), never a per-axis `[0, 1]²` (a thin-plate spline is not invariant under it).

- **Breaking: `shape_matrix` is removed; the factor is `shape_factor`.** `shape_matrix` returned the FACTOR `F` of an
  elliptical region's matrix `M = F·Fᵀ` — `S` for an `EllipticFrame` (`M = S²`), `A` for an `AffineFrame`,
  `R(θ)·diag(a, b)` for an `Ellipse` — under the name ellipsoidal calculus gives `M` itself (`E(c, M) = {x : (x-c)ᵀ
  M⁻¹ (x-c) ≤ 1}`). It is now `shape_factor(f)`, with `moment_matrix(f) = shape_factor(f) * shape_factor(f)'`;
  `shape_matrix` has no method and no alias (retired, `check_naming`), so every old call is an `UndefVarError` rather
  than a silent change of meaning. Migration: `shape_matrix(x)` → `shape_factor(x)`. The `FrameScaleTrait` arm that
  dispatches on it is renamed with it, `ShapeMatrixScale` → `ShapeFactorScale` (retired, no alias). **0.7.0 will
  retire `moment_matrix` and reintroduce `shape_matrix` meaning `M`** (and take `shape_matrix` off `check_naming`'s
  retired names when it returns): `M` is a region's shape, not its second moment (a uniform ellipse has covariance
  `M/4`; in affine feature detection the second-moment matrix is the structure tensor `μ ≈ M⁻¹`). The two-release
  order makes every caller fail hard once instead of compiling against a changed meaning.
- **Breaking: a `GaussianBlur` holds a covariance.** A point spread function is a probability density (where a photon
  from a point source lands), and a Gaussian blur is the density `N(0, Σ)`, so its parameter is a covariance: the
  blur stores a `CovMat{2}` (the same three storages, `ScaledIdentity` / `SDiagonal` / `SMatrix`; its frame tag the
  plane the blur lives on). The 0.5 note that a blur "is a kernel's shape, not a covariance" is reversed. The
  accessors are Statistics.jl's: `cov(b)` (Σ), `var(b)` and `std(b)` (per axis); `moment_matrix(b)` is removed.
  `GaussianBlur(Σ::CovMat{2})` keeps the frame; `GaussianBlur(σ)`, `GaussianBlur(σx, σy)` and the bare storages stay
  (a negative standard deviation is now an `ArgumentError`; zero, the identity blur, is allowed).
  `convolve(a, b)` adds the covariances and checks that the two frames meet. **`transform_blur` is retired**: a
  blur under a linear map is the covariance's push-forward, `propagate(J, b)` (a raw Jacobian, the result unlabeled)
  or `propagate(m, b)` for a framed map with a constant Jacobian (`EuclideanMap`, `SimilarityMap`, a camera's
  `FramedAffine`: the frame checked, the result in the map's destination), through `propagate`'s own law for the
  map; a singular or near-singular map's rank-deficient result is clamped onto the PSD cone (an eigenvalue negative
  only to rounding becomes zero), never refused over a rounded determinant. A print's blur holds m² quantities (`GaussianBlur(0.02mm)`): `has_units`, `ustrip(u, b)` (plain, in u²) and
  `with_unit(b, u)` are the whole value's.
- **Breaking: maps are the geometric objects, matrices only algebra.** The typed homogeneous matrices
  `HomTranslationMat`, `HomRotationMat`, `HomEuclideanMat`, `HomSimilarityMat`, `HomScalingMat`,
  `HomUniformScalingMat`, `HomAffineMat` and their abstract levels `AbstractHomMat`, `AbstractAffineMat`,
  `AbstractSimilarityMat`, `AbstractEuclideanMat`, `AbstractCalibrationMat` are removed, with their promotion
  rules, `*` and `inv`, the per-leaf constructors and `AbstractHomMat(x)`; the names are retired (`check_naming`).
  The transform class is the MAP's type: `EuclideanMap`, the new `SimilarityMap`, CoordinateTransformations'
  affines, `HomographyMap`. Migration: `HomRotationMat(R)` → `LinearMap(R)` or `EuclideanMap(R, 0)`,
  `HomTranslationMat(t)` → `Translation(t)`, `HomUniformScalingMat{2}(s)` → `LinearMap(s * I)`, a similarity
  matrix → `SimilarityMap(s, R, t)`, `HomAffineMat(M)` → `AffineMap(A, b)`; `HomographyMat` stays (a plain 3×3
  static wrapper, its product and inverse homographies).
- **Breaking: `to_homogeneous(map)` is a plain `SMatrix{N+1, N+1}`** for every map — `Translation`, `LinearMap`
  (rotation, `UniformScaling`, `Diagonal`, any matrix), `AffineMap`, `EuclideanMap` (now any `N`), `SimilarityMap`,
  a composition (`ComposedFunction` or CoordinateTransformations' `ComposedTransformation`: the product),
  `HomographyMap` (its forward) — the map's entries copied, the last row the constant `0 … 0 1`. It carries no
  class; geometric operations take the map. **`to_homogeneous` of a matrix is removed** (`to_homogeneous(H)` of
  a `HomographyMat` or a `CameraCalibrationMat` returned it as it was): a matrix is not a map, use it as it is or
  lift it (`HomographyMap(H)`). A unitful `EuclideanMap` has no plain matrix: `to_homogeneous`, and so
  `HomographyMap(E)` and `E ∘` a homography or a model matrix, is an `ArgumentError` naming `ustrip(m, E)`.
- **Breaking: a `HomographyMap` is always projective in meaning**: a `Circle` or an `Ellipse` maps to a `Conic`,
  whatever its matrix (a `HomographyMap` of an affine or similarity matrix gave a `HomEllipseMat` or a `Circle`).
  Apply the affine (or similarity) map itself to keep an ellipse an ellipse (a circle a circle). Lifting a
  structured map — `HomographyMap(a)` for a `Translation`, `LinearMap`, `AffineMap`, `EuclideanMap{2}`,
  `SimilarityMap{2}` — stores `to_homogeneous(a)` and, as its inverse, `to_homogeneous(inv(a))` (the map's
  analytic inverse, never a 3×3 inversion; a singular or non-finite linear part is an `ArgumentError`, where it
  stored an `Inf` inverse); a `CameraCalibrationMat` lifts with its analytic inverse. Mixed
  compositions (`EuclideanMap`, `SimilarityMap` or a CT leaf ∘ a `HomographyMap`, either order) are a
  `HomographyMap` with `SMatrix` storage; two `HomographyMat`-backed maps keep `HomographyMat`. The model-matrix
  and epipolar bridges (`a ∘ H`, `a ∘ F`, `a ∘ EpipolarMap`, `a ∘ RadialHomography`, `a ∘ RadialCamera`) take the
  same affine-class maps, `EuclideanMap{2}` and `SimilarityMap{2}` included. `HomographyMat`s of two eltypes
  promote to the `HomographyMat` of the promoted eltype.
- **`SimilarityMap`** (exported, `transforms/similarity_map.jl`): `x ↦ s·R·x + t` with a positive scale, a
  Rotations.jl rotation and a translation, frame-tagged exactly like `EuclideanMap` (`Src`/`Dst`, `relabel`,
  `unlabel`, `inv` swapping, `∘` junction-checked). Exact algebra from the parts: `∘` of two similarities is one
  (`s₁s₂, R₁R₂, s₁R₁t₂ + t₁`), `inv` is `(1/s, Rᵀ, −Rᵀt/s)`; `∘` with every other similarity-class map — an
  `EuclideanMap`, a `RotationMap`, a `Translation`, a rotation `LinearMap`, a uniform scaling (either order) — is a
  junction-checked `SimilarityMap`, never a lazy composition; `SimilarityMap(E)` (s = 1),
  `SimilarityMap(RotationMap(R))`, `SimilarityMap(LinearMap(R))`, `SimilarityMap(LinearMap(λI))` (a
  negative λ is the half turn) and `SimilarityMap(Translation(t))`. Accessors `scale`, `rotation`, `translation`;
  `transform_deriv`, `to_homogeneous`, `similarity_scale`, `propagate` of a `CovMat`/`PrecMat`/`Uncertain`
  (`J = s·R`, EuclideanMap's junction check), and EuclideanMap's labeling `convert` (an unlabeled map into a
  labeled slot). Plain numbers only: a quantity translation or scale, or a unitful `EuclideanMap` converted or
  composed with one, is an `ArgumentError` naming `ustrip(m, E)`.
- **Primitive actions by map class.** A similarity (`SimilarityMap`, and through it a rotation or uniform-scaling
  `LinearMap`) maps a `Circle` to a `Circle` (radius · s), an `Ellipse` to an `Ellipse` (axes · s, θ + the
  rotation's angle), a `Line2D` to a `Line2D`, a `Disc` to a `Disc` (σ · s), an `OrientedDisc`, an
  `EllipticFrame` (shape `s·R S Rᵀ`, moment `s²·R Σ Rᵀ`) and an `AffineFrame` (`s·R A`). An `AffineMap` or a
  general `LinearMap` gives a `HomEllipseMat` (as before), conjugating by its own analytic inverse — static
  (a Matrix-backed map is made static first) and checked: a singular or non-finite linear part is an
  `ArgumentError` for a circle, an ellipse and a line alike (it gave a `HomEllipseMat` of NaNs, a misleading
  'Normal vector cannot be zero' or a `SingularException`), and a linear part so ill-conditioned that the image
  reads as a parabola (κ² past the classifier's floor, `Diagonal(1e7, 1)`) is an `ArgumentError` rather than a
  `HomEllipseMat` that `classify_conic` disowns. A feature frame under a homography stays a `MethodError`.
- **Fixed: the rigid action on a `Plane3D` and a `Line2D`.** For the Hesse form `n·x = d`, `E(plane)` and
  `E(line)` are now `n′ = R·n`, `d′ = d + n′·t` (the plane through the moved points, `n′·(R x + t) = d + n′·t`);
  0.5 wrote `d′ = d − n·t`, the wrong sign and the unrotated normal, so every rigid plane and line result with a
  nonzero translation changes. The similarity action on a line promotes
  over every part (a `Dual` scale or translation differentiates; a Float64 translation over a Float32 scale and
  rotation gives a Float64 line).
- **Breaking: conic matrices are symmetric by storage.** `HomEllipseMat`, `HomCircleMat` and `Conic` store the six
  coefficients `(Q₁₁, Q₁₂, Q₂₂, Q₁₃, Q₂₃, Q₃₃)`; indexing returns the full symmetric matrix, arithmetic an
  `SMatrix`. A matrix (or 9-tuple) handed in is checked symmetric (`isapprox(Q, Qᵀ; rtol = √eps)`, otherwise an
  `ArgumentError`) and its off-diagonal pairs averaged; `HomEllipseMat(q::NTuple{6})` (and `HomCircleMat`, `Conic`)
  takes the coefficients. `Conic`'s field `Q` is gone: read `SMatrix{3, 3}(c)` or `c[i, j]`. The conic congruence
  `H⁻ᵀ C H⁻¹` computes the upper triangle only (a full 3×3 product left 913 of 1000 random results asymmetric).
- **Breaking: conic class checks.** `HomEllipseMat(…; check = true)` requires a real, non-degenerate ellipse —
  `classify_conic` answering `EllipseTrait`, the same test — and `HomCircleMat` also an
  isotropic quadratic block (within `√eps · ‖A‖`); a general conic (a hyperbola, a point or imaginary ellipse, a
  line pair) is a `Conic`, which `classify_conic` and `is_ellipse` now take directly. `Ellipse(c::Conic)` refuses a
  conic that is not an ellipse; `Circle(Q)` of an ellipse's or a conic's matrix is the `HomCircleMat` check (its
  `atol` keyword and `ErrorException` are gone). The three are one type, `SymmetricConic{K}` (`K` the class). A
  conic built from a primitive is of its class by construction (not re-checked), and an affine image is classified
  by the same test, so a stored class and `classify_conic` agree for every conic the stack builds; a
  `check = false` value is the caller's guarantee. Copying, converting the eltype (`push!` into a typed vector)
  and padding a `FixedModels` never re-check a conic or a `CameraCalibrationMat` (StaticArrays' generic forms
  rebuilt them through the checked constructor, so an unchecked value could not be copied and `zero` threw).
- **The conic classification is decided in at least Float64**, on the coefficients divided by the largest: a
  Float32 conic is classified as its stored coefficients define it (Float32 arithmetic refused 51% of 1–10 px
  Float32 blobs in a 4000×3000 image as point conics, and `direct_ellipse` returned `nothing` for 56%; it now
  classifies in its Hartley frame), `1e160·Q` and `1e-170·Q` classify as `Q`, and integer and rational matrices
  are accepted. The symmetry check is relative, in `float` of the entries, with the tolerance of the precision
  the entries were given in (Float32 data into a `Conic{Float64}`).
- **Breaking: `CameraCalibrationMat` stores its five parameters** `(fx, fy, s, cx, cy)`: indexing returns exact
  structural `0`s and `1`; isbits and eltype-generic as before. The structure (row 3 = `0 0 1`, `K[2, 1] = 0`) is
  now always checked, `check = false` skipping only the positive focals; `CameraCalibrationMat(α * I)` with
  `α ≠ 1` is refused (`K₃₃ = α`). `inv(K)` is analytic, a plain `SMatrix{3, 3}` with exact structural zeros (it
  was a `CameraCalibrationMat`: K⁻¹ is not a calibration), equal to the former affine inverse to rounding
  (`K⁻¹[2, 1]` is now `+0.0`, and `K⁻¹[1, 3]` can differ by an ulp for a skewed `K`); arithmetic on a
  `K` gives an `SMatrix`, and `K * v` is StaticArrays' product (`SMatrix(K) * v` bit for bit, the vector's type
  kept). A 3-D `AffineMap` is not a calibration matrix (`CameraModel(::AffineMap)` refuses it).
- **`EnvironmentLight`, `SphericalHarmonicMap`** (exported, `cameras/lighting.jl`): the scene's light as a radiometric
  camera model sees it — radiance from every direction, a directional map on a sphere of finite radius about a centre,
  read where a ray from the shaded point leaves the sphere (`exit_direction`); the
  map is real spherical harmonics per channel. `radiance(light, x, ω)`, the linear factor `coefficient_jacobian(light,
  x, ω)`, the accessors `coefficients`, `with_coefficients`, `max_degree`, `radiance_map`, `center`, `radius`, and
  `convert_eltype`, `ustrip` / `with_unit` (centre and radius are lengths). The basis is `sh_basis(Val(l_max), ω)`
  (orthonormal, no Condon–Shortley phase; `sh_index(l, m) = l² + l + m + 1`, `sh_count`): generated, isbits,
  allocation-free and Dual-safe at the poles. Hoisted from RadiometricCamera (`Lighting`, `y_basis`, `ω̂`, `L_SVSH`),
  bit for bit: every value and ForwardDiff Jacobian equal on random inputs for l ≤ 3. Additive.
- **Breaking: `CovMat` / `PrecMat` hold a static matrix, and its type is the storage.** `CovMat{N, T, M, C}`: `M` is an
  `SMatrix` (full), an `SDiagonal` (diagonal) or a `ScaledIdentity` (isotropic λ·I); it was `CovMat{N, T, L, C}` with
  `L = N * N`, a tuple of entries. `parent(Σ)` is the matrix, and structured arithmetic is the matrix's own; the form
  reads as its full `N × N` matrix, and arithmetic on it escapes to an `SMatrix` as before. A form keeps its storage
  under `inv` and `relabel`; `convert` and `promote` give the coarser storage (isotropic ⊂ diagonal ⊂ full), never a
  finer one, so a container of mixed storage is one concrete type; `propagate` through a map gives the full form; a unit
  change gives an isotropic form's diagonal. `CovMat{N}(σ = s)` is now isotropic and `CovMat{N}(σ = (s₁, …))` diagonal
  (they were full); the entry constructors (`CovMat{N, T}(…)`, `CovMat(::SMatrix)`) give the full form as before.
  Aliases `FullCovMat`, `DiagonalCovMat`, `IsotropicCovMat` (and `…PrecMat`); `UncertainPixel` and `UncertainPoint3`
  hold the full form. A `TargetDetections` set and a `PointTarget` keep the storage their points' covariances hold
  (they densified them), mixed storage in the coarser one.
- `TensorProductVignetting(knots_x, knots_y, P; degree)`: a surface on stated clamped knots (a file's reader rebuilds a
  surface exactly). `radius` (GeometryBasics') is public, an `EnvironmentLight`'s sphere among its methods. Migration: a spelled-out `CovMat{2, Float64, 4, F}` is `CovMat{2, Float64, SMatrix{2, 2, Float64,
  4}, F}`, and `CovMat{N, T, N * N}(data)` is `CovMat{N, T}(data)`.
- **`ScaledIdentity{N, T}`** (exported): the static `N × N` matrix `λ·I`, stored as `λ` — the sized counterpart of
  `UniformScaling`, beside `SDiagonal` and `SMatrix`; isotropic (a rotation leaves it unchanged), isbits.
- **`GaussianBlur`** (exported, `cameras/blur.jl`; `AbstractBlur`): one blur of an imaging model — a camera's in-focus
  blur, a print's — by its second-moment matrix (`moment_matrix`; px² on a sensor, m² on a print), held as the same three
  storages as a `CovMat`: round (`ScaledIdentity`), axis-aligned (`SDiagonal`) or general (a symmetric positive
  semidefinite `SMatrix`); `convolve(a, b)` adds the moments, `transform_blur(J, b)` is `J M Jᵀ`. A kernel's shape, not a
  measurement's uncertainty: not a `CovMat`.
- **Breaking: `TensorVignetting` is `TensorProductVignetting`** — a tensor-product B-spline surface, not a tensor-valued
  field; the old name is retired (`check_naming`).
- **The inverse of an affine-family transform is exactly of its class**: a structured map inverts analytically and
  its `HomographyMap` lift stores that inverse, last row exactly `0 0 1` (the 3×3 inverse left `1 ± ulp` there), the
  transpose for a rigid one; `inv(::CameraCalibrationMat)` is analytic. A homography keeps the 3×3 inverse.

## 0.5.7 (2026-11-05)

- **`TargetDetections{F, C}`** (exported, `captures.jl`): per frame index, per
  target id, the detections of the target's points as a correspondence set
  `csponds(index, observation)` — `index` 1-based into `points(target)`,
  unique; only detected points, a target not detected absent. The dimension
  is the frame's: pixels in a `PixelFrame` (plain or `UncertainPixel`, px²),
  3D points in a named `Frame{3}` (plain, or lengths in metres, an
  `Uncertain` one's covariance in m²). Optional attribute columns
  (`withattrs(cs; weights)`). One concrete set type per container
  (`Csponds{Int, P, Vector{Int}, Vector{P}, A}`): every set the same point
  form and the same columns (`ArgumentError`), all physical or all plain
  (`DimensionError`). `==`, `isequal` and `hash`, which agree (the frame
  type, frames, sets with their columns, stamps, targets: a sound Dict key),
  `has_units`, `ustrip(u, td)` and `with_unit(td, u)` for 3D,
  `convert_frame` for pixels, the Dict surface. Conversions
  `TargetDetections(::TargetPoints2D)` (the bundle `weight` a `weights`
  column) and `TargetDetections(::TargetPoints3D)`; an old value with a
  (target, index) twice in one frame or with plain and `Uncertain` points
  mixed is refused, the `ArgumentError` naming the frame, target and index
  (the old value stays valid as it is). Additive: the 0.7 wave
  replaces `TargetPoints2D`/`TargetPoints3D` by it.
- **`FrameStamps`** (exported): a capture's per-frame `images`, `sensors`,
  `times` and `time_origin`, built and checked as the captures' stamps are;
  `TargetDetections` holds one (`td.stamps`) and forwards its four fields as
  properties (`td.images`, …). Trajectory, TargetPoses and TargetPoints2D/3D
  keep their own fields until 0.7.
- `ustrip(u, layout)` and `with_unit(layout, u)` for `AprilGrid` and
  `SchoepsPattern`: the layout's lengths moved between the forms (an
  `ArgumentError` for a layout already in the form asked for), as for `Board`
  and `Canvas`; `PlanarTarget`'s unit moves use them.
- `check_naming` R11 flags a `TargetDetections` built from `ustrip(…)`
  components, as it does `TargetPoints3D`.
- Conventions: `frames.md` names `TargetDetections` among a capture's values,
  `units.md` lists a 3D `target_detections` among the physical kinds and
  public values, `persisted-geometry.md` tables the `target_detections` kind
  (the old measurement kinds read as it too) and points at VisualGeometryToolkitIO's
  file-name convention.

- `signed_distance(hyperplane, p)`: the signed distance `n·p + d`, positive on
  the normal's side (a line or a plane; the side test of a camera in front of a
  planar target).
- `distance(hyperplane, p)` is deprecated (a warning, the same signed value):
  the name read as unsigned. Use `signed_distance`, or `abs` of it. The
  hyperplane method is removed in 0.7; `distance` stays the unsigned generic.
- **A `PlanarTarget` records the layout it was built from**:
  `AbstractTargetLayout` (public), the supertype of `AprilGrid` and
  `SchoepsPattern`; `PlanarTarget(layout; calibration = Rigid(), board,
  canvas, placement = nothing)` (the `calibration` and `placement` keywords
  new) builds the same points as before and records the layout, read by
  `layout(t)` (public; `nothing` for a target given by its points or a
  reference). A unit move moves the layout with the points (rebuilding them
  from it), `with_parameters` keeps it, and `resolve`, which supplies other
  points, drops it; `==` and `hash` include it. The public positional
  constructor is unchanged. Additive: a new last field, `layout`.
- **The layouts hold the whole design**: `AprilGrid(; …, apriltag_family =
  "tag36h11", first_id = 0)` and `SchoepsPattern(; …, star_segments = 16,
  apriltag_family = "tag36h11")`, each Schöps tag `(; x, y, width, height,
  id)` (the YAML's `index`; without ids the tags are numbered 0, 1, … in
  order; ids distinct and ≥ 0, star segments even and ≥ 4). The keyword
  defaults keep every existing construction, and the points do not move. Both
  have value `==` and `hash` (a Schöps pattern's tag vector compared by value).
- `TARGET_LAYOUT_TAGS` (public): the layouts' `TagRegistry`, `"april_grid"`
  and `"schoeps_pattern"` (the tag rule's `snake_case` of the type's name),
  VisualGeometryToolkitIO's `layout` record type.
- **A new convention, `conventions/residuals-and-distances.md`**, read after
  `measurement-noise.md` (`convention_files()` and the docs' Conventions
  page): the prefix names the currency (`sampson_` geometric, `whitened_` /
  `mahalanobis_` in noise units), the noun names the shape (a residual is the
  signed vector, a distance its norm, a signed distance the residual when
  `d_g = 1`), and a square is the squared distance. It states the one current
  exception, `residuals!` and the hook `whitened_residual`, and their 0.7
  names. Documentation only: no name or value changes.
- Docstrings: `whitened_residual` and `residuals!` call `gᵀΣ̃⁻¹g` the squared
  Mahalanobis distance (they called it the Mahalanobis distance) and point at
  the new convention.

## 0.5.6 (2026-11-05)

Bug fixes to 0.5.5's additions; additive otherwise.

- `view_overlap(a, b, circle)`: the one circle now bounds both frames, as the
  vector form's one circle bounds every frame (frame `b` kept the whole sensor).
  `circle_b` defaults to `circle_a`.
- `view_overlap` on a frame whose model is not a `CameraModel` (a grid model)
  is an `ArgumentError` naming the requirement, not a `MethodError` inside the
  footprint.
- `with_control_points` promotes the field's element type with the control
  points' (`promote_type`): Int control points no longer throw `InexactError`
  on the knots, and Float32 ones no longer round the knots, centre or radius.
- A `TensorVignetting` of Dual eltype implements the support protocol
  (`support` reads the knots' values under the Duals): `free_nodes`,
  `node_mask` and `BoundaryRule` work on it.
- `RadialVignetting(n, center, radius)` accepts `px` quantities (a length is a
  `DimensionError`) and refuses a radius that is not positive and finite
  (`ArgumentError`).
- `RotationMap` hashes what `==` compares (tags and the rotation's matrix), so
  equal maps of different representations hash alike.
- `PosedCamera` admits `RotationMap{3}` only: a planar rotation is a
  `MethodError` at construction, not later in `pose`/`project`.
- `center(v)` (Core's one centre generic) for every vignetting field.
  `field_center` is deprecated (a warning, forwarding to `center`); it is
  removed in 0.7.
- `LogVignetting`'s `coefficient_jacobian` and `active_coefficients` evaluate
  the spline's basis once, bit for bit the two-evaluation form.
- Planned for 0.7 (breaking): the vignetting constructors' element type
  becomes a leading positional argument, `TensorVignetting([T = Float64], n, box; degree)`
  and `RadialVignetting([T = Float64], n, center, radius; degree)`, per the
  signature convention (slot 4); the `T` keyword stays until then.

## 0.5.5 (2026-11-04)

Additive.

- **A camera's vignetting** (cameras/vignetting.jl): the stack's one model of a
  relative illumination V on the sensor, a B-spline in V, linear in its control
  points, on clamped uniform knots, evaluated by Core's `bspline_basis`.
  `TensorVignetting(n, box; degree = 3)` (exported), a tensor surface of
  `n = (nx, ny)` control points over a sensor box, and
  `RadialVignetting(n, center, radius; degree = 3)` (exported), a profile in
  `‖u − center‖ / radius`, both under `AbstractVignetting`. Callable, `v(u)` at
  a sensor point (COLMAP pixels), constant beyond the knots; isbits, so they
  run in `pointwise!` on a device. Public: `coefficient_jacobian(v, u)`
  (`∂V/∂vec(P)`, the basis functions in `vec(control_points(v))` order),
  `active_coefficients(v, u)` (its nonzeros: the indices and values a
  scatter-add reads), `control_points`, `with_control_points`,
  `n_control_points`, `knot_vectors`, `spline_degree`, `field_center`,
  `field_radius`, `radial_coordinate`, `pole_regular_basis(v)` (the radial
  profiles smooth at their centre, `V′(0) = 0`: the first two control points
  tied) and `convert_eltype` (any real type, Duals included). A tensor
  surface implements the support protocol (`support_trait`, `nodes`,
  `support`), so `free_nodes`, `node_mask` and `BoundaryRule` apply to it.
  Hoisted from RadiometricCamera and PanoramaStitching, which each had one;
  their copies are deleted. RadiometricCamera's arithmetic is kept, so its
  numbers are unchanged.
- `LogVignetting(spline)` (exported): either field read as log V, `V = exp(s(u))`
  (positive, linear in log: a self-calibration's least squares). It adds no
  basis of its own: knots, degree, support, `pole_regular_basis` and
  `log_spline` are the spline's; `coefficient_jacobian` is `V·∂s/∂P`.
- `clamped_knots(n, p, lo, hi)` (public): the clamped uniform knot vector
  (BSplineKit's) as the static vector `bspline_basis` takes.
- `RotationMap{N, T, R, Src, Dst}` (exported, transforms/rotation_map.jl): a
  frame-tagged rotation about the origin, `x ↦ R·x`, whose one field is the
  rotation, so it has no translation rather than a zero one. The rotation is
  stored by EuclideanMap's rule. It provides relabel/unlabel, `inv` (tags
  swapped), and a junction-checked `∘`: with itself it gives a RotationMap, and
  with an EuclideanMap in either order it gives an EuclideanMap. Also
  `rotation`, `transform_deriv`, `to_homogeneous`, `convert_eltype`, `==`,
  `isapprox` and show. It holds no length (`has_units` false; `ustrip` and
  `with_unit` return it) and is isbits. `PosedCamera`'s pose parameter admits
  it (`Rt <: Union{EuclideanMap, RotationMap}`), and
  `OrientedCamera{M} = PosedCamera{M, <:RotationMap{3}}` (exported, an alias)
  is a camera that only turns about its centre. It projects, back-projects and
  unprojects about the origin; `extrinsics` and `pose` are RotationMaps, and a
  camera → world RotationMap labeled `CameraFrame → WorldFrame` is inverted
  into the extrinsics slot. These are refused by name because they need a
  position: the plane z = 0 verbs (`HomographyMat(camera)`, `RadialCamera`,
  `transfer_ellipse`, `project` of a plane conic or rect), a rig member, and a
  NoncentralGenericModel's pose. Plotting draws an oriented camera at the
  origin. Additive: every EuclideanMap-posed camera is unchanged.
- `view_overlap(cameras, [circles]; boundary_points = 512)` (exported,
  cameras/view_overlap.jl): for every pair i < j of a panorama's frames (each a
  `BoundedCamera` of an `OrientedCamera`; cameras may differ), a
  `StructVector` row `(; i, j, solid_angle, fraction_i, fraction_j)`, with
  K(K−1)/2 rows. The solid angle (sr) is that of the directions both frames
  image, and each fraction is it over that frame's footprint
  (`imaged_region(camera, circle) ∩ valid_domain(camera)`).
  `view_overlap(a, b, [circle_a, circle_b])` gives the one row. A camera posed
  by an EuclideanMap, or unposed, is refused by name. Boundary-sampled: frame
  j's ring is drawn into frame i's image and clipped by Core's
  Sutherland–Hodgman against Ω_i. Where frame i cannot image the ring, it
  continues on a far circle, and one exact membership test fixes the winding.
  The area is a fan of Van Oosterom–Strackee triangles from the optical axis,
  so a field of view past 180° is measured; a footprint within 0.05 rad of its
  backward axis is refused. Exact for perspective cameras, and O(1/n²) on
  curved boundaries. For PanoramaStitching's pair selection.
  `_convex_clip_area!` (primitives/operators.jl) now wraps
  `_convex_clip!(subject, clip)`, which returns the clipped ring with the
  subject's orientation kept (the area path is unchanged), and the spherical fan
  `_spherical_signed_area(apex, ring)` sits beside `_shoelace_signed_area`.

## 0.5.4 (2026-11-04)

Additive: every model built without a mask is unchanged, bit for bit.

- **A grid model holds the nodes its imaged region reaches.**
  `CentralGenericModel` and `NoncentralGenericModel` take `present` (a `Bool`
  matrix of the grid's size, `[gy, gx]`; `nothing`, the default, is every
  node, and an all-true mask is stored as `nothing`). The controls stay a
  dense matrix; an absent node's control is never read and has no meaningful
  value. The mask a fit over the imaged region takes is the footprint rule,
  `node_mask(m, imaged_region(m))`, under which every pixel of the region
  reads present nodes only, so the masked model evaluates the region bit for
  bit as the full one.
- **Where a masked model evaluates**: `in_present_domain(m, pixel)` (public),
  the calibrated area with every node of the pixel's 4×4 window present.
  Outside it the model answers as outside the calibrated area: `backproject`
  and `unproject` give `nothing`, `backproject_with_jacobian` a NaN pair,
  `project` (scalar, batched and the undistortion's coarse inverse) no pixel,
  the raw splines NaN, `spline_support` `nothing` and `spline_weights` an
  `ArgumentError`. Never a throw from an evaluation.
- `present_nodes(m)` (public): the mask, for a model or a `GenericModelWindow`
  (which carries its block of it). `free_nodes` and `node_mask` never name an
  absent node. `convert_eltype` and `Adapt` carry the mask (a dense `Bool`
  array on the device; the kernels read it as the host does).
- **A capture's camera settings, per frame** (captures.jl, exported):
  `ExposureSettings(exposure_time, f_number, iso)` (a time quantity, in
  seconds), `LensSettings(focal_length, focus_distance = nothing)` (the
  NOMINAL, marked focal length; an unknown or "infinity" focus is `nothing`),
  `FrameSettings`, and `CaptureSettings(frames; body, lens)`, an
  `AbstractDict` from frame index to `(; exposure, lens_settings)` with the
  body and lens `DeviceIdentity(make, model, serial)` beside it. Values are
  math, identity is an attribute: camera, sensor and lens models stay
  name-free. `is_constant(cs)` (public): every frame exposed alike. Every
  value is checked positive and finite; a plain exposure time is a
  `DimensionError`. conventions/persisted-geometry.md names the
  `capture_settings` kind and a calibration block's `device`.

## 0.5.3 (2026-11-04)

Additive.

- **`stable_seed(seed, keys...)`** (public, not exported): the seed of every
  random stream in the stack, the first eight bytes of the SHA-256 of `seed`
  and `keys` printed as text and separated by `\x1f`. It is fixed across Julia
  versions and keyed by name, so no arithmetic coincidence gives two keys one
  stream (as `seed + 1000i + j` did). Hoisted from VisualGeometryToolkitEval's `Stats`,
  which now re-exports this one; the same numbers. A package outside the
  evaluation stack (PanoramaStitching) seeds a StableRNGs `StableRNG` with it
  without depending on Eval. New dependency: SHA (a standard library).

## 0.5.2 (2026-11-04)

Additive.

- **`OnBackend(backend; workgroupsize = nothing)`**, an `ExecutionPolicy`
  of `pointwise!` (public, not exported): one kernel launched on the given
  KernelAbstractions backend value, so its compiler options hold
  (`CUDABackend(; always_inline = true)`), with an optional fixed workgroup
  size. A device broadcast launches on `get_backend(out)`, a backend with
  the defaults. Each element is `f(xs…[I])` over inputs of the output's
  shape; a functor reaches the kernel as its fields (separate kernel
  arguments, rebuilt on the device), since one large aggregate argument is
  copied to local memory; the launch is asynchronous like a broadcast's.
  RadiometricCamera's
  per-pixel flux, linear-block parts and dual Jacobian run through it
  instead of kernels of their own.

## 0.5.1 (2026-11-04)

Additive.

- **`ControlPatchManifold`'s ray-normal point chart.** A trailing
  `ControlPointChart` argument chooses how a non-central patch's point
  controls move: `AdditivePoints()` (the default, `p + (Δx, Δy, Δz)`, as
  before) or the new `RayNormalPoints()`, `p + a·t₁ + b·t₂` with
  `(t₁, t₂) = direction_tangents(normalize(d))` of the node's own direction
  control: 4 coordinates per non-central node `(δ₁, δ₂, a, b)`, with
  `retract`, `retract_jacobian` (`(t₁ t₂)` per point) and `local_coordinates`
  (the offset's dot products with `t₁`, `t₂`). Public, not exported:
  `ControlPointChart`, `AdditivePoints`, `RayNormalPoints`.
- **A grid lens renders on a GPU.** `Adapt.adapt_structure` for `BoundedMap`
  adapts its map's `ComposedTransformation` chain link by link, so a
  `SplineProjection` in an `image_warp` chain reaches `image_warp!`'s kernel
  with its control grid on the device; `ChartSpline`'s device form skips the
  validating constructor (KernelAbstractions' `@Const` re-adapts inside the
  kernel, where its error message cannot compile). The device warp equals
  the host warp.
- **`cubic_bspline_weights(u)`** (public, not exported): the four weights of
  a uniform cubic B-spline span at its local coordinate `u ∈ [0, 1)`,
  kernel-lawful (in `u`'s type, no allocation, no branch). It is the private
  `_cubic_weights` renamed, whose callers (the grid models' windows,
  `spline_weights`, `ChartSpline`) form `u` exactly as before: their weights
  are unchanged, bit for bit.
- **`mad_scale(x)`** (public, not exported): the centred, normal-consistent
  median absolute deviation `median(|x − median(x)|) / Φ⁻¹(3/4)` (1.4826·MAD,
  the constant exact). `noise_sigma` remains the uncentred median of residual
  norms.

## 0.5.0 (2026-11-04)

Breaking: one composition, CoordinateTransformations' `∘`, replaces the two
chain types, and CT's conventions hold for it.

- **A target calibration is a CoordinateTransformations `Transformation`.**
  `TargetCalibration` is now the union of the kinds: `Rigid`, the affine
  factors (a planar `EuclideanMap`, `Scale`, `Aspect`, `Skew`),
  `ThinPlateSpline`, `HeightField`, and their compositions. A `PlanarTarget`
  takes one kind, the affine factors in their canonical order
  `EuclideanMap ∘ Scale ∘ Aspect ∘ Skew` (any of them left out), or a
  `HeightField` after those or after a `ThinPlateSpline`; it refuses any other
  composition with an `ArgumentError` and stores a composition
  right-associated, whatever association `∘` built. A downstream calibration
  kind is no longer possible (the union is closed).
- **`AffineMap(K)` of a 3×3 matrix is `affine_part(K)`** (exported): the
  method `CoordinateTransformations.AffineMap(::StaticMatrix{3, 3})` was Core's
  method on CT's function and CT's type (type piracy) and is deleted, with no
  alias. `affine_part(M)` returns the same `AffineMap` (the 2×2 linear block and
  the translation column; the bottom row is not read), for a camera calibration
  matrix, a `HomAffineMat` or any 3×3 static matrix. `AffineMap(A, b)` is CT's
  own and unchanged. `CoordinateTransformations.AffineMap` leaves Aqua's
  `treat_as_own` list in the quality test: no piracy of it remains.
- **Removed `CalibrationChain`**: `CalibrationChain(A, H)` is `H ∘ A`
  (CoordinateTransformations' `∘`, applied right to left). A composition's
  parameters, names, bounds, Jacobian columns, length mask and
  regularizer rows are in CT's order, the OUTER map's (the height field's)
  first — the reverse of the chain's application order. Names keep their tag
  prefixes (`:height_field_h1`, …, `:aspect_a1`).
- **Removed `Affine`, and the affine kind with it: the affine print error is
  its gauge-free Iwasawa factors.** `A = s·R·D·U` (the KAN decomposition): new
  N-generic CoordinateTransformations kinds `Aspect` (`x ↦ D x`, `D` a det-1
  diagonal, its `N − 1` free log-stretches `(:a1, …)`), `Skew` (`x ↦ U x`, `U`
  unit upper-triangular, its `N(N−1)/2` strictly-upper entries row by row
  `(:u12, :u13, …, :u23, …)`) and `Scale` (`x ↦ e^k x`, `(:log_scale,)`),
  and Core's planar `EuclideanMap` as the rigid member (`(:angle, :tx, :ty)`;
  new `transform_deriv_params(::EuclideanMap{2}, x)`, minimal, and
  `transform_deriv(::EuclideanMap, x)`). The rigid member's ANGLE is its
  value: it holds its rotation as a Rotations.jl `Angle2d` (`with_parameters`
  builds `Angle2d(ϑ)`, `parameter_vector` reads `rotation_angle`, the stored
  ϑ), so `parameter_vector(with_parameters(c, θ)) == θ` bit for bit for every
  θ — a `RotMatrix{2}` misses ϑ by an ulp through `atan` for about one angle
  in ten. A member built from any other `Rotation{2}` is converted once, where
  a `PlanarTarget` stores it (its `rotation_angle` becomes the value), and
  `EuclideanMap` now stores an `Angle2d` rotation as given, as it does a
  `RotMatrix` or a `QuatRotation`. None has gauges. With the pose
  estimated jointly the print error is `Aspect ∘ Skew` (two parameters, where
  the affine kind had six and four gauges); a measured length adds `Scale`, a
  known pose the rigid member: `EuclideanMap ∘ Scale ∘ Aspect ∘ Skew`. A
  `PlanarTarget` refuses an `AffineMap` calibration with an `ArgumentError`
  naming the composition. The tag `"affine"` is retired in
  `TARGET_CALIBRATION_TAGS` (successor `"aspect"`, the composition's outer
  member); new tags `"euclidean"`, `"scale"`, `"aspect"`, `"skew"`. The
  registry no longer holds `"calibration_chain"`, and `calibration_tag` of a
  composition is an `ArgumentError`. Core no longer adds methods to
  `transform_deriv_params` or `FlexUnits.ustrip` on CoordinateTransformations'
  `AffineMap`: that type piracy is gone, and `AffineMap` stays on Aqua's
  `treat_as_own` list for one constructor only, `AffineMap(::StaticMatrix{3, 3})`
  (the pixel affine of a calibration matrix).
- **The thin-plate kinds are pure bending, and no kind has a gauge.**
  `ThinPlateSpline`'s displacements and `HeightField`'s heights have no affine
  term in their interpolant (the spline's affine coefficients `a` are zero:
  pure bending, intrinsic to the spline): the constructor and
  `with_parameters` project each nodal value vector, `v ↦ v − M a(v)` with
  `M = [1 u v]` at the nodes, which keeps the bending (values whose affine
  term is within 1e-12 of them are kept bit for bit, so the projection is
  idempotent, a stored spline rebuilds exactly, and RadiometricCamera's board
  heights — its spline's bending part at its chart nodes — are kept unchanged;
  "no affine term" is defined on the nodes the spline is built on, since the
  kernel r² log r moves a piston under a scaling of them: build on a
  dimensionless chart).
  A spline built from an affine displacement is the identity, a height field
  built from a plane of heights flat. The canonical print error is
  `HeightField ∘ ThinPlateSpline ∘ Aspect ∘ Skew` (a `ThinPlateSpline` now
  composes before the affine factors). The stored parameters stay the 2K (K)
  nodal values, with 2K − 6 (K − 3) free directions; a covariance over them
  has that rank. **Removed `gauge_names` and `gauge_directions`** (retired in
  `check_naming`) and `TargetManifold`'s `free` keyword: `TargetManifold(t)`'s
  basis is the identity for the affine factors and, for the thin-plate kinds,
  an orthonormal null-space basis of the interpolant's affine-coefficient rows
  from a pivoted Householder QR (reproducible; the 0.4 default fixed the
  gauge columns orthogonally, another complement of the same plane fields).
  `restrict(ch, names)` keeps the constraint: the named parameters'
  directions that stay pure bending.
- **`ChartPlacement(origin, side)`: a calibration on the board's dimensionless
  chart.** A `PlanarTarget` takes `placement = ChartPlacement(o, s)` (new
  field, accessor `placement(t)`, public; the positional form's sixth
  argument): its calibration then acts in the chart `c = (p − o)/s`, plain
  numbers whatever the points' form (a calibration with units is an
  `ArgumentError`), and `points(t)` is `p + s·(C(c) − c)` in the plane and
  `s·h(c)` along the normal — exact for an in-plane identity (a height field
  alone keeps the design point bit for bit). The placement is fixed data in the
  points' form, carries the lengths, moves with `ustrip`/`with_unit` (the
  calibration does not), and adds no parameters: the calibration's names,
  manifold and regularizer are unchanged, `action_jacobian` scales by `s`.
  `surface` re-expresses a placed height field on the design chart exactly
  (nodes `o + s cᵢ`, weights `wᵢ/s`, the kernel's constant `−s ln s Σ wᵢ|cᵢ|²`).
  This is how RadiometricCamera's board warping — pure bending on its page's
  chart `[0, 1]²`, rows down: the reflected nodes `(u, −v)` placed at the
  page's top-left corner — keeps its heights bit for bit and its surface,
  piston included, which building the field on the page's design nodes loses
  to the pure-bending projection.
- **Removed `calibration_jacobian`** (retired in `check_naming`): a
  calibration's parameter Jacobian is CT's `transform_deriv_params(c, x)`, its
  point Jacobian CT's `transform_deriv(c, x)`. `ThinPlateSpline` and
  `HeightField` keep their `physical` flag (their spline solver, coefficients
  and bending factor are plain arrays shared bit for bit with
  RadiometricCamera, and their rebuilds run on ForwardDiff Duals).
- `Base.hash` of a composition of the stack's kinds agrees with CT's
  structural `==`; equal targets hash equally.
- **Removed the lens `Chain`**: distortion terms are CT `Transformation`s
  (`AbstractDistortionTerm <: Transformation`, each callable on a frame
  point), and terms in sequence are CT's `∘`: `Chain(D₁, D₂)` is `D₂ ∘ D₁`
  (OpenCV's tilted model `Tilt(τx, τy) ∘ Additive(…)`, Fisheye624
  `Additive(Tangential(…), ThinPrism(…)) ∘ Radial(…)`). A composed lens's
  coefficients, names, bounds and Jacobian columns are in CT's order, the
  outer stage's first; its `lens_record` lists the stages under the new key
  `"compose"`, outer first, and `lens_from_record` refuses the retired
  `"chain"` (application order) with an `ArgumentError` naming `"compose"` — a
  lens-valued TOML config field (`toml_value`) is the same record, so a config
  that wrote `chain` lists its stages under `compose`, outer first. A composed
  distortion is stored right-associated whatever association it was built in
  (`lens_from_record` builds that form), so equal stage sequences are one
  value: `==`, `hash` and the record round trip agree. Every
  projection, back-projection and Jacobian is bit for bit the chain's.
- **The grid models' projection seed is positional**: the `init` keyword of
  `project` on `CentralGenericModel` and `NoncentralGenericModel` is removed
  (keywords are configuration only; a seed is per-call data).
  `project(m, X)` starts cold at the calibrated area's center, as before;
  `project(m, X; init = p)` is now `project(m, X, p)`, and the batched
  `project(m, rays; init = p₀s, iters, tol)` is `project(m, rays, p₀s; iters, tol)`,
  one seed per ray. Closed-form cameras take no seed.

Added:

- **`ChartSpline(area, controls)`**: a tensor-product uniform cubic B-spline
  from a 2-D chart to ℝᴷ — the grid models' lattice (`spline_value`) as a
  CoordinateTransformations `Transformation`, its `transform_deriv` the
  differentiated cubic pieces, a batch of chart points evaluated by a
  KernelAbstractions kernel.
- **`SplineProjection(chart, spline, region)`**: a forward map ray → pixel as
  a `ChartSpline` on the equidistant direction chart about the optical axis
  (`Lens(Equidistant()) ∘ LinearMap(R)`), a `Transformation` that composes in
  `image_warp` chains; `project(sp, X)` is `nothing` when the pixel leaves the
  region it was fitted over (VisualGeometryToolkitEstimation's `fit_camera`).
- **Batched `backproject(m::CentralGenericModel, pixels)`**: a KernelAbstractions
  kernel on the pixels' backend, bit for bit the scalar form on the CPU, NaN
  outside the calibrated area.
- `pixel_mask(region, width, height)`: the pixels whose centres lie in a region.
- `pixel_centres(region, us, vs)`: the COLMAP centres of the pixels of a
  sub-lattice (`u ∈ us`, `v ∈ vs`) that lie in a region, without the full
  image's mask; `pixel_mask` and it share the one centre conversion.
- **`nearest_point(lines)`**: the least-squares point nearest to N lines,
  `origin => direction` pairs (a `Csponds` or a vector of `Pair`s, any
  dimension), the minimizer of `Σ wᵢ‖(I − dᵢdᵢᵀ)(x − oᵢ)‖²` in one closed-form
  solve about the origins' centroid; per-line weights as the `weights` column;
  origins that are lengths come back in metres; parallel lines are an
  `ArgumentError`. The N-line counterpart of the two-view `triangulate`.
- **`generic_lattice(G, source, size; grid_size, calib_area, nominal, domain)`**:
  the layout `CentralGenericModel(source, size; …)` or
  `NoncentralGenericModel(source, size; …)` takes, with the constructor's
  defaults, without sampling the source (every direction control the optical
  axis, every point the origin) — the start of a fit that solves for every
  control (VisualGeometryToolkitEstimation's `fit_camera` grid targets); the
  two constructors build their seed through it.
- `difference_stencil(order)`: a lattice's forward-difference coefficients
  (`(1, −3, 3, −1)` for Δ³), the one source of `BoundaryRule`'s
  curvature-preserving extrapolation and of `fit_camera`'s P-spline penalty.
- **`backproject_with_jacobian(m::NoncentralGenericModel, u)`**: the unit
  direction of the pixel's observation line and its Jacobian `∂f/∂u`,
  analytic from the direction spline's derivative weights, tangent to the
  sphere; NaN outside the calibrated area (the central model's contract).
- **The cylindrical family of panorama projections**: `Cylindrical` (rows
  `tan φ`, the central projection onto the unit cylinder),
  `CylindricalEqualArea` (Lambert's, rows `sin φ`: equal raster areas are
  equal solid angles) and `Mercator` (rows `asinh(tan φ)`, conformal) beside
  `Equirectangular`, under the new `AbstractPanoramaProjection`. Each is a
  CoordinateTransformations `Transformation` — its call is `project`, its
  `transform_deriv` the analytic 2 × 3 Jacobian — whose `inv` is its
  `Backprojection`, so a panorama raster's pixel → ray map composes into an
  `image_warp` chain (`camera ∘ LinearMap(R') ∘ inv(projection) ∘ raster`).
  `Equirectangular` now subtypes it too (its values unchanged).
- **`Backprojection(m)`**: `backproject(m, q)` as a `Transformation` (its
  `transform_deriv` the model's own `backproject_with_jacobian` for a
  `CameraModel` or a generic grid model, ForwardDiff through `backproject` for
  any other, a `BoundedCamera` included; an integer pixel is floated first),
  for a panorama projection or a camera model
  — the pixel → ray map of a virtual camera (an azimuthal view: rectilinear,
  fisheye, stereographic, by the model's lens).
- **`image_sample!(dst, src, points; fillvalue)`**: `image_warp!`'s bilinear
  tap and boundary at scattered COLMAP points the caller holds, one
  KernelAbstractions kernel on the backend that owns `dst` (a value sampled at
  a point equals a warp landing on it).
- **`noise_sigma(norms, codim)` is public** (not exported): the robust noise
  sigma of residual norms, `median(norms) / √chisq_median(d)`, which
  downstream calibrations call qualified (`VgtkCore.noise_sigma`).
- **A target calibration composition's structure is public** (not
  exported): `members(c)`, its leaves outer first in any association;
  `TARGET_CALIBRATION_ORDER`, the canonical order as the kinds;
  `planar_kind(K, n_nodes)`, the type a planar target holds for a kind (its
  `length_mask` the record's parameter layout); `is_pure_bending(system, v)`,
  the thin-plate kinds' rule on a `ThinPlateSplineSystem`, with
  `ThinPlateSpline(system, fx, fy)` and `HeightField(system, h)` building on
  a system already formed; and `iwasawa_parameters(A)` /
  `iwasawa_parameters_with_jacobian(A)`, an affine's `A = s·R·D·U` as
  `(ϑ, log s, a₁, u₁₂)` (the factorization VisualGeometryToolkitIO's
  migration of a retired `"affine"` record held privately). A
  `ThinPlateSplineSystem` stores its plane fields `[1 x y]` and its affine
  rows (`plane`, `affine`), which the thin-plate kinds read instead of
  holding their own copy. Values are unchanged, bit for bit, except a
  composed target manifold's `action_jacobian`, which applies each member's
  action to its own rows and chains them (last-ulp differences), each member
  acting on the point in its own form (a physical target's rigid member
  translates in metres).

Fixed:

- **`Quad{T}(q)` and `HeightSurface{K, T}(s)` return the requested `T`**, a
  non-concrete one included (`Quad{Real}(q) isa Quad{Real}`); they returned
  the converted numbers' concrete type.

- **`Rect ∩ Circle`** (and `Circle ∩` any region, `Circle ∩ WholePlane()`):
  a `Circle` meets a region as its `Ellipse`, so
  `imaged_region(camera, circle::Circle)` builds instead of throwing a
  `MethodError` on `iterate` (it fell through to Base's set `intersect`).
- **Region membership takes any `StaticVector{2}`**: `p ∈ region` for a
  `RegionIntersection`, `WholePlane` and `Ellipse` is defined on
  `StaticVector{2}` (a `Point2` is one, through the same method), so
  `SVector(x, y) ∈ Rect ∩ ellipse` answers instead of throwing a `MethodError`
  on `iterate`. A `Rect` already took any static vector (GeometryBasics); a
  `Polygon` stays `Point2`-only (widening it would be type piracy).
- **A `HeightSurface`'s Dual `intersection` is the implicit derivative.**
  With ForwardDiff `Dual`s in the surface, the origin or the direction
  (nested ones and several tags included), Newton runs on the values and the
  partials are `−(∂g/∂p)/g′` at the root, one `Dual` evaluation instead of
  differentiating every iteration. The value parts are the plain result's,
  bit for bit, misses included (NaN with zero partials, as before); the
  partials agree with the old ones to rounding (3e-15 relative in Float64,
  the Float32 rounding of the same derivative in Float32), except at grazing
  incidence, where the old ones stopped up to 4e-11 short of the converged
  derivative and the new ones agree with a 200-bit reference. Plain inputs
  are unchanged, bit for bit. **`intersection(s, o, d; derivative)`** takes
  the choice, an **`ImplicitRoot(min_partials = 1)`** (public): a `Dual` with
  fewer partials runs Newton in the `Dual`s, as before. The default is the
  implicit derivative at every partial count, the CPU's choice; a kernel launch
  supplies **`implicit_root(backend)`** (public), `ImplicitRoot(8)` on a
  KernelAbstractions GPU. Measured on a 25-node surface with Duals in the
  heights and the ray, old/new: on an RTX 4090, `Dual{Float32, N}` for N = 1,
  2, 4, 6, 8, 12, 15, 32 gives 0.50, 0.55, 0.59, 1.01, 1.84, 2.26, 2.52, 2.84×
  (RadiometricCamera's GPU Jacobian at N = 4: the implicit path 10% slower);
  on the CPU, `Dual{Float64, N}` gives 1.2–3.7×. A `Quad`'s `intersection`
  takes the same keyword and ignores it.

Performance (same bits):

- **`surface(pt)` of a physical target with an `Uncertain` calibration** drops
  the covariance before the metre strip instead of rescaling it on every call
  (a 25-node height field: 27 kB a call to 6 kB, the bare calibration's).

- **`nodal_weights` of a thin-plate system allocates nothing at any node
  count.** Past 16 nodes the loop form built its basis in a heap `Vector`
  (288 B a call at 25 nodes); it is a stack `MVector` folded in the same
  order, so the weights are the same bits. A ForwardDiff `Dual` point now
  gives `Dual` weights (it was a `MethodError`).
- **A Dual camera's `backproject` inlines on the device.**
  `backproject(::CameraModel, u)` is `@inline`, and `BoundedCamera`'s
  `project`/`backproject` are explicit `@inline` forwarders instead of
  `@forward`'s; the equidistant frame's `sin`/`cos` are inlined at the call
  site. A KB camera in `Dual{Float32, 15}` (RadiometricCamera's device
  Jacobian) went from 124 registers and 3008 B of local memory a thread to
  121 and 1160 B. `evalpoly` stays a call: forcing it inline moved `muladd`
  fusing, and so last ulps.

## 0.4.0 (2026-11-02)

Breaking: ONE Sampson path. Nothing outside `base/sampson.jl` computes
`g/‖G_x‖`, its whitening or its model Jacobian.

- **NLLSsolver ≥ 5.3.1** (`[compat]`), which contains our pull requests
  #61–#65. Its typed `OptimizeCache` is always used: the cache detection and
  the uncached path are gone, and so are `lmsolve_chart!`'s `cache` keyword
  and the `cache` field of `nlls_features`, which now reports only the Schur
  system (`(; schur)`, upstream #60, still open). NLLSsolver stops at a
  non-finite start point (#65); its termination bit 11 is decoded as a failure
  (`nlls_stop_code` `-3`, `nlls_stop_reasons` `:nonfinite_start`). The chart
  LM still reads a non-finite cost as `+Inf`: NLLSsolver accepts any trial at a
  step below `xtol`, and its per-iteration `cost == cost_` check throws an
  `AssertionError` on NaN.

- **`sampson_residual_with_jacobian(M, x...; directions)` → `(s_N, J)`**, the
  Sampson residual of any constraint and its EXACT model Jacobian along the
  chart `directions` (an `nθ × N` matrix, `retract_jacobian`, or the models
  `retract_directions` returns) — for a vector constraint (a homography's two
  rows) the Cholesky factor of `G_x·G_xᵀ` differentiated with the model — from
  `constraint_with_jacobians` and a new model protocol,
  **`measurement_gradient_jacobian(M, x...)`** (`∂vec(G_x)/∂θ`), implemented by
  `FundMat`/`EssentialMat` (plane and bearings), `HomographyMat`,
  `TransferForm` (pixels and bearings), `Circle`, `RadialCamera`, `Conic` and
  `Ellipse`. A degenerate measurement has a zero row. ForwardDiff-safe and
  allocation-free.
- **A problem's constraint is `(g, G_x, G_θ)`, as a model's**: the problem
  protocol gains two hooks, **`measurement_covariance(problem, i)`** (the point's
  noise, `I` by default; a contract generic) and
  **`constraint_rounding_scale(problem, model, i)`** (its rounding scales,
  `nothing` by default). The 4- and 5-tuples that carried them are gone, and a
  longer tuple is an `ArgumentError`, never silently truncated.
- **`whitened_constraint_residual(problem, model, i)`** and
  **`whitened_constraint_rows(problem, model, i)` → `(r, J_w)`** replace the
  tuple forms: the point's constraint whitened by its `measurement_covariance`
  (one Cholesky of `G_x·Σ·G_xᵀ`), with the degeneracy rule at its rounding
  scale; the rows are its model rows under the same factor, held fixed (zero
  rows when degenerate; residual 0 when the constraint vanishes). What a
  predictive scorer reads per point. `whitened_residual`'s default is the first.
- `whiten` selects `Σ = I` by dispatch.
- **Batches over a MEASUREMENT SET**: `sampson_residuals!(r, M, xs)`,
  `whitened_residuals!(r, M, xs)`, `sampson_jacobian_rows!(J, M, directions,
  xs)` and `whitened_jacobian_rows!(J, M, directions, xs)` — `xs` a
  **`PointSet`** (new: points with attribute columns; `withattrs(points;
  …)` makes one) or a `Csponds` / vector of `Pair`s, its per-element data
  ATTRIBUTE COLUMNS: the noise (`Σ`; `Σ₁`/`Σ₂`), the constraint's keyword
  columns (**`constraint_keywords(M, xs)`**: `M₁`, `M₂` of a bearing
  `EssentialMat` or `TransferForm`) and the rows' `sqrt_weights`. Element `i`
  takes rows `dg(i−1)+1 … dg·i`; one body each, a CPU loop or one
  KernelAbstractions kernel over the per-element function; a degenerate
  element is zero (LM-safe). The whitened rows differentiate the noise's
  Cholesky factor with the model: exact. Reductions compose — the RSS is
  `sum(abs2, …)`, the pool matrix **`sampson_distances!(R, models, xs)`** (one
  kernel per model column; it reads a model's `constraint_keywords` columns, so
  an essential matrix on bearings takes its `M₁`/`M₂`).
  The image-plane and tangent epipolar row functions that restated the
  residual and its contracted Jacobian, the public per-row forms and the
  generic Sampson-Jacobian helper are removed; the bilinear contraction lives
  on as the epipolar models' way of evaluating `(∂G_x/∂θ)ᵀG_x`.
- **The epipolar degeneracy scales are the models' `constraint_rounding_scale`**
  (the separate public scale function is removed).
- **The whitening-policy types and their row verb are removed.** A frozen
  whitening (an iteratively reweighted fit's rows, a Gauss–Newton row) is
  `whitening_factor(G_x[, Σ_x])` — now also for a scalar constraint's
  gradient, a number — times the algebraic rows; the whitened residual of a
  problem's point is `whitened_constraint_residual(problem, model, i)`.
- **The epipolar and homography constraints promote** a model and points of
  different element types themselves (the Sampson layers' private promotion
  hook is gone); `EpipolarForm` forwards `measurement_gradient_jacobian`.
- **Rounding scales** for `Circle` and `RadialCamera` (relative to their
  entries; the exact-zero default stays for `Line2D`, whose gradient is a unit
  normal, and `TransferForm`, whose gradient blows up rather than vanishes); a
  `Conic`'s reuses the epipolar bilinear bound. `gradient(::Conic, p)` is the
  point-on-conic constraint's `G_x`.
- **The homography transfer rows with a whitening frozen at the anchor are
  removed** (`transfer_whitening!`, `transfer_rows!`, `transfer_jacobian_rows!`):
  an LM on a homography or a rotation descends the exact whitened rows of the
  constraint it is scored with. `TransferForm` stays, the transfer constraint.
- Removed: `sampson_rows!` (the parallel-array forms; `sampson_residuals!` over
  a set), `sampson_rss` (`sum(abs2, …)` of the batch), the n × M
  `sampson_residuals!(R, models, u₁, u₂)` (`sampson_distances!`).
- **GPU**: the 1 × 1 and 2 × 2 whitening Cholesky factors are written out with
  IEEE `sqrt` (StaticArrays' takes `Base.FastMath.sqrt_fast`, approximate on
  CUDA: a homography's Sampson residual was off by 5e-5 relative on the
  device); the CPU factors are StaticArrays' bit for bit. On an RTX 4090 every
  batch function agrees with the CPU to ≤ 4e-14 (Float64) for F, E, H and a conic.
- Removed (private): `_propagate_scalar`.
- Removed: `sampson_residual_jacobian`, `sampson_residual_row`,
  `tangent_sampson_residual_row` (all three are `sampson_residual_with_jacobian`);
  `epipolar_rounding` (`constraint_rounding_scale`); `WhiteningPolicy`,
  `NoWhitening`, `SampsonWhitening`, `whitened_rows` (`whitening_factor`,
  `whitened_constraint_residual`).
- `check_naming` reads a `CHANGELOG.md` for banned words but not for retired
  names: a changelog records what each release removed.

## 0.3.0 (2026-11-02)

Breaking: the Sampson distance is geometric, and the noise whitening is a
second layer over it.

- **`sampson_residual_with_normal(M, x...)` → `(s_N, N)`**, written once for
  every model with a `constraint_with_jacobians`: the Sampson residual (always
  an `SVector{dg}`) in an orthonormal basis `N` of the constraint's normal
  space, with the one degeneracy rule at the model's `constraint_rounding_scale`
  (now varargs, `(M, x...; kw...)`, and a contract generic).
  `sampson_residual` is its first part and **`sampson_distance` its norm, for
  any such model** (F, E on the plane or on bearings, H, the forms, a `Conic`,
  an `Ellipse`, a `Circle`, a `Line2D`). **A measurement's covariance no longer
  enters `sampson_distance`**: an `Uncertain` or `exact` point is read as its
  point.
- **`whiten(s_N, N, Σ)`** (was `whiten(g, G_x, G_θ, Σ_x, ρ) -> (r, J, ℓ)`):
  the Sampson residual whitened by `NᵀΣN`, exactly the constraint whitened by
  `G_x·Σ·G_xᵀ`. **`mahalanobis_distance(M, x...)`** reads `Σ` from the
  measurements (plain `I`, `Uncertain` its own, `exact` 0): the distance
  `sampson_distance` used to return for uncertain points. The log-determinant
  the old `whiten` returned is gone (nothing in the stack read it).
- **`constraint(M, x...)`**: `g` alone, the `<f>` of `constraint_with_jacobians`
  (algebraic, model-scale dependent: not a distance).
- **`constraint_with_jacobians(c::Conic, x)`** and the rounding scales of a
  `Conic` and an `Ellipse` (an ellipse's centre is degenerate: `Inf`).
- `whitened_rows(SampsonWhitening(), g, G_x, G_θ, Σ_x)` whitens a
  heteroscedastic constraint; `whitened_constraint_residual` reads a 4- or
  5-tuple's `Σ_x` and rounding scale as before.
- `test_local_optimization_contract([T = Float64,] lo, problem, θ, scoring;
  allocations)`: the element type is an optional leading positional (slot 4 of
  the signature rule), no longer the keyword `T`.
- `mean_average_accuracy(errors; max_error)` and `maa_score(e; max_error)`:
  the threshold keyword `T` is `max_error` (one name per quantity; `T` is the
  type-parameter letter).
- Retired: `signed_sampson_distance` (`only(sampson_residual(M, x...))`), the
  `Σ` keyword of the essential matrix's distance (`whiten(sampson_residual_with_normal(E, f₁, f₂; M₁, M₂)..., Σ)`),
  `signed_sampson_jacobian` → `sampson_residual_jacobian`,
  `signed_sampson_row` → `sampson_residual_row`,
  `signed_tangent_sampson_row` → `tangent_sampson_residual_row`.

## 0.2.1 (2026-10-29)

- `conventions_dir()` is resolved from the package's source location instead of
  `pkgdir`, which may be `nothing`; the result is the same folder.

## 0.2.0 (2026-10-29)

Behaviour changes a caller can observe, stated here because the commits that
introduced them did not state them all.

- **`Equirectangular`**, the latitude/longitude projection of directions
  (`project` → `(λ, φ)` in radians, `backproject` → the unit direction), in
  the camera frame's axes: longitude 0 forward and +π/2 right, latitude +π/2
  up (−y). A projection beside the lens families, not a lens.
- **`valid_domain(camera)`, `image_circle(camera, sensor, diameter)` and
  `imaged_region(camera, circle)`: where a lens model is valid is not where
  light falls.** `valid_domain` is the image of the lens's
  `view_angle_limit` cone (its fold, pole or bound; an `Ellipse`, inscribed for
  a lens that is not radially symmetric, the `WholePlane` without a limit):
  where the MODEL is radially injective and finite. `image_circle` is only the
  datasheet diameter through the pitch; there is no form that predicts it from
  the lens. `imaged_region` takes the image circle (a datasheet's or a
  measured one) as an argument. A generic grid model built from a parametric
  camera no longer takes the lens's fold as its `domain`: it takes the
  `domain` keyword, `WholePlane()` by default.
- **`Rotation2pt`**, the minimal 2-point rotation solver on bearing pairs
  (`solve(Rotation2pt(), cs, idx)`: `kabsch` on the two unit pairs; a
  parallel or antiparallel pair in either view is `count == 0`).
- **The transfer constraint on unit bearings**:
  `constraint_with_jacobians(TransferForm(H), f₁, f₂; M₁, M₂)` — the infinite
  homography (a rotation under a pure rotation) as the transferred ray in
  `f₂`'s tangent plane, `G_x` in pixels through the back-projection Jacobians
  (the Sampson distance of the joint measurement, `Σ = I`), `G_θ` w.r.t. `vec(H)`; and the bearing
  methods `transfer_whitening!(W, H, f₁, f₂; M₁, M₂)` and
  `transfer_jacobian_rows!(J, H, dH, W, f₁, f₂, sqrtw)` beside the pixel ones
  (`transfer_rows!` takes bearings as it is). The pixel rows' internal row
  function now takes `u₂` (unread for pixels).
- **`spectral_rotation_averaging(relative[, weights])`**, spectral chordal
  rotation averaging: the absolute rotations of a set of frames from their
  relative rotations (`f_j = R_ij f_i`), the leading eigenvectors of the
  degree-normalized block matrix projected onto SO(3), gauged at the first
  frame — the seed of a robust rotation averaging (it replaces the gauged
  linear `chordal_averaging` of this release).
- **`RotationChart{P, Rot}` and `SO3Manifold`**, the 3-dof rotation member of the
  rigid chart family (`R(ω) = ΔR(ω)·R`, over the family's increment, logarithm
  and left Jacobian; `SO3Manifold = RotationChart{ExpParam}`), with closed-form
  `retract_jacobian`, `action_jacobian` and `local_coordinates`;
  `minimal_manifold` of a rotation.

- **`parameter_bounds` states which endpoints are closed, and a closed
  endpoint is a mirror in `ParameterManifold`'s chart.** The closed endpoints
  are the values a model takes: the double sphere's `α = 0` and `α = 1`, the
  extended unified model's `α = 0` (the pinhole), `α = 1` and `β = 0`, and
  the field-of-view term's `ω = 0` (the identity). There, the chart is the
  open ends' chart (`t₀ + v` when there are none) reflected at the endpoint:
  a value past it folds back inside (with two closed ends, with period
  `2(b − a)`), and the slope keeps its magnitude with its sign flipped. A
  clamp would leave a flat residual past the end, at which a fixed-chart fit
  stalls. Open endpoints keep the log/logit chart and its 10⁻⁶ anchor
  margin. As a result:
  - an anchor at `α = 0` or `ω = 0` stays there and moves with slope 1
    (`ω`: slope `π − ω₀`). Before, it was moved to 10⁻⁶ of the interval with
    a slope of about 3·10⁻⁶.
  - the double sphere's `α` chart is now additive. It is no longer the logit
    that 0.1 used, so `retract` values on `α` differ from 0.1's.
  - `β` of the extended unified model is additive and reflected at 0. It is
    no longer `β₀·exp(v)`.
- **A negative field-of-view `ω` is anchored as `|ω|`.** `ω` and `−ω` are one
  map, so this is the same lens. Before, `FieldOfView(−0.3)` was anchored at
  `ω ≈ 3·10⁻⁶`, a different lens.
- **`ω = 0` is a stationary point of the field-of-view model itself.** Its map
  is even in `ω`, so `∂project/∂ω = 0` there whatever the chart is. Start a
  fit of `ω` at a small `ω ≠ 0`.
- **A coordinate with no step keeps its value exactly.** `retract` with a zero
  chart offset returns the anchor's value bit for bit, so a coordinate that
  `restrict` holds fixed never drifts across recenters. Before, a fixed
  coordinate under a log or logit chart moved by an ulp on each recenter:
  `ξ = 0.1` became `0.10000000000000009`.
- **`Adapt.adapt` of a generic grid model replaces a domain that is not isbits
  with `WholePlane()` on the device copy.** An isbits domain, such as an
  ellipse or a rectangle, travels with the model. The domain is host-side
  bookkeeping that no device evaluation reads.

### The LM driver: NLLSsolver (2026-10-27)

Every nonlinear least-squares fit now runs on NLLSsolver, through
`lmsolve_chart!` and `nlls_options`. The previous driver is no longer a
dependency. What changed for a caller:

- `max_iter` counts **accepted** iterations. The previous driver also counted
  rejected candidate steps.
- `xtol` is **absolute**: the max-norm of the step in chart units, `max|pᵢ| <
  xtol`. The previous driver's test was relative per component, `|pᵢ| <
  xtol·(xtol + |xᵢ|)`. The same value is now stricter far from the origin and
  looser near it.
- `ftol` is NLLSsolver's relative cost decrease. The previous driver required
  both the actual and the predicted reduction of an accepted step to fall
  below `ftol·F`. A step whose cost is NaN is still rejected (the adapter
  reads it as +Inf).
- There is no gradient tolerance. The stop codes that the stall postcondition
  reads (step-size stop `2`, cost-decrease stop `1`) keep their meaning.

Measured against the previous driver:

- **Chart LM, E refits.** On 317 captured E refits of the PhotoTourism
  replay at `ftol = 1e-10`, the returned chart vector agreed within 2.0e-7
  (max-norm; the median refit's ‖x‖∞ is 1.8e-3) and the cost within 2.0e-11
  relative, with every refit converged. Time to convergence was 0.84× the
  previous driver's when Jᵀr and JᵀJ are accumulated in one sweep, and 0.97×
  when NLLSsolver forms JᵀJ itself.
- **`ftol` for the chart LM.** The previous driver's default, eps, ran every
  refit to its rounding floor. On the PhotoTourism E replay (360 Welsch-IRLS
  LO calls, 2026-10-07) a refit took 28.6 residual and 8.0 Jacobian sweeps,
  about 20 of them rejected steps at the floor. At 1e-10 it took 5.6 and 4.1
  sweeps and was 2.3× faster, with the returned E within 7.8e-7 (unit
  Frobenius, the maximum over the 360).
- **Relative-pose bundle.** Measured on the A/B scenes (800 px, σ = 0.5 px,
  N = 100 / 1000 / 10000):
  - At eps, the previous driver took 21–34 iterations. At 1e-10 it stopped
    in 5, with the cost within 4.2e-14 relative and the pose within 6e-9
    chart units of the eps run.
  - NLLSsolver also stops in 5 at 1e-10. Both drivers reach the same optimum:
    the cost agrees to ≤ 3e-14 relative and the rotation to ≤ 6e-11°.
  - At the stop the agreement is: NLLSsolver cost ≤ 6.2e-14 relative,
    rotation ≤ 2.1e-8°, translation direction ≤ 1.8e-7°; the previous
    driver 4.6e-14, 1.8e-8° and 1.6e-7°.
  - Time relative to the previous driver was 0.88 / 1.04 / 1.00× with the
    cached problem and 1.78 / 1.49 / 1.44× without it.
  - Allocations per call: 19 / 20 / 20 now, against the previous driver's
    53 / 54 / 54.
- **Evaluation-engine golden.** The IRLS errors moved by at most 3.1e-6
  relative (1.1e-5 absolute), and the trial counts did not move.
- **Pixel LM kernel.** The generic models' kernel (`Cameras._pixel_lm`, the
  one device-side LM) keeps the previous driver's static bounded algorithm.
  It is gated at 2e-6 px against a recorded run of that routine
  (`test/fixtures/pixel_lm`).
