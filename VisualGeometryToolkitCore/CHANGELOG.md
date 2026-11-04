# VisualGeometryToolkitCore changelog

## Unreleased

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
