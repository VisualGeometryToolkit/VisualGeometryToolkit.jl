# VisualGeometryToolkitCore changelog

## 0.10.3

A patch, additive: darks at several exposures in one characterisation entry, the defect mask and a linear view's
quantum public, and the log-polar map's differential and inverse (Float32 through `image_warp`).

### Added
- **`Darks`** (public), a black-slot component: the master darks of one body at one ISO, one `Dark` per exposure
  band, with ONE shared per-channel black as the fallback. The bands are disjoint: two bands sharing an exposure,
  an endpoint included, are an `ArgumentError`. The maps are of one size, and every Dark is of the component's
  source. `sensor_model` resolves a frame to the one Dark whose band serves its exposure (`Matched`; the model and its
  `content_digest` are the ones that Dark alone gives), else to the shared black with `DarkOutOfBand`. The selection
  is inside the one component: the slot's precedence of sources (`_pick`) is unchanged. A bare `Dark` stays a `Dark`;
  a `Darks` of one band is valid. `sensor_fixture` gains `darks` (a vector of the `dark` named tuples). Tested: each
  band's frame its own Dark, between bands / below the shortest / past the longest the shared black with
  `DarkOutOfBand`, overlapping, touching and repeated bands refused, any order the same `Darks`, Adapt and equality.
- **`defect_mask(model_or_defects, mosaic; backend)`** (public): the defect slot as a `Bool` mask over the visible
  raw, on the backend, the mask `LinearRaw` joins with its saturation mask; `nothing` for `NoDefects`. It was the
  internal `_defect_mask`; LinearRaw and the estimators are unchanged.
- **`linear_quantum(view, i, j)`** (public): one RAW code in a linear view's units at photosite `(i, j)`,
  `1 / (white − black[c])` by its colour (the span; a `Dark` changes the black, not the span), so a caller derives a
  resolution in linear units (an LM's `residual_resolution`) without reading the view's private fields.
- **`image_warp(Float32, …)` of a log-polar chain** keeps both log-polar links single-precision (their stored log
  radius converted), as every other link already was.
- **The log-polar map's differential and inverse.** `transform_deriv(::LogPolarToCartesian, p)` (analytic: the radial
  column `r·(−log r_in)/2·(cos θ, sin θ)`, the angular `π·r·(−sin θ, cos θ)`) and `inv(::LogPolarToCartesian)`, a new
  `CartesianToLogPolar` with its own `transform_deriv`. Through CoordinateTransformations' composition rules
  `transform_deriv(logpolar_map(g, rect, r_in), p)` and `inv(logpolar_map(…))` now work; both maps are isbits and run in
  a kernel. Tested against ForwardDiff, `inv ∘ map = id` through the composed map, and on the device. Additive (before,
  both raised CoordinateTransformations' "not defined" errors). VisualGeometryToolkitFeatures' `detect_image_circle`
  uses them for its radial derivatives and bin areas.

## 0.10.2

A patch: the success rule's exact-fit test and the data's resolution, a visible sensor-model fallback, and a choice
of LM damping.

### Fixed
- **The LM's exact-fit test has its own threshold.** `LmStops.ftol` is the RELATIVE cost-decrease stop; it was also read
  as the success rule's ABSOLUTE exact-fit threshold (`F < ftol`), so any stop at a cost below it was a `Success`
  however far from stationary (a downstream fixture ending at a cost of about 2e-8). The exact-fit test is now the run's
  own resolution of its starting cost, `F < m·ε·F₀` over `m` residuals (`_exact_fit_threshold`: a cost of `m` squares is
  evaluated to about `m·ε` relative), in the chart, robust and multi-block postconditions alike. A low-cost stop short
  of stationarity is now `Stalled` (tested); an exact fit is still a `Success`. Measured on RadiometricCamera 0.9's CPU
  suite (rec bb478c0eb, 68 LM runs): the fixtures that end at about 1e-8 to 2e-8 sit about six orders ABOVE `m·ε·F₀`
  (F = 9.8e-9 against 4.2e-14, m = 2970). They are still a `Success`, now by the column cosine; their 2e-8 runs stop
  at the iteration cap, which the rule never reads. One run of 56 changes: a restart from an exact fit (below).

### Added
- **`ModelReason` value `DarkOutOfBand`** (public): `sensor_model` matched the body and the ISO, but the frame's exposure
  is outside the entry's `Dark`'s exposure band, so the model's black is the Dark's `ChannelBlack` (the entry's other
  components kept). The fallback was silent (`Matched`); it is now in every model's `reason` and `component_sources`.
  Additive for a caller that does not match the values exhaustively; no stack package does (searched: the monorepo, PS,
  RC, BB, CB, LODR, RSDR).
- **The LM damping is a strategy type** (public `LmDamping`, `MarquardtDamping`, `LevenbergDamping`): `LmStops(; damping)`
  and `LmProblem(…; damping)` select it by dispatch, each NLLSsolver's own public iterator. `MarquardtDamping()`
  (λ·DᵀD, `:levenbergmarquardt_dtd`) is the default, and with it every fit is 0.10.1's bit for bit (tested against
  values measured on 0.10.1). `LevenbergDamping()` (λ·I, `:levenbergmarquardt`) is opt-in. Additive: `LmStops` gains a
  type parameter and a field, and a caller that builds it by keyword is unchanged.
- **`LmStops(; residual_resolution = 0)`**, the residual norm the data resolve, as in `StaticLmStops`: the exact-fit
  test is `F < max(m·ε·F₀, residual_resolution²)`. It is needed because the run's own resolution is relative to its
  start: a run that STARTS at an exact fit (RC's termination control re-solves from a solve's minimum, F₀ = 4.2e-29)
  has `m·ε·F₀` far below its rounding and the cosine says nothing there, so it is `Stalled`. With the data's
  resolution stated it is a `Success` (tested). Only the success rule reads it; the iterations are unchanged. The
  default `0` is the rule above. A caller states a DERIVED resolution (its data's noise in the residual's units),
  never a tuned literal.
- **Measured, so it is not retried blindly:** on RC's refinements, neither damping wins everywhere. λ·DᵀD reaches a
  lower cost on the fisheye case. λ·I converges on a non-central origin fit where DᵀD stalls at 1.2e-8 of its start
  cost, and it reaches a slightly lower cost 1.6–1.8× faster on the Tamron and grid cases. Hence a choice, not a new
  default. Moré's scaling (the DᵀD running max without NLLSsolver's 1e-8 floor) was also measured and REJECTED: a
  column whose diag(JᵀJ) is zero leaves `H + λDᵀD` singular (the grid and non-central factorizations fail), and it was
  no faster than the floored DᵀD. The measurements are in the `LmDamping` docstring.

## 0.10.1

The photon transfer of a dedicated capture (owner rulings of 2026-11-13/14, the FX3 characterisation: the single-frame
estimator's K drifted from 31 to 13 e⁻/code over the exposures of a flat-field series and refused 69 of its 111 frames).
Purely additive: the single-frame method keeps its signature and return type.

### New
- **`photon_transfer(groups, darks; sensor_model, block, normalize, significance)`** — the photon-transfer series:
  `groups` exposure => the frames at that exposure (at least 2; pairs are the case of 2), `darks` exposure => the
  `ReadNoise` of `master_dark` over the darks at that exposure. Per green photosite its TEMPORAL variance over a group's
  frames (which cancels the scene and every fixed pattern; each frame first scaled to its group's mean, `normalize`),
  against its mean. The intercept is FIXED at each group's own exposure, σ₀²(t), the greens' read noise² of its darks
  (read noise and dark-current noise); a group whose exposure has no darks is refused, never interpolated or
  substituted. The fit is the slope alone, `v − σ₀²(t) = μ/K`, over a range derived from the data: the upper end below
  the variance's rollover, the lower end the lowest from which the μ² test on the line succeeds, each search at
  `significance / m` (Bonferroni, `m` its candidate tests). K's standard error is the line's from the counts, times the
  Birge ratio R = √(χ²/dof) of the fitted points when R > 1 (never shrunk; R recorded). Returns the stack's `FitResult`: its estimate a measured
  `PoissonGaussian`, its metadata a `PhotonTransferMetadata` (public): the range in codes and as fractions of the white
  level, the intercept, the μ² statistic, each search's `m` and per-test level, the points offered. The single-frame and
  series inputs share ONE block kernel (a cell rule per input) and ONE line fit.
- **Measured limitation (the FX3, ISO 100, E4):** the temporal variance is not one line over the whole signal range: the
  slope `(v − σ₀²)/μ` rises from 0.065 at 41 codes to 0.088 above about 25 % of the white level, so `σ_r² + μ/K`
  overpredicts the variance by up to about 26 % below 5 % of white. The derived range keeps the fit where one line holds;
  no low-signal noise term is modelled (its cause is not established).

### Added
- **`LmProblem(variables, blocks; fixed)`**: `fixed` holds variables (their indices) as constants the blocks read —
  NLLSsolver's unfixed mask on the one construction path (`_lm_problem`); the stationarity postcondition reads the free
  variables' columns only. The default `nothing` holds none, the problem as before, bit for bit. An index outside the
  variables, a name, or every variable held is an `ArgumentError`.

### Fixed
- **The single-frame `photon_transfer(mosaic)` refuses a μ² term**: its points fitted again as `v = a + μ/K + cμ²` at the
  same weights, a `c` significant at `significance` (a new keyword, default 0.001: |t| > 3.29, two-sided, by its own
  standard error) is a `NoiseEstimateFailed` naming the term, never a biased K (a fixed pattern between the two greens,
  which the single-frame difference does not cancel, on a bright flat). Where it accepts, its estimate is 0.10.0's, bit
  for bit (tested).

### Deprecated
- **The single-frame `photon_transfer(mosaic)`**: unreliable on flats; scheduled for removal in the next breaking wave;
  use the multi-frame method. A single flat frame cannot certify its own K (its brightness range gives the μ² test
  little power); for flat fields use the multi-frame photon-transfer method.

## 0.10.0 (unreleased)

The breaking wave of 2026-11-13 (wave 10): the terminology taxonomy, one observation container, the FeatureFrame
interface with PointFrame first-class, sensor noise, and the conventions of the IO breaking wave.

### Breaking
- **One LM: one adapter, one owner of its stop settings, one success rule** (the owner's "NO DUPLICATIONS" ruling,
  2026-11-13; src/robust/lm_core.jl, configs/default.toml `[lm]`).
  - `LmStops(; max_iter, ftol, xtol, stationarity_tol, max_time)` is the ONE owner of the host LM's knobs; its defaults are
    Core's `[lm]` table (1000, 1e-10, 1e-10, 0.05: VisualGeometryToolkitEstimation's `[chart_lm]` values, moved here
    with their measurements). `nlls_options` is DELETED (retired name; `LmStops`, which the adapter maps itself).
  - `lmsolve_chart!` takes `stops::LmStops` (positional, default `LmStops()`) in place of the keywords `maxit`, `ftol`,
    `xtol`, `stationarity_tol`, in both chart forms; the robust form now keeps its problem and NLLSsolver cache in the
    workspace as the plain one does (one caching path).
  - NEW: the multi-block form, `lmsolve_chart!(p::LmProblem, stops; callback) -> (; code, rule, iterations, cost)`, on
    `LmProblem(variables, blocks)` (heterogeneous variables and residual blocks, each block's robustifier; kept and
    reused), for the bundles that built their own NLLSsolver problems. An indefinite or singular normal system in the
    step solve is `(Failure, NonFinite)`.
  - THE SUCCESS RULE: a stop is `Success` only where the stationarity postcondition holds at the returned point — now
    the cost-decrease stop too (it was converged unchecked), so a `CostDecrease` stop short of stationarity is
    `Stalled` (code `-2`). The multi-block and robust postcondition is `is_stationary(blocks, …)`; the robust chart
    LM's own copy (`_robust_stationary`) is deleted.
  - `nlls_termination` and `nlls_stop_code` are no longer public (the adapter's decode); `nlls_stop_reasons` stays.
  - `static_lm` moves to lm_core.jl (still `VisualGeometryToolkitCore.static_lm`; `Cameras` no longer exports it) and
    takes `StaticLmStops(scale; acceptance, residual_scale, cost_floor, max_iter, stationarity_tol)` in place of
    `s, maxiter, tol2, fs, …, cost_floor`: `static_lm(fj, x₀, lb, ub, stops[, bound])`. Its `converged` is the same
    success rule: stationary AND `F < acceptance`; acceptance alone no longer succeeds. Stationary on the device: its
    exact-fit or gradient floor, the exact-fit floor no finer than the data resolve (`residual_resolution`: a Float32
    ray or Float32 controls in a Float64 solve), the undamped Gauss–Newton step below its step tolerance (the device
    form of the host's exact-fit test: at the root of a square system the cosine says nothing), or the host's column
    cosine; on its box, a coordinate a bound holds drops out (the bounded problem's KKT stationarity). Its `cost_floor` defaults to 0 (resolved to rounding); the spline lenses
    give the reference's `eps(Float64)`. Its iteration cap defaults to `[lm] device_max_iter` (100), which replaces the
    spline lenses' `_PROJECT_MAXITER` and the mapped surfaces' `_BOARD_MAXITER`.
  - THE LM GATE: `check_duplication`'s retired pattern `:lm_primitive` fails any call of NLLSsolver's
    `optimize!`/`NLLSProblem`/`OptimizeCache`/`NLLSOptions`, of `nlls_options`/`_nlls_options`, `nlls_termination`,
    `nlls_stop_code`, `_lm_run!` or `_static_lm_increment` in any file of any stack package (src/, test/, benchmark/)
    outside lm_core.jl and its own tests (test/lm_driver_tests.jl, test/static_lm_tests.jl).
- **The terminology taxonomy** (the owner's ruling of 2026-11-13; conventions/naming.md): a detection, a correspondence,
  an observation. Renamed, no aliases; each old public name is in `check_naming`'s retired names with its replacement:
  `Cspond` → `Correspondence`, `Csponds` → `Correspondences`, `csponds` → `correspondences`, `ScoredCspond` →
  `ScoredCorrespondence`, `CspondNormalization` → `CorrespondenceNormalization`, `bearing_csponds`, `pixel_csponds`,
  `pair_csponds` → `bearing_correspondences`, `pixel_correspondences`, `pair_correspondences`; `PoolPair`'s field and
  keyword `csponds` → `correspondences`; `measurement_gradient_jacobian` → `observation_gradient_jacobian`; the
  "measurement noise" rule → "observation noise" (conventions/observation-noise.md, base/observation_noise.jl); every
  private `*cspond*`/`*measurement*` name likewise.
- **`TargetDetections` → `TargetObservations`**, and `TargetPoints2D`, `TargetPoints3D`, `TargetPoint2D`, `TargetPoint3D`
  are DELETED, folded into it (a 2D bundle weight is a `weights` column; the conversions from the old types go with them:
  VisualGeometryToolkitIO reads the old kinds through version steps).
- **`check_naming` R14**: no bound name or quoted symbol holds "measurement", the old pairing abbreviation, "sighting"
  or "keypoint"; "measured" is a provenance value only (a binding holding it is flagged). Retired names added for the
  renames above and for the three public aliases of the wave (`LiftableProblem`, `ScalarOrQuantity`, `POOL_RNG`).
  Published names that spell an old word (`csponds`, `eth3d_splg_csponds`) are `FOREIGN_FORMAT_TOKENS`, accepted in
  string literals, quoted TOML values, yaml/json fixtures and an `Artifacts.toml`.
- **The FeatureFrame interface, every verb once** (conventions/feature-frames.md): deleted per-type copies —
  `Ellipse(::Uncertain{<:PointFrame}; sigma)` (now one method for every frame, its centre's error ellipse) and the two
  detection forwarders `Ellipse(::Attributed{<:Uncertain{<:PointFrame}}; sigma)` (call `Ellipse(d.value; sigma)`);
  `AffineMap`, `scale`, `shape_matrix` of `Disc`/`OrientedDisc`; the per-type `ustrip`, `*`, `/`, similarity actions,
  `extrema(::Disc)`, `Rect2(::Disc)`, `Rect2{Int}(::Disc)`, `bbox(::Disc)`, `MvNormal(::Disc)`, the per-type `inv`
  and the two `Circle` arms — each now one method on `FeatureFrame` or its trait (`MvNormal(::OrientedDisc)` and
  `bbox(::PointFrame; scale)` now work; `Circle` of every scaled frame strips its units). The
  scaled frames' verbs are absent from a `PointFrame` by design (a `MethodError` with a hint).
- **`conventions/`**: `measurement-noise.md` is `observation-noise.md`; `feature-frames.md` joins the reading order
  (`convention_files()` lists 14). `persisted-geometry.md` states the IO wave's rules: one `units` and one `conventions`
  object per file, every `Size2` a `{"width", "height"}` object, every index 1-based (`"index_base": 1`), one
  `covariance` key with a producer-stated `covariance_status` when absent, matrices in rows, `v<major>.<minor>` schema
  tags, a stored quaternion with `w < 0` refused, the JCS identity in metres, the `observations`, `detection` and
  `sensor_characterisation` kinds.

- **The Student-t scorer's workspace records its codimension** (review C2): `ScoringWorkspace` holds `dg`;
  `alloc_scoring_workspace`/`scoring_workspace(::StudentTUniformScoring, n, d_g)` refuse a `d_g` other than the scorer's
  codimension, and `check_workspace` compares it. The bare `sweep!(sws, q, n; model_dof, log_two_outlier_halfwidth)` reads
  it; its `codimension` keyword is retired (refused). A scorer built from a bare model count (codimension 1) on a 2-D
  problem now throws at allocation instead of sweeping the wrong log Γ table.
- **The label prior is applied everywhere** (C3): the bare `sweep!` applies the workspace's prior, and a partial-buffer
  sweep scores with the prior of the swept n (results change for a non-flat prior only).
- **Batched `backproject`/`backproject_with_jacobian(::CameraModel, pixels)` run on the pixels' backend** by default
  (`get_backend(pixels)`, not `default_device()`) (C10).
- **`OnBackend` compiles every CUDA launch with `always_inline = true`**, whatever `CUDABackend` value it is given (C8).
- **check_duplication** exempts a transfer/allocation helper only by its whole name (C7); **check_naming** R12 also flags
  `thresholds`/`Threshold`, R13 a typed keyword with no default (C15).
- **The weak dependencies' floors are the pre-raise ones** (CUDA "5.11 - 5": CUDA ≥ 5.11.2 does not resolve beside cuDNN 1.x;
  Aqua "0.8.18", ChainRulesCore "1.26", Makie "0.24.16").

### Fixed
- Core's splats of a data-length collection (`check_splat`): the capture containers' element type
  (`mapreduce(typeof, typejoin, …)`), `PointTarget`, `Rig`'s form check and `ids`, `merge(::TagRegistry…)` and the LO and
  loss registries (the vector form), a frame or `Disc` from an `AbstractVector` centre (`SVector{2}(center)`: a centre
  of the wrong length is a `DimensionMismatch`), `to_affine(::AbstractVector)`, the induced plane homography,
  `check_naming`'s supertype table and the quality gates' module walk (Core's `_module_tree`, one copy).
- `axis_gauged` of a posed `BoundedCamera` turns its pose with its model (C1). `plane_map` and `transfer_ellipse` of an
  off-axis SplineLens camera (C4). `center_gradient` of a log-domain radial vignetting field, through the new
  `with_center(::LogVignetting{<:RadialVignetting}, c)` (C5). `has_spline_gauge` builds the constructor's own lattice
  (`unit_controls`) (C6). `radix_sortperm!` at n = 0 (C11); its docstrings claim bit-identity for an injective key only
  (C12). `bending_rows` of a mask under 4 × 4 has no rows (C13). `map_estimate` is inferred (C14). The shell lexer reads
  `$#` (C15). The `:pool` exact selection launches once over all pairs (5–12× faster on a GPU) (C9).

- **`linearview(m; sensor_model)` is the lazy uncertain image** (owner ruling D14): `LinearRawView{T, A, D, L} <:
  AbstractMatrix{T}` (its element type now first) holds the noise slot's variance law. Under a `PoissonGaussian` or
  `RelativeNoise` its elements are `Uncertain{Float64}`, each value with its scalar variance computed on the read by the
  one law `LinearRaw`'s kernel writes per plane (`_value_variance`); `copy` of it is the materialised `StructArray`
  (value and variance planes, on the view's device). Without a variance law, `Float64` as before, bit for bit.
- **`RelativeNoise` is `SensorSource.specified` only**: any other source is an `ArgumentError` naming it. `LinearRaw`'s
  cell image under a gain stays the materialised uncertain image (its binned variances computed once by its kernel).
- **`block_coordinate_result` takes the stop's rule** (a trailing `stop_rule::StopRule`): `block_coordinate_descent!` records
  WHICH member of the `StopWhenAny` fired (the first whose `stop_reason` is not `:continue`) and hands on its
  `stop_rule(criterion, method)`. Every method of `block_coordinate_result` adds the argument; `initialize_solver!` and
  `stop_solver!` return `(reason, rule)`.

### New
- **`stop_rule(criterion::StoppingCriterion, method) -> StopRule`**, the generic (a contract generic; `NoRule` by default,
  `StopAfterIteration` → `IterationCap`, `StopWhenSolveFailed` → `NonFinite`), extended by each package for its criteria.
- **`check_splat(M; extensions, unknown, root)` → `Vector{SplatViolation}`, `test_splat(M; mode)`, and the `splat` gate of
  `test_quality_gates` (default `:report`)** — no call splats a collection whose length is data (Features 0.8.3's
  `zip(map(_identity_fields, ids)...)`: one `zip` per capture size, 518 s at 171 frames). Typed: every method of the
  package (and its loaded extensions) is inferred at its own signature and each `Core._apply_iterate` whose splatted
  argument is not `<:Union{Tuple, NamedTuple, StaticArray, Pair}` is a finding (a bare name too; a call returning a
  tuple is not; a splat into a builtin or `Expr` is not; a value inferred `Any` only with `unknown = true`). Exceptions
  are `[[splat]] file, function, reason` entries of the repository's `DUPLICATION.toml`; a stale one is a finding.
  Core's gate is `:enforce`.
- `TagRegistry(family, suffix, pairs::AbstractVector{<:Pair}; retired, paths)`, a registry from a table.
- **`center(u::Uncertain{<:FeatureFrame})`, `center(d::Attributed{<:Uncertain{<:FeatureFrame}})`** — THE detection → point
  observation conversion, every frame type: the centre with the first 2×2 block of its covariance (storage kept; a
  precision's marginal). `frame_dof` (public) — a frame's degrees of freedom, its covariance's size, the centre first.
- **`transfer(m, f::FeatureFrame)`** — a frame through any point map (a PointFrame exactly; a scaled frame to first
  order, an `AffineFrame`; exact through a similarity; a `BoundedCamera` refuses it, naming the plane maps: a camera images 3-D points). `propagate` of an `Uncertain{PointFrame}` through
  `HomographyMap` and `FramedAffine`; the similarity actions, `extrema`, `Rect2`, `bbox`, `ustrip`, `*`, `/` for every
  frame, PointFrame included; `shape_factor` of a `Disc` (`σI`) and an `OrientedDisc` (`σR(θ)`); the DOF trait
  `PointShape`.
- **The sensor model** (spec 2026-11-13-sensor-noise, Phase 1, owner rulings A and D1–D13), in three levels: `Sensor`
  (geometry, unchanged); `SensorCharacterisation(body; entries)`, ONE body's per-ISO components (host-side, keyed by
  its `DeviceIdentity`, held beside the camera); `SensorModel{B,N,D}`, ONE frame's black, noise and defect components,
  isbits when they are, `Adapt` for the maps. Each slot's component is a type, absence included: black `FileBlack` (the
  RAW's own) | `ChannelBlack(values, std_error; source)` | `Dark(black, offset, variance, exposure_band, frames;
  source)` (all dark information in ONE black-slot component, D2); noise `NoNoise` | `ReadNoise(σ, std_error;
  source)` | `PoissonGaussian(gain, σ, std_error; source)` | `RelativeNoise(σ_rel; source)`; defects `NoDefects` |
  `DefectList(x, y, onset; size, source)`. Every present component states its `source`, a `SensorSource.T`
  (`measured`, `estimated`, `specified`, `file`). THE resolver `sensor_model(characterisation, capture, t)` is a
  lookup (D1): the body (serial, else make and model), the ISO exactly, per slot specified over measured over
  estimated (`overrides_measured` notes it, D5), a `Dark` only inside its exposure band, a `DefectList` cut to the
  sites whose onset is at most the exposure; with no characterisation, another body or another ISO, the absent model
  `SensorModel()` and its `ModelReason` (`Matched`, `NoCharacterisation`, `UnmatchedISO`, `OtherBody`, D4).
  `content_digest(model)`, the model's identity: `sha256_hex(canonical_json(tree))`, the stack's one content-identity
  scheme (D9), with `canonical_json` now Core's (hoisted from VisualGeometryToolkitIO).
  `LinearRaw(mosaic; mode, gain, sensor_model)` and `linearview(mosaic; sensor_model)` (the keyword is `sensor_model`,
  D6) subtract the black slot's black in their one kernel (a `Dark`'s offset per photosite), mask a `DefectList`'s
  photosites, and under a `PoissonGaussian` or `RelativeNoise` return an UNCERTAIN image: a `StructArray` of
  `Uncertain{Float64}` values, each value's variance from the same kernel (D11; no separate variance function);
  without a model, bit for bit the RAW's linearisation. The estimators build a characterisation's components offline:
  `photon_transfer(mosaic; sensor_model)` an estimated `PoissonGaussian`, `master_dark(frames; exposure_band)` a `Dark`
  (with its `ChannelBlack`), a `ReadNoise` and the band's defect mask, `defect_onsets(exposure => mask, …)` one
  `DefectList` (D3); `NoiseEstimateFailed`. Tests build components only through `sensor_fixture` (D13), and
  `check_duplication` refuses a component built elsewhere (`:sensor_component`), a hand-written value variance
  (`:value_variance_law`) and a hand-indexed channel black (`:black_subtraction`). None of wave 10's earlier sensor-noise
  names (never released) remains; `check_naming` retires them with their replacements.
- **`check_region_support(support, region; who)`** and its typed **`RegionBeyondSupport`** (public): THE refusal of a
  fit over a region beyond the source's calibrated support. It reads the ONE region containment, `a ⊆ b`
  (`Base.issubset`), over every pair of the regions a fit and a support take: a `Rect`, an `Ellipse` (an ellipse in an
  ellipse by the 1-D grid–Brent minimizer), their `RegionIntersection`, `WholePlane`.
- **`TargetPlaneFrame{id}`** (public): a target's plane named by its id, as `TargetFrame{id}` names its 3-D frame;
  `TargetHomographies` mints `TargetPlaneFrame{id} → F`. The bare `TargetPlaneFrame` stays a plane the value does not name.
- **`green_plane(L::LinearRaw)`** and **`poisson_slope(L, model::SensorModel)`** (public): the green plane of a `:bin2`
  or `:green_cells` image, and each plane's Poisson slope α, the one expression the variance kernel writes with.
  `LinearRaw`'s geometry holds `plane_sites` (each photosite's colour and span, every plane); the wave's interim
  `green_dn`/`green_colors` fields are retired names.
- **`iou_below(e1::Ellipse, e2::Ellipse, t)`** (public; hoisted from BlobBoards' `_iou_below`), and the README-test
  helpers **`page_examples(examples, page; lang)`** and **`install_lines(root, page)`** (BlobBoards' and CaliBlobs'
  restated copies). Retired names: `BoardDetection`, `load_cspond_sets`, `BlobBoards.write_tiff`, PanoramaStitching's
  `MEASUREMENT_NOISES` and `measurement_noise`.
- `configs/default.toml` `[lm] device_acceptance = inf`: `StaticLmStops`' default acceptance (success by the
  stationarity rule alone; a root-finding caller states its bound).
- **`component_sources(m::SensorModel)`** (public): the one provenance view of a frame's model — each slot's `SensorSource.T`
  (`nothing` when absent), its `ModelReason` and `specified_override` — for a run record (ruling A).
- **`Statistics.var(u::Uncertain{<:Real})`**: THE scalar variance of an uncertain value (a pixel of an uncertain image),
  `cov(u)[1, 1]`; consumers read `value_of(x)` and `var(x)`, never `cov(x)[1]`.
- **`poisson_slope(L, model, plane)`**: one plane's Poisson slope, by index or colour (`:R`, `:G`, `:B`; `:G` the green
  plane), an `ArgumentError` for a colour the image does not hold.
- **`refine_width(lo, scale, width, d)`** and **`assert_unit_bearings(fs, which, who; remedy)`**, public: the refine
  width rule and the unit-bearing check PoseLib and VisualGeometryToolkitEstimation each kept a copy of (review E11, E12).
- **`BinnedRadialProfile`** (public) — a measured radial profile, binned in a chart's coordinate, and the `:equisolid` chart
  (`θ = 2 asin(d / 2f)`) in the one radial-chart registry `RADIAL_CHARTS` (a fitted `RadialVignetting` refuses it: not a
  normalized radius); `radial_coordinate(p, u)`. VisualGeometryToolkitIO's `radial_profile` kind serializes it.
- From the 0.9.7 patch: `reduce_normal_equations(row!, Val(N), args...; partials)` and `reduce_cost(row!, args...; partials)`,
  a device LM state's row launch and reduction in one call (VisualGeometryToolkitEstimation's lens regression and
  RadiometricCamera's radiometry LM each kept the pair).
- From the 0.9.6 patch: `reduce_normal_equations(partials, Val(N))`, `reduce_cost(partials)` (public), and the weak CUDA
  compat `"5.11 - 5"` with the cuDNN-1 resolve test.
- **The GPU test tier** (hoisted from RadiometricCamera 0.8.0's test/gpu_decision.jl): `gpu_tier(prefix)`,
  `gpu_problem(prefix)`, `require_gpu(prefix)` (public): one decision for every stack suite, never a silent skip.

## 0.9.9 (unreleased)

A patch. Additive: no result changes.

- **New: `pointwise_kernel(f, out, xs...; exec) -> (kernel, args, ndrange)`, public** — the one launch `pointwise!`
  makes, unlaunched; `pointwise!` is it followed by the launch, so its results are the same bits. A caller sizes a
  `pointwise!` launch with `k, a, n = pointwise_kernel(f, out, xs...); device_resident_threads(k, a...; ndrange = n)`:
  the count of the very kernel `pointwise!` runs, without restating how it is built.

## 0.9.8 (unreleased)

A patch. Additive: no result changes.

- **New: `device_resident_threads(kernel, args...; ndrange) -> Int`, public** (base/device.jl) — how many threads of a
  KernelAbstractions kernel instance (`OnBackend(dev)(f!)`, launched as `kernel(args...; ndrange)`) its device holds
  at once, register-aware. On a CUDA device it is the compiled kernel's occupancy at the launch's block size, times the
  block size, times the multiprocessors; never the raw maximum of threads per multiprocessor, which overstates a
  register-heavy kernel (255 registers a thread: about 8×). On the host it is `Threads.nthreads()`; another backend is
  refused. The accelerator extension answers it beside `device_free_bytes`.

## 0.9.7 (unreleased)

A patch. Additive: no result changes.

- **New methods: `reduce_normal_equations(row!, Val(N), args...; partials)` and `reduce_cost(row!, args...; partials)`**
  — a device LM state's launch and reduction in one call: `row!(partials, w, args...)` for every work item as one
  kernel on `partials`' backend (the stack's one row launch, KernelAbstractions on every backend), then the existing
  reduction of the partial sums. The one copy of the launch-then-reduce pair VisualGeometryToolkitEstimation's lens
  regression and RadiometricCamera's radiometry LM each wrote; `partials` is the caller's scratch (a keyword).

## 0.9.6 (unreleased)

A patch: two bug fixes, and two new public names (additive).

- **Fixed: Core resolves beside cuDNN 1.x again.** 0.9.5's weak `[compat] CUDA = "5.11.3"` (the wave-113 floor raise)
  needs CUDA_Runtime_Discovery 2, which no cuDNN 1.x accepts, so no environment holding cuDNN 1 (BlobBoards,
  CaliBlobs) could resolve Core 0.9.5, nor IO 0.9.3, Features 0.8.3, Estimation 0.10.1 or Eval 0.7.2 through it. The
  floor is 5.11 again (`"5.11 - 5"`: CUDA 5.11.0–5.11.1 use CUDA_Runtime_Discovery 1), and a test resolves Core with
  cuDNN 1 from the registries. No other weak floor of the six packages blocks (checked: BlobBoards' and CaliBlobs'
  environments resolve with the patched packages).

- **New: `reduce_normal_equations(partials, Val(N)) -> (½‖r‖², Jᵀr::SVector{N}, JᵀJ::SMatrix{N, N})` and
  `reduce_cost(partials) -> ½‖r‖²`, public** (robust/lm_core.jl) — a device LM state's `residual_normal_equations!`
  and `residual_cost!` from its per-work-item partial sums (column `w`: `Σ‖rᵢ‖²`, the `N` of `ΣJᵢᵀrᵢ`, the `N × N` of
  `ΣJᵢᵀJᵢ` column-major), summed on the array's backend. The one copy of the unpack VisualGeometryToolkitEstimation's
  lens regression wrote by hand (Estimation 0.10.2 calls it, the same bits); RadiometricCamera's radiometry LM is next.

- **Fixed: `axis_gauged` of a POSED `BoundedCamera` now turns its pose with its rays.** It turned the model only (through
  `with_model`, which keeps the extrinsics), so every world point projected elsewhere. It now gauges the camera the
  bound wraps, as the `PosedCamera` method and the docstring say, so every projection is kept. An unposed bounded camera
  (the only kind VisualGeometryToolkitEstimation gauges) comes back identical to 0.9.5's: no Estimation result changes.

## 0.9.5 (unreleased)

A patch. Additive: the pieces VisualGeometryToolkitFeatures 0.8.3's opt-in continuous DoG refinement reads, each the one
implementation of its concept (the owner's rulings of 2026-11-13); no existing result changes.

- **New: `static_lm`, public** — the stack's one device-side Levenberg–Marquardt, until now the private `_spline_lm`
  (renamed, no alias), at any static N: `static_lm(fj, x₀::SVector{N}, lb, ub, s, maxiter, tol2[, fs, bound, cost_floor])
  -> (x, converged, J, e, rule)`. The increment is an unrolled N × N solve (`_static_lm_increment`, formerly
  `_spline_lm_increment`). It also returns the stop that fired, a `StopRule`. At N = 2 every operation is the old one's:
  the spline lenses' inverse, the MappedSurface root and the pixel-LM reference give the same bits (tested `===` against
  0.9.3's `_spline_lm` on the pixel_lm fixture: 0 of 12 000 runs and 0 of 100 000 increments differ).
- **New: `static_lm_termination(converged, rule) -> (ReturnCode.T, StopRule)`, public** — a `static_lm` solve's §14
  record: `Success` when converged, `MaxIters` at the iteration cap, `Stalled` at any other stop.
- **New: `cubic_bspline_weight_derivatives(u) -> (d₁, d₂)`, public** — the first and second derivatives of the four
  `cubic_bspline_weights`. It is the former private `_cubic_bspline_weights_deriv` (its first derivative, the same
  expressions) made public with the second added; its callers (the lattice window weights, `spline_weights`) read its
  first part.
- **New: `spline_window_sums(control, zero, wx, wy, dwx, dwy[, ddwx, ddwy])`, public** — the one 4 × 4 cubic B-spline
  window sweep: the value, its two partials and, with the second-derivative weights, its three second partials. The
  spline lattices' value-and-partials sweep is this function, its operations in the same order (the same bits); the
  private `_window_sums` that called it is deleted, its one remaining caller (`_projection_jacobians`, the SplineLens
  bundle's differential) reading `_spline_value_and_partials` like every other.
- **New: `cubic_bspline_prefilter_taps(K)`, public** — the cubic B-spline interpolation prefilter on an unbounded
  grid, `√3·z₁^|k|` truncated at K and renormalised: `_prefilter_matrix`'s concept as a local filter a kernel applies.
- **Changed (additive): `is_positive_definite` at static N.** A static `N × N` matrix tests all N leading minors
  (Sylvester); at N = 2 the expression is the old one. Any other matrix still tests its leading 2×2 block.
- **Docs:** a generic docstring for `score_sq_residuals!` (its methods keep theirs), so a cross-reference to the
  function names one docstring.
- **New: `farthest_radius(region, center[, direction])`, public, not exported** — how far a region reaches from a
  centre: the largest distance to a point of it (a `Rect`'s farthest corner exactly; any other region's exit radius on
  720 azimuths, the largest refined by `grid_brent_minimum`), or the exit radius along one direction (bisected, in
  units of the direction's length). Beside `imaged_region`, whose sensor `Rect` clips it: an image circle cut by the
  sensor reaches only as far as the sensor. The one implementation of the concept: VisualGeometryToolkitEstimation's
  `fit_camera` exit radii (its private `_exit_radii`, deleted) read it, and RadiometricCamera's `vignetting_radius` and
  its rings' disc will. Core's own farthest-corner sums in both `division_bounded` methods read it too (the
  coefficient form's `r_max` is the same bits; the lens form's normalized radius may move by an ulp).
- **New: `bending_rows(mask; spacing)`, a method of the public `bending_rows`** — the bending energy's rows over the
  cells of a uniform cubic lattice whose 16 nodes `mask` holds, as one sparse operator: `bending_stencils` assembled
  over a mask, beside the masked `difference_rows(mask, order)`. RadiometricCamera's grid prior's `_bending_operator`
  moves here, the same rows bit for bit.
- **Changed (additive): `backproject(::NoncentralSplineModel, u)` at any number type.** It returns a `Ray3{T}`, `T`
  the promoted type of the pixel and the model (`Float32` for a `Float32` model, a ForwardDiff `Dual` through the
  pixel), and runs in a KernelAbstractions kernel on the model's device copy. `Ray3{T}` takes any `T <: Real` (it took
  `T <: AbstractFloat`). A `Float64` model at a `Float64` pixel gives the same `Ray3{Float64}` bits as before.
- **Changed: the CUDA extension's default device is `CUDABackend(; always_inline = true)`**, so every kernel launched on
  `default_device()` inlines its calls. `on_device` compares an array's backend with the target by backend type (the
  memory it lives in), not by value, so an array already on the GPU is not copied because its `get_backend` reports
  the backend without the option. A device the caller names is used as given.
- **Julia 1.13** (the owner's ruling of 2026-11-13: the whole stack requires 1.13): `[compat] julia = "1.13"` in the
  package and in every environment of its workspace; CI runs 1.13 only.
- **Every `[compat]` floor raised to the gated resolution** (the owner's ruling of 2026-11-13): a stack dependency at
  its registry latest or its in-wave version, a third-party one at the version the release gate resolved (a floor
  only, the semver upper bound kept); the monorepo's Downgrade workflow is deleted.
- **Internal: private forwards and aliases deleted** (the owner's no-shim order), every caller calling the public
  function directly, the same bits: `_strip_length` (→ `strip_or_bare`), SpatialQueries' `_Leaf`, `_Branch`, `_Rect`
  and `_mbr` (→ SpatialIndexing's, qualified), check_naming's `_line_of` (→ `searchsortedlast`), the SIFT landing's
  `_window_sums` (above), and `_homography_point_jacobian` (→ `transform_deriv` of the 2-D homography map, below).
- **New: `transform_deriv(PerspectiveMap() ∘ h::HomographyMap{3}, x::StaticVector{2})`** — the 2 × 2 Jacobian of the
  dehomogenised 2-D homography map, `transform_deriv(PerspectiveMap(), H·x̃) · H[:, 1:2]` in the promoted number type:
  the one point Jacobian of a homography. The private `_homography_point_jacobian` is folded into it; `propagate` of an
  uncertain point through a `HomographyMap{3}` reads it (the same bits).
- **New: `plane_map(camera::PosedCamera)`, public, not exported** — the board-plane → pixel map of a posed camera of
  ANY model: the plane lift `q ↦ (q, 0)`, the rigid extrinsics, then the model's projection (`project` as a map, the
  forward twin of `Backprojection`); `plane_map(camera)(q) == project(camera, [q; 0])` and
  `transform_deriv(plane_map(camera), q)` is the chain rule. For a central `CameraModel` it is, by dispatch, the same
  map on the plane points in front of the camera in its homography form, `distortion_map(m) ∘ (PerspectiveMap() ∘
  HomographyMap(pinhole twin))` (a pinhole: the homography map alone): the composition `transfer_ellipse(::PosedCamera,
  _)` built inline, which now reads it (the same bits; the cheirality test kept).
- **New: `project(::NoncentralSplineModel, X)` through ForwardDiff Duals** — the plain solve on the point's values, its
  partials by the implicit function theorem at the solved point (∂p/∂X = (∂r/∂p)⁻¹ [t₁ t₂]ᵀ, the LM's own analytic
  Jacobian), as a SplineLens's `project` already did; so a non-central camera's projection differentiates like every
  other model's.
- **`check_duplication`: the idiom cap counts this repository's definitions only.** A shingle in more than 40 of the
  checked repository's definitions is an idiom; the stack repositories of `VGTK_STACK_PATHS` no longer count towards
  it, so a pair inside the repository is found alike with and without them (CI reads no stack, a release gate does).
  The monorepo's two test-fixture `[[similar]]` entries are restored in DUPLICATION.toml.

## 0.9.4 (unreleased)

A patch. Additive: `is_inside_texel_centres` and the mode methods of `raw_frame_map` and `raw_provenance`; otherwise no
API change.

- **New: `is_inside_texel_centres(uv, (W, H))`** (public): whether `uv` lies in the closed rectangle of the texel
  centres of `bilinear_sample`'s half-texel chart, exactly the positions where `bilinear_weights` interpolates
  without extending the edge. It reads the same private chart (size check, centre coordinates, clamp) as
  `bilinear_weights`, whose arithmetic is unchanged; RadiometricCamera's `texel_weights` restated the chart beside
  its `bilinear_weights` call.
- **New: `raw_frame_map(::Type{LinearRaw{mode}})`**, a method of the exported `raw_frame_map`: a cell mode's frame map
  without an image (`x ↦ 2x`, `BayerCellFrame{:G}` → `PixelFrame{:colmap}`, for `:green_cells` and `:bin2`; `:demosaic`
  refused, as its images are). It is the one source of the cell images' map: every `LinearRaw` (and every channel map of
  a mosaic) is built as `raw_frame_map(mosaic) ∘ raw_frame_map(LinearRaw{mode})`, so a whole mosaic's image has the mode's
  map bit for bit (tested over every Bayer phase, visible-origin parity and window parity). RadiometricCamera read the
  map off a hand-made one-cell mosaic for frames with no RAW behind them.
- **New: `raw_provenance(::Type{LinearRaw{mode}})`**, a method of the public `raw_provenance`: the part of an image's
  provenance the mode alone determines, `(; mode, frame_map)`, without an image (`frame_map` the record of
  `raw_frame_map(LinearRaw{mode})`; `nothing` for `:demosaic`). An image's `raw_provenance` takes its `mode` from it and
  writes its `frame_map` by the same private rule (`_frame_map_record`), so every whole-mosaic image records the mode's
  `mode` and `frame_map` field for field (tested over every Bayer phase and visible-origin parity); a window's record differs
  only by its offset in `translation`. RadiometricCamera built its one-cell probe mosaic for this record too.

## 0.9.3 (unreleased)

A patch. Additive: `require_extension`; otherwise no API change.

- **New: `require_extension`, public** (the owner's ruling, 2026-11-12): the stack's one check for a package extension a
  call needs, hoisted from VisualGeometryToolkitFeatures' and BlobBoards' private copies.
  `require_extension(missing_error, parent, extension)` returns `nothing` when `parent`'s extension is loaded and
  otherwise throws `missing_error()`, the caller's own exception naming what to install and load.
  `require_extension(missing_error, f, args, parent, extension)` is the fallback method of a function whose methods live
  in the extension. Without the extension it throws `missing_error()`; with it loaded, the arguments fit no method of
  the extension, so it throws their `MethodError(f, args)` at the caller's world age, which keeps Julia's "method too
  new" hint. - **Fixed: one surface and one ray have one hit, whatever code calls it.** The `MappedSurface` root
  (`intersection`, `chart`, `normal`, `surface_point`) is inlined into its callers, and its arithmetic was built on
  operations LLVM may contract into FMAs as each compiled copy decides (StaticArrays' `muladd` products, its `@simd`
  `dot`, the LM increment's `muladd`s): two hits of the same surface and ray in one function came out 2–3 ulp apart (a
  height field alone or composed, Julia 1.12 and 1.13, with or without `--check-bounds=yes`). The root now reads the
  chart map through its own traversal with explicit fused products (`fma` chains in a fixed order; everything else plain
  arithmetic, which Julia never contracts), as does the 2-D LM it shares with the spline lenses' projection, so it stays
  inlined — a device kernel's evaluation keeps its speed — and gives the same bits in every copy. Hits, and the spline
  lenses' projections (the same 2-D LM), move by rounding only against 0.9.2; their bits no longer depend on the call
  site. Tested bit for bit over 10⁴ calls from five call sites per surface (`mapped_surface_determinism_tests.jl`), a
  Dual call (the heights differentiated) included. - **One copy of each piece of that fix.** The fused products
  (`_fma_dot`, `_fma_mul`, `_fma_tmul`) live in `base/fused_products.jl`, and a `Quad`'s plane coordinates have one body
  (`_plane_coordinates`, now with the fused dot; the root's copy of it is gone), so `chart(::Quad)` moves by rounding
  only. The root's traversal of the chart map is tested against the map's own call and `transform_deriv`, member by
  member (`mapped_surface_traversal_tests.jl`). The private `_chart_affine`, unused since the height path reads the
  chart map directly, is deleted.

## 0.9.2 (2026-11-12)

Additive. The sensor image moves here from VisualGeometryToolkitFeatures (Features 0.8.0 keeps the LibRaw decoder and
the display demosaic; it no longer exports these names, Core alone does), so every package below Features reads a
RAW image without the decoder, and it gains a channel-first photosite view with an exact map per channel (the Bayer
photosite radiometry spec, 2026-11-12, §1 and §4.1).

- **Moved from Features: `RawMosaic`, `LinearRaw`** (the type, the linearisation and its KernelAbstractions cell kernel),
  **`BayerCellFrame`, `raw_frame_map`, `bayer_plane_map`, `green_cell_sites`, `raw_coordinates`, `photosite_colors`**,
  and with them `cfa_string` (the mosaic's `show` reads it) and `raw_provenance` (an accessor of `LinearRaw` alone)
  (`src/raw/`). No new dependency: Adapt, KernelAbstractions and ImageCore were Core's already. Every assertion of
  Features' former raw tests holds unchanged here (`test/raw_mosaic_tests.jl`, only the import moved), CPU and GPU.
  `LinearRaw`'s inner constructor now checks the gain (positive and finite), the check `LinearRaw(mosaic; gain)` made,
  so the display image Features builds takes the same rule.
- **New: `channelview(m::RawMosaic)`, a method of ImageCore's `channelview`** (exported, the same binding as
  ImageCore's): the photosites channel-first, `4 × H/2 × W/2`, each 2×2 cell a pixel and its four photosites its
  channels (spatially interleaved), in the cell's column-major phase order — a reshape of the photosite matrix to
  `2 × H/2 × 2 × W/2`, a `PermutedDimsArray` to `2 × 2 × H/2 × W/2` and a reshape, nothing copied; a device mosaic's
  view is a view of its device array (`get_backend` the device's, `adapt` moves it whole). An odd last row or column
  is left out, as by `LinearRaw`. A plain matrix keeps ImageCore's method. **A GPU kernel reads the photosites by the
  view's index rule, not through the view** (`benchmark/raw_channelview.jl`, B200, 2848 × 4256, the cell kernel): Base's
  `ReshapedArray` indexing compiles in a CUDA kernel over a whole mosaic but at 509 GB/s against 1391 GB/s for the held
  matrix (1426 for a packed `4 × h × w` copy, 1371 for the view's own `PermutedDimsArray`), and over a window or an
  odd-sized mosaic (a `SubArray` inside) not at all (`InvalidIRError`, and a broadcast neither). `LinearRaw`'s cell
  kernel therefore reads `parent(m)[2i − 1 + di, 2j − 1 + dj]`, the view's `[k, i, j]` by the one rule, as before.
- **New: `photositeview(v)`** (exported): the inverse view, `4 × h × w` cells → the `2h × 2w` photosite matrix (of a
  mosaic's `channelview`, its held array again).
- **New: `bayer_channels(m)`, `bayer_plane_map(m::RawMosaic, k)`** (public): each channel's colour (`:R`, `:G1`, `:G2`,
  `:B`; the greens in the cell's order) and its isbits `SimilarityMap` from `BayerCellFrame{C}` onto the sensor's
  `PixelFrame{:colmap}` (exact for every Bayer phase and either parity of the visible or the window origin; the R and B
  channels' maps ARE the `:bin2` R and B planes'). `BayerCellFrame` gains the tags `:G1` and `:G2` (the two green
  photosites' lattices).
- **New: `linearview(m)`** (exported) **and its type `LinearRawView`** (public): the mosaic's linear values on the raw
  grid, lazily — each photosite's `(raw − black[c]) / (white − black[c])` (what Features' private `_linear` computed;
  that function is gone, its expression moved here once, as Core's private `_linear_value`).
  `normedview`-style (a lazy number view, no copy) and composing with ImageCore's `channelview`:
  `channelview(linearview(m))` is the linear values channel-first, and `LinearRaw{:green_cells}` is the mean of its two
  G channels, bit for bit (a test, CPU = GPU). A broadcast and `copy` run on the mosaic's device, and a KernelAbstractions
  kernel reads the view as it is (two parities and one linearisation per element). Not a `MappedArrays.mappedarray`:
  the black level depends on the photosite's position (its colour in the phase), which a value-only mapped function cannot
  see, and Core takes no new dependency.
- **`photosite_colors(m)` is lazy and isbits** (public; was a `Matrix{UInt8}` built per call in Features): a
  `PhotositeColors <: AbstractMatrix{UInt8}` holding the size, the pattern (an `NTuple`) and the phase, indexed by the
  one phase rule — the same values (every former test unchanged), and on a device mosaic it fuses into a broadcast with
  the device arrays and runs there: a photosite mask (RC's validity mask: the greens, less `raw == 0` and `raw ≥ white`)
  is ONE broadcast, no host round-trip and no kernel of its own; CPU = GPU, bit for bit (a test, 4 phases × visible and
  window parities). `collect` gives the matrix. `RawMosaic.pattern` stays the host `Matrix{UInt8}` the decoder fills;
  the device-facing form is `photosite_colors(m)`, and the black levels are an `NTuple` already.
- **New: `saturationview(m)`** (exported) **and its type `SaturationView`** (public): the saturation flags on the raw
  grid, lazily, as a `Bool` matrix matching `linearview` (same grid, same device behaviour, `channelview` of it) — THE
  saturation rule `raw ≥ white`, now one private predicate that `LinearRaw`'s cell kernel reads too. `photosite_colors`
  is public, and a colour mask is a plain broadcast over it (`((col .== 0x01) .| (col .== 0x03)) .& .!saturationview(m)`):
  no mask helper exists, by the owner's ruling.
- **One public path per concept** (the duplication check before the release):
  - the green-cell image: `LinearRaw(m; mode = :green_cells)` (its `image[:, :, 1]`) — the constructor the docs and
    RC use; it carries the saturation flags, the frame map, the green footprint and the gain, and runs on the mosaic's
    device. The plane accessor `green_cells(L)` written for this release is not kept (a second public spelling of the
    same image); Features' `variance_stabilize` finds `:bin2`'s green plane by its frame (`bayer_plane_map(L, c) ===
    raw_frame_map(L)`), the public frame maps;
  - the linear value of a photosite: `linearview(m)` (`linearview(m)[i, j]`, or `channelview(linearview(m))[k, i, j]`
    channel-first). The scalar expression (`_linear_value`) and the per-channel levels (`_bayer_levels`) are private:
    the levels are the mosaic's public fields `m.black` (per colour index) and `m.white`, and a second public spelling
    of either would be the duplicate.
- **One phase rule.** The colour lookup, the cell's four sites in column-major order and each site's offset from the
  cell centre are stated once (`src/raw/raw_mosaic.jl`); `channelview`'s channels, `bayer_plane_map` of a mosaic and of
  a `LinearRaw`, `green_cell_sites` and the cell kernel read them. `bayer_plane_map` and `green_cell_sites` are
  rewritten on the one sample-lattice map, bit-identical to 0.9.1's (a test writes the former expressions out and
  compares with `===`), and `LinearRaw{:green_cells}` is the mean of the two G channels of the linearised view, bit for
  bit (a test, CPU and GPU).
- **New: a `SimilarityMap` acts on a `Rect` and on a `RegionIntersection`** (the spec's §4.1; Core had the `Ellipse`,
  `Circle` and `WholePlane` actions, none for these): a `Rect` maps to the `Rect` of its mapped corners when the
  rotation is exactly a multiple of 90° (any other turn is an `ArgumentError`: a turned rectangle is no `Rect`); an
  intersection maps part by part.
- **New: `pixel_mask(region, frame_map, width, height)`**: the mask of an image on another pixel lattice than its
  region's (a Bayer plane or channel against a region on the sensor), each pixel's mapped centre tested in the region's
  own frame.
- **`check_raw_decoding` allows Core's sensor image** (`src/raw/`, its test) beside Features' raw module: the one
  place, now per owning package.

ImageCore's conventions for the moved types (owner ruling 2026-11-12; ImageCore is a dependency of Core, so its own
functions are used where the concept is the same):

| ImageCore | adopted? | how, or why not |
|---|---|---|
| `channelview`, channel-first `C × H × W` | adopted, by a method | `channelview(::RawMosaic)` is ImageCore's function: the channels of each pixel (a 2×2 cell), first, as a view |
| views, not copies | adopted | `channelview(m)`, `photositeview`, mosaic windows by `view` copy no photosite, on the host or a device |
| `colorview`, the inverse view of `channelview` | not extended: `photositeview`, a name of its own | `colorview(C, A)` makes colorants of channels; the inverse of a mosaic's view makes the photosite matrix of cells, with no colorant type |
| `rawview` | not extended | a mosaic IS its raw `UInt16` codes: `rawview` exposes the integer storage of a `Normed` array, and there is nothing to unwrap |
| `normedview(N0f16, A)` | not adopted for the linear value (usable as ImageCore defines it, on `parent(m)`) | the raw codes are NOT `[0, 1]`-normalised by the black and white levels: `normedview(N0f16, A)` reads a code as `code / 65535`, so the black level is not 0 and the white level is not 1, and each channel has its own black level; the linear value is `(code − black[c]) / (white − black[c])`, `linearview` (a lazy view in `normedview`'s style) |
| channel-first multi-channel arrays | adopted for the new view; **not** for `LinearRaw{:bin2}`'s stored image | `L.image` stays `H × W × 3` (planes last) in 0.9.2: moving it to `3 × H × W` changes a stored layout every reader indexes (`image_warp!(dst, warps, L.image)`, `L.image[:, :, c]`, the decode cache) — a Breaking item for the next minor release, not an additive one |
| ImageCore-compatible names only where the concept matches | adopted | no `channelview`-like local name exists; `photositeview`, `linearview` and `saturationview` are named apart because their concepts differ (above) |

The channel order is the cell's own column-major phase order (`R, G1, G2, B` for RGGB), not a fixed `R, G1, G2, B`
for every phase: reordering channels per phase is a gather, which no reshape or permutation of dimensions expresses,
so it could not be a view. `bayer_channels(m)` names each channel from the same rule.

Also additive: public names for private functions a downstream package (RadiometricCamera) copied or reached by their
private names. Each is the function Core's own callers use, unchanged; the private name is gone (no alias).

- **New: `with_parameters(s::MappedSurface, θ)`**: the surface with its calibration member rebuilt by the
  calibration's `with_parameters` (`θ` in its `parameter_names` order), the pose, placement, lift and frame kept; on
  the device and in any element type (Duals). RadiometricCamera rebuilt its warped board this way through the private
  `_map_members`.
- **`normalized_point(intrinsics, 𝐩)`, public** (was `_normalized_point`, Cameras): `K⁻¹𝐩`, the pixel-frame half of
  `backproject` and the point a spline lattice is read at. VisualGeometryToolkitEstimation's callers use the public name.
- **`bilinear_weights(uv, (W, H))`, public** (was `_bilinear_setup(H, W, uv)`; the size now follows the position, as
  `(W, H)`): the texel block and blend fractions `bilinear_sample` reads, for a caller that spreads a position over
  the four texels itself.
- **`julia_blocks(path)`, public** (the Test extension; was a closure of `test_readme_example`, over a page's text):
  the code of a page's `julia` fenced blocks, in order.
- **`signed_area(ring)`, public** (was `_shoelace_signed_area`, Primitives): the signed shoelace area of a closed ring of
  2-D points, the one body of `area(::Polygon)`, the view-overlap clip's orientation and VisualGeometryToolkitFeatures'
  `detect_image_circle` hole size (which restated it on a contour, no `Polygon`). Bit for bit the same sum.
- **The dependency allow-list names VisualGeometryToolkitEstimation for VisualGeometryToolkitFeatures**
  (`configs/stack_dependencies.toml`): a weak dependency there, the extension behind `detect_image_circle`'s fit
  (VisualGeometryToolkitFeatures 0.8.0). No API change.

## 0.9.1 (2026-11-12)

A patch: the test extension's naming checker, and tests hardened for Julia 1.12. No API change.

- **Fixed: `check_naming` no longer scans `.release/`** (the top-level directory only), the release tooling's records,
  as it already skipped `docs/superpowers/`: a record's first line is in the tooling's fixed format, which rule R7's
  banned stem cannot rename, so a root-repo package's own naming test failed on it. A `.release/` deeper in a tree is
  still scanned (`naming_scope`'s docstring; tested).
- **Fixed: R9 flags a retired name in a `@testset`/`@testitem` name on Julia 1.12 too.** Base's JuliaSyntax 1.x (Julia
  1.12) gives a macro's name the token kind `MacroName`, 2.x (Julia 1.13) `Identifier`; the lexer accepted only the
  latter, so on 1.12 a test's name was read as a plain string and the retired name went unflagged.
- **New: a foreign program's command-line value** (`FOREIGN_FORMAT_TOKENS`, `accepted_in = :cli_argument`): the retired
  `central_generic` is accepted as the value of a `--flag` (`--model central_generic`, Schöps et al.'s
  camera_calibration model) on a shell script's command line (`.sh`, not a comment) and inside a TOML string (a quoted
  command line); every other use (a Julia name or string, a TOML key or plain value, a comment, prose) is still refused.
- **Tests hardened for Julia 1.12** (CI runs 1.12.7): three bit-for-bit comparisons that the compiler's floating-point
  contraction moves by an ulp now hold to a few eps (`whitened_residuals!` of a vector constraint, the `SplineLens`
  Dual projection's value, the `MappedSurface` intersection in a KernelAbstractions kernel; its NaN-miss pattern still
  exact).

## 0.9.0 (2026-11-12)

Breaking (the next release is a minor bump). A bounded camera carries its LENS COVERAGE — where the lens covers its
sensor — so every consumer asks the camera what it images instead of supplying an image circle of its own.

- **New: `bearing_csponds(camera1, camera2, cs_px)`** (public, Cameras; moved from Synthetic): pixel correspondences as
  bearing correspondences, each endpoint back-projected with its pixel Jacobian (the `M₁`/`M₂` attribute columns).
- **New: `prefix_sweep`** (public): the device-callable prefix sweep of a sorted column, the one body behind
  `ig_sweep_kernel!` and the host `sweep!` (equal output, `===`, tested on the CPU and a CUDA kernel).
- **New: `check_keys(...; renamed = (old => new, …))`** refuses a renamed TOML key by naming its successor (no alias).
- **New: `install_artifact` and `huggingface_token` are Core's** (base/artifacts.jl, moved from
  VisualGeometryToolkitEval, which re-exports the same bindings). Core gains the stdlibs `Artifacts`, `Downloads` and
  `Pkg` as dependencies. `huggingface_token` reads `HF_TOKEN`, then `$HF_HOME/token`, and is sent only to huggingface.co.
- **`efficiency_constant(lo::AbstractOptimizationStrategy, d)`**: the efficiency constant a strategy deploys at a
  scale, codimension `d` — its own `EfficiencyConstant(c)`, else the table's `efficiency_constant(K, d)` of its shape
  (`lo_efficiency_constant_shape`); refused for a truncation that states none and for an absolute `LossWidth`. The one
  rule of Estimation's refines and fits and PoseLib's refiners (their private copies deleted); values bit-identical.
- **`map_estimate(f, r::FitResult)`, public**: a result with its estimate mapped, the rest as it was (a RANSAC
  result's `nothing` kept); Estimation's metre and frame re-wrappers and PoseLib's physical `refine_absolute_pose` use it.
- **`RetiredKeyword()` and `refuse_retired_keyword(value, old, new)`, public** (were Core's `_NoFree` and Estimation's
  `_RetiredKeyword`): the stack's one refusal of a retired keyword. `LensSpec(…; free)`'s refusal now reads
  "the keyword `free` is retired: give `fixed`" (was "LensSpec: … the coefficients held (every other one moves)").
  `check_naming`'s R13 reads the refusal slot qualified too (`VgtkCore.RetiredKeyword()`).
- **`tree_sum!` on a group of one** (`nt == 1`) returns at once, taking no barrier: KernelAbstractions' CPU backend
  splits a kernel only at a `@synchronize` written in its body, so a CPU kernel calls the helper with one work item a
  group (the stack's CPU group size for its per-blob kernels); it errored there before. Tested on the CPU and on CUDA.
- **Breaking: `lm_retcode` is removed** (public in 0.8; the termination record is the one map). The chart LM's outcome is
  the `(code, rule)` record `lmsolve_chart!` returns (from `nlls_termination`); a caller that still wants the former
  Symbol reads `retcode_symbol(code, rule)`.
- **`reset_posterior!(pws; …)`: a keyword left out keeps the state's current value** (the owner's ruling, 2026-11-12;
  `prior` was required and `codimension`, `inlier_prior` defaulted to 1 and ½), so a caller re-seeding one parameter
  states only that one. A call that gives every keyword is unchanged.

- **`rms(residuals, n = length(residuals))`**: the fits' root-mean-square residual `√(Σ r² / n)`, the one computation
  (a method of the `rms` accessor); VisualGeometryToolkitEstimation's `fit_camera`, `refine_camera_and_poses` (spline),
  and `fit_division_camera`'s step use it.

- **`reproject_so3(R)`, public** (was the private `_reproject_so3`): a drifted rotation matrix back on SO(3) by its
  unit quaternion, the charts' one re-anchoring and a caller's before an `EuclideanMap`'s rotation check.
- **`inv(pose::Uncertain{<:EuclideanMap{3}, <:CovMat{6}})`**: the inverse of an uncertain rigid pose with its
  covariance carried through in the `RigidExpManifold6` chart (hoisted from BlobBoards' bmvc2026
  `inv_pose_covariance`).

- **New dependency: GPUArrays** (11.5), for `free_device_buffers!(xs...)` (public): the ONE eager release of a device
  buffer (GPUArrays' `unsafe_free!`; a host array untouched), hoisted from RANSACScoringDoneRight. KernelAbstractions'
  `unsafe_free!`, which VisualGeometryToolkitFeatures called, is not extended by CUDA.jl 5.11 and freed nothing on the GPU.
- **`@threshold_scoring_constructors S`, public**: a threshold-scorer family's constructors (`S{DG}()`, `S()`,
  `S(dg; threshold)`, `S(problem; threshold)`, `S(P; threshold)`) and its `with_width`, generated once; `RansacScoring`
  and `MsacScoring` are built by it (bit-identical), and a downstream family by the same line.
- **`RadixPermBuffers{K, I}`**: the radix sort's permutation of any integer type `I` (`radix_sortperm!(perm::Vector{I},
  …)`); `RadixPermBuffers{K}(n)` is `I = Int`, bit-identical.

- **New methods of existing functions (hoists from downstream copies).** `is_stationary(blocks, variables, free; tol,
  ftol)`: the stall postcondition of an NLLSsolver problem's residual blocks in the columns of its free variables, the
  √ρ′-weighted gradient and column norms accumulated block by block (each block's `robustkernel`), the problem's
  Jacobian never held — the form RadiometricCamera's `lm_stationary` faked a two-row Jacobian for and
  PanoramaStitching's rotation bundle cut its held columns for. `lens_coverage(camera, rim_margin)`: the lens coverage
  less a rim margin at the lens's edge (a `Circle`'s radius, an `Ellipse`'s semi-axes, a `Rect`'s sides inside the
  sensor), or `nothing` — PanoramaStitching's frame-region rim and RadiometricCamera's image circle less its rim.
- **One kernel-launch policy for the stack: `OnBackend(backend; workgroupsize)(kernel)`** (the owner's ruling, 2026-11-12).
  The public `OnBackend` now instantiates a KernelAbstractions kernel (a call method; `(exec::OnBackend)(kernel,
  ndrange)` for a statically sized one), and every stack kernel is instantiated through it: Core's `pointwise!`, row,
  conversion, CSR, warp, sample, chart-spline, spline-lens, pool-selection and sweep kernels, VisualGeometryToolkitFeatures'
  and RANSACScoringDoneRight's. The workgroup is per backend: `nothing` the backend's choice; ONE work item (`1`,
  `(1, 1)`, …) one item per workgroup on the CPU (a kernel whose item is a long loop then runs on every thread) and one
  warp (32 items) per workgroup on a device; any other size as given. The pool
  kernels (`exact_pm_score_kernel!`, `argmax_models_kernel!`, `ig_sweep_kernel!`) are launched at one item: on the CPU
  unchanged, on a device `select_on_grid!`'s exact selection and RANSACScoringDoneRight's grids no longer run one
  thread per workgroup (RANSACScoringDoneRight's grids on a B200, 128 pairs × 800 points × 48 models: the Student-t
  grid 1.55 ms at the backend's choice, 30.6 at one item, 1.35 at one warp; its mixture marginal 90.4, 62.1, 17.7). Every other
  launch keeps its workgroup. Results bit for bit (every launch site, CPU and CUDA, before and after).
  One warp rather than the backend's choice: at the backend's choice RANSACScoringDoneRight's Welsch and mixture
  marginals ran as at its last release (86.6 / 90.4 ms) but slower than at one item per workgroup (67.7 / 62.1 ms);
  at one warp all five of its grids are faster than both (64, 128 and 256 items within 15 % of 32).

- **A SplineLens projects a ray carrying ForwardDiff Duals.** `project(::SplineLens, X)` of a Dual ray projects its
  values by the plain solve and carries its partials by the solve's implicit derivative (`_projection_jacobians`' ∂x/∂t),
  one Dual layer at a time, so nested Duals differentiate exactly; no Dual enters the LM inverse. With a forward spline the
  value and its partials are the forward's, the Duals straight through its chart and spline. A pose refine on a
  SplineLens camera (`refine_absolute_pose`, its `transform_deriv`) threw a MethodError in the Float64 cold start.

- **`Attributed` delegates by `MacroTools.@forward`**, the stack's one idiom (as `BoundedCamera`): the hand-written
  `iterate` of a wrapped Pair and `getindex`/`length`/`size` of a wrapped vector are two `@forward` lines. Behaviour is
  unchanged (the same types; `a.value` is `getfield(a, :value)`).
- **`chisq_median` is retired** (refused by name, `check_naming`'s retired list): the median of `χ²_d` is
  Distributions' `quantile(Chisq(d), 0.5)`, called directly as every χ² quantile in the stack is (owner, 2026-11-12: one
  χ² quantile, Distributions'). `mad_scale` and `_mad_scale_norms` compute it inline, bit-identical (the same call).
- **`check_naming` retires `LO_TAGS`** (VisualGeometryToolkitEstimation's strategy registry, now
  `OPTIMIZATION_STRATEGY_TAGS`; beside LocalOptimizationDoneRight's `LODR_STRATEGY_TAGS` and PoseLib's
  `POSELIB_STRATEGY_TAGS`).
- **`scale_source(::ScaleInput)`**: a stage's parsed scale input answers the stack's accessor pair as a result does —
  `scale_source(input)` its source (`ScaleNone` for a width; `nothing`, no input, too) and `noise_scale(input)` the
  scale it was given (refused for a width) — so a stage records what it was given through accessors, never the
  fields (a downstream bundle reading VisualGeometryToolkitEstimation's `bundle_loss`).
- **A no-width `Irls{K}()`/`Triggs{K}()` of a shape with no decision width (Huber) runs at the table's
  `efficiency_constant(K, d)` times the consensus scale** (`loss_at`, `d` the scorer's codimension; the owner's
  ruling, 2026-11-12). It was a `MethodError` from `decision_width`. The shapes with a `decision_width` method (Welsch,
  Geman–McClure, Cauchy, smooth TLS, TLS, Barron) follow the scorer's decision point as before; a shape with neither is an `ArgumentError`
  naming it and the remedy.
- `Triggs{K}()` is constructible with no efficiency constant (`width === nothing`): a refine
  (VisualGeometryToolkitEstimation's `refine_*`, whose default it is) runs it at `efficiency_constant(K, d)` times the
  caller's `scale`, and a RANSAC local optimization follows the scorer's `decision_boundary`, as `Irls{K}()` does.
  It was an `UndefKeywordError`.
- **The predictive Student-t scorer decides a sample point held once, from its score**: its scores and penalties are
  zeroed together by `zero_sample!(q, penalties, sample)` (new two-buffer form), so a point with a finite score beside
  a non-finite penalty (or the reverse) is never half held. `sample_point_held` is documented with the scoring
  contract (robust/scoring_contract.jl, the Robust Estimation API page).
- **A minimal-sample point with a non-finite residual is not held** (`sample_point_held`, the one rule every scorer
  reads: the threshold families' consensus partition and score correction, `zero_sample!` and so the Student-t
  scorer's zeroed scores and the pool staging). Such a point is one the model does not reach (a lattice detection a
  pose leaves unassigned); forced into the consensus set, its Inf residual made `consensus_s2` Inf and counted a dof it
  does not have (RadiometricCamera's grid index, 2026-11-12: 86 of 142 photographs of one set). It now stays an
  outlier; a finite sample point is held at zero residual as before, so every result without such a point is unchanged.
- **`BoundedCamera` forwards every camera verb** (a BoundedCamera is the type for pixel operations): its one
  `MacroTools.@forward BoundedCamera.camera` list, moved to the end of the Cameras module so it can name every verb,
  adds `backproject_with_jacobian` (a SplineLens BoundedCamera was a `MethodError`), `constraint_with_jacobians`,
  `imaging_transform`, `imaging_chain`, `distortion_map`, `has_distortion`, `fov_radius`, `horizon_radius`,
  `valid_rect`, `undistorted_rect`, `division_bounded`, `undistort`, `p3p`, `transfer` and `radial_homography` to the
  accessors, `unproject` and `is_in_front`; `project`/`backproject` stay `@inline` (device kernels). As for those two,
  no forwarded verb applies the bound. The model constructors (`pinhole`, `window`, `with_parameters`) and the
  parameter protocol stay the model's: call them on `model(bc)`. A quality test checks that every exported or public
  verb with a `CameraModel` method has a BoundedCamera one, or is a listed exception.
- **`check_duplication(dir)` never compares a file with itself**: a subdirectory of a repository (a paper project in
  a package's repository) lies inside the repository's own tree, so its files were scanned twice and every one was
  reported as a whole-file copy of itself (`X ⇔ X`). Each file is now scanned once, by its real path, on its own side.
- **`HartleyNormalization` applies to 3-D points**: `AffineMap(HartleyNormalization(), points3d[, weights])` maps
  `StaticVector{3}`s about their (weighted) centroid to mean distance √3 (Hartley & Zisserman), the normalization of a
  least squares over lengths; a known centre stays 2-D (`ArgumentError`). Two bodies (`_hartley_map`, `_hartley_map_3d`)
  share one acceptance rule (`_hartley_accept`) and degeneracy guard; the 2-D map is unchanged, bit for bit.
- **`check_naming` retires `bundle_refine_relpose` and `filtered_rotation_averaging`** (VisualGeometryToolkitEstimation's
  `refine_relative_pose` and `fit_filtered_rotation_averaging`); conventions/units.md names the new one.
- **`check_naming` retires `default_local_optimization` and `_fit_local_optimization`** (VisualGeometryToolkitEstimation's
  renames to the strategy keywords' spelling).
- **Breaking: the Student-t priors are required everywhere** (the owner's ruling (c), 2026-11-11), as the public
  scorer already required them: `PosteriorWeightState(n; …, prior)`, `reset_posterior!(pws; …, prior)`,
  `posterior_weights!`/`posterior_inlier_probs!(…; prior)`, `median_seeded_posterior(sq, n; outlier_halfwidth, ν,
  prior)` (its `ν = 1e4` and `IGPrior(0, 0)` defaults removed too) and `default_scoring_workspace(n; dg, scale_prior,
  label_prior)` (its label prior was hard-wired flat) have no prior defaults; leaving one out is an
  `UndefKeywordError`. A caller that relied on the default supplies `prior = JeffreysPrior()` /
  `label_prior = FlatLabelPrior()`, bit for bit.
- **`median_seeded_posterior` seeds at `mad_scale`²** (the no-duplication rule): its seed scale is the median residual
  NORM over the median of `χ_d`, squared — Core's one median scale — instead of a private `median(r²)/chisq_median(d)`.
  The two agree for an odd count up to rounding; for an even count the median of the norms averages the two middle
  norms, not their squares, so the seed moves slightly (the EM it starts converges to the same fixed point).
  Measured (dup-items-a probe, the DivisionBoards scene of the tests): `fit_division_camera`'s focal 1219.8623257467295 →
  1219.8623257468823 px, its division coefficient −0.9621859413513908 → −0.9621859413516484, rms 0.5449892554591349 →
  0.5449892554590682 px, noise scale 0.38533670719050517 → 0.38533670719045765 (24 iterations either way); a seed at an
  even count moves by ~1.5e-5 relative (0.08892803174088 → 0.08892669060497886, n = 40), at an odd count by an ulp.
  Estimation's SimFitPP keeps the median of the SQUARED kept residuals: there it is the truncated-median quantile
  correction's own statistic (`unbiased_median_variance`), not this seed.
- **Breaking: `CSRMatrix` and the `:pool` grid selection default to `default_device()`, and the `:pool` result gains a
  `backend` field** (a caller comparing a whole `stage_selection(:pool, …)` result with `==` changes) (the stack's GPU-first rule):
  `CSRMatrix(A; device = default_device())` (was `KA.CPU()`), and `stage_selection(:pool, pairs, residuals; backend)`
  carries its backend (`default_device()` when none is supplied), on which `select_on_grid!` runs the exact kernels
  of a residual-quality family (they were hard-wired to `KA.CPU()`). On the host the selections are unchanged, bit for
  bit.
- **The batched `backproject`/`backproject_with_jacobian` run one kernel on every backend** (the one-kernel rule, the
  no-duplication gate's `backend_branch` at camera_models.jl): the `device isa KA.CPU` test is gone. The model goes
  in its device form (isbits, or its adapt rule; the identity on the CPU), so the CPU path and its results are
  unchanged bit for bit; a model with no device form now fails on an accelerator as every `pointwise!` kernel's does,
  instead of running as a host map there.
- **Breaking: the scorer's decision point is `decision_boundary`** (the owner's ruling). `inlier_threshold(scoring, buf)`
  → `decision_boundary(scoring, buf)`, `has_inlier_threshold(S)` → `has_decision_boundary(S)`, and every
  `decision_threshold` method — the Gaussian–uniform pair's `(::GauInlier, outlier_halfwidth)` and the loss shapes'
  `(shape, loss_width)` — is a method of the same `decision_boundary` generic (no signature clashes: a scorer with
  its buffers, a `GauInlier`, a loss shape lead their own methods; one name for "the residual where the inlier/outlier
  decision flips"). The docstring states both behaviours: a threshold scorer's boundary is its threshold τ, a
  constant; the Student-t + uniform scorer's is the MAP crossover at the CURRENT consensus scale ŝ and its outlier
  half-width h, an estimate recomputed with every score. `decision_width(m, boundary)` keeps its name (its argument
  renamed). The old names are retired (`RETIRED_NAMES`); `CONTRACT_GENERICS`, the LO contract's "follow the scorer"
  width and `check_composition`'s message say "decision boundary". Behaviour is unchanged, bit for bit.

- **Breaking: `LensSpec` holds coefficients by `fixed`** (was `free`, the coefficients moved; the owner's ruling, the
  stack's one parameter-holding keyword). `LensSpec(family; fixed = (), structure...)`, `LensSpec(lens; fixed)` and the
  table key `"fixed"` name the coefficients HELD; `free_names(spec)` (unchanged, a contract generic) is the complement in
  `parameter_names` order. The keyword `free` and the table key `"free"` are `ArgumentError`s naming `fixed`; a
  calibration manifest of an earlier version is read through VisualGeometryToolkitIO's version step. `toml_value`
  writes `"fixed"` only. Rule R13 also covers `LensSpec`/`LensTarget` calls and lens tables.
- **`free_parameter_names(model, fixed)` (public): the stack's one expansion of a `fixed` list.** `parameter_names(model)`
  (or a names tuple) in its order, less `:skew` (the gauge, held always: naming it is accepted, so `fixed =
  parameter_names(model)` holds everything) and less `fixed`, whose entries are parameter names or groups (a
  coefficient family's stem, `:k` for `k1…`, `:d`, `:p`, `:sx`, `:sy`). An entry that names nothing, or a parameter
  named twice (directly or through a group), is an `ArgumentError`. `LensSpec`'s `fixed` and `free_names` resolve
  through it, so a `LensSpec` now takes groups too.
- **`check_naming` rule R13: the one parameter-holding keyword is `fixed`.** `free`, `fix`, `fixed_params` and `hold`
  are flagged as a KEYWORD of a `fit_*`/`refine_*` call or signature (and their `_`-prefixed internals) and as a key of a
  refine or LM TOML table; the bare word (comments such as "CUDA-free", locals, fields, accessors such as `free_names`,
  other functions) is not. A refusal slot (`free = RetiredKeyword()`) and a `@test_throws` call are no finding.
- **New: the core optimization strategies `Irls{K}` and `Triggs{K}` are VisualGeometryToolkitCore's** (moved from
  VisualGeometryToolkitEstimation, so PoseLib and every estimator take the same configuration types), with `LOSS_TAGS`,
  their width modes and identity (`strategy_loss`, `lo_loss`, `fixed_width`, `lo_width_value`, `lo_admits_shape`), the
  strategy registry machinery every package registers its strategies with (`@lo_registry`, `lo_construct` and the hooks
  `lo_tag`, `lo_defaults`, `lo_parametric`, `lo_admits`, `lo_key`, `lo_field`, `lo_field_from_toml`, `lo_absent`,
  `lo_validate`, `lo_efficiency_constant_shape`, now contract generics; `lo_has_default`, `lo_takes_width`) and their
  registry `STRATEGY_TAGS`. Their keyword defaults are Core's new `configs/default.toml` (`[optimization_strategy.irls]
  max_iter = 25`). A strategy is configuration only; the algorithms that run it stay the estimators'. `Triggs{K}`
  selects the loss that `loss_robustifier` hands NLLSsolver.jl; nothing in Core forms the robustified LM.
- **Breaking: one optimization-strategy family, `AbstractOptimizationStrategy`** (was `AbstractLocalOptimization`; the
  owner's ruling): Irls, Triggs, the LO wrappers and the downstream strategies alike, the contract gate checking which
  subset each verb accepts. No local optimization is `nothing` (TOML `"none"`): Core answers the LO contract for it
  (`locally_optimize!` returns the seed, `:no_lo`; `lo_workspace`, `lo_width`, `RefitFamily`, `optimization_path`
  `:seed`, `scale_interface` `NoScale()`, `lo_label` `"seed"`, `loss_family` `:none`). The retired names are in
  `RETIRED_NAMES`.
- **Breaking: a width multiple is the `efficiency_constant`, always the PLAIN multiple** (the owner's Q7 ruling). The
  keyword, field and TOML key `scale_multiple` is `efficiency_constant` (`deploy(shape, s2; codim,
  efficiency_constant)`, a `DeployedLoss`'s `efficiency_constant`), the width mode `ScaleMultiple` is
  `EfficiencyConstant` (its persisted axis name `:efficiency_constant`), and it defaults to
  `efficiency_constant(shape, codim)`. The per-shape currencies are retired: the LO contract's
  `deployed_scale_multiple` hook is removed, so `loss_at` deploys `c·ŝ` for every shape; a Geman–McClure multiple is
  no longer read as a half-weight depth. Stored values were migrated once at full precision (`decision_width(GM, m)`;
  Barron `m/√2` downstream), so every result is bit-identical. `EfficiencyConstant(c)` refuses a non-positive or
  non-finite `c` (give no width to follow the scorer). The retired names are in `check_naming`'s `RETIRED_NAMES`, with
  the call forms of `noise_sigma`.
- **New: every fit's termination record, `ReturnCode` and `StopRule`; Breaking: `converged` is strict** (the
  robust-estimation contract §14, owner rulings 2026-11-11 and 2026-11-12: a released version carries no
  transitional inconsistency).
  - `ReturnCode` is SciMLBase's return-code interface, MATCHED without the dependency: a module-scoped enum
    (`ReturnCode.T`, `ReturnCode.Success`) with SciMLBase's names, integer values and `successful_retcode` rule, for
    the codes the stack produces: `Default`, `Success`, `Terminated`, `MaxIters`, `MaxTime`, `Stalled`, `InitialFailure`,
    `Failure`, `InternalLinearSolveFailed`. A test pins the names and values to SciMLBase 3.58's.
  - `StopRule` is the detail, which rule fired: `NoRule`, `ClosedForm`, `CostDecrease`, `IncrementSize`, `RejectionCap`,
    `ParameterChange`, `NoImprovement`, `ExactFit`, `ToleranceMet`, `LadderConverged`, `LeftBasin`, `ShapeGate`, `IterationCap`,
    `TimeCap`, `NonFinite`, `Callback`, `TrialBudget`, `AdaptiveCount`, `SeedKept`.
  - **Breaking: the FitResult core holds the record** (the transition is resolved: every producer records a code, so
    `converged` agrees with `successful_retcode(retcode(r))` for every result and every code, a closed form's
    included). The core field `retcode::Symbol` is `retcode::ReturnCode.T`, beside a new core field
    `stop_rule::StopRule`; `retcode(r)` returns the `ReturnCode.T` and `stop_rule(r)` the rule for every result, each
    inferred. There is no metadata copy: the `return_code` generic is removed. `retcode_symbol(r)` is the legacy
    Symbol spelling (`retcode_symbol(code, rule)`, the one map back). The termination types load before `FitResult`
    (base/return_code.jl); the accessors are in base/termination_accessors.jl.
  - **Breaking: `converged(r)` (and `r.converged`) is `successful_retcode(retcode(r))`, strict**: an iteration or
    RANSAC trial cap (`MaxIters`) is not converged, a closed-form solve (`Success`, `ClosedForm`; the old `:solved`)
    is. `found_model(r)` (`estimate(r) !== nothing`) is the old "a RANSAC found a model".
  - Migration, one line per old form:
    - `FitResult(x, :sym, iters, rms, n, md)` → `FitResult(x, retcode_from_symbol(:sym, rule)..., iters, rms, n, md)`
      (`rule` the loop's success rule), or `FitResult(x, code, rule, iters, rms, n, md)`;
    - `lo_estimate(θ, :sym; kw...)` → `lo_estimate(θ, :sym, rule; kw...)`, or `lo_estimate(θ, code, rule; kw...)`
      (an LO status no fit reports, LODR's `:left_basin`, states its record: `(ReturnCode.Terminated, LeftBasin)`);
    - `retcode(r) === :sym` → `retcode_symbol(r) === :sym`, or `retcode(r) == ReturnCode.X`;
    - a rebuild `FitResult(f(estimate(r)), retcode(r), iterations(r), …)` → `FitResult(f(estimate(r)), retcode(r),
      stop_rule(r), iterations(r), …)`;
    - `converged(r)` read as "a RANSAC found a model" → `found_model(r)`;
    - `return_code(m::MyMetadata) = …` → delete it (the record is the core's).
    - `nlls_retcode(result, J, r; tol, ftol)` (removed: no consumer left) → `retcode_symbol(nlls_termination(result,
      () -> is_stationary(J, r; tol, ftol))...)`, or the record itself.
    - `BundleMetadata(w, r, fixed, dof, width, before, after)` → `BundleMetadata(w, r, fixed, dof, width, before, after;
      scale, scale_source)` (below).
  - **Breaking: `BundleMetadata` records the bundle's scale** (the robust-estimation contract's scale record, as
    VisualGeometryToolkitEstimation's `CameraPosesRefineMetadata` does): required keywords `scale` (the noise scale the
    bundle ran with; `NaN` under least squares) and `scale_source` (`ScaleGiven`, `ScaleUpstream`, or `ScaleNone`), read
    by `noise_scale` and `scale_source`. Its termination record is the FitResult core's, as every result's. It moved to
    base/bundle_metadata.jl (after the `ScaleSource` it holds).
  - The no-local-optimization strategy `nothing` records `(Success, ClosedForm)` (the seed returned as it is).
  - **`ReturnCode.Default`** (SciMLBase's, value 0: "the solving process has not been run"; not successful) is the
    record of a solve that never ran — a RANSAC result's local-optimization record when none ran. **`StopRule`
    `SeedKept`**: a RANSAC's score guard kept the seed (recorded with the refit's own code).
    `retcode_from_symbol` also reads a RANSAC's former local-optimization statuses: `:not_run` → `(Default, NoRule)`,
    `:guard_kept_seed` → `(Success, SeedKept)` (both read back by `retcode_symbol`), `:no_lo` → `(Success, ClosedForm)`,
    `:thin_partition` → `(InitialFailure, NoRule)`.
  - `test_termination_record(r)` checks a result's record: `retcode`, `stop_rule`, `converged` and
    `retcode_symbol(r)` inferred, `converged(r) == successful_retcode(retcode(r))`. `test_local_optimization_contract`
    runs it on the refit it checks.
  - `ReturnCode`'s type and every value are public inside its module, so another package's qualified
    `ReturnCode.Success` is a public access.
  - `nlls_termination(result_or_bits, postcondition)` maps NLLSsolver's termination bits to `(ReturnCode, StopRule)`
    in `nlls_stop_code`'s one order. `lmsolve_chart!` returns it as a fifth value, under its own stall postcondition.
    `retcode_from_symbol(s, rule)` is `retcode_symbol`'s inverse, for a loop that still reports a Symbol.
    `worst_retcode` combines the solves of a multi-solve fit (the most severe; the later on a tie).
- **New: the robust stages' scale hand-off, `ScaleSource`, `noise_scale`, `scale_source` and `scale_input`**
  (public; the robust-estimation contract, owner rulings Q1/Q2). Every robust result will record ONE scale — the one it
  ran with, or the one it estimated — and where it came from:
  - `@enum ScaleSource ScaleGiven ScaleUpstream ScaleEstimated ScaleNone` (isbits, so a metadata that carries it stays
    concrete; the prefix keeps clear of the update-policy type `Estimated`);
  - `noise_scale(result)` / `scale_source(result)` read the metadata's `scale` and `scale_source` (contract generics,
    for a metadata that records them otherwise); `noise_scale` refuses a `ScaleNone` result (its scale is `NaN`) and a
    result without metadata, so a NaN is never handed to the next stage;
  - `scale_input(scale, width) -> ScaleInput{T}` parses a stage's keywords: exactly one of `scale = σ`,
    `scale = result` (`ScaleUpstream`) or `width = w`, each positive and finite; neither or both is an `ArgumentError`.
    A width-only stage records `ScaleNone` and carries the width: a width may be cross-validated rather than
    generative, so no scale is ever backed out of it.

  No estimator records a scale yet; the verbs adopt it in their own changes.
- **Breaking: `mad_scale(residuals)` is the robust noise scale of residual VECTORS; `noise_sigma` is removed** (the
  owner's ruling). `mad_scale(rs)` takes `StaticVector{d}` residuals, reads `d` from their type, and returns
  `median(‖rᵢ‖) / √chisq_median(d)`: centred on zero (the model), never on the residuals' own median. A kernel that
  holds only norms calls the stack-internal `_mad_scale_norms(; norms, codimension)` (not public); there is no public
  norms form, so a reader always sees that nothing is re-centred. The numbers are `noise_sigma(norms, d)`'s, bit for bit (Estimation's three
  callers switched). The former centred, normal-consistent `mad_scale(x)` of scalars is gone (StatsBase's
  `mad(x; normalize = true)` is that estimator); its scalar call shape is a `MethodError`, not silently re-meant.
- **New: per-datum weights carry their kind as a trait** (public; the robust-estimation contract, Q3). `WeightKind` —
  `BinaryWeights()` (a threshold scorer; an L2 result), `PosteriorWeights()` (the Student-t + uniform scorer, an EM
  fit), `IrlsWeights()` (an M-estimator's loss weight) — is read by `weight_kind(result)` from the metadata's TYPE,
  which answers from its producer's (a contract generic). `inlier_mask(result)` dispatches on it: `w > 0`, the MAP
  rule `is_map_inlier`, or an `ArgumentError` for IRLS weights, which have no cut of their own;
  `inlier_mask(result, cut)` takes an explicit cut for any kind. `RansacScoring`/`MsacScoring` answer binary and
  `StudentTUniformScoring` posterior. No result type adopts it yet.
- **New: an L2 result's weights are lazy unit weights, binary by kind.** `unit_weights([T], n)` is
  `FillArrays.Fill(one(T), n)` (no allocation; FillArrays is a new dependency), what an L2 producer stores as its
  `weights`; an L2 producer is `nothing`, and `weight_kind(Nothing)` is `BinaryWeights()`, so its `inlier_mask` is every
  datum. `weights` is a contract generic (a metadata that stores its weights otherwise adds a method), and a result
  without metadata refuses `weight_kind`.
- **New: `efficiency_constant(shape, d)` on a shape instance** — the entry point a deployment calls on the shape it
  deploys (and `efficiency_constant(shape)` at d = 1); the type forms stay. `efficiency_constant` is a contract
  generic: a shape of another package states its constant. No value changed.
- **The `efficiency_constant` table is re-derived by a test** (`test/efficiency_constant_tests.jl`): the
  quadrature solves eff(c, d) = 0.95 under χ_d for Huber, Cauchy, Welsch and Geman–McClure at d = 1…3 and matches
  the table to its 4 decimals, so the provenance lives in the repository rather than in a scratch script. No value
  changed.
- **A height field's ray hit with Duals takes the fast path: the 1-D Newton hit plus ONE implicit-function step.**
  `intersection` of a `MappedSurface` whose chart map is a height field over an affine chart (`_HeightOnlyMap`), with
  ForwardDiff Duals in the heights, the placement, the pose or the ray, no longer falls back to the generic 2 × 2 LM from
  the plane hit. The values' hit is the plain 1-D Newton's (`_height_root`, the fast path's); the partials come from ONE
  evaluation of the chart map on the Dual parts at that chart point — ∂p = −(UᵀJ)⁻¹ ∂r in the ray's tangent basis U, the
  point's by the chain rule 𝐱 = 𝐗 + J (p − p_V) (`_height_implicit`). The Dual type is read off the members' and the ray's
  types at compile time (`_hit_numbers`), so the path compiles for a device. Nested Duals (a Hessian) keep the generic root,
  seeded at the primal fast hit. The generic root's implicit-function step (`_implicit_partials`) evaluates the residual
  only (`_value_residual`), not its Jacobian, on the Dual parts.

  Values and partials agree with the generic root's to 1e-12 (the golden `test/fixtures/height_surface/dual_hit.jl`,
  recorded at 4c9252d6), on the host and on a CUDA device (Float64; Float32 to its rounding), Dual heights and a Dual ray.
  With 25 height partials the hit costs ≈ 1.6× the plain hit (the generic root: ≈ 3× on a hit, ≈ 36× on a miss, ≈ 5× over
  the fixture's rays; the test gates 3×), no allocation. RadiometricCamera's `HeightBoardRays` (its own fast hit plus
  implicit-function step for the board heights' Jacobian columns) is retired onto it; RC's Float64 height kernel on a
  12 048-px view on a B200: 67 ms with `HeightBoardRays`, 51 ms with this.

- **NLLSsolver ≥ 5.3.4** (`[compat]` floor, the package and its tests): 5.3.4 adds our pull request #69, the run-time
  block views (a residual block of a run-time size assembled without a per-block allocation).

- **New: `axis_gauged(camera, R, K′)`, a SplineLens camera in the frame turned so ẑ is its optical axis**
  (public, exported from `Cameras`). It is an exact re-gauge, not a fit:
  - every backward control is turned (`with_directions`, `c ↦ R·c`, every ray `R·d` to rounding);
  - `K` becomes the skew-0 `K′`, the symmetric camera's fitted principal point, focal lengths and aspect
    (`CameraModel(camera, K′)`), so no ray changes;
  - the stale forward spline is dropped.

  `axis_gauged(extrinsics, R)` turns a world → camera pose by the same rotation (`(R·R_p, R·t)`, the frame tags
  kept), so every image is kept. A posed camera turns its extrinsics with its model; a bounded camera keeps its bound.
  A spline that is never gauged keeps its source's ẑ.
- **New: the chart LM's two reductions, `residual_cost!(r, lm, θ)` and `residual_normal_equations!(r, J, lm, θ)`**
  (public, contract generics). They are what `lmsolve_chart!` reads (the cost; ½‖r‖², Jᵀr, JᵀJ).
  - The defaults reduce `residual_rows!` and `residual_jacobian_rows!` on the host, as before.
  - A state whose rows live on a device overrides both with one kernel, and its rows never reach the host;
    Estimation's `fit_camera` regression is the first.
  - The stall postcondition reads them too: the cosine from `g` and `diag(H)`.
- **Changed: the scoring contract's workspace readers have fallbacks**: `score!(W, scoring, scores, residuals, n)`,
  `inlier_count(scoring, W)` and `consensus_order(scoring, W)` on a workspace another scorer built are an
  `ArgumentError` naming the fix (was a `MethodError`); `check_workspace` ignores the fallback. A caller that holds a
  scorer of an abstract type (a refine's anchoring scorer) analyses clean under JET.
- **New naming rule R12: the Student-t + uniform scorer has no threshold** (the owner's ruling: every scorer takes its
  own width keyword exactly). `check_naming` flags the word `threshold` within six tokens of the scorer's type name or
  its TOML tag, in code, comments, docs and configuration (a changelog excepted); `inlier_threshold` and
  `decision_threshold`, the contract's decision point, are other names. `conventions/naming.md` states the rule.
  Downstream quality gates see it on their next Core: a refusal test spells the word apart (`"thres" * "hold"`).

- **New: the Student-t + uniform marginal scorer, `StudentTUniformScoring`, moved here from RANSACScoringDoneRight**
  (the owner's ruling: hoisted with no duplication; RSDR deletes its copies and imports these). Moved, not rewritten
  (`robust/partition_sweep.jl`, `robust/student_t_uniform_scoring.jl`): the model-certain and `Predictive` scorer and
  its scoring contract, `score_residuals!`, `score_sq_residuals!`, `ScoringWorkspace` (`sweep_order`, `penalties`,
  `sort_key`, `lg_table`, `resize!`), `alloc_scoring_workspace`, `default_scoring_workspace`, `sweep!` with the exact
  parametric sweep, `partition_score`, the one prefix kernel `ig_sweep_kernel!` (a pool grid launches it per
  (pair, model, width); `sweep!` as one work item on the CPU backend), the radix sort (`RadixPermBuffers`,
  `radix_sortperm!`, `radix_key`), `log_gamma_table`, the label priors (`AbstractLabelPrior`, `FlatLabelPrior`,
  `ProfiledLabelPrior`, `BetaLabelPrior`, `label_prior_table`, `LABEL_PRIOR_TAGS`), `AbstractOutlierModel` and the
  supplied half-width `FixedHalfwidth`, and `suffix_sums!`. An outlier model that ESTIMATES the support (RSDR's
  extreme-value and generalized Pareto models) implements the new contract generics `outlier_cost`,
  `scored_outlier_halfwidth` and optionally `outlier_context!` (`estimates_outlier_support`) and is scored by Core's
  estimated-support sweep. `SCORING_TAGS` gains `"student_t_uniform"`. The scorer's priors are REQUIRED keywords
  (`scale_prior`, renamed from `prior`, and `label_prior`), as are `default_scoring_workspace`'s `scale_prior` and
  `BetaLabelPrior`'s parameters: the stack's default is a configuration's (VisualGeometryToolkitEstimation's
  `[scoring.student_t_uniform]`, the owner's ruling (c)). The supplied-width `score!` no longer allocates: the
  one-work-item launch is a statically sized kernel instance on the CPU backend with its buffers on the workspace
  (RSDR's dynamic launch allocated 3.9 kB per hypothesis, failing its allocation tests). Results are RSDR's bit for bit
  (its tests, and 1026 scoring results plus a sample of neurips2026's pool selections, compared before and after).
  The empty `GauScoring` supertype is not carried over (nothing dispatched on it): the scorer is an `AbstractScoring`.

- **Breaking: `pointwise!` is one KernelAbstractions kernel on every backend; the host's execution policies are
  retired** (the owner's ruling on the twin audit's V4). `pointwise!(f, out, xs...)` and the lens batches (`project!`,
  `backproject!`, `distort_normalized!`, `undistort_normalized!`) launch the one kernel on the output's backend, the
  CPU's included; `exec` still takes `OnBackend(backend; workgroupsize)` (a backend value with its compiler options,
  a workgroup size), now a plain struct. `ExecutionPolicy`, `Auto`, `Serial`, `Threaded` and `THREADED_MIN_LENGTH`
  are gone (RETIRED_NAMES); no stack package supplied them. The inputs share the output's shape (no broadcasting, a
  `DimensionMismatch`); a device array now runs the KernelAbstractions kernel instead of a GPUArrays broadcast (a
  B200 undistortion agrees with the host's per-point map to 3e-16). On the CPU the results are the per-point map's bit
  for bit, as every policy's were. The CPU is slower: a 5-coefficient OpenCV undistortion of 256 / 1024 / 10⁶
  points took 61 µs / 0.64 ms / 4.0 ms with the threaded broadcast and 117 µs / 0.61 ms / 78 ms as a kernel
  (medians, in a 224-thread process).

- **Changed: every LM row fill is one KernelAbstractions kernel on every backend** (the owner's ruling on the twin
  audit's V3: one kernel per job, the slower CPU accepted). `_foreach_row!`, which ran the rows as a plain loop on the
  CPU and as `_row_kernel!` elsewhere, is retired (RETIRED_NAMES): `_launch_rows!` launches `_row_kernel!` on the
  output's backend, the CPU's included (Sampson, tangent Sampson, the homography and radial transfer rows, the per-pair
  Sampson squares). The rows equal the plain loop's to rounding (1 ulp in 10 of 100 Sampson rows; a test checks
  1e-14). The CPU launch is slower: Sampson residuals of 100 / 1000 correspondences took 0.18 / 1.6 µs as a loop and
  14 / 90 µs as a kernel (medians, in a 224-thread process: KernelAbstractions' CPU launch spawns a task per thread).

- **Faster: the CPU launches of the row and pointwise kernels specialize** (fixing the V3/V4 slowdowns above).
  `_row_kernel!` took `args...` and `pointwise!`'s field kernels a function field they only hand on; Julia compiled
  both for a widened signature (`Vararg{Any}`, `Function`) and KernelAbstractions' CPU backend ran every row or point
  through it as a dynamic call with boxed arguments (≈150 B a Sampson row, ≈380 B an undistorted point). Type
  parameters on those arguments force the specialization; the device kernels are unchanged and the results are the
  same bit for bit. Sampson residuals of 100 / 1000 rows: 8.4 / 83 µs → 0.25 / 1.4 µs (a plain loop: 0.13 / 1.3 µs);
  the OpenCV undistortion of 256 / 1024 / 10⁶ points: 137 µs / 0.54 ms / 62 ms → 79 µs / 0.31 ms / 4.6 ms (minima,
  224 threads; at 10⁶ points the kernel matches a hand-threaded host loop). KernelAbstractions' CPU default already
  runs ≤ 1024 items as one workgroup with no task spawned.

- **Fixed: an LM row fill allocates nothing on the host again.** KernelAbstractions' kernel call is itself a vararg
  method, compiled for `Vararg{Any}` when it is given more than one argument, so every `_row_kernel!` launch boxed its
  arguments and dispatched dynamically (≈290 B a launch on the CPU). `_row_kernel!` now takes one argument, the job
  tuple `(row!, out, args...)`, and the launch allocates nothing. The allocation-free refits are allocation-free again:
  Estimation's RotationProblem and CircleProblem refits and its whitened rows allocated 335664 / 149472 / 10816 B
  before the specialization fix, 3056 / 2816 / 288 B after it, and now 0 B.

- **Faster: `ka_upload` / `on_device` of an integer image converts it on the device** (the GPU-first rule): a `UInt8`,
  `UInt16`, `N0f8` or `N0f16` image (any integer or fixed-point eltype narrower than the float target) crosses as it is
  and becomes the target float type through one KernelAbstractions kernel on every backend, the CPU's included, bit for
  bit the host's `T.(A)`; no host-converted copy. A 50 Mpx 8-bit image to Float32 on a B200: 113 ms → 3.2 ms (median
  of 7; 87 ms of the old time was the host's conversion, and it moved 4 B/px instead of 1).

- **Faster: `ka_upload` / `on_device` of a reinterpreted or reshaped dense host image uploads its memory as it is.** A
  host wrapper that is a dense array's memory in order (BlobBoards' `reinterpret(N0f8, ::Matrix{Gray{N0f8}})`, a
  reshape of a reinterpret, a contiguous view) goes to the device by one `copyto!` from an `Array` over that memory,
  with no host copy, and then through the one conversion kernel; a strided view or an adjoint is still materialized.
  Bit for bit the host's `T.(A)`. A 50 Mpx 8-bit Gray{N0f8} image to Float32 on a B200: 46 → 5.1 ms (median; min
  35 → 3.0 ms; the dense UInt8 array: 3.8 ms), host allocation 50 MB → 2.3 KB. An empty integer image no longer
  launches the conversion over an empty range (a `DivideError` on CUDA).

- **New: the exact grid selection's kernels, `exact_pm_score_kernel!` and `argmax_models_kernel!`** (public, not
  exported; robust/pool_grid_kernels.jl), moved from RANSACScoringDoneRight (the twin audit's V1): per (pair, model) the
  sum of a family's per-residual quality at every width of a grid, then per (pair, width) the first strictly best model,
  one KernelAbstractions kernel each for every backend. `select_on_grid!` on the `:pool` staging runs them on the CPU
  backend for a scoring with `has_residual_quality` (RANSAC and MSAC here; RANSACScoringDoneRight's loss-width
  families add theirs) on a loss-width axis, on `sorted_squared_residuals(pair, R)` (the squared residuals with the
  minimal sample at zero, sorted: the staging of both); `support_sq_loss_width(scoring, max_loss_width)` is where a
  compact family's quality ends (both in `CONTRACT_GENERICS`: another package's family adds its methods). `best_pool_model`
  stays the selection of a scoring without a per-residual quality.
  The selections are `best_pool_model`'s (a test checks every width).

- **New: `RadialVignetting(…; chart = :projective)`, a radial profile on the projective chart** of its distance from the
  centre, `ρ = d / (radius + d)` (½ at the radius, → 1 at infinity), so no sensor point is beyond the knots whatever the
  centre; its knots are `projective_knots(n, degree)` (uniform in the distance up to the radius, one tail span beyond).
  Ported from RadiometricCamera's `ProjectiveRadialVignetting` (254a3a3b) with its math, parameter order and defaults,
  values and Jacobians bit for bit (an equivalence test against RC's outputs, recorded once with their generating
  script, test/data/radial_projective_rc/). With it, public: `RADIAL_CHARTS`, `projective_chart(r)`,
  `projective_knots(n, p)`, `radial_chart(v)`, `with_center(v, center)` (the profile about another centre, chart kept)
  and `center_gradient(v, u)` (`∂V/∂c`, isbits). The knots form `RadialVignetting(knots, center, radius, P; degree,
  chart)` rebuilds a recorded profile. `RadialVignetting` gains a fifth type parameter, the chart (`:linear` or
  `:projective`); the four-parameter field constructor `RadialVignetting{T, N, K, DP}(center, radius, knots, P)` is the
  linear chart's. RadiometricCamera deletes its own type.

- **New in `check_duplication`: backend branches and CPU/GPU twins** (the owner's policy: a GPU branch's CPU branch is
  the SAME KernelAbstractions kernel on the CPU backend). `:backend_branch` flags a definition that tests the
  backend to run other code (`backend isa CPU`/`GPU`, `get_backend(x) isa …`, `x isa CuArray` and the other GPU
  array types, `== CPU()`) and a pair of methods on a CPU and a GPU type (`::CPU` beside `::GPU`, `::Backend`,
  `::CuArray`); the same kernel launched on whatever backend, and the transfer, allocation and device-query helpers
  (by their names), are not flagged. `:twin_name` flags two bindings whose names differ only by a CPU/GPU marker
  (`_cpu`/`_gpu`, `_host`/`_device`, `cpu_`/`gpu_`, `_cuda` beside the plain name). Exceptions are DUPLICATION.toml
  `[[backend]]` (file, function, reason) and `[[twin]]` (names, reason) entries. Breaking for a package's quality
  gate: this monorepo's own hits are open items (Core's `_foreach_row!` and pointwise policy, Features' ASIFT
  `_scheduled` pair and `_chunk_warps`).
  A `[[backend]]` entry is stale only in a run that scanned its file (a package's own run reads the monorepo's
  DUPLICATION.toml, whose entries about another package's files it cannot match). The rules' syntax-tree signature
  test is `check_duplication`'s own (its first form replaced `check_naming`'s `_is_signature`, which every package's
  naming gate then failed on).

- **`free_names` is a contract generic** (`CONTRACT_GENERICS`): a calibrator's own lens target (CaliBlobs'
  `LensTarget`) adds its method to Core's function instead of a namesake.

- **New: `required_key(d, key, where)` and `is_finite_real(x)`** (public, not exported; base/config_toml.jl beside
  `check_keys`): a key a record or a config must hold, refused by name (an `ArgumentError` naming `where` and the
  key, never a `KeyError`), and a number a record may hold (a finite `Real`, not a `Bool`, finite as a Float64).
  VisualGeometryToolkitIO's external-calibration readers and validator and CaliBlobs' session record restated them
  (`_need`, `_isreal`, `_isnum`); IO calls these.

- **New: the README's examples, run as a reader runs them** (public, not exported; the Test extension):
  `test_readme_example(root; index, install_pages, developed, checks)` — the README's one example, its install line
  adding exactly what it loads, run expression by expression (each `# true` a claim) in an environment built from
  that line — and the install-row toolkit `ReadmeExample`, `readme_examples`, `install_table`, `install_packages`,
  `loaded_packages`, `readme_environment`, `run_readme_child`, `test_readme_row`. Every stack package's readme test
  restated them (this monorepo's six, RANSACScoringDoneRight's, LocalOptimizationDoneRight's, PanoramaStitching's;
  BlobBoards' and CaliBlobs' install-row runners); the monorepo's readme tests call them.

- **Breaking: a SplineLens lattice's image size is a `Size2` and its calibrated area a `Rect`, one spelling each.**
  `CameraModel(SplineLens, source | controls, size)`, `spline_lattice`, `has_spline_gauge` and
  `NoncentralSplineModel(source, size)` take `size::Size2` (a `(width, height)` tuple is a `MethodError`) and
  `calib_area::Rect` of whole pixels (default `Rect(size)`; the inclusive 4-tuple `(min_x, min_y, max_x, max_y)` is
  gone: it is `Rect(min_x, min_y, max_x − min_x + 1, max_y − min_y + 1)`). The normalizers of the second spellings
  and the default repeated in each signature are gone. VisualGeometryToolkitIO's readers build the `Rect` from a
  record's inclusive bounds.
- **Breaking: `ControlPointChart`, `AdditivePoints` and `RayNormalPoints` are removed**, and `ControlPatchManifold`
  loses its point-chart type parameter and field (`ControlPatchManifold{PX, PY, L, T, P}`; the three-argument
  constructor is gone). The ray-normal chart had no caller but its tests; a point control moves additively,
  `p + (Δx, Δy, Δz)`, as before by default. Downstream port: RadiometricCamera's `_NoncentralPatch` alias
  (src/calibration_pipeline/grid_tiles.jl, also in the rc-abc and rc-flat checkouts) drops its
  `VgtkCore.AdditivePoints` parameter.
- **New: `has_spline_gauge(controls, size; calib_area, present)`** (public): whether a SplineLens pixel lattice
  gives its own gauge K (its 4 × 4 window at the gauge's pixel complete), the condition under which the controls
  constructor refuses to build without `intrinsics`. VisualGeometryToolkitIO's old-grid reader asks it instead of
  matching the refusal's text.
- **Changed: the batched `backproject(camera, pixels; device)` runs on a KernelAbstractions backend for every
  `CameraModel`**, with the new batched `backproject_with_jacobian(camera, pixels; device)` beside it (and a
  `NoncentralSplineModel`'s, its central model's): one `pointwise!` over a device copy of the model, each ray an
  `SVector{3}` (was a host map returning `Vec3`s), back in the pixels' array kind, bit for bit the scalar map on the
  CPU. `device` defaults to `default_device()`; a model with no device form runs as a host map. A SplineLens camera
  keeps its own kernel.
- **New: `device_zeros(T, device, dims...)`** (public, not exported; base/device.jl beside `ka_upload`): a zero array of
  element type `T` on a KernelAbstractions backend, the stack's one spelling of it (KernelAbstractions' own `zeros` is
  not public, so ExplicitImports refuses it across a package boundary). PanoramaStitching's vignetting fit wrote it as
  `_device_zeros`; this package's cgls, B-spline evaluation and ChainRulesCore extension, and
  VisualGeometryToolkitFeatures' spread selection call it instead of `KernelAbstractions.zeros`.

- **Tests: the LO kernel benches' scene is the bench record's `essential_bench_scene(rng; n = 1000) -> (; u₁, u₂, K, R,
  t)`** (test/lo_timing/bench_record.jl, which every timing bench includes from `pkgdir(VisualGeometryToolkitCore)`):
  the seeded 1000-point pinhole E scene with 30 % outliers that this package's lo_timing bench and
  LocalOptimizationDoneRight's each built as `scene()`. Its draws are the old ones, so the baselines hold.

- **New: `test_recorded(make; nested = RECORDED_NESTED_STATES)`** (public, not exported; the Test extension): the
  stack's one check that a `Recorded` run of `block_coordinate_descent!` is the bare run — the same estimate, retcode,
  iteration count and final state, one recorded iterate per good iteration, and the bare run truncated after every
  `k` at the recorded `k`-th cost. `make()` returns a fresh `(state, likelihood, method, problem, θ₀, max_iter)`.
  VisualGeometryToolkitEstimation's and LocalOptimizationDoneRight's recorded-BCD tests each carried it as
  `check_recorded` (with its `snap` helpers); both call this one.

- **New: the stack's git hooks, `sync_manifest`, `install_githooks`, `githooks_dir`** (public, not exported;
  `githooks/` and src/quality/sync_manifest.jl). `githooks/sync-manifest` (with `post-checkout`, `post-merge` and
  `post-rewrite` linking to it) re-resolves a repository's untracked root Manifest when a checkout, merge or rebase
  changes a Project.toml at any depth — `Pkg.update()`, or `Pkg.instantiate()` for a tracked Manifest — through
  `sync_manifest(mode = "update"; io, repo) -> (; created, moved, seconds)`, which reports the direct dependencies
  that moved. The hook includes the dependency-free sync_manifest.jl by path instead of loading the package (its
  Manifest may be the illegal one). `install_githooks(repo)` points a downstream repository's `core.hooksPath` at
  this copy, and each re-resolve moves it to the newly resolved VisualGeometryToolkitCore's. One copy for the stack:
  the monorepo's `.githooks` is a link to `VisualGeometryToolkitCore/githooks`, and RANSACScoringDoneRight's and
  BlobBoards' byte-identical `.githooks/` (sync-manifest.jl and the shell hooks) are deleted for it.
- **Breaking: the SplineLens source constructor loses `unit_controls`** (`CameraModel(SplineLens, source, size)`,
  `NoncentralSplineModel(source, size)`): no caller supplied it, and its default followed `init`. The prefilter's
  controls are re-normalized and the quasi-interpolant's kept, as before. The controls constructor keeps its
  `unit_controls` (a reader's choice).
- **Breaking: `init = :sample` is removed** from `CameraModel(SplineLens, source, size)` (and so from
  `NoncentralSplineModel(source, size)`): the node samples taken as controls (Schoenberg's approximation) had one
  test caller and no other in the stack or downstream. `init` is `:interpolate` (the cubic prefilter, default) or
  `:quasi_interpolate`; anything else is an `ArgumentError` naming the two.
- **Changed: a SplineLens lattice's absent nodes are read from its `present` mask alone.** The gauge
  (`CameraModel(SplineLens, controls, size; present)`) and the off-axis check of `angle_of_view`, `image_circle`,
  `undistort`, `valid_rect`, `view_overlap` and `horizon_radius` read the mask; they read NaN controls before, so a lens
  built by `SplineLens(backward; present)` (which keeps the absent slots' values) succeeded in the axis check with ẑ under a
  hole. The NaN in absent slots that `with_present` and the controls constructor write is a debugging aid only. The
  gauge no longer maps the whole lattice to test one window.
- **Breaking: `SplineLens` has no `chart` field** (and so no fourth type parameter, `SplineLens{B, F, P}`). The
  forward spline's chart is the same constant for every lens, the equidistant chart about ẑ; it is the new public
  `forward_chart(SplineLens)`, which `project` evaluates a forward on and VisualGeometryToolkitEstimation's
  `fit_camera` fits one on (it read Core's private `_forward_chart`). The public constructor checks the presence mask
  once (it did twice).
- **Breaking: `finite_linearization` is removed.** NLLSsolver 5.3.3 (#68, the floor) stops on a non-finite step or
  gradient at an accepted point itself; no block wraps its linearization. Its termination bit 12 is decoded
  (`nlls_stop_reasons`: `:nonfinite_gradient`; `nlls_stop_code`: a failed solve).
- **New: `nlls_rows_and_jacobian(blocks, variables)`** (public): the rows of NLLSsolver residual blocks and their sparse
  Jacobian in the variables' chart coordinates, the stall postcondition's input for every NLLSsolver fit of the stack.
- **Breaking: the run-time-index method `restrict(ch, idx::Vector{Int})` is removed**, together with its
  `SubsetManifold{C, D, Vector{Int}}` and its run-time `NLLSsolver.nvars`/`update`. It was added earlier in this
  release for the SplineLens refine's patches, and that refine now restricts statically (NLLSsolver assembles its
  system from statically sized blocks), so nothing in the stack calls it.
  - `SubsetManifold{C, D, I}`'s index is an `SVector{M, Int}`.
  - `restrict` itself stays, with its mask, static-index, name, `()` and subspace methods.
- **Breaking: `spline_weights(cs, x)` loses its held-window argument** (no caller in the stack; a refinement holds its
  window by solving on the window's own cell). The projection Jacobians form the window's basis weights once.
- **New: `device_free_bytes(dev)`, `reclaim_device!(dev)`, `device_names()`** (public): the accelerator's free memory,
  its pool handed back to the device and its devices' names, answered by Core's CUDA extension (assigned at load, as the
  device resolver is; CUDA 5's `CUDA` or 6's `CUDACore`). VisualGeometryToolkitFeatures' chunk sizing and plan release and
  VisualGeometryToolkitEval's GPU record read them (each had its own name-sniffing lookup of the CUDA module).
- **Internal: one implementation each.** A rectangle's centre is `centroid` (`_rect_centre` retired), the lens
  coverage's rectangle test is `_in_rect`, a rectangle mapped through K or K⁻¹ is one `_mapped_rect`, a 2 × 2
  determinant and identity are StaticArrays' `det` and `one`, and a lattice's node counts at a spacing are one
  `_node_counts` (the reference's 64 px grid and VisualGeometryToolkitEstimation's spacing search). Values unchanged.
- **Changed: the batched SplineLens `project`** takes a non-finite seed as the scalar `project`'s own start (the pinhole
  point where the lens images it, else the area's centre; was the area's centre). The same kernel serves a refine's
  normalized projections (with the lattice's fold measure).

- **Changed: `pool_frame_calibrations` moves K through `reframed_calibration`**, the one change of image frame of a
  calibration, instead of multiplying by the stored map by hand. The values are the same bit for bit; a stored map
  that is not an isotropic scale and a translation (the pool frame always is) is now an `ArgumentError`.
- **Changed: `spectral_rotation_averaging` no longer checks that its pairs connect every frame.** The caller checks it on
  its graph — VisualGeometryToolkitEstimation's `fit_rotation_averaging`, its only stack caller, already refuses a split
  `ViewGraph` before seeding — so the hand-written breadth-first search that ran the check a second time is deleted. A
  direct call on a split graph (two components, or a zero-weight bridge) now returns meaningless rotations instead of
  an `ArgumentError`; check connectivity first.
- **Internal: one implementation each, found by the duplication gate.** A physical camera's and an oriented camera's
  refusals of the plane-coordinate verbs (`HomographyMat(camera)`, `transfer_ellipse`, `project` of a conic or a
  rectangle) are one set of methods with each kind's reason; a lens leaf's and a pointwise functor's field tuples share
  one generated body. The frozen -O2 reproducer test/fixtures/repro_o2_miscompile.jl (a copy of the spline LM step, run
  by no test) is deleted; it is kept at commit a1da853f.
- **New: `test_duplication(M; kwargs...)` and `thrown_message(f)`** (public, not exported; Test extension).
  `test_duplication` is `check_duplication` as a test, as `test_naming` is `check_naming`'s; every monorepo package's
  quality item runs it, and the monorepo's few genuine exceptions are in the root `DUPLICATION.toml`. `thrown_message`
  is the message of the `ArgumentError` a call throws (`""` when none; with `shown = true`, any exception's shown text),
  which a dozen test files across the monorepo had each written for themselves. The packages whose naming item restated `test_naming`'s body call it.
- **New: `map_frames(X) -> (Src, Dst)`** (public, not exported): the source and destination frame types of a labeled
  `EuclideanMap`, `RotationMap` or `HomographyMap`, plain or `Uncertain` — the labels `relabel` sets. The one reader of
  them: Core's own `_em_frames`, `_rotation_frames`, `_homography_frames` and `_pose_frames`, Estimation's and IO's copies
  are deleted.
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
- **New: `image_warp!(dst, warps::Tuple, src)`, a multi-plane image warped plane by plane, each plane through its own
  warp** (`dst`, `src` `H × W × P`, one `BoundedMap` per plane; the 2-D `image_warp!` of each plane, on the backend
  that owns `dst`). For an image whose planes are sampled at different points — a Bayer image's R, G and B — each warp
  composes its plane's frame map, `image_warp(T, inv(plane_map) ∘ math, rect)` (VisualGeometryToolkitFeatures'
  `bayer_plane_map`). `plane_offsets` and `raw_scale` join the retired names.
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
- **Breaking: `ThinPlateSplineSystem` has no `affine` field.** It held `coeff`'s affine rows a second time; read them
  as `coeff[N+1:N+3, :]` (the pure-bending rule and the target calibration's affine-row basis do).
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
