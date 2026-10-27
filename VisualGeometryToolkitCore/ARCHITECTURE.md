# VisualGeometryToolkitCore Architecture

## Module Dependency Graph

```
                    VisualGeometryToolkitCore (main module)
                              │
     ┌────────────┬───────────┼────────────┬─────────────┐
     │            │           │            │             │
     ▼            ▼           ▼            ▼             ▼
 base/        robust/    transforms   ┌──────────┐  ┌──────────┐
 (types,      (losses,   (matrices,   │Primitives│  │ Cameras  │
  frames,      BCD loop,  maps, warp, └──────────┘  └──────────┘
  units,       LM core)   batched)         ▲              │
  FitResult)                               └──────────────┘
                              │
                              ▼
                    Base types & interface
                    (base/interface.jl, base/types.jl)
```

Submodules: `Primitives`, `Cameras`, the small `Synthetic` evaluation module,
`Pools` (the benchmark pool record) and `SpatialQueries` (allocation-free
R-tree queries); everything else is included directly into the main module in the
order below. Feature detection, scale space and descriptor matching live in
VisualGeometryToolkitFeatures; RANSAC problem types live in VisualGeometryToolkitEstimation.

## Include Order

The include order in `src/VisualGeometryToolkitCore.jl` is dependency-driven. The
authoritative list is the include block in that file (each line carries a
comment saying what it provides); this is the shape of it.

### 1. Base (loaded first)
```julia
include("base/units.jl")         # unit constants px, mm, dpi; px, pd, dpi, pt registered at __init__ — no dependencies
include("base/size2.jl")         # Size2
include("base/feature_coords.jl")   # _strip_coords, the feature-coordinate compute boundary
include("base/frames.jl")        # Frame{N} tag hierarchy (before types.jl: CovMat/PrecMat frame params)
include("base/image_frames.jl")  # 2-D image frames (pixel/image/gnomonic/retinal/distorted/target-plane) + FrameDependence
include("base/types.jl")         # interface.jl + interface_functions.jl, AbstractDetection, conic matrices
include("base/cspond_set.jl")    # Csponds columnar container + sources/targets protocol
include("base/coordinates.jl")   # to_affine / to_euclidean
include("base/fit_result.jl")    # FitResult{P,M} — the one iterative-fit return type
include("base/utils.jl")         # geometry utilities
include("base/config_toml.jl")   # TOML config skeleton shared by every package with a configuration of record
include("base/device.jl")        # default_device — the family's one KA backend resolver (CUDA ext registers)
include("base/svd_workspace.jl") # SvdWorkspace + svd_nullvec!
include("base/propagate.jl")     # propagate(J, Σ) + whitening policies
include("base/manifold.jl")      # the Manifold contract (retract + transform_covariance); instances in geometry/manifolds/
```

**Why first?** These define the foundational types (`AbstractDetection`,
`HomEllipseMat`, `Size2`, `Uncertain`, `FitResult`) and the interface stubs
every later file extends.

### 2. Robust-estimation core
```julia
include("robust/losses.jl")         # AbstractLossShape family (LOSSES.md)
include("robust/width_axes.jl")     # WidthAxis tags, width_axis ⇄ width_axis_name, to_native(::LossWidth)
include("robust/estimation.jl")     # RANSAC/LO interface hierarchy (types + contracts)
include("robust/block_coordinate_descent.jl") # block_coordinate_descent! + extension verbs
include("robust/rows.jl")           # the chart-rows contract (rows_provider, lift_* verbs), RowsWorkspace, task_buffers
include("robust/lm_core.jl")        # the chart LM: NLLSsolver on the row protocol (ManifoldLmWorkspace)
include("robust/problem.jl")          # the constraint-problem protocol (residuals!, refine!)
include("robust/scale.jl")            # IGPrior, fit_scale, the consensus Student-t scale
include("robust/mixture.jl")          # StudentTMixture, GauInlier, decision_threshold
include("robust/posterior.jl")        # PosteriorWeightState, the nuisance policies, mixture_ecm!
include("robust/normaliser.jl")       # truncated_log_normaliser
include("robust/scoring_contract.jl") # the scoring contract, ScoreBuffers, score_model!
include("robust/baseline_scorers.jl") # RansacScoring, MsacScoring
include("robust/lo_contract.jl")      # the local-optimization and external-refit contracts, loss_at
```

**Why here?** Contracts, the robust-statistics toolkit and the baseline
scorers — no geometry. VisualGeometryToolkitEstimation implements the problem protocol and
runs its one RANSAC loop over these verbs; RANSACScoringDoneRight implements
the scoring contract; LocalOptimizationDoneRight implements the
local-optimization contract and owns the lifts and continuation schedules. The
pool-selection verbs (`robust/pool_selection.jl`: `best_pool_model`,
`select_on_grid!`) are included after the `Pools` submodule they read.

### 3. Transforms (matrix types and rigid maps)
```julia
include("geometry/transforms/matrices.jl")       # Transform matrix types + AbstractHomMat hierarchy
include("geometry/transforms/euclidean_map.jl")  # EuclideanMap for poses
include("geometry/transforms/lie.jl")            # SO(3)/SE(3) exp/log
include("geometry/transforms/rotation_fits.jl")  # Kabsch/Wahba, chordal mean
```

**Why before Primitives?** `matrices.jl` defines the `AbstractHomMat` type
hierarchy; `EuclideanMap` is used by Primitives (3D conics) and Cameras.

### 4. Primitives submodule
```julia
include("geometry/primitives/primitives.jl")
using .Primitives
```

**Why here?** Primitives uses the conic matrix types from step 1 and
`EuclideanMap` from step 3; the remaining transforms need `Ellipse` from it.

### 5. Remaining transforms
```julia
include("geometry/transforms/conversions.jl")      # to_homogeneous, AffineMap(::StaticMatrix)
include("geometry/transforms/ct_contravariant.jl") # CT maps callable on Circle/Ellipse/Line2D
include("geometry/transforms/coord_maps.jl")       # canonical-frame AffineMap constructors
include("geometry/transforms/region_maps.jl")      # canonical_map, logpolar_map
include("geometry/transforms/homography_map.jl")   # HomographyMap (points/lines/conics)
include("geometry/transforms/propagate_framed.jl") # frame-labeled propagate over framed maps
include("geometry/transforms/bounded_map.jl")      # BoundedMap (output extent decorator)
```

### 6. Solvers and the solver layer
```julia
include("geometry/solvers/solvers.jl")               # forward declarations for backend-provided solvers
include("geometry/solvers/p3p_native.jl")            # native P3P (PoseLib port)
include("geometry/solvers/essential_5pt_native.jl")  # native 5-pt essential
include("geometry/solvers/homography_solvers.jl")    # homography_4pt, DLT, Hartley frames
include("geometry/solvers/homography_jacobian.jl")   # analytic Jacobian of homography_4pt
include("geometry/solvers/fundmat.jl")               # F/E matrices, 7-pt, motion()
include("geometry/solvers/tls_line.jl")              # the total-least-squares line (moved up from RVG, taxonomy v2)
include("geometry/radial_models.jl")                 # AbstractRadialModel — the plane→pencil family (BEFORE the 5-pt radial solver)
include("geometry/solvers/homography_5pt_radial.jl") # 5-pt one-sided radial homography
include("geometry/forms.jl")                 # slot-labeled epipolar forms; frame changes are ∘ (EpipolarMap, TransferForm ↔ HomographyMap)
include("geometry/manifolds/subset.jl")      # SubsetManifold — restrict() (before the instances, which extend it)
include("geometry/manifolds/essential.jl")   # EssentialManifold5/6
include("geometry/manifolds/fundmat.jl")     # FundMatManifold7
include("geometry/manifolds/homography.jl")  # HomographyManifold8 + sl(3) basis
include("geometry/manifolds/rigid.jl")       # RigidManifold6, LeftRigidManifold6, RigidExpManifold6
include("geometry/manifolds/sphere.jl")      # SphereManifold (a model up to scale: RadialCamera)
include("geometry/manifolds/product.jl")     # EuclideanManifold (the ambient chart), ProductManifold
include("geometry/manifolds/line.jl")        # Line2DManifold
include("geometry/manifolds/covariance.jl")  # implicit_covariance: the estimate's covariance (chart, blocks)
include("geometry/confidence_regions.jl")    # confidence ellipses, epipolar bands
include("geometry/model_validity.jl")        # test_sample/test_model/test_consensus
include("geometry/solvers/solver_layer.jl")          # the MinimalSolver protocol + registry (the normalizations: homography_solvers.jl)
```

The three covariance verbs (GTSAM's vocabulary in parentheses): `propagate`
through a map (a function's Jacobian, `H·Σ·Hᵀ`), `transform_covariance`
between charts of one model — the ambient space is the chart
`EuclideanManifold(ch)` (`TransformCovariance`) — and `implicit_covariance`
of an estimate: a minimal solve, or the whitened least squares on n ≥ minimal
(`Marginals`).


### 7. Warping, pointwise maps, numerics
```julia
include("geometry/transforms/warp.jl")           # image_warp adapter + KernelAbstractions warp kernels
include("geometry/transforms/pointwise.jl")      # pointwise!: one broadcast driver, every backend
include("numerics/bspline.jl")                   # B-spline surfaces (grid camera models)
include("numerics/special.jl")                   # device-safe lgamma/digamma/trigamma
include("numerics/small_lu.jl")                  # in-register partial-pivot LU for fixed-size systems
include("numerics/ka_reduce.jl")                 # KernelAbstractions workgroup tree reduction
include("numerics/thin_plate_spline.jl")         # thin-plate spline: host-side solver, isbits evaluator
include("base/raster.jl")                        # array plotting without Makie (masks, debug frames)
include("geometry/primitives/bbox.jl")           # shape → SpatialIndexing.Rect (needs frames + dilate)
include("geometry/spatial_queries.jl")           # SpatialQueries submodule: allocation-free R-tree queries
```

`warp.jl` defines the single adapter `image_warp(math, output_rect)` that wraps
a math-space `Transformation` into a `BoundedMap` callable from Julia
`(row, col)` indices. VGC math is always COLMAP; the array-side conventions
(`_RC_SWAP` + COLMAP↔Julia `Translation(±0.5)`) are baked in with zero kwargs.

### 8. Cameras submodule, Synthetic, Pools, visualization stubs, quality contracts
```julia
include("geometry/cameras/cameras.jl")   # models, lenses, imaging chain, calibration, rigs, stereo
using .Cameras
include("synthetic/synthetic.jl")        # evaluation metrics (pose_error, imc_pose_error, mean_average_accuracy)
include("pools/pools.jl")                # Pools submodule: PoolPair (the pool record, NormalizedImageFrame) + StereoRawPair
include("visualization_stubs.jl")        # `function f end` stubs the Makie/GLMakie extensions fill
include("quality/naming.jl")             # NamingViolation + the check_naming/naming_scope stubs (the linter: ext/VisualGeometryToolkitCoreTestExt, loads with Test)
include("quality/test_helpers.jl")       # stubs: rows_cost, check_rows_jacobian, test_naming (TestExt); test_quality_gates (QualityExt: Test + Aqua + ExplicitImports + JET)
include("quality/tag_registry.jl")       # snake_case, TagRegistry (the one TOML tag → type map) and its source/config scans
include("quality/stack_contracts.jl")    # the dependency allow-list (configs/stack_dependencies.toml), private-name scan, CONTRACT_GENERICS
include("quality/agent_docs.jl")         # agent_doc_violations: what CLAUDE.md, AGENTS.md and .claude/ may not carry
include("quality/conventions.jl")        # conventions_dir, convention_files: the one copy of the stack conventions
```

**Why last?** Cameras uses the solvers (P3P, radial calibration), the batched
maps (lens kernels), and the B-spline numerics (grid models). `check_naming`
depends on nothing in the package; it sits last because nothing in the package
calls it — every stack package's quality test does. `stack_contracts.jl` is
the very last file: `CONTRACT_GENERICS` lists functions from every section above.

## Submodule vs Non-Submodule

### Submodules (separate namespace)
- **Primitives**: self-contained geometric primitives
- **Cameras**: camera models, lenses, projection, calibration, rigs
- **Synthetic**: evaluation metrics on synthetic data
- **Pools**: the benchmark pool record and the raw-pair interface (its
  accessor names stay out of `using VisualGeometryToolkitCore`)

### Non-Submodules (direct includes)
- **Transforms** (`geometry/transforms/*.jl`): NOT a submodule because
  Primitives needs the matrix types (loaded before it) and `HomographyMap`
  needs `Ellipse` (loaded after it).
- **Robust** (`robust/*.jl`), the **solvers** (`geometry/solvers/*.jl`) and the
  rest of `geometry/` outside the submodules (forms, radial models, manifolds,
  confidence regions, model validity): contracts
  and kernels consumed by the Cameras submodule and by downstream packages.

## Interface Dispatch Table

| Function | Defined in | Implemented by |
|----------|------------|----------------|
| `centroid` | `base/interface_functions.jl` | Primitives (Ellipse, Circle, Rect, Disc) |
| `project` | `base/interface_functions.jl` | Cameras (`project(camera, X)`) |
| `distance` | `base/interface_functions.jl` | Primitives (`distance(plane, p)`) |
| `closest_point` | `base/interface_functions.jl` | Primitives (`closest_point(plane, p)`) |
| `dilate` | `base/interface_functions.jl` | Primitives (Circle, Ellipse) |
| `intersects` | `base/interface_functions.jl` | Primitives (Rect, Circle) |
| `normal` | `base/interface_functions.jl` | Primitives (Hyperplane, Ellipse3D) |
| `boundary_point` | `base/interface_functions.jl` | Primitives (Ellipse, Ellipse3D) |
| `AffineMap(target)` | `coord_maps.jl` | Rect, Ellipse, Circle, FeatureFrames — canonical [-1,1]² → target frame |
| `EuclideanMap(PixelFrame{A}, PixelFrame{B})` | `coord_maps.jl` | Image-origin conversion (:opencv/:colmap/:julia) — a labeled, unitless rigid map (`Translation(::Symbol, ::Symbol)` was removed in 0.7) |
| `coeffs` | Primitives (`feature_frames.jl`) | `Paraboloid3D{T,P}` → `SVector{P,T}` |

## Holy Traits

Compile-time dispatch via trait types defined in `base/traits.jl` (loaded by
`base/interface.jl`), plus `FrameDependence` in `base/image_frames.jl`:

| Trait | Values | Used by |
|-------|--------|---------|
| `ConicTrait` | `CircleTrait`, `EllipseTrait`, `ParabolaTrait`, `HyperbolaTrait`, `DegenerateTrait` | `classify_conic` (Primitives) |
| `HasBoundaryTrait` | `HasBoundary`, `NoBoundary` | Primitives (boundary-sampled `overlap_area` fallback) |
| `OverlapMethod` | `AnalyticOverlap`, `BoundarySampled` | `overlap_method` / `overlap_area` (Primitives) |
| `BoundsTrait` | `HasBounds`, `NoBounds` | `extent` on `BoundedCamera`, `BoundedMap` |
| `FrameScaleTrait` | `IsotropicScale`, `ShapeMatrixScale` | `FeatureFrame` accessors (Primitives `feature_frames.jl`) |
| `AbstractSolverBackend` | `NativeSolver`, `PoseLibSolver` | `MinimalSolver` verb layer, positional solver kernels |
| `FrameDependence` | `GeometryDefined`, `ModelDefined` | `frame_transition`, sphere sections (Cameras) |

The transform class (Euclidean ⊂ similarity ⊂ affine ⊂ projective) is not a
trait: the `AbstractHomMat` hierarchy in `matrices.jl` carries it, and
`HomographyMap` dispatches its conic push-forward on the matrix type.

### Trait-based Dispatch Example

```julia
# In Primitives (feature_frames.jl)
# Dispatch on FrameScaleTrait: one accessor, one method per scale kind
characteristic_scale(f::FeatureFrame) = _characteristic_scale(frame_scale(f), f)

_characteristic_scale(::ShapeMatrixScale, f) = sqrt(abs(det(shape_matrix(f))))
_characteristic_scale(::IsotropicScale, f) = f.σ
```

## Image warp

A single function `image_warp(math, output_rect)` wraps a math-space
`Transformation` into a `BoundedMap` callable from Julia `(row, col)` indices.
The composition is FIXED (no kwargs) — VGC math is always COLMAP, the array
boundaries are always Julia, and `image_warp` always bakes the same bridge:

```
_RC_SWAP ∘ Translation(+0.5) ∘ math ∘ Translation(-0.5) ∘ _RC_SWAP
```

Math construction is the caller's responsibility — the two-step pattern is:

```julia
math = canonical_map(geometry, 64; bound=1.0)   # COLMAP-math patch transform
bm   = image_warp(math, 64)                     # array-callable BoundedMap
```

Log-polar patches use the analogous chain via `logpolar_map(geom, n, inner_radius)`.
Any other math chain works the same way — a K-based homography just
becomes `image_warp(H, output_rect)`.

Both `canonical_map` and `logpolar_map` also accept an explicit
`HyperRectangle{2}` as the patch domain (for non-square or non-standard
extents); the size-form is sugar over the rect-form for the common case.

## Key Files Reference

| File | Purpose |
|------|---------|
| `src/VisualGeometryToolkitCore.jl` | Entry point, exports, include ordering |
| `src/base/interface.jl` | Loads `traits.jl` + `interface_functions.jl`; camera/detection interface stubs |
| `src/base/traits.jl` | Holy Traits (`ConicTrait`, `HasBoundaryTrait`, `OverlapMethod`, `BoundsTrait`, `FrameScaleTrait`, `AbstractSolverBackend`) |
| `src/base/types.jl` | Loads interface.jl, defines `AbstractDetection`, `Uncertain`, conic matrices |
| `src/base/units.jl` | unit constants, the stack's four units registered into FlexUnits' registry, `parse_unit`/`unit_string` |
| `src/base/size2.jl` | Size2 type |
| `src/base/feature_coords.jl` | `_strip_coords`: pixels in px, lengths in mm |
| `src/geometry/transforms/matrices.jl` | Transform matrices + `AbstractHomMat` hierarchy (composition promotion) |
| `src/geometry/transforms/euclidean_map.jl` | `EuclideanMap` for camera poses |
| `src/geometry/transforms/homography_map.jl` | `HomographyMap` for point/line/conic transforms |
| `src/geometry/transforms/warp.jl` | `image_warp` adapter + `image_warp_batch_to!` batched primitive |
| `src/base/cspond_set.jl` | Correspondence containers (`Csponds`, `Cspond`, `Attributed`) |
| `src/base/fit_result.jl` | `FitResult{P,M}`, the iterative-fit return type |
| `src/geometry/primitives/primitives.jl` | Primitives submodule entry |
| `src/robust/losses.jl` | Robust-loss family (`AbstractLossShape`) |
| `src/geometry/cameras/cameras.jl` | Cameras submodule entry |

## Extension Points

### Adding a New Lens Model

1. Most models are a `Lens(projection, distortion)` of existing parts or a
   `LiftLens(radial)` — no new type (see the Lens Families guide).

2. A new projection or distortion term is a LEAF (`cameras/lenses.jl`): a
   struct of numbers and tuples of numbers with its number type as the last
   type parameter — its coefficients, their names, its zero, its record tag and
   payload all follow from the fields (add it to `_LEAVES` in
   `lens_chart.jl` to read it back) — and its map: a term's `_distort`
   (or `_delta` with `isdelta`), a projection's `_project_ray` / `_lift_point`. A lens of a new KIND implements the
   contract on rays directly:
   ```julia
   project(lens::MyLens, X::StaticVector{3}) -> Vec2       # ray → image point
   backproject(lens::MyLens, xd::StaticVector{2}) -> Vec3  # image point → unit ray
   ```
   NaN where the lens does not image, generic in the number type.

3. Everything else is derived, with no per-lens methods: the perspective views
   (`distort_normalized`, `undistort_normalized`, the functor, `InverseLens`),
   `transform_deriv` / `transform_deriv_params` / `project_with_jacobians`,
   `projected_radius` / `view_angle`, `unproject`, `imaging_transform`
   (`AffineMap(K) ∘ lens`), the batches (`pointwise!`) and warps, `RayMap`,
   the framed stages, and — from the leaves — `parameter_vector` /
   `with_parameters` / names / bounds (a leaf with a bounded
   coefficient adds its `_bound_pairs` method), the record, the manifold and `_precision`.

### Adding a New Primitive

1. Define struct in `primitives/` directory
2. Implement the accessor interface (`centroid`, `area`, `boundary_point`)
   and `has_boundary` (or `frame_scale` for a feature frame) where applicable
   — traits only when a dispatch site consumes them
3. Implement interface functions: `centroid`, `boundary_point`, etc.
4. Add to exports in `primitives.jl`

### Adding a New Transform Matrix Type

1. Use the two-argument `@smatrix_wrapper` form (macro in `base/types.jl`) in
   `matrices.jl`, parenting the type at its tightest abstract level:
   ```julia
   @smatrix_wrapper MyTransformMat AbstractAffineMat   # 3×3: MyTransformMat{T} <: AbstractAffineMat{T}
   ```

2. Add it to the same-type `Base.promote_rule` loop in `matrices.jl`. There is
   no trait to declare: composition result types and `HomographyMap`'s conic
   push-forward dispatch on the abstract hierarchy.
