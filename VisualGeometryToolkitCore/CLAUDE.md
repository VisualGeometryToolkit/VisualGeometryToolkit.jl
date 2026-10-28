# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Build & Development Commands

```bash
# Once per clone: let git regenerate the Manifest whenever a pull, checkout or
# rebase moves Project.toml (compat bumps from a release cut elsewhere).
# Without this every such pull leaves the Manifest illegal. See .githooks/.
git config core.hooksPath .githooks

# Run all tests — ReTestItems: every @testitem in test/*_tests.jl, in parallel
# worker processes (RETESTITEMS_NWORKERS overrides the count in
# test/runtests.jl; 0 runs serially in-process)
julia --project=. -e 'using Pkg; Pkg.test()'

# Run only the items whose name matches a regex (an item is named after its
# file: transforms_tests.jl → "transforms"; the quality gates are "quality: *")
julia --project=. -e 'using Pkg; Pkg.test(; test_args = ["transforms|p3p"])'

# Strict run: every upstream deprecation becomes a test failure (workers
# inherit the flag). Green on the full suite as of 2026-09-14 — run before
# each release.
julia --project=. -e 'using Pkg; Pkg.test(; julia_args = ["--depwarn=error"])'

# Run tests interactively with Revise (ReTestItems and the item's test deps
# must be loadable, e.g. from the global environment)
julia --project=.
julia> using Revise, ReTestItems, VisualGeometryToolkitCore
julia> runtests("test/p3p_native_tests.jl")             # one file
julia> runtests(VisualGeometryToolkitCore; name = r"^lens")    # items by name

# Build documentation. --startup-file=no is REQUIRED, not hygiene: a
# ~/.julia/config/startup.jl that calls Pkg.activate runs after --project is
# applied and overrides it, so the build silently uses the ROOT environment
# (Pkg.project().path then prints Project.toml, not docs/Project.toml) and
# fails with "Package Documenter not found in current path". Pkg.instantiate
# inherits the same hijack and reports success for the wrong environment, so
# the obvious next step confirms a false diagnosis. Pkg.test is unaffected:
# ReTestItems' workers already launch with --startup-file=no.
julia --project=docs/ --startup-file=no docs/make.jl

# Start Julia with GLMakie for visualization
julia --project=. -e 'using GLMakie; using VisualGeometryToolkitCore'
```

## Architecture

VisualGeometryToolkitCore provides computational geometry, camera modeling, and the robust-estimation core for the stack (VisualGeometryToolkitFeatures, VisualGeometryToolkitEstimation, BlobBoards). Feature detection lives in VisualGeometryToolkitFeatures.

It owns the CONTRACTS of the composable RANSAC (`src/robust/`): the constraint-problem protocol (`problem.jl`), the scoring contract (`scoring_contract.jl`, `estimation.jl`) with the `RansacScoring`/`MsacScoring` baselines (`baseline_scorers.jl`), the local-optimization contract — `lo_workspace(::Type{P}, ::Type{T}, lo, n)`, `locally_optimize!`, the width modes resolved by `loss_at`, `RefitFamily` = `NativeRefit` | `ExternalRefit` and the external refit `external_workspace`/`prepare_external!`/`refine_external!` (`lo_contract.jl`) — the chart-rows contract (`rows.jl`), pool selection (`pool_selection.jl`: `best_pool_model`, `select_on_grid!`), the width axes (`width_axes.jl`) and the robust-statistics toolkit (`scale.jl`, `mixture.jl`, `posterior.jl`). VisualGeometryToolkitEstimation runs its one RANSAC loop over these verbs; RANSACScoringDoneRight implements the scoring and pool-selection contracts and LocalOptimizationDoneRight the local-optimization contract, neither depending on the other. `TagRegistry` (`src/quality/tag_registry.jl`) is the one TOML tag → type map, and `CONTRACT_GENERICS` with the dependency allow-list (`src/quality/stack_contracts.jl`, `configs/stack_dependencies.toml`) are what the quality gates enforce across packages.

**See [ARCHITECTURE.md](ARCHITECTURE.md)** for detailed module dependency graph, include order rationale, and extension points, and [LOSSES.md](LOSSES.md) for the robust-loss family (`AbstractLossShape`: a normalized scale family whose scale parameter is `loss_width`, composed on by `deploy(shape, s2; scale_multiple)`) shared with VisualGeometryToolkitEstimation and RANSACScoringDoneRight.

### Module Structure

**Main Module** with four submodules:

1. **Primitives** (`src/geometry/primitives/`) - Geometric primitives: `Ellipse`, `Circle`, `Rect`, `Disc`, `Hyperplane`, `Line2D`, `Plane3D`, `Ellipse3D`, `Paraboloid3D{T,P}`
2. **Cameras** (`src/geometry/cameras/`) - Camera models with pose, sensors (`CMOS_SENSORS`), intrinsics, lens and projection models, calibration, rigs, stereo
3. **Synthetic** (`src/synthetic/`) - Evaluation metrics and synthetic-data generators
4. **Pools** (`src/pools/`) - The benchmark pool record `PoolPair` (stored in `NormalizedImageFrame`) and the raw-pair interface `StereoRawPair`; not `using`'d into the main module

The minimal-solver kernels (P3P, homography, fundamental/essential) are not in
Cameras: they live in `src/geometry/solvers/`, direct includes of the main
module. Cameras holds only camera-level wrappers over them (`p3p(model, csponds)`,
`sample_p3p`) and the radial calibration solver `radial_camera_5pt`.

**Transforms** (`src/geometry/transforms/`) are NOT a submodule due to dependency ordering:
- `matrices.jl` → defines transform matrix types and their abstract hierarchy (`AbstractHomMat` ⊃ … ⊃ `AbstractEuclideanMat`)
- `euclidean_map.jl` → defines `EuclideanMap` for camera poses
- Primitives submodule → uses conic matrix types
- `coord_maps.jl`, `region_maps.jl` → use Ellipse from Primitives

### Include Order

The include order in `src/VisualGeometryToolkitCore.jl` is managed via early-loaded
types: base (units, frames, charts, types, utilities) → the robust-estimation
core → matrix transforms and `EuclideanMap` → the Primitives submodule → the
remaining transforms (which need `Ellipse`) → solvers, forms and manifolds →
warping, batched maps and numerics → the Cameras submodule → Synthetic.
The include block in `src/VisualGeometryToolkitCore.jl` is authoritative;
ARCHITECTURE.md mirrors it file by file with the reason for each section.

### Holy Traits System

Compile-time dispatch via traits defined in `src/base/traits.jl` (loaded by
`src/base/interface.jl`; `FrameDependence` lives in `src/base/image_frames.jl`):

| Trait | Purpose |
|-------|---------|
| `ConicTrait` (`CircleTrait`, `EllipseTrait`, …) | Conic classification (`classify_conic`) |
| `HasBoundaryTrait` | Boundary-sampling fallback |
| `OverlapMethod` (`AnalyticOverlap`, `BoundarySampled`) | `overlap_area` method selection |
| `BoundsTrait` | `BoundedCamera` extent dispatch |
| `FrameScaleTrait` (`IsotropicScale`, `ShapeMatrixScale`) | Feature-frame scale dispatch |
| `AbstractSolverBackend` (`NativeSolver`, `PoseLibSolver`) | Minimal-solver backend |
| `FrameDependence` (`GeometryDefined`, `ModelDefined`) | Frame-tag commensurability (`frame_transition`) |

The transform class (Euclidean ⊂ similarity ⊂ affine ⊂ projective) is not a
trait: the abstract matrix hierarchy in `matrices.jl` carries it.

Traits are added ONLY when a dispatch site consumes them — the 2026-07 audit
deleted ten trait families that had implementations but zero consumers.

### Makie Extension

Visualization is lazy-loaded via package extensions: `ext/VisualGeometryToolkitCoreMakieExt.jl` implements all plotting (code in `src/plotting/`) and loads with any Makie backend; `ext/VisualGeometryToolkitCoreGLMakieExt.jl` only adds the GLMakie window helpers `window!` and `closeall`.

**Architecture**: Stub functions are defined in VisualGeometryToolkitCore (`src/visualization_stubs.jl`, for LSP indexing), and the extensions add method implementations when a Makie backend is loaded.

```julia
using GLMakie
using VisualGeometryToolkitCore

# Simple Axis-based image display (exported)
fig, ax, plt = imshow(img)
scatter!(ax, [100, 200], [150, 250]; color=:red)

# SpecApi composition with campixel camera
using VisualGeometryToolkitCore: CampixelImshow, poly!, LScene3D, camera!

fig, ax, plt = imshow(img)
poly!(ax, ellipse; strokecolor=:red)     # overlays go on the Axis (forwards to Makie.poly!)
scene = CampixelImshow(img)              # SpecApi scene: push S.Poly(...) onto scene.plots
scene3d = LScene3D()
camera!(scene3d, camera, bounds; frustum_color=:orange)   # pushes specs onto scene3d.plots
```

Key functions:
- **Exported**: `imshow`, `imshow!`, `CampixelImshow`, `campixel_imshow`, `campixel_imshow!`, `plot_rectification`, `matches`, `matches!`, `ellipses!`, `LScene3D`, `camera!`, `cameraplot!`, `frameplot!`, `Frustum`, and the GLMakie helpers `window!`, `closeall`
- **Public, not exported**: `cameraplot`, `frameplot`, `FrameSpec`, and `poly!` (to avoid Makie conflicts)

Image display functions:
- `imshow(img)` - Axis-based display, returns (Figure, Axis, Plot)
- `imshow!(ax, img)` - Add to existing Axis
- `CampixelImshow(img)` - SpecApi LScene spec with campixel! camera
- `campixel_imshow(img)` - LScene-based, returns (Figure, LScene, Plot)
- `campixel_imshow!(lscene, img)` - Add to existing LScene

Correspondence visualization is VGC's `matches!`/`matches`;
VisualGeometryToolkitFeatures extends them by dispatch for its feature types (the old
`cspond!`/`matchlines` names are gone from both packages).

Other functions:
- `poly!(lscene, shape)` - Add geometric primitives to scene
- `LScene3D()` - 3D scene with Camera3D controls
- `camera!(lscene3d, camera, bounds)` - Camera visualization pushed as specs onto a SpecApi `LScene3D`
- `cameraplot!(scene, camera, bounds)` - Camera visualization recipe (center, axes, frustum) on a realized scene

## Unit System

Units are FlexUnits.jl as it ships, with no stack quantity type, dimension set or registry: `src/base/units.jl` holds the unit constants and registers `px` (dimensionless), `pd`, `dpi` and `pt` into FlexUnits' own registry at `__init__` (a registration made while precompiling does not persist). Dimensions are runtime values, not type parameters: `10.0mm`, `3px` and `300dpi` all share FlexUnits' base-form type `Quantity{Float64, Dimensions{FixRat32}}` (spelled out, never aliased), and mismatches raise `DimensionError`/`ConversionError` at runtime. A pixel is dimensionless, so it cannot be told from a radian; a pixel-sized parameter refuses a length with `DimensionError`. Values are stored in base units (metres), so a bare `ustrip` of a length is in metres: name the unit (`ustrip(mm, q)`). Angles are plain radians. See `conventions/units.md`.

**Logical units**: `px` (pixel), `dpi` (px/inch density)

**Physical units**: `mm`, `cm`, `m`, `μm`, `inch`, `ft`, `pt` (point = 1/72 inch)

Key types and helpers:
- `Size2{T}` - width/height with optional units (`src/base/size2.jl`)
- `BaseQuantity` (`typeof(ubase(1.0mm))`, i.e. `Quantity{Float64, Dimensions{FixRat32}}`) - FlexUnits' base-form quantity, held in structs; boundaries accept any FlexUnits `Quantity` and store `base_quantity(q)` (= `float(ubase(q))`: `142mm` is an exact Rational until stored)
- `StackUnits` - the stack's unit registry (FlexUnits' `RegistryTools`, `AffineTransform{Rational{Int}}`): every scale an exact rational (mm = 1//1000, inch = 127//5000); `registry_defaults!` is not used
- `@ud_str`, `uparse`, `qparse` - `StackUnits`' generated parsers, re-exported; a unit name from outside is read ONLY by `parse_unit` (the guarded `uparse`) and written by `unit_string` — `check_naming` R11
- `strip_or_bare(u, x)` - a quantity (or frame) stripped in `u`, plain numbers as they are
- `dimension(q) == dimension(u)` - the dimension check at constructor boundaries
- `ustrip(u, q)` - strip in a target unit; a px ↔ length crossing is a product with a density (`q * 300dpi`)

## Code Conventions

- **Functions**: `snake_case` (`detect_features`, `focal_length`)
- **Types**: `CamelCase` (`Ellipse`, `HomTranslationMat`)
- **Constants**: `UPPERCASE_SNAKE_CASE` (`CMOS_SENSORS`)
- **Section dividers**: `# =============================================================================`

## Stack Conventions

The conventions every stack package follows (naming and the estimator verbs,
signatures, data arguments, coordinate frames and normalization, pixel origins,
bearings, measurement noise, Jacobians, units, stack contracts) live ONCE, in
`conventions/` at the repository root; `conventions/README.md` indexes them. They ship with the
package (`VisualGeometryToolkitCore.conventions_dir()`), the other packages' CLAUDE.md
files point at them, and the docs' Conventions page is generated from them.
Edit them there, never in a CLAUDE.md. Imported in full:

@conventions/overview.md
@conventions/naming.md
@conventions/signatures.md
@conventions/data-arguments.md
@conventions/frames.md
@conventions/pixel-origins.md
@conventions/bearings.md
@conventions/measurement-noise.md
@conventions/persisted-geometry.md
@conventions/jacobians.md
@conventions/units.md
@conventions/stack-contracts.md

## Git Commit Conventions

- **NEVER** add "Co-Authored-By: Claude" or similar AI attribution to commit messages
- **NEVER** add "🤖 Generated with Claude Code" or similar footers
- Write clean, concise commit messages focused on the change itself

## Design Conventions

### Coordinate Frames and Normalization in this package

VisualGeometryToolkitCore owns the strategies and `CspondNormalization`. Its `!` forms,
its validity predicates (`test_sample`, `test_model`, `test_consensus`,
`test_conditioning`) and its minimal kernels are inner-loop pieces: each
docstring names the normalization it expects. **Their tolerances do not all
live in the same frame**: the collinearity ratio (1e-3) is scale-free, the
conditioning bound (κ = 100) is in the normalized frame, and the coincidence
tolerance (`COINCIDENT_PX`, 0.1 px) is in the CALLER'S data frame, forwarded as
`tol_px` — a duplicate detection is a pixel fact, so it is the one gate whose
unit is not the estimator's. `NormalizedImageFrame` is the pool-storage frame only;
no estimator conditions in it.

### Naming Precision

- **`centroid` function** (not `center` function): Use `centroid(x)` to get the geometric center/midpoint of primitives. The `center` field in structs is consistent with GeometryBasics. Example: `centroid(rect)` returns midpoint (computed), `centroid(ellipse)` returns `ellipse.center`, `origin(rect)` returns corner.

### Composition Over Inheritance

Prefer composition with property forwarding over deep inheritance hierarchies:
```julia
# Good: Attributed wraps any value with attributes
struct Attributed{V,A}
    value::V
    attributes::A
end
# Forward properties: a.field → a.attributes.field, then a.value.field

# Avoid: Deep inheritance chains
# abstract type AbstractCspond end
# struct Cspond <: AbstractCspond ... end
# struct AttributedCspond <: AbstractCspond ... end  # duplicates fields
```

### Accessors, Not Field Reflection

Reach a struct's state through accessors — a getter, and a setter when the state
is settable. Do NOT reflect over `fieldnames`/`hasfield`, and do not rebuild a
struct by splatting its fields positionally.

```julia
# Good: a declared set accessor, implemented BESIDE the struct
with_width(lo::Irls{K}, w::WidthAxis) where {K} =
    Irls{K}(; max_iter = lo.max_iter, width = w)

# Avoid: a reflective rebuild
T((f === :width ? w : getfield(lo, f) for f in fieldnames(T))...)
```

The reflective form couples the caller to the field's NAME, its ORDER, and the
existence of a positional constructor over every field. Worse, it breaks at
RUNTIME in the consuming package while that package's own suite stays green —
nothing there need call the function to compile. Implementing the setter beside
the struct keeps a field rename and its setter in view of each other.

`getfield(x, :sym)` is legitimate where a type overloads `getproperty` and
internal code must bypass it; that is the idiom, not a violation.

### Type Aliases for Simple Cases

Use type aliases when a simple existing type suffices:
```julia
const Cspond{S,T} = Pair{S,T}  # Correspondence is just a Pair
```

### Parameter Structs with `@kwdef`

Use `Base.@kwdef` for configuration structs with sensible defaults:
```julia
# Good: Self-documenting, easy to construct
Base.@kwdef struct DetectorParams
    peak_threshold::Float64 = 0.003
    edge_threshold::Float64 = 10.0
    first_octave::Int = -1
end

# Usage: DetectorParams(peak_threshold=0.01)  # others use defaults

# Avoid: Positional constructors with many arguments
struct DetectorParams
    peak_threshold::Float64
    edge_threshold::Float64
    first_octave::Int
end
# Usage: DetectorParams(0.003, 10.0, -1)  # what do these mean?
```

### Functor Pattern for Callable Structs

Make structs callable when they represent reusable operators:
```julia
# Good: ScaleSpace as a functor that applies kernels
(ss::ScaleSpace)(kernel) = ScaleSpaceResponse(ss, kernel)

# Usage: ixx = ss(DERIVATIVE_KERNELS.xx)

# Also good: Detector as a functor for repeated use
(d::HessianLaplaceDetector)(img) = detect_features!(d, img)

# Usage:
detector = HessianLaplaceDetector(template_img)
features1 = detector(img1)  # reuses buffers
features2 = detector(img2)
```

### Prefer Broadcasting Over Comprehensions

Use broadcasting with `vcat.` instead of list comprehensions for element-wise operations:
```julia
# Good: Broadcasting
X = Point3.(vcat.(centroid.(blobs), 0.0))

# Avoid: List comprehension
X = [Point3(b.center..., 0.0) for b in blobs]
```

### Functional Iteration Patterns

Prefer functional constructs over imperative loops when they improve clarity:

**`foreach` for side-effect iteration** (mutating, no return value):
```julia
# Good: foreach with zip for paired iteration
foreach(zip(dest.levels, src.levels)) do (d, s)
    apply_transform!(d.data, s.data, kernel)
end

# Avoid: index-based loop
for i in eachindex(dest.levels)
    apply_transform!(dest.levels[i].data, src.levels[i].data, kernel)
end
```

**`mapreduce` for transform-and-combine** (single sweep, no intermediate allocation):
```julia
# Good: mapreduce for finding extrema of transformed values
minDiff = mapreduce(neigh -> center - neigh, min, neighbors)
maxDiff = mapreduce(neigh -> center - neigh, max, neighbors)

# Avoid: loop with accumulator
minDiff = Inf
for neigh in neighbors
    minDiff = min(minDiff, center - neigh)
end
```

**When to use each pattern:**
| Pattern | Use When | Returns |
|---------|----------|---------|
| `for` loop | Need `break`/`continue`, complex control flow | nothing |
| `map` | Transform each element, collect results | new array |
| `foreach` | Side effects only (mutating, printing) | nothing |
| `mapreduce` | Transform + combine into single value | single value |

### Interface Pattern

1. Define function stubs in `base/interface.jl` / `base/interface_functions.jl` (loaded early)
2. Submodules extend these functions for their types
3. Traits are defined in `base/traits.jl` (loaded by interface.jl), implementations in submodules

```julia
# In base/interface.jl / interface_functions.jl:
function centroid end
function distance end

# In Primitives submodule:
centroid(e::Ellipse) = e.center
distance(h::Hyperplane, p::Point2) = ...
coeffs(p::Paraboloid3D{T,P}) -> SVector{P,T}  # paraboloid free parameters
```

### Export Organization

Exports in modules and submodules follow a consistent format:

**Structure:**
1. Main section header with `=` dividers
2. Numbered category subsections with `-` dividers
3. Inline comments explaining non-obvious exports

```julia
# =============================================================================
# EXPORTS - Organized by category
# =============================================================================

# -----------------------------------------------------------------------------
# 1. GEOMETRIC PRIMITIVES - Points, shapes, and coordinate types
# -----------------------------------------------------------------------------
export Point2, Circle, Ellipse         # Core types
export centroid, area                  # Accessors

# -----------------------------------------------------------------------------
# 2. TRANSFORMS - Maps for transforming geometry
# -----------------------------------------------------------------------------
export HomographyMap, EuclideanMap    # Transform types
# Coordinate mappings:
# - AffineMap(::HyperRectangle{2}/::Ellipse/::Circle) — canonical → primitive
#   (extends a CoordinateTransformations constructor)
# - EuclideanMap(PixelFrame{A}, PixelFrame{B}) — image-origin conventions
```

**Guidelines:**
- Group related exports together (types, then functions)
- Number categories to show logical ordering
- Add brief comments for exports whose purpose isn't obvious
- Put abstract types before concrete types
- Put constructors/accessors after their associated types

### Include Order Documentation

Include sections in entry point files must document dependencies:

**Structure:**
1. ASCII dependency graph showing load order
2. Numbered list of critical dependencies
3. Inline comments on each include explaining what it provides/uses

```julia
# =============================================================================
# INCLUDE ORDER - Dependency-aware loading
# =============================================================================
#
# DEPENDENCY GRAPH (arrows show "depends on"):
#
#   conic.jl           (Ellipse, Circle)
#        ↓
#   blob_types.jl      (Disc, Gaussian2D)
#        ↓              Uses: Ellipse
#   rectangles.jl      (Rect constructors)
#                       Uses: Ellipse, Disc
#
# CRITICAL DEPENDENCIES:
# 1. conic.jl MUST be first (defines Ellipse used everywhere)
# 2. blob_types.jl MUST come after conic.jl (boundary returns Ellipse)
#
# =============================================================================

# Conic sections: Ellipse struct, Circle interface
# NOTE: Extends HomEllipseMat from parent module
include("conic.jl")

# Blob types: Disc (3 DOF), Gaussian2D (5 DOF)
# NOTE: region(::Disc, k) returns Circle
include("blob_types.jl")
```

**Guidelines:**
- Document why the order matters, not just what the order is
- Use arrows (↓) in dependency graphs to show direction
- List critical dependencies with numbered explanations
- Add `NOTE:` comments for non-obvious dependencies

## Key Files

| File | Purpose |
|------|---------|
| `src/VisualGeometryToolkitCore.jl` | Entry point, exports, include ordering |
| `src/base/interface.jl` | Loads `traits.jl` + `interface_functions.jl`; camera/detection interface stubs |
| `src/base/traits.jl` | Holy Traits definitions |
| `src/base/types.jl` | Loads interface.jl, defines `AbstractDetection`, `Uncertain`, conic matrices |
| `src/base/units.jl` | the registry `StackUnits` (exact rational scales), unit constants, `base_quantity`, `parse_unit`/`unit_string` (`Size2` in `size2.jl`) |
| `src/geometry/transforms/matrices.jl` | Transform matrices + abstract hierarchy (composition promotion) |
| `src/geometry/transforms/euclidean_map.jl` | `EuclideanMap` for camera poses |
| `src/geometry/transforms/homography_map.jl` | `HomographyMap` for points/lines/conics |
| `src/geometry/primitives/primitives.jl` | Primitives submodule entry |
| `src/geometry/cameras/cameras.jl` | Cameras submodule entry (`AbstractCamera`) |
| `ext/VisualGeometryToolkitCoreTestExt/check_naming.jl` | `check_naming` (a Test extension; stubs and `NamingViolation` in `src/quality/naming.jl`): the stack's naming rules and the one list of banned words and retired names |

## Benchmarking

**ALWAYS use `@btime` or `@benchmark` from BenchmarkTools.jl** for performance measurements. Never use `@time` or `@elapsed` for benchmarking as they include JIT compilation time on first run.

```julia
using BenchmarkTools

# Good: @btime handles warmup automatically, shows minimum time
@btime my_function($input)  # $ interpolates to avoid benchmarking global access

# Good: @benchmark for detailed statistics
result = @benchmark my_function($input)
println("Median: ", median(result.times) / 1e6, " ms")

# Avoid: @time includes JIT compilation
@time my_function(input)  # First run includes compilation!

# Avoid: @elapsed without proper warmup
t = @elapsed my_function(input)  # Unreliable
```

**Key rules:**
1. Use `$` to interpolate inputs (avoids measuring global variable access)
2. `@btime` returns minimum time (best for comparing implementations)
3. `@benchmark` returns full statistics (median, memory, allocations)

**Type stability analysis with Cthulhu:**
```julia
using Cthulhu
@descend my_function(input)  # Interactive descent into type inference
```
