# RANSAC

```@meta
CurrentModule = VisualGeometryToolkitEstimation
```

## Main Function

```@docs
ransac
ransac!
locally_optimize
```

## Problem Interface

```@docs
AbstractRansacProblem
AbstractCspondProblem
MinimalSolveProblem
PoseContext
minimal_sample_size
data_size
model_type
codimension
solve
fit_hypothesis
residuals!
test_sample
test_model
test_consensus
refine!
draw_sample!
constraint_with_jacobians
solver_covariance
provides_solver_covariance
VisualGeometryToolkitCore.test_conditioning
VisualGeometryToolkitCore.test_cheirality
```

## Solver + correspondence protocol

The minimal-solver and correspondence verbs the problem layer is built on.
They are VisualGeometryToolkitCore's; documented here because the problem interface
above refers to them.

```@docs
VisualGeometryToolkitCore.MinimalSolver
VisualGeometryToolkitCore.RefitFamily
sources
targets
```

## Residual protocol

The per-point residual protocol (VisualGeometryToolkitCore's) that `residuals!`
runs, and the residual degrees of freedom.

```@docs
VisualGeometryToolkitCore.whitened_residual
VisualGeometryToolkitCore.residual_context
residual_dof
oracle_residual_dof
```

## Local-optimization result

`RansacEstimate`, `RansacMetadata`, `RansacWorkspace` and `inlier_ratio` are
documented under "Result Types" and "Workspaces" below; the LO half of the
result record is here.

```@docs
lo_estimate
```

## Solver Backends

The minimal-sample solver used by `HomographyProblem`, `FundMatProblem`, and
`AbsolutePoseProblem` is selected by supplying a backend to the constructor:

```julia
using VisualGeometryToolkitCore: NativeSolver, PoseLibSolver
HomographyProblem(pairs; backend=NativeSolver())     # default — pure-Julia
HomographyProblem(pairs; backend=PoseLibSolver())    # PoseLib C++ binding
```

The following names are defined and exported by VisualGeometryToolkitCore (RVG
re-exports nothing; import them from there). See the VisualGeometryToolkitCore docs for
their full API.

- **`AbstractSolverBackend`** — abstract supertype for backends
- **`NativeSolver`** — pure-Julia implementations of `homography_4pt`,
  `fundamental_7pt`, `p3p`, etc.
- **`PoseLibSolver`** — bindings to the PoseLib C++ library
- **`essential_5pt(x1, x2)`** — five-point essential matrix solver (native
  pure-Julia by default; `essential_5pt(PoseLibSolver(), x1, x2)` with `using PoseLib`)
- **`essential_dlt(x1, x2)`** — eight-point essential matrix solver (PoseLib only)

## Samplers

```@docs
AbstractSampler
UniformSampler
ProsacSampler
GaussianSampler
ExhaustiveSampler
trial_budget
adapts
sampler
draw!
reset!
```

## Validity regions

A problem can declare a region of validity, such as the image extent. The
consensus check then counts only the points inside it.

```@docs
BoundedProblem
ExtentTrait
NoExtent
WithExtent
extent_trait
```

## Relative-pose bundle

```@docs
RelposeBundleProblem
bundle_refine_relpose
SchurSolver
```

## Configuration and Results

```@docs
RansacConfig
RansacEstimate
RansacMetadata
```

## Inlier Ratio

```@docs
inlier_ratio
```

## Workspaces

`RansacWorkspace{K, M, T, W, L}` is the composed workspace of one RANSAC run:
the score buffers (VisualGeometryToolkitCore's `ScoreBuffers{T, W}` — residuals,
per-point scores, consensus mask and the scorer's workspace `W`), the local
optimizer's workspace `L` (`lo_workspace(P, T, lo, n)`), the incumbent and the
minimal sample. Build it from the scorer and the strategy it will run,
`RansacWorkspace(problem, scoring, local_optimization)`; `RansacWorkspace(P,
n)` holds the default MSAC scorer's. The core strategies' workspace is
[`LoBuffers`](@ref): the refit workspace `refine_workspace(P, n)` —
VisualGeometryToolkitCore's `ManifoldLmWorkspace{N}` for the chart-LM refits (e.g.
N = 8 for `HomographyProblem`, 7 for `FundMatProblem`, 6 for
`AbsolutePoseProblem`) — and the posterior-weight buffers. `DltWorkspace`
selects the algebraic DLT step, `refine!(ws::DltWorkspace, p, θ, weights)`
with θ unread, which LO does not call.

```@docs
RansacWorkspace
RansacResultBuffers
LoBuffers
refine_workspace
DltWorkspace
```

## SVD Workspace

Pre-allocated workspace for zero-allocation SVD computation. Defined in VisualGeometryToolkitCore
(import it from there); the RANSAC solvers use it.

- `SvdWorkspace{T}(max_rows, ncols)` — pre-allocate LAPACK buffers
- `svd_nullvec!(ws, A, m, Val(N))` — compute null-space vector of first `m` rows

## Fixed Models

**`FixedModels{N, M}`** — Stack-allocated container holding up to `N` candidate models of type `M`.
Defined in VisualGeometryToolkitCore (import it from there). Supports `iterate`, `length`, `eltype`, `collect`.

Used as the return type of all `solve()` methods:
- `HomographyProblem` → `FixedModels{1, HomographyMat{T}}`
- `FundMatProblem` → `FixedModels{3, FundMat{T}}`
- `AbsolutePoseProblem` → `FixedModels{4, Extrinsics3}`

## Benchmark pools

A scoring pool (VisualGeometryToolkitCore `Pools.PoolPair`, stored in
`NormalizedImageFrame`) is written here. `draw_hypotheses` repeats the RANSAC
trial — the same draw, sample gate, solve and model gate `ransac` runs — with
no scoring. The record's constructor `PoolPair(rng, problem, raw; target,
max_trials, variance)` draws the hypotheses of a pixel-frame problem on a raw
pair, carries them into the pool frame once, and returns the record with its
residual matrix; `EssentialProblem(pair)`, `FundMatProblem(pair)` and
`HomographyProblem(pair)` are the record's pool-frame problems; the essential
one's sample gate reads 0.1 px through `pool_pixel_scale(pair)` (F's and H's
read the scale-free tolerance on their normalized view).

```@docs
draw_hypotheses
Hypotheses
PoolVariance
InlierVariance
TrackVariance
```

## Configuration

`configs/default.toml` is the config of record: `RansacConfig()`, every core
local-optimization strategy's keyword defaults, and every `find_*` pipeline's
`threshold` and `local_optimization` read it at precompile time, so there is
no second set of Julia literals to drift from it. Single
values are overridden by keyword. `RansacConfig(path)` loads another file's
`[ransac]` table over the shipped one and `local_optimization_from_toml` builds
a strategy from its table — absent keys inherit, present keys override,
unknown keys or tags are refused.

A strategy is selected in TOML by tag:

```toml
[pipelines.homography]
threshold          = 3.0                                 # px, the default MSAC scorer's
local_optimization = { type = "irls", loss = "tls" }     # no width key: follow the scorer
# explicit widths: { type = "irls", loss = "welsch", scale_multiple = 2.5 }
#              or  { type = "irls", loss = "welsch", loss_width = 2.0 }   (both keys: ArgumentError)
```

`type` is a tag of the strategy registry — the core [`LO_TAGS`](@ref)
(`"no_local_optimization"`, `"irls"`), or its explicit merge with a downstream
package's — and `loss` a tag of [`LOSS_TAGS`](@ref). The registries are
VisualGeometryToolkitCore `TagRegistry` values; a retired tag is refused with its
successor's name. The paper strategies and their cross-validated operating
points (`scale_multiple = "cross_validated"`) are LocalOptimizationDoneRight's:

```julia
using VisualGeometryToolkitEstimation: local_optimization_from_toml, LO_TAGS
import LocalOptimizationDoneRight

lo = local_optimization_from_toml(table; class = :homography,
                                  tags = merge(LO_TAGS, LocalOptimizationDoneRight.LO_TAGS))
```

### Registering strategies

The strategy machinery — the registry, the keyword constructor, the TOML
reader and the writer — is one public API that any package registers its
strategies with. A package states one row per strategy and its defaults
table; `@lo_registry` defines the row facts and the keyword constructors and
returns the package's registry:

```julia
import VisualGeometryToolkitEstimation as VgtkEstimation

#   type     tag        path   scale  family  label   shape  parametric
const ROWS = ((MyArm, "my_arm", :irls, nothing, nothing, "Mine", nothing, false),)
const MY_LO_TAGS = VgtkEstimation.@lo_registry ROWS defaults = MY_DEFAULTS["local_optimization"]
```

A strategy's exceptions are methods of the hooks — [`lo_admits`](@ref),
[`lo_key`](@ref), [`lo_field`](@ref), [`lo_field_from_toml`](@ref),
[`lo_absent`](@ref), [`lo_validate`](@ref), [`lo_concrete`](@ref),
[`lo_scale_multiple_shape`](@ref) and [`lo_resolve_scale_multiple`](@ref) —
all of them [`CONTRACT_GENERICS`](@ref); the reader and writer are the same for
every strategy.

```@docs
default_config_path
local_optimization_from_toml
lo_from_toml
config_dict
lo_config_dict
LO_TAGS
@lo_registry
lo_construct
lo_tag
lo_defaults
lo_parametric
lo_admits
lo_admits_shape
lo_key
lo_field
lo_field_from_toml
lo_absent
lo_validate
lo_concrete
lo_scale_multiple_shape
lo_resolve_scale_multiple
LOSS_TAGS
MODEL_UPDATE_TAGS
NUISANCE_TAGS
CONTRACT_GENERICS
```

