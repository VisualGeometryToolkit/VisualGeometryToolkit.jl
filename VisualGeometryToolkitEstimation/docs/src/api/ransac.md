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
solve
fit_hypothesis
test_sample
test_model
test_consensus
draw_sample!
constraint_with_jacobians
solver_covariance
```

The rest of the problem protocol is VisualGeometryToolkitCore's, documented in
its manual: [`data_size`](@extref VisualGeometryToolkitCore.data_size),
[`refine!`](@extref VisualGeometryToolkitCore.refine!) (the local-optimization step this package's problems implement),
[`model_type`](@extref VisualGeometryToolkitCore.model_type),
[`codimension`](@extref VisualGeometryToolkitCore.codimension),
[`mahalanobis_distances!`](@extref VisualGeometryToolkitCore.mahalanobis_distances!),
[`provides_solver_covariance`](@extref VisualGeometryToolkitCore.provides_solver_covariance),
[`test_conditioning`](@extref VisualGeometryToolkitCore.test_conditioning) and
[`test_cheirality`](@extref `VisualGeometryToolkitCore.test_cheirality-Tuple{HomographyMat, Any, AbstractVector{<:Integer}}`).

## Solver + correspondence protocol

The minimal-solver and correspondence verbs the problem layer is built on are
VisualGeometryToolkitCore's:
[`MinimalSolver`](@extref VisualGeometryToolkitCore.MinimalSolver),
[`RefitFamily`](@extref VisualGeometryToolkitCore.RefitFamily),
[`sources`](@extref VisualGeometryToolkitCore.sources) and
[`targets`](@extref VisualGeometryToolkitCore.targets).

## Residual protocol

The per-point protocol that `mahalanobis_distances!` runs is VisualGeometryToolkitCore's
(`mahalanobis_distance(problem, ctx, i)`,
the signed [`whitened_residual`](@extref VisualGeometryToolkitCore.whitened_residual),
[`residual_context`](@extref VisualGeometryToolkitCore.residual_context)); the
residual degrees of freedom are this package's.

```@docs
residual_dof
oracle_residual_dof
```

## Local-optimization result

`RansacEstimate`, `RansacMetadata`, `RansacWorkspace` and `inlier_ratio` are
documented under "Result Types" and "Workspaces" below. The LO half of the
result record is VisualGeometryToolkitCore's
[`lo_estimate`](@extref VisualGeometryToolkitCore.lo_estimate).

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
with_extent
ExtentTrait
NoExtent
WithExtent
extent_trait
```

## Relative-pose bundle

```@docs
RelposeBundleProblem
refine_relative_pose
```

## Configuration and Results

```@docs
RansacConfig
RansacEstimate
RansacMetadata
```

Every RANSAC-class `refine_*` returns a `FitResult` with [`RefineMetadata`](@ref)
(the refine contract: no scorer, the caller's scale or width):

```@docs
RefineMetadata
```

## Inlier Ratio

```@docs
inlier_ratio
```

## One noise scale for a set of results

`pooled_noise_scale(results)` pools the scales several RANSAC results
originated (each `ScaleEstimated`, over its own dof) into one, for a downstream
refine of them all (`scale = pooled`, recorded `ScaleUpstream`); a result
without a scale is left out and counted.

```@docs
pooled_noise_scale
```

## Workspaces

`RansacWorkspace{K, M, T, W, L}` is the composed workspace of one RANSAC run:
the score buffers (VisualGeometryToolkitCore's `ScoreBuffers{T, W}` — residuals,
per-point scores, consensus mask and the scorer's workspace `W`), the local
optimizer's workspace `L` (`lo_workspace(P, T, lo, n)`), the incumbent and the
minimal sample. Build it from the scorer and the strategy it will run,
`RansacWorkspace(problem, scoring, local_optimization_strategy)`; `RansacWorkspace(P,
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
`scoring` and `local_optimization_strategy` read it at precompile time, so there is
no second set of Julia literals to drift from it. Single
values are overridden by keyword. `RansacConfig(path)` loads another file's
`[ransac]` table over the shipped one and `optimization_strategy_from_toml` builds
a strategy from its table — absent keys inherit, present keys override,
unknown keys or tags are refused.

A strategy is selected in TOML by tag:

```toml
[pipelines.homography]
scoring            = { type = "msac", threshold = 3.0 }  # px; a Student-t: { type = "student_t_uniform", outlier_halfwidth = 20.0 }
local_optimization_strategy = { type = "irls", loss = "tls" }     # no width key: follow the scorer
# explicit widths: { type = "irls", loss = "welsch", efficiency_constant = 2.5 }
#              or  { type = "irls", loss = "welsch", loss_width = 2.0 }   (both keys: ArgumentError)
```

`type` is a tag of the strategy registry — the core [`OPTIMIZATION_STRATEGY_TAGS`](@ref)
(`"none"` for `nothing`, `"irls"`, `"triggs"`, `"terminal_lo"`), or its explicit merge with a downstream
package's — and `loss` a tag of [`LOSS_TAGS`](@extref VisualGeometryToolkitCore.LOSS_TAGS). The registries are
VisualGeometryToolkitCore `TagRegistry` values; a retired tag is refused with its
successor's name. The paper strategies and their cross-validated operating
points (`efficiency_constant = "cross_validated"`) are LocalOptimizationDoneRight's:

```julia
using VisualGeometryToolkitEstimation: optimization_strategy_from_toml, OPTIMIZATION_STRATEGY_TAGS
import LocalOptimizationDoneRight

lo = optimization_strategy_from_toml(table; class = :homography,
                                  tags = merge(OPTIMIZATION_STRATEGY_TAGS, LocalOptimizationDoneRight.LODR_STRATEGY_TAGS))
```

The scorer is selected the same way, by one reader through a tag registry
(VisualGeometryToolkitCore's `SCORING_TAGS`, or a downstream merge such as
RANSACScoringDoneRight's), every key required:

```julia
using VisualGeometryToolkitEstimation: scoring_from_toml
scoring = scoring_from_toml(Dict("type" => "msac", "threshold" => 3.0); problem = HomographyProblem)
```

### Registering strategies

The strategy machinery — the registry, the keyword constructor, the TOML
reader and the writer — is one public API that any package registers its
strategies with. A package states one row per strategy and its defaults
table; `@lo_registry` defines the row facts and the keyword constructors and
returns the package's registry:

```julia
import VisualGeometryToolkitCore as VgtkCore   # the machinery is VisualGeometryToolkitCore's

#   type     tag        path   scale  family  label   shape  parametric
const ROWS = ((MyArm, "my_arm", :irls, nothing, nothing, "Mine", nothing, false),)
const MY_STRATEGY_TAGS = VgtkCore.@lo_registry ROWS defaults = MY_DEFAULTS["optimization_strategy"]
```

The machinery is VisualGeometryToolkitCore's —
[`@lo_registry`](@extref `VisualGeometryToolkitCore.@lo_registry-Tuple{Any, Vararg{Any}}`),
[`lo_construct`](@extref VisualGeometryToolkitCore.lo_construct),
[`lo_tag`](@extref VisualGeometryToolkitCore.lo_tag),
[`lo_defaults`](@extref VisualGeometryToolkitCore.lo_defaults),
[`lo_parametric`](@extref VisualGeometryToolkitCore.lo_parametric),
[`lo_admits_shape`](@extref VisualGeometryToolkitCore.lo_admits_shape) and the
loss registry [`LOSS_TAGS`](@extref VisualGeometryToolkitCore.LOSS_TAGS); this
package holds the TOML reader and writer below.
A strategy's exceptions are methods of the hooks — [`lo_admits`](@extref VisualGeometryToolkitCore.lo_admits),
[`lo_key`](@extref VisualGeometryToolkitCore.lo_key), [`lo_field`](@extref VisualGeometryToolkitCore.lo_field), [`lo_field_from_toml`](@extref VisualGeometryToolkitCore.lo_field_from_toml),
[`lo_absent`](@extref VisualGeometryToolkitCore.lo_absent), [`lo_validate`](@extref VisualGeometryToolkitCore.lo_validate), [`lo_concrete`](@ref),
[`lo_efficiency_constant_shape`](@extref VisualGeometryToolkitCore.lo_efficiency_constant_shape) and [`lo_resolve_efficiency_constant`](@ref) —
all of them [`CONTRACT_GENERICS`](@ref); the reader and writer are the same for
every strategy.

```@docs
default_config_path
optimization_strategy_from_toml
lo_from_toml
scoring_from_toml
config_dict
lo_config_dict
OPTIMIZATION_STRATEGY_TAGS
lo_concrete
lo_resolve_efficiency_constant
MODEL_UPDATE_TAGS
NUISANCE_TAGS
STOP_TAGS
CONTRACT_GENERICS
```

## Synthetic test scenes

The scenes every stack package's tests draw from (loaded with `Test`, the
Test extension):

```@docs
synthetic_scenes
```

