# =============================================================================
# The finds and the seeded local optimization as algorithms — Ransac, LocalOptimization
# =============================================================================
#
#     solve(EstimationProblem(f, p), Ransac(; scoring, local_optimization_strategy, config); rng)
#     solve(RefineProblem(f, θ₀, p), LocalOptimization(strategy; scoring, score_guard))
#     init(problem, algorithm; rng) → solve!(cache), reinit!(cache[, θ₀]; f, p, alg, rng)
#
# `f` is a RANSAC family problem (`HomographyProblem(cs)`, …: the public `f`s, which carry the data and their
# own construction context); `p` the solve's context, `(; extent)` or `NullParameters()`. Core owns `solve`/`init` (the
# kind law, the problem recorded on the result); this file adds `__init` and the cache's `solve!`/`reinit!` for the
# two algorithms, which run the one loop (loop.jl) on the one composed workspace (`RansacWorkspace`).
#
# A family problem's CLASS (`_problem_class`) names its configuration of record: `[pipelines.<class>]` (the
# scorer and local optimization a `ClassDefault()` field resolves to) and `[refine.<class>]` (the strategy a
# LevenbergMarquardt refine's `ClassDefault()` resolves to, `default_optimization_strategy`).
# =============================================================================

# -----------------------------------------------------------------------------
# The class of a family problem, and its configuration of record
# -----------------------------------------------------------------------------

# THE table of the family problems' classes: the problem type, its class — its `configs/default.toml` tables' name
# (`[pipelines.<class>]`, `[refine.<class>]`) — and whether its model class has an extent check (a context `extent` of
# any other problem is refused). One row per class; `_problem_class`, `_REFINE_CLASSES` and `_takes_extent` read it.
const _CLASS_TABLE = (
    (HomographyProblem, :homography, true), (FundMatProblem, :fundmat, false), (EssentialProblem, :essential, false),
    (RotationProblem, :rotation, false), (CircleProblem, :circle, false), (EllipseProblem, :ellipse, false),
    (AbsolutePoseProblem, :absolute_pose, true), (BearingPoseProblem, :bearing_pose, false),
    (RadialHomographyProblem, :radial_homography, true), (RadialCameraProblem, :radial_camera, true),
    (HandEyeProblem, :hand_eye, false))
# The class of a problem type, or `nothing` (no class: a caller's own problem, the line problems, a pool-frame problem
# names its scorer).
_problem_class(p) = _problem_class(typeof(p))
_problem_class(::Type) = nothing
_problem_class(::Type{<:BoundedProblem{P}}) where {P} = _problem_class(P)
_takes_extent(_f) = false
for (P, class, extent) in _CLASS_TABLE
    @eval _problem_class(::Type{<:$P}) = $(QuoteNode(class))
    extent && @eval _takes_extent(::$P) = true
end

# The scorer a find runs: the algorithm's, or (`ClassDefault()`) the class's scorer of record — configs/default.toml
# `[pipelines.<class>] scoring`, read by the one scorer reader for the problem's type — its width in the problem's
# residual unit (`_in_problem_unit`).
_class_scoring(_problem, scoring::AbstractScoring) = scoring
_class_scoring(problem, ::ClassDefault) = _class_scoring(problem, _problem_class(problem), ClassDefault())
_class_scoring(problem, ::Nothing, ::ClassDefault) = throw(ArgumentError(
    "a $(nameof(typeof(problem))) has no class of record to default its scorer from: give `Ransac(; scoring = …)` " *
        "(a threshold scorer such as MsacScoring, the Student-t + uniform scorer with its outlier half-width, …)"))
_class_scoring(problem, class::Symbol, ::ClassDefault) =
    _in_problem_unit(scoring_from_toml(_dflt("pipelines.$class", "scoring"); problem = typeof(problem),
                                       section = "pipelines.$class.scoring"), problem)

# The local optimization a find runs: the algorithm's, or (`ClassDefault()`) the class's of record (configs/default.toml
# `[pipelines.<class>] local_optimization_strategy`); a problem without a class takes its scorer's
# (`default_local_optimization_strategy(scoring)`).
_class_lo(_problem, lo, _scoring) = lo
_class_lo(problem, ::ClassDefault, scoring) = _class_lo(_problem_class(problem), scoring)
_class_lo(::Nothing, scoring) = default_local_optimization_strategy(scoring)
_class_lo(class::Symbol, _scoring) = _dflt_lo(class)

# The unit of a class's file width (`threshold` or `outlier_halfwidth`): the problem's residual unit (pixels) for every
# pixel-data class — E and PnP included, whose residuals are whitened into pixels by the cameras — where the file value
# is the width itself. Two classes state a UNIT-FREE width: hand-eye's is a FRACTION of the problem's characteristic
# length `L` (median rig translation per motion), so the same in metres or millimetres; the bearing pose's a fraction of
# its residual's bound (problems/bearing_pose.jl). A caller's scorer is the caller's, in the residual's unit, never
# rescaled.
_in_problem_unit(sc, problem) = _in_unit(sc, _width_unit(problem))
_in_unit(sc, ::Nothing) = sc
function _in_unit(sc::StudentTUniformScoring, u::Float64)
    h = outlier_halfwidth(sc)   # a class table states a supplied half-width; an estimated support has none to scale
    h === nothing && throw(ArgumentError("a class scorer whose outlier model estimates its support has no unit-free width to scale"))
    return VgtkCore.with_width(sc, VgtkCore.OutlierHalfwidth(h * u))
end
_in_unit(sc::RansacScoring{DG}, u::Float64) where {DG} = RansacScoring(DG; threshold = sc.threshold * u)
_in_unit(sc::MsacScoring{DG}, u::Float64) where {DG} = MsacScoring(DG; threshold = sc.threshold * u)
_width_unit(_problem) = nothing   # a unit-free class's width unit is its problem's (hand_eye.jl, bearing_pose.jl)

# The strategy of record of a refine class: configs/default.toml [refine.<class>] `optimization_strategy`. Read once,
# into the constant `_REFINE_STRATEGY_DEFAULTS` (after config_toml.jl), and looked up by a class known at compile time
# (`Val`), so the default is inferred.
const _REFINE_CLASSES = map(row -> row[2], _CLASS_TABLE)
_dflt_strategy(section::AbstractString, class::Symbol) =
    optimization_strategy_from_toml(_dflt(section, "optimization_strategy"); class, section = "$section.optimization_strategy")
_read_refine_strategy_defaults() =
    NamedTuple{_REFINE_CLASSES}(map(c -> _dflt_strategy("refine.$c", c), _REFINE_CLASSES))
_refine_default(::Val{C}) where {C} = getfield(_REFINE_STRATEGY_DEFAULTS, C)

# The LevenbergMarquardt refine's `ClassDefault()` of a family problem (Core's contract generic); a problem without a
# class has none (Core's refusal: "has no class of record").
default_optimization_strategy(p::AbstractRansacProblem) = _refine_default(p, _problem_class(p))
_refine_default(p, ::Nothing) = @invoke default_optimization_strategy(p::Any)
_refine_default(_p, class::Symbol) = _refine_default(Val(class))

# The one parameter-holding keyword is `fixed` (the owner's ruling, 2026-11-11: NLLSsolver's word), naming the
# exceptions; its other spellings, `free`, `fix`, `fixed_params` and `hold`, are refused by Core's one refusal of a
# retired keyword (`refuse_retired_keyword`), never aliased. Every fit and refine that takes `fixed` calls it.
function _refuse_held_spellings(free, fix, fixed_params, hold)
    refuse_retired_keyword(free, :free, :fixed)
    refuse_retired_keyword(fix, :fix, :fixed)
    refuse_retired_keyword(fixed_params, :fixed_params, :fixed)
    refuse_retired_keyword(hold, :hold, :fixed)
    return nothing
end

# A scorer carries its own width — a threshold scorer's `threshold`, the Student-t + uniform scorer's
# `outlier_halfwidth` — so no RANSAC algorithm takes a bare width keyword: each is refused with the way to say it,
# never read into a default scorer (one schema: the scorer object, or its TOML table).
function _refuse_width_keywords(threshold, outlier_halfwidth)
    threshold === nothing || throw(ArgumentError(
        "`threshold` is no keyword of a RANSAC algorithm: it is a threshold scorer's width — give " *
            "`scoring = MsacScoring(problem; threshold)` (or `RansacScoring`); the class default is " *
            "configs/default.toml [pipelines.<class>] `scoring`"))
    outlier_halfwidth === nothing || throw(ArgumentError(
        "`outlier_halfwidth` is no keyword of a RANSAC algorithm: it is the Student-t + uniform " *
            "scorer's width — give `scoring = StudentTUniformScoring(problem, outlier_halfwidth; " *
            "scale_prior, label_prior)`"))
    return nothing
end

# A problem's refusal of a scoring it cannot mean — the ONE check every run (`Ransac`, `LocalOptimization`) runs on the
# scoring it is about to use, whoever built it. No-op by default; the essential problem refuses a sub-detector
# half-width (problems/essential.jl), the leftover of its calibrated-unit era.
_check_scoring(_problem, _scoring) = nothing

# -----------------------------------------------------------------------------
# The solve's context: `p = (; extent)`
# -----------------------------------------------------------------------------

# The problem a run is on: `f`, or — `p = (; extent)` — `BoundedProblem(f, extent)`, whose `extent_trait` makes the
# post-local-optimization consensus hook validate the fitted model over the region (`_check_extent`). A physical
# problem's extent is a physical rectangle, stripped in its unit (`_context_extent`).
_with_context(f, ::NullParameters) = f
function _with_context(f, p::NamedTuple)
    keys(p) === (:extent,) || throw(ArgumentError(string(
        "the context `p` of a RANSAC problem is `(; extent)` (a validity region of the model) or none; got the key(s) ",
        join(keys(p), ", "))))
    _takes_extent(f) || throw(ArgumentError(string("a ", nameof(typeof(f)), " has no extent check: its model is ",
        "validated over no region (an extent context is taken by the homography, the absolute pose, the radial homography ",
        "and the radial camera)")))
    return with_extent(f, _context_extent(_problem_unit(f), p.extent))
end
_context_extent(::Nothing, ::Nothing) = nothing
_context_extent(::Nothing, extent::HyperRectangle{2, <:Real}) = extent
_context_extent(::_Metres, extent::HyperRectangle{2, <:Quantity}) = _plain_metres(extent)

"""
    with_extent(problem, extent) -> problem or BoundedProblem

`problem` unchanged for `extent = nothing`, else [`BoundedProblem`](@ref)`(problem, extent)`, whose `extent_trait`
makes RANSAC's post-local-optimization extent-consensus hook validate the fitted model over the region (the
`_check_extent` method for the model class): what a solve's context `p = (; extent)` does to its family problem.
"""
with_extent(p, ::Nothing) = p
with_extent(p, extent) = BoundedProblem(p, extent)

# -----------------------------------------------------------------------------
# Ransac
# -----------------------------------------------------------------------------

"""
    Ransac([T = Float32]; scoring = ClassDefault(), local_optimization_strategy = ClassDefault(),
           config = RansacConfig(), score_guard = false, on_improvement = nothing, on_hypothesis = nothing)

The RANSAC find, a [`FindKind`](@extref VisualGeometryToolkitCore.FindKind): `solve(EstimationProblem(f, p), Ransac(…);
rng)` on a family problem `f` (`HomographyProblem(cs)`, `EssentialProblem(camera1, camera2, cs)`, …, or a caller's
own `AbstractRansacProblem`), `p = (; extent)` for a validity region of the model (or none). Returns a
[`RansacEstimate`](@ref).

For each minimal sample it fits θₜ (`sample_models`), scores it, and takes the scorer's consensus set; S_gl gates entry
into local optimization and S_lo tracks the best locally optimized score; the adaptive trial count uses the
hypergeometric distribution (`config.success_prob`), within `config.max_trials` and at least `config.min_trials`.

Fields (configuration):
- `scoring`: the scorer, an object of VisualGeometryToolkitCore's scoring contract carrying its own width IN THE
  PROBLEM'S RESIDUAL UNIT — a threshold scorer's `threshold` (`MsacScoring(HomographyProblem; threshold = 2.0)`,
  `RansacScoring`), the Student-t + uniform scorer's `outlier_halfwidth`, a paper's scorer. `ClassDefault()`: the
  problem class's scorer of record (`configs/default.toml [pipelines.<class>] scoring`, its width in the problem's
  unit); a problem without a class needs one. A bare `threshold` or `outlier_halfwidth` keyword is refused
  (`ArgumentError`): a width belongs to its scorer.
- `local_optimization_strategy`: the LO-RANSAC strategy, run in the loop's Phase 2 through `locally_optimize!`, or — a
  [`TerminalLO`](@ref) ([`lo_placement`](@ref)) — once on the final incumbent of a raw race; `nothing` is none.
  `ClassDefault()`: the class's (`[pipelines.<class>] local_optimization_strategy`), or the scorer's
  (`default_local_optimization_strategy(scoring)`) for a problem without a class.
- `config`: the [`RansacConfig`](@ref) (`max_trials`, `min_trials`, `success_prob`).
- `score_guard`: per-candidate ScoreGuarded acceptance: a refined model scoring BELOW its own seed re-enters the race as
  the RAW hypothesis (its record's rule `SeedKept`).
- `on_improvement`: an anytime-trace hook `(trial, score, ws) -> nothing`, called exactly when the tracked best is
  replaced; `on_hypothesis`: `(trial, model) -> nothing` for every candidate that succeeds at `test_model`.
- `T`: the per-point storage precision of the workspace and of the result's `resid` and weights.

Solve keyword (run setting): `rng`, the source of every random draw (`supports_solve_rng`); a seeded rng
(`StableRNG(seed)`) makes the run bit-reproducible, the default is the task-local `Random.default_rng()`.

Workspace reuse is `init(problem, Ransac(…); rng)` → `solve!(cache)` → `reinit!(cache; f, alg, rng)` with new data,
a new configuration of the same `Ransac` type or a new rng: the cache holds the composed workspace and the result
buffers, and the returned estimate's `resid` and weights ARE those buffers, overwritten by the next `solve!` (a caller
that keeps a result across solves copies it). Fewer data than a minimal sample: the `:no_model` result of a search that
drew no trial (`stats.nsteps` 0).
"""
struct Ransac{T, S, L, I, H} <: AbstractEstimationAlgorithm
    scoring::S
    local_optimization_strategy::L
    config::RansacConfig
    score_guard::Bool
    on_improvement::I
    on_hypothesis::H
end
function Ransac(::Type{T}; scoring = ClassDefault(), local_optimization_strategy = ClassDefault(),
        config::RansacConfig = RansacConfig(), score_guard::Bool = false, on_improvement = nothing, on_hypothesis = nothing,
        threshold = nothing, outlier_halfwidth = nothing,   # refused, with the way to say it
        local_optimization = RetiredKeyword()) where {T <: AbstractFloat}
    _refuse_width_keywords(threshold, outlier_halfwidth)
    refuse_retired_keyword(local_optimization, :local_optimization, :local_optimization_strategy)
    scoring isa Union{ClassDefault, AbstractScoring} || throw(ArgumentError(
        "Ransac: `scoring` is a scorer of the scoring contract or ClassDefault(), got a $(typeof(scoring))"))
    local_optimization_strategy isa Union{ClassDefault, Nothing, AbstractOptimizationStrategy} || throw(ArgumentError(
        "Ransac: `local_optimization_strategy` is an optimization strategy, nothing or ClassDefault(), got a " *
            "$(typeof(local_optimization_strategy))"))
    return Ransac{T, typeof(scoring), typeof(local_optimization_strategy), typeof(on_improvement), typeof(on_hypothesis)}(
        scoring, local_optimization_strategy, config, score_guard, on_improvement, on_hypothesis)
end
Ransac(; kwargs...) = Ransac(_DEFAULT_STORAGE_ELTYPE; kwargs...)
estimator_kind(::Ransac) = FindKind()

"""
    LocalOptimization(strategy; scoring, score_guard = false)

The RANSAC local-optimization step from a caller's seed, a [`RefineKind`](@extref VisualGeometryToolkitCore.RefineKind):
`solve(RefineProblem(f, θ₀, p), LocalOptimization(strategy; scoring))` runs the same `score → locally_optimize! →
re-score → test_consensus → finalize` machinery a RANSAC trial runs, driven by θ₀ instead of a minimal sample (no
sampling, no consensus search). `strategy` is REQUIRED (`nothing`: no optimization, the seed scored); so is `scoring`
(a scorer object with its own width): there is no class to default either from. If the refined model fails
`test_consensus` the seed is returned; with `score_guard = true` the refined model is kept only if it does not lower
the seed's score. Returns a [`RansacEstimate`](@ref).

`θ₀` may be a model (an `Uncertain` one brings its own covariance to a predictive scorer), or a `FixedModels` of
candidates (the planar-pose ambiguity): each is scored against the consensus set, and only the best that succeeds at
`test_consensus` is locally optimized (no candidate clears: a result with no model, `Failure`). `p = (; extent)`
validates the refined model over a region, as for [`Ransac`](@ref).

The strategy receives its loss width through its width mode (VGC `loss_at`): a consensus-anchored strategy
(`EfficiencyConstant(m)`) reads the consensus set's ŝ², an absolute one (`LossWidth`, `with_width`) runs at the
caller's width with the scoring untouched, and a strategy with no width follows the scorer's `decision_boundary`.

Workspace reuse is `init(RefineProblem(f, θ₀), LocalOptimization(…))` → `solve!(cache)` → `reinit!(cache, θ₁; alg)`,
a new seed and a new configuration of the same type (the scorer at another width: the refit layer of an evaluation);
the result's `resid` and weights ARE the cache's buffers until the next `solve!` (copy what you keep). The timing
bracket of one refit (the LO timing gate, an evaluation's per-call wall clock) is `solve!(reinit!(cache, θ₁))`:
the workspace is built once by `init`, outside it.
"""
struct LocalOptimization{L, S} <: AbstractEstimationAlgorithm
    strategy::L
    scoring::S
    score_guard::Bool
end
function LocalOptimization(strategy::Union{Nothing, AbstractOptimizationStrategy}; scoring::AbstractScoring,
        score_guard::Bool = false, threshold = nothing, outlier_halfwidth = nothing, local_optimization = RetiredKeyword())
    _refuse_width_keywords(threshold, outlier_halfwidth)
    refuse_retired_keyword(local_optimization, :local_optimization, :local_optimization_strategy)
    return LocalOptimization(strategy, scoring, score_guard)
end
estimator_kind(::LocalOptimization) = RefineKind()

# -----------------------------------------------------------------------------
# The cache: init → solve! → reinit!
# -----------------------------------------------------------------------------

"""
    RansacCache

The cache of [`Ransac`](@ref) and [`LocalOptimization`](@ref) (`init`'s): the problem the run is on (its context
applied), the scorer and local optimization resolved for the problem (from the
algorithm's fields, `ClassDefault()` against the problem's class and data), the composed `RansacWorkspace`, the result
buffers, the rng and the seed (`nothing` for a find). `solve!` runs it — a find on fewer data than a minimal sample is
the `:no_model` result of no trial; `reinit!(cache[, θ₀]; f, p, alg, rng)` points it at a new seed, context, data,
algorithm configuration or rng, keeping its workspace (grown only when the new data outgrow it).

A result of `solve!` ALIASES the cache's result buffers — its `resid` and weights ARE them — until the next `solve!`
(SciML's cache rule): a caller that keeps a result across solves copies what it keeps.
"""
mutable struct RansacCache{P, S, L, W, R, G, A, Q}
    problem::P
    scoring::S
    lo::L
    ws::W
    result::R
    rng::G
    alg::A
    seed::Q
end

function __init(prob::EstimationProblem{<:Any, <:AbstractRansacProblem}, alg::Ransac{T};
        rng::Random.AbstractRNG = Random.default_rng()) where {T}
    problem, scoring, lo = _resolve(alg, prob.f, prob.p)
    ws = RansacWorkspace(T, problem, scoring, lo)
    return RansacCache(problem, scoring, lo, ws, _result_buffers(ws, problem), rng, alg, nothing)
end
function __init(prob::RefineProblem{<:AbstractRansacProblem}, alg::LocalOptimization)
    problem, scoring, lo = _resolve(alg, prob.f, prob.p)
    ws = RansacWorkspace(problem, scoring, lo)
    return RansacCache(problem, scoring, lo, ws, _result_buffers(ws, problem), nothing, alg, prob.u0)
end
# The run's problem (its context applied), scorer and local optimization, resolved from the algorithm's fields for the
# data `f` — `ClassDefault()` against the problem's class and data (a class width can read the data: hand-eye's
# characteristic length) — and the scorer checked against the problem: at `init` and at every `reinit!` with new data.
function _resolve(alg::Ransac, f, p)
    problem = _with_context(f, p)
    scoring = _class_scoring(f, alg.scoring)
    lo = _class_lo(f, alg.local_optimization_strategy, scoring)
    _check_scoring(problem, scoring)
    return problem, scoring, lo
end
function _resolve(alg::LocalOptimization, f, p)
    problem = _with_context(f, p)
    _check_scoring(problem, alg.scoring)
    return problem, alg.scoring, alg.strategy
end

# A find on fewer data than a minimal sample is the `:no_model` result of a search that drew no trial. A physical
# problem's model comes back in metres (`_with_units`, units.jl), with or without one.
solve!(c::RansacCache) = _with_units(c.problem, _run!(c, c.alg))
_run!(c::RansacCache, alg::Ransac) =
    _holds_minimal_sample(c.problem, data_size(c.problem)) ? _ransac!(
        c.result, c.ws, c.problem, c.scoring; local_optimization_strategy = c.lo, config = alg.config,
        on_improvement = alg.on_improvement, on_hypothesis = alg.on_hypothesis, score_guard = alg.score_guard, rng = c.rng) :
    _no_model_result(model_type(c.problem), c.scoring, c.result, 0, NaN)
function _run!(c::RansacCache, alg::LocalOptimization)
    _entry_workspace(c.ws, c.problem, c.scoring, c.lo)
    return _seed_run!(c, c.seed)
end
_seed_run!(c::RansacCache, models::FixedModels) =
    _select_then_optimize!(c.ws, c.problem, c.scoring, c.lo, models; result = c.result)
# The seed through the problem's `guess_model` (the refine contract's one reading of a guess: the workspace model type
# may be a labeled alias, `CameraExtrinsics`, while a caller hands in an unlabeled seed or a physical one).
_seed_run!(c::RansacCache, θ₀) = _locally_optimize_seed!(c.ws, c.problem, c.scoring, c.lo,
    _seed_model(c.problem, θ₀); score_guard = c.alg.score_guard, result = c.result)

"""
    reinit!(cache::RansacCache[, θ₀]; f, p, alg = cache's, rng = cache's) -> cache

Point a find's or a local optimization's cache at new data `f` (a family problem of the same type), a new context `p`
(the caller's form: Core's `reinit!(::EstimationCache)` hands over both, the cache's problem's where the call names none),
(a local optimization) a new seed `θ₀`, a new algorithm `alg` or (a find) a new `rng`.

`alg` is a new configuration of the cache's algorithm — a value of its concrete type (a `Ransac` or
`LocalOptimization` of other field types is an `ArgumentError`): a scorer at another width, another strategy setting,
another `config`. The scorer and local optimization are re-resolved from the (new) algorithm's fields for the (new) data
(a class width can read the data) and checked against the problem; the workspace and result buffers are kept, and grown only when the new data
outgrow them. `rng` replaces a find's rng (of the same type; a local optimization draws nothing and takes none). Every
refusal of the call's arguments comes before anything changes. A scorer whose workspace tables depend on its
configuration (a scale prior's) is checked against the kept workspace where every run is, at the next `solve!`
(`check_workspace`, an `ArgumentError`: init a new cache for it).

The next `solve!`'s result aliases the cache's result buffers until the solve after it.
"""
function reinit!(c::RansacCache{P, S, L, W, R, G, A}, θ₀...; f, p, alg = c.alg, rng = c.rng) where {P, S, L, W, R, G, A}
    length(θ₀) <= (c.seed === nothing ? 0 : 1) || throw(ArgumentError("reinit!: a find's cache takes no seed"))
    alg isa A || throw(ArgumentError(string("reinit!: the cache was initialized for a ", A, "; the new algorithm is a ",
        typeof(alg), " (init a new cache for an algorithm of another type)")))
    rng === c.rng || rng isa G || throw(ArgumentError(c.rng === nothing ?
        "reinit!: a local optimization draws nothing and takes no rng" :
        string("reinit!: the cache's rng is a ", G, "; the new one is a ", typeof(rng), " (init a new cache)")))
    problem, scoring, lo = _resolve(alg, f, p)
    n = data_size(problem)
    n > _capacity(c.ws) && (c.ws = RansacWorkspace(_storage(c), problem, scoring, lo))
    length(c.result.residuals) == n || (c.result = _result_buffers(c.ws, problem))
    c.problem, c.scoring, c.lo, c.alg, c.rng = problem, scoring, lo, alg, rng
    isempty(θ₀) || (c.seed = only(θ₀))
    return c
end
_storage(c::RansacCache) = eltype(c.result.residuals)
_capacity(ws::RansacWorkspace) = length(ws.best_residuals)
