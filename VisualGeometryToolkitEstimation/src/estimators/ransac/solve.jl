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
# A family problem's CLASS (`_problem_class`) names its configuration of record, the problem's tables
# (`_class_table`): `[<problem>.ransac]` (the scorer and local optimization a `ClassDefault()` field of `Ransac`
# resolves to) and `[<problem>.levenberg_marquardt]` (the strategy a LevenbergMarquardt refine's `ClassDefault()`
# resolves to, `default_optimization_strategy`).
# =============================================================================

# -----------------------------------------------------------------------------
# The class of a family problem, and its configuration of record
# -----------------------------------------------------------------------------

# THE table of the family problems' classes: the problem type, its class (the name the per-class tuning constants are
# keyed by, LocalOptimizationDoneRight's too), its `configs/default.toml` table — the problem type's tag (`snake_case`
# of its name: `[<problem>.ransac]`, `[<problem>.levenberg_marquardt]`) — and whether its model class has an extent
# check (a context `extent` of any other problem is refused). One row per class; `_problem_class`, `_REFINE_CLASSES`,
# `_class_table` and `_takes_extent` read it.
const _CLASS_TABLE = (
    (HomographyProblem, :homography, "homography_problem", true), (FundMatProblem, :fundmat, "fund_mat_problem", false),
    (EssentialProblem, :essential, "essential_problem", false), (RotationProblem, :rotation, "rotation_problem", false),
    (CircleProblem, :circle, "circle_problem", false), (EllipseProblem, :ellipse, "ellipse_problem", false),
    (AbsolutePoseProblem, :absolute_pose, "absolute_pose_problem", true),
    (BearingPoseProblem, :bearing_pose, "bearing_pose_problem", false),
    (RadialHomographyProblem, :radial_homography, "radial_homography_problem", true),
    (RadialCameraProblem, :radial_camera, "radial_camera_problem", true), (HandEyeProblem, :hand_eye, "hand_eye_problem", false))
# The class of a problem type, or `nothing` (no class: a caller's own problem, the line problems, a pool-frame problem
# names its scorer).
_problem_class(p) = _problem_class(typeof(p))
_problem_class(::Type) = nothing
_problem_class(::Type{<:BoundedProblem{P}}) where {P} = _problem_class(P)
_takes_extent(_f) = false
for (P, class, _, extent) in _CLASS_TABLE
    @eval _problem_class(::Type{<:$P}) = $(QuoteNode(class))
    extent && @eval _takes_extent(::$P) = true
end
# A class's configuration table, its problem's tag.
const _CLASS_TABLES = NamedTuple{map(row -> row[2], _CLASS_TABLE)}(map(row -> row[3], _CLASS_TABLE))
_class_table(class::Symbol) = getfield(_CLASS_TABLES, class)::String

# The scorer a find runs: the algorithm's, or (`ClassDefault()`) the class's scorer of record — configs/default.toml
# `[<problem>.ransac] scoring`, read by the one scorer reader for the problem's type — its width in the problem's
# residual unit (`_in_problem_unit`).
_class_scoring(_problem, scoring::AbstractScoring) = scoring
_class_scoring(problem, ::ClassDefault) = _class_scoring(problem, _problem_class(problem), ClassDefault())
_class_scoring(problem, ::Nothing, ::ClassDefault) = throw(ArgumentError(
    "a $(nameof(typeof(problem))) has no class of record to default its scorer from: give `Ransac(; scoring = …)` " *
        "(a threshold scorer such as MsacScoring, the Student-t + uniform scorer with its outlier half-width, …)"))
_class_scoring(problem, class::Symbol, ::ClassDefault) =
    _in_problem_unit(scoring_from_toml(_dflt("$(_class_table(class)).ransac", "scoring"); problem = typeof(problem),
                                       section = "$(_class_table(class)).ransac.scoring"), problem)

# The local optimization a find runs: the algorithm's, or (`ClassDefault()`) the class's of record (configs/default.toml
# `[<problem>.ransac] local_optimization_strategy`); a problem without a class takes its scorer's
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

# The strategy of record of a refine class: configs/default.toml [<problem>.levenberg_marquardt] `optimization_strategy`. Read once,
# into the constant `_REFINE_STRATEGY_DEFAULTS` (after config_toml.jl), and looked up by a class known at compile time
# (`Val`), so the default is inferred.
const _REFINE_CLASSES = map(row -> row[2], _CLASS_TABLE)
_dflt_strategy(section::AbstractString, class::Symbol) =
    optimization_strategy_from_toml(_dflt(section, "optimization_strategy"); class, section = "$section.optimization_strategy")
_read_refine_strategy_defaults() =
    NamedTuple{_REFINE_CLASSES}(map(c -> _dflt_strategy("$(_class_table(c)).levenberg_marquardt", c), _REFINE_CLASSES))
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
            "configs/default.toml [<problem>.ransac] `scoring`"))
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
_context_extent(::_Meters, extent::HyperRectangle{2, <:Quantity}) = _plain_meters(extent)

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
  problem class's scorer of record (`configs/default.toml [<problem>.ransac] scoring`, its width in the problem's
  unit); a problem without a class needs one. A bare `threshold` or `outlier_halfwidth` keyword is refused
  (`ArgumentError`): a width belongs to its scorer.
- `local_optimization_strategy`: the LO-RANSAC strategy, run in the loop's Phase 2 through `locally_optimize!`, or — a
  [`TerminalLO`](@ref) ([`lo_placement`](@ref)) — once on the final incumbent of a raw race; `nothing` is none.
  `ClassDefault()`: the class's (`[<problem>.ransac] local_optimization_strategy`), or the scorer's
  (`default_local_optimization_strategy(scoring)`) for a problem without a class.
- `config`: the [`RansacConfig`](@ref) (`max_trials`, `min_trials`, `success_prob`).
- `score_guard`: per-candidate ScoreGuarded acceptance: a refined model scoring BELOW its own seed re-enters the race as
  the RAW hypothesis (its record's rule `SeedKept`).
- `on_improvement`: an anytime-trace hook `(trial, score, ws) -> nothing`, called exactly when the tracked best is
  replaced; `on_hypothesis`: `(trial, model) -> nothing` for every candidate that succeeds at `test_model`.
- `T`: the per-point storage precision of the workspace and of the result's `resid` and weights.

Solve keywords (run settings): `rng`, the source of every random draw (`supports_solve_rng`); a seeded rng
(`StableRNG(seed)`) makes the run bit-reproducible, the default is the task-local `Random.default_rng()`. `stops`, the
LOCAL OPTIMIZATION's stop settings (the trial budget is `config`'s): the strategy at those stops, VisualGeometryToolkitCore's
`with_stops(strategy, stops)` — a block-coordinate strategy's `StopAfterIteration(n)` (its `max_iter`), a
`BackendRefit`'s `LmStops`; `nothing` (the default) keeps the strategy's own budget.

Workspace reuse is `init(problem, Ransac(…); rng)` → `solve!(cache)` → `reinit!(cache; f, alg, rng)` with new data,
a new configuration of the same `Ransac` type or a new rng: the cache holds the composed workspace and the result
buffers, and the returned estimate's `resid` and weights ARE those buffers, overwritten by the next `solve!` (a caller
that keeps a result across solves copies it). Fewer data than a minimal sample: the no-model result (`(Failure, TrialBudget)`) of a search that
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

Solve keyword `stops`: the local optimization's stop settings, the strategy at them (VisualGeometryToolkitCore's
`with_stops`: a block-coordinate strategy's `StopAfterIteration(n)`, a `BackendRefit`'s `LmStops`); `nothing` (the
default) keeps the strategy's own. `reinit!(cache, θ₁; stops)` changes them.

# An external refit

A strategy of the `ExternalRefit` family (VisualGeometryToolkitCore's `RefitFamily`: a `BackendRefit` — PoseLib's
LM and Le–Zach refines —, an adaptive-width arm, a paper's classical LO) runs through the same front end, as its
contract's model → model refine at an absolute width, no seed score and no consensus test:

    c = init(RefineProblem(f, θ₀, (; frame)), LocalOptimization(strategy; scoring); width)   # external_workspace
    r = solve!(reinit!(c, θ₁; width))   # reinit! runs prepare_external!, solve! the external_refit! (the bracket)

`width` (required) is the contract's absolute width, a `LossWidth` (`LossWidth(w)`, the loss width `w`) or a fixed
noise scale `FixedS2(s2)`; `p = (; frame)` the two-view frame the refiner reads beyond the problem (`frame.Kc1`,
`frame.Kc2`, `frame.cs`: an evaluation's pair geometry), none (`NullParameters()`) for a refiner that reads none;
`scoring` the refiner's scoring template, or `nothing` for a refiner that anchors on none. `init` builds the
refiner's workspace once (`external_workspace`), `reinit!` points it at a seed, width or configuration of the same
type and runs the refiner's preparation (`prepare_external!`: a consensus-scale bootstrap, OUTSIDE the timing bracket
of one refit, which is `solve!` alone); a one-shot `solve` prepares then refits. The result is the refiner's own
`FitResult` (`external_refit!`), or — the preparation declined the seed — no refit: `u = nothing` (`found_model` false),
`(InitialFailure, NoRule)`, `stats.nsteps` 0 (the record of a local optimization that did not run). Both are ONE type,
the refiner's record type with its model, residuals and raw output widened by `Nothing`, so `solve!` is inferred and
its reuse boxes nothing.
"""
struct LocalOptimization{L, S} <: AbstractEstimationAlgorithm
    strategy::L
    scoring::S
    score_guard::Bool
end
function LocalOptimization(strategy::Union{Nothing, AbstractOptimizationStrategy}; scoring::Union{Nothing, AbstractScoring},
        score_guard::Bool = false, threshold = nothing, outlier_halfwidth = nothing, local_optimization = RetiredKeyword())
    _refuse_width_keywords(threshold, outlier_halfwidth)
    refuse_retired_keyword(local_optimization, :local_optimization, :local_optimization_strategy)
    scoring === nothing && RefitFamily(strategy) isa NativeRefit && throw(ArgumentError(
        "LocalOptimization: a native local optimization scores its seed and its refit: give `scoring` (a scorer with " *
            "its own width); only an external refit (an ExternalRefit strategy) may run without one"))
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
the no-model result (`(Failure, TrialBudget)`) of no trial; `reinit!(cache[, θ₀]; f, p, alg, rng)` points it at a new seed, context, data,
algorithm configuration or rng, keeping its workspace (grown only when the new data outgrow it).

A result of `solve!` ALIASES the cache's result buffers — its `resid` and weights ARE them — until the next `solve!`
(SciML's cache rule): a caller that keeps a result across solves copies what it keeps.
"""
mutable struct RansacCache{P, S, L, W, R, G, A, Q, C}
    problem::P
    scoring::S
    lo::L
    ws::W
    result::R
    rng::G
    alg::A
    seed::Q
    stops::C      # the local optimization's stop settings (`nothing`: the strategy's own)
end

__init(prob::EstimationProblem{<:Any, <:AbstractRansacProblem}, alg::Ransac; rng::Random.AbstractRNG = Random.default_rng(),
       stops = nothing) = _ransac_cache(prob, alg, false, rng, stops)
# The one-shot solve: the same run on a workspace built for this solve alone (`_one_shot_workspace`: the task's refit
# and block-coordinate caches, no LM built that it does not run).
__solve(prob::EstimationProblem{<:Any, <:AbstractRansacProblem}, alg::Ransac; rng::Random.AbstractRNG = Random.default_rng(),
        stops = nothing) = solve!(_ransac_cache(prob, alg, true, rng, stops))
function _ransac_cache(prob, alg::Ransac{T}, one_shot::Bool, rng, stops) where {T}
    problem, scoring, lo = _resolve(alg, prob.f, prob.p, stops)
    ws = one_shot ? _one_shot_workspace(T, problem, scoring, lo) : RansacWorkspace(T, problem, scoring, lo)
    _check_entry_composition(ws, scoring, lo)
    return RansacCache(problem, scoring, lo, ws, _result_buffers(ws, problem), rng, alg, nothing, stops)
end
# A local optimization runs natively (the RANSAC loop's local optimization from the seed) or, an `ExternalRefit` strategy, as its
# external refit (`_ExternalRefitCache`, below): the family of its strategy (VisualGeometryToolkitCore's `RefitFamily`).
__init(prob::RefineProblem{<:AbstractRansacProblem}, alg::LocalOptimization; stops = nothing, width = nothing) =
    _lo_cache(RefitFamily(alg.strategy), prob, alg, stops, width, false)
__solve(prob::RefineProblem{<:AbstractRansacProblem}, alg::LocalOptimization; stops = nothing, width = nothing) =
    solve!(_lo_cache(RefitFamily(alg.strategy), prob, alg, stops, width, true))   # one-shot: the task's refit caches
function _lo_cache(::NativeRefit, prob, alg::LocalOptimization, stops, width, one_shot::Bool)
    width === nothing || throw(ArgumentError(
        "LocalOptimization: a native local optimization takes its loss width from its strategy's width mode " *
            "(`with_width(strategy, w)`), not a `width` keyword; only an external refit takes one"))
    problem, scoring, lo = _resolve(alg, prob.f, prob.p, stops)
    ws = one_shot ? _one_shot_workspace(_DEFAULT_STORAGE_ELTYPE, problem, scoring, lo) : RansacWorkspace(problem, scoring, lo)
    _check_entry_composition(ws, scoring, lo)
    return RansacCache(problem, scoring, lo, ws, _result_buffers(ws, problem), nothing, alg, prob.u0, stops)
end
# The run's problem (its context applied), scorer and local optimization, resolved from the algorithm's fields for the
# data `f` — `ClassDefault()` against the problem's class and data (a class width can read the data: hand-eye's
# characteristic length) — at the solve's stop settings (`with_stops`), and the scorer checked against the problem: at
# `init` and at every `reinit!` with new data.
function _resolve(alg::Ransac, f, p, stops)
    problem = _with_context(f, p)
    scoring = _class_scoring(f, alg.scoring)
    lo = with_stops(_class_lo(f, alg.local_optimization_strategy, scoring), stops)
    _check_scoring(problem, scoring)
    return problem, scoring, lo
end
function _resolve(alg::LocalOptimization, f, p, stops)
    problem = _with_context(f, p)
    _check_scoring(problem, alg.scoring)
    return problem, alg.scoring, with_stops(alg.strategy, stops)
end

# A find on fewer data than a minimal sample is the no-model result (`(Failure, TrialBudget)`) of a search that drew no trial. A physical
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
    reinit!(cache::RansacCache[, θ₀]; f, p, alg = cache's, rng = cache's, stops = cache's) -> cache

Point a find's or a local optimization's cache at new data `f` (a family problem of the same type), a new context `p`
(the caller's form: Core's `reinit!(::EstimationCache)` hands over both, the cache's problem's where the call names none),
(a local optimization) a new seed `θ₀`, a new algorithm `alg` or (a find) a new `rng`.

`alg` is a new configuration of the cache's algorithm — a value of its concrete type (a `Ransac` or
`LocalOptimization` of other field types is an `ArgumentError`): a scorer at another width, another strategy setting,
another `config`; `stops` new stop settings of the local optimization (`with_stops`), of the type `init` was given. The scorer and local optimization are re-resolved from the (new) algorithm's fields for the (new) data
(a class width can read the data) and checked against the problem; the workspace and result buffers are kept, and grown only when the new data
outgrow them. `rng` replaces a find's rng (of the same type; a local optimization draws nothing and takes none). Every
refusal of the call's arguments comes before anything changes. A scorer whose workspace tables depend on its
configuration (a scale prior's) is checked against the kept workspace when the composition changes — at `init`, and
at a `reinit!` with a new algorithm, new stops or a grown workspace (`check_workspace`, an `ArgumentError`: init a new
cache for it) — never per `solve!`.

The next `solve!`'s result aliases the cache's result buffers until the solve after it.
"""
function reinit!(c::RansacCache{P, S, L, W, R, G, A, Q, C}, θ₀...; f, p, alg = c.alg, rng = c.rng, stops = c.stops) where {P, S, L, W, R, G, A, Q, C}
    length(θ₀) <= (c.seed === nothing ? 0 : 1) || throw(ArgumentError("reinit!: a find's cache takes no seed"))
    stops isa C || throw(ArgumentError(string("reinit!: the cache's local-optimization stops are a ", C, "; the new ones a ",
        typeof(stops), " (init a new cache with stops of that type)")))
    alg isa A || throw(ArgumentError(string("reinit!: the cache was initialized for a ", A, "; the new algorithm is a ",
        typeof(alg), " (init a new cache for an algorithm of another type)")))
    rng === c.rng || rng isa G || throw(ArgumentError(c.rng === nothing ?
        "reinit!: a local optimization draws nothing and takes no rng" :
        string("reinit!: the cache's rng is a ", G, "; the new one is a ", typeof(rng), " (init a new cache)")))
    problem, scoring, lo = _resolve(alg, f, p, stops)
    n = data_size(problem)
    grown = n > _capacity(c.ws)
    ws = grown ? RansacWorkspace(_storage(c), problem, scoring, lo) : c.ws
    (grown || alg !== c.alg || stops !== c.stops) && _check_entry_composition(ws, scoring, lo)   # a new composition only
    c.ws = ws
    length(c.result.residuals) == n || (c.result = _result_buffers(c.ws, problem))
    c.problem, c.scoring, c.lo, c.alg, c.rng, c.stops = problem, scoring, lo, alg, rng, stops
    isempty(θ₀) || (c.seed = only(θ₀))
    return c
end
_storage(c::RansacCache) = eltype(c.result.residuals)
_capacity(ws::RansacWorkspace) = length(ws.best_residuals)

# -----------------------------------------------------------------------------
# The external refit through `LocalOptimization` (an `ExternalRefit` strategy)
# -----------------------------------------------------------------------------

# The cache of an external refit: the problem, the scorer template, the strategy at the solve's stops, the refiner's
# workspace (`external_workspace`, built once for the frame), the seed and the width, and whether the refiner's
# preparation accepted this seed (`nothing`: not yet prepared — the one-shot solve prepares at its `solve!`).
mutable struct _ExternalRefitCache{P, X, S, L, W, Q, A, C}
    problem::P
    context::X     # the solve's context `p`: `(; frame)` or none
    scoring::S
    lo::L
    ws::W
    seed::Q
    width::Union{LossWidth{NTuple{2, Float64}}, FixedS2}   # a call's absolute width, or a fixed noise scale
    alg::A
    stops::C      # the strategy's stop settings (`nothing`: its own)
    prepared::Union{Nothing, Bool}
end
function _lo_cache(::ExternalRefit, prob, alg::LocalOptimization, stops, width, _one_shot::Bool)
    w = _external_width(width)
    problem, frame = prob.f, _external_frame(prob.p)   # the scorer is the refiner's template: it scores nothing here
    lo = with_stops(alg.strategy, stops)
    ws = external_workspace(lo, problem, frame, alg.scoring)
    return _ExternalRefitCache(problem, prob.p, alg.scoring, lo, ws, prob.u0, w, alg, stops, nothing)
end
_external_width(w::Union{LossWidth{NTuple{2, Float64}}, FixedS2}) = w
_external_width(w) = throw(ArgumentError(string(
    "LocalOptimization: an external refit runs at an absolute width — give `width = LossWidth(w)` (the loss width) or ",
    "a fixed noise scale `width = FixedS2(s2)`; got ", w === nothing ? "none" : repr(w))))
# The two-view frame an external refiner reads beyond the problem: the context's `frame`, or none.
_external_frame(::NullParameters) = nothing
function _external_frame(p::NamedTuple)
    keys(p) === (:frame,) || throw(ArgumentError(string("the context `p` of an external refit is `(; frame)` (the two-view ",
        "frame its refiner reads) or none; got the key(s) ", join(keys(p), ", "))))
    return p.frame
end

function reinit!(c::_ExternalRefitCache{P, X, S, L, W, Q, A, C}, θ₀ = c.seed; f = c.problem, p = c.context, alg = c.alg,
        width = c.width, stops::C = c.stops) where {P, X, S, L, W, Q, A, C}
    alg isa A || throw(ArgumentError(string("reinit!: the cache was initialized for a ", A, "; the new algorithm is a ",
        typeof(alg), " (init a new cache for an algorithm of another type)")))
    w = _external_width(width)
    lo = with_stops(alg.strategy, stops)
    if f !== c.problem || p !== c.context                   # new data or frame: the refiner's workspace rebuilt
        c.ws = external_workspace(lo, f, _external_frame(p), alg.scoring)
    end
    c.problem, c.context, c.scoring, c.lo, c.seed, c.width, c.alg, c.stops = f, p, alg.scoring, lo, θ₀, w, alg, stops
    c.prepared = prepare_external!(c.ws, c.lo, c.problem, c.seed, c.width, c.scoring)   # outside the timing bracket
    return c
end
function solve!(c::_ExternalRefitCache)
    kept = c.prepared
    prepared::Bool = kept === nothing ? prepare_external!(c.ws, c.lo, c.problem, c.seed, c.width, c.scoring) : kept
    c.prepared = nothing                                     # a later solve! without a reinit! prepares again
    T = _external_record_type(c)
    prepared || return _declined_record(T)                   # no refit made
    return _as_record(T, _external_refit(c, c.width))
end
# The refit at the call's width, the width's concrete type dispatched (the cache holds a small Union).
_external_refit(c, w) = external_refit!(c.ws, c.lo, c.problem, c.seed, w, c.scoring)

# ONE record type for the refit and the declined preparation, so the cache's `solve!` (and the front end's recorded
# result) is inferred and boxes nothing: the refiner's own `FitResult` type — inferred for both width forms, which must
# agree — with its model, residuals, problem, algorithm and raw output widened by `Nothing`. The declined record is that
# type with every one of them `nothing` (`u = nothing`: `found_model` reads it as no refit), `(InitialFailure, NoRule)`,
# no step. A refiner whose record type is not inferred keeps its own (the declined record then the plain no-model one).
function _external_record_type(c::_ExternalRefitCache{P, X, S, L, W, Q}) where {P, X, S, L, W, Q}
    R1 = Core.Compiler.return_type(external_refit!, Tuple{W, L, P, Q, LossWidth{NTuple{2, Float64}}, S})
    R2 = Core.Compiler.return_type(external_refit!, Tuple{W, L, P, Q, FixedS2, S})
    return R1 === R2 ? _widened(R1) : nothing
end
_widened(::Type{FitResult{U, M, R, Prob, Alg, O}}) where {U, M, R, Prob, Alg, O} =
    FitResult{Union{Nothing, U}, Union{Nothing, M}, Union{Nothing, R}, Union{Nothing, Prob}, Union{Nothing, Alg}, Union{Nothing, O}}
_widened(_) = nothing
_declined_record(::Type{T}) where {T <: FitResult} =
    T(nothing, nothing, ReturnCode.InitialFailure, SolveStats(), nothing, nothing, nothing, NoRule, NaN, 0, nothing)
_declined_record(::Nothing) = FitResult(nothing, ReturnCode.InitialFailure, NoRule, SolveStats(), NaN, 0, nothing)
_as_record(::Type{T}, r::FitResult) where {T <: FitResult} =
    T(r.u, r.resid, r.retcode, r.stats, r.prob, r.alg, r.original, r.stop_rule, r.rms, r.n, r.metadata)
_as_record(::Nothing, r) = r
