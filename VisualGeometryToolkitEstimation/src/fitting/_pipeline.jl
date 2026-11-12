# =============================================================================
# Shared RANSAC fitting pipeline
# =============================================================================
#
# `_ransac_pipeline` is the shared body of every `find_*` (every `refine_*`'s is
# `_guarded_refine`, _refine.jl). Each public wrapper
# builds its problem and names its model CLASS once; the class selects the
# wrapper's `configs/default.toml` `[pipelines.<class>]` table, which supplies
# both defaults — the `scoring` table (read by `scoring_from_toml`) and the
# `local_optimization_strategy` table (read by `optimization_strategy_from_toml`): the one
# robust-estimation schema. The shared keywords are documented once, on
# [`find_homography`](@ref).
# =============================================================================

"""
    _ransac_pipeline(problem, class::Symbol; scoring, local_optimization_strategy,
                     max_trials, min_trials, config, workspace, rng)
        -> RansacEstimate

Run RANSAC with the standard defaults shared by every `find_*` pipeline, read
from `[pipelines.<class>]` for the class the wrapper names:

- `scoring` defaults to the class's scorer (`_class_scoring`: its `scoring`
  table, read by [`scoring_from_toml`](@ref) for the problem's type, its width
  in the problem's residual unit). A scorer carries its own width — a threshold
  scorer's `threshold`, the Student-t + uniform scorer's `outlier_halfwidth` —
  so a bare `threshold` or `outlier_halfwidth` keyword is refused
  (`ArgumentError`, naming `scoring`).
- `local_optimization_strategy` defaults to the class's strategy (`_dflt_lo(class)`).
- `config` resolves to `RansacConfig(; max_trials, min_trials)`.
"""
function _ransac_pipeline(
        problem, class::Symbol;
        scoring::Union{AbstractScoring, Nothing} = nothing,
        # The class's strategy of record (configs/default.toml [pipelines.<class>]).
        local_optimization_strategy::Union{Nothing, AbstractOptimizationStrategy} = _dflt_lo(class),
        local_optimization = _RetiredKeyword(),   # refused, naming its successor
        # Defaults are configs/default.toml [ransac] — the same authority
        # RansacConfig() reads, so there is no second "default" to disagree.
        max_trials::Int = _dflt("ransac", "max_trials"),
        min_trials::Int = _dflt("ransac", "min_trials"),
        config::Union{RansacConfig, Nothing} = nothing,
        workspace = nothing,
        rng::Random.AbstractRNG = Random.default_rng(),
        threshold = nothing, outlier_halfwidth = nothing,   # refused, with the way to say it (`_refuse_width_keywords`)
    )
    _refuse_width_keywords(threshold, outlier_halfwidth)
    _refuse_retired_keyword(local_optimization, :local_optimization, :local_optimization_strategy)
    scoring = _class_scoring(problem, class, scoring)
    config = something(config, RansacConfig(; max_trials, min_trials))
    problem isa _TooFew && return _too_few(problem, scoring, workspace)
    return ransac(problem, scoring; config, local_optimization_strategy, workspace, rng)
end

# The scoring `_ransac_pipeline` runs: the caller's, or the class's scorer of
# record — configs/default.toml `[pipelines.<class>] scoring`, read by the one
# scorer reader for the problem's type — its width in the problem's residual
# unit (`_in_problem_unit`).
_class_scoring(_problem, _class::Symbol, scoring::AbstractScoring) = scoring
_class_scoring(problem, class::Symbol, ::Nothing) =
    _in_problem_unit(scoring_from_toml(_dflt("pipelines.$class", "scoring"); problem = typeof(problem),
                                       section = "pipelines.$class.scoring"), problem)

# The unit of a class's file width (`threshold` or `outlier_halfwidth`): the
# problem's residual unit (pixels) for every pixel-data class — E and PnP
# included, whose residuals are whitened into pixels by the cameras — where the
# file value is the width itself. Two classes state a UNIT-FREE width:
# hand-eye's is a FRACTION of the problem's characteristic length `L` (median
# rig translation per motion), so the same in metres or millimetres; the
# bearing pose's a fraction of its residual's bound (problems/bearing_pose.jl).
# A caller's scorer is the caller's, in the residual's unit, never rescaled.
_in_problem_unit(sc, problem) = _in_unit(sc, _width_unit(problem))
_in_unit(sc, ::Nothing) = sc
function _in_unit(sc::StudentTUniformScoring, u::Float64)
    h = outlier_halfwidth(sc)   # a class table states a supplied half-width; an estimated support has none to scale
    h === nothing && throw(ArgumentError("a class scorer whose outlier model estimates its support has no unit-free width to scale"))
    return VgtkCore.with_width(sc, VgtkCore.OutlierHalfwidth(h * u))
end
_in_unit(sc::RansacScoring{DG}, u::Float64) where {DG} = RansacScoring(DG; threshold = sc.threshold * u)
_in_unit(sc::MsacScoring{DG}, u::Float64) where {DG} = MsacScoring(DG; threshold = sc.threshold * u)
_width_unit(_problem) = nothing
_width_unit(problem::HandEyeProblem) = Float64(problem.char_length)

# The too-little-data rule of the `find_*` pipelines is
# `_holds_minimal_sample` (estimators/ransac/interface.jl): fewer data than the
# problem type's minimal sample hold no hypothesis, so the pipeline returns the
# `:no_model` result of a search that drew no trial — `iterations` 0, no
# sample-acceptance rate — rather than let the problem's constructor refuse them.
# The pipeline runs on `_TooFew{P}(n)` in place of the problem, so its keywords
# are read and refused exactly as on enough data (a retired, a width or an
# unknown keyword raises the same error); only then is the result built from the
# problem TYPE (a constructor may throw on so few data), in the storage eltype of
# the caller's `workspace`, else `RansacWorkspace`'s default, and under the
# scorer the search would run (the caller's, or the class's scorer of record —
# its TYPE is all a result without a model reads, so the problem-unit rescale of
# its width is not needed), so it is the type a search on the same data returns.
struct _TooFew{P}
    n::Int
end
_too_few(t::_TooFew{P}, scoring::AbstractScoring, workspace) where {P} =
    _no_model_result(model_type(P), scoring, RansacResultBuffers{_storage_eltype(workspace)}(t.n), 0, NaN)
_class_scoring(::_TooFew{P}, class::Symbol, ::Nothing) where {P} =
    scoring_from_toml(_dflt("pipelines.$class", "scoring"); problem = P, section = "pipelines.$class.scoring")

# The guarded entry of a `find_*` on `n` data of the problem type `P`: the shared
# pipeline on `problem()`, built only when they hold a minimal sample (a
# constructor may refuse so few data), else on `_TooFew{P}(n)`. The refine_*
# entry is `_guarded_refine` (_refine.jl).
_guarded_ransac(problem, ::Type{P}, n::Integer, class::Symbol; kwargs...) where {P} =
    _holds_minimal_sample(P, n) ? _ransac_pipeline(problem(), class; kwargs...) : _ransac_pipeline(_TooFew{P}(n), class; kwargs...)
_storage_eltype(::Nothing) = _DEFAULT_STORAGE_ELTYPE   # RansacWorkspace([T = Float32,] P, n_data)
_storage_eltype(::RansacWorkspace{<:Any, <:Any, T}) where {T} = T

# The `extent` keyword of the `find_*` pipelines: when given, the problem is
# wrapped in `BoundedProblem`, whose `extent_trait` makes RANSAC's post-IRLS
# extent-consensus hook validate the fitted model over the region (the
# `_check_extent` method for the model class, extent_traits.jl and the problem
# files).
_with_extent(p, ::Nothing) = p
_with_extent(p, extent) = BoundedProblem(p, extent)

# A supplied workspace carries the workspace of the scorer it was built for —
# `RansacWorkspace(P, n_data)` the default MSAC scoring's. Each entry checks it
# against the scoring it runs (VisualGeometryToolkitCore `check_workspace`); for another
# scorer or local optimizer build it from them
# (`RansacWorkspace(problem, scoring, local_optimization_strategy)`).
