# =============================================================================
# Shared RANSAC fitting pipeline
# =============================================================================
#
# `_ransac_pipeline` is the shared body of every `find_*`. Each public wrapper
# builds its problem and names its model CLASS once; the class selects the
# wrapper's `configs/default.toml` `[pipelines.<class>]` table, which supplies
# both defaults — the MSAC `threshold` and `local_optimization`. The shared
# keywords are documented once, on [`find_homography`](@ref).
# =============================================================================

"""
    _ransac_pipeline(problem, class::Symbol; threshold, local_optimization,
                     max_trials, min_trials, scoring, config, workspace, rng)
        -> RansacEstimate

Run RANSAC with the standard defaults shared by every `find_*` pipeline, read
from `[pipelines.<class>]` for the class the wrapper names:

- `scoring` resolves to MSAC at `threshold` (`MsacScoring(problem; threshold)`,
  VisualGeometryToolkitCore) when omitted, the caller's `threshold` else the class's
  file value — a convention, not a tuned value — in the problem's residual unit
  (`_threshold_unit`: pixels for the pixel-data classes). A caller's `scoring`
  carries its own width, so supplying `threshold` with it is an `ArgumentError`.
- `local_optimization` defaults to the class's strategy (`_dflt_lo(class)`).
- `config` resolves to `RansacConfig(; max_trials, min_trials)`.
"""
function _ransac_pipeline(
        problem, class::Symbol;
        threshold::Union{Real, Nothing} = nothing,
        outlier_halfwidth::Union{Real, Nothing} = nothing,   # refused (`_default_scoring`), with the way to say it
        # The class's strategy of record (configs/default.toml [pipelines.<class>]).
        local_optimization::AbstractLocalOptimization = _dflt_lo(class),
        # Defaults are configs/default.toml [ransac] — the same authority
        # RansacConfig() reads, so there is no second "default" to disagree.
        max_trials::Int = _dflt("ransac", "max_trials"),
        min_trials::Int = _dflt("ransac", "min_trials"),
        scoring::Union{AbstractScoring, Nothing} = nothing,
        config::Union{RansacConfig, Nothing} = nothing,
        workspace = nothing,
        rng::Random.AbstractRNG = Random.default_rng()
    )
    scoring = _class_scoring(problem, class, scoring, threshold, outlier_halfwidth)
    config = something(config, RansacConfig(; max_trials, min_trials))
    return ransac(problem, scoring; config, local_optimization, workspace, rng)
end

"""
    _refine_pipeline(problem, class::Symbol, θ₀; threshold, local_optimization,
                     scoring, workspace) -> RansacEstimate

The shared body of every `refine_*`: the class's RANSAC local-optimization step
run ONCE from the given model `θ₀` — what the loop runs on its hypothesis, with
no sampling ([`locally_optimize`](@ref): score θ₀, take ITS consensus set under
the scorer, `locally_optimize!` from θ₀, re-score, `test_consensus`; the seed
comes back when the refined model is not kept). The defaults are the class's
`find_*`'s, read from the same `[pipelines.<class>]` table (`threshold`,
`local_optimization`); the keywords are `_ransac_pipeline`'s minus the
sampling ones (`max_trials`, `min_trials`, `config` — a `RansacConfig` is the
trial schedule and nothing else — and `rng`: no local optimization draws from
the loop's generator; `locally_optimize!` takes none).
"""
function _refine_pipeline(
        problem, class::Symbol, θ₀;
        threshold::Union{Real, Nothing} = nothing,
        outlier_halfwidth::Union{Real, Nothing} = nothing,   # refused (`_default_scoring`), with the way to say it
        local_optimization::AbstractLocalOptimization = _dflt_lo(class),
        scoring::Union{AbstractScoring, Nothing} = nothing,
        workspace = nothing
    )
    scoring = _class_scoring(problem, class, scoring, threshold, outlier_halfwidth)
    return locally_optimize(problem, θ₀; local_optimization, scoring, workspace)
end

# The scoring both pipelines run: the caller's, or MSAC at the caller's
# `threshold`, else at the class's file value in the problem's residual unit.
function _class_scoring(problem, class::Symbol, scoring, threshold, outlier_halfwidth)
    scoring === nothing && outlier_halfwidth === nothing && (
        threshold = something(threshold, _dflt("pipelines.$class", "threshold") * _threshold_unit(problem))
    )
    return _default_scoring(problem, scoring, threshold, outlier_halfwidth)
end

# The unit of a class's file `threshold`: the problem's residual unit (pixels)
# for every pixel-data class — E and PnP included, whose residuals are whitened
# into pixels by the cameras. Hand-eye's file value is unit-free, a FRACTION of
# the problem's characteristic length `L` (median rig translation per motion),
# so its threshold is the same in metres or millimetres.
_threshold_unit(_problem) = 1.0
_threshold_unit(problem::HandEyeProblem) = problem.char_length

# The default scorer of the core entries: MSAC at threshold `τ` — the same
# truncated quadratic the default local optimization, `Irls{TLSShape}`, refits.
_default_scoring_at(problem, τ::Float64) = MsacScoring(problem; threshold = τ)

# The too-little-data rule of the `find_*` pipelines is `_holds_minimal_sample`
# (estimators/ransac/interface.jl): fewer correspondences than the problem type's minimal
# sample hold no hypothesis, so the pipeline returns `nothing` rather than let the
# problem's constructor refuse them.

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
# (`RansacWorkspace(problem, scoring, local_optimization)`).
