# =============================================================================
# Circle Fitting — the 2D circle through contaminated points by RANSAC
# =============================================================================
#
#   find_circle — thin wrapper around ransac() on the `CircleProblem`
#                 (problems/circle.jl): the 3-point circle, the Sampson
#                 distance of the point-on-circle constraint in pixels, the
#                 chart-LM refit.
# =============================================================================

"""
    find_circle(points; scoring = MsacScoring(problem; threshold = 1.0),
        max_trials = 10000, min_trials = 100,
        local_optimization_strategy = Triggs{WelschShape}(efficiency_constant = 2.5), config = nothing)

Robust estimation of a 2D circle through points (a contour, edge points) by
RANSAC on the [`CircleProblem`](@ref): 3-point samples (VisualGeometryToolkitCore's
`Circle3pt`, the circumcircle), MSAC scoring of the Sampson distance of
`‖x − c‖² − r²` — the geometric distance to the circle to first order — and,
as local optimization, the robustified LM ([`Triggs`](@ref)) on the centre
and log radius: one Levenberg–Marquardt solve from the hypothesis over all the
points under the Welsch loss at `efficiency_constant·ŝ`, ŝ the Student-t scale of the
hypothesis's Sampson errors. Every default
shown is `configs/default.toml`'s (`[ransac]`, `[pipelines.circle]`); a keyword
overrides.

Returns a `RansacEstimate` (its `estimate` a `Circle{Float64}`), the `:no_model` result (`iterations` 0) for
fewer than 3 points. `iterations(result)` is the number of trials (hypotheses
drawn); test `converged(result)` before reading the model (see
[`RansacEstimate`](@ref)).

# Arguments
- `points`: 2D points, `AbstractVector` of `StaticVector{2}` (`Point2{Float64}`
  storage canonicalized internally), plain: unit isotropic noise.
- `scoring`: the scorer, its width **in PIXELS** (the points' unit), as [`find_homography`](@ref); default `[pipelines.circle] scoring`.
- `workspace`: as [`find_homography`](@ref), built with `RansacWorkspace([T,] CircleProblem, n_data)`.
- `local_optimization_strategy`: default `Triggs{WelschShape}(efficiency_constant = 2.5)`; any
  other strategy as [`find_homography`](@ref).
- the other keywords as [`find_homography`](@ref).
"""
function find_circle(points::AbstractVector{<:StaticArrays.StaticVector{2, <:Real}}; kwargs...)
    return _guarded_ransac(() -> CircleProblem(points), CircleProblem, length(points), :circle; kwargs...)
end

"""
    refine_circle(circle, points; optimization_strategy = Triggs{WelschShape}(), scale | width, workspace = nothing)
        -> FitResult{M, RefineMetadata}

The 2D circle through `points` refined from the given `circle` (the points'
distances to it, pixels; codimension 1): the keywords, the result and the refusals are those of every `refine_*`,
documented on [`refine_homography`](@ref) (default strategy:
`[refine.circle] optimization_strategy`). Fewer than 3 points: the guess back
unrefined, `retcode` `:degenerate`.
"""
function refine_circle(circle::VgtkCore.Circle, points::AbstractVector{<:StaticArrays.StaticVector{2, <:Real}}; kwargs...)
    return _guarded_refine(() -> CircleProblem(points), CircleProblem, length(points), Val(:circle), circle; kwargs...)
end
