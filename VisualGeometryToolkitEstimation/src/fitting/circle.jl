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
    find_circle(points; threshold = 1.0,
        max_trials = 10000, min_trials = 100,
        local_optimization = TriggsLM{CauchyShape}(scale_multiple = 2.5),
        scoring = nothing, config = nothing)

Robust estimation of a 2D circle through points (a contour, edge points) by
RANSAC on the [`CircleProblem`](@ref): 3-point samples (VisualGeometryToolkitCore's
`Circle3pt`, the circumcircle), MSAC scoring of the Sampson distance of
`‖x − c‖² − r²` — the geometric distance to the circle to first order — and,
as local optimization, the robustified LM ([`TriggsLM`](@ref)) on the centre
and log radius: one Levenberg–Marquardt solve from the hypothesis over all the
points under the Cauchy loss at `scale_multiple·ŝ`, ŝ the Student-t scale of the
hypothesis's Sampson errors. Every default
shown is `configs/default.toml`'s (`[ransac]`, `[pipelines.circle]`); a keyword
overrides.

Returns a `RansacEstimate` (its `estimate` a `Circle{Float64}`) or `nothing` if
fewer than 3 points. `iterations(result)` is the number of trials (hypotheses
drawn); test `converged(result)` before reading the model (see
[`RansacEstimate`](@ref)).

# Arguments
- `points`: 2D points, `AbstractVector` of `StaticVector{2}` (`Point2{Float64}`
  storage canonicalized internally), plain: unit isotropic noise.
- `threshold`: **in PIXELS** (the points' unit), as [`find_homography`](@ref).
- `workspace`: as [`find_homography`](@ref), built with `RansacWorkspace(CircleProblem, n_data[, T])`.
- `local_optimization`: default `TriggsLM{CauchyShape}(scale_multiple = 2.5)`; any
  other strategy as [`find_homography`](@ref).
- the other keywords as [`find_homography`](@ref).
"""
function find_circle(points::AbstractVector{<:StaticArrays.StaticVector{2, <:Real}}; kwargs...)
    _holds_minimal_sample(CircleProblem, length(points)) || return nothing
    return _ransac_pipeline(CircleProblem(points), :circle; kwargs...)
end
