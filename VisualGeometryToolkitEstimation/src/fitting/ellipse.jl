# =============================================================================
# Ellipse fitting — find_ellipse (RANSAC) and refine_ellipse (its step alone)
# =============================================================================
#
# Both run on the `EllipseProblem` (problems/ellipse.jl): Sampson scoring, and a
# refit on the points' orthogonal distances to the ellipse (the maximum-
# likelihood objective under isotropic Gaussian noise on the points).
# =============================================================================

# =============================================================================
# find_ellipse — the 2D ellipse through contaminated points by RANSAC
# =============================================================================
#
#   find_ellipse — thin wrapper around ransac() on the `EllipseProblem`
#                  (problems/ellipse.jl): the 5-point ellipse, the Sampson
#                  distance of the point-on-ellipse constraint in pixels, the
#                  chart-LM refit — as `find_circle` (circle.jl).
# =============================================================================

"""
    find_ellipse(points; scoring,
        max_trials, min_trials,
        local_optimization_strategy, config = nothing)

Robust estimation of a 2D ellipse through points (a contour, edge points) by
RANSAC on the [`EllipseProblem`](@ref): 5-point samples (VisualGeometryToolkitCore's
`Ellipse5pt`, the conic through them when it is an ellipse), MSAC scoring of the
Sampson distance of `(x − c)ᵀ Q (x − c) − 1` — the geometric distance to the
ellipse to first order — and, as local optimization, the robustified LM
([`Triggs`](@extref VisualGeometryToolkitCore.Triggs)) on the centre and log shape of the points' ORTHOGONAL
distances to the ellipse (the problem's refit rows): one Levenberg–Marquardt
solve from the hypothesis over all the points under the Welsch loss at
`efficiency_constant·ŝ`, ŝ the Student-t scale of the hypothesis's Sampson errors.
Every default is `configs/default.toml`'s (`[ransac]`,
`[pipelines.ellipse]`); a keyword overrides. The circle's counterpart is
[`find_circle`](@ref).

Returns a `RansacEstimate` (its `estimate` an `Ellipse{Float64}`), the `:no_model` result (`iterations` 0)
for fewer than 5 points. `iterations(result)` is the number of trials (hypotheses
drawn); test `found_model(result)` before reading the model (see
[`RansacEstimate`](@ref)). From a given ellipse, the same step without the
sampling: [`refine_ellipse`](@ref).

# Arguments
- `points`: 2D points, `AbstractVector` of `StaticVector{2}` (`Point2{Float64}`
  storage canonicalized internally), plain: unit isotropic noise.
- `scoring`: the scorer, its width **in PIXELS** (the points' unit), as [`find_homography`](@ref); default `[pipelines.ellipse] scoring`.
- `workspace`: as [`find_homography`](@ref), built with `RansacWorkspace([T,] EllipseProblem, n_data)`.
- `local_optimization_strategy`: default `[pipelines.ellipse] local_optimization_strategy`; any
  other strategy as [`find_homography`](@ref).
- the other keywords as [`find_homography`](@ref).
"""
function find_ellipse(points::AbstractVector{<:StaticArrays.StaticVector{2, <:Real}}; kwargs...)
    return _guarded_ransac(() -> EllipseProblem(points), EllipseProblem, length(points), :ellipse; kwargs...)
end

"""
    refine_ellipse(ellipse, points; optimization_strategy, scale | width, workspace = nothing)
        -> FitResult{M, RefineMetadata}

The 2D ellipse through `points` refined from the given `ellipse`. Every
strategy descends the problem's refit rows, the points' ORTHOGONAL distances to
the ellipse (the distance to the foot point, the closest boundary point;
codimension 1), on the log-shape chart (VisualGeometryToolkitCore's
`EllipseManifold`, smooth at a circle): the keywords, the result and the refusals are those of every `refine_*`,
documented on [`refine_homography`](@ref) (default strategy:
`[refine.ellipse] optimization_strategy`). Fewer than 5 points: the guess back
unrefined, record `(ReturnCode.InitialFailure, NoRule)` (`retcode_symbol` `:degenerate`).

The maximum-likelihood ellipse under isotropic Gaussian noise on outlier-free
points is `refine_ellipse(direct_ellipse(points), points; optimization_strategy =
nothing)`: least squares over every point. Its covariance is the one model
covariance, `s²·solver_covariance(EllipseProblem(points), e, eachindex(points))`
(in `ambient_coordinates(e)`), `s²` the residual variance.
"""
function refine_ellipse(ellipse::VgtkCore.Ellipse, points::AbstractVector{<:StaticArrays.StaticVector{2, <:Real}}; kwargs...)
    return _guarded_refine(() -> EllipseProblem(points), EllipseProblem, length(points), Val(:ellipse), ellipse; kwargs...)
end
