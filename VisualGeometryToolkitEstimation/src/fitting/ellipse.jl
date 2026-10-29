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
    find_ellipse(points; threshold = 1.0,
        max_trials = 10000, min_trials = 100,
        local_optimization = Triggs{WelschShape}(scale_multiple = 2.5),
        scoring = nothing, config = nothing)

Robust estimation of a 2D ellipse through points (a contour, edge points) by
RANSAC on the [`EllipseProblem`](@ref): 5-point samples (VisualGeometryToolkitCore's
`Ellipse5pt`, the conic through them when it is an ellipse), MSAC scoring of the
Sampson distance of `(x − c)ᵀ Q (x − c) − 1` — the geometric distance to the
ellipse to first order — and, as local optimization, the robustified LM
([`Triggs`](@ref)) on the centre and log shape of the points' ORTHOGONAL
distances to the ellipse (the problem's refit rows): one Levenberg–Marquardt
solve from the hypothesis over all the points under the Welsch loss at
`scale_multiple·ŝ`, ŝ the Student-t scale of the hypothesis's Sampson errors.
Every default shown is `configs/default.toml`'s (`[ransac]`,
`[pipelines.ellipse]`); a keyword overrides. The circle's counterpart is
[`find_circle`](@ref).

Returns a `RansacEstimate` (its `estimate` an `Ellipse{Float64}`) or `nothing`
if fewer than 5 points. `iterations(result)` is the number of trials (hypotheses
drawn); test `converged(result)` before reading the model (see
[`RansacEstimate`](@ref)). From a given ellipse, the same step without the
sampling: [`refine_ellipse`](@ref).

# Arguments
- `points`: 2D points, `AbstractVector` of `StaticVector{2}` (`Point2{Float64}`
  storage canonicalized internally), plain: unit isotropic noise.
- `threshold`: **in PIXELS** (the points' unit), as [`find_homography`](@ref).
- `workspace`: as [`find_homography`](@ref), built with `RansacWorkspace(EllipseProblem, n_data[, T])`.
- `local_optimization`: default `Triggs{WelschShape}(scale_multiple = 2.5)`; any
  other strategy as [`find_homography`](@ref).
- the other keywords as [`find_homography`](@ref).
"""
function find_ellipse(points::AbstractVector{<:StaticArrays.StaticVector{2, <:Real}}; kwargs...)
    _holds_minimal_sample(EllipseProblem, length(points)) || return nothing
    return _ransac_pipeline(EllipseProblem(points), :ellipse; kwargs...)
end

"""
    refine_ellipse(ellipse, points; threshold = 1.0,
        local_optimization = Triggs{WelschShape}(scale_multiple = 2.5),
        scoring = nothing, workspace = nothing) -> RansacEstimate

The 2D ellipse through `points` refined from the given `ellipse`:
[`find_ellipse`](@ref)'s step without the sampling. `local_optimization` is the
ellipse's RANSAC local-optimization step, run here alone from the given model
(no sampling): `ellipse` is scored, its consensus set under the scorer is the
one a consensus-anchored strategy reads its scale from, and the strategy runs
once from it. Every strategy descends the problem's refit rows, the points'
ORTHOGONAL distances to the ellipse (the distance to the foot point, the
closest boundary point), on the log-shape chart (VisualGeometryToolkitCore's
`EllipseManifold`, smooth at a circle). Every default is `find_ellipse`'s
(`[pipelines.ellipse]`); the keywords are those of every `refine_*`, documented
on [`refine_homography`](@ref). `nothing` if fewer than 5 points.

The maximum-likelihood ellipse under isotropic Gaussian noise on outlier-free
points is `refine_ellipse(direct_ellipse(points), points; local_optimization =
Irls{GaussianShape}(), threshold = τ)`, τ above every residual: least squares over every point. Its
covariance is the one model covariance on the inliers,
`s²·solver_covariance(EllipseProblem(points), e, findall(result.inlier_mask))`
(in `ambient_coordinates(e)`), `s² = result.sigma²`.
"""
function refine_ellipse(ellipse::VgtkCore.Ellipse, points::AbstractVector{<:StaticArrays.StaticVector{2, <:Real}}; kwargs...)
    _holds_minimal_sample(EllipseProblem, length(points)) || return nothing
    return _refine_pipeline(EllipseProblem(points), :ellipse, ellipse; kwargs...)
end
