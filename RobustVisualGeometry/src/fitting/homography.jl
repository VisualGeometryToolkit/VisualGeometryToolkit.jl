# =============================================================================
# Homography Fitting
# =============================================================================
#
# Robust homography estimation via RANSAC.
#
#   find_homography — thin wrapper around ransac() with composable scoring
#                    and local optimization
#
# =============================================================================

# =============================================================================
# Public API
# =============================================================================

"""
    find_homography(correspondences; threshold=3.0,
        extent=nothing, workspace=nothing,
        max_trials=10000, min_trials=100,
        local_optimization=Irls{TLSShape}(),
        scoring=nothing, config=nothing)

Robust homography estimation via RANSAC with MSAC scoring and least-squares-on-the-inliers local optimization.
Every default shown is `configs/default.toml`'s (`[ransac]`,
`[pipelines.homography]`); a keyword overrides.

Returns a `RansacEstimate` or `nothing` if fewer than 4 correspondences.
`iterations(result)` is the number of trials (hypotheses drawn); test
`converged(result)` before reading the model (see [`RansacEstimate`](@ref)).

# Arguments
- `correspondences`: 2D-2D correspondences — `Csponds`, `StructArray{<:Pair}` or a
  Pair vector (`Point2` storage canonicalized internally). A `Csponds` `:score`
  attribute column is the per-match prior (PROSAC, descending score); uniform
  otherwise. Endpoints carry their MEASUREMENT NOISE (VisualGeometryCore
  `conventions/measurement-noise.md`): a plain point is unit isotropic noise in
  its own frame, `Uncertain(p, CovMat(Σ))` its own covariance, `exact(p)` none.
  Image↔image matches stay plain; a board, model or world plane is `exact` —
  `find_homography(exact.(board) .=> pixels)` scores the transfer error in
  pixels whatever the board's units (a plain metre board would be scored in
  metres, its threshold silently scaled by the board→image scale).
- `threshold`: the MSAC inlier threshold of the default scoring, **in PIXELS** — the unit of this problem's residuals (every pixel-data `find_*` pipeline states its threshold by reference to this paragraph). The file default is a convention (OpenCV's), not a tuned value. Builds the default scoring only; supplying it with `scoring` is an `ArgumentError`, and so is the retired `outlier_halfwidth` (a marginal scorer's half-width belongs to that scoring).
- `extent`: optional `HyperRectangle{2}` world-plane region; when given the problem
  is wrapped in `BoundedProblem` so the fitted H is validated over the rectangle.
- `workspace`: optional pre-allocated `RansacWorkspace` to reuse —
  `RansacWorkspace(HomographyProblem{Float64}, n_data[, T = Float32])`, `T` the
  per-point storage precision, holds the DEFAULT (MSAC) scorer's workspace; for
  another scoring or local optimization build it from them,
  `RansacWorkspace(problem, scoring, local_optimization)`. Each call checks the
  workspace against its scoring (VisualGeometryCore `check_workspace`).
- `max_trials`, `min_trials`: RANSAC trial bounds (`[ransac]`).
- `local_optimization`: LO-RANSAC strategy (default: the class's
  `[pipelines.<class>]` strategy — here `Irls{TLSShape}()`, least squares on
  the inliers with its truncation at the scorer's decision point; any other
  `AbstractLocalOptimization` changes mechanism — LocalOptimizationDoneRight's
  half-quadratic lifts among them).
- `scoring`: Override scoring function (default: VisualGeometryCore's `MsacScoring(problem; threshold)`; a paper's scorer, e.g. RansacScoringDoneRight's `StudentTUniformScoring`, plugs in here).
- `config`: Override `RansacConfig` (ignores `max_trials`/`min_trials`).
- `rng`: the sampler's random number generator.

These keywords are shared by every `find_*` pipeline; the others document
theirs by reference to this list, stating only their class's defaults.

# Examples
```julia
# Bare pairs, no prior (uniform sampling)
result = find_homography(pairs)

# Descriptor-similarity prior as a `:score` column (PROSAC sampling) — the
# matcher's `Csponds(...; score, ratio)` output carries it already; from bare
# pairs, build the columnar container (`withattrs` takes a `Csponds`)
result = find_homography(Csponds(sources(pairs), targets(pairs); score = similarities))
```
"""
function find_homography(
        pairs::AbstractVector{<:Pair{<:_PlanePoint, <:_PlanePoint}};
        extent::Union{Nothing, HyperRectangle{2}} = nothing,
        kwargs...
    )
    _holds_minimal_sample(HomographyProblem, length(pairs)) || return nothing
    return _ransac_pipeline(_with_extent(HomographyProblem(pairs), extent), :homography; kwargs...)
end
