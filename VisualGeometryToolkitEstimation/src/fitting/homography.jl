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
    find_homography(correspondences; scoring = MsacScoring(problem; threshold = 3.0),
        extent=nothing, workspace=nothing,
        max_trials=10000, min_trials=100,
        local_optimization_strategy=Irls{TLSShape}(), config=nothing)

Robust homography estimation via RANSAC with MSAC scoring and least-squares-on-the-inliers local optimization.
Every default shown is `configs/default.toml`'s (`[ransac]`,
`[pipelines.homography]`); a keyword overrides.

Returns a `RansacEstimate`, the `:no_model` result (`iterations` 0) for fewer than 4 correspondences.
`iterations(result)` is the number of trials (hypotheses drawn); test
`converged(result)` before reading the model (see [`RansacEstimate`](@ref)).

# Arguments
- `correspondences`: 2D-2D correspondences — `Csponds`, `StructArray{<:Pair}` or a
  Pair vector (`Point2` storage canonicalized internally). A `Csponds` `:score`
  attribute column is the per-match prior (PROSAC, descending score); uniform
  otherwise. Endpoints carry their MEASUREMENT NOISE (VisualGeometryToolkitCore
  `conventions/measurement-noise.md`): a plain point is unit isotropic noise in
  its own frame, `Uncertain(p, CovMat(Σ))` its own covariance, `exact(p)` none.
  Image↔image matches stay plain; a board, model or world plane is `exact` —
  `find_homography(exact.(board) .=> pixels)` scores the transfer error in
  pixels whatever the board's units (a plain metre board would be scored in
  metres, its threshold silently scaled by the board→image scale).
- `scoring`: the RANSAC scorer, an object of VisualGeometryToolkitCore's scoring contract carrying its own width **in PIXELS** — the unit of this problem's residuals (every pixel-data `find_*` pipeline states its width by reference to this paragraph): a threshold scorer's `threshold` (`MsacScoring(HomographyProblem; threshold = 2.0)`, `RansacScoring`), the Student-t + uniform scorer's `outlier_halfwidth` (`StudentTUniformScoring(HomographyProblem, 5.0; scale_prior, label_prior)`), a paper's scorer (RANSACScoringDoneRight's families). Default: the class's scorer of record, `configs/default.toml` `[pipelines.homography] scoring` (`{ type = "msac", threshold = 3.0 }`: OpenCV's convention, not a tuned value), read by [`scoring_from_toml`](@ref). A bare `threshold` or `outlier_halfwidth` keyword is refused (`ArgumentError`): a width belongs to its scorer.
- `extent`: optional `HyperRectangle{2}` world-plane region; when given the problem
  is wrapped in `BoundedProblem` so the fitted H is validated over the rectangle.
- `workspace`: optional pre-allocated `RansacWorkspace` to reuse —
  `RansacWorkspace([T = Float32,] HomographyProblem{Float64}, n_data)`, `T` the
  per-point storage precision, holds the DEFAULT (MSAC) scorer's workspace; for
  another scoring or local optimization build it from them,
  `RansacWorkspace(problem, scoring, local_optimization_strategy)`. Each call checks the
  workspace against its scoring (VisualGeometryToolkitCore `check_workspace`).
- `max_trials`, `min_trials`: RANSAC trial bounds (`[ransac]`).
- `local_optimization_strategy`: LO-RANSAC strategy (default: the class's
  `[pipelines.<class>]` strategy — here `Irls{TLSShape}()`, least squares on
  the inliers with its truncation at the scorer's decision point: a threshold
  scorer's threshold, the Student-t scorer's MAP crossover at the consensus
  partition's scale; any other
  `AbstractOptimizationStrategy` changes mechanism — LocalOptimizationDoneRight's
  half-quadratic lifts among them).
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

Units: the source points are plain numbers. A homography maps from the source
plane, so its entries carry that plane's unit, which a matrix cannot record:
physical (unitful) source points are an `ArgumentError` — strip them in the
unit the rest of your plain data uses — metres, as the stack's boundaries strip
(`ustrip.(m, X)`).
"""
function find_homography(
        pairs::AbstractVector{<:Pair{<:_PlanePoint, <:_PlanePoint}};
        extent::Union{Nothing, HyperRectangle{2}} = nothing,
        kwargs...
    )
    _refuse_physical_plane(pairs, "find_homography")
    return _guarded_ransac(() -> _with_extent(HomographyProblem(pairs), extent), HomographyProblem{_cspond_scalar(pairs)}, length(pairs), :homography; kwargs...)
end

"""
    refine_homography(H, correspondences; scoring = MsacScoring(problem; threshold = 3.0), extent = nothing,
        optimization_strategy = Irls{TLSShape}(), workspace = nothing)
        -> RansacEstimate

The homography `H` refined against `correspondences`: [`find_homography`](@ref)'s
step without the sampling. Every `refine_*` is its `find_*`'s RANSAC
local-optimization step run ONCE from a given model — [`locally_optimize`](@ref)
with the `find_*`'s defaults — so the guess is required and leads (a refine: the
model family is kept), and the data follow as the `find_*` takes them.

`optimization_strategy` is the homography's RANSAC local-optimization step, run
here alone from the given model (no sampling): `H` is scored under the scorer,
ITS consensus set is the one a consensus-anchored strategy reads its scale ŝ
from (the efficiency-constant widths, [`Triggs`](@ref)), exactly as the RANSAC loop
scores a hypothesis, and the strategy runs once from `H`. The refined model is
re-scored and kept when it succeeds at the problem's consensus test; otherwise
`H` comes back. When `H` holds no consensus beyond a minimal sample, nothing is
refined. `lo_retcode` on the result is the strategy's stop reason.

Returns a `RansacEstimate` (one trial; `retcode` `:converged`, the consensus
set, residuals and weights of the returned model under the scorer), the `:no_model` result (`iterations` 0)
for fewer than 4 correspondences.

# Keywords — shared by every `refine_*`
- `scoring`: any scorer of VisualGeometryToolkitCore's contract, its width in
  the problem's residual unit (here PIXELS), as [`find_homography`](@ref);
  default the class's (`[pipelines.<class>] scoring`). A bare `threshold` or
  `outlier_halfwidth` keyword is refused.
- `optimization_strategy`: any `AbstractOptimizationStrategy` the problem hosts —
  `Irls{K}`, `Triggs{K}`, `nothing`, LocalOptimizationDoneRight's
  half-quadratic lifts and GNC; a strategy the problem cannot host fails as it
  would inside `find_*`. Default: the class's `[pipelines.<class>]` strategy.
  The strategy's own knobs are its own (`Irls`'s `max_iter`; the chart LM's
  cap is `[chart_lm]`): a `refine_*` has no `max_iter`.
- `workspace`: a `RansacWorkspace` to reuse, as [`find_homography`](@ref).

The sampling keywords of `find_*` (`max_trials`, `min_trials`, `config`, `rng`)
have no meaning here and are not accepted. Model-specific keywords are those of
the `find_*` (here `extent`).
"""
function refine_homography(
        H, pairs::AbstractVector{<:Pair{<:_PlanePoint, <:_PlanePoint}};
        extent::Union{Nothing, HyperRectangle{2}} = nothing,
        kwargs...
    )
    _refuse_physical_plane(pairs, "refine_homography")
    return _guarded_refine(() -> _with_extent(HomographyProblem(pairs), extent), HomographyProblem{_cspond_scalar(pairs)}, length(pairs), :homography, H; kwargs...)
end
