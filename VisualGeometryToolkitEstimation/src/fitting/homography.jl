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
    find_homography(correspondences; scoring,
        extent=nothing, workspace=nothing,
        max_trials, min_trials,
        local_optimization_strategy, config=nothing)

Robust homography estimation via RANSAC with MSAC scoring and least-squares-on-the-inliers local optimization.
Every default is `configs/default.toml`'s (`[ransac]`,
`[pipelines.homography]`); a keyword overrides.

Returns a `RansacEstimate`, the `:no_model` result (`iterations` 0) for fewer than 4 correspondences.
`iterations(result)` is the number of trials (hypotheses drawn); test
`found_model(result)` before reading the model (see [`RansacEstimate`](@ref)).

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
- `scoring`: the RANSAC scorer, an object of VisualGeometryToolkitCore's scoring contract carrying its own width **in PIXELS** — the unit of this problem's residuals (every pixel-data `find_*` pipeline states its width by reference to this paragraph): a threshold scorer's `threshold` (`MsacScoring(HomographyProblem; threshold = 2.0)`, `RansacScoring`), the Student-t + uniform scorer's `outlier_halfwidth` (`StudentTUniformScoring(HomographyProblem, 5.0; scale_prior, label_prior)`), a paper's scorer (RANSACScoringDoneRight's families). Default: the class's scorer of record, `configs/default.toml` `[pipelines.homography] scoring` (OpenCV's convention, not a tuned value), read by [`scoring_from_toml`](@ref). A bare `threshold` or `outlier_halfwidth` keyword is refused (`ArgumentError`): a width belongs to its scorer.
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
  `[pipelines.<class>]` strategy — here least squares (`Irls` with the TLS loss) on
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
    return _guarded_ransac(() -> with_extent(HomographyProblem(pairs), extent), HomographyProblem{_cspond_scalar(pairs)}, length(pairs), :homography; kwargs...)
end

"""
    refine_homography(H, correspondences; optimization_strategy, scale | width, workspace = nothing)
        -> FitResult{HomographyMat{Float64}, RefineMetadata}

The homography `H` refined against `correspondences` from `H` itself — a refine:
the guess is required and leads, the model family is kept, and nothing is
estimated from scratch. It runs NO scorer: it estimates no scale, forms no
consensus set and drops no datum. Its robust loss runs over every datum at the
caller's scale, frozen. (The RANSAC local-optimization step from a seed — a
scorer, its consensus set, the strategy anchored on it — is
[`locally_optimize`](@ref), not a refine.)

Returns a `FitResult` with [`RefineMetadata`](@ref): the refined model, its
termination record (`retcode`, `stop_rule`: the LM's, whose `retcode_symbol` is
`:converged`, `:max_iterations`, `:stalled` or `:solve_failed`), `iterations` (the LM's accepted
iterations), `rms` the weighted residual rms
`√(Σ wᵢrᵢ² / Σ wᵢ)` and `n` the support (`count(w > 0)`), and in the metadata
the per-datum `weights` and `residuals` at the estimate and the scale record
(`noise_scale(result)`, `scale_source(result)`, `loss_width`). There is no inlier
mask: a loss weight has no cut of its own (`inlier_mask(result, cut)` states one).
Fewer than 4 correspondences: the guess back unrefined, record `(ReturnCode.InitialFailure, NoRule)` (`retcode_symbol` `:degenerate`),
`n` 0.

# Keywords — shared by every `refine_*`
- `optimization_strategy`: `Triggs{Shape}(…)` — ONE robustified chart LM over
  every datum (Core's `loss_robustifier`, handed to NLLSsolver); the default,
  `Triggs{WelschShape}()`, is the class's `configs/default.toml
  [refine.<class>] optimization_strategy` — or `nothing`, plain least squares
  over every datum (one chart LM). An `Irls{Shape}(…)` is an `ArgumentError`
  naming `Triggs{Shape}` (a refine runs one LM, no reweighting cycles); any
  other strategy needs a consensus set: an `ArgumentError` naming
  `locally_optimize`. Least squares is spelled `nothing`, not a Gaussian
  shape.
- `scale` or `width`: EXACTLY ONE for a robust strategy, neither for least
  squares; anything else is an `ArgumentError` (Core's `scale_input`).
  `scale = σ` is the residual's noise scale (here PIXELS), or an upstream
  robust result, whose `noise_scale` it takes (`scale = find_homography(…)`).
  The loss width is then the strategy's efficiency constant times σ: Core's
  `efficiency_constant(Shape, d)` at the problem's codimension `d`, or the
  constant the strategy states (`Triggs{K}(efficiency_constant = c)`, the plain
  multiple; TLS has none and needs one, or `width`). `width = w` is the loss
  width itself; the result then records no scale (`ScaleNone`), so a downstream
  stage needs its own.
- `workspace`: a `RansacWorkspace` (its chart-LM buffers are reused) or a
  chart-LM workspace; `nothing` draws the task's own.

Refused, each with the way to say it (`ArgumentError`): `scoring`, `threshold`,
`outlier_halfwidth` (a refine runs no scorer), `loss`, `loss_width`,
`efficiency_constant` (the loss is the strategy, its width `scale`/`width`),
`local_optimization`, `local_optimization_strategy` and `robust` (the strategy is `optimization_strategy`),
`extent` (an extent check validates a consensus set:
`locally_optimize(BoundedProblem(problem, extent), θ₀; …)`) and the sampling
keywords of `find_*` (`max_trials`, `min_trials`, `config`, `rng`).
"""
function refine_homography(
        H, pairs::AbstractVector{<:Pair{<:_PlanePoint, <:_PlanePoint}};
        kwargs...
    )
    _refuse_physical_plane(pairs, "refine_homography")
    return _guarded_refine(() -> HomographyProblem(pairs), HomographyProblem{_cspond_scalar(pairs)}, length(pairs), Val(:homography), H; kwargs...)
end
