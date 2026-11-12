# =============================================================================
# Radial-Homography Fitting — find_radial_homography
# =============================================================================
#
# Robust one-sided radially-distorted homography estimation via RANSAC —
# the (H, k) sibling of `find_homography`, over `RadialHomographyProblem`.
# =============================================================================

"""
    find_radial_homography(correspondences; center,
        scoring, extent=nothing,
        workspace=nothing, extra_check=nothing, minimal=:radial,
        local_optimization_strategy, max_trials, min_trials, config=nothing, rng=...)

`scoring` (its width in pixels) and `local_optimization_strategy` default to
`configs/default.toml` `[pipelines.radial_homography]`, `max_trials` and
`min_trials` to its `[ransac]` table; a keyword overrides. A bare `threshold`
or `outlier_halfwidth` keyword is refused, as [`find_homography`](@ref). `extra_check`
is the radial 5-point kernel's root filter (see
[`RadialHomographyProblem`](@ref)) and is refused with `minimal = :homography`.

Robust radially distorted homography estimation (one-sided one-parameter
division model on the TARGET side; [Wadenback2026](@citet)) via RANSAC
with MSAC scoring. `minimal = :homography` pairs the global search for a plain homography (four-point
hypotheses, `Homography4pt`) with this problem's (H, k) local optimization: the
hypothesis carries no distortion term, k is seeded on its consensus by the
closed-form least squares in k given H and the joint manifold step estimates both,
and the correspondences a plane homography cannot hold under distortion enter at
the re-score. The default `:radial` samples five for the 5-point kernel, whose
root pick fails on boards too small to identify k from alone.

Returns a `RansacEstimate` whose model is
a `VisualGeometryToolkitCore.RadialHomography` in the DATA (pixel) frame — `H`,
`k`, and the distortion `center` all carried in the model — the `:no_model` result (`iterations` 0)
for fewer than a minimal sample (5 correspondences, 4 with `minimal = :homography`).
`iterations(result)` is the number of trials (hypotheses drawn); test
`converged(result)` before reading the model (see [`RansacEstimate`](@ref)).

`center` — the distortion center in target pixels — is REQUIRED: the 5-pt
solver runs in the Hartley frame about it.

`local_optimization_strategy` defaults to `Irls{TLSShape}()` (least squares on the
inliers) on the 9-dof sl(3) ⊕ k manifold, via the joint 9-dof LM M-step. The
chart-rows provider makes LocalOptimizationDoneRight's lifts available on the
same manifold.

The other keywords as [`find_homography`](@ref).

Units: the source points are plain numbers. A radial homography maps from the source
plane, so its entries carry that plane's unit, which a matrix cannot record:
physical (unitful) source points are an `ArgumentError` — strip them in the
unit the rest of your plain data uses — metres, as the stack's boundaries strip
(`ustrip.(m, X)`).
"""
function find_radial_homography(
        pairs::AbstractVector{
            <:Pair{<:_PlanePoint, <:_PlanePoint},
        };
        center,
        extent::Union{Nothing, HyperRectangle{2}} = nothing,
        extra_check::Union{Nothing, Bool} = nothing,
        minimal::Symbol = :radial,
        kwargs...
    )
    _refuse_physical_plane(pairs, "find_radial_homography")
    return _guarded_ransac(() -> _with_extent(RadialHomographyProblem(pairs; center, extra_check, minimal), extent), RadialHomographyProblem{_cspond_scalar(pairs), _radial_sample_size(minimal)}, length(pairs), :radial_homography; kwargs...)
end

"""
    refine_radial_homography(model, correspondences; optimization_strategy, scale | width, workspace = nothing)
        -> FitResult{M, RefineMetadata}

The radially distorted homography `model` (VisualGeometryToolkitCore
`RadialHomography{<:Any, 1}`: `H`, `k` and the distortion centre, in the data
frame; a model with more division coefficients has no method, a `MethodError`)
refined against `correspondences` from `model` itself (the reprojection pair,
codimension 2). The problem's distortion centre is the model's own
(`center(model)`): the keywords, the result and the refusals are those of every `refine_*`,
documented on [`refine_homography`](@ref) (default strategy:
`[refine.radial_homography] optimization_strategy`). Fewer than 5 correspondences: the guess back
unrefined, `retcode` `:degenerate`.
"""
function refine_radial_homography(
        model::VgtkCore.RadialHomography{<:Any, 1}, pairs::AbstractVector{<:Pair{<:_PlanePoint, <:_PlanePoint}};
        kwargs...
    )
    _refuse_physical_plane(pairs, "refine_radial_homography")
    return _guarded_refine(() -> RadialHomographyProblem(pairs; center = center(model)), RadialHomographyProblem{_cspond_scalar(pairs), _radial_sample_size(:radial)}, length(pairs), Val(:radial_homography), model; kwargs...)
end
