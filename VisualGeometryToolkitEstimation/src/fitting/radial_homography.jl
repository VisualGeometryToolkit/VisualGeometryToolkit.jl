# =============================================================================
# Radial-Homography Fitting — find_radial_homography
# =============================================================================
#
# Robust one-sided radially-distorted homography estimation via RANSAC —
# the (H, k) sibling of `find_homography`, over `RadialHomographyProblem`.
# =============================================================================

"""
    find_radial_homography(correspondences; center,
        threshold, extent=nothing,
        workspace=nothing, extra_check=nothing, minimal=:radial,
        local_optimization, max_trials, min_trials, config=nothing, rng=...)

`threshold` and `local_optimization` default to
`configs/default.toml` `[pipelines.radial_homography]`, `max_trials` and
`min_trials` to its `[ransac]` table; a keyword overrides. `threshold`
builds the default scoring only; supplying it with `scoring` is an `ArgumentError`. `extra_check`
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
`k`, and the distortion `center` all carried in the model — or `nothing`
for fewer than 5 correspondences.
`iterations(result)` is the number of trials (hypotheses drawn); test
`converged(result)` before reading the model (see [`RansacEstimate`](@ref)).

`center` — the distortion center in target pixels — is REQUIRED: the 5-pt
solver runs in the Hartley frame about it.

`local_optimization` defaults to `Irls{TLSShape}()` (least squares on the
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
    _holds_minimal_sample(_radial_sample_size(minimal), length(pairs)) || return nothing
    problem = _with_extent(RadialHomographyProblem(pairs; center, extra_check, minimal), extent)
    return _ransac_pipeline(problem, :radial_homography; kwargs...)
end
