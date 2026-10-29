# =============================================================================
# Essential Matrix Fitting
# =============================================================================
#
# Robust essential-matrix estimation via 5-point RANSAC.
#
#   find_essential — thin wrapper around ransac() with the settled scoring/LO
#                    formulation (the ONLY class with a NATIVE CV'd operating
#                    point — PT-E is the main-paper cell)
#
# =============================================================================

# =============================================================================
# Public API
# =============================================================================

"""
    find_essential(camera1, camera2, pixel_cs; threshold=1.0,
        max_trials=10000, min_trials=100,
        local_optimization=Irls{TLSShape}(),
        scoring=nothing, config=nothing)

Robust essential-matrix estimation via 5-point RANSAC with MSAC scoring. Every default shown is `configs/default.toml`'s (`[ransac]`,
`[pipelines.essential]`); a keyword overrides.

Takes PIXEL correspondences and the two cameras that produced them. Each view
is back-projected with its OWN camera, so the per-view Jacobians land with the
right endpoints by construction rather than by the caller remembering — a swap
is norm-preserving and invisible to Monte-Carlo calibration, showing only in
the tangency contract `fᵀM = 0`.

Two cameras rather than one because they are genuinely independent, and because
a single-camera convenience form would make that swap undetectable in exactly
the tests that ought to catch it.

DO NOT ADD a single-camera overload as an ergonomics improvement. Collapsing
two distinct inputs into one erases the asymmetry a swap needs in order to be
visible — the same reason an `M`-wiring harness needs two different focal
lengths. A stereo pair with one camera repeated is written `find_essential(cam,
cam, cs)`, which costs three characters and keeps the asymmetry available.

Returns a `RansacEstimate` or `nothing` if fewer than 5 correspondences.
`iterations(result)` is the number of trials (hypotheses drawn); test
`converged(result)` before reading the model (see [`RansacEstimate`](@ref)).

# Arguments
- `camera1`, `camera2`: the cameras of the source and target views (any
  camera `backproject_with_jacobian` accepts).
- `pixel_cs`: PIXEL correspondences, `u₁ => u₂` in the two cameras' images
  (`Csponds`, `StructArray{<:Pair}` or a Pair vector); a `Csponds` `:score`
  column is the per-match prior (PROSAC; uniform otherwise) and travels to the
  bearing problem with every other attribute column.
- `threshold`: the MSAC inlier threshold, **in PIXELS**. `M` picks the unit and the call cannot: the residual is whitened by
  the back-projection Jacobian, so the threshold it is compared against is in
  pixels; an orthonormal tangent basis would put the same number in radians.
  THE UNIT CHANGED — this was in calibrated coordinates until the bearing port,
  and no pixel value reproduces the old `0.02` across cameras because `0.02·f`
  is f-dependent (≈16 px at f = 800, ≈24 px at f = 1200). Values below
  VisualGeometryToolkitFeatures' 0.06 px refinement bound — a threshold or a
  scorer's outlier half-width — are REFUSED
  (`ArgumentError`) rather than silently honoured, since they are almost
  always a calibrated leftover — and refused by every entry that scores an
  `EssentialProblem`, not only this keyword: a `scoring` supplied here, or
  `ransac` / `locally_optimize` called directly.
  Builds the default scoring only; supplying it with `scoring` is an `ArgumentError`.
- `local_optimization`: LO-RANSAC strategy (default `Irls{TLSShape}()`, as
  [`find_homography`](@ref)).
- the other keywords as [`find_homography`](@ref).

# Examples
```julia
result = find_essential(cam1, cam2, pixel_pairs)
result = find_essential(cam1, cam2, Csponds(sources(pixel_pairs), targets(pixel_pairs); score = similarities))
```
"""
function find_essential(
        camera1, camera2,
        pixel_cs;
        kwargs...
    )
    # The sub-detector half-width refusal is the problem's (`_check_scoring`,
    # problems/essential.jl), run on whatever scoring the pipeline ends up with.
    _holds_minimal_sample(EssentialProblem, length(pixel_cs)) || return nothing
    return _ransac_pipeline(EssentialProblem(camera1, camera2, pixel_cs), :essential; kwargs...)
end


"""
    refine_essential(E, camera1, camera2, pixel_cs; threshold = 1.0,
        local_optimization = Irls{TLSShape}(), scoring = nothing, workspace = nothing)
        -> RansacEstimate

The essential matrix `E` refined against the PIXEL correspondences `pixel_cs`
of the cameras `camera1`, `camera2`: [`find_essential`](@ref)'s step without the
sampling, on the same bearing problem (each view back-projected with its own
camera; the threshold in pixels, sub-detector values refused).
`local_optimization` is the essential matrix's RANSAC local-optimization step,
run here alone from the given model (no sampling). Every default is
`find_essential`'s (`[pipelines.essential]`); the keywords are those of every
`refine_*`, documented on [`refine_homography`](@ref). `nothing` for fewer than
5 correspondences.
"""
function refine_essential(E, camera1, camera2, pixel_cs; kwargs...)
    _holds_minimal_sample(EssentialProblem, length(pixel_cs)) || return nothing
    return _refine_pipeline(EssentialProblem(camera1, camera2, pixel_cs), :essential, E; kwargs...)
end
