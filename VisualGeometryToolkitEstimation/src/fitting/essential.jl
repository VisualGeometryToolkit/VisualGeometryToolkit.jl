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
    find_essential(camera1, camera2, pixel_cs; scoring,
        max_trials, min_trials,
        local_optimization_strategy, config=nothing)

Robust essential-matrix estimation via 5-point RANSAC with MSAC scoring. Every default is `configs/default.toml`'s (`[ransac]`,
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

Returns a `RansacEstimate`, the `:no_model` result (`iterations` 0) for fewer than 5 correspondences.
`iterations(result)` is the number of trials (hypotheses drawn); test
`found_model(result)` before reading the model (see [`RansacEstimate`](@ref)).

# Arguments
- `camera1`, `camera2`: the cameras of the source and target views (any
  camera `backproject_with_jacobian` accepts).
- `pixel_cs`: PIXEL correspondences, `u₁ => u₂` in the two cameras' images
  (`Csponds`, `StructArray{<:Pair}` or a Pair vector); a `Csponds` `:score`
  column is the per-match prior (PROSAC; uniform otherwise) and travels to the
  bearing problem with every other attribute column.
- `scoring`: the scorer, its width **in PIXELS** (default `[pipelines.essential] scoring`, MSAC at 1 px). `M` picks the unit and the call cannot: the residual is whitened by
  the back-projection Jacobian, so the width it is compared against is in
  pixels; an orthonormal tangent basis would put the same number in radians.
  THE UNIT CHANGED — this was in calibrated coordinates until the bearing port,
  and no pixel value reproduces the old `0.02` across cameras because `0.02·f`
  is f-dependent (≈16 px at f = 800, ≈24 px at f = 1200). Values below
  VisualGeometryToolkitFeatures' 0.06 px refinement bound — a threshold
  scorer's threshold or the Student-t scorer's outlier half-width — are REFUSED
  (`ArgumentError`) rather than silently honoured, since they are almost
  always a calibrated leftover — and refused by every entry that scores an
  `EssentialProblem`: a `scoring` supplied here, the class default, or
  `ransac` / `locally_optimize` called directly.
- `local_optimization_strategy`: LO-RANSAC strategy (default `[pipelines.essential] local_optimization_strategy`, as
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
    return _guarded_ransac(() -> EssentialProblem(camera1, camera2, pixel_cs), EssentialProblem{Float64}, length(pixel_cs), :essential; kwargs...)
end


"""
    refine_essential(E, camera1, camera2, pixel_cs; optimization_strategy, scale | width, workspace = nothing)
        -> FitResult{M, RefineMetadata}

The essential matrix `E` refined against the PIXEL correspondences `pixel_cs`
of the cameras `camera1`, `camera2`, from `E` itself, on [`find_essential`](@ref)'s
bearing problem (each view back-projected with its own camera; the residual
whitened into pixels, codimension 1): the keywords, the result and the refusals are those of every `refine_*`,
documented on [`refine_homography`](@ref) (default strategy:
`[refine.essential] optimization_strategy`). Fewer than 5 correspondences: the guess back
unrefined, record `(ReturnCode.InitialFailure, NoRule)` (`retcode_symbol` `:degenerate`).
"""
function refine_essential(E, camera1, camera2, pixel_cs; kwargs...)
    return _guarded_refine(() -> EssentialProblem(camera1, camera2, pixel_cs), EssentialProblem{Float64}, length(pixel_cs), Val(:essential), E; kwargs...)
end
