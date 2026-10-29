# =============================================================================
# Fundamental Matrix Fitting — RANSAC
# =============================================================================

"""
    find_fundmat(correspondences; threshold=3.0,
        max_trials=10000, min_trials=100,
        local_optimization=Irls{TLSShape}(),
        scoring=nothing, config=nothing)

Robust fundamental matrix estimation via 7-point RANSAC with MSAC
scoring and least-squares-on-the-inliers local optimization. Every default shown is
`configs/default.toml`'s (`[ransac]`, `[pipelines.fundmat]`); a keyword overrides.

Returns a `RansacEstimate` or `nothing` if fewer than 7 correspondences.
`iterations(result)` is the number of trials (hypotheses drawn); test
`converged(result)` before reading the model (see [`RansacEstimate`](@ref)).

# Arguments
- `correspondences`: as [`find_homography`](@ref); a `Csponds` `:score` column is the
  per-match prior (PROSAC; uniform otherwise).
- `threshold`: **in PIXELS**, as [`find_homography`](@ref).
- `workspace`: as [`find_homography`](@ref), built with `RansacWorkspace(FundMatProblem{Float64}, n_data[, T])`.
- `local_optimization`: default `Irls{TLSShape}()`, as [`find_homography`](@ref).
- the other keywords as [`find_homography`](@ref).
"""
function find_fundmat(
        pairs::AbstractVector{<:Pair{<:StaticArrays.StaticVector{2}, <:StaticArrays.StaticVector{2}}};
        kwargs...
    )
    _holds_minimal_sample(FundMatProblem, length(pairs)) || return nothing
    return _ransac_pipeline(FundMatProblem(pairs), :fundmat; kwargs...)
end

"""
    refine_fundmat(F, correspondences; threshold = 3.0,
        local_optimization = Irls{TLSShape}(), scoring = nothing, workspace = nothing)
        -> RansacEstimate

The fundamental matrix `F` refined against `correspondences`:
[`find_fundmat`](@ref)'s step without the sampling. `local_optimization` is the
fundamental matrix's RANSAC local-optimization step, run here alone from the
given model (no sampling). Every default is `find_fundmat`'s
(`[pipelines.fundmat]`); the keywords are those of every `refine_*`, documented
on [`refine_homography`](@ref). `nothing` for fewer than 7 correspondences.
"""
function refine_fundmat(
        F, pairs::AbstractVector{<:Pair{<:StaticArrays.StaticVector{2}, <:StaticArrays.StaticVector{2}}};
        kwargs...
    )
    _holds_minimal_sample(FundMatProblem, length(pairs)) || return nothing
    return _refine_pipeline(FundMatProblem(pairs), :fundmat, F; kwargs...)
end
