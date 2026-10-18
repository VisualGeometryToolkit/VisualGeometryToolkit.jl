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
