# =============================================================================
# The least-squares point nearest to N lines (one closed-form step): NearestPoint's kernel
# =============================================================================
#
# The N-line generalization of the midpoint of two rays. `Triangulation` is the
# two-view, ray-domain form (the point AND its depths, from a pose); this is
# the point alone, from any number of lines in one frame. One deterministic
# linear solve with FIXED weights: an algorithm, named for what it computes.

# Kernel of [`NearestPoint`](@ref), whose docstring is the one copy of its documentation: `_nearest_point(lines) -> SVector{N}`.
function _nearest_point(lines::AbstractVector{<:Pair{<:StaticVector{N, <:Quantity}, <:StaticVector{N}}}) where {N}
    # Origins that are lengths: stripped in metres once, the plain solve, metres back.
    stripped = [strip_meters(first(l)) => last(l) for l in lines]
    w = weights_column(lines)
    return _nearest_point(w === nothing ? stripped : withattrs(Correspondences(sources(stripped), targets(stripped)); weights = w)) .* m
end
function _nearest_point(lines::AbstractVector{<:Pair{<:StaticVector{N}, <:StaticVector{N}}}) where {N}
    w = check_weights(weights_column(lines), length(lines))
    weight(i) = w === nothing ? 1.0 : Float64(w[i])
    n₊ = w === nothing ? length(lines) : count(>(0), w)
    n₊ >= 2 || throw(ArgumentError("NearestPoint requires at least 2 lines of positive weight, got $n₊"))
    origin(i) = SVector{N, Float64}(first(lines[i]))
    # the weighted centroid of the origins: the solve's own origin
    wsum = 0.0
    ō = zero(SVector{N, Float64})
    for i in eachindex(lines)
        wᵢ = weight(i)
        wᵢ > 0 || continue
        wsum += wᵢ
        ō += wᵢ * origin(i)
    end
    ō /= wsum
    A = zero(SMatrix{N, N, Float64})
    b = zero(SVector{N, Float64})
    Id = SMatrix{N, N, Float64}(I)
    for i in eachindex(lines)
        wᵢ = weight(i)
        wᵢ > 0 || continue
        d = SVector{N, Float64}(last(lines[i]))
        nd = norm(d)
        (isfinite(nd) && nd > 0) || throw(ArgumentError("NearestPoint: line $i has direction $(last(lines[i])); a direction must be finite and nonzero"))
        d = d / nd
        P = Id - d * d'
        A += wᵢ * P
        b += wᵢ * (P * (origin(i) - ō))
    end
    λ = eigen(Symmetric(A)).values   # ascending
    λ[1] > 1.0e-12 * λ[end] ||
        throw(ArgumentError("NearestPoint: the lines do not determine a point (parallel; the normal matrix's eigenvalues $(λ[1]) ≪ $(λ[end]))"))
    return ō + A \ b
end
