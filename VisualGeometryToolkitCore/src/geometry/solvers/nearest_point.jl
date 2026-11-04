# =============================================================================
# nearest_point — the least-squares point nearest to N lines (one closed-form step)
# =============================================================================
#
# The N-line generalization of the midpoint of two rays. `triangulate` is the
# two-view, ray-domain form (the point AND its depths, from a pose); this is
# the point alone, from any number of lines in one frame. One deterministic
# linear solve with FIXED weights: an algorithm, named for what it computes.

"""
    nearest_point(lines) -> SVector{N}

The point `x` nearest to the `lines` in the least-squares sense: the minimizer
of `Σᵢ wᵢ ‖(I − dᵢdᵢᵀ)(x − oᵢ)‖²`, the weighted sum of squared distances from
`x` to each line, by one closed-form solve of the normal equations
`(Σ wᵢPᵢ) x = Σ wᵢPᵢoᵢ`, `Pᵢ = I − dᵢdᵢᵀ` (the projector onto the line's normal
space).

`lines` is a vector of `origin => direction` pairs (a [`Csponds`](@ref) by
`csponds(origins, directions)`, or a plain vector of `Pair`s): `oᵢ` any point
on the line, `dᵢ` its direction, normalized inside (any nonzero length).
`N`-dimensional: 3-D lines and 2-D lines alike. Per-line weights are the
`weights` attribute column (`withattrs(lines; weights = w)`): FIXED,
non-negative, a zero weight drops the line; without it every line weighs 1.

Units: origins that are lengths (`SVector(1.0, 2.0, 3.0) .* mm`) are stripped
in metres at entry and the point comes back in metres; plain origins give a
plain point in the caller's own unit. The solve runs about the weighted
centroid of the origins, so translating every line by `t` translates the point
by `t` to rounding.

Throws `ArgumentError` for fewer than two lines of positive weight, a zero or
non-finite direction, or lines that do not determine a point — all parallel,
the normal matrix singular to `1e-12` of its largest eigenvalue.

```julia
lines = csponds([SVector(0.0, 0.0, 0.0), SVector(1.0, 0.0, 0.0)], [SVector(0.0, 0.0, 1.0), SVector(-1.0, 0.0, 1.0)])
nearest_point(lines)   # [0.0, 0.0, 1.0]: where the two lines meet
```
"""
function nearest_point(lines::AbstractVector{<:Pair{<:StaticVector{N, <:Quantity}, <:StaticVector{N}}}) where {N}
    # Origins that are lengths: stripped in metres once, the plain solve, metres back.
    stripped = [strip_metres(first(l)) => last(l) for l in lines]
    w = weights_column(lines)
    return nearest_point(w === nothing ? stripped : withattrs(Csponds(sources(stripped), targets(stripped)); weights = w)) .* m
end
function nearest_point(lines::AbstractVector{<:Pair{<:StaticVector{N}, <:StaticVector{N}}}) where {N}
    w = check_weights(weights_column(lines), length(lines))
    weight(i) = w === nothing ? 1.0 : Float64(w[i])
    n₊ = w === nothing ? length(lines) : count(>(0), w)
    n₊ >= 2 || throw(ArgumentError("nearest_point requires at least 2 lines of positive weight, got $n₊"))
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
        (isfinite(nd) && nd > 0) || throw(ArgumentError("nearest_point: line $i has direction $(last(lines[i])); a direction must be finite and nonzero"))
        d = d / nd
        P = Id - d * d'
        A += wᵢ * P
        b += wᵢ * (P * (origin(i) - ō))
    end
    λ = eigen(Symmetric(A)).values   # ascending
    λ[1] > 1.0e-12 * λ[end] ||
        throw(ArgumentError("nearest_point: the lines do not determine a point (parallel; the normal matrix's eigenvalues $(λ[1]) ≪ $(λ[end]))"))
    return ō + A \ b
end
