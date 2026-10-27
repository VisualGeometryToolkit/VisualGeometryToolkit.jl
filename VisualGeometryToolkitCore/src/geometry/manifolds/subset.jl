# =============================================================================
# SubsetManifold — a manifold on some of its parent's coordinates.
# `restrict` itself is declared with the contract in base/manifold.jl.
#
# The free/fixed choice every fitter makes (a target's complexity, a frozen
# focal, a pinned camera) is a manifold operation, not per-consumer bookkeeping:
# `restrict(ch, mask)` is the manifold of `ch` whose coordinates outside `mask`
# stay at zero. `tangent_size` = the free count; the Jacobians are the parent's columns;
# re-anchoring re-anchors the parent through the lifted coordinates. A
# restriction to no coordinates is the pinned model (tangent size 0).
# -----------------------------------------------------------------------------

"""
    SubsetManifold{C, D, M} <: Manifold

[`restrict`](@ref)'s manifold: the parent `manifold::C` of tangent size `D` on its `M` free
coordinates `idx` (ascending). `retract(rc, v)` is the parent's at `v`
lifted into `D` coordinates (zeros elsewhere), the Jacobians are the parent's
columns `idx`, `recenter` re-anchors the parent.
"""
struct SubsetManifold{C <: Manifold, D, M} <: Manifold
    manifold::C
    idx::SVector{M, Int}
    function SubsetManifold(manifold::C, idx::SVector{M, Int}) where {C <: Manifold, M}
        D = tangent_size(manifold)
        # strictly ascending, pair by pair: `allunique(idx)` collected the SVector
        # into an Array on Julia 1.12 (64 bytes per recenter at M = 4)
        ascending = true
        for j in 1:(M - 1)
            ascending &= idx[j] < idx[j + 1]
        end
        ascending ||
            throw(ArgumentError("restrict: coordinate indices must be ascending and distinct, got $idx"))
        all(i -> 1 <= i <= D, idx) || throw(ArgumentError("restrict: coordinate indices $idx outside 1:$D"))
        return new{C, D, M}(manifold, idx)
    end
end

restrict(ch::Manifold, idx::SVector{M, Int}) where {M} = SubsetManifold(ch, idx)
function restrict(ch::Manifold, free::SVector{D, Bool}) where {D}
    D == tangent_size(ch) || throw(ArgumentError("restrict: a $D-entry mask for a manifold of tangent size $(tangent_size(ch))"))
    return SubsetManifold(ch, SVector{count(free), Int}(findall(free)))
end
restrict(ch::Manifold, ::Tuple{}) = SubsetManifold(ch, SVector{0, Int}())

tangent_size(::SubsetManifold{C, D, M}) where {C, D, M} = M
model(rc::SubsetManifold) = model(rc.manifold)

# The M free coordinates lifted into the parent's D, the fixed ones at zero.
@inline function _lift(rc::SubsetManifold{C, D, M}, v::AbstractVector) where {C, D, M}
    full = zero(SVector{D, promote_type(eltype(v), Float64)})   # an empty v has eltype Union{}
    @inbounds for j in 1:M
        full = Base.setindex(full, v[j], rc.idx[j])
    end
    return full
end

retract(rc::SubsetManifold, v) = retract(rc.manifold, _lift(rc, v))
recenter(rc::SubsetManifold, v) = SubsetManifold(recenter(rc.manifold, _lift(rc, v)), rc.idx)
retract_jacobian(rc::SubsetManifold) = retract_jacobian(rc.manifold)[:, rc.idx]
retract_jacobian(rc::SubsetManifold, v::StaticVector) = retract_jacobian(rc.manifold, _lift(rc, v))[:, rc.idx]
action_jacobian(rc::SubsetManifold, x) = action_jacobian(rc.manifold, x)[:, rc.idx]
action_jacobian(rc::SubsetManifold, x, v::StaticVector) = action_jacobian(rc.manifold, x, _lift(rc, v))[:, rc.idx]
