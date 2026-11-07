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

# -----------------------------------------------------------------------------
# LinearSubspaceManifold — a manifold on a linear subspace of its parent's
# coordinates: `v = Z y` (a control-point basis, a pole-regular subspace).
# -----------------------------------------------------------------------------

"""
    LinearSubspaceManifold{C, D, M} <: Manifold

`restrict(ch, Z)`'s manifold: the parent `manifold::C` of tangent size `D` on
the `M` coordinates `y` of the subspace spanned by the columns of
`Z::SMatrix{D, M}`, the parent's coordinates `v = Z y`. At the anchor the
model is the parent's, so the parameters move as `θ = θ₀ + Z y` through the
parent's chart (exactly, for an unbounded parent). `retract`, `recenter` and
the Jacobians are the parent's composed with `Z` (`retract_jacobian(rc) =
retract_jacobian(ch) * Z`); `local_coordinates` is the least-squares `y` of
the parent's coordinates, exact for a model in the subspace.
"""
struct LinearSubspaceManifold{C <: Manifold, D, M, L} <: Manifold
    manifold::C
    basis::SMatrix{D, M, Float64, L}
    function LinearSubspaceManifold(manifold::C, Z::SMatrix{D, M, Float64, L}) where {C <: Manifold, D, M, L}
        D == tangent_size(manifold) ||
            throw(ArgumentError("restrict: a $D-row basis for a manifold of tangent size $(tangent_size(manifold))"))
        return new{C, D, M, L}(manifold, Z)
    end
end

"""
    restrict(ch::Manifold, Z::AbstractMatrix) -> LinearSubspaceManifold

The manifold `ch` on the subspace of its coordinates spanned by the columns of
`Z` (`tangent_size(ch)` rows, any number of columns): its coordinates `y`
move the parent's as `v = Z y`. A fit whose parameters are `vec(P) = vec(P₀) +
Z y` — a vignetting's control points on a regular subspace (a radial profile's
[`pole_regular_basis`](@ref)), a target's deformation basis — is the
parameter manifold of `P₀` restricted to `Z`. `Z` is stored as a static
matrix, so its size is part of the manifold's type.
"""
restrict(ch::Manifold, Z::AbstractMatrix) =
    LinearSubspaceManifold(ch, SMatrix{size(Z, 1), size(Z, 2), Float64}(Z))

tangent_size(::LinearSubspaceManifold{C, D, M}) where {C, D, M} = M
model(rc::LinearSubspaceManifold) = model(rc.manifold)
@inline _lift(rc::LinearSubspaceManifold, y::AbstractVector) = rc.basis * SVector{size(rc.basis, 2)}(Tuple(y))
retract(rc::LinearSubspaceManifold, y) = retract(rc.manifold, _lift(rc, y))
recenter(rc::LinearSubspaceManifold, y) = LinearSubspaceManifold(recenter(rc.manifold, _lift(rc, y)), rc.basis)
retract_jacobian(rc::LinearSubspaceManifold) = retract_jacobian(rc.manifold) * rc.basis
retract_jacobian(rc::LinearSubspaceManifold, y::StaticVector) = retract_jacobian(rc.manifold, _lift(rc, y)) * rc.basis
action_jacobian(rc::LinearSubspaceManifold, x) = action_jacobian(rc.manifold, x) * rc.basis
action_jacobian(rc::LinearSubspaceManifold, x, y::StaticVector) = action_jacobian(rc.manifold, x, _lift(rc, y)) * rc.basis
local_coordinates(rc::LinearSubspaceManifold, x) = _subspace_coordinates(rc, local_coordinates(rc.manifold, x))
# a matrix model: the parent's matrix form, with its keywords
local_coordinates(rc::LinearSubspaceManifold, x::AbstractMatrix; kw...) =
    _subspace_coordinates(rc, local_coordinates(rc.manifold, x; kw...))
_subspace_coordinates(rc::LinearSubspaceManifold, v) = (Z = rc.basis; (Z' * Z) \ (Z' * v))
