using LinearAlgebra: pinv

# =============================================================================
# EuclideanManifold and ProductManifold — a plain vector as a chart, and the
# chart of a model made of independent parts (a homography ⊕ a division
# coefficient, a rig of poses, a pose and its points).
# =============================================================================

"""
    EuclideanManifold(x::StaticVector{N}) <: Manifold

`Rᴺ` at `x`: `retract(ch, v) = x + v`, the identity differential,
`local_coordinates(ch, y) = y − x`. The chart of a model's plain-vector part — a
division coefficient, a 3D point — in a [`ProductManifold`](@ref).
"""
struct EuclideanManifold{N, T} <: Manifold
    x::SVector{N, T}
end
EuclideanManifold(x::StaticVector{N}) where {N} = EuclideanManifold(SVector{N, float(eltype(x))}(x))

"""
    EuclideanManifold(ch::Manifold)

The AMBIENT space of the chart `ch` as a chart: `Rᴺ` anchored at the ambient
coordinates of `model(ch)` (vec(M) for a 3×3 model, `[vec R; t]` for a pose, a
lens's parameters, …) — where [`transform_covariance`](@ref) pushes a covariance
to, `transform_covariance(ch, EuclideanManifold(ch), Σv)`, and pulls it from,
`transform_covariance(EuclideanManifold(ch), ch, Σ)`.
"""
EuclideanManifold(ch::Manifold) = EuclideanManifold(_ambient_coordinates(model(ch)))

tangent_size(::EuclideanManifold{N}) where {N} = N
model(ch::EuclideanManifold) = ch.x
retract(ch::EuclideanManifold, v) = ch.x + v
recenter(ch::EuclideanManifold, v) = EuclideanManifold(retract(ch, v))
retract_jacobian(::EuclideanManifold{N, T}, v::StaticVector = zero(SVector{N, T})) where {N, T} = SMatrix{N, N, T}(I)
local_coordinates(ch::EuclideanManifold{N}, y::StaticVector{N}) where {N} = y - ch.x

# To and from the ambient chart: the push T·Σ·Tᵀ and the pull T⁺·Σ·T⁺ᵀ alone (the
# ambient chart's own differential is the identity).
function transform_covariance(from::Manifold, to::EuclideanManifold{N}, Σ::AbstractMatrix) where {N}
    T = retract_jacobian(from)
    size(T, 1) == N || throw(DimensionMismatch("transform_covariance: a $(size(T, 1))-dim ambient space is not the $N-dim chart"))
    return propagate(T, Σ)
end
function transform_covariance(from::EuclideanManifold{N}, to::Manifold, Σ::AbstractMatrix) where {N}
    T = retract_jacobian(to)
    size(T, 1) == N || throw(DimensionMismatch("transform_covariance: the $N-dim chart is not a $(size(T, 1))-dim ambient space"))
    return propagate(pinv(T), Σ)
end
function transform_covariance(from::EuclideanManifold{N}, to::EuclideanManifold{M}, Σ::AbstractMatrix) where {N, M}
    N == M || throw(DimensionMismatch("transform_covariance: a $N-dim chart is not an $M-dim one"))
    return Σ
end

"""
    ProductManifold(charts::Manifold...; embed = identity) <: Manifold

The chart of a model made of independent parts, one chart each: coordinates the
concatenation of the parts' (in order), `retract(ch, v) = embed(parts)` with
`parts` the tuple of the parts' retractions, `recenter` part by part,
`retract_jacobian` block diagonal over the parts' ambient spaces (an `SMatrix`
when the parts' are static and the whole stays small, a dense `Matrix`
otherwise — products run to many parts: a rig of poses, a pose and its points),
and `local_coordinates(ch, parts::Tuple)` the concatenation. `embed` assembles
the model from its parts: [`minimal_manifold`](@ref)`(::RadialHomography)` is
sl(3) ⊕ k assembled into the `RadialHomography` with its centre.
"""
struct ProductManifold{C <: Tuple, F} <: Manifold
    charts::C
    embed::F
end
ProductManifold(charts::Manifold...; embed = identity) = ProductManifold(charts, embed)

tangent_size(p::ProductManifold) = sum(tangent_size, p.charts)
model(p::ProductManifold) = p.embed(map(model, p.charts))
retract(p::ProductManifold, v) = p.embed(map(retract, p.charts, manifold_parts(p, v)))
recenter(p::ProductManifold, v) = ProductManifold(map(recenter, p.charts, manifold_parts(p, v)), p.embed)
retract_jacobian(p::ProductManifold) = _blockdiag(map(retract_jacobian, p.charts))
retract_jacobian(p::ProductManifold, v::AbstractVector) = _blockdiag(map(retract_jacobian, p.charts, manifold_parts(p, v)))
retract_jacobian(p::ProductManifold, v::StaticVector) = _blockdiag(map(retract_jacobian, p.charts, manifold_parts(p, v)))   # beside the generic default
local_coordinates(p::ProductManifold, parts::Tuple) = vcat(map(local_coordinates, p.charts, parts)...)

# The coordinates of each part: consecutive slices of v, sized by the parts' charts.
# The sizes go through `Val` so each slice is a statically sized `SVector` of v's
# eltype (a chart's `tangent_size` is a constant of its type): the parts, and the
# retraction of a product, infer — ForwardDiff Duals included (and a part with no
# coordinates, a parameterless lens family, is an empty vector of v's eltype).
function manifold_parts(p::ProductManifold, v)
    offsets = Base.front(accumulate(+, (0, map(tangent_size, p.charts)...)))
    return map(p.charts, offsets) do ch, o
        SVector{tangent_size(ch), eltype(v)}(ntuple(j -> v[o + j], Val(tangent_size(ch))))
    end
end

# Block-diagonal assembly of the parts' differentials: static while the result
# stays within `_STATIC_BLOCKDIAG_CELLS` entries (the size read off the block
# types), dense beyond it or when any part is dense.
const _STATIC_BLOCKDIAG_CELLS = 256

_blockdiag(blocks::Tuple) = _dense_blockdiag(blocks)
function _blockdiag(blocks::Tuple{Vararg{StaticMatrix}})
    m, n = _static_extent(blocks)
    m * n <= _STATIC_BLOCKDIAG_CELLS || return _dense_blockdiag(blocks)
    return foldl(_static_blockdiag, blocks)
end

_static_extent(::Tuple{}) = (0, 0)
@inline function _static_extent(bs::Tuple{StaticMatrix{m, n}, Vararg{StaticMatrix}}) where {m, n}
    r, c = _static_extent(Base.tail(bs))
    return (m + r, n + c)
end

@inline function _static_blockdiag(A::StaticMatrix{ma, na}, B::StaticMatrix{mb, nb}) where {ma, na, mb, nb}
    T = promote_type(eltype(A), eltype(B))
    return vcat(
        hcat(SMatrix{ma, na, T}(A), zeros(SMatrix{ma, nb, T})),
        hcat(zeros(SMatrix{mb, na, T}), SMatrix{mb, nb, T}(B)),
    )
end

function _dense_blockdiag(blocks::Tuple)
    T = mapreduce(eltype, promote_type, blocks)
    J = zeros(T, sum(b -> size(b, 1), blocks), sum(b -> size(b, 2), blocks))
    r = c = 0
    for b in blocks
        m, n = size(b)
        J[(r + 1):(r + m), (c + 1):(c + n)] .= b
        r += m
        c += n
    end
    return J
end
