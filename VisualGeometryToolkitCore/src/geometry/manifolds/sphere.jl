# =============================================================================
# SphereManifold — the unit sphere S^{N−1} ⊂ Rᴺ: the chart of a model known only
# up to scale and sign (a projective vector — a RadialCamera's 2×3 rows), when
# the model has no finer structure to chart.
# =============================================================================

using LinearAlgebra: normalize

"""
    SphereManifold(q::StaticVector{N}, embed = identity) <: Manifold

The unit sphere at `q` (normalized on construction): coordinates `v ∈ R^{N−1}`
along an orthonormal basis `B` of `q`'s complement — the last `N − 1` columns of
the Householder reflection taking `e₁` to `∓q`, built from `q + sign(q₁)·e₁` so it
is stable for every `q` — and `retract(ch, v) = embed(normalize(q + B·v))`.

`embed` turns the unit vector into the model: the identity for a bare vector, a
`RadialCamera` with its centre for [`minimal_manifold`](@ref)`(::RadialCamera)`.
The ambient space is the vector's: `retract_jacobian(ch, v)` is
`(I − ŷŷᵀ)·B/‖q + B·v‖` (`N × (N − 1)`, `B` at the anchor). `local_coordinates(ch,
y)` is closed form and invariant to the scale and sign of `y`: `Bᵀy/(qᵀy)`.
`recenter` re-anchors at the retracted point, with a new basis.
"""
struct SphereManifold{N, D, T, L, F} <: Manifold
    q::SVector{N, T}
    B::SMatrix{N, D, T, L}
    embed::F
end
# `embed::F`: a Function argument that is only handed on is not specialized on by
# default, and the unspecialized call boxed u, B and the result (176 bytes per
# recenter at N = 3, 624 at N = 6).
function SphereManifold(q::StaticVector{N}, embed::F = identity) where {N, F}
    T = float(eltype(q))
    u = SVector{N, T}(q) / norm(q)
    return SphereManifold(u, _householder_complement(u), embed)
end

# The last N − 1 columns of H = I − 2wwᵀ/(wᵀw), w = u + sign(u₁)·e₁: H·e₁ = ∓u, so
# they are an orthonormal basis of u's complement; wᵀw = 2(1 + |u₁|) ≥ 2.
function _householder_complement(u::SVector{N, T}) where {N, T}
    σ = u[1] >= 0 ? one(T) : -one(T)
    w = Base.setindex(u, u[1] + σ, 1)
    H = SMatrix{N, N, T}(I) - (2 / dot(w, w)) * (w * w')
    return H[:, SVector(ntuple(i -> i + 1, Val(N - 1)))]
end

tangent_size(::SphereManifold{N, D}) where {N, D} = D
model(ch::SphereManifold) = ch.embed(ch.q)
retract(ch::SphereManifold, v) = ch.embed(normalize(ch.q + ch.B * v))
recenter(ch::SphereManifold, v) = SphereManifold(normalize(ch.q + ch.B * v), ch.embed)

function retract_jacobian(ch::SphereManifold{N, D, T}, v::StaticVector = zero(SVector{D, T})) where {N, D, T}
    y = ch.q + ch.B * v
    n = norm(y)
    ŷ = y / n
    return (SMatrix{N, N, eltype(ŷ)}(I) - ŷ * ŷ') * ch.B / n
end

# A target of any scale or sign; a model with an `embed` adds the method that
# reads its vector (`local_coordinates(ch, ::RadialCamera)`).
local_coordinates(ch::SphereManifold{N}, y::StaticVector{N}) where {N} = ch.B' * y / dot(ch.q, y)
