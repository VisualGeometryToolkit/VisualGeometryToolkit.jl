# =============================================================================
# SimilarityMap — scale, rotation, translation (x ↦ s·R·x + t)
# =============================================================================
#
# The similarity class as a map: its TYPE states that it preserves angles and
# ratios of lengths, so a circle stays a circle and an ellipse an ellipse
# (ct_contravariant.jl), and the class is closed under ∘ and inv exactly —
# (s₁, R₁, t₁) ∘ (s₂, R₂, t₂) = (s₁s₂, R₁R₂, s₁R₁t₂ + t₁), inv = (1/s, Rᵀ,
# −Rᵀt/s) — never by a 3×3 product or inversion. Frame tags exactly like
# EuclideanMap's (Src/Dst, UnlabeledFrame default, the junction checked by ∘,
# inv swapping them, relabel/unlabel).

"""
    SimilarityMap{N,T,S,R<:Rotation{N,T},V<:StaticVector{N},Src<:Frame{N},Dst<:Frame{N}} <: Transformation

The similarity transformation `x ↦ s·R·x + t` of `N`-space: a positive scale
`s`, a rotation `R` (a Rotations.jl `Rotation{N}`) and a translation `t`.

```julia
S = SimilarityMap(2.0, RotMatrix{2}(π / 6), SVector(1.0, -3.0))
S(circle)                  # a Circle: centre mapped, radius · 2
S ∘ inv(S)                 # a SimilarityMap (the identity, up to rounding)
SimilarityMap(E)           # an EuclideanMap, s = 1 (also a rotation LinearMap, a Translation, a uniform scaling)
```

Accessors [`scale`](@ref), [`rotation`](@ref), [`translation`](@ref).
`∘` with a similarity, an [`EuclideanMap`](@ref) or a `Translation` (either
order) is a `SimilarityMap`, and so is `inv`, both computed from the parts.
It acts on primitives and feature frames in its class (the class table of
the Transforms guide); [`to_homogeneous`](@ref) is `[s·R t; 0 1]`.

**Plain numbers only.** No consumer holds a physical similarity, so a
quantity translation or scale (or a unitful `EuclideanMap` converted or
composed with one) is refused with an `ArgumentError` naming the fix: strip it
in metres first, `ustrip(m, t)` or `ustrip(m, E)`.
`check = true` checks the rotation and that `s` is positive and finite.

The trailing `Src`/`Dst` parameters are coordinate-[`Frame`](@ref) tags, as
for `EuclideanMap`: constructors that state none give `UnlabeledFrame{N}`,
`inv` swaps them, `∘` checks a labeled junction (`MethodError` on a
mismatch), [`relabel`](@ref)/[`unlabel`](@ref) are the escape hatches.
"""
struct SimilarityMap{
        N, T, S <: Real, R <: Rotation{N, T}, V <: StaticVector{N, <:Real},
        Src <: Frame{N}, Dst <: Frame{N},
    } <: AbstractMatrixMap
    s::S
    R::R
    t::V
end

# Gradual-typing default: the 5-parameter surface is UNLABELED.
@inline function (::Type{SimilarityMap{N, T, S, R, V}})(
        s, Rv, tv
    ) where {N, T, S <: Real, R <: Rotation{N, T}, V <: StaticVector{N, <:Real}}
    return SimilarityMap{N, T, S, R, V, UnlabeledFrame{N}, UnlabeledFrame{N}}(s, Rv, tv)
end

scale(S::SimilarityMap) = S.s
rotation(S::SimilarityMap) = S.R
translation(S::SimilarityMap) = S.t
has_units(::SimilarityMap) = false

@inline function _assert_similarity_scale(s::Real)
    (s > 0 && isfinite(s)) || _refuse_similarity_scale(s)
    return nothing
end

# The refusal apart from the check (the message is built only here: the check compiles into a device kernel).
@noinline _refuse_similarity_scale(s) = throw(ArgumentError("a SimilarityMap's scale must be positive and finite, got $(s)"))

# THE refusal of a physical (unitful) value where only plain numbers can be
# held — a similarity, a homography, a planar primitive's action — naming the
# operation the caller asked for and the fix (units.md: a model that cannot
# record a unit refuses unitful input, the fix stated).
@noinline _refuse_unitful(what) = throw(
    ArgumentError(
        "$what takes plain numbers, and the map or translation given has units. " *
            "Strip it in metres first, the unit the stack's boundaries use: `ustrip(m, E)` for a map, `ustrip(m, t)` for a translation."
    )
)

"""
    SimilarityMap(s::Real, R::Rotation{N}, t::AbstractVector{<:Real}; check = true)

The similarity `x ↦ s·R·x + t`. A `RotMatrix`, `QuatRotation` or `Angle2d`
is stored as given, any other `Rotation` as a `RotMatrix`; an integer scale
or translation becomes its float type.
"""
@inline function SimilarityMap(s::Number, R::Rotation{N, T}, t::AbstractVector; check::Bool = true) where {N, T}
    (has_units(s) || has_units(t)) && _refuse_unitful("SimilarityMap")
    check && (_assert_rotation(R); _assert_similarity_scale(s))
    R_typed = _stored_rotation(R)
    sf = float(s)
    t_typed = t isa StaticVector{N} ? _float_translation(t) : SVector{N, float(eltype(t))}(t)
    return SimilarityMap{N, T, typeof(sf), typeof(R_typed), typeof(t_typed)}(sf, R_typed, t_typed)
end

SimilarityMap(S::SimilarityMap) = S

"""
    SimilarityMap(E::EuclideanMap) -> SimilarityMap

The rigid map as a similarity of scale 1, its frame tags kept. A physical
(unitful) map is refused: strip it first, `SimilarityMap(ustrip(m, E))`.
"""
function SimilarityMap(E::EuclideanMap{N, T, R, V, Src, Dst}) where {N, T, R, V, Src, Dst}
    has_units(E) && _refuse_unitful("SimilarityMap(E)")
    t = E.t
    sf = one(float(eltype(t)))
    return SimilarityMap{N, T, typeof(sf), R, V, Src, Dst}(sf, E.R, t)
end

# The similarity subgroup of CoordinateTransformations' maps: the rigid ones
# through their EuclideanMap, the uniform scaling the one case of its own.
SimilarityMap(L::CoordinateTransformations.LinearMap{<:Rotation{N, T}}) where {N, T} =
    SimilarityMap(EuclideanMap(L.linear, zero(SVector{N, float(T)}); check = false))
function SimilarityMap(Tx::CoordinateTransformations.Translation{<:StaticVector{N}}) where {N}
    T = float(eltype(Tx.translation))
    return SimilarityMap(EuclideanMap(RotMatrix{N, T}(SMatrix{N, N, T}(I)), Tx.translation; check = false))
end
function SimilarityMap(L::CoordinateTransformations.LinearMap{<:UniformScaling})
    λ = float(L.linear.λ)
    (iszero(λ) || !isfinite(λ)) && throw(ArgumentError("SimilarityMap: a uniform scaling by $(λ) is not a similarity"))
    T = typeof(λ)
    # a negative uniform scaling of the plane is the half turn times |λ|
    R = RotMatrix{2, T}(λ > 0 ? SMatrix{2, 2, T}(I) : -SMatrix{2, 2, T}(I))
    return SimilarityMap(abs(λ), R, zero(SVector{2, T}); check = false)
end

# Frame labeling escape hatches (zero-cost; the parts are reused bit for bit)
@inline function relabel(
        ::Type{Src}, ::Type{Dst}, S::SimilarityMap{N, T, Sc, R, V}
    ) where {N, T, Sc, R, V, Src <: Frame{N}, Dst <: Frame{N}}
    return SimilarityMap{N, T, Sc, R, V, Src, Dst}(S.s, S.R, S.t)
end
@inline unlabel(S::SimilarityMap{N}) where {N} = relabel(UnlabeledFrame{N}, UnlabeledFrame{N}, S)

# Operations
(S::SimilarityMap)(x) = S.s * (S.R * x) + S.t
CoordinateTransformations.transform_deriv(S::SimilarityMap{N}, x) where {N} = S.s * SMatrix{N, N}(S.R)

to_homogeneous(S::SimilarityMap{N}) where {N} = to_homogeneous(S.s * SMatrix{N, N}(S.R), S.t)

# inv = (1/s, R⁻¹, −R⁻¹t/s): the rotation keeps its representation (a
# RotMatrix inverts to its transpose, a QuatRotation to its conjugate)
function Base.inv(S::SimilarityMap{N, T, Sc, R, V, Src, Dst}) where {N, T, Sc, R, V, Src, Dst}
    Rin = inv(S.R)
    si = inv(S.s)
    tin = -(si * (Rin * S.t))
    return relabel(Dst, Src, SimilarityMap(si, Rin, SVector{N}(tin); check = false))
end

@inline function _compose_similarity(A::SimilarityMap{N}, B::SimilarityMap{N}, ::Type{Src}, ::Type{Dst}) where {N, Src, Dst}
    s = A.s * B.s
    R = A.R * B.R
    t = A.s * (A.R * B.t) + A.t
    return relabel(Src, Dst, SimilarityMap(s, R, SVector{N}(t); check = false))
end

function Base.:∘(
        A::SimilarityMap{N, TA, SA, RA, VA, SrcA, DstA}, B::SimilarityMap{N, TB, SB, RB, VB, SrcB, DstB}
    ) where {N, TA, SA, RA, VA, SrcA <: Frame{N}, DstA <: Frame{N}, TB, SB, RB, VB, SrcB <: Frame{N}, DstB <: Frame{N}}
    _check_junction(SrcA, DstB, ∘, (A, B))
    return _compose_similarity(A, B, SrcB, DstA)
end
# a rigid map and a similarity compose to a similarity, either order
Base.:∘(A::EuclideanMap{N}, B::SimilarityMap{N}) where {N} =
    (has_units(A) && _refuse_unitful("∘ with a SimilarityMap"); SimilarityMap(A) ∘ B)
Base.:∘(A::SimilarityMap{N}, B::EuclideanMap{N}) where {N} =
    (has_units(B) && _refuse_unitful("∘ with a SimilarityMap"); A ∘ SimilarityMap(B))
# and so do the rest of the similarity class — a translation, a rotation
# LinearMap, a uniform scaling (the class table, ct_contravariant.jl) — never
# CoordinateTransformations' lazy composition, which skips the junction check.
# (A RotationMap's ∘ is beside it, rotation_map.jl.)
const _SimilarityCTMap{N} = Union{
    CoordinateTransformations.Translation{<:StaticVector{N}},
    CoordinateTransformations.LinearMap{<:Rotation{N}},
}
Base.:∘(A::_SimilarityCTMap{N}, B::SimilarityMap{N}) where {N} = SimilarityMap(A) ∘ B
Base.:∘(A::SimilarityMap{N}, B::_SimilarityCTMap{N}) where {N} = A ∘ SimilarityMap(B)
Base.:∘(A::CoordinateTransformations.LinearMap{<:UniformScaling}, B::SimilarityMap{2}) = SimilarityMap(A) ∘ B
Base.:∘(A::SimilarityMap{2}, B::CoordinateTransformations.LinearMap{<:UniformScaling}) = A ∘ SimilarityMap(B)

# Storage-boundary conversion, EuclideanMap's frame policy: it may LABEL an
# unlabeled map (or keep the frames), never relabel or erase one; the parts
# are converted to the slot's types.
@inline function Base.convert(
        ::Type{SimilarityMap{N, T, S, R, V, Src, Dst}},
        m::SimilarityMap{N, <:Any, <:Real, <:Rotation{N}, <:StaticVector{N}, Src0, Dst0}
    ) where {
        N, T, S, R <: Rotation{N, T}, V <: StaticVector{N},
        Src <: Frame{N}, Dst <: Frame{N},
        Src0 <: Union{Src, UnlabeledFrame{N}}, Dst0 <: Union{Dst, UnlabeledFrame{N}},
    }
    return SimilarityMap{N, T, S, R, V, Src, Dst}(S(m.s), R(m.R), V(m.t))
end
# (propagate through a SimilarityMap is beside EuclideanMap's, propagate_framed.jl;
# similarity_scale beside the framed affine's, cameras/imaging_chain.jl.)
