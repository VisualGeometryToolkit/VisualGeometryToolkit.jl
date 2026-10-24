# =============================================================================
# N-way correspondences
# =============================================================================
#
# The pairwise layer — `Cspond{S,T} = Pair{S,T}`, the `csponds` builder and the
# columnar `Csponds` it returns — is VisualGeometryToolkitCore's; nn_match.jl adds the
# `csponds` method for NNMatch lists. CspondSet is the N-way form, defined here.
#
# CspondSet wraps any AbstractVector — fixed-size (SVector) or growable (Vector).
#
#   CspondSet((a, b, c))          # fixed: SVector{3,T}
#   CspondSet([a, b])             # variable: Vector{T}, supports push!
#   CspondSet(T)                  # empty growable set of eltype T
#

using VisualGeometryToolkitCore: Cspond
import VisualGeometryToolkitCore: csponds

"""
    CspondSet(features::AbstractVector)
    CspondSet(features::Tuple)
    CspondSet(T::Type)

An N-way correspondence: the same scene point observed as one feature in each
of several images, where `Cspond` is the two-way case. It wraps a vector
of features and iterates, indexes and has a length like one. A tuple becomes a
fixed-size `SVector`; a `Vector` (or `CspondSet(T)`, an empty one of element
type `T`) grows with `push!` and `append!`.

```jldoctest
julia> using VisualGeometryToolkitFeatures

julia> track = CspondSet(Int);

julia> push!(track, 3); push!(track, 7);

julia> length(track), track[2], collect(track)
(2, 7, [3, 7])

julia> length(CspondSet((1, 2, 3)))
3
```
"""
struct CspondSet{C <: AbstractVector}
    features::C
end
CspondSet(features::Tuple) = CspondSet(SVector(features))
CspondSet(::Type{T}) where {T} = CspondSet(T[])

Base.getindex(cs::CspondSet, i::Int) = cs.features[i]
Base.length(cs::CspondSet) = length(cs.features)
Base.size(cs::CspondSet) = (length(cs),)
Base.iterate(cs::CspondSet, state = 1) = state > length(cs) ? nothing : (cs.features[state], state + 1)
Base.push!(cs::CspondSet{<:Vector}, x) = (push!(cs.features, x); cs)
Base.append!(cs::CspondSet{<:Vector}, xs) = (append!(cs.features, xs); cs)
Base.isempty(cs::CspondSet) = isempty(cs.features)
Base.eltype(::Type{CspondSet{C}}) where {C} = eltype(C)

# NOTE (#21): `point_csponds` is deleted — endpoint extraction is VGC's lazy
# lift layer now: `points(cs)` (centroid both sides), `planar_lift(cs)`
# (z=0-lifted 3D sources for absolute pose), or `lift(cs; source, target)`
# for custom extractors. Attribute columns (score/ratio) travel with the
# lift automatically.
