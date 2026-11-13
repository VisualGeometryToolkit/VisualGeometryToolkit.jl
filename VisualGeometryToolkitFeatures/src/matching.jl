# =============================================================================
# N-way correspondences
# =============================================================================
#
# The pairwise layer — `Correspondence{S,T} = Pair{S,T}`, the `correspondences` builder and the
# columnar `Correspondences` it returns — is VisualGeometryToolkitCore's; nn_match.jl adds the
# `correspondences` method for NNMatch lists. CorrespondenceSet is the N-way form, defined here.
#
# CorrespondenceSet wraps any AbstractVector — fixed-size (SVector) or growable (Vector).
#
#   CorrespondenceSet((a, b, c))          # fixed: SVector{3,T}
#   CorrespondenceSet([a, b])             # variable: Vector{T}, supports push!
#   CorrespondenceSet(T)                  # empty growable set of eltype T
#

using VisualGeometryToolkitCore: Correspondence
import VisualGeometryToolkitCore: correspondences

"""
    CorrespondenceSet(features::AbstractVector)
    CorrespondenceSet(features::Tuple)
    CorrespondenceSet(T::Type)

An N-way correspondence: the same scene point observed as one feature in each
of several images, where `Correspondence` is the two-way case. It wraps a vector
of features and iterates, indexes and has a length like one. A tuple becomes a
fixed-size `SVector`; a `Vector` (or `CorrespondenceSet(T)`, an empty one of element
type `T`) grows with `push!` and `append!`.

```jldoctest
julia> using VisualGeometryToolkitFeatures

julia> track = CorrespondenceSet(Int);

julia> push!(track, 3); push!(track, 7);

julia> length(track), track[2], collect(track)
(2, 7, [3, 7])

julia> length(CorrespondenceSet((1, 2, 3)))
3
```
"""
struct CorrespondenceSet{C <: AbstractVector}
    features::C
end
CorrespondenceSet(features::Tuple) = CorrespondenceSet(SVector(features))
CorrespondenceSet(::Type{T}) where {T} = CorrespondenceSet(T[])

Base.getindex(cs::CorrespondenceSet, i::Int) = cs.features[i]
Base.length(cs::CorrespondenceSet) = length(cs.features)
Base.size(cs::CorrespondenceSet) = (length(cs),)
Base.iterate(cs::CorrespondenceSet, state = 1) = state > length(cs) ? nothing : (cs.features[state], state + 1)
Base.push!(cs::CorrespondenceSet{<:Vector}, x) = (push!(cs.features, x); cs)
Base.append!(cs::CorrespondenceSet{<:Vector}, xs) = (append!(cs.features, xs); cs)
Base.isempty(cs::CorrespondenceSet) = isempty(cs.features)
Base.eltype(::Type{CorrespondenceSet{C}}) where {C} = eltype(C)

# Endpoint extraction is VGC's lazy
# lift layer now: `points(cs)` (centroid both sides), `planar_lift(cs)`
# (z=0-lifted 3D sources for absolute pose), or `lift(cs; source, target)`
# for custom extractors. Attribute columns (score/ratio) travel with the
# lift automatically.
