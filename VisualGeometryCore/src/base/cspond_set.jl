# =============================================================================
# Csponds — columnar correspondences with attribute columns
# =============================================================================
#
# The container half of the solver/correspondence layering (#21):
#
#   Csponds{S,T}       columnar source/target storage + index-aligned
#                        attribute columns (score, ratio, weights, …)
#   sources / targets    THE access protocol — replaces `.first`/`.second`
#                        duck-typing; has methods for Csponds, the legacy
#                        StructArray{<:Pair} layout, Attributed-wrapped
#                        (scored) vectors, and plain Pair vectors
#   lift / points /      lazy endpoint views (MappedColumn) — replace VGF's
#   planar_lift          point_csponds and the hand-written z=0 lifts
#
# Correspondence ELEMENTS stay `Cspond{S,T} = Pair{S,T}` — `cs[i]` returns a
# Pair, so element-wise consumers (sampson_distance, constraint_with_jacobians)
# are unaffected by the container.

# -----------------------------------------------------------------------------
# MappedColumn — lazy element-wise view used by lift()
# -----------------------------------------------------------------------------

"""
    MappedColumn(f, parent::AbstractVector)

Lazy `map(f, parent)` view: `m[i] == f(parent[i])`, no allocation, no
materialization. Element type is inferred via `Base.promote_op`.

Used by [`lift`](@ref) so that endpoint conversions (centroid extraction,
z=0 planar lifts, bearing lifts) cost nothing until a solver actually
indexes the column.
"""
struct MappedColumn{T, F, P <: AbstractVector} <: AbstractVector{T}
    f::F
    parent::P
end

function MappedColumn(f::F, parent::P) where {F, P <: AbstractVector}
    T = Base.promote_op(f, eltype(P))
    # Abstract-eltype parents (e.g. Vector{SomeUnionAllAlias}) defeat
    # inference; narrow from the first element — the `collect` convention —
    # so typed downstream signatures still match. Access converts, so an
    # inhomogeneous parent fails loudly rather than lying about eltype.
    if !isconcretetype(T) && !isempty(parent)
        T = typeof(f(@inbounds parent[firstindex(parent)]))
    end
    return MappedColumn{T, F, P}(f, parent)
end

Base.size(m::MappedColumn) = size(getfield(m, :parent))
Base.IndexStyle(::Type{<:MappedColumn}) = IndexLinear()
Base.@propagate_inbounds Base.getindex(m::MappedColumn{T}, i::Int) where {T} =
    convert(T, getfield(m, :f)(getfield(m, :parent)[i]))

# -----------------------------------------------------------------------------
# Csponds
# -----------------------------------------------------------------------------

"""
    Csponds(sources, targets; attrs...)

Columnar correspondence set: parallel `sources`/`targets` vectors plus any
number of index-aligned attribute columns (`score`, `ratio`, `weights`,
`covariance`, …) that filter and view IN LOCKSTEP with the endpoints.

`cs[i]` returns the i-th correspondence as a `Cspond` (= `Pair`);
`cs[mask]` / `view(cs, idx)` subset every column together — no more
hand-written `scored[keep]; weights[keep]` co-filtering.

Attribute columns are reached by property: `cs.score`, `cs.ratio`;
presence is queried with [`hasattr`](@ref). Endpoint columns are reached
through the protocol verbs [`sources`](@ref) / [`targets`](@ref) (NOT
`.first`/`.second` — that duck-typing is retired).

# Example
```julia
cs = Csponds(src_pts, dst_pts; score = scores, ratio = ratios)
sources(cs)          # the source column
cs.score             # the score column
inliers = cs[mask]   # endpoints AND attributes filtered together
```
"""
struct Csponds{S, T, VS <: AbstractVector{S}, VT <: AbstractVector{T}, A <: NamedTuple} <:
    AbstractVector{Cspond{S, T}}
    sources::VS
    targets::VT
    attrs::A

    function Csponds(
            sources::VS, targets::VT, attrs::A
        ) where {S, T, VS <: AbstractVector{S}, VT <: AbstractVector{T}, A <: NamedTuple}
        n = length(sources)
        length(targets) == n ||
            throw(ArgumentError("sources ($n) and targets ($(length(targets))) must have the same length"))
        for (k, col) in pairs(attrs)
            col isa AbstractVector ||
                throw(ArgumentError("attribute column :$k must be an AbstractVector"))
            length(col) == n ||
                throw(ArgumentError("attribute column :$k has length $(length(col)), expected $n"))
        end
        return new{S, T, VS, VT, A}(sources, targets, attrs)
    end
end

Csponds(sources::AbstractVector, targets::AbstractVector; attrs...) =
    Csponds(sources, targets, values(attrs))

Base.size(cs::Csponds) = (length(getfield(cs, :sources)),)
Base.IndexStyle(::Type{<:Csponds}) = IndexLinear()
Base.@propagate_inbounds Base.getindex(cs::Csponds, i::Int) =
    getfield(cs, :sources)[i] => getfield(cs, :targets)[i]

# Whole-set subsetting: every column filters together.
Base.@propagate_inbounds Base.getindex(cs::Csponds, I::AbstractVector) =
    Csponds(
    getfield(cs, :sources)[I], getfield(cs, :targets)[I],
    map(col -> col[I], getfield(cs, :attrs))
)

# `I` typed like `getindex`'s: untyped, it was ambiguous with any package's
# `view(::AbstractArray, ::SpecialIndex)` (BandedMatrices' `Band`, flagged by Aqua).
Base.view(cs::Csponds, I::Union{AbstractVector, Colon}) =
    Csponds(
    view(getfield(cs, :sources), I), view(getfield(cs, :targets), I),
    map(col -> view(col, I), getfield(cs, :attrs))
)

function Base.getproperty(cs::Csponds, k::Symbol)
    k === :sources && return getfield(cs, :sources)
    k === :targets && return getfield(cs, :targets)
    k === :attrs && return getfield(cs, :attrs)
    return getproperty(getfield(cs, :attrs), k)
end

Base.propertynames(cs::Csponds) =
    (:sources, :targets, :attrs, keys(getfield(cs, :attrs))...)

"""
    hasattr(cs::Csponds, k::Symbol) -> Bool

Whether the correspondence set carries the attribute column `k`
(e.g. `hasattr(cs, :score)` before reading `cs.score`).
"""
hasattr(cs::Csponds, k::Symbol) = haskey(getfield(cs, :attrs), k)

"""
    withattrs(cs::Csponds; attrs...) -> Csponds

A copy of `cs` with the given attribute columns merged in (same-name
columns replaced). Endpoint columns are shared, not copied. The idiom for
attaching derived per-correspondence data — e.g. IRLS weights after a
fit — so it subsets in lockstep from then on.
"""
withattrs(cs::Csponds; attrs...) =
    Csponds(
    getfield(cs, :sources), getfield(cs, :targets),
    merge(getfield(cs, :attrs), values(attrs))
)

# -----------------------------------------------------------------------------
# sources / targets — the access protocol
# -----------------------------------------------------------------------------

"""
    sources(cs) -> AbstractVector

The source column of a correspondence container. Protocol verb — defined
for [`Csponds`](@ref), the legacy `StructArray{<:Pair}` columnar layout,
`Attributed`-wrapped (scored) correspondence vectors, and plain
`AbstractVector{<:Pair}` (lazy view). Solver code MUST go through
`sources`/`targets` rather than `.first`/`.second`.
"""
sources(cs::Csponds) = getfield(cs, :sources)

"""
    targets(cs) -> AbstractVector

The target column of a correspondence container. See [`sources`](@ref).
"""
targets(cs::Csponds) = getfield(cs, :targets)

sources(cs::StructArrays.StructArray{<:Pair}) = cs.first
targets(cs::StructArrays.StructArray{<:Pair}) = cs.second

# NamedTuple-element columnar layout ((first = …, second = …)) — the other
# spelling the old `.first` duck-typing admitted.
sources(cs::StructArrays.StructArray{<:NamedTuple{(:first, :second)}}) = cs.first
targets(cs::StructArrays.StructArray{<:NamedTuple{(:first, :second)}}) = cs.second

# Scored legacy layout: StructArray{Attributed{Cspond,⋅}} nests a
# StructArray{Cspond} under :value.
sources(cs::StructArrays.StructArray{<:Attributed{<:Pair}}) = sources(cs.value)
targets(cs::StructArrays.StructArray{<:Attributed{<:Pair}}) = targets(cs.value)

# Row-major fallbacks: lazy, allocation-free.
sources(cs::AbstractVector{<:Pair}) = MappedColumn(first, cs)
targets(cs::AbstractVector{<:Pair}) = MappedColumn(last, cs)
sources(cs::AbstractVector{<:Attributed{<:Pair}}) =
    MappedColumn(a -> first(getfield(a, :value)), cs)
targets(cs::AbstractVector{<:Attributed{<:Pair}}) =
    MappedColumn(a -> last(getfield(a, :value)), cs)

# Attribute columns survive lift(); legacy containers contribute none
# (their scores are adapted at problem construction, not here).
attrs_of(cs::Csponds) = getfield(cs, :attrs)
attrs_of(::AbstractVector) = NamedTuple()

# A per-correspondence attribute column by name, `nothing` when the container
# carries none: the one reader of an optional column (`weights`, a PROSAC
# `score`, …). The attribute names are in the Csponds type, so the branch folds.
attribute_column(cs::Csponds, k::Symbol) = hasattr(cs, k) ? getproperty(cs, k) : nothing
attribute_column(::Any, ::Symbol) = nothing

# The per-correspondence `weights` column: the one place an estimator reads
# per-cspond weights (never a keyword).
weights_column(cs) = attribute_column(cs, :weights)

# A `weights` vector's contract, the one check for every estimator that takes
# FIXED weights (a `weights` column, `tls_line`'s, `zhang_intrinsics`'): every
# entry finite and non-negative (+Inf included: rejected), and with `n` given,
# one weight per datum. A violation is the caller's error, named with its index —
# never `sqrt`'s `DomainError` or a silently NaN solve downstream. `nothing` (no
# weights) is admissible. Returns `w`.
function check_weights(w::AbstractVector)
    for i in eachindex(w)
        (isfinite(w[i]) && w[i] >= 0) || throw(
            ArgumentError("weights[$i] = $(w[i]); weights must be finite and non-negative")
        )
    end
    return w
end
check_weights(::Nothing) = nothing
function check_weights(w::AbstractVector, n::Integer)
    length(w) == n || throw(ArgumentError("$(length(w)) weights for $n data; need one weight per datum"))
    return check_weights(w)
end
check_weights(::Nothing, ::Integer) = nothing

# -----------------------------------------------------------------------------
# lift / points / planar_lift — lazy endpoint views
# -----------------------------------------------------------------------------

_lift_col(::typeof(identity), col) = col
_lift_col(f, col) = MappedColumn(f, col)

"""
    lift(cs; source = identity, target = identity) -> Csponds

Lazy endpoint conversion: returns a [`Csponds`](@ref) whose source/target
columns apply `source`/`target` element-wise on access (no materialization);
attribute columns are forwarded untouched. Works on any container that
satisfies the [`sources`](@ref)/[`targets`](@ref) protocol.

This replaces eager extraction helpers (`point_csponds`) and hand-written
z=0 lifts: compose whatever endpoint conversion the solver needs at the
call site, pay only when the solver indexes.
"""
lift(cs; source = identity, target = identity) =
    Csponds(
    _lift_col(source, sources(cs)),
    _lift_col(target, targets(cs)),
    attrs_of(cs)
)

"""
    points(cs) -> Csponds

[`lift`](@ref) both endpoints through [`centroid`](@ref): detections,
frames, discs, `Attributed`/`Uncertain` wrappers — anything with a
`centroid` — become bare points, lazily.
"""
points(cs) = lift(cs; source = centroid, target = centroid)

"""
    planar_lift(cs) -> Csponds

[`points`](@ref) with the source side additionally lifted onto the z=0
plane: 2D source centroids become `Point3(x, y, 0)` (the planar-board
world convention), targets become 2D centroids. The correspondence set is
then shaped for absolute-pose estimation.
"""
planar_lift(cs) = lift(cs; source = _z0_lift, target = centroid)

@inline function _z0_lift(s)
    p = centroid(s)
    z = zero(eltype(p))
    return Point3{eltype(p)}(p[1], p[2], z)
end

# -----------------------------------------------------------------------------
# centroid — endpoint-anchor closure of the protocol
# -----------------------------------------------------------------------------
# `points`/`planar_lift` need `centroid` to be total over everything that
# appears in an endpoint column. Primitives and detections already have
# methods; these close the remaining cases (plain points are their own
# centroid; Attributed unwraps — more specific downstream methods, e.g.
# VGF's Attributed{<:FeatureFrame}, still win).

centroid(p::StaticVector{2}) = p
centroid(p::StaticVector{3}) = p
centroid(a::Attributed) = centroid(getfield(a, :value))

# -----------------------------------------------------------------------------
# Grouped data — the block order of a dict of groups
# -----------------------------------------------------------------------------

"""
    block_order(groups::AbstractDict, given::AbstractDict...) -> Vector{K}

The block order of grouped data: the sorted keys of `groups` (the order a
grouped estimator stacks its per-group unknowns and stores on its result). Every
`given` dict — the per-group models a caller supplies, keyed by the same `K` —
must have exactly the same key set: an `ArgumentError` otherwise.
"""
function block_order(groups::AbstractDict, given::AbstractDict...)
    ids = sort!(collect(keys(groups)))
    for g in given
        (length(g) == length(ids) && all(in(keys(g)), ids)) || throw(
            ArgumentError("grouped data: key sets differ — $(ids) vs $(sort!(collect(keys(g))))")
        )
    end
    return ids
end

"""
    split_blocks(v::AbstractVector, ids::AbstractVector{K}, counts) -> Dict{K, Vector}

The inverse of stacking per-group vectors in block order: `v`'s consecutive
blocks of `counts[j]` entries, keyed by `ids[j]` (a copy per group). A
`DimensionMismatch` unless `ids` and `counts` have one entry per group and the
counts sum to `length(v)`.
"""
function split_blocks(v::AbstractVector, ids::AbstractVector{K}, counts) where {K}
    length(ids) == length(counts) || throw(
        DimensionMismatch("split_blocks: $(length(ids)) ids for $(length(counts)) counts")
    )
    sum(counts; init = 0) == length(v) || throw(
        DimensionMismatch("split_blocks: counts sum to $(sum(counts; init = 0)), the vector has $(length(v)) entries")
    )
    out = Dict{K, Vector{eltype(v)}}()
    o = firstindex(v) - 1
    for (id, c) in zip(ids, counts)
        out[id] = v[(o + 1):(o + c)]
        o += c
    end
    return out
end
