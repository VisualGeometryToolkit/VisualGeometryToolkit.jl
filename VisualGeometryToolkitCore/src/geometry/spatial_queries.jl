# =============================================================================
# SpatialQueries — allocation-free queries on a SpatialIndexing.jl R-tree,
# and its in-place bulk load
# =============================================================================
#
# SpatialIndexing.jl's own region queries (`intersects_with`, `contained_in`)
# are iterators whose state is a heap object per step, and it has no
# nearest-neighbour query. The hot loops that index detections (BlobBoards'
# guided association and tracking) need both without allocating, so they live
# here, on SpatialIndexing's public types (`RTree`, `SpatialElem`, `Rect`) and
# its node internals (`Leaf`, `Branch`, `children`, `mbr`, `id`, `tree.root`),
# which have been stable since SpatialIndexing 0.1.0.
#
# Adapted from github.com/prittjam/SpatialIndexing.jl (an MIT-licensed fork of
# github.com/alyst/SpatialIndexing.jl, Copyright (c) 2018 Alexey Stukalov):
# `foreach_intersecting`, `nearest_neighbor` and its reusable state, there
# added to `src/rtree/query.jl`. Two changes: the fork narrowed `RTree.root` to
# `Union{Branch, Leaf}`, so its queries read the root concretely, while
# upstream declares it the abstract `Node`; here each query splits the root by
# `isa` before the traversal (no dynamic dispatch, no boxing). And the reusable
# state is the workspace form `nearest_neighbor!(state, tree, point)`.
#
# `bulk_load!` is SpatialIndexing 0.1.6's OMT bulk load (`load_omt!` and
# `omt_subtree` in `src/rtree/bulk.jl`) with the fork's changes: it sorts the
# caller's element vector in place instead of sorting copies at every level,
# and sizes each node's child vector up front. Same tree, node for node, at
# about 70% of the time and half the memory of `SpatialIndexing.load!` on
# BlobBoards' detection trees (measured 2026-10-07 on DSC02847: 1.5–1.9k
# points, 145 µs vs 188 µs minimum). It reads more internals: `acquire`,
# `_attach!`, `capacity`, `omt_branch_fill`, `nelems`, `nnodes_perlevel`.
#
# `f::F`: Julia does not specialize a method on a `Function` argument that the
# method only hands on to another call, and `foreach_intersecting` calls `f`
# only through its helpers, so without the type parameter a closure arrives as
# `::Function` and every call to it allocates.
#
# The union split: `Branch.children` is `Union{Vector{Branch}, Vector{Leaf}}`.
# Each traversal tests it once per branch and inlines the leaf level into the
# twig (level-2) loop, because a call across the union boundary boxes its
# arguments.
# =============================================================================

"""
    SpatialQueries

Allocation-free queries on a SpatialIndexing.jl `RTree` (`using
VisualGeometryToolkitCore.SpatialQueries`):

- [`foreach_intersecting`](@ref): call a function on every element whose
  bounding box intersects a query `Rect`, with early exit;
- [`nearest_neighbor`](@ref): the element nearest a point (branch-and-bound),
  within an optional squared radius;
- [`nearest_neighbor!`](@ref) with a [`NearestNeighborState`](@ref): the same
  query with its state supplied, so a batch of queries allocates nothing;
- [`bulk_load!`](@ref): SpatialIndexing's OMT bulk load, sorting the
  caller's elements in place.

The trees are SpatialIndexing's own (`RTree`, `SpatialElem`, `Rect`); build
them with `bulk_load!`, or SpatialIndexing's `insert!` or `load!`.
VisualGeometryToolkitCore's [`bbox`](@ref VisualGeometryToolkitCore.bbox) gives a shape's `Rect`.
"""
module SpatialQueries

using SpatialIndexing: SpatialIndexing, RTree

export foreach_intersecting, nearest_neighbor, nearest_neighbor!, NearestNeighborState, bulk_load!

# The node types, read from SpatialIndexing's internals (see the header).
const _Leaf = SpatialIndexing.Leaf
const _Branch = SpatialIndexing.Branch
const _Rect = SpatialIndexing.Rect
_mbr(x) = SpatialIndexing.mbr(x)

# =============================================================================
# Intersection query
# =============================================================================

"""
    foreach_intersecting(f, tree::RTree{T,N}, rect::SpatialIndexing.Rect{T,N}) -> Bool

Call `f(elem)` on every element of `tree` whose bounding box intersects `rect`
(closed boxes: touching counts). `f` returns `true` to stop the query early and
`false` to continue. Returns `true` when `f` stopped it.

The elements are SpatialIndexing's (`SpatialElem`: `SpatialIndexing.id(elem)`,
`SpatialIndexing.mbr(elem)`), visited in tree order. Allocation-free when `f`
is: unlike `intersects_with`, no iterator state is built per element.

```julia
hits = 0
foreach_intersecting(tree, SpatialIndexing.Rect((0.0, 0.0), (1.0, 1.0))) do elem
    hits += 1
    false                                   # keep going
end
```
"""
function foreach_intersecting(f::F, tree::RTree{T, N, V}, rect::_Rect{T, N}) where {F, T, N, V}
    isempty(tree) && return false
    lo = rect.low
    hi = rect.high
    root = tree.root
    if root isa _Leaf{T, N, V}
        _intersects(_mbr(root), lo, hi) || return false
        return _intersecting_leaf(f, root, lo, hi)
    else
        root = root::_Branch{T, N, V}
        _intersects(_mbr(root), lo, hi) || return false
        return _intersecting_branch(f, root, lo, hi)
    end
end

@inline function _intersects(r::_Rect{T, N}, lo::NTuple{N}, hi::NTuple{N}) where {T, N}
    @inbounds for i in 1:N
        (r.low[i] > hi[i] || r.high[i] < lo[i]) && return false
    end
    return true
end

@inline function _intersecting_leaf(f::F, leaf, lo, hi) where {F}
    @inbounds for elem in leaf.children
        if _intersects(_mbr(elem), lo, hi)
            f(elem) && return true
        end
    end
    return false
end

function _intersecting_branch(f::F, node::_Branch{T, N, V}, lo, hi) where {F, T, N, V}
    cs = node.children
    if cs isa Vector{_Leaf{T, N, V}}
        # Twig: the leaf loop is written out here (see the header).
        @inbounds for leaf in cs
            _intersects(_mbr(leaf), lo, hi) || continue
            for elem in leaf.children
                if _intersects(_mbr(elem), lo, hi)
                    f(elem) && return true
                end
            end
        end
    else
        @inbounds for child in cs::Vector{_Branch{T, N, V}}
            _intersects(_mbr(child), lo, hi) || continue
            _intersecting_branch(f, child, lo, hi) && return true
        end
    end
    return false
end

# =============================================================================
# Nearest-neighbour query (branch-and-bound)
# =============================================================================

"""
    NearestNeighborState(tree::RTree)

The reusable state of [`nearest_neighbor!`](@ref): build one per querying task
and every query through it is allocation-free. It is overwritten by each
query, so a state is never shared between threads.
"""
mutable struct NearestNeighborState{V, F}
    best_sqdist::F
    best::V
    found::Bool
    function NearestNeighborState{V, F}() where {V, F}
        state = new{V, F}()
        state.best_sqdist = convert(F, Inf)
        state.found = false
        return state
    end
end

NearestNeighborState(::RTree{T, N, V}) where {T, N, V} = NearestNeighborState{V, float(T)}()

"""
    nearest_neighbor(tree::RTree{T,N}, point::NTuple{N,<:Number}; max_sqdist = Inf) -> (id, sqdist) | nothing

The element of `tree` nearest `point`: its `SpatialIndexing.id` and its squared
Euclidean distance, measured to the element's bounding box (for point
elements, the point-to-point distance). `nothing` when no element lies within
`max_sqdist` (inclusive) or the tree is empty. Ties keep the element visited
last in tree order.

Depth-first branch-and-bound: a subtree whose bounding box is farther than the
best distance so far (seeded by `max_sqdist`) is skipped, so a query touches
only the neighbourhood of `point`. Allocates its one state; a batch of queries
supplies one to [`nearest_neighbor!`](@ref) instead.
"""
nearest_neighbor(tree::RTree{T, N}, point::NTuple{N, Number}; max_sqdist::Real = Inf) where {T, N} =
    nearest_neighbor!(NearestNeighborState(tree), tree, point; max_sqdist)

"""
    nearest_neighbor!(state::NearestNeighborState, tree::RTree{T,N}, point::NTuple{N,<:Number}; max_sqdist = Inf) -> (id, sqdist) | nothing

[`nearest_neighbor`](@ref) with its state supplied: allocation-free. `state`
comes from [`NearestNeighborState`](@ref)`(tree)` (or from any tree of the same
element type) and is overwritten.
"""
function nearest_neighbor!(
        state::NearestNeighborState{V, F}, tree::RTree{T, N, V}, point::NTuple{N, Number};
        max_sqdist::Real = Inf
    ) where {T, N, V, F}
    isempty(tree) && return nothing
    pt = map(x -> convert(F, x), point)
    state.best_sqdist = convert(F, max_sqdist)
    state.found = false
    root = tree.root
    if root isa _Leaf{T, N, V}
        _nearest_leaf!(state, root, pt)
    else
        _nearest_branch!(state, root::_Branch{T, N, V}, pt)
    end
    state.found || return nothing
    return (SpatialIndexing.id(state.best::V), state.best_sqdist)
end

@inline function _min_sqdist(r::_Rect{T, N}, pt::NTuple{N}) where {T, N}
    F = eltype(pt)
    s = zero(F)
    @inbounds for i in 1:N
        lo = convert(F, r.low[i])
        hi = convert(F, r.high[i])
        d = pt[i] < lo ? lo - pt[i] : (pt[i] > hi ? pt[i] - hi : zero(F))
        s = muladd(d, d, s)
    end
    return s
end

@inline function _nearest_leaf!(state::NearestNeighborState{V, F}, leaf, pt::NTuple{N, F}) where {V, F, N}
    @inbounds for elem in leaf.children
        d2 = _min_sqdist(_mbr(elem), pt)
        if d2 <= state.best_sqdist
            state.best_sqdist = d2
            state.best = elem
            state.found = true
        end
    end
    return nothing
end

function _nearest_branch!(
        state::NearestNeighborState{V, F}, node::_Branch{T, N, V}, pt::NTuple{N, F}
    ) where {V, F, T, N}
    cs = node.children
    if cs isa Vector{_Leaf{T, N, V}}
        @inbounds for leaf in cs
            _min_sqdist(_mbr(leaf), pt) <= state.best_sqdist || continue
            _nearest_leaf!(state, leaf, pt)
        end
    else
        @inbounds for child in cs::Vector{_Branch{T, N, V}}
            _min_sqdist(_mbr(child), pt) <= state.best_sqdist || continue
            _nearest_branch!(state, child, pt)
        end
    end
    return nothing
end

# =============================================================================
# Bulk load (OMT), in place
# =============================================================================

"""
    bulk_load!(tree::RTree{T,N,V}, elems::Vector{V}) -> tree

Bulk-load `elems` into the empty `tree` by SpatialIndexing's OMT method
(Lee & Lee, "OMT: Overlap Minimizing Top-down Bulk Loading Algorithm for
R-tree"), sorting `elems` in place: the caller hands the vector over and must
not rely on its order afterwards. The tree is the one
`SpatialIndexing.load!(tree, elems)` builds, node for node, without the copies
`load!` sorts at every level.

```julia
tree = RTree{Float64, 2}(Int, Nothing)
bulk_load!(tree, [SpatialElem(SpatialIndexing.Rect(p, p), j, nothing) for (j, p) in enumerate(points)])
```
"""
function bulk_load!(tree::RTree{T, N, V}, elems::Vector{V}) where {T, N, V}
    isempty(tree) || throw(ArgumentError("bulk_load!: the tree is not empty"))
    isempty(elems) && return tree
    leaf_fill = SpatialIndexing.capacity(_Leaf, tree)
    branch_fill = SpatialIndexing.omt_branch_fill(tree)
    nbranch_subtrees = min(branch_fill[1] * branch_fill[2], SpatialIndexing.capacity(_Branch, tree))
    height = max(ceil(Int, (log(length(elems)) - log(leaf_fill)) / log(nbranch_subtrees)), 0) + 1
    if height > 1
        maxelems_subtree = leaf_fill * nbranch_subtrees^(height - 2)
        nroot_subtrees = fld1(length(elems), maxelems_subtree)
        nroot_slices = floor(Int, sqrt(nroot_subtrees))
    else
        nroot_subtrees = length(elems)
        nroot_slices = 1
    end
    _sort_by_center!(elems, mod1(height, N))
    resize!(tree.nnodes_perlevel, height)
    fill!(tree.nnodes_perlevel, 0)
    tree.root = _omt_subtree!(tree, elems, height, nroot_slices, nroot_subtrees, leaf_fill, branch_fill)
    tree.nnodes_perlevel[end] = 1
    return tree
end

_sort_by_center!(elems, dim::Int) = sort!(elems; by = elem -> _mbr(elem).low[dim] + _mbr(elem).high[dim])

# The subtree of height `lev` over `elems` (a view into the caller's vector,
# sorted in place by the level above), its root holding `nsubtrees` children
# in `nslices` slices.
function _omt_subtree!(
        tree::RTree, elems::AbstractVector, lev::Integer, nslices::Integer, nsubtrees::Integer,
        leaf_fill::Integer, branch_fill::Tuple{Integer, Integer}
    )
    if length(elems) <= nsubtrees
        if lev == 1         # a leaf holding every element
            node = SpatialIndexing.acquire(tree, _Leaf, lev)
            sizehint!(node.children, length(elems))
            for elem in elems
                SpatialIndexing._attach!(node, elem, tree)
            end
            tree.nelems += length(elems)
            return node
        else                # a branch with a single child
            child = _omt_subtree!(tree, elems, lev - 1, nslices, nsubtrees, leaf_fill, branch_fill)
            node = SpatialIndexing.acquire(tree, _Branch, lev)
            sizehint!(node.children, 1)
            SpatialIndexing._attach!(node, child, tree)
            tree.nnodes_perlevel[lev - 1] += 1
            return node
        end
    end
    node = SpatialIndexing.acquire(tree, _Branch, lev)
    sizehint!(node.children, fld1(length(elems), fld1(length(elems), nsubtrees)))
    nelems_subtree = fld1(length(elems), nsubtrees)
    nelems_slice = nelems_subtree * fld1(nsubtrees, nslices)
    dim = mod1(lev, ndims(tree))   # the sort dimension cycles through the levels
    child_subtrees = min(branch_fill[1] * branch_fill[2], SpatialIndexing.capacity(_Branch, tree))
    for i in 1:nelems_slice:length(elems)
        slice = view(elems, i:min(i + nelems_slice - 1, length(elems)))
        _sort_by_center!(slice, dim)
        for j in 1:nelems_subtree:length(slice)
            child = _omt_subtree!(
                tree, view(slice, j:min(j + nelems_subtree - 1, length(slice))),
                lev - 1, branch_fill[1], child_subtrees, leaf_fill, branch_fill
            )
            SpatialIndexing._attach!(node, child, tree)
            tree.nnodes_perlevel[lev - 1] += 1
        end
    end
    return node
end

end # module SpatialQueries
