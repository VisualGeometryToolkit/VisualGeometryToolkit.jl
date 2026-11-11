# =============================================================================
# ViewGraph — the frames of a sequence and their verified pairs
# =============================================================================
#
# One vertex per frame (any label type), one edge per verified pair keyed by the
# canonical pair `(i, j)`, `i < j`, carrying the caller's edge data. The name and
# the shape are the field's (Theia's and GLOMAP's `ViewGraph`, keyed by an image
# pair). The graph algorithms are Graphs.jl's; what is written here is the
# label bookkeeping and the geometry they cannot know: chaining relative
# transforms along a spanning tree, and the loop and orientation checks of
# rotation edges.
#
# The edge dictionary is the source of truth: `edgelist` returns it sorted, and
# a ViewGraph is rebuilt from it, so the graph needs no file format of its own.
# =============================================================================

"""
    ViewGraph{K, E}

The view graph of a set of frames: vertices labelled `K` (frame ids of any
ordered type), one edge per verified pair `(i, j)` with `i < j`, carrying data
`E`. An edge's data is its relative transform, or VisualGeometryToolkitCore's
`Attributed(transform, (; weight))` carrying a fixed non-negative weight (an
inlier count; a bare transform weighs 1). The transform takes frame `i`'s
frame into frame `j`'s — for a rotation `R_ij`, `f_j = R_ij·f_i` — as COLMAP's
`cam2_from_cam1`.

Build it with [`ViewGraph(views, edges)`](@ref ViewGraph(::AbstractVector, ::AbstractDict))
or from the edges alone. Read it with `vg[i, j]`, `haskey(vg, i, j)`,
[`views`](@ref), [`edgelist`](@ref) and [`adjacent_views`](@ref); its structure
with [`components`](@ref), [`articulation`](@ref) and [`bridges`](@ref).
"""
struct ViewGraph{K, E}
    views::Vector{K}
    index::Dict{K, Int}               # cache: views → vertex, built once by the constructor
    graph::Graphs.SimpleGraph{Int}    # cache: the structure of `edges`, for Graphs.jl's algorithms
    edges::Dict{Tuple{K, K}, E}
end

"""
    ViewGraph(views, edges) -> ViewGraph
    ViewGraph(edges) -> ViewGraph

The view graph of `views` (frame labels, kept in the given order) with the
`edges`, an `AbstractDict` keyed by pairs `(i, j)`, `i < j` (a pair given the
other way round is an `ArgumentError`: its data would have to be inverted,
which only the caller knows how to do). A pair naming a frame not in `views`,
or a frame paired with itself, is an `ArgumentError`. The one-argument form
takes the views from the edges, sorted.
"""
function ViewGraph(views::AbstractVector, edges::AbstractDict)
    K = eltype(views)
    vs = collect(K, views)
    index = Dict(k => v for (v, k) in enumerate(vs))
    length(index) == length(vs) || throw(ArgumentError("ViewGraph: a view is listed twice"))
    E = valtype(edges)
    g = Graphs.SimpleGraph(length(vs))
    es = Dict{Tuple{K, K}, E}()
    for ((i, j), e) in edges
        (haskey(index, i) && haskey(index, j)) ||
            throw(ArgumentError("ViewGraph: the pair $((i, j)) names a view that is not in the graph"))
        isless(i, j) ||
            throw(ArgumentError("ViewGraph: the pair $((i, j)) is not canonical; give it as $((j, i)) with its data inverted"))
        Graphs.add_edge!(g, index[i], index[j])
        es[(i, j)] = e
    end
    return ViewGraph{K, E}(vs, index, g, es)
end
ViewGraph(edges::AbstractDict) = ViewGraph(sort!(unique!([k for p in keys(edges) for k in p])), edges)

"""
    views(vg::ViewGraph) -> Vector

The frame labels of `vg`, in the order it was built with.
"""
views(vg::ViewGraph) = vg.views

"""
    adjacent_views(vg::ViewGraph, v) -> Vector

The views paired with `v`, sorted.
"""
adjacent_views(vg::ViewGraph, v) = sort!([vg.views[u] for u in Graphs.neighbors(vg.graph, vg.index[v])])

"""
    label_components(vg::ViewGraph) -> Dict

Each view's component number, `1` the largest (GLOMAP's `MarkConnectedComponents`):
the policy of one gauge per component.
"""
label_components(vg::ViewGraph) = Dict(v => c for (c, comp) in enumerate(components(vg)) for v in comp)

"""
    induced(vg::ViewGraph, keep) -> ViewGraph

The view graph on the views in `keep` and the edges between them, the views in
`vg`'s order.
"""
function induced(vg::ViewGraph{K, E}, keep) where {K, E}
    ks = Set(keep)
    vs = [v for v in vg.views if v in ks]
    return ViewGraph(vs, Dict{Tuple{K, K}, E}(p => e for (p, e) in vg.edges if p[1] in ks && p[2] in ks))
end

"""
    keep_largest_component(vg::ViewGraph) -> ViewGraph

The largest connected component (Theia's `GetLargestConnectedComponentIds`,
OpenCV's `leaveBiggestComponent`): the SfM policy.
"""
keep_largest_component(vg::ViewGraph) = isempty(vg.views) ? vg : induced(vg, first(components(vg)))

_pair(vg::ViewGraph, a::Int, b::Int) =                 # a graph edge as its label pair (the labelled-graph core, observation_graph.jl)
    (i = vg.views[a]; j = vg.views[b]; isless(i, j) ? (i, j) : (j, i))

# An edge's relative transform and its weight are VGC's `value_of` and `weight_of`: the value and the
# `weight` attribute of an `Attributed` edge; a bare transform is its own value and weighs 1.

"""
    spanning_tree(vg::ViewGraph) -> Vector{Tuple}

The maximum spanning forest by the edges' weights (Graphs.jl's `kruskal_mst`,
maximizing): the strongest edges that tie every component together, as sorted
pairs. Theia and OpenCV seed absolute orientations from it, weighted by
inlier counts. An edge of non-positive weight is an `ArgumentError`.
"""
function spanning_tree(vg::ViewGraph)
    n = length(vg.views)
    I = Int[]; J = Int[]; V = Float64[]
    for ((i, j), e) in vg.edges
        a, b = vg.index[i], vg.index[j]
        x = VgtkCore.weight_of(e)
        x > 0 || throw(ArgumentError("spanning_tree: the edge $((i, j)) has a non-positive weight $x"))
        append!(I, (a, b)); append!(J, (b, a)); append!(V, (x, x))
    end
    w = SparseArrays.sparse(I, J, V, n, n)
    return sort!([_pair(vg, Graphs.src(e), Graphs.dst(e)) for e in Graphs.kruskal_mst(vg.graph, w; minimize = false)])
end

"""
    chain(vg::ViewGraph, root; compose = *, inverse = inv, identity) -> Dict

Absolute transforms from relative ones by walking the maximum spanning tree
([`spanning_tree`](@ref)) out of `root`: each view `k` of `root`'s component
gets `world_from_k`, `root` the `identity`, an edge's transform `j_from_i`
taking view `i`'s frame into view `j`'s. Going from `i` to `j` along the tree,
`world_from_j = world_from_i ∘ inverse(j_from_i)`; from `j` to `i`,
`world_from_i = world_from_j ∘ j_from_i`. Views outside `root`'s component get
no entry. With rotations (`identity = one(RotMatrix{3, Float64})`) this is the
spanning-tree seed of rotation averaging (`d = R_k·f_k`).
"""
function chain(vg::ViewGraph, root; compose = *, inverse = inv, identity)
    haskey(vg.index, root) || throw(ArgumentError("chain: the root $root is not a view of the graph"))
    tree = Graphs.SimpleGraph(length(vg.views))
    foreach(((i, j),) -> Graphs.add_edge!(tree, vg.index[i], vg.index[j]), spanning_tree(vg))
    r = vg.index[root]
    parent = Graphs.bfs_parents(tree, r)                               # 0: outside root's component
    depth = Graphs.gdistances(tree, r)
    out = Dict{eltype(vg.views), Any}(root => identity)               # untyped: `identity` (e.g. `I`) and the composed transforms may differ in type; narrowed once below
    for v in sort!(filter(v -> v != r && parent[v] != 0, eachindex(parent)); by = v -> depth[v])   # a parent before its children
        u, w = vg.views[parent[v]], vg.views[v]
        out[w] = isless(u, w) ? compose(out[u], inverse(VgtkCore.value_of(vg[u, w]))) : compose(out[u], VgtkCore.value_of(vg[w, u]))
    end
    return Dict(k => T for (k, T) in out)                             # the narrowest value type the transforms share
end

"""
    triangles(vg::ViewGraph) -> Vector{NTuple{3}}

The 3-cycles `(i, j, k)`, `i < j < k`, every pair an edge, sorted.
"""
function triangles(vg::ViewGraph)
    out = NTuple{3, eltype(vg.views)}[]
    for (i, j) in keys(vg.edges), u in Graphs.neighbors(vg.graph, vg.index[j])
        k = vg.views[u]
        isless(j, k) && haskey(vg, i, k) && push!(out, (i, j, k))
    end
    return sort!(out)
end

"""
    filter_cycles(vg::ViewGraph; tol, residual = rotation_loop_angle, keep_untested = true) -> ViewGraph

The view graph without the edges no consistent loop supports (Theia's
`FilterViewGraphCyclesByRotation`): over the [`triangles`](@ref) `(i, j, k)`,
`residual(vg[i, j], vg[j, k], vg[i, k])` measures how far the loop is from
closing, and an edge stays when at least one of its triangles has
`residual <= tol`. An edge in no triangle stays when `keep_untested` (a
sequence's chain edges have none) and goes otherwise. The default residual is
[`rotation_loop_angle`](@ref), for rotation edges.

A triangle cannot say which of its edges is wrong, so a correct edge whose
every triangle runs through a wrong one goes too, and wrong edges clustered on
a sequence's backbone can disconnect the graph; a removal that worst-first
re-checks does no better there (the edges' loops all run through the cluster).
"""
function filter_cycles(vg::ViewGraph{K, E}; tol::Real, residual = rotation_loop_angle, keep_untested::Bool = true) where {K, E}
    tested = Set{Tuple{K, K}}(); good = Set{Tuple{K, K}}()
    for (i, j, k) in triangles(vg)
        loop = ((i, j), (j, k), (i, k))
        union!(tested, loop)
        residual(vg[i, j], vg[j, k], vg[i, k]) <= tol && union!(good, loop)
    end
    keep = Dict{Tuple{K, K}, E}(p => e for (p, e) in vg.edges if p in good || (keep_untested && p ∉ tested))
    return ViewGraph(vg.views, keep)
end

"""
    rotation_loop_angle(ij, jk, ik) -> Float64

The angle (radians) by which the rotations of a triangle's edges fail to close:
`f_k = R_jk·R_ij·f_i` should equal `R_ik·f_i`, so the angle of `R_ikᵀ·R_jk·R_ij`.
The residual of [`filter_cycles`](@ref) for rotation edges (each edge a rotation
or an `Attributed` one).
"""
rotation_loop_angle(ij, jk, ik) = Rotations.rotation_angle(RotMatrix{3}(_m3(VgtkCore.value_of(ik))' * _m3(VgtkCore.value_of(jk)) * _m3(VgtkCore.value_of(ij))))

_m3(R) = SMatrix{3, 3, Float64, 9}(R)

"""
    filter_by_orientation(vg::ViewGraph, world_from; tol) -> ViewGraph

The view graph without the rotation edges that disagree with the absolute
rotations `world_from` (a `Dict` view => `R_k`, `d = R_k·f_k`, as
[`fit_rotation_averaging`](@ref) returns): an edge `(i, j)` goes when the angle of
`R_j·R_ij·R_iᵀ` exceeds `tol` radians (Theia's `FilterViewPairsFromOrientation`).
An edge with a view missing from `world_from` stays.
"""
function filter_by_orientation(vg::ViewGraph{K, E}, world_from::AbstractDict; tol::Real) where {K, E}
    agrees(i, j, e) = !(haskey(world_from, i) && haskey(world_from, j)) ||
        Rotations.rotation_angle(RotMatrix{3}(_m3(world_from[j]) * _m3(VgtkCore.value_of(e)) * _m3(world_from[i])')) <= tol
    return ViewGraph(vg.views, Dict{Tuple{K, K}, E}(p => e for (p, e) in vg.edges if agrees(p..., e)))
end
