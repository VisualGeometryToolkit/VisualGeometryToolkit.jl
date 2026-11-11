# =============================================================================
# ObservationGraph — what each frame observes
# =============================================================================
#
# The bipartite graph a calibration or a reconstruction lives on: one vertex per
# OBSERVER (a frame, a capture instant) and per OBSERVED thing (a target, a
# board, a track), one edge per observation, carrying the caller's data (the
# observation's correspondences). OpenSfM's tracks manager (shots × tracks) and
# a multi-camera calibration's frame × board graph have this shape; no library
# gives the type a standard name. The two sides keep their own labels, so a
# frame `1` and a target `1` never collide.
#
# Projecting onto one side gives that side's co-visibility as a ViewGraph
# (OpenSfM's `GetAllPairsConnectivity`). The observation dictionary is the source
# of truth: `edgelist` returns it, sorted, and the graph is rebuilt from it.
# =============================================================================

"""
    ObservationGraph{A, B, E}

The bipartite graph of observations: observers labelled `A` (frames), observed
things labelled `B` (targets, boards, tracks), one edge per observation `(a, b)`
carrying data `E`. Build it with [`ObservationGraph(edges)`](@ref
ObservationGraph(::AbstractDict)); read it with `og[a, b]`, `haskey(og, a, b)`,
[`observers`](@ref), [`observed`](@ref), [`edgelist`](@ref),
[`observed_in`](@ref) and [`observers_of`](@ref); its structure with
[`components`](@ref) and [`articulation`](@ref); its co-visibility with
[`covisibility`](@ref).
"""
struct ObservationGraph{A, B, E}
    observers::Vector{A}
    observed::Vector{B}
    observer_index::Dict{A, Int}
    observed_index::Dict{B, Int}     # graph vertex = length(observers) + this index (`_vertex`)
    graph::Graphs.SimpleGraph{Int}
    edges::Dict{Tuple{A, B}, E}
end

"""
    ObservationGraph(edges) -> ObservationGraph

The observation graph of `edges`, an `AbstractDict` keyed by `(observer,
observed)` pairs; the observers and the observed things are those the keys
name, each side sorted.
"""
function ObservationGraph(edges::AbstractDict)
    KT = keytype(edges)
    (KT <: Tuple && length(KT.parameters) == 2) ||
        throw(ArgumentError("ObservationGraph: the edges must be keyed by (observer, observed) pairs, not $KT"))
    A, B = KT.parameters
    E = valtype(edges)
    as = sort!(unique!(A[p[1] for p in keys(edges)]))
    bs = sort!(unique!(B[p[2] for p in keys(edges)]))
    ai = Dict(a => k for (k, a) in enumerate(as))
    bi = Dict(b => k for (k, b) in enumerate(bs))
    g = Graphs.SimpleGraph(length(as) + length(bs))
    og = ObservationGraph{A, B, E}(as, bs, ai, bi, g, Dict{Tuple{A, B}, E}(edges))
    for (a, b) in keys(edges)
        Graphs.add_edge!(g, ai[a], _vertex(og, b))
    end
    return og
end

"""
    observers(og::ObservationGraph) -> Vector

The observers (frames), sorted.
"""
observers(og::ObservationGraph) = og.observers

"""
    observed(og::ObservationGraph) -> Vector

The observed things (targets, boards, tracks), sorted.
"""
observed(og::ObservationGraph) = og.observed

# Graph vertices: the observers first, then the observed things.
_vertex(og::ObservationGraph, b) = length(og.observers) + og.observed_index[b]
_vertex_label(og::ObservationGraph, v::Int) =
    v <= length(og.observers) ? (true, og.observers[v]) : (false, og.observed[v - length(og.observers)])

"""
    observed_in(og::ObservationGraph, a) -> Vector

The things observer `a` observes, sorted.
"""
observed_in(og::ObservationGraph, a) = sort!([_vertex_label(og, v)[2] for v in Graphs.neighbors(og.graph, og.observer_index[a])])

"""
    observers_of(og::ObservationGraph, b) -> Vector

The observers that observe `b`, sorted.
"""
observers_of(og::ObservationGraph, b) = sort!([og.observers[v] for v in Graphs.neighbors(og.graph, _vertex(og, b))])

function _vertex_labels(og::ObservationGraph{A, B}, vs) where {A, B}
    as = A[]; bs = B[]
    for v in vs
        side, l = _vertex_label(og, v)
        side ? push!(as, l) : push!(bs, l)
    end
    return (; observers = sort!(as), observed = sort!(bs))
end

"""
    label_components(og::ObservationGraph) -> @NamedTuple{observers::Dict, observed::Dict}

Each observer's and each observed thing's component number, `1` the largest.
"""
function label_components(og::ObservationGraph{A, B}) where {A, B}
    la = Dict{A, Int}(); lb = Dict{B, Int}()
    for (k, c) in enumerate(components(og))
        foreach(a -> (la[a] = k), c.observers); foreach(b -> (lb[b] = k), c.observed)
    end
    return (; observers = la, observed = lb)
end

"""
    covisibility(og::ObservationGraph; side = :observers) -> ViewGraph

The co-visibility of one side as a [`ViewGraph`](@ref): two observers (`side =
:observers`) are paired when they observe a common thing, two observed things
(`side = :observed`) when a common observer observes both. Each edge is
`Attributed(shared, (; weight = length(shared)))`, `shared` the sorted labels of
the other side they share and its weight their count (OpenSfM's pair
connectivity), so [`spanning_tree`](@ref) keeps the best-connected pairs.
"""
function covisibility(og::ObservationGraph; side::Symbol = :observers)
    if side === :observers
        labels = og.observers; through = og.observed; members = b -> observers_of(og, b)
    elseif side === :observed
        labels = og.observed; through = og.observers; members = a -> observed_in(og, a)
    else
        throw(ArgumentError("covisibility: side must be :observers or :observed, not :$side"))
    end
    K = eltype(labels); T = eltype(through)
    shared = Dict{Tuple{K, K}, Vector{T}}()
    for x in through
        ms = members(x)
        for p in 1:length(ms), q in (p + 1):length(ms)
            push!(get!(Vector{T}, shared, (ms[p], ms[q])), x)
        end
    end
    return ViewGraph(labels, Dict(pair => VgtkCore.Attributed(xs, (; weight = length(xs))) for (pair, xs) in shared))   # xs is sorted: `through` is
end

# =============================================================================
# The labelled-graph core: what a ViewGraph and an ObservationGraph share
# =============================================================================
#
# Both hold a Graphs.SimpleGraph (`graph`) and the edge dictionary keyed by label pairs (`edges`), the
# source of truth. Each type says only how a vertex set reads as labels (`_vertex_labels`: a ViewGraph's
# sorted views, an ObservationGraph's sorted observers and observed things), how a graph edge reads as
# its label pair (`_pair`), and a component's size and first labels (`_component_size`,
# `_component_labels`); the queries below are written once over them.

const _LabelledGraph = Union{ViewGraph, ObservationGraph}

_vertex_labels(vg::ViewGraph, vs) = sort!(vg.views[vs])
_pair(og::ObservationGraph, a::Int, b::Int) =
    ((s, d) = minmax(a, b); (og.observers[s], og.observed[d - length(og.observers)]))   # observers come first among the vertices
_component_size(c::AbstractVector) = length(c)
_component_size(c::NamedTuple) = length(c.observers) + length(c.observed)
_component_labels(c::AbstractVector) = c
_component_labels(c::NamedTuple) = c.observers

"""
    edgelist(g) -> Vector{Pair}

The edges of a [`ViewGraph`](@ref) (its verified pairs) or an
[`ObservationGraph`](@ref) (its observations) as `(i, j) => data` pairs, sorted
by `(i, j)`: the persistence form the graph is rebuilt from.
"""
edgelist(g::_LabelledGraph) = sort!([p => e for (p, e) in g.edges]; by = first)

Base.getindex(g::_LabelledGraph, i, j) = g.edges[(i, j)]
Base.haskey(g::_LabelledGraph, i, j) = haskey(g.edges, (i, j))

"""
    nedges(g) -> Int

The number of edges of a [`ViewGraph`](@ref) (its verified pairs) or an
[`ObservationGraph`](@ref) (its observations).
"""
nedges(g::_LabelledGraph) = length(g.edges)

"""
    components(g) -> Vector

The connected components, largest first by vertex count, equal sizes by their
smallest label (a [`ViewGraph`](@ref)'s view, an [`ObservationGraph`](@ref)'s
observer). A ViewGraph's component is its sorted views; an ObservationGraph's
is `(; observers, observed)`, each sorted. Poses are comparable only within one
component: a calibration gives each its own gauge.
"""
components(g::_LabelledGraph) =
    sort!([_vertex_labels(g, c) for c in Graphs.connected_components(g.graph)];
          by = c -> (-_component_size(c), first(_component_labels(c))))

"""
    articulation(g)

The cut vertices, whose removal splits their component: a [`ViewGraph`](@ref)'s
sorted views, or an [`ObservationGraph`](@ref)'s `(; observers, observed)` (the
fragile links of a session), each sorted.
"""
articulation(g::_LabelledGraph) = _vertex_labels(g, Graphs.articulation(g.graph))

"""
    bridges(g) -> Vector{Tuple}

The bridge edges, whose removal splits their component, as sorted label pairs:
a [`ViewGraph`](@ref)'s `(i, j)`, `i < j`, an [`ObservationGraph`](@ref)'s
`(a, b)`.
"""
bridges(g::_LabelledGraph) = sort!([_pair(g, Graphs.src(e), Graphs.dst(e)) for e in Graphs.bridges(g.graph)])

"""
    require_connected(g; what = "graph") -> g

`g` (a [`ViewGraph`](@ref) or an [`ObservationGraph`](@ref)) when it is
connected, otherwise an `ArgumentError` that names its components (an
observation graph's by their observers): the policy of a method whose frames
need one common gauge.
"""
function require_connected(g::_LabelledGraph; what::AbstractString = "graph")
    comps = components(g)
    length(comps) <= 1 && return g
    throw(ArgumentError("the $what is not connected; its components are " *
                        join(["[" * join(string.(_component_labels(c)), ", ") * "]" for c in comps], ", ")))
end
