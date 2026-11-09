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

"""
    edgelist(og::ObservationGraph) -> Vector{Pair}

The observations as `(a, b) => data` pairs, sorted by `(a, b)`: the persistence
form the graph is rebuilt from.
"""
edgelist(og::ObservationGraph) = sort!([p => e for (p, e) in og.edges]; by = first)

Base.getindex(og::ObservationGraph, a, b) = og.edges[(a, b)]
Base.haskey(og::ObservationGraph, a, b) = haskey(og.edges, (a, b))
nedges(og::ObservationGraph) = length(og.edges)

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

function _split(og::ObservationGraph{A, B}, vs) where {A, B}
    as = A[]; bs = B[]
    for v in vs
        side, l = _vertex_label(og, v)
        side ? push!(as, l) : push!(bs, l)
    end
    return (; observers = sort!(as), observed = sort!(bs))
end

"""
    components(og::ObservationGraph) -> Vector{@NamedTuple{observers, observed}}

The connected components, each its sorted observers and observed things,
largest first by vertex count (equal sizes by their smallest observer, as a
ViewGraph's by their smallest view). Poses are comparable only within one
component: a calibration gives each its own gauge.
"""
components(og::ObservationGraph) =
    sort!([_split(og, c) for c in Graphs.connected_components(og.graph)];
          by = c -> (-(length(c.observers) + length(c.observed)), first(c.observers)))

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
    articulation(og::ObservationGraph) -> @NamedTuple{observers, observed}

The cut vertices: observers and observed things whose removal splits their
component, the fragile links of a session, sorted.
"""
articulation(og::ObservationGraph) = _split(og, Graphs.articulation(og.graph))

"""
    bridges(og::ObservationGraph) -> Vector{Tuple}

The bridge observations: `(a, b)` pairs whose removal splits their component,
sorted.
"""
function bridges(og::ObservationGraph)
    pairs = map(Graphs.bridges(og.graph)) do e
        s, d = minmax(Graphs.src(e), Graphs.dst(e))      # observers come first among the vertices
        (_vertex_label(og, s)[2], _vertex_label(og, d)[2])
    end
    return sort!(pairs)
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

"""
    require_connected(g; what = "graph") -> g

`g` (a [`ViewGraph`](@ref) or an [`ObservationGraph`](@ref)) when it is
connected, otherwise an `ArgumentError` that names its components (an
observation graph's by their observers): the policy of a method whose frames
need one common gauge.
"""
function require_connected(g::Union{ViewGraph, ObservationGraph}; what::AbstractString = "graph")
    comps = components(g)
    length(comps) <= 1 && return g
    throw(ArgumentError("the $what is not connected; its components are " *
                        join(["[" * join(string.(_component_labels(c)), ", ") * "]" for c in comps], ", ")))
end
_component_labels(c::AbstractVector) = c
_component_labels(c::NamedTuple) = c.observers
