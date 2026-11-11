# =============================================================================
# fit_rotation_averaging — robust rotation averaging on a view graph
# =============================================================================
#
# The absolute rotations of a set of frames from their pairwise relative
# rotations (the edges of a view graph, some of them wrong), after Chatterjee &
# Govindu (ICCV 2013, TPAMI 2018): IRLS in the Lie algebra.
#
#   seed      VisualGeometryToolkitCore's `spectral_rotation_averaging` (every edge).
#   repeat    per edge (i, j) the residual rotation E_ij = R_j·R_ij·R_iᵀ (the
#             identity when the edge agrees) and its log r_ij (radians, a
#             3-vector, codimension 3); the robust scale σ̂ = median‖r‖ /
#             √chisq_median(3); the IRLS weights of the `Irls{K}` strategy's
#             loss at σ̂ (irls_weights.jl); the weighted sparse linear least squares
#             of the first-order update ω_j − ω_i = −r_ij over every edge,
#             the gauge frame's ω = 0, and every frame the edges of positive
#             weight no longer tie to the gauge held where it is (one anchor
#             per such component); R_k ← exp(ω_k)·R_k.
#   stop      when an update moves every frame by less than `tol` radians, or at
#             the strategy's `max_iter`.
# =============================================================================

"""
    RotationAveragingMetadata{K}

`FitResult` metadata of [`fit_rotation_averaging`](@ref) at the returned
rotations: per edge `(i, j)`, `residuals` the angle (radians) of the residual
rotation `R_j·R_ij·R_iᵀ` and `irls_weights` the loss's IRLS weight times the
edge's weight; `sigma` the robust scale σ̂ (radians). They forward as properties.
"""
struct RotationAveragingMetadata{K}
    residuals::Dict{Tuple{K, K}, Float64}
    irls_weights::Dict{Tuple{K, K}, Float64}
    sigma::Float64
end

"""
    fit_rotation_averaging(relative; optimization_strategy, tol, seed = :spectral)
    fit_rotation_averaging(vg::ViewGraph; kwargs...)
        -> FitResult{Dict{K, RotMatrix{3, Float64}}, RotationAveragingMetadata{K}}

Robust rotation averaging on a view graph: the absolute rotations `R_k` of the
frames (`d = R_k·f_k`, every frame's directions into the common frame, the
first of the sorted frames the identity — the gauge) from the relative
rotations `relative`, an `AbstractDict` keyed by frame pairs `(i, j)` (or the
edges of a [`ViewGraph`](@ref)) whose value `R_ij` takes frame `i`'s directions
into frame `j`'s (`f_j = R_ij·f_i`, so `R_i = R_j·R_ij`), some of them wrong.
Each value is the rotation, or VisualGeometryToolkitCore's
`Attributed(R_ij, (; weight))` with a fixed non-negative edge weight (an inlier
count); a bare rotation weighs 1. A fit: it seeds itself with
VisualGeometryToolkitCore's `spectral_rotation_averaging` and iterates a reweighted
first-order Lie-algebra least squares over every edge (Chatterjee & Govindu
2013), each a sparse linear solve.

- The edge weights: a zero weight drops its edge. The edges of positive weight
  must connect every frame (an `ArgumentError` otherwise — a zero-weight bridge
  splits the graph), and all-zero weights are refused.
- `optimization_strategy` — the stack's robust-estimation strategy, an
  [`Irls`](@ref)`{K}`: its loss `K`, its width and its `max_iter` (default
  configs/default.toml `[fit_rotation_averaging] optimization_strategy`, Huber).
  `efficiency_constant = m` puts the width at `m · σ̂` (σ̂ the median of the
  residual angles over `√chisq_median(3)`; `m` the plain multiple, for every
  loss), `loss_width = w` at `w` radians; with no
  width — a fit has no scorer to follow — the shape's efficiency constant at
  codimension 3, which only an M-estimator has.
- `tol` — converged when an iteration moves every frame by less than `tol`
  radians (configs/default.toml [fit_rotation_averaging]).
- `seed` — where the iteration starts: `:spectral` (the default,
  VisualGeometryToolkitCore's `spectral_rotation_averaging` over every edge),
  `:spanning_tree` (the rotations chained along the maximum spanning tree of
  the edges of positive weight, by weight: Theia's and OpenCV's
  initialization, see [`chain`](@ref)), or an `AbstractDict` frame => rotation
  given by the caller. Every seed is re-gauged so the first sorted frame is the
  identity.

When the loss's weights drop every edge of a frame (a redescending loss past
its cutoff), that frame is held at its current rotation for the iteration
rather than left free in a singular solve; more generally, in each part of
the graph that the edges of positive weight no longer tie to the gauge frame,
the first sorted frame is held and the rest move with it.

`retcode` is `:converged` or `:max_iterations`; `rms` the root mean square
residual angle, and `n` the count, over the edges of positive weight. A
disconnected graph is an `ArgumentError`, counting only the edges of positive
weight; the `ViewGraph` form also refuses a view in no edge
([`require_connected`](@ref)).
"""
function fit_rotation_averaging(
        relative::AbstractDict{<:Tuple{K, K}};
        optimization_strategy::Union{Nothing, AbstractOptimizationStrategy} = _fit_local_optimization("fit_rotation_averaging"),
        local_optimization = _RetiredKeyword(),   # refused, naming its successor
        tol::Real = _dflt("fit_rotation_averaging", "tol"),
        seed::Union{Symbol, AbstractDict} = :spectral,
    ) where {K}
    _refuse_retired_keyword(local_optimization, :local_optimization, :optimization_strategy)
    lo = _check_fit_width("fit_rotation_averaging", optimization_strategy)
    edges = sort!(collect(keys(relative)))
    p = [VgtkCore.weight_of(relative[e]) for e in edges]
    all(>=(0), p) || throw(ArgumentError("fit_rotation_averaging: an edge weight is negative"))
    any(>(0), p) || throw(ArgumentError("fit_rotation_averaging: every edge weight is zero"))
    Rrel = [SMatrix{3, 3, Float64, 9}(VgtkCore.value_of(relative[e])) for e in edges]
    start = _averaging_seed(seed, edges, Rrel, p)
    frames = sort!(collect(keys(start)))
    R = Dict(k => SMatrix{3, 3, Float64, 9}(start[k]) for k in frames)
    gauge = R[first(frames)]'                                          # one rule for every seed: the first sorted frame is the identity
    foreach(k -> (R[k] = gauge * R[k]), frames)
    m = length(edges); r = Vector{SVector{3, Float64}}(undef, m); w = copy(p)
    edge_residuals!() = for (q, (i, j)) in enumerate(edges)
        r[q] = log_so3(R[j] * Rrel[q] * R[i]')
    end
    edge_residuals!()
    retcode = :max_iterations; iter = 0; σ = NaN
    while iter < lo.max_iter
        iter += 1
        σ = _fit_weights!(w, r, p, lo, 3)
        iszero(σ) && (retcode = :converged; break)                     # every edge agrees
        col, nunk = _averaging_columns(frames, edges, w)
        ω = _averaging_update(edges, col, r, w, nunk)
        Δ = 0.0
        for k in frames
            c = col[k]; c < 0 && continue                              # the gauge, or held
            δ = SVector(ω[c + 1], ω[c + 2], ω[c + 3])
            R[k] = SMatrix{3, 3, Float64, 9}(VgtkCore.exp_so3(δ)) * R[k]
            Δ = max(Δ, norm(δ))
        end
        edge_residuals!()
        Δ < tol && (retcode = :converged; break)
    end
    σ = _fit_weights!(w, r, p, lo, 3)
    out = Dict(k => RotMatrix{3, Float64, 9}(Rotations.nearest_rotation(R[k])) for k in frames)
    meta = RotationAveragingMetadata{K}(Dict(e => norm(r[q]) for (q, e) in enumerate(edges)), Dict(e => w[q] for (q, e) in enumerate(edges)), σ)
    live = [q for q in 1:m if p[q] > 0]                                # a zero weight drops its edge
    return FitResult(out, retcode, iter, sqrt(sum(q -> norm(r[q])^2, live) / length(live)), length(live), meta)
end

# The update's unknowns at the current weights: the column offset of each frame
# (-1: no columns). The edges of weight > 0 split the frames into components;
# each component's anchor — the gauge frame where it lies there, else its first
# sorted frame — has no columns and keeps its rotation, so the normal equations
# are nonsingular (a frame whose every edge weighs 0 is its own anchor, held).
function _averaging_columns(frames, edges, w)
    root = Dict(k => k for k in frames)
    anchor(k) = (while root[k] != k; root[k] = root[root[k]]; k = root[k]; end; k)
    for (q, (i, j)) in enumerate(edges)
        w[q] > 0 || continue
        a, b = anchor(i), anchor(j)
        a == b || (a < b ? (root[b] = a) : (root[a] = b))              # the smaller frame roots
    end
    col = Dict{eltype(frames), Int}(); nunk = 0
    for k in frames                                                    # sorted: frames[1] is the gauge
        anchor(k) == k ? (col[k] = -1) : (col[k] = nunk; nunk += 3)
    end
    return col, nunk
end

# The weighted sparse least squares of ω_j − ω_i = −r_ij over the edges (3 rows
# each, the gauge frame's columns absent): its normal equations, sparse.
function _averaging_update(edges, col, r, w, nunk)
    Is = Int[]; Js = Int[]; Vs = Float64[]; b = zeros(3 * length(edges))
    for (q, (i, j)) in enumerate(edges)
        s = sqrt(w[q])
        for a in 1:3
            row = 3 * (q - 1) + a
            col[j] >= 0 && (push!(Is, row); push!(Js, col[j] + a); push!(Vs, s))
            col[i] >= 0 && (push!(Is, row); push!(Js, col[i] + a); push!(Vs, -s))
            b[row] = -s * r[q][a]
        end
    end
    A = sparse(Is, Js, Vs, 3 * length(edges), nunk)
    return Vector((A' * A) \ (A' * b))
end

# The rotations the IRLS starts from (`seed`, above), for exactly the frames of the edges. Every seed checks what the spectral
# seed checks: the edges of positive weight connect every frame.
function _averaging_seed(seed, edges, Rrel, p)
    seed isa Symbol && seed ∉ (:spectral, :spanning_tree) &&
        throw(ArgumentError("fit_rotation_averaging: seed must be :spectral, :spanning_tree or a Dict, not :$seed"))
    frames = sort!(unique!([k for e in edges for k in e]))
    # the edges of positive weight on a ViewGraph, each pair canonical (i < j): a pair given as (j, i) carries R_ijᵀ
    live = Dict((isless(i, j) ? (i, j) : (j, i)) => VgtkCore.Attributed(isless(i, j) ? R : R', (; weight = q))
                for ((i, j), R, q) in zip(edges, Rrel, p) if q > 0)
    vg = ViewGraph(frames, live)
    require_connected(vg; what = "fit_rotation_averaging: graph of the edges of positive weight")   # names the components
    seed === :spectral &&
        return VgtkCore.spectral_rotation_averaging(live)
    seed === :spanning_tree && return chain(vg, first(frames); identity = one(SMatrix{3, 3, Float64, 9}), inverse = transpose)
    missing_frames = [k for k in frames if !haskey(seed, k)]
    isempty(missing_frames) || throw(ArgumentError("fit_rotation_averaging: the seed has no rotation for the frames $missing_frames"))
    return Dict(k => seed[k] for k in frames)                         # a frame in no edge is not the averaging's
end

function fit_rotation_averaging(vg::ViewGraph; kwargs...)
    require_connected(vg; what = "view graph")
    return fit_rotation_averaging(vg.edges; kwargs...)
end

"""
    filtered_rotation_averaging(vg::ViewGraph; cycle_tol = nothing, orientation_tol = nothing, kwargs...)
        -> (; fit, graph)

Rotation averaging behind the two view-graph filters, as Theia's global
pipeline runs them: [`filter_cycles`](@ref) at `cycle_tol` before the
averaging, then [`filter_by_orientation`](@ref) at `orientation_tol` against
the averaged rotations and, when it dropped an edge, the averaging again on
what is left. Both tolerances are radians; `nothing` skips that filter.
`kwargs` go to every [`fit_rotation_averaging`](@ref). Returns the fit and the
view graph it was fitted on (`graph`: a caller drops whatever it keyed by a
removed edge, a sequence's tracks). A graph a filter disconnects is an
`ArgumentError` naming that filter ([`require_connected`](@ref)).
"""
function filtered_rotation_averaging(vg::ViewGraph; cycle_tol::Union{Nothing, Real} = nothing,
                                     orientation_tol::Union{Nothing, Real} = nothing, kwargs...)
    for (name, t) in (("cycle_tol", cycle_tol), ("orientation_tol", orientation_tol))
        t === nothing || t > 0 || throw(ArgumentError("filtered_rotation_averaging: $name = $t is not positive"))
    end
    graph = cycle_tol === nothing ? vg :
        require_connected(filter_cycles(vg; tol = cycle_tol); what = "view graph after the cycle filter")
    fit = fit_rotation_averaging(graph; kwargs...)
    orientation_tol === nothing && return (; fit, graph)
    kept = filter_by_orientation(graph, VgtkCore.estimate(fit); tol = orientation_tol)
    nedges(kept) == nedges(graph) && return (; fit, graph)
    kept = require_connected(kept; what = "view graph after the orientation filter")
    return (; fit = fit_rotation_averaging(kept; kwargs...), graph = kept)
end
