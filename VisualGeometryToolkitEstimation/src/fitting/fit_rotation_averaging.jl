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
#             3-vector, codimension 3); the IRLS weights of the `Irls{K}`
#             strategy's loss at ONE loss width, set once from the caller's
#             scale input (`scale = σ` or an upstream result: the efficiency
#             constant × σ, Huber's 1.628 at codimension 3; or `width = w`)
#             and never re-estimated (irls_weights.jl); the weighted sparse linear least squares
#             of the first-order update ω_j − ω_i = −r_ij over every edge,
#             the gauge frame's ω = 0, and every frame the edges of positive
#             weight no longer tie to the gauge held where it is (one anchor
#             per such component); R_k ← exp(ω_k)·R_k.
#   stop      when an update moves every frame by less than `tol` radians, or at
#             the strategy's `max_iter`.
#   optimization_strategy = nothing: the seed itself (spectral, spanning tree or the caller's),
#             no update.
# =============================================================================

"""
    RotationAveragingMetadata{K, L, W}

`FitResult` metadata of [`fit_rotation_averaging`](@ref) at the returned
rotations (`L` the `optimization_strategy`'s type, `Nothing` for the seed alone):

- `edges::Vector{Tuple{K, K}}` — the edges, sorted; the order of `weights`.
- `weights::W` — per edge, the loss weight `ρ′(r²)` normalized to a maximum
  of 1 (`IrlsWeights`; a lazy all-ones vector, `BinaryWeights`, for
  `optimization_strategy = nothing`). The solve's weight of an edge is its own weight times
  this.
- `residuals::Dict{Tuple{K, K}, Float64}` — per edge `(i, j)`, the angle
  (radians) of the residual rotation `R_j·R_ij·R_iᵀ`.
- `scale`, `scale_source` — the scale record (radians; `noise_scale(result)`),
  `NaN` and `ScaleNone` for `optimization_strategy = nothing` or a `width`-only fit.
- `loss_width` — radians, `NaN` for `optimization_strategy = nothing`.
"""
struct RotationAveragingMetadata{K, L, W <: AbstractVector{Float64}}
    edges::Vector{Tuple{K, K}}
    weights::W
    residuals::Dict{Tuple{K, K}, Float64}
    scale::Float64
    scale_source::VgtkCore.ScaleSource
    loss_width::Float64
end
RotationAveragingMetadata{K, L}(e, w::W, r, s, src, lw) where {K, L, W} = RotationAveragingMetadata{K, L, W}(e, w, r, s, src, lw)
VgtkCore.weight_kind(::Type{<:RotationAveragingMetadata{K, L}}) where {K, L} = _strategy_weight_kind(L)

"""
    fit_rotation_averaging(relative; optimization_strategy, scale | width, tol, seed = :spectral)
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
count); a bare rotation weighs 1. A fit: it seeds itself (below) and iterates
a reweighted first-order Lie-algebra least squares over every edge
(Chatterjee & Govindu 2013), each a sparse linear solve, at ONE loss width
fixed by the caller's scale input.

- An edge may carry its rotation's covariance:
  `Attributed(Uncertain(R_ij, Σ), (; weight))` (or a bare `Uncertain(R_ij, Σ)`),
  Σ at unit scale — 3×3 in the so(3) chart of `R_ij` (left exp,
  `R_ij → exp(δ)·R_ij`), or 9×9 over `vec(R_ij)` as
  `solver_covariance(problem::RotationProblem, estimate(res), findall(inlier_mask(res)))`
  returns it for a pairwise `find_rotation`. The noise is `σ₀²Σ` with σ₀ the
  `scale` (the variance factor: with Σ from `solver_covariance`, which is in
  unit pixel noise, σ₀ is the keypoints' pixel noise). Each edge residual is
  whitened by its covariance (VisualGeometryToolkitCore's `whitening_factor`
  of `R_j·Σ·R_jᵀ`, refreshed with `R_j`) before the loss and in the update, so a
  weak edge (few inliers, a wide Σ) weighs less. A bare rotation is Σ = I; a
  graph of bare rotations is solved exactly as before.
- The edge weights: a zero weight drops its edge. The edges of positive weight
  must connect every frame (an `ArgumentError` otherwise — a zero-weight bridge
  splits the graph), and all-zero weights are refused.
- `optimization_strategy` — the robust weighting, an [`Irls`](@ref)`{K}` strategy (its loss,
  its efficiency constant and its `max_iter`; default configs/default.toml
  `[fit_rotation_averaging] optimization_strategy`, Huber), or `nothing` for the seed alone.
- `scale` | `width` — with an `Irls` strategy, EXACTLY ONE of: `scale = σ`
  (the edges' residual rotation noise scale per axis component, radians),
  `scale = result` (an upstream robust result's `noise_scale`, e.g. the
  pairwise `find_rotation` results' scale), or `width = w` (the loss width,
  radians). The width is `w`, or `c · σ` with `c` the strategy's efficiency
  constant, by default the shape's at codimension 3
  (`efficiency_constant(K, 3)`: Huber 1.628). Neither, both, or one given with
  `optimization_strategy = nothing` is an `ArgumentError`.
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

`retcode` is `:converged`, `:max_iterations` or `:solved` (`optimization_strategy = nothing`);
`n` the support (the edges of positive solve weight) and `rms` the
weight-weighted root mean square residual angle over them. A disconnected
graph is an `ArgumentError`, counting only the edges of positive weight; the
`ViewGraph` form also refuses a view in no edge ([`require_connected`](@ref)).
"""
function fit_rotation_averaging(
        relative::AbstractDict{<:Tuple{K, K}};
        optimization_strategy::Union{Nothing, AbstractOptimizationStrategy} = _fit_strategy_default("fit_rotation_averaging"),
        local_optimization = _RetiredKeyword(),   # refused, naming its successor
        scale = nothing, width::Union{Nothing, Real} = nothing,
        tol::Real = _dflt("fit_rotation_averaging", "tol"),
        seed::Union{Symbol, AbstractDict} = :spectral,
    ) where {K}
    _refuse_retired_keyword(local_optimization, :local_optimization, :optimization_strategy)
    f = _fit_irls("fit_rotation_averaging", optimization_strategy, scale, width, 3)
    edges = sort!(collect(keys(relative)))
    meas = [_edge_measurement(relative[e]) for e in edges]
    p = [last(x) for x in meas]
    all(>=(0), p) || throw(ArgumentError("fit_rotation_averaging: an edge weight is negative"))
    any(>(0), p) || throw(ArgumentError("fit_rotation_averaging: every edge weight is zero"))
    Rrel = [SMatrix{3, 3, Float64, 9}(_edge_rotation(first(x))) for x in meas]
    Σrel = _edge_covariances([first(x) for x in meas], Rrel)          # nothing: every edge plain (Σ = I), today's path
    start = _averaging_seed(seed, edges, Rrel, p)
    frames = sort!(collect(keys(start)))
    R = Dict(k => SMatrix{3, 3, Float64, 9}(start[k]) for k in frames)
    gauge = R[first(frames)]'                                          # one rule for every seed: the first sorted frame is the identity
    foreach(k -> (R[k] = gauge * R[k]), frames)
    m = length(edges); r = Vector{SVector{3, Float64}}(undef, m); w = _fit_weights_storage(f, m)
    edge_residuals!() = for (q, (i, j)) in enumerate(edges)
        r[q] = log_so3(R[j] * Rrel[q] * R[i]')
    end
    edge_residuals!()
    retcode = f === nothing ? :solved : :max_iterations; iter = 0
    while iter < _fit_max_iter(optimization_strategy)
        iter += 1
        W = _edge_whitening(Σrel, edges, R)
        _irls_weights!(w, f, _whitened_norms(W, r))
        sw = _solve_weights(p, w)
        col, nunk = _averaging_columns(frames, edges, sw)
        ω = _averaging_update(edges, col, r, sw, nunk, W)
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
    _irls_weights!(w, f, _whitened_norms(_edge_whitening(Σrel, edges, R), r))   # the metadata at the returned rotations
    out = Dict(k => RotMatrix{3, Float64, 9}(Rotations.nearest_rotation(R[k])) for k in frames)
    meta = RotationAveragingMetadata{K, typeof(optimization_strategy)}(edges, w, Dict(e => norm(r[q]) for (q, e) in enumerate(edges)), _fit_scale_record(f)...)
    sw = _solve_weights(p, w)
    live = [q for q in 1:m if sw[q] > 0]                               # a zero weight drops its edge
    rms = sqrt(sum(q -> sw[q] * norm(r[q])^2, live; init = 0.0) / max(sum(sw), eps()))
    return FitResult(out, retcode, iter, rms, length(live), meta)
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

# =============================================================================
# Uncertain edges: each relative rotation with its own covariance
# =============================================================================
#
# An edge `Attributed(Uncertain(R_ij, Σ), (; weight))` (or a bare
# `Uncertain(R_ij, Σ)`) carries its rotation's covariance Σ, unit scale: the
# noise is σ₀²Σ, σ₀ the call's `scale` (the variance factor). Σ is 3×3 in the
# so(3) chart of R_ij (left exp, R_ij → exp(δ)·R_ij: VGC's `SO3Manifold`), or
# 9×9 over vec(R_ij) as `solver_covariance(::RotationProblem, …)` returns it,
# read into the chart as `Jᵀ·Σ·J/4` (`J` the chart's `retract_jacobian`, whose
# columns are orthogonal of norm √2). To first order the edge residual
# r_ij = log(R_j·R_ij·R_iᵀ) moves by R_j·δ, so its covariance is R_j·Σ·R_jᵀ and
# its whitening VGC's `whitening_factor(R_j, Σ)` (the inverse Cholesky factor of
# that propagated covariance), refreshed at every iteration's R_j. The loss reads
# the whitened norm and the update solves the whitened rows. A plain edge is
# Σ = I; when every edge is plain, nothing is whitened (today's path, bit for bit).
_edge_measurement(e::VgtkCore.Uncertain) = (e, 1.0)
_edge_measurement(e) = (VgtkCore.value_of(e), Float64(VgtkCore.weight_of(e)))
_edge_rotation(x::VgtkCore.Uncertain) = VgtkCore.value_of(x)
_edge_rotation(x) = x

_edge_covariances(xs, Rrel) = any(x -> x isa VgtkCore.Uncertain, xs) ? [_edge_covariance(x, Rq) for (x, Rq) in zip(xs, Rrel)] : nothing
_edge_covariance(_x, _R) = SMatrix{3, 3, Float64, 9}(LinearAlgebra.I)
_edge_covariance(x::VgtkCore.Uncertain, R) = _chart_covariance(SMatrix{size(cov(x))..., Float64}(cov(x)), R)
_chart_covariance(Σ::SMatrix{3, 3, Float64}, _R) = Σ
function _chart_covariance(Σ::SMatrix{9, 9, Float64}, R)
    J = VgtkCore.retract_jacobian(VgtkCore.SO3Manifold(R))
    return SMatrix{3, 3, Float64, 9}(VgtkCore.propagate(J' / 2, Σ))
end
_chart_covariance(Σ, _R) = throw(ArgumentError(
    "fit_rotation_averaging: an edge covariance is 3×3 (the so(3) chart) or 9×9 (vec(R), solver_covariance's), got $(size(Σ))"))

_edge_whitening(::Nothing, _edges, _R) = nothing
_edge_whitening(Σrel, edges, R) = [VgtkCore.whitening_factor(R[j], Σrel[q]) for (q, (_, j)) in enumerate(edges)]

_whitened_norms(::Nothing, r) = norm.(r)
_whitened_norms(W, r) = [norm(W[q] * r[q]) for q in eachindex(r)]

# The weighted sparse least squares of ω_j − ω_i = −r_ij over the edges (3 rows
# each, the gauge frame's columns absent): its normal equations, sparse. With
# Uncertain edges each edge's rows are its whitening W_ij times √w.
_averaging_update(edges, col, r, w, nunk, ::Nothing) = _averaging_update(edges, col, r, w, nunk)
function _averaging_update(edges, col, r, w, nunk, W::AbstractVector)
    Is = Int[]; Js = Int[]; Vs = Float64[]; b = zeros(3 * length(edges))
    for (q, (i, j)) in enumerate(edges)
        S = sqrt(w[q]) * W[q]
        z = S * r[q]
        for a in 1:3
            row = 3 * (q - 1) + a
            for c in 1:3
                col[j] >= 0 && (push!(Is, row); push!(Js, col[j] + c); push!(Vs, S[a, c]))
                col[i] >= 0 && (push!(Is, row); push!(Js, col[i] + c); push!(Vs, -S[a, c]))
            end
            b[row] = -z[a]
        end
    end
    A = sparse(Is, Js, Vs, 3 * length(edges), nunk)
    return Vector((A' * A) \ (A' * b))
end

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
`kwargs` go to every [`fit_rotation_averaging`](@ref) (its `optimization_strategy` and its
`scale` or `width` among them). Returns the fit and the
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
