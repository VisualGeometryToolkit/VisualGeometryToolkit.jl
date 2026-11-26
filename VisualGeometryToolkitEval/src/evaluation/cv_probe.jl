# =============================================================================
# src/evaluation/cv_probe.jl — the coarse-fine search: the one CV path of
# every tuned hyperparameter, a scalar or a joint setting of several.
#
# A `[tuning.<name>]` spec of scheme `joint_coarse_fine` (Tuning
# `JointCoarseFine`: named axes, each a coarse axis, an extension rule, a fine
# spacing and whether it is expensive) or `coarse_fine` (`CoarseFine`, its
# one-axis case) carries every parameter, and is the algorithm:
# `solve(EstimationProblem(Hyperparameter, errorsfn, (; cells)), scheme)`. One internal function per stage:
#
#   1. COARSE — `_coarse_search`: the objective over the product of the axes'
#      coarse grids. While the joint argmin lies within `edge_margin` points
#      of a face, that axis `extend`s by `extension.npts` points on that side
#      (lo before hi, one side per axis per round, at most
#      `extension.max_rounds` per side) and only the new slab is evaluated.
#      Windows are located by expansion, never hand-widened; an axis whose
#      argmin is still within the margin after its budget is a boundary hit.
#   2. FINE — `_fine_search`: the product of each axis's `fine` spacing over
#      the coarse argmin's two neighbours on that axis; its free argmin is the
#      selection.
#      With an incumbent declared, the selection replaces it only when a paired
#      unit bootstrap puts the gain above zero with the spec's confidence.
#   3. BOOTSTRAP — Tuning `SelectionBootstrap` over the evaluated cells' unit
#      statistics: the sampling distribution of the joint selection, read per
#      axis (a solve of its own).
#
# Persisted by `save_coarse_fine_artifact`; deployed through one gated
# reader, `deployed_setting`, which refuses a boundary hit on any axis
# (`enforce_probe_interior`).
#
# The objective: `errorsfn(points)` — `points::Dict{K,Vector{NamedTuple}}`,
# per search key `K` (independent searches sharing one evaluation: the
# selectors of one validation sweep, the arms of one detection run) the cells
# to evaluate, each a NamedTuple of axis values — returns per key the
# validation ERRORS at each cell (a vector, or a `units × items` matrix); the
# spec's criterion reduces them (Tuning `tune_objective`, MINIMISED; NaN
# counts as +Inf, never selected). The search makes one call per round per
# distinct value of the expensive axes, so a caller that keeps its expensive
# run per value pays one run per expensive value visited. No experiment
# implements its own sweep/argmin loop.
# =============================================================================

# The axes of a search in its order: a CoarseFine's one axis under the name
# its cells give it; a JointCoarseFine's axes by name, sorted.
function _search_axes(s::CoarseFine, names)
    length(names) == 1 || throw(ArgumentError("a coarse_fine search has one axis; the cells name $(collect(names))"))
    return [Symbol(only(names)) => CoarseFineAxis(s.coarse, s.extension, s.fine, false)]
end
function _search_axes(s::JointCoarseFine, names)
    Set(String.(collect(names))) == Set(keys(s.axes)) ||
        throw(ArgumentError("the cells name the axes $(sort!(String.(collect(names)))), the spec $(sort!(collect(keys(s.axes))))"))
    return [Symbol(n) => s.axes[n] for n in sort!(collect(keys(s.axes)))]
end

_argmin(Y) = argmin(map(y -> isnan(y) ? Inf : y, Y))
_point(names, gs, I) = NamedTuple{names}(ntuple(i -> Float64(gs[i][I[i]]), length(names)))

# `Y` over the grids of `specs`, its old block at offset `off` (the points
# prepended per axis): the grown objective (NaN where not yet evaluated) and
# the cells outside the old block.
function _grow(Y::Array{Float64}, off, specs)
    Y′ = fill(NaN, Tuple(length.(grid.(specs))))
    old = CartesianIndices(ntuple(i -> off[i] .+ (1:size(Y, i)), ndims(Y′)))
    Y′[old] = Y
    return Y′, [I for I in CartesianIndices(Y′) if !(I in old)]
end
_empty(d) = Array{Float64,d}(undef, ntuple(_ -> 0, d))

# Evaluate the cells `todo[k]` of every search `k` (grids `gs[k]`): one
# `errorsfn` call per distinct value of the expensive axes. The objective goes
# to `Y[k]`, `point => errors` to `ev[k]`.
function _evaluate!(Y, ev, errorsfn, todo, gs, names, expensive, criterion)
    pts = Dict(k => [_point(names, gs[k], I) for I in todo[k]] for k in keys(todo))
    xval(p) = Tuple(p[i] for i in eachindex(expensive) if expensive[i])
    for e in sort!(unique(xval(p) for ps in values(pts) for p in ps))
        idx = Dict(k => findall(p -> xval(p) == e, ps) for (k, ps) in pts)
        filter!(kv -> !isempty(kv.second), idx)
        out = errorsfn(Dict(k => pts[k][ix] for (k, ix) in idx))
        for (k, ix) in idx
            haskey(out, k) && length(out[k]) == length(ix) ||
                error("objective: search $(repr(k)) asked for $(length(ix)) cells, got $(haskey(out, k) ? length(out[k]) : "none")")
            for (j, i) in enumerate(ix)
                Y[k][todo[k][i]] = tune_objective(criterion, out[k][j] isa AbstractMatrix ? vec(out[k][j]) : out[k][j])
                push!(ev[k], pts[k][i] => out[k][j])
            end
        end
    end
end

# Stage 1 of the coarse-fine search (module header): per search key `k`, `cells[k]` names each axis's coarse
# `SpacedGrid`; the objective over their product, each axis extended on a side while the joint argmin lies within the
# spec's `edge_margin` of that face. Per key `(; names, coarse, coarse_objective, n_extensions, coarse_argmin,
# boundary_hit, evaluated)`.
function _coarse_search(errorsfn, cells::AbstractDict{K,<:NamedTuple}, scheme::Union{CoarseFine,JointCoarseFine};
                        label = string) where {K}
    isempty(cells) && throw(ArgumentError("the coarse-fine search: no cells"))
    ax = _search_axes(scheme, keys(first(values(cells))))
    names = Tuple(first.(ax)); d = length(names); m = scheme.edge_margin
    specs = Dict{K,Vector{SpacedGrid}}(k => [c[n] for n in names] for (k, c) in cells)
    rounds = Dict{K,Matrix{Int}}(k => zeros(Int, d, 2) for k in keys(cells))
    Y = Dict{K,Array{Float64,d}}(); todo = Dict{K,Vector{CartesianIndex{d}}}()
    ev = Dict{K,Vector{Pair{NamedTuple,Any}}}(k => Pair{NamedTuple,Any}[] for k in keys(cells))
    for k in keys(cells)
        Y[k], todo[k] = _grow(_empty(d), zeros(Int, d), specs[k])
    end
    while !isempty(todo)
        _evaluate!(Y, ev, errorsfn, todo, Dict(k => grid.(specs[k]) for k in keys(todo)), names,
                   [a.expensive for a in last.(ax)], scheme.criterion)
        empty!(todo)
        for k in keys(cells)
            J = _argmin(Y[k]); off = zeros(Int, d); grown = false
            for (i, a) in enumerate(last.(ax))
                ext = a.extension
                side = J[i] <= 1 + m && rounds[k][i, 1] < ext.max_rounds ? 1 :
                       J[i] >= size(Y[k], i) - m && rounds[k][i, 2] < ext.max_rounds ? 2 : 0
                side == 0 && continue
                specs[k][i], off[i] = extend(specs[k][i], (side == 1 ? :lo : :hi,); npts = ext.npts)
                rounds[k][i, side] += 1; grown = true
                @info @sprintf("[%s] extend %s %s (argmin idx=%d/%d)", label(k), names[i],
                               side == 1 ? "lo" : "hi", J[i], size(Y[k], i))
            end
            grown && ((Y[k], todo[k]) = _grow(Y[k], off, specs[k]))
        end
        flush(stdout)
    end
    return Dict{K,NamedTuple}(k => let J = _argmin(Y[k])
            hit = NamedTuple{names}(ntuple(i -> J[i] <= 1 + m || J[i] >= size(Y[k], i) - m, d))
            x = _point(names, grid.(specs[k]), J)
            @info "[$(label(k))] coarse argmin" x objective = Y[k][J] boundary_hit = hit
            (; names, coarse = specs[k], coarse_objective = Y[k], n_extensions = rounds[k],
               coarse_argmin = x, boundary_hit = hit, evaluated = ev[k])
        end for k in keys(cells))
end

# Stage 2 of the coarse-fine search: per key, each axis's `fine` spacing over the two coarse neighbours of the coarse
# argmin on that axis, their product evaluated once (no extension), its free argmin the selection; each record gains
# `fine`, `fine_objective` and `selected`, and with a spec's incumbent `incumbent`, `incumbent_errors`, `gain` and
# `kept_incumbent` (`selected` the incumbent when kept).
function _fine_search(errorsfn, coarse::AbstractDict{K,<:NamedTuple}, scheme::Union{CoarseFine,JointCoarseFine}) where {K}
    names = first(values(coarse)).names; d = length(names)
    ax = _search_axes(scheme, names)
    bracket(g, j, s) = (xs = grid(g); SpacedGrid(xs[max(j - 1, 1)], xs[min(j + 1, length(xs))], s))
    specs = Dict{K,Vector{SpacedGrid}}(k => let J = _argmin(r.coarse_objective)
            [bracket(r.coarse[i], J[i], a.fine) for (i, a) in enumerate(last.(ax))]
        end for (k, r) in coarse)
    Y = Dict{K,Array{Float64,d}}(); todo = Dict{K,Vector{CartesianIndex{d}}}()
    for k in keys(coarse)
        Y[k], todo[k] = _grow(_empty(d), zeros(Int, d), specs[k])
    end
    ev = Dict{K,Vector{Pair{NamedTuple,Any}}}(k => copy(r.evaluated) for (k, r) in coarse)
    _evaluate!(Y, ev, errorsfn, todo, Dict(k => grid.(specs[k]) for k in keys(todo)), names,
               [a.expensive for a in last.(ax)], scheme.criterion)
    recs = Dict{K,NamedTuple}(k => merge(r, (; fine = specs[k], fine_objective = Y[k],
                                             selected = _point(names, grid.(specs[k]), _argmin(Y[k])),
                                             evaluated = ev[k]))
                              for (k, r) in coarse)
    inc = scheme isa JointCoarseFine ? scheme.incumbent : nothing
    inc === nothing && return recs
    p = NamedTuple{names}(Tuple(inc.values[String(n)] for n in names))
    E = errorsfn(Dict(k => [p] for k in keys(coarse)))
    return Dict{K,NamedTuple}(k => _keep_incumbent(r, p, only(E[k]), inc, scheme.criterion) for (k, r) in recs)
end

# The incumbent rule: the selection replaces the incumbent `p` only when a
# paired bootstrap of the units puts its objective below the incumbent's with
# probability ≥ `inc.confidence`. `gain` is the incumbent's objective minus
# the selection's (positive: the selection is better) and that probability.
function _keep_incumbent(r, p, E, inc::Incumbent, criterion)
    S = unit_statistics(criterion, last(r.evaluated[findlast(e -> e.first == r.selected, r.evaluated)]))
    Si = unit_statistics(criterion, E); rng = StableRNG(inc.seed); n = size(Si, 1)
    p_gain = mean(1:inc.n_resamples) do _
        rows = rand(rng, 1:n, n)
        stats_objective(criterion, S[rows, :]) < stats_objective(criterion, Si[rows, :])
    end
    kept = p_gain < inc.confidence
    return merge(r, (; incumbent = p, incumbent_errors = E, kept_incumbent = kept, selected = kept ? p : r.selected,
                     gain = (objective = stats_objective(criterion, Si) - stats_objective(criterion, S), p_gain)))
end

# A coarse-to-fine tuning problem: `f` maps a cells Dict to validation errors, `p` holds the start cells.
const _CellsProblem = EstimationProblem{Hyperparameter, <:Any, <:NamedTuple{(:cells,)}}

"""
    solve(EstimationProblem(Hyperparameter, errorsfn, (; cells)), scheme::Union{CoarseFine, JointCoarseFine})

The coarse-fine search, both stages (module header): per search key `k` of `cells` — independent searches sharing
one evaluation (the selectors of one validation sweep, the arms of one detection run) — `cells[k]` names each axis's
coarse [`SpacedGrid`](@ref) (`grid_spec` of the spec's axis); `scheme` is a `JointCoarseFine` or a `CoarseFine` (one
axis, named by the cells). `errorsfn(points)` is the objective (module header).

`u` is `Dict(k => selected)`, each a NamedTuple of axis values; `metadata = (; records)`, per key the search's record
`(; names, coarse, coarse_objective, n_extensions, coarse_argmin, boundary_hit, fine, fine_objective, selected,
evaluated)` (plus the incumbent's fields with a declared incumbent), what [`save_coarse_fine_artifact`](@ref)
persists; `stats.nf` the cells evaluated. The record is `(Success, ToleranceMet)`, or `(MaxIters, IterationCap)` when
any key's coarse argmin stayed within the margin of a face after its extension budget (`boundary_hit`: a clipped
window, which [`enforce_probe_interior`](@ref) refuses on read).
"""
__solve(prob::_CellsProblem, scheme::CoarseFine) = _coarse_fine_solve(prob, scheme)
__solve(prob::_CellsProblem, scheme::JointCoarseFine) = _coarse_fine_solve(prob, scheme)
function _coarse_fine_solve(prob, scheme::Union{CoarseFine,JointCoarseFine})
    errorsfn = prob.f
    records = _fine_search(errorsfn, _coarse_search(errorsfn, prob.p.cells, scheme), scheme)
    K = keytype(records)
    selected = Dict{K,NamedTuple}(k => r.selected for (k, r) in records)
    hit = any(r -> any(values(r.boundary_hit)), values(records))
    nf = sum(r -> length(r.evaluated), values(records); init = 0)
    return FitResult(selected, hit ? ReturnCode.MaxIters : ReturnCode.Success, hit ? IterationCap : ToleranceMet,
                     SolveStats(; nf), NaN, 0, (; records))
end

"""
    enforce_probe_interior(rec_or_flag; context, allow_edge = false)

Read-side gate of the coarse-fine search: HARD ERROR when a record (or an
explicit `boundary_hit::Bool`) is flagged — the coarse window clipped the
optimum even after the extension budget, so the tuned value is suspect and
must never be silently consumed. Re-run the search (it expands itself; if it
still hits the budget, the objective has no interior optimum on any
reachable window and the method/cell needs attention). `allow_edge = true`
downgrades to a loud warn — FORENSIC reads only, never production. Mirrors
the LO engine's `_enforce_interior_argmin` gate on the product-grid readers.
"""
function enforce_probe_interior(hit::Bool; context::AbstractString,
                                allow_edge::Bool = false)
    hit || return nothing
    msg = "$context: boundary_hit = true — the coarse CV window clipped the " *
          "optimum (extension budget exhausted); the tuned value is suspect. " *
          "Re-run the search; consuming a clipped value violates the CV " *
          "protocol (interior-argmin invariant, single-CV-path discipline)."
    allow_edge || error(msg)
    @warn "$msg [allow_edge = true FORENSIC read — never production]"
    return nothing
end
enforce_probe_interior(rec::AbstractDict; context::AbstractString, allow_edge::Bool = false) =
    enforce_probe_interior(Bool(get(rec, :boundary_hit, get(rec, "boundary_hit", false))); context, allow_edge)
