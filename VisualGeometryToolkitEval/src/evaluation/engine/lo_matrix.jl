# =============================================================================
# evaluation/engine/lo_matrix.jl — post-LO error matrices from a selection sweep.
#
# A selection sweep (VisualGeometryToolkitCore `select_on_grid!`) gives, per
# (pair p, grid column t):
#     idxs[p,t] — the pool index j* of the model the scorer picked at column t
#                 (0 ⇒ no consensus / no model)
# `apply_lo_refits` turns a VECTOR of refit specs into (pairs × |grid|) imc
# matrices — one generic applier over the RefitFamily trait, building the
# per-pair pool-frame geometry (`PairCtx`) once per (pair, spec) task and
# reusing it across that spec's columns.
# =============================================================================

"""
    LoRefitSpec(label, lo, axis, grid, idxs; bundle=false)

One post-LO refit of a selection sweep: refine the selected model
`idxs[p, t]` with strategy `lo` at native parameter
`consumed_native(lo, axis, grid[t])`. `axis` is the SELECTOR grid's native axis
(a [`NativeAxis`](@ref): `OutlierHalfwidth()` | `LossWidth()`) — cross-axis conversion happens per
column inside the refit layer, so callers never pre-convert grids.
`lo = NoLocalOptimization()` records the selection imc (no refit).

Column `t` refits the model selected at column `seedcols[t]` of `idxs` at the
EXPLICIT native value `natives[t]` when those fields are set (no `consumed_native`
conversion — the value is already in the refiner's units). Both default to the
standard coupling (`seedcols[t] = t`, `natives === nothing` ⇒ derive from
`grid[t]`). This is what the tuning grids (one fixed width parameter
per spec) and the fixed-width deployment evaluations (seed at the ζ-scaled
column, refit at the exactly-scaled native) are built from.

`widths[t]` (an absolute loss width): refit at the loss width `widths[t]`
whatever the refiner — the BITWISE-identical call [`solo_refit_imc!`](@ref)
makes, so the tuning and the ladder's test side deliver the width
identically. A native refiner sweeps its seed at the column's `nat′` (from
`grid[t]` — scoring only, never a width) under the context's solo template and
deploys `with_width(lo, fixed_width(lo, widths[t]²))`; an external refiner
refines at `widths[t]`. Mutually exclusive with `natives`.

`columns` restricts which grid columns are EVALUATED (default: all). The
others keep the imc matrix's `NaN` fill. The matrix stays full-width and every
column index keeps its meaning, so boundary detection, the evaluation's seed
columns and the checkpoint are untouched — this only stops computing cells that
a windowed tune never reads, because it argmins inside the standardized window and
the grid is far wider than that window. Measured on the PT-E 5% run of
2026-10-02: 45,567 of 386,000 cells per pair were inside the searched windows;
the other 88% were computed and discarded. (A cell outside the searched window
is never read by a windowed argmin.)

`fixed_s2s[t]` (a fixed external scale): refit at the column's native
exactly as the default path (the refit template), with the strategy deployed
at that scale, `with_width(lo, fixed_width(lo, fixed_s2s[t]))` — the
`refit_imc!(…; s2)` conduit — so its loss width is `scale_multiple·√s2`
instead of the consensus set's `scale_multiple·ŝ`. The consensus-anchoring
ablation evaluation ([`miscal_evaluation_refit`](@ref) with `plugin_scale`) is built
from it. Combines with `natives`; mutually exclusive with `widths`.
"""
struct LoRefitSpec
    label::String
    lo::AbstractLocalOptimization
    axis::NativeAxis
    grid::Vector{Float64}
    idxs::Matrix{Int32}
    bundle::Bool
    seedcols::Union{Nothing,Vector{Int}}
    natives::Union{Nothing,Vector{Float64}}
    widths::Union{Nothing,Vector{Float64}}
    fixed_s2s::Union{Nothing,Vector{Float64}}
    columns::Union{Nothing,Vector{Int}}
end
function LoRefitSpec(label::AbstractString, lo::AbstractLocalOptimization, axis::WidthAxis,
                     grid::AbstractVector, idxs::AbstractMatrix; bundle::Bool = false,
                     seedcols::Union{Nothing,AbstractVector} = nothing,
                     natives::Union{Nothing,AbstractVector} = nothing,
                     widths::Union{Nothing,AbstractVector} = nothing,
                     fixed_s2s::Union{Nothing,AbstractVector} = nothing,
                     columns::Union{Nothing,AbstractVector} = nothing)
    natives !== nothing && widths !== nothing &&
        throw(ArgumentError("LoRefitSpec: natives and widths are mutually exclusive"))
    widths !== nothing && fixed_s2s !== nothing &&
        throw(ArgumentError("LoRefitSpec: widths and fixed_s2s are mutually exclusive"))
    # The axis is the selection grid's NATIVE axis, never a width parameterization.
    axis isa NativeAxis ||
        throw(ArgumentError("LoRefitSpec: axis must be the selection grid's native axis, OutlierHalfwidth() or LossWidth(); got $axis"))
    g = collect(Float64, grid)
    cols = columns === nothing ? nothing : collect(Int, columns)
    # Validated here so the row loops can stay @inbounds: an out-of-range column
    # would otherwise write past the imc matrix rather than erroring.
    cols === nothing || all(c -> c in eachindex(g), cols) ||
        throw(ArgumentError("LoRefitSpec $label: `columns` outside the grid's " *
                            "$(eachindex(g)) — got $(filter(c -> !(c in eachindex(g)), cols))"))
    LoRefitSpec(String(label), lo, axis, g, Matrix{Int32}(idxs), bundle,
                seedcols === nothing ? nothing : collect(Int, seedcols),
                natives === nothing ? nothing : collect(Float64, natives),
                widths === nothing ? nothing : collect(Float64, widths),
                fixed_s2s === nothing ? nothing : collect(Float64, fixed_s2s), cols)
end

"""
    apply_lo_refits(problem_type, specs::Vector{LoRefitSpec}, pairs;
                    scoring, solo_scoring = scoring,
                    operating_scale_multiple = OPERATING_SCALE_MULTIPLE, on_fail = nothing)
        -> Dict{String, Matrix{Float64}}

Run every refit spec over `pairs` and return each spec's post-LO error matrix.
`scoring` is the refit scoring TEMPLATE (rebuilt at each refit's own outlier
half-width), `solo_scoring` the absolute-width (`widths`) columns' template.
The run is threaded `:greedy` over the flat (pair × spec)
product so neither a subsampled validation side nor the arms' very uneven
per-spec cost leaves threads idle (see the comment in the body), with one
`PairCtx` built per task at exactly that spec's `bundle`. `pairs` are light pool
records (`PoolPair`): a refit never reads a residual matrix. `on_fail = nothing` ⇒ seed-error fallback (LO never
worsens the running best); supply `NaN` for the ladder convention (failed refit
≠ success). Returns `label => (P × T) imc matrix`.
"""
function apply_lo_refits(problem_type::Type, specs::AbstractVector{LoRefitSpec}, pairs;
                         scoring = nothing, solo_scoring = scoring,
                         operating_scale_multiple::Real = OPERATING_SCALE_MULTIPLE,
                         on_fail = nothing)
    P = length(pairs)
    out = Dict{String,Matrix{Float64}}(
        spec.label => fill(NaN, P, length(spec.grid)) for spec in specs)
    for spec in specs
        size(spec.idxs, 1) == P ||
            error("apply_lo_refits: spec \"$(spec.label)\" idxs has $(size(spec.idxs,1)) rows ≠ $P pairs")
    end
    isempty(specs) && return out
    # The task space is the flat (pair x spec) product, and the schedule is
    # :greedy. Two separate corrections to what was here:
    #
    # 1. Threading over pairs alone made the iteration count `P`, which is small
    #    whenever the validation side is subsampled — a 5% PT rehearsal gives
    #    P = 110 against 48 threads, each iteration ~9.6 core-minutes, so the
    #    span is one straggler pair. Measured 2026-10-02 on the PT-E 5% cv:
    #    8.82 h wall at ~21 of 48 cores busy.
    # 2. :dynamic does NOT work-steal, which the previous comment claimed. Base:
    #    ":dynamic ... assumes that the workload for each iteration is uniform"
    #    and "each task processes contiguous regions". Per-pair refit cost varies
    #    ~10x with #correspondences and per-spec cost by orders of magnitude (a
    #    NoLocalOptimization selection row against an HQ or bundle refit), so the
    #    uniformity assumption is false in both indices. :greedy is Base's answer
    #    for "workload ... not uniform/has a large spread" (Julia >= 1.11).
    #
    # Together these make the span one (pair, spec) task rather than one pair,
    # with no chunk-count knob to tune: the earlier fix bolted a thread-derived
    # chunking of the spec list onto a pair-keyed space, which re-introduced the
    # same imbalance at chunk granularity because `Iterators.partition` cuts
    # equal-COUNT pieces of a cost-heterogeneous list grouped by arm.
    #
    # The ctx is now built per (pair, spec) rather than per pair. That is the
    # cost, and it is negligible by three orders of magnitude: the ctx is
    # O(#correspondences) — one back-projection of the csponds plus a PairGeom —
    # against a spec's |columns| refits at ~1.6 ms each. It is also now built
    # with exactly the `bundle` THIS spec needs, so a non-bundle spec never pays
    # for a RelposeBundleProblem, which a per-pair or per-chunk `any(...)` made
    # it do.
    #
    # BITWISE-IDENTICAL to the pair-only loop: task (p, spec) writes only
    # `out[spec.label][p, :]` — disjoint in both indices — from read-only specs
    # plus a freshly built ctx, and the refit path draws no RNG (`refit_seed` is
    # a pure function of the ctx and the pool model). So the matrices do not
    # depend on the task->thread mapping or on completion order.
    tasks = vec([(p, i) for p in 1:P, i in eachindex(specs)])
    Threads.@threads :greedy for (p, i) in tasks
        spec = specs[i]
        ctx = build_pair_ctx(problem_type, pairs[p]; scoring, solo_scoring, bundle = spec.bundle)
        # Function barrier: `spec.lo` is abstractly typed on the struct; the row
        # worker specializes the whole column loop on the CONCRETE LO type (one
        # dynamic dispatch per refit row, none per column).
        _apply_refit_row!(out[spec.label], ctx, spec.lo, spec, p,
                         Float64(operating_scale_multiple), on_fail)
    end
    return out
end

# The columns a spec evaluates: its restriction, else the whole grid. A column
# outside it keeps the matrix's NaN fill and is never read — a windowed tune
# argmins inside the SEARCHED window, so the matrix stays full-width and every
# index (boundary detection, the evaluation's seed columns, the checkpoint) keeps
# its meaning. Restricting is therefore invisible to every consumer; it only
# stops computing cells nothing looks at.
@inline _refit_columns(spec::LoRefitSpec) =
    spec.columns === nothing ? eachindex(spec.grid) : spec.columns

# Selection row: recompute the imc of the selected model (`model_imc`: the H
# decomposition metric on an H pool, the E/F pose metric otherwise). NOT the
# raw sweep errs — the LO rows score through `model_imc` too, so selection is
# on the identical metric. Until
# 2026-09-21 this row always took the E/F metric: on an H pool VGC's loose
# `EssentialMat(::AbstractMatrix, K1, K2)` wrapped the homography as a
# fundamental matrix and scored nonsense.
function _apply_refit_row!(M::Matrix{Float64}, ctx::PairCtx, lo::NoLocalOptimization,
                           spec::LoRefitSpec, p::Int, operating_scale_multiple::Float64, on_fail)
    sc = spec.seedcols
    @inbounds for t in _refit_columns(spec)
        j = Int(spec.idxs[p, sc === nothing ? t : sc[t]]); j == 0 && continue
        M[p, t] = model_imc(ctx.pair.models[j], ctx.geom)
    end
    return nothing
end

function _apply_refit_row!(M::Matrix{Float64}, ctx::PairCtx, lo::AbstractLocalOptimization,
                           spec::LoRefitSpec, p::Int, operating_scale_multiple::Float64, on_fail)
    sc = spec.seedcols; nv = spec.natives; wv = spec.widths; fv = spec.fixed_s2s
    @inbounds for t in _refit_columns(spec)
        j = Int(spec.idxs[p, sc === nothing ? t : sc[t]]); j == 0 && continue
        seed_E = refit_seed(ctx, ctx.pair.models[j])
        nat′ = nv === nothing ? consumed_native(lo, spec.axis, spec.grid[t]; operating_scale_multiple, d_g = codimension(ctx.problem_type)) : nv[t]
        # An absolute width is the exact solo_refit_imc! call (bitwise; see the
        # LoRefitSpec doc); otherwise the refit template, at the fixed external
        # scale when set.
        M[p, t] = first(wv === nothing ?
            refit_imc!(ctx, lo, seed_E, nat′; bundle = spec.bundle, on_fail,
                       s2 = fv === nothing ? nothing : fv[t]) :
            _width_refit_imc!(ctx, RefitFamily(lo), lo, seed_E, wv[t], nat′; bundle = spec.bundle, on_fail))
    end
    return nothing
end
