# =============================================================================
# The chart-rows contract — what a lifted local optimizer reads of a problem
# (the half-quadratic lifts and GNC ports live in LocalOptimizationDoneRight;
# the providers of each problem live with the problem, VisualGeometryToolkitEstimation).
# =============================================================================
#
# A chart problem provides, per local-optimization call, a PROVIDER over its
# whitened residual rows at a chart anchor:
#
#   rows_provider(problem) -> prov
#   lift_init(prov, model)       -> the chart at the model
#   lift_refresh!(prov, ch)      rows (and per-point norms for codim 2) at ch
#   lift_jacobian!(prov, ch)     the rows' Jacobian at ch
#   lift_retract(prov, ch, Δθ)   -> the chart re-anchored at the step
#   lift_model(prov, ch)         -> the model
#   lift_valid(prov)             participation mask (default `nothing`)
#   lift_rows(prov)              -> RowsWorkspace{D, C}: r, J, rn, valid, onev
#   lift_residuals(prov)         -> the residual currency (r for codim 1, rn for 2)
#   lift_size(prov)              -> the point count n
#
# The rows are stored ROW-contiguous (`J` the transpose of a `D × C·n` matrix),
# so a row kernel writes one row and an accumulation reads one point's row as
# one contiguous load.
# =============================================================================

# =============================================================================
# Task-local buffers — ONE reuse policy
# =============================================================================

"""
    task_buffers(alloc, key)

The buffer set stored under `key` in the current task's local storage, built by
`alloc()` on first use. A workspace keyed by its type and sizes is then built
once per (task, size) and reused by every later call of that size on that task
— task-local, so threaded callers (`Threads.@threads` bodies are tasks) never
share one. The one memo every per-call scratch workspace of the stack draws from.
"""
function task_buffers(alloc, key)
    tls = task_local_storage()
    buf = get(tls, key, nothing)
    buf === nothing || return buf
    return tls[key] = alloc()
end

"""
    task_buffers(::Type{V}, alloc, key) -> V

The typed form: the buffer set of type `V` stored under `key` (a tuple of
sizes, any isbits value) in the current task, built by `alloc()` on first use.
Allocation-free on reuse: the task's storage is keyed by a singleton per
`(V, typeof(key))` and holds a `Dict{typeof(key), V}`, so no key is boxed.
The form every per-call scratch with a size in its key uses.
"""
function task_buffers(::Type{V}, alloc, key::K) where {V, K}
    tls = task_local_storage()
    slot = _TaskBufferSlot{V, K}()
    d = get(tls, slot, nothing)
    if d === nothing
        d = Dict{K, V}()
        tls[slot] = d
    end
    d = d::Dict{K, V}
    buf = get(d, key, nothing)
    buf === nothing || return buf
    return d[key] = alloc()::V
end
struct _TaskBufferSlot{V, K} end

"""
    RowsWorkspace{D, C}
    RowsWorkspace{D, C}(n)

The whitened residual rows of a `D`-dof chart with codimension-`C` residuals
for `n` points: the stacked rows `r` and their Jacobian `J` (`C·n × D`,
row-contiguous), the per-point residual norms `rn` (the codim-2 currency), the
per-iteration participation mask `valid` and a unit √weight vector `onev` for
the row kernels. [`rows_workspace`](@ref) builds one per (task, size).
"""
struct RowsWorkspace{D, C}
    r::Vector{Float64}                             # C·n
    J::Transpose{Float64, Matrix{Float64}}         # C·n × D, row-contiguous
    rn::Vector{Float64}                            # n: per-point residual norms
    valid::Vector{Bool}                            # n: per-iteration participation (PnP)
    onev::Vector{Float64}                          # C·n: unit √weights
end
function RowsWorkspace{D, C}(n::Int) where {D, C}
    rows = C * n
    return RowsWorkspace{D, C}(
        Vector{Float64}(undef, rows), transpose(Matrix{Float64}(undef, D, rows)),
        Vector{Float64}(undef, n), Vector{Bool}(undef, n), ones(rows)
    )
end

"""
    rows_workspace(::Val{D}, ::Val{C}, n) -> RowsWorkspace{D, C}

The rows of `n` points under the one reuse policy ([`task_buffers`](@ref)).
"""
rows_workspace(::Val{D}, ::Val{C}, n::Int) where {D, C} =
    task_buffers(RowsWorkspace{D, C}, () -> RowsWorkspace{D, C}(n), n)

@inline _row_count(::Val{C}) where {C} = C
@inline _row_count(d::Int) = d

# Point i's C rows of a stacked C·n-row layout, as ONE strided dot product:
# `_rows_dot(x, y, i, C) = Σⱼ x[C(i−1)+j]·y[C(i−1)+j]`. Written first-term-seeded,
# left to right, so C = 2 is exactly x₁y₁ + x₂y₂. `C` is a `Val` or an `Int`.
Base.@propagate_inbounds function _rows_dot(x, y, i::Int, C::Union{Val, Int})
    c = _row_count(C)
    k = c * (i - 1)
    s = x[k + 1] * y[k + 1]
    for j in 2:c
        s += x[k + j] * y[k + j]
    end
    return s
end

"""
    point_norms!(rn, r, n, d) -> rn

Per-point norms of `d`-row residual groups: `rn[i] = ‖r[d(i−1)+1 : d·i]‖`.
"""
@inline function point_norms!(rn::AbstractVector, r::AbstractVector, n::Int, d::Int)
    @inbounds for i in 1:n
        rn[i] = sqrt(_rows_dot(r, r, i, d))
    end
    return rn
end

"""
    rows_provider(problem) -> prov

The chart-rows provider of a chart problem for one local-optimization call
(see the contract at the head of this file). The problem's owner implements it.
"""
function rows_provider end
"The chart at the model `model` (a provider's starting anchor)."
function lift_init end
"Fill the provider's rows (and, at codimension 2, the per-point norms) at the chart `ch`."
function lift_refresh! end
"Fill the provider's rows' Jacobian at the chart `ch`."
function lift_jacobian! end
"The chart re-anchored at the tangent step `Δθ` from `ch`."
function lift_retract end
"The model at the chart `ch`."
function lift_model end
"The provider's per-iteration participation mask, or `nothing` (every point participates)."
lift_valid(_prov) = nothing
"The provider's rows, a [`RowsWorkspace`](@ref)."
function lift_rows end
"The provider's residual currency: the rows at codimension 1, the per-point norms at 2."
function lift_residuals end
"The provider's point count."
function lift_size end
