# =============================================================================
# Exact pool selection on a width grid — KernelAbstractions, CPU() or a GPU.
#
# A loss-width family with a per-residual quality (`residual_quality(scoring, d², loss_width)`,
# `has_residual_quality`) scores a stored model by the SUM of that quality over the model's
# residuals; the selection over a loss-width grid is two kernels, written once for every backend:
#
#   exact_pm_score_kernel!   per (pair, model): every residual through the quality, at every width
#   argmax_models_kernel!    per (pair, width): the FIRST strictly best model, and its error
#
# over a staged batch of pairs (`stage_sorted_residuals`): per model its squared residuals with
# its minimal sample at zero, sorted ascending (so a family of compact support,
# `support_sq_loss_width`, stops at the largest width), flat, with the per-pair offsets and the
# per-model errors the argmax gathers. `select_on_grid!` on the `:pool` staging launches them for
# such a family; `best_pool_model` (per pair, through `score!`) stays the selection of a scoring
# without a per-residual quality. RANSACScoringDoneRight's grid families run the same kernels on
# its staging (its LogBinned aggregation is its own).
# =============================================================================

"""
    has_residual_quality(scoring) -> Bool

Whether `scoring` scores a model by the sum of a per-residual quality,
[`residual_quality`](@ref)`(scoring, d², loss_width)` (Core's RANSAC and MSAC;
RANSACScoringDoneRight's loss-width families add their methods), so its grid
selection is [`exact_pm_score_kernel!`](@ref) then [`argmax_models_kernel!`](@ref).
`false` by default.
"""
has_residual_quality(_scoring) = false
has_residual_quality(::Union{RansacScoring, MsacScoring}) = true

"""
    support_sq_loss_width(scoring, max_loss_width) -> Float64

The squared residual from which `scoring`'s quality is exactly 0 at every width
up to `max_loss_width`: `max_loss_width²` for a family of compact support
(RANSAC, MSAC; RANSACScoringDoneRight's MAGSAC++ and smooth TLS), `Inf` for one
that reaches 0 only asymptotically (the default). On sorted residuals the exact
kernel stops there.
"""
support_sq_loss_width(_scoring, ::Float64) = Inf
support_sq_loss_width(::Union{RansacScoring, MsacScoring}, max_loss_width::Float64) = max_loss_width * max_loss_width

"""
    exact_pm_score_kernel!(backend)(scores_out, sorted_sq_flat, pair_offset, K_per_pair, M_per_pair,
                                    loss_widths, T_val, p_offset, scoring, support_sq; ndrange = (n, M_max))

Per (pair, model) of a chunk of `n` pairs: `scores_out[p, m, t]` (Float64, `(chunk, M_max, T)`)
the sum of `residual_quality(scoring, d², loss_widths[t])` over the model's squared residuals
(`sorted_sq_flat`, ascending, from `pair_offset[p]`, `K_per_pair[p]` per model), evaluated and
accumulated in Float64. A non-finite residual is an outlier (skipped); the loop stops at
`support_sq` ([`support_sq_loss_width`](@ref)). Chunk row `p` is pair `p + p_offset`; a model
beyond `M_per_pair[p]` is left alone.
"""
@kernel function exact_pm_score_kernel!(
        scores_out,                       # (chunk_size, M_max, T) Float64
        @Const(sorted_sq_flat),
        @Const(pair_offset),
        @Const(K_per_pair),
        @Const(M_per_pair),
        @Const(loss_widths),               # (T,) Float64
        T_val::Int32,
        p_offset::Int32,                  # chunk-local p_local + p_offset → p_global
        scoring,                          # the loss-width family (a singleton the kernel specialises on)
        support_sq::Float64)              # max_loss_width² for compact families, Inf otherwise
    p_local, m = @index(Global, NTuple)
    p = Int(p_local) + Int(p_offset)
    M = Int(M_per_pair[p])
    if m <= M
        K = Int(K_per_pair[p]); Tn = Int(T_val)
        col_base = pair_offset[p] + (Int64(m) - Int64(1)) * Int64(K)
        @inbounds for t in 1:Tn
            scores_out[p_local, m, t] = 0.0
        end
        @inbounds for k in 1:K
            d2 = Float64(sorted_sq_flat[col_base + k])
            isfinite(d2) || continue
            d2 >= support_sq && break
            for t in 1:Tn
                scores_out[p_local, m, t] += residual_quality(scoring, d2, loss_widths[t])
            end
        end
    end
end

"""
    argmax_models_kernel!(backend)(errors_out, idxs_out, scores_pmt, M_per_pair, errors_imc_flat,
                                   errors_offset, p_offset; ndrange = (n, T))

Per (pair, width) of a chunk: the FIRST strictly best model of `scores_pmt[p, :, t]` over the
pair's `M_per_pair[p]` models (`1` when none scores above −Inf), its error
`errors_imc_flat[errors_offset[p] + m]` into `errors_out[p, t]` and, when `idxs_out` has rows,
its index into `idxs_out[p, t]`. Padded work items exit untouched.
"""
@kernel function argmax_models_kernel!(
        errors_out,                       # (chunk_size, T) Float64
        idxs_out,                         # (chunk_size, T) Int32  (0-size when unused)
        @Const(scores_pmt),               # (chunk_size, M_max, T) Float64
        @Const(M_per_pair),
        @Const(errors_imc_flat),
        @Const(errors_offset),
        p_offset::Int32)                  # chunk_local p_local + p_offset → p_global
    p_local, t = @index(Global, NTuple)
    p = Int(p_local) + Int(p_offset)      # global pair index for staging-array reads
    # GPU backends pad ndrange to workgroup multiples: padded threads MUST exit
    # before touching memory (they read garbage M and corrupt errors/idxs rows).
    if p_local <= size(errors_out, 1) && t <= size(errors_out, 2) && p <= length(M_per_pair)
    M = Int(M_per_pair[p])
    best_score = -Inf
    best_m = Int32(1)
    @inbounds for m in Int32(1):Int32(M)
        s = scores_pmt[p_local, m, t]      # chunk-local read
        if s > best_score
            best_score = s
            best_m = m
        end
    end
    @inbounds errors_out[p_local, t] = errors_imc_flat[errors_offset[p] + best_m]
    if size(idxs_out, 1) > 0
        @inbounds idxs_out[p_local, t] = best_m
    end
    end   # ndrange padding guard
end
