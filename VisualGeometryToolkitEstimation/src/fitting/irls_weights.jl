# =============================================================================
# The IRLS weight step of every reweighted fit — fit_linear (codimension 1),
# fit_rotation (2) and fit_rotation_averaging (3). Each fit takes the stack's
# one robust-estimation schema, an `Irls{K}` strategy (its loss, its width mode
# and its `max_iter`; TOML `[<fit>] local_optimization = { type = "irls", … }`,
# read by `local_optimization_from_toml`), and only its model update and its
# stop rule are its own. Per iteration: the noise scale σ̂ from the residual
# norms (VisualGeometryToolkitCore's `noise_sigma`), the strategy's loss at σ̂²,
# and the loss's weights times the caller's prior.
# =============================================================================

# A fit's default strategy: the `local_optimization` table of its
# configs/default.toml table `fit`, read by the one strategy reader.
_fit_local_optimization(fit::AbstractString) =
    local_optimization_from_toml(_dflt(fit, "local_optimization"); class = Symbol(fit),
                                 section = "$fit.local_optimization")

# The loss strategy `lo` deploys at the fit's noise scale σ̂² = `s2`, re-estimated
# every iteration. A width mode (`scale_multiple = m`: m·σ̂ in the loss's
# currency, `plain_scale_multiple`; `loss_width = w`: w) resolves through
# `lo_loss` (Core's `loss_at`) at the fit's scale; the least-squares arm has no
# width. With no width — a fit has no scorer to follow — the shape's efficiency
# constant at the fit's codimension (Core's `deploy` default), which only an
# M-estimator has (`_check_fit_width`).
_fit_loss(lo::Irls, s2, codim) = _fit_loss(lo_width(lo), lo, s2, codim)
_fit_loss(lo::Irls{GaussianShape}, s2, _codim) = _fit_loss(lo, s2)
_fit_loss(::Nothing, lo, s2, codim) = deploy(strategy_loss(lo)(), s2; codim)
_fit_loss(_width, lo, s2, _codim) = _fit_loss(lo, s2)
_fit_loss(lo, s2) = lo_loss(lo, ConsensusSet(; count = 0, s2, epoch = 0), nothing, nothing)

# The fit's entry check: a strategy with no width needs an efficiency default.
function _check_fit_width(fit::AbstractString, lo::Irls)
    lo isa Irls{GaussianShape} || lo_width(lo) !== nothing || strategy_loss(lo) <: VgtkCore.AbstractMEstimator ||
        throw(ArgumentError("$fit: the loss $(nameof(strategy_loss(lo))) has no efficiency default; give the " *
                            "strategy its width, Irls{$(nameof(strategy_loss(lo)))}(scale_multiple = m)"))
    return lo
end

# σ̂ of the residual `norms` over the entries of positive `prior` — Core's
# `noise_sigma` at codimension `codim` (an excluded entry's residual does not
# widen the loss) — and the IRLS weights of `lo`'s loss at σ̂² times the prior
# into `w`; the prior itself on an exact fit (σ̂ = 0). Returns σ̂.
function _fit_weights!(w, norms, prior, lo::Irls, codim::Integer)
    σ = VgtkCore.noise_sigma((norms[i] for i in eachindex(norms) if prior[i] > 0), codim)
    if iszero(σ)
        w .= prior
        return σ
    end
    loss = _fit_loss(lo, σ^2, codim)
    shape, c² = loss_shape(loss), sq_loss_width(loss)
    @inbounds for i in eachindex(w, norms, prior)
        w[i] = prior[i] * weight(shape, norms[i]^2, c²)
    end
    return σ
end
