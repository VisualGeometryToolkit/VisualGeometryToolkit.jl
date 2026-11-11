# =============================================================================
# The IRLS weight step of every reweighted fit — fit_linear (codimension 1),
# fit_rotation (2) and fit_rotation_averaging (3): the noise scale σ̂ from the
# residual norms (VisualGeometryToolkitCore's `noise_sigma`), the loss deployed
# at σ̂², and the loss's weights times the caller's prior.
# =============================================================================

# σ̂ of the residual `norms` over the entries of positive `prior` — Core's
# `noise_sigma` at codimension `codim` (an excluded entry's residual does not
# widen the loss) — and the loss's IRLS weights times the prior into `w`; the
# prior itself on an exact fit (σ̂ = 0). Returns σ̂.
function _irls_weights!(w, norms, prior, loss, scale_multiple, codim::Integer)
    σ = VgtkCore.noise_sigma((norms[i] for i in eachindex(norms) if prior[i] > 0), codim)
    if iszero(σ)
        w .= prior
        return σ
    end
    c² = VgtkCore.sq_loss_width(_irls_loss(loss, σ^2, scale_multiple, codim))
    @inbounds for i in eachindex(w, norms, prior)
        w[i] = prior[i] * weight(loss, norms[i]^2, c²)
    end
    return σ
end

# The loss at the noise scale s2: an M-estimator's width from its efficiency at
# codimension `codim` (or the given multiple); any other shape at the given
# multiple (it has no efficiency, so no codimension: its fit requires the multiple).
_irls_loss(loss::VgtkCore.AbstractMEstimator, s2, ::Nothing, codim) = deploy(loss, s2; codim)
_irls_loss(loss::VgtkCore.AbstractMEstimator, s2, scale_multiple::Real, codim) = deploy(loss, s2; codim, scale_multiple)
_irls_loss(loss::VgtkCore.AbstractLossShape, s2, scale_multiple::Real, codim) = deploy(loss, s2; scale_multiple)
