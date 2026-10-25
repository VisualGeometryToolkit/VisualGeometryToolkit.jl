# =============================================================================
# VisualGeometryToolkitCoreNLLSsolverExt — NLLSsolver's variable protocol on
# VisualGeometryToolkitCore's `Manifold` (base/manifold.jl): a variable's size is
# the manifold's `tangent_size` (a constant of its type, so static) and an
# update re-anchors the manifold at the stepped point (`recenter`). It attaches
# when NLLSsolver is loaded, so VisualGeometryToolkitCore carries no solver
# dependency of its own.
#
# `update` takes the variable's `tangent_part` of the solver's increment.
# NLLSsolver's automatic Jacobian differentiates `update`, so `recenter` accepts
# a Dual increment and returns a Dual-valued manifold; a residual on manifold
# variables then needs only `computeresidual`, written generic in the variables'
# element type. A residual may still supply `computeresjac` when a hand-written
# Jacobian is faster.
# =============================================================================
module VisualGeometryToolkitCoreNLLSsolverExt

using ForwardDiff: ForwardDiff
using NLLSsolver: NLLSsolver
using VisualGeometryToolkitCore: AbstractLossShape, Manifold, recenter, tangent_part, tangent_size, weight, ρ
import VisualGeometryToolkitCore: loss_robustifier   # the stub this extension adds its method to

NLLSsolver.nvars(m::Manifold) = NLLSsolver.static(tangent_size(m))
NLLSsolver.update(m::Manifold, δ, start = 1) = recenter(m, tangent_part(m, δ, start - 1))

# A Core loss shape as NLLSsolver's robustifier: ρ(s) of the squared residual
# norm s, ρ′ = `weight` and ρ″ its derivative (Core has no closed ρ″ per
# shape; ForwardDiff differentiates the weight).
struct LossRobustifier{S <: AbstractLossShape, T <: Real} <: NLLSsolver.AbstractRobustifier
    shape::S
    sq_loss_width::T
end
loss_robustifier(shape::AbstractLossShape, loss_width::Real) =
    LossRobustifier(shape, Float64(loss_width)^2)

NLLSsolver.robustify(k::LossRobustifier, cost) = ρ(k.shape, cost, k.sq_loss_width)
function NLLSsolver.robustifydcost(k::LossRobustifier, cost)
    ρ′ = weight(k.shape, cost, k.sq_loss_width)
    ρ″ = ForwardDiff.derivative(s -> weight(k.shape, s, k.sq_loss_width), cost)
    return ρ(k.shape, cost, k.sq_loss_width), ρ′, ρ″
end

end
