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

using NLLSsolver: NLLSsolver
using VisualGeometryToolkitCore: Manifold, recenter, tangent_part, tangent_size

NLLSsolver.nvars(m::Manifold) = NLLSsolver.static(tangent_size(m))
NLLSsolver.update(m::Manifold, δ, start = 1) = recenter(m, tangent_part(m, δ, start - 1))

end
