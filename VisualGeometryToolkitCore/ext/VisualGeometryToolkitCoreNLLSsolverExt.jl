# =============================================================================
# VisualGeometryToolkitCoreNLLSsolverExt — NLLSsolver's variable protocol on
# VisualGeometryToolkitCore's `Manifold` (base/manifold.jl): a variable's size is
# the manifold's `tangent_size` (a constant of its type, so static) and an
# update re-anchors the manifold at the stepped point (`recenter`). It attaches
# when NLLSsolver is loaded, so VisualGeometryToolkitCore carries no solver
# dependency of its own.
# =============================================================================
module VisualGeometryToolkitCoreNLLSsolverExt

using NLLSsolver: NLLSsolver
using VisualGeometryToolkitCore: Manifold, recenter, tangent_part, tangent_size

NLLSsolver.nvars(m::Manifold) = NLLSsolver.static(tangent_size(m))
NLLSsolver.update(m::Manifold, δ, start = 1) = recenter(m, tangent_part(m, δ, start - 1))

end
