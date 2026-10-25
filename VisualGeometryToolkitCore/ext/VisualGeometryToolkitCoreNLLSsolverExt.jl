# =============================================================================
# VisualGeometryToolkitCoreNLLSsolverExt — NLLSsolver's variable protocol on
# VisualGeometryToolkitCore's `ManifoldBlock` (base/manifold.jl): the block size
# is the manifold's dof and an update re-anchors the manifold at the stepped
# point (`recenter`). It attaches when NLLSsolver is loaded, so
# VisualGeometryToolkitCore carries no solver dependency of its own.
# =============================================================================
module VisualGeometryToolkitCoreNLLSsolverExt

using NLLSsolver: NLLSsolver
using StaticArrays: SVector
using VisualGeometryToolkitCore: ManifoldBlock, recenter

NLLSsolver.nvars(::ManifoldBlock{D}) where {D} = NLLSsolver.static(D)
NLLSsolver.update(b::ManifoldBlock{D}, δ, start = 1) where {D} =
    ManifoldBlock(recenter(b.manifold, SVector{D, eltype(δ)}(ntuple(i -> δ[start + i - 1], Val(D)))))

end
