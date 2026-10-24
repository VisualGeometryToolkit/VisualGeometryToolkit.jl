@testitem "p3p_jacobian" setup = [P3PScenes, Cameras] begin
    import VisualGeometryToolkitCore as VgtkCore
using Test
using StaticArrays
using Random
using StableRNGs
using FiniteDiff: FiniteDiff
using VisualGeometryToolkitCore: VisualGeometryToolkitCore
using VisualGeometryToolkitCore: LeftRigidManifold6   # the PnP refit chart (public)
using VisualGeometryToolkitEstimation: _lm_state

# The analytic `residual_jacobian_rows!` of the PnP refit against central
# differences of its own residual rows, at a general (nonzero) chart point.
# (Central differences, not ForwardDiff: the rows read the pose through VGC's
# Float64 point constraint, which does not take Dual poses.)
@testset "P3P Jacobian (analytic vs central differences)" begin
    m = 20
    R₀, t₀, prob, sqrtw = p3p_scene(StableRNG(13), IDCAM, m; σ = 0.01)   # σ ≠ 0: g ≠ 0
    lm = _lm_state(prob, LeftRigidManifold6(R₀, t₀), sqrtw)
    x = SVector{6, Float64}(0.04, -0.03, 0.06, 0.1, -0.05, 0.08)
    @test VgtkCore.check_rows_jacobian(lm, x, 2m; rtol = 1.0e-7, reference = (f, θ) -> FiniteDiff.finite_difference_jacobian(f, θ, Val(:central)))
end
end
