@testitem "homography_lm" setup = [HomographyScenes, Contamination] begin
    import VisualGeometryToolkitCore as VgtkCore
# =============================================================================
# homography_lm_tests.jl — minimal sl(3) homography refit (NonlinLsq).
# Gates: chart minimality (traceless basis, no scale direction), analytic
# Jacobian vs ForwardDiff, whitened refit cost decrease from perturbed seeds.
# =============================================================================
using Test, Random, LinearAlgebra, StaticArrays
using VisualGeometryToolkitCore: VisualGeometryToolkitCore
using VisualGeometryToolkitCore: SL3_BASIS, HomographyManifold8   # the sl(3) chart and its basis (public)
using VisualGeometryToolkitEstimation
const RVGH = VisualGeometryToolkitEstimation

@testset "sl(3) minimal homography refit" begin
    rng = MersenneTwister(7)
    @test all(abs(tr(G)) < eps() for G in SL3_BASIS)
    @test length(SL3_BASIS) == 8

    H_gt = RVGH.HomographyMat(Tuple(SMatrix{3, 3, Float64, 9}(1.0, 0.1, 1.0e-3, -0.15, 0.9, 2.0e-3, 0.2, -0.1, 1.0)))
    n = 50
    u1 = [SVector(randn(rng), randn(rng)) for _ in 1:n]
    pairs, _ = contaminated_pairs(rng, u1 .=> homography_warp(H_gt).(u1); σ = 0.001)
    prob = RVGH.HomographyProblem(pairs)
    w = ones(n)
    sqrtw = ones(2n)

    # The rows at a general chart point, the whitening frozen at the anchor H_gt.
    lm = RVGH._lm_state(prob, HomographyManifold8(SMatrix{3, 3, Float64, 9}(H_gt)), sqrtw)
    a0 = SVector{8, Float64}(0.01, -0.02, 0.015, 0.01, -0.005, 0.02, -0.01, 0.008)
    @test VgtkCore.check_rows_jacobian(lm, a0, 2n; atol = 1.0e-8)

    # Refit from a perturbed seed: strict cost decrease, lands near GT optimum.
    H_seed = RVGH.HomographyMat(Tuple(SMatrix{3, 3, Float64, 9}(H_gt) + 0.05 * SMatrix{3, 3, Float64, 9}(randn(rng, 3, 3))))
    cost(H) = VgtkCore.rows_cost(RVGH._lm_state(prob, RVGH._chart(prob, H), sqrtw), zeros(SVector{8, Float64}), 2n)
    H_ref = RVGH._refine!(RVGH.NonlinLsq(), nothing, prob, H_seed, w)
    @test H_ref !== nothing
    @test cost(H_ref) < 0.5 * cost(H_seed)
    @test cost(H_ref) < 1.5 * cost(H_gt)
end
end
