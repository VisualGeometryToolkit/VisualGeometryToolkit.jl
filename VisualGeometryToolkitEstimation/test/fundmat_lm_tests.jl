@testitem "fundmat_lm" setup = [EpipolarScenes] begin
    import VisualGeometryToolkitCore as VgtkCore
# =============================================================================
# fundmat_lm_tests.jl — Bartoli–Sturm orthonormal F refit (NonlinLsq, PAMI'04).
# Gates: chart roundtrip (scale-invariant), analytic Jacobian vs ForwardDiff,
# and cost decrease of the weighted LM refit from a perturbed seed.
# =============================================================================
using Test, Random, LinearAlgebra, StaticArrays
using Rotations: RotMatrix, MRP
using VisualGeometryToolkitCore: VisualGeometryToolkitCore
using VisualGeometryToolkitCore: estimate   # VGC accessors (public)
using VisualGeometryToolkitCore: EssentialMat, FundMat
using VisualGeometryToolkitCore.Synthetic: projective_distance
using VisualGeometryToolkitCore: FundMatManifold7, retract   # the F chart (public)
using VisualGeometryToolkitCore: MsacScoring   # owned and exported by VisualGeometryToolkitCore
using VisualGeometryToolkitEstimation
const RVGF = VisualGeometryToolkitEstimation

@testset "Bartoli–Sturm orthonormal F refit" begin
    rng = MersenneTwister(42)
    # Ground-truth F from a random pose (rank-2 by construction).
    R = RotMatrix(MRP(0.1, -0.2, 0.15))
    t = normalize(SVector(0.3, -0.1, 1.0))
    F_gt = SMatrix{3, 3, Float64}(FundMat(EssentialMat(R, t), SMatrix{3, 3, Float64}(I)))   # F at K = I

    ch = FundMatManifold7(F_gt)
    F0 = retract(ch, zero(SVector{7, Float64}))
    @test 0 < ch.σ <= 1 + 1.0e-12
    # Roundtrip up to scale: F(0) ∝ F_gt.
    @test projective_distance(F0, F_gt) < 1.0e-10

    # Synthetic correspondences: calibrated two-view pairs of that pose (so on
    # F_gt's epipolar constraint) + noise.
    u1, u2 = calibrated_pairs(rng, R, t; n = 60, σ = 0.001)
    n = length(u1)
    w = ones(n); sqrtw = sqrt.(w)
    prob = RVGF.FundMatProblem(u1 .=> u2)

    # Analytic Jacobian vs ForwardDiff at a nonzero chart point.
    a0 = SVector(0.02, -0.01, 0.03, 0.015, -0.02, 0.01, -0.05)
    @test VgtkCore.check_rows_jacobian(RVGF.ChartLm(ch, prob, sqrtw), a0, n; atol = 1.0e-8)

    # Refit from a perturbed seed strictly reduces the weighted Sampson cost.
    F_seed = RVGF.FundMat(Tuple(F_gt + 0.05 * SMatrix{3, 3, Float64}(randn(rng, 3, 3))))
    cost(F) = VgtkCore.rows_cost(RVGF.ChartLm(RVGF._chart(prob, F), prob, sqrtw), zero(SVector{7, Float64}), n)
    F_ref = RVGF._refine!(RVGF.NonlinLsq(), nothing, prob, F_seed, w)
    @test F_ref !== nothing
    @test cost(F_ref) < 0.5 * cost(F_seed)
    @test cost(F_ref) < 1.5 * cost(RVGF.FundMat(Tuple(F_gt)))   # near the GT optimum

    # The core's static Welsch IRLS on the same chart: locally_optimize must run
    # on FundMatProblem and improve the cost from the perturbed seed. (The
    # joint-(θ,z) Welsch HQ lift on F is LocalOptimizationDoneRight's test.)
    sc = MsacScoring(typeof(prob); threshold = 0.05)   # outlier-free scene: wide enough for the perturbed seed's consensus
    ws = RVGF.RansacWorkspace(prob, sc)
    rlift = RVGF.locally_optimize(
        prob, F_seed; local_optimization = RVGF.Irls{RVGF.WelschShape}(max_iter = 25, scale_multiple = 1.5),
        scoring = sc, workspace = ws
    )
    @test estimate(rlift) !== nothing
    @test cost(estimate(rlift)) < cost(F_seed)
end

# The rows' degeneracy rule (a zero row, never a floor) is VGC's
# `sampson_rows!` / `sampson_jacobian_rows!`, pinned in VGC's sampson_rows_tests.
end
