@testitem "ransac_fundmat" setup = [EpipolarScenes] begin
    using VisualGeometryToolkitCore: inlier_mask   # the robust contract's accessors
    import VisualGeometryToolkitEstimation as VgtkEstimation
using Test
using LinearAlgebra
using StaticArrays
using Random
using StableRNGs
using Statistics: median
using VisualGeometryToolkitCore
using VisualGeometryToolkitCore: correspondences
using VisualGeometryToolkitCore: converged, estimate   # VGC accessors (public)
using VisualGeometryToolkitCore: mahalanobis_distances!   # the constraint-problem contract (VGC)
using VisualGeometryToolkitCore: MsacScoring   # owned and exported by VisualGeometryToolkitCore
using VisualGeometryToolkitEstimation: inlier_ratio
using VisualGeometryToolkitCore: refine!
using VisualGeometryToolkitEstimation
using VisualGeometryToolkitEstimation: minimal_sample_size, data_size, model_type

# =============================================================================
# Test utilities
# =============================================================================

# THE two-camera F scene: EpipolarScenes' `fundmat_scene` (StableRNG(42), the draw of record); `exact(n)` its noise- and
# outlier-free pairs.
exact(n) = fundmat_scene(StableRNG(42); n_inliers = n, n_outliers = 0, σ = 0.0)

@testset "RANSAC Fundamental Matrix" begin

    @testset "FundMatProblem construction" begin
        src, dst = exact(10)
        p = FundMatProblem(correspondences(src, dst))
        @test minimal_sample_size(p) == 7
        @test data_size(p) == 10
        @test model_type(p) == FundMat{Float64}
        @test p isa FundMatProblem{Float64}

        @test_throws ArgumentError FundMatProblem(correspondences(src[1:6], dst[1:6]))
    end

    @testset "7-point solver — exact data" begin
        src, dst = exact(7)
        p = FundMatProblem(correspondences(src, dst))

        solutions = solve(p, collect(1:7))
        @test !isnothing(solutions)
        @test length(solutions) >= 1

        # At least one of the (up to 3) candidate solutions satisfies the
        # epipolar constraint on the fitting points.
        best_err = minimum(
            maximum(sampson_distance(F, s, d) for (s, d) in zip(src, dst))
                for F in solutions
        )
        @test best_err < 1.0e-6
    end

    @testset "Sampson distance on exact data" begin
        src, dst, F_true = exact(10)
        p = FundMatProblem(correspondences(src, dst))

        r = Vector{Float64}(undef, 10)
        mahalanobis_distances!(r, p, F_true)
        @test all(r .< 1.0e-6)
    end

    @testset "fit — DLT roundtrip" begin
        src, dst = exact(20)
        p = FundMatProblem(correspondences(src, dst))
        # Algebraic no-θ fit needs the DLT workspace explicitly — the problem's
        # LO workspace is now the Bartoli–Sturm LM buffer (NonlinLsq refit_kind).
        ws = VgtkEstimation.DltWorkspace{9, Float64}(
            VgtkEstimation.codimension(p) * VgtkEstimation.data_size(p)
        )

        F_ref = refine!(ws, p, nothing, ones(20))
        @test !isnothing(F_ref)

        # Epipolar constraint on exact data
        max_err = maximum(sampson_distance(F_ref, s, d) for (s, d) in zip(src, dst))
        @test max_err < 1.0e-6
        # Rank-2 enforcement
        @test svdvals(F_ref)[3] < 1.0e-10 * svdvals(F_ref)[1]

        # Too-few-inliers boundary
        @test isnothing(refine!(ws, p, nothing, ones(7)))
    end

    @testset "RANSAC end-to-end" begin
        source_pts, target_pts, _, n_inliers = fundmat_scene(StableRNG(42); n_inliers = 100, n_outliers = 30, σ = 0.5)
        problem = FundMatProblem(correspondences(source_pts, target_pts))

        result = ransac(
            problem,
            MsacScoring(problem; threshold = 3.0);   # the class convention (noise 0.5 px)
            config = RansacConfig(max_trials = 5000, min_trials = 500),
            rng = StableRNG(191)
        )

        @test converged(result)
        sv = svdvals(estimate(result))
        @test sv[3] / sv[1] < 0.01    # rank-2 enforced
        @test inlier_ratio(result) > 0.5
        @test sum(inlier_mask(result)) >= n_inliers * 0.6

        r_inliers = [
            sampson_distance(estimate(result), source_pts[i], target_pts[i])
                for i in 1:n_inliers
        ]
        @test median(r_inliers) < 3.0
    end

    # (The Student-t + uniform IRLS arm of this run is LocalOptimizationDoneRight's test.)
    @testset "LO-RANSAC — Irls{Welsch}" begin
        source_pts, target_pts, _, n_inliers = fundmat_scene(StableRNG(42); n_inliers = 100, n_outliers = 30, σ = 0.5)
        problem = FundMatProblem(correspondences(source_pts, target_pts))

        result = ransac(
            problem,
            MsacScoring(problem; threshold = 3.0);   # the class convention (noise 0.5 px)
            local_optimization_strategy = Irls{WelschShape}(efficiency_constant = 1.5),
            config = RansacConfig(max_trials = 5000, min_trials = 500),
            rng = StableRNG(192)
        )

        @test converged(result)
        sv = svdvals(estimate(result))
        @test sv[3] / sv[1] < 0.01
        @test sum(inlier_mask(result)) >= n_inliers * 0.6
    end

end
end
