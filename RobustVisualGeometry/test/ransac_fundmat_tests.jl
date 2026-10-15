@testitem "ransac_fundmat" setup = [EpipolarScenes] begin
using Test
using Rotations: RotY
using LinearAlgebra
using StaticArrays
using Random
using Statistics: median
using VisualGeometryCore
using VisualGeometryCore: csponds
using VisualGeometryCore: converged, estimate   # VGC accessors (public)
using VisualGeometryCore: residuals!   # the constraint-problem contract (VGC)
using VisualGeometryCore: MsacScoring   # owned and exported by VisualGeometryCore
using RobustVisualGeometry: refine!, inlier_ratio   # public, not exported
using RobustVisualGeometry
using RobustVisualGeometry: minimal_sample_size, data_size, model_type

# =============================================================================
# Test utilities
# =============================================================================

# Two cameras of different intrinsics, view 2 at x₂ = R·x₁ + t; the true F is
# the owner's, F = K₂⁻ᵀ [t]× R K₁⁻¹.
const _CAM₁ = CameraModel(; focal = 500.0, pp = (320.0, 240.0))
const _CAM₂ = CameraModel(; focal = 600.0, pp = (300.0, 250.0))
const _R = RotY(0.15)
const _T = SA[1.0, 0.2, 0.1]
const _F_TRUE = FundMat(EssentialMat(_R, _T), calibration_matrix(_CAM₁), calibration_matrix(_CAM₂))
_box(rng) = SA[4.0 * (rand(rng) - 0.5), 3.0 * (rand(rng) - 0.5), 5.0 + 3.0 * rand(rng)]

function make_fundmat_data(; n_inliers = 100, n_outliers = 30, noise = 0.5, seed = 42)
    source_pts, target_pts = stereo_pixel_pairs(
        MersenneTwister(seed), _CAM₁, _CAM₂, _R, _T;
        n = n_inliers, σ₁ = noise, σ₂ = noise, point = _box, n_out = n_outliers,
        outlier = rng -> (SA[100.0 + 440.0 * rand(rng), 50.0 + 380.0 * rand(rng)], SA[80.0 + 440.0 * rand(rng), 60.0 + 380.0 * rand(rng)])
    )
    return source_pts, target_pts, _F_TRUE, n_inliers
end

make_exact_fundmat_data(; n = 7, seed = 42) =
    (stereo_pixel_pairs(MersenneTwister(seed), _CAM₁, _CAM₂, _R, _T; n, point = _box)..., _F_TRUE)

@testset "RANSAC Fundamental Matrix" begin

    @testset "FundMatProblem construction" begin
        src, dst, _ = make_exact_fundmat_data(n = 10)
        p = FundMatProblem(csponds(src, dst))
        @test minimal_sample_size(p) == 7
        @test data_size(p) == 10
        @test model_type(p) == FundMat{Float64}
        @test p isa FundMatProblem{Float64}

        @test_throws ArgumentError FundMatProblem(csponds(src[1:6], dst[1:6]))
    end

    @testset "7-point solver — exact data" begin
        src, dst, _ = make_exact_fundmat_data(n = 7)
        p = FundMatProblem(csponds(src, dst))

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
        src, dst, F_true = make_exact_fundmat_data(n = 10)
        p = FundMatProblem(csponds(src, dst))

        r = Vector{Float64}(undef, 10)
        residuals!(r, p, F_true)
        @test all(r .< 1.0e-6)
    end

    @testset "fit — DLT roundtrip" begin
        src, dst, _ = make_exact_fundmat_data(n = 20)
        p = FundMatProblem(csponds(src, dst))
        # Algebraic no-θ fit needs the DLT workspace explicitly — the problem's
        # LO workspace is now the Bartoli–Sturm LM buffer (NonlinLsq refit_kind).
        ws = RobustVisualGeometry.DltWorkspace{9, Float64}(
            RobustVisualGeometry.codimension(p) * RobustVisualGeometry.data_size(p)
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
        source_pts, target_pts, _, n_inliers = make_fundmat_data(
            n_inliers = 100, n_outliers = 30, noise = 0.5, seed = 42
        )
        problem = FundMatProblem(csponds(source_pts, target_pts))

        result = ransac(
            problem,
            MsacScoring(problem; threshold = 3.0);   # the class convention (noise 0.5 px)
            config = RansacConfig(max_trials = 5000, min_trials = 500),
            rng = MersenneTwister(191)
        )

        @test converged(result)
        sv = svdvals(estimate(result))
        @test sv[3] / sv[1] < 0.01    # rank-2 enforced
        @test inlier_ratio(result) > 0.5
        @test sum(result.inlier_mask) >= n_inliers * 0.6

        r_inliers = [
            sampson_distance(estimate(result), source_pts[i], target_pts[i])
                for i in 1:n_inliers
        ]
        @test median(r_inliers) < 3.0
    end

    # (The Student-t + uniform IRLS arm of this run is LocalOptimizationDoneRight's test.)
    @testset "LO-RANSAC — Irls{Welsch}" begin
        source_pts, target_pts, _, n_inliers = make_fundmat_data(
            n_inliers = 100, n_outliers = 30, noise = 0.5, seed = 42
        )
        problem = FundMatProblem(csponds(source_pts, target_pts))

        result = ransac(
            problem,
            MsacScoring(problem; threshold = 3.0);   # the class convention (noise 0.5 px)
            local_optimization = Irls{WelschShape}(scale_multiple = 1.5),
            config = RansacConfig(max_trials = 5000, min_trials = 500),
            rng = MersenneTwister(192)
        )

        @test converged(result)
        sv = svdvals(estimate(result))
        @test sv[3] / sv[1] < 0.01
        @test sum(result.inlier_mask) >= n_inliers * 0.6
    end

end
end
