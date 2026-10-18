@testitem "noiseless_ransac" setup = [EpipolarScenes, HomographyScenes] begin
using Test
using Rotations: RotY
using StaticArrays
using StructArrays
using Random
using LinearAlgebra

using VisualGeometryToolkitCore
using VisualGeometryToolkitCore: converged, estimate   # VGC accessors (public)
using VisualGeometryToolkitCore: minimal_sample_size   # the constraint-problem contract (VGC)
using VisualGeometryToolkitCore: MsacScoring   # owned and exported by VisualGeometryToolkitCore
using VisualGeometryToolkitEstimation

# =============================================================================
# The noiseless F scene — two views of a Gaussian cloud, plus uniform-ish
# outliers — as the StructArray{Pair} the problem constructors take.
# =============================================================================

const _FCAM = CameraModel(; focal = 500.0, pp = (320.0, 240.0))

function noiseless_fundmat_pairs(rng, n_inliers, n_outliers)
    src, dst = stereo_pixel_pairs(
        rng, _FCAM, _FCAM, RotY(0.3), SA[1.0, 0.2, 0.1];
        n = n_inliers, n_out = n_outliers,
        point = rng -> SA[randn(rng) * 2, randn(rng) * 2, randn(rng) * 0.5 + 5.0],
        outlier = rng -> (randn(rng, 2) * 200 .+ 320, randn(rng, 2) * 200 .+ 320)
    )
    return StructArray(src .=> dst)
end

# =============================================================================
# DLT precision on pure inlier data (no RANSAC)
# =============================================================================

@testset "F-matrix DLT — noiseless precision (no RANSAC)" begin
    rng = MersenneTwister(42)
    cs = noiseless_fundmat_pairs(rng, 100, 0)

    F_est = fundamental_dlt(cs)

    max_sampson = 0.0
    for i in eachindex(cs)
        max_sampson = max(max_sampson, sampson_distance(F_est, cs[i].first, cs[i].second))
    end
    @test max_sampson < 1.0e-10
end

# =============================================================================
# Noiseless RANSAC theoretical guarantees
# =============================================================================
#
# For each problem we keep three points on the outlier-rate curve:
#   0%   — bottom of the easy regime
#   50%  — the canonical "RANSAC works" benchmark
#   75%  — high-outlier-rate stress test
# =============================================================================

@testset "Homography RANSAC — noiseless theoretical guarantees" begin
    rng = MersenneTwister(42)
    H_true = SMatrix{3, 3, Float64, 9}(H_PX)

    scenarios = [
        (100, 0, "0% outliers"),
        (100, 100, "50% outliers"),
        (50, 150, "75% outliers"),
    ]

    for (n_in, n_out, label) in scenarios
        @testset "$label" begin
            cs = StructArray(Pair.(homography_pairs(rng, H_PX; n_in, n_out, source = gaussian_source)...))
            problem = HomographyProblem(cs)
            config = RansacConfig(; max_trials = 10_000, success_prob = 0.9999, min_trials = 100)
            # The problem form: the codimension (2 for H) from the problem.
            scoring = MsacScoring(problem; threshold = 1.0e-4)
            result = ransac(problem, scoring; config, rng = MersenneTwister(421))

            @test converged(result)
            H_est_n = estimate(result) / estimate(result)[3, 3]
            H_ref = H_true / H_true[3, 3]
            @test norm(H_est_n - H_ref) / norm(H_ref) < 1.0e-8
            @test sum(result.inlier_mask) >= n_in
        end
    end
end

@testset "F-matrix RANSAC — noiseless theoretical guarantees" begin
    rng = MersenneTwister(123)

    scenarios = [
        (100, 0, "0% outliers"),
        (100, 100, "50% outliers"),
        (50, 150, "75% outliers"),
    ]

    for (n_in, n_out, label) in scenarios
        @testset "$label" begin
            cs = noiseless_fundmat_pairs(rng, n_in, n_out)
            problem = FundMatProblem(cs)
            config = RansacConfig(; max_trials = 200_000, success_prob = 0.9999, min_trials = 500)
            scoring = MsacScoring(problem; threshold = 1.0e-4)
            result = ransac(problem, scoring; config, rng = MersenneTwister(422))

            @test converged(result)
            F_est = estimate(result)
            @test sum(result.inlier_mask) >= n_in

            max_sampson = 0.0
            for i in eachindex(cs)
                if result.inlier_mask[i]
                    max_sampson = max(
                        max_sampson,
                        sampson_distance(F_est, cs[i].first, cs[i].second)
                    )
                end
            end
            @test max_sampson < 1.0e-6
        end
    end
end
end
