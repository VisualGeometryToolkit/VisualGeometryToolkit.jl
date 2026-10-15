# =============================================================================
# homography_decomposition_tests.jl — native Malis–Vargas motion(::HomographyMat)
#
# PoseLib-independent: validates the decomposition by reconstruction
# (H̄ = R + t·nᵀ), proper-rotation candidates, unit normals, and the degenerate
# single-candidate path. (Numerical equivalence to PoseLib is checked separately
# in an RVG-env script, since PoseLib is not a VGC test dependency.)
# =============================================================================

@testitem "homography_decomposition" begin
    using Test
    using VisualGeometryCore
    using VisualGeometryCore: HomographyMat, motion
    using VisualGeometryCore.Synthetic: optimistic_pose_error
    using LinearAlgebra, StaticArrays, Random
    using Rotations: RotMatrix, RotXYZ, rotation_angle

    @testset "motion(::HomographyMat) — Malis–Vargas decomposition" begin
        rng = MersenneTwister(7)

        @testset "4 candidates: reconstruct H̄, proper rotations, unit normals" begin
            for _ in 1:300
                R = rand(rng, RotMatrix{3, Float64})
                n = normalize(SVector{3, Float64}(randn(rng), randn(rng), randn(rng)))
                t = SVector{3, Float64}(randn(rng), randn(rng), randn(rng))
                H = SMatrix{3, 3, Float64, 9}(R) + t * n'         # calibrated homography
                decs = motion(HomographyMat{Float64}(H))
                @test length(decs) == 4
                HH = det(H) < 0 ? -H : H
                H2 = HH / svd(HH).S[2]                         # normalized homography
                for (Rr, tr, nr) in decs
                    Rm = SMatrix{3, 3, Float64, 9}(Rr)
                    @test norm(Rm'Rm - I) < 1.0e-9               # orthonormal
                    @test abs(det(Rm) - 1) < 1.0e-9              # proper (det +1)
                    @test isapprox(norm(nr), 1.0; atol = 1.0e-9) # unit plane normal
                    @test norm((Rm + tr * nr') - H2) < 1.0e-9    # reconstructs H̄
                end
                # Candidate order: the two ± twins share a rotation.
                @test decs[1][1] == decs[2][1]
                @test decs[3][1] == decs[4][1]
                @test decs[1][2] ≈ -decs[2][2]                 # t flips with n
                @test decs[1][3] ≈ -decs[2][3]
            end
        end

        @testset "degenerate (pure rotation) ⇒ single candidate" begin
            R = rand(rng, RotMatrix{3, Float64})
            decs = motion(HomographyMat{Float64}(SMatrix{3, 3, Float64, 9}(R)))
            @test length(decs) == 1
        end

        @testset "optimistic_pose_error(::HomographyMat) — per-axis minima" begin
            for _ in 1:100
                R_gt = SMatrix{3, 3, Float64, 9}(rand(rng, RotMatrix{3, Float64}))
                t_gt = normalize(SVector{3, Float64}(randn(rng), randn(rng), randn(rng)))
                R = rand(rng, RotMatrix{3, Float64})
                n = normalize(SVector{3, Float64}(randn(rng), randn(rng), randn(rng)))
                t = SVector{3, Float64}(randn(rng), randn(rng), randn(rng))
                H = HomographyMat{Float64}(SMatrix{3, 3, Float64, 9}(R) + t * n')
                Δr, Δt = optimistic_pose_error(H, R_gt, t_gt)
                @test 0.0 <= Δr <= 180.0
                @test 0.0 <= Δt <= 180.0
                # Δr is the per-axis minimum rotation error over the candidates
                decs = motion(H)
                Δr_ref = minimum(
                    rad2deg(rotation_angle(RotMatrix{3}(SMatrix{3, 3, Float64, 9}(Rk)' * R_gt)))
                        for (Rk, _t, _n) in decs
                )
                @test Δr ≈ Δr_ref
            end
            # the pixel-space form calibrates by the intrinsics: inv(K₂)·H·K₁
            K1 = SMatrix{3, 3, Float64}(800.0, 0, 0, 0, 810.0, 0, 320.0, 240.0, 1.0)
            K2 = SMatrix{3, 3, Float64}(700.0, 0, 0, 0, 705.0, 0, 300.0, 250.0, 1.0)
            R = rand(rng, RotMatrix{3, Float64}); t = SVector(0.3, -0.1, 0.2); n = normalize(SVector(0.1, 0.2, 1.0))
            Hcal = HomographyMat{Float64}(SMatrix{3, 3, Float64, 9}(R) + t * n')
            Hpx = HomographyMat{Float64}(Tuple(K2 * SMatrix{3, 3, Float64, 9}(Hcal) * inv(K1)))
            # against a ground truth a few degrees off (errors well away from 0)
            R_gt = SMatrix{3, 3}(R) * SMatrix{3, 3, Float64}(RotMatrix{3, Float64}(RotXYZ(0.05, -0.03, 0.04)))
            t_gt = normalize(t + SVector(0.05, 0.02, -0.03))
            e_px = optimistic_pose_error(Hpx, K1, K2, R_gt, t_gt)
            e_cal = optimistic_pose_error(Hcal, R_gt, t_gt)
            @test e_px[1] ≈ e_cal[1] rtol = 1.0e-8
            @test e_px[2] ≈ e_cal[2] rtol = 1.0e-8
        end
    end
end
