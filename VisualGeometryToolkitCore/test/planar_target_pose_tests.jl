# =============================================================================
# planar_target_pose_tests.jl — Extrinsics3(H, intrinsics): the pose of a
# planar target from its plane → image homography, the inverse of
# HomographyMat(camera). Exact on an exact H; on an H fitted to noisy pixels it
# recovers the true pose to within the noise.
# =============================================================================

@testitem "planar target pose from a homography" begin
    using Test
    using VisualGeometryToolkitCore
    using VisualGeometryToolkitCore: HomographyMat, Extrinsics3, CameraCalibrationMat, LogicalIntrinsics,
        homography_dlt, Cspond, Point2
    using LinearAlgebra, StaticArrays, Random
    using Statistics: median
    using Rotations: RotMatrix, RotXYZ, rotation_angle

    K = CameraCalibrationMat(SMatrix{3, 3, Float64}(1200.0, 0, 0, 0, 1180.0, 0, 640.0, 480.0, 1))
    intr = LogicalIntrinsics(K)
    Km = SMatrix{3, 3, Float64}(K)
    rng = MersenneTwister(11)
    # A board of 150 mm seen from 0.4–3 m at up to 60° obliqueness.
    randpose() = (RotMatrix{3, Float64}(RotXYZ(0.9 * (rand(rng) - 0.5), 0.9 * (rand(rng) - 0.5), 2π * rand(rng))),
                  SVector(200 * randn(rng), 150 * randn(rng), 400 + 2600 * rand(rng)))
    H_of(R, t) = HomographyMat{Float64}(Km * hcat(R[:, 1], R[:, 2], t))

    @testset "exact H: the identity" begin
        for _ in 1:200
            R, t = randpose()
            X = Extrinsics3(H_of(R, t), intr)
            @test X isa Extrinsics3
            @test rotation_angle(RotMatrix{3}(SMatrix{3, 3}(X.R)' * R)) < 1.0e-9
            @test norm(SVector{3}(X.t) - t) < 1.0e-8 * norm(t)
            # Any scale and sign of H is the same plane map.
            Xs = Extrinsics3(HomographyMat{Float64}(-3.7 * SMatrix{3, 3}(H_of(R, t))), intr)
            @test SMatrix{3, 3}(Xs.R) ≈ SMatrix{3, 3}(X.R) && SVector{3}(Xs.t) ≈ SVector{3}(X.t)
        end
    end

    @testset "a homography fitted to noisy pixels recovers the true pose" begin
        board = [SVector(x, y) for x in 0.0:15.0:150.0 for y in 0.0:15.0:150.0]
        dr = Float64[]; dt = Float64[]
        for _ in 1:100
            R, t = randpose()
            t = SVector(t[1] / 4, t[2] / 4, 400 + mod(t[3], 600))   # 0.4–1 m: the board spans ≥ 180 px
            px = [let c = Km * (R * SVector(b[1], b[2], 0.0) + t); SVector(c[1] / c[3], c[2] / c[3]) + 0.3 * randn(rng, SVector{2}) end
                  for b in board]
            Hf = homography_dlt([Cspond(Point2(b...), Point2(p...)) for (b, p) in zip(board, px)])
            X = Extrinsics3(Hf, intr)
            push!(dr, rad2deg(rotation_angle(RotMatrix{3}(SMatrix{3, 3}(X.R)' * R))))
            push!(dt, norm(SVector{3}(X.t) - t) / norm(t))
        end
        # 0.3 px of noise on 121 points of a board spanning ≥ 180 px: the closed
        # form is within half a degree and half a percent of the truth.
        @test median(dr) < 0.5
        @test median(dt) < 5.0e-3
    end
end
