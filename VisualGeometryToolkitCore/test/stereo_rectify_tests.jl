@testitem "stereo_rectify" begin
    import VisualGeometryToolkitCore as VgtkCore
    using Test
    using VisualGeometryToolkitCore
    using VisualGeometryToolkitCore: model
    using StaticArrays
    using LinearAlgebra: norm, I
    using Rotations: RotMatrix, RotXYZ

    # =============================================================================
    # Stereo rectification (0.16) — rectify(rig) on the rig types
    # =============================================================================
    #
    # Covers: the rectified rig's relative pose (pure x-baseline, identity
    # rotation), row alignment and the disparity law fx·B/z on synthetic pinhole
    # correspondences (the decisive numerical test of the frame construction),
    # and the image form through a DIVISION lens — a bright dot placed by the
    # lens projection in each view must land on the same rectified row with the
    # predicted disparity, exercising the fused K∘lens∘homography warp (the lens
    # takes the rectified ray itself, no division) and the rectified frame both FramedImages share.


    @testset "Stereo rectification (0.16)" begin
        K1 = @SMatrix [600.0 0.0 320.0; 0.0 600.0 240.0; 0.0 0.0 1.0]
        K2 = @SMatrix [700.0 0.0 330.0; 0.0 700.0 250.0; 0.0 0.0 1.0]
        R = RotMatrix{3, Float64}(RotXYZ(0.05, -0.1, 0.03))
        t = SVector(-500.0, 20.0, 5.0)                     # mostly-horizontal baseline
        rig = StereoRig(K1, K2, SMatrix{3, 3, Float64}(R), t; ids = (:left, :right))

        res = rectify(rig)
        B = norm(t)

        @testset "rectified rig: pure baseline" begin
            Xr = extrinsics(res.rig)
            @test RotMatrix(Xr.R) ≈ SMatrix{3, 3, Float64}(I) atol = 1.0e-12
            @test abs(Xr.t[2]) < 1.0e-9 && abs(Xr.t[3]) < 1.0e-9
            @test abs(abs(Xr.t[1]) - B) < 1.0e-9
            @test ids(res.rig) == (:left, :right)
            @test model(source(res.rig)).lens isa VgtkCore.Pinhole
            # Q₂·R_rel = Q₁ — the two rectifying rotations agree on world directions
            @test SMatrix{3, 3, Float64}(res.R[2]) * SMatrix{3, 3, Float64}(R) ≈
                SMatrix{3, 3, Float64}(res.R[1]) atol = 1.0e-12
        end

        @testset "row alignment + disparity = fx·B/z" begin
            s = sign(extrinsics(res.rig).t[1])             # baseline sign in the rectified frame
            for X1 in (
                    SVector(300.0, -150.0, 4000.0), SVector(-700.0, 400.0, 6000.0),
                    SVector(1200.0, 100.0, 9000.0), SVector(-100.0, -600.0, 3500.0),
                )
                u1 = PerspectiveMap()(K1 * X1)
                u2 = PerspectiveMap()(K2 * (R * X1 + t))
                p1 = PerspectiveMap()(res.K * SMatrix{3, 3, Float64}(res.R[1]) * inv(K1) * SVector(u1[1], u1[2], 1.0))
                p2 = PerspectiveMap()(res.K * SMatrix{3, 3, Float64}(res.R[2]) * inv(K2) * SVector(u2[1], u2[2], 1.0))
                @test abs(p1[2] - p2[2]) < 1.0e-8          # SAME ROW
                zr = (SMatrix{3, 3, Float64}(res.R[1]) * X1)[3]
                @test p1[1] - p2[1] ≈ -s * res.K[1, 1] * B / zr rtol = 1.0e-9
            end
        end

        @testset "image form through a division lens" begin
            W, H = 640, 480
            Kd = @SMatrix [400.0 0.0 320.0; 0.0 400.0 240.0; 0.0 0.0 1.0]
            m1 = CameraModel(CameraCalibrationMat(Kd); lens = Division(-0.15))
            m2 = CameraModel(CameraCalibrationMat(Kd); lens = Division(-0.15))
            cam1 = PosedCamera(m1, EuclideanMap(one(RotMatrix{3, Float64}), zero(Vec3{Float64})))
            cam2 = PosedCamera(m2, EuclideanMap(RotMatrix{3, Float64}(RotXYZ(0.02, -0.04, 0.01)), Vec3(-300.0, 5.0, 2.0)))
            drig = StereoRig(cam1, cam2; ids = (:l, :r))

            X = SVector(200.0, -100.0, 3000.0)             # view-1 coordinates
            pix(m, Xc) = begin                             # forward: persp → lens → K
                n = PerspectiveMap()(Xc)
                d = m.lens(n)
                PerspectiveMap()(SMatrix{3, 3, Float64}(m.intrinsics.K) * SVector(d[1], d[2], 1.0))
            end
            Xrel = extrinsics(drig)
            u1 = pix(m1, X)
            u2 = pix(m2, RotMatrix(Xrel.R) * X + Xrel.t)
            imA = zeros(Float64, H, W); imB = zeros(Float64, H, W)
            imA[round(Int, u1[2] + 0.5), round(Int, u1[1] + 0.5)] = 1.0
            imB[round(Int, u2[2] + 0.5), round(Int, u2[1] + 0.5)] = 1.0

            out = rectify(drig, (imA, imB))
            ciA, ciB = out.images
            @test ciA isa FramedImage && ciB isa FramedImage
            @test ciA.rect === ciB.rect                    # ONE shared rectified frame
            aA, aB = argmax(ciA.data), argmax(ciB.data)
            @test abs(aA[1] - aB[1]) <= 1                  # same row, ±1 px (bilinear + rounding)
            Bd = norm(Xrel.t)
            zr = (SMatrix{3, 3, Float64}(out.R[1]) * X)[3]
            s = sign(extrinsics(out.rig).t[1])
            @test abs((aA[2] - aB[2]) - (-s * out.K[1, 1] * Bd / zr)) <= 1.5
        end
    end

    @testset "a pinhole view warps through the same K ∘ lens ∘ homography chain" begin
        Ki = @SMatrix [300.0 0.0 80.0; 0.0 300.0 60.0; 0.0 0.0 1.0]
        Kr_inv = inv(@SMatrix [280.0 0.0 82.0; 0.0 280.0 58.0; 0.0 0.0 1.0])
        Qi = RotMatrix{3, Float64}(RotXYZ(0.01, -0.02, 0.005))
        m = CameraModel(CameraCalibrationMat(Ki); lens = Pinhole())
        math = VgtkCore.Cameras._rectify_math(m, Ki, Qi, Kr_inv)
        H = HomographyMap(Ki * SMatrix{3, 3, Float64}(Qi)' * Kr_inv)
        for u in (SVector(10.0, 20.0), SVector(80.0, 60.0), SVector(150.0, 110.0))
            @test math(u) ≈ PerspectiveMap()(H(u)) atol = 1.0e-9
        end
        img = [sin(x / 7) * cos(y / 5) for y in 1:120, x in 1:160]
        a = image_warp!(similar(img), image_warp(Float64, math, Rect(0, 0, 160, 120)), img)
        b = image_warp!(similar(img), image_warp(Float64, H, Rect(0, 0, 160, 120)), img)
        @test maximum(abs.(a .- b)) < 1.0e-9
    end
    # the rectification of two SplineLens cameras: spline_lens_tests.jl
end
