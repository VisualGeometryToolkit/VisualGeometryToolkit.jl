@testitem "stereo_rectify" begin
    using Test
    using VisualGeometryCore
    using VisualGeometryCore: model
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
            @test model(source(res.rig)).lens isa VisualGeometryCore.Pinhole
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
            cam1 = Camera(m1, EuclideanMap(one(RotMatrix{3, Float64}), zero(Vec3{Float64})))
            cam2 = Camera(m2, EuclideanMap(RotMatrix{3, Float64}(RotXYZ(0.02, -0.04, 0.01)), Vec3(-300.0, 5.0, 2.0)))
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
        math = VisualGeometryCore.Cameras._rectify_math(m, Ki, Qi, Kr_inv)
        H = HomographyMap(Ki * SMatrix{3, 3, Float64}(Qi)' * Kr_inv)
        for u in (SVector(10.0, 20.0), SVector(80.0, 60.0), SVector(150.0, 110.0))
            @test math(u) ≈ PerspectiveMap()(H(u)) atol = 1.0e-9
        end
        img = [sin(x / 7) * cos(y / 5) for y in 1:120, x in 1:160]
        a = image_warp!(similar(img), image_warp(Float64, math, Rect(0, 0, 160, 120)), img)
        b = image_warp!(similar(img), image_warp(Float64, H, Rect(0, 0, 160, 120)), img)
        @test maximum(abs.(a .- b)) < 1.0e-9
    end

    @testset "rectify with grid-model sensors" begin
        Ws, Hs = 160, 120
        Ks = @SMatrix [80.0 0.0 80.0; 0.0 80.0 60.0; 0.0 0.0 1.0]
        mdiv = CameraModel(CameraCalibrationMat(Ks); lens = VisualGeometryCore.Division(-0.12))
        g1 = CentralGenericModel(mdiv, (Ws, Hs); grid_size = (10, 8))
        g2 = CentralGenericModel(mdiv, (Ws, Hs); grid_size = (10, 8))
        X = EuclideanMap(RotMatrix{3, Float64}(RotXYZ(0.0, 0.03, 0.0)), SVector(-40.0, 0.0, 0.0))
        rig_par = StereoRig(Camera(mdiv, EuclideanMap(one(RotMatrix{3, Float64}), zeros(SVector{3, Float64}))), Camera(mdiv, X); ids = (:l, :r))
        rig_gen = StereoRig(Camera(g1, EuclideanMap(one(RotMatrix{3, Float64}), zeros(SVector{3, Float64}))), Camera(g2, X); ids = (:l, :r))
        # geometry form: a grid sensor cannot derive the shared K — loud contract
        @test_throws ArgumentError rectify(rig_gen)
        geo = rectify(rig_par)
        imgt = [
            exp(-((x - 70)^2 + (y - 55)^2) / 400) + 0.2 * sin(x / 9) * cos(y / 7)
                for y in 1:Hs, x in 1:Ws
        ]
        rp = rectify(rig_par, (imgt, imgt))
        rg = rectify(rig_gen, (imgt, imgt); K = geo.K)
        @test rg.K ≈ geo.K
        for i in 1:2
            @test rg.images[i].rect == rp.images[i].rect
            a = rg.images[i].data; b = rp.images[i].data
            @test size(a) == size(b)
            di = abs.(a[20:(end - 20), 20:(end - 20)] .- b[20:(end - 20), 20:(end - 20)])
            # grid-fit floor + remap interpolation vs the exact parametric warp
            @test maximum(di) < 0.03
        end
        # Float32: the grid view's dense kernel runs in the requested precision, as
        # its parametric sibling's warp does (the host remap used to refuse it)
        rg32 = rectify(Float32, rig_gen, (Float32.(imgt), Float32.(imgt)); K = geo.K)
        for i in 1:2
            @test eltype(rg32.images[i].data) == Float32
            @test maximum(abs.(rg32.images[i].data .- rg.images[i].data)) < 1.0e-4
        end
        # The grid view takes the output size `rectify` computed for the pair,
        # not a height re-derived from the horizontal scale: at scale 3 on this
        # 160×120 window that re-derivation gave 41 rows beside the parametric
        # view's 40.
        for s in (2.5, 3)
            rp_s = rectify(rig_par, (imgt, imgt); scale = s)
            rg_s = rectify(rig_gen, (imgt, imgt); K = geo.K, scale = s)
            for i in 1:2
                a = rg_s.images[i].data; b = rp_s.images[i].data
                @test size(a) == size(b)
                @test maximum(abs.(a[8:(end - 8), 8:(end - 8)] .- b[8:(end - 8), 8:(end - 8)])) < 0.03
            end
        end
    end
end
