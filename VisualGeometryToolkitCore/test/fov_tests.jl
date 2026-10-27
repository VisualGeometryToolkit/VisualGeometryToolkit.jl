# fov (cameras/utilities.jl): the full angle a bounded camera's extent
# subtends, through the camera's backproject; the inverse of focal_length(resolution, θ) for a centred pinhole.
@testitem "fov" begin
    using Test
    using VisualGeometryToolkitCore
    using VisualGeometryToolkitCore: view_angle_limit, view_angle, Pinhole, KannalaBrandt
    using StaticArrays

    mk(fx, fy, cx, cy, l) = CameraModel(CameraCalibrationMat(@SMatrix [fx 0.0 cx; 0.0 fy cy; 0.0 0.0 1.0]); lens = l)
    s = Sensor(Size2(width = 1920px, height = 1200px), Size2(width = 3.0μm / px, height = 4.0μm / px))

    @testset "the inverse of focal_length" begin
        θ = deg2rad(60)
        fh = ustrip(px, focal_length(s.resolution, θ))
        fv = ustrip(px, focal_length(s.resolution, θ; dimension = :vertical))
        @test fov(BoundedCamera(mk(fh, fh, 960.0, 600.0, Pinhole()), s)) ≈ θ rtol = 1.0e-12
        @test fov(BoundedCamera(mk(fv, fv, 960.0, 600.0, Pinhole()), s); dimension = :vertical) ≈ θ rtol = 1.0e-12
        # the metric form: f in mm through the pitch of that axis is the same camera
        @test ustrip(μm, focal_length(s, θ; dimension = :vertical)) ≈ fv * 4.0
    end

    @testset "a circular fisheye: the lens's own limit" begin
        fold = KannalaBrandt(-4 / (3π^2))                # folds at 90°, r = π/3 there
        c = BoundedCamera(mk(400.0, 400.0, 960.0, 600.0, fold), s)
        @test image_circle(c).a < 600                    # 419 px: inside the sensor
        θl = view_angle_limit(lens(c))
        @test fov(c) == 2θl
        @test fov(c; dimension = :diagonal) == 2θl
        @test fov(c; dimension = :vertical) == 2θl
        # the fold between the edges: the horizontal edge (960 px) lies past the
        # fold (733 px) and clips to the limit; the vertical edge (600 px) is read
        p = BoundedCamera(mk(700.0, 700.0, 960.0, 600.0, fold), s)
        @test 600 < image_circle(p).a < 960
        @test fov(p) == 2view_angle_limit(fold)
        @test fov(p; dimension = :vertical) ≈ 2view_angle(fold, 600 / 700) rtol = 1.0e-9
        # off-centre: one edge inside the circle, the other past it — the nearer one is read
        o = BoundedCamera(mk(700.0, 700.0, 700.0, 600.0, fold), s)
        @test fov(o) ≈ 2view_angle(fold, 1.0) rtol = 1.0e-9
        # an undistorted fisheye (the whole plane): the true angle comes back
        w = BoundedCamera(mk(500.0, 500.0, 960.0, 600.0, Lens(Equidistant())), s)
        @test fov(w) ≈ 2 * 960 / 500 rtol = 1.0e-9
        @test fov(w; dimension = :vertical) ≈ 2 * 600 / 500 rtol = 1.0e-9
    end

    @testset "a skewed K: the camera's own backprojection" begin
        sk = 40.0
        K = @SMatrix [1000.0 sk 960.0; 0.0 900.0 600.0; 0.0 0.0 1.0]
        c = BoundedCamera(CameraModel(CameraCalibrationMat(K); lens = Pinhole()), s)
        @test fov(c) ≈ 2atan(960 / 1000) rtol = 1.0e-12  # on the principal row skew does not enter
        θv(v) = (y = (v - 600) / 900; x = -sk * y / 1000; atan(hypot(x, y)))
        @test fov(c; dimension = :vertical) ≈ 2min(θv(0.0), θv(1200.0)) rtol = 1.0e-12
    end

    @testset "an off-centre principal point: the nearer edge" begin
        c = BoundedCamera(mk(1000.0, 1000.0, 800.0, 650.0, Pinhole()), s)
        @test fov(c) ≈ 2atan(800 / 1000) rtol = 1.0e-12
        @test fov(c; dimension = :vertical) ≈ 2atan(550 / 1000) rtol = 1.0e-12
    end

    @testset "the diagonal through K⁻¹ of the corners" begin
        c = BoundedCamera(mk(1000.0, 1000.0, 960.0, 600.0, Pinhole()), s)
        @test fov(c; dimension = :diagonal) ≈ 2atan(hypot(960, 600) / 1000) rtol = 1.0e-12
        a = BoundedCamera(mk(1000.0, 800.0, 960.0, 600.0, Pinhole()), s)
        @test fov(a; dimension = :diagonal) ≈ 2atan(hypot(960 / 1000, 600 / 800)) rtol = 1.0e-12
        @test fov(a; dimension = :vertical) ≈ 2atan(600 / 800) rtol = 1.0e-12
        @test_throws ArgumentError fov(c; dimension = :radial)
    end
end
