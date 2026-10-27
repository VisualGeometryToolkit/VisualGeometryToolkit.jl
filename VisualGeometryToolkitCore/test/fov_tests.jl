# fov (cameras/utilities.jl): the full angle a bounded camera's extent
# subtends, through K⁻¹ and the lens; the inverse of focal_length(sensor, θ).
@testitem "fov" begin
    using Test
    using VisualGeometryToolkitCore
    using VisualGeometryToolkitCore: view_angle_limit, Pinhole
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
        c = BoundedCamera(mk(150.0, 150.0, 960.0, 600.0, Lens(Equidistant())), s)
        @test image_circle(c).a < 600                    # inside the sensor
        θl = view_angle_limit(lens(c))
        @test fov(c) == 2θl
        @test fov(c; dimension = :diagonal) == 2θl
        @test fov(c; dimension = :vertical) == 2θl
        # a cropped fisheye: the edge is imaged, the true angle comes back
        w = BoundedCamera(mk(500.0, 500.0, 960.0, 600.0, Lens(Equidistant())), s)
        @test fov(w) ≈ 2 * 960 / 500 rtol = 1.0e-9
        @test fov(w; dimension = :vertical) ≈ 2 * 600 / 500 rtol = 1.0e-9
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
