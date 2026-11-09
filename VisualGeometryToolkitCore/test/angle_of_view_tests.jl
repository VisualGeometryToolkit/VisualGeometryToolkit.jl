# angle_of_view (cameras/utilities.jl): the full angle a bounded camera IMAGES — its extent ∩ its lens coverage —
# through the camera's backproject; the inverse of focal_length(resolution, θ) for a centred pinhole. And
# field_of_view: the size that angle spans at a distance.
@testitem "angle_of_view" begin
    using Test
    using VisualGeometryToolkitCore
    using VisualGeometryToolkitCore: view_angle_limit, view_angle, Pinhole, KannalaBrandt, Equidistant, Lens
    using StaticArrays

    mk(fx, fy, cx, cy, l) = CameraModel(CameraCalibrationMat(@SMatrix [fx 0.0 cx; 0.0 fy cy; 0.0 0.0 1.0]); lens = l)
    s = Sensor(Size2(width = 1920px, height = 1200px), Size2(width = 3.0μm / px, height = 4.0μm / px))

    @testset "the inverse of focal_length" begin
        θ = deg2rad(60)
        fh = ustrip(px, focal_length(s.resolution, θ))
        fv = ustrip(px, focal_length(s.resolution, θ; dimension = :vertical))
        @test angle_of_view(BoundedCamera(mk(fh, fh, 960.0, 600.0, Pinhole()), s)) ≈ θ rtol = 1.0e-12
        @test angle_of_view(BoundedCamera(mk(fv, fv, 960.0, 600.0, Pinhole()), s); dimension = :vertical) ≈ θ rtol = 1.0e-12
        # the metric form: f in mm through the pitch of that axis is the same camera
        @test ustrip(μm, focal_length(s, θ; dimension = :vertical)) ≈ fv * 4.0
    end

    @testset "a circular fisheye: the lens's own limit" begin
        fold = KannalaBrandt(-4 / (3π^2))                # folds at 90°, r = π/3 there
        c = BoundedCamera(mk(400.0, 400.0, 960.0, 600.0, fold), s)
        @test valid_domain(c).a < 600                    # 419 px: the fold, inside the sensor
        θl = view_angle_limit(lens(c))
        @test angle_of_view(c) == 2θl
        @test angle_of_view(c; dimension = :diagonal) == 2θl
        @test angle_of_view(c; dimension = :vertical) == 2θl
        # the fold between the edges: the horizontal edge (960 px) lies past the
        # fold (733 px) and clips to the limit; the vertical edge (600 px) is read
        p = BoundedCamera(mk(700.0, 700.0, 960.0, 600.0, fold), s)
        @test 600 < valid_domain(p).a < 960
        @test angle_of_view(p) == 2view_angle_limit(fold)
        @test angle_of_view(p; dimension = :vertical) ≈ 2view_angle(fold, 600 / 700) rtol = 1.0e-9
        # off-centre: one edge inside the circle, the other past it — the nearer one is read
        o = BoundedCamera(mk(700.0, 700.0, 700.0, 600.0, fold), s)
        @test angle_of_view(o) ≈ 2view_angle(fold, 1.0) rtol = 1.0e-9
        # an undistorted fisheye (the whole plane): the true angle comes back
        w = BoundedCamera(mk(500.0, 500.0, 960.0, 600.0, Lens(Equidistant())), s)
        @test angle_of_view(w) ≈ 2 * 960 / 500 rtol = 1.0e-9
        @test angle_of_view(w; dimension = :vertical) ≈ 2 * 600 / 500 rtol = 1.0e-9
    end

    @testset "a skewed K: the camera's own backprojection" begin
        sk = 40.0
        K = @SMatrix [1000.0 sk 960.0; 0.0 900.0 600.0; 0.0 0.0 1.0]
        c = BoundedCamera(CameraModel(CameraCalibrationMat(K); lens = Pinhole()), s)
        @test angle_of_view(c) ≈ 2atan(960 / 1000) rtol = 1.0e-12  # on the principal row skew does not enter
        θv(v) = (y = (v - 600) / 900; x = -sk * y / 1000; atan(hypot(x, y)))
        @test angle_of_view(c; dimension = :vertical) ≈ 2min(θv(0.0), θv(1200.0)) rtol = 1.0e-12
    end

    @testset "an off-centre principal point: the nearer edge" begin
        c = BoundedCamera(mk(1000.0, 1000.0, 800.0, 650.0, Pinhole()), s)
        @test angle_of_view(c) ≈ 2atan(800 / 1000) rtol = 1.0e-12
        @test angle_of_view(c; dimension = :vertical) ≈ 2atan(550 / 1000) rtol = 1.0e-12
    end

    @testset "the diagonal through K⁻¹ of the corners" begin
        c = BoundedCamera(mk(1000.0, 1000.0, 960.0, 600.0, Pinhole()), s)
        @test angle_of_view(c; dimension = :diagonal) ≈ 2atan(hypot(960, 600) / 1000) rtol = 1.0e-12
        a = BoundedCamera(mk(1000.0, 800.0, 960.0, 600.0, Pinhole()), s)
        @test angle_of_view(a; dimension = :diagonal) ≈ 2atan(hypot(960 / 1000, 600 / 800)) rtol = 1.0e-12
        @test angle_of_view(a; dimension = :vertical) ≈ 2atan(600 / 800) rtol = 1.0e-12
        @test_throws ArgumentError angle_of_view(c; dimension = :radial)
    end
    @testset "a projective camera has no angle of view: an ArgumentError, as fov_radius" begin
        pc = BoundedCamera(CameraModel(ProjectiveIntrinsics(Point2(960.0, 600.0)), Pinhole()), s)
        @test_throws ArgumentError angle_of_view(pc)
        @test_throws ArgumentError angle_of_view(pc; dimension = :diagonal)
        err = try angle_of_view(pc); catch e; e; end
        @test occursin("needs a metric camera", sprint(showerror, err))
        @test_throws ArgumentError fov_radius(model(pc), 1.0)
    end

    @testset "the lens coverage bounds the angle: the edge of what is imaged, not the sensor's" begin
        c = mk(1000.0, 1000.0, 960.0, 600.0, Pinhole())
        whole = BoundedCamera(c, s)
        # a disc of 500 px about the principal point: every direction's edge is the disc's
        disc = BoundedCamera(c, s; lens_coverage = Circle(Point2(960.0, 600.0), 500.0))
        for d in (:horizontal, :vertical, :diagonal)
            @test angle_of_view(disc; dimension = d) ≈ 2atan(500 / 1000) rtol = 1.0e-9
        end
        # an off-centre disc: the nearer of its two edges (460 px left of the principal point, 540 right)
        off = BoundedCamera(c, s; lens_coverage = Circle(Point2(1000.0, 600.0), 500.0))
        @test angle_of_view(off) ≈ 2atan(460 / 1000) rtol = 1.0e-9
        # a coverage beyond the sensor changes nothing: the sensor's edges are read, bit for bit
        big = BoundedCamera(c, s; lens_coverage = Circle(Point2(960.0, 600.0), 5000.0))
        for d in (:horizontal, :vertical, :diagonal)
            @test angle_of_view(big; dimension = d) == angle_of_view(whole; dimension = d)
        end
        # the 14 mm's case: a disc wider than the sensor is tall, narrower than it is wide
        cut = BoundedCamera(c, s; lens_coverage = Circle(Point2(960.0, 600.0), 800.0))
        @test angle_of_view(cut; dimension = :vertical) == angle_of_view(whole; dimension = :vertical)
        @test angle_of_view(cut) ≈ 2atan(800 / 1000) rtol = 1.0e-9
        # nothing imaged at the principal point: no angle about it
        @test_throws ArgumentError angle_of_view(BoundedCamera(c, s; lens_coverage = Circle(Point2(100.0, 100.0), 50.0)))
    end

    @testset "field_of_view: the size the angle of view spans at a distance" begin
        c = BoundedCamera(mk(1000.0, 1000.0, 960.0, 600.0, Pinhole()), s)
        @test field_of_view(c, 2.0m) ≈ 2 * 2.0m * (960 / 1000) rtol = 1.0e-12
        @test field_of_view(c, 2.0m; dimension = :vertical) ≈ 2 * 2.0m * (600 / 1000) rtol = 1.0e-12
        @test field_of_view(c, 3.0) ≈ 2 * 3.0 * (960 / 1000) rtol = 1.0e-12            # a bare number: the same unit back
        @test ustrip(m, field_of_view(BoundedCamera(mk(1000.0, 1000.0, 960.0, 600.0, Pinhole()), s;
                                                   lens_coverage = Circle(Point2(960.0, 600.0), 500.0)), 2.0m)) ≈ 2.0 rtol = 1.0e-9   # ≈ in metres (FlexUnits)
        # 180° or wider spans no finite size on a plane
        w = BoundedCamera(mk(250.0, 250.0, 960.0, 600.0, Lens(Equidistant())), s)
        @test angle_of_view(w) >= π
        @test_throws ArgumentError field_of_view(w, 1.0m)
        @test_throws ArgumentError field_of_view(c, -1.0m)
    end

    @testset "no old name: fov is gone" begin
        @test !isdefined(VisualGeometryToolkitCore, :fov)
    end
end
