# image_circle (cameras/image_circle.jl): the region of the sensor a lens
# lights, predicted from the lens (view_angle_limit through the radial
# projection and K) or from a datasheet diameter through the sensor's pitch —
# the same Ellipse either way. `nothing` when the lens has no limit of its own.
@testitem "image_circle" begin
    using Test
    using VisualGeometryToolkitCore
    using VisualGeometryToolkitCore: view_angle_limit, projected_radius, Pinhole, KannalaBrandt, Division, RegionIntersection
    using FlexUnits: DimensionError
    using StaticArrays

    mk(fx, fy, cx, cy, l) = CameraModel(CameraCalibrationMat(@SMatrix [fx 0.0 cx; 0.0 fy cy; 0.0 0.0 1.0]); lens = l)
    imx250 = CMOS_SENSORS["Sony"]["IMX250"]              # 2448 × 2048 @ 3.45 μm
    d780 = CMOS_SENSORS["Nikon"]["D780"]                 # 6048 × 4024 full frame

    @testset "equidistant fisheye, isotropic K: radius f·θ_max about the principal point" begin
        c = mk(800.0, 800.0, 1224.0, 1024.0, Lens(Equidistant()))
        θ = view_angle_limit(lens(c))
        @test θ == π - 1.0e-3                            # the sphere's edge: a finite disc
        q = image_circle(c)
        @test q isa Ellipse{Float64}
        @test q.a ≈ 800 * θ rtol = 1.0e-12
        @test q.b ≈ 800 * θ rtol = 1.0e-12
        @test centroid(q) ≈ Point2(1224.0, 1024.0) atol = 1.0e-9
        @test Point2(1224.0 + 800θ - 0.5, 1024.0) ∈ q
        @test Point2(1224.0 + 800θ + 0.5, 1024.0) ∉ q
        @test Point2(1224.0, 1024.0 - 800θ + 0.5) ∈ q
        @test image_circle(BoundedCamera(c, imx250)) == q   # through the wrapper
    end

    @testset "anisotropic K: the ellipse with semi-axes fx·ρ, fy·ρ" begin
        l = KannalaBrandt(-0.1)                          # folds at θ = √(1/0.3)
        ρ = projected_radius(l, view_angle_limit(l))
        q = image_circle(mk(900.0, 600.0, 640.0, 480.0, l))
        @test q.a ≈ 900ρ rtol = 1.0e-9
        @test q.b ≈ 600ρ rtol = 1.0e-9
        @test mod(q.θ, π) ≈ 0 atol = 1.0e-9
        @test centroid(q) ≈ Point2(640.0, 480.0) atol = 1.0e-9
    end

    @testset "no limit of its own: nothing" begin
        @test image_circle(mk(800.0, 800.0, 640.0, 480.0, Pinhole())) === nothing
        @test image_circle(mk(800.0, 800.0, 640.0, 480.0, Lens(Perspective(), Radial((0.05,))))) === nothing   # monotone to the edge
        @test image_circle(mk(800.0, 800.0, 640.0, 480.0, Division(-0.3))) === nothing   # a lift that never folds
        # a fold (perspective, a division lift, a Kannala–Brandt fisheye) is a limit: a conic
        @test image_circle(mk(800.0, 800.0, 640.0, 480.0, Lens(Perspective(), Radial((-0.2,))))) isa Ellipse
        @test image_circle(mk(800.0, 800.0, 640.0, 480.0, Division(-0.3, 0.05))) isa Ellipse
        @test image_circle(mk(800.0, 800.0, 640.0, 480.0, KannalaBrandt(-0.1))) isa Ellipse
        # nothing restricts nothing
        @test (Rect(0, 0, 1280, 960) ∩ image_circle(mk(800.0, 800.0, 640.0, 480.0, Pinhole()))) == Rect(0, 0, 1280, 960)
    end

    @testset "not radially symmetric: an inscribed bound" begin
        l = Lens(Perspective(), Additive(Radial((-0.3, 0.04)), Tangential(1.0e-3, -5.0e-4)))
        c = mk(800.0, 800.0, 640.0, 480.0, l)
        q = image_circle(c)
        θ = view_angle_limit(l)
        # the image of the θ_max cone lies on or outside the circle, so its inside is imaged
        rim = [project(c, SVector(sin(θ) * cos(ψ), sin(θ) * sin(ψ), cos(θ))) for ψ in range(0, 2π; length = 997)]
        @test q.a ≈ q.b rtol = 1.0e-12
        @test all(u -> hypot((u .- centroid(q))...) >= q.a * (1 - 1.0e-6), rim)
        @test minimum(u -> hypot((u .- centroid(q))...), rim) ≈ q.a rtol = 1.0e-4
    end

    @testset "the datasheet route, through the pitch" begin
        c = mk(800.0, 800.0, 1224.0, 1024.0, Lens(Equidistant()))
        q = image_circle(c, imx250, 17.3mm)
        @test q.a ≈ 2507.246376811594 rtol = 1.0e-12   # 17.3 mm / (2 · 3.45 μm)
        @test q.b == q.a
        @test centroid(q) == Point2(1224.0, 1024.0)
        @test image_circle(BoundedCamera(c, imx250), 17.3mm) == q
        @test image_circle(c, imx250, 1.73cm).a ≈ q.a
        # an anisotropic pitch: an ellipse
        s = Sensor(Size2(width = 2000px, height = 1500px), Size2(width = 3.0μm / px, height = 4.0μm / px))
        qa = image_circle(c, s, 6.0mm)
        @test qa.a ≈ 1000.0 && qa.b ≈ 750.0
        # the two routes agree where the fitted θ_max corresponds to the stated diameter
        θ = π - 1.0e-3
        f = 2507.246376811594 / θ
        cf = mk(f, f, 1224.0, 1024.0, Lens(Equidistant()))
        qf = image_circle(cf)
        @test qf.a ≈ q.a rtol = 1.0e-12
        @test qf.b ≈ q.b rtol = 1.0e-12
        @test centroid(qf) ≈ centroid(q) atol = 1.0e-9
        # refusals: a non-length diameter, a camera with no pitch
        @test_throws DimensionError image_circle(c, imx250, 17.3px)
        @test_throws DimensionError image_circle(c, imx250, 17.3dpi)
        @test_throws ArgumentError image_circle(BoundedCamera(c, Size2(2448, 2048)), 17.3mm)
    end

    @testset "is the whole sensor lit? Rect(extent) ∈ circle" begin
        kb = BoundedCamera(mk(800.0, 800.0, 640.0, 480.0, KannalaBrandt(-0.1)), Size2(1280, 960))
        q = image_circle(kb)
        @test q.a > hypot(640, 480)                      # the fold lies beyond the sensor
        @test Rect(extent(kb)) ∈ q                       # a conic, covering: nothing to restrict
        fish = BoundedCamera(mk(800.0, 800.0, 3024.0, 2012.0, Lens(Equidistant())), d780)
        qf = image_circle(fish)
        @test Rect(extent(fish)) ∉ qf                    # a circular fisheye on full frame
        domain = Rect(extent(fish)) ∩ qf
        @test domain isa RegionIntersection
        @test Point2(3024.0, 2012.0) ∈ domain && Point2(10.0, 10.0) ∉ domain
    end
end
