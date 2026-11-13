# A bounded camera's bound and its lens coverage (cameras.jl): the bound is the pixel extent, or the physical Sensor whose
# resolution IS that extent; the lens coverage is where light reaches the sensor — a Rect2, a Circle or an Ellipse, by default
# the sensor's own Rect, never "none". Both are type parameters, so every bounded camera is a concrete, isbits value whose
# accessors infer, whichever bound and region it carries.
@testitem "BoundedCamera: bound and lens coverage" begin
    using Test
    using VisualGeometryToolkitCore
    using VisualGeometryToolkitCore: model, with_model, convert_eltype, lens_coverage, with_lens_coverage, WholePlane
    using VisualGeometryToolkitCore.Cameras: _with_extrinsics
    using StaticArrays
    using Rotations: RotMatrix, RotXYZ

    # the Sony FX3 with the Sigma 14 mm of 2026-10-27 (its KB fit), 4256 × 2848, 5.95 μm pitch
    m = CameraModel(1754.7px, SA[2133.6px, 1422.5px]; lens = KannalaBrandt(0.2468, 0.1505, -0.0214, 0.1308))
    W, H = 4256, 2848
    bounds = (Size2(W, H), Sensor(Size2(W, H), 5.95μm))
    regions = (nothing,
               Circle(Point2(2131.1, 1407.9), 1680.1),                       # the panorama config's circle
               Ellipse(Point2(2133.9, 1423.7), 1611.8, 1607.8, 1.5777))      # the grid's measured domain
    expected(::Nothing) = Rect2{Float64}(0, 0, W, H)
    expected(r) = r
    E = EuclideanMap(RotMatrix{3, Float64}(RotXYZ(0.1, -0.2, 0.05)), SVector(0.3, -0.1, 2.0))
    u = SVector(2500.0, 1500.0)

    for bound in bounds, region in regions
        bc = BoundedCamera(m, bound; lens_coverage = region)
        @testset "bound $(nameof(typeof(bound))), region $(nameof(typeof(region)))" begin
            @test isconcretetype(typeof(bc))
            @test isbits(bc)
            @test (@inferred extent(bc)) == Size2(W, H)
            @test (@inferred sensor(bc)) == (bound isa Sensor ? bound : nothing)
            @test (@inferred lens_coverage(bc)) == expected(region)
            @test (@inferred backproject(bc, u)) == backproject(m, u)
            @test (@inferred project(bc, backproject(m, u))) ≈ u
            # every rebuild keeps the bound and the region
            for rb in (convert_eltype(Float32, bc), with_model(bc, m))
                @test lens_coverage(rb) == lens_coverage(bc) && extent(rb) == extent(bc) && sensor(rb) == sensor(bc)
            end
            posed = BoundedCamera(PosedCamera(m, E), bound; lens_coverage = region)
            reposed = _with_extrinsics(posed, inv(E))
            @test lens_coverage(reposed) == lens_coverage(posed) && sensor(reposed) == sensor(posed)
        end
    end

    @testset "a degenerate coverage is refused (a writer would store what its reader refuses)" begin
        for bad in (Circle(Point2(5.0, 5.0), 0.0), Circle(Point2(5.0, 5.0), NaN), Circle(Point2(NaN, 5.0), 3.0),
                    Ellipse(Point2(5.0, 5.0), 3.0, 0.0, 0.0), Ellipse(Point2(5.0, 5.0), 3.0, -1.0, 0.0),
                    Rect2(0.0, 0.0, 0.0, 10.0), Rect2(0.0, 0.0, Inf, 10.0))
            @test_throws ArgumentError BoundedCamera(m, Size2(10, 10); lens_coverage = bad)
        end
    end

    @testset "the region is a geometric primitive in Float64" begin
        @test_throws ArgumentError BoundedCamera(m, Size2(10, 10); lens_coverage = WholePlane())
        @test lens_coverage(BoundedCamera(m, Size2(10, 10); lens_coverage = Circle(Point2(5, 5), 4))) === Circle(Point2(5.0, 5.0), 4.0)   # kept a Circle, in Float64
        b = with_lens_coverage(BoundedCamera(m, Size2(10, 10)), Circle(Point2(5.0, 5.0), 4.0))
        @test lens_coverage(b) == Circle(Point2(5.0, 5.0), 4.0) && extent(b) == Size2(10, 10)
    end
end

# What a bounded camera images (image_circle.jl): the sensor ∩ its lens coverage ∩ where its lens model is valid — one
# function of the camera alone, so no consumer carries an image circle of its own.
@testitem "imaged_region: sensor ∩ lens coverage ∩ valid domain" begin
    using Test
    using VisualGeometryToolkitCore
    using VisualGeometryToolkitCore: lens_coverage, with_lens_coverage, relabel, CameraFrame, WorldFrame, RotationMap, RegionIntersection
    using StaticArrays
    using Rotations: RotY

    W, H = 4256, 2848
    m = CameraModel(1754.7px, SA[2133.6px, 1422.5px]; lens = KannalaBrandt(0.2468, 0.1505, -0.0214, 0.1308))

    @testset "a lit disc cut by the sensor: clipped to the sensor" begin
        bc = BoundedCamera(m, Size2(W, H); lens_coverage = Circle(Point2(2140.8, 1375.8), 1742.2))   # the 14 mm's measured disc
        # the one small Union kept on purpose (owner, 2026-11-09): valid_domain(::CameraModel) is an Ellipse or WholePlane
        # by the lens's coefficients, so imaged_region is one of two concrete RegionIntersections — pinned here, no wider
        r = @inferred RegionIntersection imaged_region(bc)
        @test isconcretetype(typeof(r)) && isbits(r)
        @test Base.return_types(imaged_region, (typeof(bc),)) |> only |> (T -> T isa Union && length(Base.uniontypes(T)) == 2)
        b = Rect2(r)
        @test all(minimum(b) .>= 0) && maximum(b)[1] <= W && maximum(b)[2] <= H
        @test Point2(2140.8, 1375.8) ∈ r && Point2(10.0, 10.0) ∉ r
    end

    @testset "a spline lens: lit but outside where its grid is defined is not imaged" begin
        # the grid calibrated over a band of the sensor (x from 500 to 3755): its valid domain is that area
        g = CameraModel(SplineLens, m, Size2(W, H); grid_size = (35, 25), init = :quasi_interpolate, calib_area = Rect(500, 0, 3755 - 500 + 1, H - 1 - 0 + 1))
        bc = BoundedCamera(g, Size2(W, H); lens_coverage = Circle(Point2(2140.8, 1375.8), 1742.2))
        @test Point2(450.0, 1423.7) ∉ imaged_region(bc)                 # lit, left of the grid's area
        @test Point2(550.0, 1423.7) ∈ imaged_region(bc)
        area = Rect2(500.0, 0.0, 3256.0, Float64(H))
        r = valid_domain(g)
        @test r.origin ≈ area.origin && r.widths ≈ area.widths
        for p in (Point2(2133.9, 1423.7), Point2(450.0, 1423.7), Point2(3800.0, 5.0))
            @test (p ∈ valid_domain(g)) == (p ∈ area)
        end
    end

    @testset "a lens with no limit, the whole sensor lit: every pixel centre" begin
        pin = CameraModel(800.0px, SA[50.0px, 40.0px])
        r = imaged_region(BoundedCamera(pin, Size2(100, 80)))
        @test all(Point2(u - 0.5, v - 0.5) ∈ r for u in 1:100, v in 1:80)
    end

    @testset "view_overlap: each frame's own lens coverage, of different types" begin
        fish = CameraModel(; focal = 250.0, pp = (640.0, 480.0), lens = KannalaBrandt(-4 / (3π^2)))
        frame(R, region) = BoundedCamera(PosedCamera(fish, relabel(CameraFrame, WorldFrame, RotationMap(SMatrix{3, 3, Float64, 9}(R)))),
                                         Size2(1280, 960); lens_coverage = region)
        ca = Circle(Point2(640.0, 480.0), 240.0); eb = Ellipse(Point2(645.0, 478.0), 250.0, 230.0, 0.3)
        rows = view_overlap([frame(RotY(0.0), ca), frame(RotY(0.6), eb), frame(RotY(1.1), ca)])
        # the 0.8 two-argument form on the same frames and circles (view_overlap([A, B, C], [ca, eb, ca]), 2026-11-09)
        expected = [(1, 2, 2.7412784749222254, 0.7000022505703593, 0.6949793635464715),
                    (1, 3, 1.8977069809115283, 0.4845911021859392, 0.4845911021859392),
                    (2, 3, 3.099614193899441, 0.7858260003216053, 0.791505471435549)]
        @test length(rows) == 3
        for (r, e) in zip(rows, expected)
            @test (r.i, r.j) == e[1:2]
            @test isapprox(SVector(r.solid_angle, r.fraction_i, r.fraction_j), SVector(e[3:5]...); rtol = 1e-12)
        end
    end
end

# A bounded camera of a spline lens on a device: `Adapt.adapt` moves the wrapped camera's arrays (a pose kept) and keeps the
# bound and the lens coverage, so a kernel reads where the camera images (inside its coverage, its backprojection finite).
@testitem "BoundedCamera: Adapt keeps the bound and the lens coverage; a kernel reads where it images (CUDA availability-gated)" begin
    using Test
    using VisualGeometryToolkitCore
    using VisualGeometryToolkitCore: KannalaBrandt, lens_coverage, with_present, node_mask
    using StaticArrays
    using Adapt: adapt
    using KernelAbstractions
    using KernelAbstractions: @kernel, @index, @Const
    using Rotations: RotXYZ

    @kernel function imaged!(ok, @Const(ps), c)
        i = @index(Global)
        p = ps[i]
        d = backproject(c, p)
        ok[i] = p ∈ lens_coverage(c) && isfinite(d[1]) && isfinite(d[2]) && isfinite(d[3])
    end
    W, H = 1280, 960
    fish = CameraModel(; focal = 250.0, pp = (640.0, 480.0), lens = KannalaBrandt(-4 / (3π^2)))
    disc = Circle(Point2(640.0, 480.0), 250π / 3)
    m = CameraModel(SplineLens, fish, Size2(W, H); grid_size = (24, 19), init = :quasi_interpolate)
    m = with_present(m, node_mask(m, valid_domain(m) ∩ disc))
    X = EuclideanMap(RotXYZ(0.1, -0.2, 0.3), SVector(0.5, -0.1, 2.0))
    cams = (BoundedCamera(m, Size2(W, H); lens_coverage = disc), BoundedCamera(PosedCamera(m, X), Sensor(Size2(W, H), 5.95μm); lens_coverage = disc))
    ps = vec([SVector(x, y) for x in 0.25:19.5:(W - 0.25), y in 0.25:19.5:(H - 0.25)])
    host(c) = [p ∈ lens_coverage(c) && all(isfinite, backproject(c, p)) for p in ps]
    @test any(host(cams[1])) && !all(host(cams[1]))
    cuda = try
        mod = Base.require(Main, :CUDA)
        (mod !== nothing && Base.invokelatest(getproperty(mod, :functional))) ? mod : nothing
    catch
        nothing
    end
    if cuda === nothing
        @test_skip cuda !== nothing      # no functional GPU here: recorded as Broken, not as a success
    else
        CuArray = getproperty(cuda, :CuArray)
        gpu = Base.invokelatest(getproperty(cuda, :CUDABackend))
        for c in cams
            d = Base.invokelatest(adapt, CuArray, c)
            @test d isa BoundedCamera && d.bound === c.bound && lens_coverage(d) === lens_coverage(c)
            @test !(model(d).lens.backward.controls isa Array) && !(model(d).lens.present isa Array)
            @test isequal(Base.invokelatest(Array, model(d).lens.backward.controls), model(c).lens.backward.controls)
            c.camera isa PosedCamera && @test d.camera.extrinsics === c.camera.extrinsics
            ok = Base.invokelatest(CuArray, falses(length(ps)))
            Base.invokelatest(imaged!(gpu, 64), ok, Base.invokelatest(CuArray, ps), d; ndrange = length(ps))
            Base.invokelatest(KernelAbstractions.synchronize, gpu)
            @test Base.invokelatest(Array, ok) == host(c)
        end
    end
end

# The lens coverage less a rim margin (cameras.jl, hoisted from PanoramaStitching's frame region and RadiometricCamera's
# image circle less its rim): the kind kept, a circle's radius and an ellipse's semi-axes less the margin, a Rect inset
# only on a side inside the sensor (the sensor's own edge is not the lens's), `nothing` when nothing is left.
@testitem "lens_coverage less a rim margin" begin
    using Test
    using VisualGeometryToolkitCore
    using VisualGeometryToolkitCore: lens_coverage, with_lens_coverage
    using StaticArrays

    m = CameraModel(1754.7px, SA[2133.6px, 1422.5px]; lens = KannalaBrandt(0.2468, 0.1505, -0.0214, 0.1308))
    W, H = 4256, 2848
    circ = BoundedCamera(m, Size2(W, H); lens_coverage = Circle(Point2(2131.1, 1407.9), 1680.1))
    c = lens_coverage(circ, 24.0)
    @test c isa Circle{Float64} && c.center == Point2(2131.1, 1407.9) && c.r == 1680.1 - 24.0
    @test lens_coverage(circ, 0.0) == lens_coverage(circ)
    @test lens_coverage(circ, 1680.1) === nothing
    ell = BoundedCamera(m, Size2(W, H); lens_coverage = Ellipse(Point2(2133.9, 1423.7), 1611.8, 1607.8, 1.5777))
    e = lens_coverage(ell, 10)
    @test e isa Ellipse{Float64} && e.a == lens_coverage(ell).a - 10 && e.b == lens_coverage(ell).b - 10 && e.θ == lens_coverage(ell).θ && e.center == Point2(2133.9, 1423.7)
    @test lens_coverage(ell, 1607.8) === nothing
    # the whole sensor lit: nothing lost to the margin
    whole = BoundedCamera(m, Size2(W, H))
    @test lens_coverage(whole, 24.0) == Rect2{Float64}(0, 0, W, H)
    # a lit Rect inside the sensor on its left and top: inset there only
    part = BoundedCamera(m, Size2(W, H); lens_coverage = Rect2{Float64}(100, 50, W - 100, H - 50))
    @test lens_coverage(part, 24.0) == Rect2{Float64}(124, 74, W - 124, H - 74)
    @test lens_coverage(BoundedCamera(m, Size2(W, H); lens_coverage = Rect2{Float64}(100, 50, 30, 30)), 24.0) === nothing
    # what the camera images away from the rim: the imaged region of the camera with that coverage
    r = imaged_region(with_lens_coverage(circ, c))
    @test Point2(2131.1, 1407.9) ∈ r && !(Point2(2131.1 + 1680.1 - 20.0, 1407.9) ∈ r) && Point2(2131.1 + 1680.1 - 20.0, 1407.9) ∈ imaged_region(circ)
end

# How far an imaged region reaches from a centre (image_circle.jl): RadiometricCamera's radial vignetting radius and its
# rings' disc, and Estimation's fit_camera exit radii, read it.
@testitem "farthest_radius: an imaged region's reach from a centre, clipped by the sensor" begin
    using Test
    using VisualGeometryToolkitCore
    using VisualGeometryToolkitCore: farthest_radius, WholePlane
    using StaticArrays
    W, H = 520, 348
    c = SVector(W / 2, H / 2)
    sensor = Rect(0, 0, W, H)
    # the sensor alone: its farthest corner, exactly (the half diagonal from its centre)
    @test farthest_radius(sensor, c) == hypot(W, H) / 2
    @test farthest_radius(sensor, SVector(10.0, 20.0)) == hypot(W - 10.0, H - 20.0)
    # a circle centred on the centre inside the sensor: its radius
    circle = Circle(Point2(c...), 150.0)
    @test farthest_radius(sensor ∩ circle, c) ≈ 150.0 rtol = 1.0e-9
    # off centre: the farthest point of the arc inside the sensor (RadiometricCamera's 190 px case: 40 + 150)
    off = sensor ∩ Ellipse(Point2(300.0, 174.0), 150.0, 150.0, 0.0)
    @test farthest_radius(off, c) ≈ 190.0 rtol = 1.0e-9
    # a circle larger than the sensor is clipped by it: the corners, not the rim (a corner is a kink of the reach over the
    # azimuth, which Brent's method resolves to √eps in the azimuth: first order in the radius)
    big = sensor ∩ Circle(Point2(c...), 400.0)
    @test farthest_radius(big, c) ≈ hypot(W, H) / 2 rtol = 1.0e-7
    @test farthest_radius(big, c) <= hypot(W, H) / 2
    # a circle the sensor cuts: the farthest point is where the rim meets the sensor's edge
    cut = sensor ∩ Circle(Point2(c...), 200.0)
    @test farthest_radius(cut, c) ≈ 200.0 rtol = 1.0e-9
    cut2 = sensor ∩ Circle(Point2(c...), 280.0)       # past the half width (260), short of the half diagonal (313)
    @test farthest_radius(cut2, c) ≈ 280.0 rtol = 1.0e-9
    # along a direction: the exit radius, in units of the direction's length
    @test farthest_radius(sensor, c, SVector(1.0, 0.0)) ≈ W / 2 rtol = 1.0e-12
    @test farthest_radius(sensor, c, SVector(2.0, 0.0)) ≈ W / 4 rtol = 1.0e-12
    @test farthest_radius(sensor ∩ circle, c, SVector(0.0, -1.0)) ≈ 150.0 rtol = 1.0e-12
    # the refusals: an unbounded region, a centre outside the region, a zero direction
    @test_throws ArgumentError farthest_radius(WholePlane(), c)
    @test_throws ArgumentError farthest_radius(off, SVector(5.0, 5.0))
    @test farthest_radius(sensor, SVector(-1.0, 5.0)) == hypot(W + 1.0, H - 5.0)   # a Rect's farthest corner from any centre
    @test_throws ArgumentError farthest_radius(sensor, c, SVector(0.0, 0.0))
    # an imaged region: the sensor ∩ the lens coverage ∩ the valid domain
    cam = BoundedCamera(CameraModel(416.0px, SA[260.0px, 174.0px]; lens = KannalaBrandt(-0.02)), Size2(W, H);
                        lens_coverage = Circle(Point2(260.0, 174.0), 150.0))
    @test farthest_radius(imaged_region(cam), c) ≈ 150.0 rtol = 1.0e-9
end
