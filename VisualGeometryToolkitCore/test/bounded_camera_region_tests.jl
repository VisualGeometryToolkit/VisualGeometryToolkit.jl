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
