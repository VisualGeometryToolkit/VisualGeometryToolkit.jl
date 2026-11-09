# A bounded camera's bound and its lit region (cameras.jl): the bound is the pixel extent, or the physical Sensor whose
# resolution IS that extent; the lit region is where light reaches the sensor — a Rect2, a Circle or an Ellipse, by default
# the sensor's own Rect, never "none". Both are type parameters, so every bounded camera is a concrete, isbits value whose
# accessors infer, whichever bound and region it carries.
@testitem "BoundedCamera: bound and lit region" begin
    using Test
    using VisualGeometryToolkitCore
    using VisualGeometryToolkitCore: model, with_model, convert_eltype, lit_region, with_lit_region, WholePlane
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
        bc = BoundedCamera(m, bound; region)
        @testset "bound $(nameof(typeof(bound))), region $(nameof(typeof(region)))" begin
            @test isconcretetype(typeof(bc))
            @test isbits(bc)
            @test (@inferred extent(bc)) == Size2(W, H)
            @test (@inferred sensor(bc)) == (bound isa Sensor ? bound : nothing)
            @test (@inferred lit_region(bc)) == expected(region)
            @test (@inferred backproject(bc, u)) == backproject(m, u)
            @test (@inferred project(bc, backproject(m, u))) ≈ u
            # every rebuild keeps the bound and the region
            for rb in (convert_eltype(Float32, bc), with_model(bc, m))
                @test lit_region(rb) == lit_region(bc) && extent(rb) == extent(bc) && sensor(rb) == sensor(bc)
            end
            posed = BoundedCamera(PosedCamera(m, E), bound; region)
            reposed = _with_extrinsics(posed, inv(E))
            @test lit_region(reposed) == lit_region(posed) && sensor(reposed) == sensor(posed)
        end
    end

    @testset "the region is a geometric primitive in Float64" begin
        @test_throws ArgumentError BoundedCamera(m, Size2(10, 10); region = WholePlane())
        @test lit_region(BoundedCamera(m, Size2(10, 10); region = Circle(Point2(5, 5), 4))) === Circle(Point2(5.0, 5.0), 4.0)   # kept a Circle, in Float64
        b = with_lit_region(BoundedCamera(m, Size2(10, 10)), Circle(Point2(5.0, 5.0), 4.0))
        @test lit_region(b) == Circle(Point2(5.0, 5.0), 4.0) && extent(b) == Size2(10, 10)
    end
end

# What a bounded camera images (image_circle.jl): the sensor ∩ its lit region ∩ where its lens model is valid — one
# function of the camera alone, so no consumer carries an image circle of its own.
@testitem "imaged_region: sensor ∩ lit region ∩ valid domain" begin
    using Test
    using VisualGeometryToolkitCore
    using VisualGeometryToolkitCore: lit_region, with_lit_region, relabel, CameraFrame, WorldFrame, RotationMap, RegionIntersection
    using VisualGeometryToolkitCore.Cameras: _calibrated_rect
    using StaticArrays
    using Rotations: RotY

    W, H = 4256, 2848
    m = CameraModel(1754.7px, SA[2133.6px, 1422.5px]; lens = KannalaBrandt(0.2468, 0.1505, -0.0214, 0.1308))

    @testset "a lit disc cut by the sensor: clipped to the sensor" begin
        bc = BoundedCamera(m, Size2(W, H); region = Circle(Point2(2140.8, 1375.8), 1742.2))   # the 14 mm's measured disc
        # the one small Union kept on purpose (owner, 2026-11-09): valid_domain(::CameraModel) is an Ellipse or WholePlane
        # by the lens's coefficients, so imaged_region is one of two concrete RegionIntersections — pinned here, no wider
        r = @inferred RegionIntersection imaged_region(bc)
        @test isconcretetype(typeof(r)) && isbits(r)
        @test Base.return_types(imaged_region, (typeof(bc),)) |> only |> (T -> T isa Union && length(Base.uniontypes(T)) == 2)
        b = Rect2(r)
        @test all(minimum(b) .>= 0) && maximum(b)[1] <= W && maximum(b)[2] <= H
        @test Point2(2140.8, 1375.8) ∈ r && Point2(10.0, 10.0) ∉ r
    end

    @testset "a grid: lit but outside where the grid is defined is not imaged" begin
        dom = Ellipse(Point2(2133.9, 1423.7), 1611.8, 1607.8, 1.5777)
        g = CentralGenericModel(m, (W, H); grid_size = (35, 25), init = :quasi_interpolate, domain = dom)
        bc = BoundedCamera(g, Size2(W, H); region = Circle(Point2(2140.8, 1375.8), 1742.2))
        @test Point2(2133.9 + 1650.0, 1423.7) ∉ imaged_region(bc)       # lit, past the grid's domain
        @test Point2(2133.9 + 1550.0, 1423.7) ∈ imaged_region(bc)
        for p in (Point2(2133.9, 1423.7), Point2(2133.9 + 1650.0, 1423.7), Point2(5.0, 5.0))
            @test (p ∈ valid_domain(g)) == (p ∈ _calibrated_rect(g) && p ∈ dom)
        end
    end

    @testset "a lens with no limit, the whole sensor lit: every pixel centre" begin
        pin = CameraModel(800.0px, SA[50.0px, 40.0px])
        r = imaged_region(BoundedCamera(pin, Size2(100, 80)))
        @test all(Point2(u - 0.5, v - 0.5) ∈ r for u in 1:100, v in 1:80)
    end

    @testset "view_overlap: each frame's own lit region, of different types" begin
        fish = CameraModel(; focal = 250.0, pp = (640.0, 480.0), lens = KannalaBrandt(-4 / (3π^2)))
        frame(R, region) = BoundedCamera(PosedCamera(fish, relabel(CameraFrame, WorldFrame, RotationMap(SMatrix{3, 3, Float64, 9}(R)))),
                                         Size2(1280, 960); region)
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
