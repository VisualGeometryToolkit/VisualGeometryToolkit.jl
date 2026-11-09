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
        @test lit_region(BoundedCamera(m, Size2(10, 10); region = Circle(Point2(5, 5), 4))) isa Circle{Float64}
        b = with_lit_region(BoundedCamera(m, Size2(10, 10)), Circle(Point2(5.0, 5.0), 4.0))
        @test lit_region(b) == Circle(Point2(5.0, 5.0), 4.0) && extent(b) == Size2(10, 10)
    end
end
