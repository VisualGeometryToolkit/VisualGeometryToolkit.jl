# Supports on the sensor (cameras/support.jl): which of a model's parameters a
# camera can see. A global model is all free; a lattice's node is free when its
# footprint meets the imaged region `Rect(extent) ∩ image_circle`.
@testitem "support" begin
    using Test
    using VisualGeometryToolkitCore
    using VisualGeometryToolkitCore: KannalaBrandt, Pinhole, WholePlane, RegionIntersection, SupportTrait,
        GlobalSupport, LocalSupport, support_trait, nodes, support, node_position, cubic_bspline_footprint, CONTRACT_GENERICS

    cam(f, l) = CameraModel(; focal = f, pp = (640.0, 480.0), lens = l)
    fold = KannalaBrandt(-4 / (3π^2))                      # a 180° fisheye, r(90°) = π/3

    @testset "imaged_region: the extent ∩ the image circle" begin
        bc = BoundedCamera(cam(250.0, fold), Size2(1280, 960))
        R = imaged_region(bc)
        @test R isa RegionIntersection
        @test R.a == Rect(0, 0, 1280, 960) && R.b == image_circle(bc)
        @test Point2(640.0, 480.0) ∈ R && Point2(5.0, 5.0) ∉ R
        @test imaged_region(BoundedCamera(cam(800.0, Pinhole()), Size2(1280, 960))) === Rect(0, 0, 1280, 960)
    end

    @testset "the trait and its contract" begin
        @test support_trait(fold) === GlobalSupport() && support_trait(typeof(fold)) === GlobalSupport()
        @test support_trait(cam(800.0, Pinhole())) === GlobalSupport()
        @test all(f -> any(g -> g === f, CONTRACT_GENERICS), (support_trait, nodes, support, node_position))
        @test node_position(fold, 1) === nothing
        # a global model: every parameter is free, one support covering everything
        R = imaged_region(BoundedCamera(cam(250.0, fold), Size2(1280, 960)))
        @test free_nodes(fold, R) === Colon()
        @test node_mask(fold, R) == trues() && ndims(node_mask(fold, R)) == 0
        # the uniform cubic B-spline footprint: center ± 2 cells
        @test cubic_bspline_footprint((100.0, 50.0), (10.0, 5.0)) == Rect2(80.0, 40.0, 40.0, 20.0)
    end
end
