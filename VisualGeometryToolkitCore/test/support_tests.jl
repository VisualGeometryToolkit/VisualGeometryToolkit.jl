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
        @test support_trait(CentralGenericModel) === LocalSupport()
        @test support_trait(NoncentralGenericModel) === LocalSupport()
        @test all(f -> any(g -> g === f, CONTRACT_GENERICS), (support_trait, nodes, support, node_position))
        @test node_position(fold, 1) === nothing
        # a global model: every parameter is free, one support covering everything
        R = imaged_region(BoundedCamera(cam(250.0, fold), Size2(1280, 960)))
        @test free_nodes(fold, R) === Colon()
        @test node_mask(fold, R) == trues() && ndims(node_mask(fold, R)) == 0
        # the uniform cubic B-spline footprint: center ± 2 cells
        @test cubic_bspline_footprint((100.0, 50.0), (10.0, 5.0)) == Rect2(80.0, 40.0, 40.0, 20.0)
    end

    @testset "a circular fisheye's grid model: nodes frozen where the footprint misses the circle" begin
        g = CentralGenericModel(cam(250.0, fold), (1280, 960); grid_size = (20, 16))
        R = imaged_region(BoundedCamera(cam(250.0, fold), Size2(1280, 960)))
        @test nodes(g) == CartesianIndices((16, 20))
        m = node_mask(g, R)
        @test size(m) == size(g.grid) == (16, 20)
        @test sum(m) == 108                                    # of 320
        @test free_nodes(g, R) == [i for i in nodes(g) if m[i]]
        # frozen exactly where the footprint misses the circle: brute-force sampling of each footprint
        q = image_circle(BoundedCamera(cam(250.0, fold), Size2(1280, 960)))
        for i in nodes(g)
            s = support(g, i)
            pts = [Point2(x, y) for x in range(minimum(s)[1], maximum(s)[1]; length = 31), y in range(minimum(s)[2], maximum(s)[2]; length = 31)]
            m[i] || @test !any(p -> p ∈ q, pts)
            m[i] && @test any(p -> p ∈ q, pts)
        end
        # the node sits at the centre of its footprint (interior nodes)
        i = CartesianIndex(8, 10)
        @test node_position(g, i) ≈ Point2((minimum(support(g, i)) .+ maximum(support(g, i))) ./ 2...)
        # the noncentral model has the same lattice
        gn = NoncentralGenericModel(cam(250.0, fold), (1280, 960); grid_size = (20, 16))
        @test node_mask(gn, R) == m
        # a rectilinear lens images the whole extent: no node is frozen
        gr = CentralGenericModel(cam(800.0, Pinhole()), (1280, 960); grid_size = (20, 16))
        @test all(node_mask(gr, imaged_region(BoundedCamera(cam(800.0, Pinhole()), Size2(1280, 960)))))
    end

    @testset "the sampler clamps nodes inside the lit field" begin
        # every node of the circular fisheye's grid backprojects finitely, outside the circle too
        g = CentralGenericModel(cam(250.0, fold), (1280, 960); grid_size = (20, 16))
        @test all(v -> all(isfinite, v), g.grid)
        # the margin is a keyword: at 1 the corner nodes sample at the fold itself
        g1 = CentralGenericModel(cam(250.0, fold), (1280, 960); grid_size = (20, 16), sampling_margin = 0.9)
        @test g1.grid != g.grid
    end
end
