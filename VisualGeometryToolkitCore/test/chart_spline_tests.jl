@testitem "chart_spline" begin
    import VisualGeometryToolkitCore as VgtkCore
    using Test
    using VisualGeometryToolkitCore
    using VisualGeometryToolkitCore: ChartSpline, SplineProjection, CentralGenericModel, KannalaBrandt, Lens, Equidistant,
        Ellipse, Rect2, spline_value, px, WholePlane
    using CoordinateTransformations: LinearMap, transform_deriv
    using StaticArrays
    using LinearAlgebra: I, norm
    using Random: MersenneTwister
    using FiniteDiff: FiniteDiff

    # =============================================================================
    # ChartSpline, SplineProjection, and the grid model's batched backprojection
    # =============================================================================
    #
    # The spline is the grid models' lattice as a primitive (`spline_value`), its
    # Jacobian the differentiated cubic pieces; a SplineProjection is that spline
    # on the equidistant direction chart, `nothing` outside its region; the
    # batched kernels equal their scalar forms bit for bit on the CPU backend.

    rng = MersenneTwister(11)
    area = Rect2(-1.0, -0.5, 2.0, 1.0)
    controls = [SVector(randn(rng), randn(rng)) for _ in 1:7, _ in 1:9]     # 7 rows (second coordinate) × 9 columns
    cs = ChartSpline(area, controls)
    points = [SVector(2.4 * rand(rng) - 1.2, 1.2 * rand(rng) - 0.6) for _ in 1:300]   # inside and past the area

    @testset "the value is the lattice's spline_value; the Jacobian its derivative" begin
        @test all(c -> cs(c) == spline_value(controls, area, c), points)
        for c in points[1:40]
            J = transform_deriv(cs, c)
            Jfd = FiniteDiff.finite_difference_jacobian(x -> Vector(cs(SVector{2}(x))), Vector(c))
            @test J ≈ Jfd rtol = 1.0e-6 atol = 1.0e-8
        end
        @test_throws ArgumentError ChartSpline(area, controls[1:3, :])
        @test_throws ArgumentError ChartSpline(Rect2(0.0, 0.0, 0.0, 1.0), controls)
    end

    @testset "the batch kernel equals the scalar evaluation (CPU backend)" begin
        @test cs(points) == cs.(points)
        @test isempty(cs(SVector{2, Float64}[]))
    end

    @testset "SplineProjection: ray → chart → pixel; nothing outside its region" begin
        chart = Lens(Equidistant()) ∘ LinearMap(SMatrix{3, 3}(1.0I))
        pix = [SVector(500.0 + 300 * c[1], 500.0 + 300 * c[2]) for c in [SVector(x, y) for y in range(-1.5, 1.5; length = 7), x in range(-1.5, 1.5; length = 9)]]
        lattice = ChartSpline(Rect2(-1.2, -1.2, 2.4, 2.4), reshape(pix, 7, 9))
        region = Rect(0, 0, 1000, 1000) ∩ Ellipse(SVector(500.0, 500.0), 400.0, 400.0, 0.0)
        sp = SplineProjection(chart, lattice, region)
        X = SVector(0.2, -0.1, 1.0)
        @test sp(X) == lattice(chart(X))
        @test project(sp, X) == sp(X)
        @test project(sp, SVector(1.0, 0.0, 0.05)) === nothing        # 87° off axis: its pixel lies outside the disc
        @test sp(SVector(1.0, 0.0, 0.05)) isa SVector{2}                # the map itself continues
        # a ray far outside the fitted chart area is not imaged even where its
        # extrapolated pixel lands in the region (the whole plane here); within one
        # knot cell of the area (0.4 rad) it still is
        wide = SplineProjection(chart, lattice, WholePlane())
        far = SVector(sind(170.0), 0.0, cosd(170.0))                     # 170° off axis, the chart at 2.97 rad
        @test wide(far) isa SVector{2} && project(wide, far) === nothing
        edge = SVector(sin(1.5), 0.0, cos(1.5))                           # 1.5 rad: 0.3 past the area, inside its cell
        @test project(wide, edge) == wide(edge)
        @test project(wide, SVector(sin(1.7), 0.0, cos(1.7))) === nothing   # 0.5 rad past: beyond one cell
        J = transform_deriv(sp, X)
        Jfd = FiniteDiff.finite_difference_jacobian(x -> Vector(sp(SVector{3}(x))), Vector(X))
        @test J ≈ Jfd rtol = 1.0e-6
    end

    @testset "batched backproject of a grid model equals the scalar one, bit for bit" begin
        model = CameraModel(300.0px, SA[500.0px, 500.0px]; lens = KannalaBrandt(-0.02))
        grid = CentralGenericModel(model, (1000, 1000); grid_size = (40, 40))
        pix = [SVector(1010 * rand(rng) - 5, 1010 * rand(rng) - 5) for _ in 1:2000]
        push!(pix, SVector(NaN, 3.0), SVector(0.0, 0.0), SVector(999.999, 999.999), SVector(1000.0, 10.0))
        rays = backproject(grid, pix)
        scalar = [backproject(grid, p) for p in pix]
        @test eltype(rays) === SVector{3, Float64} && eltype(scalar) === SVector{3, Float64}
        @test all(i -> isequal(rays[i], scalar[i]), eachindex(pix))
        @test count(r -> all(isnan, r), rays) == count(r -> all(isnan, r), scalar) > 0
    end

    @testset "GPU (availability-gated)" begin
        cuda = try
            m = Base.require(Main, :CUDA)
            (m !== nothing && Base.invokelatest(getproperty(m, :functional))) ? m : nothing
        catch
            nothing
        end
        if cuda === nothing
            # CUDA is in the test target, so this skip means no functional GPU here —
            # recorded as Broken, not as a success, so it shows in the summary
            @test_skip cuda !== nothing
        else
            CuArray = getproperty(cuda, :CuArray)
            model = CameraModel(300.0px, SA[500.0px, 500.0px]; lens = KannalaBrandt(-0.02))
            grid = CentralGenericModel(model, (1000, 1000); grid_size = (40, 40))
            pix = [SVector(1000 * rand(rng), 1000 * rand(rng)) for _ in 1:4096]
            dev = Base.invokelatest(Array, Base.invokelatest(backproject, grid, Base.invokelatest(CuArray, pix)))
            host = backproject(grid, pix)
            @test maximum(i -> norm(host[i] - dev[i]), eachindex(pix)) < 1.0e-12
            dvals = Base.invokelatest(Array, Base.invokelatest(cs, Base.invokelatest(CuArray, points)))
            # FMA contraction on the device: a few ulp of the value
            @test maximum(i -> norm(dvals[i] - cs(points[i])) / max(1, norm(cs(points[i]))), eachindex(points)) < 1.0e-11
        end
    end
end

@testitem "chart_spline: a SplineProjection's image_warp on the device" begin
    using Test
    using VisualGeometryToolkitCore
    using VisualGeometryToolkitCore: ChartSpline, SplineProjection, Lens, Equidistant, Backprojection, Equirectangular, Rect2,
        WholePlane, BoundedMap, Ellipse
    using CoordinateTransformations: LinearMap, AffineMap
    using GeometryBasics: GeometryBasics
    using Adapt: adapt
    using StaticArrays
    using LinearAlgebra: I
    using Random: MersenneTwister

    # A grid lens's forward map in a warp: the BoundedMap's chain is adapted link by
    # link (the spline's control grid to the device), and the spline's device form
    # skips the validating constructor, which cannot compile inside the kernel.
    chart = Lens(Equidistant()) ∘ LinearMap(SMatrix{3, 3}(1.0I))
    pix = [SVector(500.0 + 300 * c[1], 500.0 + 300 * c[2]) for c in [SVector(x, y) for y in range(-1.5, 1.5; length = 7), x in range(-1.5, 1.5; length = 9)]]
    sp = SplineProjection(chart, ChartSpline(Rect2(-1.2, -1.2, 2.4, 2.4), reshape(pix, 7, 9)), WholePlane())
    to_angles = AffineMap(SMatrix{2, 2}(0.02, 0.0, 0.0, 0.02), SVector(-0.64, -0.64))     # output pixels → (λ, φ), ±0.64 rad
    src = rand(MersenneTwister(5), 1000, 1000)
    host = image_warp!(zeros(64, 64), image_warp(Float64, sp ∘ Backprojection(Equirectangular()) ∘ to_angles, Rect(0, 0, 64, 64)), src)
    @test count(!iszero, host) > 4000                                                  # the warp samples the source
    # the device form keeps an isbits region and leaves one holding arrays behind (WholePlane)
    disc = Ellipse(SVector(500.0, 500.0), 400.0, 400.0, 0.0)
    @test adapt(Array, SplineProjection(chart, sp.spline, disc)).region == disc
    poly = GeometryBasics.Polygon([GeometryBasics.Point2(0.0, 0.0), GeometryBasics.Point2(1.0, 0.0), GeometryBasics.Point2(1.0, 1.0)])
    @test adapt(Array, SplineProjection(chart, sp.spline, poly)).region isa WholePlane

    cuda = try
        m = Base.require(Main, :CUDA)
        (m !== nothing && Base.invokelatest(getproperty(m, :functional))) ? m : nothing
    catch
        nothing
    end
    if cuda === nothing
        # CUDA is in the test target, so this skip means no functional GPU here —
        # recorded as Broken, not as a success, so it shows in the summary
        @test_skip cuda !== nothing
    else
        CuArray = getproperty(cuda, :CuArray)
        dsp = Base.invokelatest(adapt, Base.invokelatest(getproperty(cuda, :CUDABackend)), sp)
        @test !(dsp.spline.controls isa Matrix)                                          # the controls on the device
        bm = image_warp(Float64, dsp ∘ Backprojection(Equirectangular()) ∘ to_angles, Rect(0, 0, 64, 64))
        @test bm isa BoundedMap
        dst = Base.invokelatest(CuArray, zeros(64, 64))
        dev = Base.invokelatest(Array, Base.invokelatest(image_warp!, dst, bm, Base.invokelatest(CuArray, src)))
        @test maximum(abs, dev - host) < 1.0e-12                                        # FMA contraction on the device at most
    end
end
