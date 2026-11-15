# The valid domain polygon on a device (Core 0.10.10): `Adapt.adapt` moves a decentred lens's `PolygonRegion` (its vertex
# ring) and an imaged region holding it to the device, and membership evaluated inside a KernelAbstractions kernel answers
# as the host's `∈` on a grid of points. A frame region that reaches a device kernel (a mask, a sampling predicate) is
# such a region for every lens that is not radially symmetric.

@testitem "valid_domain polygon: adapted to the device, membership in a kernel matches the host" begin
    using Test
    using VisualGeometryToolkitCore
    using VisualGeometryToolkitCore: PolygonRegion, RegionIntersection
    using StaticArrays
    using Adapt: adapt
    using KernelAbstractions
    @kernel function inside!(ok, @Const(ps), region)
        i = @index(Global)
        ok[i] = ps[i] ∈ region
    end
    # an OpenCV-style lens (radial, tangential, thin prism): the domain is the star polygon through each azimuth's fold
    ocv = Lens(Perspective(), Additive(Radial((-0.35, 0.03)), Tangential(-4.0e-3, 6.0e-3), ThinPrism((3.0e-3, 1.0e-3), (-2.0e-3, 1.0e-3))))
    W, H = 1280, 960
    cam = CameraModel(CameraCalibrationMat(@SMatrix([800.0 0.0 640.0; 0.0 800.0 480.0; 0.0 0.0 1.0])); lens = ocv)
    q = valid_domain(cam)
    r = imaged_region(BoundedCamera(cam, Size2(W, H)))
    @test q isa PolygonRegion && r isa RegionIntersection
    # a grid over the sensor and past it: the polygon's annulus (between its inscribed and circumscribed discs) included
    ps = vec([SVector(x, y) for x in -200.25:7.5:(W + 200), y in -200.25:7.5:(H + 200)])
    host(region) = [p ∈ region for p in ps]
    @test any(host(q)) && !all(host(q)) && any(p -> q.inner < hypot((p - q.centre)...) < q.outer, ps)
    @test adapt(Array, q) == q && adapt(Array, r) == r            # the host form is the region itself
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
        for region in (q, r)
            d = Base.invokelatest(adapt, CuArray, region)
            dq = d isa RegionIntersection ? d.b : d
            @test dq isa PolygonRegion && !(dq.vertices isa Array)    # the vertex ring is on the device
            @test Base.invokelatest(Array, dq.vertices) == q.vertices
            ok = Base.invokelatest(CuArray, falses(length(ps)))
            Base.invokelatest(inside!(gpu, 64), ok, Base.invokelatest(CuArray, ps), d; ndrange = length(ps))
            Base.invokelatest(KernelAbstractions.synchronize, gpu)
            @test Base.invokelatest(Array, ok) == host(region)
        end
    end
end
