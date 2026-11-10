@testitem "device" begin
    using Test
    using VisualGeometryToolkitCore: default_device, reset_default_device!, _DEVICE_RESOLVER
    import KernelAbstractions as KA

    @testset "default_device — the family's one backend resolver" begin
        # The host is the default while no accelerator extension has registered a
        # resolver. Under ReTestItems another item on this worker may have loaded
        # CUDA (pointwise's GPU smoke), and its extension registers the GPU —
        # so the claim holds only when that extension is absent.
        if Base.get_extension(VisualGeometryToolkitCore, :VisualGeometryToolkitCoreCUDAExt) === nothing
            reset_default_device!()
            @test default_device() === KA.CPU()
        end
        # Memoized: the resolver runs once per session.
        calls = Ref(0)
        old = _DEVICE_RESOLVER[]
        try
            _DEVICE_RESOLVER[] = () -> (calls[] += 1; KA.CPU())
            reset_default_device!()
            default_device(); default_device()
            @test calls[] == 1
            # An accelerator extension registers by ASSIGNMENT; a reset re-resolves.
            _DEVICE_RESOLVER[] = () -> (calls[] += 1; KA.CPU())
            reset_default_device!()
            default_device()
            @test calls[] == 2
        finally
            _DEVICE_RESOLVER[] = old
            reset_default_device!()
        end
    end
end

@testitem "ka_upload and on_device: dense, a view, an adjoint, a reinterpret (host; CUDA availability-gated)" begin
    using Test, StaticArrays
    import KernelAbstractions as KA
    using VisualGeometryToolkitCore: ka_upload, on_device, CameraModel, KannalaBrandt, px, undistort
    # The one upload takes any host array: a dense one of the target eltype as it is,
    # anything else materialized first (a device `copyto!` from the host takes a dense
    # array only). `on_device` keeps an array already on the device.
    A = reshape(collect(Float32, 1:48), 6, 8)
    wrapped = (A, view(A, 2:5, 3:7), A', reinterpret(Float32, reinterpret(UInt32, A)), Float64.(A))
    for W in wrapped
        h = ka_upload(Float32, KA.CPU(), W)
        @test h isa Matrix{Float32} && h == Float32.(W)
        @test h !== W                                          # always a copy
    end
    @test on_device(KA.CPU(), A) === A
    @test on_device(KA.CPU(), view(A, 1:2, 1:2)) isa SubArray  # already on the host: kept
    cam = CameraModel(300.0px, SA[60.0px, 50.0px]; lens = KannalaBrandt(0.02))
    img = rand(120, 140)
    host_ref = undistort(cam, img[11:110, 21:140]; device = KA.CPU())
    @test undistort(cam, view(img, 11:110, 21:140); device = KA.CPU()) == host_ref
    cuda = try
        mod = Base.require(Main, :CUDA)
        (mod !== nothing && Base.invokelatest(getproperty(mod, :functional))) ? mod : nothing
    catch
        nothing
    end
    if cuda === nothing
        @test_skip cuda !== nothing      # no functional GPU here: recorded as Broken, not as a success
    else
        gpu = Base.invokelatest(getproperty(cuda, :CUDABackend))
        for W in wrapped
            d = Base.invokelatest(ka_upload, Float32, gpu, W)
            @test !(d isa Array)
            @test Base.invokelatest(Array, d) == Float32.(W)
        end
        dA = Base.invokelatest(ka_upload, Float32, gpu, A)
        @test Base.invokelatest(on_device, gpu, dA) === dA
        # a host view and an adjoint through undistort on the GPU: an `Array` back, within 1e-4
        for v in (view(img, 11:110, 21:140), collect(img[11:110, 21:140]')')
            out = Base.invokelatest(undistort, cam, v; device = gpu)
            @test out isa Array
            @test maximum(abs.(out .- host_ref)) <= 1.0e-4
        end
    end
end
