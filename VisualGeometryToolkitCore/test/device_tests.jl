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

# An integer image (UInt8, UInt16, N0f8, N0f16, a view of one) goes up as it is and becomes the target float type on the
# device, through the one conversion kernel on every backend (the CPU's included): bit-identical to the host's `T.(A)`,
# and on a GPU no host-side converted copy (the host allocates no 4 B/px array).
@testitem "ka_upload: an integer image converted on the device, bit-identical to the host's conversion (CUDA availability-gated)" begin
    using Test
    import KernelAbstractions as KA
    using FixedPointNumbers: N0f8, N0f16
    using VisualGeometryToolkitCore: ka_upload, on_device
    import VisualGeometryToolkitCore as VgtkCore
    @test isdefined(VgtkCore, :_convert_elements!)                       # the one conversion kernel
    raw8 = rand(UInt8, 300, 200); raw16 = rand(UInt16, 300, 200)
    imgs = (raw8, raw16, reinterpret(N0f8, raw8), reinterpret(N0f16, raw16), view(raw8, 11:290, 21:180))
    for A in imgs, T in (Float32, Float64)
        h = ka_upload(T, KA.CPU(), A)
        @test h isa Matrix{T} && h == T.(A) && reinterpret(UInt8, vec(h)) == reinterpret(UInt8, vec(T.(A)))
    end
    @test on_device(Float32, KA.CPU(), raw8) == Float32.(raw8)
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
        for A in imgs, T in (Float32, Float64)
            d = Base.invokelatest(ka_upload, T, gpu, A)
            @test !(d isa Array) && eltype(d) === T
            h = Base.invokelatest(Array, d)
            @test reinterpret(UInt8, vec(h)) == reinterpret(UInt8, vec(T.(A)))      # bit for bit
        end
        big = rand(UInt8, 2000, 3000)
        Base.invokelatest(ka_upload, Float32, gpu, big)                            # compiled
        # the host moves the 1 B/px image only: no host-side Float32 copy (24 MB) is allocated
        @test (@allocated Base.invokelatest(ka_upload, Float32, gpu, big)) < sizeof(big)
    end
end

# The accelerator's memory and devices: the host's free memory, the CUDA extension's query for its backend, and a backend
# no extension answers for warned once and left unchunked (typemax).
@testitem "device_free_bytes, reclaim_device!, device_names: the accelerator extension's queries" begin
    import KernelAbstractions as KA
    using VisualGeometryToolkitCore: device_free_bytes, reclaim_device!, device_names
    struct UnknownGPU <: KA.GPU end
    @test device_free_bytes(KA.CPU()) > 0
    @test (@test_logs (:warn, r"no free-memory query") device_free_bytes(UnknownGPU())) == typemax(Int)
    @test reclaim_device!(KA.CPU()) === nothing && reclaim_device!(UnknownGPU()) === nothing
    @test device_names() isa Vector{String}
    using CUDA: CUDA
    if CUDA.functional()
        @test 0 < device_free_bytes(CUDA.CUDABackend()) < typemax(Int)
        @test reclaim_device!(CUDA.CUDABackend()) === nothing
        @test length(device_names()) == length(CUDA.devices())
    end
end

@testitem "device_zeros: a zero array on a backend (host)" begin
    # KernelAbstractions' own `zeros` is not public, so a package cannot call it
    # across the boundary; PanoramaStitching, Core's cgls and Features' spread
    # each wrote `fill!(KA.allocate(dev, T, dims...), zero(T))`. One copy, here.
    using Test
    using VisualGeometryToolkitCore: device_zeros
    import KernelAbstractions as KA
    z = device_zeros(Float32, KA.CPU(), 3, 2)
    @test z isa Matrix{Float32} && size(z) == (3, 2) && all(iszero, z)
    @test device_zeros(UInt64, KA.CPU(), 0) == UInt64[]
    @test device_zeros(Float64, KA.CPU(), (2, 2)) == zeros(2, 2)
end

# The LM row fills are one kernel on every backend (the owner's one-kernel policy: the CPU's plain loop is retired):
# the rows equal the per-row function run in a plain loop, to rounding (the kernel's compiled row may round an ulp apart).
@testitem "LM rows: one kernel on the CPU, the per-row function to rounding" begin
    using Test, StaticArrays, Random
    import VisualGeometryToolkitCore as VgtkCore
    @test !isdefined(VgtkCore, Symbol("_foreach" * "_row!"))
    rng = Xoshiro(3)
    F = VgtkCore.FundMat(SMatrix{3, 3, Float64}(randn(rng, 9)...))
    for n in (1, 100, 1000)
        xs = VgtkCore.csponds([SVector(rand(rng), rand(rng)) for _ in 1:n], [SVector(rand(rng), rand(rng)) for _ in 1:n])
        r = VgtkCore.sampson_residuals!(zeros(n), F, xs)
        ref = zeros(n)
        cols = VgtkCore._measurement_columns(xs); kw = VgtkCore.constraint_keywords(F, xs)
        for i in 1:n
            VgtkCore._sampson_residual_row!(ref, i, F, cols, kw, nothing, nothing)
        end
        @test r ≈ ref rtol = 1.0e-14
    end
    @test VgtkCore.sampson_residuals!(Float64[], F, VgtkCore.csponds(SVector{2, Float64}[], SVector{2, Float64}[])) == Float64[]
end
