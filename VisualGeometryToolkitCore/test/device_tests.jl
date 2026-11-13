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

# A same-memory wrapper of a dense host array — a reinterpret (BlobBoards' N0f8 view of its Gray{N0f8} image, a UInt8 buffer
# as N0f8), a reshape of one — IS the dense memory: it goes up as it is, with no host copy, then through the one conversion
# kernel. Bit-identical to the host's `T.(A)`; the host allocates nothing proportional to the image (two sizes compared).
@testitem "ka_upload: a reinterpreted or reshaped dense image goes up as its memory, no host copy (CUDA availability-gated)" begin
    using Test
    import KernelAbstractions as KA
    using FixedPointNumbers: N0f8
    using Colors: Gray
    using VisualGeometryToolkitCore: ka_upload, on_device
    import VisualGeometryToolkitCore as VgtkCore
    views(raw) = (reinterpret(N0f8, Gray{N0f8}.(reinterpret(N0f8, raw))), reinterpret(N0f8, raw),
        reshape(reinterpret(N0f8, vec(raw)), size(raw)), reinterpret(reshape, N0f8, Gray{N0f8}.(reinterpret(N0f8, raw))))
    small, large = rand(UInt8, 200, 300), rand(UInt8, 800, 1200)
    for W in views(small), T in (Float32, Float64)
        h = ka_upload(T, KA.CPU(), W)
        @test h isa Matrix{T} && reinterpret(UInt8, vec(h)) == reinterpret(UInt8, vec(T.(W)))
    end
    # the host: the converted array is the only allocation that grows with the image (the 1 B/px source is read in place)
    host_extra(W) = (ka_upload(Float32, KA.CPU(), W); (@allocated ka_upload(Float32, KA.CPU(), W)) - sizeof(Float32) * length(W))
    for (Ws, Wl) in zip(views(small), views(large))
        @test host_extra(Wl) - host_extra(Ws) < length(Wl) - length(Ws)
    end
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
        for W in views(small), T in (Float32, Float64)
            d = Base.invokelatest(ka_upload, T, gpu, W)
            @test !(d isa Array) && eltype(d) === T
            @test reinterpret(UInt8, vec(Base.invokelatest(Array, d))) == reinterpret(UInt8, vec(T.(W)))   # bit for bit
            # the wrapper's own eltype goes up as it is too (the narrow source of the conversion above)
            @test Base.invokelatest(Array, Base.invokelatest(on_device, N0f8, gpu, W)) == W
        end
        # an empty image: nothing to launch
        @test size(Base.invokelatest(ka_upload, Float32, gpu, reinterpret(N0f8, UInt8[]))) == (0,)
        # the device: the host allocates the same at 0.06 and 0.96 Mpx (before: an N0f8 host copy, 1 B/px)
        dev_allocs(W) = (Base.invokelatest(ka_upload, Float32, gpu, W); @allocated Base.invokelatest(ka_upload, Float32, gpu, W))
        for (Ws, Wl) in zip(views(small), views(large))
            @test dev_allocs(Wl) - dev_allocs(Ws) < (length(Wl) - length(Ws)) ÷ 16
        end
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

@testitem "device_resident_threads: a kernel's threads resident at once, register-aware (CUDA availability-gated)" begin
    import KernelAbstractions as KA
    using KernelAbstractions: @kernel, @index
    using StaticArrays: SMatrix
    using VisualGeometryToolkitCore: device_resident_threads, OnBackend
    @kernel function light!(out)
        i = @index(Global)
        @inbounds out[i] = 2 * out[i]
    end
    # a 12 × 12 Float64 product held in registers per thread: 288 values live, past the 255-register cap, so its
    # occupancy is a fraction of a light kernel's
    @kernel function heavy!(out, @Const(x))
        i = @index(Global)
        @inbounds begin
            A = SMatrix{12, 12, Float64}(ntuple(k -> x[i] * k, Val(144)))
            B = A * A' * A
            out[i] = sum(B)
        end
    end
    # the host: the threads its kernels run on
    v = zeros(64)
    @test device_resident_threads(OnBackend(KA.CPU())(light!), v; ndrange = 64) == Threads.nthreads()
    struct UnknownGPU <: KA.GPU end
    @test_throws r"no occupancy query" device_resident_threads(light!(UnknownGPU()), v; ndrange = 64)
    using CUDA: CUDA
    if CUDA.functional()
        dev = CUDA.CUDABackend(; always_inline = true)
        n = 1 << 20
        out = CUDA.zeros(Float64, n); x = CUDA.rand(Float64, n)
        sms = CUDA.attribute(CUDA.device(), CUDA.DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT)
        raw = CUDA.attribute(CUDA.device(), CUDA.DEVICE_ATTRIBUTE_MAX_THREADS_PER_MULTIPROCESSOR) * sms
        nl = device_resident_threads(OnBackend(dev)(light!), out; ndrange = n)
        nh = device_resident_threads(OnBackend(dev)(heavy!), out, x; ndrange = n)
        @test 0 < nl <= raw && 0 < nh
        @test nh < raw ÷ 2 && nh < nl      # registers bound the heavy kernel's occupancy, far below the raw maximum
        # a static workgroup: whole blocks of it, at most the raw maximum
        ns = device_resident_threads(OnBackend(dev; workgroupsize = 128)(light!), out; ndrange = n)
        @test ns % 128 == 0 && 0 < ns <= raw
        # it launches as queried
        OnBackend(dev)(heavy!)(out, x; ndrange = n); KA.synchronize(dev)
        @test all(isfinite, Array(out))
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
        xs = VgtkCore.correspondences([SVector(rand(rng), rand(rng)) for _ in 1:n], [SVector(rand(rng), rand(rng)) for _ in 1:n])
        r = VgtkCore.sampson_residuals!(zeros(n), F, xs)
        ref = zeros(n)
        cols = VgtkCore._observation_columns(xs); kw = VgtkCore.constraint_keywords(F, xs)
        for i in 1:n
            VgtkCore._sampson_residual_row!(ref, i, F, cols, kw, nothing, nothing)
        end
        @test r ≈ ref rtol = 1.0e-14
    end
    @test VgtkCore.sampson_residuals!(Float64[], F, VgtkCore.correspondences(SVector{2, Float64}[], SVector{2, Float64}[])) == Float64[]
end

# The eager release of a device buffer (base/device.jl, hoisted from RANSACScoringDoneRight): GPUArrays' unsafe_free!, so a
# CuArray's memory is released at once (KernelAbstractions' unsafe_free! is not extended by CUDA.jl and freed nothing);
# a host array is left alone.
@testitem "free_device_buffers!: a device buffer released at once, a host array untouched (CUDA availability-gated)" begin
    using Test
    import KernelAbstractions as KA
    using VisualGeometryToolkitCore: free_device_buffers!, ka_upload
    h = rand(Float32, 64)
    c = copy(h)
    @test free_device_buffers!(h, zeros(3)) === nothing && h == c                 # host: nothing happens, still readable
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
        d = Base.invokelatest(ka_upload, Float32, gpu, rand(Float32, 1 << 20))
        r = reshape(d, 1024, 1024)                               # derived: shares d's memory
        @test !d.data.freed
        Base.invokelatest(free_device_buffers!, d)
        @test d.data.freed                                       # this reference released at once
        @test_throws ArgumentError Base.invokelatest(sum, d)    # a freed buffer is never read again
        Base.invokelatest(free_device_buffers!, r)
        @test r.data.freed
        # another buffer allocates and is released as the first was
        e = Base.invokelatest(ka_upload, Float32, gpu, rand(Float32, 1 << 20))
        @test !e.data.freed
        Base.invokelatest(free_device_buffers!, e)
    end
end

# The CUDA extension's default backend inlines every kernel call (`always_inline = true`), and an array already on the device
# is that backend's memory: `on_device` hands it back as it is, never a copy (the backend VALUES differ: `get_backend` of a
# CuArray is `CUDABackend()`).
@testitem "default_device: the GPU backend with always_inline, the same memory as its arrays (CUDA availability-gated)" begin
    using Test
    import KernelAbstractions as KA
    using VisualGeometryToolkitCore: default_device, reset_default_device!, on_device
    using CUDA: CUDA
    a = rand(Float32, 8)
    @test on_device(KA.CPU(), a) === a && on_device(KA.CPU(; static = true), a) === a
    if CUDA.functional()
        reset_default_device!()
        dev = default_device()
        @test dev isa CUDA.CUDABackend && dev.always_inline
        d = on_device(dev, a)
        @test d isa CUDA.CuArray && KA.get_backend(d) != dev                    # the array's backend value is the default one
        @test on_device(dev, d) === d && on_device(KA.get_backend(d), d) === d   # the same memory: no copy either way
        @test Array(on_device(KA.CPU(), d)) == a
    end
end

# One launch value per backend type: a kernel on the same data compiles once, one way, whether the caller holds
# `default_device()` or `get_backend` of its array (which never reports compiler options).
@testitem "OnBackend: one launch value per backend type (CUDA availability-gated)" begin
    using Test
    using StaticArrays: SVector
    import KernelAbstractions as KA
    import VisualGeometryToolkitCore as VgtkCore
    using VisualGeometryToolkitCore: OnBackend, default_device, pointwise!
    @test OnBackend(KA.CPU()).backend === KA.CPU() && OnBackend(KA.CPU(), (1, 1)).workgroupsize === (1, 1)
    using CUDA: CUDA
    if CUDA.functional()
        VgtkCore.reset_default_device!()          # resolved afresh with the CUDA extension loaded (an earlier item may have cached the CPU)
        d = default_device()
        @test d === CUDA.CUDABackend(; always_inline = true)
        x = CUDA.CuArray([SVector(0.1i, -0.2i) for i in 1:100])
        @test OnBackend(KA.get_backend(x)).backend === d
        @test OnBackend(CUDA.CUDABackend(; prefer_blocks = true); workgroupsize = 1).backend ===
              CUDA.CUDABackend(; prefer_blocks = true, always_inline = true)
        f(p) = SVector(sin(p[1]) * p[2], p[1] / (1 + p[2]^2))
        a = similar(x); b = similar(x)
        pointwise!(f, a, x); pointwise!(f, b, x; exec = OnBackend(d))
        KA.synchronize(d)
        @test isequal(Array(a), Array(b))
    end
end
