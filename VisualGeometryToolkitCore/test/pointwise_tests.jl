# =============================================================================
# A lens over a batch of points: pointwise! runs the per-point maps as ONE
# KernelAbstractions kernel on every backend (the CPU's included; the owner's
# one-kernel policy retired the host's Serial/Threaded/Auto broadcast paths),
# bit for bit the per-point map.
# =============================================================================

@testitem "pointwise" begin
    using Test
    using VisualGeometryToolkitCore
    using VisualGeometryToolkitCore: _precision, OnBackend
    using KernelAbstractions: KernelAbstractions
    struct VecAffine
        w::Vector{Float64}
    end
    (a::VecAffine)(x) = a.w[1] * x + a.w[2]
    # type parameters in another order than the fields; one parameter (T) no field's, equal to a field's type
    struct Swapped{B, A}
        a::A
        b::B
    end
    (s::Swapped)(i) = s.a[i] + s.b
    struct Tagged{T, V}
        v::V
    end
    (t::Tagged)(i) = 2 * t.v[i]
    using StaticArrays: SVector
    using LinearAlgebra: norm
    using Random: Xoshiro

    BC = Lens(Perspective(), Additive(Radial((-0.28, 0.07, -0.01)), Tangential(1.0e-3, -5.0e-4)))
    KB = KannalaBrandt(-0.02, 0.005, -1.0e-3, 2.0e-4)
    TPF = Lens(Equidistant(), Additive(Radial((-0.01, 0.005, -1.0e-3, 2.0e-4)), Tangential(1.0e-3, -5.0e-4), ThinPrism((2.0e-4,), (-1.0e-4,))))
    # OpenCV's full model: rational radial, tangential and thin prism, then the sensor tilt
    OCV = Lens(Perspective(), (Tilt(0.01, -0.02) ∘ Additive(Radial((-0.12, 0.03, 0.004), (0.02, -0.001, 0.0003)), Tangential(1.0e-3, -5.0e-4),
        ThinPrism((-5.5e-3, 7.0e-4), (2.2e-3, -1.2e-3)))))
    lenses = (Pinhole(), BC, KB, TPF, Division(-0.31, 0.024), Scaramuzza(-0.17, -0.006, 0.035), Lens(DoubleSphere(-0.2, 0.59)),
        Lens(Equidistant()), Lens(Perspective(), Tangential(1.0e-3, -5.0e-4)), OCV)

    @testset "perspective views: round trip at both precisions" begin
        function worst_roundtrip(lens, ::Type{T}) where {T}
            lens_T = _precision(T, lens)
            worst = zero(T)
            for u in range(T(-0.5), T(0.5); length = 21), v in range(T(-0.5), T(0.5); length = 21)
                x = SVector{2, T}(u, v)
                fx = lens_T(x)
                xu = undistort_normalized(lens_T, SVector{2, T}(fx[1], fx[2]))
                @test eltype(xu) === T
                worst = max(worst, abs(xu[1] - x[1]), abs(xu[2] - x[2]))
            end
            return worst
        end
        for lens in (BC, KB, TPF)
            @test worst_roundtrip(lens, Float64) <= 1.0e-9
            @test worst_roundtrip(lens, Float32) <= 1.0f-6
        end
        # NaN where the view does not exist, never a throw
        @test all(isnan, undistort_normalized(BC, SVector(50.0, 50.0)))
        eq = Lens(Equidistant())
        b = backproject(eq, SVector(1.7, 0.0))
        @test all(isfinite, b) && b[3] < 0
        @test all(isnan, undistort_normalized(eq, SVector(1.7, 0.0)))
    end

    @testset "the one kernel is the per-point map, bitwise" begin
        # the host's broadcast policies are retired: one launch path
        for name in (:Serial, :Threaded, :Auto, Symbol("Execution" * "Policy"), Symbol("THREADED" * "_MIN_LENGTH"))
            @test !isdefined(VisualGeometryToolkitCore, name)
        end
        cpu = KernelAbstractions.CPU()
        rng = Xoshiro(3)
        n = 2065                                      # several workgroups, a ragged tail
        rays = [(θ = 1.6 * rand(rng); ψ = 2π * rand(rng); SVector(sin(θ) * cos(ψ), sin(θ) * sin(ψ), cos(θ))) for _ in 1:n]
        pts = [SVector(0.9 * randn(rng), 0.9 * randn(rng)) for _ in 1:n]
        push!(pts, SVector(50.0, 50.0)); push!(rays, SVector(0.0, 0.0, -1.0))   # not imaged by most lenses
        for lens in lenses
            for (f!, f, xs, T) in ((project!, project, rays, SVector{2, Float64}), (backproject!, backproject, pts, SVector{3, Float64}),
                    (distort_normalized!, distort_normalized, pts, SVector{2, Float64}), (undistort_normalized!, undistort_normalized, pts, SVector{2, Float64}))
                ref = [f(lens, x) for x in xs]
                out = Vector{T}(undef, length(xs))
                @test f!(out, lens, xs) === out
                @test isequal(out, ref)
                for exec in (OnBackend(cpu), OnBackend(cpu; workgroupsize = 64))
                    out = Vector{T}(undef, length(xs))
                    @test f!(out, lens, xs; exec) === out
                    @test isequal(out, ref)
                end
            end
        end
        # a camera is a per-point map too: K ∘ lens, through the same driver
        cam = CameraModel(; focal = 800.0, pp = (640.0, 480.0), lens = KB)
        u = Vector{SVector{2, Float64}}(undef, length(rays))
        project!(u, cam, rays)
        @test isequal(u, [project(cam, X) for X in rays])
        # pointwise! itself: any per-point function, several inputs
        a, b = randn(rng, 100), randn(rng, 100)
        @test pointwise!(hypot, similar(a), a, b) == hypot.(a, b)
        @test_throws DimensionMismatch pointwise!(hypot, zeros(3), zeros(3), zeros(1))         # no broadcasting
        # one kernel on a given backend value (here KernelAbstractions' CPU), with and without a workgroup size, any shape
        for exec in (OnBackend(cpu), OnBackend(cpu; workgroupsize = 16))
            out = similar(rays, SVector{2, Float64})
            pointwise!(Base.Fix1(project, KB), out, rays; exec); KernelAbstractions.synchronize(cpu)
            @test isequal(out, [project(KB, X) for X in rays])
            A, B = randn(rng, 7, 5), randn(rng, 7, 5)
            O = pointwise!(hypot, similar(A), A, B; exec); KernelAbstractions.synchronize(cpu)
            @test O == hypot.(A, B)
            # a functor goes as its fields where its type is rebuilt from them, else as one argument: the same values
            scaled = VecAffine([2.0, -1.0])                            # its field's type, a Vector, is no type parameter: one argument
            @test VisualGeometryToolkitCore._splattable(typeof(Base.Fix1(project, KB))) && !VisualGeometryToolkitCore._splattable(typeof(scaled))
            O = pointwise!(scaled, similar(A), A; exec); KernelAbstractions.synchronize(cpu)
            @test O == scaled.(A)
            t = randn(rng, 9); idx = collect(1:9)
            for g in (Swapped(t, 0.5), Tagged{typeof(t), typeof(t)}(t))
                @test VisualGeometryToolkitCore._splattable(typeof(g))
                o = pointwise!(g, similar(t), idx; exec); KernelAbstractions.synchronize(cpu)
                @test o == g.(idx)
            end
        end
        @test pointwise!(hypot, Float64[], Float64[], Float64[]; exec = OnBackend(cpu)) == Float64[]
        @test_throws DimensionMismatch pointwise!(hypot, zeros(3), zeros(4), zeros(4); exec = OnBackend(cpu))
        @test_throws DimensionMismatch pointwise!(hypot, zeros(3), zeros(3), zeros(1); exec = OnBackend(cpu))   # no broadcasting
    end

    @testset "a Float32 batch stays Float32: the lens converts once" begin
        xs = [SVector{2, Float32}(0.01f0 * i, -0.007f0 * i) for i in 1:50]
        out = similar(xs)
        undistort_normalized!(out, BC, xs)
        @test eltype(out) === SVector{2, Float32}
        @test out == [undistort_normalized(_precision(Float32, BC), x) for x in xs]
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
            rays = [SVector{3, Float32}(0.01f0 * i, -0.02f0 * i, 1.0f0) for i in 1:256]
            for lens in lenses
                host = similar(rays, SVector{2, Float32})
                project!(host, lens, rays)
                dev = Base.invokelatest(CuArray, similar(host))
                Base.invokelatest(project!, dev, lens, Base.invokelatest(CuArray, rays))
                got = Base.invokelatest(Array, dev)
                @test all(i -> (all(isnan, got[i]) && all(isnan, host[i])) || norm(got[i] - host[i]) <= 1.0f-5 * max(1, norm(host[i])), eachindex(host))
                back = Base.invokelatest(CuArray, similar(rays))
                Base.invokelatest(backproject!, back, lens, Base.invokelatest(CuArray, host))
                hb = similar(rays); backproject!(hb, lens, host)
                gb = Base.invokelatest(Array, back)
                @test all(i -> (all(isnan, gb[i]) && all(isnan, hb[i])) || norm(gb[i] - hb[i]) <= 1.0f-4, eachindex(hb))
            end
            # OnBackend: one kernel on the caller's backend value (always_inline, a fixed workgroup) gives the broadcast's values
            cbe = Base.invokelatest(getproperty(cuda, :CUDABackend); always_inline = true)
            drays = Base.invokelatest(CuArray, rays)
            bcast = Base.invokelatest(CuArray, similar(rays, SVector{2, Float32}))
            Base.invokelatest(project!, bcast, KB, drays)
            for exec in (OnBackend(cbe), OnBackend(cbe; workgroupsize = 128))
                dev = Base.invokelatest(CuArray, similar(rays, SVector{2, Float32}))
                Base.invokelatest(project!, dev, KB, drays; exec)
                Base.invokelatest(KernelAbstractions.synchronize, cbe)
                @test isequal(Base.invokelatest(Array, dev), Base.invokelatest(Array, bcast))
            end
            # functors holding a device array, launched as their fields: the array arrives as the kernel's form of it, the type
            # parameters bound in their declared order (a parameter no field declares, here equal to the array's type, kept)
            tab = Float32.(1:64) ./ 7; dtab = Base.invokelatest(CuArray, tab); didx = Base.invokelatest(CuArray, collect(1:64))
            for (g, h) in ((Swapped(dtab, 0.5f0), Swapped(tab, 0.5f0)), (Tagged{typeof(dtab), typeof(dtab)}(dtab), Tagged{Nothing, typeof(tab)}(tab)))
                @test VisualGeometryToolkitCore._splattable(typeof(g))
                o = Base.invokelatest(CuArray, zeros(Float32, 64))
                Base.invokelatest(pointwise!, g, o, didx; exec = OnBackend(cbe))
                Base.invokelatest(KernelAbstractions.synchronize, cbe)
                @test Base.invokelatest(Array, o) == h.(1:64)
            end
            # a strong OpenCV chain out to its branch's edge, where the inverse
            # continues from the axis (the device runs the same fallback)
            strong = _precision(Float32, Lens(Perspective(), (Tilt(-0.00992587, -0.0062454) ∘ Additive(Radial((-0.150994, -0.0501823, 0.023916), (0.187097, 0.0255086, -0.00128661)),
                Tangential(0.00244469, 0.00575199), ThinPrism((-0.00157781, -0.000969799), (-0.000762879, 0.00210571))))))
            pts = vec([SVector{2, Float32}(u, v) for u in range(-4.0f0, 4.0f0; length = 33), v in range(-4.0f0, 4.0f0; length = 33)])
            hb = similar(pts, SVector{3, Float32}); backproject!(hb, strong, pts)
            back = Base.invokelatest(CuArray, hb)
            Base.invokelatest(backproject!, back, strong, Base.invokelatest(CuArray, pts))
            gb = Base.invokelatest(Array, back)
            @test count(b -> all(isfinite, b), hb) > 0 && count(b -> all(isnan, b), hb) > 0
            @test all(i -> (all(isnan, gb[i]) && all(isnan, hb[i])) || norm(gb[i] - hb[i]) <= 1.0f-4, eachindex(hb))
        end
    end
end
