# =============================================================================
# A lens over a batch of points: pointwise! broadcasts the per-point maps — the
# same function on the host (serial or split across threads) and on a GPU —
# and every execution policy gives identical results.
# =============================================================================

@testitem "pointwise" begin
    using Test
    using VisualGeometryToolkitCore
    using VisualGeometryToolkitCore: _precision, Serial, Threaded, Auto, THREADED_MIN_LENGTH
    using StaticArrays: SVector
    using LinearAlgebra: norm
    using Random: Xoshiro

    BC = Lens(Perspective(), Additive(Radial((-0.28, 0.07, -0.01)), Tangential(1.0e-3, -5.0e-4)))
    KB = KannalaBrandt(-0.02, 0.005, -1.0e-3, 2.0e-4)
    TPF = Lens(Equidistant(), Additive(Radial((-0.01, 0.005, -1.0e-3, 2.0e-4)), Tangential(1.0e-3, -5.0e-4), ThinPrism((2.0e-4,), (-1.0e-4,))))
    lenses = (Pinhole(), BC, KB, TPF, Division(-0.31, 0.024), Scaramuzza(-0.17, -0.006, 0.035), Lens(DoubleSphere(-0.2, 0.59)),
        Lens(Equidistant()), Lens(Perspective(), Tangential(1.0e-3, -5.0e-4)))

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

    @testset "every policy is the per-point map, bitwise" begin
        rng = Xoshiro(3)
        n = 2 * THREADED_MIN_LENGTH + 17              # several chunks, a ragged tail
        rays = [(θ = 1.6 * rand(rng); ψ = 2π * rand(rng); SVector(sin(θ) * cos(ψ), sin(θ) * sin(ψ), cos(θ))) for _ in 1:n]
        pts = [SVector(0.9 * randn(rng), 0.9 * randn(rng)) for _ in 1:n]
        push!(pts, SVector(50.0, 50.0)); push!(rays, SVector(0.0, 0.0, -1.0))   # not imaged by most lenses
        for lens in lenses
            for (f!, f, xs, T) in ((project!, project, rays, SVector{2, Float64}), (backproject!, backproject, pts, SVector{3, Float64}),
                    (distort_normalized!, distort_normalized, pts, SVector{2, Float64}), (undistort_normalized!, undistort_normalized, pts, SVector{2, Float64}))
                ref = [f(lens, x) for x in xs]
                for exec in (Serial(), Threaded(), Auto())
                    out = Vector{T}(undef, length(xs))
                    @test f!(out, lens, xs; exec) === out
                    @test isequal(out, ref)
                end
            end
        end
        # a camera is a per-point map too: K ∘ lens, through the same driver
        cam = CameraModel(; focal = 800.0, pp = (640.0, 480.0), lens = KB)
        u = Vector{SVector{2, Float64}}(undef, length(rays))
        project!(u, cam, rays; exec = Threaded())
        @test isequal(u, [project(cam, X) for X in rays])
        # pointwise! itself: any per-point function, several inputs
        a, b = randn(rng, 100), randn(rng, 100)
        @test pointwise!(hypot, similar(a), a, b; exec = Threaded()) == hypot.(a, b)
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
        end
    end
end
