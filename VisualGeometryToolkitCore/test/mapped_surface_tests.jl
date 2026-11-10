# =============================================================================
# The board's deformation on the device, and the surface of its chart map:
# the thin-plate kinds' device form (Adapt, convert_eltype), the static
# pure-bending coefficients, fit_height_field, MappedSurface (the surface of
# pose ∘ S₃ ∘ lift ∘ C ∘ S₂⁻¹, geometry/cameras/mapped_surface.jl) and the
# deformation's freedom from the pose's gauge.
# =============================================================================

@testitem "board deformation: the thin-plate kinds on the device" begin
    using Test, StaticArrays, Random
    using Adapt: adapt
    using KernelAbstractions
    using KernelAbstractions: @kernel, @index, @Const
    using VisualGeometryToolkitCore
    using VisualGeometryToolkitCore: grid_nodes, convert_eltype, transform_deriv, surface_point
    using Rotations: RotMatrix

    @kernel function field_points!(y, J, @Const(ps), c)
        i = @index(Global)
        y[i] = c(ps[i])
        J[i] = transform_deriv(c, ps[i])
    end
    rng = Xoshiro(21)
    nodes = grid_nodes(5; lo = -0.5, hi = 0.5)
    H = HeightField(nodes, 0.01 .* randn(rng, 25))
    T = ThinPlateSpline(nodes, 0.01 .* randn(rng, 25), 0.01 .* randn(rng, 25))
    ps = [SVector(rand(rng) - 0.5, rand(rng) - 0.5) for _ in 1:200]
    backend = KernelAbstractions.CPU()
    for (c, N) in ((H, 3), (T, 2))
        d = adapt(backend, c)
        @test d == c && d.system.nodes === c.system.nodes
        for F in (Float64, Float32)
            cF = convert_eltype(F, d)
            @test eltype(cF.system.coeff) == F && eltype(parameter_vector(cF)) == F
            psF = SVector{2, F}.(ps)
            y = zeros(SVector{N, F}, length(ps)); J = zeros(SMatrix{N, 2, F, 2N}, length(ps))
            field_points!(backend, 64)(y, J, psF, cF; ndrange = length(ps))
            KernelAbstractions.synchronize(backend)
            host = [c(p) for p in ps]
            tol = F === Float64 ? 1.0e-12 : 1.0e-5
            @test all(isapprox(Float64.(a), b; rtol = tol, atol = tol * 1.0e-2) for (a, b) in zip(y, host))
            @test all(isapprox(Float64.(a), transform_deriv(c, p); rtol = tol, atol = tol) for (a, p) in zip(J, ps))
        end
    end

    cuda = try
        m = Base.require(Main, :CUDA)
        (m !== nothing && Base.invokelatest(getproperty(m, :functional))) ? m : nothing
    catch
        nothing
    end
    if cuda === nothing
        @test_skip cuda !== nothing   # CUDA is in the test target: no functional GPU here
    else
        CuArray = getproperty(cuda, :CuArray)
        gpu = Base.invokelatest(getproperty(cuda, :CUDABackend))
        # On a device a kind is applied through the chart map (`MappedSurface`, its member `_apply`s the kind): the
        # call `c(p)` carries the one-form check, whose DimensionError cannot be compiled for a device.
        @kernel function field_on_device!(y, J, @Const(ps), s)
            i = @index(Global)
            y[i] = surface_point(s, ps[i])
            J[i] = transform_deriv(s.map, ps[i])
        end
        I3 = SMatrix{3, 3, Float64}(1, 0, 0, 0, 1, 0, 0, 0, 1)
        for c in (H, T), F in (Float64, Float32)
            cF = convert_eltype(F, c)
            s = MappedSurface(EuclideanMap(RotMatrix{3, F}(I3), zero(SVector{3, F})), cF, nothing, Rect2{F}(-0.5, -0.5, 1.0, 1.0))
            d = Base.invokelatest(adapt, gpu, s)
            @test !(Base.invokelatest(adapt, gpu, cF).system.coeff isa Matrix)   # the matrices on the device
            psF = SVector{2, F}.(ps)
            y = Base.invokelatest(CuArray, zeros(SVector{3, F}, length(ps)))
            J = Base.invokelatest(CuArray, zeros(SMatrix{3, 2, F, 6}, length(ps)))
            Base.invokelatest(field_on_device!(gpu, 64), y, J, Base.invokelatest(CuArray, psF), d; ndrange = length(ps))
            Base.invokelatest(KernelAbstractions.synchronize, gpu)
            tol = F === Float64 ? 1.0e-12 : 1.0e-5
            lift(q) = length(q) == 2 ? SVector(q[1], q[2], 0.0) : q
            @test all(isapprox(Float64.(a), lift(c(p)); rtol = tol, atol = tol * 1.0e-2) for (a, p) in zip(Base.invokelatest(Array, y), ps))
        end
    end
end

@testitem "board deformation: the static pure-bending coefficients" begin
    using Test, StaticArrays, Random, LinearAlgebra
    using ForwardDiff
    using Adapt: adapt
    using KernelAbstractions
    using KernelAbstractions: @kernel, @index, @Const
    using VisualGeometryToolkitCore
    using VisualGeometryToolkitCore: grid_nodes, ThinPlateSplineSystem, tps_coefficients, tps_pure_bending, tps_bending_coefficients,
        tps_evaluate, is_pure_bending, convert_eltype

    rng = Xoshiro(22)
    nodes = grid_nodes(5; lo = 0.0, hi = 1.0)
    sys = ThinPlateSplineSystem(nodes)
    M = [ones(25) first.(nodes) last.(nodes)]
    bend = SVector{25}(0.01 .* randn(rng, 25))
    raw = bend + SVector{25}(M * [0.3, -0.2, 0.7])                         # a plane on top: the pose's piston and tilts
    # (a) one implementation: the host kinds hold exactly what the static path gives
    v, w, a = tps_bending_coefficients(sys, raw)
    H = HeightField(nodes, raw)
    @test parameter_vector(H) == v && H.w == w && H.a == a
    @test (v, w, a) == (tps_pure_bending(sys, raw), tps_coefficients(sys, tps_pure_bending(sys, raw))...)
    @test is_pure_bending(sys, v) && norm(a) <= 1.0e-14 * norm(v)
    @test tps_pure_bending(sys, v) === v                                    # pure bending is kept bit for bit
    @test norm(tps_pure_bending(sys, SVector{25}(M * [1.0, 2.0, -3.0]))) <= 1.0e-13   # a plane is all pose: nothing left
    # against the matrix (BLAS) form of the same map: to 1e-12
    wa = sys.coeff * Vector(v)
    @test isapprox(vcat(collect(w), collect(a)), wa; atol = 1.0e-12 * norm(wa))
    p = SVector(0.31, 0.77)
    @test H(p)[3] ≈ tps_evaluate(sys.nodes, SVector{25}(wa[1:25]), SVector{3}(wa[26:28]), p) atol = 1.0e-12

    # (b) a ForwardDiff Dual through it against central differences (the projection's derivative included)
    g(h) = ((vv, ww, aa) = tps_bending_coefficients(sys, h); tps_evaluate(sys.nodes, ww, aa, p))
    G = ForwardDiff.gradient(g, Vector(raw))
    ε = 1.0e-6
    Gfd = [(g(Vector(raw) + ε * (1:25 .== k)) - g(Vector(raw) - ε * (1:25 .== k))) / 2ε for k in 1:25]
    @test G ≈ Gfd atol = 1.0e-9
    @test abs(sum(G .* (M * [1.0, 0.5, -0.25]))) <= 1.0e-12               # a plane direction moves nothing
    # at pure-bending values a Dual passes through as it is: the derivative along the subspace (the nodal
    # weights, the kinds' transform_deriv_params), equal to the projected one on every pure-bending direction
    Gv = ForwardDiff.gradient(g, Vector(v))
    @test Gv ≈ collect(VisualGeometryToolkitCore.nodal_weights(sys, p)) atol = 1.0e-12
    b = tps_pure_bending(sys, SVector{25}(randn(rng, 25)))
    @test dot(Gv, b) ≈ dot(G, b) atol = 1.0e-12

    # (c) allocation-free on the CPU
    allocs(fn, args...) = (fn(args...); @allocated fn(args...))
    @test allocs(tps_bending_coefficients, sys, raw) == 0
    @test allocs(tps_pure_bending, sys, raw) == 0 && allocs(tps_coefficients, sys, raw) == 0
    d = ForwardDiff.Dual.(raw, 1.0)
    @test allocs(tps_bending_coefficients, sys, d) == 0

    # (d) in a KernelAbstractions kernel, the system adapted, in Float64 and Float32
    @kernel function bending_kernel!(out, @Const(values), s)
        i = @index(Global)
        _, ww, _ = tps_bending_coefficients(s, values[i])
        out[i] = ww
    end
    backend = KernelAbstractions.CPU()
    vals = [SVector{25}(0.01 .* randn(rng, 25)) + SVector{25}(M * randn(rng, 3)) for _ in 1:64]
    for F in (Float64, Float32)
        sF = adapt(backend, convert_eltype(F, sys))
        valsF = [SVector{25, F}(x) for x in vals]
        out = zeros(SVector{25, F}, length(vals))
        bending_kernel!(backend, 16)(out, valsF, sF; ndrange = length(vals))
        KernelAbstractions.synchronize(backend)
        @test out == [tps_bending_coefficients(sF, x)[2] for x in valsF]
        host = [tps_bending_coefficients(sys, x)[2] for x in vals]
        # within the summation's rounding in F: 64 eps(F) of the terms' magnitudes Σⱼ |Cᵢⱼ vⱼ| (the projection's too)
        C = abs.(sys.coeff[1:25, :])
        @test all(all(abs.(Float64.(a) - b) .<= 64 * eps(F) .* (C * abs.(x) .+ norm(b))) for (a, b, x) in zip(out, host, vals))
    end
end

@testitem "board deformation: fit_height_field" begin
    using Test, StaticArrays, Random, LinearAlgebra
    using VisualGeometryToolkitCore
    using VisualGeometryToolkitCore: grid_nodes, is_pure_bending, mm, m, has_units

    rng = Xoshiro(23)
    nodes = grid_nodes(4; lo = 0.0, hi = 1.0)
    truth = HeightField(nodes, 0.01 .* randn(rng, 16))                   # pure bending
    pts = [SVector(rand(rng), rand(rng)) for _ in 1:200]
    plane(p) = 0.02 - 0.05p[1] + 0.03p[2]
    z = [truth(p)[3] + plane(p) for p in pts]
    H = fit_height_field(nodes, pts, z)
    @test H isa HeightField{16} && is_pure_bending(H.system, parameter_vector(H))
    @test norm(parameter_vector(H) - parameter_vector(truth)) <= 1.0e-10 * norm(parameter_vector(truth))
    # a planar input: zero bending
    Hp = fit_height_field(nodes, pts, plane.(pts))
    @test norm(parameter_vector(Hp)) <= 1.0e-12
    # smoothing pulls the field towards a plane: less bending energy
    Hs = fit_height_field(nodes, pts, z .+ 1.0e-3 .* randn(rng, length(z)); smoothing = 1.0e-3)
    using VisualGeometryToolkitCore: bending_energy
    @test bending_energy(Hs.system, parameter_vector(Hs)) < bending_energy(truth.system, parameter_vector(truth))
    # lengths: the physical field
    Hm = fit_height_field([n .* m for n in nodes], [p .* m for p in pts], z .* m)
    @test has_units(Hm) && isapprox(parameter_vector(Hm), parameter_vector(H); rtol = 1.0e-9)
    @test_throws ArgumentError fit_height_field(nodes, pts[1:5], z[1:5])
    @test_throws ArgumentError fit_height_field(nodes, pts, z; smoothing = -1)
    @test_throws DimensionMismatch fit_height_field(nodes, pts, z[1:3])
end

@testitem "board deformation: MappedSurface, the surface of the chart map" begin
    using Test, StaticArrays, Random, LinearAlgebra
    using ForwardDiff
    using Rotations: RotMatrix, RotXYZ
    using KernelAbstractions
    using KernelAbstractions: @kernel, @index, @Const
    using Adapt: adapt
    using VisualGeometryToolkitCore
    using VisualGeometryToolkitCore: intersection, chart, bounds, normal, surface, surface_point, grid_nodes, convert_eltype,
        with_parameters, tps_evaluate, tps_radial_basis, ThinPlateSplineSystem, tps_coefficients

    allocs(f, args...) = (f(args...); @allocated f(args...))
    rng = Xoshiro(24)
    nodes = grid_nodes(4; lo = -0.6, hi = 0.6)
    H = HeightField(nodes, 0.01 .* randn(rng, 16))
    T = ThinPlateSpline(nodes, 0.01 .* randn(rng, 16), 0.01 .* randn(rng, 16))
    A = Aspect(0.01) ∘ Skew(-0.003)
    X = EuclideanMap(RotMatrix{3}(RotXYZ(0.1, -0.2, 0.05)), SVector(0.02, -0.01, 2.0))
    rect = Rect2(-0.5, -0.5, 1.0, 1.0)
    rays = [(SVector(0.3randn(rng), 0.3randn(rng), 0.0), SVector(0.2randn(rng), 0.2randn(rng), 1.0)) for _ in 1:400]

    @testset "the RadiometricCamera warped board, placed on its page" begin
        include(joinpath(pkgdir(VisualGeometryToolkitCore), "test", "fixtures", "height_surface", "reference.jl"))
        REF = HEIGHT_SURFACE_REFERENCE
        # its page [0, 1]² of side 2 at (-1, -1): the field on the chart's nodes (pure bending there, kept bit for
        # bit), heights in chart units, the placement scaling them back
        Hc = HeightField(REF.nodes, REF.bending_heights ./ 2)
        P = SimilarityMap(2.0, one(RotMatrix{3, Float64}), SVector(-1.0, -1.0, 0.0))
        s = MappedSurface(EuclideanMap(one(RotMatrix{3, Float64}), SVector(0.0, 0.0, 0.0)), Hc, P, Rect2(-1.0, -1.0, 2.0, 2.0))
        hits = [intersection(s, o, d) for (o, d) in zip(REF.ray_origins, REF.ray_directions)]
        @test length(REF.t) == 100 && all(isfinite, first.(hits))
        # RadiometricCamera's own Newton on its own spline: rounding apart (measured 2.2e-16 in t and x)
        @test maximum(abs.(first.(hits) - REF.t)) <= 64eps()
        @test maximum(maximum(abs, x - r) for (x, r) in zip(last.(hits), REF.x)) <= 64eps()
        @test maximum(maximum(abs, chart(s, x) - (REF.origin[SOneTo(2)] + 2 .* c)) for (x, c) in zip(last.(hits), REF.chart)) <= 1.0e-12
        @test maximum(maximum(abs, normal(s, x) - r) for (x, r) in zip(last.(hits), REF.surface_normal)) <= 1.0e-12
        # misses: outside the rectangle, behind the origin, parallel, a NaN origin
        @test all(isnan, intersection(s, SVector(1.5, 0.0, 1.0), SVector(0.0, 0.0, -1.0))[2])
        @test all(isnan, intersection(s, SVector(0.1, 0.2, 1.0), SVector(0.0, 0.0, 1.0))[2])
        @test all(isnan, intersection(s, SVector(0.1, 0.2, 1.0), SVector(1.0, 0.0, 0.0))[2])
        @test all(isnan, intersection(s, SVector(NaN, 0.2, 1.0), SVector(0.0, 0.0, -1.0))[2])
    end

    @testset "an identity in-plane warp is the height-only surface, bit for bit" begin
        s_h = MappedSurface(X, H, nothing, rect)
        s_id = MappedSurface(X, H ∘ ThinPlateSpline(nodes), nothing, rect)
        generic(s, o, d) = VisualGeometryToolkitCore.Cameras._mapped_intersection(s, o, d)
        for (o, d) in rays
            @test isequal(intersection(s_id, o, d), generic(s_h, o, d))     # the same 2 × 2 root
        end
        @test count(r -> isfinite(intersection(s_h, r...)[1]), rays) > 100
    end

    @testset "a height field over an affine chart: the 1-D fast path is the 2 × 2 root to rounding" begin
        generic(s, o, d) = VisualGeometryToolkitCore.Cameras._mapped_intersection(s, o, d)
        P = SimilarityMap(0.9, one(RotMatrix{3, Float64}), SVector(-0.4, -0.45, 0.0))
        for (c, Pc) in ((H, nothing), (H ∘ A, nothing), (H, P))
            s = MappedSurface(X, c, Pc, rect)
            for (o, d) in rays
                tf, xf = intersection(s, o, d)
                tg, xg = generic(s, o, d)
                @test isnan(tf) == isnan(tg)
                isnan(tf) || @test maximum(abs, xf - xg) <= 8eps()
            end
            o, d = rays[1]
            @test allocs(intersection, s, o, d) == 0
            t32, x32 = @inferred intersection(convert_eltype(Float32, s), Float32.(o), Float32.(d))
            @test t32 isa Float32
        end
    end

    @testset "a pure in-plane warp with zero height: the plane hit at the warped point" begin
        s = MappedSurface(X, T, nothing, rect)
        q = surface(PosedTarget(PlanarTarget([SVector(-0.5, -0.5), SVector(0.5, 0.5)]), X))   # the unwarped plane
        for (o, d) in rays
            t, x = intersection(s, o, d)
            tq, xq = intersection(q, o, d)
            (isnan(t) || isnan(tq)) && continue                              # (the two rectangles' edges differ by the warp)
            @test x ≈ xq atol = 1.0e-14                                      # the same plane
            c = chart(s, x)
            @test T(c) ≈ chart(q, x) atol = 1.0e-13                          # its chart point is the warp's preimage
        end
    end

    @testset "Jacobians against central differences; nested Duals" begin
        c = H ∘ T ∘ A
        s = MappedSurface(X, c, nothing, rect)
        o, d = SVector(0.05, -0.1, 0.0), SVector(0.1, 0.2, 1.0)
        hit(od) = intersection(s, SVector{3}(od[1:3]), SVector{3}(od[4:6]))[2]
        cfd(f, x, ε) = reduce(hcat, [(f(x + ε * (1:length(x) .== k)) - f(x - ε * (1:length(x) .== k))) / 2ε for k in eachindex(x)])
        @test ForwardDiff.jacobian(hit, [o; d]) ≈ cfd(hit, [o; d], 1.0e-6) atol = 1.0e-8
        # along the calibration's free directions (its manifold's basis B: the pure-bending values and the
        # factors' parameters), where with_parameters is the identity on the values
        θ = Vector(parameter_vector(c))
        B = Matrix(VisualGeometryToolkitCore.retract_jacobian(VisualGeometryToolkitCore.minimal_manifold(PlanarTarget(
            [SVector(-0.5, -0.5), SVector(0.5, 0.5)]; calibration = c))))
        hitθ(y) = intersection(MappedSurface(X, with_parameters(c, θ + B * y), nothing, rect), o, d)[2]
        chartθ(y) = (sθ = MappedSurface(X, with_parameters(c, θ + B * y), nothing, rect); chart(sθ, intersection(sθ, o, d)[2]))
        y0 = zeros(size(B, 2))
        @test ForwardDiff.jacobian(hitθ, y0) ≈ cfd(hitθ, y0, 1.0e-7) atol = 1.0e-7
        @test ForwardDiff.jacobian(chartθ, y0) ≈ cfd(chartθ, y0, 1.0e-7) atol = 1.0e-7
        # a Dual pose (the joint refinement's): its frame's values seed the root
        hitX(v) = intersection(MappedSurface(EuclideanMap(RotMatrix{3}(RotXYZ(0.1 + v[1], -0.2, 0.05)), SVector(0.02, -0.01 + v[2], 2.0)),
                                             c, nothing, rect), o, d)[2]
        @test ForwardDiff.jacobian(hitX, [0.0, 0.0]) ≈ cfd(hitX, [0.0, 0.0], 1.0e-6) atol = 1.0e-8
        # nested Duals (a Hessian): each layer by the implicit function theorem, exact to every order
        z(od) = hit(od)[3]
        Hs = ForwardDiff.hessian(z, [o; d])
        @test Hs ≈ cfd(od -> ForwardDiff.gradient(z, od), [o; d], 1.0e-5) atol = 1.0e-6
        # Dual hits are inferred and allocation-free
        od = ForwardDiff.Dual.(o, 1.0), ForwardDiff.Dual.(d, 0.5)
        @test (@inferred intersection(s, od...))[1] isa ForwardDiff.Dual && allocs(intersection, s, od...) == 0
    end

    @testset "Float32, inferred and allocation-free" begin
        s = MappedSurface(X, H ∘ T ∘ A, nothing, rect)
        s32 = convert_eltype(Float32, s)
        for (o, d) in rays[1:50]
            t, x = @inferred intersection(s, o, d)
            t32, x32 = @inferred intersection(s32, Float32.(o), Float32.(d))
            @test t32 isa Float32 && isnan(t32) == isnan(t)
            isnan(t) || @test abs(t32 - t) <= 1.0f-5 * abs(t)
        end
        o, d = rays[1]
        @test allocs(intersection, s, o, d) == 0 && allocs(intersection, s32, Float32.(o), Float32.(d)) == 0
        x = surface_point(s, SVector(0.1, 0.2))
        @test isapprox((@inferred chart(s, x)), SVector(0.1, 0.2); atol = 1.0e-13) && allocs(chart, s, x) == 0
        @test (@inferred normal(s, x)) isa SVector{3, Float64} && allocs(normal, s, x) == 0
        @test bounds(s) == rect
    end

    @testset "the builder: surface(pt) calls it; in a KernelAbstractions kernel" begin
        c = H ∘ T ∘ A
        design = [SVector(x, y) for x in -0.5:0.25:0.5 for y in -0.5:0.25:0.5]
        pt = PosedTarget(PlanarTarget(design; calibration = c), X)
        s = surface(pt)
        @test s == MappedSurface(X, VisualGeometryToolkitCore.Cameras._stored_calibration(c), nothing, bounds(s))
        # rebuilt from new nodal heights inside a kernel (with_parameters on the field, the builder)
        @kernel function rebuild_hits!(t, @Const(heights), field, rest, X, rect, o, d)
            i = @index(Global)
            si = MappedSurface(X, with_parameters(field, heights[i]) ∘ rest, nothing, rect)
            t[i] = intersection(si, o, d)[1]
        end
        o, d = SVector(0.05, -0.1, 0.0), SVector(0.1, 0.2, 1.0)
        hs = [SVector{16}(0.01 .* randn(rng, 16)) for _ in 1:32]
        backend = KernelAbstractions.CPU()
        Hd = adapt(backend, H)
        t = zeros(length(hs))
        rebuild_hits!(backend, 8)(t, hs, Hd, T ∘ A, X, bounds(s), o, d; ndrange = length(hs))
        KernelAbstractions.synchronize(backend)
        @test isequal(t, [intersection(MappedSurface(X, with_parameters(H, h) ∘ (T ∘ A), nothing, bounds(s)), o, d)[1] for h in hs])
    end

    @testset "in a KernelAbstractions kernel, and on CUDA" begin
        @kernel function hits!(t, x, @Const(origins), @Const(directions), s)
            i = @index(Global)
            ti, xi = intersection(s, origins[i], directions[i])
            t[i] = ti
            x[i] = xi
        end
        s = MappedSurface(X, H ∘ T ∘ A, nothing, rect)
        os, ds = first.(rays), last.(rays)
        backend = KernelAbstractions.CPU()
        for F in (Float64, Float32)
            sF = convert_eltype(F, s)
            oF, dF = SVector{3, F}.(os), SVector{3, F}.(ds)
            t = zeros(F, length(os)); x = zeros(SVector{3, F}, length(os))
            hits!(backend, 64)(t, x, oF, dF, sF; ndrange = length(os))
            KernelAbstractions.synchronize(backend)
            ref = [intersection(sF, a, b) for (a, b) in zip(oF, dF)]
            @test isequal(t, first.(ref)) && isequal(x, last.(ref))         # bit for bit, NaN misses included
        end
        cuda = try
            m = Base.require(Main, :CUDA)
            (m !== nothing && Base.invokelatest(getproperty(m, :functional))) ? m : nothing
        catch
            nothing
        end
        if cuda === nothing
            @test_skip cuda !== nothing   # CUDA is in the test target: no functional GPU here
        else
            CuArray = getproperty(cuda, :CuArray)
            gpu = Base.invokelatest(getproperty(cuda, :CUDABackend))
            for F in (Float64, Float32)
                sF = convert_eltype(F, s)
                oF, dF = SVector{3, F}.(os), SVector{3, F}.(ds)
                ref = first.([intersection(sF, a, b) for (a, b) in zip(oF, dF)])
                t = Base.invokelatest(CuArray, zeros(F, length(os))); x = Base.invokelatest(CuArray, zeros(SVector{3, F}, length(os)))
                Base.invokelatest(hits!(gpu, 64), t, x, Base.invokelatest(CuArray, oF), Base.invokelatest(CuArray, dF),
                                  Base.invokelatest(adapt, gpu, sF); ndrange = length(os))
                tg = Base.invokelatest(Array, t)
                @test isequal(isnan.(tg), isnan.(ref))
                @test maximum(abs, filter(isfinite, tg - ref)) <= 4 * eps(F) * 2   # FMA contraction on the device at most
            end
        end
    end
end

@testitem "board deformation: the height fast path continues past four iterations, or misses" begin
    using Test, StaticArrays, LinearAlgebra
    using VisualGeometryToolkitCore
    using VisualGeometryToolkitCore: intersection, surface_point, ThinPlateSplineSystem, tps_coefficients, tps_evaluate,
        tps_gradient, grid_nodes

    # a 10 cm bump on a 1 m quad (slope ≈ 0.4), a ray at 10° elevation aimed at its flank
    nodes = SVector{25}(grid_nodes(5; lo = 0.0, hi = 1.0))
    h = SVector{25}([i == 13 ? 0.1 : 0.0 for i in 1:25])
    w, a = tps_coefficients(ThinPlateSplineSystem(nodes), Vector(h))
    using Rotations: RotMatrix
    s = MappedSurface(EuclideanMap(one(RotMatrix{3, Float64}), SVector(0.0, 0.0, 0.0)), HeightField(collect(nodes), h; project = false),
                      nothing, Rect2(0.0, 0.0, 1.0, 1.0))
    θ = deg2rad(10)
    d = SVector(cos(θ), 0.0, -sin(θ))
    on_surface(x) = abs(x[3] - tps_evaluate(nodes, w, a, x[SOneTo(2)]))
    # the four iterations alone (RadiometricCamera's intersect_board)
    function four(o, d)
        t = -o[3] / d[3]
        for _ in 1:4
            x = o + t * d
            m = x[SOneTo(2)]
            g = t * d[3] + o[3] - tps_evaluate(nodes, w, a, m)
            gp = d[3] - dot(tps_gradient(nodes, w, a, m), d[SOneTo(2)])
            t -= g / gp
        end
        return o + t * d
    end
    # aimed at the flank at u = 0.4: four iterations stop 0.4 μm off the surface, the continuation lands on it
    P = surface_point(s, SVector(0.4, 0.5))
    o = P - 0.5 * d
    @test on_surface(four(o, d)) > 1.0e-7
    t, x = intersection(s, o, d)
    @test isfinite(t) && on_surface(x) < 1.0e-12 && norm(x - (o + t * d)) < 1.0e-15
    # aimed at u = 0.35 behind the crest: four iterations report a point 7.6 cm off the surface
    # (23 cm from the aimed point); the continuation does not converge, and the ray is a miss
    P = surface_point(s, SVector(0.35, 0.5))
    o = P - 0.5 * d
    @test on_surface(four(o, d)) > 0.05
    t, x = intersection(s, o, d)
    @test isnan(t) && all(isnan, x)
    # the grazing limit: a miss or a point on the surface, never one off it
    rs = [begin
        φ = deg2rad(elevation)
        dg = SVector(cos(φ), 0.0, -sin(φ))
        intersection(s, surface_point(s, SVector(u, 0.5)) - 0.5 * dg, dg)
    end for elevation in (10, 5, 2, 1), u in 0.1:0.1:0.9]
    @test all(r -> isnan(r[1]) || on_surface(r[2]) < 1.0e-12, rs)
end


@testitem "board deformation: no gauge freedom with respect to the pose" begin
    # Every deformation parameter changes the posed board in a way no pose
    # change reproduces: the deformation's and the pose's tangent spans
    # intersect trivially. Not Euclidean orthogonality in point space — a
    # pure-bending spline is the OBLIQUE complement of the plane fields (its
    # values orthogonal to the interpolant's affine rows), and a bump correlates
    # with a piston at the sampled points — so the test asserts (1) full column
    # rank of the stacked point Jacobian [∂X/∂deformation | ∂X/∂pose] with a
    # condition-number bound and the smallest principal angle between the spans
    # bounded away from zero, (2) the pose's nodal fields annihilated by the
    # pure-bending projection, (3) the manifold's basis orthogonal to the affine
    # rows, (4) a pose-reproducible member refused without what fixes it.
    using Test, StaticArrays, Random, LinearAlgebra
    using Rotations: RotMatrix, RotXYZ
    using VisualGeometryToolkitCore
    using VisualGeometryToolkitCore: grid_nodes, minimal_manifold, action_jacobian, retract_jacobian, tps_pure_bending

    rng = Xoshiro(25)
    nodes = grid_nodes(4; lo = -0.6, hi = 0.6)
    H = HeightField(nodes, 0.01 .* randn(rng, 16))
    T = ThinPlateSpline(nodes, 0.01 .* randn(rng, 16), 0.01 .* randn(rng, 16))
    design = [SVector(x, y) for x in range(-0.5, 0.5, length = 7) for y in range(-0.4, 0.4, length = 6)]
    sk(v) = [0 -v[3] v[2]; v[3] 0 -v[1]; -v[2] v[1] 0]
    lift3(q) = length(q) == 2 ? SVector(q[1], q[2], 0.0) : q
    poses = [i == 1 ? Matrix(1.0I, 3, 3) : Matrix(RotMatrix{3}(RotXYZ(0.3randn(rng), 0.3randn(rng), 0.3randn(rng)))) for i in 1:3]
    function stats(cal)
        t = PlanarTarget(design; calibration = cal)
        ch = minimal_manifold(t)
        pts = points(t)
        D = [let a = Matrix(action_jacobian(ch, p)); size(a, 1) == 2 ? vcat(a, zeros(1, size(a, 2))) : a end for p in design]
        m, n, F = size(D[1], 2), length(design), length(poses)
        J = zeros(3n * F, m + 6F)
        for f in 1:F, k in 1:n
            r = (3n * (f - 1) + 3k - 2):(3n * (f - 1) + 3k)
            J[r, 1:m] = poses[f] * D[k]
            J[r, (m + 6f - 5):(m + 6f)] = hcat(-sk(poses[f] * lift3(pts[k])), Matrix(1.0I, 3, 3))
        end
        J1 = J[1:3n, 1:(m + 6)]                                             # one pose: the identity
        Qd = Matrix(qr(J1[:, 1:m]).Q)[:, 1:m]
        Qp = Matrix(qr(J1[:, (m + 1):end]).Q)[:, 1:6]
        return (m = m, cond = cond(J), cond1 = cond(J1), angle = acos(min(1.0, opnorm(Qd' * Qp))))
    end
    # measured 2026-11-10 (cond over three poses, cond at the identity pose alone, smallest principal angle in rad):
    #   height only        40.4  54.6  0.095
    #   in-plane only      35.6  54.6  0.095
    #   Aspect∘Skew∘TPS    44.1  62.6  0.086
    #   full chain         44.0  62.4  0.086
    #   Skew alone          3.7   5.2  0.88   (the Iwasawa N: its antisymmetric part is no gauge of the in-plane rotation)
    # pinned at twice the conditions and half the angles
    for (name, cal, κ, κ1, α) in (("height only", H, 81, 110, 0.047), ("in-plane only", T, 72, 110, 0.047),
                                  ("Aspect∘Skew∘TPS", T ∘ Aspect(0.01) ∘ Skew(-0.003), 89, 126, 0.043),
                                  ("full chain", H ∘ T ∘ Aspect(0.01) ∘ Skew(-0.003), 88, 125, 0.043),
                                  ("Skew alone", Skew(-0.003), 7.4, 10.4, 0.44))
        r = stats(cal)
        @testset "$name" begin
            @test r.cond <= κ && r.cond1 <= κ1 && r.angle >= α
        end
    end

    # (2) the pose's fields at the nodes are all affine: the pure-bending projection leaves nothing of them
    sys = H.system
    M = [ones(16) first.(nodes) last.(nodes)]
    for α in ([1.0, 0, 0], [0, 1.0, 0], [0, 0, 1.0], randn(rng, 3))         # piston, the tilts (heights); translations and the rotation (in-plane)
        @test norm(tps_pure_bending(sys, M * α)) <= 1.0e-13
    end
    ω = 0.01
    @test norm(tps_pure_bending(sys, -ω .* last.(nodes) .+ 0.003)) <= 1.0e-13 && norm(tps_pure_bending(sys, ω .* first.(nodes) .- 0.002)) <= 1.0e-13

    # (3) the manifold's basis is orthogonal to the affine rows: K⁻¹-orthogonal to the pose's nodal fields
    B = Matrix(retract_jacobian(minimal_manifold(PlanarTarget(design; calibration = H ∘ T))))
    Aff = sys.affine
    @test opnorm(cat(Aff, Aff, Aff; dims = (1, 2)) * B) <= 1.0e-13

    # (4) the pose's own members are refused without what fixes them
    @test_throws ArgumentError minimal_manifold(PlanarTarget(design; calibration = H ∘ T ∘ Scale(0.001) ∘ Aspect(0.01)))
    @test_throws ArgumentError minimal_manifold(PlanarTarget(design; calibration = H ∘ T ∘ EuclideanMap(RotMatrix{2}(0.01), SVector(0.0, 0.0))))
    @test minimal_manifold(PlanarTarget(design; calibration = H ∘ T ∘ Scale(0.001)); known = (:length,)) isa VisualGeometryToolkitCore.TargetManifold
end
