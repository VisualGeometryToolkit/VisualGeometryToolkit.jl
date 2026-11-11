# =============================================================================
# The board's deformation on the device, and the surface of its chart map:
# the thin-plate kinds' device form (Adapt, convert_eltype), the static
# pure-bending coefficients, a HeightField from scattered heights, MappedSurface (the surface of
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
    using VisualGeometryToolkitCore: grid_nodes, ThinPlateSplineSystem, tps_coefficients, tps_pure_bending,
        tps_evaluate, is_pure_bending, convert_eltype

    rng = Xoshiro(22)
    nodes = grid_nodes(5; lo = 0.0, hi = 1.0)
    sys = ThinPlateSplineSystem(nodes)
    M = [ones(25) first.(nodes) last.(nodes)]
    bend = SVector{25}(0.01 .* randn(rng, 25))
    raw = bend + SVector{25}(M * [0.3, -0.2, 0.7])                         # a plane on top: the pose's piston and tilts
    # (a) one implementation: the host kinds hold exactly what the static path gives
    v = tps_pure_bending(sys, raw)
    w, a = tps_coefficients(sys, v)
    H = HeightField(nodes, raw)
    @test parameter_vector(H) == v && H.w == w && H.a == a
    # a is exactly zero for a pure-bending v; computed, it is the rounding of the (n+3)-term
    # products coeff·v: |δa| <= γ_(n+3) ‖coeff‖ ‖v‖ with γ_k ≈ k·eps (the standard bound), so the
    # bound scales with the system's own norm instead of an absolute 1e-14 (FMA and summation
    # order move the measured value by tens of eps: 25·eps here)
    rounding_bound = size(sys.coeff, 1) * eps() * opnorm(Matrix(sys.coeff)) * norm(v)
    @test is_pure_bending(sys, v) && norm(a) <= rounding_bound
    @test tps_pure_bending(sys, v) === v                                    # pure bending is kept bit for bit
    @test norm(tps_pure_bending(sys, SVector{25}(M * [1.0, 2.0, -3.0]))) <= 1.0e-13   # a plane is all pose: nothing left
    # against the matrix (BLAS) form of the same map: to 1e-12
    wa = sys.coeff * Vector(v)
    @test isapprox(vcat(collect(w), collect(a)), wa; atol = 1.0e-12 * norm(wa))
    p = SVector(0.31, 0.77)
    @test H(p)[3] ≈ tps_evaluate(sys.nodes, SVector{25}(wa[1:25]), SVector{3}(wa[26:28]), p) atol = 1.0e-12

    # (b) a ForwardDiff Dual through it against central differences (the projection's derivative included)
    g(h) = ((ww, aa) = tps_coefficients(sys, tps_pure_bending(sys, h)); tps_evaluate(sys.nodes, ww, aa, p))
    G = ForwardDiff.gradient(g, Vector(raw))
    ε = 1.0e-6
    Gfd = [(g(Vector(raw) + ε * (1:25 .== k)) - g(Vector(raw) - ε * (1:25 .== k))) / 2ε for k in 1:25]
    @test G ≈ Gfd atol = 1.0e-9
    @test abs(sum(G .* (M * [1.0, 0.5, -0.25]))) <= 1.0e-12               # a plane direction moves nothing
    # at pure-bending values the values are kept bit for bit and a Dual's partials projected: the
    # derivative is the projection's there too (the projected nodal weights, the kinds' transform_deriv_params),
    # the nodal weights themselves on every pure-bending direction
    Gv = ForwardDiff.gradient(g, Vector(v))
    @test Gv ≈ collect(VisualGeometryToolkitCore._pure_bending_weights(sys, p)) atol = 1.0e-12
    @test Gv ≈ [(g(Vector(v) + ε * (1:25 .== k)) - g(Vector(v) - ε * (1:25 .== k))) / 2ε for k in 1:25] atol = 1.0e-9
    b = tps_pure_bending(sys, SVector{25}(randn(rng, 25)))
    @test dot(Gv, b) ≈ dot(G, b) atol = 1.0e-12
    @test dot(Gv, b) ≈ dot(VisualGeometryToolkitCore.nodal_weights(sys, p), b) atol = 1.0e-12

    # (c) allocation-free on the CPU
    allocs(fn, args...) = (fn(args...); @allocated fn(args...))
    @test allocs(tps_pure_bending, sys, raw) == 0 && allocs(tps_coefficients, sys, raw) == 0
    d = ForwardDiff.Dual.(raw, 1.0)
    @test allocs(tps_pure_bending, sys, d) == 0 && allocs(tps_coefficients, sys, d) == 0

    # (d) in a KernelAbstractions kernel, the system adapted, in Float64 and Float32
    @kernel function bending_kernel!(out, @Const(values), s)
        i = @index(Global)
        ww, _ = tps_coefficients(s, tps_pure_bending(s, values[i]))
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
        @test out == [tps_coefficients(sF, tps_pure_bending(sF, x))[1] for x in valsF]
        host = [tps_coefficients(sys, tps_pure_bending(sys, x))[1] for x in vals]
        # within the summation's rounding in F: 64 eps(F) of the terms' magnitudes Σⱼ |Cᵢⱼ vⱼ| (the projection's too)
        C = abs.(sys.coeff[1:25, :])
        @test all(all(abs.(Float64.(a) - b) .<= 64 * eps(F) .* (C * abs.(x) .+ norm(b))) for (a, b, x) in zip(out, host, vals))
    end
end

@testitem "board deformation: HeightField from scattered heights" begin
    using Test, StaticArrays, Random, LinearAlgebra
    using VisualGeometryToolkitCore
    using VisualGeometryToolkitCore: grid_nodes, is_pure_bending, mm, m, has_units

    rng = Xoshiro(23)
    nodes = grid_nodes(4; lo = 0.0, hi = 1.0)
    truth = HeightField(nodes, 0.01 .* randn(rng, 16))                   # pure bending
    pts = [SVector(rand(rng), rand(rng)) for _ in 1:200]
    plane(p) = 0.02 - 0.05p[1] + 0.03p[2]
    z = [truth(p)[3] + plane(p) for p in pts]
    H = HeightField(nodes, pts .=> z)
    @test H isa HeightField{16} && is_pure_bending(H.system, parameter_vector(H))
    @test norm(parameter_vector(H) - parameter_vector(truth)) <= 1.0e-10 * norm(parameter_vector(truth))
    # a planar input: zero bending
    Hp = HeightField(nodes, pts .=> plane.(pts))
    @test norm(parameter_vector(Hp)) <= 1.0e-12
    # smoothing pulls the field towards a plane: less bending energy
    Hs = HeightField(nodes, pts .=> z .+ 1.0e-3 .* randn(rng, length(z)); smoothing = 1.0e-3)
    using VisualGeometryToolkitCore: bending_energy
    @test bending_energy(Hs.system, parameter_vector(Hs)) < bending_energy(truth.system, parameter_vector(truth))
    # lengths: the physical field
    Hm = HeightField([n .* m for n in nodes], [p .* m for p in pts] .=> z .* m)
    @test has_units(Hm) && isapprox(parameter_vector(Hm), parameter_vector(H); rtol = 1.0e-9)
    @test_throws ArgumentError HeightField(nodes, pts[1:5] .=> z[1:5])
    @test_throws ArgumentError HeightField(nodes, pts .=> z; smoothing = -1)
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
            # Float32 takes the fast path too, and it is the Float32 root to Float32 rounding
            s32 = convert_eltype(Float32, s)
            @test VisualGeometryToolkitCore.Cameras._plain_fast(typeof(s32.map), SVector{3, Float32}, SVector{3, Float32})
            for (o, d) in rays
                tf, xf = @inferred intersection(s32, Float32.(o), Float32.(d))
                tg, xg = generic(s32, Float32.(o), Float32.(d))
                @test tf isa Float32
                if isnan(tf) != isnan(tg)
                    # a hit and a miss differ only on the rectangle's edge, within Float32's rounding of the chart point
                    x64 = generic(s, o, d)[2]
                    c64 = isnan(x64[1]) ? generic(s, o, d)[2] : chart(s, x64)
                    lo, hi = rect.origin, rect.origin + rect.widths
                    @test isnan(c64[1]) || minimum(abs, vcat(c64 - lo, hi - c64)) <= 1.0e-5
                else
                    isnan(tf) || @test maximum(abs, xf - xg) <= 64 * eps(Float32) * 2
                end
            end
        end
    end

    @testset "Float32 resolves a sub-millimetre relief (the LM's precision-aware stops)" begin
        # 0.2 mm heights: the plane-hit seed is within a few tenths of a millimetre of the surface, which the LM's
        # former absolute eps(Float32) stop on the squared residual accepted as it was
        Hl = HeightField(nodes, 2.0e-4 .* randn(rng, 16))
        s = MappedSurface(X, Hl ∘ ThinPlateSpline(nodes), nothing, rect)    # the 2 × 2 root
        s32 = convert_eltype(Float32, s)
        plane = MappedSurface(X, ThinPlateSpline(nodes), nothing, rect)
        e32, eplane = Float64[], Float64[]
        for (o, d) in rays
            t, x = intersection(s, o, d)
            isnan(t) && continue
            t32, x32 = intersection(s32, Float32.(o), Float32.(d))
            push!(e32, norm(Float64.(x32) - x))
            push!(eplane, norm(intersection(plane, o, d)[2] - x))
        end
        L = hypot(rect.widths...)
        @test length(e32) > 100
        @test maximum(e32) <= 2 * 16 * eps(Float32) * L                     # within the stop's floor (twice: rounding)
        @test sort(e32)[end ÷ 2] < sort(eplane)[end ÷ 2] / 100              # far nearer the surface than the plane is
    end

    @testset "the iteration is plain: no Dual reaches the LM" begin
        c = H ∘ T ∘ A
        s = MappedSurface(X, c, nothing, rect)
        o, d = rays[1]
        ForwardDiff.jacobian(od -> intersection(s, SVector{3}(od[1:3]), SVector{3}(od[4:6]))[2], [o; d])
        ForwardDiff.hessian(od -> intersection(s, SVector{3}(od[1:3]), SVector{3}(od[4:6]))[2][3], [o; d])
        # every method instance the LM was compiled for (dispatch tuples: the concrete calls) is Dual-free
        lm = VisualGeometryToolkitCore.Cameras._spline_lm
        calls = [mi.specTypes for m in methods(lm) for mi in Base.specializations(m) if Base.isdispatchtuple(mi.specTypes)]
        @test !isempty(calls) && !any(t -> occursin("ForwardDiff.Dual", string(t)), calls)
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
        # rebuilt from new nodal heights inside a kernel: with_parameters on the adapted field, then the
        # unchecked builder with a placement (the parts checked once, on the host), in the parts' element type
        build = VisualGeometryToolkitCore.Cameras._mapped_surface
        # (a kernel adapts each argument: the in-plane spline is given as itself — a CT composition has
        # no Adapt rule — and composed in the kernel)
        @kernel function rebuild_hits!(t, @Const(heights), field, spline, affine, X, S, rect, o, d)
            i = @index(Global)
            Hi = with_parameters(field, heights[i])
            si = build(X, spline === nothing ? Hi : Hi ∘ (spline ∘ affine), S, rect)
            t[i] = intersection(si, o, d)[1]
        end
        o, d = SVector(0.05, -0.1, 0.0), SVector(0.1, 0.2, 1.0)
        hs = [SVector{16}(0.01 .* randn(rng, 16)) for _ in 1:32]
        S = SimilarityMap(0.9, one(RotMatrix{3, Float64}), SVector(-0.4, -0.45, 0.0))
        host(h, spline, F) = intersection(convert_eltype(F, MappedSurface(X, spline === nothing ? with_parameters(H, h) : with_parameters(H, h) ∘ (spline ∘ A),
                                                                          S, bounds(s))), SVector{3, F}(o), SVector{3, F}(d))[1]
        parts(F, spline) = (convert_eltype(F, H), spline === nothing ? nothing : convert_eltype(F, spline), convert_eltype(F, A), convert_eltype(F, X),
                            VisualGeometryToolkitCore.Cameras._member_numbers(v -> F(v), S), Rect2{F}(bounds(s)), SVector{3, F}(o), SVector{3, F}(d))
        backend = KernelAbstractions.CPU()
        for spline in (nothing, T), F in (Float64, Float32)                  # the fast path, the 2 × 2 root
            Hd, TF, AF, XF, SF, rF, oF, dF = parts(F, spline)
            t = zeros(F, length(hs))
            rebuild_hits!(backend, 8)(t, [SVector{16, F}(h) for h in hs], adapt(backend, Hd), TF, AF, XF, SF, rF, oF, dF; ndrange = length(hs))
            KernelAbstractions.synchronize(backend)
            ref = [host(SVector{16, F}(h), spline, F) for h in hs]
            @test isequal(isnan.(t), isnan.(ref)) && maximum(abs, filter(isfinite, t - ref)) <= 64 * eps(F)
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
            for spline in (nothing, T), F in (Float64, Float32)
                Hd, TF, AF, XF, SF, rF, oF, dF = parts(F, spline)
                t = Base.invokelatest(CuArray, zeros(F, length(hs)))
                Base.invokelatest(rebuild_hits!(gpu, 8), t, Base.invokelatest(CuArray, [SVector{16, F}(h) for h in hs]),
                                  Base.invokelatest(adapt, gpu, Hd), TF === nothing ? nothing : Base.invokelatest(adapt, gpu, TF), AF, XF, SF, rF, oF, dF;
                                  ndrange = length(hs))
                tg = Base.invokelatest(Array, t)
                ref = [host(SVector{16, F}(h), spline, F) for h in hs]
                @test isequal(isnan.(tg), isnan.(ref)) && maximum(abs, filter(isfinite, tg - ref)) <= 64 * eps(F)
            end
        end
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
        # the condition numbers of the column-normalized Jacobians (a radian, a metre and a deformation
        # coordinate are not commensurate: unscaled, cond measures the units); the angle is scale-free
        unit(A) = A ./ [norm(col) for col in eachcol(A)]'
        J1 = J[1:3n, 1:(m + 6)]                                             # one pose: the identity
        Qd = Matrix(qr(J1[:, 1:m]).Q)[:, 1:m]
        Qp = Matrix(qr(J1[:, (m + 1):end]).Q)[:, 1:6]
        F1 = svd(Qd' * Qp)
        # the principal deformation direction (nearest the pose's span), as the calibration's parameter change
        y = J1[:, 1:m] \ (Qd * F1.U[:, 1])
        mode = Matrix(retract_jacobian(ch)) * y
        return (m = m, cond = cond(unit(J)), cond1 = cond(unit(J1)), angle = acos(min(1.0, F1.S[1])), mode = mode / norm(mode))
    end
    # measured 2026-11-10 (cond of the column-normalized Jacobian over three poses, at the identity pose alone, the
    # smallest principal angle in rad):
    #   height only        26.3  26.3  0.095
    #   in-plane only      27.2  26.9  0.095
    #   Aspect∘Skew∘TPS    31.3  31.6  0.086
    #   full chain         31.1  31.6  0.086
    #   Skew alone          2.2   2.1  0.88   (the Iwasawa N: its antisymmetric part is no gauge of the in-plane rotation)
    #   height, nodes inside the data   6.2  5.7  0.36
    #   full chain, nodes inside        8.5  8.4  0.29
    # pinned at twice the conditions and half the angles
    ni = grid_nodes(4; lo = -0.4, hi = 0.4)                                # nodes inside the sampled rectangle
    for (name, cal, κ, κ1, α) in (("height only", H, 53, 53, 0.047), ("in-plane only", T, 55, 54, 0.047),
                                  ("Aspect∘Skew∘TPS", T ∘ Aspect(0.01) ∘ Skew(-0.003), 63, 64, 0.043),
                                  ("full chain", H ∘ T ∘ Aspect(0.01) ∘ Skew(-0.003), 63, 64, 0.043),
                                  ("Skew alone", Skew(-0.003), 4.4, 4.3, 0.44),
                                  ("height, nodes inside the data", HeightField(ni, zeros(16)), 13, 12, 0.18),
                                  ("full chain, nodes inside the data", HeightField(ni, zeros(16)) ∘ ThinPlateSpline(ni) ∘ Aspect(0.01) ∘ Skew(-0.003), 17, 17, 0.145))
        r = stats(cal)
        @testset "$name" begin
            @test r.cond <= κ && r.cond1 <= κ1 && r.angle >= α
        end
    end
    # THE NEAR-GAUGE, identified: with the nodes outside the data (±0.6 against samples to ±0.5, ±0.4) the
    # deformation direction nearest the pose's span is a nearly CONSTANT field of nodal heights — a pure-bending
    # bowl whose curvature lies beyond the samples, so on the data it is almost a piston (0.095 rad from it, a
    # variance inflation of 1/sin² ≈ 110 along it); with the nodes inside the data the angle is 0.36 rad.
    mode = stats(H).mode
    @test abs(sum(mode)) / (sqrt(16) * norm(mode)) > 0.9                  # its correlation with a constant
    @test stats(HeightField(ni, zeros(16))).angle > 3 * stats(H).angle

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
    Aff = sys.coeff[(end - 2):end, :]
    @test opnorm(cat(Aff, Aff, Aff; dims = (1, 2)) * B) <= 1.0e-13

    # (4) the pose's own members are refused without what fixes them
    @test_throws ArgumentError minimal_manifold(PlanarTarget(design; calibration = H ∘ T ∘ Scale(0.001) ∘ Aspect(0.01)))
    @test_throws ArgumentError minimal_manifold(PlanarTarget(design; calibration = H ∘ T ∘ EuclideanMap(RotMatrix{2}(0.01), SVector(0.0, 0.0))))
    @test minimal_manifold(PlanarTarget(design; calibration = H ∘ T ∘ Scale(0.001)); known = (:length,)) isa VisualGeometryToolkitCore.TargetManifold

    # a placed target's action through the composition's members, planar (2 rows) and lifted (3 rows)
    using ForwardDiff
    P = SimilarityMap(0.9, one(RotMatrix{3, Float64}), SVector(-0.4, -0.45, 0.0))
    for cal in (T, H ∘ T)
        tp = PlanarTarget(design; calibration = cal, placement = P)
        chp = minimal_manifold(tp)
        Jp = ForwardDiff.jacobian(θ -> points(with_parameters(tp, θ))[3], Vector(parameter_vector(tp)))
        @test action_jacobian(chp, design[3]) ≈ Jp * Matrix(retract_jacobian(chp)) rtol = 1.0e-10
    end
end

# The generic ray–board root resolves a nearly flat surface to rounding: a residual of the plane-hit seed below √eps of
# the chart's diagonal is not the root (heights of 1e-9 move the hit by 1e-9), so the hit is a smooth function of the
# heights and its finite differences are its derivative.
@testitem "board deformation: the generic root resolves tiny heights (finite differences, the fast path)" begin
    using Test, StaticArrays, Random, LinearAlgebra
    using ForwardDiff
    using Rotations: RotMatrix, RotXYZ
    using VisualGeometryToolkitCore
    using VisualGeometryToolkitCore: intersection, grid_nodes, chart

    rng = Xoshiro(31)
    nodes = grid_nodes(4; lo = -0.6, hi = 0.6)
    X = EuclideanMap(RotMatrix{3}(RotXYZ(0.1, -0.2, 0.05)), SVector(0.02, -0.01, 2.0))
    rect = Rect2(-0.5, -0.5, 1.0, 1.0)
    generic(s, o, d) = VisualGeometryToolkitCore.Cameras._mapped_intersection(s, o, d)
    warped(h) = MappedSurface(X, HeightField(nodes, h) ∘ ThinPlateSpline(nodes), nothing, rect)   # the generic root
    rays = [(SVector(0.2randn(rng), 0.2randn(rng), 0.0), SVector(0.1randn(rng), 0.1randn(rng), 1.0)) for _ in 1:20]
    for scale in (1.0e-9, 1.0e-8, 1.0e-7)
        h = scale .* randn(rng, 16)
        for (o, d) in rays
            tg, xg = generic(warped(h), o, d)
            tf, xf = intersection(MappedSurface(X, HeightField(nodes, h), nothing, rect), o, d)   # the 1-D fast path
            isnan(tf) && continue
            @test maximum(abs, xg - xf) <= 16eps()
            @test maximum(abs, chart(warped(h), xg) - chart(MappedSurface(X, HeightField(nodes, h), nothing, rect), xf)) <= 1.0e-12
        end
        # the hit's derivative by the heights: central differences at a step of the heights' own size against the
        # implicit derivative (a Dual through the root)
        o, d = first(r for r in rays if isfinite(generic(warped(h), r...)[1]))
        hit(v) = generic(warped(v), o, d)[2]
        Jd = ForwardDiff.jacobian(hit, h)
        δ = scale
        Jfd = reduce(hcat, [(hit(h .+ δ .* (1:16 .== k)) - hit(h .- δ .* (1:16 .== k))) ./ 2δ for k in 1:16])
        @test maximum(abs, Jfd - Jd) <= 1.0e-5 * maximum(abs, Jd)
    end
end

# A thin-plate kind's derivative by its parameters is the derivative of the projection its `with_parameters` applies:
# a ForwardDiff Dual through it, at pure-bending values (kept bit for bit) and off them, agrees with central
# differences of the function, and `transform_deriv_params` is that Jacobian.
@testitem "board deformation: the thin-plate kinds' parameter derivative is the projection's (Dual, central differences)" begin
    using Test, StaticArrays, Random, LinearAlgebra
    using ForwardDiff
    using VisualGeometryToolkitCore
    using VisualGeometryToolkitCore: grid_nodes, with_parameters, parameter_vector, transform_deriv_params, tps_pure_bending

    rng = Xoshiro(32)
    nodes = grid_nodes(4; lo = -0.6, hi = 0.6)
    H = HeightField(nodes, 0.01 .* randn(rng, 16))
    T = ThinPlateSpline(nodes, 0.01 .* randn(rng, 16), 0.01 .* randn(rng, 16))
    q = SVector(0.13, -0.27)
    for c in (H, T)
        n = length(parameter_vector(c))
        f(θ) = with_parameters(c, θ)(q)
        central(θ, ε) = reduce(hcat, [(f(θ .+ ε .* (1:n .== k)) - f(θ .- ε .* (1:n .== k))) ./ 2ε for k in 1:n])
        for θ in (Vector(parameter_vector(c)), 0.01 .* randn(rng, n))    # pure bending (kept), and with a plane part
            J = ForwardDiff.jacobian(f, θ)
            @test maximum(abs, J - central(θ, 1.0e-6)) <= 1.0e-8 * maximum(abs, J)
            @test transform_deriv_params(with_parameters(c, θ), q) ≈ J atol = 1.0e-12
        end
    end
    # pure-bending values are kept bit for bit under a Dual, their partials projected
    sys = H.system
    v = parameter_vector(H)
    d = tps_pure_bending(sys, ForwardDiff.Dual.(v, 1.0))
    @test ForwardDiff.value.(d) == v
    @test ForwardDiff.partials.(d, 1) ≈ tps_pure_bending(sys, ones(16)) atol = 1.0e-14
end

@testitem "board deformation: Dual heights take the fast hit and one implicit-function step" begin
    using Test, StaticArrays, LinearAlgebra, ForwardDiff
    using KernelAbstractions
    using KernelAbstractions: @kernel, @index, @Const
    using Adapt: adapt
    using VisualGeometryToolkitCore: convert_eltype
    using Rotations: RotMatrix, RotXYZ
    using VisualGeometryToolkitCore
    using VisualGeometryToolkitCore: intersection, grid_nodes, with_parameters

    dir = joinpath(pkgdir(VisualGeometryToolkitCore), "test", "fixtures", "height_surface")
    include(joinpath(dir, "dual_hit_surface.jl"))
    include(joinpath(dir, "dual_hit.jl"))
    REF = DUAL_HIT_REFERENCE
    s = dual_hit_surface(dual_hit_heights(REF.heights))
    s0 = dual_hit_surface(REF.heights)
    rays = [(SVector{3}(o), SVector{3}(d)) for (o, d) in zip(REF.origins, REF.directions)]

    # the generic root's values and partials (generate_dual_hit.jl, before the fast path took Duals), to 1e-12
    @testset "the generic root's hit, value and partials" begin
        @test count(isfinite, REF.t) == length(REF.t) - 1          # one ray misses the page
        for (k, (o, d)) in enumerate(rays)
            t, x = intersection(s, o, d)
            if isnan(REF.t[k])
                @test isnan(ForwardDiff.value(t)) && all(isnan ∘ ForwardDiff.value, x)
            else
                @test abs(ForwardDiff.value(t) - REF.t[k]) <= 1.0e-12
                @test maximum(abs.(collect(ForwardDiff.partials(t).values) .- REF.dt_dh[k])) <= 1.0e-12
                @test abs(ForwardDiff.value(t) - intersection(s0, o, d)[1]) <= 1.0e-15   # the plain fast hit, to rounding
            end
        end
    end

    # the cost: the plain fast hit plus one Dual evaluation of the residual, not the generic 2 × 2 LM from the plane hit —
    # over these rays ≈ 1.6× the plain hit with 25 partials (the LM: ≈ 3× on a hit, ≈ 36× on the miss, ≈ 5× in all) — and
    # no allocation
    @testset "the cost of the partials" begin
        o, d = rays[findfirst(isfinite, REF.t)]
        @test (intersection(s, o, d); @allocated intersection(s, o, d)) == 0
        all_rays(s) = for (o, d) in rays; intersection(s, o, d); end
        timed(s) = (t0 = time_ns(); for _ in 1:50; all_rays(s); end; time_ns() - t0)
        all_rays(s); all_rays(s0)
        tD, tP = typemax(UInt64), typemax(UInt64)
        for _ in 1:9                                   # interleaved, the best of each: a busy machine slows both alike
            tD = min(tD, timed(s)); tP = min(tP, timed(s0))
        end
        @test tD < 3.0 * tP
    end

    # on a device: the Dual heights' hit (Float64) and a Dual ray's (a pose's partials; Float64 and Float32) compile there and
    # agree with the host's
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
        @kernel function hits!(t, @Const(os), @Const(ds), s)
            i = @index(Global)
            t[i] = intersection(s, os[i], ds[i])[1]
        end
        on_device(s, os, ds) = (out = Base.invokelatest(CuArray, Vector{eltype(eltype(os))}(undef, length(os)));
                                Base.invokelatest(hits!(gpu, 64), out, Base.invokelatest(CuArray, os), Base.invokelatest(CuArray, ds),
                                                  Base.invokelatest(adapt, gpu, s); ndrange = length(os));
                                Base.invokelatest(KernelAbstractions.synchronize, gpu); Base.invokelatest(Array, out))
        DH = eltype(dual_hit_heights(REF.heights))
        td = on_device(s, [map(DH, o) for (o, _) in rays], [map(DH, d) for (_, d) in rays])
        for (k, t) in enumerate(td)
            isnan(REF.t[k]) && (@test isnan(ForwardDiff.value(t)); continue)
            @test abs(ForwardDiff.value(t) - REF.t[k]) <= 1.0e-12
            @test maximum(abs.(collect(ForwardDiff.partials(t).values) .- REF.dt_dh[k])) <= 1.0e-12
        end
        for F in (Float64, Float32)
            tol = F === Float64 ? 1.0e-12 : 1.0e-3
            Dr = ForwardDiff.Dual{Nothing, F, 6}
            seed(v, k) = SVector(ntuple(i -> Dr(F(v[i]), ForwardDiff.Partials(ntuple(j -> F(j == k + i), 6))), 3))
            tr = on_device(convert_eltype(F, s0), [seed(o, 0) for (o, _) in rays], [seed(d, 3) for (_, d) in rays])
            th = [intersection(s0, map(x -> ForwardDiff.Dual{Nothing}(Float64(ForwardDiff.value(x)), Float64.(ForwardDiff.partials(x).values)...), seed(o, 0)),
                               map(x -> ForwardDiff.Dual{Nothing}(Float64(ForwardDiff.value(x)), Float64.(ForwardDiff.partials(x).values)...), seed(d, 3)))[1] for (o, d) in rays]
            for (a, b) in zip(tr, th)
                isnan(ForwardDiff.value(b)) && (@test isnan(ForwardDiff.value(a)); continue)
                @test abs(ForwardDiff.value(a) - ForwardDiff.value(b)) <= tol
                @test maximum(abs.(Float64.(collect(ForwardDiff.partials(a).values)) .- collect(ForwardDiff.partials(b).values))) <= tol
            end
        end
    end
end

@testitem "board deformation: nested Duals (a Hessian) through a height field's ray hit" begin
    using Test, StaticArrays, LinearAlgebra, ForwardDiff
    using Rotations: RotMatrix, RotXYZ
    using VisualGeometryToolkitCore
    using VisualGeometryToolkitCore: intersection, grid_nodes

    dir = joinpath(pkgdir(VisualGeometryToolkitCore), "test", "fixtures", "height_surface")
    include(joinpath(dir, "dual_hit_surface.jl"))
    include(joinpath(dir, "dual_hit.jl"))
    REF = DUAL_HIT_REFERENCE
    generic(s, o, d) = VisualGeometryToolkitCore.Cameras._mapped_intersection(s, o, d)
    # the ray parameter as a function of three nodal heights (the rest at the fixture's), through `hit`
    t_of(hit, o, d) = v -> (h = [i <= 3 ? REF.heights[i] + v[i] : oftype(v[1], REF.heights[i]) for i in eachindex(REF.heights)];
                            hit(dual_hit_surface(h), o, d)[1])
    for k in findall(isfinite, REF.t)[1:4]
        o, d = SVector{3}(REF.origins[k]), SVector{3}(REF.directions[k])
        # nested Duals: the fast path's fallback, the generic root seeded at the primal hit
        H = ForwardDiff.hessian(t_of(intersection, o, d), zeros(3))
        # the generic root from the plane hit, unseeded (the implicit function theorem layer by layer)
        @test maximum(abs, H - ForwardDiff.hessian(t_of(generic, o, d), zeros(3))) <= 1.0e-12 * max(1.0, maximum(abs, H))
        # central differences of the (first-order Dual) gradient, step 1e-5 m: truncation O(h²), well under 1e-6
        g(v) = ForwardDiff.gradient(t_of(intersection, o, d), v)
        Hfd = reduce(hcat, ((g(1.0e-5 .* (1:3 .== j)) - g(-1.0e-5 .* (1:3 .== j))) ./ 2.0e-5 for j in 1:3))
        @test maximum(abs, H - Hfd) <= 1.0e-6 * max(1.0, maximum(abs, H))
        @test H ≈ H' atol = 1.0e-12
    end
end
