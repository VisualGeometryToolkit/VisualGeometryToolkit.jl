# =============================================================================
# The surface a ray meets (geometry/primitives/surfaces.jl): Quad and
# HeightSurface, and surface(::PosedTarget) (geometry/cameras/target_surfaces.jl).
#
# A Quad's intersection is ray_plane_intersection's arithmetic, bit for bit,
# NaN outside its rectangle. A HeightSurface's is RadiometricCamera's
# `intersect_board` on its own warped test board: test/fixtures/height_surface/
# reference.jl holds what RadiometricCamera returned (generate.jl beside it).
# =============================================================================

@testitem "surfaces: a Quad is ray_plane_intersection, bounded" begin
    using Test, StaticArrays, LinearAlgebra, Random
    using Rotations: RotMatrix, QuatRotation
    using VisualGeometryToolkitCore
    using VisualGeometryToolkitCore: intersection, chart, bounds, normal, ray_plane_intersection

    rng = Xoshiro(20261026)
    count = Dict(:hit => 0, :miss => 0, :border => 0, :grazing => 0)   # mutated, never rebound (soft scope)
    for k in 1:100
        R = RotMatrix(QuatRotation(randn(rng, 4)...))
        o = SVector{3}(randn(rng, 3))
        rect = Rect2(randn(rng), randn(rng), 0.5 + rand(rng), 0.5 + rand(rng))
        e_u, e_v, n = R[:, 1], R[:, 2], R[:, 3]
        q = Quad(o, e_u, e_v, rect)
        @test isbits(q) && bounds(q) == rect
        @test normal(q) ≈ n && normal(q, o) === normal(q.plane)
        lo, w = rect.origin, rect.widths
        for j in 1:100
            # aim at a chart point of the rectangle grown by 30 % on each side
            c = SVector(lo[1] + w[1] * (1.6 * rand(rng) - 0.3), lo[2] + w[2] * (1.6 * rand(rng) - 0.3))
            X = o + c[1] * e_u + c[2] * e_v
            ro = X + R * SVector(randn(rng), randn(rng), (rand(rng, Bool) ? 1 : -1) * (0.2 + 2 * rand(rng)))
            d = X - ro
            if j % 10 == 0          # grazing: in the plane up to ε (ε = 0: parallel to rounding)
                count[:grazing] += 1
                ε = (0.0, 1.0e-14, 1.0e-10, 1.0e-6)[mod1(j ÷ 10, 4)]
                d = randn(rng) * e_u + randn(rng) * e_v + ε * n
            end
            x1, x2, x3, tr = ray_plane_intersection(q.plane, ro, d[1], d[2], d[3])
            xr = SVector(x1, x2, x3)
            cr = (R' * (xr - o))[SOneTo(2)]          # the chart, computed independently
            margin = min(cr[1] - lo[1], lo[1] + w[1] - cr[1], cr[2] - lo[2], lo[2] + w[2] - cr[2])
            t, x = intersection(q, ro, d)
            if isfinite(margin) && margin > 1.0e-9
                count[:hit] += 1
                @test (t, x) === (tr, xr)                  # bit for bit
                @test chart(q, x) ≈ cr atol = 1.0e-12
            elseif !isfinite(margin) || margin < -1.0e-9
                count[:miss] += 1
                @test isnan(t) && all(isnan, x)
            else
                count[:border] += 1
            end
        end
        # round trips
        u, v = lo[1] + w[1] * rand(rng), lo[2] + w[2] * rand(rng)
        @test chart(q, o + u * e_u + v * e_v) ≈ SVector(u, v) atol = 1.0e-14
        t, x = intersection(q, o + u * e_u + v * e_v + 2n, -n)
        @test t ≈ 2 && x ≈ o + u * e_u + v * e_v && chart(q, x) ≈ SVector(u, v)
    end
    @test count[:hit] + count[:miss] + count[:border] == 10^4 && count[:hit] > 3000 && count[:miss] > 1000 &&
        count[:border] < 10 && count[:grazing] == 1000

    # a ray exactly parallel to an axis-aligned quad divides by zero: a miss, not an error
    q = Quad(SVector(0.0, 0.0, 1.0), SVector(1.0, 0.0, 0.0), SVector(0.0, 1.0, 0.0), Rect2(0.0, 0.0, 1.0, 1.0))
    t, x = intersection(q, SVector(0.5, 0.5, 2.0), SVector(1.0, 0.0, 0.0))
    @test isnan(t) && all(isnan, x)
    # the rectangle is closed: its edge is a hit
    t, x = intersection(q, SVector(1.0, 0.0, 2.0), SVector(0.0, 0.0, -1.0))
    @test t == 1.0 && x == SVector(1.0, 0.0, 1.0)
    @test_throws ArgumentError Quad(SVector(0.0, 0.0, 0.0), SVector(1.0, 0.0, 0.0), SVector(1.0, 1.0, 0.0), Rect2(0.0, 0.0, 1.0, 1.0))
end

@testitem "surfaces: a HeightSurface meets RadiometricCamera's warped board" begin
    using Test, StaticArrays, LinearAlgebra
    using VisualGeometryToolkitCore
    using VisualGeometryToolkitCore: intersection, chart, bounds, normal, ThinPlateSplineSystem, tps_coefficients, tps_evaluate

    include(joinpath(pkgdir(VisualGeometryToolkitCore), "test", "fixtures", "height_surface", "reference.jl"))
    REF = HEIGHT_SURFACE_REFERENCE
    # RadiometricCamera's board is the square [-1, 1]² of z = 0, normal +z, its
    # chart [0, 1]²; here the chart is metric, the target's own (x, y), so the
    # nodes are -1 + 2 b and the heights the bending part at them (generate.jl)
    nodes = SVector{25}([REF.origin[SOneTo(2)] + 2 .* b for b in REF.nodes])
    h = SVector{25}(REF.bending_heights)
    w, a = tps_coefficients(ThinPlateSplineSystem(nodes), Vector(h))
    quad = Quad(SVector(0.0, 0.0, 0.0), SVector(1.0, 0.0, 0.0), SVector(0.0, 1.0, 0.0), Rect2(-1.0, -1.0, 2.0, 2.0))
    s = HeightSurface(quad, nodes, h, w, a)
    @test isbits(s) && s isa HeightSurface{25, Float64} && bounds(s) == bounds(quad)
    # (RadiometricCamera's spline over [0, 1]² through these heights has no
    # affine part; over the metric chart the scale's ln 2 lands in the constant a₁)

    hits = [intersection(s, o, d) for (o, d) in zip(REF.ray_origins, REF.ray_directions)]
    xs = last.(hits)
    Δt = maximum(abs.(first.(hits) - REF.t))
    Δx = maximum(maximum(abs, x - r) for (x, r) in zip(xs, REF.x))
    Δc = maximum(maximum(abs, chart(s, x) - (REF.origin[SOneTo(2)] + 2 .* c)) for (x, c) in zip(xs, REF.chart))
    Δn = maximum(maximum(abs, normal(s, x) - r) for (x, r) in zip(xs, REF.surface_normal))
    @test all(abs(x[3] - tps_evaluate(nodes, w, a, chart(s, x))) < 1.0e-15 for x in xs)   # on the surface
    @test length(REF.t) == 100
    @test Δt <= 1.0e-15 && Δx <= 1.0e-15 && Δc <= 1.0e-15 && Δn <= 1.0e-15
    # the normal leans against the slope: (−∂h/∂u, −∂h/∂v, 1), normalised
    x = intersection(s, REF.ray_origins[1], REF.ray_directions[1])[2]
    ε = 1.0e-6
    hu = (tps_evaluate(nodes, w, a, chart(s, x) + SVector(ε, 0)) - tps_evaluate(nodes, w, a, chart(s, x) - SVector(ε, 0))) / 2ε
    hv = (tps_evaluate(nodes, w, a, chart(s, x) + SVector(0, ε)) - tps_evaluate(nodes, w, a, chart(s, x) - SVector(0, ε))) / 2ε
    @test normal(s, x) ≈ normalize(SVector(-hu, -hv, 1.0)) atol = 1.0e-9
    # a hit outside the rectangle is a miss
    t, x = intersection(s, SVector(1.5, 0.0, 1.0), SVector(0.0, 0.0, -1.0))
    @test isnan(t) && all(isnan, x)
end

@testitem "surfaces: Float32 and ForwardDiff, inferred and allocation-free" begin
    using Test, StaticArrays, LinearAlgebra, Random
    using ForwardDiff
    using Rotations: RotMatrix, QuatRotation
    using VisualGeometryToolkitCore
    using VisualGeometryToolkitCore: intersection, chart, bounds, normal, ThinPlateSplineSystem, tps_coefficients, grid_nodes

    rng = Xoshiro(3)
    R = RotMatrix(QuatRotation(1.0, 0.1, -0.2, 0.05))
    o = SVector(0.1, -0.2, 0.3)
    quad = Quad(o, R[:, 1], R[:, 2], Rect2(-0.5, -0.5, 1.0, 1.0))
    nodes = SVector{9}(grid_nodes(3; lo = -0.5, hi = 0.5))
    sys = ThinPlateSplineSystem(nodes)
    h = SVector{9}(0.01 .* randn(rng, 9))
    surf = HeightSurface(quad, nodes, h, tps_coefficients(sys, Vector(h))...)
    ro = o + R * SVector(0.1, 0.05, 1.5)
    d = R * normalize(SVector(-0.05, 0.1, -1.0))

    allocs(f, args...) = (f(args...); @allocated f(args...))

    for p in (quad, surf)
        t, x = @inferred intersection(p, ro, d)
        @test isfinite(t) && allocs(intersection, p, ro, d) == 0
        @test (@inferred normal(p, x)) isa SVector{3, Float64} && allocs(normal, p, x) == 0
        @test (@inferred chart(p, x)) isa SVector{2, Float64} && allocs(chart, p, x) == 0
        @test (@inferred bounds(p)) == Rect2(-0.5, -0.5, 1.0, 1.0)

        # Float32: the primitive and the ray converted
        p32 = p isa Quad ? Quad{Float32}(p) : HeightSurface{9, Float32}(p)
        @test isbits(p32)
        t32, x32 = @inferred intersection(p32, Float32.(ro), Float32.(d))
        @test t32 isa Float32 && x32 isa SVector{3, Float32} && allocs(intersection, p32, Float32.(ro), Float32.(d)) == 0
        @test abs(t32 - t) < 1.0f-5 && maximum(abs, x32 - x) < 1.0f-5
        @test (@inferred normal(p32, x32)) isa SVector{3, Float32} && (@inferred chart(p32, x32)) isa SVector{2, Float32}

        # Dual rays: ∂x/∂(o, d) against central differences
        J = ForwardDiff.jacobian(od -> intersection(p, od[SOneTo(3)], od[SVector(4, 5, 6)])[2], [ro; d])
        ε = 1.0e-6
        Jfd = reduce(hcat, [begin
            e = SVector{6}(ntuple(j -> j == k ? ε : 0.0, 6))
            (intersection(p, (([ro; d] + e)[SOneTo(3)]), ([ro; d] + e)[SVector(4, 5, 6)])[2] -
             intersection(p, (([ro; d] - e)[SOneTo(3)]), ([ro; d] - e)[SVector(4, 5, 6)])[2]) / 2ε
        end for k in 1:6])
        @test J ≈ Jfd atol = 1.0e-7
        od = ForwardDiff.Dual.(ro, 1.0), ForwardDiff.Dual.(d, 0.5)
        td, xd = @inferred intersection(p, od...)
        @test td isa ForwardDiff.Dual && allocs(intersection, p, od...) == 0
        # a Quad's normal does not depend on the point: plain; a HeightSurface's carries the partials
        @test (@inferred normal(p, xd)) isa (p isa Quad ? SVector{3, Float64} : SVector{3, <:ForwardDiff.Dual}) && allocs(normal, p, xd) == 0
    end

    # Dual heights: the hit's derivative in the nodal heights
    hit(hv) = intersection(HeightSurface(quad, nodes, SVector{9}(hv), tps_coefficients(sys, hv)...), ro, d)[2]
    J = ForwardDiff.jacobian(hit, Vector(h))
    ε = 1.0e-7
    Jfd = reduce(hcat, [(hit(Vector(h) + ε * (1:9 .== k)) - hit(Vector(h) - ε * (1:9 .== k))) / 2ε for k in 1:9])
    @test J ≈ Jfd atol = 1.0e-7
    hd = ForwardDiff.Dual.(Vector(h), 1.0)
    sd = HeightSurface(quad, nodes, SVector{9}(hd), tps_coefficients(sys, hd)...)
    @test isbits(sd) && allocs(intersection, sd, ro, d) == 0
    @test (@inferred intersection(sd, ro, d))[1] isa ForwardDiff.Dual
end

@testitem "surfaces: both primitives in a KernelAbstractions kernel (CPU)" begin
    using Test, StaticArrays, LinearAlgebra, Random
    using KernelAbstractions
    using KernelAbstractions: @kernel, @index, @Const
    using VisualGeometryToolkitCore
    using VisualGeometryToolkitCore: intersection, normal, ThinPlateSplineSystem, tps_coefficients, grid_nodes

    @kernel function surface_hits!(t, x, nrm, @Const(origins), @Const(directions), s)
        i = @index(Global)
        ti, xi = intersection(s, origins[i], directions[i])
        t[i] = ti
        x[i] = xi
        nrm[i] = normal(s, xi)
    end

    rng = Xoshiro(5)
    quad = Quad(SVector(0.0, 0.0, 0.0), SVector(1.0, 0.0, 0.0), SVector(0.0, 1.0, 0.0), Rect2(-0.5, -0.5, 1.0, 1.0))
    nodes = SVector{9}(grid_nodes(3; lo = -0.5, hi = 0.5))
    h = SVector{9}(0.01 .* randn(rng, 9))
    surf = HeightSurface(quad, nodes, h, tps_coefficients(ThinPlateSplineSystem(nodes), Vector(h))...)
    origins = [SVector(0.3 * randn(rng), 0.3 * randn(rng), 1.0 + rand(rng)) for _ in 1:257]
    directions = [normalize(SVector(0.4 * rand(rng) - 0.2, 0.4 * rand(rng) - 0.2, 0.0) - o) for o in origins]
    backend = KernelAbstractions.CPU()
    for T in (Float64, Float32), p in (quad, surf)
        pT = p isa Quad ? Quad{T}(p) : HeightSurface{9, T}(p)
        oT, dT = SVector{3, T}.(origins), SVector{3, T}.(directions)
        t = zeros(T, length(oT)); x = zeros(SVector{3, T}, length(oT)); nrm = similar(x)
        surface_hits!(backend, 64)(t, x, nrm, oT, dT, pT; ndrange = length(oT))
        KernelAbstractions.synchronize(backend)
        scalar = [intersection(pT, oi, di) for (oi, di) in zip(oT, dT)]
        @test t == first.(scalar) && x == last.(scalar)          # bit for bit (no ray here misses)
        @test nrm == [normal(pT, xi) for xi in x]
        @test all(isfinite, t)
    end
end
