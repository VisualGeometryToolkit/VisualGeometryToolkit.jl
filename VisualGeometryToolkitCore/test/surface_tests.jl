# =============================================================================
# The surface a ray meets (geometry/primitives/surfaces.jl): Quad and
# HeightSurface, and surface(::PosedTarget) (geometry/cameras/target_surfaces.jl).
#
# A Quad's intersection is ray_plane_intersection's arithmetic, bit for bit,
# NaN behind the ray's origin or outside its rectangle. A HeightSurface's is RadiometricCamera's
# `intersect_board` on its own warped test board: test/fixtures/height_surface/
# reference.jl holds what RadiometricCamera returned (generate.jl beside it).
# =============================================================================

@testitem "surfaces: a Quad is ray_plane_intersection, bounded" begin
    using Test, StaticArrays, LinearAlgebra, Random
    using Rotations: RotMatrix, QuatRotation
    using VisualGeometryToolkitCore
    using VisualGeometryToolkitCore: intersection, chart, bounds, normal, surface_point, ray_plane_intersection

    rng = Xoshiro(20261026)
    count = Dict(:hit => 0, :miss => 0, :behind => 0, :border => 0, :grazing => 0)   # mutated, never rebound (soft scope)
    for k in 1:100
        R = RotMatrix(QuatRotation(randn(rng, 4)...))
        o = SVector{3}(randn(rng, 3))
        rect = Rect2(randn(rng), randn(rng), 0.5 + rand(rng), 0.5 + rand(rng))
        e_u, e_v, n = R[:, 1], R[:, 2], R[:, 3]
        q = Quad(o, R, rect)
        @test isbits(q) && bounds(q) == rect && hash(q) == hash(Quad(o, R, rect)) && q == Quad(o, R, rect)
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
            elseif j % 7 == 0       # turned away: the plane is behind the origin
                d = -d
            end
            x1, x2, x3, tr = ray_plane_intersection(q.plane, ro, d[1], d[2], d[3])
            xr = SVector(x1, x2, x3)
            cr = (R' * (xr - o))[SOneTo(2)]          # the chart, computed independently
            margin = min(cr[1] - lo[1], lo[1] + w[1] - cr[1], cr[2] - lo[2], lo[2] + w[2] - cr[2])
            t, x = intersection(q, ro, d)
            if tr < 0
                count[:behind] += 1
                @test isnan(t) && all(isnan, x)                # a ray is a half-line
            elseif isfinite(margin) && margin > 1.0e-9
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
        @test surface_point(q, SVector(u, v)) ≈ o + u * e_u + v * e_v && chart(q, surface_point(q, SVector(u, v))) ≈ SVector(u, v)
        t, x = intersection(q, o + u * e_u + v * e_v + 2n, -n)
        @test t ≈ 2 && x ≈ o + u * e_u + v * e_v && chart(q, x) ≈ SVector(u, v)
    end
    @test count[:hit] + count[:miss] + count[:behind] + count[:border] == 10^4 && count[:hit] > 2500 &&
        count[:miss] > 1000 && count[:behind] > 1000 && count[:border] < 10 && count[:grazing] == 1000

    I3 = SMatrix{3, 3}(1.0I)
    # a ray exactly parallel to an axis-aligned quad divides by zero: a miss, not an error
    q = Quad(SVector(0.0, 0.0, 1.0), I3, Rect2(0.0, 0.0, 1.0, 1.0))
    t, x = intersection(q, SVector(0.5, 0.5, 2.0), SVector(1.0, 0.0, 0.0))
    @test isnan(t) && all(isnan, x)
    # the rectangle is closed: its edge is a hit; behind the origin is a miss
    t, x = intersection(q, SVector(1.0, 0.0, 2.0), SVector(0.0, 0.0, -1.0))
    @test t == 1.0 && x == SVector(1.0, 0.0, 1.0)
    @test all(isnan, intersection(q, SVector(0.5, 0.5, 2.0), SVector(0.0, 0.0, 1.0))[2])
    # the normal is geometric, e_u × e_v, whichever side the ray comes from
    @test normal(q, SVector(0.5, 0.5, 1.0)) == SVector(0.0, 0.0, 1.0)
    # refused: a non-orthonormal frame (to 1e-12), a left-handed one, a negative width, a singular affine
    @test_throws ArgumentError Quad(SVector(0.0, 0.0, 0.0), SMatrix{3, 3}(1.0, 0, 0, 1.0e-9, 1.0, 0, 0, 0, 1.0), Rect2(0.0, 0.0, 1.0, 1.0))
    @test_throws ArgumentError Quad(SVector(0.0, 0.0, 0.0), SMatrix{3, 3}(0.0, 1, 0, 1, 0, 0, 0, 0, 1), Rect2(0.0, 0.0, 1.0, 1.0))
    @test_throws ArgumentError Quad(SVector(0.0, 0.0, 0.0), I3, Rect2(0.0, 0.0, -1.0, 1.0))
    @test_throws ArgumentError Quad(SVector(0.0, 0.0, 0.0), I3, Rect2(0.0, 0.0, 1.0, 1.0), SMatrix{2, 2}(1.0, 1.0, 1.0, 1.0))

    # an in-plane affine: the chart is the design's, m = A c + b
    A = SMatrix{2, 2}(1.01, -0.003, 0.002, 0.99)
    b = SVector(0.004, -0.002)
    qa = Quad(SVector(0.0, 0.0, 1.0), I3, Rect2(0.0, 0.0, 1.0, 1.0), A, b)
    c = SVector(0.3, 0.7)
    mc = A * c + b
    t, x = intersection(qa, SVector(mc[1], mc[2], 3.0), SVector(0.0, 0.0, -1.0))
    @test t == 2.0 && x == SVector(mc[1], mc[2], 1.0)
    @test chart(qa, x) ≈ c atol = 1.0e-15
    @test surface_point(qa, c) ≈ x atol = 1.0e-15
    # the hit test is in the design chart: a plane point inside [0, 1]² whose design coordinates are not
    edge = A * SVector(1.0, 0.5) + b + SVector(0.001, 0.0)
    @test all(isnan, intersection(qa, SVector(edge[1], edge[2], 2.0), SVector(0.0, 0.0, -1.0))[2])
end

@testitem "surfaces: a HeightSurface meets RadiometricCamera's warped board" begin
    using Test, StaticArrays, LinearAlgebra
    using VisualGeometryToolkitCore
    using VisualGeometryToolkitCore: intersection, chart, bounds, normal, surface_point, ThinPlateSplineSystem,
        tps_coefficients, tps_evaluate, grid_nodes

    include(joinpath(pkgdir(VisualGeometryToolkitCore), "test", "fixtures", "height_surface", "reference.jl"))
    REF = HEIGHT_SURFACE_REFERENCE
    # RadiometricCamera's board is the square [-1, 1]² of z = 0, normal +z, its
    # chart [0, 1]²; here the chart is metric, the target's own (x, y), so the
    # nodes are -1 + 2 b and the heights the bending part at them (generate.jl)
    nodes = SVector{25}([REF.origin[SOneTo(2)] + 2 .* b for b in REF.nodes])
    h = SVector{25}(REF.bending_heights)
    w, a = tps_coefficients(ThinPlateSplineSystem(nodes), Vector(h))
    I3 = SMatrix{3, 3}(1.0I)
    quad = Quad(SVector(0.0, 0.0, 0.0), I3, Rect2(-1.0, -1.0, 2.0, 2.0))
    s = HeightSurface(quad, nodes, h, w, a)
    @test isbits(s) && s isa HeightSurface{25, Float64} && bounds(s) == bounds(quad)
    @test s == HeightSurface(quad, nodes, h, w, a) && hash(s) == hash(HeightSurface(quad, nodes, h, w, a))
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
    # misses: outside the rectangle, behind the origin, parallel, a NaN origin
    t, x = intersection(s, SVector(1.5, 0.0, 1.0), SVector(0.0, 0.0, -1.0))
    @test isnan(t) && all(isnan, x)
    @test all(isnan, intersection(s, SVector(0.1, 0.2, 1.0), SVector(0.0, 0.0, 1.0))[2])
    @test all(isnan, intersection(s, SVector(0.1, 0.2, 1.0), SVector(1.0, 0.0, 0.0))[2])
    @test all(isnan, intersection(s, SVector(NaN, 0.2, 1.0), SVector(0.0, 0.0, -1.0))[2])
    # surface_point is chart's inverse, on the surface
    for uv in (SVector(0.3, -0.4), SVector(-0.9, 0.8))
        p = surface_point(s, uv)
        @test chart(s, p) ≈ uv atol = 1.0e-15
        @test p[3] == tps_evaluate(nodes, w, a, uv)
        @test intersection(s, p + SVector(0.0, 0.0, 1.0), SVector(0.0, 0.0, -1.0))[2] ≈ p atol = 1.0e-15
    end
end

@testitem "surfaces: Newton continues past four iterations, or misses" begin
    using Test, StaticArrays, LinearAlgebra
    using VisualGeometryToolkitCore
    using VisualGeometryToolkitCore: intersection, surface_point, ThinPlateSplineSystem, tps_coefficients, tps_evaluate,
        tps_gradient, grid_nodes

    # a 10 cm bump on a 1 m quad (slope ≈ 0.4), a ray at 10° elevation aimed at its flank
    nodes = SVector{25}(grid_nodes(5; lo = 0.0, hi = 1.0))
    h = SVector{25}([i == 13 ? 0.1 : 0.0 for i in 1:25])
    w, a = tps_coefficients(ThinPlateSplineSystem(nodes), Vector(h))
    s = HeightSurface(Quad(SVector(0.0, 0.0, 0.0), SMatrix{3, 3}(1.0I), Rect2(0.0, 0.0, 1.0, 1.0)), nodes, h, w, a)
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

@testitem "surfaces: Float32 and ForwardDiff, inferred and allocation-free" begin
    using Test, StaticArrays, LinearAlgebra, Random
    using ForwardDiff
    using Rotations: RotMatrix, QuatRotation
    using VisualGeometryToolkitCore
    using VisualGeometryToolkitCore: intersection, chart, bounds, normal, ThinPlateSplineSystem, tps_coefficients, grid_nodes

    rng = Xoshiro(3)
    R = RotMatrix(QuatRotation(1.0, 0.1, -0.2, 0.05))
    o = SVector(0.1, -0.2, 0.3)
    quad = Quad(o, R, Rect2(-0.5, -0.5, 1.0, 1.0))
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
    quad = Quad(SVector(0.0, 0.0, 0.0), SMatrix{3, 3}(1.0I), Rect2(-0.5, -0.5, 1.0, 1.0))
    nodes = SVector{9}(grid_nodes(3; lo = -0.5, hi = 0.5))
    h = SVector{9}(0.01 .* randn(rng, 9))
    surf = HeightSurface(quad, nodes, h, tps_coefficients(ThinPlateSplineSystem(nodes), Vector(h))...)
    origins = [SVector(0.3 * randn(rng), 0.3 * randn(rng), 1.0 + rand(rng)) for _ in 1:257]
    # every fifth ray aimed off the rectangle, every seventh turned away: the miss branch
    directions = [normalize((k % 5 == 0 ? SVector(0.9, 0.0, 0.0) : SVector(0.4 * rand(rng) - 0.2, 0.4 * rand(rng) - 0.2, 0.0)) - o) *
                  (k % 7 == 0 ? -1 : 1) for (k, o) in enumerate(origins)]
    backend = KernelAbstractions.CPU()
    for T in (Float64, Float32), p in (quad, surf)
        pT = p isa Quad ? Quad{T}(p) : HeightSurface{9, T}(p)
        oT, dT = SVector{3, T}.(origins), SVector{3, T}.(directions)
        t = zeros(T, length(oT)); x = zeros(SVector{3, T}, length(oT)); nrm = similar(x)
        surface_hits!(backend, 64)(t, x, nrm, oT, dT, pT; ndrange = length(oT))
        KernelAbstractions.synchronize(backend)
        scalar = [intersection(pT, oi, di) for (oi, di) in zip(oT, dT)]
        @test isequal(t, first.(scalar)) && isequal(x, last.(scalar))          # bit for bit, NaN misses included
        @test isequal(nrm, [normal(pT, xi) for xi in x])
        @test count(isnan, t) == count(k -> k % 5 == 0 || k % 7 == 0, 1:257) && count(isfinite, t) > 150
    end
end

@testitem "surfaces: surface(::PosedTarget) by the calibration's kind" begin
    using Test, StaticArrays, LinearAlgebra, Random
    using Rotations: RotMatrix, QuatRotation
    using VisualGeometryToolkitCore
    using VisualGeometryToolkitCore: intersection, chart, bounds, normal, surface, surface_point, board, grid_nodes, mm, m,
        ThinPlateSplineSystem, tps_coefficients, tps_evaluate, with_unit

    rng = Xoshiro(17)
    R = RotMatrix(QuatRotation(1.0, 0.2, -0.1, 0.3))
    tr = SVector(0.2, -0.1, 1.5)
    X = EuclideanMap(R, tr)
    pts = [SVector(0.2 * i - 0.5, 0.1 * j - 0.3) for i in 0:5 for j in 0:6]   # [-0.5, 0.5] × [-0.3, 0.3]
    # the design points' box, grown by the rounding margin surface() adds
    function bbox(ps)
        lo = SVector(minimum(p[1] for p in ps), minimum(p[2] for p in ps))
        hi = SVector(maximum(p[1] for p in ps), maximum(p[2] for p in ps))
        δ = 16 * eps(Float64) * max(maximum(abs, lo), maximum(abs, hi), hi[1] - lo[1], hi[2] - lo[2])
        return Rect2(lo[1] - δ, lo[2] - δ, hi[1] - lo[1] + 2δ, hi[2] - lo[2] + 2δ)
    end
    nodes = grid_nodes(3; lo = -0.6, hi = 0.6)
    H = HeightField(nodes, 1.0e-3 .* randn(rng, 9))
    A = Affine([1.01 0.002; -0.003 0.99], [0.004, -0.002])

    @testset "Rigid: a Quad in the pose's frame" begin
        q = @inferred surface(PosedTarget(PlanarTarget(pts), X))
        @test q isa Quad{Float64} && bounds(q) == bbox(pts) && bounds(q) ≈ Rect2(-0.5, -0.3, 1.0, 0.6)
        @test q.origin == tr && q.e_u == R[:, 1] && q.e_v == R[:, 2] && normal(q) ≈ R[:, 3]
        @test all(chart(q, X(SVector(p[1], p[2], 0.0))) ≈ p for p in pts)          # the chart is the design's: TargetFrame (x, y) here
        # a ray at a design point meets it
        p = SVector(pts[9][1], pts[9][2], 0.0)
        t, x = intersection(q, X(p) + R * SVector(0.0, 0.0, 2.0), -R[:, 3])
        @test t ≈ 2 && x ≈ X(p)
    end

    @testset "Affine: a Quad whose chart is the design's" begin
        t = PlanarTarget(pts; calibration = A)
        q = surface(PosedTarget(t, X))
        @test q isa Quad && bounds(q) == bbox(pts) && q.A == A.A && q.b == A.b
        # a ray at a calibrated point meets it; its chart is the nominal point
        for (p, P) in zip(pts, points(t))
            XP = X(SVector(P[1], P[2], 0.0))
            tt, x = intersection(q, XP + R * SVector(0.05, -0.02, 1.0), -R * SVector(0.05, -0.02, 1.0))
            @test x ≈ XP atol = 1.0e-15
            @test chart(q, x) ≈ p atol = 1.0e-14
        end
    end

    @testset "HeightField: a HeightSurface with the field's heights and weights" begin
        s = @inferred surface(PosedTarget(PlanarTarget(pts; calibration = H), X))
        @test s isa HeightSurface{9, Float64}
        @test s.heights == H.h && s.nodes == SVector{9}(nodes) && s.w === H.w && s.a === H.a
        @test bounds(s) == bbox(pts) && s.quad == surface(PosedTarget(PlanarTarget(pts), X))
        # a hit lies on the surface: in TargetFrame its z is h(x, y)
        p = SVector(0.13, -0.07)
        t, x = intersection(s, X(SVector(p[1], p[2], 0.0)) + R * SVector(0.1, 0.0, 1.0), R * normalize(SVector(-0.1, 0.0, -1.0)))
        y = inv(X)(x)
        @test isfinite(t) && abs(y[3] - H(SVector(y[1], y[2]))[3]) < 1.0e-15 && chart(s, x) ≈ y[SOneTo(2)]
    end

    @testset "CalibrationChain(Affine, HeightField): a HeightSurface over the calibrated plane" begin
        t = PlanarTarget(pts; calibration = CalibrationChain(A, H))
        s = surface(PosedTarget(t, X))
        @test s isa HeightSurface{9, Float64} && s.heights == H.h && s.w === H.w
        @test bounds(s) == bbox(pts) && s.quad.A == A.A && s.quad.b == A.b
        # a ray aimed at a calibrated point (a 3-vector: the height applied) meets it; its chart is the nominal point
        for (p, P) in zip(pts, points(t))
            XP = X(P)
            dd = R * normalize(SVector(0.05, -0.02, -1.0))
            tt, x = intersection(s, XP - 0.7 * dd, dd)
            @test x ≈ XP atol = 1.0e-15
            @test chart(s, x) ≈ p atol = 1.0e-14
            @test surface_point(s, p) ≈ XP atol = 1.0e-15
        end
    end

    @testset "an in-plane ThinPlateSpline: not supported yet" begin
        T = ThinPlateSpline(nodes, 1.0e-4 .* randn(rng, 9), 1.0e-4 .* randn(rng, 9))
        @test_throws ArgumentError surface(PosedTarget(PlanarTarget(pts; calibration = T), X))
        @test_throws ArgumentError surface(PosedTarget(PlanarTarget(pts; calibration = CalibrationChain(A, T, H)), X))
    end

    @testset "a board bounds the surface" begin
        pts_mm = [p .* 1000 .* mm for p in pts]
        bd = Board(Size2(width = 1200mm, height = 800mm), [-600.0mm, -400.0mm])
        Am = Affine(A.A, A.b .* 1000 .* mm)                          # the same print error, physical
        t = PlanarTarget(pts_mm; calibration = Am, board = bd)
        @test board(t) === bd && Rect(bd) ≈ Rect2(-0.6, -0.4, 1.2, 0.8)
        s = surface(PosedTarget(t, EuclideanMap(R, tr .* 1000 .* mm)))
        @test s isa Quad && bounds(s) == Rect(bd)
        # a ray to the board's margin, outside the design points: a hit
        XP = X(SVector((A.A * SVector(0.55, 0.35) + A.b)..., 0.0))
        tt, x = intersection(s, XP + R * SVector(0.0, 0.0, 1.0), -R[:, 3])
        @test x ≈ XP atol = 1.0e-14
        @test chart(s, x) ≈ SVector(0.55, 0.35) atol = 1.0e-14
    end

    @testset "the RadiometricCamera fixture board, posed" begin
        include(joinpath(pkgdir(VisualGeometryToolkitCore), "test", "fixtures", "height_surface", "reference.jl"))
        REF = HEIGHT_SURFACE_REFERENCE
        rn = [REF.origin[SOneTo(2)] + 2 .* b for b in REF.nodes]
        Hr = HeightField(rn, REF.bending_heights)
        pt = PosedTarget(PlanarTarget(grid_nodes(3; lo = -1.0, hi = 1.0); calibration = Hr), EuclideanMap(one(RotMatrix{3, Float64}), SVector(0.0, 0.0, 0.0)))
        s = surface(pt)
        w, a = tps_coefficients(ThinPlateSplineSystem(rn), Vector(REF.bending_heights))
        quad = Quad(SVector(0.0, 0.0, 0.0), SMatrix{3, 3}(1.0I), bbox(grid_nodes(3; lo = -1.0, hi = 1.0)))
        @test s == HeightSurface(quad, SVector{25}(rn), SVector{25}(REF.bending_heights), w, a)
        hits = [intersection(s, o, d) for (o, d) in zip(REF.ray_origins, REF.ray_directions)]
        @test maximum(abs.(first.(hits) - REF.t)) <= 1.0e-15
        @test maximum(maximum(abs, x - r) for (x, r) in zip(last.(hits), REF.x)) <= 1.0e-15
    end

    @testset "a physical target: plain metres; an uncertain pose: its value" begin
        pts_mm = [p .* 1000 .* mm for p in pts]
        H_mm = HeightField([p .* 1000 .* mm for p in nodes], H.h .* 1000 .* mm)
        X_mm = EuclideanMap(R, tr .* 1000 .* mm)
        s = surface(PosedTarget(PlanarTarget(pts_mm; calibration = H_mm), X_mm))
        s_m = surface(PosedTarget(PlanarTarget(pts; calibration = H), X))
        @test s isa HeightSurface{9, Float64}
        @test s.quad.origin ≈ tr && s.nodes ≈ s_m.nodes && s.heights ≈ s_m.heights && bounds(s).origin ≈ bounds(s_m).origin
        o, d = tr + R * SVector(0.1, 0.0, 1.0), R * normalize(SVector(-0.1, 0.05, -1.0))
        @test intersection(s, o, d)[2] ≈ intersection(s_m, o, d)[2] atol = 1.0e-14
        Xu = Uncertain(X, CovMat(SMatrix{6, 6}(1.0e-6 * I)))
        @test surface(PosedTarget(PlanarTarget(pts), Xu)) == surface(PosedTarget(PlanarTarget(pts), X))
    end

    @testset "a point target has no surface" begin
        @test_throws MethodError surface(PosedTarget(PointTarget([SVector(0.0, 0.0, 0.0), SVector(1.0, 0.0, 0.0)]), X))
    end
end

@testitem "surfaces: a planar target's Board" begin
    using Test, StaticArrays
    using Rotations: RotMatrix
    using VisualGeometryToolkitCore
    using VisualGeometryToolkitCore: board, mm, m, px, with_parameters, parameter_vector
    using FlexUnits: DimensionError, ustrip

    bd = Board(Size2(width = 297mm, height = 210mm), [-10.0mm, -5.0mm])
    @test bd.origin == SVector(-0.01, -0.005) && ustrip(mm, bd.size.width) ≈ 297
    @test aspect_ratio(bd) ≈ 297 / 210 && Rect(bd) ≈ Rect2(-0.01, -0.005, 0.297, 0.21)
    @test Board(Size2(width = 0.297m, height = 0.21m), SVector(-0.01, -0.005)) == bd
    @test hash(Board(Size2(width = 0.297m, height = 0.21m), SVector(-0.01, -0.005))) == hash(bd)
    @test Rect(Board(Size2(width = 1.0m, height = 2.0m))) == Rect2(0.0, 0.0, 1.0, 2.0)
    @test_throws DimensionError Board(Size2(width = 297px, height = 210px))
    @test_throws ArgumentError Board(Size2(width = -1.0mm, height = 210mm))

    pts_mm = [SVector(0.0, 0.0) .* mm, SVector(250.0, 180.0) .* mm]
    t = PlanarTarget(pts_mm; board = bd)
    @test board(t) === bd && board(PlanarTarget(pts_mm)) === nothing
    @test t == PlanarTarget(pts_mm; board = bd) && hash(t) == hash(PlanarTarget(pts_mm; board = bd))
    @test t != PlanarTarget(pts_mm) && t != PlanarTarget(pts_mm; board = Board(Size2(width = 300mm, height = 210mm), [-10.0mm, -5.0mm]))
    @test board(resolve(PlanarTarget(:b; board = bd), pts_mm)) === bd
    tA = PlanarTarget(pts_mm; calibration = VisualGeometryToolkitCore.with_unit(Affine([1.0 0.0; 0.0 1.0], [0.0, 0.0]), mm), board = bd)
    @test board(with_parameters(tA, parameter_vector(tA))) === bd
    @test_throws DimensionError PlanarTarget([SVector(0.0, 0.0)]; board = bd)            # a board is lengths: the points too
    @test_throws ArgumentError PlanarTarget([SVector(300.0, 0.0) .* mm]; board = bd)     # off the board
    # ustrip moves the board with the points, and with_unit brings both back
    tp = ustrip(mm, t)
    @test board(tp) == Board(Size2(width = 297.0, height = 210.0), [-10.0, -5.0]) && Rect(board(tp)) ≈ Rect2(-10.0, -5.0, 297.0, 210.0)
    @test VisualGeometryToolkitCore.with_unit(tp, mm) == t
    tq = PlanarTarget(:b; board = bd)                                                    # unresolved: the board moves too
    @test board(ustrip(mm, tq)) == board(tp) && VisualGeometryToolkitCore.with_unit(ustrip(mm, tq), mm) == tq
    # a unitless board: plain numbers, for a plain design
    bu = Board(Size2(width = 297.0, height = 210.0), [-10.0, -5.0])
    @test !VisualGeometryToolkitCore.has_units(bu) && aspect_ratio(bu) ≈ 297 / 210 && bu.origin == SVector(-10.0, -5.0)
    tu = PlanarTarget([SVector(0.0, 0.0), SVector(250.0, 180.0)]; board = bu)
    @test board(tu) === bu && Rect(board(tu)) == Rect2(-10.0, -5.0, 297.0, 210.0)
    @test Board(Size2(width = 297, height = 210)) == Board(Size2(width = 297.0, height = 210.0))
    @test_throws DimensionError PlanarTarget(pts_mm; board = bu)                       # physical points, a plain board
    @test_throws ArgumentError Board(Size2(width = 297.0, height = 210.0), [-10.0mm, -5.0mm])   # a mix
    q = VisualGeometryToolkitCore.surface(PosedTarget(tu, EuclideanMap(one(RotMatrix{3, Float64}), SVector(0.0, 0.0, 1000.0))))
    @test VisualGeometryToolkitCore.bounds(q) == Rect(bu)
end
