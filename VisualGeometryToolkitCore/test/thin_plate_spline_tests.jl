# =============================================================================
# Thin-plate spline (numerics/thin_plate_spline.jl; hoisted from
# RadiometricCamera src/camera/tps.jl and src/board/warping.jl).
#
# Bit identity: test/fixtures/thin_plate_spline/reference.jl holds, as
# literals, what RadiometricCamera's OWN functions return for fixed nodal
# values at 20 query points (on nodes, inside the hull, a hair off a node and
# far outside it), on grid_nodes(3) and grid_nodes(5): U, the coefficients,
# the evaluation, nodal_weights and the bending energy, its Float32 focus
# path, and on its 5 × 5 height grid over [0, 1]² `board_height` and
# `board_height_grad` — the gradient's source. generate.jl beside it wrote
# it, in RadiometricCamera's environment. Every Float64 value is compared
# with `==`.
# =============================================================================

@testitem "thin_plate_spline" begin
    using Test
    using VisualGeometryToolkitCore
    using VisualGeometryToolkitCore: ThinPlateSpline, tps_radial_basis, grid_nodes, tps_coefficients,
        tps_evaluate, tps_gradient, nodal_weights, bending_energy   # public, not exported
    using StaticArrays: SVector
    using LinearAlgebra: norm
    using ForwardDiff
    using KernelAbstractions
    using KernelAbstractions: @kernel, @index, @Const

    include(joinpath(pkgdir(VisualGeometryToolkitCore), "test", "fixtures", "thin_plate_spline", "reference.jl"))
    const REF = TPS_REFERENCE
    const POINTS = REF.points

    # Σ of the magnitudes of the evaluation's terms: the scale its rounding
    # (and its cancellation far outside the hull) is measured against.
    term_scale(nodes, w, a, p) =
        abs(a[1]) + abs(a[2] * p[1]) + abs(a[3] * p[2]) + sum(i -> abs(w[i] * tps_radial_basis(p, nodes[i])), eachindex(w))
    gradient_scale(nodes, w, a, p) =
        abs(a[2]) + abs(a[3]) + sum(i -> abs(w[i]) * norm(p - nodes[i]) * (abs(log(max(norm(p - nodes[i])^2, floatmin()))) + 1), eachindex(w))

    @testset "bit identity with RadiometricCamera: $name" for (name, n) in (("grid_3", 3), ("grid_5", 5))
        ref = getproperty(REF, Symbol(name))
        nodes = grid_nodes(n)
        @test nodes == ref.nodes
        tps = ThinPlateSpline(nodes)
        w, a = tps_coefficients(tps, ref.values)
        @test collect(w) == ref.w
        @test collect(a) == ref.a
        @test [tps_evaluate(tps, w, a, p) for p in POINTS] == ref.evaluate
        @test [tps_evaluate(tps.nodes, w, a, p) for p in POINTS] == ref.evaluate
        @test [collect(nodal_weights(tps, p)) for p in POINTS] == ref.nodal_weights
        @test bending_energy(tps, ref.values) == ref.bending_energy
        @test [[tps_radial_basis(p, q) for q in tps.nodes] for p in POINTS] == ref.radial_basis

        # RadiometricCamera's Float32 path: the Float64 solver, Float32 (nodes, w, a, 𝐩).
        w32, a32 = tps_coefficients(tps, Float32.(ref.values))
        @test eltype(w32) == Float32 && eltype(a32) == Float32
        @test collect(w32) == ref.w_f32
        @test collect(a32) == ref.a_f32
        nodes32 = SVector{n * n}(SVector{2, Float32}.(tps.nodes))
        f32 = [tps_evaluate(nodes32, w32, a32, SVector{2, Float32}(p)) for p in POINTS]
        @test f32 isa Vector{Float32}
        @test f32 == ref.evaluate_f32
    end

    @testset "bit identity with RadiometricCamera: height grid gradient" begin
        ref = REF.height
        nodes = grid_nodes(5; lo = 0.0, hi = 1.0)
        @test nodes == ref.nodes
        snodes = ThinPlateSpline(nodes).nodes
        w = SVector{25}(ref.w)
        a0 = zero(SVector{3, Float64})   # the height field is the spline without its affine part
        @test [tps_evaluate(snodes, w, a0, b) for b in ref.points] == ref.height
        @test [tps_gradient(snodes, w, a0, b) for b in ref.points] == ref.gradient
    end

    @testset "radial basis at a node" begin
        p = SVector(0.25, -0.5)
        @test tps_radial_basis(p, p) === 0.0
        @test tps_radial_basis(SVector(1.0f0, 2.0f0), SVector(1.0f0, 2.0f0)) === 0.0f0
        d = ForwardDiff.gradient(q -> tps_radial_basis(q, p), p)
        @test d == [0.0, 0.0]   # zero partials, not NaN
    end

    @testset "Float32 within eps of Float64: $name" for (name, n) in (("grid_3", 3), ("grid_5", 5))
        ref = getproperty(REF, Symbol(name))
        tps = ThinPlateSpline(grid_nodes(n))
        w32, a32 = tps_coefficients(tps, Float32.(ref.values))
        nodes32 = SVector{n * n}(SVector{2, Float32}.(tps.nodes))
        # The same (Float32-rounded) inputs evaluated in Float64: what differs is
        # the arithmetic alone, bounded by the summation's rounding.
        nodes64 = SVector{n * n}(SVector{2, Float64}.(nodes32))
        w64, a64 = Float64.(w32), Float64.(a32)
        for p in POINTS
            p32 = SVector{2, Float32}(p)
            p64 = SVector{2, Float64}(p32)
            f32 = tps_evaluate(nodes32, w32, a32, p32)
            f64 = tps_evaluate(nodes64, w64, a64, p64)
            bound = 2 * (n * n + 3) * eps(Float32) * term_scale(nodes64, w64, a64, p64)
            @test abs(f32 - f64) <= bound
            g32 = tps_gradient(nodes32, w32, a32, p32)
            g64 = tps_gradient(nodes64, w64, a64, p64)
            @test g32 isa SVector{2, Float32}
            @test norm(g32 - g64) <= 4 * (n * n + 3) * eps(Float32) * gradient_scale(nodes64, w64, a64, p64)
        end
    end

    @testset "ForwardDiff: the analytic gradient and nodal_weights: $name" for (name, n) in (("grid_3", 3), ("grid_5", 5))
        ref = getproperty(REF, Symbol(name))
        tps = ThinPlateSpline(grid_nodes(n))
        w, a = tps_coefficients(tps, ref.values)
        for p in POINTS
            # Through tps_evaluate with Dual coordinates (on a node too: the r² > 0 guard).
            g_ad = ForwardDiff.gradient(q -> tps_evaluate(tps.nodes, w, a, SVector(q[1], q[2])), collect(p))
            g = tps_gradient(tps, w, a, p)
            @test all(isfinite, g_ad)
            @test norm(g_ad - g) <= 64 * eps() * gradient_scale(tps.nodes, w, a, p)

            # ∂f(𝐩)/∂valuesᵢ through tps_coefficients with Dual values.
            nw_ad = ForwardDiff.gradient(v -> tps_evaluate(tps.nodes, tps_coefficients(tps, v)..., p), ref.values)
            φ_scale = (sum(q -> abs(tps_radial_basis(p, q)), tps.nodes) + 1 + abs(p[1]) + abs(p[2])) * maximum(abs, tps.coeff)
            @test norm(nw_ad - nodal_weights(tps, p), Inf) <= 64 * eps() * φ_scale
        end
        # Interpolation: the spline passes through its nodal values.
        @test [tps_evaluate(tps, w, a, q) for q in tps.nodes] ≈ ref.values rtol = 1.0e-12
        # Affine values bend nothing.
        affine = [1 + 2q[1] - 3q[2] for q in tps.nodes]
        @test abs(bending_energy(tps, affine)) <= 1.0e-10
    end

    @testset "isbits and allocation-free" begin
        tps = ThinPlateSpline(grid_nodes(5))
        w, a = tps_coefficients(tps, REF.grid_5.values)
        nodes, p = tps.nodes, SVector(0.123, -0.377)
        @test isbitstype(typeof(nodes)) && isbitstype(typeof(w)) && isbitstype(typeof(a))
        allocations(f, args...) = (f(args...); @allocated f(args...))
        @test allocations(tps_evaluate, nodes, w, a, p) == 0
        @test allocations(tps_gradient, nodes, w, a, p) == 0
        nodes32 = SVector{25}(SVector{2, Float32}.(nodes))
        w32, a32 = tps_coefficients(tps, Float32.(REF.grid_5.values))
        @test allocations(tps_evaluate, nodes32, w32, a32, SVector{2, Float32}(p)) == 0
        @test allocations(tps_gradient, nodes32, w32, a32, SVector{2, Float32}(p)) == 0
    end

    @testset "inside a KernelAbstractions kernel (CPU backend)" begin
        @kernel function tps_points_kernel!(f, g, @Const(points), nodes, w, a)
            i = @index(Global)
            f[i] = tps_evaluate(nodes, w, a, points[i])
            g[i] = tps_gradient(nodes, w, a, points[i])
        end
        tps = ThinPlateSpline(grid_nodes(5))
        for T in (Float64, Float32)
            w, a = tps_coefficients(tps, T.(REF.grid_5.values))
            nodes = SVector{25}(SVector{2, T}.(tps.nodes))
            points = SVector{2, T}.(POINTS)
            f = zeros(T, length(points))
            g = zeros(SVector{2, T}, length(points))
            backend = KernelAbstractions.CPU()
            tps_points_kernel!(backend)(f, g, points, nodes, w, a; ndrange = length(points))
            KernelAbstractions.synchronize(backend)
            @test f == [tps_evaluate(nodes, w, a, p) for p in points]
            @test g == [tps_gradient(nodes, w, a, p) for p in points]
        end
        @test [tps_evaluate(tps, tps_coefficients(tps, REF.grid_5.values)..., p) for p in POINTS] == REF.grid_5.evaluate
    end
end
