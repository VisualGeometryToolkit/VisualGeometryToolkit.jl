# =============================================================================
# Thin-plate spline (numerics/thin_plate_spline.jl; hoisted from
# RadiometricCamera src/camera/tps.jl and src/board/warping.jl).
#
# Against RadiometricCamera: test/fixtures/thin_plate_spline/reference.jl
# holds, as literals, what RadiometricCamera's OWN functions return for fixed
# nodal values at 20 query points (on nodes, inside the hull, a hair off a
# node and far outside it), on grid_nodes(3) and grid_nodes(5): U, the
# coefficients, the evaluation, nodal_weights and the bending energy, its
# Float32 focus path, and on its 5 × 5 height grid over [0, 1]²
# `board_height` and `board_height_grad` — the gradient's source.
# generate.jl beside it wrote it, in RadiometricCamera's environment.
#
# What is pinned exactly and what is bounded. Everything AFTER (w, a) — the
# radial basis, the evaluation, the gradient — is plain scalar arithmetic and
# is compared `==`, fed the fixture's own w and a. The coefficients, the
# nodal weights and the bending energy go through `inv` and BLAS products, so
# their last bits belong to the BLAS build (OpenBLAS picks its kernels by
# CPU): they are compared within bounds measured on the recording machine
# against two other solvers of the same system (Julia's generic LU, and
# BigFloat rounded), whose largest deviations were, in eps() × the sum of
# the magnitudes of the terms: w and a 9.3, nodal_weights 27.9, bending
# energy 0.17. The bounds are 16, 32 and 1. On the recording machine the
# deviation is zero; TPS_BITWISE=1 turns those comparisons back into `==`.
# The gradient with a nonzero affine part has no RadiometricCamera
# counterpart: it is bit-pinned only for a = 0 (the height grid), and checked
# against ForwardDiff otherwise.
# =============================================================================

@testitem "thin_plate_spline" begin
    using Test
    using VisualGeometryToolkitCore
    using VisualGeometryToolkitCore: ThinPlateSplineSystem, tps_radial_basis, grid_nodes, tps_coefficients,
        tps_evaluate, tps_gradient, nodal_weights, bending_energy   # public, not exported
    using StaticArrays: SVector
    using LinearAlgebra: norm
    using ForwardDiff
    using KernelAbstractions
    using KernelAbstractions: @kernel, @index, @Const

    include(joinpath(pkgdir(VisualGeometryToolkitCore), "test", "fixtures", "thin_plate_spline", "reference.jl"))
    REF = TPS_REFERENCE
    POINTS = REF.points
    BITWISE = get(ENV, "TPS_BITWISE", "") == "1"   # the recording machine: the BLAS-dependent values `==` too

    # Within `c` eps() of the sum of the term magnitudes `scale` (elementwise),
    # or `==` under TPS_BITWISE.
    within(x, ref, scale, c) = BITWISE ? x == ref : all(abs.(x .- ref) .<= c * eps() .* scale)

    # Σ of the magnitudes of the evaluation's terms: the scale its rounding
    # (and its cancellation far outside the hull) is measured against.
    term_scale(nodes, w, a, p) =
        abs(a[1]) + abs(a[2] * p[1]) + abs(a[3] * p[2]) + sum(i -> abs(w[i] * tps_radial_basis(p, nodes[i])), eachindex(w))
    gradient_scale(nodes, w, a, p) =
        abs(a[2]) + abs(a[3]) + sum(i -> abs(w[i]) * norm(p - nodes[i]) * (abs(log(max(norm(p - nodes[i])^2, floatmin()))) + 1), eachindex(w))

    @testset "against RadiometricCamera: $name" for (name, n) in (("grid_3", 3), ("grid_5", 5))
        ref = getproperty(REF, Symbol(name))
        N = n * n
        nodes = grid_nodes(n)
        @test nodes == ref.nodes
        tps = ThinPlateSplineSystem(nodes)
        t = ref.values

        # Through `inv` and BLAS: bounded (see the header).
        w, a = tps_coefficients(tps, t)
        coeff_scale = [sum(abs.(tps.coeff[r, :]) .* abs.(t)) for r in 1:(N + 3)]
        @test within(vcat(collect(w), collect(a)), vcat(ref.w, ref.a), coeff_scale, 16)
        for (k, p) in enumerate(POINTS)
            φ = [[abs(tps_radial_basis(p, q)) for q in tps.nodes]; 1.0; abs(p[1]); abs(p[2])]
            @test within(collect(nodal_weights(tps, p)), ref.nodal_weights[k], [sum(abs.(tps.coeff[:, i]) .* φ) for i in 1:N], 32)
        end
        @test within(bending_energy(tps, t), ref.bending_energy, sum(abs.(t) .* (abs.(tps.Q) * abs.(t))), 1)

        # Plain arithmetic, fed the fixture's (w, a): bit-identical.
        rw, ra = SVector{N}(ref.w), SVector{3}(ref.a)
        @test [tps_evaluate(tps, rw, ra, p) for p in POINTS] == ref.evaluate
        @test [tps_evaluate(tps.nodes, rw, ra, p) for p in POINTS] == ref.evaluate
        @test [[tps_radial_basis(p, q) for q in tps.nodes] for p in POINTS] == ref.radial_basis

        # RadiometricCamera's Float32 path: the Float64 solver, Float32 (nodes, w, a, 𝐩).
        # The coefficients are a Float64 BLAS product rounded to Float32: within
        # one Float32 ulp plus the Float64 bound.
        w32, a32 = tps_coefficients(tps, Float32.(t))
        @test eltype(w32) == Float32 && eltype(a32) == Float32
        ref32 = vcat(ref.w_f32, ref.a_f32)
        coeff_scale32 = [sum(abs.(tps.coeff[r, :]) .* abs.(Float32.(t))) for r in 1:(N + 3)]
        @test BITWISE ? vcat(collect(w32), collect(a32)) == ref32 :
            all(abs.(Float64.(vcat(collect(w32), collect(a32))) .- ref32) .<= eps.(ref32) .+ 16 * eps() .* coeff_scale32)
        nodes32 = SVector{N}(SVector{2, Float32}.(tps.nodes))
        f32 = [tps_evaluate(nodes32, SVector{N}(ref.w_f32), SVector{3}(ref.a_f32), SVector{2, Float32}(p)) for p in POINTS]
        @test f32 isa Vector{Float32}
        @test f32 == ref.evaluate_f32
    end

    # The gradient is bit-pinned here only, with a = 0: RadiometricCamera has no
    # gradient with an affine part (ForwardDiff checks that case below).
    @testset "bit identity with RadiometricCamera: height grid gradient" begin
        ref = REF.height
        nodes = grid_nodes(5; lo = 0.0, hi = 1.0)
        @test nodes == ref.nodes
        snodes = ThinPlateSplineSystem(nodes).nodes
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
        tps = ThinPlateSplineSystem(grid_nodes(n))
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
        tps = ThinPlateSplineSystem(grid_nodes(n))
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
        # Interpolation: the spline interpolates its nodal values.
        @test [tps_evaluate(tps, w, a, q) for q in tps.nodes] ≈ ref.values rtol = 1.0e-12
        # Affine values bend nothing.
        affine = [1 + 2q[1] - 3q[2] for q in tps.nodes]
        @test abs(bending_energy(tps, affine)) <= 1.0e-10
    end

    @testset "isbits and allocation-free" begin
        tps = ThinPlateSplineSystem(grid_nodes(5))
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

    @testset "runs in a KernelAbstractions kernel on the CPU backend" begin
        @kernel function tps_points_kernel!(f, g, @Const(points), nodes, w, a)
            i = @index(Global)
            f[i] = tps_evaluate(nodes, w, a, points[i])
            g[i] = tps_gradient(nodes, w, a, points[i])
        end
        tps = ThinPlateSplineSystem(grid_nodes(5))
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
    end
end
