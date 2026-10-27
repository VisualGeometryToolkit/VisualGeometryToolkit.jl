# =============================================================================
# Iterative Cox–de Boor B-spline kernels (numerics/bspline.jl, 0.14; hoisted
# from RayTracedCalibration) + the ChainRulesCore rrules of the extension.
#
# The BSplineKit cross-check is the hoisted downstream validation test: the
# knot vector is the calibration demo's ACTUAL θ knot vector (from
# RayTracedCalibration data/spline.json), evaluated here on the CPU backend —
# the exact code path GPU backends compile.
# =============================================================================

@testitem "bspline" begin
    using Test
    using VisualGeometryToolkitCore
    using VisualGeometryToolkitCore: bspline_4d_partial, bspline_basis_iterative, bspline_surface_partial, contract_basis_4d_weighted, evaluate_bspline_4d, evaluate_bspline_surface   # public, not exported
    using BSplineKit: BSplineBasis, BSplineOrder
    using FiniteDiff: FiniteDiff
    using LinearAlgebra: diag
    import ChainRulesCore

    # The partials are checked against FiniteDiff's central Jacobian in Float32
    # (the kernels' GPU element type), at relstep = 1f-3 rather than the default
    # cbrt(eps(Float32)) ≈ 4.9f-3: a degree-2 spline is only C¹ at its knots
    # (v = 1.0 below sits on one), so a central difference straddling a knot is
    # off by O(h·jump in S″). Measured max |fd − partial| over 200 random
    # control nets: surface 1.4e-2 / 2.9e-3 / 1.5e-3 and 4-D 9.2e-3 / 4.9e-4 /
    # 3.5e-4 at relstep 4.9f-3 / 1f-3 / 5f-4, Float32 rounding taking over below
    # (2f-4: 1.8e-3, 9.2e-4). 1f-3 is the step these checks always used.
    const FD_RELATIVE_H = 1.0f-3

    # The downstream demo's θ knot vector (degree 3, clamped, non-uniform).
    const THETA_KNOTS_64 = [
        0.0, 0.0, 0.0, 0.0,
        0.1582347789314227, 0.23735216839713408, 0.3164695578628454,
        0.39558694732855676, 0.47470433679426816, 0.5538217262599795,
        0.7120565051914022, 0.7120565051914022, 0.7120565051914022,
        0.7120565051914022,
    ]

    # Clamped uniform knot vector on [lo, hi] for n control points, degree p.
    function clamped_knots(n::Int, p::Int, lo::Float32, hi::Float32)
        interior = collect(range(lo, hi; length = n - p + 1))
        return Float32.(vcat(fill(lo, p), interior, fill(hi, p)))
    end

    @testset "basis agrees with BSplineKit on the demo θ knots" begin
        degree = 3
        order = degree + 1
        bk_basis = BSplineBasis(BSplineOrder(order), copy(THETA_KNOTS_64); augment = Val(false))
        n_basis = length(bk_basis)

        knots = Float32.(THETA_KNOTS_64)
        for θ in Float32[0.05, 0.1, 0.2, 0.35, 0.5, 0.6]
            N = bspline_basis_iterative(Float32[θ], degree, knots)
            vals = Float64[N[i][1] for i in 1:length(N)]

            last_i, bk_nonzero = bk_basis(Float64(θ))
            bk_full = zeros(Float64, n_basis)
            for k in 1:order
                bk_full[last_i - k + 1] = bk_nonzero[k]
            end

            @test length(vals) == n_basis
            @test sum(vals) ≈ 1.0 atol = 1.0e-5     # partition of unity
            @test sum(bk_full) ≈ 1.0 atol = 1.0e-12
            @test vals ≈ bk_full atol = 1.0e-5
        end
    end

    @testset "surface evaluation matches the naive tensor sum" begin
        p = 2
        n_u, n_v = 5, 4
        ku = clamped_knots(n_u, p, 0.0f0, 1.0f0)
        kv = clamped_knots(n_v, p, 0.0f0, 2.0f0)
        cp = randn(Float32, n_u, n_v)
        u = Float32[0.05, 0.3, 0.55, 0.7, 0.95]
        v = Float32[0.1, 1.0, 1.5, 1.9, 0.4]

        S = evaluate_bspline_surface(u, v, p, p, ku, kv, cp)

        N_u = bspline_basis_iterative(u, p, ku)
        N_v = bspline_basis_iterative(v, p, kv)
        naive = zeros(Float32, length(u))
        for i in 1:n_u, j in 1:n_v
            naive .+= N_u[i] .* N_v[j] .* cp[i, j]
        end
        @test S ≈ naive atol = 1.0f-6

        # partial derivatives against central finite differences: S[s] depends
        # on (u[s], v[s]) alone, so the partial is the Jacobian's diagonal
        for along in (1, 2)
            d = bspline_surface_partial(u, v, p, p, ku, kv, cp, along)
            J = along == 1 ?
                FiniteDiff.finite_difference_jacobian(x -> evaluate_bspline_surface(x, v, p, p, ku, kv, cp), u, Val(:central); relstep = FD_RELATIVE_H) :
                FiniteDiff.finite_difference_jacobian(x -> evaluate_bspline_surface(u, x, p, p, ku, kv, cp), v, Val(:central); relstep = FD_RELATIVE_H)
            @test d ≈ diag(J) atol = 5.0f-3
        end
    end

    @testset "4D spline: GEMM factorization matches the naive quadruple sum" begin
        p = 2
        n = (4, 3, 3, 4)
        lims = ((0.0f0, 1.0f0), (0.0f0, 1.0f0), (-1.0f0, 1.0f0), (-1.0f0, 1.0f0))
        ks = ntuple(i -> clamped_knots(n[i], p, lims[i][1], lims[i][2]), 4)
        cp = randn(Float32, n...)
        m = 7
        u1 = rand(Float32, m) .* 0.9f0 .+ 0.05f0
        u2 = rand(Float32, m) .* 0.9f0 .+ 0.05f0
        u3 = rand(Float32, m) .* 1.8f0 .- 0.9f0
        u4 = rand(Float32, m) .* 1.8f0 .- 0.9f0
        degrees = (p, p, p, p)

        S = evaluate_bspline_4d(u1, u2, u3, u4, degrees, ks..., cp)

        N1 = bspline_basis_iterative(u1, p, ks[1])
        N2 = bspline_basis_iterative(u2, p, ks[2])
        N3 = bspline_basis_iterative(u3, p, ks[3])
        N4 = bspline_basis_iterative(u4, p, ks[4])
        naive = zeros(Float32, m)
        for i in 1:n[1], j in 1:n[2], k in 1:n[3], l in 1:n[4]
            naive .+= N1[i] .* N2[j] .* N3[k] .* N4[l] .* cp[i, j, k, l]
        end
        @test S ≈ naive atol = 1.0f-5

        # weighted basis contraction against the naive accumulation
        w = randn(Float32, m)
        C = contract_basis_4d_weighted(u1, u2, u3, u4, degrees, ks..., w)
        naive_c = zeros(Float32, n...)
        for i in 1:n[1], j in 1:n[2], k in 1:n[3], l in 1:n[4]
            naive_c[i, j, k, l] = sum(w .* N1[i] .* N2[j] .* N3[k] .* N4[l])
        end
        @test C ≈ naive_c atol = 1.0f-4

        # 4D partials against central finite differences (the Jacobian's diagonal)
        us = (u1, u2, u3, u4)
        for dim in 1:4
            d = bspline_4d_partial(u1, u2, u3, u4, degrees, ks..., cp, dim)
            J = FiniteDiff.finite_difference_jacobian(
                x -> evaluate_bspline_4d(ntuple(i -> i == dim ? x : us[i], 4)..., degrees, ks..., cp),
                us[dim], Val(:central); relstep = FD_RELATIVE_H)
            @test d ≈ diag(J) atol = 5.0f-3
        end
    end

    @testset "ChainRulesCore rrules (package extension)" begin
        p = 2
        n_u, n_v = 4, 3
        ku = clamped_knots(n_u, p, 0.0f0, 1.0f0)
        kv = clamped_knots(n_v, p, 0.0f0, 1.0f0)
        cp = randn(Float32, n_u, n_v)
        u = Float32[0.1, 0.4, 0.8]
        v = Float32[0.2, 0.6, 0.9]

        out = ChainRulesCore.rrule(evaluate_bspline_surface, u, v, p, p, ku, kv, cp)
        @test out !== nothing
        S, pb = out
        @test S ≈ evaluate_bspline_surface(u, v, p, p, ku, kv, cp)
        ȳ = ones(Float32, length(u))
        _, du, dv, _, _, _, _, dcp = pb(ȳ)
        # CP cotangent: ∂(Σ S)/∂cp[i,j] = Σ_s N_u[i](s)·N_v[j](s). Σ S is LINEAR
        # in cp, so the central difference has no truncation error and FiniteDiff's
        # default Float32 step is the right one (measured max error vs the basis
        # sum 1.1e-4, against 5.8e-4 at the old h = 1f-3).
        fd = FiniteDiff.finite_difference_gradient(c -> sum(evaluate_bspline_surface(u, v, p, p, ku, kv, c)), cp, Val(:central))
        for i in 1:n_u, j in 1:n_v
            @test dcp[i, j] ≈ fd[i, j] atol = 5.0f-3
        end
        # input cotangents are the partials (ȳ = 1)
        @test du ≈ bspline_surface_partial(u, v, p, p, ku, kv, cp, 1) atol = 1.0f-5
        @test dv ≈ bspline_surface_partial(u, v, p, p, ku, kv, cp, 2) atol = 1.0f-5

        # 4D rrule: CP cotangent equals the weighted basis contraction
        n4 = (3, 3, 3, 3)
        ks = ntuple(_ -> clamped_knots(3, p, 0.0f0, 1.0f0), 4)
        cp4 = randn(Float32, n4...)
        m = 5
        uu = ntuple(_ -> rand(Float32, m) .* 0.9f0 .+ 0.05f0, 4)
        degrees = (p, p, p, p)
        out4 = ChainRulesCore.rrule(evaluate_bspline_4d, uu..., degrees, ks..., cp4)
        @test out4 !== nothing
        S4, pb4 = out4
        w = randn(Float32, m)
        tang = pb4(w)
        dcp4 = tang[end]
        @test dcp4 ≈ contract_basis_4d_weighted(uu..., degrees, ks..., w) atol = 1.0f-5
        for dim in 1:4
            @test tang[1 + dim] ≈
                bspline_4d_partial(uu..., degrees, ks..., cp4, dim) .* w atol = 1.0f-5
        end
    end
end

# The scalar basis, exact against the recursive Cox–de Boor definition (an
# independent reference: the textbook recursion with 0/0 := 0 and the last
# non-empty span closed on the right) at Float32 and Float64, with the
# partition of unity, and the broadcast form equal to it.
@testitem "bspline: the scalar basis against the recursion" begin
    using Test, StaticArrays
    using ForwardDiff: ForwardDiff
    using VisualGeometryToolkitCore
    using VisualGeometryToolkitCore: bspline_basis, clamped_uniform_knots, bspline_basis_iterative

    function cdb(i, p, x, k)
        if p == 0
            last = findlast(j -> k[j] < k[j + 1], 1:(length(k) - 1))
            return (k[i] <= x < k[i + 1]) || (x == k[end] && i == last) ? one(x) : zero(x)
        end
        a = k[i + p] - k[i] > 0 ? (x - k[i]) / (k[i + p] - k[i]) * cdb(i, p - 1, x, k) : zero(x)
        b = k[i + p + 1] - k[i + 1] > 0 ? (k[i + p + 1] - x) / (k[i + p + 1] - k[i + 1]) * cdb(i + 1, p - 1, x, k) : zero(x)
        return a + b
    end

    kn = clamped_uniform_knots(7, 3, -0.5, 0.5)
    @test kn == SVector(-0.5, -0.5, -0.5, -0.5, -0.25, 0.0, 0.25, 0.5, 0.5, 0.5, 0.5)
    @test_throws ArgumentError clamped_uniform_knots(3, 3, 0.0, 1.0)
    nonuniform = SVector(0.0, 0.0, 0.0, 0.1, 0.35, 0.4, 0.9, 1.0, 1.0, 1.0)   # degree 2, a near-double interior
    xs = vcat(collect(range(-0.5, 0.5; length = 101)), [-0.25, 0.0, 0.25])     # the knots themselves included
    for T in (Float64, Float32)
        for (k, p, pts) in ((kn, 3, xs), (nonuniform, 2, collect(range(0.0, 1.0; length = 101))), (clamped_uniform_knots(5, 1, 0.0, 2.0), 1, collect(range(0.0, 2.0; length = 41))))
            kT = SVector{length(k), T}(k)
            for x in T.(pts)
                N = bspline_basis(kT, Val(p), x)
                @test N isa SVector{length(k) - p - 1, T}
                @test N == [cdb(i, p, x, collect(kT)) for i in 1:(length(k) - p - 1)]     # exact: the same operations
                @test abs(sum(N) - 1) <= 4 * eps(T)
            end
        end
    end
    @test bspline_basis(kn, 3, 0.1) == bspline_basis(kn, Val(3), 0.1)
    @test bspline_basis(kn, Val(3), 0.7) == bspline_basis(kn, Val(3), 0.5)           # clamped onto the interval
    @test bspline_basis(kn, Val(3), 0.5)[end] == 1                                    # the closed right end
    # allocation-free and inferred; ForwardDiff: the derivative is the differenced-basis one
    @inferred bspline_basis(kn, Val(3), 0.1)
    alloc(k, x) = @allocated bspline_basis(k, Val(3), x)
    alloc(kn, 0.1)
    @test alloc(kn, 0.1) == 0
    d = ForwardDiff.derivative(x -> bspline_basis(kn, Val(3), x), 0.13)
    kd = SVector{9}(kn[2:(end - 1)])
    lower = bspline_basis(kd, Val(2), 0.13)
    ref = [3 * ((i >= 2 ? lower[i - 1] / (kn[i + 3] - kn[i]) : 0.0) - (i <= 6 ? lower[i] / (kn[i + 4] - kn[i + 1]) : 0.0)) for i in 1:7]
    @test d ≈ ref rtol = 1.0e-12
    # the broadcast form is the scalar basis at every point
    u = Float32.(range(-0.5, 0.5; length = 37))
    Nb = bspline_basis_iterative(u, 3, Float32.(collect(kn)))
    @test all(Nb[i][j] == bspline_basis(SVector{11, Float32}(kn), Val(3), u[j])[i] for i in 1:7, j in eachindex(u))
end
