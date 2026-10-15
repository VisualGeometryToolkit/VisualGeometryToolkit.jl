# The whitened homography transfer rows: the transfer-form constraint and its
# derivatives, the frozen whitening factor, and the row/Jacobian kernels through
# the sl(3) chart — against ForwardDiff and finite differences.
@testitem "transfer_rows" begin
    using Test
    using LinearAlgebra
    using StaticArrays
    using Random
    using ForwardDiff: ForwardDiff
    using FiniteDiff: FiniteDiff
    using VisualGeometryCore
    using VisualGeometryCore: TransferForm, whitening_factor, transfer_whitening!,
        transfer_rows!, transfer_jacobian_rows!, retract, retract_jacobian, whiten

    rng = Xoshiro(3)
    H = HomographyMat(SMatrix{3, 3, Float64}(1.1, 0.02, 1.0e-4, -0.03, 0.95, 2.0e-4, 12.0, -7.0, 1.0))
    n = 40
    u₁ = [SVector(50 + 400rand(rng), 50 + 400rand(rng)) for _ in 1:n]
    u₂ = [(PerspectiveMap()(SMatrix{3, 3}(H) * SVector(u[1], u[2], 1.0)) + 0.5 * SVector(randn(rng), randn(rng))) for u in u₁]
    sqrtw = sqrt.(rand(rng, 2n))

    @testset "the transfer form against ForwardDiff" begin
        u, v = u₁[3], u₂[3]
        g, Gx, Gθ = constraint_with_jacobians(TransferForm(H), u, v)
        # the stack's one perspective division, `PerspectiveMap()` (multiplies by 1/h₃)
        gf(z) = (h = SMatrix{3, 3}(H) * SVector(z[1], z[2], 1.0); PerspectiveMap()(h) - SVector(z[3], z[4]))
        @test g == gf(SVector(u[1], u[2], v[1], v[2]))
        @test Gx ≈ ForwardDiff.jacobian(gf, SVector(u[1], u[2], v[1], v[2])) rtol = 1.0e-12
        gθ(θ) = PerspectiveMap()(SMatrix{3, 3}(θ...) * SVector(u[1], u[2], 1.0)) - SVector(v[1], v[2])
        @test Gθ ≈ ForwardDiff.jacobian(gθ, SVector{9}(Tuple(H))) rtol = 1.0e-12
    end

    # (moved from RobustVisualGeometry's homography_jacobian item: the math is VGC's)
    @testset "the algebraic form against ForwardDiff; whitened, its square is the Sampson distance²" begin
        Hs = SMatrix{3, 3}(H)
        for i in (1, 7, 23)
            u, v = u₁[i], u₂[i]
            g, Gx, Gθ = constraint_with_jacobians(H, u, v)
            # the owner's constraint g, as a function of the measurement and of vec(H)
            gz(z) = constraint_with_jacobians(HomographyMat{eltype(z)}(Tuple(Hs)), SVector(z[1], z[2]), SVector(z[3], z[4]))[1]
            gθ(θ) = constraint_with_jacobians(HomographyMat{eltype(θ)}(Tuple(θ)), SVector{2, eltype(θ)}(u), SVector{2, eltype(θ)}(v))[1]
            @test Gx ≈ ForwardDiff.jacobian(gz, SVector(u[1], u[2], v[1], v[2])) rtol = 1.0e-12
            @test Gθ ≈ ForwardDiff.jacobian(gθ, SVector{9}(Tuple(Hs))) rtol = 1.0e-12
            @test sum(abs2, whiten(g, Gx, Gθ)[1]) ≈ sampson_distance(H, u, v)^2 rtol = 1.0e-10
        end
    end

    @testset "a 3-vector bearing is refused, never read as its first two components" begin
        b = SVector(0.1, 0.2, 1.0)
        @test_throws MethodError constraint_with_jacobians(TransferForm(H), b, u₂[1])
        @test_throws MethodError constraint_with_jacobians(TransferForm(H), u₁[1], b)
        @test_throws MethodError transfer_jacobian_rows!(zeros(2, 1), H, (zero(SMatrix{3, 3, Float64}),),
            [one(SMatrix{2, 2, Float64})], [b], ones(2))
    end

    @testset "whitening_factor: L⁻¹ of Σ_g, the same factor whiten applies" begin
        _, Gx, Gθ = constraint_with_jacobians(TransferForm(H), u₁[5], u₂[5])
        W = whitening_factor(Gx)
        @test W * (Gx * Gx') * W' ≈ I rtol = 1.0e-12
        A = Gx[:, SOneTo(2)]
        @test W == inv(SMatrix{2, 2}(cholesky(Symmetric(A * A' + I)).L))    # RVG's frozen factor, bit for bit
        g, _, _ = constraint_with_jacobians(TransferForm(H), u₁[5], u₂[5])
        r, Jw, _ = whiten(g, Gx, Gθ)
        @test W * g ≈ r rtol = 1.0e-12
        @test W * Gθ ≈ Jw rtol = 1.0e-12
        @test whitening_factor(zero(SMatrix{2, 4, Float64})) == zero(SMatrix{2, 2, Float64})   # degenerate: a zero factor
        @test all(isnan, whitening_factor(fill(NaN, SMatrix{2, 4, Float64})))
    end

    @testset "rows and chart Jacobian (central differences), kernels = fast path" begin
        ch = HomographyManifold8(SMatrix{3, 3}(H))
        v = SVector(0.01, -0.02, 0.005, 0.01, -1.0e-3, 2.0e-3, 0.03, -0.02)
        Hv = retract(ch, v)
        dH = VisualGeometryCore.retract_directions(ch, v)
        W = Vector{SMatrix{2, 2, Float64, 4}}(undef, n)
        transfer_whitening!(W, H, u₁)                          # frozen at the seed
        r = zeros(2n); J = zeros(2n, 8)
        transfer_rows!(r, Hv, W, u₁, u₂, sqrtw)
        transfer_jacobian_rows!(J, Hv, dH, W, u₁, sqrtw)
        rv(w) = (Hw = retract(ch, w); out = zeros(2n); transfer_rows!(out, Hw, W, u₁, u₂, sqrtw); out)
        # the rows divide by h₃ at ~400 px, so their third derivative in the chart is
        # large: truncation-bound, FiniteDiff's default step (6e-6) misses by 6.6e-6
        # in norm, a 1e-7 step by 1.8e-9
        fd = FiniteDiff.finite_difference_jacobian(rv, v, Val(:central); relstep = 1.0e-7, absstep = 1.0e-7)
        @test J ≈ fd rtol = 1.0e-6 atol = 1.0e-9
        for i in (1, 17)
            g, _, _ = constraint_with_jacobians(TransferForm(Hv), u₁[i], u₂[i])
            @test r[(2i - 1):(2i)] ≈ sqrtw[(2i - 1):(2i)] .* (W[i] * g) rtol = 1.0e-14   # muladd fuses per inlining site
        end
        # the device kernels, launched on KA.CPU() (the public functions take the CPU fast path)
        KA = VisualGeometryCore.KA
        Wk = similar(W); rk = zeros(2n); Jk = zeros(2n, 8)
        VisualGeometryCore._row_kernel!(KA.CPU())(VisualGeometryCore._transfer_whitening_row!, Wk, H, u₁; ndrange = n)
        VisualGeometryCore._row_kernel!(KA.CPU())(VisualGeometryCore._transfer_rows!, rk, Hv, W, u₁, u₂, sqrtw; ndrange = n)
        VisualGeometryCore._row_kernel!(KA.CPU())(VisualGeometryCore._transfer_jacobian_rows!, Jk, Hv, dH, W, u₁, sqrtw; ndrange = n)
        KA.synchronize(KA.CPU())
        # ELEMENTWISE, AT THE MEASURED MARGIN. Each side runs the same per-row
        # scalar function under a different driver (KA kernel vs the CPU fast
        # path of `_foreach_row!`); the drivers inline it at different sites, so
        # muladd fusion differs within a row. Measured 2026-09-30: at most 3 ulps
        # over the 160 whitening entries and 1 ulp over the 80 residuals (6
        # differ). Re-measured 2026-10-05 (Julia 1.12.6, the 0.30.0 Manifest):
        # 3 ulps on the whitening, 18 ulps on ONE residual (r[73] = -13.73, an
        # absolute 3.6e-16 of max|r|; the next is 6). If this fails, MEASURE
        # the margin again and loosen to it — never re-run until green.
        ulps(a, b) = abs(a - b) / eps(max(abs(a), abs(b)))
        @test all(i -> all(ulps.(Wk[i], W[i]) .<= 4), 1:n)
        @test all(ulps.(rk, r) .<= 18)
        @test Jk ≈ J rtol = 1.0e-14
    end
end
