# The homography transfer form: the constraint and its derivatives, the frozen
# whitening factor, and its exact Sampson rows (the batch rows of the one path)
# through the sl(3) chart — against ForwardDiff and finite differences.
@testitem "transfer_rows" begin
    import VisualGeometryToolkitCore as VgtkCore
    using Test
    using LinearAlgebra
    using StaticArrays
    using Random
    using ForwardDiff: ForwardDiff
    using FiniteDiff: FiniteDiff
    using VisualGeometryToolkitCore
    using VisualGeometryToolkitCore: TransferForm, whitening_factor, sampson_residuals!, sampson_jacobian_rows!,
        measurement_gradient_jacobian, retract, retract_jacobian, whiten, Csponds, withattrs

    rng = Xoshiro(3)
    H = HomographyMat(SMatrix{3, 3, Float64}(1.1, 0.02, 1.0e-4, -0.03, 0.95, 2.0e-4, 12.0, -7.0, 1.0))
    n = 40
    u₁ = [SVector(50 + 400rand(rng), 50 + 400rand(rng)) for _ in 1:n]
    u₂ = [(PerspectiveMap()(SMatrix{3, 3}(H) * SVector(u[1], u[2], 1.0)) + 0.5 * SVector(randn(rng), randn(rng))) for u in u₁]
    sqrtw = sqrt.(rand(rng, n))

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

    # (moved from VisualGeometryToolkitEstimation's homography_jacobian item: the math is VGC's)
    @testset "the algebraic form against ForwardDiff; whitened, its square is the Sampson distance²" begin
        Hs = SMatrix{3, 3}(H)
        for i in (1, 7, 23)
            u, v = u₁[i], u₂[i]
            g, Gx, Gθ = constraint_with_jacobians(H, u, v)
            # the owner's constraint g, as a function of the measurement and of vec(H)
            gz(z) = VgtkCore.constraint(HomographyMat{eltype(z)}(Tuple(Hs)), SVector(z[1], z[2]), SVector(z[3], z[4]))
            gθ(θ) = VgtkCore.constraint(HomographyMat{eltype(θ)}(Tuple(θ)), SVector{2, eltype(θ)}(u), SVector{2, eltype(θ)}(v))
            @test Gx ≈ ForwardDiff.jacobian(gz, SVector(u[1], u[2], v[1], v[2])) rtol = 1.0e-12
            @test Gθ ≈ ForwardDiff.jacobian(gθ, SVector{9}(Tuple(Hs))) rtol = 1.0e-12
            @test sum(abs2, cholesky(Symmetric(Gx * Gx')).L \ g) ≈ sampson_distance(H, u, v)^2 rtol = 1.0e-10
        end
    end

    @testset "a 3-vector bearing is refused, never read as its first two components" begin
        b = SVector(0.1, 0.2, 1.0)
        @test_throws MethodError constraint_with_jacobians(TransferForm(H), b, u₂[1])
        @test_throws MethodError constraint_with_jacobians(TransferForm(H), u₁[1], b)
    end

    @testset "whitening_factor: L⁻¹ of Σ_g, the same factor the Sampson rows apply" begin
        _, Gx, Gθ = constraint_with_jacobians(TransferForm(H), u₁[5], u₂[5])
        W = whitening_factor(Gx)
        @test W * (Gx * Gx') * W' ≈ I rtol = 1.0e-12
        A = Gx[:, SOneTo(2)]
        # RVG's frozen factor, to the last ulps: bit for bit on x86-64, but FMA on
        # aarch64 rounds the two evaluation orders differently
        @test W ≈ inv(SMatrix{2, 2}(cholesky(Symmetric(A * A' + I)).L)) rtol = 8eps()
        g = VgtkCore.constraint(TransferForm(H), u₁[5], u₂[5])
        # the frozen factor whitens the constraint to its Sampson residual at the model
        @test W * g ≈ sampson_residual(TransferForm(H), u₁[5], u₂[5]) rtol = 1.0e-12
        @test whitening_factor(zero(SMatrix{2, 4, Float64})) == zero(SMatrix{2, 2, Float64})   # degenerate: a zero factor
        @test all(isnan, whitening_factor(fill(NaN, SMatrix{2, 4, Float64})))
    end

    @testset "the exact rows: chart Jacobian (central differences, ForwardDiff), kernels = fast path" begin
        ch = HomographyManifold8(SMatrix{3, 3}(H))
        v = SVector(0.01, -0.02, 0.005, 0.01, -1.0e-3, 2.0e-3, 0.03, -0.02)
        Hv = retract(ch, v)
        xs = withattrs(Csponds(u₁, u₂); sqrt_weights = sqrtw)
        r = zeros(2n); J = zeros(2n, 8)
        sampson_residuals!(r, TransferForm(Hv), xs)
        sampson_jacobian_rows!(J, TransferForm(Hv), retract_jacobian(ch, v), xs)
        rv(w) = (out = zeros(eltype(w), 2n); sampson_residuals!(out, TransferForm(retract(ch, w)), xs); out)
        # the rows divide by h₃ at ~400 px, so their third derivative in the chart is
        # large: truncation-bound at FiniteDiff's default step, so a 1e-7 step
        fd = FiniteDiff.finite_difference_jacobian(w -> rv(SVector{8}(w)), v, Val(:central); relstep = 1.0e-7, absstep = 1.0e-7)
        @test J ≈ fd rtol = 1.0e-6 atol = 1.0e-9
        @test J ≈ ForwardDiff.jacobian(rv, v) rtol = 1.0e-10 atol = 1.0e-12
        for i in (1, 17)
            @test r[(2i - 1):(2i)] ≈ sqrtw[i] .* sampson_residual(TransferForm(Hv), u₁[i], u₂[i]) rtol = 1.0e-14
        end
        # the transfer form's ∂vec(G_x)/∂vec(H): ForwardDiff of G_x itself
        G(h) = vec(constraint_with_jacobians(TransferForm(HomographyMat(Tuple(h))), u₁[3], u₂[3])[2])
        @test measurement_gradient_jacobian(TransferForm(Hv), u₁[3], u₂[3]) ≈ ForwardDiff.jacobian(G, SVector{9}(Tuple(SMatrix{3, 3}(Hv)))) rtol = 1.0e-14
        # the device kernels, launched on KA.CPU() (the public functions take the CPU fast path)
        KA = VgtkCore.KA
        rk = zeros(2n); Jk = zeros(2n, 8)
        VgtkCore._row_kernel!(KA.CPU())(VgtkCore._sampson_residual_row!, rk, TransferForm(Hv), (u₁, u₂), NamedTuple(), nothing, sqrtw; ndrange = n)
        VgtkCore._row_kernel!(KA.CPU())(VgtkCore._sampson_jacobian_row!, Jk, TransferForm(Hv),
            VgtkCore._direction_rows(retract_jacobian(ch, v)), (u₁, u₂), NamedTuple(), nothing, sqrtw; ndrange = n)
        KA.synchronize(KA.CPU())
        # the drivers inline the same row function at different sites, so muladd
        # fusion may differ within a row: bounded relative to max|r| (where fusion
        # error lives), not per entry
        @test maximum(abs.(rk .- r)) <= 64 * eps(maximum(abs, r))
        @test Jk ≈ J rtol = 1.0e-12
    end
end
