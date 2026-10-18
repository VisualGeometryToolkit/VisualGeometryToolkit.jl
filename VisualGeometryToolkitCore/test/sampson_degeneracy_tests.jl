# One degeneracy rule for the Sampson distance, its signed form and whitening:
# where the measurement gradient vanishes to rounding the distance is 0 when the
# constraint vanishes with it (the correspondence satisfies it exactly) and Inf when
# it does not; the signed distance is ±Inf, and `whiten` returns that residual with
# zero Jacobian rows. Every test is relative, so a homogeneous model's scale never
# decides it — the absolute `eps` thresholds called every correspondence degenerate
# for a model scaled by 1e-9, and the signed distance's `+ eps` floor moved it.
@testitem "sampson_degeneracy" begin
    import VisualGeometryToolkitCore as VgtkCore
    using Test
    using VisualGeometryToolkitCore
    using VisualGeometryToolkitCore: FundMat, HomographyMat, sampson_distance, signed_sampson_distance,
        whiten, constraint_with_jacobians
    using StaticArrays: SMatrix, SVector
    using LinearAlgebra: I, norm

    @testset "epipolar: 0 at the epipoles, Inf for a line at infinity" begin
        F = FundMat(SMatrix{3, 3, Float64}(0, 1, 0, -1, 0, 0, 0, 0, 0))   # [t]× with t = (0, 0, 1): both epipoles at the origin
        o = SVector(0.0, 0.0)
        @test sampson_distance(F, o, o) == 0.0
        @test sampson_distance(F, SVector(3.0, 4.0), SVector(6.0, 8.0)) == 0.0    # on its epipolar line: zero, and not degenerate
        @test sampson_distance(F, SVector(3.0, 4.0), SVector(4.0, 3.0)) > 0
        # both epipolar lines at infinity with a nonzero constraint: no nearby pair satisfies it
        Finf = FundMat(SMatrix{3, 3, Float64}(0, 1, 0, -1, 0, 0, 0, 0, 1))
        @test sampson_distance(Finf, o, o) == Inf
    end

    @testset "scale-free: a model's homogeneous scale changes nothing" begin
        F = FundMat(SMatrix{3, 3, Float64}(1.0e-6, -2.0e-5, 3.0e-3, 4.0e-5, 1.0e-6, -2.0e-2, -1.0e-3, 2.0e-2, 1.0))
        u, v = SVector(310.0, 205.0), SVector(298.0, 211.0)
        d = sampson_distance(F, u, v)
        @test isfinite(d)
        for k in (1.0e-9, 1.0e9)
            @test sampson_distance(FundMat(k * SMatrix{3, 3, Float64}(F)), u, v) ≈ d rtol = 1.0e-9
        end
        H = HomographyMat(SMatrix{3, 3, Float64}(1.1, 0.02, 1.0e-4, -0.03, 0.95, 2.0e-4, 12.0, -7.0, 1.0))
        h = sampson_distance(H, u, v)
        @test isfinite(h)
        for k in (1.0e-9, 1.0e9)
            @test sampson_distance(HomographyMat(k * SMatrix{3, 3, Float64}(H)), u, v) ≈ h rtol = 1.0e-9
        end
    end

    @testset "homography: G_x loses rank at a point mapped to infinity" begin
        # rows x+1, x, x (columns (1,1,1), (0,0,0), (1,0,0)): at x = 0, w = 0 and both
        # constraint rows' gradients lie along x — rank one
        H = HomographyMat(SMatrix{3, 3, Float64}(1, 1, 1, 0, 0, 0, 1, 0, 0))
        @test sampson_distance(H, SVector(0.0, 2.0), SVector(5.0, 5.0)) == Inf   # g₁ = 1: nothing nearby satisfies it
        H0 = HomographyMat(SMatrix{3, 3, Float64}(1, 1, 1, 0, 0, 0, 0, 0, 0))
        @test sampson_distance(H0, SVector(0.0, 2.0), SVector(5.0, 5.0)) == 0.0  # g = 0 with it
    end

    @testset "whiten: a rescaled model whitens to the same residual" begin
        F = FundMat(SMatrix{3, 3, Float64}(1.0e-6, -2.0e-5, 3.0e-3, 4.0e-5, 1.0e-6, -2.0e-2, -1.0e-3, 2.0e-2, 1.0))
        u, v = SVector(310.0, 205.0), SVector(298.0, 211.0)
        r1 = whiten(constraint_with_jacobians(F, u, v)...)[1]
        r9 = whiten(constraint_with_jacobians(FundMat(1.0e-9 * SMatrix{3, 3, Float64}(F)), u, v)...)[1]
        @test r1 != 0
        @test r9 ≈ r1 rtol = 1.0e-9
    end

    @testset "signed: the unsigned rule with the sign of g" begin
        F = FundMat(SMatrix{3, 3, Float64}(0, 1, 0, -1, 0, 0, 0, 0, 0))
        o = SVector(0.0, 0.0)
        @test signed_sampson_distance(F, o, o) == 0.0
        Finf = FundMat(SMatrix{3, 3, Float64}(0, 1, 0, -1, 0, 0, 0, 0, 1))
        @test signed_sampson_distance(Finf, o, o) == Inf
        @test signed_sampson_distance(FundMat(-SMatrix{3, 3, Float64}(Finf)), o, o) == -Inf
        # scale-free, and |signed| is the unsigned distance
        F = FundMat(SMatrix{3, 3, Float64}(1.0e-6, -2.0e-5, 3.0e-3, 4.0e-5, 1.0e-6, -2.0e-2, -1.0e-3, 2.0e-2, 1.0))
        u, v = SVector(310.0, 205.0), SVector(298.0, 211.0)
        d = signed_sampson_distance(F, u, v)
        @test abs(d) ≈ sampson_distance(F, u, v) rtol = 1.0e-12
        for k in (1.0e-9, 1.0e9)
            @test signed_sampson_distance(FundMat(k * SMatrix{3, 3, Float64}(F)), u, v) ≈ d rtol = 1.0e-9
        end
        @test signed_sampson_distance(FundMat(SMatrix{3, 3, Float32}(F)), Float32.(u), Float32.(v)) isa Float32
        # the unsigned distance keeps the element type too
        @test sampson_distance(FundMat(SMatrix{3, 3, Float32}(F)), Float32.(u), Float32.(v)) isa Float32
        @test sampson_distance(FundMat(SMatrix{3, 3, Float32}(F)), Float32.(u), Float32.(v)) ≈ abs(d) rtol = 1.0e-4
    end

    @testset "whiten: a degenerate measurement is 0 or Inf with zero Jacobian rows" begin
        r, J, ℓ = whiten(0.7, SVector(0.0, 0.0, 0.0, 0.0), SVector(1.0, 2.0, 3.0))
        @test r == Inf && all(iszero, J) && ℓ == 0
        @test whiten(-0.7, SVector(0.0, 0.0, 0.0, 0.0), SVector(1.0, 2.0, 3.0))[1] == -Inf
        @test whiten(0.0, SVector(0.0, 0.0, 0.0, 0.0), SVector(1.0, 2.0, 3.0))[1] == 0
        g = SVector(0.3, -0.2)
        Gx = SMatrix{2, 4, Float64}(0, 0, 0, 0, 0, 0, 0, 0)
        Gθ = SMatrix{2, 3, Float64}(1, 2, 3, 4, 5, 6)
        rv, Jv, ℓv = whiten(g, Gx, Gθ)
        @test rv == SVector(Inf, -Inf) && all(iszero, Jv) && ℓv == 0
        @test all(iszero, whiten(zero(g), Gx, Gθ)[1])
        # the degenerate homography pair above: whiten and sampson_distance agree
        H = HomographyMat(SMatrix{3, 3, Float64}(1, 1, 1, 0, 0, 0, 1, 0, 0))
        rH = whiten(constraint_with_jacobians(H, SVector(0.0, 2.0), SVector(5.0, 5.0))...)[1]
        @test sqrt(sum(abs2, rH)) == sampson_distance(H, SVector(0.0, 2.0), SVector(5.0, 5.0)) == Inf
    end

    @testset "whiten at the rounding scale: the one rule, whatever route" begin
        # A pair whose gradient and value both vanish only to ROUNDING: exact
        # zeros (whiten's default) call it unsatisfiable, the model's rounding
        # scale calls it satisfied — the Sampson distance's verdict. Threading
        # `constraint_rounding_scale` into `whiten` makes the two routes agree.
        F = FundMat(SMatrix{3, 3, Float64}(0, 1, 0, -1, 0, 0, 0, 0, 1.0e-17))
        o = SVector(0.0, 0.0)
        g, Gx, Gθ = constraint_with_jacobians(F, o, o)
        ρ = VgtkCore.constraint_rounding_scale(F, o, o)
        @test whiten(g, Gx, Gθ)[1] == Inf                        # exact rule: g ≠ 0
        @test whiten(g, Gx, Gθ, I, ρ)[1] == 0 == sampson_distance(F, o, o)
        @test signed_sampson_distance(F, o, o) == 0
        # a regular pair: the rounding scale changes nothing
        F = FundMat(SMatrix{3, 3, Float64}(1.0e-6, -2.0e-5, 3.0e-3, 4.0e-5, 1.0e-6, -2.0e-2, -1.0e-3, 2.0e-2, 1.0))
        u, v = SVector(310.0, 205.0), SVector(298.0, 211.0)
        g, Gx, Gθ = constraint_with_jacobians(F, u, v)
        ρ = VgtkCore.constraint_rounding_scale(F, u, v)
        @test whiten(g, Gx, Gθ, I, ρ) == whiten(g, Gx, Gθ)
        @test whiten(g, Gx, Gθ)[1] == signed_sampson_distance(F, u, v)
        # the homography: its rows' rounding bound decides the degenerate pair's value
        H = HomographyMat(SMatrix{3, 3, Float64}(1, 1, 1, 0, 0, 0, 1, 0, 0))
        a, b = SVector(0.0, 2.0), SVector(5.0, 5.0)
        gH, GxH, GθH = constraint_with_jacobians(H, a, b)
        @test norm(whiten(gH, GxH, GθH, I, VgtkCore.constraint_rounding_scale(H, a, b))[1]) == sampson_distance(H, a, b)
    end
end

# The scalar whitening is generic in its element type, so a chart's scalar rows
# (a radial camera's, a Sampson row) differentiate by ForwardDiff: through
# `whiten` it agrees with central differences, and with Float64 the result is
# the Float64 path's, bit for bit.
@testitem "whiten_ad" begin
    using Test
    using VisualGeometryToolkitCore
    using VisualGeometryToolkitCore: whiten, _whiten_residual
    using StaticArrays: SVector, SMatrix
    using ForwardDiff: ForwardDiff
    using FiniteDiff: FiniteDiff

    # a scalar constraint g(θ) = θ₁θ₂ − 1 measured through G_x(θ) = (θ₁, θ₂²), θ-dependent both
    g(θ) = θ[1] * θ[2] - 1
    Gx(θ) = SVector(θ[1], θ[2]^2)
    Gθ(θ) = SVector(θ[2], θ[1])
    r(θ) = first(whiten(g(θ), Gx(θ), Gθ(θ)))
    θ0 = SVector(0.7, 1.9)
    @test ForwardDiff.gradient(r, θ0) ≈ FiniteDiff.finite_difference_gradient(r, Vector(θ0)) rtol = 1.0e-7
    @test ForwardDiff.gradient(θ -> _whiten_residual(g(θ), Gx(θ)), θ0) ≈ ForwardDiff.gradient(r, θ0)
    @test r(θ0) ≈ g(θ0) / sqrt(Gx(θ0)[1]^2 + Gx(θ0)[2]^2) rtol = 1.0e-15    # Float64: the plain path
    # the degeneracy rule under duals: a vanishing gradient with g ≠ 0 is ±Inf
    @test isinf(ForwardDiff.value(first(whiten(ForwardDiff.Dual(1.0, 1.0), SVector(ForwardDiff.Dual(0.0, 0.0), ForwardDiff.Dual(0.0, 0.0)),
        SVector(ForwardDiff.Dual(1.0, 0.0), ForwardDiff.Dual(1.0, 0.0))))))
end
