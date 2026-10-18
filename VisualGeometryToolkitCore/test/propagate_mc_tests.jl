# The frame-labeled transports of a measurement's uncertainty against Monte
# Carlo: `propagate` through a rigid map (linear — exact), a collineation (the
# point-dependent Jacobian of a nonlinear action) and a batched lens stage, each
# law (covariance pushed forward, precision pulled back); and the kσ ellipses
# built from a measured covariance, against their containment law.
@testitem "propagate_mc" begin
    import VisualGeometryToolkitCore as VgtkCore
    using Test
    using LinearAlgebra
    using StaticArrays
    using Random
    using Rotations
    using VisualGeometryToolkitCore
    using VisualGeometryToolkitCore: CovMat, PrecMat, Uncertain, cov, prec, propagate, Attributed,
        boundary
    const CHar = VgtkCore.Synthetic

    # One measurement x₀ with covariance shape Σ₀: draws x = x₀ + σ·L·n through
    # `act`, the error act(x) − act(x₀) against σ²·Σ′ (Σ′ the transported shape).
    function transport_trial(act, x₀, Σ₀, Σ′, σ)
        L = cholesky(Symmetric(Matrix(Σ₀))).L
        y₀ = act(x₀)
        N = length(x₀)
        return rng -> begin
            x = x₀ + σ * SVector{N}(L * randn(rng, N))
            (Vector(act(x) - y₀), σ^2 .* Matrix(Σ′))
        end
    end

    @testset "rigid map (3D): R·Σ·Rᵀ and the pulled-back precision — exact" begin
        m = EuclideanMap(RotMatrix{3, Float64}(RotXYZ(0.4, -0.3, 0.8)), SVector(1.0, -2.0, 0.5))
        x₀ = SVector(0.3, -0.7, 2.0)
        Σ₀ = SMatrix{3, 3}(2.0, 0.3, -0.1, 0.3, 1.0, 0.2, -0.1, 0.2, 0.5)
        u = propagate(m, Uncertain(x₀, CovMat(Σ₀)))
        @test CHar.is_calibrated(CHar.run_calibration(transport_trial(m, x₀, Σ₀, cov(u), 0.1), MersenneTwister(71), 4000, 3))
        up = propagate(m, Uncertain(x₀, PrecMat(inv(Σ₀))))
        @test CHar.is_calibrated(CHar.run_calibration(transport_trial(m, x₀, Σ₀, inv(Matrix(prec(up))), 0.1), MersenneTwister(72), 4000, 3))
    end

    @testset "collineation (pixel H, perspective): J·Σ·Jᵀ at the point, both laws" begin
        Hm = @SMatrix [1.1 0.2 5.0; -0.3 0.9 -3.0; 1.0e-3 2.0e-3 1.0]
        h = HomographyMap(Hm)
        act(x) = PerspectiveMap()(Hm * SVector(x[1], x[2], 1.0))
        x₀ = SVector(120.0, 80.0)
        Σ₀ = SMatrix{2, 2}(1.0, 0.3, 0.3, 0.6)
        u = propagate(h, Uncertain(x₀, CovMat(Σ₀)))
        @test CHar.is_calibrated(CHar.run_calibration(transport_trial(act, x₀, Σ₀, cov(u), 0.5), MersenneTwister(73), 4000, 2))
        up = propagate(h, Uncertain(x₀, PrecMat(inv(Σ₀))))
        @test CHar.is_calibrated(CHar.run_calibration(transport_trial(act, x₀, Σ₀, inv(Matrix(prec(up))), 0.5), MersenneTwister(74), 4000, 2))
    end

    @testset "a lens: the per-point J·Σ·Jᵀ, broadcast over a batch" begin
        lens = Lens(Perspective(), Additive(Radial((-0.28, 0.07, -0.006)), Tangential(0.0009, -0.0004)))
        act(x) = distort_normalized(lens, x)
        push(x, Σ) = (J = transform_deriv(lens, x); J * Σ * J')
        Σ₀ = SMatrix{2, 2}(1.0, -0.2, -0.2, 0.7)
        σ = 1.0e-3                                        # normalized units (≈ 0.6 px at f = 600)
        x₀s = [SVector(0.1, -0.05), SVector(0.45, 0.3)]   # near the centre and toward the rim
        Σos = push.(x₀s, Ref(Σ₀))
        for (x₀, Σo) in zip(x₀s, Σos)
            @test CHar.is_calibrated(CHar.run_calibration(transport_trial(act, x₀, Σ₀, Matrix(Σo), σ), MersenneTwister(75), 4000, 2))
        end
    end

    @testset "kσ ellipses from a covariance: containment 1 − e^(−k²/2)" begin
        c = Point2(3.0, -1.0)
        Σ = SMatrix{2, 2}(4.0, 1.0, 1.0, 9.0)
        L = cholesky(Symmetric(Matrix(Σ))).L
        for k in (1.0, 2.0, 3.0)
            level = 1 - exp(-k^2 / 2)
            eΣ = Ellipse(Attributed(c, CovMat(Σ)); sigma = k)
            eΛ = Ellipse(Attributed(c, PrecMat(inv(Σ))); sigma = k)
            eb = boundary(c, Σ, k)
            draw(rng) = Point2(c + SVector{2}(L * randn(rng, 2)))
            for (e, seed) in ((eΣ, 81), (eΛ, 82), (eb, 83))
                r = CHar.run_containment(rng -> (draw(rng) ∈ e,), MersenneTwister(seed), 20000)
                @test CHar.containment_nominal(r, 1; level)
            end
        end
    end
end
