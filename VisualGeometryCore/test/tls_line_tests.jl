# tls_line — the total-least-squares line, moved up from RobustVisualGeometry's
# unweighted fit_line with its tests and its Monte Carlo cell.

@testitem "tls_line" begin
    using Test
    using LinearAlgebra
    using StaticArrays
    using Random
    using VisualGeometryCore
    using VisualGeometryCore: coeffs, normal, geom, cov, tls_line_statistics
    const CHar = VisualGeometryCore.Synthetic

    _dist(line, p) = abs(coeffs(line)[1] * p[1] + coeffs(line)[2] * p[2] + coeffs(line)[3])

    @testset "exact points: the line, and a zero covariance" begin
        pts = [Point2{Float64}(Float64(i), 2.0i + 1.0) for i in 1:12]       # 2x − y + 1 = 0
        u = tls_line(pts)
        @test u isa Uncertain
        for p in pts
            @test _dist(geom(u), p) < 1.0e-10
        end
        nv = normal(geom(u))                                                 # ±(2, −1)/√5
        @test abs(abs(nv[1]) - 2 / sqrt(5)) < 1.0e-10
        @test abs(abs(nv[2]) - 1 / sqrt(5)) < 1.0e-10
        @test norm(SMatrix{2, 2, Float64}(cov(u))) < 1.0e-20                 # s² = 0 on exact data
        @test_throws ArgumentError tls_line(pts[1:2])
    end

    @testset "noisy: the covariance is s²(JᵀJ)⁻¹ with an honest s" begin
        rng = MersenneTwister(23)
        pts = [Point2{Float64}(Float64(i), 2.0i + 1.0 + 0.05 * randn(rng)) for i in 1:200]
        u = tls_line(pts)
        l = geom(u); nv = normal(l); tv = SVector(-nv[2], nv[1])
        r = [dot(nv, p) + coeffs(l)[3] for p in pts]
        s² = sum(abs2, r) / (length(pts) - 2)
        # s is the residual scale; the perpendicular noise is σ/√5 for this slope
        @test 0.5 * (0.05 / sqrt(5)) < sqrt(s²) < 2.0 * (0.05 / sqrt(5))
        t = [dot(tv, p) for p in pts]
        JtJ = SMatrix{2, 2}(sum(abs2, t), sum(t), sum(t), Float64(length(pts)))
        C = SMatrix{2, 2, Float64}(cov(u))
        @test C ≈ s² * inv(JtJ) rtol = 1.0e-10
        @test issymmetric(C) && all(≥(0), eigvals(Symmetric(C)))
    end

    @testset "weighted: fixed weights, a zero weight drops the point" begin
        rng = MersenneTwister(31)
        pts = [Point2{Float64}(Float64(i), 2.0i + 1.0 + 0.05 * randn(rng)) for i in 1:40]
        # unit weights are the unweighted step, bit for bit
        u1 = tls_line(pts); uw = tls_line(pts, ones(40))
        @test coeffs(geom(uw)) == coeffs(geom(u1)) && cov(uw) == cov(u1)
        # a zero weight is the point removed; integer weights are repeats
        bad = copy(pts); bad[7] = Point2(100.0, -300.0)
        w = ones(40); w[7] = 0.0
        @test coeffs(geom(tls_line(bad, w))) ≈ coeffs(geom(tls_line(deleteat!(copy(bad), 7)))) rtol = 1.0e-12
        w2 = [isodd(i) ? 2.0 : 1.0 for i in 1:40]
        rep = vcat(pts, pts[1:2:end])
        @test coeffs(geom(tls_line(pts, w2))) ≈ coeffs(geom(tls_line(rep))) rtol = 1.0e-12
        # the statistics: S and JᵀWJ at the returned line, the covariance made of them
        line, S, J = tls_line_statistics(pts, w2)
        nv = normal(line); tv = SVector(-nv[2], nv[1])
        r = [dot(nv, p) + coeffs(line)[3] for p in pts]; t = [dot(tv, p) for p in pts]
        @test S ≈ sum(w2 .* r .^ 2) rtol = 1.0e-12
        @test J ≈ SMatrix{2, 2}(sum(w2 .* t .^ 2), sum(w2 .* t), sum(w2 .* t), sum(w2)) rtol = 1.0e-12
        @test cov(tls_line(pts, w2)) ≈ S / (40 - 2) * inv(J) rtol = 1.0e-12
        # a given normal: only the offset is estimated, through the weighted centroid
        n0 = SVector(-2.0, 1.0) / sqrt(5)
        l0, _, _ = tls_line_statistics(pts, w2, n0)
        @test normal(l0) ≈ n0
        c = sum(w2 .* pts) / sum(w2)
        @test abs(dot(normal(l0), c) + coeffs(l0)[3]) < 1.0e-10
        # refusals
        @test_throws ArgumentError tls_line(pts, [1.0, 1.0, zeros(38)...])   # 2 of positive weight
        @test_throws ArgumentError tls_line(pts, -ones(40))
        @test_throws ArgumentError tls_line(pts, ones(39))
        @test_throws ArgumentError tls_line(pts, [Inf, ones(39)...])            # the one weights contract: finite
        @test_throws ArgumentError tls_line_statistics(pts, [NaN, ones(39)...])
        @test_throws ArgumentError tls_line_statistics(pts, zeros(40))          # no support: not a NaN line
        @test_throws ArgumentError tls_line_statistics(pts, zeros(40), n0)
    end

    @testset "Monte Carlo: the (φ, d) covariance is calibrated" begin
        # 30 points with isotropic 2D noise; the reported covariance in the
        # Hesse chart (φ, d) against the estimate's error (the normal's sign
        # aligned to the truth), χ²₂. Moved from RobustVisualGeometry (seed 801,
        # 600 trials: NEES 2.179/2 there).
        function tls_line_trial(; σ = 0.02, n = 30)
            return function (rng)
                φ = 2π * rand(rng); nv = SVector(cos(φ), sin(φ)); d = randn(rng)
                tv = SVector(-nv[2], nv[1]); p0 = -d * nv
                pts = [Point2{Float64}(p0 + (4 * rand(rng) - 2) * tv + σ * SVector(randn(rng), randn(rng))) for _ in 1:n]
                est = tls_line(pts)
                l = geom(est); n̂ = normal(l); d̂ = coeffs(l)[3]
                dot(n̂, nv) < 0 && (n̂ = -n̂; d̂ = -d̂)
                δφ = atan(nv[1] * n̂[2] - nv[2] * n̂[1], dot(nv, n̂))        # φ̂ − φ, wrapped
                return [δφ, d̂ - d], Matrix(cov(est))
            end
        end
        r = CHar.run_calibration(tls_line_trial(), MersenneTwister(801), 600, 2)
        @test CHar.is_calibrated(r)
    end
end
