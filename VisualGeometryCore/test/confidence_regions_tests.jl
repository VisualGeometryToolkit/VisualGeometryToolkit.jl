# =============================================================================
# Image-domain confidence regions — geometry + statistical containment
#
# • confidence_ellipse: boundary is the exact χ²₂ level set; ~95% of known-
#   Gaussian samples fall inside.
# • epipolar band (H–Z envelope of lines): boundary points satisfy the conic
#   equation; pointwise containment is χ²₁-exact; whole-line containment
#   matches the χ²₂ convention to first order.
# =============================================================================

@testitem "confidence_regions" begin
    using Test
    using VisualGeometryCore: band_conic   # public, not exported
    using LinearAlgebra
    using Random
    using StaticArrays
    using VisualGeometryCore                      # exports Rect2
    using VisualGeometryCore.GeometryBasics: coordinates

    @testset "confidence_ellipse — geometry and containment" begin
        rng = Random.MersenneTwister(21)
        μ = SVector(3.0, -2.0)
        A = [2.0 0.3; 0.3 0.5]
        Σ = A * A'
        e = confidence_ellipse(μ, Σ; level = 0.95)
        k2 = 5.991464547107979                       # χ²₂(0.95)
        Σi = inv(Σ)
        # boundary points sit on the Mahalanobis level set
        for p in coordinates(e, 64)
            d = [p[1] - μ[1], p[2] - μ[2]]
            @test dot(d, Σi * d) ≈ k2 rtol = 1.0e-8
        end
        # ~95% containment of Gaussian samples (binomial 4σ tolerance at N=20000)
        L = cholesky(Σ).L
        N = 20000
        hits = count(1:N) do _
            d = L * randn(rng, 2)
            dot(d, Σi * d) <= k2
        end
        @test abs(hits / N - 0.95) < 4 * sqrt(0.95 * 0.05 / N)
        # a covariance that is not positive definite is refused, as by Ellipse(μ, Symmetric(Σ)),
        # never clamped into a degenerate ellipse
        @test_throws ArgumentError confidence_ellipse(μ, zeros(2, 2))
        @test_throws ArgumentError confidence_ellipse(μ, [1.0 0.0; 0.0 -1.0])
        e′ = Ellipse(Point2{Float64}(μ...), Symmetric(k2 * SMatrix{2, 2}(Σ)))
        @test e.center == e′.center && e.a ≈ e′.a && e.b ≈ e′.b && e.θ ≈ e′.θ
    end

    @testset "epipolar band — conic geometry" begin
        rng = Random.MersenneTwister(22)
        # random uncertain-F setup: F + small ambient covariance
        F = FundMat{Float64}(Tuple(SMatrix{3, 3, Float64, 9}(randn(rng, 3, 3))))
        B = randn(rng, 9, 6)
        Σθ = 1.0e-6 .* (B * B')                       # rank-6 ambient model covariance
        u = Point2{Float64}(120.0, 200.0)
        band = epipolar_band(F, Σθ, u; level = 0.95)
        @test band.k2 ≈ 5.991464547107979 rtol = 1.0e-9
        rect = Rect2(0.0, 0.0, 500.0, 500.0)
        upper, lower = band_boundary(band, rect; n = 41)
        @test !isempty(upper) && !isempty(lower)
        C = band_conic(band)
        nb = 0
        for p in vcat(upper, lower)
            isnan(p[1]) && continue
            x̃ = SVector(p[1], p[2], 1.0)
            @test abs(dot(x̃, C * x̃)) < 1.0e-6 * band.k2 * dot(x̃, band.Σl * x̃)
            nb += 1
        end
        @test nb > 20
        # nominal line is inside its own band
        seg_mid = let l = band.l
            # a point on the line: least-squares foot from origin
            n2 = l[1]^2 + l[2]^2
            Point2(-l[1] * l[3] / n2, -l[2] * l[3] / n2)
        end
        @test in_band(band, seg_mid)
    end

    @testset "epipolar band — statistical containment (χ²₁ pointwise, χ²₂ whole-line)" begin
        rng = Random.MersenneTwister(23)
        l = SVector(0.3, -1.0, 80.0)
        Bm = randn(rng, 3, 3)
        Σl = 1.0e-4 .* (Bm * Bm')
        band1 = VisualGeometryCore.LineConfidenceBand(l, SMatrix{3, 3, Float64, 9}(Σl), 3.841458820694124)  # χ²₁(0.95)
        band2 = VisualGeometryCore.LineConfidenceBand(l, SMatrix{3, 3, Float64, 9}(Σl), 5.991464547107979)  # χ²₂(0.95)
        L = cholesky(Symmetric(Σl)).L
        N = 20000
        # pointwise: at a fixed x on the nominal line, |x̃ᵀm| ≤ √(χ²₁·x̃ᵀΣx̃) for 95%
        x̃ = let t = 137.0                          # x on l: l = (0.3, -1, 80) → y = 0.3x + 80
            SVector(t, 0.3 * t + 80.0, 1.0)
        end
        @test abs(dot(x̃, l)) < 1.0e-9
        hits1 = count(1:N) do _
            m = l + L * randn(rng, 3)
            dot(x̃, m)^2 <= band1.k2 * dot(x̃, Σl * x̃)
        end
        @test abs(hits1 / N - 0.95) < 4 * sqrt(0.95 * 0.05 / N)
        # whole-line: an MC line stays inside the χ²₂ band along its visible span
        # for ≈95% of draws (first-order statement — 1.5% slack on top of binomial)
        xs = collect(range(-500.0, 1000.0; length = 60))
        hits2 = count(1:N) do _
            m = l + L * randn(rng, 3)
            all(xs) do t                           # point on m: m₁x + m₂y + m₃ = 0
                ỹ = SVector(t, -(m[1] * t + m[3]) / m[2], 1.0)
                dot(ỹ, band2.l)^2 - band2.k2 * dot(ỹ, band2.Σl * ỹ) <= 0
            end
        end
        @test abs(hits2 / N - 0.95) < 0.015 + 4 * sqrt(0.95 * 0.05 / N)
    end
end
