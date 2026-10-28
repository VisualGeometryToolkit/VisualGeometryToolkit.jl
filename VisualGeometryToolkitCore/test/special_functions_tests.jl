# =============================================================================
# The device-safe special functions hoisted from RadiometricCamera
# (numerics/special.jl): the normal CDF by Abramowitz & Stegun 7.1.26 and the
# dilogarithm, against independent references.
# =============================================================================

@testitem "special functions: normcdf_as and dilog" begin
    using Test
    using ForwardDiff: ForwardDiff
    using SpecialFunctions: erfc
    using QuadGK: quadgk
    using VisualGeometryToolkitCore: normcdf_as, dilog

    # Φ(x) = erfc(−x/√2)/2, in BigFloat; A&S 7.1.26 promises |error| < 1.5e-7 (on erf, so 7.5e-8 on Φ)
    Φ(x) = Float64(erfc(-big(x) / sqrt(big(2))) / 2)
    xs = range(-6.0, 6.0; length = 1201)
    @test maximum(abs(normcdf_as(x) - Φ(x)) for x in xs) <= 7.5e-8
    @test normcdf_as(0.0) ≈ 0.5 atol = 1.0e-9
    @test all(normcdf_as(x) + normcdf_as(-x) ≈ 1 for x in xs)                    # symmetric
    # Float32 stays Float32 and agrees to its precision; the derivative is φ to the approximation's accuracy
    @test normcdf_as(0.3f0) isa Float32
    @test maximum(abs(normcdf_as(Float32(x)) - Φ(x)) for x in xs) <= 2.0e-7
    φ(x) = exp(-x^2 / 2) / sqrt(2π)
    @test maximum(abs(ForwardDiff.derivative(normcdf_as, x) - φ(x)) for x in xs) <= 3.0e-6     # measured 2.5e-6

    # Li₂(x) = −∫₀ˣ ln(1 − t)/t dt, the integral in 256-bit BigFloat
    li2_ref(x) = setprecision(BigFloat, 256) do
        Float64(-first(quadgk(t -> log1p(-t) / t, big(0), big(x); rtol = big(1.0e-40))))
    end
    @test dilog(0.0) == 0
    for x in (1.0e-8, 0.1, 0.25, 0.4999, 0.5, 0.5001, 0.7, 0.9, 0.99, 0.999999)
        @test dilog(x) ≈ li2_ref(x) rtol = 4 * eps() atol = 4 * eps()
    end
    @test dilog(1.0) == π^2 / 6 && dilog(1.5) == π^2 / 6                          # x ≥ 1 reads Li₂(1)
    @test dilog(0.5) ≈ π^2 / 12 - log(2)^2 / 2 rtol = 2 * eps()                     # Euler's value
    # the derivative is −ln(1 − x)/x; Float32 stays Float32
    for x in (0.2, 0.5, 0.8)
        @test ForwardDiff.derivative(dilog, x) ≈ -log1p(-x) / x rtol = 1.0e-13
    end
    @test dilog(0.3f0) isa Float32 && dilog(0.8f0) isa Float32
    @test dilog(0.3f0) ≈ Float32(li2_ref(0.3)) rtol = 4 * eps(Float32)
    @test dilog(0.8f0) ≈ Float32(li2_ref(0.8)) rtol = 8 * eps(Float32)
end

# The quadrature rules hoisted from RadiometricCamera (numerics/quadrature.jl).
@testitem "quadrature: gauss_hermite_prob and hammersley" begin
    using Test
    using VisualGeometryToolkitCore: gauss_hermite_prob, hammersley

    # E[X^k] of N(0, 1): 0 for odd k, (k − 1)!! for even k — exact to degree 2n − 1
    moment(k) = isodd(k) ? 0.0 : Float64(prod(1:2:(k - 1); init = 1))
    for n in 1:10
        x, w = gauss_hermite_prob(n)
        @test length(x) == n == length(w) && issorted(x) && all(>(0), w)
        @test sum(w) ≈ 1 atol = 1.0e-14
        for k in 0:(2n - 1)
            @test sum(w .* x .^ k) ≈ moment(k) atol = 1.0e-13 * max(1, sum(w .* abs.(x) .^ k))   # rounding of the terms summed
        end
        @test !(sum(w .* x .^ (2n)) ≈ moment(2n))                                  # and not beyond
    end
    @test_throws ArgumentError gauss_hermite_prob(0)
    x3, w3 = gauss_hermite_prob(3)
    @test x3 == [-sqrt(3.0), 0.0, sqrt(3.0)] && w3 == [1 / 6, 2 / 3, 1 / 6]

    # Hammersley: u₁ = (i − ½)/N, u₂ the radical inverse of i − 1 in base 2
    N = 8
    @test [hammersley(i, N) for i in 1:N] == [((i - 0.5) / N, r) for (i, r) in zip(1:N, (0.0, 0.5, 0.25, 0.75, 0.125, 0.625, 0.375, 0.875))]
    @test hammersley(Float32, 3, 4) === (0.625f0, 0.25f0)
    @test sort([hammersley(i, 64)[2] for i in 1:64]) == [(k - 1) / 64 for k in 1:64]   # a stratified permutation
    alloc() = @allocated hammersley(Float32, 7, 16)
    alloc()
    @test alloc() == 0
end

@testitem "square_rule and unit_disc_rule: the 2-D product rules" begin
    using Test, StaticArrays
    using QuadGK: gauss
    using VisualGeometryToolkitCore: square_rule, unit_disc_rule, gauss_hermite_prob

    # the tensor rule over [−½, ½]² of Gauss–Legendre: Σw = 1, exact for x^a y^b up to a, b ≤ 2n − 1 (odd moments 0, x² → 1/12)
    x, w = gauss(3); x = x ./ 2; w = w ./ 2
    nodes, weights = square_rule(x, w)
    @test length(nodes) == 9 && nodes[2] == SVector(x[2], x[1]) && weights[2] == w[2] * w[1]      # the first coordinate fastest
    @test sum(weights) ≈ 1
    m(a, b) = sum(wi * p[1]^a * p[2]^b for (p, wi) in zip(nodes, weights))
    @test m(2, 0) ≈ 1 / 12 && m(2, 2) ≈ 1 / 144 && m(4, 0) ≈ 1 / 80 && abs(m(1, 3)) < 1e-16
    # of the Gauss–Hermite rule: the standard bivariate normal's moments
    xh, wh = gauss_hermite_prob(4)
    nh, wh2 = square_rule(xh, wh)
    @test sum(wh2) ≈ 1 && sum(wi * p[1]^2 * p[2]^2 for (p, wi) in zip(nh, wh2)) ≈ 1 && sum(wi * p[1]^4 for (p, wi) in zip(nh, wh2)) ≈ 3
    @test_throws DimensionMismatch square_rule(x, w[1:2])

    # the polar rule over the unit disc: Σw = π (the area); ∫ρ² = π/2, ∫x² = π/4, ∫x²y² = π/24, ∫ρ⁴ = π/3; odd moments vanish
    nd, wd = unit_disc_rule(4, 12)
    @test length(nd) == 48 && all(p -> sum(abs2, p) < 1, nd)
    @test sum(wd) ≈ π
    d(f) = sum(wi * f(p) for (p, wi) in zip(nd, wd))
    @test d(p -> p[1]^2 + p[2]^2) ≈ π / 2 && d(p -> p[1]^2) ≈ π / 4 && d(p -> p[1]^2 * p[2]^2) ≈ π / 24 && d(p -> (p[1]^2 + p[2]^2)^2) ≈ π / 3
    @test abs(d(p -> p[1])) < 1e-14 && abs(d(p -> p[1] * p[2]^2)) < 1e-14
    # radius-major: node (k − 1)·n_φ + j at radius ρ_k, angle 2π(j − ½)/n_φ
    xr, wr = gauss(4); ρ = (xr .+ 1) ./ 2
    @test nd[12 + 3] ≈ ρ[2] .* SVector(cos(2π * 2.5 / 12), sin(2π * 2.5 / 12)) && wd[12 + 3] ≈ ρ[2] * wr[2] / 2 * 2π / 12
    @test_throws ArgumentError unit_disc_rule(0, 4)
end
