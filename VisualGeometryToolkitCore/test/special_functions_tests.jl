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
