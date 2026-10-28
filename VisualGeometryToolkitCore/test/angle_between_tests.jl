# =============================================================================
# angle_between (geometry/transforms/euclidean_map.jl): the atan form against
# the acos form near 0 and π, scale-freeness, 2-vectors, the edge cases, and
# translation_angle through it.
# =============================================================================

@testitem "angle_between: the atan form, accurate at every angle" begin
    using Test, StaticArrays, LinearAlgebra
    using VisualGeometryToolkitCore: angle_between, translation_angle, m
    import ForwardDiff

    acos_form(u, v) = acos(clamp(dot(normalize(u), normalize(v)), -1.0, 1.0))
    @testset "near 0 and π the acos form loses half the digits; this one does not" begin
        for θ in (1.0e-8, 3.0e-7, 1.0e-5)
            u = SVector(1.0, 0.0, 0.0); v = SVector(cos(θ), sin(θ), 0.0)
            @test angle_between(u, v) ≈ θ rtol = 1.0e-14
            @test angle_between(u, -v) ≈ π - θ rtol = 1.0e-15
        end
        # the acos form at 1e-8 rad: cos θ rounds to 1, the angle to 0
        u = SVector(1.0, 0.0, 0.0); v = SVector(cos(1.0e-8), sin(1.0e-8), 0.0)
        @test acos_form(u, v) == 0.0
        @test angle_between(u, v) ≈ 1.0e-8 rtol = 1.0e-14
        # and at a general angle the two agree
        a = normalize(SVector(0.3, -1.2, 0.7)); b = normalize(SVector(-0.4, 0.5, 1.1))
        @test angle_between(a, b) ≈ acos_form(a, b) rtol = 1.0e-13
    end
    @testset "scale-free, symmetric, 2-vectors, zero and NaN" begin
        a = SVector(0.3, -1.2, 0.7); b = SVector(-0.4, 0.5, 1.1)
        @test angle_between(3a, 0.5b) ≈ angle_between(a, b) rtol = 1.0e-15
        @test angle_between(a, b) == angle_between(b, a)
        @test angle_between([0.3, -1.2, 0.7], [-0.4, 0.5, 1.1]) == angle_between(a, b)
        @test angle_between(SVector(1.0, 0.0), SVector(0.0, 2.0)) ≈ π / 2
        @test angle_between(SVector(1.0, 0.0), SVector(-1.0, 1.0e-9)) ≈ π - 1.0e-9 rtol = 1.0e-15
        @test angle_between(SVector(1.0, 0.0), SVector(1.0, -1.0)) ≈ π / 4      # unsigned
        @test angle_between(zero(a), b) == 0.0
        @test isnan(angle_between(SVector(NaN, 0.0, 1.0), b))
        @test_throws DimensionMismatch angle_between(SVector(1.0, 0.0), a)
        @test_throws DimensionMismatch angle_between(SVector(1.0, 0.0, 0.0, 0.0), SVector(0.0, 1.0, 0.0, 0.0))
    end
    @testset "duals flow through; translation_angle is it" begin
        a = SVector(0.3, -1.2, 0.7); b = SVector(-0.4, 0.5, 1.1)
        g = ForwardDiff.derivative(s -> angle_between(a + s * b, b), 0.0)
        h = 1.0e-6
        @test g ≈ (angle_between(a + h * b, b) - angle_between(a - h * b, b)) / 2h rtol = 1.0e-7
        @test translation_angle(a, b) == angle_between(a, b)
        @test translation_angle(a .* m, b .* m) == angle_between(a, b)
    end
end
