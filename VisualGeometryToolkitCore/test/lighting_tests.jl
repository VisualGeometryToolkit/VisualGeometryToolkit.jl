# The scene's light (cameras/lighting.jl): the real spherical-harmonic basis against closed forms (the convention: no
# Condon–Shortley phase) and orthonormality on the sphere, the exit direction's geometry, the miss case's finite derivative,
# linearity in the coefficients, the constant and distant lights, units, eltype conversion and the kernel laws.
@testitem "lighting" begin
    using Test
    using StaticArrays: SVector, SMatrix
    using LinearAlgebra: dot, norm, normalize, I
    using VisualGeometryToolkitCore: EnvironmentLight, SphericalHarmonicMap, radiance, exit_direction, radiance_map, coefficients,
        with_coefficients, max_degree, sh_basis, sh_count, sh_index, coefficient_jacobian, center, convert_eltype, with_unit, has_units, m, mm
    import GeometryBasics
    import FlexUnits
    import ForwardDiff

    unit3(v) = SVector{3}(normalize(collect(Float64, v)))

    @testset "the basis: closed forms for l ≤ 2, no Condon–Shortley phase" begin
        @test sh_count(0) == 1 && sh_count(2) == 9 && sh_index(0, 0) == 1 && sh_index(1, -1) == 2 && sh_index(2, 2) == 9
        ref(x, y, z) = [1 / sqrt(4π),
                        sqrt(3 / 4π) * y, sqrt(3 / 4π) * z, sqrt(3 / 4π) * x,
                        sqrt(15 / π) / 2 * x * y, sqrt(15 / π) / 2 * y * z, sqrt(5 / π) / 4 * (3z^2 - 1),
                        sqrt(15 / π) / 2 * x * z, sqrt(15 / π) / 4 * (x^2 - y^2)]
        for ω in (unit3((0.3, -0.5, 0.8)), unit3((-0.9, 0.1, -0.2)), unit3((1.0, 0.0, 0.0)), unit3((0.0, -1.0, 0.0)))
            @test sh_basis(Val(2), ω) ≈ ref(ω...) atol = 1.0e-14
        end
        # at a pole the polar angle is clamped off it (acos's derivative kept finite): within √(2 eps) of the closed form
        @test sh_basis(Val(2), SVector(0.0, 0.0, 1.0)) ≈ ref(0.0, 0.0, 1.0) atol = 1.0e-7
    end

    @testset "the basis: orthonormal on the sphere (l ≤ 4, a midpoint rule in cos θ and φ)" begin
        nz, nφ = 1200, 200   # the midpoint rule is exact in φ for these degrees; its error in cos θ is O(1/nz²)
        G = zeros(sh_count(4), sh_count(4))
        for i in 1:nz, k in 1:nφ
            z = -1 + (i - 0.5) * 2 / nz; φ = (k - 0.5) * 2π / nφ; s = sqrt(1 - z^2)
            y = sh_basis(Val(4), SVector(s * cos(φ), s * sin(φ), z))
            G .+= (y * y') .* (2 / nz) .* (2π / nφ)
        end
        @test maximum(abs.(G - I)) < 1.0e-4
    end

    @testset "the basis: finite derivatives at the poles, isbits, allocation-free" begin
        J = ForwardDiff.jacobian(w -> sh_basis(Val(3), SVector{3}(w)), [0.0, 0.0, 1.0])
        @test all(isfinite, J)
        @test isbitstype(typeof(sh_basis(Val(3), unit3((1, 2, 3)))))
        f() = sh_basis(Val(3), SVector(0.6, 0.0, 0.8)); f()
        @test @allocated(f()) == 0
    end

    a = SMatrix{3, 4}(0.5 .+ 0.1 .* reshape(collect(1.0:12.0), 3, 4))
    M = SphericalHarmonicMap(a, 1)
    c = SVector(0.1, -0.2, 0.3); r = 2.0
    light = EnvironmentLight(M, c, r)

    @testset "the map and the light's accessors" begin
        @test max_degree(M) == 1 && coefficients(M) == a && max_degree(light) == 1 && coefficients(light) == a
        @test center(light) == c && GeometryBasics.radius(light) == r && radiance_map(light) === M
        @test_throws DimensionMismatch SphericalHarmonicMap(zeros(3, 5), 1)
        ω = unit3((1, 1, 0))
        @test M(ω) ≈ a * sh_basis(Val(1), ω)
        b = 2 .* a
        @test coefficients(with_coefficients(light, b)) == b && center(with_coefficients(light, b)) == c
        # the constant map: the same radiance from every direction, at every point
        K = SphericalHarmonicMap(0.7, 2)
        @test all(radiance(EnvironmentLight(K, c, r), SVector(0.2, 0.1, 0.0), unit3(d)) ≈ fill(0.7, 3) for d in ((1, 0, 0), (0, -1, 1), (3, 2, -1)))
    end

    @testset "exit direction: on the sphere, ahead of the point; distant light; the miss" begin
        for (x, d) in ((SVector(0.0, 0.0, 0.0), (1, 0, 0)), (SVector(0.5, 0.4, -0.3), (-1, 2, 0.5)), (SVector(1.0, -1.0, 1.0), (0, 0, 1)))
            ω = unit3(d)
            ω̂ = exit_direction(light, x, ω)
            @test norm(ω̂) ≈ 1
            p = c + r * ω̂                                  # the exit point
            μ = dot(p - x, ω)
            @test μ > 0 && p ≈ x + μ * ω                     # on the ray, ahead of x
        end
        far = EnvironmentLight(M, c, Inf)
        @test exit_direction(far, SVector(5.0, 1.0, 2.0), unit3((1, 2, 2))) ≈ unit3((1, 2, 2))
        @test radiance(far, SVector(5.0, 1.0, 2.0), unit3((1, 2, 2))) ≈ M(unit3((1, 2, 2)))
        # a point outside the sphere whose line misses it: read at the line's closest point to the centre, derivatives finite
        x = SVector(5.0, 0.0, 0.0); ω = unit3((0, 1, 0))
        @test exit_direction(light, x, ω) ≈ normalize((x - c) - dot(x - c, ω) * ω)
        @test all(isfinite, ForwardDiff.jacobian(p -> exit_direction(light, SVector{3}(p), ω), collect(x)))
        # outside, pointing away, the line crossing the sphere: read at the crossing nearer to the point, behind it (μ < 0)
        ω = unit3((1, 0.2, 0))
        p = c + r * exit_direction(light, x, ω)
        @test norm(p - c) ≈ r && dot(p - x, ω) < 0 && norm((p - x) - dot(p - x, ω) * ω) < 1.0e-12
    end

    @testset "radiance is linear in the coefficients" begin
        x = SVector(0.3, 0.2, -0.1); ω = unit3((0.2, -0.7, 0.4))
        @test coefficients(light) * coefficient_jacobian(light, x, ω) ≈ radiance(light, x, ω)
        g = ForwardDiff.jacobian(v -> radiance(with_coefficients(light, reshape(v, 3, 4)), x, ω), vec(collect(a)))
        y = coefficient_jacobian(light, x, ω)
        @test g ≈ kron(y', Matrix(1.0I, 3, 3))
    end

    @testset "units: metres in a public value, plain in a kernel" begin
        q = with_unit(light, mm)
        @test has_units(q) && !has_units(light)
        p = FlexUnits.ustrip(m, q)
        @test center(p) ≈ c ./ 1000 && GeometryBasics.radius(p) ≈ r / 1000
        back = FlexUnits.ustrip(mm, q)
        @test center(back) ≈ c && GeometryBasics.radius(back) ≈ r && coefficients(back) == a
        @test_throws ArgumentError FlexUnits.ustrip(m, light)
    end

    @testset "eltype conversion; isbits; allocation-free" begin
        l32 = convert_eltype(Float32, light)
        @test eltype(coefficients(l32)) == Float32 && eltype(center(l32)) == Float32
        @test radiance(l32, SVector{3, Float32}(0, 0, 0), SVector{3, Float32}(1, 0, 0)) ≈ radiance(light, SVector(0.0, 0, 0), SVector(1.0, 0, 0)) rtol = 1.0e-5
        @test isbitstype(typeof(light))
        x = SVector(0.3, 0.2, -0.1); ω = unit3((0.2, -0.7, 0.4))
        allocs(l, x, ω) = (radiance(l, x, ω); @allocated radiance(l, x, ω))
        @test allocs(light, x, ω) == 0
    end
end
