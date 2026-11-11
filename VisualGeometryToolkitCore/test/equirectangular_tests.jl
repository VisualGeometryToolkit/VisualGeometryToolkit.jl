# =============================================================================
# The panorama projections (geometry/cameras/equirectangular.jl): the axes,
# the round trip both ways, scale-freeness, the poles and the refusals; the
# cylindrical family's vertical coordinates, Jacobians, known raster pixels and
# the equal-area property; Backprojection of a projection and of a camera.
# =============================================================================

@testitem "Equirectangular: axes, round trips, poles" begin
    using Test, StaticArrays, LinearAlgebra, Random
    using VisualGeometryToolkitCore: Equirectangular, project, backproject, angle_between

    E = Equirectangular()
    @testset "the axes: forward, right, up, back" begin
        @test project(E, SVector(0.0, 0.0, 1.0)) == SVector(0.0, 0.0)
        @test project(E, SVector(1.0, 0.0, 0.0)) ≈ SVector(π / 2, 0.0)
        @test project(E, SVector(-1.0, 0.0, 0.0)) ≈ SVector(-π / 2, 0.0)
        @test project(E, SVector(0.0, -1.0, 0.0)) ≈ SVector(0.0, π / 2)     # up = −y
        @test project(E, SVector(0.0, 1.0, 0.0)) ≈ SVector(0.0, -π / 2)
        @test project(E, SVector(0.0, 0.0, -1.0)) ≈ SVector(π, 0.0)
        @test backproject(E, SVector(π / 2, 0.0)) ≈ SVector(1.0, 0.0, 0.0) atol = 1.0e-16
        @test backproject(E, SVector(0.0, π / 2)) ≈ SVector(0.0, -1.0, 0.0) atol = 1.0e-16
    end
    @testset "round trips, and the length of X does not enter" begin
        rng = Xoshiro(3)
        for _ in 1:1000
            X = randn(rng, SVector{3, Float64})
            p = project(E, X)
            @test -π < p[1] <= π && -π / 2 <= p[2] <= π / 2
            @test angle_between(backproject(E, p), X) < 1.0e-14
            @test project(E, 7.5 * X) ≈ p rtol = 1.0e-14
            @test norm(backproject(E, p)) ≈ 1 rtol = 1.0e-15
            q = SVector(2π * rand(rng) - π, π * rand(rng) - π / 2)
            @test project(E, backproject(E, q)) ≈ q atol = 1.0e-13
        end
        @test backproject(E, SVector(0.3 + 2π, 0.2)) ≈ backproject(E, SVector(0.3, 0.2)) atol = 1.0e-15
    end
    @testset "a zero or non-finite direction is NaN; Float32 stays Float32" begin
        @test all(isnan, project(E, SVector(0.0, 0.0, 0.0)))
        @test all(isnan, project(E, SVector(NaN, 0.0, 1.0)))
        @test all(isnan, project(E, SVector(0.0, Inf, 1.0)))
        @test eltype(project(E, SVector(1.0f0, 0.0f0, 1.0f0))) === Float32
        @test eltype(backproject(E, SVector(0.1f0, 0.2f0))) === Float32
    end
end

@testitem "the cylindrical family: vertical coordinates, round trips, Jacobians, the inverse map" begin
    using Test, StaticArrays, LinearAlgebra, Random
    using ForwardDiff: ForwardDiff
    using CoordinateTransformations: LinearMap, transform_deriv
    using VisualGeometryToolkitCore: Equirectangular, Cylindrical, CylindricalEqualArea, Mercator, Backprojection,
        project, backproject, angle_between

    # each projection's vertical coordinate of the latitude φ, and the range of v a test draws from
    family = ((Equirectangular(), φ -> φ, 1.5), (Cylindrical(), tan, 3.0), (CylindricalEqualArea(), sin, 0.99),
              (Mercator(), φ -> asinh(tan(φ)), 2.5))
    rng = Xoshiro(5)
    E = Equirectangular()
    for (P, vert, vmax) in family
        @testset "$(nameof(typeof(P))): the axes, the longitude is shared, the vertical is v(φ)" begin
            @test project(P, SVector(0.0, 0.0, 1.0)) == SVector(0.0, 0.0)
            @test project(P, SVector(1.0, 0.0, 0.0)) ≈ SVector(π / 2, 0.0)
            @test project(P, SVector(0.0, -1.0, 1.0)) ≈ SVector(0.0, vert(π / 4))      # 45° up
            @test project(P, SVector(0.0, 1.0, 1.0)) ≈ SVector(0.0, -vert(π / 4))      # 45° down
            for _ in 1:500
                X = randn(rng, SVector{3, Float64})
                p, e = project(P, X), project(E, X)
                @test p[1] == e[1]
                @test p[2] ≈ vert(e[2]) rtol = 1.0e-12 atol = 1.0e-15
            end
        end
        @testset "$(nameof(typeof(P))): round trips both ways, scale-freeness" begin
            for _ in 1:500
                X = randn(rng, SVector{3, Float64})
                p = project(P, X)
                @test angle_between(backproject(P, p), X) < 1.0e-14
                @test project(P, 7.5 * X) ≈ p rtol = 1.0e-14
                @test norm(backproject(P, p)) ≈ 1 rtol = 1.0e-15
                q = SVector(2π * rand(rng) - π, vmax * (2rand(rng) - 1))
                @test project(P, backproject(P, q)) ≈ q atol = 1.0e-12
            end
        end
        @testset "$(nameof(typeof(P))): transform_deriv is the Jacobian of the projection and of its inverse" begin
            for _ in 1:100
                X = randn(rng, SVector{3, Float64})
                @test transform_deriv(P, X) ≈ ForwardDiff.jacobian(v -> project(P, v), X) rtol = 1.0e-10
                q = project(P, X)
                @test transform_deriv(inv(P), q) ≈ ForwardDiff.jacobian(p -> SVector{3}(backproject(P, p)), q) rtol = 1.0e-12
            end
        end
        @testset "$(nameof(typeof(P))): inv is its Backprojection, a Transformation" begin
            @test inv(P) isa Backprojection && inv(inv(P)) === P
            X = SVector(0.3, -0.2, 0.9)
            @test P(X) == project(P, X) && inv(P)(P(X)) ≈ normalize(X)
            @test (LinearMap(SMatrix{3, 3}(1.0I)) ∘ inv(P))(P(X)) ≈ normalize(X)
            @test eltype(project(P, SVector(1.0f0, 0.0f0, 1.0f0))) === Float32
            @test eltype(backproject(P, SVector(0.1f0, 0.2f0))) === Float32
            @test all(isnan, project(P, SVector(0.0, 0.0, 0.0))) && all(isnan, project(P, SVector(NaN, 0.0, 1.0)))
        end
    end
    @testset "the closed-form Jacobians are ForwardDiff's, at fixed points from the equator to near the poles" begin
        for (P, _, _) in family, lat in (0.0, 0.3, -0.9, 1.4, -1.55, 1.5707), lon in (0.0, 0.7, -2.9, π)
            X = SVector(cos(lat) * sin(lon), -sin(lat), cos(lat) * cos(lon))   # unit direction at (lon, lat)
            @test transform_deriv(P, X) ≈ ForwardDiff.jacobian(v -> project(P, v), X) rtol = 1.0e-9
            @test transform_deriv(P, 3.0 * X) ≈ ForwardDiff.jacobian(v -> project(P, v), 3.0 * X) rtol = 1.0e-9   # off the sphere
            q = project(P, X)
            all(isfinite, q) || continue
            @test transform_deriv(inv(P), q) ≈ ForwardDiff.jacobian(p -> SVector{3}(backproject(P, p)), q) rtol = 1.0e-12
        end
    end
    @testset "the poles: no height on the central cylinder or Mercator's; ±1 on the equal-area one" begin
        @test all(isnan, project(Cylindrical(), SVector(0.0, -1.0, 0.0)))
        @test all(isnan, project(Mercator(), SVector(0.0, -1.0, 0.0)))
        @test project(CylindricalEqualArea(), SVector(0.0, -2.0, 0.0)) == SVector(0.0, 1.0)
        @test all(isnan, backproject(CylindricalEqualArea(), SVector(0.0, 1.5)))
    end
end

@testitem "the cylindrical family: the rays of known raster pixels; equal area" begin
    using Test, StaticArrays, LinearAlgebra
    using CoordinateTransformations: AffineMap
    using VisualGeometryToolkitCore: Cylindrical, CylindricalEqualArea, Mercator, Equirectangular, project

    # a raster of 0.5°/px: columns from longitude −90°, rows from v = v₁ downwards (v step = 0.5° in radians)
    δ = deg2rad(0.5)
    raster(v1) = AffineMap(SMatrix{2, 2}(δ, 0.0, 0.0, -δ), SVector(-π / 2, v1))   # COLMAP (x, y) → (λ, v)
    @testset "$(nameof(typeof(P)))" for P in (Cylindrical(), Mercator(), CylindricalEqualArea(), Equirectangular())
        v45 = project(P, SVector(0.0, -1.0, 1.0))[2]
        ray = inv(P) ∘ raster(v45)
        @test ray(SVector(0.0, 0.0)) ≈ SVector(-1.0, -1.0, 0.0) / sqrt(2) atol = 1.0e-15  # the top-left corner: 90° left, 45° up
        @test ray(SVector(180.0, v45 / δ)) ≈ SVector(0.0, 0.0, 1.0) atol = 1.0e-14        # forward, on the horizon
        @test P(ray(SVector(123.25, 77.5))) ≈ raster(v45)(SVector(123.25, 77.5)) atol = 1.0e-13
    end
    @testset "CylindricalEqualArea: every pixel of a uniform raster subtends the same solid angle" begin
        # the solid angle of a pixel [λ, λ + δλ] × [φ₀, φ₁] is ∫∫ cos φ dφ dλ = δλ (sin φ₁ − sin φ₀) (Archimedes): with the
        # pixel's latitudes from the projection's inverse, every row of a uniform raster subtends δλ·δv
        P = CylindricalEqualArea()
        δλ = deg2rad(0.5); δv = 0.01
        pix(v) = (φ0 = project(Equirectangular(), inv(P)(SVector(0.0, v)))[2]; φ1 = project(Equirectangular(), inv(P)(SVector(0.0, v + δv)))[2];
                  δλ * (sin(φ1) - sin(φ0)))
        @test all(v -> pix(v) ≈ δλ * δv, -0.99:0.01:0.98)
        # the equirectangular raster is not equal-area: its polar pixels are smaller
        pixE(φ) = δλ * (sin(φ + 0.01) - sin(φ))
        @test pixE(1.4) < 0.2 * pixE(0.0)
    end
end

@testitem "Backprojection of a camera model: a virtual camera's pixel → ray map" begin
    using Test, StaticArrays, LinearAlgebra
    using ForwardDiff: ForwardDiff
    using CoordinateTransformations: LinearMap, transform_deriv
    using VisualGeometryToolkitCore: Backprojection, CameraModel, Lens, Perspective, Equidistant, px, backproject, project,
        backproject_with_jacobian
    for lens in (Lens(Perspective()), Lens(Equidistant()))
        m = CameraModel(300.0px, SA[320.0px, 240.0px]; lens)
        b = Backprojection(m)
        u = SVector(400.0, 100.0)
        @test b(u) ≈ backproject(m, u) && project(m, b(u)) ≈ u
        # its Jacobian is the camera model's own back-projection Jacobian, ForwardDiff's of the unit ray
        @test transform_deriv(b, u) == last(backproject_with_jacobian(m, u))
        @test transform_deriv(b, u) ≈ ForwardDiff.jacobian(p -> SVector{3}(Tuple(backproject(m, p))), u) rtol = 1.0e-10
        R = SMatrix{3, 3}(0.0, 0.0, -1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0)            # aimed 90° to the side
        @test (LinearMap(R) ∘ b)(SVector(320.0, 240.0)) ≈ R * SVector(0.0, 0.0, 1.0)  # the principal point looks along R·z
    end
end

@testitem "Backprojection of a camera without its own Jacobian, and of Int pixels" begin
    # A model with no backproject_with_jacobian method (a BoundedCamera) takes
    # ForwardDiff through its backproject; an integer pixel is floated first.
    using Test, StaticArrays
    using ForwardDiff: ForwardDiff
    using CoordinateTransformations: transform_deriv
    using VisualGeometryToolkitCore: Backprojection, CameraModel, BoundedCamera, NoncentralSplineModel, SplineLens, Lens, Equidistant, Size2, px,
        backproject_with_jacobian
    cm = CameraModel(300.0px, SA[320.0px, 240.0px]; lens = Lens(Equidistant()))
    u = SVector(400.0, 100.0)
    J = last(backproject_with_jacobian(cm, u))
    bc = BoundedCamera(cm, Size2(640, 480))
    @test transform_deriv(Backprojection(bc), u) ≈ J rtol = 1.0e-10
    @test transform_deriv(Backprojection(bc), SVector(400, 100)) ≈ J rtol = 1.0e-10
    @test transform_deriv(Backprojection(cm), SVector(400, 100)) ≈ J rtol = 1.0e-10
    # a spline camera, and a non-central model, take their own (analytic) Jacobian
    gen = CameraModel(SplineLens, cm, Size2(640, 480); grid_size = (6, 5))
    @test transform_deriv(Backprojection(gen), SVector(400, 100)) == last(backproject_with_jacobian(gen, u))
    nc = NoncentralSplineModel(cm, Size2(640, 480); grid_size = (6, 5))
    @test transform_deriv(Backprojection(nc), SVector(400, 100)) == last(backproject_with_jacobian(nc, u))
end
