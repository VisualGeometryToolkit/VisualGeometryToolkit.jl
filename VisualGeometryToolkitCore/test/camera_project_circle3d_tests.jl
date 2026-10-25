@testitem "camera_project_circle3d" begin
    using Test
    using VisualGeometryToolkitCore
    using GeometryBasics: Point2, Point3, Circle
    using StaticArrays
    using LinearAlgebra: I, normalize

    @testset "project(camera, ::Circle3D)" begin
        K = SMatrix{3, 3, Float64}(600.0, 0, 0, 0, 600.0, 0, 320.0, 240.0, 1.0)
        R = SMatrix{3, 3, Float64}(1.0I)
        t = SVector(0.0, 0.0, 8.0)
        cam = PosedCamera(K, R, t)

        @testset "z=0 agreement with existing Circle method" begin
            c2 = Circle(Point2(0.0, 0.0), 1.0)
            c3 = Circle3D(Point3(0.0, 0.0, 0.0), 1.0)  # default normal = +z
            e2 = project(cam, c2)
            e3 = project(cam, c3)
            @test e3 isa Ellipse
            @test centroid(e3) ≈ centroid(e2) atol = 1.0e-8
            @test e3.a ≈ e2.a atol = 1.0e-8
            @test e3.b ≈ e2.b atol = 1.0e-8
        end

        @testset "tilted plane matches point sampling" begin
            n = normalize(SVector(0.3, -0.2, 1.0))
            c3 = Circle3D(Point3(0.5, -0.3, 0.0), 0.75, Vec3(n...))
            ell = project(cam, c3)
            @test ell isa Ellipse

            # Every sampled world-space boundary point should project onto the image ellipse
            Q = SMatrix{3, 3, Float64}(HomEllipseMat(ell))
            for θ in range(0, 2π; length = 32)[1:(end - 1)]
                X_world = boundary_point(c3, θ)
                u = project(cam, Point3{Float64}(X_world...))   # world position → Point3
                h = SVector{3, Float64}(u[1], u[2], 1.0)         # homogeneous lift = numerical buffer
                # Point lies on ellipse iff hᵀ Q h ≈ 0 (scale normalized)
                residual = h' * Q * h
                scale = max(abs(h[1]), abs(h[2]), 1.0)^2
                @test abs(residual) / scale < 1.0e-6
            end
        end

        @testset "offset circle, non-identity rotation" begin
            # Rotate camera 20° about y, translate
            θ = deg2rad(20.0)
            Ry = SMatrix{3, 3, Float64}(cos(θ), 0, -sin(θ), 0, 1.0, 0, sin(θ), 0, cos(θ))
            cam2 = PosedCamera(K, Ry, SVector(1.0, 0.5, 6.0))
            c3 = Circle3D(Point3(0.2, 0.1, 0.0), 0.5, Vec3(0.0, 0.0, 1.0))
            ell = project(cam2, c3)
            @test ell isa Ellipse
            @test ell.a > 0 && ell.b > 0
        end
    end
end
