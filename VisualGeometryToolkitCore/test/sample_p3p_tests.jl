# sample_p3p: random P3P configurations. Every overload drew its image points
# with `rand(rng, ::Rect)`, which has no method, so none had ever run; these are
# its first tests.
@testitem "sample_p3p" begin
    using Test
    using VisualGeometryToolkitCore
    using VisualGeometryToolkitCore: sample_p3p, sensor, focal_length_for_fov, CameraModel, Camera, project, centroid, px, Plane3D
    using GeometryBasics: Rect, Point3, widths
    using StaticArrays: SVector
    using LinearAlgebra: norm
    using Random: Xoshiro

    s = sensor("Sony", "IMX174")
    bounds = Rect(s)
    model = CameraModel(focal_length_for_fov(s, deg2rad(40.0); dimension = :horizontal), s.pitch, centroid(bounds) .* px)
    rng = Xoshiro(2)
    X = [Point3(200 * randn(rng), 200 * randn(rng), 0.0) for _ in 1:40]
    X3 = SVector{3}(X[1], X[2], X[3])

    @testset "every solution reprojects its correspondences" begin
        exts, cs = sample_p3p(Xoshiro(7), model, X, bounds)
        @test !isempty(exts) && length(cs) == 3
        @test all(norm(project(Camera(model, e), c.first) - c.second) < 1.0e-6 for e in exts, c in cs)
        @test all(c -> all(0 .<= c.second .<= widths(bounds)), cs)   # image points drawn inside the sensor
    end

    @testset "plane variants: a camera in front of the plane" begin
        for (cam, cs) in (
                sample_p3p(Xoshiro(7), model, X, bounds, Plane3D()),
                sample_p3p(Xoshiro(7), model, X3, bounds, Plane3D()),
            )
            @test cam isa Camera
            @test all(norm(project(cam, c.first) - c.second) < 1.0e-6 for c in cs)
        end
    end
end
