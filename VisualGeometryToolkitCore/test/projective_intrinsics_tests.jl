@testitem "projective_intrinsics" begin
    # =============================================================================
    # ProjectiveIntrinsics: the principal point alone. A projective camera computes
    # exactly what the unit-focal LogicalIntrinsics placeholder computes, and has
    # no focal length.
    # =============================================================================
    using Test
    using VisualGeometryToolkitCore
    using VisualGeometryToolkitCore: EuclideanMap, PixelFrame, ParameterManifold, action_jacobian, convert_eltype, parameter_names, intrinsics, lens
    using StaticArrays
    using GeometryBasics: Point2
    using Random: MersenneTwister

    lens0 = Division(-2.0e-8, 3.0e-15)
    cam = CameraModel(ProjectiveIntrinsics(Point2(2120.0, 1416.0)), lens0)
    ref = CameraModel(LogicalIntrinsics(CameraCalibrationMat([1.0 0 2120; 0 1.0 1416; 0 0 1])), lens0)

    @testset "exactly the unit-focal reference" begin
        rng = MersenneTwister(7)
        for _ in 1:20
            r = SVector(0.5 * randn(rng), 0.5 * randn(rng), 1.0 + rand(rng))
            @test project(cam, r) == project(ref, r)
            x = SVector(4000 * rand(rng), 2800 * rand(rng))
            @test backproject(cam, x) == backproject(ref, x)
        end
    end

    @testset "accessors" begin
        @test parameter_names(intrinsics(cam)) == (:cx, :cy)
        @test principal_point(intrinsics(cam)) == Point2(2120.0, 1416.0)
        @test_throws MethodError focal_length(intrinsics(cam))
        @test ProjectiveIntrinsics(Point2(1, 2)) isa ProjectiveIntrinsics{Float64}
        @test ProjectiveIntrinsics(Point2(1.0, 2.0)) == ProjectiveIntrinsics{Float64}(Point2(1.0, 2.0))
        @test sprint(show, intrinsics(cam)) == "ProjectiveIntrinsics(cx = 2120.0, cy = 1416.0)"
    end

    @testset "parameter chart and eltype" begin
        θ = parameter_vector(cam)
        @test with_parameters(cam, θ) == cam
        @test convert_eltype(Float32, intrinsics(cam)) isa ProjectiveIntrinsics{Float32}
        @test convert_eltype(Float32, cam) isa CameraModel{<:ProjectiveIntrinsics{Float32}}
    end

    @testset "convert_frame moves the centre, not the lens" begin
        cv = convert_frame(PixelFrame{:opencv}, cam)
        shift = EuclideanMap(PixelFrame{:colmap}, PixelFrame{:opencv})(SVector(2120.0, 1416.0))
        @test principal_point(intrinsics(cv)) == Point2(shift[1], shift[2])
        @test principal_point(intrinsics(cv)) != principal_point(intrinsics(cam))
        @test lens(cv) == lens(cam)
    end

    @testset "undistort is the reference's" begin
        img = rand(MersenneTwister(3), 24, 32)
        cam_s = CameraModel(ProjectiveIntrinsics(Point2(16.0, 12.0)), Division(-2.0e-4, 0.0))
        ref_s = CameraModel(LogicalIntrinsics(CameraCalibrationMat([1.0 0 16; 0 1.0 12; 0 0 1])), Division(-2.0e-4, 0.0))
        @test undistort(cam_s, img) == undistort(ref_s, img)
    end

    @testset "parameter Jacobian under ForwardDiff" begin
        using ForwardDiff
        using FiniteDiff
        X = SVector(0.1, 0.2, 1.0)
        J = action_jacobian(ParameterManifold(cam), X)
        @test size(J) == (2, 4) && all(isfinite, J)
        f(θ) = project(with_parameters(cam, θ), X)
        @test J ≈ FiniteDiff.finite_difference_jacobian(θ -> collect(f(θ)), collect(parameter_vector(cam))) rtol = 1.0e-5
    end

    @testset "no metric operation" begin
        @test_throws MethodError calibration_matrix(cam)
        @test_throws MethodError unproject(cam, SVector(1.0, 2.0), 1.0)
    end
end
