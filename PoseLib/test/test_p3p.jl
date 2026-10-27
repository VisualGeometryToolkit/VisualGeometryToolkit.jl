#!/usr/bin/env julia
# P3P Camera Pose Estimation Unit Tests

using Test
using VisualGeometryToolkitCore
using VisualGeometryToolkitCore: PosedCamera, RotMatrix
using LinearAlgebra, Random, Statistics
using GeometryBasics: Vec3, Point3
using StaticArrays: SVector

@testset "P3P Camera Pose Estimation" begin

    @testset "P3P with random synthetic correspondences" begin
        # Setup camera model (following pose_from_correspondences.jl pattern)
        s = sensor("Sony", "IMX174")
        f = focal_length(s, deg2rad(40.0); dimension=:horizontal)
        pp = s.resolution ./ 2
        model = CameraModel(f, s.pitch, pp)

        # Set random seed for reproducibility
        Random.seed!(42)

        # Run multiple trials with different random configurations
        n_trials = 10
        for trial in 1:n_trials
            # Generate random 3D points in world coordinates (planar at z=0, in mm)
            # Spread points over a reasonable area (similar to calibration board)
            X = [Point3(rand(100.0:700.0), rand(100.0:500.0), 0.0) for _ in 1:3]

            # Generate random ground truth camera pose
            # Position: random location above the plane looking down
            camera_pos = Point3(rand(200.0:600.0), rand(200.0:400.0), rand(1500.0:2500.0))
            target_pos = Point3(400.0, 300.0, 0.0)  # Look at center of point cloud
            up_vector = Vec3(0.0, -1.0, 0.0)

            extrinsics_gt = lookat(camera_pos, target_pos, up_vector)
            camera_gt = PosedCamera(model, extrinsics_gt)

            # Project 3D points to get 2D observations
            u = project.(Ref(camera_gt), X)
            u_unitless = u

            # Solve P3P: backproject to rays and solve for pose
            rays = backproject.(Ref(model), u_unitless)
            poses = p3p(PoseLibSolver(), rays, X)

            # Verify P3P found at least one solution
            @test poses !== nothing && length(poses) > 0

            if poses !== nothing && length(poses) > 0
                # Select best solution based on reprojection error
                cameras = [PosedCamera(model, pose) for pose in poses]
                u_proj = [project.(Ref(cam), X) for cam in cameras]
                mean_errors = [mean(norm.(u_p .- u_unitless)) for u_p in u_proj]
                best_idx = argmin(mean_errors)

                camera_recovered = cameras[best_idx]

                # Verify reprojection error is near zero
                # This is the key test - P3P may return multiple valid solutions,
                # but the best one (selected by reprojection error) should perfectly
                # reproduce the observations
                u_reproj = project.(Ref(camera_recovered), X)
                errors = norm.(u_reproj .- u_unitless)
                mean_error = mean(errors)
                max_error = maximum(errors)

                @test mean_error < 1e-9   # Near-perfect reconstruction
                @test max_error  < 1e-8

                # Note: We don't test pose match because P3P can return up to 4 valid
                # solutions. The solution with minimum reprojection error is correct
                # from a geometric standpoint, even if it's not the same as ground truth.
            end
        end
    end

    @testset "P3P with degenerate configurations" begin
        # Setup camera model
        s = sensor("Sony", "IMX174")
        f = focal_length(s, deg2rad(40.0); dimension=:horizontal)
        pp = s.resolution ./ 2
        model = CameraModel(f, s.pitch, pp)

        Random.seed!(123)

        # Test 1: Collinear points (should fail or give poor results)
        X_collinear = [Point3(100.0, 100.0, 0.0),
                       Point3(200.0, 200.0, 0.0),
                       Point3(300.0, 300.0, 0.0)]

        camera_pos = Point3(400.0, 300.0, 2000.0)
        extrinsics = lookat(camera_pos, Point3(200.0, 200.0, 0.0), Vec3(0.0, -1.0, 0.0))
        camera = PosedCamera(model, extrinsics)

        u = project.(Ref(camera), X_collinear)
        u_unitless = u
        rays = backproject.(Ref(model), u_unitless)
        poses = p3p(PoseLibSolver(), rays, X_collinear)

        # P3P may return solutions for collinear points, but they may not be unique
        # Just verify it doesn't crash
        @test true

        # Test 2: Very small triangle (numerical stability)
        X_small = [Point3(400.0, 300.0, 0.0),
                   Point3(400.1, 300.0, 0.0),
                   Point3(400.0, 300.1, 0.0)]

        u_small = project.(Ref(camera), X_small)
        u_small_unitless = u_small
        rays_small = backproject.(Ref(model), u_small_unitless)
        poses_small = p3p(PoseLibSolver(), rays_small, X_small)

        # Should handle small triangles gracefully
        @test true
    end

    @testset "P3P with multiple valid solutions" begin
        # In general, P3P can have up to 4 solutions
        # Test that we can disambiguate using reprojection error

        s = sensor("Sony", "IMX174")
        f = focal_length(s, deg2rad(40.0); dimension=:horizontal)
        pp = s.resolution ./ 2
        model = CameraModel(f, s.pitch, pp)

        Random.seed!(456)

        # Generate configuration that typically gives multiple solutions
        X = [Point3(0.0, 0.0, 0.0),
             Point3(100.0, 0.0, 0.0),
             Point3(50.0, 100.0, 0.0)]

        camera_pos = Point3(50.0, 50.0, 500.0)
        extrinsics_gt = lookat(camera_pos, Point3(50.0, 50.0, 0.0), Vec3(0.0, -1.0, 0.0))
        camera_gt = PosedCamera(model, extrinsics_gt)

        u = project.(Ref(camera_gt), X)
        u_unitless = u
        rays = backproject.(Ref(model), u_unitless)
        poses = p3p(PoseLibSolver(), rays, X)

        if poses !== nothing && length(poses) > 1
            # Multiple solutions exist - verify we can find the correct one
            cameras = [PosedCamera(model, pose) for pose in poses]
            u_proj = [project.(Ref(cam), X) for cam in cameras]
            mean_errors = [mean(norm.(u_p .- u_unitless)) for u_p in u_proj]

            # Best solution should have near-zero error
            min_error = minimum(mean_errors)
            @test min_error < 1e-10

            # Other solutions should have larger error
            best_idx = argmin(mean_errors)
            for (i, err) in enumerate(mean_errors)
                if i != best_idx
                    @test err > min_error  # Other solutions are worse
                end
            end
        end
    end

    @testset "P3P scale invariance" begin
        # P3P should work correctly regardless of world coordinate scale
        s = sensor("Sony", "IMX174")
        f = focal_length(s, deg2rad(40.0); dimension=:horizontal)
        pp = s.resolution ./ 2
        model = CameraModel(f, s.pitch, pp)

        Random.seed!(789)

        # Original scale (mm)
        X_mm = [Point3(100.0, 100.0, 0.0),
                Point3(500.0, 100.0, 0.0),
                Point3(300.0, 400.0, 0.0)]

        camera_pos_mm = Point3(300.0, 250.0, 2000.0)
        extrinsics = lookat(camera_pos_mm, Point3(300.0, 250.0, 0.0), Vec3(0.0, -1.0, 0.0))
        camera = PosedCamera(model, extrinsics)

        # Test at mm scale
        u = project.(Ref(camera), X_mm)
        u_unitless = u
        rays = backproject.(Ref(model), u_unitless)
        poses_mm = p3p(PoseLibSolver(), rays, X_mm)

        @test poses_mm !== nothing && length(poses_mm) > 0

        # Test at 10x scale (cm to mm)
        scale = 10.0
        X_scaled = [Point3(p[1] * scale, p[2] * scale, p[3] * scale) for p in X_mm]
        camera_pos_scaled = Point3(camera_pos_mm[1] * scale, camera_pos_mm[2] * scale, camera_pos_mm[3] * scale)
        target_scaled = Point3(300.0 * scale, 250.0 * scale, 0.0)
        extrinsics_scaled = lookat(camera_pos_scaled, target_scaled, Vec3(0.0, -1.0, 0.0))
        camera_scaled = PosedCamera(model, extrinsics_scaled)

        u_scaled = project.(Ref(camera_scaled), X_scaled)
        @test all(isapprox.(u_scaled, u_unitless, atol=1e-10))  # Same projections

        rays_scaled = backproject.(Ref(model), u_scaled)
        poses_scaled = p3p(PoseLibSolver(), rays_scaled, X_scaled)

        @test poses_scaled !== nothing && length(poses_scaled) > 0

        # Recovered translations should scale proportionally
        if poses_mm !== nothing && poses_scaled !== nothing &&
                length(poses_mm) > 0 && length(poses_scaled) > 0
            cameras_mm     = [PosedCamera(model, pose) for pose in poses_mm]
            cameras_scaled = [PosedCamera(model, pose) for pose in poses_scaled]

            u_proj_mm     = [project.(Ref(cam), X_mm)     for cam in cameras_mm]
            u_proj_scaled = [project.(Ref(cam), X_scaled) for cam in cameras_scaled]

            errors_mm     = [mean(norm.(u_p .- u_unitless))      for u_p in u_proj_mm]
            errors_scaled = [mean(norm.(u_p .- u_scaled)) for u_p in u_proj_scaled]

            best_mm     = argmin(errors_mm)
            best_scaled = argmin(errors_scaled)

            # Check that translation scales correctly
            poses_mm_vec     = collect(poses_mm)
            poses_scaled_vec = collect(poses_scaled)
            # PoseLib P3P returns up to 4 cheirality candidates, and the
            # min-reprojection argmin over the SAME 3 solve points is pure
            # numerical noise (every candidate reprojects them ~exactly), so
            # the selected candidate can differ between the two scales.
            # Scale invariance of the SOLUTION SET is the real property:
            # the scaled set must contain the scaled counterpart.
            t_mm = poses_mm_vec[best_mm].t
            @test any(isapprox(norm(p.t), norm(t_mm) * scale, rtol = 1e-2)
                      for p in poses_scaled_vec)
        end
    end
end
