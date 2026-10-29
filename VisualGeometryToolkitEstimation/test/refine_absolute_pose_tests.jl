# refine_absolute_pose — find_absolute_pose's local optimization alone, from a
# given pose, on the AbsolutePoseProblem's rows (the raw-pixel reprojection
# error through the full camera).
#
# Correctness: a perturbed pose is recovered to the LM's tolerance from exact
# pixels — a pinhole camera and a fisheye whose rays reach past 90°, under plain
# least squares (`Irls{GaussianShape}`, the non-robust LM the bearing form of
# 2026-10-27 ran, which recovered these scenes to the same 1e-9) and under the
# class default; the input's frames come back, and physical inputs in metres.

@testsetup module PixelPoseScenes
    using StaticArrays: SVector
    using Rotations: RotMatrix, RotXYZ
    using VisualGeometryToolkitCore: EuclideanMap, exp_so3, project
    export true_pose, perturbed, pose_errors, scene_points, pixel_pairs

    true_pose() = EuclideanMap(RotMatrix(RotXYZ(0.3, -0.2, 0.1)), SVector(0.1, -0.05, 2.0))
    # the truth moved by 0.08 rad about a skew axis and 0.15 in translation
    function perturbed(E)
        ΔR = exp_so3(SVector(0.05, -0.04, 0.05))
        return EuclideanMap(RotMatrix{3, Float64}(ΔR * E.R), E.t + SVector(0.1, -0.08, 0.07))
    end
    pose_errors(E, F) = (maximum(abs, E.R - F.R), maximum(abs, E.t - F.t))
    # world points whose camera-frame directions span ±`θmax` off the optical axis
    function scene_points(E, θmax; n = 40, depth = 3.0)
        pts = SVector{3, Float64}[]
        for k in 1:n
            θ = θmax * (k / n); φ = 2.399963 * k                     # golden-angle spiral
            d = SVector(sin(θ) * cos(φ), sin(θ) * sin(φ), cos(θ))
            Xc = (depth * (0.7 + 0.6 * ((7k) % 10) / 10)) * d
            push!(pts, E.R' * (Xc - E.t))                            # back into the world
        end
        return pts
    end
    # `X => u` through the camera, the points it can image
    function pixel_pairs(cam, E, Xs)
        keep = [X for X in Xs if all(isfinite, project(cam, E.R * X + E.t))]
        return [X => SVector{2, Float64}(project(cam, E.R * X + E.t)) for X in keep]
    end
end

@testitem "refine_absolute_pose: pinhole and fisheye past 90°, any strategy" setup = [PixelPoseScenes] begin
    using StaticArrays: SVector
    using VisualGeometryToolkitCore: CameraModel, Lens, Equidistant, GaussianShape, CauchyShape, estimate, converged, rms
    using VisualGeometryToolkitEstimation: refine_absolute_pose, Irls, Triggs
    E = true_pose()
    ls = (; local_optimization = Irls{GaussianShape}(), threshold = 1.0e4)   # plain least squares over every point
    pin = CameraModel(; focal = 800.0, pp = (640.0, 480.0))
    fish = CameraModel(; focal = 400.0, pp = (800.0, 800.0), lens = Lens(Equidistant()))
    for (cam, θmax) in ((pin, 0.5), (fish, deg2rad(110)))
        cs = pixel_pairs(cam, E, scene_points(E, θmax))
        cam === fish && @test count(((X, _),) -> (E.R * X + E.t)[3] < 0, cs) >= 6     # rays past 90°
        res = refine_absolute_pose(perturbed(E), cam, cs; ls...)
        @test converged(res) && res.lo_retcode === :converged
        eR, et = pose_errors(estimate(res), E)
        @test eR < 1.0e-9 && et < 1.0e-9
        @test rms(res) < 1.0e-6                                            # px
        # under the class default (Irls{TLSShape} at 8 px) from a near pose
        near = refine_absolute_pose(E, cam, [X => u + SVector(0.3 * sin(7i), 0.3 * cos(11i)) for (i, (X, u)) in enumerate(cs)])
        eR, et = pose_errors(estimate(near), E)
        @test count(near.inlier_mask) == length(cs) && eR < 1.0e-3 && et < 1.0e-3
    end
end

@testitem "refine_absolute_pose: frames, units, too few" setup = [PixelPoseScenes] begin
    using StaticArrays: SVector
    using VisualGeometryToolkitCore: CameraModel, GaussianShape, estimate, EuclideanMap, relabel, WorldFrame, CameraFrame, m, ustrip
    using VisualGeometryToolkitEstimation: refine_absolute_pose, Irls
    E = true_pose()
    cam = CameraModel(; focal = 800.0, pp = (640.0, 480.0))
    cs = pixel_pairs(cam, E, scene_points(E, 0.6; n = 12))
    ls = (; local_optimization = Irls{GaussianShape}(), threshold = 1.0e4)
    # a labeled guess is accepted; the pose comes back as find_absolute_pose's
    El = relabel(WorldFrame, CameraFrame, perturbed(E))
    @test estimate(refine_absolute_pose(El, cam, cs; ls...)) isa EuclideanMap{3, Float64}
    # physical pose and points: read in metres, the pose back in metres
    Ep = EuclideanMap(perturbed(E).R, perturbed(E).t .* m)
    res = refine_absolute_pose(Ep, cam, [X .* m => u for (X, u) in cs]; ls...)
    @test maximum(abs, ustrip(m, estimate(res)).t - E.t) < 1.0e-9
    @test refine_absolute_pose(E, cam, cs[1:2]) === nothing
end
