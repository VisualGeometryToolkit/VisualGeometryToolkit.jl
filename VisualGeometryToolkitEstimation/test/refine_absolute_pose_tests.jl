# refine_absolute_pose — the bearing (tangent-plane) pose refinement.
#
# Correctness: a perturbed pose is recovered to the LM's tolerance from exact
# bearings — a pinhole camera, a fisheye whose rays reach past 90° and one ray
# behind the camera (no z = 1 lift anywhere), and a generalized camera with
# distinct ray origins; the back-projection Jacobians are tangent per view
# (fᵀM = 0) and whiten the residual into pixels, checked against the
# reprojection error computed independently through the camera.

@testsetup module BearingPoseScenes
    using StaticArrays: SVector
    using LinearAlgebra: normalize
    using Rotations: RotMatrix, RotXYZ
    using VisualGeometryToolkitCore: EuclideanMap, exp_so3
    export true_pose, perturbed, pose_errors, scene_points

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
end

@testitem "refine_absolute_pose: pinhole, pixel-whitened" setup = [BearingPoseScenes] begin
    using StaticArrays: SVector
    using LinearAlgebra: normalize, dot, norm
    using VisualGeometryToolkitCore: CameraModel, Csponds, withattrs, project, backproject_with_jacobian, estimate, retcode, rms
    using VisualGeometryToolkitEstimation: refine_absolute_pose
    cam = CameraModel(; focal = 800.0, pp = (640.0, 480.0))
    E = true_pose()
    Xs = scene_points(E, 0.5)
    us = [SVector{2, Float64}(project(cam, E.R * X + E.t)) for X in Xs]
    fM = [backproject_with_jacobian(cam, u) for u in us]
    fs = first.(fM); Ms = last.(fM)
    @test maximum(abs(dot(f, M[:, k])) for (f, M) in fM for k in 1:2) < 1.0e-12     # tangent: fᵀM = 0
    cs = withattrs(Csponds(Xs, fs); M = Ms)
    res = refine_absolute_pose(perturbed(E), cs)
    @test retcode(res) === :converged
    eR, et = pose_errors(estimate(res), E)
    @test eR < 1.0e-9 && et < 1.0e-9
    @test rms(res) < 1.0e-6                                          # px

    # the pixel whitening against the reprojection error itself: with noisy
    # pixels, the refined rms (px) is the camera's own reprojection rms there
    noise = [SVector(0.3 * sin(7i), 0.3 * cos(11i)) for i in eachindex(us)]
    fMn = [backproject_with_jacobian(cam, u + δ) for (u, δ) in zip(us, noise)]
    resn = refine_absolute_pose(E, withattrs(Csponds(Xs, first.(fMn)); M = last.(fMn)))
    En = estimate(resn)
    reproj = sqrt(sum(norm(SVector{2, Float64}(project(cam, En.R * X + En.t)) - (u + δ))^2 for (X, u, δ) in zip(Xs, us, noise)) / length(Xs))
    @test isapprox(rms(resn), reproj; rtol = 1.0e-3)
    @test 0.1 < reproj < 0.5
    # without M the residual is the angle (radians): at f = 800 px, ≈ px / 800
    resr = refine_absolute_pose(E, Csponds(Xs, first.(fMn)))
    @test isapprox(rms(resr), reproj / 800; rtol = 0.1)
end

@testitem "refine_absolute_pose: fisheye past 90° and a ray behind the camera" setup = [BearingPoseScenes] begin
    using StaticArrays: SVector
    using LinearAlgebra: normalize, dot
    using VisualGeometryToolkitCore: CameraModel, Csponds, withattrs, Lens, Equidistant, project, backproject_with_jacobian,
        estimate, retcode, rms
    using VisualGeometryToolkitEstimation: refine_absolute_pose
    E = true_pose()
    # bearings out to 110° off the axis (z < 0 for the outer ones): no z = 1 point exists for them
    Xs = scene_points(E, deg2rad(110))
    fs = [normalize(E.R * X + E.t) for X in Xs]
    @test count(f -> f[3] < 0, fs) >= 6
    # and one ray BEHIND the camera, 160° off the axis
    Xb = E.R' * (SVector(0.3, 0.1, -2.5) - E.t)
    push!(Xs, Xb); push!(fs, normalize(E.R * Xb + E.t))
    res = refine_absolute_pose(perturbed(E), Csponds(Xs, fs))
    @test retcode(res) === :converged
    eR, et = pose_errors(estimate(res), E)
    @test eR < 1.0e-9 && et < 1.0e-9
    # the same rays through a fisheye CAMERA (equidistant, 190° field): its
    # back-projection's bearings and Jacobians, tangent, whiten into its pixels
    cam = CameraModel(; focal = 400.0, pp = (800.0, 800.0), lens = Lens(Equidistant()))
    keep = [i for i in 1:(length(Xs) - 1) if all(isfinite, project(cam, E.R * Xs[i] + E.t))]
    us = [SVector{2, Float64}(project(cam, E.R * Xs[i] + E.t)) for i in keep]
    fM = [backproject_with_jacobian(cam, u) for u in us]
    @test count(fm -> fm[1][3] < 0, fM) >= 6                              # past 90°, through the camera
    @test maximum(abs(dot(f, M[:, k])) for (f, M) in fM for k in 1:2) < 1.0e-10
    resf = refine_absolute_pose(perturbed(E), withattrs(Csponds(Xs[keep], first.(fM)); M = last.(fM)))
    @test retcode(resf) === :converged
    eR, et = pose_errors(estimate(resf), E)
    @test eR < 1.0e-8 && et < 1.0e-8
    @test rms(resf) < 1.0e-6
end

@testitem "refine_absolute_pose: generalized camera (ray origins)" setup = [BearingPoseScenes] begin
    using StaticArrays: SVector
    using LinearAlgebra: normalize
    using VisualGeometryToolkitCore: Csponds, withattrs, estimate, retcode, rms
    using VisualGeometryToolkitEstimation: refine_absolute_pose
    E = true_pose()
    Xs = scene_points(E, deg2rad(100))
    os = [SVector(0.02 * sin(3i), 0.02 * cos(5i), 0.01 * sin(i)) for i in eachindex(Xs)]   # distinct origins, ~2 cm
    fs = [normalize(E.R * X + E.t - o) for (X, o) in zip(Xs, os)]
    res = refine_absolute_pose(perturbed(E), withattrs(Csponds(Xs, fs); origins = os))
    @test retcode(res) === :converged
    eR, et = pose_errors(estimate(res), E)
    @test eR < 1.0e-9 && et < 1.0e-9
    # the central form on the same rays cannot fit them: the origins matter
    @test rms(refine_absolute_pose(perturbed(E), Csponds(Xs, fs))) > 1.0e-4
end

@testitem "refine_absolute_pose: refusals, frames and units" setup = [BearingPoseScenes] begin
    using StaticArrays: SVector, SMatrix
    using LinearAlgebra: normalize
    using VisualGeometryToolkitCore: Csponds, withattrs, estimate, EuclideanMap, relabel, WorldFrame, CameraFrame, m, ustrip
    using VisualGeometryToolkitEstimation: refine_absolute_pose
    E = true_pose()
    Xs = scene_points(E, 0.6; n = 12)
    fs = [normalize(E.R * X + E.t) for X in Xs]
    # a z = 1 chart point is not a bearing
    zs = [f / f[3] for f in fs]
    @test_throws ArgumentError refine_absolute_pose(E, Csponds(Xs, zs))
    # a Jacobian that is not tangent to its bearing (here: the bearing itself added to M)
    Mbad = [hcat(normalize(SVector(1.0, 0.0, 0.0) - f[1] * f) + f, normalize(SVector(0.0, 1.0, 0.0) - f[2] * f)) for f in fs]
    @test_throws ArgumentError refine_absolute_pose(E, withattrs(Csponds(Xs, fs); M = Mbad))
    @test_throws ArgumentError refine_absolute_pose(E, Csponds(Xs[1:2], fs[1:2]))
    # the input's frames come back; physical inputs come back in metres
    El = relabel(WorldFrame, CameraFrame, perturbed(E))
    @test estimate(refine_absolute_pose(El, Csponds(Xs, fs))) isa EuclideanMap{3, Float64, <:Any, <:Any, WorldFrame, CameraFrame}
    Ep = EuclideanMap(perturbed(E).R, perturbed(E).t .* m)
    res = refine_absolute_pose(Ep, Csponds([X .* m for X in Xs], fs))
    @test maximum(abs, ustrip(m, estimate(res)).t - E.t) < 1.0e-9
end
