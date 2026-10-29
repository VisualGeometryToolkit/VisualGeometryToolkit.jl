# refine_absolute_pose — find_absolute_pose's local optimization alone, from a
# given pose, on the AbsolutePoseProblem's rows (the raw-pixel reprojection
# error through the full camera).
#
# Correctness: a perturbed pose is recovered to the LM's tolerance from exact
# pixels — a pinhole camera and a fisheye whose rays reach past 90°, under plain
# least squares (`Irls{GaussianShape}`, the non-robust LM the bearing form of
# 2026-10-27 ran, which recovered these scenes to the same 1e-9) and under the
# class default; the input's frames come back, and physical inputs in metres.
#
# The bearing form (a BearingPoseProblem: X => unit bearing, optional ray
# origins, pixel Jacobians and weights) recovers a perturbed pose from exact
# bearings — pinhole, a fisheye past 90° with a ray behind the camera, a
# generalized camera — under its default (least squares over every
# correspondence; the same pose as the pre-2026-10-29 bearing refine to
# ≤ 8e-14, measured on these scenes), and a robust strategy (Triggs) plugged in
# on contaminated rays reaches the inliers' least squares.

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
    using VisualGeometryToolkitCore: CameraModel, GaussianShape, estimate, EuclideanMap, relabel, unlabel, WorldFrame, CameraFrame, m, ustrip
    using VisualGeometryToolkitEstimation: refine_absolute_pose, Irls
    E = true_pose()
    cam = CameraModel(; focal = 800.0, pp = (640.0, 480.0))
    cs = pixel_pairs(cam, E, scene_points(E, 0.6; n = 12))
    ls = (; local_optimization = Irls{GaussianShape}(), threshold = 1.0e4)
    # the guess's frames come back (frames.md rule 1), and an unlabeled guess comes back unlabeled
    El = relabel(WorldFrame, CameraFrame, perturbed(E))
    Pl = estimate(refine_absolute_pose(El, cam, cs; ls...))
    @test Pl isa EuclideanMap{3, Float64, <:Any, SVector{3, Float64}, WorldFrame, CameraFrame}
    @test maximum(abs, Pl.R - E.R) < 1.0e-9
    @test typeof(estimate(refine_absolute_pose(perturbed(E), cam, cs; ls...))) === typeof(unlabel(E))
    # physical pose and points: read in metres, the pose back in metres
    Ep = EuclideanMap(perturbed(E).R, perturbed(E).t .* m)
    res = refine_absolute_pose(Ep, cam, [X .* m => u for (X, u) in cs]; ls...)
    @test maximum(abs, ustrip(m, estimate(res)).t - E.t) < 1.0e-9
    @test refine_absolute_pose(E, cam, cs[1:2]) === nothing
end

@testitem "refine_absolute_pose bearings: pinhole, pixel-whitened" setup = [PixelPoseScenes] begin
    using StaticArrays: SVector
    using LinearAlgebra: dot, norm
    using VisualGeometryToolkitCore: CameraModel, Csponds, withattrs, project, backproject_with_jacobian, estimate, converged, rms
    using VisualGeometryToolkitEstimation: refine_absolute_pose
    cam = CameraModel(; focal = 800.0, pp = (640.0, 480.0))
    E = true_pose()
    Xs = scene_points(E, 0.5)
    us = [SVector{2, Float64}(project(cam, E.R * X + E.t)) for X in Xs]
    fM = [backproject_with_jacobian(cam, u) for u in us]
    @test maximum(abs(dot(f, M[:, k])) for (f, M) in fM for k in 1:2) < 1.0e-12     # tangent: fᵀM = 0
    res = refine_absolute_pose(perturbed(E), withattrs(Csponds(Xs, first.(fM)); M = last.(fM)))
    @test converged(res) && res.lo_retcode === :converged && res.n == length(Xs)   # every correspondence, by default
    eR, et = pose_errors(estimate(res), E)
    @test eR < 1.0e-9 && et < 1.0e-9
    @test rms(res) < 1.0e-6                                                        # px
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
    @test isapprox(rms(refine_absolute_pose(E, Csponds(Xs, first.(fMn)))), reproj / 800; rtol = 0.1)
end

@testitem "refine_absolute_pose bearings: fisheye past 90° and a ray behind the camera" setup = [PixelPoseScenes] begin
    using StaticArrays: SVector
    using LinearAlgebra: normalize, dot
    using VisualGeometryToolkitCore: CameraModel, Csponds, withattrs, Lens, Equidistant, project, backproject_with_jacobian,
        estimate, converged, rms
    using VisualGeometryToolkitEstimation: refine_absolute_pose
    E = true_pose()
    Xs = scene_points(E, deg2rad(110))
    fs = [normalize(E.R * X + E.t) for X in Xs]
    @test count(f -> f[3] < 0, fs) >= 6                                    # no z = 1 point exists for these
    Xb = E.R' * (SVector(0.3, 0.1, -2.5) - E.t)                            # and one ray BEHIND the camera, 160° off the axis
    push!(Xs, Xb); push!(fs, normalize(E.R * Xb + E.t))
    res = refine_absolute_pose(perturbed(E), Csponds(Xs, fs))
    @test converged(res) && res.n == length(Xs)
    eR, et = pose_errors(estimate(res), E)
    @test eR < 1.0e-9 && et < 1.0e-9
    # the same rays through a fisheye CAMERA (equidistant, 190° field): its
    # back-projection's bearings and Jacobians, tangent, whiten into its pixels
    cam = CameraModel(; focal = 400.0, pp = (800.0, 800.0), lens = Lens(Equidistant()))
    keep = [i for i in 1:(length(Xs) - 1) if all(isfinite, project(cam, E.R * Xs[i] + E.t))]
    fM = [backproject_with_jacobian(cam, SVector{2, Float64}(project(cam, E.R * Xs[i] + E.t))) for i in keep]
    @test count(fm -> fm[1][3] < 0, fM) >= 6
    @test maximum(abs(dot(f, M[:, k])) for (f, M) in fM for k in 1:2) < 1.0e-10
    resf = refine_absolute_pose(perturbed(E), withattrs(Csponds(Xs[keep], first.(fM)); M = last.(fM)))
    @test converged(resf)
    eR, et = pose_errors(estimate(resf), E)
    @test eR < 1.0e-8 && et < 1.0e-8
    @test rms(resf) < 1.0e-6
end

@testitem "refine_absolute_pose bearings: generalized camera (ray origins), refine only" setup = [PixelPoseScenes] begin
    using StaticArrays: SVector
    using LinearAlgebra: normalize
    using VisualGeometryToolkitCore: Csponds, withattrs, estimate, converged, rms
    using VisualGeometryToolkitEstimation: refine_absolute_pose, BearingPoseProblem
    import VisualGeometryToolkitEstimation as VgtkEstimation
    E = true_pose()
    Xs = scene_points(E, deg2rad(100))
    os = [SVector(0.02 * sin(3i), 0.02 * cos(5i), 0.01 * sin(i)) for i in eachindex(Xs)]   # distinct origins, ~2 cm
    fs = [normalize(E.R * X + E.t - o) for (X, o) in zip(Xs, os)]
    cs = withattrs(Csponds(Xs, fs); origins = os)
    res = refine_absolute_pose(perturbed(E), cs)
    @test converged(res)
    eR, et = pose_errors(estimate(res), E)
    @test eR < 1.0e-9 && et < 1.0e-9
    # the central form on the same rays cannot fit them: the origins matter
    @test rms(refine_absolute_pose(perturbed(E), Csponds(Xs, fs))) > 1.0e-4
    # no generalized minimal solver: refine only; the central problem solves by P3P
    @test_throws ArgumentError VgtkEstimation.solve(BearingPoseProblem(cs), [1, 2, 3])
    sols = VgtkEstimation.solve(BearingPoseProblem(Csponds(Xs, [normalize(E.R * X + E.t) for X in Xs])), [1, 9, 17])
    @test any(P -> maximum(abs, P.R - E.R) < 1.0e-8, sols)
    # a ray origin on its point at the guess
    @test_throws ArgumentError refine_absolute_pose(E, withattrs(Csponds(Xs, fs); origins = [i == 1 ? E.R * Xs[1] + E.t : o for (i, o) in enumerate(os)]))
end

@testitem "refine_absolute_pose bearings: Triggs on contaminated rays" setup = [PixelPoseScenes] begin
    using StaticArrays: SVector, SMatrix
    using LinearAlgebra: normalize
    using Random: Xoshiro, randperm
    using VisualGeometryToolkitCore: Csponds, withattrs, estimate, converged, CauchyShape, WelschShape, s2_tangent_basis, EuclideanMap, exp_so3
    using Rotations: RotMatrix
    using VisualGeometryToolkitEstimation: refine_absolute_pose, Triggs
    E = true_pose(); rng = Xoshiro(3)
    Xs = scene_points(E, deg2rad(100))
    os = [SVector(0.02 * sin(3i), 0.02 * cos(5i), 0.01 * sin(i)) for i in eachindex(Xs)]
    Bof(f) = (b = s2_tangent_basis(f); hcat(SVector{3}(b[1]), SVector{3}(b[2])))
    fs = [(f = normalize(E.R * X + E.t - o); normalize(f + Bof(f) * (0.3 / 800 * SVector(randn(rng), randn(rng))))) for (X, o) in zip(Xs, os)]
    bad = randperm(rng, length(Xs))[1:10]
    for i in bad
        fs[i] = normalize(SVector(randn(rng), randn(rng), randn(rng)))
    end
    good = setdiff(eachindex(Xs), bad)
    Ms = [Bof(f) / 800 for f in fs]                                        # an 800 px pixel Jacobian, tangent
    cs = withattrs(Csponds(Xs, fs); origins = os, M = Ms)
    ls = estimate(refine_absolute_pose(E, withattrs(Csponds(Xs[good], fs[good]); origins = os[good], M = Ms[good])))
    guess = EuclideanMap(RotMatrix{3, Float64}(exp_so3(SVector(0.002, -0.0015, 0.002)) * E.R), E.t + SVector(0.004, -0.003, 0.003))
    @test pose_errors(estimate(refine_absolute_pose(guess, cs)), E)[1] > 0.1          # least squares is ruined
    for lo in (Triggs{CauchyShape}(scale_multiple = 2.5), Triggs{WelschShape}(scale_multiple = 2.5))
        r = refine_absolute_pose(guess, cs; local_optimization = lo, threshold = 8.0)   # px
        @test converged(r) && r.lo_retcode === :converged
        @test sort(findall(r.inlier_mask)) == sort(good)
        eR, et = pose_errors(estimate(r), ls)
        @test eR < 1.0e-4 && et < 1.0e-4                                   # the inliers' least squares
    end
end

@testitem "refine_absolute_pose bearings: refusals, frames and units" setup = [PixelPoseScenes] begin
    using StaticArrays: SVector
    using LinearAlgebra: normalize
    using VisualGeometryToolkitCore: Csponds, withattrs, estimate, EuclideanMap, relabel, unlabel, WorldFrame, CameraFrame, m, ustrip
    using VisualGeometryToolkitEstimation: refine_absolute_pose
    E = true_pose()
    Xs = scene_points(E, 0.6; n = 12)
    fs = [normalize(E.R * X + E.t) for X in Xs]
    # a z = 1 chart point is not a bearing
    @test_throws ArgumentError refine_absolute_pose(E, Csponds(Xs, [f / f[3] for f in fs]))
    # a Jacobian that is not tangent to its bearing (here: the bearing itself added to M)
    Mbad = [hcat(normalize(SVector(1.0, 0.0, 0.0) - f[1] * f) + f, normalize(SVector(0.0, 1.0, 0.0) - f[2] * f)) for f in fs]
    @test_throws ArgumentError refine_absolute_pose(E, withattrs(Csponds(Xs, fs); M = Mbad))
    @test refine_absolute_pose(E, Csponds(Xs[1:2], fs[1:2])) === nothing
    # the guess's frames come back (frames.md rule 1); an unlabeled guess comes back unlabeled
    El = relabel(WorldFrame, CameraFrame, perturbed(E))
    @test estimate(refine_absolute_pose(El, Csponds(Xs, fs))) isa EuclideanMap{3, Float64, <:Any, SVector{3, Float64}, WorldFrame, CameraFrame}
    @test typeof(estimate(refine_absolute_pose(perturbed(E), Csponds(Xs, fs)))) === typeof(unlabel(E))
    # physical pose, points and origins: read in metres, the pose back in metres
    os = [SVector(0.01, 0.0, 0.0) for _ in Xs]
    fo = [normalize(E.R * X + E.t - o) for (X, o) in zip(Xs, os)]
    Ep = EuclideanMap(perturbed(E).R, perturbed(E).t .* m)
    res = refine_absolute_pose(Ep, withattrs(Csponds([X .* m for X in Xs], fo); origins = [o .* m for o in os]))
    @test maximum(abs, ustrip(m, estimate(res)).t - E.t) < 1.0e-9
end
