# the AbsolutePoseProblem refine — the refine contract (no scorer; the caller's scale or
# width), from a given pose, on the AbsolutePoseProblem's rows (the raw-pixel reprojection
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

    import VisualGeometryToolkitEstimation as VgtkEstimation
    include(joinpath(pkgdir(VgtkEstimation), "test", "fixtures", "pixel_pose_scenes.jl"))   # shared with the golden records
    pose_errors(E, F) = (maximum(abs, E.R - F.R), maximum(abs, E.t - F.t))
end

@testitem "the AbsolutePoseProblem refine: pinhole and fisheye past 90°, any strategy" setup = [PixelPoseScenes] begin
    using VisualGeometryToolkitCore: solve, RefineProblem, LevenbergMarquardt, successful_retcode
    using VisualGeometryToolkitEstimation: AbsolutePoseProblem
    using VisualGeometryToolkitCore: inlier_mask
    using StaticArrays: SVector
    using VisualGeometryToolkitCore: CameraModel, Lens, Equidistant, GaussianShape, CauchyShape, rms, inlier_mask
    
    using VisualGeometryToolkitCore: Irls, Triggs
    E = true_pose()
    ls = (; optimization_strategy = nothing)   # plain least squares over every point
    pin = CameraModel(; focal = 800.0, pp = (640.0, 480.0))
    fish = CameraModel(; focal = 400.0, pp = (800.0, 800.0), lens = Lens(Equidistant()))
    for (cam, θmax) in ((pin, 0.5), (fish, deg2rad(110)))
        cs = pixel_pairs(cam, E, scene_points(E, θmax))
        cam === fish && @test count(((X, _),) -> (E.R * X + E.t)[3] < 0, cs) >= 6     # rays past 90°
        res = solve(RefineProblem(AbsolutePoseProblem(cam, cs), perturbed(E)), LevenbergMarquardt(; ls...))
        @test successful_retcode(res)
        eR, et = pose_errors(res.u, E)
        @test eR < 1.0e-9 && et < 1.0e-9
        @test res.rms < 1.0e-6                                            # px
        # under the class default (Triggs{Welsch}) at the pixels' scale, from a near pose
        near = solve(RefineProblem(AbsolutePoseProblem(cam, [X => u + SVector(0.3 * sin(7i), 0.3 * cos(11i)) for (i, (X, u)) in enumerate(cs)]), E), LevenbergMarquardt(); scale = 0.3)
        eR, et = pose_errors(near.u, E)
        @test count(inlier_mask(near, 0.5)) == length(cs) && eR < 1.0e-3 && et < 1.0e-3
    end
end

@testitem "the AbsolutePoseProblem refine: frames, units, too few" setup = [PixelPoseScenes] begin
    import VisualGeometryToolkitCore as VgtkCore
    using VisualGeometryToolkitCore: solve, RefineProblem, LevenbergMarquardt
    using VisualGeometryToolkitEstimation: AbsolutePoseProblem
    
    using StaticArrays: SVector
    using VisualGeometryToolkitCore: CameraModel, GaussianShape, EuclideanMap, relabel, unlabel, WorldFrame, CameraFrame, m, ustrip, MsacScoring
    
    using VisualGeometryToolkitCore: Irls
    E = true_pose()
    cam = CameraModel(; focal = 800.0, pp = (640.0, 480.0))
    cs = pixel_pairs(cam, E, scene_points(E, 0.6; n = 12))
    ls = (; optimization_strategy = nothing)
    # the guess's frames come back (frames.md rule 1), and an unlabeled guess comes back unlabeled
    El = relabel(WorldFrame, CameraFrame, perturbed(E))
    Pl = solve(RefineProblem(AbsolutePoseProblem(cam, cs), El), LevenbergMarquardt(; ls...)).u
    @test Pl isa EuclideanMap{3, Float64, <:Any, SVector{3, Float64}, WorldFrame, CameraFrame}
    @test maximum(abs, Pl.R - E.R) < 1.0e-9
    @test typeof(solve(RefineProblem(AbsolutePoseProblem(cam, cs), perturbed(E)), LevenbergMarquardt(; ls...)).u) === typeof(unlabel(E))
    # physical pose and points: read in metres, the pose back in metres
    Ep = EuclideanMap(perturbed(E).R, perturbed(E).t .* m)
    res = solve(RefineProblem(AbsolutePoseProblem(cam, [X .* m => u for (X, u) in cs]), Ep), LevenbergMarquardt(; ls...))
    @test maximum(abs, ustrip(m, res.u).t - E.t) < 1.0e-9
    @test let r = solve(RefineProblem(AbsolutePoseProblem(cam, cs[1:2]), E), LevenbergMarquardt(; ls...))
        (r.retcode, r.stop_rule) === (VgtkCore.ReturnCode.InitialFailure, VgtkCore.NoRule) && r.stats.nsteps == 0 && r.u.R == E.R
    end
end

@testitem "the AbsolutePoseProblem refine bearings: pinhole, pixel-whitened" setup = [PixelPoseScenes] begin
    using VisualGeometryToolkitCore: solve, RefineProblem, LevenbergMarquardt, successful_retcode
    using VisualGeometryToolkitEstimation: BearingPoseProblem
    using StaticArrays: SVector
    using LinearAlgebra: dot, norm
    using VisualGeometryToolkitCore: CameraModel, Correspondences, withattrs, project, backproject_with_jacobian, rms
    
    cam = CameraModel(; focal = 800.0, pp = (640.0, 480.0))
    E = true_pose()
    Xs = scene_points(E, 0.5)
    us = [SVector{2, Float64}(project(cam, E.R * X + E.t)) for X in Xs]
    fM = [backproject_with_jacobian(cam, u) for u in us]
    @test maximum(abs(dot(f, M[:, k])) for (f, M) in fM for k in 1:2) < 1.0e-12     # tangent: fᵀM = 0
    res = solve(RefineProblem(BearingPoseProblem(withattrs(Correspondences(Xs, first.(fM)); M = last.(fM))), perturbed(E)), LevenbergMarquardt(; optimization_strategy = nothing))
    @test successful_retcode(res) && res.n == length(Xs)   # every correspondence, under least squares
    eR, et = pose_errors(res.u, E)
    @test eR < 1.0e-9 && et < 1.0e-9
    @test res.rms < 1.0e-6                                                        # px
    # the pixel whitening against the reprojection error itself: with noisy
    # pixels, the refined rms (px) is the camera's own reprojection rms there
    noise = [SVector(0.3 * sin(7i), 0.3 * cos(11i)) for i in eachindex(us)]
    fMn = [backproject_with_jacobian(cam, u + δ) for (u, δ) in zip(us, noise)]
    resn = solve(RefineProblem(BearingPoseProblem(withattrs(Correspondences(Xs, first.(fMn)); M = last.(fMn))), E), LevenbergMarquardt(; optimization_strategy = nothing))
    En = resn.u
    reproj = sqrt(sum(norm(SVector{2, Float64}(project(cam, En.R * X + En.t)) - (u + δ))^2 for (X, u, δ) in zip(Xs, us, noise)) / length(Xs))
    @test isapprox(resn.rms, reproj; rtol = 1.0e-3)
    @test 0.1 < reproj < 0.5
    # without M the residual is the angle (radians): at f = 800 px, ≈ px / 800
    @test isapprox(solve(RefineProblem(BearingPoseProblem(Correspondences(Xs, first.(fMn))), E), LevenbergMarquardt(; optimization_strategy = nothing)).rms, reproj / 800; rtol = 0.1)
end

@testitem "the AbsolutePoseProblem refine bearings: fisheye past 90° and a ray behind the camera" setup = [PixelPoseScenes] begin
    using VisualGeometryToolkitCore: solve, RefineProblem, LevenbergMarquardt, successful_retcode
    using VisualGeometryToolkitEstimation: BearingPoseProblem
    using StaticArrays: SVector
    using LinearAlgebra: normalize, dot
    using VisualGeometryToolkitCore: CameraModel, Correspondences, withattrs, Lens, Equidistant, project, backproject_with_jacobian, rms
    
    E = true_pose()
    Xs = scene_points(E, deg2rad(110))
    fs = [normalize(E.R * X + E.t) for X in Xs]
    @test count(f -> f[3] < 0, fs) >= 6                                    # no z = 1 point exists for these
    Xb = E.R' * (SVector(0.3, 0.1, -2.5) - E.t)                            # and one ray BEHIND the camera, 160° off the axis
    push!(Xs, Xb); push!(fs, normalize(E.R * Xb + E.t))
    res = solve(RefineProblem(BearingPoseProblem(Correspondences(Xs, fs)), perturbed(E)), LevenbergMarquardt(; optimization_strategy = nothing))
    @test successful_retcode(res) && res.n == length(Xs)
    eR, et = pose_errors(res.u, E)
    @test eR < 1.0e-9 && et < 1.0e-9
    # the same rays through a fisheye CAMERA (equidistant, 190° field): its
    # back-projection's bearings and Jacobians, tangent, whiten into its pixels
    cam = CameraModel(; focal = 400.0, pp = (800.0, 800.0), lens = Lens(Equidistant()))
    keep = [i for i in 1:(length(Xs) - 1) if all(isfinite, project(cam, E.R * Xs[i] + E.t))]
    fM = [backproject_with_jacobian(cam, SVector{2, Float64}(project(cam, E.R * Xs[i] + E.t))) for i in keep]
    @test count(fm -> fm[1][3] < 0, fM) >= 6
    @test maximum(abs(dot(f, M[:, k])) for (f, M) in fM for k in 1:2) < 1.0e-10
    resf = solve(RefineProblem(BearingPoseProblem(withattrs(Correspondences(Xs[keep], first.(fM)); M = last.(fM))), perturbed(E)), LevenbergMarquardt(; optimization_strategy = nothing))
    @test successful_retcode(resf)
    eR, et = pose_errors(resf.u, E)
    @test eR < 1.0e-8 && et < 1.0e-8
    @test resf.rms < 1.0e-6
end

@testitem "the AbsolutePoseProblem refine bearings: generalized camera (ray origins), refine only" setup = [PixelPoseScenes] begin
    using VisualGeometryToolkitEstimation: sample_models
    using VisualGeometryToolkitCore: solve, RefineProblem, LevenbergMarquardt, successful_retcode
    using StaticArrays: SVector
    using LinearAlgebra: normalize
    using VisualGeometryToolkitCore: Correspondences, withattrs, rms
    using VisualGeometryToolkitEstimation: BearingPoseProblem
    import VisualGeometryToolkitEstimation as VgtkEstimation
    E = true_pose()
    Xs = scene_points(E, deg2rad(100))
    os = [SVector(0.02 * sin(3i), 0.02 * cos(5i), 0.01 * sin(i)) for i in eachindex(Xs)]   # distinct origins, ~2 cm
    fs = [normalize(E.R * X + E.t - o) for (X, o) in zip(Xs, os)]
    cs = withattrs(Correspondences(Xs, fs); origins = os)
    res = solve(RefineProblem(BearingPoseProblem(cs), perturbed(E)), LevenbergMarquardt(; optimization_strategy = nothing))
    @test successful_retcode(res)
    eR, et = pose_errors(res.u, E)
    @test eR < 1.0e-9 && et < 1.0e-9
    # the central form on the same rays cannot fit them: the origins matter
    @test solve(RefineProblem(BearingPoseProblem(Correspondences(Xs, fs)), perturbed(E)), LevenbergMarquardt(; optimization_strategy = nothing)).rms > 1.0e-4
    # no generalized minimal solver: refine only; the central problem solves by P3P
    @test_throws ArgumentError VgtkEstimation.sample_models(BearingPoseProblem(cs), [1, 2, 3])
    sols = VgtkEstimation.sample_models(BearingPoseProblem(Correspondences(Xs, [normalize(E.R * X + E.t) for X in Xs])), [1, 9, 17])
    @test any(P -> maximum(abs, P.R - E.R) < 1.0e-8, sols)
    # a ray origin on its point at the guess
    @test_throws ArgumentError solve(RefineProblem(BearingPoseProblem(withattrs(Correspondences(Xs, fs); origins = [i == 1 ? E.R * Xs[1] + E.t : o for (i, o) in enumerate(os)])), E), LevenbergMarquardt(; optimization_strategy = nothing))
end

@testitem "the AbsolutePoseProblem refine bearings: Triggs on contaminated rays" setup = [PixelPoseScenes] begin
    using VisualGeometryToolkitCore: weights
    using VisualGeometryToolkitCore: solve, RefineProblem, LevenbergMarquardt, successful_retcode
    using VisualGeometryToolkitEstimation: BearingPoseProblem
    using VisualGeometryToolkitCore: inlier_mask
    using StaticArrays: SVector, SMatrix
    using LinearAlgebra: normalize
    using Random: Xoshiro, randperm
    using VisualGeometryToolkitCore: Correspondences, withattrs, CauchyShape, WelschShape, s2_tangent_basis, EuclideanMap, exp_so3, inlier_mask
    using Rotations: RotMatrix
    
    using VisualGeometryToolkitCore: Triggs
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
    cs = withattrs(Correspondences(Xs, fs); origins = os, M = Ms)
    ls = solve(RefineProblem(BearingPoseProblem(withattrs(Correspondences(Xs[good], fs[good]); origins = os[good], M = Ms[good])), E), LevenbergMarquardt(; optimization_strategy = nothing)).u
    guess = EuclideanMap(RotMatrix{3, Float64}(exp_so3(SVector(0.002, -0.0015, 0.002)) * E.R), E.t + SVector(0.004, -0.003, 0.003))
    # least squares is ruined: over ten times the inliers' own least-squares error
    @test pose_errors(solve(RefineProblem(BearingPoseProblem(cs), guess), LevenbergMarquardt(; optimization_strategy = nothing)).u, E)[1] > 10 * pose_errors(ls, E)[1]
    # Triggs at the rays' pixel noise (σ = 0.3 px through the 800 px Jacobians): the outliers at no weight, the
    # pose as close to the truth as the inliers' least squares (a loss weight is not a 0/1 inlier set)
    for lo in (Triggs{CauchyShape}(efficiency_constant = 2.5), Triggs{WelschShape}())
        r = solve(RefineProblem(BearingPoseProblem(cs), guess), LevenbergMarquardt(; optimization_strategy = lo); scale = 0.3)
        @test successful_retcode(r)
        @test all(<(0.05), weights(r)[bad]) && count(>(0.05), weights(r)[good]) >= length(good) - 1
        @test pose_errors(r.u, E)[1] < 3 * pose_errors(ls, E)[1] + 1.0e-6
    end
end

@testitem "the AbsolutePoseProblem refine bearings: refusals, frames and units" setup = [PixelPoseScenes] begin
    import VisualGeometryToolkitCore as VgtkCore
    using VisualGeometryToolkitCore: solve, RefineProblem, LevenbergMarquardt
    using VisualGeometryToolkitEstimation: BearingPoseProblem
    
    using StaticArrays: SVector
    using LinearAlgebra: normalize
    using VisualGeometryToolkitCore: Correspondences, withattrs, EuclideanMap, relabel, unlabel, WorldFrame, CameraFrame, m, ustrip
    
    E = true_pose()
    Xs = scene_points(E, 0.6; n = 12)
    fs = [normalize(E.R * X + E.t) for X in Xs]
    # a z = 1 chart point is not a bearing
    @test_throws ArgumentError solve(RefineProblem(BearingPoseProblem(Correspondences(Xs, [f / f[3] for f in fs])), E), LevenbergMarquardt(; optimization_strategy = nothing))
    # a Jacobian that is not tangent to its bearing (here: the bearing itself added to M)
    Mbad = [hcat(normalize(SVector(1.0, 0.0, 0.0) - f[1] * f) + f, normalize(SVector(0.0, 1.0, 0.0) - f[2] * f)) for f in fs]
    @test_throws ArgumentError solve(RefineProblem(BearingPoseProblem(withattrs(Correspondences(Xs, fs); M = Mbad)), E), LevenbergMarquardt(; optimization_strategy = nothing))
    @test let r = solve(RefineProblem(BearingPoseProblem(Correspondences(Xs[1:2], fs[1:2])), E), LevenbergMarquardt(; optimization_strategy = nothing))
        (r.retcode, r.stop_rule) === (VgtkCore.ReturnCode.InitialFailure, VgtkCore.NoRule) && r.stats.nsteps == 0 && r.u.R == E.R
    end
    # the guess's frames come back (frames.md rule 1); an unlabeled guess comes back unlabeled
    El = relabel(WorldFrame, CameraFrame, perturbed(E))
    @test solve(RefineProblem(BearingPoseProblem(Correspondences(Xs, fs)), El), LevenbergMarquardt(; optimization_strategy = nothing)).u isa EuclideanMap{3, Float64, <:Any, SVector{3, Float64}, WorldFrame, CameraFrame}
    @test typeof(solve(RefineProblem(BearingPoseProblem(Correspondences(Xs, fs)), perturbed(E)), LevenbergMarquardt(; optimization_strategy = nothing)).u) === typeof(unlabel(E))
    # physical pose, points and origins: read in metres, the pose back in metres
    os = [SVector(0.01, 0.0, 0.0) for _ in Xs]
    fo = [normalize(E.R * X + E.t - o) for (X, o) in zip(Xs, os)]
    Ep = EuclideanMap(perturbed(E).R, perturbed(E).t .* m)
    res = solve(RefineProblem(BearingPoseProblem(withattrs(Correspondences([X .* m for X in Xs], fo); origins = [o .* m for o in os])), Ep), LevenbergMarquardt(; optimization_strategy = nothing))
    @test maximum(abs, ustrip(m, res.u).t - E.t) < 1.0e-9
end

@testitem "the AbsolutePoseProblem refine: a SplineLens camera (its projection differentiated by the solve's implicit derivative)" setup = [PixelPoseScenes] begin
    using VisualGeometryToolkitCore: solve, RefineProblem, LevenbergMarquardt, successful_retcode
    using VisualGeometryToolkitEstimation: AbsolutePoseProblem
    # The pose refine's rows through a SplineLens BoundedCamera: ForwardDiff through `project` (Core's Dual method: the
    # plain solve, the partials by `_projection_jacobians`), the pose recovered from exact pixels as for the parametric
    # lenses; and with a fitted forward spline (the value and the partials the forward's).
    using StaticArrays: SVector
    using VisualGeometryToolkitCore: CameraModel, SplineLens, KannalaBrandt, BoundedCamera, Size2, rms, model, imaged_region, project
    import ForwardDiff
    using VisualGeometryToolkitEstimation: CameraRegressionProblem, CameraRegression
    using VisualGeometryToolkitCore: solve, EstimationProblem, RefineProblem, LevenbergMarquardt, LmStops, successful_retcode, RadialCamera, CameraModel
    E = true_pose()
    src = CameraModel(; focal = 420.0, pp = (640.0, 480.0), lens = KannalaBrandt(-0.02, 0.004, -0.001, 0.0002))
    cam = BoundedCamera(CameraModel(SplineLens, src, Size2(1280, 960); grid_size = (12, 10)), Size2(1280, 960))
    fwd = BoundedCamera(solve(EstimationProblem(CameraRegressionProblem(SplineLens, cam, imaged_region(cam))), CameraRegression(; tolerance = 1.0e-3)).u, Size2(1280, 960))
    # with a forward the value and the partials are both the forward's: one map, so the same 1e-9 as the backward's solve
    J(c, X) = ForwardDiff.jacobian(t -> project(model(c), t), X)
    Xt = SVector(0.2, -0.1, 1.0); h = 1.0e-6
    Jfd(c, X) = reduce(hcat, [(project(model(c), X + h * e) - project(model(c), X - h * e)) / 2h for e in (SVector(1.0, 0, 0), SVector(0, 1.0, 0), SVector(0, 0, 1.0))])
    @test isapprox(J(fwd, Xt), Jfd(fwd, Xt); rtol = 1.0e-6) && isapprox(J(cam, Xt), Jfd(cam, Xt); rtol = 1.0e-6)
    for (c, tol) in ((cam, 1.0e-9), (fwd, 1.0e-9))
        cs = pixel_pairs(model(c), E, scene_points(E, deg2rad(70)))
        @test length(cs) >= 30
        res = solve(RefineProblem(AbsolutePoseProblem(model(c), cs), perturbed(E)), LevenbergMarquardt(; optimization_strategy = nothing))
        @test successful_retcode(res)
        eR, et = pose_errors(res.u, E)
        @test eR < tol && et < tol
        @test res.rms < 1.0e-6
    end
end

# Plain least squares minimises the PIXEL reprojection residual, the one Triggs robustifies (`_triggs_state`): at an
# unbounded width Huber is L2, so the two agree to the LM's tolerance. On an anisotropic pinhole (fx ≠ fy) the
# calibrated-plane rows L2 descended before weigh the two axes differently, a different minimum on noisy pixels.
@testitem "the AbsolutePoseProblem refine: L2 minimises the pixels Triggs does" setup = [PixelPoseScenes] begin
    using VisualGeometryToolkitCore: solve, RefineProblem, LevenbergMarquardt, successful_retcode
    using VisualGeometryToolkitEstimation: AbsolutePoseProblem
    using StaticArrays: SVector
    using VisualGeometryToolkitCore: CameraModel, HuberShape, Triggs
    
    E = true_pose()
    cam = CameraModel(; focal = (1200.0, 400.0), pp = (640.0, 480.0))
    cs = [X => u + SVector(2.0 * sin(7i), 2.0 * cos(11i)) for (i, (X, u)) in enumerate(pixel_pairs(cam, E, scene_points(E, 0.5)))]
    l2 = solve(RefineProblem(AbsolutePoseProblem(cam, cs), perturbed(E)), LevenbergMarquardt(; optimization_strategy = nothing))
    tr = solve(RefineProblem(AbsolutePoseProblem(cam, cs), perturbed(E)), LevenbergMarquardt(; optimization_strategy = Triggs{HuberShape}()); width = 1.0e9)
    @test successful_retcode(l2) && successful_retcode(tr)
    eR, et = pose_errors(l2.u, tr.u)
    @test eR < 1.0e-9 && et < 1.0e-9
end
