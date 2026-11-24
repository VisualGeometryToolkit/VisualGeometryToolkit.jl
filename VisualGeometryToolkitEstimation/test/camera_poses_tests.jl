# CameraPosesProblem — a camera and one pose per view from 3D → 2D
# correspondences.
#
# Correctness: from exact pixels, a perturbed camera (intrinsics and lens) and
# perturbed poses are recovered to the LM's tolerance — a pinhole camera, and a
# Kannala–Brandt fisheye whose rays reach past 90°. The Jacobian the stall
# postcondition reads is in the charts' tangent coordinates: its pose block is
# the projection's Jacobian times the left-perturbation action
# (`Jπ·[−[X_c]×  I₃]`), its camera block the retraction's central difference.
# The held camera, the dict and vector forms, lengths, and the refusals.
#
# A SplineLens camera (its K a frozen gauge, its parameters the backward controls
# of its present nodes): from exact pixels of a KB-derived spline camera, its
# perturbed controls and perturbed poses are refined, K bit-identical, the lens
# axis kept on ẑ by the rotation gauge, the block Jacobian (implicit
# differentiation of the projection solve) against finite differences of the
# residual, the poses alone, a forward spline refitted, a masked lattice.

@testsetup module CameraPoseScenes
    import VisualGeometryToolkitEstimation as VgtkEstimation
    include(joinpath(pkgdir(VgtkEstimation), "test", "fixtures", "camera_pose_scenes.jl"))   # shared with the golden records
    export views_of, truth_poses, perturbed_pose, pinhole_camera, fisheye_camera
end

@testitem "CameraPosesProblem: a pinhole camera and its poses recovered" setup = [CameraPoseScenes] begin
    using VisualGeometryToolkitCore: dof, s2
    using VisualGeometryToolkitCore: retcode_symbol
    using StaticArrays: SVector
    using VisualGeometryToolkitCore: CameraModel, Pinhole, parameter_vector, parameter_names, with_parameters
    using VisualGeometryToolkitEstimation: CameraPosesProblem
    using VisualGeometryToolkitCore: solve, EstimationProblem, RefineProblem, LevenbergMarquardt, LmStops, successful_retcode, RadialCamera, CameraModel
    cam = pinhole_camera(); Es = truth_poses(5)
    views = views_of(cam, Es, 0.6)
    θ = parameter_vector(cam)
    cam0 = with_parameters(cam, θ .* SVector(1.04, 1.01, 1.0, 1.0, 1.0) .+ SVector(0.0, 0.0, 0.0, 6.0, -5.0))   # f, aspect, skew, cx, cy
    res = solve(RefineProblem(CameraPosesProblem(views), (; camera = cam0, poses = [perturbed_pose(E, v) for (v, E) in enumerate(Es)])), LevenbergMarquardt(; optimization_strategy = nothing))
    @test retcode_symbol(res) === :converged
    est = res.u
    @test maximum(abs.(parameter_vector(est.camera) - θ) ./ max.(abs.(θ), 1.0)) < 1.0e-9
    @test maximum(maximum(abs, P.R - E.R) for (P, E) in zip(est.poses, Es)) < 1.0e-9
    @test maximum(maximum(abs, P.t - E.t) for (P, E) in zip(est.poses, Es)) < 1.0e-9
    @test res.rms < 1.0e-6 && length(res.metadata.residuals) == 2 * 5 * 60 && res.metadata.free == (:f, :aspect, :cx, :cy) && dof(res) == 600 - 4 - 30
    @test parameter_vector(est.camera)[3] == 0.0                                       # the skew is held by default
end

# Core 0.11.1: an exact fit is converged. CaliBlobs' README refine of a Division camera and six board poses from exact
# correspondences ends at rms 1.8e-13 px with NLLSsolver's bit 10 (a rounding-level cost disagreement) beside its step
# stop: under Core 0.11.0 it read (Stalled, NonFinite), `successful_retcode(res)` false — and from other starts and view counts
# (Failure, NonFinite) — every residual finite.
@testitem "CameraPosesProblem: an exact fit from exact correspondences is converged (Core 0.11.1)" begin
    using VisualGeometryToolkitCore: dof, s2
    using VisualGeometryToolkitCore: CameraModel, LogicalIntrinsics, CameraCalibrationMat, Division, EuclideanMap, Vec3, project, px, ReturnCode, NonFinite, parameter_vector
    using Rotations: RotationVec
    using VisualGeometryToolkitEstimation: CameraPosesProblem
    using VisualGeometryToolkitCore: solve, EstimationProblem, RefineProblem, LevenbergMarquardt, LmStops, successful_retcode, RadialCamera, CameraModel
    truth = CameraModel(LogicalIntrinsics(CameraCalibrationMat(900.0px, [2120.0px, 1416.0px])), Division(-0.2, -0.08))
    board = [Vec3(x, y, 0.0) for x in -60.0:30.0:60.0 for y in -60.0:30.0:60.0]
    poses = Dict(t => EuclideanMap(RotationVec(0.2sin(t), 0.2cos(t), 0.1t), Vec3(40.0t - 140, 20.0 - 10t, 300.0)) for t in 1:6)
    views = Dict(t => [X => project(truth, P(X)) for X in board] for (t, P) in poses)
    for (n, f0) in ((6, 945.0), (6, 930.0), (4, 945.0), (3, 945.0))   # 0.11.0: Stalled, Stalled, Failure, Success
        start = CameraModel(LogicalIntrinsics(CameraCalibrationMat(f0 * px, [2120.0px, 1416.0px])), Division(-0.18, -0.072))
        res = solve(RefineProblem(CameraPosesProblem(Dict(t => views[t] for t in 1:n); fixed = (:aspect,)), (; camera = start, poses = Dict(t => poses[t] for t in 1:n))), LevenbergMarquardt(; optimization_strategy = nothing))
        @test res.retcode === ReturnCode.Success && successful_retcode(res)
        @test res.stop_rule !== NonFinite && res.rms < 1.0e-10
        @test abs(parameter_vector(res.u.camera)[1] - 900.0) < 1.0e-9 * 900.0      # f recovered
    end
end

@testitem "CameraPosesProblem: a fisheye past 90°, camera and poses recovered" setup = [CameraPoseScenes] begin
    using VisualGeometryToolkitCore: dof, s2
    using VisualGeometryToolkitCore: retcode_symbol
    using StaticArrays: SVector
    using LinearAlgebra: norm
    using VisualGeometryToolkitCore: parameter_vector, with_parameters, KannalaBrandt, lens, free_parameter_names
    using VisualGeometryToolkitEstimation: CameraPosesProblem
    using VisualGeometryToolkitCore: solve, EstimationProblem, RefineProblem, LevenbergMarquardt, LmStops, successful_retcode, RadialCamera, CameraModel
    cam = fisheye_camera(); Es = truth_poses(4)
    views = views_of(cam, Es, deg2rad(100))                                      # rays up to 100° off the axis
    θ = parameter_vector(cam)
    δ = SVector(1.03, 1.0, 1.0, 1.0, 1.0, 0.8, 1.2, 0.9, 1.1)                    # f, aspect, skew, cx, cy, k1 … k4
    cam0 = with_parameters(cam, θ .* δ .+ SVector(0.0, 0.0, 0.0, -4.0, 3.0, 0.0, 0.0, 0.0, 0.0))
    res = solve(RefineProblem(CameraPosesProblem(views), (; camera = cam0, poses = [perturbed_pose(E, v) for (v, E) in enumerate(Es)])), LevenbergMarquardt(; optimization_strategy = nothing))
    @test retcode_symbol(res) === :converged
    est = res.u
    @test lens(est.camera) isa KannalaBrandt                                    # the family is kept
    @test maximum(abs.(parameter_vector(est.camera) - θ) ./ max.(abs.(θ), 1.0e-2)) < 1.0e-7
    @test maximum(maximum(abs, P.R - E.R) for (P, E) in zip(est.poses, Es)) < 1.0e-9
    @test maximum(norm(P.t - E.t) for (P, E) in zip(est.poses, Es)) < 1.0e-9
    @test res.rms < 1.0e-6
    # `fixed` takes a group, by Core's one expansion: `:k` holds k1 … k4 (and `:skew`, the gauge, is accepted)
    held = solve(RefineProblem(CameraPosesProblem(views; fixed = (:k, :skew)), (; camera = cam0, poses = [perturbed_pose(E, v) for (v, E) in enumerate(Es)])), LevenbergMarquardt(; optimization_strategy = nothing))
    @test held.metadata.free == (:f, :aspect, :cx, :cy) == free_parameter_names(cam0, (:k,))
    @test parameter_vector(held.u.camera)[6:9] == parameter_vector(cam0)[6:9]
end

@testitem "CameraPosesProblem: the Jacobian is in the charts' tangent coordinates" setup = [CameraPoseScenes] begin
    using VisualGeometryToolkitCore: dof, s2
    using StaticArrays: SVector, SMatrix
    using LinearAlgebra: norm
    import ForwardDiff
    using VisualGeometryToolkitCore: LeftRigidManifold6, ParameterManifold, restrict, retract, project, action_jacobian, reprojection_error, model
    using VisualGeometryToolkitEstimation: VisualGeometryToolkitEstimation as VE
    cam = fisheye_camera(); Es = truth_poses(2)
    views = views_of(cam, Es, deg2rad(95); n = 8)
    poses0 = [perturbed_pose(E, v) for (v, E) in enumerate(Es)]
    free = (:f, :aspect, :cx, :cy, :k1, :k2, :k3, :k4)
    chc = restrict(ParameterManifold(cam), free)
    chs = [LeftRigidManifold6(P) for P in poses0]
    world = model(LeftRigidManifold6(SMatrix{3, 3, Float64}(1.0, 0, 0, 0, 1.0, 0, 0, 0, 1.0), SVector(0.0, 0.0, 0.0)))   # the held identity board
    costs = [VE._BoardViewResidual{2, (true, true, false), typeof(cam)}(SVector(1, 1 + v), cam, world, world, SVector{3, Float64}(first(c)), last(c))
             for v in 1:2 for c in views[v]]
    using VisualGeometryToolkitCore: nlls_rows_and_jacobian
    r, J = nlls_rows_and_jacobian(costs, [chc; chs])                 # Core's assembler of the stall postcondition's rows
    @test size(J) == (32, 8 + 12)
    for (k, c) in enumerate(costs)
        v = c.varind[2] - 1
        P = model(chs[v]); Xc = P.R * c.X + P.t
        @test r[(2k - 1):(2k)] ≈ reprojection_error(cam, P, c.X, c.u)
        # the pose block: Jπ · [−[X_c]× I₃], the left perturbation's action (zero outside the view's columns)
        Jπ = ForwardDiff.jacobian(x -> project(cam, x), Xc)
        @test J[(2k - 1):(2k), (8 + 6v - 5):(8 + 6v)] ≈ Jπ * action_jacobian(LeftRigidManifold6(P), c.X) rtol = 1.0e-10
        @test iszero(J[(2k - 1):(2k), (8 + 6(3 - v) - 5):(8 + 6(3 - v))])
        # the camera block: the central difference of the retraction
        h = 1.0e-6
        fd = reduce(hcat, [begin
            e = SVector{8}(ntuple(i -> i == j ? h : 0.0, 8))
            (reprojection_error(retract(chc, e), P, c.X, c.u) - reprojection_error(retract(chc, -e), P, c.X, c.u)) / 2h
        end for j in 1:8])
        @test J[(2k - 1):(2k), 1:8] ≈ fd rtol = 1.0e-6
    end
end

@testitem "CameraPosesProblem: a held camera, dict and vector forms, lengths, refusals" setup = [CameraRecipes, CameraPoseScenes] begin
    using VisualGeometryToolkitCore: dof, s2
    using VisualGeometryToolkitCore: retcode_symbol
    using StaticArrays: SVector
    using VisualGeometryToolkitCore: parameter_vector, parameter_names, with_parameters, EuclideanMap, m, ustrip, with_unit, Correspondences
    using VisualGeometryToolkitEstimation: CameraPosesProblem
    using VisualGeometryToolkitCore: solve, EstimationProblem, RefineProblem, LevenbergMarquardt, LmStops, successful_retcode, RadialCamera, CameraModel
    cam = pinhole_camera(); Es = truth_poses(3)
    views = views_of(cam, Es, 0.5)
    poses0 = [perturbed_pose(E, v) for (v, E) in enumerate(Es)]
    # the camera held: the poses alone (the camera comes back as given)
    held = solve(RefineProblem(CameraPosesProblem(views; fixed = parameter_names(cam)), (; camera = cam, poses = poses0)), LevenbergMarquardt(; optimization_strategy = nothing))
    @test retcode_symbol(held) === :converged && held.u.camera === cam && held.metadata.free == ()
    # `fixed` names the held parameters, the complement of the refined set (the retired `free`), bit for bit:
    # holding the aspect refines (f, cx, cy) exactly as the former `free = (:f, :cx, :cy)` did
    part = solve(RefineProblem(CameraPosesProblem(views; fixed = (:aspect,)), (; camera = cam, poses = poses0)), LevenbergMarquardt(; optimization_strategy = nothing))
    @test part.metadata.free == (:f, :cx, :cy) && parameter_vector(part.u.camera)[2] === parameter_vector(cam)[2]
    @test solve(RefineProblem(CameraPosesProblem(views; fixed = ()), (; camera = cam, poses = poses0)), LevenbergMarquardt(; optimization_strategy = nothing)).metadata.free ==
        solve(RefineProblem(CameraPosesProblem(views), (; camera = cam, poses = poses0)), LevenbergMarquardt(; optimization_strategy = nothing)).metadata.free
    # the other spellings are refused, naming `fixed`: Core's one refusal of a retired keyword
    for kw in (:free, :fix, :fixed_params, :hold)
        @test_throws ArgumentError("the keyword `$kw` is retired: give `fixed`") solve(RefineProblem(CameraPosesProblem(views; (; kw => ())...), (; camera = cam, poses = poses0)), LevenbergMarquardt(; optimization_strategy = nothing))
    end
    @test maximum(maximum(abs, P.t - E.t) for (P, E) in zip(held.u.poses, Es)) < 1.0e-9
    # a dict keyed by view: the same estimate, keyed back; a Correspondences view is one too
    cam0 = with_parameters(cam, parameter_vector(cam) .* SVector(1.02, 1.0, 1.0, 1.0, 1.0))
    rv = solve(RefineProblem(CameraPosesProblem(views), (; camera = cam0, poses = poses0)), LevenbergMarquardt(; optimization_strategy = nothing))
    rd = solve(RefineProblem(CameraPosesProblem(Dict(:a => views[1], :b => views[2], :c => Correspondences(first.(views[3]), last.(views[3])))), (; camera = cam0, poses = Dict(:c => poses0[3], :a => poses0[1], :b => poses0[2]))), LevenbergMarquardt(; optimization_strategy = nothing))
    @test parameter_vector(rd.u.camera) ≈ parameter_vector(rv.u.camera) rtol = 1.0e-12
    @test rd.u.poses[:b].t ≈ rv.u.poses[2].t rtol = 1.0e-12
    # lengths in, lengths out (metres), the same refinement
    rp = solve(RefineProblem(CameraPosesProblem([[X .* m => u for (X, u) in cs] for cs in views]), (; camera = cam0, poses = [with_unit(P, m) for P in poses0])), LevenbergMarquardt(; optimization_strategy = nothing))
    @test ustrip(m, rp.u.poses[1]).t ≈ rv.u.poses[1].t rtol = 1.0e-12
    # refusals
    @test_throws ArgumentError solve(RefineProblem(CameraPosesProblem(Dict(:b => views[1])), (; camera = cam0, poses = Dict(:a => poses0[1]))), LevenbergMarquardt(; optimization_strategy = nothing))
    @test_throws ArgumentError solve(RefineProblem(CameraPosesProblem(views), (; camera = cam0, poses = poses0[1:2])), LevenbergMarquardt(; optimization_strategy = nothing))
    @test_throws ArgumentError solve(RefineProblem(CameraPosesProblem(views; fixed = (:k9,)), (; camera = cam0, poses = poses0)), LevenbergMarquardt(; optimization_strategy = nothing))
    @test_throws ArgumentError solve(RefineProblem(CameraPosesProblem(views), (; camera = cam0, poses = [EuclideanMap(Es[1].R, SVector(0.0, 0.0, -50.0)); poses0[2:3]])), LevenbergMarquardt(; optimization_strategy = nothing))   # behind the camera
    @test_throws ArgumentError solve(RefineProblem(CameraPosesProblem([views[1][1:2]]), (; camera = cam0, poses = poses0[1:1])), LevenbergMarquardt(; optimization_strategy = nothing))   # underdetermined
    @test_throws ArgumentError solve(RefineProblem(CameraPosesProblem(views; fixed = parameter_names(cam0), box = (; f = (1.0, 2.0))), (; camera = cam0, poses = poses0)), LevenbergMarquardt(; optimization_strategy = nothing))   # a box with nothing to bound
    # the wall-clock cap reaches the LM: a zero cap stops after the first iteration, :max_time; Inf is no cap
    rt = solve(RefineProblem(CameraPosesProblem(views), (; camera = cam0, poses = poses0)), LevenbergMarquardt(; optimization_strategy = nothing); stops = poses_stops(; max_time = 0.0))
    @test retcode_symbol(rt) === :max_time && rt.stats.nsteps <= 1
    @test solve(RefineProblem(CameraPosesProblem(views), (; camera = cam0, poses = poses0)), LevenbergMarquardt(; optimization_strategy = nothing); stops = poses_stops(; max_time = Inf)).u.camera == rv.u.camera
end

@testsetup module SplineCameraPoseScenes
    using StaticArrays: SVector
    using LinearAlgebra: normalize, norm
    using VisualGeometryToolkitCore: CameraModel, SplineLens, ChartSpline, KannalaBrandt, Size2, project, reprojection_error, backproject, angle_between
    using Test: @test, @test_throws
    using VisualGeometryToolkitCore: retcode_symbol
    using VisualGeometryToolkitEstimation: CameraPosesProblem, CameraRegressionProblem, OpticalAxis
    using VisualGeometryToolkitCore: solve, EstimationProblem, RefineProblem, LevenbergMarquardt, LmStops, successful_retcode, RadialCamera, CameraModel
    using VisualGeometryToolkitCore: axis_gauged, calibration_matrix, rotation, BoundedCamera, Size2, model
    using Rotations: RotMatrix
    import VisualGeometryToolkitEstimation as VgtkEstimation
    include(joinpath(pkgdir(VgtkEstimation), "test", "fixtures", "camera_pose_scenes.jl"))   # shared with the golden records
    export spline_truth, perturbed_controls, spline_views, spline_rms, ray_error, entry_gauge, truth_in_frame, bounded, model

    # a SplineLens camera is refined as a bounded camera (2026-11-11, owner: its optical axis is fitted over the camera's
    # own bound, never a default region): the fixture's whole 1280 × 960 sensor
    bounded(c) = BoundedCamera(c, Size2(1280, 960))

    # the camera and the rotation the refine's entry gives a guess: its optical axis (`OpticalAxis`), then Core's
    # `axis_gauged` — the frame and the K the refine then holds exactly
    function entry_gauge(cam)
        b = cam isa BoundedCamera ? cam : bounded(cam)
        r = solve(EstimationProblem(CameraRegressionProblem(b)), OpticalAxis())
        K = calibration_matrix(r.u)
        R = rotation(r.metadata.rotation)
        return axis_gauged(model(b), R, K), R
    end
    # the truth and its poses in an estimate's frame (2026-11-11: the refine holds its guess's axis-gauged frame, no
    # longer the truth's canonical one): turned by the rotation Q taking the truth's first pose onto the estimate's — the
    # view whose rotation the joint solve holds; a turn of the frame changes no image
    function truth_in_frame(truth, Es, est)
        Q = RotMatrix{3, Float64}(est.poses[1].R * Es[1].R')
        K = calibration_matrix(model(est.camera))
        return axis_gauged(truth, Q, K), [axis_gauged(E, Q) for E in Es]
    end

    # the largest angle between two cameras' rays at the detections
    ray_error(a, b, views) = maximum(angle_between(backproject(a, last(x)), backproject(b, last(x))) for cs in views for x in cs)
    spline_rms(camera, poses, views) =
        sqrt(sum(sum(abs2, reprojection_error(camera, P, first(c), last(c))) for (P, cs) in zip(poses, views) for c in cs) / sum(length, views))
end

# A robust refine (`optimization_strategy`, the refine contract): Triggs{Shape} (the default, Triggs{Welsch}) or
# Triggs{Huber} at the caller's `scale` times the strategy's efficiency constant, or at the caller's `width`, from the
# caller's start (no least-squares stage); a refine estimates no scale; `nothing` is plain least squares; the refusals.
@testitem "CameraPosesProblem: optimization_strategy = Triggs{Huber} / Triggs{Welsch} from the caller's start at the caller's scale, no least-squares stage; least squares unchanged" setup = [CameraRecipes, CameraPoseScenes] begin
    using VisualGeometryToolkitCore: dof, s2
    using VisualGeometryToolkitCore: retcode_symbol
    using StaticArrays: SVector
    using LinearAlgebra: normalize
    using Random: Xoshiro
    using VisualGeometryToolkitCore: parameter_vector, with_parameters, WelschShape, HuberShape, noise_scale, scale_source, weight_kind, weights, ScaleGiven, ScaleNone, ScaleUpstream, IrlsWeights, BinaryWeights, efficiency_constant
    using VisualGeometryToolkitEstimation: CameraPosesProblem, optimization_strategy_from_toml, config_dict, CameraPosesRefineMetadata
    using VisualGeometryToolkitCore: solve, EstimationProblem, RefineProblem, LevenbergMarquardt, LmStops, successful_retcode, RadialCamera, CameraModel
    using VisualGeometryToolkitCore: Irls, Triggs
    using VisualGeometryToolkitCore: EuclideanMap, exp_so3
    using Rotations: RotMatrix
    cam = pinhole_camera(); Es = truth_poses(5); views = views_of(cam, Es, 0.6)
    θ = parameter_vector(cam)
    cam0 = with_parameters(cam, θ .* SVector(1.04, 1.01, 1.0, 1.0, 1.0) .+ SVector(0.0, 0.0, 0.0, 6.0, -5.0))
    poses0 = [perturbed_pose(E, v) for (v, E) in enumerate(Es)]
    rng = Xoshiro(5)
    # σ = 0.2 px detections, 8 % of them moved 20–60 px in a random direction
    noisy = [[X => u + 0.2 * SVector(randn(rng), randn(rng)) +
                    (rand(rng) < 0.08 ? (20 + 40rand(rng)) * normalize(SVector(randn(rng), randn(rng))) : zero(SVector{2}))
              for (X, u) in cs] for cs in views]
    err(r) = maximum(abs.(parameter_vector(r.u.camera) - θ)[[1, 4, 5]])      # f, cx, cy, pixels
    ls = solve(RefineProblem(CameraPosesProblem(noisy), (; camera = cam0, poses = poses0)), LevenbergMarquardt(; optimization_strategy = nothing))
    @test err(ls) > 3.0 && scale_source(ls) === ScaleNone && scale_source(ls) === ScaleNone && isnan(ls.metadata.loss_width)
    @test all(==(1.0), weights(ls)) && weight_kind(ls) === BinaryWeights() && sizeof(weights(ls)) < 64
    @test_throws ArgumentError noise_scale(ls)
    @test length(ls.metadata.residual_norms) == 300 && ls.metadata.residual_norms[7] == sqrt(ls.metadata.residuals[13]^2 + ls.metadata.residuals[14]^2)
    # the default is the refine contract's: Triggs{Welsch}, so it needs exactly one of scale or width
    @test_throws r"scale = σ" solve(RefineProblem(CameraPosesProblem(noisy), (; camera = cam0, poses = poses0)), LevenbergMarquardt())
    rd = solve(RefineProblem(CameraPosesProblem(noisy), (; camera = cam0, poses = poses0)), LevenbergMarquardt(); scale = 0.2)
    @test rd.metadata isa CameraPosesRefineMetadata{<:Triggs{WelschShape}}
    @test rd.metadata.loss_width ≈ efficiency_constant(WelschShape, 2) * 0.2
    @test (@inferred noise_scale(rd)) === 0.2 && (@inferred scale_source(rd)) === ScaleGiven && weight_kind(rd) === IrlsWeights()
    # A robust refine runs NO least-squares stage (the owner's ruling) and estimates no scale: its width is the
    # strategy's efficiency constant times the caller's `scale` (the detections' σ). From this far start (every residual
    # 20–80 px, median 30): the truth, to the noise; the outliers' weights low (Triggs{Huber} 0.175 px; Triggs{Welsch}
    # 0.30 px: one Welsch LM from a start 25 px off)
    for lo in (Triggs{HuberShape}(efficiency_constant = 1.345), Triggs{WelschShape}(efficiency_constant = 2.98))
        rs = solve(RefineProblem(CameraPosesProblem(noisy), (; camera = cam0, poses = poses0)), LevenbergMarquardt(; optimization_strategy = lo); scale = 0.2)
        @test retcode_symbol(rs) === :converged && err(rs) < 0.5 && noise_scale(rs) == 0.2
        @test rs.metadata.loss_width ≈ (lo isa Triggs{HuberShape} ? 1.345 : 2.98) * 0.2
        @test count(<(0.5), weights(rs)) >= 20 && length(weights(rs)) == 300
        # `n` the weights' support and `rms` the weighted one (the stack's one rule): Welsch gives the 20–60 px outliers
        # no weight at all, so the support is short of the 300 correspondences
        w = weights(rs)
        @test rs.n == count(>(0), w) && (lo isa Triggs{WelschShape} ? rs.n < 300 : true)
        @test rs.rms ≈ sqrt(sum(w .* rs.metadata.residual_norms .^ 2) / sum(w))
    end
    # least squares: every correspondence, the plain rms
    @test ls.n == 300 && ls.rms ≈ sqrt(sum(abs2, ls.metadata.residuals) / 300)
    # From a start near the data (f 0.5 %, 1 px, poses 2 mrad / 5 mm off: residuals median 3.3 px, a caller's linear
    # calibration's), at the caller's σ (Triggs{Huber} 0.175 px, Triggs{Welsch} 0.202 px)
    near(E, v) = EuclideanMap(RotMatrix{3, Float64}(exp_so3(SVector(0.002, -0.0015v / 4, 0.002)) * E.R),
                              E.t + SVector(0.004, -0.003, 0.005 - 0.001v))
    cam1 = with_parameters(cam, θ .* SVector(1.005, 1.0, 1.0, 1.0, 1.0) .+ SVector(0.0, 0.0, 0.0, 1.0, -1.0))
    poses1 = [near(E, v) for (v, E) in enumerate(Es)]
    for lo in (Triggs{HuberShape}(efficiency_constant = 1.345), Triggs{WelschShape}(efficiency_constant = 2.98))
        r = solve(RefineProblem(CameraPosesProblem(noisy), (; camera = cam1, poses = poses1)), LevenbergMarquardt(; optimization_strategy = lo); scale = 0.2)
        @test retcode_symbol(r) === :converged && err(r) < 0.4 && noise_scale(r) == 0.2
    end
    # the one fixed-scale robust path is ONE robustified LM: an Irls (reweighting cycles) is refused, naming its Triggs;
    # the record is the LM's own (NLLSsolver's bits under Core's one success rule), its iteration cap included
    @test_throws r"Triggs\{HuberShape\}" solve(RefineProblem(CameraPosesProblem(noisy), (; camera = cam1, poses = poses1)), LevenbergMarquardt(; optimization_strategy = Irls{HuberShape}()); scale = 0.2)
    @test retcode_symbol(solve(RefineProblem(CameraPosesProblem(noisy), (; camera = cam1, poses = poses1)), LevenbergMarquardt(; optimization_strategy = Triggs{HuberShape}()); scale = 0.2, stops = poses_stops(; max_iter = 1))) ===
        :max_iterations
    # a width alone: the loss at it, no scale recorded; the same loss as the scale that gives it (the record aside)
    rw = solve(RefineProblem(CameraPosesProblem(noisy), (; camera = cam1, poses = poses1)), LevenbergMarquardt(; optimization_strategy = Triggs{HuberShape}()); width = 1.345 * 0.2)
    @test retcode_symbol(rw) === :converged && err(rw) < 0.4 && scale_source(rw) === ScaleNone && rw.metadata.loss_width ≈ 1.345 * 0.2
    @test_throws ArgumentError noise_scale(rw)
    ru = solve(RefineProblem(CameraPosesProblem(noisy), (; camera = cam1, poses = poses1)), LevenbergMarquardt(); scale = rd)       # an upstream result's scale, handed on
    @test noise_scale(ru) === 0.2 && scale_source(ru) === ScaleUpstream
    # the strategy from the TOML reader, and back
    t = Dict("type" => "triggs", "loss" => "welsch", "efficiency_constant" => 2.98)
    lo = optimization_strategy_from_toml(t; class = :CameraPosesProblem, section = "CameraPosesProblem.optimization_strategy")
    @test lo == Triggs{WelschShape}(efficiency_constant = 2.98) && config_dict(lo) == t
    # refusals: a strategy anchored on a RANSAC consensus set; neither or both of scale and width; a scale for least
    # squares; a non-positive scale; a scorer (a refine takes none); the retired keywords
    @test_throws r"consensus set" solve(RefineProblem(CameraPosesProblem(noisy), (; camera = cam0, poses = poses0)), LevenbergMarquardt(; optimization_strategy = VisualGeometryToolkitEstimation.TerminalLO(Irls{HuberShape}())))
    @test_throws r"not both" solve(RefineProblem(CameraPosesProblem(noisy), (; camera = cam0, poses = poses0)), LevenbergMarquardt(); scale = 0.2, width = 0.5)
    @test_throws r"reads no scale" solve(RefineProblem(CameraPosesProblem(noisy), (; camera = cam0, poses = poses0)), LevenbergMarquardt(; optimization_strategy = nothing); scale = 0.2)
    @test_throws MethodError solve(RefineProblem(CameraPosesProblem(noisy; scoring = nothing), (; camera = cam1, poses = poses1)), LevenbergMarquardt(; optimization_strategy = lo); scale = 0.2)
    @test_throws r"positive and finite" solve(RefineProblem(CameraPosesProblem(noisy), (; camera = cam0, poses = poses0)), LevenbergMarquardt(; optimization_strategy = lo); scale = -1.0)
    @test_throws r"`robust` is retired: give `optimization_strategy`" solve(RefineProblem(CameraPosesProblem(noisy; robust = nothing), (; camera = cam1, poses = poses1)), LevenbergMarquardt(; optimization_strategy = nothing))
    @test_throws r"`local_optimization` is retired" solve(RefineProblem(CameraPosesProblem(noisy; local_optimization = lo), (; camera = cam1, poses = poses1)), LevenbergMarquardt(); scale = 0.2)
end

# Several boards (owner decision 20): the board poses a block of the same bundle, each board → world a variable unless
# held; each connected component of the view–board graph fixes its gauge by a held pose, exactly (no pin, no prior).
@testsetup module BoardScenes
    import VisualGeometryToolkitEstimation as VgtkEstimation
    include(joinpath(pkgdir(VgtkEstimation), "test", "fixtures", "camera_pose_scenes.jl"))   # shared with the golden records
    export board_points, truth_boards, view_poses, board_views, nudged
end

@testitem "CameraPosesProblem: board poses, recovered from exact and noisy pixels, the gauge held exactly" setup = [CameraPoseScenes, BoardScenes] begin
    using VisualGeometryToolkitCore: dof, s2
    using VisualGeometryToolkitCore: retcode_symbol
    using StaticArrays: SVector
    using LinearAlgebra: norm
    using Random: Xoshiro
    using VisualGeometryToolkitCore: parameter_vector, with_parameters, rotation, translation, project
    using VisualGeometryToolkitEstimation: CameraPosesProblem
    using VisualGeometryToolkitCore: solve, EstimationProblem, RefineProblem, LevenbergMarquardt, LmStops, successful_retcode, RadialCamera, CameraModel
    cam = pinhole_camera(); boards = truth_boards(); views = view_poses()
    θ = parameter_vector(cam)
    cam0 = with_parameters(cam, θ .* SVector(1.03, 1.0, 1.0, 1.0, 1.0) .+ SVector(0.0, 0.0, 0.0, 5.0, -4.0))
    views0 = Dict(v => nudged(E, v) for (v, E) in views)
    keyn = Dict(:A => 1, :B => 2, :C => 3)
    boards0 = Dict(b => b === :A ? E : nudged(E, 7 + keyn[b]) for (b, E) in boards)     # board A at the truth: the held gauge
    Δ(P, E) = max(maximum(abs, rotation(P) - rotation(E)), norm(translation(P) - translation(E)))   # ≈ rad, m
    # exact pixels: everything recovered to the LM's tolerance, board A (each component's first board) held exactly
    res = solve(RefineProblem(CameraPosesProblem(board_views(cam, views, boards)), (; camera = cam0, poses = views0, board_poses = boards0)), LevenbergMarquardt(; optimization_strategy = nothing))
    est = res.u
    @test retcode_symbol(res) === :converged && res.rms < 1.0e-9
    @test maximum(abs.(parameter_vector(est.camera) - θ) ./ max.(abs.(θ), 1.0)) < 1.0e-9
    @test est.board_poses[:A] === boards0[:A]
    @test maximum(Δ(est.board_poses[b], boards[b]) for b in (:B, :C)) < 1.0e-9
    @test maximum(Δ(est.poses[v], views[v]) for v in 1:6) < 1.0e-9
    @test dof(res) == 2 * 63 * 12 - (4 + 6 * 6 + 6 * 2)
    # noisy pixels (σ = 0.2 px per coordinate): from the perturbed start, the optimum the truth start reaches (the
    # estimator's own, recovered), its rms at the noise (σ√2 per correspondence), the camera and poses to the noise
    # (measured: f, cx, cy within 4.2 px, the boards within 1.5 mrad / mm, the views within 1 cm: the boards are near
    # fronto-parallel, so the views' depth trades with f)
    noisy = board_views(cam, views, boards; σ = 0.2, rng = Xoshiro(7))
    rn = solve(RefineProblem(CameraPosesProblem(noisy), (; camera = cam0, poses = views0, board_poses = boards0)), LevenbergMarquardt(; optimization_strategy = nothing))
    rt = solve(RefineProblem(CameraPosesProblem(noisy), (; camera = cam, poses = views, board_poses = boards)), LevenbergMarquardt(; optimization_strategy = nothing))
    en, et = rn.u, rt.u
    @test retcode_symbol(rn) === :converged && retcode_symbol(rt) === :converged
    @test parameter_vector(en.camera) ≈ parameter_vector(et.camera) rtol = 1.0e-8
    @test maximum(Δ(en.poses[v], et.poses[v]) for v in 1:6) < 1.0e-6
    @test maximum(Δ(en.board_poses[b], et.board_poses[b]) for b in (:A, :B, :C)) < 1.0e-6
    @test 0.24 < rn.rms < 0.32
    @test maximum(abs.(parameter_vector(en.camera) - θ)[[1, 4, 5]]) < 8.0          # f, cx, cy: 1 % of f
    @test maximum(Δ(en.board_poses[b], boards[b]) for b in (:B, :C)) < 5.0e-3
    @test maximum(Δ(en.poses[v], views[v]) for v in 1:6) < 2.0e-2
    # the gauge held by a view instead: the same solution up to the gauge (every view ∘ board the same map)
    rv = solve(RefineProblem(CameraPosesProblem(noisy; held_boards = (), held_views = (1,)), (; camera = cam0, poses = views0, board_poses = boards0)), LevenbergMarquardt(; optimization_strategy = nothing))
    ev = rv.u
    @test ev.poses[1] == views0[1] && retcode_symbol(rv) === :converged
    @test rv.rms ≈ rn.rms rtol = 1.0e-8
    for v in 1:6, b in (:A, :B, :C)
        @test Δ(ev.poses[v] ∘ ev.board_poses[b], en.poses[v] ∘ en.board_poses[b]) < 1.0e-6
    end
    # a second component (view 7 sees board D alone) is held by its own first board
    vd = merge(views, Dict(7 => views[2])); bd = merge(boards, Dict(:D => nudged(boards[:B], 1)))
    obs = merge(board_views(cam, views, boards), Dict((7, :D) => [X => SVector{2, Float64}(project(cam, vd[7](bd[:D](X))))
                                                                   for X in board_points()]))
    r2 = solve(RefineProblem(CameraPosesProblem(obs), (; camera = cam0, poses = merge(views0, Dict(7 => nudged(views[2], 3))), board_poses = merge(boards0, Dict(:D => bd[:D])))), LevenbergMarquardt(; optimization_strategy = nothing))
    @test retcode_symbol(r2) === :converged && r2.u.board_poses[:D] === bd[:D] && r2.rms < 1.0e-9
    # refusals: a component with no held pose, a key without a pose, a pose without correspondences
    @test_throws r"hold no pose" solve(RefineProblem(CameraPosesProblem(noisy; held_boards = ()), (; camera = cam0, poses = views0, board_poses = boards0)), LevenbergMarquardt(; optimization_strategy = nothing))
    @test_throws r"no pose" solve(RefineProblem(CameraPosesProblem(noisy), (; camera = cam0, poses = views0, board_poses = Dict(:A => boards[:A], :B => boards[:B]))), LevenbergMarquardt(; optimization_strategy = nothing))
    @test_throws r"no correspondence" solve(RefineProblem(CameraPosesProblem(noisy), (; camera = cam0, poses = merge(views0, Dict(9 => views[1])), board_poses = boards0)), LevenbergMarquardt(; optimization_strategy = nothing))
end

# The image-space refine of a SplineLens camera's controls is retired (owner, 2026-11-13; 0.10.0): a call that would
# free the controls is refused by name, before the entry's optical-axis fit, and the camera is refined only by
# back-projection (`CameraRegression(SplineLens, …)`, `PSpline`) or held (`fixed = (:controls,)`).
@testitem "CameraPosesProblem: a SplineLens camera's free controls are refused by name — refined only by back-projection, or held" setup = [CameraPoseScenes, SplineCameraPoseScenes] begin
    using VisualGeometryToolkitCore: dof, s2
    using VisualGeometryToolkitEstimation: CameraPosesProblem
    using VisualGeometryToolkitCore: solve, EstimationProblem, RefineProblem, LevenbergMarquardt, LmStops, successful_retcode, RadialCamera, CameraModel
    truth = spline_truth(); Es = truth_poses(3)
    views = spline_views(truth, Es, 1.1)
    cam0 = perturbed_controls(truth, 0.01); poses0 = [perturbed_pose(E, v) for (v, E) in enumerate(Es)]
    refusal = r"image-space refine of a SplineLens camera's controls is retired .* refined only by back-projection, or held"
    @test_throws refusal solve(RefineProblem(CameraPosesProblem(views), (; camera = bounded(cam0), poses = poses0)), LevenbergMarquardt(; optimization_strategy = nothing))                    # the default `fixed = ()`
    @test_throws refusal solve(RefineProblem(CameraPosesProblem(views; fixed = ()), (; camera = bounded(cam0), poses = poses0)), LevenbergMarquardt(; optimization_strategy = nothing))
    @test_throws r"`fixed = \(\)`" solve(RefineProblem(CameraPosesProblem(views), (; camera = bounded(cam0), poses = poses0)), LevenbergMarquardt(; optimization_strategy = nothing))
    @test_throws r"fixed = \(:controls,\)" solve(RefineProblem(CameraPosesProblem(views), (; camera = bounded(cam0), poses = poses0)), LevenbergMarquardt(; optimization_strategy = nothing))  # the remedy named
    # that route's smoothness weight, in any form, with the camera held or not
    for penalty in (:gcv, 1.0e-8, 0)
        @test_throws refusal solve(RefineProblem(CameraPosesProblem(views; fixed = (:controls,), penalty), (; camera = bounded(cam0), poses = poses0)), LevenbergMarquardt(; optimization_strategy = nothing))
        @test_throws r"a `penalty`" solve(RefineProblem(CameraPosesProblem(views; penalty), (; camera = bounded(cam0), poses = poses0)), LevenbergMarquardt(; optimization_strategy = nothing))
    end
    # refused first: a bare model (which the axis fit would refuse, naming BoundedCamera) gets the refusal
    @test_throws refusal solve(RefineProblem(CameraPosesProblem(views), (; camera = cam0, poses = poses0)), LevenbergMarquardt(; optimization_strategy = nothing))
    @test_throws refusal solve(RefineProblem(CameraPosesProblem(Dict(v => views[v] for v in 1:3); fixed = ()), (; camera = cam0, poses = Dict(v => poses0[v] for v in 1:3))), LevenbergMarquardt(; optimization_strategy = nothing))   # the keyed form too
end

# The held route on the SplineLens cameras CameraRegression and the constructors make (non-unit controls): the poses of
# exact data come back the truth's, the camera exactly as given (no turn of its frame: its controls are no variable,
# so there is no gauge to fix).
@testitem "CameraPosesProblem: a held SplineLens camera of CameraRegression and the constructors, non-unit controls" setup = [CameraPoseScenes, SplineCameraPoseScenes] begin
    using VisualGeometryToolkitCore: dof, s2
    using VisualGeometryToolkitCore: retcode_symbol
    using LinearAlgebra: norm
    using VisualGeometryToolkitCore: VisualGeometryToolkitCore as VC, CameraModel, SplineLens, ChartSpline, KannalaBrandt, BoundedCamera, Size2, backproject, imaged_region, lens
    using VisualGeometryToolkitEstimation: CameraPosesProblem, CameraRegressionProblem, CameraRegression
    using VisualGeometryToolkitCore: solve, EstimationProblem, RefineProblem, LevenbergMarquardt, LmStops, successful_retcode, RadialCamera, CameraModel
    kb = CameraModel(; focal = 420.0, pp = (640.0, 480.0), lens = KannalaBrandt(-0.02, 0.004, -0.001, 0.0002))
    bk = BoundedCamera(kb, Size2(1280, 960))
    fitted = solve(EstimationProblem(CameraRegressionProblem(SplineLens, bk, imaged_region(bk))), CameraRegression(; tolerance = 1.0e-2, forward = false)).u   # the P-spline's controls, 1.8e-2 off unit
    quasi = CameraModel(SplineLens, u -> backproject(kb, u), Size2(1280, 960); init = :quasi_interpolate)          # 7.7e-3 off unit
    t = spline_truth(); bt = lens(t).backward
    scaled = CameraModel(t.intrinsics, SplineLens(ChartSpline(bt.area, map(c -> 2.0 * c, bt.controls))))        # every norm 2
    Es = truth_poses(3); poses0 = [perturbed_pose(E, v) for (v, E) in enumerate(Es)]
    for truth in (fitted, quasi, scaled)
        @test maximum(c -> abs(norm(c) - 1), lens(truth).backward.controls) > 1.0e-3
        views = spline_views(truth, Es, 1.1)
        bc = bounded(truth)
        res = solve(RefineProblem(CameraPosesProblem(views; fixed = (:controls,)), (; camera = bc, poses = poses0)), LevenbergMarquardt(; optimization_strategy = nothing))
        est = res.u
        @test retcode_symbol(res) === :converged && res.metadata.free == ()
        @test est.camera === bc                                                                # the camera as given
        @test res.rms < 1.0e-8 < spline_rms(truth, poses0, views)
        @test maximum(maximum(abs, P.R - E.R) + maximum(abs, P.t - E.t) for (P, E) in zip(est.poses, Es)) < 1.0e-8
    end
end

# A HELD SplineLens camera is a constant: its rows project through the whole camera (`project`, every node), so no
# window is declared or rebuilt and no node is a column. Until 2026-11-12 the held route's blocks each carried the
# 4 × 4 window of its point's cell at the guess and extrapolated its cubic pieces when the pose moved the point into
# another cell (never rebuilt: the held problem has no rounds), so the poses came back optimal for those windows, not
# for the camera. The poses move ≈ 20 px here (`perturbed_pose`), across many cells.
@testitem "CameraPosesProblem: a held SplineLens camera is projected whole — no window, no node column, its own rows" setup = [CameraPoseScenes, SplineCameraPoseScenes] begin
    using VisualGeometryToolkitCore: dof, s2
    using StaticArrays: SVector
    using LinearAlgebra: norm
    using VisualGeometryToolkitCore: VisualGeometryToolkitCore as VC, retcode_symbol, reprojection_error, model, lens, spline_window, project
    using VisualGeometryToolkitEstimation: VisualGeometryToolkitEstimation as VE, CameraPosesProblem
    using VisualGeometryToolkitCore: solve, EstimationProblem, RefineProblem, LevenbergMarquardt, LmStops, successful_retcode, RadialCamera, CameraModel
    truth = spline_truth(); Es = truth_poses(3)
    views = spline_views(truth, Es, 1.1)
    poses0 = [perturbed_pose(E, v) for (v, E) in enumerate(Es)]
    # the held problem's filter: every correspondence kept (a full lattice), each on its view and the one held board
    p = VE._spline_problem(VE._without_forward(truth), poses0, views)
    @test length(p.obs) == 360 && all(o -> o[2] == 1, p.obs) && p.dropped == 0 && p.unprojected == 0
    # many points change cells between the guess and the truth
    b = lens(truth).backward
    cell(P, X) = spline_window(b.area, size(b.controls), project(lens(truth), P.R * X + P.t))[1:2]
    crossed = sum(count(c -> cell(poses0[v], first(c)) != cell(Es[v], first(c)), views[v]) for v in eachindex(views))
    @test crossed > 20
    held = solve(RefineProblem(CameraPosesProblem(views; fixed = (:controls,)), (; camera = bounded(truth), poses = poses0)), LevenbergMarquardt(; optimization_strategy = nothing))
    est = held.u
    @test retcode_symbol(held) === :converged && held.metadata.free == ()
    @test dof(held) == 2 * 360 - 6 * 3                                                         # the poses' columns alone: no node
    # the rows are the camera's own reprojection errors at the estimate, and the exact data's optimum is the truth
    own = reduce(vcat, [reduce(vcat, [Vector(reprojection_error(model(est.camera), est.poses[v], first(c), last(c))) for c in views[v]]) for v in eachindex(views)])
    @test maximum(abs, held.metadata.residuals - own) <= 1.0e-9
    @test maximum(abs, held.metadata.residuals) <= 1.0e-8
    @test maximum(maximum(abs, P.R - E.R) + maximum(abs, P.t - E.t) for (P, E) in zip(est.poses, Es)) < 1.0e-9
end

# The held SplineLens camera comes back AS GIVEN (VisualGeometryToolkitEstimation 0.11.0): its controls are no variable,
# so there is no gauge to fix and no turn onto the optical axis; a forward spline it holds is kept, never refitted.
@testitem "CameraPosesProblem: a SplineLens camera's poses alone, the camera returned as given, a masked lattice, refusals" setup = [CameraPoseScenes, SplineCameraPoseScenes] begin
    using VisualGeometryToolkitCore: dof, s2
    using VisualGeometryToolkitCore: retcode_symbol
    using StaticArrays: SVector
    using VisualGeometryToolkitCore: VisualGeometryToolkitCore as VC, CameraModel, SplineLens, ChartSpline, BoundedCamera, Size2, lens, model, with_present, spline_window, imaged_region
    using VisualGeometryToolkitEstimation: VisualGeometryToolkitEstimation, CameraPosesProblem, CameraRegressionProblem, CameraRegression
    using VisualGeometryToolkitCore: solve, EstimationProblem, RefineProblem, LevenbergMarquardt, LmStops, successful_retcode, RadialCamera, CameraModel
    truth = spline_truth(); Es = truth_poses(3)
    views = spline_views(truth, Es, 1.1)
    poses0 = [perturbed_pose(E, v) for (v, E) in enumerate(Es)]
    # the poses alone: the camera comes back as given, the poses recovered in its own frame
    bt = bounded(truth)
    held = solve(RefineProblem(CameraPosesProblem(views; fixed = (:controls,)), (; camera = bt, poses = poses0)), LevenbergMarquardt(; optimization_strategy = nothing))
    @test retcode_symbol(held) === :converged && held.metadata.free == ()
    @test held.u.camera === bt
    @test maximum(maximum(abs, P.t - E.t) for (P, E) in zip(held.u.poses, Es)) < 1.0e-6
    @test keys(held.metadata.spline) == (:n_dropped_window, :n_dropped_unprojected)
    # a bare model needs no bound (no axis is fitted): returned as given too
    bare = solve(RefineProblem(CameraPosesProblem(views; fixed = (:controls,)), (; camera = truth, poses = poses0)), LevenbergMarquardt(; optimization_strategy = nothing))
    @test bare.u.camera === truth && bare.u.poses == held.u.poses
    # a camera holding a forward spline: the camera, its forward included, returned as given
    cam0 = perturbed_controls(truth, 0.01)
    bc = BoundedCamera(solve(EstimationProblem(CameraRegressionProblem(SplineLens, BoundedCamera(cam0, Size2(1280, 960)), imaged_region(BoundedCamera(cam0, Size2(1280, 960))))), CameraRegression(; tolerance = 1.0e-3)).u, Size2(1280, 960))
    @test lens(model(bc)).forward isa ChartSpline{2}
    rf = solve(RefineProblem(CameraPosesProblem(views; fixed = (:controls,)), (; camera = bc, poses = poses0)), LevenbergMarquardt(; optimization_strategy = nothing))
    @test retcode_symbol(rf) === :converged && rf.u.camera === bc
    # the forward refit's keywords are retired, naming why
    @test_throws r"`refit_forward` is retired: give `ForwardSpline" solve(RefineProblem(CameraPosesProblem(views; fixed = (:controls,), refit_forward = false), (; camera = bc, poses = poses0)), LevenbergMarquardt(; optimization_strategy = nothing))
    @test_throws r"`forward_tolerance` is retired: give `ForwardSpline" solve(RefineProblem(CameraPosesProblem(views; fixed = (:controls,), forward_tolerance = 1.0e-3), (; camera = bc, poses = poses0)), LevenbergMarquardt(; optimization_strategy = nothing))
    # the default strategy is the bundle's (Triggs{Welsch}): it needs a scale or a width
    @test_throws r"scale = σ" solve(RefineProblem(CameraPosesProblem(views; fixed = (:controls,)), (; camera = bt, poses = poses0)), LevenbergMarquardt())
    # a masked lattice: the detections on an absent window are dropped and counted
    present = trues(10, 12); present[1:3, 9:12] .= false
    masked = with_present(cam0, present)
    b = lens(masked).backward
    absent(u) = (w = spline_window(b.area, size(b.controls), VC.normalized_point(masked.intrinsics, u)); !all(present[(w[2] - 2):(w[2] + 1), (w[1] - 2):(w[1] + 1)]))
    n_absent = count(absent(last(x)) for cs in views for x in cs)
    @test n_absent > 0
    rm = solve(RefineProblem(CameraPosesProblem(views; fixed = (:controls,)), (; camera = bounded(masked), poses = poses0)), LevenbergMarquardt(; optimization_strategy = nothing))
    @test rm.metadata.spline.n_dropped_window == n_absent
    @test rm.n == 360 - n_absent - rm.metadata.spline.n_dropped_unprojected
    @test rm.metadata.spline.n_dropped_unprojected <= 10        # those the camera images onto an absent window, within a cell of one (the guess's taken back)
    @test lens(model(rm.u.camera)).present == present
    # refusals: a box, a `fixed` other than (:controls,), a retired spelling (Core's one refusal; the free controls and a
    # `penalty`: the refusal by name, its own item above)
    @test_throws r"`box` has nothing to bound" solve(RefineProblem(CameraPosesProblem(views; fixed = (:controls,), box = (; c_1_1_x = (0.0, 1.0))), (; camera = bt, poses = poses0)), LevenbergMarquardt(; optimization_strategy = nothing))
    @test_throws r"is \(:controls,\)" solve(RefineProblem(CameraPosesProblem(views; fixed = (:c_1_1_x,)), (; camera = bt, poses = poses0)), LevenbergMarquardt(; optimization_strategy = nothing))
    @test_throws ArgumentError("the keyword `free` is retired: give `fixed`") solve(RefineProblem(CameraPosesProblem(views; free = ()), (; camera = bt, poses = poses0)), LevenbergMarquardt(; optimization_strategy = nothing))
    # a lattice that folds where a point projects (every control ẑ: the backward's Jacobian exactly 0 there) is not
    # linearized: the error names the fold
    flat = CameraModel(truth.intrinsics, SplineLens(ChartSpline(b.area, fill(SVector(0.0, 0.0, 1.0), size(b.controls)))))
    P = poses0[1]
    one_point = [[P.R' * (SVector(0.0, 0.0, 2.0) - P.t) => SVector(640.0, 480.0)]]
    @test_throws r"lattice folds where the point .* projects at the guess" solve(RefineProblem(CameraPosesProblem(one_point; fixed = (:controls,)), (; camera = bounded(flat), poses = [P])), LevenbergMarquardt(; optimization_strategy = nothing))
end

# A robust refine of the poses through a held SplineLens camera: the bundle's one solve, its strategy included
# (VisualGeometryToolkitEstimation 0.11.0; the route refused every strategy but least squares).
@testitem "CameraPosesProblem: a held SplineLens camera under a robust strategy — the outliers down-weighted" setup = [CameraPoseScenes, SplineCameraPoseScenes] begin
    using VisualGeometryToolkitCore: dof, s2
    using VisualGeometryToolkitCore: weights, noise_scale
    using VisualGeometryToolkitCore: retcode_symbol
    using StaticArrays: SVector
    using LinearAlgebra: normalize
    using Random: Xoshiro
    using VisualGeometryToolkitCore: Triggs, HuberShape, scale_source, ScaleGiven
    using VisualGeometryToolkitEstimation: CameraPosesProblem, CameraPosesRefineMetadata
    using VisualGeometryToolkitCore: solve, EstimationProblem, RefineProblem, LevenbergMarquardt, LmStops, successful_retcode, RadialCamera, CameraModel
    truth = spline_truth(); Es = truth_poses(3)
    views = spline_views(truth, Es, 1.1)
    poses0 = [perturbed_pose(E, v) for (v, E) in enumerate(Es)]
    rng = Xoshiro(5)
    # σ = 0.2 px detections; of the inner 80 per view (θ ≤ 0.9 rad, every moved pixel still inside the lattice) about
    # 8 % moved 20–60 px in a random direction
    noisy = [[(k <= 80 && rand(rng) < 0.08) ? (X => u + 0.2 * SVector(randn(rng), randn(rng)) + (20 + 40rand(rng)) * normalize(SVector(randn(rng), randn(rng)))) :
                                              (X => u + 0.2 * SVector(randn(rng), randn(rng)))
              for (k, (X, u)) in enumerate(cs)] for cs in views]
    perr(r) = maximum(maximum(abs, P.t - E.t) for (P, E) in zip(r.u.poses, Es))
    ls = solve(RefineProblem(CameraPosesProblem(noisy; fixed = (:controls,)), (; camera = bounded(truth), poses = poses0)), LevenbergMarquardt(; optimization_strategy = nothing))
    rs = solve(RefineProblem(CameraPosesProblem(noisy; fixed = (:controls,)), (; camera = bounded(truth), poses = poses0)), LevenbergMarquardt(; optimization_strategy = Triggs{HuberShape}(efficiency_constant = 1.345)); scale = 0.2)
    @test retcode_symbol(rs) === :converged && rs.metadata isa CameraPosesRefineMetadata{<:Triggs{HuberShape}}
    @test noise_scale(rs) == 0.2 && scale_source(rs) === ScaleGiven && rs.metadata.loss_width ≈ 1.345 * 0.2
    @test count(<(0.5), weights(rs)) >= 10 && length(weights(rs)) == 360
    @test perr(rs) < perr(ls) / 3
end

@testitem "gcv_curve, gcv_select: the stack's one GCV of a penalty weight" begin
    using VisualGeometryToolkitCore: dof
    using LinearAlgebra: I, Symmetric, eigvals, tr
    using Random: Xoshiro, randn
    using SparseArrays: sparse
    using VisualGeometryToolkitCore: difference_rows
    using VisualGeometryToolkitEstimation: gcv_curve, gcv_select, gcv_score, check_penalty_weight
    # the closed-form curve equals the direct solve at every λ: rss of the penalized fit and tr H
    rng = Xoshiro(3)
    J = randn(rng, 40, 8); r = randn(rng, 40); Dm = Matrix(difference_rows((1, 8), 3)); c = zeros(size(Dm, 1))
    curve = gcv_curve(J' * J, J' * r, sum(abs2, r), Dm' * Dm, Dm' * c, 40)
    for λ in (1.0e-3, 0.1, 10.0)
        A = J' * J + λ * Dm' * Dm
        y = A \ (J' * r)
        H = J * (A \ J')
        e = curve(λ)
        @test e.rss ≈ sum(abs2, r - J * y) rtol = 1.0e-8
        @test e.dof ≈ tr(H) rtol = 1.0e-8
        @test e.gcv ≈ 40 * e.rss / (40 - e.dof)^2 rtol = 1.0e-12
    end
    # with a target c, normal-equation and rows forms; the rows form past `exact_max` (a sparse Cholesky per λ, its
    # symbolic analysis once, a Hutchinson trace) gives the same rss and tr H within 3σ
    c2 = randn(rng, size(Dm, 1))
    ch = gcv_curve(J' * J, J' * r, sum(abs2, r), Dm' * Dm, Dm' * c2, 40)
    cr = gcv_curve(J, r, Dm, c2); cs = gcv_curve(J, r, Dm, c2; exact_max = 0, probes = 4000)
    for λ in (1.0e-3, 0.1, 10.0)
        A = J' * J + λ * Dm' * Dm
        y = A \ (J' * r + λ * Dm' * c2)
        e = ch(λ)
        @test e.rss ≈ sum(abs2, r - J * y) rtol = 1.0e-8
        @test e.dof ≈ tr(J * (A \ J')) rtol = 1.0e-8
        @test cr(λ).rss ≈ e.rss rtol = 1.0e-10
        @test cr(λ).dof ≈ e.dof rtol = 1.0e-10
        @test cs(λ).rss ≈ e.rss rtol = 1.0e-8
        @test e.y ≈ y rtol = 1.0e-8                                                     # the solution at λ, every form
        @test cr(λ).y ≈ y rtol = 1.0e-8
        @test cs(λ).y ≈ y rtol = 1.0e-8
        @test abs(cs(λ).dof - e.dof) <= 3 * sqrt(2 * e.dof / 4000)
    end
    # a parameter no row reads (a zero column of J and of D) is dropped by every form, the pencil and the per-λ
    # Cholesky alike: 0 in y, the rest as without it
    Ju = hcat(J, zeros(40)); Du = hcat(Dm, zeros(size(Dm, 1)))
    for exact_max in (3000, 0)
        cu = gcv_curve(Ju, r, Du, c2; exact_max, probes = 4000)
        @test cu(0.1).y[end] == 0.0
        @test cu(0.1).y[1:8] ≈ cr(0.1).y rtol = 1.0e-8
        @test cu(0.1).rss ≈ cr(0.1).rss rtol = 1.0e-8
    end
    # a direction neither term reads but the parameters share (two equal columns, the penalty blind to their difference):
    # a gauge, refused — never ridged
    Jg = hcat(J[:, 1:7], J[:, 7]); Dg = hcat(Dm[:, 1:6], zeros(size(Dm, 1), 2))
    @test_throws r"not positive definite" gcv_curve(Jg, r, Dg, c)
    # the search, a coarse scan and Brent's method: a minimum inside the bracket to Brent's floor (no tolerance to
    # choose), in a handful of evaluations past the scan; one at an end reported there
    calls = Ref(0)
    s = gcv_select(λ -> (calls[] += 1; (; gcv = (log10(λ) + 2.3)^2, rss = 0.0, dof = 0.0)), 1.0; log10_bracket = (-8.0, 4.0), points = 25)
    @test abs(log10(s.β) + 2.3) < 1.0e-7 && !s.at_bracket_end && s.λ == s.β && calls[] <= 25 + 12
    e = gcv_select(λ -> (; gcv = log10(λ), rss = 0.0, dof = 0.0), 1.0; log10_bracket = (-8.0, 4.0), points = 25)
    @test e.at_bracket_end && e.β == 1.0e-8
    # a minimum inside the scan's first interval, sampled lowest at the scan's end: refined there and interior
    s1 = gcv_select(λ -> (; gcv = (log10(λ) - 3.8)^2, rss = 0.0, dof = 0.0), 1.0; log10_bracket = (-8.0, 4.0), points = 25)
    @test abs(log10(s1.β) - 3.8) < 1.0e-7 && !s1.at_bracket_end
    # a weight the caller cannot evaluate (Inf) is never the minimum, even beside the basin
    s2 = gcv_select(λ -> (; gcv = log10(λ) < -3 ? Inf : (log10(λ) + 2.3)^2, rss = 0.0, dof = 0.0), 1.0; log10_bracket = (-8.0, 4.0), points = 25)
    @test abs(log10(s2.β) + 2.3) < 1.0e-7 && isfinite(s2.gcv)
    @test gcv_score(40, 2.0, 8.0) == 40 * 2.0 / 32^2
    @test check_penalty_weight(:gcv, "w") === :gcv && check_penalty_weight(0.5, "w") == 0.5
    @test_throws ArgumentError check_penalty_weight(nothing, "w")                              # RC's options: no `nothing`
    @test_throws r"`allow_nothing` is retired" check_penalty_weight(nothing, "w"; allow_nothing = true)   # an inert knob, no caller
    @test_throws ArgumentError check_penalty_weight(-1.0, "w")
    @test_throws ArgumentError check_penalty_weight(:aic, "w")
end
# The rotation-centre views (fitting/rotation_center_views.jl): a held camera turned about one fixed point, x_k =
# R_k (X − c) + d — one rotation per view, the centre c and the offset d shared; the residual is each dot's bearing
# (back-projected once, whitened into pixels), for every camera. From exact pixels and a perturbed
# start the truth is recovered; from noisy, contaminated pixels a robust refine at an upstream scale recovers it to the
# noise and records ScaleUpstream; a one-axis set, two views, a free camera and mismatched keys are refused.
@testsetup module RotationCenterScenes
    import VisualGeometryToolkitEstimation as VgtkEstimation
    include(joinpath(pkgdir(VgtkEstimation), "test", "fixtures", "camera_pose_scenes.jl"))   # shared with the golden records
    export center_truth, center_views
end

@testitem "CameraPosesProblem: rotation-centre views — the centre, the offset and the rotations recovered" setup = [CameraPoseScenes, RotationCenterScenes] begin
    using VisualGeometryToolkitCore: dof, s2
    using VisualGeometryToolkitCore: weights, noise_scale
    using StaticArrays: SVector, SMatrix
    using StableRNGs: StableRNG
    using LinearAlgebra: norm
    using VisualGeometryToolkitCore: parameter_names, exp_so3, FitResult, noise_scale, scale_source, ScaleUpstream, ScaleEstimated, ScaleNone, ScaleGiven, HuberShape, Irls, Triggs, ReturnCode, ClosedForm, SolveStats
    using VisualGeometryToolkitEstimation: CameraPosesProblem, RotationCenterViews, rotation_center_pose, held_parameters
    using VisualGeometryToolkitCore: solve, EstimationProblem, RefineProblem, LevenbergMarquardt, LmStops, successful_retcode, RadialCamera, CameraModel
    using VisualGeometryToolkitCore: EuclideanMap, rotation, translation, BoundedCamera, Size2
    cam = pinhole_camera(); held = held_parameters(cam)
    @test held == Tuple(parameter_names(cam)) && held_parameters(BoundedCamera(cam, Size2(1280, 960))) == held   # every parameter, by name
    axes = [SVector(0.0, 0.0, 0.0), SVector(0.08, 0.0, 0.0), SVector(0.0, 0.08, 0.0), SVector(-0.06, 0.05, 0.03), SVector(0.05, -0.07, -0.04),
            SVector(0.0, 0.0, 0.1)]
    t = center_truth(axes)
    start = RotationCenterViews(Dict(k => SMatrix{3, 3, Float64, 9}(exp_so3(SVector(0.002, -0.001, 0.0015))) * R for (k, R) in t.R),
                                t.c + SVector(0.01, -0.008, 0.012), t.d + SVector(-0.005, 0.004, 0.006))
    rot_err(est) = maximum(norm(est.views.rotations[k] - R) for (k, R) in t.R)
    @testset "exact pixels: the truth, to the LM's tolerance" begin
        r = solve(RefineProblem(CameraPosesProblem(center_views(cam, t); fixed = held), (; camera = cam, views = start)), LevenbergMarquardt(; optimization_strategy = nothing))
        @test successful_retcode(r)
        est = r.u
        @test est.camera === cam
        @test norm(est.views.center - t.c) < 1.0e-8 && norm(est.views.offset - t.d) < 1.0e-8 && rot_err(est) < 1.0e-9
        @test dof(r) == 2 * 6 * 63 - (6 + 3 * 6) && r.rms < 1.0e-8 && scale_source(r) === ScaleNone
        @test sort(collect(keys(est.views.rotations))) == collect(1:6)
        # a view's pose from the refined views (the public rotation_center_pose): (R_k, d − R_k c), the truth's camera points
        v = est.views
        P = rotation_center_pose(v.rotations[3], v.center, v.offset)
        @test P isa EuclideanMap && rotation(P) == v.rotations[3] && translation(P) == v.offset - SMatrix{3, 3, Float64, 9}(v.rotations[3]) * v.center
        X = SVector(0.04, -0.08, 0.0)
        @test norm(P(X) - (t.R[3] * (X - t.c) + t.d)) < 1.0e-8
    end
    @testset "noisy pixels with 5% outliers: Triggs{Huber} at an upstream scale" begin
        upstream = FitResult(0.3, ReturnCode.Success, ClosedForm, SolveStats(), NaN, 6,
                             (; scale = 0.3, scale_source = ScaleEstimated))
        views = center_views(cam, t; σ = 0.3, outliers = 0.05)
        r = solve(RefineProblem(CameraPosesProblem(views; fixed = held), (; camera = cam, views = start)), LevenbergMarquardt(; optimization_strategy = Triggs{HuberShape}(efficiency_constant = 3.0)); scale = upstream)
        @test successful_retcode(r) && scale_source(r) === ScaleUpstream && noise_scale(r) == 0.3 && r.metadata.loss_width ≈ 0.9
        est = r.u
        @test norm(est.views.center - t.c) < 1.4e-2 && norm(est.views.offset - t.d) < 1.4e-2
        @test count(<(1), weights(r)) >= 10                                           # the outliers down-weighted
        ls = solve(RefineProblem(CameraPosesProblem(views; fixed = held), (; camera = cam, views = start)), LevenbergMarquardt(; optimization_strategy = nothing))   # least squares: pulled by them
        @test norm(ls.u.views.center - t.c) > norm(est.views.center - t.c)
        # Triggs{Huber} at 1.345σ given (ScaleGiven), and a strategy at an absolute width (ScaleNone, the width recorded): the
        # same views. (`ri`'s references were an Irls{Huber}'s at the same width until 0.9: one robustified LM lands within
        # 2e-8 m of it, the rms within 3e-8 relative.)
        ri = solve(RefineProblem(CameraPosesProblem(views; fixed = held), (; camera = cam, views = start)), LevenbergMarquardt(; optimization_strategy = Triggs{HuberShape}(efficiency_constant = 1.345)); scale = 0.3)
        @test successful_retcode(ri) && scale_source(ri) === ScaleGiven && noise_scale(ri) == 0.3 && ri.metadata.loss_width ≈ 1.345 * 0.3
        # The bound is this draw's (StableRNG(2): Huber at 1.345σ 9.0 mm, at 3σ 8.7 mm, least squares 118 mm), with a 1.6× margin. The centre
        # is weakly fixed here: it slides with the offset (their errors equal) under rotations of at most 0.1 rad, so over 20
        # draws of this set (σ = 0.3 px, 6 × 63 points, 5% outliers at 39 px; StableRNG(101:120)) both robust fits spread 0.5–12 mm
        # while least squares spreads 16–128 mm (at least 2.8× a robust error). What holds over draws is the second clause: closer
        # than least squares.
        @test norm(ri.u.views.center - t.c) < min(1.4e-2, norm(ls.u.views.center - t.c))
        @test norm(ri.u.views.offset - t.d) < 1.4e-2
        rw = solve(RefineProblem(CameraPosesProblem(views; fixed = held), (; camera = cam, views = start)), LevenbergMarquardt(; optimization_strategy = Triggs{HuberShape}()); width = 0.9)
        @test successful_retcode(rw) && scale_source(rw) === ScaleNone && rw.metadata.loss_width ≈ 0.9
        @test norm(rw.u.views.center - t.c) < 1.4e-2 && norm(rw.u.views.offset - t.d) < 1.4e-2
        # Against the pixel residual this refine had until 2026-11-12 (125d80dc, measured on this draw): the bearing residual
        # (the tangent angle whitened into pixels by the back-projection Jacobian) agrees with it to first order, so the robust
        # estimates move by at most 0.09 mm (Huber at 3σ 0.09 mm, at 1.345σ 0.06 mm; their errors from the truth ~ 9 mm), held here to
        # 0.3 mm, and the rms by under 0.02% (held to 0.2%); least squares, pulled by the 39 px outliers where the two whitenings
        # differ at second order, moves 2.7 mm of its 118 mm error, held to 6 mm.
        pixel = (; triggs = (SVector(0.004542457182526425, 0.05992413930977435, 0.5412536292628963), SVector(0.004441893165669695, -0.060187182358688224, 0.0587542470305566), 1.6222332631675085),
                   narrow = (SVector(0.004159982519303803, 0.0603976193754511, 0.540992910097172), SVector(0.004160219288152445, -0.06039753472075275, 0.058968691998531116), 1.1572328472850872),
                   ls = (SVector(0.046913821623391486, 0.03285096285063938, 0.44088480955963727), SVector(0.041888728477604775, -0.04349052504441306, 0.16042921368413593), 9.541040680183073))
        for (res, ref, tol) in ((r, pixel.triggs, 3.0e-4), (rw, pixel.triggs, 3.0e-4), (ri, pixel.narrow, 3.0e-4), (ls, pixel.ls, 6.0e-3))
            @test norm(res.u.views.center - ref[1]) < tol && norm(res.u.views.offset - ref[2]) < tol
        end
        @test abs(r.rms - pixel.triggs[3]) < 2.0e-3 * pixel.triggs[3] && abs(ri.rms - pixel.narrow[3]) < 2.0e-3 * pixel.narrow[3]
    end
    @testset "least squares on outlier-free, low-noise pixels: the bearing residual's estimate is the pixel residual's" begin
        # σ = 0.1 px, no outliers (StableRNG(7)): the two whitenings differ only at second order in the residual, so the least-
        # squares estimates agree far below a millimetre — measured 0.03 µm in c and 0.06 µm in d against the pixel residual's
        # (125d80dc), held here to 10 µm; the rms to 1e-4 relative (measured 1.8e-6). The 2.7 mm of the contaminated set above is
        # the 39 px outliers' full influence under least squares, not a disagreement of the residuals.
        clean = center_views(cam, t; σ = 0.1, rng = StableRNG(7))
        r = solve(RefineProblem(CameraPosesProblem(clean; fixed = held), (; camera = cam, views = start)), LevenbergMarquardt(; optimization_strategy = nothing))
        c_px = SVector(0.003972430523787387, 0.060992540771253746, 0.5521073428426978)
        d_px = SVector(0.003969954547206083, -0.061055755407834915, 0.047875584955742075)
        @test successful_retcode(r) && norm(r.u.views.center - c_px) < 1.0e-5 && norm(r.u.views.offset - d_px) < 1.0e-5
        @test abs(r.rms - 0.13664940184183744) < 1.0e-4 * 0.13664940184183744
        # the record is in pixels (the bearing rows whitened into px): at σ = 0.1 px a dot's 2-vector residual has rms
        # σ√2 ≈ 0.14 px, so the rms, √s² and the per-dot norms read in pixels (a tangent angle would read ~1e-4)
        md = r.metadata
        @test 0.5 * 0.1 * sqrt(2) < r.rms < 2 * 0.1 * sqrt(2) && 0.5 * 0.1 * sqrt(2) < sqrt(s2(r)) < 2 * 0.1 * sqrt(2)
        @test length(md.residual_norms) == r.n && 0.5 * 0.1 < sum(md.residual_norms) / r.n < 4 * 0.1
    end
    @testset "the gauge: a one-axis set and two views are refused; a free camera; the keys" begin
        one_axis = center_truth([SVector(0.0, 0.0, a) for a in -0.1:0.05:0.1])
        s1 = RotationCenterViews(one_axis.R, one_axis.c, one_axis.d)
        @test_throws r"turned about one axis" solve(RefineProblem(CameraPosesProblem(center_views(cam, one_axis); fixed = held), (; camera = cam, views = s1)), LevenbergMarquardt(; optimization_strategy = nothing))
        two = center_truth(axes[1:2])
        @test_throws r"at least 3 views" solve(RefineProblem(CameraPosesProblem(center_views(cam, two); fixed = held), (; camera = cam, views = RotationCenterViews(two.R, two.c, two.d))), LevenbergMarquardt(; optimization_strategy = nothing))
        @test_throws r"HELD camera" solve(RefineProblem(CameraPosesProblem(center_views(cam, t)), (; camera = cam, views = start)), LevenbergMarquardt(; optimization_strategy = nothing))
        @test_throws ArgumentError solve(RefineProblem(CameraPosesProblem(Dict(k => v for (k, v) in center_views(cam, t) if k != 3); fixed = held), (; camera = cam, views = start)), LevenbergMarquardt(; optimization_strategy = nothing))
    end
end

@testitem "CameraPosesProblem: rotation-centre views of a held SplineLens camera — the bearing residual, no projection in the solve" setup = [CameraPoseScenes, SplineCameraPoseScenes, RotationCenterScenes] begin
    using VisualGeometryToolkitCore: dof, s2
    using VisualGeometryToolkitCore: weights, noise_scale
    using StaticArrays: SVector, SMatrix
    using LinearAlgebra: norm, normalize
    using StableRNGs: StableRNG
    import NLLSsolver
    using VisualGeometryToolkitCore: exp_so3, scale_source, noise_scale, ScaleGiven, ScaleNone, HuberShape, Irls, Triggs, lens, with_present, spline_window, backproject, project, ReturnCode, NoRule, retcode_symbol, successful_retcode, CameraModel
    import VisualGeometryToolkitCore
    using VisualGeometryToolkitEstimation: CameraPosesProblem, RotationCenterViews
    using VisualGeometryToolkitCore: solve, EstimationProblem, RefineProblem, LevenbergMarquardt, LmStops, successful_retcode, RadialCamera, CameraModel
    using VisualGeometryToolkitEstimation: VisualGeometryToolkitEstimation as VE
    # a camera whose back-projection Jacobian is zero at one pixel (the singular-Jacobian testset)
    struct SingularAt{C}
        camera::C
        u::SVector{2, Float64}
    end
    VisualGeometryToolkitCore.backproject_with_jacobian(c::SingularAt, u) =
        u == c.u ? (first(VisualGeometryToolkitCore.backproject_with_jacobian(c.camera, u)), zero(SMatrix{3, 2, Float64, 6})) :
                   VisualGeometryToolkitCore.backproject_with_jacobian(c.camera, u)
    cam = spline_truth()
    # rotations to 0.4 rad (the parametric item's axes × 4): the centre is held well against the offset
    axes = 4 .* [SVector(0.0, 0.0, 0.0), SVector(0.08, 0.0, 0.0), SVector(0.0, 0.08, 0.0), SVector(-0.06, 0.05, 0.03), SVector(0.05, -0.07, -0.04)]
    t = center_truth(axes)
    start = RotationCenterViews(Dict(k => SMatrix{3, 3, Float64, 9}(exp_so3(SVector(0.002, -0.001, 0.0015))) * R for (k, R) in t.R),
                                t.c + SVector(0.01, -0.008, 0.012), t.d + SVector(-0.005, 0.004, 0.006))
    rot_err(est) = maximum(norm(est.views.rotations[k] - R) for (k, R) in t.R)
    exact = center_views(cam, t)
    @testset "exact pixels: the truth, the camera held as given, every dot a bearing; the termination record" begin
        r = solve(RefineProblem(CameraPosesProblem(exact; fixed = (:controls,)), (; camera = cam, views = start)), LevenbergMarquardt(; optimization_strategy = nothing))
        @test successful_retcode(r) && scale_source(r) === ScaleNone
        @test r.retcode == ReturnCode.Success && r.stop_rule != NoRule && retcode_symbol(r) === :converged     # §14: (code, rule)
        @test successful_retcode(r) == successful_retcode(r.retcode)
        est = r.u
        @test est.camera === cam
        # the floor is the pixels' (`project`'s LM, converged below its tolerance): the bearings of the given pixels are exact
        @test norm(est.views.center - t.c) < 2.0e-6 && norm(est.views.offset - t.d) < 2.0e-6 && rot_err(est) < 3.0e-6
        @test r.n == 5 * 63 && dof(r) == 2 * 5 * 63 - (6 + 3 * 5) && r.rms < 1.0e-4
        @test r.metadata.n_without_bearing == 0 && isempty(r.metadata.without_bearing) && r.metadata.n_without_bearing == 0
    end
    @testset "the rim off the pixel domain at the start (a frame turn, as on the chrome fisheyes): converged" begin
        # dots up to 8 px from the image's border, the start's frame turned 50 mrad against the truth's (R_k ← Q R_k, d ← Q d): the
        # start's pose predicts the rim dots off the lens's pixel domain, which the pixel residual (the projection through the
        # spline) could not start from — refused, "does not project at the guess", on 125d80dc. The bearing residual never projects.
        Q = SMatrix{3, 3, Float64, 9}(exp_so3(SVector(0.03, -0.035, 0.015)))
        px = [SVector(x, y) for x in range(8.0, 1272.0; length = 12) for y in range(8.0, 952.0; length = 9)]
        rays = [normalize(backproject(cam, u)) for u in px]
        rim = Dict(k => [(R' * (2.5 * f - (t.d - R * t.c))) => u for (f, u) in zip(rays, px)] for (k, R) in t.R)
        turned = RotationCenterViews(Dict(k => Q * R for (k, R) in t.R), t.c, Q * t.d)
        @test any(k -> any(c -> !all(isfinite, project(cam, Q * t.R[k] * (first(c) - t.c) + Q * t.d)), rim[k]), keys(rim))   # off the domain at the start
        r = solve(RefineProblem(CameraPosesProblem(rim; fixed = (:controls,)), (; camera = cam, views = turned)), LevenbergMarquardt(; optimization_strategy = nothing))
        @test successful_retcode(r) && r.metadata.n_without_bearing == 0
        est = r.u
        @test norm(est.views.center - t.c) < 1.0e-6 && norm(est.views.offset - t.d) < 1.0e-6 && rot_err(est) < 1.0e-6 && r.rms < 1.0e-4
    end
    @testset "noisy pixels: Triggs{Huber} at two given scales' widths, and an absolute width" begin
        noisy = center_views(cam, t; σ = 0.3, rng = StableRNG(5))
        for (strategy, input) in ((Triggs{HuberShape}(efficiency_constant = 3.0), (; scale = 0.3)),
                                  (Triggs{HuberShape}(efficiency_constant = 1.345), (; scale = 0.3)),
                                  (Triggs{HuberShape}(), (; width = 0.9)))
            r = solve(RefineProblem(CameraPosesProblem(noisy; fixed = (:controls,)), (; camera = cam, views = start)), LevenbergMarquardt(; optimization_strategy = strategy); input...)
            @test successful_retcode(r)
            @test haskey(input, :scale) ? (scale_source(r) === ScaleGiven && noise_scale(r) == 0.3) : (scale_source(r) === ScaleNone && r.metadata.loss_width ≈ 0.9)
            # σ = 0.3 px on 5 × 63 dots, no outliers: c, d errors of a few mm (the pixel residual's measured 2.5–3.6 mm), R ≤ 5e-3
            est = r.u
            @test norm(est.views.center - t.c) < 6.0e-3 && norm(est.views.offset - t.d) < 6.0e-3 && rot_err(est) < 8.0e-3
        end
    end
    @testset "5% outliers: Triggs{Huber} down-weights them, least squares is pulled" begin
        bad = center_views(cam, t; σ = 0.3, outliers = 0.05, rng = StableRNG(3))
        r = solve(RefineProblem(CameraPosesProblem(bad; fixed = (:controls,)), (; camera = cam, views = start)), LevenbergMarquardt(; optimization_strategy = Triggs{HuberShape}(efficiency_constant = 3.0)); scale = 0.3)
        ls = solve(RefineProblem(CameraPosesProblem(bad; fixed = (:controls,)), (; camera = cam, views = start)), LevenbergMarquardt(; optimization_strategy = nothing))
        @test successful_retcode(r) && count(<(1), weights(r)) >= 10
        @test norm(r.u.views.center - t.c) < norm(ls.u.views.center - t.c)
    end
    @testset "a masked lattice: a dot the camera does not back-project makes no row and is counted, named" begin
        present = trues(10, 12); present[8, 10] = false          # one node toward the lower-right corner: its 16 windows absent
        masked = with_present(cam, present)
        nob = [(k, j) for (k, cs) in exact for (j, c) in enumerate(cs) if !all(isfinite, backproject(masked, last(c)))]
        @test !isempty(nob)
        r = solve(RefineProblem(CameraPosesProblem(exact; fixed = (:controls,)), (; camera = masked, views = start)), LevenbergMarquardt(; optimization_strategy = nothing))
        @test r.metadata.n_without_bearing == length(nob) && Set(r.metadata.without_bearing) == Set(nob) && r.n == 5 * 63 - length(nob)
        @test successful_retcode(r) && norm(r.u.views.center - t.c) < 2.0e-6 && r.u.camera === masked
    end
    @testset "a dot whose back-projection Jacobian is singular is counted, never an abort" begin
        # the camera's back-projection with one dot's Jacobian zeroed (finite, singular on the tangent plane): BearingPoseProblem
        # refuses such a Jacobian, so built into a view's problem it would abort the whole refine; the blocks find it first,
        # by the problem's own test (`pixel_whitening_defect`), and count it
        u0 = last(exact[3][30])
        singular = SingularAt(cam, u0)
        fM = VisualGeometryToolkitCore.backproject_with_jacobian(singular, u0)
        f0 = normalize(fM[1])
        @test VE.pixel_whitening_defect(f0, fM[2]) === :singular
        fg, Mg = VisualGeometryToolkitCore.backproject_with_jacobian(cam, last(exact[3][1]))
        @test VE.pixel_whitening_defect(normalize(fg), Mg) === nothing
        @test VE.pixel_whitening_defect(normalize(fg), Mg .* NaN) === :nonfinite
        @test VE.pixel_whitening_defect(normalize(fg), hcat(normalize(fg), Mg[:, 2])) === :not_tangent
        good = [VisualGeometryToolkitCore.backproject_with_jacobian(cam, last(c)) for c in exact[3][1:3]]
        @test_throws r"singular" VE.BearingPoseProblem(VisualGeometryToolkitCore.Correspondences(
            [first(c) for c in exact[3][[1, 2, 3, 30]]], push!([normalize(g[1]) for g in good], f0); M = push!([g[2] for g in good], fM[2])))
        ids = sort(collect(keys(t.R)))
        blocks, without = VE._rotation_center_bearing_blocks(singular, ids, exact)
        @test without == [(3, 30)] && length(blocks) == 5 * 63 - 1
    end
    @testset "held_parameters: the `fixed` that holds the camera" begin
        @test VE.held_parameters(cam) == (:controls,)
        r = solve(RefineProblem(CameraPosesProblem(exact; fixed = VE.held_parameters(cam)), (; camera = cam, views = start)), LevenbergMarquardt(; optimization_strategy = nothing))
        @test successful_retcode(r) && r.u.camera === cam
    end
    @testset "refusals: a free camera, a box, a `fixed` that is not the controls" begin
        @test_throws r"HELD camera" solve(RefineProblem(CameraPosesProblem(exact; fixed = ()), (; camera = cam, views = start)), LevenbergMarquardt(; optimization_strategy = nothing))
        @test_throws r"box" solve(RefineProblem(CameraPosesProblem(exact; fixed = (:controls,), box = (; f = (100.0, 900.0))), (; camera = cam, views = start)), LevenbergMarquardt(; optimization_strategy = nothing))
        @test_throws r"one parameter group is its controls" solve(RefineProblem(CameraPosesProblem(exact; fixed = (:controls, :f)), (; camera = cam, views = start)), LevenbergMarquardt(; optimization_strategy = nothing))
    end
    @testset "the bearing block: its Jacobian (ForwardDiff through the bearing residual) matches finite differences; type-stable" begin
        ids = sort(collect(keys(t.R)))
        blocks, without = VE._rotation_center_bearing_blocks(cam, ids, exact)
        @test length(blocks) == 5 * 63 && isempty(without)
        vars = Union{SVector{3, Float64}, VE._ViewRotationChart}[start.center, start.offset]
        foreach(k -> push!(vars, VE._ViewRotationChart(SMatrix{3, 3, Float64, 9}(start.rotations[k]))), ids)
        h = 1.0e-6
        for b in blocks[1:9:end]
            vs = NLLSsolver.getvars(b, vars)
            @test (@inferred NLLSsolver.getvars(b, vars)) isa Tuple{VE._ViewRotationChart, SVector{3, Float64}, SVector{3, Float64}}
            @test (@inferred NLLSsolver.computeresidual(b, vs...)) isa SVector{2, Float64}
            res, J = @inferred NLLSsolver.computeresjac(NLLSsolver.static(7), b, vs...)
            @test res ≈ NLLSsolver.computeresidual(b, vs...) && size(J) == (2, 9)
            col = 0
            for k in 1:3, j in 1:3
                col += 1
                e = zeros(3); e[j] = h
                f(δ) = NLLSsolver.computeresidual(b, ntuple(i -> i == k ? NLLSsolver.update(vs[i], δ) : vs[i], 3)...)
                fd = (f(e) - f(-e)) / 2h
                @test norm(J[:, col] - fd) <= 1.0e-6 * max(norm(fd), maximum(abs, J))
            end
        end
    end
end

# 0.11.6: `ReprojectionResidual()` — the bearing refine, then a second stage from its estimate on the exact pixel
# reprojection error (a SplineLens camera projected by inverting its backward spline, its forward dropped). From noisy
# pixels the bearing estimate is first order in the pixel residual; the pixel stage lowers the pixel residual from it,
# and the projection round trip it relies on is exact.
@testitem "CameraPosesProblem: rotation-centre views — the ReprojectionResidual stage from the bearing estimate" setup = [CameraPoseScenes, SplineCameraPoseScenes, RotationCenterScenes] begin
    using VisualGeometryToolkitCore: dof, s2
    using StaticArrays: SVector, SMatrix
    using LinearAlgebra: norm, normalize
    using VisualGeometryToolkitCore: exp_so3, backproject, project, reprojection_error
    using VisualGeometryToolkitEstimation: CameraPosesProblem, RotationCenterViews, rotation_center_pose, BearingResidual, ReprojectionResidual
    using VisualGeometryToolkitCore: solve, EstimationProblem, RefineProblem, LevenbergMarquardt, LmStops, successful_retcode, RadialCamera, CameraModel
    import VisualGeometryToolkitEstimation as VE
    cam = spline_truth()
    axes = 4 .* [SVector(0.0, 0.0, 0.0), SVector(0.08, 0.0, 0.0), SVector(0.0, 0.08, 0.0), SVector(-0.06, 0.05, 0.03), SVector(0.05, -0.07, -0.04)]
    t = center_truth(axes)
    start = RotationCenterViews(Dict(k => SMatrix{3, 3, Float64, 9}(exp_so3(SVector(0.002, -0.001, 0.0015))) * R for (k, R) in t.R),
                                t.c + SVector(0.01, -0.008, 0.012), t.d + SVector(-0.005, 0.004, 0.006))
    views = center_views(cam, t; σ = 0.5)
    exact_cam = VE._without_forward(cam)
    # the pixel residual of a result's views: the rms over every dot of the exact reprojection error
    function pixel_rms(v)
        e = [reprojection_error(exact_cam, rotation_center_pose(v.rotations[k], v.center, v.offset), X, u) for k in keys(views) for (X, u) in views[k]]
        return sqrt(sum(x -> sum(abs2, x), e) / length(e))
    end
    rb = solve(RefineProblem(CameraPosesProblem(views; fixed = (:controls,)), (; camera = cam, views = start)), LevenbergMarquardt(; optimization_strategy = nothing))
    rd = solve(RefineProblem(CameraPosesProblem(views; fixed = (:controls,), residual = BearingResidual()), (; camera = cam, views = start)), LevenbergMarquardt(; optimization_strategy = nothing))
    @test rd.u.views.center == rb.u.views.center          # the default is the bearing residual
    rp = solve(RefineProblem(CameraPosesProblem(views; fixed = (:controls,), residual = ReprojectionResidual()), (; camera = cam, views = start)), LevenbergMarquardt(; optimization_strategy = nothing))
    @test successful_retcode(rb) && successful_retcode(rp)
    @test isempty(rp.metadata.without_projection) && rp.metadata.n_without_bearing == 0 && isempty(rb.metadata.without_projection)
    # the pixel stage lowers the pixel residual from the bearing estimate, and reports it (its rms IS the pixel residual)
    @test pixel_rms(rp.u.views) < pixel_rms(rb.u.views)
    @test isapprox(rp.rms, pixel_rms(rp.u.views); rtol = 1.0e-6)
    # the round trip the pixel residual relies on: a dot back-projected and projected again through the exact projection
    us = [u for k in keys(views) for (_, u) in views[k]]
    @test maximum(u -> norm(project(exact_cam, normalize(backproject(cam, u))) - u), us) < 1.0e-9
end
