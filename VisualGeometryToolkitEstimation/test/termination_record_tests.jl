# =============================================================================
# The termination record of EVERY producer (the robust-estimation contract, §14):
# each fit, find and refine algorithm, `Ransac` and `LocalOptimization` (seeded,
# fixed candidates, terminal), the one-step and closed-form results, the bundles
# and the camera verbs record a `ReturnCode` and a `StopRule` (never a Symbol,
# never `nothing`), and `successful_retcode(r) == successful_retcode(r.retcode)` for
# every one of them (VisualGeometryToolkitCore's `test_termination_record`).
# Each item lists the producers it covers and checks that every one was reached,
# on a success path and, where the producer has one, a failure path (too few
# data, no model, a cap).
# =============================================================================

@testsetup module TerminationRecord
    using Test
    import VisualGeometryToolkitCore as VgtkCore
    import VisualGeometryToolkitEstimation as VgtkEstimation
    export record!, check_covered

    # One result of the producer `name`: its record checked, the producer marked as reached.
    function record!(seen::Set{Symbol}, name::Symbol, r)
        @test r isa VgtkCore.FitResult
        code = r.retcode
        @test code isa VgtkCore.ReturnCode.T                           # a code, never a Symbol or `nothing`
        @test r.stop_rule isa VgtkCore.StopRule
        @test VgtkCore.successful_retcode(r) === VgtkCore.successful_retcode(code)
        VgtkCore.test_termination_record(r)
        if r.metadata isa VgtkEstimation.RansacMetadata
            # the local optimization's outcome is the same record, typed: SciMLBase's Default exactly when none ran
            lo_code = Test.@inferred (x -> x.metadata.lo_retcode)(r)
            lo_rule = Test.@inferred (x -> x.metadata.lo_stop_rule)(r)
            @test lo_code isa VgtkCore.ReturnCode.T && lo_rule isa VgtkCore.StopRule
            VgtkCore.found_model(r) || @test (lo_code, lo_rule) == (VgtkCore.ReturnCode.Default, VgtkCore.NoRule)
            lo_code == VgtkCore.ReturnCode.Default && @test lo_rule == VgtkCore.NoRule
        end
        push!(seen, name)
        return r
    end
    check_covered(seen, expected) = @test isempty(setdiff(Set(expected), seen))
end

@testitem "termination record: every find and refine, Ransac and LocalOptimization" setup = [TerminationRecord, RadialCameraScene,
        EpipolarScenes, ConicScenes, RotationScene, TooFewData, FindScenes] begin
    import VisualGeometryToolkitEstimation as VgtkEstimation; include(joinpath(pkgdir(VgtkEstimation), "test", "fixtures", "thin_homography.jl"))   # `thin_homography_correspondences`
    using Test, Random, LinearAlgebra, StaticArrays, StructArrays
    using StableRNGs
    using Rotations: RotMatrix, RotXYZ, RotX, RotZ
    import VisualGeometryToolkitCore as VgtkCore
    import VisualGeometryToolkitEstimation as VgtkEstimation
    using VisualGeometryToolkitCore: CameraModel, Vec3, correspondences, radial_transfer, RadialHomography, HomographyMat, MsacScoring, Irls, Triggs, TLSShape, WelschShape, ReturnCode, stop_rule, found_model, Point2, Correspondences
    using VisualGeometryToolkitEstimation: RansacConfig, HomographyProblem, TerminalLO, Ransac, LocalOptimization, FundMatProblem,
        EssentialProblem, RotationProblem, CircleProblem, EllipseProblem, AbsolutePoseProblem, RadialHomographyProblem,
        RadialCameraProblem, HandEyeProblem
    using VisualGeometryToolkitCore: solve, EstimationProblem, RefineProblem, LevenbergMarquardt, successful_retcode

    seen = Set{Symbol}()
    # --- the scenes: FindScenes (core_defaults_tests.jl's), and the conic and rotation scenes ---------------------------
    cs_h, _ = h_scene()
    u₁, u₂, _ = stereo_scene(PCAM); cs_f = StructArray(u₁ .=> u₂)
    v₁, v₂, _ = stereo_scene(ECAM; seed = 14); cs_e = correspondences(v₁, v₂)
    cs_p = pnp_scene()
    rh, _ = radial_homography_scene(); c_rh = (RW / 2, RH / 2)
    rc, _ = radial_camera_pairs()
    he, _ = hand_eye_scene()
    pts_c, _ = circle_scene(Xoshiro(33), 300; outl = 0.3)
    pts_e, _ = ellipse_scene(Xoshiro(34), 300; outl = 0.3)
    cs_r = rotation_scene(Xoshiro(35), RotMatrix(RotXYZ(0.1, -0.2, 0.3)), 120; outl = 0.3)[1]
    # hand-eye with no model: every rig motion turns about z, so every sample's axes are parallel (ransac_no_model_tests.jl)
    he_parallel = [VgtkCore.EuclideanMap(RotMatrix(RotZ(0.35i)), SVector(0.1i, 0.05i^2, 0.2)) =>
                   VgtkCore.EuclideanMap(RotMatrix(RotX(0.35i)), SVector(0.2, 0.1i, -0.03i)) for i in 1:8]

    # --- every family problem's find and refine: the search, the refine from its model, and the too-few-data paths ----
    # (class, the problem `f`, the same family on too few data for a minimal sample)
    classes = (
        (:homography, HomographyProblem(cs_h), HomographyProblem(few_correspondences(3))),
        (:fundmat, FundMatProblem(cs_f), FundMatProblem(few_correspondences(6))),
        (:essential, EssentialProblem(ECAM, ECAM, cs_e), EssentialProblem(ECAM, ECAM, few_correspondences(4))),
        (:rotation, RotationProblem(FISHEYE, FISHEYE, cs_r), RotationProblem(FISHEYE, FISHEYE, few_correspondences(1))),
        (:circle, CircleProblem(pts_c), CircleProblem(few_points(2))),
        (:ellipse, EllipseProblem(pts_e), EllipseProblem(few_points(4))),
        (:absolute_pose, AbsolutePoseProblem(PCAM, cs_p), AbsolutePoseProblem(PCAM, cs_p[1:2])),
        (:radial_homography, RadialHomographyProblem(rh; center = c_rh), RadialHomographyProblem(rh[1:4]; center = c_rh)),
        (:radial_camera, RadialCameraProblem(rc; center = RC_CENTER), RadialCameraProblem(rc[1:4]; center = RC_CENTER)),
        (:hand_eye, HandEyeProblem(he), HandEyeProblem(he_parallel)),
    )
    for (seed, (class, f, few)) in enumerate(classes)
        @testset "$class" begin
            r = record!(seen, Symbol(:find_, class), solve(EstimationProblem(f), Ransac(); rng = StableRNG(seed)))
            @test found_model(r)
            # a capped search: the trial budget drawn without the confidence is MaxIters (not converged), a model found
            capped = record!(seen, Symbol(:find_, class),
                solve(EstimationProblem(f), Ransac(; config = RansacConfig(max_trials = 2, min_trials = 2)); rng = StableRNG(seed)))
            @test capped.stop_rule in (VgtkCore.TrialBudget, VgtkCore.AdaptiveCount)
            capped.stop_rule == VgtkCore.TrialBudget && found_model(capped) && @test capped.retcode == ReturnCode.MaxIters
            # no model: too few data for a minimal sample (hand-eye: no sample holds a model)
            none = record!(seen, Symbol(:find_, class), solve(EstimationProblem(few), Ransac()))
            @test !found_model(none) && none.retcode == ReturnCode.Failure && !successful_retcode(none)
            # the refine from the found model: least squares, and the class's default robust strategy at a scale
            θ = r.u
            record!(seen, Symbol(:refine_, class), solve(RefineProblem(f, θ), LevenbergMarquardt(; optimization_strategy = nothing)))
            record!(seen, Symbol(:refine_, class), solve(RefineProblem(f, θ), LevenbergMarquardt(); scale = 1.0))
        end
    end
    # the refine's too-few-data path (the guess unrefined: InitialFailure)
    H = HomographyMat{Float64}(Tuple(SMatrix{3, 3, Float64}(I)))
    deg = record!(seen, Symbol(:refine_, :homography), solve(RefineProblem(HomographyProblem(few_correspondences(3)), H), LevenbergMarquardt(); scale = 1.0))
    @test deg.retcode == ReturnCode.InitialFailure && deg.stop_rule == VgtkCore.NoRule

    # --- Ransac on a problem, LocalOptimization (seeded, fixed candidates) and a terminal LO ---------------------------------------
    p = HomographyProblem(cs_h)
    sc = MsacScoring(p; threshold = 3.0)
    rr = record!(seen, :ransac, solve(EstimationProblem(p), Ransac(; scoring = sc, local_optimization_strategy = Irls{TLSShape}()); rng = StableRNG(1)))
    @test successful_retcode(rr) && rr.stop_rule == VgtkCore.AdaptiveCount
    record!(seen, :ransac, solve(EstimationProblem(p), Ransac(; scoring = sc, local_optimization_strategy = nothing); rng = StableRNG(1)))                           # no LO
    record!(seen, :ransac, solve(EstimationProblem(p), Ransac(; scoring = sc, local_optimization_strategy = TerminalLO(Irls{TLSShape}())); rng = StableRNG(1)))
    θ = rr.u
    for lo in (nothing, Irls{TLSShape}(), Triggs{WelschShape}())
        record!(seen, :LocalOptimization, solve(RefineProblem(p, θ), LocalOptimization(lo; scoring = sc)))
    end
    record!(seen, :LocalOptimization, solve(RefineProblem(p, VgtkCore.FixedModels(1, (θ, θ))), LocalOptimization(nothing; scoring = sc)))
    # a terminal LO on a thin incumbent (m + 1 points: no residual dof): the refit did not run, its record
    # (InitialFailure, NoRule); the result keeps the search's record
    pt = HomographyProblem(thin_homography_correspondences()); sct = MsacScoring(pt; threshold = 0.5)   # the suite's fixture file
    cfgt = RansacConfig(max_trials = 5000, min_trials = 3000)
    thin = record!(seen, :ransac, solve(EstimationProblem(pt), Ransac(; scoring = sct, local_optimization_strategy = TerminalLO(Irls{WelschShape}(efficiency_constant = 1.5)), config = cfgt); rng = StableRNG(1)))
    @test (thin.metadata.lo_retcode, thin.metadata.lo_stop_rule) == (ReturnCode.InitialFailure, VgtkCore.NoRule)
    # the per-candidate score guard
    record!(seen, :ransac, solve(EstimationProblem(p), Ransac(; scoring = sc, local_optimization_strategy = Irls{TLSShape}(), score_guard = true); rng = StableRNG(1)))

    # the adaptive count's boundary: a confidence count equal to `max_trials` was reached AT the cap (AdaptiveCount);
    # one past it was not (the budget drawn without it)
    c, reached = VgtkEstimation._adaptive_count(60, 100, 4, RansacConfig(max_trials = 10^6, min_trials = 1))
    @test reached && c > 1
    @test VgtkEstimation._adaptive_count(60, 100, 4, RansacConfig(max_trials = c, min_trials = 1)) == (c, true)
    @test VgtkEstimation._adaptive_count(60, 100, 4, RansacConfig(max_trials = c - 1, min_trials = 1)) == (c - 1, false)

    check_covered(seen, (Symbol.(:find_, first.(classes))..., Symbol.(:refine_, first.(classes))..., :ransac, :LocalOptimization))
end

@testitem "termination record: every fit_* and one-step result" setup = [TerminationRecord, FitContractScenes, DivisionBoards] begin
    using VisualGeometryToolkitCore: successful_retcode
    using Test, LinearAlgebra
    using StableRNGs
    using StaticArrays: SVector, SMatrix
    using Rotations: RotXYZ
    import VisualGeometryToolkitCore as VgtkCore
    using VisualGeometryToolkitCore: Point2, Uncertain, CovMat, RadialCamera, StudentTUniformScoring, Irls, HuberShape, ReturnCode, stop_rule
    using VisualGeometryToolkitEstimation: RadialCameraFit, DivisionCameraFit, ViewGraph, _heteroscedastic_line, ReweightedLeastSquares, LinearModel, ResidualMixture, FilteredRotationAveraging
    using VisualGeometryToolkitCore: solve, EstimationProblem, RefineProblem, LevenbergMarquardt, LmStops, successful_retcode, RadialCamera, CameraModel
    using VisualGeometryToolkitCore: EstimationProblem, solve, PosteriorWeightState, AbsoluteRotations
    include(joinpath(pkgdir(VisualGeometryToolkitEstimation), "test", "fixtures", "reweighted_fits.jl"))

    seen = Set{Symbol}()
    # the five reweighted fits: least squares (a closed form, converged) and a robust IRLS, and a degenerate input
    for (name, codim, fit) in CONTRACT_FITS
        @testset "$name" begin
            l2 = record!(seen, name, fit((; optimization_strategy = nothing)))
            @test successful_retcode(l2)
            record!(seen, name, fit((; optimization_strategy = Irls{HuberShape}(), scale = 1.0e-3)))
            capped = record!(seen, name, fit((; optimization_strategy = Irls{HuberShape}(max_iter = 1), scale = 1.0e-9)))
            capped.retcode == ReturnCode.MaxIters && @test !successful_retcode(capped)
        end
    end
    @test linear_fit(contract_linear()...; optimization_strategy = nothing).stop_rule == VgtkCore.ClosedForm
    deg = record!(seen, :linear, linear_fit(zeros(2, 3), zeros(2); optimization_strategy = nothing))
    @test deg.retcode == ReturnCode.InitialFailure && !successful_retcode(deg)
    # a weighted solve that cannot be solved inside the loop (every TLS weight zero at a width far below the residuals):
    # SciMLBase's InternalLinearSolveFailed, not a degenerate start
    rng_l = StableRNG(9)
    ils = record!(seen, :linear, linear_fit(randn(rng_l, 8, 3), randn(rng_l, 8);
                                             optimization_strategy = Irls{VgtkCore.TLSShape}(efficiency_constant = 3.0), scale = 1.0e-9))
    @test ils.retcode == ReturnCode.InternalLinearSolveFailed && !successful_retcode(ils) && (ils.retcode, ils.stop_rule) === (VgtkCore.ReturnCode.InternalLinearSolveFailed, VgtkCore.NoRule)
    # the heteroscedastic line: the one covariance-reweighted solve (a closed form)
    pts = [Uncertain(Point2{Float64}(Float64(i), 2.0i + 1.0 + 0.01 * sin(i)), CovMat(SMatrix{2, 2, Float64}(1.0e-4, 0.0, 0.0, 1.0e-4)))
           for i in 1:20]
    rl = record!(seen, :heteroscedastic_line, _heteroscedastic_line(pts))
    @test successful_retcode(rl) && rl.stop_rule == VgtkCore.ClosedForm
    # RadialCameraFit: the iterated solve (closed form) and a degenerate configuration
    cam = board_camera(RotXYZ(BOARD_ANGLES[1]...), BOARD_TS[1])
    pp = board_pixels(cam)
    record!(seen, :RadialCameraFit, solve(EstimationProblem(RadialCamera, pp, (; center = SVector(0.0, 0.0))), RadialCameraFit()))
    collinear = [SVector(Float64(i), 0.0) => SVector(10.0i, 0.0) for i in 1:8]
    record!(seen, :RadialCameraFit, solve(EstimationProblem(RadialCamera, collinear, (; center = SVector(0.0, 0.0))), RadialCameraFit()))
    # DivisionCameraFit: the EM over a focal stack
    cams = [board_camera(RotXYZ(a...), t) for (a, t) in zip(BOARD_ANGLES, BOARD_TS)]
    boards, _ = noisy_board_views(StableRNG(3), cams; σ = 0.4, ε = 0.08)
    rdc = record!(seen, :DivisionCameraFit, solve(EstimationProblem(CameraModel, (Dict(enumerate(RadialCamera.(cams))), Dict(enumerate(boards))), (; image_size = (4000, 3000))), DivisionCameraFit()))
    # ResidualMixture: the EM on a held model's distances
    r = abs.(vcat(0.5 .* randn(StableRNG(5), 200), 40.0 .* rand(StableRNG(6), 40)))
    mix = solve(EstimationProblem(PosteriorWeightState, r, (; model_dof = 4)),
                ResidualMixture(; weighting = StudentTUniformScoring(4, 40.0; codimension = 1, scale_prior = VgtkCore.JeffreysPrior(),
                                                                      label_prior = VgtkCore.FlatLabelPrior())))
    record!(seen, :residual_mixture, mix)
    # FilteredRotationAveraging: its result is its last RotationAveraging's
    Rs = Dict(k => SMatrix{3, 3, Float64, 9}(RotXYZ(0.3k, -0.2k, 0.1k^2)) for k in 1:5)
    vg = ViewGraph(Dict((i, j) => Rs[j] * Rs[i]' for i in 1:5 for j in (i + 1):5))
    record!(seen, :FilteredRotationAveraging, solve(EstimationProblem(AbsoluteRotations, vg), FilteredRotationAveraging(); scale = 1.0e-3))

    check_covered(seen, (first.(CONTRACT_FITS)..., :heteroscedastic_line, :RadialCameraFit, :DivisionCameraFit, :residual_mixture,
                         :FilteredRotationAveraging))
end

@testitem "termination record: the camera verbs" setup = [CameraRecipes, TerminationRecord, FitCameraFisheye] begin
    using VisualGeometryToolkitCore: successful_retcode
    using Test
    using StaticArrays: SVector
    import VisualGeometryToolkitCore as VgtkCore
    import VisualGeometryToolkitEstimation as VgtkEstimation
    using VisualGeometryToolkitCore: CameraModel, KannalaBrandt, SplineLens, BoundedCamera, Size2, Division, stop_rule, with_lens_coverage
    using VisualGeometryToolkitEstimation: CameraRegressionProblem, CameraRegression, OpticalAxis, PSpline
    using VisualGeometryToolkitCore: solve, EstimationProblem, RefineProblem, LevenbergMarquardt, LmStops, successful_retcode, RadialCamera, CameraModel

    seen = Set{Symbol}()
    bc = BoundedCamera(FISHEYE, Size2(1000, 1000))
    # the linear fit (a closed form), its refine (the chart LM) and the two in one
    lin = record!(seen, :CameraRegression, solve(EstimationProblem(CameraRegressionProblem(KannalaBrandt{4}, FISHEYE, REGION)), CameraRegression()))
    @test successful_retcode(lin) && lin.stop_rule == VgtkCore.ClosedForm
    record!(seen, :CameraRegressionRefine, solve(RefineProblem(CameraRegressionProblem(FISHEYE, REGION), lin.u), LevenbergMarquardt()))
    record!(seen, :CameraRegressionRefine, solve(RefineProblem(CameraRegressionProblem(FISHEYE, REGION), lin.u), LevenbergMarquardt(); stops = LmStops(; max_iter = 1)))              # the cap
    # nothing free (every parameter held on a physical K): the guess returned, no LM — a closed form
    held = record!(seen, :CameraRegressionRefine, solve(RefineProblem(CameraRegressionProblem(FISHEYE, REGION; fixed = Tuple(VgtkCore.parameter_names(lin.u))), lin.u), LevenbergMarquardt()))
    @test held.retcode == VgtkCore.ReturnCode.Success && held.stop_rule == VgtkCore.ClosedForm && successful_retcode(held) &&
        held.stats.nsteps == 0
    record!(seen, :FitThenRefine, solve(EstimationProblem(CameraRegressionProblem(Division{2}, FISHEYE, REGION)), CameraRegression(; refine = LevenbergMarquardt())))
    # the spline fits: the spacing search to a tolerance, the one lattice solve, and the optical axis
    rs = record!(seen, :CameraRegression, solve(EstimationProblem(CameraRegressionProblem(SplineLens, bc, REGION)), CameraRegression(; tolerance = 1.0e-3)))
    capped = record!(seen, :CameraRegression, solve(EstimationProblem(CameraRegressionProblem(SplineLens, bc, REGION)), CameraRegression(; tolerance = 1.0e-12, forward = false)))   # the finest spacing misses it
    gw, gh = rs.metadata.family.grid_size
    record!(seen, :PSpline, solve(EstimationProblem(CameraRegressionProblem(SplineLens, bc, REGION)), PSpline(; grid_size = (gw, gh))))
    record!(seen, :OpticalAxis, solve(EstimationProblem(CameraRegressionProblem(with_lens_coverage(GRID_BOUNDED, DISC))), OpticalAxis()))
    # the stages' records are carried through as records, by Core's one rule of several solves (`worst_retcode`): the
    # most severe, rule included — a Stalled optical-axis gauge stays Stalled (never re-read as a tolerance search's) —
    # and an earlier stage's failure is never replaced by a later, less severe one (the private `_spline_record` let a
    # later Stalled replace an earlier MaxIters; it is deleted)
    @test !isdefined(VgtkEstimation, :_spline_record)
    wr = VgtkCore.worst_retcode
    @test wr((VgtkCore.ReturnCode.Success, VgtkCore.ToleranceMet), (VgtkCore.ReturnCode.Stalled, VgtkCore.IncrementSize)) ==
        (VgtkCore.ReturnCode.Stalled, VgtkCore.IncrementSize)
    @test wr((VgtkCore.ReturnCode.MaxIters, VgtkCore.IterationCap), (VgtkCore.ReturnCode.Success, VgtkCore.NoRule)) ==
        (VgtkCore.ReturnCode.MaxIters, VgtkCore.IterationCap)
    @test wr((VgtkCore.ReturnCode.MaxIters, VgtkCore.IterationCap), (VgtkCore.ReturnCode.Stalled, VgtkCore.IncrementSize)) ==
        (VgtkCore.ReturnCode.MaxIters, VgtkCore.IterationCap)
    # a tie of severity keeps the LATER stage's record (the former rule kept the earlier on a success): two successes,
    # the closed-form stage after the tolerance search
    @test wr((VgtkCore.ReturnCode.Success, VgtkCore.ToleranceMet), (VgtkCore.ReturnCode.Success, VgtkCore.ClosedForm)) ==
        (VgtkCore.ReturnCode.Success, VgtkCore.ClosedForm)
    # the backward search capped at the finest spacing: the fit does not report a success, and its record is at least as
    # severe as a capped search's
    @test !VgtkCore.successful_retcode(capped.retcode) && !successful_retcode(capped)
    @test wr((VgtkCore.ReturnCode.MaxIters, VgtkCore.IterationCap), (capped.retcode, capped.stop_rule)) ==
        (capped.retcode, capped.stop_rule)
    check_covered(seen, (:CameraRegression, :CameraRegressionRefine, :FitThenRefine, :PSpline, :OpticalAxis))
end

@testitem "termination record: the non-central spline fit" setup = [TerminationRecord, FitCameraNoncentral] begin
    using Test
    using VisualGeometryToolkitCore: NoncentralSplineModel
    using VisualGeometryToolkitEstimation: CameraRegressionProblem, CameraRegression
    using VisualGeometryToolkitCore: solve, EstimationProblem, RefineProblem, LevenbergMarquardt, LmStops, successful_retcode, RadialCamera, CameraModel
    seen = Set{Symbol}()
    record!(seen, :CameraRegression, solve(EstimationProblem(CameraRegressionProblem(NoncentralSplineModel, SOURCE, REGION)), CameraRegression(; tolerance = 1.0e-3)))
    check_covered(seen, (:CameraRegression,))
end

@testitem "termination record: the bundles" setup = [CameraRecipes, TerminationRecord, CameraPoseScenes, SplineCameraPoseScenes, BoardScenes, PoseScenes,
        RotationCenterScenes] begin
    using Test, LinearAlgebra
    using StaticArrays: SVector, SMatrix
    using StableRNGs
    using Rotations: AngleAxis
    import VisualGeometryToolkitCore as VgtkCore
    import VisualGeometryToolkitEstimation as VgtkEstimation
    using VisualGeometryToolkitCore: CameraModel, parameter_vector, with_parameters, Triggs, HuberShape, ReturnCode, project
    using VisualGeometryToolkitEstimation: CameraPosesProblem, RobotWorldHandEyeProblem, RobotWorldHandEyeCalibration
    using VisualGeometryToolkitCore: solve, EstimationProblem, RefineProblem, LevenbergMarquardt, LmStops, successful_retcode, RadialCamera, CameraModel
    using VisualGeometryToolkitCore: RefineProblem, LevenbergMarquardt, solve

    seen = Set{Symbol}()
    # the parametric bundle: least squares, the robust Triggs (and its LM's iteration cap), the board form
    cam = pinhole_camera(); Es = truth_poses(5)
    views = views_of(cam, Es, 0.6)
    θ = parameter_vector(cam)
    cam0 = with_parameters(cam, θ .* SVector(1.04, 1.01, 1.0, 1.0, 1.0) .+ SVector(0.0, 0.0, 0.0, 6.0, -5.0))
    poses0 = [perturbed_pose(E, v) for (v, E) in enumerate(Es)]
    l2 = record!(seen, :CameraPosesProblem, solve(RefineProblem(CameraPosesProblem(views), (; camera = cam0, poses = poses0)), LevenbergMarquardt(; optimization_strategy = nothing)))
    @test successful_retcode(l2)
    record!(seen, :CameraPosesProblem, solve(RefineProblem(CameraPosesProblem(views), (; camera = cam0, poses = poses0)), LevenbergMarquardt(; optimization_strategy = Triggs{HuberShape}()); scale = 0.2))
    capped = record!(seen, :CameraPosesProblem,
        solve(RefineProblem(CameraPosesProblem(views), (; camera = cam0, poses = poses0)), LevenbergMarquardt(; optimization_strategy = Triggs{HuberShape}()); scale = 0.2, stops = poses_stops(; max_iter = 1)))
    @test capped.retcode == ReturnCode.MaxIters && !successful_retcode(capped)
    record!(seen, :CameraPosesProblem, solve(RefineProblem(CameraPosesProblem(views), (; camera = cam0, poses = poses0)), LevenbergMarquardt(; optimization_strategy = nothing); stops = poses_stops(; max_iter = 1)))
    boards = truth_boards(); vposes = view_poses()
    views0 = Dict(v => nudged(E, v) for (v, E) in vposes)
    keyn = Dict(:A => 1, :B => 2, :C => 3)
    boards0 = Dict(b => b === :A ? E : nudged(E, 7 + keyn[b]) for (b, E) in boards)
    record!(seen, :CameraPosesProblem, solve(RefineProblem(CameraPosesProblem(board_views(cam, vposes, boards)), (; camera = cam0, poses = views0, board_poses = boards0)), LevenbergMarquardt(; optimization_strategy = nothing)))
    # rotation-centre views (RotationCenterViews): the views' rotations about one centre
    axes = [SVector(0.0, 0.0, 0.0), SVector(0.08, 0.0, 0.0), SVector(0.0, 0.08, 0.0), SVector(-0.06, 0.05, 0.03), SVector(0.05, -0.07, -0.04),
            SVector(0.0, 0.0, 0.1)]
    t = center_truth(axes)
    start = VgtkEstimation.RotationCenterViews(
        Dict(k => SMatrix{3, 3, Float64, 9}(VgtkCore.exp_so3(SVector(0.002, -0.001, 0.0015))) * R for (k, R) in t.R),
        t.c + SVector(0.01, -0.008, 0.012), t.d + SVector(-0.005, 0.004, 0.006))
    held = Tuple(VgtkCore.parameter_names(cam))
    record!(seen, :CameraPosesProblem, solve(RefineProblem(CameraPosesProblem(center_views(cam, t); fixed = held), (; camera = cam, views = start)), LevenbergMarquardt(; optimization_strategy = nothing)))
    # Core's bundle metadata (the board bundles of CaliBlobs and BlobBoards): the record in the core, the scale record in it
    s_ = (; rms = 0.3, median = 0.25, p90 = 0.6)
    bm = VgtkCore.BundleMetadata([1.0, 0.5], [0.2, 0.9], [1], 10, 1.5, s_, s_; scale = 0.4, scale_source = VgtkCore.ScaleGiven)
    rb = record!(seen, :BundleMetadata, VgtkCore.FitResult(:calibration, ReturnCode.Success, VgtkCore.CostDecrease, VgtkCore.SolveStats(; nsteps = 12), 0.3, 2, bm))
    @test VgtkCore.noise_scale(rb) === 0.4 && VgtkCore.scale_source(rb) === VgtkCore.ScaleGiven
    l2m = VgtkCore.BundleMetadata([1.0, 1.0], [0.2, 0.9], [1], 10, NaN, s_, s_; scale = NaN, scale_source = VgtkCore.ScaleNone)
    record!(seen, :BundleMetadata, VgtkCore.FitResult(:calibration, ReturnCode.MaxIters, VgtkCore.IterationCap, VgtkCore.SolveStats(; nsteps = 50), 0.3, 2, l2m))
    @test_throws ArgumentError VgtkCore.noise_scale(VgtkCore.FitResult(:c, ReturnCode.Success, VgtkCore.CostDecrease, VgtkCore.SolveStats(; nsteps = 1), 0.3, 2, l2m))
    # the SplineLens bundle (the camera held: refined only by back-projection, or held)
    truth = spline_truth(); Es3 = truth_poses(3)
    sviews = spline_views(truth, Es3, 1.1)
    record!(seen, :CameraPosesProblem, solve(RefineProblem(CameraPosesProblem(sviews; fixed = (:controls,)), (; camera = bounded(perturbed_controls(truth, 0.01)), poses = [perturbed_pose(E, v) for (v, E) in enumerate(Es3)])), LevenbergMarquardt(; optimization_strategy = nothing)))
    # the robot-world hand-eye bundle: markers' features from a seed near the truth
    rng = StableRNG(3)
    camera = CameraModel(; focal = 600.0, pp = (400.0, 300.0))
    grid = [SVector(0.03 * (i - 3), 0.03 * (j - 3), 0.0) for i in 1:5 for j in 1:5]
    Y = rigid(AngleAxis(0.7, normalize(SVector(0.3, -0.2, 0.9))...), SVector(0.2, -0.1, 0.3))
    X = [rigid(AngleAxis(0.2, normalize(SVector(1.0, 0.5, 0.2))...), SVector(0.08 * (m - 1.5), 0.0, 0.0)) for m in 1:2]
    A = [Y ∘ rigid(AngleAxis(0.3 * rand(rng), normalize(randn(rng, 3))...), SVector(0.1 * randn(rng), 0.1 * randn(rng), 1.2)) for _ in 1:10]
    markers = Dict(m => [A[i] => [P => SVector{2, Float64}(project(camera, marker_pose(Y, A[i], X[m])(P))) + 0.3 * randn(rng, SVector{2, Float64})
                                  for P in grid] for i in eachindex(A)] for m in 1:2)
    seed = RobotWorldHandEyeCalibration(perturb_pose(rng, Y; σ_R = 0.01, σ_t = 0.01),
                                        Dict(m => perturb_pose(rng, X[m]; σ_R = 0.01, σ_t = 0.01) for m in 1:2))
    rwp = RefineProblem(RobotWorldHandEyeProblem(camera, markers), seed)
    record!(seen, :robot_world_hand_eye_refine, solve(rwp, LevenbergMarquardt(; optimization_strategy = nothing)))
    record!(seen, :robot_world_hand_eye_refine, solve(rwp, LevenbergMarquardt(; optimization_strategy = Triggs{HuberShape}()); scale = 0.3))
    check_covered(seen, (:CameraPosesProblem, :robot_world_hand_eye_refine, :BundleMetadata))
end

@testitem "termination record: DivisionCameraFit audits no failed leg" begin
    # A leg whose solve failed mid-loop (the IRLS step's InternalLinearSolveFailed, the owner's ruling) is neither
    # branch-audited nor resumed: the failed-leg predicate is `_failed_leg(record...)`, so no failure code (a non-finite start, a failed solve) is audited.
    using VisualGeometryToolkitEstimation
    import VisualGeometryToolkitCore as VgtkCore
    failed(code, rule) = VisualGeometryToolkitEstimation._failed_leg(code, rule)
    @test failed(VgtkCore.ReturnCode.InternalLinearSolveFailed, VgtkCore.NoRule)
    @test failed(VgtkCore.ReturnCode.Failure, VgtkCore.NoRule)
    @test failed(VgtkCore.ReturnCode.InitialFailure, VgtkCore.NonFinite)
    @test !failed(VgtkCore.ReturnCode.Success, VgtkCore.ParameterChange)
    @test !failed(VgtkCore.ReturnCode.MaxIters, VgtkCore.IterationCap)
    @test !failed(VgtkCore.ReturnCode.Stalled, VgtkCore.IncrementSize)
    @test !failed(VgtkCore.ReturnCode.Failure, VgtkCore.TrialBudget)
    @test !failed(VgtkCore.ReturnCode.Failure, VgtkCore.SeedKept)
    # every record against the ruling written out on its own (not through a symbol mapping): a leg failed when its code
    # is a failure — not a stop that kept a usable estimate (Success, Terminated, the caps, Stalled), not a run that
    # never started (Default) — and its rule is not a guard that kept the seed, not a no-model verdict (a trial budget
    # or count exhausted, a sampler's) and, for InitialFailure, a non-finite start (a degenerate one is no solve)
    RC = VgtkCore.ReturnCode
    kept = (RC.Default, RC.Success, RC.Terminated, RC.MaxIters, RC.MaxTime, RC.Stalled)
    expected(code, rule) = !(code in kept) && rule != VgtkCore.SeedKept &&
        !(code == RC.Failure && rule in (VgtkCore.TrialBudget, VgtkCore.AdaptiveCount)) &&
        !(code == RC.InitialFailure && rule != VgtkCore.NonFinite)
    for code in instances(RC.T), rule in instances(VgtkCore.StopRule)
        @test failed(code, rule) == expected(code, rule)
    end
end

# The reweighted fits' termination record is public (the owner's ruling on the BlobBoards duplication audit, ask 6;
# two scalar fields since the conformance work, no tuple): a downstream reweighted loop (BlobBoards' per-blob IRLS)
# records its stops through them, an increment below its tolerance as `IncrementSize`.
@testitem "irls_retcode, irls_stop_rule: the reweighted fits' one termination record, public" begin
    using Test
    import VisualGeometryToolkitEstimation as VgtkEstimation
    using VisualGeometryToolkitCore: ReturnCode, IncrementSize, ClosedForm, IterationCap
    @test Base.ispublic(VgtkEstimation, :irls_retcode) && Base.ispublic(VgtkEstimation, :irls_stop_rule)
    record(c, cf) = (VgtkEstimation.irls_retcode(c, cf), VgtkEstimation.irls_stop_rule(c, cf))
    @test record(true, false) === (ReturnCode.Success, IncrementSize)
    @test record(true, true) === (ReturnCode.Success, IncrementSize)
    @test record(false, true) === (ReturnCode.Success, ClosedForm)
    @test record(false, false) === (ReturnCode.MaxIters, IterationCap)
end
