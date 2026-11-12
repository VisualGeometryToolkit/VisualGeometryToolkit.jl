# =============================================================================
# The termination record of EVERY producer (the robust-estimation contract, §14):
# each `fit_*`, `find_*` and `refine_*`, `ransac` and `locally_optimize` (seeded,
# fixed candidates, terminal), the one-step and closed-form results, the bundles
# and the camera verbs record a `ReturnCode` and a `StopRule` (never a Symbol,
# never `nothing`), and `converged(r) == successful_retcode(retcode(r))` for
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
        code = VgtkCore.retcode(r)
        @test code isa VgtkCore.ReturnCode.T                           # a code, never a Symbol or `nothing`
        @test VgtkCore.stop_rule(r) isa VgtkCore.StopRule
        @test VgtkCore.converged(r) === VgtkCore.successful_retcode(code)
        VgtkCore.test_termination_record(r)
        if VgtkCore.metadata(r) isa VgtkEstimation.RansacMetadata
            # the local optimization's outcome is the same record, typed: SciMLBase's Default exactly when none ran
            lo_code = Test.@inferred (x -> x.lo_retcode)(r)
            lo_rule = Test.@inferred (x -> x.lo_stop_rule)(r)
            @test lo_code isa VgtkCore.ReturnCode.T && lo_rule isa VgtkCore.StopRule
            VgtkCore.found_model(r) || @test (lo_code, lo_rule) == (VgtkCore.ReturnCode.Default, VgtkCore.NoRule)
            lo_code == VgtkCore.ReturnCode.Default && @test lo_rule == VgtkCore.NoRule
        end
        push!(seen, name)
        return r
    end
    check_covered(seen, expected) = @test isempty(setdiff(Set(expected), seen))
end

@testitem "termination record: every find_*, refine_*, ransac and locally_optimize" setup = [TerminationRecord, RadialCameraScene,
        EpipolarScenes, ConicScenes, RotationScene, TooFewData, FindScenes] begin
    using Test, Random, LinearAlgebra, StaticArrays, StructArrays
    using StableRNGs
    using Rotations: RotMatrix, RotXYZ, RotX, RotZ
    import VisualGeometryToolkitCore as VgtkCore
    import VisualGeometryToolkitEstimation as VgtkEstimation
    using VisualGeometryToolkitCore: CameraModel, Vec3, csponds, radial_transfer, RadialHomography, HomographyMat, estimate,
        MsacScoring, Irls, Triggs, TLSShape, WelschShape, ReturnCode, retcode, stop_rule, converged, found_model,
        Point2, Csponds
    using VisualGeometryToolkitEstimation: find_homography, find_fundmat, find_essential, find_rotation, find_circle, find_ellipse,
        find_absolute_pose, find_radial_homography, find_radial_camera, find_hand_eye, refine_homography, refine_fundmat,
        refine_essential, refine_rotation, refine_circle, refine_ellipse, refine_absolute_pose, refine_radial_homography,
        refine_radial_camera, refine_hand_eye, ransac, locally_optimize, RansacConfig, HomographyProblem, TerminalLO

    seen = Set{Symbol}()
    # --- the scenes: FindScenes (core_defaults_tests.jl's), and the conic and rotation scenes ---------------------------
    cs_h, _ = h_scene()
    u₁, u₂, _ = stereo_scene(PCAM); cs_f = StructArray(u₁ .=> u₂)
    v₁, v₂, _ = stereo_scene(ECAM; seed = 14); cs_e = csponds(v₁, v₂)
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

    # --- every find_* and its refine_*: the search, the refine from its model, and the too-few-data paths ------------
    classes = (
        (:homography, kw -> find_homography(cs_h; kw...), (θ, kw) -> refine_homography(θ, cs_h; kw...),
         kw -> find_homography(few_csponds(3); kw...)),
        (:fundmat, kw -> find_fundmat(cs_f; kw...), (θ, kw) -> refine_fundmat(θ, cs_f; kw...), kw -> find_fundmat(few_csponds(6); kw...)),
        (:essential, kw -> find_essential(ECAM, ECAM, cs_e; kw...), (θ, kw) -> refine_essential(θ, ECAM, ECAM, cs_e; kw...),
         kw -> find_essential(ECAM, ECAM, few_csponds(4); kw...)),
        (:rotation, kw -> find_rotation(FISHEYE, FISHEYE, cs_r; kw...), (θ, kw) -> refine_rotation(θ, FISHEYE, FISHEYE, cs_r; kw...),
         kw -> find_rotation(FISHEYE, FISHEYE, few_csponds(1); kw...)),
        (:circle, kw -> find_circle(pts_c; kw...), (θ, kw) -> refine_circle(θ, pts_c; kw...), kw -> find_circle(few_points(2); kw...)),
        (:ellipse, kw -> find_ellipse(pts_e; kw...), (θ, kw) -> refine_ellipse(θ, pts_e; kw...), kw -> find_ellipse(few_points(4); kw...)),
        (:absolute_pose, kw -> find_absolute_pose(PCAM, cs_p; kw...), (θ, kw) -> refine_absolute_pose(θ, PCAM, cs_p; kw...),
         kw -> find_absolute_pose(PCAM, cs_p[1:2]; kw...)),
        (:radial_homography, kw -> find_radial_homography(rh; center = c_rh, kw...), (θ, kw) -> refine_radial_homography(θ, rh; kw...),
         kw -> find_radial_homography(rh[1:4]; center = c_rh, kw...)),
        (:radial_camera, kw -> find_radial_camera(rc; center = RC_CENTER, kw...), (θ, kw) -> refine_radial_camera(θ, rc; kw...),
         kw -> find_radial_camera(rc[1:4]; center = RC_CENTER, kw...)),
        (:hand_eye, kw -> find_hand_eye(he; kw...), (θ, kw) -> refine_hand_eye(θ, he; kw...), kw -> find_hand_eye(he_parallel; kw...)),
    )
    for (seed, (class, find, refine, few)) in enumerate(classes)
        @testset "$class" begin
            r = record!(seen, Symbol(:find_, class), find((; rng = StableRNG(seed))))
            @test found_model(r)
            # a capped search: the trial budget drawn without the confidence is MaxIters (not converged), a model found
            capped = record!(seen, Symbol(:find_, class), find((; rng = StableRNG(seed), config = RansacConfig(max_trials = 2, min_trials = 2))))
            @test stop_rule(capped) in (VgtkCore.TrialBudget, VgtkCore.AdaptiveCount)
            stop_rule(capped) == VgtkCore.TrialBudget && found_model(capped) && @test retcode(capped) == ReturnCode.MaxIters
            # no model: too few data for a minimal sample
            none = record!(seen, Symbol(:find_, class), few((;)))
            @test !found_model(none) && retcode(none) == ReturnCode.Failure && !converged(none)
            # the refine from the found model: least squares, and the class's default robust strategy at a width
            θ = estimate(r)
            record!(seen, Symbol(:refine_, class), refine(θ, (; optimization_strategy = nothing)))
            record!(seen, Symbol(:refine_, class), refine(θ, (; scale = 1.0)))
        end
    end
    # the refine_* too-few-data path (the guess unrefined: InitialFailure)
    H = HomographyMat{Float64}(Tuple(SMatrix{3, 3, Float64}(I)))
    deg = record!(seen, :refine_homography, refine_homography(H, few_csponds(3); scale = 1.0))
    @test retcode(deg) == ReturnCode.InitialFailure && stop_rule(deg) == VgtkCore.NoRule

    # --- ransac, locally_optimize (seeded, fixed candidates) and a terminal LO ---------------------------------------
    p = HomographyProblem(cs_h)
    sc = MsacScoring(p; threshold = 3.0)
    rr = record!(seen, :ransac, ransac(p, sc; local_optimization_strategy = Irls{TLSShape}(), rng = StableRNG(1)))
    @test converged(rr) && stop_rule(rr) == VgtkCore.AdaptiveCount
    record!(seen, :ransac, ransac(p, sc; rng = StableRNG(1)))                           # no LO
    record!(seen, :ransac, ransac(p, sc; local_optimization_strategy = TerminalLO(Irls{TLSShape}()), rng = StableRNG(1)))
    θ = estimate(rr)
    for lo in (nothing, Irls{TLSShape}(), Triggs{WelschShape}())
        record!(seen, :locally_optimize, locally_optimize(p, θ; local_optimization_strategy = lo, scoring = sc))
    end
    record!(seen, :locally_optimize, locally_optimize(p, VgtkCore.FixedModels(1, (θ, θ)); local_optimization_strategy = nothing, scoring = sc))
    # a terminal LO on a thin incumbent (m + 1 points: no residual dof): the refit did not run, its record
    # (InitialFailure, NoRule); the result keeps the search's record
    rngt = StableRNG(3)
    Ht = @SMatrix [1.1 0.05 3.0; -0.02 0.95 -2.0; 1.0e-4 2.0e-4 1.0]
    apt(x) = (y = Ht * SVector(x[1], x[2], 1.0); SVector(y[1] / y[3], y[2] / y[3]))
    srct = [SVector(100rand(rngt), 100rand(rngt)) for _ in 1:5]; dstt = apt.(srct)
    for _ in 1:12
        push!(srct, SVector(100rand(rngt), 100rand(rngt))); push!(dstt, SVector(100rand(rngt), 100rand(rngt)))
    end
    pt = HomographyProblem(csponds(srct, dstt)); sct = MsacScoring(pt; threshold = 0.5)
    cfgt = RansacConfig(max_trials = 5000, min_trials = 3000)
    thin = record!(seen, :ransac, ransac(pt, sct; local_optimization_strategy = TerminalLO(Irls{WelschShape}(efficiency_constant = 1.5)),
                                         config = cfgt, rng = StableRNG(1)))
    @test (thin.lo_retcode, thin.lo_stop_rule) == (ReturnCode.InitialFailure, VgtkCore.NoRule)
    # the per-candidate score guard
    record!(seen, :ransac, ransac(p, sc; local_optimization_strategy = Irls{TLSShape}(), score_guard = true, rng = StableRNG(1)))

    # the adaptive count's boundary: a confidence count equal to `max_trials` was reached AT the cap (AdaptiveCount);
    # one past it was not (the budget drawn without it)
    c, reached = VgtkEstimation._adaptive_count(60, 100, 4, RansacConfig(max_trials = 10^6, min_trials = 1))
    @test reached && c > 1
    @test VgtkEstimation._adaptive_count(60, 100, 4, RansacConfig(max_trials = c, min_trials = 1)) == (c, true)
    @test VgtkEstimation._adaptive_count(60, 100, 4, RansacConfig(max_trials = c - 1, min_trials = 1)) == (c - 1, false)
    @test first(VgtkEstimation._adaptive_count(60, 100, 4, RansacConfig(max_trials = c, min_trials = 1))) ==
        VgtkEstimation._adaptive_trials(60, 100, 4, RansacConfig(max_trials = c, min_trials = 1))

    check_covered(seen, (Symbol.(:find_, first.(classes))..., Symbol.(:refine_, first.(classes))..., :ransac, :locally_optimize))
end

@testitem "termination record: every fit_* and one-step result" setup = [TerminationRecord, FitContractScenes, DivisionBoards] begin
    using Test, LinearAlgebra
    using StableRNGs
    using StaticArrays: SVector, SMatrix
    using Rotations: RotXYZ
    import VisualGeometryToolkitCore as VgtkCore
    using VisualGeometryToolkitCore: Point2, Uncertain, CovMat, RadialCamera, StudentTUniformScoring, Irls,
        HuberShape, ReturnCode, retcode, stop_rule, converged
    using VisualGeometryToolkitEstimation: fit_line, fit_radial_camera, fit_division_camera, fit_residual_mixture,
        fit_filtered_rotation_averaging, ViewGraph

    seen = Set{Symbol}()
    # the five reweighted fits: least squares (a closed form, converged) and a robust IRLS, and a degenerate input
    for (name, codim, fit) in CONTRACT_FITS
        @testset "$name" begin
            l2 = record!(seen, name, fit((; optimization_strategy = nothing)))
            @test converged(l2)
            record!(seen, name, fit((; optimization_strategy = Irls{HuberShape}(), scale = 1.0e-3)))
            capped = record!(seen, name, fit((; optimization_strategy = Irls{HuberShape}(max_iter = 1), scale = 1.0e-9)))
            retcode(capped) == ReturnCode.MaxIters && @test !converged(capped)
        end
    end
    @test stop_rule(VgtkEstimation.fit_linear(contract_linear()...; optimization_strategy = nothing)) == VgtkCore.ClosedForm
    deg = record!(seen, :fit_linear, VgtkEstimation.fit_linear(zeros(2, 3), zeros(2); optimization_strategy = nothing))
    @test retcode(deg) == ReturnCode.InitialFailure && !converged(deg)
    # a weighted solve that cannot be solved inside the loop (every TLS weight zero at a width far below the residuals):
    # SciMLBase's InternalLinearSolveFailed, not a degenerate start
    rng_l = StableRNG(9)
    ils = record!(seen, :fit_linear, VgtkEstimation.fit_linear(randn(rng_l, 8, 3), randn(rng_l, 8);
                                                                optimization_strategy = Irls{VgtkCore.TLSShape}(efficiency_constant = 3.0), scale = 1.0e-9))
    @test retcode(ils) == ReturnCode.InternalLinearSolveFailed && !converged(ils) && VgtkCore.retcode_symbol(ils) === :solve_failed
    # fit_line: the one covariance-reweighted solve (a closed form)
    pts = [Uncertain(Point2{Float64}(Float64(i), 2.0i + 1.0 + 0.01 * sin(i)), CovMat(SMatrix{2, 2, Float64}(1.0e-4, 0.0, 0.0, 1.0e-4)))
           for i in 1:20]
    rl = record!(seen, :fit_line, fit_line(pts))
    @test converged(rl) && stop_rule(rl) == VgtkCore.ClosedForm
    # fit_radial_camera: the iterated solve (closed form) and a degenerate configuration
    cam = board_camera(RotXYZ(BOARD_ANGLES[1]...), BOARD_TS[1])
    pp = board_pixels(cam)
    record!(seen, :fit_radial_camera, fit_radial_camera(pp; center = SVector(0.0, 0.0)))
    collinear = [SVector(Float64(i), 0.0) => SVector(10.0i, 0.0) for i in 1:8]
    record!(seen, :fit_radial_camera, fit_radial_camera(collinear; center = SVector(0.0, 0.0)))
    # fit_division_camera: the EM over a focal stack
    cams = [board_camera(RotXYZ(a...), t) for (a, t) in zip(BOARD_ANGLES, BOARD_TS)]
    boards, _ = noisy_board_views(StableRNG(3), cams; σ = 0.4, ε = 0.08)
    rdc = record!(seen, :fit_division_camera, fit_division_camera(Dict(enumerate(RadialCamera.(cams))), Dict(enumerate(boards));
                                                                  image_size = (4000, 3000)))
    # fit_residual_mixture: the EM on a held model's distances
    r = abs.(vcat(0.5 .* randn(StableRNG(5), 200), 40.0 .* rand(StableRNG(6), 40)))
    mix = fit_residual_mixture(r; weighting = StudentTUniformScoring(4, 40.0; codimension = 1, scale_prior = VgtkCore.JeffreysPrior(),
                                                                                label_prior = VgtkCore.FlatLabelPrior()), model_dof = 4)
    record!(seen, :fit_residual_mixture, mix)
    # fit_filtered_rotation_averaging: its fit is fit_rotation_averaging's
    Rs = Dict(k => SMatrix{3, 3, Float64, 9}(RotXYZ(0.3k, -0.2k, 0.1k^2)) for k in 1:5)
    vg = ViewGraph(Dict((i, j) => Rs[j] * Rs[i]' for i in 1:5 for j in (i + 1):5))
    record!(seen, :fit_filtered_rotation_averaging, fit_filtered_rotation_averaging(vg; scale = 1.0e-3).fit)

    check_covered(seen, (first.(CONTRACT_FITS)..., :fit_line, :fit_radial_camera, :fit_division_camera, :fit_residual_mixture,
                         :fit_filtered_rotation_averaging))
end

@testitem "termination record: the camera verbs" setup = [TerminationRecord, FitCameraFisheye] begin
    using Test
    using StaticArrays: SVector
    import VisualGeometryToolkitCore as VgtkCore
    import VisualGeometryToolkitEstimation as VgtkEstimation
    using VisualGeometryToolkitCore: CameraModel, KannalaBrandt, SplineLens, BoundedCamera, Size2, Division, estimate, converged,
        stop_rule, with_lens_coverage
    using VisualGeometryToolkitEstimation: fit_camera, refine_camera, approximate_camera, optical_axis, pspline_grid

    seen = Set{Symbol}()
    bc = BoundedCamera(FISHEYE, Size2(1000, 1000))
    # the linear fit (a closed form), its refine (the chart LM) and the two in one
    lin = record!(seen, :fit_camera, fit_camera(KannalaBrandt{4}, FISHEYE, REGION))
    @test converged(lin) && stop_rule(lin) == VgtkCore.ClosedForm
    record!(seen, :refine_camera, refine_camera(estimate(lin), FISHEYE, REGION))
    record!(seen, :refine_camera, refine_camera(estimate(lin), FISHEYE, REGION; max_iter = 1))              # the cap
    # nothing free (every parameter held on a physical K): the guess returned, no LM — a closed form
    held = record!(seen, :refine_camera, refine_camera(estimate(lin), FISHEYE, REGION; fixed = Tuple(VgtkCore.parameter_names(estimate(lin)))))
    @test VgtkCore.retcode(held) == VgtkCore.ReturnCode.Success && stop_rule(held) == VgtkCore.ClosedForm && converged(held) &&
        VgtkCore.iterations(held) == 0
    record!(seen, :approximate_camera, approximate_camera(Division{2}, FISHEYE, REGION))
    # the spline fits: the spacing search to a tolerance, the one lattice solve, and the optical axis
    rs = record!(seen, :fit_camera, fit_camera(SplineLens, bc, REGION; tolerance = 1.0e-3))
    capped = record!(seen, :fit_camera, fit_camera(SplineLens, bc, REGION; tolerance = 1.0e-12, forward = false))   # the finest spacing misses it
    gw, gh = VgtkCore.metadata(rs).family.grid_size
    record!(seen, :pspline_grid, pspline_grid(SplineLens, bc, REGION, (gw, gh)))
    record!(seen, :optical_axis, optical_axis(with_lens_coverage(GRID_BOUNDED, DISC)))
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
    @test !VgtkCore.successful_retcode(VgtkCore.retcode(capped)) && !VgtkCore.converged(capped)
    @test wr((VgtkCore.ReturnCode.MaxIters, VgtkCore.IterationCap), (VgtkCore.retcode(capped), stop_rule(capped))) ==
        (VgtkCore.retcode(capped), stop_rule(capped))
    check_covered(seen, (:fit_camera, :refine_camera, :approximate_camera, :pspline_grid, :optical_axis))
end

@testitem "termination record: the non-central spline fit" setup = [TerminationRecord, FitCameraNoncentral] begin
    using Test
    using VisualGeometryToolkitCore: NoncentralSplineModel
    using VisualGeometryToolkitEstimation: fit_camera
    seen = Set{Symbol}()
    record!(seen, :fit_camera, fit_camera(NoncentralSplineModel, SOURCE, REGION; tolerance = 1.0e-3))
    check_covered(seen, (:fit_camera,))
end

@testitem "termination record: the bundles" setup = [TerminationRecord, CameraPoseScenes, SplineCameraPoseScenes, BoardScenes, PoseScenes,
        RotationCenterScenes] begin
    using Test, LinearAlgebra
    using StaticArrays: SVector, SMatrix
    using StableRNGs
    using Rotations: AngleAxis
    import VisualGeometryToolkitCore as VgtkCore
    import VisualGeometryToolkitEstimation as VgtkEstimation
    using VisualGeometryToolkitCore: CameraModel, parameter_vector, with_parameters, Triggs, HuberShape, ReturnCode, retcode,
        converged, project
    using VisualGeometryToolkitEstimation: refine_camera_and_poses, refine_robot_world_hand_eye, RobotWorldHandEyeCalibration

    seen = Set{Symbol}()
    # the parametric bundle: least squares, the robust Triggs (and its LM's iteration cap), the board form
    cam = pinhole_camera(); Es = truth_poses(5)
    views = views_of(cam, Es, 0.6)
    θ = parameter_vector(cam)
    cam0 = with_parameters(cam, θ .* SVector(1.04, 1.01, 1.0, 1.0, 1.0) .+ SVector(0.0, 0.0, 0.0, 6.0, -5.0))
    poses0 = [perturbed_pose(E, v) for (v, E) in enumerate(Es)]
    l2 = record!(seen, :refine_camera_and_poses, refine_camera_and_poses(cam0, poses0, views; optimization_strategy = nothing))
    @test converged(l2)
    record!(seen, :refine_camera_and_poses, refine_camera_and_poses(cam0, poses0, views; optimization_strategy = Triggs{HuberShape}(), scale = 0.2))
    capped = record!(seen, :refine_camera_and_poses,
        refine_camera_and_poses(cam0, poses0, views; optimization_strategy = Triggs{HuberShape}(), scale = 0.2, max_iter = 1))
    @test retcode(capped) == ReturnCode.MaxIters && !converged(capped)
    record!(seen, :refine_camera_and_poses, refine_camera_and_poses(cam0, poses0, views; optimization_strategy = nothing, max_iter = 1))
    boards = truth_boards(); vposes = view_poses()
    views0 = Dict(v => nudged(E, v) for (v, E) in vposes)
    keyn = Dict(:A => 1, :B => 2, :C => 3)
    boards0 = Dict(b => b === :A ? E : nudged(E, 7 + keyn[b]) for (b, E) in boards)
    record!(seen, :refine_camera_and_poses, refine_camera_and_poses(cam0, views0, boards0, board_views(cam, vposes, boards); optimization_strategy = nothing))
    # rotation-centre views (RotationCenterViews): the views' rotations about one centre
    axes = [SVector(0.0, 0.0, 0.0), SVector(0.08, 0.0, 0.0), SVector(0.0, 0.08, 0.0), SVector(-0.06, 0.05, 0.03), SVector(0.05, -0.07, -0.04),
            SVector(0.0, 0.0, 0.1)]
    t = center_truth(axes)
    start = VgtkEstimation.RotationCenterViews(
        Dict(k => SMatrix{3, 3, Float64, 9}(VgtkCore.exp_so3(SVector(0.002, -0.001, 0.0015))) * R for (k, R) in t.R),
        t.c + SVector(0.01, -0.008, 0.012), t.d + SVector(-0.005, 0.004, 0.006))
    held = Tuple(VgtkCore.parameter_names(cam))
    record!(seen, :refine_camera_and_poses, refine_camera_and_poses(cam, start, center_views(cam, t); fixed = held, optimization_strategy = nothing))
    # Core's bundle metadata (the board bundles of CaliBlobs and BlobBoards): the record in the core, the scale record in it
    s_ = (; rms = 0.3, median = 0.25, p90 = 0.6)
    bm = VgtkCore.BundleMetadata([1.0, 0.5], [0.2, 0.9], [1], 10, 1.5, s_, s_; scale = 0.4, scale_source = VgtkCore.ScaleGiven)
    rb = record!(seen, :BundleMetadata, VgtkCore.FitResult(:calibration, ReturnCode.Success, VgtkCore.CostDecrease, 12, 0.3, 2, bm))
    @test VgtkCore.noise_scale(rb) === 0.4 && VgtkCore.scale_source(rb) === VgtkCore.ScaleGiven
    l2m = VgtkCore.BundleMetadata([1.0, 1.0], [0.2, 0.9], [1], 10, NaN, s_, s_; scale = NaN, scale_source = VgtkCore.ScaleNone)
    record!(seen, :BundleMetadata, VgtkCore.FitResult(:calibration, ReturnCode.MaxIters, VgtkCore.IterationCap, 50, 0.3, 2, l2m))
    @test_throws ArgumentError VgtkCore.noise_scale(VgtkCore.FitResult(:c, ReturnCode.Success, VgtkCore.CostDecrease, 1, 0.3, 2, l2m))
    # the SplineLens bundle
    truth = spline_truth(); Es3 = truth_poses(3)
    sviews = spline_views(truth, Es3, 1.1)
    record!(seen, :refine_camera_and_poses, refine_camera_and_poses(bounded(perturbed_controls(truth, 0.01)),
                                                                    [perturbed_pose(E, v) for (v, E) in enumerate(Es3)], sviews))
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
    record!(seen, :refine_robot_world_hand_eye, refine_robot_world_hand_eye(seed, camera, markers; optimization_strategy = nothing))
    record!(seen, :refine_robot_world_hand_eye, refine_robot_world_hand_eye(seed, camera, markers; optimization_strategy = Triggs{HuberShape}(), scale = 0.3))
    check_covered(seen, (:refine_camera_and_poses, :refine_robot_world_hand_eye, :BundleMetadata))
end

@testitem "termination record: fit_division_camera audits no failed leg" begin
    # A leg whose solve failed mid-loop (the IRLS step's InternalLinearSolveFailed, the owner's ruling) is neither
    # branch-audited nor resumed: the failed-leg predicate is `retcode_symbol(record...) === :solve_failed` (Core's one
    # map), so no failure code (a non-finite start, a failed solve) is audited.
    using VisualGeometryToolkitEstimation
    import VisualGeometryToolkitCore as VgtkCore
    failed(code, rule) = VgtkCore.retcode_symbol(code, rule) === :solve_failed
    @test failed(VgtkCore.ReturnCode.InternalLinearSolveFailed, VgtkCore.NoRule)
    @test failed(VgtkCore.ReturnCode.Failure, VgtkCore.NoRule)
    @test failed(VgtkCore.ReturnCode.InitialFailure, VgtkCore.NonFinite)
    @test !failed(VgtkCore.ReturnCode.Success, VgtkCore.ParameterChange)
    @test !failed(VgtkCore.ReturnCode.MaxIters, VgtkCore.IterationCap)
    @test !failed(VgtkCore.ReturnCode.Stalled, VgtkCore.IncrementSize)
    @test !failed(VgtkCore.ReturnCode.Failure, VgtkCore.TrialBudget)
    @test !failed(VgtkCore.ReturnCode.Failure, VgtkCore.SeedKept)
    # every record against the ruling written out on its own (not through retcode_symbol): a leg failed when its code
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

# The reweighted fits' termination record is public (the owner's ruling on the BlobBoards duplication audit, ask 6):
# a downstream reweighted loop (BlobBoards' per-blob IRLS) records its stops through it, an increment below its
# tolerance as `IncrementSize`, a solve that failed as SciMLBase's `InternalLinearSolveFailed`.
@testitem "irls_termination: the reweighted fits' one termination record, public" begin
    using Test
    import VisualGeometryToolkitEstimation as VgtkEstimation
    using VisualGeometryToolkitCore: ReturnCode, IncrementSize, ClosedForm, IterationCap, NoRule
    @test Base.ispublic(VgtkEstimation, :irls_termination)
    @test !isdefined(VgtkEstimation, Symbol("_irls" * "_record"))
    @test VgtkEstimation.irls_termination(true, false) === (ReturnCode.Success, IncrementSize)
    @test VgtkEstimation.irls_termination(false, true) === (ReturnCode.Success, ClosedForm)
    @test VgtkEstimation.irls_termination(false, false) === (ReturnCode.MaxIters, IterationCap)
    @test VgtkEstimation.irls_termination(true, false; solve_failed = true) === (ReturnCode.InternalLinearSolveFailed, NoRule)
end
