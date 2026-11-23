# The old pose estimators at 3d8a06da: find_absolute_pose / refine_absolute_pose (pixel form with camera and extent,
# plain and physical; the bearing form of refine_absolute_pose), find_hand_eye / refine_hand_eye,
# find_radial_homography / refine_radial_homography and find_radial_camera / refine_radial_camera.
using VisualGeometryToolkitCore: estimate, MsacScoring, RansacScoring, Irls, Triggs, TLSShape, WelschShape, CauchyShape,
    HuberShape, GaussianShape
using VisualGeometryToolkitEstimation: find_absolute_pose, refine_absolute_pose, find_hand_eye, refine_hand_eye,
    find_radial_homography, refine_radial_homography, find_radial_camera, refine_radial_camera, AbsolutePoseProblem,
    HandEyeProblem, RadialHomographyProblem, RadialCameraProblem, RansacConfig, RansacWorkspace, TerminalLO, scoring_from_toml,
    data_size

const PO_LOS = [
    "classdefault" => (;),
    "none" => (; local_optimization_strategy = nothing),
    "irls_tls" => (; local_optimization_strategy = Irls{TLSShape}()),
    "irls_welsch_ec1.5" => (; local_optimization_strategy = Irls{WelschShape}(efficiency_constant = 1.5)),
    "triggs_welsch" => (; local_optimization_strategy = Triggs{WelschShape}()),
    "terminal_irls_tls" => (; local_optimization_strategy = TerminalLO(Irls{TLSShape}())),
]

# Every find_* setting of one class on one scene (`full`: the whole matrix, else the default and no LO).
# `problem(fx)` is the class's problem, for an explicit scorer and a workspace; `thr` the class threshold in the
# problem's residual unit.
function po_find_cases!(cases, family, name, fx, find, problem, thr; full = true)
    for (lbl, kw) in (full ? PO_LOS : PO_LOS[1:2]), seed in SEEDS
        case!(cases, "$family/$name/lo_$lbl/seed$seed", fx, fx -> find(fx; rng = StableRNG(seed), kw...))
    end
    full || return cases
    for seed in SEEDS
        case!(cases, "$family/$name/msac_class_threshold/seed$seed", fx,
            fx -> find(fx; scoring = MsacScoring(problem(fx); threshold = thr(fx)), rng = StableRNG(seed)))
        case!(cases, "$family/$name/ransac_scoring/seed$seed", fx,
            fx -> find(fx; scoring = RansacScoring(problem(fx); threshold = 2thr(fx)), rng = StableRNG(seed)))
        case!(cases, "$family/$name/student_t_uniform_7.25/seed$seed", fx,
            fx -> find(fx; scoring = scoring_from_toml(Dict("type" => "student_t_uniform", "outlier_halfwidth" => 7.25);
                                                       problem = typeof(problem(fx))), rng = StableRNG(seed)))
        case!(cases, "$family/$name/capped_2_2/seed$seed", fx,
            fx -> find(fx; config = RansacConfig(max_trials = 2, min_trials = 2), rng = StableRNG(seed)))
        case!(cases, "$family/$name/max_min_trials_4000_200/seed$seed", fx,
            fx -> find(fx; max_trials = 4000, min_trials = 200, rng = StableRNG(seed)))
        case!(cases, "$family/$name/workspace/seed$seed", fx,
            fx -> (p = problem(fx); find(fx; workspace = RansacWorkspace(typeof(p), data_size(p)), rng = StableRNG(seed))))
        case!(cases, "$family/$name/workspace_f64/seed$seed", fx,
            fx -> (p = problem(fx); find(fx; workspace = RansacWorkspace(Float64, typeof(p), data_size(p)), rng = StableRNG(seed))))
    end
    return cases
end

# The refine_* settings from each seed's find (its model and its search's scale), and from the raw incumbent.
function po_refine_cases!(cases, family, name, fx, find, refine; full = true, problem = nothing)
    for seed in SEEDS
        r = find(fx; rng = StableRNG(seed))
        θ = estimate(r)
        θ === nothing && continue
        case!(cases, "$family/$name/scale_upstream/seed$seed", (fx, θ, r), ((fx, θ, r),) -> refine(fx, θ; scale = r))
        case!(cases, "$family/$name/ls/seed$seed", (fx, θ), ((fx, θ),) -> refine(fx, θ; optimization_strategy = nothing))
        case!(cases, "$family/$name/scale1/seed$seed", (fx, θ), ((fx, θ),) -> refine(fx, θ; scale = 1.0))
        full || continue
        case!(cases, "$family/$name/width2/seed$seed", (fx, θ), ((fx, θ),) -> refine(fx, θ; width = 2.0))
        case!(cases, "$family/$name/triggs_cauchy_ec2.5_upstream/seed$seed", (fx, θ, r),
            ((fx, θ, r),) -> refine(fx, θ; optimization_strategy = Triggs{CauchyShape}(efficiency_constant = 2.5), scale = r))
        case!(cases, "$family/$name/triggs_tls_width3/seed$seed", (fx, θ),
            ((fx, θ),) -> refine(fx, θ; optimization_strategy = Triggs{TLSShape}(), width = 3.0))
        case!(cases, "$family/$name/triggs_huber_scale1/seed$seed", (fx, θ),
            ((fx, θ),) -> refine(fx, θ; optimization_strategy = Triggs{HuberShape}(), scale = 1.0))
        problem === nothing || case!(cases, "$family/$name/workspace_scale1/seed$seed", (fx, θ),
            ((fx, θ),) -> (p = problem(fx); refine(fx, θ; scale = 1.0, workspace = RansacWorkspace(typeof(p), data_size(p)))))
        raw_r = find(fx; local_optimization_strategy = nothing, rng = StableRNG(seed))
        raw = estimate(raw_r)
        raw === nothing && continue
        case!(cases, "$family/$name/from_raw_upstream/seed$seed", (fx, raw, raw_r), ((fx, θ, r),) -> refine(fx, θ; scale = r))
        case!(cases, "$family/$name/from_raw_cauchy_ec2.5/seed$seed", (fx, raw, raw_r),
            ((fx, θ, r),) -> refine(fx, θ; optimization_strategy = Triggs{CauchyShape}(efficiency_constant = 2.5), scale = r))
    end
    return cases
end

# The refusals every find_* and refine_* shares (core_defaults, refine_contract, ransac_no_model).
function po_refusal_cases!(cases, family, fx, find, refine, θ)
    if find !== nothing
        case!(cases, "$family/refused/find_threshold_kw", fx, fx -> find(fx; threshold = 3.0))
        case!(cases, "$family/refused/find_outlier_halfwidth_kw", fx, fx -> find(fx; outlier_halfwidth = 20.0))
        case!(cases, "$family/refused/find_retired_local_optimization", fx, fx -> find(fx; local_optimization = Irls{TLSShape}()))
        case!(cases, "$family/refused/find_unknown_kw", fx, fx -> find(fx; no_such_keyword = 1))
    end
    for (lbl, kw) in (
            "no_scale" => (;), "scale_and_width" => (; scale = 1.0, width = 1.0), "negative_scale" => (; scale = -1.0),
            "nan_scale" => (; scale = NaN), "zero_width" => (; width = 0.0), "ls_with_scale" => (; optimization_strategy = nothing, scale = 1.0),
            "tls_scale_no_constant" => (; optimization_strategy = Triggs{TLSShape}(), scale = 1.0),
            "constant_with_width" => (; optimization_strategy = Triggs{WelschShape}(efficiency_constant = 2.5), width = 1.0),
            "irls_huber" => (; optimization_strategy = Irls{HuberShape}(), scale = 1.0),
            "irls_gaussian" => (; optimization_strategy = Irls{GaussianShape}(), scale = 1.0),
            "terminal_lo" => (; optimization_strategy = TerminalLO(Irls{TLSShape}()), scale = 1.0),
            "scoring_kw" => (; scale = 1.0, scoring = MsacScoring(AbsolutePoseProblem; threshold = 3.0)),
            "rng_kw" => (; scale = 1.0, rng = StableRNG(1)), "max_trials_kw" => (; scale = 1.0, max_trials = 10),
            "extent_kw" => (; scale = 1.0, extent = nothing), "unknown_kw" => (; scale = 1.0, nonsense = 1))
        case!(cases, "$family/refused/refine_$lbl", (fx, θ), ((fx, θ),) -> refine(fx, θ; kw...))
    end
    return cases
end

function record_est_pose(cases)
    # ---------------------------------------------------------------- the absolute pose, pixel form
    afind((cam, cs); kw...) = find_absolute_pose(cam, cs; kw...)
    arefine((cam, cs), θ; kw...) = refine_absolute_pose(θ, cam, cs; kw...)
    aprob((cam, cs)) = AbsolutePoseProblem(cam, cs)
    for (name, fx) in po_abspose_scenes()
        if name == "pnp_120+60"
            po_find_cases!(cases, "abspose_find", name, fx, afind, aprob, fx -> 8.0)
            po_refine_cases!(cases, "abspose_refine", name, fx, afind, arefine; problem = aprob)
        elseif name in ("division2", "opencv5")   # pnp_lens_tests: MSAC at 5 px, two strategies
            for (lbl, lo) in ("irls_tls" => Irls{TLSShape}(), "irls_welsch_ec1.5" => Irls{WelschShape}(efficiency_constant = 1.5)), seed in SEEDS
                case!(cases, "abspose_find/$name/msac5_$lbl/seed$seed", fx,
                    fx -> afind(fx; scoring = MsacScoring(AbsolutePoseProblem; threshold = 5.0), local_optimization_strategy = lo, rng = StableRNG(seed)))
            end
            po_refine_cases!(cases, "abspose_refine", name, fx, afind, arefine; full = false)
        else
            po_find_cases!(cases, "abspose_find", name, fx, afind, aprob, fx -> 8.0; full = false)
            name == "units_80_pairs" && po_refine_cases!(cases, "abspose_refine", name, fx, afind, arefine; full = false)
            name == "units_80_metres" && po_refine_cases!(cases, "abspose_refine", name, fx, afind,
                (fx, θ; kw...) -> arefine(fx, θ; kw...); full = false)
        end
    end
    X, u = po_units_pnp()
    for seed in SEEDS   # an extent: plain with plain, physical with physical
        case!(cases, "abspose_find/units_80_pairs/extent/seed$seed", (PO_PCAM, X .=> u, po_abspose_extent()),
            ((cam, cs, ext),) -> find_absolute_pose(cam, cs; extent = ext, rng = StableRNG(seed)))
        case!(cases, "abspose_find/units_80_metres/extent_metres/seed$seed", (PO_PCAM, [x .* po_m for x in X] .=> u, po_abspose_extent_m()),
            ((cam, cs, ext),) -> find_absolute_pose(cam, cs; extent = ext, rng = StableRNG(seed)))
    end
    case!(cases, "abspose/refused/metres_with_plain_extent", (PO_PCAM, [x .* po_m for x in X] .=> u, po_abspose_extent()),
        ((cam, cs, ext),) -> find_absolute_pose(cam, cs; extent = ext, rng = StableRNG(3)))
    case!(cases, "abspose/refused/plain_with_metres_extent", (PO_PCAM, X .=> u, po_abspose_extent_m()),
        ((cam, cs, ext),) -> find_absolute_pose(cam, cs; extent = ext, rng = StableRNG(3)))
    for (name, fx) in po_abspose_degenerate(), seed in SEEDS
        case!(cases, "abspose_find/$name/seed$seed", fx, fx -> afind(fx; rng = StableRNG(seed)))
    end
    pfx = last(po_abspose_scenes()[1])
    po_refusal_cases!(cases, "abspose", pfx, afind, arefine, estimate(afind(pfx; rng = StableRNG(1))))
    # refine_absolute_pose_tests' pixel fixtures: least squares, the default at the pixels' scale, a Cauchy
    for (name, (g, cam, cs)) in po_pixel_refine_fixtures()
        fx = (g, cam, cs)
        case!(cases, "abspose_refine/$name/ls", fx, ((g, cam, cs),) -> refine_absolute_pose(g, cam, cs; optimization_strategy = nothing))
        case!(cases, "abspose_refine/$name/scale0.3", fx, ((g, cam, cs),) -> refine_absolute_pose(g, cam, cs; scale = 0.3))
        case!(cases, "abspose_refine/$name/scale1", fx, ((g, cam, cs),) -> refine_absolute_pose(g, cam, cs; scale = 1.0))
        startswith(name, "pinhole_") && case!(cases, "abspose_refine/$name/triggs_huber_width1e9", fx,
            ((g, cam, cs),) -> refine_absolute_pose(g, cam, cs; optimization_strategy = Triggs{HuberShape}(), width = 1.0e9))
    end

    # ---------------------------------------------------------------- the absolute pose, bearing form (refine only)
    brefine(cs, θ; kw...) = refine_absolute_pose(θ, cs; kw...)
    for (name, (g, cs)) in po_bearing_refine_fixtures()
        fx = (g, cs)
        case!(cases, "bearingpose_refine/$name/ls", fx, ((g, cs),) -> refine_absolute_pose(g, cs; optimization_strategy = nothing))
        case!(cases, "bearingpose_refine/$name/scale1", fx, ((g, cs),) -> refine_absolute_pose(g, cs; scale = 1.0))
        if name in ("contaminated_rays", "pinhole_noisy_M")
            for (lbl, lo) in ("triggs_cauchy_ec2.5" => Triggs{CauchyShape}(efficiency_constant = 2.5), "triggs_welsch" => Triggs{WelschShape}())
                case!(cases, "bearingpose_refine/$name/$(lbl)_scale0.3", fx,
                    ((g, cs),) -> refine_absolute_pose(g, cs; optimization_strategy = lo, scale = 0.3))
            end
            case!(cases, "bearingpose_refine/$name/triggs_tls_width3", fx,
                ((g, cs),) -> refine_absolute_pose(g, cs; optimization_strategy = Triggs{TLSShape}(), width = 3.0))
        end
    end
    g, bcs = last(po_bearing_refine_fixtures()[1])
    po_refusal_cases!(cases, "bearingpose", bcs, nothing, brefine, g)

    # ---------------------------------------------------------------- hand-eye
    hfind(cs; kw...) = find_hand_eye(cs; kw...)
    hrefine(cs, θ; kw...) = refine_hand_eye(θ, cs; kw...)
    hprob(cs) = HandEyeProblem(cs; min_rel_angle = deg2rad(VgtkEstimation._dflt("robot_world_hand_eye", "min_rel_angle_deg")))
    for (name, cs) in po_hand_eye_scenes()
        physical = occursin("metres", name) || occursin("millimetres", name)
        if name == "find_scene_30"
            po_find_cases!(cases, "handeye_find", name, cs, hfind, hprob, cs -> 0.01 * hprob(cs).char_length)
            po_refine_cases!(cases, "handeye_refine", name, cs, hfind, hrefine; problem = hprob)
        else
            po_find_cases!(cases, "handeye_find", name, cs, hfind, hprob, nothing; full = false)
            name in ("outliers_120+15", "units_40_metres") && po_refine_cases!(cases, "handeye_refine", name, cs, hfind, hrefine; full = false)
        end
        if name in ("allpairs_20", "find_scene_30")
            for seed in SEEDS
                case!(cases, "handeye_find/$name/motion_pairs_all/seed$seed", cs, cs -> find_hand_eye(cs; motion_pairs = :all, rng = StableRNG(seed)))
            end
            θ = estimate(find_hand_eye(cs; motion_pairs = :all, rng = StableRNG(1)))
            case!(cases, "handeye_refine/$name/motion_pairs_all_scale1", (cs, θ), ((cs, θ),) -> refine_hand_eye(θ, cs; motion_pairs = :all, scale = 1.0))
        end
        if name in ("units_40", "units_40_metres")   # a caller's char_length and a scorer in metres
            L = physical ? 0.4 * po_m : 0.4
            for seed in SEEDS
                case!(cases, "handeye_find/$name/char_length0.4_msac2mm/seed$seed", (cs, L),
                    ((cs, L),) -> find_hand_eye(cs; scoring = MsacScoring(HandEyeProblem; threshold = 0.002), char_length = L, rng = StableRNG(seed)))
                case!(cases, "handeye_find/$name/char_length0.4/seed$seed", (cs, L), ((cs, L),) -> find_hand_eye(cs; char_length = L, rng = StableRNG(seed)))
            end
        end
    end
    slow = po_slow_rotation_set()
    case!(cases, "handeye/refused/below_min_rel_angle", slow, cs -> find_hand_eye(cs; rng = StableRNG(1)))
    for seed in SEEDS
        case!(cases, "handeye_find/slow_rotation_30/min_rel_angle2deg/seed$seed", slow,
            cs -> find_hand_eye(cs; min_rel_angle = deg2rad(2), rng = StableRNG(seed)))
    end
    θs = estimate(find_hand_eye(slow; min_rel_angle = deg2rad(2), rng = StableRNG(1)))
    case!(cases, "handeye_refine/slow_rotation_30/min_rel_angle2deg_scale1", (slow, θs),
        ((cs, θ),) -> refine_hand_eye(θ, cs; min_rel_angle = deg2rad(2), scale = 1.0))
    par = po_hand_eye_parallel()
    for seed in SEEDS
        case!(cases, "handeye_find/parallel_axes_8/seed$seed", par, cs -> find_hand_eye(cs; rng = StableRNG(seed)))
        case!(cases, "handeye_find/parallel_axes_8_metres/seed$seed", po_phys.(first.(par)) .=> po_phys.(last.(par)),
            cs -> find_hand_eye(cs; rng = StableRNG(seed)))
    end
    Xhe, hecs = po_hand_eye_contract()
    case!(cases, "handeye_refine/contract_3/scale1", (hecs, Xhe), ((cs, X),) -> refine_hand_eye(X, cs; scale = 1.0))
    case!(cases, "handeye_refine/contract_3/ls", (hecs, Xhe), ((cs, X),) -> refine_hand_eye(X, cs; optimization_strategy = nothing))
    case!(cases, "handeye/refused/two_frames", hecs[1:2], cs -> find_hand_eye(cs; rng = StableRNG(1)))
    case!(cases, "handeye/refused/motion_pairs_other", hecs, cs -> find_hand_eye(cs; motion_pairs = :other, rng = StableRNG(1)))
    A, B = po_units_hand_eye()
    case!(cases, "handeye/refused/metres_threshold_kw", po_phys.(A) .=> po_phys.(B), cs -> find_hand_eye(cs; threshold = 2.0 * po_mm))
    case!(cases, "handeye/refused/metres_plain_char_length", po_phys.(A) .=> po_phys.(B), cs -> find_hand_eye(cs; char_length = 0.4))
    case!(cases, "handeye/refused/mixed_forms", A .=> po_phys.(B), cs -> find_hand_eye(cs))
    hfx = last(po_hand_eye_scenes()[1])
    po_refusal_cases!(cases, "handeye", hfx, hfind, hrefine, estimate(hfind(hfx; rng = StableRNG(1))))

    # ---------------------------------------------------------------- the radial homography
    rhfind((cs, c); kw...) = find_radial_homography(cs; center = c, kw...)
    rhrefine((cs, c), θ; kw...) = refine_radial_homography(θ, cs; kw...)
    rhprob((cs, c)) = RadialHomographyProblem(cs; center = c)
    for (name, fx) in po_radial_homography_scenes()
        if name == "find_scene_70+30"
            po_find_cases!(cases, "radialh_find", name, fx, rhfind, rhprob, fx -> 3.0)
            po_refine_cases!(cases, "radialh_refine", name, fx, rhfind, rhrefine; problem = rhprob)
        else
            po_find_cases!(cases, "radialh_find", name, fx, rhfind, rhprob, nothing; full = false)
        end
        for seed in SEEDS   # the suite's settings on its own draws
            if startswith(name, "exact_70")
                for (lbl, kw) in ("msac2_4000_200" => (;), "msac2_4000_200_irls_welsch_ec1.5" => (; local_optimization_strategy = Irls{WelschShape}(efficiency_constant = 1.5)))
                    case!(cases, "radialh_find/$name/$lbl/seed$seed", fx, fx -> rhfind(fx; scoring = MsacScoring(RadialHomographyProblem; threshold = 2.0),
                        max_trials = 4000, min_trials = 200, rng = StableRNG(seed), kw...))
                end
            elseif name == "noisy_120+20"
                case!(cases, "radialh_find/$name/msac2.5_4000/seed$seed", fx,
                    fx -> rhfind(fx; scoring = MsacScoring(RadialHomographyProblem; threshold = 2.5), max_trials = 4000, rng = StableRNG(seed)))
            elseif name == "extent_60+15"
                case!(cases, "radialh_find/$name/msac2_extent_3000/seed$seed", (fx, VgtkCore.Rect(0.0, 0.0, 200.0, 150.0)),
                    ((fx, rect),) -> rhfind(fx; scoring = MsacScoring(RadialHomographyProblem; threshold = 2.0), extent = rect, max_trials = 3000, rng = StableRNG(seed)))
            elseif name == "four_80+25"
                case!(cases, "radialh_find/$name/minimal_homography_msac3/seed$seed", fx,
                    fx -> rhfind(fx; minimal = :homography, scoring = MsacScoring(RadialHomographyProblem; threshold = 3.0), rng = StableRNG(seed)))
                case!(cases, "radialh_find/$name/minimal_homography/seed$seed", fx, fx -> rhfind(fx; minimal = :homography, rng = StableRNG(seed)))
            elseif name == "clean_20"
                case!(cases, "radialh_find/$name/extra_check/seed$seed", fx, fx -> rhfind(fx; extra_check = true, rng = StableRNG(seed)))
                case!(cases, "radialh_find/$name/extra_check_false/seed$seed", fx, fx -> rhfind(fx; extra_check = false, rng = StableRNG(seed)))
            end
        end
        if name == "noisy_40"   # radial_homography_tests: a seed with 1.5 × the true k
            m_gt = _PO_R.radial_homography_truth(-0.25)
            θ0 = VgtkCore.RadialHomography{Float64}(VgtkCore.homography(m_gt), VgtkCore.division_coeff(m_gt) * 1.5, VgtkCore.center(m_gt))
            case!(cases, "radialh_refine/$name/k1.5_ls", (fx, θ0), ((fx, θ),) -> rhrefine(fx, θ; optimization_strategy = nothing))
            case!(cases, "radialh_refine/$name/k1.5_scale0.3", (fx, θ0), ((fx, θ),) -> rhrefine(fx, θ; scale = 0.3))
        end
    end
    four = last(po_radial_homography_scenes()[end])
    case!(cases, "radialh/refused/minimal_homography_extra_check_false", four, fx -> rhfind(fx; minimal = :homography, extra_check = false))
    case!(cases, "radialh/refused/minimal_other", four, fx -> rhfind(fx; minimal = :other))
    c = VgtkCore.Point2(500.0, 400.0)
    for seed in SEEDS
        case!(cases, "radialh_find/too_few_4/seed$seed", (po_few_correspondences(4), c), fx -> rhfind(fx; rng = StableRNG(seed)))
        case!(cases, "radialh_find/too_few_3_minimal_homography/seed$seed", (po_few_correspondences(3), c),
            fx -> rhfind(fx; minimal = :homography, rng = StableRNG(seed)))
    end
    H = VgtkCore.HomographyMat{Float64}(Tuple(SMatrix{3, 3, Float64}(I)))
    case!(cases, "radialh_refine/too_few_4/scale1", (po_few_correspondences(4), VgtkCore.RadialHomography{Float64}(H, 0.0)),
        ((cs, θ),) -> refine_radial_homography(θ, cs; scale = 1.0))
    m2 = VgtkCore.RadialHomography{Float64}(
        VgtkCore.HomographyMat{Float64}(Tuple(SMatrix{3, 3, Float64}(4.0, -0.2, 5.0e-5, 0.3, 4.2, -4.0e-5, 260.0, 200.0, 1.0))),
        (1.0e-7, 1.0e-13), SVector(PO_RW / 2, PO_RH / 2))
    case!(cases, "radialh/refused/two_coefficient_model", ([SVector(1.0 * i, 2.0 * i) => SVector(3.0 * i, 4.0 * i) for i in 1:8], m2),
        ((cs, θ),) -> refine_radial_homography(θ, cs))
    rh, _ = po_radial_homography_scenes()[1]
    case!(cases, "radialh/refused/metres_plane", [SVector(1.0 * i, 2.0 * i) .* po_m => SVector(3.0 * i, 4.0 * i) for i in 1:8],
        cs -> find_radial_homography(cs; center = (600.0, 450.0)))
    rhfx = last(po_radial_homography_scenes()[1])
    po_refusal_cases!(cases, "radialh", rhfx, rhfind, rhrefine, estimate(rhfind(rhfx; rng = StableRNG(1))))

    # ---------------------------------------------------------------- the radial camera
    rcfind((cs, c); kw...) = find_radial_camera(cs; center = c, kw...)
    rcrefine((cs, c), θ; kw...) = refine_radial_camera(θ, cs; kw...)
    rcprob((cs, c)) = RadialCameraProblem(cs; center = c)
    for (k, (name, fx)) in enumerate(po_radial_camera_scenes())
        if k == 1
            po_find_cases!(cases, "radialcam_find", name, fx, rcfind, rcprob, fx -> 3.0)
            po_refine_cases!(cases, "radialcam_refine", name, fx, rcfind, rcrefine; problem = rcprob)
        else
            po_find_cases!(cases, "radialcam_find", name, fx, rcfind, rcprob, nothing; full = false)
            for seed in SEEDS
                case!(cases, "radialcam_find/$name/msac5/seed$seed", fx,
                    fx -> rcfind(fx; scoring = MsacScoring(RadialCameraProblem; threshold = 5.0), rng = StableRNG(seed)))
            end
            k == 2 && po_refine_cases!(cases, "radialcam_refine", name, fx, rcfind, rcrefine; full = false)
        end
    end
    for seed in SEEDS   # an extent over the board's plane
        case!(cases, "radialcam_find/find_scene_81+20/extent_board/seed$seed", (last(po_radial_camera_scenes()[1]), VgtkCore.Rect(-100.0, -100.0, 200.0, 200.0)),
            ((fx, rect),) -> rcfind(fx; extent = rect, rng = StableRNG(seed)))
        case!(cases, "radialcam_find/too_few_4/seed$seed", (po_few_correspondences(4), c), fx -> rcfind(fx; rng = StableRNG(seed)))
    end
    case!(cases, "radialcam_refine/too_few_4/scale1", (po_few_correspondences(4), VgtkCore.RadialCamera(H, c)),
        ((cs, θ),) -> refine_radial_camera(θ, cs; scale = 1.0))
    case!(cases, "radialcam_refine/too_few_4/ls", (po_few_correspondences(4), VgtkCore.RadialCamera(H, c)),
        ((cs, θ),) -> refine_radial_camera(θ, cs; optimization_strategy = nothing))
    rcfx = last(po_radial_camera_scenes()[1])
    po_refusal_cases!(cases, "radialcam", rcfx, rcfind, rcrefine, estimate(rcfind(rcfx; rng = StableRNG(1))))
    return cases
end
