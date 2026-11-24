# The pose estimators on the new API (record/est_pose.jl, case for case): the RANSAC find and the LevenbergMarquardt
# refine of the absolute-pose (pixel and bearing), hand-eye, radial-homography and radial-camera problems. A problem's
# own construction keywords (`center`, `minimal`, `extra_check`, `motion_pairs`, `char_length`, `min_rel_angle`) are
# its constructor's; the extent is the find's context `p`.
using VisualGeometryToolkitCore: MsacScoring, RansacScoring, Irls, Triggs, TLSShape, WelschShape, CauchyShape, HuberShape,
    GaussianShape
using VisualGeometryToolkitEstimation: AbsolutePoseProblem, BearingPoseProblem, HandEyeProblem, RadialHomographyProblem,
    RadialCameraProblem, RansacConfig, RansacWorkspace, TerminalLO, scoring_from_toml, data_size
Base.include(@__MODULE__, joinpath(@__DIR__, "_common.jl"))

const PO_LOS = [
    "classdefault" => (;),
    "none" => (; local_optimization_strategy = nothing),
    "irls_tls" => (; local_optimization_strategy = Irls{TLSShape}()),
    "irls_welsch_ec1.5" => (; local_optimization_strategy = Irls{WelschShape}(efficiency_constant = 1.5)),
    "triggs_welsch" => (; local_optimization_strategy = Triggs{WelschShape}()),
    "terminal_irls_tls" => (; local_optimization_strategy = TerminalLO(Irls{TLSShape}())),
]

function po_find_replays!(cases, family, name, fx, find, problem, thr; full = true)
    for (lbl, kw) in (full ? PO_LOS : PO_LOS[1:2]), seed in SEEDS
        # hand-eye has no Triggs local optimization: the refusal names the problem type, which gained a parameter
        m = family == "handeye_find" && lbl == "triggs_welsch" ? message_free(FIND_MAP) : FIND_MAP
        replay!(cases, "$family/$name/lo_$lbl/seed$seed", fx, fx -> find(fx; rng = StableRNG(seed), kw...); map = m)
    end
    full || return cases
    for seed in SEEDS
        replay!(cases, "$family/$name/msac_class_threshold/seed$seed", fx,
            fx -> find(fx; scoring = MsacScoring(problem(fx); threshold = thr(fx)), rng = StableRNG(seed)); map = FIND_MAP)
        replay!(cases, "$family/$name/ransac_scoring/seed$seed", fx,
            fx -> find(fx; scoring = RansacScoring(problem(fx); threshold = 2thr(fx)), rng = StableRNG(seed)); map = FIND_MAP)
        replay!(cases, "$family/$name/student_t_uniform_7.25/seed$seed", fx,
            fx -> find(fx; scoring = scoring_from_toml(Dict("type" => "student_t_uniform", "outlier_halfwidth" => 7.25);
                                                       problem = typeof(problem(fx))), rng = StableRNG(seed)); map = FIND_MAP)
        replay!(cases, "$family/$name/capped_2_2/seed$seed", fx,
            fx -> find(fx; config = RansacConfig(max_trials = 2, min_trials = 2), rng = StableRNG(seed)); map = FIND_MAP)
        replay!(cases, "$family/$name/max_min_trials_4000_200/seed$seed", fx,
            fx -> find(fx; max_trials = 4000, min_trials = 200, rng = StableRNG(seed)); map = FIND_MAP)
        replay!(cases, "$family/$name/workspace/seed$seed", fx,
            fx -> (p = problem(fx); find(fx; workspace = RansacWorkspace(typeof(p), data_size(p)), rng = StableRNG(seed))); map = FIND_MAP)
        replay!(cases, "$family/$name/workspace_f64/seed$seed", fx,
            fx -> (p = problem(fx); find(fx; workspace = RansacWorkspace(Float64, typeof(p), data_size(p)), rng = StableRNG(seed))); map = FIND_MAP)
    end
    return cases
end

# A hand-eye problem from the old keyword set: its construction keywords to the constructor, the rest to the solve.
function _hand_eye(cs; motion_pairs = nothing, char_length = nothing, min_rel_angle = nothing, kw...)
    pkw = (; (motion_pairs === nothing ? (;) : (; motion_pairs))..., (char_length === nothing ? (;) : (; char_length))...,
            (min_rel_angle === nothing ? (;) : (; min_rel_angle))...)
    return HandEyeProblem(cs; pkw...), kw
end
function _radial_h(cs, center; extra_check = nothing, minimal = nothing, kw...)
    pkw = (; center, (extra_check === nothing ? (;) : (; extra_check))..., (minimal === nothing ? (;) : (; minimal))...)
    return RadialHomographyProblem(cs; pkw...), kw
end

function replay_est_pose(cases)
    # ---------------------------------------------------------------- the absolute pose, pixel form
    afind((cam, cs); kw...) = ransac_solve(AbsolutePoseProblem(cam, cs); kw...)
    arefine((cam, cs), θ; kw...) = lm_solve(AbsolutePoseProblem(cam, cs), θ; kw...)
    aprob((cam, cs)) = AbsolutePoseProblem(cam, cs)
    for (name, fx) in po_abspose_scenes()
        if name == "pnp_120+60"
            po_find_replays!(cases, "abspose_find", name, fx, afind, aprob, fx -> 8.0)
            refine_replays!(cases, "abspose_refine", name, fx, afind, arefine; workspace = true)
        elseif name in ("division2", "opencv5")
            for (lbl, lo) in ("irls_tls" => Irls{TLSShape}(), "irls_welsch_ec1.5" => Irls{WelschShape}(efficiency_constant = 1.5)), seed in SEEDS
                replay!(cases, "abspose_find/$name/msac5_$lbl/seed$seed", fx,
                    fx -> afind(fx; scoring = MsacScoring(AbsolutePoseProblem; threshold = 5.0), local_optimization_strategy = lo, rng = StableRNG(seed));
                    map = FIND_MAP)
            end
            refine_replays!(cases, "abspose_refine", name, fx, afind, arefine; full = false)
        else
            po_find_replays!(cases, "abspose_find", name, fx, afind, aprob, fx -> 8.0; full = false)
            name in ("units_80_pairs", "units_80_metres") && refine_replays!(cases, "abspose_refine", name, fx, afind, arefine; full = false)
        end
    end
    X, u = po_units_pnp()
    for seed in SEEDS
        replay!(cases, "abspose_find/units_80_pairs/extent/seed$seed", (PCAM, X .=> u, po_abspose_extent()),
            ((cam, cs, ext),) -> afind((cam, cs); extent = ext, rng = StableRNG(seed)); map = FIND_MAP)
        replay!(cases, "abspose_find/units_80_metres/extent_metres/seed$seed", (PCAM, [x .* po_m for x in X] .=> u, po_abspose_extent_m()),
            ((cam, cs, ext),) -> afind((cam, cs); extent = ext, rng = StableRNG(seed)); map = FIND_MAP)
    end
    replay!(cases, "abspose/refused/metres_with_plain_extent", (PCAM, [x .* po_m for x in X] .=> u, po_abspose_extent()),
        ((cam, cs, ext),) -> afind((cam, cs); extent = ext, rng = StableRNG(3)); map = REFUSAL_MAP)
    replay!(cases, "abspose/refused/plain_with_metres_extent", (PCAM, X .=> u, po_abspose_extent_m()),
        ((cam, cs, ext),) -> afind((cam, cs); extent = ext, rng = StableRNG(3)); map = REFUSAL_MAP)
    for (name, fx) in po_abspose_degenerate(), seed in SEEDS
        replay!(cases, "abspose_find/$name/seed$seed", fx, fx -> afind(fx; rng = StableRNG(seed)); map = FIND_MAP)
    end
    pfx = last(po_abspose_scenes()[1])
    refusal_replays!(cases, "abspose", pfx, afind, arefine, afind(pfx; rng = StableRNG(1)).u, AbsolutePoseProblem)
    for (name, (g, cam, cs)) in po_pixel_refine_fixtures()
        fx = (g, cam, cs)
        replay!(cases, "abspose_refine/$name/ls", fx, ((g, cam, cs),) -> arefine((cam, cs), g; optimization_strategy = nothing); map = REFINE_MAP)
        replay!(cases, "abspose_refine/$name/scale0.3", fx, ((g, cam, cs),) -> arefine((cam, cs), g; scale = 0.3); map = REFINE_MAP)
        replay!(cases, "abspose_refine/$name/scale1", fx, ((g, cam, cs),) -> arefine((cam, cs), g; scale = 1.0); map = REFINE_MAP)
        startswith(name, "pinhole_") && replay!(cases, "abspose_refine/$name/triggs_huber_width1e9", fx,
            ((g, cam, cs),) -> arefine((cam, cs), g; optimization_strategy = Triggs{HuberShape}(), width = 1.0e9); map = REFINE_MAP)
    end
    # ---------------------------------------------------------------- the absolute pose, bearing form (refine only)
    brefine(cs, θ; kw...) = lm_solve(BearingPoseProblem(cs), θ; kw...)
    for (name, (g, cs)) in po_bearing_refine_fixtures()
        fx = (g, cs)
        # the guess's coincidence refusal is now the problem's (its message names BearingPoseProblem)
        bmap = name == "origin_on_point" ? message_free(REFINE_MAP) : REFINE_MAP
        replay!(cases, "bearingpose_refine/$name/ls", fx, ((g, cs),) -> brefine(cs, g; optimization_strategy = nothing); map = bmap)
        replay!(cases, "bearingpose_refine/$name/scale1", fx, ((g, cs),) -> brefine(cs, g; scale = 1.0); map = bmap)
        if name in ("contaminated_rays", "pinhole_noisy_M")
            for (lbl, lo) in ("triggs_cauchy_ec2.5" => Triggs{CauchyShape}(efficiency_constant = 2.5), "triggs_welsch" => Triggs{WelschShape}())
                replay!(cases, "bearingpose_refine/$name/$(lbl)_scale0.3", fx,
                    ((g, cs),) -> brefine(cs, g; optimization_strategy = lo, scale = 0.3); map = REFINE_MAP)
            end
            replay!(cases, "bearingpose_refine/$name/triggs_tls_width3", fx,
                ((g, cs),) -> brefine(cs, g; optimization_strategy = Triggs{TLSShape}(), width = 3.0); map = REFINE_MAP)
        end
    end
    g, bcs = last(po_bearing_refine_fixtures()[1])
    refusal_replays!(cases, "bearingpose", bcs, nothing, brefine, g, AbsolutePoseProblem)
    # ---------------------------------------------------------------- hand-eye
    hfind(cs; kw...) = ((p, rest) = _hand_eye(cs; kw...); ransac_solve(p; rest...))
    hrefine(cs, θ; kw...) = ((p, rest) = _hand_eye(cs; kw...); lm_solve(p, θ; rest...))
    hprob(cs) = HandEyeProblem(cs; min_rel_angle = deg2rad(VgtkEstimation._dflt("robot_world_hand_eye", "min_rel_angle_deg")))
    for (name, cs) in po_hand_eye_scenes()
        physical = occursin("metres", name) || occursin("millimetres", name)
        if name == "find_scene_30"
            po_find_replays!(cases, "handeye_find", name, cs, hfind, hprob, cs -> 0.01 * hprob(cs).char_length)
            refine_replays!(cases, "handeye_refine", name, cs, hfind, hrefine; workspace = true)
        else
            po_find_replays!(cases, "handeye_find", name, cs, hfind, hprob, nothing; full = false)
            name in ("outliers_120+15", "units_40_metres") && refine_replays!(cases, "handeye_refine", name, cs, hfind, hrefine; full = false)
        end
        if name in ("allpairs_20", "find_scene_30")
            for seed in SEEDS
                replay!(cases, "handeye_find/$name/motion_pairs_all/seed$seed", cs, cs -> hfind(cs; motion_pairs = :all, rng = StableRNG(seed)); map = FIND_MAP)
            end
            θ = hfind(cs; motion_pairs = :all, rng = StableRNG(1)).u
            replay!(cases, "handeye_refine/$name/motion_pairs_all_scale1", (cs, θ), ((cs, θ),) -> hrefine(cs, θ; motion_pairs = :all, scale = 1.0);
                map = REFINE_MAP)
        end
        if name in ("units_40", "units_40_metres")
            L = physical ? 0.4 * po_m : 0.4
            for seed in SEEDS
                replay!(cases, "handeye_find/$name/char_length0.4_msac2mm/seed$seed", (cs, L),
                    ((cs, L),) -> hfind(cs; scoring = MsacScoring(HandEyeProblem; threshold = 0.002), char_length = L, rng = StableRNG(seed)); map = FIND_MAP)
                replay!(cases, "handeye_find/$name/char_length0.4/seed$seed", (cs, L), ((cs, L),) -> hfind(cs; char_length = L, rng = StableRNG(seed));
                    map = FIND_MAP)
            end
        end
    end
    slow = po_slow_rotation_set()
    replay!(cases, "handeye/refused/below_min_rel_angle", slow, cs -> hfind(cs; rng = StableRNG(1)); map = REFUSAL_MAP)
    for seed in SEEDS
        replay!(cases, "handeye_find/slow_rotation_30/min_rel_angle2deg/seed$seed", slow,
            cs -> hfind(cs; min_rel_angle = deg2rad(2), rng = StableRNG(seed)); map = FIND_MAP)
    end
    θs = hfind(slow; min_rel_angle = deg2rad(2), rng = StableRNG(1)).u
    replay!(cases, "handeye_refine/slow_rotation_30/min_rel_angle2deg_scale1", (slow, θs),
        ((cs, θ),) -> hrefine(cs, θ; min_rel_angle = deg2rad(2), scale = 1.0); map = REFINE_MAP)
    par = po_hand_eye_parallel()
    for seed in SEEDS
        replay!(cases, "handeye_find/parallel_axes_8/seed$seed", par, cs -> hfind(cs; rng = StableRNG(seed)); map = FIND_MAP)
        replay!(cases, "handeye_find/parallel_axes_8_metres/seed$seed", po_phys.(first.(par)) .=> po_phys.(last.(par)),
            cs -> hfind(cs; rng = StableRNG(seed)); map = FIND_MAP)
    end
    Xhe, hecs = po_hand_eye_contract()
    replay!(cases, "handeye_refine/contract_3/scale1", (hecs, Xhe), ((cs, X),) -> hrefine(cs, X; scale = 1.0); map = REFINE_MAP)
    replay!(cases, "handeye_refine/contract_3/ls", (hecs, Xhe), ((cs, X),) -> hrefine(cs, X; optimization_strategy = nothing); map = REFINE_MAP)
    replay!(cases, "handeye/refused/two_frames", hecs[1:2], cs -> hfind(cs; rng = StableRNG(1)); map = REFUSAL_MAP)
    replay!(cases, "handeye/refused/motion_pairs_other", hecs, cs -> hfind(cs; motion_pairs = :other, rng = StableRNG(1)); map = REFUSAL_MAP)
    A, B = po_units_hand_eye()
    replay!(cases, "handeye/refused/metres_threshold_kw", po_phys.(A) .=> po_phys.(B), cs -> hfind(cs; threshold = 2.0 * po_mm); map = REFUSAL_MAP)
    replay!(cases, "handeye/refused/metres_plain_char_length", po_phys.(A) .=> po_phys.(B), cs -> hfind(cs; char_length = 0.4); map = REFUSAL_MAP)
    replay!(cases, "handeye/refused/mixed_forms", A .=> po_phys.(B), cs -> hfind(cs); map = REFUSAL_MAP)
    hfx = last(po_hand_eye_scenes()[1])
    refusal_replays!(cases, "handeye", hfx, hfind, hrefine, hfind(hfx; rng = StableRNG(1)).u, AbsolutePoseProblem)
    # ---------------------------------------------------------------- the radial homography
    rhfind((cs, c); kw...) = ((p, rest) = _radial_h(cs, c; kw...); ransac_solve(p; rest...))
    rhrefine((cs, c), θ; kw...) = lm_solve(RadialHomographyProblem(cs; center = VgtkCore.center(θ)), θ; kw...)
    rhprob((cs, c)) = RadialHomographyProblem(cs; center = c)
    for (name, fx) in po_radial_homography_scenes()
        if name == "find_scene_70+30"
            po_find_replays!(cases, "radialh_find", name, fx, rhfind, rhprob, fx -> 3.0)
            refine_replays!(cases, "radialh_refine", name, fx, rhfind, rhrefine; workspace = true)
        else
            po_find_replays!(cases, "radialh_find", name, fx, rhfind, rhprob, nothing; full = false)
        end
        for seed in SEEDS
            if startswith(name, "exact_70")
                for (lbl, kw) in ("msac2_4000_200" => (;), "msac2_4000_200_irls_welsch_ec1.5" => (; local_optimization_strategy = Irls{WelschShape}(efficiency_constant = 1.5)))
                    replay!(cases, "radialh_find/$name/$lbl/seed$seed", fx, fx -> rhfind(fx; scoring = MsacScoring(RadialHomographyProblem; threshold = 2.0),
                        max_trials = 4000, min_trials = 200, rng = StableRNG(seed), kw...); map = FIND_MAP)
                end
            elseif name == "noisy_120+20"
                replay!(cases, "radialh_find/$name/msac2.5_4000/seed$seed", fx,
                    fx -> rhfind(fx; scoring = MsacScoring(RadialHomographyProblem; threshold = 2.5), max_trials = 4000, rng = StableRNG(seed)); map = FIND_MAP)
            elseif name == "extent_60+15"
                replay!(cases, "radialh_find/$name/msac2_extent_3000/seed$seed", (fx, VgtkCore.Rect(0.0, 0.0, 200.0, 150.0)),
                    ((fx, rect),) -> rhfind(fx; scoring = MsacScoring(RadialHomographyProblem; threshold = 2.0), extent = rect, max_trials = 3000, rng = StableRNG(seed));
                    map = FIND_MAP)
            elseif name == "four_80+25"
                replay!(cases, "radialh_find/$name/minimal_homography_msac3/seed$seed", fx,
                    fx -> rhfind(fx; minimal = :homography, scoring = MsacScoring(RadialHomographyProblem; threshold = 3.0), rng = StableRNG(seed)); map = FIND_MAP)
                replay!(cases, "radialh_find/$name/minimal_homography/seed$seed", fx, fx -> rhfind(fx; minimal = :homography, rng = StableRNG(seed)); map = FIND_MAP)
            elseif name == "clean_20"
                replay!(cases, "radialh_find/$name/extra_check/seed$seed", fx, fx -> rhfind(fx; extra_check = true, rng = StableRNG(seed)); map = FIND_MAP)
                replay!(cases, "radialh_find/$name/extra_check_false/seed$seed", fx, fx -> rhfind(fx; extra_check = false, rng = StableRNG(seed)); map = FIND_MAP)
            end
        end
        if name == "noisy_40"
            m_gt = _PO_R.radial_homography_truth(-0.25)
            θ0 = VgtkCore.RadialHomography{Float64}(VgtkCore.homography(m_gt), VgtkCore.division_coeff(m_gt) * 1.5, VgtkCore.center(m_gt))
            replay!(cases, "radialh_refine/$name/k1.5_ls", (fx, θ0), ((fx, θ),) -> rhrefine(fx, θ; optimization_strategy = nothing); map = REFINE_MAP)
            replay!(cases, "radialh_refine/$name/k1.5_scale0.3", (fx, θ0), ((fx, θ),) -> rhrefine(fx, θ; scale = 0.3); map = REFINE_MAP)
        end
    end
    four = last(po_radial_homography_scenes()[end])
    replay!(cases, "radialh/refused/minimal_homography_extra_check_false", four, fx -> rhfind(fx; minimal = :homography, extra_check = false); map = REFUSAL_MAP)
    replay!(cases, "radialh/refused/minimal_other", four, fx -> rhfind(fx; minimal = :other); map = REFUSAL_MAP)
    c = VgtkCore.Point2(500.0, 400.0)
    for seed in SEEDS
        replay!(cases, "radialh_find/too_few_4/seed$seed", (few_correspondences(4), c), fx -> rhfind(fx; rng = StableRNG(seed)); map = FIND_MAP)
        replay!(cases, "radialh_find/too_few_3_minimal_homography/seed$seed", (few_correspondences(3), c),
            fx -> rhfind(fx; minimal = :homography, rng = StableRNG(seed)); map = FIND_MAP)
    end
    H = VgtkCore.HomographyMat{Float64}(Tuple(SMatrix{3, 3, Float64}(I)))
    replay!(cases, "radialh_refine/too_few_4/scale1", (few_correspondences(4), VgtkCore.RadialHomography{Float64}(H, 0.0)),
        ((cs, θ),) -> rhrefine((cs, nothing), θ; scale = 1.0); map = REFINE_MAP)
    m2 = VgtkCore.RadialHomography{Float64}(
        VgtkCore.HomographyMat{Float64}(Tuple(SMatrix{3, 3, Float64}(4.0, -0.2, 5.0e-5, 0.3, 4.2, -4.0e-5, 260.0, 200.0, 1.0))),
        (1.0e-7, 1.0e-13), SVector(RW / 2, RH / 2))
    replay!(cases, "radialh/refused/two_coefficient_model", ([SVector(1.0 * i, 2.0 * i) => SVector(3.0 * i, 4.0 * i) for i in 1:8], m2),
        ((cs, θ),) -> rhrefine((cs, nothing), θ); map = REFUSAL_MAP)
    replay!(cases, "radialh/refused/metres_plane", [SVector(1.0 * i, 2.0 * i) .* po_m => SVector(3.0 * i, 4.0 * i) for i in 1:8],
        cs -> rhfind((cs, (600.0, 450.0))); map = REFUSAL_MAP)
    rhfx = last(po_radial_homography_scenes()[1])
    refusal_replays!(cases, "radialh", rhfx, rhfind, rhrefine, rhfind(rhfx; rng = StableRNG(1)).u, AbsolutePoseProblem)
    # ---------------------------------------------------------------- the radial camera
    rcfind((cs, c); kw...) = ransac_solve(RadialCameraProblem(cs; center = c); kw...)
    rcrefine((cs, c), θ; kw...) = lm_solve(RadialCameraProblem(cs; center = VgtkCore.center(θ)), θ; kw...)
    rcprob((cs, c)) = RadialCameraProblem(cs; center = c)
    for (k, (name, fx)) in enumerate(po_radial_camera_scenes())
        if k == 1
            po_find_replays!(cases, "radialcam_find", name, fx, rcfind, rcprob, fx -> 3.0)
            refine_replays!(cases, "radialcam_refine", name, fx, rcfind, rcrefine; workspace = true)
        else
            po_find_replays!(cases, "radialcam_find", name, fx, rcfind, rcprob, nothing; full = false)
            for seed in SEEDS
                replay!(cases, "radialcam_find/$name/msac5/seed$seed", fx,
                    fx -> rcfind(fx; scoring = MsacScoring(RadialCameraProblem; threshold = 5.0), rng = StableRNG(seed)); map = FIND_MAP)
            end
            k == 2 && refine_replays!(cases, "radialcam_refine", name, fx, rcfind, rcrefine; full = false)
        end
    end
    for seed in SEEDS
        replay!(cases, "radialcam_find/find_scene_81+20/extent_board/seed$seed", (last(po_radial_camera_scenes()[1]), VgtkCore.Rect(-100.0, -100.0, 200.0, 200.0)),
            ((fx, rect),) -> rcfind(fx; extent = rect, rng = StableRNG(seed)); map = FIND_MAP)
        replay!(cases, "radialcam_find/too_few_4/seed$seed", (few_correspondences(4), c), fx -> rcfind(fx; rng = StableRNG(seed)); map = FIND_MAP)
    end
    replay!(cases, "radialcam_refine/too_few_4/scale1", (few_correspondences(4), VgtkCore.RadialCamera(H, c)),
        ((cs, θ),) -> rcrefine((cs, nothing), θ; scale = 1.0); map = REFINE_MAP)
    replay!(cases, "radialcam_refine/too_few_4/ls", (few_correspondences(4), VgtkCore.RadialCamera(H, c)),
        ((cs, θ),) -> rcrefine((cs, nothing), θ; optimization_strategy = nothing); map = REFINE_MAP)
    rcfx = last(po_radial_camera_scenes()[1])
    refusal_replays!(cases, "radialcam", rcfx, rcfind, rcrefine, rcfind(rcfx; rng = StableRNG(1)).u, AbsolutePoseProblem)
    return cases
end
