# The old PoseLib estimators at 3d8a06da: the four truncated-loss refiners (refine_relpose in its camera-aware pixel
# and bearing forms, refine_fundamental, refine_homography, refine_absolute_pose in its plain, physical and labeled
# forms), and PoseLib's PoseLibSolver backends of Core's kernels: essential_dlt (also the backend-less form, which
# PoseLib defines), essential_5pt, homography_4pt, fundamental_7pt, p3p, each as the kernel and as
# solve(<Solver>(PoseLibSolver()), cs, idx).
# The refiners are called qualified, `PoseLib.refine_*`: VisualGeometryToolkitEstimation exports a `refine_homography` and
# a `refine_absolute_pose` of its own.
import PoseLib
using PoseLib: LeZach
using VisualGeometryToolkitCore: Irls, Triggs, TLSShape, HuberShape, CauchyShape, GaussianShape, WelschShape,
    EfficiencyConstant, LossWidth, efficiency_constant, PoseLibSolver, Essential5pt, Homography4pt, Fundamental7pt, P3P

const POSELIB_STRATEGIES = (tls = Irls{TLSShape}(), huber = Irls{HuberShape}(), cauchy = Irls{CauchyShape}(),
                            lezach = LeZach(), ls = nothing)
const POSELIB_FAMILY = (relpose_cam = "relpose_pixels_refine", relpose_bearing = "relpose_bearings_refine", fundamental = "fundamental_refine",
                        homography = "homography_refine", absolute_pose = "absolute_pose_refine")

# test_contract.jl's `_baseline_calls`: (refiner family, the call on a contract scene and keywords)
const POSELIB_CALLS = (
    relpose_cam = (d, kw) -> PoseLib.refine_relpose(d.seed, d.K1, d.K2, d.rel; max_iter = 25, kw...),
    relpose_bearing = (d, kw) -> PoseLib.refine_relpose(d.seed, d.bear; max_iter = 25, kw...),
    fundamental = (d, kw) -> PoseLib.refine_fundamental(d.F0, d.rel; max_iter = 25, kw...),
    homography = (d, kw) -> PoseLib.refine_homography(d.H0, d.hom; max_iter = 25, kw...),
    absolute_pose = (d, kw) -> PoseLib.refine_absolute_pose(d.seed, d.K1, d.abs_cs; max_iter = 25, kw...),
)

function _record_poselib_contract!(cases)
    for (scene, d) in poselib_contract_scenes()
        # the suite's draw is the fixture `contract`; a noise draw `contract_noise` at its seed
        fx, sfx = scene == "test" ? ("contract", "") : ("contract_noise", "/" * scene)
        # every refiner × strategy at the suite's width (40 px for the absolute pose), least squares widthless
        for name in keys(POSELIB_CALLS), l in keys(POSELIB_STRATEGIES)
            call = POSELIB_CALLS[name]; lo = POSELIB_STRATEGIES[l]
            w = name === :absolute_pose ? 40.0 : 2.0
            kw = lo === nothing ? (; optimization_strategy = lo) : (; optimization_strategy = lo, width = w)
            case!(cases, "$(POSELIB_FAMILY[name])/$fx/$(l)_width$(lo === nothing ? "none" : w)$sfx", d, d -> call(d, kw))
            # a scale σ: the shape's efficiency constant, then the strategy's own (3.0)
            if lo isa Irls{<:Union{HuberShape, CauchyShape}}
                S = typeof(lo).parameters[1]
                case!(cases, "$(POSELIB_FAMILY[name])/$fx/$(l)_scale0.75$sfx", d,
                    d -> call(d, (; optimization_strategy = lo, scale = 0.75)))
                case!(cases, "$(POSELIB_FAMILY[name])/$fx/$(l)_ec3_scale0.75$sfx", d,
                    d -> call(d, (; optimization_strategy = Irls{S}(25, EfficiencyConstant(3.0)), scale = 0.75)))
            end
        end
    end
    d = last(first(poselib_contract_scenes()))   # the suite's own draw
    H = (d, kw) -> POSELIB_CALLS.homography(d, kw)
    # the result record: the budget used up; an upstream result's scale handed on
    case!(cases, "homography_refine/contract/ls_maxiter1", d,
        d -> PoseLib.refine_homography(d.H0, d.hom; max_iter = 1, optimization_strategy = nothing))
    case!(cases, "homography_refine/contract/huber_scale_upstream0.5", d,
        d -> H(d, (; optimization_strategy = Irls{HuberShape}(), scale = H(d, (; optimization_strategy = Irls{HuberShape}(), scale = 0.5)))))
    # the physical absolute pose (metre points, metre seed)
    for (l, kw) in ("tls_width40.0" => (; optimization_strategy = Irls{TLSShape}(), width = 40.0), "ls_widthnone" => (; optimization_strategy = nothing))
        case!(cases, "absolute_pose_refine/contract_metres/$l", d,
            d -> PoseLib.refine_absolute_pose(d.seedm, d.K1, d.abs_cs_m; max_iter = 25, kw...))
    end
    # below the minimal sample: the seed back, InitialFailure
    case!(cases, "relpose_pixels_refine/contract_first4/ls", d, d -> PoseLib.refine_relpose(d.seed, d.K1, d.K2, d.rel[1:4]; max_iter = 25, optimization_strategy = nothing))
    case!(cases, "relpose_bearings_refine/contract_first4/ls", d, d -> PoseLib.refine_relpose(d.seed, d.bear[1:4]; max_iter = 25, optimization_strategy = nothing))
    case!(cases, "fundamental_refine/contract_first6/tls_width2.0", d, d -> PoseLib.refine_fundamental(d.F0, d.rel[1:6]; max_iter = 25, optimization_strategy = Irls{TLSShape}(), width = 2.0))
    case!(cases, "homography_refine/contract_first3/ls", d, d -> PoseLib.refine_homography(d.H0, d.hom[1:3]; max_iter = 25, optimization_strategy = nothing))
    case!(cases, "homography_refine/contract_empty/ls", d, d -> PoseLib.refine_homography(d.H0, d.hom[1:0]; max_iter = 25, optimization_strategy = nothing))
    case!(cases, "homography_refine/contract_first4/ls", d, d -> PoseLib.refine_homography(d.H0, d.hom[1:4]; max_iter = 25, optimization_strategy = nothing))
    case!(cases, "absolute_pose_refine/contract_first2/ls", d, d -> PoseLib.refine_absolute_pose(d.seed, d.K1, d.abs_cs[1:2]; max_iter = 25, optimization_strategy = nothing))
    # refusals: the retired keywords and the missing strategy, on every refiner
    for name in keys(POSELIB_CALLS)
        call = POSELIB_CALLS[name]
        case!(cases, "$(POSELIB_FAMILY[name])/refused/retired_loss", d, d -> call(d, (; loss = TLSShape(), loss_width = 2.0)))
        case!(cases, "$(POSELIB_FAMILY[name])/refused/retired_loss_width", d, d -> call(d, (; optimization_strategy = Irls{TLSShape}(), loss_width = 2.0)))
        case!(cases, "$(POSELIB_FAMILY[name])/refused/no_strategy", d, d -> call(d, (; width = 2.0)))
    end
    refusals = [
        "retired_local_optimization" => (; local_optimization = Irls{TLSShape}(), width = 2.0),
        "unknown_keyword" => (; optimization_strategy = Irls{TLSShape}(), width = 2.0, tolerance = 1.0),
        "huber_no_width" => (; optimization_strategy = Irls{HuberShape}()),
        "huber_scale_and_width" => (; optimization_strategy = Irls{HuberShape}(), scale = 1.0, width = 2.0),
        "huber_negative_width" => (; optimization_strategy = Irls{HuberShape}(), width = -2.0),
        "huber_zero_scale" => (; optimization_strategy = Irls{HuberShape}(), scale = 0.0),
        "ls_width" => (; optimization_strategy = nothing, width = 2.0),
        "ls_scale" => (; optimization_strategy = nothing, scale = 1.0),
        "tls_scale" => (; optimization_strategy = Irls{TLSShape}(), scale = 1.0),
        "lezach_scale" => (; optimization_strategy = LeZach(), scale = 1.0),
        "huber_ec2_width" => (; optimization_strategy = Irls{HuberShape}(25, EfficiencyConstant(2.0)), width = 2.0),
        "huber_lw2_scale" => (; optimization_strategy = Irls{HuberShape}(25, LossWidth(2.0)), scale = 1.0),
        "triggs_tls" => (; optimization_strategy = Triggs{TLSShape}(nothing), width = 2.0),
        "triggs_huber" => (; optimization_strategy = Triggs{HuberShape}(nothing), width = 2.0),
        "triggs_cauchy" => (; optimization_strategy = Triggs{CauchyShape}(nothing), width = 2.0),
        "welsch" => (; optimization_strategy = Irls{WelschShape}(), width = 2.0),
        "gaussian" => (; optimization_strategy = Irls{GaussianShape}()),
        "bare_shape" => (; optimization_strategy = TLSShape(), width = 2.0),
    ]
    for (lbl, kw) in refusals
        case!(cases, "homography_refine/refused/$lbl", d, d -> H(d, kw))
    end
    case!(cases, "homography_refine/refused/huber_scale_from_ls_result", d,
        d -> H(d, (; optimization_strategy = Irls{HuberShape}(), scale = H(d, (; optimization_strategy = nothing)))))
    return cases
end

function _record_poselib_refine_scene!(cases)
    s = poselib_refine_scene()
    kw(w) = (; max_iter = 25, optimization_strategy = Irls{TLSShape}(), width = w)
    # relative pose: the z = 1 chart refused (both containers); camera-aware pixels; K1 = K2 pixels vs identity cameras
    case!(cases, "relpose_bearings_refine/refused/chart_correspondences", s, s -> PoseLib.refine_relpose(s.seed, Correspondences(s.chart.x1, s.chart.x2); kw(3e-3)...))
    case!(cases, "relpose_bearings_refine/refused/chart_pairs", s, s -> PoseLib.refine_relpose(s.seed, s.chart.x1 .=> s.chart.x2; kw(3e-3)...))
    case!(cases, "relpose_pixels_refine/refine_scene/tls_width2.0", s, s -> PoseLib.refine_relpose(s.seed, s.K1, s.K2, s.relcam.cs; kw(2.0)...))
    case!(cases, "relpose_pixels_refine/refine_scene_K1K1/tls_width2.0", s, s -> PoseLib.refine_relpose(s.seed, s.K1, s.K1, s.relcam.cs_K1; kw(2.0)...))
    case!(cases, "relpose_pixels_refine/refine_scene_identity_cameras/tls_calibrated_width", s,
        s -> PoseLib.refine_relpose(s.seed, s.relcam.I3, s.relcam.I3, s.relcam.cs_cal; kw(PoseLib.calibrated_loss_width(s.K1, s.K1, 2.0))...))
    case!(cases, "relpose_pixels_refine/refused/physical_seed", s,
        s -> PoseLib.refine_relpose(VgtkCore.EuclideanMap(s.seed.R, s.seed.t .* VgtkCore.m), s.K1, s.K2,
            Correspondences([SVector(0.0, 0.0)], [SVector(0.0, 0.0)]); max_iter = 1, optimization_strategy = Irls{TLSShape}(), width = 1.0))
    # fundamental, homography (and their 3×3 refusals)
    case!(cases, "fundamental_refine/refine_scene/tls_width2.0", s, s -> PoseLib.refine_fundamental(s.fund.F0, s.fund.cs; kw(2.0)...))
    case!(cases, "fundamental_refine/refused/seed_2x3", s, s -> PoseLib.refine_fundamental(s.fund.F0_2x3, s.fund.cs; kw(2.0)...))
    case!(cases, "homography_refine/refine_scene/tls_width2.0", s, s -> PoseLib.refine_homography(s.hom.H0, s.hom.cs; kw(2.0)...))
    case!(cases, "homography_refine/refused/seed_2x2", s, s -> PoseLib.refine_homography(s.hom.H0_2x2, s.hom.cs; kw(2.0)...))
    # bearings with tangent Jacobians, and the pixel twin they reduce to; the refusals
    b = s.bearings
    case!(cases, "relpose_pixels_refine/refine_scene_Kb2/tls_width2.0", s, s -> PoseLib.refine_relpose(s.seed, s.K1, s.bearings.Kb2, s.bearings.pix; kw(2.0)...))
    case!(cases, "relpose_bearings_refine/refine_scene/tls_width2.0", s, s -> PoseLib.refine_relpose(s.seed, s.bearings.cs; kw(2.0)...))
    case!(cases, "relpose_bearings_refine/refused/no_jacobians", s, s -> PoseLib.refine_relpose(s.seed, s.bearings.no_jacobians; kw(2.0)...))
    case!(cases, "relpose_bearings_refine/refused/chart_sources", s, s -> PoseLib.refine_relpose(s.seed, s.bearings.chart_sources; kw(2.0)...))
    case!(cases, "relpose_bearings_refine/refused/chart_targets", s, s -> PoseLib.refine_relpose(s.seed, s.bearings.chart_targets; kw(2.0)...))
    case!(cases, "relpose_bearings_refine/refused/nan_source", s, s -> PoseLib.refine_relpose(s.seed, s.bearings.nan_source; kw(2.0)...))
    # past 90°: radians width; its chart refused
    case!(cases, "relpose_bearings_refine/past90/tls_width0.5deg", s, s -> PoseLib.refine_relpose(s.past90.seed, s.past90.cs; kw(deg2rad(0.5))...))
    case!(cases, "relpose_bearings_refine/refused/past90_chart", s, s -> PoseLib.refine_relpose(s.past90.seed, s.past90.chart; kw(deg2rad(0.5))...))
    # absolute pose: the chart refused; pixels; K 3×4; the wrong unit and the dropped principal point; metres; labeled
    a = s.absolute
    case!(cases, "absolute_pose_refine/refused/chart", s, s -> PoseLib.refine_absolute_pose(s.seed, s.absolute.chart; kw(0.05)...))
    case!(cases, "absolute_pose_refine/refine_scene/tls_width40.0", s, s -> PoseLib.refine_absolute_pose(s.seed, s.K1, s.absolute.cs; kw(s.absolute.f * 0.05)...))
    case!(cases, "absolute_pose_refine/refused/K_3x4", s, s -> PoseLib.refine_absolute_pose(s.seed, s.absolute.K_3x4, s.absolute.cs; kw(s.absolute.f * 0.05)...))
    case!(cases, "absolute_pose_refine/refine_scene/tls_width0.05_wrong_unit", s, s -> PoseLib.refine_absolute_pose(s.seed, s.K1, s.absolute.cs; kw(0.05)...))
    case!(cases, "absolute_pose_refine/refine_scene_no_principal_point/tls_width40.0", s, s -> PoseLib.refine_absolute_pose(s.seed, s.absolute.Kbad, s.absolute.cs; kw(s.absolute.f * 0.05)...))
    case!(cases, "absolute_pose_refine/refine_scene_metres/tls_width40.0", s, s -> PoseLib.refine_absolute_pose(s.absolute.seedm, s.K1, s.absolute.cs_m; kw(s.absolute.f * 0.05)...))
    case!(cases, "absolute_pose_refine/refine_scene_metres_labeled/tls_width40.0", s, s -> PoseLib.refine_absolute_pose(s.absolute.labeled, s.K1, s.absolute.cs_m; kw(s.absolute.f * 0.05)...))
    case!(cases, "absolute_pose_refine/refused/metre_seed_plain_points", s, s -> PoseLib.refine_absolute_pose(s.absolute.seedm, s.K1, s.absolute.cs; kw(1.0)...))
    case!(cases, "absolute_pose_refine/refused/plain_seed_metre_points", s, s -> PoseLib.refine_absolute_pose(s.seed, s.K1, s.absolute.cs_m; kw(1.0)...))
    # the eight-point E (backend-less and PoseLibSolver forms) and its refusal below eight
    case!(cases, "essential8pt/refine_scene_first20/correspondences", s, s -> VgtkCore.essential_dlt(Correspondences(s.b1[1:20], s.b2[1:20])))
    case!(cases, "essential8pt/refine_scene_first20/poselib_pairs", s, s -> VgtkCore.essential_dlt(PoseLibSolver(), s.b1[1:20] .=> s.b2[1:20]))
    case!(cases, "essential8pt/refine_scene_first20/poselib_correspondences", s, s -> VgtkCore.essential_dlt(PoseLibSolver(), Correspondences(s.b1[1:20], s.b2[1:20])))
    case!(cases, "essential8pt/refine_scene_all200/correspondences", s, s -> VgtkCore.essential_dlt(Correspondences(s.b1, s.b2)))
    case!(cases, "essential8pt/refused/first7", s, s -> VgtkCore.essential_dlt(Correspondences(s.b1[1:7], s.b2[1:7])))
    # the 5-point kernel and its solve, every container
    idx = [3, 17, 42, 88, 150]
    case!(cases, "essential5pt/refine_scene_idx/kernel_ntuple", s,
        s -> VgtkCore.essential_5pt(PoseLibSolver(), ntuple(k -> s.b1[idx[k]], Val(5)), ntuple(k -> s.b2[idx[k]], Val(5))))
    case!(cases, "essential5pt/refine_scene_idx/solve_pairs", s, s -> solve(Essential5pt(PoseLibSolver()), s.b1 .=> s.b2, idx))
    case!(cases, "essential5pt/refine_scene_idx/solve_correspondences", s, s -> solve(Essential5pt(PoseLibSolver()), Correspondences(s.b1, s.b2), idx))
    case!(cases, "essential5pt/refine_scene_idx/kernel_pairs", s, s -> VgtkCore.essential_5pt(PoseLibSolver(), s.b1 .=> s.b2, idx))
    case!(cases, "essential5pt/refine_scene_idx/kernel_correspondences", s, s -> VgtkCore.essential_5pt(PoseLibSolver(), Correspondences(s.b1, s.b2), idx))
    # the calibrated kernels refuse 2-vectors (MethodError)
    case!(cases, "essential5pt/refused/chart_kernel_pairs", s, s -> VgtkCore.essential_5pt(PoseLibSolver(), s.c1 .=> s.c2, 1:5))
    case!(cases, "essential5pt/refused/chart_kernel_correspondences", s, s -> VgtkCore.essential_5pt(PoseLibSolver(), Correspondences(s.c1, s.c2), 1:5))
    case!(cases, "essential5pt/refused/chart_solve_pairs", s, s -> solve(Essential5pt(PoseLibSolver()), s.c1 .=> s.c2, 1:5))
    case!(cases, "essential8pt/refused/chart_poselib", s, s -> VgtkCore.essential_dlt(PoseLibSolver(), Correspondences(s.c1, s.c2)))
    case!(cases, "pose3pt/refused/chart_kernel", s, s -> VgtkCore.p3p(PoseLibSolver(), s.c1[1:3], s.X[1:3]))
    return cases
end

function _record_poselib_typed_loss!(cases)
    t = poselib_typed_loss_scene()
    fit(cs, lo, w) = PoseLib.refine_homography(t.H0, cs; max_iter = 100, optimization_strategy = lo, width = w)
    for (lbl, lo, w) in (("tls_width2.5", Irls{TLSShape}(), 2.5), ("tls_width3.5", Irls{TLSShape}(), 3.5),
                          ("lezach_width1.0", LeZach(), 1.0), ("lezach_width3.5", LeZach(), 3.5))
        case!(cases, "homography_refine/typed_loss_outliers3px/$lbl", t, t -> fit(t.outliers, lo, w))
    end
    for sc in (:noisy, :noisy_out)
        case!(cases, "homography_refine/typed_loss_$sc/ls_maxiter100", t,
            t -> PoseLib.refine_homography(t.H0, getproperty(t, sc); max_iter = 100, optimization_strategy = nothing))
        case!(cases, "homography_refine/typed_loss_$sc/huber_width2.0", t, t -> fit(getproperty(t, sc), Irls{HuberShape}(), 2.0))
    end
    case!(cases, "homography_refine/typed_loss_noisy_out/huber_width6.0", t, t -> fit(t.noisy_out, Irls{HuberShape}(), 6.0))
    case!(cases, "homography_refine/typed_loss_noisy_out/cauchy_width1e6", t, t -> fit(t.noisy_out, Irls{CauchyShape}(), 1.0e6))
    case!(cases, "homography_refine/typed_loss_noisy_out/cauchy_width2.0", t, t -> fit(t.noisy_out, Irls{CauchyShape}(), 2.0))
    return cases
end

function _record_poselib_kernels!(cases)
    sm = poselib_smoke_scene()
    case!(cases, "pose3pt/smoke_identity/kernel", sm, sm -> VgtkCore.p3p(PoseLibSolver(), sm.rays, sm.X))
    case!(cases, "pose3pt/refused_length/kernel_two_points", sm, sm -> VgtkCore.p3p(PoseLibSolver(), sm.rays[1:2], sm.X[1:2]))
    case!(cases, "homography4pt/smoke_square/kernel", sm,
        sm -> VgtkCore.homography_4pt(PoseLibSolver(), sm.s[1], sm.s[2], sm.s[3], sm.s[4], sm.d[1], sm.d[2], sm.d[3], sm.d[4]))
    case!(cases, "homography4pt/smoke_square/solve", sm, sm -> solve(Homography4pt(PoseLibSolver()), Correspondences(sm.s, sm.d), 1:4))
    for (k, seed) in enumerate(SEEDS)
        case!(cases, "essential5pt/smoke_random/kernel/seed$seed", sm, sm -> VgtkCore.essential_5pt(PoseLibSolver(), sm.bearings5[k]...))
    end
    for (name, (rays, X)) in poselib_p3p_scenes()
        case!(cases, "pose3pt/lookat_$name/kernel", (rays, X), ((rays, X),) -> VgtkCore.p3p(PoseLibSolver(), rays, X))
    end
    # 300 random minimal samples of the contract scene through each backend's solve (and the cs/idx kernel)
    for (scene, d) in poselib_contract_scenes()
        scene == "test" || continue   # the suite's draw only
        rng = StableRNG(1)
        draw(n, k) = [randperm(rng, n)[1:k] for _ in 1:300]
        hidx = draw(length(d.hom), 4); fidx = draw(length(d.rel), 7); eidx = draw(length(d.bear), 5); pidx = draw(length(d.abs_bear), 3)
        hom, rel, bear, ab = d.hom, d.rel, Correspondences(VgtkCore.sources(d.bear), VgtkCore.targets(d.bear)), d.abs_bear
        case!(cases, "homography4pt/contract/300samples_solve", (hom, hidx), ((cs, I),) -> [solve(Homography4pt(PoseLibSolver()), cs, i) for i in I])
        # the cs/idx kernels read `.first`/`.second` columns: a StructArray of pairs, as Core's solve hands them
        homsa = StructArray(VgtkCore.sources(hom) .=> VgtkCore.targets(hom)); relsa = StructArray(VgtkCore.sources(rel) .=> VgtkCore.targets(rel))
        case!(cases, "homography4pt/contract/300samples_kernel", (homsa, hidx), ((cs, I),) -> [VgtkCore.homography_4pt(PoseLibSolver(), cs, i) for i in I])
        case!(cases, "fundamental7pt/contract/300samples_solve", (rel, fidx), ((cs, I),) -> [solve(Fundamental7pt(PoseLibSolver()), cs, i) for i in I])
        case!(cases, "fundamental7pt/contract/300samples_kernel", (relsa, fidx), ((cs, I),) -> [VgtkCore.fundamental_7pt(PoseLibSolver(), cs, i) for i in I])
        case!(cases, "essential5pt/contract/300samples_solve", (bear, eidx), ((cs, I),) -> [solve(Essential5pt(PoseLibSolver()), cs, i) for i in I])
        case!(cases, "pose3pt/contract/300samples_solve", (ab, pidx), ((cs, I),) -> [solve(P3P(PoseLibSolver()), cs, i) for i in I])
    end
    return cases
end

function record_poselib(cases)
    _record_poselib_contract!(cases)
    _record_poselib_refine_scene!(cases)
    _record_poselib_typed_loss!(cases)
    _record_poselib_kernels!(cases)
    return cases
end
