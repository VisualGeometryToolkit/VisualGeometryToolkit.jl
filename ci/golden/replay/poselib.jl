# PoseLib on Core 0.14 (record/poselib.jl, case for case). The four refiners are Core's LevenbergMarquardt with PoseLib's
# backend on Core's RefineProblem of the correspondences and a guess (spec §6, §9): `refine_homography(H, cs)` →
# `solve(RefineProblem(cs, HomographyMat(H)), …)`, `refine_fundamental(F, cs)` → `RefineProblem(cs, FundMat(F))`,
# `refine_relpose(pose, K1, K2, cs)` → `RefineProblem(cs, pose, (; K1, K2))`, `refine_relpose(pose, bearings)` →
# `RefineProblem(bearings, pose)`, `refine_absolute_pose(pose, K, cs)` → `RefineProblem(cs, pose, (; K))`, each solved by
# `LevenbergMarquardt(PoseLibSolver(); optimization_strategy)` with `stops = LmStops(; max_iter)` and the scale input as
# solve keywords (Core forwards it to PoseLib's `lm_backend_refine`). `essential_dlt([PoseLibSolver(),] cs)` →
# `solve(EstimationProblem(EssentialMat, cs), DLT(PoseLibSolver()))` (Core's DLT over PoseLib's `essential_8pt`);
# `solve(<Solver>(PoseLibSolver()), cs, idx)` → `solve(MinimalProblem(M, cs, idx), <Solver>(PoseLibSolver()))`. The
# positional kernels are unchanged.
import PoseLib
using PoseLib: LeZach
using VisualGeometryToolkitCore: Irls, Triggs, TLSShape, HuberShape, CauchyShape, GaussianShape, WelschShape,
    EfficiencyConstant, LossWidth, PoseLibSolver, Essential5pt, Homography4pt, Fundamental7pt, P3P, DLT, LevenbergMarquardt,
    EstimationProblem, RefineProblem, MinimalProblem, NullParameters, ClassDefault, model_type, solve, HomographyMat, FundMat,
    EssentialMat, LmStops
Base.include(@__MODULE__, joinpath(@__DIR__, "_common.jl"))

const POSELIB_STRATEGIES = (tls = Irls{TLSShape}(), huber = Irls{HuberShape}(), cauchy = Irls{CauchyShape}(),
                            lezach = LeZach(), ls = nothing)
const POSELIB_FAMILY = (relpose_cam = "relpose_pixels_refine", relpose_bearing = "relpose_bearings_refine", fundamental = "fundamental_refine",
                        homography = "homography_refine", absolute_pose = "absolute_pose_refine")

# The old refine_* keywords as the new solve: PoseLib's suite's refine shim, `poselib_refine` (the strategy is the
# algorithm's field, `max_iter` the stops', the scale input a solve keyword; any other keyword goes to the solve as
# given: a refusal, no such keyword). A refine's FitResult replaces the old one under `RESULT_MAP`, a refusal under
# `REFUSAL_MAP`.
Base.include(@__MODULE__, joinpath(@__DIR__, "..", "..", "..", "PoseLib", "test", "refine_shim.jl"))

# A closed form that returned a bare model (or `nothing`): `u` is it, every other field new.
const PL_BARE = FieldMap(; renames = ["u" => ""], new_only = ["", "resid", "retcode", "stats", "prob", "alg", "original",
                                                              "stop_rule", "rms", "n", "metadata"])

# The shape of a loss-parametric strategy.
_strategy_shape(::Irls{S}) where {S} = S

const POSELIB_CALLS = (
    relpose_cam = (d, kw) -> poselib_refine(d.rel, d.seed, (; d.K1, d.K2); max_iter = 25, kw...),
    relpose_bearing = (d, kw) -> poselib_refine(d.bear, d.seed; max_iter = 25, kw...),
    fundamental = (d, kw) -> poselib_refine(d.rel, FundMat(d.F0); max_iter = 25, kw...),
    homography = (d, kw) -> poselib_refine(d.hom, HomographyMat(d.H0); max_iter = 25, kw...),
    absolute_pose = (d, kw) -> poselib_refine(d.abs_cs, d.seed, (; K = d.K1); max_iter = 25, kw...),
)

function _replay_poselib_contract!(cases)
    M, R = RESULT_MAP, REFUSAL_MAP
    for (scene, d) in poselib_contract_scenes()
        fx, sfx = scene == "test" ? ("contract", "") : ("contract_noise", "/" * scene)
        for name in keys(POSELIB_CALLS), l in keys(POSELIB_STRATEGIES)
            call = POSELIB_CALLS[name]; lo = POSELIB_STRATEGIES[l]
            w = name === :absolute_pose ? 40.0 : 2.0
            kw = lo === nothing ? (; optimization_strategy = lo) : (; optimization_strategy = lo, width = w)
            replay!(cases, "$(POSELIB_FAMILY[name])/$fx/$(l)_width$(lo === nothing ? "none" : w)$sfx", d, d -> call(d, kw); map = M)
            if lo isa Irls{<:Union{HuberShape, CauchyShape}}
                S = _strategy_shape(lo)
                replay!(cases, "$(POSELIB_FAMILY[name])/$fx/$(l)_scale0.75$sfx", d,
                    d -> call(d, (; optimization_strategy = lo, scale = 0.75)); map = M)
                replay!(cases, "$(POSELIB_FAMILY[name])/$fx/$(l)_ec3_scale0.75$sfx", d,
                    d -> call(d, (; optimization_strategy = Irls{S}(25, EfficiencyConstant(3.0)), scale = 0.75)); map = M)
            end
        end
    end
    d = last(first(poselib_contract_scenes()))
    H = POSELIB_CALLS.homography
    replay!(cases, "homography_refine/contract/ls_maxiter1", d,
        d -> poselib_refine(d.hom, HomographyMat(d.H0); max_iter = 1, optimization_strategy = nothing); map = M)
    replay!(cases, "homography_refine/contract/huber_scale_upstream0.5", d,
        d -> H(d, (; optimization_strategy = Irls{HuberShape}(), scale = H(d, (; optimization_strategy = Irls{HuberShape}(), scale = 0.5)))); map = M)
    for (l, kw) in ("tls_width40.0" => (; optimization_strategy = Irls{TLSShape}(), width = 40.0), "ls_widthnone" => (; optimization_strategy = nothing))
        replay!(cases, "absolute_pose_refine/contract_metres/$l", d,
            d -> poselib_refine(d.abs_cs_m, d.seedm, (; K = d.K1); max_iter = 25, kw...); map = M)
    end
    replay!(cases, "relpose_pixels_refine/contract_first4/ls", d,
        d -> poselib_refine(d.rel[1:4], d.seed, (; d.K1, d.K2); max_iter = 25, optimization_strategy = nothing); map = M)
    replay!(cases, "relpose_bearings_refine/contract_first4/ls", d,
        d -> poselib_refine(d.bear[1:4], d.seed; max_iter = 25, optimization_strategy = nothing); map = M)
    replay!(cases, "fundamental_refine/contract_first6/tls_width2.0", d,
        d -> poselib_refine(d.rel[1:6], FundMat(d.F0); max_iter = 25, optimization_strategy = Irls{TLSShape}(), width = 2.0); map = M)
    replay!(cases, "homography_refine/contract_first3/ls", d,
        d -> poselib_refine(d.hom[1:3], HomographyMat(d.H0); max_iter = 25, optimization_strategy = nothing); map = M)
    replay!(cases, "homography_refine/contract_empty/ls", d,
        d -> poselib_refine(d.hom[1:0], HomographyMat(d.H0); max_iter = 25, optimization_strategy = nothing); map = M)
    replay!(cases, "homography_refine/contract_first4/ls", d,
        d -> poselib_refine(d.hom[1:4], HomographyMat(d.H0); max_iter = 25, optimization_strategy = nothing); map = M)
    replay!(cases, "absolute_pose_refine/contract_first2/ls", d,
        d -> poselib_refine(d.abs_cs[1:2], d.seed, (; K = d.K1); max_iter = 25, optimization_strategy = nothing); map = M)
    # refusals: the retired keywords (now keywords the solve does not take) and the missing strategy, on every refiner
    for name in keys(POSELIB_CALLS)
        call = POSELIB_CALLS[name]
        replay!(cases, "$(POSELIB_FAMILY[name])/refused/retired_loss", d, d -> call(d, (; loss = TLSShape(), loss_width = 2.0)); map = R)
        replay!(cases, "$(POSELIB_FAMILY[name])/refused/retired_loss_width", d,
            d -> call(d, (; optimization_strategy = Irls{TLSShape}(), loss_width = 2.0)); map = R)
        replay!(cases, "$(POSELIB_FAMILY[name])/refused/no_strategy", d, d -> call(d, (; width = 2.0)); map = R)
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
        replay!(cases, "homography_refine/refused/$lbl", d, d -> H(d, kw); map = R)
    end
    replay!(cases, "homography_refine/refused/huber_scale_from_ls_result", d,
        d -> H(d, (; optimization_strategy = Irls{HuberShape}(), scale = H(d, (; optimization_strategy = nothing)))); map = R)
    return cases
end

function _replay_poselib_refine_scene!(cases)
    M, R, B = RESULT_MAP, REFUSAL_MAP, PL_BARE
    s = poselib_refine_scene()
    kw(w) = (; max_iter = 25, optimization_strategy = Irls{TLSShape}(), width = w)
    edlt(cs) = solve(EstimationProblem(EssentialMat, cs), DLT(PoseLibSolver()))
    msolve(sv, cs, idx) = solve(MinimalProblem(model_type(sv), cs, idx), sv)
    replay!(cases, "relpose_bearings_refine/refused/chart_correspondences", s,
        s -> poselib_refine(Correspondences(s.chart.x1, s.chart.x2), s.seed; kw(3e-3)...); map = R)
    replay!(cases, "relpose_bearings_refine/refused/chart_pairs", s, s -> poselib_refine(s.chart.x1 .=> s.chart.x2, s.seed; kw(3e-3)...); map = R)
    replay!(cases, "relpose_pixels_refine/refine_scene/tls_width2.0", s, s -> poselib_refine(s.relcam.cs, s.seed, (; s.K1, s.K2); kw(2.0)...); map = M)
    replay!(cases, "relpose_pixels_refine/refine_scene_K1K1/tls_width2.0", s,
        s -> poselib_refine(s.relcam.cs_K1, s.seed, (; s.K1, K2 = s.K1); kw(2.0)...); map = M)
    replay!(cases, "relpose_pixels_refine/refine_scene_identity_cameras/tls_calibrated_width", s,
        s -> poselib_refine(s.relcam.cs_cal, s.seed, (; K1 = s.relcam.I3, K2 = s.relcam.I3); kw(PoseLib.calibrated_loss_width(s.K1, s.K1, 2.0))...); map = M)
    replay!(cases, "relpose_pixels_refine/refused/physical_seed", s,
        s -> poselib_refine(Correspondences([SVector(0.0, 0.0)], [SVector(0.0, 0.0)]), VgtkCore.EuclideanMap(s.seed.R, s.seed.t .* VgtkCore.m),
            (; s.K1, s.K2); max_iter = 1, optimization_strategy = Irls{TLSShape}(), width = 1.0); map = R)
    replay!(cases, "fundamental_refine/refine_scene/tls_width2.0", s, s -> poselib_refine(s.fund.cs, FundMat(s.fund.F0); kw(2.0)...); map = M)
    replay!(cases, "fundamental_refine/refused/seed_2x3", s, s -> poselib_refine(s.fund.cs, FundMat(s.fund.F0_2x3); kw(2.0)...); map = R)
    replay!(cases, "homography_refine/refine_scene/tls_width2.0", s, s -> poselib_refine(s.hom.cs, HomographyMat(s.hom.H0); kw(2.0)...); map = M)
    replay!(cases, "homography_refine/refused/seed_2x2", s, s -> poselib_refine(s.hom.cs, HomographyMat(s.hom.H0_2x2); kw(2.0)...); map = R)
    replay!(cases, "relpose_pixels_refine/refine_scene_Kb2/tls_width2.0", s,
        s -> poselib_refine(s.bearings.pix, s.seed, (; s.K1, K2 = s.bearings.Kb2); kw(2.0)...); map = M)
    replay!(cases, "relpose_bearings_refine/refine_scene/tls_width2.0", s, s -> poselib_refine(s.bearings.cs, s.seed; kw(2.0)...); map = M)
    replay!(cases, "relpose_bearings_refine/refused/no_jacobians", s, s -> poselib_refine(s.bearings.no_jacobians, s.seed; kw(2.0)...); map = R)
    replay!(cases, "relpose_bearings_refine/refused/chart_sources", s, s -> poselib_refine(s.bearings.chart_sources, s.seed; kw(2.0)...); map = R)
    replay!(cases, "relpose_bearings_refine/refused/chart_targets", s, s -> poselib_refine(s.bearings.chart_targets, s.seed; kw(2.0)...); map = R)
    replay!(cases, "relpose_bearings_refine/refused/nan_source", s, s -> poselib_refine(s.bearings.nan_source, s.seed; kw(2.0)...); map = R)
    replay!(cases, "relpose_bearings_refine/past90/tls_width0.5deg", s, s -> poselib_refine(s.past90.cs, s.past90.seed; kw(deg2rad(0.5))...); map = M)
    replay!(cases, "relpose_bearings_refine/refused/past90_chart", s, s -> poselib_refine(s.past90.chart, s.past90.seed; kw(deg2rad(0.5))...); map = R)
    a(s) = s.absolute
    replay!(cases, "absolute_pose_refine/refused/chart", s, s -> poselib_refine(a(s).chart, s.seed; kw(0.05)...); map = R)
    replay!(cases, "absolute_pose_refine/refine_scene/tls_width40.0", s,
        s -> poselib_refine(a(s).cs, s.seed, (; K = s.K1); kw(a(s).f * 0.05)...); map = M)
    replay!(cases, "absolute_pose_refine/refused/K_3x4", s, s -> poselib_refine(a(s).cs, s.seed, (; K = a(s).K_3x4); kw(a(s).f * 0.05)...); map = R)
    replay!(cases, "absolute_pose_refine/refine_scene/tls_width0.05_wrong_unit", s,
        s -> poselib_refine(a(s).cs, s.seed, (; K = s.K1); kw(0.05)...); map = M)
    replay!(cases, "absolute_pose_refine/refine_scene_no_principal_point/tls_width40.0", s,
        s -> poselib_refine(a(s).cs, s.seed, (; K = a(s).Kbad); kw(a(s).f * 0.05)...); map = M)
    replay!(cases, "absolute_pose_refine/refine_scene_metres/tls_width40.0", s,
        s -> poselib_refine(a(s).cs_m, a(s).seedm, (; K = s.K1); kw(a(s).f * 0.05)...); map = M)
    replay!(cases, "absolute_pose_refine/refine_scene_metres_labeled/tls_width40.0", s,
        s -> poselib_refine(a(s).cs_m, a(s).labeled, (; K = s.K1); kw(a(s).f * 0.05)...); map = M)
    replay!(cases, "absolute_pose_refine/refused/metre_seed_plain_points", s, s -> poselib_refine(a(s).cs, a(s).seedm, (; K = s.K1); kw(1.0)...); map = R)
    replay!(cases, "absolute_pose_refine/refused/plain_seed_metre_points", s, s -> poselib_refine(a(s).cs_m, s.seed, (; K = s.K1); kw(1.0)...); map = R)
    # the eight-point E: one entry, DLT(PoseLibSolver()) (the backend-less form routed to it); its refusal below eight
    replay!(cases, "essential8pt/refine_scene_first20/correspondences", s, s -> edlt(Correspondences(s.b1[1:20], s.b2[1:20])); map = B)
    replay!(cases, "essential8pt/refine_scene_first20/poselib_pairs", s, s -> edlt(s.b1[1:20] .=> s.b2[1:20]); map = B)
    replay!(cases, "essential8pt/refine_scene_first20/poselib_correspondences", s, s -> edlt(Correspondences(s.b1[1:20], s.b2[1:20])); map = B)
    replay!(cases, "essential8pt/refine_scene_all200/correspondences", s, s -> edlt(Correspondences(s.b1, s.b2)); map = B)
    replay!(cases, "essential8pt/refused/first7", s, s -> edlt(Correspondences(s.b1[1:7], s.b2[1:7])); map = R)
    idx = [3, 17, 42, 88, 150]
    replay!(cases, "essential5pt/refine_scene_idx/kernel_ntuple", s,
        s -> VgtkCore.essential_5pt(PoseLibSolver(), ntuple(k -> s.b1[idx[k]], Val(5)), ntuple(k -> s.b2[idx[k]], Val(5))))
    replay!(cases, "essential5pt/refine_scene_idx/solve_pairs", s, s -> msolve(Essential5pt(PoseLibSolver()), s.b1 .=> s.b2, idx))
    replay!(cases, "essential5pt/refine_scene_idx/solve_correspondences", s, s -> msolve(Essential5pt(PoseLibSolver()), Correspondences(s.b1, s.b2), idx))
    replay!(cases, "essential5pt/refine_scene_idx/kernel_pairs", s, s -> VgtkCore.essential_5pt(PoseLibSolver(), s.b1 .=> s.b2, idx))
    replay!(cases, "essential5pt/refine_scene_idx/kernel_correspondences", s, s -> VgtkCore.essential_5pt(PoseLibSolver(), Correspondences(s.b1, s.b2), idx))
    replay!(cases, "essential5pt/refused/chart_kernel_pairs", s, s -> VgtkCore.essential_5pt(PoseLibSolver(), s.c1 .=> s.c2, 1:5))
    replay!(cases, "essential5pt/refused/chart_kernel_correspondences", s, s -> VgtkCore.essential_5pt(PoseLibSolver(), Correspondences(s.c1, s.c2), 1:5))
    # the solve's 2-vector refusal is Core's `__solve` (its bearing read), no longer the old `solve`'s
    replay!(cases, "essential5pt/refused/chart_solve_pairs", s, s -> msolve(Essential5pt(PoseLibSolver()), s.c1 .=> s.c2, 1:5); map = R)
    replay!(cases, "essential8pt/refused/chart_poselib", s, s -> edlt(Correspondences(s.c1, s.c2)); map = R)
    replay!(cases, "pose3pt/refused/chart_kernel", s, s -> VgtkCore.p3p(PoseLibSolver(), s.c1[1:3], s.X[1:3]))
    return cases
end

function _replay_poselib_typed_loss!(cases)
    M = RESULT_MAP
    t = poselib_typed_loss_scene()
    fit(cs, lo, w) = poselib_refine(cs, HomographyMat(t.H0); max_iter = 100, optimization_strategy = lo, width = w)
    for (lbl, lo, w) in (("tls_width2.5", Irls{TLSShape}(), 2.5), ("tls_width3.5", Irls{TLSShape}(), 3.5),
                          ("lezach_width1.0", LeZach(), 1.0), ("lezach_width3.5", LeZach(), 3.5))
        replay!(cases, "homography_refine/typed_loss_outliers3px/$lbl", t, t -> fit(t.outliers, lo, w); map = M)
    end
    for sc in (:noisy, :noisy_out)
        replay!(cases, "homography_refine/typed_loss_$sc/ls_maxiter100", t,
            t -> poselib_refine(getproperty(t, sc), HomographyMat(t.H0); max_iter = 100, optimization_strategy = nothing); map = M)
        replay!(cases, "homography_refine/typed_loss_$sc/huber_width2.0", t, t -> fit(getproperty(t, sc), Irls{HuberShape}(), 2.0); map = M)
    end
    replay!(cases, "homography_refine/typed_loss_noisy_out/huber_width6.0", t, t -> fit(t.noisy_out, Irls{HuberShape}(), 6.0); map = M)
    replay!(cases, "homography_refine/typed_loss_noisy_out/cauchy_width1e6", t, t -> fit(t.noisy_out, Irls{CauchyShape}(), 1.0e6); map = M)
    replay!(cases, "homography_refine/typed_loss_noisy_out/cauchy_width2.0", t, t -> fit(t.noisy_out, Irls{CauchyShape}(), 2.0); map = M)
    return cases
end

function _replay_poselib_kernels!(cases)
    sm = poselib_smoke_scene()
    msolve(sv, cs, idx) = solve(MinimalProblem(model_type(sv), cs, idx), sv)
    replay!(cases, "pose3pt/smoke_identity/kernel", sm, sm -> VgtkCore.p3p(PoseLibSolver(), sm.rays, sm.X))
    replay!(cases, "pose3pt/refused_length/kernel_two_points", sm, sm -> VgtkCore.p3p(PoseLibSolver(), sm.rays[1:2], sm.X[1:2]))
    replay!(cases, "homography4pt/smoke_square/kernel", sm,
        sm -> VgtkCore.homography_4pt(PoseLibSolver(), sm.s[1], sm.s[2], sm.s[3], sm.s[4], sm.d[1], sm.d[2], sm.d[3], sm.d[4]))
    replay!(cases, "homography4pt/smoke_square/solve", sm, sm -> msolve(Homography4pt(PoseLibSolver()), Correspondences(sm.s, sm.d), 1:4))
    for (k, seed) in enumerate(SEEDS)
        replay!(cases, "essential5pt/smoke_random/kernel/seed$seed", sm, sm -> VgtkCore.essential_5pt(PoseLibSolver(), sm.bearings5[k]...))
    end
    for (name, (rays, X)) in poselib_p3p_scenes()
        replay!(cases, "pose3pt/lookat_$name/kernel", (rays, X), ((rays, X),) -> VgtkCore.p3p(PoseLibSolver(), rays, X))
    end
    for (scene, d) in poselib_contract_scenes()
        scene == "test" || continue
        rng = StableRNG(1)
        draw(n, k) = [randperm(rng, n)[1:k] for _ in 1:300]
        hidx = draw(length(d.hom), 4); fidx = draw(length(d.rel), 7); eidx = draw(length(d.bear), 5); pidx = draw(length(d.abs_bear), 3)
        hom, rel, bear, ab = d.hom, d.rel, Correspondences(VgtkCore.sources(d.bear), VgtkCore.targets(d.bear)), d.abs_bear
        replay!(cases, "homography4pt/contract/300samples_solve", (hom, hidx), ((cs, I),) -> [msolve(Homography4pt(PoseLibSolver()), cs, i) for i in I])
        homsa = StructArray(VgtkCore.sources(hom) .=> VgtkCore.targets(hom)); relsa = StructArray(VgtkCore.sources(rel) .=> VgtkCore.targets(rel))
        replay!(cases, "homography4pt/contract/300samples_kernel", (homsa, hidx), ((cs, I),) -> [VgtkCore.homography_4pt(PoseLibSolver(), cs, i) for i in I])
        replay!(cases, "fundamental7pt/contract/300samples_solve", (rel, fidx), ((cs, I),) -> [msolve(Fundamental7pt(PoseLibSolver()), cs, i) for i in I])
        replay!(cases, "fundamental7pt/contract/300samples_kernel", (relsa, fidx), ((cs, I),) -> [VgtkCore.fundamental_7pt(PoseLibSolver(), cs, i) for i in I])
        replay!(cases, "essential5pt/contract/300samples_solve", (bear, eidx), ((cs, I),) -> [msolve(Essential5pt(PoseLibSolver()), cs, i) for i in I])
        replay!(cases, "pose3pt/contract/300samples_solve", (ab, pidx), ((cs, I),) -> [msolve(P3P(PoseLibSolver()), cs, i) for i in I])
    end
    return cases
end

function replay_poselib(cases)
    _replay_poselib_contract!(cases)
    _replay_poselib_refine_scene!(cases)
    _replay_poselib_typed_loss!(cases)
    _replay_poselib_kernels!(cases)
    return cases
end
