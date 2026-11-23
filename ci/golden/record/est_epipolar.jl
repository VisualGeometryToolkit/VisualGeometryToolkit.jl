# The old epipolar estimators at 3d8a06da: find_fundmat / refine_fundmat, find_essential / refine_essential,
# find_rotation / refine_rotation (camera and bearing forms) and refine_relative_pose.
using VisualGeometryToolkitCore: estimate, MsacScoring, RansacScoring, StudentTUniformScoring, Irls, Triggs, TLSShape,
    WelschShape, CauchyShape, HuberShape, GaussianShape
using VisualGeometryToolkitEstimation: find_fundmat, refine_fundmat, find_essential, refine_essential, find_rotation,
    refine_rotation, refine_relative_pose, FundMatProblem, EssentialProblem, RotationProblem, RansacConfig, RansacWorkspace,
    TerminalLO, scoring_from_toml

# The local-optimization strategies the suite runs a find_* under (core_defaults, ransac_fundmat, termination_record).
const EP_LOS = [
    "classdefault" => (;),
    "none" => (; local_optimization_strategy = nothing),
    "irls_tls" => (; local_optimization_strategy = Irls{TLSShape}()),
    "irls_welsch_ec1.5" => (; local_optimization_strategy = Irls{WelschShape}(efficiency_constant = 1.5)),
    "triggs_welsch" => (; local_optimization_strategy = Triggs{WelschShape}()),
    "terminal_irls_tls" => (; local_optimization_strategy = TerminalLO(Irls{TLSShape}())),
]

# Every find_* setting of one class on one scene: `find(fx; kw...)` the old pipeline, `problem(fx)` the class's
# problem (for an explicit scorer of record), `thr` the class threshold, `P` the workspace's problem type.
function ep_find_cases!(cases, family, name, fx, find, problem, thr, P, n; full = true)
    for (lbl, kw) in (full ? EP_LOS : EP_LOS[1:2]), seed in SEEDS
        case!(cases, "$family/$name/lo_$lbl/seed$seed", fx, fx -> find(fx; rng = StableRNG(seed), kw...))
    end
    full || return cases
    for seed in SEEDS
        case!(cases, "$family/$name/msac_class_threshold/seed$seed", fx,
            fx -> find(fx; scoring = MsacScoring(problem(fx); threshold = thr), rng = StableRNG(seed)))
        case!(cases, "$family/$name/ransac_scoring/seed$seed", fx,
            fx -> find(fx; scoring = RansacScoring(problem(fx); threshold = 2thr), rng = StableRNG(seed)))
        case!(cases, "$family/$name/student_t_uniform_7.25/seed$seed", fx,
            fx -> find(fx; scoring = scoring_from_toml(Dict("type" => "student_t_uniform", "outlier_halfwidth" => 7.25);
                                                       problem = typeof(problem(fx))), rng = StableRNG(seed)))
        case!(cases, "$family/$name/capped_2_2/seed$seed", fx,
            fx -> find(fx; config = RansacConfig(max_trials = 2, min_trials = 2), rng = StableRNG(seed)))
        case!(cases, "$family/$name/cfg5000_500_irls_welsch/seed$seed", fx,
            fx -> find(fx; config = RansacConfig(max_trials = 5000, min_trials = 500),
                       local_optimization_strategy = Irls{WelschShape}(efficiency_constant = 1.5), rng = StableRNG(seed)))
        case!(cases, "$family/$name/workspace/seed$seed", fx,
            fx -> find(fx; workspace = RansacWorkspace(P, n), rng = StableRNG(seed)))
        case!(cases, "$family/$name/workspace_f64/seed$seed", fx,
            fx -> find(fx; workspace = RansacWorkspace(Float64, P, n), rng = StableRNG(seed)))
    end
    return cases
end

# The refine_* settings from each seed's find (its model and its search's scale), and from the raw incumbent.
# `fxof(fx, θ, r)` is the refine's fixture, `refine(rfx, θ; kw...)` the old refine on it.
function ep_refine_cases!(cases, family, name, fx, find, refine, P, n; full = true)
    for seed in SEEDS
        r = find(fx; rng = StableRNG(seed))
        θ = estimate(r)
        θ === nothing && continue
        rfx = (fx, θ, r)
        case!(cases, "$family/$name/scale_upstream/seed$seed", rfx, ((fx, θ, r),) -> refine(fx, θ; scale = r))
        case!(cases, "$family/$name/ls/seed$seed", (fx, θ), ((fx, θ),) -> refine(fx, θ; optimization_strategy = nothing))
        case!(cases, "$family/$name/scale1/seed$seed", (fx, θ), ((fx, θ),) -> refine(fx, θ; scale = 1.0))
        full || continue
        case!(cases, "$family/$name/width2/seed$seed", (fx, θ), ((fx, θ),) -> refine(fx, θ; width = 2.0))
        case!(cases, "$family/$name/triggs_cauchy_ec2.5_upstream/seed$seed", rfx,
            ((fx, θ, r),) -> refine(fx, θ; optimization_strategy = Triggs{CauchyShape}(efficiency_constant = 2.5), scale = r))
        case!(cases, "$family/$name/triggs_tls_width3/seed$seed", (fx, θ),
            ((fx, θ),) -> refine(fx, θ; optimization_strategy = Triggs{TLSShape}(), width = 3.0))
        case!(cases, "$family/$name/triggs_huber_scale1/seed$seed", (fx, θ),
            ((fx, θ),) -> refine(fx, θ; optimization_strategy = Triggs{HuberShape}(), scale = 1.0))
        case!(cases, "$family/$name/workspace_scale1/seed$seed", (fx, θ),
            ((fx, θ),) -> refine(fx, θ; scale = 1.0, workspace = RansacWorkspace(P, n)))
        # from the raw RANSAC incumbent (no local optimization) at its search's scale (core_defaults)
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
function ep_refusal_cases!(cases, family, fx, find, refine, θ)
    case!(cases, "$family/refused/find_threshold_kw", fx, fx -> find(fx; threshold = 3.0))
    case!(cases, "$family/refused/find_outlier_halfwidth_kw", fx, fx -> find(fx; outlier_halfwidth = 20.0))
    case!(cases, "$family/refused/find_retired_local_optimization", fx, fx -> find(fx; local_optimization = Irls{TLSShape}()))
    case!(cases, "$family/refused/find_unknown_kw", fx, fx -> find(fx; no_such_keyword = 1))
    rfx = (fx, θ)
    for (lbl, kw) in (
            "no_scale" => (;), "scale_and_width" => (; scale = 1.0, width = 1.0), "negative_scale" => (; scale = -1.0),
            "nan_scale" => (; scale = NaN), "zero_width" => (; width = 0.0), "ls_with_scale" => (; optimization_strategy = nothing, scale = 1.0),
            "tls_scale_no_constant" => (; optimization_strategy = Triggs{TLSShape}(), scale = 1.0),
            "constant_with_width" => (; optimization_strategy = Triggs{WelschShape}(efficiency_constant = 2.5), width = 1.0),
            "irls_huber" => (; optimization_strategy = Irls{HuberShape}(), scale = 1.0),
            "irls_gaussian" => (; optimization_strategy = Irls{GaussianShape}(), scale = 1.0),
            "terminal_lo" => (; optimization_strategy = TerminalLO(Irls{TLSShape}()), scale = 1.0),
            "scoring_kw" => (; scale = 1.0, scoring = MsacScoring(FundMatProblem; threshold = 3.0)),
            "rng_kw" => (; scale = 1.0, rng = StableRNG(1)), "max_trials_kw" => (; scale = 1.0, max_trials = 10),
            "extent_kw" => (; scale = 1.0, extent = nothing), "unknown_kw" => (; scale = 1.0, nonsense = 1))
        case!(cases, "$family/refused/refine_$lbl", rfx, ((fx, θ),) -> refine(fx, θ; kw...))
    end
    return cases
end

function record_est_epipolar(cases)
    I3 = SMatrix{3, 3, Float64}(I)
    X3 = SMatrix{3, 3, Float64}(0, 0, 0, 0, 0, -1, 0, 1, 0)

    # ---------------------------------------------------------------- fundamental matrix
    ffind(cs; kw...) = find_fundmat(cs; kw...)
    frefine(cs, θ; kw...) = refine_fundmat(θ, cs; kw...)
    for (k, (name, cs)) in enumerate(ep_fundmat_scenes())
        full = k == 1 || k == 2
        ep_find_cases!(cases, "fundmat_find", name, cs, ffind, FundMatProblem, 3.0, FundMatProblem{Float64}, length(cs); full)
        ep_refine_cases!(cases, "fundmat_refine", name, cs, ffind, frefine, FundMatProblem{Float64}, length(cs); full = k == 1)
    end
    few = ep_few_correspondences(6)
    for seed in SEEDS
        case!(cases, "fundmat_find/too_few_6/seed$seed", few, cs -> find_fundmat(cs; rng = StableRNG(seed)))
    end
    F0 = VgtkCore.FundMat{Float64}(Tuple(X3))
    case!(cases, "fundmat_refine/too_few_6/scale1", (few, F0), ((cs, F),) -> refine_fundmat(F, cs; scale = 1.0))
    case!(cases, "fundmat_refine/too_few_6/ls", (few, F0), ((cs, F),) -> refine_fundmat(F, cs; optimization_strategy = nothing))
    cs1 = last(first(ep_fundmat_scenes()))
    ep_refusal_cases!(cases, "fundmat", cs1, ffind, frefine, estimate(find_fundmat(cs1; rng = StableRNG(1))))
    case!(cases, "fundmat/refused/find_zero_threshold", cs1, cs -> find_fundmat(cs; scoring = MsacScoring(FundMatProblem(cs); threshold = 0.0)))

    # ---------------------------------------------------------------- essential matrix
    efind((c1, c2, cs); kw...) = find_essential(c1, c2, cs; kw...)
    erefine((c1, c2, cs), θ; kw...) = refine_essential(θ, c1, c2, cs; kw...)
    eprob((c1, c2, cs)) = EssentialProblem(c1, c2, cs)
    for (k, (name, fx)) in enumerate(ep_essential_scenes())
        full = k <= 2
        n = length(fx[3])
        ep_find_cases!(cases, "essential_find", name, fx, efind, eprob, 1.0, EssentialProblem{Float64}, n; full)
        ep_refine_cases!(cases, "essential_refine", name, fx, efind, erefine, EssentialProblem{Float64}, n; full = k == 1)
        if name == "calib_ecam_120+50"
            for seed in SEEDS   # fitting_pipelines: MSAC at 5 px on the 1.6 px scene
                case!(cases, "essential_find/$name/msac5/seed$seed", fx,
                    fx -> efind(fx; scoring = MsacScoring(EssentialProblem; threshold = 5.0), rng = StableRNG(seed)))
            end
        end
    end
    fx4 = (EP_ECAM, EP_ECAM, ep_few_correspondences(4))
    for seed in SEEDS
        case!(cases, "essential_find/too_few_4/seed$seed", fx4, fx -> efind(fx; rng = StableRNG(seed)))
    end
    E0 = VgtkCore.EssentialMat{Float64}(Tuple(X3))
    case!(cases, "essential_refine/too_few_4/scale1", (fx4, E0), ((fx, E),) -> erefine(fx, E; scale = 1.0))
    case!(cases, "essential_refine/too_few_4/ls", (fx4, E0), ((fx, E),) -> erefine(fx, E; optimization_strategy = nothing))
    efx = last(ep_essential_scenes()[2])
    ep_refusal_cases!(cases, "essential", efx, efind, erefine, estimate(efind(efx; rng = StableRNG(1))))
    for thr in (0.02, 0.05)   # the sub-detector threshold, refused on every entry (V17)
        case!(cases, "essential/refused/find_msac_threshold$thr", efx, fx -> efind(fx; scoring = MsacScoring(eprob(fx); threshold = thr)))
    end
    case!(cases, "essential/refused/find_msac_type_threshold0.02", efx, fx -> efind(fx; scoring = MsacScoring(EssentialProblem; threshold = 0.02)))
    for seed in SEEDS
        case!(cases, "essential_find/epix_cam800_150+60/msac_floor0.06/seed$seed", efx,
            fx -> efind(fx; scoring = MsacScoring(eprob(fx); threshold = 0.06), rng = StableRNG(seed)))
    end

    # ---------------------------------------------------------------- pure rotation (camera and bearing forms)
    rfind((c1, c2, cs); kw...) = find_rotation(c1, c2, cs; kw...)
    rrefine((c1, c2, cs), θ; kw...) = refine_rotation(θ, c1, c2, cs; kw...)
    rprob((c1, c2, cs)) = RotationProblem(c1, c2, cs)
    bfind(b; kw...) = find_rotation(b; kw...)
    brefine(b, θ; kw...) = refine_rotation(θ, b; kw...)
    for (k, (name, fx)) in enumerate(ep_rotation_scenes())
        n = length(fx[3])
        ep_find_cases!(cases, "rotation_find", name, fx, rfind, rprob, 1.0, RotationProblem, n; full = k == 1)
        ep_refine_cases!(cases, "rotation_refine", name, fx, rfind, rrefine, RotationProblem, n; full = k == 1)
        b = ep_bearings(fx...)
        ep_find_cases!(cases, "rotation_find", "$(name)_bearings", b, bfind, RotationProblem, 1.0, RotationProblem, n; full = k == 1)
        ep_refine_cases!(cases, "rotation_refine", "$(name)_bearings", b, bfind, brefine, RotationProblem, n; full = false)
    end
    for (name, fx) in ("pcam_1" => (EP_PCAM, EP_PCAM, ep_few_correspondences(1)),
                       "fisheye_1" => (EP_FISHEYE, EP_FISHEYE, VgtkCore.Correspondences([SA[600.0, 600.0]], [SA[600.0, 600.0]])))
        for seed in SEEDS
            case!(cases, "rotation_find/too_few_$name/seed$seed", fx, fx -> rfind(fx; rng = StableRNG(seed)))
        end
        R1 = one(RotMatrix{3, Float64})
        case!(cases, "rotation_refine/too_few_$name/scale1", (fx, R1), ((fx, R),) -> rrefine(fx, R; scale = 1.0))
    end
    b1 = VgtkCore.Correspondences([SA[0.0, 0.0, 1.0]], [SA[0.0, 0.0, 1.0]])
    for seed in SEEDS
        case!(cases, "rotation_find/too_few_bearings_1/seed$seed", b1, b -> find_rotation(b; rng = StableRNG(seed)))
    end
    case!(cases, "rotation_refine/too_few_bearings_1/scale1", (b1, one(RotMatrix{3, Float64})), ((b, R),) -> refine_rotation(R, b; scale = 1.0))
    case!(cases, "rotation_refine/too_few_bearings_1/ls", (b1, one(RotMatrix{3, Float64})),
        ((b, R),) -> refine_rotation(R, b; optimization_strategy = nothing))
    rfx = last(first(ep_rotation_scenes()))
    ep_refusal_cases!(cases, "rotation", rfx, rfind, rrefine, estimate(rfind(rfx; rng = StableRNG(1))))
    case!(cases, "rotation/refused/find_msac_threshold0.01", rfx,
        fx -> rfind(fx; scoring = MsacScoring(RotationProblem; threshold = 0.01)))
    case!(cases, "rotation/refused/bearings_pairs_vector", b1,
        b -> find_rotation(collect(VgtkCore.sources(b)) .=> collect(VgtkCore.targets(b)); rng = StableRNG(1)))

    # ---------------------------------------------------------------- the relative-pose bundle (deterministic)
    for (name, fx) in ep_relpose_fixtures()
        case!(cases, "relpose_bundle/$name/default", fx, ((pose, K1, K2, cs),) -> refine_relative_pose(pose, K1, K2, cs))
        name == "exact_30" && case!(cases, "relpose_bundle/$name/max_iter200", fx,
            ((pose, K1, K2, cs),) -> refine_relative_pose(pose, K1, K2, cs; max_iter = 200))
        name == "harness_100" && case!(cases, "relpose_bundle/$name/max_iter1", fx,
            ((pose, K1, K2, cs),) -> refine_relative_pose(pose, K1, K2, cs; max_iter = 1))
    end
    return cases
end
