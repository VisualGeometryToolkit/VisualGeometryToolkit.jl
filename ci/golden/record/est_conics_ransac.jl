# The old conic estimators and RANSAC core at 3d8a06da: find_circle / refine_circle, find_ellipse / refine_ellipse;
# ransac and ransac! (with workspaces and result buffers), locally_optimize (a model and FixedModels),
# locally_optimize_rescored!, fit_hypothesis, and the minimal solves solve(problem, idx) of every RANSAC problem type.
using VisualGeometryToolkitCore: estimate, MsacScoring, RansacScoring, Irls, Triggs, TLSShape, WelschShape, CauchyShape,
    HuberShape, GaussianShape, FixedModels
using VisualGeometryToolkitEstimation: find_circle, refine_circle, find_ellipse, refine_ellipse, ransac, ransac!,
    locally_optimize, locally_optimize_rescored!, fit_hypothesis, CircleProblem, EllipseProblem, LineFittingProblem,
    InhomLineFittingProblem, HomographyProblem, FundMatProblem, EssentialProblem, AbsolutePoseProblem, BearingPoseProblem,
    RotationProblem, RadialHomographyProblem, RadialCameraProblem, HandEyeProblem, BoundedProblem,
    RansacConfig, RansacWorkspace, RansacResultBuffers, TerminalLO, CountedLO, scoring_from_toml, data_size, minimal_sample_size

const CR_LOS = [
    "classdefault" => (;),
    "none" => (; local_optimization_strategy = nothing),
    "irls_tls" => (; local_optimization_strategy = Irls{TLSShape}()),
    "irls_welsch_ec1.5" => (; local_optimization_strategy = Irls{WelschShape}(efficiency_constant = 1.5)),
    "triggs_welsch" => (; local_optimization_strategy = Triggs{WelschShape}()),
    "terminal_irls_tls" => (; local_optimization_strategy = TerminalLO(Irls{TLSShape}())),
]
# The strategies the core loop and locally_optimize are run under (`nothing` is no local optimization).
const CR_STRATEGIES = [
    "none" => nothing,
    "irls_tls" => Irls{TLSShape}(),
    "irls_welsch_ec1.5" => Irls{WelschShape}(efficiency_constant = 1.5),
    "irls_huber_ec1.345" => Irls{HuberShape}(efficiency_constant = 1.345),
    "triggs_welsch" => Triggs{WelschShape}(),
    "triggs_cauchy_ec2.5" => Triggs{CauchyShape}(efficiency_constant = 2.5),
    "terminal_irls_welsch_ec1.5" => TerminalLO(Irls{WelschShape}(efficiency_constant = 1.5)),
]

# --- find_circle / find_ellipse and their refines: the shared settings -----------------------------------------------
function cr_find_cases!(cases, family, name, fx, find, problem, P; full = true)
    for (lbl, kw) in (full ? CR_LOS : CR_LOS[1:2]), seed in SEEDS
        case!(cases, "$family/$name/lo_$lbl/seed$seed", fx, fx -> find(fx; rng = StableRNG(seed), kw...))
    end
    full || return cases
    for seed in SEEDS
        case!(cases, "$family/$name/msac_class_threshold/seed$seed", fx, fx -> find(fx; scoring = MsacScoring(problem(fx); threshold = 1.0), rng = StableRNG(seed)))
        case!(cases, "$family/$name/ransac_scoring2/seed$seed", fx, fx -> find(fx; scoring = RansacScoring(problem(fx); threshold = 2.0), rng = StableRNG(seed)))
        case!(cases, "$family/$name/student_t_uniform_7.25/seed$seed", fx,
            fx -> find(fx; scoring = scoring_from_toml(Dict("type" => "student_t_uniform", "outlier_halfwidth" => 7.25); problem = P), rng = StableRNG(seed)))
        case!(cases, "$family/$name/capped_2_2/seed$seed", fx, fx -> find(fx; config = RansacConfig(max_trials = 2, min_trials = 2), rng = StableRNG(seed)))
        case!(cases, "$family/$name/max_trials500/seed$seed", fx, fx -> find(fx; max_trials = 500, rng = StableRNG(seed)))
        case!(cases, "$family/$name/min_trials50/seed$seed", fx, fx -> find(fx; min_trials = 50, rng = StableRNG(seed)))
        case!(cases, "$family/$name/cfg400_20/seed$seed", fx, fx -> find(fx; config = RansacConfig(max_trials = 400, min_trials = 20), rng = StableRNG(seed)))
        case!(cases, "$family/$name/workspace/seed$seed", fx, fx -> find(fx; workspace = RansacWorkspace(P, length(fx)), rng = StableRNG(seed)))
        case!(cases, "$family/$name/workspace_f64/seed$seed", fx, fx -> find(fx; workspace = RansacWorkspace(Float64, P, length(fx)), rng = StableRNG(seed)))
    end
    return cases
end
function cr_refine_cases!(cases, family, name, fx, find, refine, P; full = true)
    for seed in SEEDS
        r = find(fx; rng = StableRNG(seed))
        θ = estimate(r)
        θ === nothing && continue
        case!(cases, "$family/$name/scale_upstream/seed$seed", (fx, θ, r), ((fx, θ, r),) -> refine(fx, θ; scale = r))
        case!(cases, "$family/$name/ls/seed$seed", (fx, θ), ((fx, θ),) -> refine(fx, θ; optimization_strategy = nothing))
        case!(cases, "$family/$name/scale0.5/seed$seed", (fx, θ), ((fx, θ),) -> refine(fx, θ; scale = 0.5))
        full || continue
        case!(cases, "$family/$name/width2/seed$seed", (fx, θ), ((fx, θ),) -> refine(fx, θ; width = 2.0))
        case!(cases, "$family/$name/triggs_cauchy_ec2.5_upstream/seed$seed", (fx, θ, r),
            ((fx, θ, r),) -> refine(fx, θ; optimization_strategy = Triggs{CauchyShape}(efficiency_constant = 2.5), scale = r))
        case!(cases, "$family/$name/triggs_tls_width1.5/seed$seed", (fx, θ), ((fx, θ),) -> refine(fx, θ; optimization_strategy = Triggs{TLSShape}(), width = 1.5))
        case!(cases, "$family/$name/triggs_huber_scale0.5/seed$seed", (fx, θ), ((fx, θ),) -> refine(fx, θ; optimization_strategy = Triggs{HuberShape}(), scale = 0.5))
        case!(cases, "$family/$name/workspace_scale0.5/seed$seed", (fx, θ), ((fx, θ),) -> refine(fx, θ; scale = 0.5, workspace = RansacWorkspace(P, length(fx))))
        raw_r = find(fx; local_optimization_strategy = nothing, rng = StableRNG(seed))
        raw = estimate(raw_r)
        raw === nothing && continue
        case!(cases, "$family/$name/from_raw_upstream/seed$seed", (fx, raw, raw_r), ((fx, θ, r),) -> refine(fx, θ; scale = r))
        case!(cases, "$family/$name/from_raw_cauchy_ec2.5/seed$seed", (fx, raw, raw_r),
            ((fx, θ, r),) -> refine(fx, θ; optimization_strategy = Triggs{CauchyShape}(efficiency_constant = 2.5), scale = r))
    end
    return cases
end
function cr_refusal_cases!(cases, family, fx, find, refine, θ, P)
    case!(cases, "$family/refused/find_threshold_kw", fx, fx -> find(fx; threshold = 2.0))
    case!(cases, "$family/refused/find_outlier_halfwidth_kw", fx, fx -> find(fx; outlier_halfwidth = 5.0))
    case!(cases, "$family/refused/find_threshold_and_scoring", fx, fx -> find(fx; threshold = 2.0, scoring = MsacScoring(P; threshold = 1.0)))
    case!(cases, "$family/refused/find_retired_local_optimization", fx, fx -> find(fx; local_optimization = Irls{TLSShape}()))
    case!(cases, "$family/refused/find_extent_kw", fx, fx -> find(fx; extent = nothing))
    case!(cases, "$family/refused/find_zero_threshold", fx, fx -> find(fx; scoring = MsacScoring(P; threshold = 0.0)))
    for (lbl, kw) in (
            "no_scale" => (;), "scale_and_width" => (; scale = 1.0, width = 1.0), "negative_scale" => (; scale = -1.0),
            "nan_scale" => (; scale = NaN), "zero_width" => (; width = 0.0), "ls_with_scale" => (; optimization_strategy = nothing, scale = 1.0),
            "tls_scale_no_constant" => (; optimization_strategy = Triggs{TLSShape}(), scale = 1.0),
            "constant_with_width" => (; optimization_strategy = Triggs{WelschShape}(efficiency_constant = 2.5), width = 1.0),
            "irls_huber" => (; optimization_strategy = Irls{HuberShape}(), scale = 1.0),
            "irls_gaussian" => (; optimization_strategy = Irls{GaussianShape}(), scale = 1.0),
            "irls_welsch_loss_width" => (; optimization_strategy = Irls{WelschShape}(loss_width = 1.0)),
            "terminal_lo" => (; optimization_strategy = TerminalLO(Irls{TLSShape}()), scale = 1.0),
            "scoring_kw" => (; scale = 0.5, scoring = MsacScoring(P; threshold = 1.0)),
            "threshold_kw" => (; scale = 0.5, threshold = 2.0), "outlier_halfwidth_kw" => (; scale = 0.5, outlier_halfwidth = 5.0),
            "config_kw" => (; scale = 0.5, config = RansacConfig()), "min_trials_kw" => (; scale = 0.5, min_trials = 1),
            "rng_kw" => (; scale = 0.5, rng = StableRNG(1)), "max_trials_kw" => (; scale = 0.5, max_trials = 10),
            "extent_kw" => (; scale = 0.5, extent = nothing), "unknown_kw" => (; scale = 0.5, nonsense = 1))
        case!(cases, "$family/refused/refine_$lbl", (fx, θ), ((fx, θ),) -> refine(fx, θ; kw...))
    end
    return cases
end

# --- the RANSAC loop ---------------------------------------------------------------------------------------------
# Every ransac / ransac! setting on one problem: `mk(data)` builds the problem, `thr` its MSAC threshold.
function cr_ransac_cases!(cases, name, data, mk, thr; full = true, cfg = RansacConfig())
    fam = "ransac/$name"
    for (lbl, lo) in (full ? CR_STRATEGIES : CR_STRATEGIES[1:2]), seed in SEEDS
        case!(cases, "$fam/msac_$lbl/seed$seed", data,
            d -> (p = mk(d); ransac(p, MsacScoring(p; threshold = thr); local_optimization_strategy = lo, config = cfg, rng = StableRNG(seed))))
    end
    for seed in SEEDS
        case!(cases, "$fam/msac_scoring_default_lo/seed$seed", data, d -> (p = mk(d); ransac(p, MsacScoring(p; threshold = thr); config = cfg, rng = StableRNG(seed))))
    end
    full || return cases
    for seed in SEEDS
        sc(p) = MsacScoring(p; threshold = thr)
        lo = Irls{TLSShape}()
        case!(cases, "$fam/ransac_scoring_irls_tls/seed$seed", data,
            d -> (p = mk(d); ransac(p, RansacScoring(p; threshold = thr); local_optimization_strategy = lo, config = cfg, rng = StableRNG(seed))))
        case!(cases, "$fam/capped_2_2/seed$seed", data,
            d -> (p = mk(d); ransac(p, sc(p); local_optimization_strategy = lo, config = RansacConfig(max_trials = 2, min_trials = 2), rng = StableRNG(seed))))
        case!(cases, "$fam/cfg5000_500/seed$seed", data,
            d -> (p = mk(d); ransac(p, sc(p); local_optimization_strategy = lo, config = RansacConfig(max_trials = 5000, min_trials = 500), rng = StableRNG(seed))))
        case!(cases, "$fam/score_guard_irls_tls/seed$seed", data,
            d -> (p = mk(d); ransac(p, sc(p); local_optimization_strategy = lo, score_guard = true, config = cfg, rng = StableRNG(seed))))
        case!(cases, "$fam/workspace/seed$seed", data,
            d -> (p = mk(d); ransac(p, sc(p); local_optimization_strategy = lo, workspace = RansacWorkspace(p, sc(p), lo), config = cfg, rng = StableRNG(seed))))
        case!(cases, "$fam/workspace_f64/seed$seed", data,
            d -> (p = mk(d); ransac(p, sc(p); local_optimization_strategy = lo, workspace = RansacWorkspace(Float64, p, sc(p), lo), config = cfg, rng = StableRNG(seed))))
        case!(cases, "$fam/workspace_type_n/seed$seed", data,
            d -> (p = mk(d); ransac(p, sc(p); local_optimization_strategy = lo, workspace = RansacWorkspace(typeof(p), data_size(p), sc(p), lo), config = cfg, rng = StableRNG(seed))))
        case!(cases, "$fam/workspace_reused_second_run/seed$seed", data,
            d -> (p = mk(d); ws = RansacWorkspace(p, sc(p), lo);
                  ransac(p, sc(p); local_optimization_strategy = lo, workspace = ws, config = cfg, rng = StableRNG(seed + 100));
                  ransac(p, sc(p); local_optimization_strategy = lo, workspace = ws, config = cfg, rng = StableRNG(seed))))
        case!(cases, "$fam/bang_result_buffers/seed$seed", data,
            d -> (p = mk(d); ws = RansacWorkspace(p, sc(p), lo);
                  ransac!(RansacResultBuffers(ws), ws, p, sc(p); local_optimization_strategy = lo, config = cfg, rng = StableRNG(seed))))
        case!(cases, "$fam/bang_reused_second_run/seed$seed", data,
            d -> (p = mk(d); ws = RansacWorkspace(p, sc(p), lo); res = RansacResultBuffers(ws);
                  ransac!(res, ws, p, sc(p); local_optimization_strategy = lo, config = cfg, rng = StableRNG(seed + 100));
                  ransac!(res, ws, p, sc(p); local_optimization_strategy = lo, config = cfg, rng = StableRNG(seed))))
        case!(cases, "$fam/bang_f64_no_lo/seed$seed", data,
            d -> (p = mk(d); ws = RansacWorkspace(Float64, p, sc(p));
                  ransac!(RansacResultBuffers(ws), ws, p, sc(p); local_optimization_strategy = nothing, config = cfg, rng = StableRNG(seed))))
        case!(cases, "$fam/on_improvement_trace/seed$seed", data,
            d -> (p = mk(d); trace = Any[];
                  r = ransac(p, sc(p); local_optimization_strategy = lo, config = cfg, rng = StableRNG(seed),
                             on_improvement = (trial, score, ws) -> push!(trace, (trial, score, ws.best_model, ws.best_lo_retcode, ws.best_lo_stop_rule)));
                  (r, trace)))
        case!(cases, "$fam/on_hypothesis_trace/seed$seed", data,
            d -> (p = mk(d); trace = Any[];
                  r = ransac(p, sc(p); local_optimization_strategy = nothing, config = cfg, rng = StableRNG(seed),
                             on_hypothesis = (trial, model) -> push!(trace, (trial, model)));
                  (r, trace)))
        case!(cases, "$fam/counted_irls_welsch/seed$seed", data,
            d -> (p = mk(d); c = CountedLO(Irls{WelschShape}(efficiency_constant = 1.5));
                  r = ransac(p, sc(p); local_optimization_strategy = c, config = cfg, rng = StableRNG(seed));
                  (r, c.calls[], c.iterations[], c.call_records)))
    end
    return cases
end

# --- locally_optimize, a model and FixedModels; locally_optimize_rescored! ------------------------------------------
function cr_lo_cases!(cases, name, data, mk, thr)
    fam = "locally_optimize/$name"
    for seed in SEEDS
        p = mk(data)
        θ = estimate(ransac(p, MsacScoring(p; threshold = thr); local_optimization_strategy = nothing, rng = StableRNG(seed)))
        θ2 = estimate(ransac(p, MsacScoring(p; threshold = thr); local_optimization_strategy = nothing, rng = StableRNG(seed + 10)))
        (θ === nothing || θ2 === nothing) && continue
        fx = (data, θ)
        for (lbl, lo) in CR_STRATEGIES
            case!(cases, "$fam/model_$lbl/seed$seed", fx, ((d, θ),) -> (p = mk(d); locally_optimize(p, θ; scoring = MsacScoring(p; threshold = thr), local_optimization_strategy = lo)))
        end
        case!(cases, "$fam/model_ransac_scoring_irls_tls/seed$seed", fx,
            ((d, θ),) -> (p = mk(d); locally_optimize(p, θ; scoring = RansacScoring(p; threshold = thr), local_optimization_strategy = Irls{TLSShape}())))
        case!(cases, "$fam/model_score_guard_irls_welsch/seed$seed", fx,
            ((d, θ),) -> (p = mk(d); locally_optimize(p, θ; scoring = MsacScoring(p; threshold = thr), local_optimization_strategy = Irls{WelschShape}(efficiency_constant = 1.5), score_guard = true)))
        case!(cases, "$fam/model_workspace_irls_tls/seed$seed", fx,
            ((d, θ),) -> (p = mk(d); sc = MsacScoring(p; threshold = thr); lo = Irls{TLSShape}();
                          locally_optimize(p, θ; scoring = sc, local_optimization_strategy = lo, workspace = RansacWorkspace(p, sc, lo))))
        case!(cases, "$fam/model_counted_irls_welsch/seed$seed", fx,
            ((d, θ),) -> (p = mk(d); c = CountedLO(Irls{WelschShape}(efficiency_constant = 1.5));
                          r = locally_optimize(p, θ; scoring = MsacScoring(p; threshold = thr), local_optimization_strategy = c); (r, c.calls[], c.iterations[], c.call_records)))
        fx2 = (data, θ, θ2)
        for (lbl, lo) in ("none" => nothing, "irls_tls" => Irls{TLSShape}(), "triggs_welsch" => Triggs{WelschShape}(),
                          "irls_welsch_ec1.5" => Irls{WelschShape}(efficiency_constant = 1.5))
            case!(cases, "$fam/fixed2_$lbl/seed$seed", fx2,
                ((d, θ, θ2),) -> (p = mk(d); locally_optimize(p, FixedModels(2, (θ, θ2)); scoring = MsacScoring(p; threshold = thr), local_optimization_strategy = lo)))
            case!(cases, "$fam/fixed1of2_$lbl/seed$seed", fx2,
                ((d, θ, θ2),) -> (p = mk(d); locally_optimize(p, FixedModels(1, (θ, θ2)); scoring = MsacScoring(p; threshold = thr), local_optimization_strategy = lo)))
        end
        case!(cases, "$fam/fixed2_workspace_irls_tls/seed$seed", fx2,
            ((d, θ, θ2),) -> (p = mk(d); sc = MsacScoring(p; threshold = thr); lo = Irls{TLSShape}();
                              locally_optimize(p, FixedModels(2, (θ, θ2)); scoring = sc, local_optimization_strategy = lo, workspace = RansacWorkspace(p, sc, lo))))
        # locally_optimize_rescored!: the loop's Phase-2 step under another scorer, on a fresh workspace
        for (lbl, lo) in ("irls_tls" => Irls{TLSShape}(), "irls_welsch_ec1.5" => Irls{WelschShape}(efficiency_constant = 1.5),
                          "triggs_welsch" => Triggs{WelschShape}(), "none" => nothing)
            for (slbl, mult) in ("same" => 1.0, "double" => 2.0, "half" => 0.5)
                case!(cases, "rescored/$name/$(lbl)_msac_$slbl/seed$seed", fx,
                    ((d, θ),) -> (p = mk(d); ws = RansacWorkspace(p, MsacScoring(p; threshold = thr), lo);
                                  locally_optimize_rescored!(ws.lo_ws, lo, p, θ, ws.buf, MsacScoring(p; threshold = mult * thr))))
            end
            case!(cases, "rescored/$name/$(lbl)_ransac_scoring/seed$seed", fx,
                ((d, θ),) -> (p = mk(d); ws = RansacWorkspace(p, MsacScoring(p; threshold = thr), lo);
                              locally_optimize_rescored!(ws.lo_ws, lo, p, θ, ws.buf, RansacScoring(p; threshold = thr))))
        end
    end
    p = mk(data)
    θ = estimate(ransac(p, MsacScoring(p; threshold = thr); local_optimization_strategy = nothing, rng = StableRNG(1)))
    case!(cases, "$fam/refused/no_scoring", (data, θ), ((d, θ),) -> locally_optimize(mk(d), θ; local_optimization_strategy = nothing))
    case!(cases, "$fam/refused/threshold_kw", (data, θ),
        ((d, θ),) -> (p = mk(d); locally_optimize(p, θ; scoring = MsacScoring(p; threshold = thr), local_optimization_strategy = nothing, threshold = 3.0)))
    case!(cases, "$fam/refused/outlier_halfwidth_kw", (data, θ),
        ((d, θ),) -> (p = mk(d); locally_optimize(p, FixedModels(1, (θ, θ)); scoring = MsacScoring(p; threshold = thr), local_optimization_strategy = nothing, outlier_halfwidth = 20.0)))
    case!(cases, "$fam/refused/no_strategy", (data, θ), ((d, θ),) -> (p = mk(d); locally_optimize(p, θ; scoring = MsacScoring(p; threshold = thr))))
    case!(cases, "$fam/refused/retired_local_optimization", (data, θ),
        ((d, θ),) -> (p = mk(d); locally_optimize(p, θ; scoring = MsacScoring(p; threshold = thr), local_optimization_strategy = nothing,
                                                  local_optimization = Irls{TLSShape}())))
    return cases
end

function record_est_conics_ransac(cases)
    # ================================================================ circle
    cfind(pts; kw...) = find_circle(pts; kw...)
    crefine(pts, θ; kw...) = refine_circle(θ, pts; kw...)
    for (k, (name, pts)) in enumerate(cr_circle_scenes())
        cr_find_cases!(cases, "circle_find", name, pts, cfind, CircleProblem, CircleProblem; full = k <= 2)
        cr_refine_cases!(cases, "circle_refine", name, pts, cfind, crefine, CircleProblem; full = k == 2)
    end
    for seed in SEEDS
        case!(cases, "circle_find/collinear_10/seed$seed", cr_collinear(), pts -> find_circle(pts; rng = StableRNG(seed)))
        case!(cases, "circle_find/too_few_2/seed$seed", few_points(2), pts -> find_circle(pts; rng = StableRNG(seed)))
        case!(cases, "circle_find/too_few_2_workspace_f64/seed$seed", few_points(2),
            pts -> find_circle(pts; workspace = RansacWorkspace(Float64, CircleProblem, 2), rng = StableRNG(seed)))
    end
    for (name, (g, pts)) in cr_circle_refine_fixtures()
        fx = (g, pts)
        for (lbl, kw) in ("ls" => (; optimization_strategy = nothing), "scale0.5" => (; scale = 0.5), "scale0.3" => (; scale = 0.3),
                          "triggs_tls_width1.5" => (; optimization_strategy = Triggs{TLSShape}(), width = 1.5),
                          "workspace_scale0.5" => (; scale = 0.5, workspace = RansacWorkspace(CircleProblem, length(pts))))
            case!(cases, "circle_refine/$name/$lbl", fx, ((g, pts),) -> refine_circle(g, pts; kw...))
        end
        name == "too_few_1" && case!(cases, "circle_refine/$name/refused_scoring", fx,
            ((g, pts),) -> refine_circle(g, pts; scale = 0.3, scoring = MsacScoring(CircleProblem; threshold = 1.0)))
    end
    # triggs_tests: every loss of Core's table as a Triggs at the plain multiple 2.5, from a perturbed guess
    g, pts = last(cr_circle_refine_fixtures()[4])
    for (K, tag) in VgtkCore._LOSS_TABLE
        case!(cases, "circle_refine/perturbed_noisy0.3_200/triggs_$(tag)_ec2.5_scale0.3", (g, pts),
            ((g, pts),) -> refine_circle(g, pts; optimization_strategy = Triggs{K}(efficiency_constant = 2.5), scale = 0.3))
    end
    cpts = last(cr_circle_scenes()[2])
    cr_refusal_cases!(cases, "circle", cpts, cfind, crefine, estimate(cfind(cpts; rng = StableRNG(1))), CircleProblem)

    # ================================================================ ellipse
    efind(pts; kw...) = find_ellipse(pts; kw...)
    erefine(pts, θ; kw...) = refine_ellipse(θ, pts; kw...)
    for (k, (name, pts)) in enumerate(cr_ellipse_scenes())
        cr_find_cases!(cases, "ellipse_find", name, pts, efind, EllipseProblem, EllipseProblem; full = k <= 2)
        cr_refine_cases!(cases, "ellipse_refine", name, pts, efind, erefine, EllipseProblem; full = k == 2)
    end
    for seed in SEEDS
        case!(cases, "ellipse_find/collinear_10/seed$seed", cr_collinear(), pts -> find_ellipse(pts; rng = StableRNG(seed)))
        case!(cases, "ellipse_find/too_few_4/seed$seed", few_points(4), pts -> find_ellipse(pts; rng = StableRNG(seed)))
        case!(cases, "ellipse_find/square_4/seed$seed", [VgtkCore.Point2(0.0, 0.0), VgtkCore.Point2(1.0, 0.0), VgtkCore.Point2(1.0, 1.0), VgtkCore.Point2(0.0, 1.0)],
            pts -> find_ellipse(pts; rng = StableRNG(seed)))
    end
    for (name, (g, pts)) in cr_ellipse_refine_fixtures()
        fx = (g, pts)
        for (lbl, kw) in ("ls" => (; optimization_strategy = nothing), "scale0.5" => (; scale = 0.5),
                          "triggs_tls_width1.5" => (; optimization_strategy = Triggs{TLSShape}(), width = 1.5),
                          "triggs_cauchy_ec2.5_scale0.5" => (; optimization_strategy = Triggs{CauchyShape}(efficiency_constant = 2.5), scale = 0.5),
                          "workspace_scale0.5" => (; scale = 0.5, workspace = RansacWorkspace(EllipseProblem, length(pts))))
            case!(cases, "ellipse_refine/$name/$lbl", fx, ((g, pts),) -> refine_ellipse(g, pts; kw...))
        end
    end
    epts = last(cr_ellipse_scenes()[2])
    cr_refusal_cases!(cases, "ellipse", epts, efind, erefine, estimate(efind(epts; rng = StableRNG(1))), EllipseProblem)

    # ================================================================ ransac / ransac!
    hdata = VgtkCore.correspondences(cr_make_homography_data()...)
    cr_ransac_cases!(cases, "homography_100+30", hdata, HomographyProblem, 3.0; cfg = RansacConfig(max_trials = 3000, min_trials = 200))
    cr_ransac_cases!(cases, "homography_h_scene", first(h_scene()), HomographyProblem, 3.0; full = false)
    cr_ransac_cases!(cases, "homography_prosac", cr_homography_scored(), HomographyProblem, 3.0; full = false,
                     cfg = RansacConfig(max_trials = 3000, min_trials = 200))
    fdata = cr_fundmat_data()
    cr_ransac_cases!(cases, "fundmat_100+30", fdata, FundMatProblem, 3.0; cfg = RansacConfig(max_trials = 5000, min_trials = 500))
    ldata = cr_make_line_data(n_inliers = 200, n_outliers = 50)
    cr_ransac_cases!(cases, "line_200+50", ldata, LineFittingProblem, 0.5; cfg = RansacConfig(max_trials = 2000, min_trials = 100))
    cr_ransac_cases!(cases, "line_50+200", cr_make_line_data(n_inliers = 50, n_outliers = 200, seed = 456), LineFittingProblem, 0.5;
                     full = false, cfg = RansacConfig(max_trials = 10_000, min_trials = 500))
    cr_ransac_cases!(cases, "line_60+180", cr_make_line_data(n_inliers = 60, n_outliers = 180, seed = 99), LineFittingProblem, 0.5;
                     full = false, cfg = RansacConfig(max_trials = 500, min_trials = 200))
    cr_ransac_cases!(cases, "line_contaminated_57+3", first(cr_contaminated_line(StableRNG(57))), LineFittingProblem, 0.1; full = false)
    cdata = last(cr_circle_scenes()[2])
    cr_ransac_cases!(cases, "circle_outl30_300", cdata, CircleProblem, 1.0)
    cr_ransac_cases!(cases, "circle_run_60+20", last(cr_circle_scenes()[5]), CircleProblem, 1.0; full = false, cfg = RansacConfig(max_trials = 200))
    cr_ransac_cases!(cases, "ellipse_outl30_300", last(cr_ellipse_scenes()[2]), EllipseProblem, 1.0; full = false)
    cr_ransac_cases!(cases, "inhom_line_50+20", cr_inhom_line_points(), InhomLineFittingProblem, 1.5; full = false,
                     cfg = RansacConfig(; max_trials = 500, success_prob = 0.999))
    cr_ransac_cases!(cases, "inhom_line_clean_40", cr_inhom_line_clean(), InhomLineFittingProblem, 0.05; full = false)
    for seed in SEEDS   # line_fitting_tests: the inhomogeneous line's failed refit under Irls{Welsch}
        case!(cases, "ransac/inhom_line_clean_40/msac_irls_welsch_ec1.5_default_cfg/seed$seed", cr_inhom_line_clean(),
            d -> (p = InhomLineFittingProblem(d); ransac(p, MsacScoring(p; threshold = 0.05); local_optimization_strategy = Irls{WelschShape}(efficiency_constant = 1.5), rng = StableRNG(seed))))
        # Xoshiro streams (ransac_tests' bit-reproducibility run)
        case!(cases, "ransac/line_60+180/msac_xoshiro/seed$seed", cr_make_line_data(n_inliers = 60, n_outliers = 180, seed = 99),
            d -> (p = LineFittingProblem(d); ransac(p, MsacScoring(p; threshold = 0.5); config = RansacConfig(max_trials = 500, min_trials = 200), rng = Xoshiro(seed))))
        # a workspace built for a small problem serves a larger one, and the reverse
        small = cr_make_line_data(n_inliers = 20, n_outliers = 5)
        case!(cases, "ransac/line_200+50/workspace_from_small/seed$seed", (small, ldata),
            ((s, l),) -> (sc = MsacScoring(2; threshold = 0.5); lo = Irls{TLSShape}();
                          ransac(LineFittingProblem(l), sc; local_optimization_strategy = lo, workspace = RansacWorkspace(LineFittingProblem(s), sc, lo), rng = StableRNG(seed))))
        case!(cases, "ransac/line_20+5/workspace_from_large/seed$seed", (small, ldata),
            ((s, l),) -> (sc = MsacScoring(2; threshold = 0.5);
                          ransac(LineFittingProblem(s), sc; workspace = RansacWorkspace(LineFittingProblem(l), sc), rng = StableRNG(seed))))
        # the thin incumbent: a terminal LO on m + 1 points does not run
        case!(cases, "ransac/homography_thin_5+12/terminal_irls_welsch_ec1.5/seed$seed", cr_homography_thin(),
            d -> (p = HomographyProblem(d); ransac(p, MsacScoring(p; threshold = 0.5); local_optimization_strategy = TerminalLO(Irls{WelschShape}(efficiency_constant = 1.5)),
                                                    config = RansacConfig(max_trials = 5000, min_trials = 3000), rng = StableRNG(seed))))
        case!(cases, "ransac/homography_thin_5+12/none/seed$seed", cr_homography_thin(),
            d -> (p = HomographyProblem(d); ransac(p, MsacScoring(p; threshold = 0.5); local_optimization_strategy = nothing,
                                                    config = RansacConfig(max_trials = 5000, min_trials = 3000), rng = StableRNG(seed))))
    end
    # the loop's no-model results (ransac_no_model_tests) and its refusals
    case!(cases, "ransac/circle_collinear_10/msac1_default", cr_collinear(), d -> (p = CircleProblem(d); ransac(p, MsacScoring(p; threshold = 1.0))))
    case!(cases, "ransac/circle_collinear_10/msac1_seed1", cr_collinear(), d -> (p = CircleProblem(d); ransac(p, MsacScoring(p; threshold = 1.0); rng = StableRNG(1))))
    case!(cases, "ransac/line_one_point_10/msac1_seed1", fill(VgtkCore.Point2(1.0, 2.0), 10),
        d -> (p = LineFittingProblem(d); ransac(p, MsacScoring(p; threshold = 1.0); rng = StableRNG(1))))
    case!(cases, "ransac/refused/terminal_lo_score_guard", hdata,
        d -> (p = HomographyProblem(d); ransac(p, MsacScoring(p; threshold = 3.0); local_optimization_strategy = TerminalLO(Irls{TLSShape}()), score_guard = true, rng = StableRNG(1))))
    case!(cases, "ransac/refused/retired_local_optimization", hdata,
        d -> (p = HomographyProblem(d); ransac(p, MsacScoring(p; threshold = 3.0); local_optimization = Irls{TLSShape}(), rng = StableRNG(1))))
    edata = cr_essential_pixels()
    for (lbl, mk) in ("msac0.02" => p -> MsacScoring(p; threshold = 0.02), "ransac0.02" => p -> RansacScoring(p; threshold = 0.02))
        case!(cases, "ransac/refused/essential_$lbl", edata, d -> (p = EssentialProblem(_CR_E.ECAM, _CR_E.ECAM, d); ransac(p, mk(p); rng = StableRNG(1))))
    end
    case!(cases, "ransac/refused/zero_threshold", ldata, d -> (p = LineFittingProblem(d); ransac(p, MsacScoring(p; threshold = 0.0); rng = StableRNG(1))))

    # ================================================================ locally_optimize, locally_optimize_rescored!
    cr_lo_cases!(cases, "homography_100+30", hdata, HomographyProblem, 3.0)
    cr_lo_cases!(cases, "fundmat_100+30", fdata, FundMatProblem, 3.0)
    cr_lo_cases!(cases, "line_200+50", ldata, LineFittingProblem, 0.5)
    cr_lo_cases!(cases, "circle_outl30_300", cdata, CircleProblem, 1.0)
    # line_fitting_tests: the contaminated line from its contaminated mean seed
    for rs in (57, 58)
        pts, θ₀ = cr_contaminated_line(StableRNG(rs))
        for (lbl, lo, thr) in (("irls_welsch_ec1.5_msac0.1", Irls{WelschShape}(efficiency_constant = 1.5), 0.1),
                               ("irls_huber_ec1.345_msac0.2", Irls{HuberShape}(efficiency_constant = 1.345), 0.2))
            case!(cases, "locally_optimize/line_contaminated_$rs/seed_mean_line_$lbl", (pts, θ₀),
                ((d, θ),) -> (p = LineFittingProblem(d); locally_optimize(p, θ; scoring = MsacScoring(p; threshold = thr), local_optimization_strategy = lo)))
            case!(cases, "rescored/line_contaminated_$rs/seed_mean_line_$lbl", (pts, θ₀),
                ((d, θ),) -> (p = LineFittingProblem(d); sc = MsacScoring(p; threshold = thr); ws = RansacWorkspace(p, sc, lo);
                              locally_optimize_rescored!(ws.lo_ws, lo, p, θ, ws.buf, sc)))
        end
    end
    pe = EssentialProblem(_CR_E.ECAM, _CR_E.ECAM, edata)
    θe = estimate(ransac(pe, MsacScoring(pe; threshold = 1.0); local_optimization_strategy = nothing, rng = StableRNG(1)))
    case!(cases, "locally_optimize/refused/essential_msac0.02", (edata, θe),
        ((d, θ),) -> (p = EssentialProblem(_CR_E.ECAM, _CR_E.ECAM, d); locally_optimize(p, θ; scoring = MsacScoring(p; threshold = 0.02), local_optimization_strategy = nothing)))
    case!(cases, "locally_optimize/essential_stereo/msac1_none", (edata, θe),
        ((d, θ),) -> (p = EssentialProblem(_CR_E.ECAM, _CR_E.ECAM, d); locally_optimize(p, θ; scoring = MsacScoring(p; threshold = 1.0), local_optimization_strategy = nothing)))
    # the thin partition (m + 1 points): the model comes back unrefined
    thin = cr_homography_thin()
    pt = HomographyProblem(thin)
    θt = estimate(ransac(pt, MsacScoring(pt; threshold = 0.5); local_optimization_strategy = nothing,
                         config = RansacConfig(max_trials = 5000, min_trials = 3000), rng = StableRNG(1)))
    lo = Irls{WelschShape}(efficiency_constant = 1.5)
    case!(cases, "rescored/homography_thin_5+12/irls_welsch_ec1.5_msac0.5", (thin, θt),
        ((d, θ),) -> (p = HomographyProblem(d); sc = MsacScoring(p; threshold = 0.5); ws = RansacWorkspace(p, sc, lo);
                      locally_optimize_rescored!(ws.lo_ws, lo, p, θ, ws.buf, sc)))
    case!(cases, "locally_optimize/homography_thin_5+12/model_irls_welsch_ec1.5", (thin, θt),
        ((d, θ),) -> (p = HomographyProblem(d); locally_optimize(p, θ; scoring = MsacScoring(p; threshold = 0.5), local_optimization_strategy = lo)))

    # ================================================================ solve(problem, idx) and fit_hypothesis
    rh = cr_radial_homography_pairs(); rhc = (600.0, 450.0)
    solve_problems = [
        "homography" => (hdata, HomographyProblem),
        "fundmat" => (fdata, FundMatProblem),
        "essential" => (edata, d -> EssentialProblem(_CR_E.ECAM, _CR_E.ECAM, d)),
        "absolute_pose" => (pnp_scene(), d -> AbsolutePoseProblem(PCAM, d)),
        "bearing_pose" => (cr_bearing_pose(), BearingPoseProblem),
        "rotation" => (cr_rotation_pixels(), d -> RotationProblem(CR_FISHEYE, CR_FISHEYE, d)),
        "radial_homography" => (rh, d -> RadialHomographyProblem(d; center = rhc)),
        "radial_homography_extra_check" => (rh, d -> RadialHomographyProblem(d; center = rhc, extra_check = true)),
        "radial_homography_minimal_homography" => (rh, d -> RadialHomographyProblem(d; center = rhc, minimal = :homography)),
        "radial_camera" => (cr_radial_board_pairs(), d -> RadialCameraProblem(d; center = _CR_D.RADIAL_BOARD_CENTER)),
        "circle" => (cdata, CircleProblem),
        "ellipse" => (last(cr_ellipse_scenes()[2]), EllipseProblem),
        "line" => (ldata, LineFittingProblem),
        "inhom_line" => (cr_inhom_line_points(), InhomLineFittingProblem),
        "hand_eye" => (cr_hand_eye_motions(), HandEyeProblem),
        "hand_eye_all_pairs" => (cr_hand_eye_motions(), d -> HandEyeProblem(d; motion_pairs = :all)),
        "bounded_homography" => (hdata, d -> BoundedProblem(HomographyProblem(d), VgtkCore.Rect(0.0, 0.0, 200.0, 150.0))),
    ]
    for (name, (data, mk)) in solve_problems
        p = mk(data)
        idxs = cr_samples(data_size(p), minimal_sample_size(p))
        case!(cases, "solve/$name/50samples", (data, idxs), ((d, idxs),) -> (p = mk(d); [VgtkEstimation.solve(p, idx) for idx in idxs]))
        case!(cases, "fit_hypothesis/$name/50samples", (data, idxs),
            ((d, idxs),) -> (p = mk(d); [[fit_hypothesis(p, θ, idx) for θ in VgtkEstimation.solve(p, idx)] for idx in idxs]))
    end
    # the generalized bearing pose has no minimal solver (refine only): refused
    let
        E = VgtkCore.EuclideanMap(RotMatrix(RotXYZ(0.3, -0.2, 0.1)), SVector(0.1, -0.05, 2.0))
        Xs = [E.R' * (SVector(0.3sin(k), 0.3cos(2k), 2.0 + 0.5sin(3k)) - E.t) for k in 1:30]
        os = [SVector(0.02 * sin(3i), 0.02 * cos(5i), 0.01 * sin(i)) for i in eachindex(Xs)]
        fs = [normalize(E.R * X + E.t - o) for (X, o) in zip(Xs, os)]
        case!(cases, "solve/refused/generalized_bearing_pose", VgtkCore.withattrs(VgtkCore.Correspondences(Xs, fs); origins = os),
            d -> VgtkEstimation.solve(BearingPoseProblem(d), [1, 2, 3]))
    end
    return cases
end
