# The old reweighted fits at 3d8a06da: fit_linear, fit_rotation, fit_rotation_averaging, fit_rigid_alignment,
# fit_robot_world_hand_eye (the robust-estimation contract's five), refine_robot_world_hand_eye,
# fit_filtered_rotation_averaging, fit_residual_mixture, simfitpp_loss_width and simfitpp_select.
using VisualGeometryToolkitCore: Irls, Triggs, HuberShape, WelschShape, CauchyShape, TLSShape, GaussianShape,
    GemanMcclureShape, Estimated, FixedPrior, FixedNu, ConsensusDof, StudentTUniformScoring, JeffreysPrior, FlatLabelPrior,
    MsacScoring, estimate, value_of
using VisualGeometryToolkitEstimation: fit_linear, fit_rotation, fit_rotation_averaging, fit_rigid_alignment,
    fit_robot_world_hand_eye, refine_robot_world_hand_eye, fit_filtered_rotation_averaging, fit_residual_mixture,
    simfitpp_loss_width, simfitpp_select, SimFitPP, with_residual_unit, MixtureEmConfig, KeepModel, WeightedRefine,
    ParameterChangeStop, mixture_em_from_toml, RotationProblem, find_rotation, uncertain_rotation

# The five contract fits, each `(fixture, kw) -> result` on its fixture
const EOF_CONTRACT_FITS = (
    ("linear", 1, s -> contract_linear(StableRNG(s)), (fx, kw) -> fit_linear(fx...; kw...)),
    ("rotation", 2, s -> contract_rotation(StableRNG(s)), (fx, kw) -> fit_rotation(fx; kw...)),
    ("rot_avg", 3, s -> contract_averaging(), (fx, kw) -> fit_rotation_averaging(fx; kw...)),
    ("rigid", 3, s -> first(contract_points(StableRNG(s))), (fx, kw) -> fit_rigid_alignment(fx; kw...)),
    ("rwhe", 6, s -> contract_handeye(StableRNG(s)), (fx, kw) -> fit_robot_world_hand_eye(fx; σ_rot = 1.0e-2, σ_trans = 1.0e-2, kw...)),
)
# the contract's settings on clean data (fit_contract_tests, termination_record_tests)
const EOF_CONTRACT_SETTINGS = [
    "l2" => (; optimization_strategy = nothing),
    "default_scale0.02" => (; scale = 0.02),
    "huber_scale1e-3" => (; optimization_strategy = Irls{HuberShape}(), scale = 1.0e-3),
    "welsch_scale0.02" => (; optimization_strategy = Irls{WelschShape}(), scale = 0.02),
    "tls_c3_scale0.02" => (; optimization_strategy = Irls{TLSShape}(efficiency_constant = 3.0), scale = 0.02),
    "huber_width0.05" => (; optimization_strategy = Irls{HuberShape}(), width = 0.05),
    "huber_cap1_scale1e-9" => (; optimization_strategy = Irls{HuberShape}(max_iter = 1), scale = 1.0e-9),
]
# the contract's refusals (fit_contract_tests "the scale input, refusals")
const EOF_CONTRACT_REFUSALS = [
    "no_scale" => (; optimization_strategy = Irls{HuberShape}()),
    "scale_and_width" => (; optimization_strategy = Irls{HuberShape}(), scale = 0.01, width = 0.02),
    "scale_zero" => (; optimization_strategy = Irls{HuberShape}(), scale = 0.0),
    "scale_negative" => (; optimization_strategy = Irls{HuberShape}(), scale = -1.0),
    "width_inf" => (; optimization_strategy = Irls{HuberShape}(), width = Inf),
    "l2_scale" => (; optimization_strategy = nothing, scale = 0.01),
    "l2_width" => (; optimization_strategy = nothing, width = 0.01),
    "triggs" => (; optimization_strategy = Triggs{WelschShape}(efficiency_constant = 2.0), scale = 0.01),
    "gaussian" => (; optimization_strategy = Irls{GaussianShape}(), scale = 0.01),
    "tls_no_constant" => (; optimization_strategy = Irls{TLSShape}(), scale = 0.01),
    "strategy_loss_width" => (; optimization_strategy = Irls{HuberShape}(loss_width = 0.1), scale = 0.01),
    "retired_irls_kw" => (; irls = Irls{HuberShape}(), scale = 0.01),
    "retired_local_optimization" => (; local_optimization = Irls{HuberShape}(), scale = 0.01),
    "efficiency_constant_kw" => (; efficiency_constant = 2.0, scale = 0.01),
    "loss_kw" => (; loss = HuberShape(), scale = 0.01),
]

function _eof_contract!(cases)
    for (fam, _, build, fit) in EOF_CONTRACT_FITS
        for seed in SEEDS
            fx = build(seed)
            sd = fam == "rot_avg" ? "" : "/seed$seed"   # the averaging scene draws no noise: one fixture
            (fam == "rot_avg" && seed > 1) && continue
            for (lbl, kw) in EOF_CONTRACT_SETTINGS
                case!(cases, "$fam/contract/$lbl$sd", fx, fx -> fit(fx, kw))
            end
            # the hand-off: an upstream result's scale (the Welsch fit at 0.02), seeded here and stored in the fixture
            up = fit(fx, (; optimization_strategy = Irls{WelschShape}(), scale = 0.02))
            case!(cases, "$fam/contract/huber_scale_upstream$sd", (fx, up),
                ((fx, up),) -> fit(fx, (; optimization_strategy = Irls{HuberShape}(), scale = up)))
        end
        fx = build(1)
        for (lbl, kw) in EOF_CONTRACT_REFUSALS
            case!(cases, "$fam/contract/refused/$lbl", fx, fx -> fit(fx, kw))
        end
        # an upstream result that recorded no scale is refused
        l2 = fit(fx, (; optimization_strategy = nothing))
        wonly = fit(fx, (; optimization_strategy = Irls{HuberShape}(), width = 0.05))
        case!(cases, "$fam/contract/refused/upstream_l2", (fx, l2), ((fx, u),) -> fit(fx, (; optimization_strategy = Irls{HuberShape}(), scale = u)))
        case!(cases, "$fam/contract/refused/upstream_width_only", (fx, wonly), ((fx, u),) -> fit(fx, (; optimization_strategy = Irls{HuberShape}(), scale = u)))
    end
    return cases
end

function _eof_linear!(cases)
    for seed in SEEDS
        sc = eof_linear_scene(seed)
        (; A) = sc
        case!(cases, "linear/clean/scale1e-3/seed$seed", (A, sc.y_clean), ((A, y),) -> fit_linear(A, y; scale = 1.0e-3))
        case!(cases, "linear/clean/l2/seed$seed", (A, sc.y_clean), ((A, y),) -> fit_linear(A, y; optimization_strategy = nothing))
        case!(cases, "linear/outliers80/scale1e-3/seed$seed", (A, sc.y_out), ((A, y),) -> fit_linear(A, y; scale = 1.0e-3))
        case!(cases, "linear/shift10_prior0/scale1e-3/seed$seed", (A, sc.y_shift, sc.p_shift), ((A, y, p),) -> fit_linear(A, y, p; scale = 1.0e-3))
        r = fit_linear(A, sc.y_loop; optimization_strategy = Irls{HuberShape}(max_iter = 7), scale = 0.01, tol = 0)
        case!(cases, "linear/loop/huber_cap7_scale0.01_tol0/seed$seed", (A, sc.y_loop),
            ((A, y),) -> fit_linear(A, y; optimization_strategy = Irls{HuberShape}(max_iter = 7), scale = 0.01, tol = 0))
        case!(cases, "linear/loop_counts/huber_cap2_scale0.01_tol0/seed$seed", (A, sc.y_loop, sc.cnt .* r.weights),
            ((A, y, p),) -> fit_linear(A, y, p; optimization_strategy = Irls{HuberShape}(max_iter = 2), scale = 0.01, tol = 0))
        case!(cases, "linear/clean/tls_c3_scale1e-3/seed$seed", (A, sc.y_clean),
            ((A, y),) -> fit_linear(A, y; optimization_strategy = Irls{TLSShape}(efficiency_constant = 3.0), scale = 1.0e-3))
        case!(cases, "linear/clean/tls_width3e-3/seed$seed", (A, sc.y_clean),
            ((A, y),) -> fit_linear(A, y; optimization_strategy = Irls{TLSShape}(), width = 3.0e-3))
        case!(cases, "linear/three_rows/scale1/seed$seed", (A, sc.y_clean, sc.p_few), ((A, y, p),) -> fit_linear(A, y, p; scale = 1.0))
        # the strategy item
        A2, y2 = eof_linear_strategy_scene(seed)
        f2 = (A2, y2)
        case!(cases, "linear/strategy/scale1e-3/seed$seed", f2, ((A, y),) -> fit_linear(A, y; scale = 1.0e-3))
        case!(cases, "linear/strategy/huber_cap3_tol0/seed$seed", f2,
            ((A, y),) -> fit_linear(A, y; optimization_strategy = Irls{HuberShape}(max_iter = 3), scale = 1.0e-3, tol = 0))
        case!(cases, "linear/strategy/welsch_c2.5/seed$seed", f2,
            ((A, y),) -> fit_linear(A, y; optimization_strategy = Irls{WelschShape}(efficiency_constant = 2.5), scale = 1.0e-3))
        case!(cases, "linear/strategy/gm_c2/seed$seed", f2,
            ((A, y),) -> fit_linear(A, y; optimization_strategy = Irls{GemanMcclureShape}(efficiency_constant = 2.0), scale = 1.0e-3))
        case!(cases, "linear/strategy/width0.004/seed$seed", f2, ((A, y),) -> fit_linear(A, y; width = 0.004))
        case!(cases, "linear/unsolvable/tls_c3_scale1e-9/seed$seed", eof_linear_unsolvable(seed),
            ((A, y),) -> fit_linear(A, y; optimization_strategy = Irls{TLSShape}(efficiency_constant = 3.0), scale = 1.0e-9))
    end
    sc = eof_linear_scene(1); A = sc.A; m = size(A, 1)
    case!(cases, "linear/refused/short_rhs", A, A -> fit_linear(A, ones(m - 1); scale = 1.0))
    case!(cases, "linear/refused/short_prior", A, A -> fit_linear(A, ones(m), ones(3); scale = 1.0))
    case!(cases, "linear/refused/tls_no_constant", (A, sc.y_clean), ((A, y),) -> fit_linear(A, y; optimization_strategy = Irls{TLSShape}(), scale = 1.0))
    case!(cases, "linear/refused/negative_prior", (A, sc.y_clean), ((A, y),) -> fit_linear(A, y, fill(-1.0, m); scale = 1.0))
    A2, y2 = eof_linear_strategy_scene(1)
    for (lbl, kw) in ("loss_kw" => (; loss = HuberShape()), "efficiency_constant_kw" => (; efficiency_constant = 2.0),
                      "max_iter_kw" => (; max_iter = 3), "irls_kw" => (; irls = Irls{HuberShape}()),
                      "retired_local_optimization" => (; local_optimization = Irls{HuberShape}()))
        case!(cases, "linear/refused/$lbl", (A2, y2), ((A, y),) -> fit_linear(A, y; kw..., scale = 1.0e-3))
    end
    case!(cases, "linear/degenerate/zeros2x3/l2", (zeros(2, 3), zeros(2)), ((A, y),) -> fit_linear(A, y; optimization_strategy = nothing))
    t = eof_linear_tiny()
    case!(cases, "linear/tiny/huber_scale0.01", t, ((A, y),) -> fit_linear(A, y; optimization_strategy = Irls{HuberShape}(), scale = 0.01))
    case!(cases, "linear/tiny/l2", t, ((A, y),) -> fit_linear(A, y; optimization_strategy = nothing))
    case!(cases, "linear/tiny/refused/triggs", t, ((A, y),) -> fit_linear(A, y; optimization_strategy = Triggs{WelschShape}(efficiency_constant = 2.5), scale = 0.01))
    case!(cases, "linear/tiny/refused/retired_local_optimization", t, ((A, y),) -> fit_linear(A, y; local_optimization = Irls{HuberShape}()))
    return cases
end

function _eof_rotation!(cases)
    cs = eof_rotation_clean()
    case!(cases, "rotation/clean/huber_scale1e-3", cs, cs -> fit_rotation(cs; optimization_strategy = Irls{HuberShape}(), scale = 1.0e-3))
    case!(cases, "rotation/clean/welsch_scale1e-3", cs, cs -> fit_rotation(cs; optimization_strategy = Irls{WelschShape}(), scale = 1.0e-3))
    case!(cases, "rotation/clean/default_scale1e-3", cs, cs -> fit_rotation(cs; scale = 1.0e-3))
    case!(cases, "rotation/clean/l2", cs, cs -> fit_rotation(cs; optimization_strategy = nothing))
    case!(cases, "rotation/clean_long/default_scale1e-3", eof_rotation_clean_long(), cs -> fit_rotation(cs; scale = 1.0e-3))
    case!(cases, "rotation/refused/one_pair", eof_rotation_one_pair(), cs -> fit_rotation(cs; scale = 1.0e-3))
    σ = 1.0e-3
    for seed in SEEDS
        cs, _ = eof_rotation_outliers(seed)
        case!(cases, "rotation/outliers20/huber_scale1e-3/seed$seed", cs, cs -> fit_rotation(cs; optimization_strategy = Irls{HuberShape}(), scale = σ))
        case!(cases, "rotation/outliers20/welsch_scale1e-3/seed$seed", cs, cs -> fit_rotation(cs; optimization_strategy = Irls{WelschShape}(), scale = σ))
        case!(cases, "rotation/outliers20_prior0/huber_scale1e-3/seed$seed", eof_rotation_outliers_prior(seed),
            cs -> fit_rotation(cs; optimization_strategy = Irls{HuberShape}(), scale = σ))
        case!(cases, "rotation/outliers20/width_rc/seed$seed", cs, cs -> fit_rotation(cs; width = 3 * 1.4826 * 1.1774 * σ))
        for k in (1, 2, 50)
            case!(cases, "rotation/outliers20/welsch_cap$(k)_scale1e-3/seed$seed", cs,
                cs -> fit_rotation(cs; optimization_strategy = Irls{WelschShape}(max_iter = k), scale = σ))
        end
        case!(cases, "rotation/outliers20/tls_no_constant/seed$seed", cs, cs -> fit_rotation(cs; optimization_strategy = Irls{TLSShape}(), scale = σ))
        case!(cases, "rotation/outliers20/tls_c3/seed$seed", cs, cs -> fit_rotation(cs; optimization_strategy = Irls{TLSShape}(efficiency_constant = 3.0), scale = σ))
        case!(cases, "rotation/outliers20/tls_c100/seed$seed", cs, cs -> fit_rotation(cs; optimization_strategy = Irls{TLSShape}(efficiency_constant = 100.0), scale = σ))
    end
    return cases
end

function _eof_rigid!(cases)
    σ = 0.005
    for seed in SEEDS
        cs = eof_rigid_cs(seed)
        case!(cases, "rigid/walk150_gross5/l2/seed$seed", cs, cs -> fit_rigid_alignment(cs))
        case!(cases, "rigid/walk145_clean/l2/seed$seed", eof_rigid_clean_subset(seed), cs -> fit_rigid_alignment(cs))
        case!(cases, "rigid/walk150_gross5/huber_scale5e-3/seed$seed", cs, cs -> fit_rigid_alignment(cs; optimization_strategy = Irls{HuberShape}(), scale = σ))
        case!(cases, "rigid/walk150_gross5/welsch_scale5e-3/seed$seed", cs, cs -> fit_rigid_alignment(cs; optimization_strategy = Irls{WelschShape}(), scale = σ))
        case!(cases, "rigid/walk150_prior0/l2/seed$seed", eof_rigid_prior(seed), cs -> fit_rigid_alignment(cs))
        pcs = eof_rigid_physical(seed)
        case!(cases, "rigid/walk150_metres/l2/seed$seed", pcs, cs -> fit_rigid_alignment(cs))
        case!(cases, "rigid/walk150_metres/huber_scale5mm/seed$seed", pcs,
            cs -> fit_rigid_alignment(cs; optimization_strategy = Irls{HuberShape}(), scale = σ * VgtkCore.m))
    end
    case!(cases, "rigid/refused/two_pairs", eof_rigid_two(), cs -> fit_rigid_alignment(cs))
    case!(cases, "rigid/refused/collinear", eof_rigid_collinear(), cs -> fit_rigid_alignment(cs))
    return cases
end

function _eof_rwhe!(cases)
    for seed in SEEDS
        case!(cases, "rwhe/noiseless_12x3/l2/seed$seed", eof_rwhe_synth(seed; N = 12, M = 3), mk -> fit_robot_world_hand_eye(mk))
        case!(cases, "rwhe/far_12x3/l2/seed$seed", eof_rwhe_far(seed), mk -> fit_robot_world_hand_eye(mk))
        mod = eof_rwhe_synth(seed; N = 40, M = 5, σ_R = 1.0e-3, σ_t = 1.0e-3, drop = 0.2)
        case!(cases, "rwhe/noisy_40x5_drop0.2/l2/seed$seed", mod, mk -> fit_robot_world_hand_eye(mk))
        case!(cases, "rwhe/noisy_40x5_drop0.2/sigma1/seed$seed", mod, mk -> fit_robot_world_hand_eye(mk; σ_rot = 1.0, σ_trans = 1.0))
        g = eof_rwhe_gross(seed)
        case!(cases, "rwhe/gross_30x2/l2/seed$seed", g, mk -> fit_robot_world_hand_eye(mk))
        case!(cases, "rwhe/gross_30x2/huber_scale1/seed$seed", g,
            mk -> fit_robot_world_hand_eye(mk; σ_rot = 2.0e-3, σ_trans = 2.0e-3, optimization_strategy = Irls{HuberShape}(), scale = 1.0))
        case!(cases, "rwhe/gross_30x2/welsch_scale1/seed$seed", g,
            mk -> fit_robot_world_hand_eye(mk; σ_rot = 2.0e-3, σ_trans = 2.0e-3, optimization_strategy = Irls{WelschShape}(), scale = 1.0))
        plain, phys = eof_rwhe_units(seed)
        case!(cases, "rwhe/units_20x3_plain/l2/seed$seed", plain, mk -> fit_robot_world_hand_eye(mk))
        case!(cases, "rwhe/units_20x3_metres/l2/seed$seed", phys, mk -> fit_robot_world_hand_eye(mk))
        case!(cases, "rwhe/units_20x3_metres/sigma_trans1mm/seed$seed", phys, mk -> fit_robot_world_hand_eye(mk; σ_trans = 1.0VgtkCore.mm))
        case!(cases, "rwhe/units_20x3_plain/sigma_trans1e-3/seed$seed", plain, mk -> fit_robot_world_hand_eye(mk; σ_trans = 1.0e-3))
    end
    case!(cases, "rwhe/pure_z/sigma1", eof_rwhe_pure_z(5), mk -> fit_robot_world_hand_eye(mk; σ_rot = 1.0, σ_trans = 1.0))
    case!(cases, "rwhe/healthy_12x2/sigma1", eof_rwhe_synth(6; N = 12, M = 2), mk -> fit_robot_world_hand_eye(mk; σ_rot = 1.0, σ_trans = 1.0))
    case!(cases, "rwhe/string_ids/l2", eof_rwhe_named(), mk -> fit_robot_world_hand_eye(mk))
    case!(cases, "rwhe/refused/underdetermined", eof_rwhe_underdetermined(), mk -> fit_robot_world_hand_eye(mk))
    case!(cases, "rwhe/refused/empty_marker", eof_rwhe_empty_marker(), mk -> fit_robot_world_hand_eye(mk))
    g = eof_rwhe_gross(1)
    case!(cases, "rwhe/refused/robust_without_sigmas", g, mk -> fit_robot_world_hand_eye(mk; optimization_strategy = Irls{HuberShape}(), scale = 1.0))
    case!(cases, "rwhe/refused/triggs", g, mk -> fit_robot_world_hand_eye(mk; σ_rot = 2.0e-3, σ_trans = 2.0e-3,
        optimization_strategy = Triggs{WelschShape}(efficiency_constant = 2.0), scale = 1.0))
    case!(cases, "rwhe/refused/l2_scale", g, mk -> fit_robot_world_hand_eye(mk; scale = 1.0))
    case!(cases, "rwhe/refused/retired_local_optimization", g, mk -> fit_robot_world_hand_eye(mk; local_optimization = Irls{HuberShape}()))
    case!(cases, "rwhe/refused/metres_sigma_trans_plain", last(eof_rwhe_units(1)), mk -> fit_robot_world_hand_eye(mk; σ_trans = 1.0e-3))
    return cases
end

function _eof_rw_refine!(cases)
    for seed in SEEDS
        case!(cases, "rwhe_refine/noiseless_12x3/l2/seed$seed", eof_rw_problem(seed; σ = 0.0, rot = 0.02, tr = 0.02),
            ((g, c, mk),) -> refine_robot_world_hand_eye(g, c, mk; optimization_strategy = nothing))
        noisy = eof_rw_problem(seed)
        case!(cases, "rwhe_refine/noisy_12x3/l2/seed$seed", noisy, ((g, c, mk),) -> refine_robot_world_hand_eye(g, c, mk; optimization_strategy = nothing))
        case!(cases, "rwhe_refine/noisy_12x3/poor_seed_l2/seed$seed", eof_rw_problem(seed; rot = 0.1, tr = 0.1),
            ((g, c, mk),) -> refine_robot_world_hand_eye(g, c, mk; optimization_strategy = nothing))
        mv = eof_rw_moved(seed)
        case!(cases, "rwhe_refine/moved5pct/l2/seed$seed", mv, ((g, c, mk),) -> refine_robot_world_hand_eye(g, c, mk; optimization_strategy = nothing))
        case!(cases, "rwhe_refine/moved5pct/default_scale0.5/seed$seed", mv, ((g, c, mk),) -> refine_robot_world_hand_eye(g, c, mk; scale = 0.5))
        case!(cases, "rwhe_refine/moved5pct/triggs_huber_scale0.5/seed$seed", mv,
            ((g, c, mk),) -> refine_robot_world_hand_eye(g, c, mk; optimization_strategy = Triggs{HuberShape}(), scale = 0.5))
        case!(cases, "rwhe_refine/moved5pct/default_scale0.5_cap1/seed$seed", mv, ((g, c, mk),) -> refine_robot_world_hand_eye(g, c, mk; scale = 0.5, max_iter = 1))
        case!(cases, "rwhe_refine/moved5pct/default_width1.5/seed$seed", mv, ((g, c, mk),) -> refine_robot_world_hand_eye(g, c, mk; width = 1.5))
    end
    mv = eof_rw_moved(1)
    for (lbl, kw) in ("no_scale" => (;), "irls" => (; optimization_strategy = Irls{HuberShape}(), scale = 0.5),
                      "l2_scale" => (; optimization_strategy = nothing, scale = 0.5), "scale_and_width" => (; scale = 0.5, width = 1.0),
                      "retired_local_optimization" => (; local_optimization = nothing))
        case!(cases, "rwhe_refine/refused/$lbl", mv, ((g, c, mk),) -> refine_robot_world_hand_eye(g, c, mk; kw...))
    end
    seed, camera, inputs = eof_rw_contract_inputs()
    for (lbl, mk) in inputs
        case!(cases, "rwhe_refine/contract/$lbl/l2", (seed, camera, mk), ((g, c, mk),) -> refine_robot_world_hand_eye(g, c, mk; optimization_strategy = nothing))
    end
    case!(cases, "rwhe_refine/pure_z/l2", eof_rw_pure_z(), ((g, c, mk),) -> refine_robot_world_hand_eye(g, c, mk; optimization_strategy = nothing))
    case!(cases, "rwhe_refine/healthy_12x2_exact_seed/l2", eof_rw_problem(8; σ = 0.0, rot = 0.0, tr = 0.0, M = 2),
        ((g, c, mk),) -> refine_robot_world_hand_eye(g, c, mk; optimization_strategy = nothing))
    for M in (6, 10)
        case!(cases, "rwhe_refine/grid_markers$M/poor_seed_l2", eof_rw_problem(1200 + M; σ = 0.0, rot = 0.1, tr = 0.1, M, layout = :grid),
            ((g, c, mk),) -> refine_robot_world_hand_eye(g, c, mk; optimization_strategy = nothing))
    end
    tr = eof_rw_fixed_scene(3, 10, 0.3)
    case!(cases, "rwhe_refine/termination_10x2/l2", tr, ((g, c, mk),) -> refine_robot_world_hand_eye(g, c, mk; optimization_strategy = nothing))
    case!(cases, "rwhe_refine/termination_10x2/triggs_huber_scale0.3", tr,
        ((g, c, mk),) -> refine_robot_world_hand_eye(g, c, mk; optimization_strategy = Triggs{HuberShape}(), scale = 0.3))
    case!(cases, "rwhe_refine/units_8x2_plain/l2", eof_rw_fixed_scene(13, 8, 0.35), ((g, c, mk),) -> refine_robot_world_hand_eye(g, c, mk; optimization_strategy = nothing))
    case!(cases, "rwhe_refine/units_8x2_metres/l2", eof_rw_physical(13), ((g, c, mk),) -> refine_robot_world_hand_eye(g, c, mk; optimization_strategy = nothing))
    case!(cases, "rwhe_refine/refused/metres_guess_plain_markers", (first(eof_rw_physical(13)), eof_rw_fixed_scene(13, 8, 0.35)[2:3]...),
        ((g, c, mk),) -> refine_robot_world_hand_eye(g, c, mk; optimization_strategy = nothing))
    return cases
end

function _eof_rot_avg!(cases)
    for seed in SEEDS
        s = eof_avg_scene(seed)
        case!(cases, "rot_avg/consistent8/default_scale2e-3/seed$seed", s.consistent, d -> fit_rotation_averaging(d; scale = 2.0e-3))
        case!(cases, "rot_avg/noisy8_bad1/cauchy_scale2e-3/seed$seed", s.noisy, d -> fit_rotation_averaging(d; optimization_strategy = Irls{CauchyShape}(), scale = 2.0e-3))
        case!(cases, "rot_avg/noisy8_bad1/default_scale2e-3/seed$seed", s.noisy, d -> fit_rotation_averaging(d; scale = 2.0e-3))
        case!(cases, "rot_avg/noisy8_bad_weight0/default_scale2e-3/seed$seed", s.dropped, d -> fit_rotation_averaging(d; scale = 2.0e-3))
        case!(cases, "rot_avg/noisy8_bad_weight0/l2/seed$seed", s.dropped, d -> fit_rotation_averaging(d; optimization_strategy = nothing))
        case!(cases, "rot_avg/hung_frame9/tls_c5_scale2e-3/seed$seed", s.hung,
            d -> fit_rotation_averaging(d; optimization_strategy = Irls{TLSShape}(efficiency_constant = 5.0), scale = 2.0e-3))
        if seed == 1
            case!(cases, "rot_avg/refused/zero_bridge", s.zero_bridge, d -> fit_rotation_averaging(d; scale = 1.0e-3))
            case!(cases, "rot_avg/refused/all_zero", s.all_zero, d -> fit_rotation_averaging(d; scale = 1.0e-3))
            case!(cases, "rot_avg/refused/disconnected", s.split, d -> fit_rotation_averaging(d; scale = 1.0e-3))
            case!(cases, "rot_avg/refused/tls_no_constant", s.single, d -> fit_rotation_averaging(d; optimization_strategy = Irls{TLSShape}(), scale = 1.0e-3))
            case!(cases, "rot_avg/refused/loss_kw", s.single, d -> fit_rotation_averaging(d; loss = CauchyShape(), scale = 1.0e-3))
            case!(cases, "rot_avg/refused/max_iter_kw", s.single, d -> fit_rotation_averaging(d; max_iter = 3, scale = 1.0e-3))
        end
        # the least squares, iterated from the spectral seed, and from that seed given (Core's spectral initializer,
        # gauged as the fit gauges it, computed here and stored in the fixture)
        ls = eof_avg_ls_scene(seed)
        case!(cases, "rot_avg/noisy8_0.02/l2/seed$seed", ls, d -> fit_rotation_averaging(d; optimization_strategy = nothing))
        S = VgtkCore.spectral_rotation_averaging(Dict(e => VgtkCore.Attributed(ls[e], (; weight = 1.0)) for e in keys(ls)))
        G = SMatrix{3, 3}(S[1])'
        sd = Dict(k => G * SMatrix{3, 3}(S[k]) for k in keys(S))
        case!(cases, "rot_avg/noisy8_0.02/l2_given_seed/seed$seed", (ls, sd), ((d, sd),) -> fit_rotation_averaging(d; optimization_strategy = nothing, seed = sd))
        # the spanning-tree seed and the ViewGraph method
        t = eof_avg_tree_scene(seed)
        case!(cases, "rot_avg/tree8/default_scale1e-3/seed$seed", t.weighted, d -> fit_rotation_averaging(d; scale = 1.0e-3))
        case!(cases, "rot_avg/tree8/spanning_tree_scale1e-3/seed$seed", t.weighted, d -> fit_rotation_averaging(d; seed = :spanning_tree, scale = 1.0e-3))
        case!(cases, "rot_avg/tree8/gauged_truth_seed/seed$seed", (t.weighted, t.gauged_truth), ((d, sd),) -> fit_rotation_averaging(d; seed = sd, scale = 1.0e-3))
        case!(cases, "rot_avg/tree8_viewgraph/default_scale1e-3/seed$seed", VgtkEstimation.ViewGraph(t.weighted), vg -> fit_rotation_averaging(vg; scale = 1.0e-3))
        case!(cases, "rot_avg/tree8_unit_weights/default_scale1e-3/seed$seed", t.noisy, d -> fit_rotation_averaging(d; scale = 1.0e-3))
        case!(cases, "rot_avg/tree8_sub/truth_seed/seed$seed", (t.sub, t.truth), ((d, sd),) -> fit_rotation_averaging(d; seed = sd, scale = 1.0e-3))
        case!(cases, "rot_avg/tree8_flipped/spanning_tree/seed$seed", t.flipped, d -> fit_rotation_averaging(d; seed = :spanning_tree, scale = 1.0e-3))
        if seed == 1
            case!(cases, "rot_avg/refused/partial_seed", (t.weighted, t.partial), ((d, sd),) -> fit_rotation_averaging(d; seed = sd, scale = 1.0e-3))
            case!(cases, "rot_avg/refused/unknown_seed_symbol", t.weighted, d -> fit_rotation_averaging(d; seed = :nonsense, scale = 1.0e-3))
            case!(cases, "rot_avg/refused/disconnected_given_seed", (t.split, t.truth), ((d, sd),) -> fit_rotation_averaging(d; seed = sd, scale = 1.0e-3))
            case!(cases, "rot_avg/refused/disconnected_spanning_tree", t.split, d -> fit_rotation_averaging(d; seed = :spanning_tree, scale = 1.0e-3))
            case!(cases, "rot_avg/refused/isolated_view", t.isolated, vg -> fit_rotation_averaging(vg; scale = 1.0e-3))
        end
        # Uncertain edges at a known covariance
        u = eof_avg_uncertain_scene(seed)
        case!(cases, "rot_avg/plain7_weak/huber_scale4e-3/seed$seed", u.rel, d -> fit_rotation_averaging(d; optimization_strategy = Irls{HuberShape}(), scale = 0.004))
        case!(cases, "rot_avg/uncertain7_weak/huber/seed$seed", u.attributed, d -> fit_rotation_averaging(d; optimization_strategy = Irls{HuberShape}()))
        case!(cases, "rot_avg/plain7_weak/default_scale4e-3/seed$seed", u.rel, d -> fit_rotation_averaging(d; scale = 0.004))
        case!(cases, "rot_avg/uncertain7_isotropic/default/seed$seed", u.eye, d -> fit_rotation_averaging(d))
        case!(cases, "rot_avg/uncertain7_vec9/default/seed$seed", u.nine, d -> fit_rotation_averaging(d))
        case!(cases, "rot_avg/uncertain7_chart3/default/seed$seed", u.three, d -> fit_rotation_averaging(d))
        seed == 1 && case!(cases, "rot_avg/refused/uncertain_2x2", u.two, d -> fit_rotation_averaging(d))
        # uncertain_rotation edges: each pixel edge's pairwise find_rotation (MSAC at 4, StableRNG(seed)) and its
        # absolute covariance, computed here and stored in the fixture
        cam = eof_edge_camera()
        pix = eof_edge_pixels(seed)
        fits = [e => (p = RotationProblem(cam, cam, cs); (p, find_rotation(cam, cam, cs; scoring = MsacScoring(p; threshold = 4.0), rng = StableRNG(seed))))
                for (e, cs) in pix]
        unc = Dict(e => uncertain_rotation(p, r) for (e, (p, r)) in fits)
        plain = Dict(e => SMatrix{3, 3}(estimate(r)) for (e, (_, r)) in fits)
        case!(cases, "rot_avg/edges5_uncertain/huber/seed$seed", unc, d -> fit_rotation_averaging(d; optimization_strategy = Irls{HuberShape}()))
        case!(cases, "rot_avg/edges5_plain/huber_scale1e-3/seed$seed", plain, d -> fit_rotation_averaging(d; optimization_strategy = Irls{HuberShape}(), scale = 1.0e-3))
        case!(cases, "rot_avg/edges5_uncertain/huber_width2/seed$seed", unc, d -> fit_rotation_averaging(d; optimization_strategy = Irls{HuberShape}(), width = 2.0))
        case!(cases, "rot_avg/edges5_uncertain/l2/seed$seed", unc, d -> fit_rotation_averaging(d; optimization_strategy = nothing))
        if seed == 1
            case!(cases, "rot_avg/refused/uncertain_with_scale", unc, d -> fit_rotation_averaging(d; optimization_strategy = Irls{HuberShape}(), scale = 1.0))
            mixed = Dict{Tuple{Int, Int}, Any}(unc); mixed[(2, 3)] = value_of(unc[(2, 3)])
            case!(cases, "rot_avg/refused/mixed_uncertain_plain", mixed, d -> fit_rotation_averaging(d; optimization_strategy = Irls{HuberShape}(), scale = 1.0e-3))
        end
    end
    return cases
end

function _eof_filtered!(cases)
    for seed in SEEDS
        vg = eof_avg_filter_graph(seed)
        case!(cases, "filtered_rot_avg/graph10_bad1/none/seed$seed", vg, vg -> fit_filtered_rotation_averaging(vg; scale = 1.0e-3))
        case!(cases, "filtered_rot_avg/graph10_bad1/orientation2deg/seed$seed", vg, vg -> fit_filtered_rotation_averaging(vg; orientation_tol = deg2rad(2.0), scale = 1.0e-3))
        case!(cases, "filtered_rot_avg/graph10_bad1/cycle2deg/seed$seed", vg, vg -> fit_filtered_rotation_averaging(vg; cycle_tol = deg2rad(2.0), scale = 1.0e-3))
        case!(cases, "filtered_rot_avg/graph10_bad1/cycle_orientation2deg/seed$seed", vg,
            vg -> fit_filtered_rotation_averaging(vg; cycle_tol = deg2rad(2.0), orientation_tol = deg2rad(2.0), scale = 1.0e-3))
        vu = eof_avg_filter_graph(seed; uncertain = true)
        case!(cases, "filtered_rot_avg/uncertain10_bad1/orientation2deg/seed$seed", vu, vg -> fit_filtered_rotation_averaging(vg; orientation_tol = deg2rad(2.0)))
        case!(cases, "filtered_rot_avg/uncertain10_bad1/cycle_orientation2deg/seed$seed", vu,
            vg -> fit_filtered_rotation_averaging(vg; cycle_tol = deg2rad(2.0), orientation_tol = deg2rad(2.0)))
    end
    vg = eof_avg_filter_graph(1)
    case!(cases, "filtered_rot_avg/refused/negative_cycle_tol", vg, vg -> fit_filtered_rotation_averaging(vg; cycle_tol = -1.0, scale = 1.0e-3))
    case!(cases, "filtered_rot_avg/refused/zero_orientation_tol", vg, vg -> fit_filtered_rotation_averaging(vg; orientation_tol = 0.0, scale = 1.0e-3))
    case!(cases, "filtered_rot_avg/consistent5/none", eof_avg_termination_graph(), vg -> fit_filtered_rotation_averaging(vg; scale = 1.0e-3))
    return cases
end

function _eof_mixture!(cases)
    a = 40.0
    for seed in SEEDS
        r = eof_mixture_residuals(seed; n_in = 400, n_out = 100, σ = 0.8, halfwidth = a)
        sc = HalfwidthScoring(2; threshold = 3.0, outlier_halfwidth = a)
        case!(cases, "residual_mix/known_400+100/halfwidth_t3/seed$seed", r, r -> fit_residual_mixture(r; weighting = sc, model_dof = 4))
        case!(cases, "residual_mix/known_400+100_f32/halfwidth_t3/seed$seed", Float32.(r), r -> fit_residual_mixture(r; weighting = sc, model_dof = 4))
        case!(cases, "residual_mix/known_400+100/thin_partition/seed$seed", r,
            r -> fit_residual_mixture(r; weighting = HalfwidthScoring(2; threshold = 1.0e-6, outlier_halfwidth = a), model_dof = 4))
        r2 = eof_mixture_residuals(seed + 6; n_in = 200, n_out = 50, σ = 1.2, halfwidth = 60.0)
        sc2 = HalfwidthScoring(2; threshold = 4.0, outlier_halfwidth = 60.0)
        case!(cases, "residual_mix/defaults_200+50/omitted_config/seed$seed", r2, r -> fit_residual_mixture(r; weighting = sc2, model_dof = 3))
        case!(cases, "residual_mix/defaults_200+50/empty_toml_config/seed$seed", r2,
            r -> fit_residual_mixture(r; weighting = sc2, model_dof = 3, config = mixture_em_from_toml(Dict{String, Any}())))
        case!(cases, "residual_mix/defaults_200+50/held_prior0.75_nu6/seed$seed", r2,
            r -> fit_residual_mixture(r; weighting = sc2, model_dof = 3,
                config = MixtureEmConfig(nuisance = (; scale = Estimated(), inlier_prior = FixedPrior(0.75), nu = FixedNu(6.0)))))
        # a calibration pipeline's call: [classification] (the image's half-diagonal, codimension 2, ν at the consensus dof)
        rc = eof_mixture_residuals(seed + 12; n_in = 900, n_out = 25, σ = 0.4, halfwidth = 300.0)
        scc = HalfwidthScoring(2; threshold = 2.0, outlier_halfwidth = hypot(4000, 3000) / 2)
        case!(cases, "residual_mix/pipeline_900+25/classification_em/seed$seed", rc,
            r -> fit_residual_mixture(r; weighting = scc, model_dof = 76, config = eof_mixture_classification_em()))
        case!(cases, "residual_mix/pipeline_900+25/consensus_dof_max_iter1/seed$seed", rc,
            r -> fit_residual_mixture(r; weighting = scc, model_dof = 76,
                config = MixtureEmConfig(nuisance = (; scale = Estimated(), inlier_prior = Estimated(), nu = ConsensusDof()), max_iter = 1)))
        rs = eof_mixture_scalar(seed)
        case!(cases, "residual_mix/scalar_200+40/student_t_uniform/seed$seed", rs,
            r -> fit_residual_mixture(r; weighting = StudentTUniformScoring(4, 40.0; codimension = 1, scale_prior = JeffreysPrior(),
                                                                            label_prior = FlatLabelPrior()), model_dof = 4))
    end
    r = eof_mixture_residuals(11; n_in = 100, n_out = 10, σ = 0.5, halfwidth = 50.0)
    case!(cases, "residual_mix/refused/weighted_refine", r,
        r -> fit_residual_mixture(r; weighting = MsacScoring(2; threshold = 2.0), model_dof = 4, config = MixtureEmConfig(model_update = WeightedRefine())))
    case!(cases, "residual_mix/refused/signed", eof_mixture_signed(), r -> fit_residual_mixture(r; weighting = MsacScoring(2; threshold = 2.0), model_dof = 4))
    sc = HalfwidthScoring(2; threshold = 3.0, outlier_halfwidth = 50.0)
    case!(cases, "residual_mix/refused/parameter_change_stop", r,
        r -> fit_residual_mixture(r; weighting = sc, model_dof = 4, config = MixtureEmConfig(stop = ParameterChangeStop(1.0e-6))))
    case!(cases, "residual_mix/refused/positional_scorer", r, r -> fit_residual_mixture(r, sc; model_dof = 4))
    case!(cases, "residual_mix/refused/scoring_kw", r, r -> fit_residual_mixture(r; weighting = sc, scoring = sc, model_dof = 4))
    return cases
end

function _eof_simfit!(cases)
    msac(D) = (lw, train) -> VgtkEstimation._msac_best_over(D, train, lw)
    sq(D) = (j, val) -> Float64[Float64(D[i, j])^2 for i in val]
    for seed in SEEDS
        D = eof_simfit_pool(seed; σ = 1.0)
        n = size(D, 1)
        case!(cases, "loss_width_cv/pool_sigma1/default/seed$seed", D, D -> simfitpp_loss_width(msac(D), sq(D), size(D, 1), 1.0, SimFitPP(); rng = StableRNG(seed)))
        case!(cases, "loss_width_cv/pool_sigma1/previous3/seed$seed", D,
            D -> simfitpp_loss_width(msac(D), sq(D), size(D, 1), 1.0, SimFitPP(); num_previous_est = 3, rng = StableRNG(seed)))
        Dsm = eof_simfit_pool(seed + 10; σ = 0.02)
        case!(cases, "loss_width_cv/pool_sigma0.02/out_of_bounds/seed$seed", Dsm,
            D -> simfitpp_loss_width(msac(D), sq(D), size(D, 1), 1.0, SimFitPP(); rng = StableRNG(seed)))
        case!(cases, "loss_width_cv/pool_sigma1/residual_unit0.5/seed$seed", D,
            D -> simfitpp_loss_width(msac(D), sq(D), size(D, 1), 1.0, with_residual_unit(SimFitPP(), 0.5); rng = StableRNG(seed)))
        case!(cases, "loss_width_cv/failing_estimator/seed$seed", n,
            n -> simfitpp_loss_width((lw, tr) -> nothing, (m, v) -> nothing, n, 1.0, SimFitPP(); rng = StableRNG(seed)))
        D8 = eof_simfit_pool(seed + 20; σ = 0.8)
        case!(cases, "pool_select/matrix_sigma0.8/default/seed$seed", D8, D -> simfitpp_select(D, 1.0; rng = StableRNG(seed)))
        fp = eof_fundmat_pool(seed)
        case!(cases, "pool_select/fundmat_pool/pixels3/seed$seed", fp, p -> simfitpp_select(p, 3.0; rng = StableRNG(seed)))
        case!(cases, "pool_select/fundmat_pool_frame/unit_pool_scale/seed$seed", fp,
            p -> simfitpp_select(VgtkCore.Pools.pixel_residuals(p) .* Float32(VgtkCore.Pools.pool_pixel_scale(p)), 0.01,
                                 with_residual_unit(SimFitPP(), VgtkCore.Pools.pool_pixel_scale(p)); rng = StableRNG(seed)))
    end
    # a pool of bare matrices has no pixel mapping: the pair method is refused (simfitpp_tests' MethodError)
    case!(cases, "pool_select/refused/bare_matrix_models", eof_identity_pool(eof_simfit_pool(21; σ = 0.8)), p -> simfitpp_select(p, 1.0; rng = StableRNG(1)))
    case!(cases, "pool_select/empty_pool", zeros(Float32, 0, 3), D -> simfitpp_select(D, 1.0; rng = StableRNG(1)))
    return cases
end

function record_est_other_fits(cases)
    _eof_contract!(cases)
    _eof_linear!(cases)
    _eof_rotation!(cases)
    _eof_rigid!(cases)
    _eof_rwhe!(cases)
    _eof_rw_refine!(cases)
    _eof_rot_avg!(cases)
    _eof_filtered!(cases)
    _eof_mixture!(cases)
    _eof_simfit!(cases)
    return cases
end
