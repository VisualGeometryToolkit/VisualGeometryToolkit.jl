# The other fits on Core 0.14 (record/est_other_fits.jl, case for case): ReweightedLeastSquares on a LinearModel, a
# rotation, a rigid alignment and a robot-world calibration; RotationAveraging and FilteredRotationAveraging; the
# LevenbergMarquardt refine of a RobotWorldHandEyeProblem; ResidualMixture; SimFitPP on a LossWidth and a
# PoolSelection problem.
using VisualGeometryToolkitCore: Irls, Triggs, HuberShape, WelschShape, CauchyShape, TLSShape, GaussianShape,
    GemanMcclureShape, Estimated, FixedPrior, FixedNu, ConsensusDof, StudentTUniformScoring, JeffreysPrior, FlatLabelPrior,
    MsacScoring, value_of, ClassDefault, LmStops, AbsoluteRotations, LossWidth, PosteriorWeightState, EuclideanMap,
    successful_retcode, weights
using VisualGeometryToolkitEstimation: ReweightedLeastSquares, LinearModel, RotationAveraging, FilteredRotationAveraging,
    RobotWorldHandEyeProblem, RobotWorldHandEyeCalibration, ResidualMixture, SimFitPP, PoolSelection, ScoreSelection, with_residual_unit,
    MixtureEmConfig, KeepModel, WeightedRefine, ParameterChangeStop, mixture_em_from_toml, RotationProblem, uncertain_rotation
using Rotations: RotMatrix
Base.include(@__MODULE__, joinpath(@__DIR__, "_common.jl"))

# A fit's old keyword set as the new solve: the algorithm's configuration (`optimization_strategy`, `tol`, `seed`) is
# its fields, the scale input (`scale`, `width`) the solve's run settings; any other keyword goes to the solve as given
# (a refusal: the solve has no such keyword).
const _FIT_FIELDS = (:optimization_strategy, :tol)
function _split(kw, fields)
    alg = NamedTuple(k => v for (k, v) in pairs(kw) if k in fields)
    run = NamedTuple(k => v for (k, v) in pairs(kw) if !(k in fields))
    return alg, run
end
fit_solve(prob; kw...) = ((a, r) = _split(kw, _FIT_FIELDS); solve(prob, ReweightedLeastSquares(; a...); r...))
avg_solve(d; kw...) = ((a, r) = _split(kw, (_FIT_FIELDS..., :seed)); solve(EstimationProblem(AbsoluteRotations, d), RotationAveraging(; a...); r...))
linear(fx; kw...) = fit_solve(EstimationProblem(LinearModel, Tuple(fx)); kw...)
rwhe(mk; σ_rot = nothing, σ_trans = nothing, min_rel_angle = nothing, kw...) =
    fit_solve(EstimationProblem(RobotWorldHandEyeCalibration, mk, NamedTuple(k => v for (k, v) in
        pairs((; σ_rot, σ_trans, min_rel_angle)) if v !== nothing)); kw...)
rwhe_refine((g, c, mk); kw...) = lm_solve(RobotWorldHandEyeProblem(c, mk), g; kw...)

const _FIT_MAP = RESULT_MAP
const _FIX2 = prefixed(RESULT_MAP, "2")
const _AVG_MAP = FieldMap(; renames = ["u" => "estimate", "stats.nsteps" => "iterations"],
                          new_only = [_NEW_FIELDS; "resid"; "metadata.graph"], retyped = ["", "metadata"])
const _AVG_FIX2 = prefixed(_AVG_MAP, "2")

const EOF_CONTRACT_FITS = (
    ("linear", 1, s -> contract_linear(StableRNG(s)), (fx, kw) -> linear(fx; kw...), _FIT_MAP),
    ("rotation", 2, s -> contract_rotation(StableRNG(s)), (fx, kw) -> fit_solve(EstimationProblem(RotMatrix{3}, fx); kw...), _FIT_MAP),
    ("rot_avg", 3, s -> contract_averaging(), (fx, kw) -> avg_solve(fx; kw...), _AVG_MAP),
    ("rigid", 3, s -> first(contract_points(StableRNG(s))), (fx, kw) -> fit_solve(EstimationProblem(EuclideanMap{3}, fx); kw...), _FIT_MAP),
    ("rwhe", 6, s -> contract_handeye(StableRNG(s)), (fx, kw) -> rwhe(fx; σ_rot = 1.0e-2, σ_trans = 1.0e-2, kw...), _FIT_MAP),
)
const EOF_CONTRACT_SETTINGS = [
    "l2" => (; optimization_strategy = nothing),
    "default_scale0.02" => (; scale = 0.02),
    "huber_scale1e-3" => (; optimization_strategy = Irls{HuberShape}(), scale = 1.0e-3),
    "welsch_scale0.02" => (; optimization_strategy = Irls{WelschShape}(), scale = 0.02),
    "tls_c3_scale0.02" => (; optimization_strategy = Irls{TLSShape}(efficiency_constant = 3.0), scale = 0.02),
    "huber_width0.05" => (; optimization_strategy = Irls{HuberShape}(), width = 0.05),
    "huber_cap1_scale1e-9" => (; optimization_strategy = Irls{HuberShape}(max_iter = 1), scale = 1.0e-9),
]
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
    for (fam, _, build, fit, map) in EOF_CONTRACT_FITS
        for seed in SEEDS
            fx = build(seed)
            sd = fam == "rot_avg" ? "" : "/seed$seed"
            (fam == "rot_avg" && seed > 1) && continue
            for (lbl, kw) in EOF_CONTRACT_SETTINGS
                # the rigid alignment's and the robot-world calibration's class default is least squares: a scale refused
                refused = fam in ("rigid", "rwhe") && lbl == "default_scale0.02"
                replay!(cases, "$fam/contract/$lbl$sd", fx, fx -> fit(fx, kw); map = refused ? REFUSAL_MAP : map)
            end
            up = fit(fx, (; optimization_strategy = Irls{WelschShape}(), scale = 0.02))
            replay!(cases, "$fam/contract/huber_scale_upstream$sd", (fx, up),
                ((fx, up),) -> fit(fx, (; optimization_strategy = Irls{HuberShape}(), scale = up)); map,
                fixture_map = prefixed(map, "2"))
        end
        fx = build(1)
        for (lbl, kw) in EOF_CONTRACT_REFUSALS
            replay!(cases, "$fam/contract/refused/$lbl", fx, fx -> fit(fx, kw); map = REFUSAL_MAP)
        end
        l2 = fit(fx, (; optimization_strategy = nothing))
        wonly = fit(fx, (; optimization_strategy = Irls{HuberShape}(), width = 0.05))
        replay!(cases, "$fam/contract/refused/upstream_l2", (fx, l2), ((fx, u),) -> fit(fx, (; optimization_strategy = Irls{HuberShape}(), scale = u));
            map = REFUSAL_MAP, fixture_map = prefixed(map, "2"))
        replay!(cases, "$fam/contract/refused/upstream_width_only", (fx, wonly), ((fx, u),) -> fit(fx, (; optimization_strategy = Irls{HuberShape}(), scale = u));
            map = REFUSAL_MAP, fixture_map = prefixed(map, "2"))
    end
    return cases
end

function _eof_linear!(cases)
    M = _FIT_MAP
    for seed in SEEDS
        sc = eof_linear_scene(seed)
        (; A) = sc
        replay!(cases, "linear/clean/scale1e-3/seed$seed", (A, sc.y_clean), fx -> linear(fx; scale = 1.0e-3); map = M)
        replay!(cases, "linear/clean/l2/seed$seed", (A, sc.y_clean), fx -> linear(fx; optimization_strategy = nothing); map = M)
        replay!(cases, "linear/outliers80/scale1e-3/seed$seed", (A, sc.y_out), fx -> linear(fx; scale = 1.0e-3); map = M)
        replay!(cases, "linear/shift10_prior0/scale1e-3/seed$seed", (A, sc.y_shift, sc.p_shift), fx -> linear(fx; scale = 1.0e-3); map = M)
        r = linear((A, sc.y_loop); optimization_strategy = Irls{HuberShape}(max_iter = 7), scale = 0.01, tol = 0)
        replay!(cases, "linear/loop/huber_cap7_scale0.01_tol0/seed$seed", (A, sc.y_loop),
            fx -> linear(fx; optimization_strategy = Irls{HuberShape}(max_iter = 7), scale = 0.01, tol = 0); map = M)
        replay!(cases, "linear/loop_counts/huber_cap2_scale0.01_tol0/seed$seed", (A, sc.y_loop, sc.cnt .* weights(r)),
            fx -> linear(fx; optimization_strategy = Irls{HuberShape}(max_iter = 2), scale = 0.01, tol = 0); map = M)
        replay!(cases, "linear/clean/tls_c3_scale1e-3/seed$seed", (A, sc.y_clean),
            fx -> linear(fx; optimization_strategy = Irls{TLSShape}(efficiency_constant = 3.0), scale = 1.0e-3); map = M)
        replay!(cases, "linear/clean/tls_width3e-3/seed$seed", (A, sc.y_clean),
            fx -> linear(fx; optimization_strategy = Irls{TLSShape}(), width = 3.0e-3); map = M)
        replay!(cases, "linear/three_rows/scale1/seed$seed", (A, sc.y_clean, sc.p_few), fx -> linear(fx; scale = 1.0); map = M)
        A2, y2 = eof_linear_strategy_scene(seed)
        f2 = (A2, y2)
        replay!(cases, "linear/strategy/scale1e-3/seed$seed", f2, fx -> linear(fx; scale = 1.0e-3); map = M)
        replay!(cases, "linear/strategy/huber_cap3_tol0/seed$seed", f2,
            fx -> linear(fx; optimization_strategy = Irls{HuberShape}(max_iter = 3), scale = 1.0e-3, tol = 0); map = M)
        replay!(cases, "linear/strategy/welsch_c2.5/seed$seed", f2,
            fx -> linear(fx; optimization_strategy = Irls{WelschShape}(efficiency_constant = 2.5), scale = 1.0e-3); map = M)
        replay!(cases, "linear/strategy/gm_c2/seed$seed", f2,
            fx -> linear(fx; optimization_strategy = Irls{GemanMcclureShape}(efficiency_constant = 2.0), scale = 1.0e-3); map = M)
        replay!(cases, "linear/strategy/width0.004/seed$seed", f2, fx -> linear(fx; width = 0.004); map = M)
        replay!(cases, "linear/unsolvable/tls_c3_scale1e-9/seed$seed", eof_linear_unsolvable(seed),
            fx -> linear(fx; optimization_strategy = Irls{TLSShape}(efficiency_constant = 3.0), scale = 1.0e-9); map = M)
    end
    sc = eof_linear_scene(1); A = sc.A; m = size(A, 1)
    replay!(cases, "linear/refused/short_rhs", A, A -> linear((A, ones(m - 1)); scale = 1.0); map = REFUSAL_MAP)
    replay!(cases, "linear/refused/short_prior", A, A -> linear((A, ones(m), ones(3)); scale = 1.0); map = REFUSAL_MAP)
    replay!(cases, "linear/refused/tls_no_constant", (A, sc.y_clean), fx -> linear(fx; optimization_strategy = Irls{TLSShape}(), scale = 1.0); map = REFUSAL_MAP)
    replay!(cases, "linear/refused/negative_prior", (A, sc.y_clean), ((A, y),) -> linear((A, y, fill(-1.0, m)); scale = 1.0); map = REFUSAL_MAP)
    A2, y2 = eof_linear_strategy_scene(1)
    for (lbl, kw) in ("loss_kw" => (; loss = HuberShape()), "efficiency_constant_kw" => (; efficiency_constant = 2.0),
                      "max_iter_kw" => (; max_iter = 3), "irls_kw" => (; irls = Irls{HuberShape}()),
                      "retired_local_optimization" => (; local_optimization = Irls{HuberShape}()))
        replay!(cases, "linear/refused/$lbl", (A2, y2), fx -> linear(fx; kw..., scale = 1.0e-3); map = REFUSAL_MAP)
    end
    replay!(cases, "linear/degenerate/zeros2x3/l2", (zeros(2, 3), zeros(2)), fx -> linear(fx; optimization_strategy = nothing); map = M)
    t = eof_linear_tiny()
    replay!(cases, "linear/tiny/huber_scale0.01", t, fx -> linear(fx; optimization_strategy = Irls{HuberShape}(), scale = 0.01); map = M)
    replay!(cases, "linear/tiny/l2", t, fx -> linear(fx; optimization_strategy = nothing); map = M)
    replay!(cases, "linear/tiny/refused/triggs", t, fx -> linear(fx; optimization_strategy = Triggs{WelschShape}(efficiency_constant = 2.5), scale = 0.01); map = REFUSAL_MAP)
    replay!(cases, "linear/tiny/refused/retired_local_optimization", t, fx -> linear(fx; local_optimization = Irls{HuberShape}()); map = REFUSAL_MAP)
    return cases
end

rot(cs; kw...) = fit_solve(EstimationProblem(RotMatrix{3}, cs); kw...)
function _eof_rotation!(cases)
    M = _FIT_MAP
    cs = eof_rotation_clean()
    replay!(cases, "rotation/clean/huber_scale1e-3", cs, cs -> rot(cs; optimization_strategy = Irls{HuberShape}(), scale = 1.0e-3); map = M)
    replay!(cases, "rotation/clean/welsch_scale1e-3", cs, cs -> rot(cs; optimization_strategy = Irls{WelschShape}(), scale = 1.0e-3); map = M)
    replay!(cases, "rotation/clean/default_scale1e-3", cs, cs -> rot(cs; scale = 1.0e-3); map = M)
    replay!(cases, "rotation/clean/l2", cs, cs -> rot(cs; optimization_strategy = nothing); map = M)
    replay!(cases, "rotation/clean_long/default_scale1e-3", eof_rotation_clean_long(), cs -> rot(cs; scale = 1.0e-3); map = M)
    replay!(cases, "rotation/refused/one_pair", eof_rotation_one_pair(), cs -> rot(cs; scale = 1.0e-3); map = REFUSAL_MAP)
    σ = 1.0e-3
    for seed in SEEDS
        cs, _ = eof_rotation_outliers(seed)
        replay!(cases, "rotation/outliers20/huber_scale1e-3/seed$seed", cs, cs -> rot(cs; optimization_strategy = Irls{HuberShape}(), scale = σ); map = M)
        replay!(cases, "rotation/outliers20/welsch_scale1e-3/seed$seed", cs, cs -> rot(cs; optimization_strategy = Irls{WelschShape}(), scale = σ); map = M)
        replay!(cases, "rotation/outliers20_prior0/huber_scale1e-3/seed$seed", eof_rotation_outliers_prior(seed),
            cs -> rot(cs; optimization_strategy = Irls{HuberShape}(), scale = σ); map = M)
        replay!(cases, "rotation/outliers20/width_rc/seed$seed", cs, cs -> rot(cs; width = 3 * 1.4826 * 1.1774 * σ); map = M)
        for k in (1, 2, 50)
            replay!(cases, "rotation/outliers20/welsch_cap$(k)_scale1e-3/seed$seed", cs,
                cs -> rot(cs; optimization_strategy = Irls{WelschShape}(max_iter = k), scale = σ); map = M)
        end
        replay!(cases, "rotation/outliers20/tls_no_constant/seed$seed", cs, cs -> rot(cs; optimization_strategy = Irls{TLSShape}(), scale = σ); map = REFUSAL_MAP)
        replay!(cases, "rotation/outliers20/tls_c3/seed$seed", cs, cs -> rot(cs; optimization_strategy = Irls{TLSShape}(efficiency_constant = 3.0), scale = σ); map = M)
        replay!(cases, "rotation/outliers20/tls_c100/seed$seed", cs, cs -> rot(cs; optimization_strategy = Irls{TLSShape}(efficiency_constant = 100.0), scale = σ); map = M)
    end
    return cases
end

rigid(cs; kw...) = fit_solve(EstimationProblem(EuclideanMap{3}, cs); kw...)
function _eof_rigid!(cases)
    M = _FIT_MAP
    σ = 0.005
    for seed in SEEDS
        cs = eof_rigid_cs(seed)
        replay!(cases, "rigid/walk150_gross5/l2/seed$seed", cs, cs -> rigid(cs); map = M)
        replay!(cases, "rigid/walk145_clean/l2/seed$seed", eof_rigid_clean_subset(seed), cs -> rigid(cs); map = M)
        replay!(cases, "rigid/walk150_gross5/huber_scale5e-3/seed$seed", cs, cs -> rigid(cs; optimization_strategy = Irls{HuberShape}(), scale = σ); map = M)
        replay!(cases, "rigid/walk150_gross5/welsch_scale5e-3/seed$seed", cs, cs -> rigid(cs; optimization_strategy = Irls{WelschShape}(), scale = σ); map = M)
        replay!(cases, "rigid/walk150_prior0/l2/seed$seed", eof_rigid_prior(seed), cs -> rigid(cs); map = M)
        pcs = eof_rigid_physical(seed)
        replay!(cases, "rigid/walk150_metres/l2/seed$seed", pcs, cs -> rigid(cs); map = M)
        replay!(cases, "rigid/walk150_metres/huber_scale5mm/seed$seed", pcs,
            cs -> rigid(cs; optimization_strategy = Irls{HuberShape}(), scale = σ * VgtkCore.m); map = M)
    end
    replay!(cases, "rigid/refused/two_pairs", eof_rigid_two(), cs -> rigid(cs); map = REFUSAL_MAP)
    replay!(cases, "rigid/refused/collinear", eof_rigid_collinear(), cs -> rigid(cs); map = REFUSAL_MAP)
    return cases
end

function _eof_rwhe!(cases)
    M = _FIT_MAP
    for seed in SEEDS
        replay!(cases, "rwhe/noiseless_12x3/l2/seed$seed", eof_rwhe_synth(seed; N = 12, M = 3), mk -> rwhe(mk); map = M)
        replay!(cases, "rwhe/far_12x3/l2/seed$seed", eof_rwhe_far(seed), mk -> rwhe(mk); map = M)
        mod = eof_rwhe_synth(seed; N = 40, M = 5, σ_R = 1.0e-3, σ_t = 1.0e-3, drop = 0.2)
        replay!(cases, "rwhe/noisy_40x5_drop0.2/l2/seed$seed", mod, mk -> rwhe(mk); map = M)
        replay!(cases, "rwhe/noisy_40x5_drop0.2/sigma1/seed$seed", mod, mk -> rwhe(mk; σ_rot = 1.0, σ_trans = 1.0); map = M)
        g = eof_rwhe_gross(seed)
        replay!(cases, "rwhe/gross_30x2/l2/seed$seed", g, mk -> rwhe(mk); map = M)
        replay!(cases, "rwhe/gross_30x2/huber_scale1/seed$seed", g,
            mk -> rwhe(mk; σ_rot = 2.0e-3, σ_trans = 2.0e-3, optimization_strategy = Irls{HuberShape}(), scale = 1.0); map = M)
        # (a redescending loss whose least-squares seed lies outside its width refuses on one draw, an ArgumentError whose
        # message named the old function: its type is compared, its message not, as every refusal's)
        replay!(cases, "rwhe/gross_30x2/welsch_scale1/seed$seed", g,
            mk -> rwhe(mk; σ_rot = 2.0e-3, σ_trans = 2.0e-3, optimization_strategy = Irls{WelschShape}(), scale = 1.0);
            map = seed == 3 ? REFUSAL_MAP : M)
        plain, phys = eof_rwhe_units(seed)
        replay!(cases, "rwhe/units_20x3_plain/l2/seed$seed", plain, mk -> rwhe(mk); map = M)
        replay!(cases, "rwhe/units_20x3_metres/l2/seed$seed", phys, mk -> rwhe(mk); map = M)
        replay!(cases, "rwhe/units_20x3_metres/sigma_trans1mm/seed$seed", phys, mk -> rwhe(mk; σ_trans = 1.0VgtkCore.mm); map = M)
        replay!(cases, "rwhe/units_20x3_plain/sigma_trans1e-3/seed$seed", plain, mk -> rwhe(mk; σ_trans = 1.0e-3); map = M)
    end
    replay!(cases, "rwhe/pure_z/sigma1", eof_rwhe_pure_z(5), mk -> rwhe(mk; σ_rot = 1.0, σ_trans = 1.0); map = M)
    replay!(cases, "rwhe/healthy_12x2/sigma1", eof_rwhe_synth(6; N = 12, M = 2), mk -> rwhe(mk; σ_rot = 1.0, σ_trans = 1.0); map = M)
    replay!(cases, "rwhe/string_ids/l2", eof_rwhe_named(), mk -> rwhe(mk); map = M)
    replay!(cases, "rwhe/refused/underdetermined", eof_rwhe_underdetermined(), mk -> rwhe(mk); map = REFUSAL_MAP)
    replay!(cases, "rwhe/refused/empty_marker", eof_rwhe_empty_marker(), mk -> rwhe(mk); map = REFUSAL_MAP)
    g = eof_rwhe_gross(1)
    replay!(cases, "rwhe/refused/robust_without_sigmas", g, mk -> rwhe(mk; optimization_strategy = Irls{HuberShape}(), scale = 1.0); map = REFUSAL_MAP)
    replay!(cases, "rwhe/refused/triggs", g, mk -> rwhe(mk; σ_rot = 2.0e-3, σ_trans = 2.0e-3,
        optimization_strategy = Triggs{WelschShape}(efficiency_constant = 2.0), scale = 1.0); map = REFUSAL_MAP)
    replay!(cases, "rwhe/refused/l2_scale", g, mk -> rwhe(mk; scale = 1.0); map = REFUSAL_MAP)
    replay!(cases, "rwhe/refused/retired_local_optimization", g, mk -> rwhe(mk; local_optimization = Irls{HuberShape}()); map = REFUSAL_MAP)
    replay!(cases, "rwhe/refused/metres_sigma_trans_plain", last(eof_rwhe_units(1)), mk -> rwhe(mk; σ_trans = 1.0e-3); map = REFUSAL_MAP)
    return cases
end

# The old `max_iter` keyword: the stops of record with that cap.
_rw_cap(fx, k) = (s = VgtkCore.default_stops(RobotWorldHandEyeProblem(fx[2], fx[3])); LmStops(; max_iter = k, ftol = s.ftol, xtol = s.xtol))
function _eof_rw_refine!(cases)
    M = _FIT_MAP
    for seed in SEEDS
        replay!(cases, "rwhe_refine/noiseless_12x3/l2/seed$seed", eof_rw_problem(seed; σ = 0.0, rot = 0.02, tr = 0.02),
            fx -> rwhe_refine(fx; optimization_strategy = nothing); map = M)
        noisy = eof_rw_problem(seed)
        replay!(cases, "rwhe_refine/noisy_12x3/l2/seed$seed", noisy, fx -> rwhe_refine(fx; optimization_strategy = nothing); map = M)
        replay!(cases, "rwhe_refine/noisy_12x3/poor_seed_l2/seed$seed", eof_rw_problem(seed; rot = 0.1, tr = 0.1),
            fx -> rwhe_refine(fx; optimization_strategy = nothing); map = M)
        mv = eof_rw_moved(seed)
        replay!(cases, "rwhe_refine/moved5pct/l2/seed$seed", mv, fx -> rwhe_refine(fx; optimization_strategy = nothing); map = M)
        replay!(cases, "rwhe_refine/moved5pct/default_scale0.5/seed$seed", mv, fx -> rwhe_refine(fx; scale = 0.5); map = M)
        replay!(cases, "rwhe_refine/moved5pct/triggs_huber_scale0.5/seed$seed", mv,
            fx -> rwhe_refine(fx; optimization_strategy = Triggs{HuberShape}(), scale = 0.5); map = M)
        replay!(cases, "rwhe_refine/moved5pct/default_scale0.5_cap1/seed$seed", mv, fx -> rwhe_refine(fx; scale = 0.5, stops = _rw_cap(fx, 1)); map = M)
        replay!(cases, "rwhe_refine/moved5pct/default_width1.5/seed$seed", mv, fx -> rwhe_refine(fx; width = 1.5); map = M)
    end
    mv = eof_rw_moved(1)
    for (lbl, kw) in ("no_scale" => (;), "irls" => (; optimization_strategy = Irls{HuberShape}(), scale = 0.5),
                      "l2_scale" => (; optimization_strategy = nothing, scale = 0.5), "scale_and_width" => (; scale = 0.5, width = 1.0),
                      "retired_local_optimization" => (; local_optimization = nothing))
        replay!(cases, "rwhe_refine/refused/$lbl", mv, fx -> rwhe_refine(fx; kw...); map = REFUSAL_MAP)
    end
    seed, camera, inputs = eof_rw_contract_inputs()
    for (lbl, mk) in inputs
        replay!(cases, "rwhe_refine/contract/$lbl/l2", (seed, camera, mk), fx -> rwhe_refine(fx; optimization_strategy = nothing); map = M)
    end
    replay!(cases, "rwhe_refine/pure_z/l2", eof_rw_pure_z(), fx -> rwhe_refine(fx; optimization_strategy = nothing); map = M)
    replay!(cases, "rwhe_refine/healthy_12x2_exact_seed/l2", eof_rw_problem(8; σ = 0.0, rot = 0.0, tr = 0.0, M = 2),
        fx -> rwhe_refine(fx; optimization_strategy = nothing); map = M)
    for M_ in (6, 10)
        replay!(cases, "rwhe_refine/grid_markers$M_/poor_seed_l2", eof_rw_problem(1200 + M_; σ = 0.0, rot = 0.1, tr = 0.1, M = M_, layout = :grid),
            fx -> rwhe_refine(fx; optimization_strategy = nothing); map = M)
    end
    tr = eof_rw_fixed_scene(3, 10, 0.3)
    replay!(cases, "rwhe_refine/termination_10x2/l2", tr, fx -> rwhe_refine(fx; optimization_strategy = nothing); map = M)
    replay!(cases, "rwhe_refine/termination_10x2/triggs_huber_scale0.3", tr,
        fx -> rwhe_refine(fx; optimization_strategy = Triggs{HuberShape}(), scale = 0.3); map = M)
    replay!(cases, "rwhe_refine/units_8x2_plain/l2", eof_rw_fixed_scene(13, 8, 0.35), fx -> rwhe_refine(fx; optimization_strategy = nothing); map = M)
    replay!(cases, "rwhe_refine/units_8x2_metres/l2", eof_rw_physical(13), fx -> rwhe_refine(fx; optimization_strategy = nothing); map = M)
    replay!(cases, "rwhe_refine/refused/metres_guess_plain_markers", (first(eof_rw_physical(13)), eof_rw_fixed_scene(13, 8, 0.35)[2:3]...),
        fx -> rwhe_refine(fx; optimization_strategy = nothing); map = REFUSAL_MAP)
    return cases
end

function _eof_rot_avg!(cases)
    M = _AVG_MAP
    for seed in SEEDS
        s = eof_avg_scene(seed)
        replay!(cases, "rot_avg/consistent8/default_scale2e-3/seed$seed", s.consistent, d -> avg_solve(d; scale = 2.0e-3); map = M)
        replay!(cases, "rot_avg/noisy8_bad1/cauchy_scale2e-3/seed$seed", s.noisy, d -> avg_solve(d; optimization_strategy = Irls{CauchyShape}(), scale = 2.0e-3); map = M)
        replay!(cases, "rot_avg/noisy8_bad1/default_scale2e-3/seed$seed", s.noisy, d -> avg_solve(d; scale = 2.0e-3); map = M)
        replay!(cases, "rot_avg/noisy8_bad_weight0/default_scale2e-3/seed$seed", s.dropped, d -> avg_solve(d; scale = 2.0e-3); map = M)
        replay!(cases, "rot_avg/noisy8_bad_weight0/l2/seed$seed", s.dropped, d -> avg_solve(d; optimization_strategy = nothing); map = M)
        replay!(cases, "rot_avg/hung_frame9/tls_c5_scale2e-3/seed$seed", s.hung,
            d -> avg_solve(d; optimization_strategy = Irls{TLSShape}(efficiency_constant = 5.0), scale = 2.0e-3); map = M)
        if seed == 1
            replay!(cases, "rot_avg/refused/zero_bridge", s.zero_bridge, d -> avg_solve(d; scale = 1.0e-3); map = REFUSAL_MAP)
            replay!(cases, "rot_avg/refused/all_zero", s.all_zero, d -> avg_solve(d; scale = 1.0e-3); map = REFUSAL_MAP)
            replay!(cases, "rot_avg/refused/disconnected", s.split, d -> avg_solve(d; scale = 1.0e-3); map = REFUSAL_MAP)
            replay!(cases, "rot_avg/refused/tls_no_constant", s.single, d -> avg_solve(d; optimization_strategy = Irls{TLSShape}(), scale = 1.0e-3); map = REFUSAL_MAP)
            replay!(cases, "rot_avg/refused/loss_kw", s.single, d -> avg_solve(d; loss = CauchyShape(), scale = 1.0e-3); map = REFUSAL_MAP)
            replay!(cases, "rot_avg/refused/max_iter_kw", s.single, d -> avg_solve(d; max_iter = 3, scale = 1.0e-3); map = REFUSAL_MAP)
        end
        ls = eof_avg_ls_scene(seed)
        replay!(cases, "rot_avg/noisy8_0.02/l2/seed$seed", ls, d -> avg_solve(d; optimization_strategy = nothing); map = M)
        S = solve(EstimationProblem(AbsoluteRotations, Dict(e => VgtkCore.Attributed(ls[e], (; weight = 1.0)) for e in keys(ls))), VgtkCore.SpectralAveraging()).u
        G = SMatrix{3, 3}(S[1])'
        sd = Dict(k => G * SMatrix{3, 3}(S[k]) for k in keys(S))
        replay!(cases, "rot_avg/noisy8_0.02/l2_given_seed/seed$seed", (ls, sd), ((d, sd),) -> avg_solve(d; optimization_strategy = nothing, seed = sd); map = M)
        t = eof_avg_tree_scene(seed)
        replay!(cases, "rot_avg/tree8/default_scale1e-3/seed$seed", t.weighted, d -> avg_solve(d; scale = 1.0e-3); map = M)
        replay!(cases, "rot_avg/tree8/spanning_tree_scale1e-3/seed$seed", t.weighted, d -> avg_solve(d; seed = :spanning_tree, scale = 1.0e-3); map = M)
        replay!(cases, "rot_avg/tree8/gauged_truth_seed/seed$seed", (t.weighted, t.gauged_truth), ((d, sd),) -> avg_solve(d; seed = sd, scale = 1.0e-3); map = M)
        replay!(cases, "rot_avg/tree8_viewgraph/default_scale1e-3/seed$seed", VgtkEstimation.ViewGraph(t.weighted), vg -> avg_solve(vg; scale = 1.0e-3); map = M)
        replay!(cases, "rot_avg/tree8_unit_weights/default_scale1e-3/seed$seed", t.noisy, d -> avg_solve(d; scale = 1.0e-3); map = M)
        replay!(cases, "rot_avg/tree8_sub/truth_seed/seed$seed", (t.sub, t.truth), ((d, sd),) -> avg_solve(d; seed = sd, scale = 1.0e-3); map = M)
        replay!(cases, "rot_avg/tree8_flipped/spanning_tree/seed$seed", t.flipped, d -> avg_solve(d; seed = :spanning_tree, scale = 1.0e-3); map = M)
        if seed == 1
            replay!(cases, "rot_avg/refused/partial_seed", (t.weighted, t.partial), ((d, sd),) -> avg_solve(d; seed = sd, scale = 1.0e-3); map = REFUSAL_MAP)
            replay!(cases, "rot_avg/refused/unknown_seed_symbol", t.weighted, d -> avg_solve(d; seed = :nonsense, scale = 1.0e-3); map = REFUSAL_MAP)
            replay!(cases, "rot_avg/refused/disconnected_given_seed", (t.split, t.truth), ((d, sd),) -> avg_solve(d; seed = sd, scale = 1.0e-3); map = REFUSAL_MAP)
            replay!(cases, "rot_avg/refused/disconnected_spanning_tree", t.split, d -> avg_solve(d; seed = :spanning_tree, scale = 1.0e-3); map = REFUSAL_MAP)
            replay!(cases, "rot_avg/refused/isolated_view", t.isolated, vg -> avg_solve(vg; scale = 1.0e-3); map = REFUSAL_MAP)
        end
        u = eof_avg_uncertain_scene(seed)
        replay!(cases, "rot_avg/plain7_weak/huber_scale4e-3/seed$seed", u.rel, d -> avg_solve(d; optimization_strategy = Irls{HuberShape}(), scale = 0.004); map = M)
        replay!(cases, "rot_avg/uncertain7_weak/huber/seed$seed", u.attributed, d -> avg_solve(d; optimization_strategy = Irls{HuberShape}()); map = M)
        replay!(cases, "rot_avg/plain7_weak/default_scale4e-3/seed$seed", u.rel, d -> avg_solve(d; scale = 0.004); map = M)
        replay!(cases, "rot_avg/uncertain7_isotropic/default/seed$seed", u.eye, d -> avg_solve(d); map = M)
        replay!(cases, "rot_avg/uncertain7_vec9/default/seed$seed", u.nine, d -> avg_solve(d); map = M)
        replay!(cases, "rot_avg/uncertain7_chart3/default/seed$seed", u.three, d -> avg_solve(d); map = M)
        seed == 1 && replay!(cases, "rot_avg/refused/uncertain_2x2", u.two, d -> avg_solve(d); map = REFUSAL_MAP)
        # uncertain_rotation edges from each pixel edge's pairwise rotation find (MSAC at 4, StableRNG(seed)) — the
        # RANSAC solve of Core 0.14
        cam = eof_edge_camera()
        pix = eof_edge_pixels(seed)
        fits = [e => (p = RotationProblem(cam, cam, cs); (p, ransac_solve(p; scoring = MsacScoring(p; threshold = 4.0), rng = StableRNG(seed))))
                for (e, cs) in pix]
        unc = Dict(e => uncertain_rotation(p, r) for (e, (p, r)) in fits)
        plain = Dict(e => SMatrix{3, 3}(r.u) for (e, (_, r)) in fits)
        replay!(cases, "rot_avg/edges5_uncertain/huber/seed$seed", unc, d -> avg_solve(d; optimization_strategy = Irls{HuberShape}()); map = M)
        replay!(cases, "rot_avg/edges5_plain/huber_scale1e-3/seed$seed", plain, d -> avg_solve(d; optimization_strategy = Irls{HuberShape}(), scale = 1.0e-3); map = M)
        replay!(cases, "rot_avg/edges5_uncertain/huber_width2/seed$seed", unc, d -> avg_solve(d; optimization_strategy = Irls{HuberShape}(), width = 2.0); map = M)
        replay!(cases, "rot_avg/edges5_uncertain/l2/seed$seed", unc, d -> avg_solve(d; optimization_strategy = nothing); map = M)
        if seed == 1
            replay!(cases, "rot_avg/refused/uncertain_with_scale", unc, d -> avg_solve(d; optimization_strategy = Irls{HuberShape}(), scale = 1.0); map = REFUSAL_MAP)
            mixed = Dict{Tuple{Int, Int}, Any}(unc); mixed[(2, 3)] = value_of(unc[(2, 3)])
            replay!(cases, "rot_avg/refused/mixed_uncertain_plain", mixed, d -> avg_solve(d; optimization_strategy = Irls{HuberShape}(), scale = 1.0e-3); map = REFUSAL_MAP)
        end
    end
    return cases
end

# The filtered averaging returned `(; fit, graph)`; its result now records the graph in its metadata. The replay
# presents the new result in the old shape (`fit` the result, `graph` a copy of its `metadata.graph` — a copy because
# the walker records a value it meets a second time as a cycle): every leaf from the new result, the recorded graph
# compared bit for bit at `graph` (the copy of `fit.metadata.graph`, which is new).
const _FILTERED_MAP = FieldMap(; renames = ["fit.u" => "fit.estimate", "fit.stats.nsteps" => "fit.iterations"],
    new_only = ["fit.prob", "fit.alg", "fit.original", "fit.stats", "fit.resid", "fit.metadata.graph"], retyped = ["", "fit", "fit.metadata"])
function filtered(vg; cycle_tol = nothing, orientation_tol = nothing, kw...)
    r = solve(EstimationProblem(AbsoluteRotations, vg), FilteredRotationAveraging(; cycle_tol, orientation_tol); kw...)
    return (; fit = r, graph = deepcopy(r.metadata.graph))
end
function _eof_filtered!(cases)
    M = _FILTERED_MAP
    for seed in SEEDS
        vg = eof_avg_filter_graph(seed)
        replay!(cases, "filtered_rot_avg/graph10_bad1/none/seed$seed", vg, vg -> filtered(vg; scale = 1.0e-3); map = M)
        replay!(cases, "filtered_rot_avg/graph10_bad1/orientation2deg/seed$seed", vg, vg -> filtered(vg; orientation_tol = deg2rad(2.0), scale = 1.0e-3); map = M)
        replay!(cases, "filtered_rot_avg/graph10_bad1/cycle2deg/seed$seed", vg, vg -> filtered(vg; cycle_tol = deg2rad(2.0), scale = 1.0e-3); map = M)
        replay!(cases, "filtered_rot_avg/graph10_bad1/cycle_orientation2deg/seed$seed", vg,
            vg -> filtered(vg; cycle_tol = deg2rad(2.0), orientation_tol = deg2rad(2.0), scale = 1.0e-3); map = M)
        vu = eof_avg_filter_graph(seed; uncertain = true)
        replay!(cases, "filtered_rot_avg/uncertain10_bad1/orientation2deg/seed$seed", vu, vg -> filtered(vg; orientation_tol = deg2rad(2.0)); map = M)
        replay!(cases, "filtered_rot_avg/uncertain10_bad1/cycle_orientation2deg/seed$seed", vu,
            vg -> filtered(vg; cycle_tol = deg2rad(2.0), orientation_tol = deg2rad(2.0)); map = M)
    end
    vg = eof_avg_filter_graph(1)
    replay!(cases, "filtered_rot_avg/refused/negative_cycle_tol", vg, vg -> filtered(vg; cycle_tol = -1.0, scale = 1.0e-3); map = REFUSAL_MAP)
    replay!(cases, "filtered_rot_avg/refused/zero_orientation_tol", vg, vg -> filtered(vg; orientation_tol = 0.0, scale = 1.0e-3); map = REFUSAL_MAP)
    replay!(cases, "filtered_rot_avg/consistent5/none", eof_avg_termination_graph(), vg -> filtered(vg; scale = 1.0e-3); map = M)
    return cases
end

mixture(r; weighting, model_dof, config = MixtureEmConfig(), kw...) =
    solve(EstimationProblem(PosteriorWeightState, r, (; model_dof)), ResidualMixture(; weighting, config, kw...))
# A partition too thin to carry a scale: the old `nothing` is the new result's `u`; the new root (the FitResult, whose
# `u` takes the old root's path) and every other field are new (`""` in `new_only` names the root path only).
const _NONE_MAP = FieldMap(; renames = ["u" => ""], new_only = ["", "resid", "retcode", "stats", "prob", "alg", "original", "stop_rule",
                                                                "rms", "n", "metadata"])
function _eof_mixture!(cases)
    M = _FIT_MAP
    a = 40.0
    for seed in SEEDS
        r = eof_mixture_residuals(seed; n_in = 400, n_out = 100, σ = 0.8, halfwidth = a)
        sc = HalfwidthScoring(2; threshold = 3.0, outlier_halfwidth = a)
        replay!(cases, "residual_mix/known_400+100/halfwidth_t3/seed$seed", r, r -> mixture(r; weighting = sc, model_dof = 4); map = M)
        replay!(cases, "residual_mix/known_400+100_f32/halfwidth_t3/seed$seed", Float32.(r), r -> mixture(r; weighting = sc, model_dof = 4); map = M)
        replay!(cases, "residual_mix/known_400+100/thin_partition/seed$seed", r,
            r -> mixture(r; weighting = HalfwidthScoring(2; threshold = 1.0e-6, outlier_halfwidth = a), model_dof = 4); map = _NONE_MAP)
        r2 = eof_mixture_residuals(seed + 6; n_in = 200, n_out = 50, σ = 1.2, halfwidth = 60.0)
        sc2 = HalfwidthScoring(2; threshold = 4.0, outlier_halfwidth = 60.0)
        replay!(cases, "residual_mix/defaults_200+50/omitted_config/seed$seed", r2, r -> mixture(r; weighting = sc2, model_dof = 3); map = M)
        replay!(cases, "residual_mix/defaults_200+50/empty_toml_config/seed$seed", r2,
            r -> mixture(r; weighting = sc2, model_dof = 3, config = mixture_em_from_toml(Dict{String, Any}())); map = M)
        replay!(cases, "residual_mix/defaults_200+50/held_prior0.75_nu6/seed$seed", r2,
            r -> mixture(r; weighting = sc2, model_dof = 3,
                config = MixtureEmConfig(nuisance = (; scale = Estimated(), inlier_prior = FixedPrior(0.75), nu = FixedNu(6.0)))); map = M)
        rc = eof_mixture_residuals(seed + 12; n_in = 900, n_out = 25, σ = 0.4, halfwidth = 300.0)
        scc = HalfwidthScoring(2; threshold = 2.0, outlier_halfwidth = hypot(4000, 3000) / 2)
        replay!(cases, "residual_mix/pipeline_900+25/classification_em/seed$seed", rc,
            r -> mixture(r; weighting = scc, model_dof = 76, config = eof_mixture_classification_em()); map = M)
        replay!(cases, "residual_mix/pipeline_900+25/consensus_dof_max_iter1/seed$seed", rc,
            r -> mixture(r; weighting = scc, model_dof = 76,
                config = MixtureEmConfig(nuisance = (; scale = Estimated(), inlier_prior = Estimated(), nu = ConsensusDof()), max_iter = 1)); map = M)
        rs = eof_mixture_scalar(seed)
        replay!(cases, "residual_mix/scalar_200+40/student_t_uniform/seed$seed", rs,
            r -> mixture(r; weighting = StudentTUniformScoring(4, 40.0; codimension = 1, scale_prior = JeffreysPrior(),
                                                              label_prior = FlatLabelPrior()), model_dof = 4); map = M)
    end
    r = eof_mixture_residuals(11; n_in = 100, n_out = 10, σ = 0.5, halfwidth = 50.0)
    replay!(cases, "residual_mix/refused/weighted_refine", r,
        r -> mixture(r; weighting = MsacScoring(2; threshold = 2.0), model_dof = 4, config = MixtureEmConfig(model_update = WeightedRefine())); map = REFUSAL_MAP)
    replay!(cases, "residual_mix/refused/signed", eof_mixture_signed(), r -> mixture(r; weighting = MsacScoring(2; threshold = 2.0), model_dof = 4); map = REFUSAL_MAP)
    sc = HalfwidthScoring(2; threshold = 3.0, outlier_halfwidth = 50.0)
    replay!(cases, "residual_mix/refused/parameter_change_stop", r,
        r -> mixture(r; weighting = sc, model_dof = 4, config = MixtureEmConfig(stop = ParameterChangeStop(1.0e-6))); map = REFUSAL_MAP)
    # the former positional scorer: the algorithm takes its weighting by keyword only
    replay!(cases, "residual_mix/refused/positional_scorer", r,
        r -> solve(EstimationProblem(PosteriorWeightState, r, (; model_dof = 4)), ResidualMixture(sc)); map = REFUSAL_MAP)
    replay!(cases, "residual_mix/refused/scoring_kw", r, r -> mixture(r; weighting = sc, scoring = sc, model_dof = 4); map = REFUSAL_MAP)
    return cases
end

# SimFitPP returned `(loss_width, success)` and `(best_idx, loss_width, success)`: the replay presents the new result
# in those shapes — `u`, and `metadata.loss_width` for a pool — with `success` its `successful_retcode`.
simfit_width(f, rs, n, init, cfg; num_previous_est = 0, rng) =
    (r = solve(EstimationProblem(LossWidth, (; fit = f, residuals_sq = rs, n), (; initial_loss_width = init, num_previous_est)), cfg; rng);
     (r.u, successful_retcode(r)))
simfit_select(D, init, cfg = SimFitPP(); rng) =
    (r = solve(EstimationProblem(PoolSelection, D, (; initial_loss_width = init)), cfg; rng);
     (r.u, r.metadata.loss_width, successful_retcode(r)))
function _eof_simfit!(cases)
    msac(D) = (lw, train) -> solve(EstimationProblem(PoolSelection, D[train, :]), ScoreSelection(MsacScoring{1, Float64}(Float64(lw)))).u
    sq(D) = (j, val) -> Float64[Float64(D[i, j])^2 for i in val]
    for seed in SEEDS
        D = eof_simfit_pool(seed; σ = 1.0)
        n = size(D, 1)
        replay!(cases, "loss_width_cv/pool_sigma1/default/seed$seed", D, D -> simfit_width(msac(D), sq(D), size(D, 1), 1.0, SimFitPP(); rng = StableRNG(seed)))
        replay!(cases, "loss_width_cv/pool_sigma1/previous3/seed$seed", D,
            D -> simfit_width(msac(D), sq(D), size(D, 1), 1.0, SimFitPP(); num_previous_est = 3, rng = StableRNG(seed)))
        Dsm = eof_simfit_pool(seed + 10; σ = 0.02)
        replay!(cases, "loss_width_cv/pool_sigma0.02/out_of_bounds/seed$seed", Dsm,
            D -> simfit_width(msac(D), sq(D), size(D, 1), 1.0, SimFitPP(); rng = StableRNG(seed)))
        replay!(cases, "loss_width_cv/pool_sigma1/residual_unit0.5/seed$seed", D,
            D -> simfit_width(msac(D), sq(D), size(D, 1), 1.0, with_residual_unit(SimFitPP(), 0.5); rng = StableRNG(seed)))
        replay!(cases, "loss_width_cv/failing_estimator/seed$seed", n,
            n -> simfit_width((lw, tr) -> nothing, (m, v) -> nothing, n, 1.0, SimFitPP(); rng = StableRNG(seed)))
        D8 = eof_simfit_pool(seed + 20; σ = 0.8)
        replay!(cases, "pool_select/matrix_sigma0.8/default/seed$seed", D8, D -> simfit_select(D, 1.0; rng = StableRNG(seed)))
        fp = eof_fundmat_pool(seed)
        replay!(cases, "pool_select/fundmat_pool/pixels3/seed$seed", fp, p -> simfit_select(p, 3.0; rng = StableRNG(seed)))
        replay!(cases, "pool_select/fundmat_pool_frame/unit_pool_scale/seed$seed", fp,
            p -> simfit_select(VgtkCore.Pools.pixel_residuals(p) .* Float32(VgtkCore.Pools.pool_pixel_scale(p)), 0.01,
                               with_residual_unit(SimFitPP(), VgtkCore.Pools.pool_pixel_scale(p)); rng = StableRNG(seed)))
    end
    replay!(cases, "pool_select/refused/bare_matrix_models", eof_identity_pool(eof_simfit_pool(21; σ = 0.8)), p -> simfit_select(p, 1.0; rng = StableRNG(1));
        map = REFUSAL_MAP)
    replay!(cases, "pool_select/empty_pool", zeros(Float32, 0, 3), D -> simfit_select(D, 1.0; rng = StableRNG(1)))
    return cases
end

function replay_est_other_fits(cases)
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
