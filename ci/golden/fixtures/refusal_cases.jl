# The refusals every find_* and refine_* shares (core_defaults, refine_contract, ransac_no_model), recorded by
# record/est_epipolar.jl and record/est_pose.jl: the RECORD side's one copy (it calls the old pipelines handed to it).
# It lives beside the fixtures because every file of record/ is a group. `find === nothing`: the class has no find_*;
# `P`, the class's problem type, names the scorer of the refused `scoring` keyword.
function refusal_cases!(cases, family, fx, find, refine, θ, P)
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
            "scoring_kw" => (; scale = 1.0, scoring = MsacScoring(P; threshold = 3.0)),
            "rng_kw" => (; scale = 1.0, rng = StableRNG(1)), "max_trials_kw" => (; scale = 1.0, max_trials = 10),
            "extent_kw" => (; scale = 1.0, extent = nothing), "unknown_kw" => (; scale = 1.0, nonsense = 1))
        case!(cases, "$family/refused/refine_$lbl", (fx, θ), ((fx, θ),) -> refine(fx, θ; kw...))
    end
    return cases
end
