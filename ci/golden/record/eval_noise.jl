# The old noise-prior estimator at 3d8a06da: VisualGeometryToolkitEval's fit_prior, every (family, fitting) its suite
# exercises, through a PriorSpec and through the (family, fitting, data) form, and the illegal combinations refused.
using VisualGeometryToolkitEval.Tuning: PriorSpec, fit_prior, Jeffreys, InverseGamma, MLE, RobustFit, Unfitted,
    JeffreysPriorSpec, IGRobustSpec

const EVAL_NOISE_SPECS = [
    "jeffreys" => JeffreysPriorSpec(),
    "ig_robust_alpha1" => IGRobustSpec(),
    "ig_robust_alpha2.5" => IGRobustSpec(2.5),
    "ig_mle" => PriorSpec(InverseGamma(), MLE()),
]

function record_eval_noise(cases)
    for (name, x) in eval_noise_samples(), (lbl, spec) in EVAL_NOISE_SPECS
        case!(cases, "scale_prior/$name/$lbl", x, x -> fit_prior(spec, x))
    end
    for seed in SEEDS, (lbl, spec) in EVAL_NOISE_SPECS
        case!(cases, "scale_prior/ig3_2e-3_200k_noise/$lbl/seed$seed", eval_noise_ig_draw(seed), x -> fit_prior(spec, x))
    end
    x = last(first(eval_noise_samples()))
    # the three-argument form, Jeffreys without data
    case!(cases, "scale_prior/no_data/jeffreys_family_form", nothing, x -> fit_prior(Jeffreys(), Unfitted(), x))
    case!(cases, "scale_prior/six_heavy_tail/ig_mle_family_form", x, x -> fit_prior(InverseGamma(), MLE(), x))
    case!(cases, "scale_prior/six_heavy_tail/ig_robust_family_form", x, x -> fit_prior(InverseGamma(), RobustFit(2.5), x))
    # the illegal combinations, refused at dispatch
    case!(cases, "scale_prior/refused/jeffreys_mle", x, x -> fit_prior(PriorSpec(Jeffreys(), MLE()), x))
    case!(cases, "scale_prior/refused/jeffreys_robust", x, x -> fit_prior(PriorSpec(Jeffreys(), RobustFit()), x))
    case!(cases, "scale_prior/refused/ig_unfitted", x, x -> fit_prior(PriorSpec(InverseGamma(), Unfitted()), x))
    case!(cases, "scale_prior/refused/jeffreys_mle_family_form", x, x -> fit_prior(Jeffreys(), MLE(), x))
    case!(cases, "scale_prior/refused/ig_unfitted_family_form", x, x -> fit_prior(InverseGamma(), Unfitted(), x))
    return cases
end
