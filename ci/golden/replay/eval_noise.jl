# The noise-prior fit on VisualGeometryToolkitEval 0.12: `fit_prior(spec, σ²s)` → `solve(EstimationProblem(IGPrior, σ²s),
# spec)`, the `PriorSpec` the closed-form algorithm; the three-argument form `fit_prior(family, fitting, σ²s)` →
# `PriorSpec(family, fitting)`. A bare `IGPrior` (or `nothing`, too few data) → a FitResult whose `u` it is.
using VisualGeometryToolkitCore: EstimationProblem, IGPrior, solve
using VisualGeometryToolkitEval.Tuning: PriorSpec, Jeffreys, InverseGamma, MLE, RobustFit, Unfitted

# A closed form that returned a bare value: `u` is it, every other field new.
const _EN_BARE = FieldMap(; renames = ["u" => ""],
    new_only = ["", "resid", "retcode", "stats", "prob", "alg", "original", "stop_rule", "rms", "n", "metadata"])

const EVAL_NOISE_SPECS = [
    "jeffreys" => PriorSpec(Jeffreys(), Unfitted()),
    "ig_robust_alpha1" => PriorSpec(InverseGamma(), RobustFit(1.0)),
    "ig_robust_alpha2.5" => PriorSpec(InverseGamma(), RobustFit(2.5)),
    "ig_mle" => PriorSpec(InverseGamma(), MLE()),
]

_en_prior(spec, x) = solve(EstimationProblem(IGPrior, x), spec)

function replay_eval_noise(cases)
    for (name, x) in eval_noise_samples(), (lbl, spec) in EVAL_NOISE_SPECS
        replay!(cases, "scale_prior/$name/$lbl", x, x -> _en_prior(spec, x); map = _EN_BARE)
    end
    for seed in SEEDS, (lbl, spec) in EVAL_NOISE_SPECS
        replay!(cases, "scale_prior/ig3_2e-3_200k_noise/$lbl/seed$seed", eval_noise_ig_draw(seed), x -> _en_prior(spec, x);
                map = _EN_BARE)
    end
    x = last(first(eval_noise_samples()))
    # the three-argument form: its (family, fitting) is the PriorSpec; Jeffreys without data
    replay!(cases, "scale_prior/no_data/jeffreys_family_form", nothing, x -> _en_prior(PriorSpec(Jeffreys(), Unfitted()), x);
            map = _EN_BARE)
    replay!(cases, "scale_prior/six_heavy_tail/ig_mle_family_form", x, x -> _en_prior(PriorSpec(InverseGamma(), MLE()), x);
            map = _EN_BARE)
    replay!(cases, "scale_prior/six_heavy_tail/ig_robust_family_form", x,
            x -> _en_prior(PriorSpec(InverseGamma(), RobustFit(2.5)), x); map = _EN_BARE)
    # the illegal combinations, refused where the spec is built (an ArgumentError, the same message)
    replay!(cases, "scale_prior/refused/jeffreys_mle", x, x -> _en_prior(PriorSpec(Jeffreys(), MLE()), x))
    replay!(cases, "scale_prior/refused/jeffreys_robust", x, x -> _en_prior(PriorSpec(Jeffreys(), RobustFit()), x))
    replay!(cases, "scale_prior/refused/ig_unfitted", x, x -> _en_prior(PriorSpec(InverseGamma(), Unfitted()), x))
    replay!(cases, "scale_prior/refused/jeffreys_mle_family_form", x, x -> _en_prior(PriorSpec(Jeffreys(), MLE()), x))
    replay!(cases, "scale_prior/refused/ig_unfitted_family_form", x, x -> _en_prior(PriorSpec(InverseGamma(), Unfitted()), x))
    return cases
end
