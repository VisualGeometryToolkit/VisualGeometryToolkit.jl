# The scale-prior strategy layer of a tuning (src/tuning/prior_specs.jl; moved
# from RANSACScoringDoneRight's eval_strategy.jl, K+ 2026-10-09): the prior
# family × fitting axes, the closed-form solve `solve(EstimationProblem(IGPrior,
# σ²s), spec)` for every legal (family, fitting), the illegal combinations
# refused at dispatch, the labels and per-axis tags, and the roster a sweep
# iterates.

@testitem "prior_specs" begin
    using Test, Statistics, Random
    using Distributions: Gamma, InverseGamma as IGDist, fit_mle, shape, scale
    using VisualGeometryToolkitCore: IGPrior, EstimationProblem, ClosedFormKind, ReturnCode, ClosedForm, NoRule, solve,
        estimator_kind
    using VisualGeometryToolkitEval.Tuning: PriorSpec
    using VisualGeometryToolkitEval.Tuning: Jeffreys, InverseGamma, MLE, RobustFit, Unfitted, JeffreysPriorSpec,
        IGRobustSpec, ALL_PRIOR_SPECS, prior_label, family_tag, fitting_tag, ScalePriorFamily, PriorFitting

    σ2s = [1.0e-4, 4.0e-4, 2.5e-5, 9.0e-4, 1.6e-4, 3.0e-2]   # one heavy-tailed value

    @testset "the axes and the named specs" begin
        @test Jeffreys() isa ScalePriorFamily && InverseGamma() isa ScalePriorFamily
        @test all(f -> f isa PriorFitting, (Unfitted(), RobustFit(), MLE()))
        @test JeffreysPriorSpec() === PriorSpec(Jeffreys(), Unfitted())
        @test IGRobustSpec() === PriorSpec(InverseGamma(), RobustFit(1.0))
        @test IGRobustSpec(2.0) === PriorSpec(InverseGamma(), RobustFit(2.0))
        @test RobustFit() === RobustFit(1.0)
        # the oracle-free roster: Jeffreys alone (the empirical-Bayes IG fits use GT σ̂²)
        @test ALL_PRIOR_SPECS === (JeffreysPriorSpec(),)
    end

    prior(spec, x) = solve(EstimationProblem(IGPrior, x), spec)

    @testset "a closed-form algorithm" begin
        @test estimator_kind(JeffreysPriorSpec()) === ClosedFormKind()
        r = prior(IGRobustSpec(), σ2s)
        @test r.retcode === ReturnCode.Success && r.stop_rule === ClosedForm && r.n == length(σ2s) && isnan(r.rms)
        @test r.prob.f === σ2s && r.alg === IGRobustSpec()
        # too few data: degenerate, no prior
        r0 = prior(IGRobustSpec(), Float64[])
        @test r0.u === nothing && r0.retcode === ReturnCode.InitialFailure && r0.stop_rule === NoRule && r0.n == 0
    end

    @testset "Jeffreys is parameterless" begin
        @test prior(JeffreysPriorSpec(), σ2s).u === IGPrior(0.0, 0.0)
        @test prior(JeffreysPriorSpec(), Float64[]).u === IGPrior(0.0, 0.0)   # data is ignored
        @test prior(JeffreysPriorSpec(), nothing).u === IGPrior(0.0, 0.0)
        @test prior(JeffreysPriorSpec(), σ2s).n == 0                          # it reads none
    end

    @testset "robust IG — fixed shape, scale from the median" begin
        for α0 in (1.0, 2.5)
            @test prior(IGRobustSpec(α0), σ2s).u === IGPrior(α0, α0 * median(σ2s))
        end
        # the median ignores the heavy tail: moving the largest value moves nothing
        @test prior(IGRobustSpec(), vcat(σ2s[1:end-1], 1.0e3)).u === prior(IGRobustSpec(), σ2s).u
        @test prior(IGRobustSpec(), Float64[]).u === nothing
    end

    @testset "IG MLE through the Gamma reparameterisation" begin
        spec = PriorSpec(InverseGamma(), MLE())
        g = fit_mle(Gamma, 1 ./ σ2s)
        @test prior(spec, σ2s).u === IGPrior(shape(g), 1 / scale(g))
        # recovers the generating IG(α, β) on a large sample
        rng = MersenneTwister(4)
        α, β = 3.0, 2.0e-3
        x = rand(rng, IGDist(α, β), 200_000)
        p = prior(spec, x).u
        @test p.alpha0 ≈ α rtol = 0.02
        @test p.beta0 ≈ β rtol = 0.02
        @test prior(spec, [1.0e-4]).u === nothing          # n < 2
        @test prior(spec, Float64[]).u === nothing
    end

    @testset "illegal combinations are refused at dispatch" begin
        @test_throws "parameterless" prior(PriorSpec(Jeffreys(), MLE()), σ2s)
        @test_throws "parameterless" prior(PriorSpec(Jeffreys(), RobustFit()), σ2s)
        @test_throws "non-trivial fitting" prior(PriorSpec(InverseGamma(), Unfitted()), σ2s)
        @test_throws ErrorException prior(PriorSpec(Jeffreys(), MLE()), σ2s)
        @test_throws MethodError PriorSpec(Jeffreys(), :mle)    # a fitting is a PriorFitting
        @test_throws ArgumentError solve(EstimationProblem(IGPrior, σ2s))   # no default: the spec is named
    end

    @testset "labels and per-axis tags" begin
        @test prior_label(JeffreysPriorSpec()) == "Jeffreys"
        @test prior_label(IGRobustSpec()) == "IG-Robust"
        @test prior_label(PriorSpec(InverseGamma(), MLE())) == "IG-MLE"
        @test_throws MethodError prior_label(PriorSpec(Jeffreys(), MLE()))   # no label for an illegal spec
        @test family_tag(Jeffreys()) === :jeffreys && family_tag(InverseGamma()) === :ig
        @test fitting_tag(Unfitted()) === :unfitted && fitting_tag(RobustFit()) === :robust && fitting_tag(MLE()) === :mle
        @test family_tag(IGRobustSpec()) === :ig && fitting_tag(IGRobustSpec()) === :robust
        @test (family_tag(JeffreysPriorSpec()), fitting_tag(JeffreysPriorSpec())) === (:jeffreys, :unfitted)
        # labels and tag pairs are distinct across the specs
        specs = (JeffreysPriorSpec(), IGRobustSpec(), PriorSpec(InverseGamma(), MLE()))
        @test allunique(prior_label.(specs))
        @test allunique([(family_tag(s), fitting_tag(s)) for s in specs])
    end
end
