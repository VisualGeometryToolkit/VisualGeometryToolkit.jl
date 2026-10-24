# The scale-prior strategy layer of a tuning (src/tuning/prior_specs.jl; moved
# from RANSACScoringDoneRight's eval_strategy.jl, K+ 2026-10-09): the prior
# family × fitting axes, `fit_prior` for every legal (family, fitting), the
# illegal combinations refused at dispatch, the labels and per-axis tags, and
# the roster a sweep iterates.

@testitem "prior_specs" begin
    using Test, Statistics, Random
    using Distributions: Gamma, InverseGamma as IGDist, fit_mle, shape, scale
    using VisualGeometryToolkitCore: IGPrior
    using VisualGeometryToolkitEval.Tuning: PriorSpec, fit_prior
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

    @testset "fit_prior: Jeffreys is parameterless" begin
        @test fit_prior(JeffreysPriorSpec(), σ2s) === IGPrior(0.0, 0.0)
        @test fit_prior(JeffreysPriorSpec(), Float64[]) === IGPrior(0.0, 0.0)   # data is ignored
        @test fit_prior(Jeffreys(), Unfitted(), nothing) === IGPrior(0.0, 0.0)
    end

    @testset "fit_prior: robust IG — fixed shape, scale from the median" begin
        for α0 in (1.0, 2.5)
            @test fit_prior(IGRobustSpec(α0), σ2s) === IGPrior(α0, α0 * median(σ2s))
        end
        # the median ignores the heavy tail: moving the largest value moves nothing
        @test fit_prior(IGRobustSpec(), vcat(σ2s[1:end-1], 1.0e3)) === fit_prior(IGRobustSpec(), σ2s)
        @test fit_prior(IGRobustSpec(), Float64[]) === nothing
    end

    @testset "fit_prior: IG MLE through the Gamma reparameterisation" begin
        spec = PriorSpec(InverseGamma(), MLE())
        g = fit_mle(Gamma, 1 ./ σ2s)
        @test fit_prior(spec, σ2s) === IGPrior(shape(g), 1 / scale(g))
        # recovers the generating IG(α, β) on a large sample
        rng = MersenneTwister(4)
        α, β = 3.0, 2.0e-3
        x = rand(rng, IGDist(α, β), 200_000)
        p = fit_prior(spec, x)
        @test p.alpha0 ≈ α rtol = 0.02
        @test p.beta0 ≈ β rtol = 0.02
        @test fit_prior(spec, [1.0e-4]) === nothing          # n < 2
        @test fit_prior(spec, Float64[]) === nothing
    end

    @testset "illegal combinations are refused at dispatch" begin
        @test_throws ErrorException fit_prior(PriorSpec(Jeffreys(), MLE()), σ2s)
        @test_throws ErrorException fit_prior(PriorSpec(Jeffreys(), RobustFit()), σ2s)
        @test_throws ErrorException fit_prior(PriorSpec(InverseGamma(), Unfitted()), σ2s)
        @test_throws "parameterless" fit_prior(Jeffreys(), MLE(), σ2s)
        @test_throws "non-trivial fitting" fit_prior(InverseGamma(), Unfitted(), σ2s)
        @test_throws MethodError PriorSpec(Jeffreys(), :mle)    # a fitting is a PriorFitting
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
