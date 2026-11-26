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
    using VisualGeometryToolkitEval.Tuning: Jeffreys, InverseGamma, MLE, RobustFit, Unfitted,
        prior_label, family_tag, fitting_tag, ScalePriorFamily, PriorFitting

    σ2s = [1.0e-4, 4.0e-4, 2.5e-5, 9.0e-4, 1.6e-4, 3.0e-2]   # one heavy-tailed value

    @testset "the axes and the named specs" begin
        @test Jeffreys() isa ScalePriorFamily && InverseGamma() isa ScalePriorFamily
        @test all(f -> f isa PriorFitting, (Unfitted(), RobustFit(), MLE()))
        @test RobustFit() === RobustFit(1.0)
    end

    prior(spec, x) = solve(EstimationProblem(IGPrior, x), spec)

    @testset "a closed-form algorithm" begin
        @test estimator_kind(PriorSpec(Jeffreys(), Unfitted())) === ClosedFormKind()
        r = prior(PriorSpec(InverseGamma(), RobustFit()), σ2s)
        @test r.retcode === ReturnCode.Success && r.stop_rule === ClosedForm && r.n == length(σ2s) && isnan(r.rms)
        @test r.prob.f === σ2s && r.alg === PriorSpec(InverseGamma(), RobustFit())
        # too few data: degenerate, no prior
        r0 = prior(PriorSpec(InverseGamma(), RobustFit()), Float64[])
        @test r0.u === nothing && r0.retcode === ReturnCode.InitialFailure && r0.stop_rule === NoRule && r0.n == 0
    end

    @testset "Jeffreys is parameterless" begin
        @test prior(PriorSpec(Jeffreys(), Unfitted()), σ2s).u === IGPrior(0.0, 0.0)
        @test prior(PriorSpec(Jeffreys(), Unfitted()), Float64[]).u === IGPrior(0.0, 0.0)   # data is ignored
        @test prior(PriorSpec(Jeffreys(), Unfitted()), nothing).u === IGPrior(0.0, 0.0)
        @test prior(PriorSpec(Jeffreys(), Unfitted()), σ2s).n == 0                          # it reads none
    end

    @testset "robust IG — fixed shape, scale from the median" begin
        for α0 in (1.0, 2.5)
            @test prior(PriorSpec(InverseGamma(), RobustFit(α0)), σ2s).u === IGPrior(α0, α0 * median(σ2s))
        end
        # the median ignores the heavy tail: moving the largest value moves nothing
        @test prior(PriorSpec(InverseGamma(), RobustFit()), vcat(σ2s[1:end-1], 1.0e3)).u === prior(PriorSpec(InverseGamma(), RobustFit()), σ2s).u
        @test prior(PriorSpec(InverseGamma(), RobustFit()), Float64[]).u === nothing
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

    @testset "illegal combinations are refused where the spec is built" begin
        @test_throws "parameterless" PriorSpec(Jeffreys(), MLE())
        @test_throws "parameterless" PriorSpec(Jeffreys(), RobustFit())
        @test_throws "non-trivial fitting" PriorSpec(InverseGamma(), Unfitted())
        @test_throws ArgumentError PriorSpec(Jeffreys(), MLE())
        @test_throws MethodError PriorSpec(Jeffreys(), :mle)    # a fitting is a PriorFitting
        @test_throws ArgumentError solve(EstimationProblem(IGPrior, σ2s))   # no default: the spec is named
    end

    @testset "labels and per-axis tags" begin
        @test prior_label(PriorSpec(Jeffreys(), Unfitted())) == "Jeffreys"
        @test prior_label(PriorSpec(InverseGamma(), RobustFit())) == "IG-Robust"
        @test prior_label(PriorSpec(InverseGamma(), MLE())) == "IG-MLE"
        @test family_tag(Jeffreys()) === :jeffreys && family_tag(InverseGamma()) === :ig
        @test fitting_tag(Unfitted()) === :unfitted && fitting_tag(RobustFit()) === :robust && fitting_tag(MLE()) === :mle
        @test family_tag(PriorSpec(InverseGamma(), RobustFit())) === :ig && fitting_tag(PriorSpec(InverseGamma(), RobustFit())) === :robust
        @test (family_tag(PriorSpec(Jeffreys(), Unfitted())), fitting_tag(PriorSpec(Jeffreys(), Unfitted()))) === (:jeffreys, :unfitted)
        # labels and tag pairs are distinct across the specs
        specs = (PriorSpec(Jeffreys(), Unfitted()), PriorSpec(InverseGamma(), RobustFit()), PriorSpec(InverseGamma(), MLE()))
        @test allunique(prior_label.(specs))
        @test allunique([(family_tag(s), fitting_tag(s)) for s in specs])
    end
end
