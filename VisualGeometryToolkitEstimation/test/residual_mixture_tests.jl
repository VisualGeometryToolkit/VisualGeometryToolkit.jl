@testitem "residual_mixture" begin
    import VisualGeometryToolkitCore as VgtkCore
# =============================================================================
# The posterior-weight iteration's bound updates, and `ResidualMixture`
# =============================================================================
#
# What this pins: the DEFAULT model update (`WeightedRefine`) and nuisance
# policies (ŝ², ν estimated; π held at the state's own value) are the
# two-part update the RANSAC local optimization always ran, bit for bit;
# `ResidualMixture` refuses a model update that is not `KeepModel`; the
# `[mixture_em]` TOML round trip, and refusal of unknown tags and keys; and
# `ResidualMixture` itself — its scale, inlier fraction and MAP labels on a
# known mixture, its `[mixture_em]` defaults, and a calibration pipeline's call —
# under a test-only scorer with a uniform outlier model (`HalfwidthScoring`
# below: the scoring contract at a threshold, plus `outlier_halfwidth`).
# =============================================================================

using Test
using Random
using StableRNGs
using VisualGeometryToolkitCore: inlier_mask, is_inlier, inlier_weight, robust_weight,
    update_nuisance!, update_weights!, successful_retcode, EstimationProblem, solve
# The robust-statistics toolkit (VisualGeometryToolkitCore): the posterior-weight
# state's accessors and updates, and the nuisance-part policies.
using VisualGeometryToolkitCore: s_hat_sq, student_t_dof, inlier_prior, sq_distances, update_scale!, update_nu!,
    Estimated, FixedS2, FixedPrior, FixedNu, ConsensusDof
using VisualGeometryToolkitCore: MsacScoring, median_seeded_posterior, JeffreysPrior
using VisualGeometryToolkitCore: weights, weight_kind, PosteriorWeights, noise_scale, scale_source, ScaleEstimated
using VisualGeometryToolkitEstimation
using VisualGeometryToolkitEstimation: PosteriorLikelihood, MixtureEmConfig, KeepModel, WeightedRefine,
    mixture_em_from_toml, config_dict, ResidualMixtureMetadata, ResidualMixture
using VisualGeometryToolkitCore: VisualGeometryToolkitCore, AbstractScoring, FitResult, PosteriorWeightState
residual_mixture(d; weighting, model_dof, kw...) =
    solve(EstimationProblem(PosteriorWeightState, d, (; model_dof)), ResidualMixture(; weighting, kw...))

# The test-only scorer with a uniform outlier model (`HalfwidthScoring`) and the reprojection-residual draw
# (`reprojection_residuals`): the suite's fixture file, shared with the golden records.
import VisualGeometryToolkitEstimation as VgtkEstimation
include(joinpath(pkgdir(VgtkEstimation), "test", "fixtures", "residual_mixture.jl"))
# A posterior seeded at the residuals' median (VisualGeometryToolkitCore's seed).
median_posterior(r, n) = median_seeded_posterior(r .^ 2, n; outlier_halfwidth = 40.0, ν = 1.0e4, prior = JeffreysPrior(), codimension = 2)

@testset "default nuisance policies == the two-part update, bit for bit" begin
    rng = StableRNG(3)
    r, _ = reprojection_residuals(rng; n_in = 300, n_out = 60, σ = 0.8, halfwidth = 40.0)
    n = length(r)
    a = median_posterior(r, n)
    b = median_posterior(r, n)
    lh = PosteriorLikelihood(a, n)
    @test lh.scale_update isa Estimated && lh.nu_update isa Estimated
    @test lh.prior_update == FixedPrior(inlier_prior(a))
    for _ in 1:6
        update_nuisance!(lh); update_weights!(lh)
        # the update the RANSAC local optimization ran before the policies existed
        update_scale!(b, n); update_nu!(b, n); update_weights!(b, n)
        @test (s_hat_sq(a), student_t_dof(a), inlier_prior(a)) === (s_hat_sq(b), student_t_dof(b), inlier_prior(b))
        @test robust_weight(a) == robust_weight(b) && inlier_weight(a) == inlier_weight(b)
    end

    # a held part is held; the others still move
    c = median_posterior(r, n)
    held = PosteriorLikelihood(c, n; scale_update = FixedS2(0.5), nu_update = FixedNu(4.0),
                               prior_update = Estimated())
    update_nuisance!(held)
    @test s_hat_sq(c) === 0.5 && student_t_dof(c) === 4.0
    @test inlier_prior(c) ≈ sum(inlier_weight(c)) / n
end

@testset "residual mixture: the model is held" begin
    # a weighted refit needs a model to refit — refused, not ignored (before
    # the scorer is read)
    r, _ = reprojection_residuals(StableRNG(11); n_in = 100, n_out = 10, σ = 0.5, halfwidth = 50.0)
    @test_throws ArgumentError residual_mixture(r; weighting = MsacScoring(2; threshold = 2.0), model_dof = 4,
                                                    config = MixtureEmConfig(model_update = WeightedRefine()))
    # the input is distances (≥ 0): a signed residual is refused, naming `abs`
    rs = copy(r); rs[3] = -rs[3]
    err = try residual_mixture(rs; weighting = MsacScoring(2; threshold = 2.0), model_dof = 4) catch e; e end
    @test err isa ArgumentError && occursin("abs", err.msg) && occursin("distances[3]", err.msg)
end

@testset "residual mixture: a held model stops on the cost, never on a parameter change" begin
    # the distances are a held model's: its parameters never move, so a parameter-change stop would end the EM at
    # iteration 1 with the nuisance unconverged — refused by name, as the division-camera fit validates its own stop
    r, _ = reprojection_residuals(StableRNG(11); n_in = 100, n_out = 10, σ = 0.5, halfwidth = 50.0)
    sc = HalfwidthScoring(2; threshold = 3.0, outlier_halfwidth = 50.0)
    @test_throws r"NoImprovementStop" residual_mixture(r; weighting = sc, model_dof = 4,
                                                           config = MixtureEmConfig(stop = VisualGeometryToolkitEstimation.ParameterChangeStop(1.0e-6)))
end

@testset "residual mixture: a known mixture" begin
    # 400 Rayleigh inliers at σ = 0.8 per axis, 100 outliers uniform over the
    # ±40 square: the EM recovers σ, the inlier fraction and every label.
    σ, a = 0.8, 40.0
    r, truth = reprojection_residuals(StableRNG(5); n_in = 400, n_out = 100, σ, halfwidth = a)
    sc = HalfwidthScoring(2; threshold = 3.0, outlier_halfwidth = a)
    fit = residual_mixture(r; weighting = sc, model_dof = 4)
    @test fit isa FitResult{Union{PosteriorWeightState, Nothing}, ResidualMixtureMetadata}
    @test successful_retcode(fit) && 1 <= fit.stats.nsteps < MixtureEmConfig().max_iter
    p = fit.u
    # σ against the inliers' own per-axis sample scale: the draw's scale
    # (0.764 for this StableRNG seed) is what the EM can recover.
    σ_sample = sqrt(sum(abs2, r[truth]) / (2 * count(truth)))
    @test sqrt(s_hat_sq(p)) ≈ σ_sample rtol = 0.05
    @test inlier_prior(p) ≈ 0.8 atol = 0.02
    @test inlier_mask(p) == truth
    @test all(i -> is_inlier(p, i) == truth[i], eachindex(r))
    # the contract's record: the posteriors as the weights (PosteriorWeights: the MAP
    # labels are inlier_mask(fit)), the E-step's ŝ estimated; n the support and rms
    # the posterior-weighted RMS; the metadata the NLL the EM lowered
    w = weights(fit)
    @test w == [x < eps() ? 0.0 : x for x in inlier_weight(p)] && weight_kind(fit) === PosteriorWeights()
    @test inlier_mask(fit) == truth
    @test noise_scale(fit) === sqrt(s_hat_sq(p)) && scale_source(fit) === ScaleEstimated
    @test @inferred(noise_scale(fit)) isa Float64
    md = fit.metadata
    @test @inferred(ResidualMixtureMetadata(md.nll, md.weights, md.scale, md.scale_source)) isa ResidualMixtureMetadata
    @test fit.n == count(>(0), w) && fit.n >= count(truth)
    @test fit.rms ≈ sqrt(sum(w .* r .^ 2) / sum(w))
    @test isfinite(fit.metadata.nll)
    # the algorithm takes its `weighting` by keyword and nothing else: no positional scorer, no `scoring`
    @test_throws MethodError solve(EstimationProblem(PosteriorWeightState, r, (; model_dof = 4)), ResidualMixture(sc))
    @test_throws MethodError residual_mixture(r; weighting = sc, scoring = sc, model_dof = 4)
    # the residuals are not moved, and the vector type does not matter
    r0 = copy(r)
    fit32 = residual_mixture(Float32.(r); weighting = sc, model_dof = 4)
    @test r == r0 && inlier_mask(fit32.u) == truth
    # a partition too thin to carry a scale: no mixture (`u` nothing, a degenerate start)
    thin = residual_mixture(r; weighting = HalfwidthScoring(2; threshold = 1.0e-6, outlier_halfwidth = a), model_dof = 4)
    @test thin.u === nothing && thin.retcode == VisualGeometryToolkitCore.ReturnCode.InitialFailure
end

@testset "residual mixture: the [mixture_em] defaults" begin
    c = MixtureEmConfig()
    @test c.model_update == KeepModel(1.0e-10) && c.max_iter == 200
    @test c.nuisance.scale isa Estimated && c.nuisance.inlier_prior isa Estimated && c.nuisance.nu isa Estimated
    # the defaults are what an omitted `config` runs
    r, _ = reprojection_residuals(StableRNG(7); n_in = 200, n_out = 50, σ = 1.2, halfwidth = 60.0)
    sc = HalfwidthScoring(2; threshold = 4.0, outlier_halfwidth = 60.0)
    a = residual_mixture(r; weighting = sc, model_dof = 3)
    b = residual_mixture(r; weighting = sc, model_dof = 3, config = mixture_em_from_toml(Dict{String, Any}()))
    @test (s_hat_sq(a.u), student_t_dof(a.u), inlier_prior(a.u), a.stats.nsteps, a.n) ===
          (s_hat_sq(b.u), student_t_dof(b.u), inlier_prior(b.u), b.stats.nsteps, b.n)
    # a held part stays held through the EM
    h = residual_mixture(r; weighting = sc, model_dof = 3,
                             config = MixtureEmConfig(nuisance = (; scale = Estimated(), inlier_prior = FixedPrior(0.75),
                                                                  nu = FixedNu(6.0))))
    @test inlier_prior(h.u) === 0.75 && student_t_dof(h.u) === 6.0
end

@testset "residual mixture: a calibration pipeline's call" begin
    # A bundle adjustment's inlier statistics: its per-observation residual
    # norms, its dof, and the `[classification]` table of its configs — the
    # outlier half-width the image's half-diagonal, codimension 2, and the EM
    # with ν held at the consensus seed's dof.
    image_size = (4000, 3000)
    classification = (;
        outlier_halfwidth = hypot(image_size[1], image_size[2]) / 2, codimension = 2,
        em = mixture_em_from_toml(Dict{String, Any}(
            "max_iter" => 200, "model_update" => Dict{String, Any}("type" => "keep_model", "tol" => 1.0e-10),
            "nuisance" => Dict{String, Any}("scale" => "estimated", "inlier_prior" => "estimated", "nu" => "consensus_dof"),
        ); section = "classification.em"),
    )
    @test classification.em.nuisance.nu isa ConsensusDof
    rng = StableRNG(13)
    residuals, truth = reprojection_residuals(rng; n_in = 900, n_out = 25, σ = 0.4, halfwidth = 300.0)
    n = length(residuals)
    dof = 2n - 150                                      # a bundle with 150 free parameters
    d = classification.codimension
    m = cld(d * n - dof, d)
    @test m == 75
    scoring = HalfwidthScoring(d; threshold = 2.0, outlier_halfwidth = classification.outlier_halfwidth)
    fit = residual_mixture(residuals; weighting = scoring, model_dof = m + 1, config = classification.em)
    @test fit !== nothing && successful_retcode(fit)
    pws = fit.u
    @test sqrt(s_hat_sq(pws)) ≈ 0.4 rtol = 0.1
    @test count(inlier_mask(fit)) == count(inlier_mask(pws)) && n - count(inlier_mask(fit)) >= 20
    @test count(inlier_mask(pws) .!= truth) <= 3
    @test fit.rms < 1.0
    # ν held at the seed's posterior-predictive dof: unchanged by the EM
    seed = residual_mixture(residuals; weighting = scoring, model_dof = m + 1,
                                config = MixtureEmConfig(nuisance = (; scale = Estimated(), inlier_prior = Estimated(),
                                                                     nu = ConsensusDof()), max_iter = 1))
    @test student_t_dof(pws) == student_t_dof(seed.u)
end

@testset "[mixture_em] TOML: round trip, refusals" begin
    c = MixtureEmConfig()
    @test mixture_em_from_toml(config_dict(c)) == c
    held = MixtureEmConfig(model_update = KeepModel(1.0e-6),
                           nuisance = (; scale = FixedS2(0.25), inlier_prior = FixedPrior(0.9), nu = FixedNu(5.0)),
                           max_iter = 7)
    d = config_dict(held)
    @test d["model_update"] == Dict("type" => "keep_model", "tol" => 1.0e-6)
    @test d["nuisance"]["scale"] == Dict("type" => "fixed_s2", "s2" => 0.25)
    @test d["nuisance"]["inlier_prior"] == Dict("type" => "fixed_prior", "inlier_prior" => 0.9)
    @test d["nuisance"]["nu"] == Dict("type" => "fixed_nu", "nu" => 5.0)
    @test mixture_em_from_toml(d) == held
    @test config_dict(MixtureEmConfig(model_update = WeightedRefine()))["model_update"] == "weighted_refine"
    @test mixture_em_from_toml(Dict("model_update" => "weighted_refine")).model_update === WeightedRefine()
    # present keys override, absent keys inherit the shipped file
    m = mixture_em_from_toml(Dict("max_iter" => 3))
    @test m.max_iter == 3 && m.model_update == c.model_update && m.nuisance == c.nuisance
    # unknown tags, keys and tag fields are refused
    @test_throws ArgumentError mixture_em_from_toml(Dict("model_update" => "refine"))
    @test_throws ArgumentError mixture_em_from_toml(Dict("model_update" => "keep_model"))    # keep_model needs its tol
    @test_throws ArgumentError mixture_em_from_toml(Dict("model_update" => Dict("type" => "keep_model", "tol" => 1e-9, "rate" => 2)))
    # a retired tag is refused, naming its successor
    @test_throws ArgumentError mixture_em_from_toml(Dict("model_update" => Dict("type" => "keep", "tol" => 1e-9)))
    @test_throws ArgumentError mixture_em_from_toml(Dict("nuisance" => Dict("scale" => "estimate", "inlier_prior" => "estimated", "nu" => "estimated")))
    @test_throws ArgumentError mixture_em_from_toml(Dict("max_iters" => 3))
    nuis = Dict("scale" => "estimated", "inlier_prior" => "estimated", "nu" => "estimated")
    @test_throws ArgumentError mixture_em_from_toml(Dict("nuisance" => merge(nuis, Dict("nu" => "learn"))))
    @test_throws ArgumentError mixture_em_from_toml(Dict("nuisance" => merge(nuis, Dict("alpha" => "estimated"))))
    @test_throws ArgumentError mixture_em_from_toml(Dict("nuisance" => delete!(copy(nuis), "nu")))
    @test_throws ArgumentError mixture_em_from_toml(Dict("nuisance" => merge(nuis, Dict("scale" => Dict("type" => "fixed_s2", "value" => 1.0)))))
    @test_throws ArgumentError mixture_em_from_toml(Dict("nuisance" => merge(nuis, Dict("inlier_prior" => Dict("type" => "fixed_prior", "inlier_prior" => 1.5)))))
end

end
