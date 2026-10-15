@testitem "residual_mixture" begin
# =============================================================================
# The posterior-weight iteration's bound updates, and `fit_residual_mixture`
# =============================================================================
#
# What this pins: the DEFAULT model update (`WeightedRefine`) and nuisance
# policies (ŝ², ν estimated; π held at the state's own value) are the
# two-part update the RANSAC local optimization always ran, bit for bit;
# `fit_residual_mixture` refuses a model update that is not `KeepModel`; the
# `[mixture_em]` TOML round trip, and refusal of unknown tags and keys; and
# `fit_residual_mixture` itself — its scale, inlier fraction and MAP labels on a
# known mixture, its `[mixture_em]` defaults, and a calibration pipeline's call —
# under a test-only scorer with a uniform outlier model (`HalfwidthScoring`
# below: the scoring contract at a threshold, plus `outlier_halfwidth`).
# =============================================================================

using Test
using Random
using VisualGeometryCore: inlier_mask, is_inlier, inlier_weight, robust_weight,
    update_nuisance!, update_weights!, retcode, iterations
# The robust-statistics toolkit (VisualGeometryCore): the posterior-weight
# state's accessors and updates, and the nuisance-part policies.
using VisualGeometryCore: s_hat_sq, student_t_dof, inlier_prior, sq_residuals, update_scale!, update_nu!,
    Estimated, FixedS2, FixedPrior, FixedNu, ConsensusDof
using VisualGeometryCore: MsacScoring, median_seeded_posterior
using RobustVisualGeometry
using RobustVisualGeometry: PosteriorLikelihood, MixtureEmConfig, KeepModel, WeightedRefine,
    mixture_em_from_toml, config_dict, ResidualMixtureMetadata
using VisualGeometryCore: VisualGeometryCore, AbstractScoring, FitResult, PosteriorWeightState, estimate, rms,
    converged, metadata

# A scorer with a uniform outlier model, test-only: its consensus set is the
# residuals below its threshold (inliers first, each part in index order), its
# per-point scores the squared residuals (what `fit_residual_mixture` seeds the
# posterior from), and its outlier half-width. Scale prior: the default.
struct HalfwidthScoring{DG} <: AbstractScoring
    threshold::Float64
    outlier_halfwidth::Float64
end
HalfwidthScoring(dg::Integer; threshold, outlier_halfwidth) =
    HalfwidthScoring{Int(dg)}(Float64(threshold), Float64(outlier_halfwidth))
mutable struct HalfwidthScoringWorkspace
    order::Vector{Int}
    k::Int
end
VisualGeometryCore.codimension(::HalfwidthScoring{DG}) where {DG} = DG
VisualGeometryCore.outlier_halfwidth(s::HalfwidthScoring) = s.outlier_halfwidth
VisualGeometryCore.scoring_workspace(::HalfwidthScoring, n::Int, _d_g::Int) =
    HalfwidthScoringWorkspace(Vector{Int}(undef, n), 0)
VisualGeometryCore.inlier_count(::HalfwidthScoring, W::HalfwidthScoringWorkspace) = W.k
VisualGeometryCore.consensus_order(::HalfwidthScoring, W::HalfwidthScoringWorkspace) = W.order
function VisualGeometryCore.score!(W::HalfwidthScoringWorkspace, s::HalfwidthScoring, scores, r, n;
                                   sample = nothing, bound = -Inf)
    sq_threshold = s.threshold^2
    k = 0
    for i in 1:n
        scores[i] = Float64(r[i])^2
        scores[i] < sq_threshold && (k += 1; W.order[k] = i)
    end
    j = k
    for i in 1:n
        scores[i] < sq_threshold || (j += 1; W.order[j] = i)
    end
    W.k = k
    return Float64(k)
end
# A posterior seeded at the residuals' median (VisualGeometryCore's seed).
median_posterior(r, n) = median_seeded_posterior(r .^ 2, n; outlier_halfwidth = 40.0, codimension = 2)

# Codimension-2 reprojection residual NORMS: Rayleigh inliers at σ, gross
# outliers uniform over the ±halfwidth square (the mixture's outlier support).
function reprojection_residuals(rng; n_in, n_out, σ, halfwidth)
    r_in = [σ * hypot(randn(rng), randn(rng)) for _ in 1:n_in]
    r_out = [hypot(halfwidth * (2rand(rng) - 1), halfwidth * (2rand(rng) - 1)) for _ in 1:n_out]
    return vcat(r_in, r_out), vcat(trues(n_in), falses(n_out))
end

@testset "default nuisance policies == the two-part update, bit for bit" begin
    rng = MersenneTwister(3)
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

@testset "fit_residual_mixture: the model is held" begin
    # a weighted refit needs a model to refit — refused, not ignored (before
    # the scorer is read)
    r, _ = reprojection_residuals(MersenneTwister(11); n_in = 100, n_out = 10, σ = 0.5, halfwidth = 50.0)
    @test_throws ArgumentError fit_residual_mixture(r, MsacScoring(2; threshold = 2.0); model_dof = 4,
                                                    config = MixtureEmConfig(model_update = WeightedRefine()))
end

@testset "fit_residual_mixture: a known mixture" begin
    # 400 Rayleigh inliers at σ = 0.8 per axis, 100 outliers uniform over the
    # ±40 square: the EM recovers σ, the inlier fraction and every label.
    σ, a = 0.8, 40.0
    r, truth = reprojection_residuals(MersenneTwister(5); n_in = 400, n_out = 100, σ, halfwidth = a)
    sc = HalfwidthScoring(2; threshold = 3.0, outlier_halfwidth = a)
    fit = fit_residual_mixture(r, sc; model_dof = 4)
    @test fit isa FitResult{PosteriorWeightState, ResidualMixtureMetadata}
    @test converged(fit) && 1 <= iterations(fit) < MixtureEmConfig().max_iter
    p = estimate(fit)
    @test sqrt(s_hat_sq(p)) ≈ σ rtol = 0.05
    @test inlier_prior(p) ≈ 0.8 atol = 0.02
    @test inlier_mask(p) == truth
    @test all(i -> is_inlier(p, i) == truth[i], eachindex(r))
    # rms and n over the MAP inliers; the metadata the NLL the EM lowered
    @test fit.n == count(truth)
    @test rms(fit) ≈ sqrt(sum(abs2, r[truth]) / count(truth))
    @test isfinite(metadata(fit).nll)
    # the residuals are not moved, and the vector type does not matter
    r0 = copy(r)
    fit32 = fit_residual_mixture(Float32.(r), sc; model_dof = 4)
    @test r == r0 && inlier_mask(estimate(fit32)) == truth
    # a partition too thin to carry a scale: nothing
    @test fit_residual_mixture(r, HalfwidthScoring(2; threshold = 1.0e-6, outlier_halfwidth = a); model_dof = 4) === nothing
end

@testset "fit_residual_mixture: the [mixture_em] defaults" begin
    c = MixtureEmConfig()
    @test c.model_update == KeepModel(1.0e-10) && c.max_iter == 200
    @test c.nuisance.scale isa Estimated && c.nuisance.inlier_prior isa Estimated && c.nuisance.nu isa Estimated
    # the defaults are what an omitted `config` runs
    r, _ = reprojection_residuals(MersenneTwister(7); n_in = 200, n_out = 50, σ = 1.2, halfwidth = 60.0)
    sc = HalfwidthScoring(2; threshold = 4.0, outlier_halfwidth = 60.0)
    a = fit_residual_mixture(r, sc; model_dof = 3)
    b = fit_residual_mixture(r, sc; model_dof = 3, config = mixture_em_from_toml(Dict{String, Any}()))
    @test (s_hat_sq(estimate(a)), student_t_dof(estimate(a)), inlier_prior(estimate(a)), iterations(a), a.n) ===
          (s_hat_sq(estimate(b)), student_t_dof(estimate(b)), inlier_prior(estimate(b)), iterations(b), b.n)
    # a held part stays held through the EM
    h = fit_residual_mixture(r, sc; model_dof = 3,
                             config = MixtureEmConfig(nuisance = (; scale = Estimated(), inlier_prior = FixedPrior(0.75),
                                                                  nu = FixedNu(6.0))))
    @test inlier_prior(estimate(h)) === 0.75 && student_t_dof(estimate(h)) === 6.0
end

@testset "fit_residual_mixture: a calibration pipeline's call" begin
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
    rng = MersenneTwister(13)
    residuals, truth = reprojection_residuals(rng; n_in = 900, n_out = 25, σ = 0.4, halfwidth = 300.0)
    n = length(residuals)
    dof = 2n - 150                                      # a bundle with 150 free parameters
    d = classification.codimension
    m = cld(d * n - dof, d)
    @test m == 75
    scoring = HalfwidthScoring(d; threshold = 2.0, outlier_halfwidth = classification.outlier_halfwidth)
    fit = fit_residual_mixture(residuals, scoring; model_dof = m + 1, config = classification.em)
    @test fit !== nothing && converged(fit)
    pws = estimate(fit)
    @test sqrt(s_hat_sq(pws)) ≈ 0.4 rtol = 0.1
    @test fit.n == count(inlier_mask(pws)) && n - fit.n >= 20
    @test count(inlier_mask(pws) .!= truth) <= 3
    @test rms(fit) < 1.0
    # ν held at the seed's posterior-predictive dof: unchanged by the EM
    seed = fit_residual_mixture(residuals, scoring; model_dof = m + 1,
                                config = MixtureEmConfig(nuisance = (; scale = Estimated(), inlier_prior = Estimated(),
                                                                     nu = ConsensusDof()), max_iter = 1))
    @test student_t_dof(pws) == student_t_dof(estimate(seed))
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
