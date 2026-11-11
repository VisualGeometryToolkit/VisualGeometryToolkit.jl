# The owner's ruling (4): fit_division_camera on the shared robust-estimation machinery. Its
# E-step is the posterior of the shared Student-t + uniform scorer (`weighting`, read by
# `scoring_from_toml`, absent keys inheriting [scoring.student_t_uniform]); its loop is the
# posterior-weight iteration (`PosteriorIrlsMethod`, `PosteriorIrlsState`) of a `MixtureEmConfig`
# (`mixture_em`, read by `mixture_em_from_toml`, absent keys inheriting [mixture_em]), stopped
# by the shared schema's parameter-change rule (STOP_TAGS "parameter_change": the last iterate).
# The private `outlier_halfwidth` and `max_iter` keys are gone: refused, naming the new form.
# Every other user of the shared iteration keeps the no-improvement rule (the incumbent).
@testitem "fit_division_camera: the shared scorer and EM tables" setup = [DivisionBoards] begin
    using Test, StableRNGs
    using TOML: TOML
    using Rotations: RotXYZ
    using VisualGeometryToolkitCore: RadialCamera, StudentTUniformScoring, JeffreysPrior, FlatLabelPrior,
        outlier_halfwidth, scale_prior, codimension, retcode, iterations
    using VisualGeometryToolkitEstimation: fit_division_camera, DivisionCameraProblem, scoring_from_toml,
        mixture_em_from_toml, config_dict, default_config_path, MixtureEmConfig, WeightedRefine, KeepModel,
        PosteriorIrlsMethod, PosteriorIrlsState, PosteriorLikelihood, FixedPrior, Estimated,
        NoImprovementStop, ParameterChangeStop, StopWhenNoImprovement, StopWhenParameterChangeLess, STOP_TAGS
    using VisualGeometryToolkitCore: stopping_criterion, weights, weight_kind, PosteriorWeights, inlier_mask, noise_scale,
        scale_source, ScaleEstimated, rms
    import VisualGeometryToolkitEstimation as VgtkEstimation

    tbl = TOML.parsefile(default_config_path())["fit_division_camera"]
    # the table keeps the fit's own keys and refers to the shared tables
    @test Set(keys(tbl)) == Set(["ncoef", "weighting", "mixture_em", "branch_audit_efficiency_constant"])
    @test !haskey(tbl, "outlier_halfwidth") && !haskey(tbl, "max_iter") && !haskey(tbl, "scoring")
    # the new form round-trips: scorer and EM read, written back, read back
    sc = scoring_from_toml(tbl["weighting"]; problem = DivisionCameraProblem)
    @test sc == StudentTUniformScoring(DivisionCameraProblem, 20.0; scale_prior = JeffreysPrior(), label_prior = FlatLabelPrior())
    @test codimension(sc) == 2 && outlier_halfwidth(sc) == 20.0
    @test scoring_from_toml(config_dict(sc); problem = DivisionCameraProblem) == sc
    @test all(config_dict(sc)[k] == v for (k, v) in tbl["weighting"])
    em = mixture_em_from_toml(tbl["mixture_em"]; section = "fit_division_camera.mixture_em")
    @test em.model_update isa WeightedRefine && em.max_iter == 30 && em.stop == ParameterChangeStop(1.0e-9)
    @test em.nuisance == (; scale = Estimated(), inlier_prior = FixedPrior(0.5), nu = Estimated())
    @test mixture_em_from_toml(config_dict(em)) == em
    @test VgtkEstimation._division_weighting_default() == sc && VgtkEstimation._division_mixture_em_default() == em
    @test config_dict(em)["stop"] == Dict("type" => "parameter_change", "rtol" => 1.0e-9)

    # the stop rule is a key of the shared schema: required (stated in the defaults file, nowhere in code),
    # its tags refused when unknown or short of a field, and every other user keeps the no-improvement rule
    @test TOML.parsefile(default_config_path())["mixture_em"]["stop"] == "no_improvement"
    @test MixtureEmConfig().stop === NoImprovementStop() && config_dict(MixtureEmConfig())["stop"] == "no_improvement"
    @test STOP_TAGS["parameter_change"] === ParameterChangeStop && STOP_TAGS["no_improvement"] === NoImprovementStop
    @test_throws ArgumentError mixture_em_from_toml(Dict("stop" => "parameter_change"))
    @test_throws ArgumentError mixture_em_from_toml(Dict("stop" => Dict("type" => "parameter_change", "tol" => 1.0)))
    @test_throws ArgumentError mixture_em_from_toml(Dict("stop" => "converged"))
    @test_throws ArgumentError ParameterChangeStop(-1.0)
    @test stopping_criterion(PosteriorIrlsMethod(nothing)) === StopWhenNoImprovement(1, 2)
    @test stopping_criterion(PosteriorIrlsMethod(nothing, KeepModel(1.0e-10))) === StopWhenNoImprovement(1, 1)
    @test stopping_criterion(PosteriorIrlsMethod(nothing, MixtureEmConfig())) === StopWhenNoImprovement(1, 1)
    @test stopping_criterion(PosteriorIrlsMethod(nothing, em)) === StopWhenParameterChangeLess(1.0e-9)

    cams = [board_camera(RotXYZ(a...), t) for (a, t) in zip(BOARD_ANGLES, BOARD_TS)]
    rc = Dict(enumerate(RadialCamera.(cams)))
    boards, _ = noisy_board_views(StableRNG(3), cams; σ = 0.4, ε = 0.08)
    vw = Dict(enumerate(boards))
    sz = (4000, 3000)
    # the old private keys are refused, naming the new form
    for (kw, says) in ((; outlier_halfwidth = 20.0) => "weighting", (; max_iter = 5) => "mixture_em", (; scoring = sc) => "weighting")
        err = try fit_division_camera(rc, vw; image_size = sz, kw...); nothing catch e; e end
        @test err isa ArgumentError && occursin("no keyword", err.msg) && occursin(says, err.msg)
    end
    # a scorer of another codimension, and an EM that holds the model, are refused
    @test_throws ArgumentError fit_division_camera(rc, vw; image_size = sz,
        weighting = StudentTUniformScoring(2, 20.0; codimension = 1, scale_prior = JeffreysPrior(), label_prior = FlatLabelPrior()))
    @test_throws ArgumentError fit_division_camera(rc, vw; image_size = sz,
        mixture_em = MixtureEmConfig(KeepModel(1.0e-9), em.nuisance, em.stop, 30))
    # and so is the incumbent rule: no cost is comparable between this fit's iterations
    @test_throws ArgumentError fit_division_camera(rc, vw; image_size = sz,
        mixture_em = MixtureEmConfig(em.model_update, em.nuisance, NoImprovementStop(), 30))
    # the defaults ARE the tables: the same fit, bit for bit
    a = fit_division_camera(rc, vw; image_size = sz)
    b = fit_division_camera(rc, vw; image_size = sz, weighting = sc, mixture_em = em)
    @test a.estimate == b.estimate && noise_scale(a) === noise_scale(b) && iterations(a) == iterations(b)
    # the scorer's half-width and the EM's cap act
    @test iterations(fit_division_camera(rc, vw; image_size = sz,
        mixture_em = MixtureEmConfig(em.model_update, em.nuisance, em.stop, 3))) <= 3
    c = fit_division_camera(rc, vw; image_size = sz,
        weighting = scoring_from_toml(Dict("type" => "student_t_uniform", "outlier_halfwidth" => 60.0); problem = DivisionCameraProblem))
    @test noise_scale(c) != noise_scale(a)
    # the contract's record of the EM: the E-step posteriors as the weights (PosteriorWeights, the MAP cut),
    # its ŝ estimated; n the support, rms posterior-weighted
    w = weights(a)
    @test weight_kind(a) === PosteriorWeights() && length(w) == sum(length, boards) && all(x -> 0 <= x <= 1, w)
    @test inlier_mask(a) == (w .> 0.5) && count(inlier_mask(a)) < length(w)
    @test noise_scale(a) === a.metadata.scale && scale_source(a) === ScaleEstimated && 0.2 < noise_scale(a) < 1.0
    sq = reduce(vcat, [a.state.sq_distances[id] for id in sort!(collect(keys(vw)))])
    @test a.n == count(>(0), w) && rms(a) ≈ sqrt(sum(w[i] * sq[i] for i in eachindex(w) if w[i] > 0) / sum(w))
    # the fit runs the shared types
    pb = VgtkEstimation.DivisionCameraProblem(VgtkEstimation._keyed_views(rc, vw)[2:3]..., nothing, 1, nothing,
        VgtkEstimation._division_normalization(VgtkEstimation._keyed_views(rc, vw)[3], VgtkEstimation._shared_center(VgtkEstimation._keyed_views(rc, vw)[2])),
        sum(length, boards), false)
    st, lh, method, _ = VgtkEstimation._division_irls(pb, sc, em, 2500.0)
    @test st isa PosteriorIrlsState && lh isa PosteriorLikelihood && method isa PosteriorIrlsMethod{Nothing, WeightedRefine, ParameterChangeStop}
end
