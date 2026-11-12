@testitem "config_toml" setup = [HomographyScenes] begin
    using VisualGeometryToolkitCore: Irls, Triggs
    using VisualGeometryToolkitCore: inlier_mask   # the robust contract's accessors
    import VisualGeometryToolkitCore as VgtkCore
    import VisualGeometryToolkitEstimation as VgtkEstimation
# =============================================================================
# TOML configuration — configs/default.toml is the ONE default authority
# =============================================================================
#
# What this pins, in order: the shipped file is what the constructors
# actually default to; the file is COMPLETE over every config field; loaders
# override per key and inherit the rest; unknown keys/tags are refused, never
# defaulted; config → table → config round-trips exactly.
# =============================================================================

using Test
using TOML
using Random
using StableRNGs
using StaticArrays
using VisualGeometryToolkitCore: converged, estimate, iterations   # VGC accessors (public)
using VisualGeometryToolkitCore: AbstractOptimizationStrategy, CauchyShape, GaussianShape, HuberShape, TLSShape, WelschShape   # owned and exported by VisualGeometryToolkitCore
using VisualGeometryToolkitCore: HomographyMat
using VisualGeometryToolkitCore: MsacScoring   # owned and exported by VisualGeometryToolkitCore
using VisualGeometryToolkitCore: toml_key   # public (the TOML key of a field)
using VisualGeometryToolkitEstimation: OPTIMIZATION_STRATEGY_TAGS, config_dict, optimization_strategy_from_toml   # public, not exported
using VisualGeometryToolkitEstimation
import VisualGeometryToolkitCore
# A width-carrying strategy's efficiency constant (its `EfficiencyConstant` width mode).
sm(lo) = VgtkCore.lo_width(lo).value
using VisualGeometryToolkitEstimation: _dflt, _TOML_DEFAULTS, _dflt_lo

@testset "TOML configuration" begin

    @testset "the default names ONE thing" begin
        # RansacConfig() IS the file.
        rc = RansacConfig()
        @test rc.success_prob == _dflt("ransac", "success_prob")
        @test rc.max_trials == _dflt("ransac", "max_trials")
        @test rc.min_trials == _dflt("ransac", "min_trials") == 100
        # The core strategy's keyword defaults ARE its table.
        @test Irls{WelschShape}(efficiency_constant = 1.0).max_iter == VgtkCore.toml_default(VgtkCore._STRATEGY_DEFAULTS, "optimization_strategy.irls", "max_iter")
        @test Irls{TLSShape}().max_iter == VgtkCore.toml_default(VgtkCore._STRATEGY_DEFAULTS, "optimization_strategy.irls", "max_iter")
        # Every pipeline deploys the file's strategy: the core's TLS IRLS,
        # following the scorer — the circle and the ellipse the robustified LM
        # at its stated multiple. (The strategies' tables and their cross-validated operating
        # points are LocalOptimizationDoneRight's.)
        for class in Symbol.(keys(_TOML_DEFAULTS["pipelines"]))
            @test _dflt_lo(class) == (class in (:circle, :ellipse) ? Triggs{WelschShape}(efficiency_constant = 2.5) :
                                      class === :bearing_pose ? Irls{GaussianShape}() : Irls{TLSShape}())
        end
        @test !haskey(_TOML_DEFAULTS, "tuning_constants")
    end

    @testset "a cross-validated efficiency constant is LocalOptimizationDoneRight's to resolve" begin
        lo_from(loss, efficiency_constant) = optimization_strategy_from_toml(
            Dict("type" => "irls", "loss" => loss, "efficiency_constant" => efficiency_constant); class = :homography
        )
        for class in Symbol.(keys(_TOML_DEFAULTS["pipelines"]))
            @test _dflt_lo(class) isa AbstractOptimizationStrategy
        end
        # The core has no tuning constants: the request is refused with the way
        # out (load through the merged registry, or give a number).
        @test_throws ArgumentError lo_from("welsch", "cross_validated")
        msg = try
            lo_from("welsch", "cross_validated"); ""
        catch e
            sprint(showerror, e)
        end
        @test occursin("LocalOptimizationDoneRight", msg)
        @test occursin("LocalOptimizationDoneRight.LODR_STRATEGY_TAGS", msg)   # its registry's name since LODR 0.9
        @test_throws ArgumentError lo_from("welsch", "not_a_number")

        # A LITERAL efficiency constant claims no provenance, so any shape may take one.
        @test sm(lo_from("geman_mcclure", 1.25)) == 1.25
        @test sm(lo_from("cauchy", 0.9)) == 0.9

        # The retired spellings fail loudly: an old `k` or `kernel` key is an
        # unknown key, never silently read.
        @test_throws ArgumentError optimization_strategy_from_toml(
            Dict("type" => "irls", "kernel" => "welsch", "efficiency_constant" => 1.0); class = :homography
        )
        @test_throws ArgumentError optimization_strategy_from_toml(
            Dict("type" => "irls", "loss" => "welsch", "k" => 1.0); class = :homography
        )
    end

    @testset "the file is complete over every config field" begin
        # A defaulted knob that is not in the file would be a second authority.
        for f in fieldnames(RansacConfig)
            @test haskey(_TOML_DEFAULTS["ransac"], toml_key(Any, f))
        end
        for f in fieldnames(VgtkEstimation.MixtureEmConfig)
            @test haskey(_TOML_DEFAULTS["mixture_em"], toml_key(Any, f))
        end
        # SIMFIT++'s constructor defaults are the file's [simfitpp] table, every field, and nothing else
        @test Set(keys(_TOML_DEFAULTS["simfitpp"])) == Set(toml_key(Any, f) for f in fieldnames(VgtkEstimation.SimFitPP))
        let cfg = VgtkEstimation.SimFitPP()
            for f in fieldnames(VgtkEstimation.SimFitPP)
                @test getfield(cfg, f) == _dflt("simfitpp", toml_key(Any, f))
            end
        end
        # Every strategy table names every keyword its constructor defaults —
        # `efficiency_constant` / `alpha` are REQUIRED per call, deliberately absent.
        # `efficiency_constant` is cross-validated per model class, never defaulted
        # in the file, for EVERY arm that takes one.
        # The terminal placement's `inner` is a strategy, never a file default.
        required = Dict("irls" => (:efficiency_constant,), "terminal_lo" => (:inner,))
        lo_table(tag) = get(VgtkCore.lo_defaults(tag == "none" ? VgtkEstimation.TerminalLO : OPTIMIZATION_STRATEGY_TAGS[tag]), tag, Dict{String, Any}())
        for (tag, T) in OPTIMIZATION_STRATEGY_TAGS
            T === nothing && continue   # "none": no local optimization
            tbl = lo_table(tag)
            for f in fieldnames(T)
                f in get(required, tag, ()) && continue
                f in VgtkCore.toml_skip_fields(T) && continue   # the scale interface is not a config key
                @test haskey(tbl, toml_key(Any, f))
            end
        end
        # ...and the converse, asserted rather than assumed: a key listed in
        # `required` is deliberately ABSENT from the file. `efficiency_constant` is
        # cross-validated per model class, so a file default for it would
        # silently defeat that resolution — which is exactly what a literal 1.5
        # did for the four fixed-loss arms.
        for (tag, fields) in required
            tbl = lo_table(tag)
            for f in fields
                @test !haskey(tbl, toml_key(Any, f))
            end
        end
        for (class, tbl) in _TOML_DEFAULTS["pipelines"]
            @test haskey(tbl, "scoring")              # the class's scorer table (scoring_from_toml)
            @test !haskey(tbl, "threshold") && !haskey(tbl, "outlier_halfwidth")   # never a bare width: it is its scorer's
            @test haskey(tbl, "local_optimization_strategy")
        end
    end

    @testset "loaders: present keys override, absent keys inherit" begin
        mktempdir() do dir
            p = joinpath(dir, "run.toml")
            write(
                p, """
                [ransac]
                max_trials = 42

                [pipelines.homography]
                local_optimization_strategy = { type = "irls", loss = "huber", efficiency_constant = 2.0 }

                [pipelines.essential]
                local_optimization_strategy = { type = "irls", loss = "welsch", max_iter = 7 }
                """
            )
            rc = RansacConfig(p)
            @test rc.max_trials == 42
            @test rc.min_trials == _dflt("ransac", "min_trials")      # inherited
            @test rc.success_prob == _dflt("ransac", "success_prob")

            # A strategy table from the file: present keys override, absent inherit.
            pipes = TOML.parsefile(p)["pipelines"]
            h = optimization_strategy_from_toml(pipes["homography"]["local_optimization_strategy"]; class = :homography)
            @test h isa Irls{HuberShape}
            @test sm(h) == 2.0
            @test h.max_iter == VgtkCore.toml_default(VgtkCore._STRATEGY_DEFAULTS, "optimization_strategy.irls", "max_iter")   # inherited

            e = optimization_strategy_from_toml(pipes["essential"]["local_optimization_strategy"]; class = :essential)
            @test e isa Irls{WelschShape}
            @test e.max_iter == 7
            @test VgtkCore.lo_width(e) === nothing                 # no width: follows the scorer
        end
    end

    @testset "bad configs are refused, never silently defaulted" begin
        mktempdir() do dir
            bad(name, text) = (p = joinpath(dir, name); write(p, text); p)
            @test_throws ArgumentError RansacConfig(bad("a.toml", "[ransac]\nmax_iteration = 5\n"))          # typo
            # A misspelled or missing [ransac] table is refused, not read as the defaults.
            @test_throws ArgumentError RansacConfig(bad("b.toml", "[ransc]\nmax_trials = 5\n"))
            @test_throws ArgumentError RansacConfig(bad("c.toml", "[RANSAC]\nmax_trials = 5\n"))
            @test_throws ArgumentError RansacConfig(bad("d.toml", "[pipelines.homography]\nthreshold = 3.0\n"))
            # Out-of-range values are refused at construction, for every sampler.
            @test_throws ArgumentError RansacConfig(bad("e.toml", "[ransac]\nmax_trials = 0\n"))
            @test_throws ArgumentError RansacConfig(max_trials = 0)
            @test_throws ArgumentError RansacConfig(min_trials = -1)
            @test_throws ArgumentError RansacConfig(success_prob = 0.0)
            @test_throws ArgumentError RansacConfig(success_prob = 1.5)
            @test RansacConfig(success_prob = 1.0).success_prob == 1.0
            lo_bad(tbl) = optimization_strategy_from_toml(tbl; class = :homography)
            @test_throws ArgumentError lo_bad(Dict("type" => "irls", "loss" => "welch", "efficiency_constant" => 1.0))               # unknown loss
            @test_throws ArgumentError lo_bad(Dict("type" => "irlss", "loss" => "welsch", "efficiency_constant" => 1.0))             # unknown type
            @test_throws ArgumentError lo_bad(Dict("type" => "irls", "efficiency_constant" => 1.0))                                  # no loss
            # no width: the strategy follows the scorer's inlier threshold (not a refusal)
            @test VgtkCore.lo_width(lo_bad(Dict("type" => "irls", "loss" => "welsch"))) === nothing
            @test_throws ArgumentError lo_bad(Dict("type" => "irls", "loss" => "welsch", "efficiency_constant" => 1.0, "loss_width" => 2.0))   # both widths
            @test_throws ArgumentError lo_bad(Dict("type" => "irls", "loss" => "welsch", "efficiency_constant" => 1.0, "max_itr" => 3)) # typo'd knob
            # a retired tag names its successor
            @test_throws ArgumentError lo_bad(Dict("type" => "no_local_optimization"))
        end
    end

    @testset "round trip" begin
        mktempdir() do dir
            # RansacConfig → table → file → RansacConfig.
            rc = RansacConfig(max_trials = 777)
            p = joinpath(dir, "cfg.toml")
            open(io -> TOML.print(io, Dict("ransac" => config_dict(rc))), p, "w")
            @test RansacConfig(p) == rc
            # config_dict is derived from fieldnames: every strategy field appears.
            for (tag, T) in OPTIMIZATION_STRATEGY_TAGS
                T === nothing && continue   # "none": no local optimization
                takes_efficiency_constant = hasfield(T, :width) || hasfield(T, :efficiency_constant)
                takes_loss = VgtkCore.lo_parametric(T)
                # the minimal table for this tag: only the keys it actually takes
                tbl = Dict{String, Any}("type" => tag)
                takes_loss && (tbl["loss"] = "welsch")
                takes_efficiency_constant && (tbl["efficiency_constant"] = 1.0)
                hasfield(T, :inner) &&
                    (tbl["inner"] = Dict{String, Any}("type" => "irls", "loss" => "welsch", "efficiency_constant" => 1.0))
                lo = optimization_strategy_from_toml(tbl; class = :homography)
                d = config_dict(lo)
                for f in fieldnames(T)
                    f in VgtkCore.toml_skip_fields(T) && continue   # scale interface: not a config key
                    @test haskey(d, toml_key(Any, f))
                end
                @test optimization_strategy_from_toml(d; class = :homography) == lo
            end
            # the null strategy and a width given as an absolute loss width round-trip too
            @test optimization_strategy_from_toml(config_dict(nothing); class = :homography) == nothing
            lw = Irls{WelschShape}(loss_width = 2.5)
            @test optimization_strategy_from_toml(config_dict(lw); class = :homography) == lw
        end
    end

    @testset "find_* defaults come from the file, and kwargs override" begin
        rng = StableRNG(3)
        H = HomographyMat{Float64}(Tuple(SA[1.1 0.0 5.0; 0.0 1.1 -3.0; 0.0 0.0 1.0]))
        src, dst = homography_pairs(rng, H; n_in = 40, source = rng -> SVector{2, Float64}(randn(rng, 2) .* 100 .+ 200))
        pairs = src .=> dst
        r_default = find_homography(pairs; rng = StableRNG(282))
        r_config = find_homography(pairs; config = RansacConfig(max_trials = 200, min_trials = 10), rng = StableRNG(283))
        @test converged(r_default) && converged(r_config)
        @test iterations(r_config) <= 200
        # `min_trials = 100` is the file's floor for the default run
        @test iterations(r_default) >= _dflt("ransac", "min_trials")

        # A width is its scorer's: a bare `threshold` is refused, alone or
        # beside a scorer; the default IS the class table's scorer.
        prob = HomographyProblem(pairs)
        sc = MsacScoring(prob; threshold = 5.0)
        @test_throws ArgumentError find_homography(pairs; threshold = 5.0, scoring = sc)
        @test_throws ArgumentError find_homography(pairs; threshold = 5.0)
        r_tbl = find_homography(pairs; rng = StableRNG(282),
            scoring = VgtkEstimation.scoring_from_toml(_dflt("pipelines.homography", "scoring"); problem = HomographyProblem{Float64}))
        @test estimate(r_tbl) == estimate(r_default) && inlier_mask(r_tbl) == inlier_mask(r_default)
        lo = nothing
        θ₀ = estimate(r_default)
        @test converged(locally_optimize(prob, θ₀; local_optimization_strategy = lo, scoring = sc))
        @test_throws ArgumentError locally_optimize(prob, θ₀; local_optimization_strategy = lo, threshold = 5.0, scoring = sc)
        @test_throws UndefKeywordError locally_optimize(prob, θ₀; local_optimization_strategy = lo)   # no scorer
    end

end
end

# The scorers on the same schema as the local optimization: a TOML table read by one reader through a tag registry,
# every key required, unknown keys refused, written back by config_dict.
@testitem "config_toml: scoring_from_toml, the scorers' tables (one schema with the local optimization)" begin
    using Test
    using VisualGeometryToolkitCore: RansacScoring, MsacScoring, SCORING_TAGS
    using VisualGeometryToolkitEstimation: scoring_from_toml, config_dict, HomographyProblem, EssentialProblem
    for (t, T) in (("msac", MsacScoring), ("ransac", RansacScoring)), P in (HomographyProblem, EssentialProblem)
        d = Dict("type" => t, "threshold" => 2.5)
        sc = scoring_from_toml(d; problem = P)
        @test sc == T(P; threshold = 2.5) && config_dict(sc) == d
        @test scoring_from_toml(config_dict(sc); problem = P) == sc
    end
    @test_throws r"needs `threshold`" scoring_from_toml(Dict("type" => "msac"); problem = HomographyProblem)
    @test_throws ArgumentError scoring_from_toml(Dict("type" => "msac", "threshold" => 1.0, "width" => 2.0); problem = HomographyProblem)
    @test_throws ArgumentError scoring_from_toml(Dict("type" => "lmeds", "threshold" => 1.0); problem = HomographyProblem)
    @test_throws r"needs a `type`" scoring_from_toml(Dict("threshold" => 1.0); problem = HomographyProblem)
end

# Every threshold scorer's table is the one reader's and writer's (AbstractThresholdScoring): a downstream family built
# by Core's @threshold_scoring_constructors, in a downstream merge of SCORING_TAGS, reads through `T(problem; threshold)`
# and writes its tag from the registry it is given.
@testitem "config_toml: a downstream threshold family reads and writes through the one threshold-scorer table" begin
    using Test
    import VisualGeometryToolkitCore as VgtkCore
    using VisualGeometryToolkitCore: AbstractThresholdScoring, @threshold_scoring_constructors, SCORING_TAGS, TagRegistry, MsacScoring
    using VisualGeometryToolkitEstimation: scoring_from_toml, config_dict, HomographyProblem
    struct HalfMsacScoring{DG, T} <: AbstractThresholdScoring{T}
        threshold::T
        HalfMsacScoring{DG, T}(threshold) where {DG, T} = new{DG, T}(threshold)
    end
    @threshold_scoring_constructors HalfMsacScoring
    tags = merge(SCORING_TAGS, TagRegistry(:scoring, "Scoring", "half_msac" => HalfMsacScoring))
    d = Dict("type" => "half_msac", "threshold" => 2.5)
    sc = scoring_from_toml(d; problem = HomographyProblem, tags)
    @test sc === HalfMsacScoring(HomographyProblem; threshold = 2.5)
    @test config_dict(sc; tags) == d && scoring_from_toml(config_dict(sc; tags); problem = HomographyProblem, tags) === sc
    @test_throws r"`outlier_halfwidth` is" scoring_from_toml(Dict("type" => "half_msac", "outlier_halfwidth" => 2.0); problem = HomographyProblem, tags)
    @test config_dict(MsacScoring(HomographyProblem; threshold = 2.0)) == Dict("type" => "msac", "threshold" => 2.0)   # Core's, unchanged
end

# The owner's ruling (c): the Student-t scorer's scale prior and label prior are required keys of its table, the stack's
# default (Jeffreys, flat) in configs/default.toml's [scoring.student_t_uniform] — a key the table leaves out inherits
# it there, as a strategy's keys do — and nothing defaulted in code; config_dict states both, and reads back.
@testitem "config_toml: the Student-t scorer's priors are its table's keys, defaulted in default.toml only" begin
    using Test
    using TOML: TOML
    import VisualGeometryToolkitCore as VgtkCore
    using VisualGeometryToolkitCore: StudentTUniformScoring, JeffreysPrior, IGPrior, FlatLabelPrior, ProfiledLabelPrior,
        BetaLabelPrior, SCORING_TAGS, scale_prior, label_prior
    using VisualGeometryToolkitEstimation: scoring_from_toml, config_dict, default_config_path, HomographyProblem, EssentialProblem
    P = EssentialProblem
    # the defaults file states them, and only there
    dflt = TOML.parsefile(default_config_path())["scoring"]["student_t_uniform"]
    @test dflt == Dict("scale_prior" => "jeffreys", "label_prior" => "flat")
    @test_throws UndefKeywordError StudentTUniformScoring(P, 20.0)
    # inherited: the stack default
    sc = scoring_from_toml(Dict("type" => "student_t_uniform", "outlier_halfwidth" => 20.0); problem = P)
    @test sc == StudentTUniformScoring(P, 20.0; scale_prior = JeffreysPrior(), label_prior = FlatLabelPrior())
    @test config_dict(sc) == Dict("type" => "student_t_uniform", "outlier_halfwidth" => 20.0,
                                  "scale_prior" => "jeffreys", "label_prior" => "flat")
    # stated: every prior, written back and read back
    for (t, sp, lp) in ((Dict("scale_prior" => "ig", "alpha0" => 2.0, "beta0" => 0.5), IGPrior(2.0, 0.5), FlatLabelPrior()),
                        (Dict("label_prior" => "profiled"), JeffreysPrior(), ProfiledLabelPrior()),
                        (Dict("scale_prior" => "jeffreys", "label_prior" => "beta", "alpha" => 0.5, "beta" => 2.0),
                         JeffreysPrior(), BetaLabelPrior(0.5, 2.0)))
        d = merge(Dict{String, Any}("type" => "student_t_uniform", "outlier_halfwidth" => 3.5), t)
        sc = scoring_from_toml(d; problem = HomographyProblem)
        @test scale_prior(sc) == sp && label_prior(sc) == lp && sc isa StudentTUniformScoring{Nothing, 2}
        @test scoring_from_toml(config_dict(sc); problem = HomographyProblem) == sc
        @test all(config_dict(sc)[k] == v for (k, v) in d)
    end
    # a prior's parameters: required with it, refused with another; unknown tags and keys refused
    T(kw...) = merge(Dict{String, Any}("type" => "student_t_uniform", "outlier_halfwidth" => 1.0), Dict(kw...))
    @test_throws r"needs `beta0`" scoring_from_toml(T("scale_prior" => "ig", "alpha0" => 1.0); problem = P)
    @test_throws r"takes none" scoring_from_toml(T("alpha0" => 1.0, "beta0" => 1.0); problem = P)
    @test_throws r"needs `alpha`" scoring_from_toml(T("label_prior" => "beta", "beta" => 1.0); problem = P)
    @test_throws r"takes none" scoring_from_toml(T("label_prior" => "flat", "alpha" => 1.0); problem = P)
    @test_throws ArgumentError scoring_from_toml(T("scale_prior" => "gamma"); problem = P)
    @test_throws ArgumentError scoring_from_toml(T("label_prior" => "uniform"); problem = P)
    @test_throws ArgumentError scoring_from_toml(T("prior" => "jeffreys"); problem = P)
    @test_throws r"needs `outlier_halfwidth`" scoring_from_toml(Dict("type" => "student_t_uniform"); problem = P)
    @test SCORING_TAGS["student_t_uniform"] === StudentTUniformScoring
end

# The scorer of record at a half-width (hoisted from BlobBoards' and CaliBlobs' copies): the Student-t table read with
# the stack's priors, the same scorer as the table read by hand.
@testitem "config_toml: default_student_t_uniform_scoring is the Student-t table of record at a half-width" begin
    using Test
    using VisualGeometryToolkitCore: StudentTUniformScoring, JeffreysPrior, FlatLabelPrior
    using VisualGeometryToolkitEstimation: default_student_t_uniform_scoring, scoring_from_toml, HomographyProblem
    sc = default_student_t_uniform_scoring(HomographyProblem{Float64}, 5)
    @test sc == scoring_from_toml(Dict("type" => "student_t_uniform", "outlier_halfwidth" => 5.0); problem = HomographyProblem{Float64})
    @test sc == StudentTUniformScoring(HomographyProblem{Float64}, 5.0; scale_prior = JeffreysPrior(), label_prior = FlatLabelPrior())
end

# The one refusal of a retired keyword (public since 0.9, so a downstream package refuses its own the same way).
@testitem "RetiredKeyword: a retired keyword refused by name, its default returning nothing" begin
    using Test
    using VisualGeometryToolkitEstimation: RetiredKeyword, refuse_retired_keyword
    f(; image_size = RetiredKeyword()) = (refuse_retired_keyword(image_size, :image_size, "the BoundedCamera's extent"); :ok)
    @test f() === :ok
    @test_throws r"the keyword `image_size` is retired: give `the BoundedCamera's extent`" f(; image_size = (640, 480))
end
