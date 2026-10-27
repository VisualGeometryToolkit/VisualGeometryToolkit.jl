@testitem "config_toml" setup = [HomographyScenes] begin
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
using VisualGeometryToolkitCore: AbstractLocalOptimization, HuberShape, TLSShape, WelschShape   # owned and exported by VisualGeometryToolkitCore
using VisualGeometryToolkitCore: HomographyMat
using VisualGeometryToolkitCore: MsacScoring   # owned and exported by VisualGeometryToolkitCore
using VisualGeometryToolkitCore: toml_key   # public (the TOML key of a field)
using VisualGeometryToolkitEstimation: LO_TAGS, config_dict, local_optimization_from_toml   # public, not exported
using VisualGeometryToolkitEstimation
import VisualGeometryToolkitCore
# A width-carrying strategy's scale multiple (its `ScaleMultiple` width mode).
sm(lo) = VgtkCore.lo_width(lo).value
using VisualGeometryToolkitEstimation: _dflt, _TOML_DEFAULTS, _dflt_lo, NoLocalOptimization

@testset "TOML configuration" begin

    @testset "the default names ONE thing" begin
        # RansacConfig() IS the file.
        rc = RansacConfig()
        @test rc.success_prob == _dflt("ransac", "success_prob")
        @test rc.max_trials == _dflt("ransac", "max_trials")
        @test rc.min_trials == _dflt("ransac", "min_trials") == 100
        # The core strategy's keyword defaults ARE its table.
        @test Irls{WelschShape}(scale_multiple = 1.0).max_iter == _dflt("local_optimization.irls", "max_iter")
        @test Irls{TLSShape}().max_iter == _dflt("local_optimization.irls", "max_iter")
        # Every pipeline deploys the file's strategy: the core's TLS IRLS,
        # following the scorer. (The strategies' tables and their cross-validated
        # operating points are LocalOptimizationDoneRight's.)
        for class in Symbol.(keys(_TOML_DEFAULTS["pipelines"]))
            @test _dflt_lo(class) == Irls{TLSShape}()
        end
        @test !haskey(_TOML_DEFAULTS, "tuning_constants")
    end

    @testset "a cross-validated scale multiple is LocalOptimizationDoneRight's to resolve" begin
        lo_from(loss, scale_multiple) = local_optimization_from_toml(
            Dict("type" => "irls", "loss" => loss, "scale_multiple" => scale_multiple); class = :homography
        )
        for class in Symbol.(keys(_TOML_DEFAULTS["pipelines"]))
            @test _dflt_lo(class) isa AbstractLocalOptimization
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
        @test occursin("LO_TAGS", msg)
        @test_throws ArgumentError lo_from("welsch", "not_a_number")

        # A LITERAL scale multiple claims no provenance, so any shape may take one.
        @test sm(lo_from("geman_mcclure", 1.25)) == 1.25
        @test sm(lo_from("cauchy", 0.9)) == 0.9

        # The retired spellings fail loudly: an old `k` or `kernel` key is an
        # unknown key, never silently read.
        @test_throws ArgumentError local_optimization_from_toml(
            Dict("type" => "irls", "kernel" => "welsch", "scale_multiple" => 1.0); class = :homography
        )
        @test_throws ArgumentError local_optimization_from_toml(
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
        # Every strategy table names every keyword its constructor defaults —
        # `scale_multiple` / `alpha` are REQUIRED per call, deliberately absent.
        # `scale_multiple` is cross-validated per model class, never defaulted
        # in the file, for EVERY arm that takes one.
        # The terminal placement's `inner` is a strategy, never a file default.
        required = Dict("irls" => (:scale_multiple,), "terminal_lo" => (:inner,))
        lo_table(tag) = get(_TOML_DEFAULTS["local_optimization"], tag, Dict{String, Any}())
        for (tag, T) in LO_TAGS
            tag == "no_local_optimization" && continue
            tbl = lo_table(tag)
            for f in fieldnames(T)
                f in get(required, tag, ()) && continue
                f in VgtkCore.toml_skip_fields(T) && continue   # the scale interface is not a config key
                @test haskey(tbl, toml_key(Any, f))
            end
        end
        # ...and the converse, asserted rather than assumed: a key listed in
        # `required` is deliberately ABSENT from the file. `scale_multiple` is
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
            @test haskey(tbl, "threshold")            # the default MSAC scorer's
            @test !haskey(tbl, "outlier_halfwidth")   # a marginal scorer's knob, not the core's
            @test haskey(tbl, "local_optimization")
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
                local_optimization = { type = "irls", loss = "huber", scale_multiple = 2.0 }

                [pipelines.essential]
                local_optimization = { type = "irls", loss = "welsch", max_iter = 7 }
                """
            )
            rc = RansacConfig(p)
            @test rc.max_trials == 42
            @test rc.min_trials == _dflt("ransac", "min_trials")      # inherited
            @test rc.success_prob == _dflt("ransac", "success_prob")

            # A strategy table from the file: present keys override, absent inherit.
            pipes = TOML.parsefile(p)["pipelines"]
            h = local_optimization_from_toml(pipes["homography"]["local_optimization"]; class = :homography)
            @test h isa Irls{HuberShape}
            @test sm(h) == 2.0
            @test h.max_iter == _dflt("local_optimization.irls", "max_iter")   # inherited

            e = local_optimization_from_toml(pipes["essential"]["local_optimization"]; class = :essential)
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
            lo_bad(tbl) = local_optimization_from_toml(tbl; class = :homography)
            @test_throws ArgumentError lo_bad(Dict("type" => "irls", "loss" => "welch", "scale_multiple" => 1.0))               # unknown loss
            @test_throws ArgumentError lo_bad(Dict("type" => "irlss", "loss" => "welsch", "scale_multiple" => 1.0))             # unknown type
            @test_throws ArgumentError lo_bad(Dict("type" => "irls", "scale_multiple" => 1.0))                                  # no loss
            # no width: the strategy follows the scorer's inlier threshold (not a refusal)
            @test VgtkCore.lo_width(lo_bad(Dict("type" => "irls", "loss" => "welsch"))) === nothing
            @test_throws ArgumentError lo_bad(Dict("type" => "irls", "loss" => "welsch", "scale_multiple" => 1.0, "loss_width" => 2.0))   # both widths
            @test_throws ArgumentError lo_bad(Dict("type" => "irls", "loss" => "welsch", "scale_multiple" => 1.0, "max_itr" => 3)) # typo'd knob
            # a retired tag names its successor
            @test_throws ArgumentError lo_bad(Dict("type" => "none"))
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
            for (tag, T) in LO_TAGS
                tag == "no_local_optimization" && continue
                takes_scale_multiple = hasfield(T, :width) || hasfield(T, :scale_multiple)
                takes_loss = VgtkEstimation.lo_parametric(T)
                # the minimal table for this tag: only the keys it actually takes
                tbl = Dict{String, Any}("type" => tag)
                takes_loss && (tbl["loss"] = "welsch")
                takes_scale_multiple && (tbl["scale_multiple"] = 1.0)
                hasfield(T, :inner) &&
                    (tbl["inner"] = Dict{String, Any}("type" => "irls", "loss" => "welsch", "scale_multiple" => 1.0))
                lo = local_optimization_from_toml(tbl; class = :homography)
                d = config_dict(lo)
                for f in fieldnames(T)
                    f in VgtkCore.toml_skip_fields(T) && continue   # scale interface: not a config key
                    @test haskey(d, toml_key(Any, f))
                end
                @test local_optimization_from_toml(d; class = :homography) == lo
            end
            # the null strategy and a width given as an absolute loss width round-trip too
            @test local_optimization_from_toml(config_dict(NoLocalOptimization()); class = :homography) == NoLocalOptimization()
            lw = Irls{WelschShape}(loss_width = 2.5)
            @test local_optimization_from_toml(config_dict(lw); class = :homography) == lw
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

        # `threshold` builds the DEFAULT scoring and nothing else: a caller's
        # `scoring` carries its own width, so it needs none, and both is
        # refused rather than one being silently ignored.
        prob = HomographyProblem(pairs)
        sc = MsacScoring(prob; threshold = 5.0)
        @test_throws ArgumentError find_homography(pairs; threshold = 5.0, scoring = sc)
        r_a = find_homography(pairs; threshold = 5.0, rng = StableRNG(285))
        r_s = find_homography(pairs; scoring = sc, rng = StableRNG(285))
        @test converged(r_s) && estimate(r_a) == estimate(r_s)      # the default IS MsacScoring(problem; threshold)
        lo = NoLocalOptimization()
        θ₀ = estimate(r_default)
        @test converged(locally_optimize(prob, θ₀; local_optimization = lo, scoring = sc))
        @test_throws ArgumentError locally_optimize(prob, θ₀; local_optimization = lo, threshold = 5.0, scoring = sc)
        @test_throws ArgumentError locally_optimize(prob, θ₀; local_optimization = lo)   # neither
    end

end
end
