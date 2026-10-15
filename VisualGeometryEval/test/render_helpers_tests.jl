# The Render module's public emission helpers (src/render): the pgfplots
# coordinate body, the evaluation curve, the facet rank, the one colour emitter,
# a PaperStyle's colours and arm records, the per-call LO time, and THE mAA rule
# in the budget panel (pooled over scenes, failures score 0).

@testitem "render_helpers" setup = [StyleFixtures] begin
    using Test, JLD2
    using VisualGeometryEval.Stats: finite_median, eligible_maa
    using VisualGeometryEval.Render: pgf_coords, evaluation_curve, family_rank, FAMILY_ORDER,
        color_defs, house_colors, paper_color_name, paper_color_defs, PANELS, PaperStyle,
        has_arm, arm_style, lo_us_per_call, budget_curves, emit_budget_faceted_curves, emit_budget_time_panel

    @testset "pgf_coords" begin
        @test pgf_coords([0.5, 1.0, 2.0], [1.0, NaN, Inf]) == "(0.500000,1.0000)"
        @test pgf_coords([10, 20], [0.25, 0.5]; fmt = "(%d,%.4f)") == "(10,0.2500) (20,0.5000)"
        @test pgf_coords(Float64[], Float64[]) == ""
    end

    @testset "evaluation_curve" begin
        ro = Dict("a" => [1.0 4.0; NaN 2.0; 3.0 Inf])
        @test evaluation_curve(ro, "a", finite_median) == [2.0, 3.0]
        @test evaluation_curve(ro, "a", eligible_maa) ≈ [(0.9 + 0.7) / 3, (0.6 + 0.8) / 3]
    end

    @testset "family_rank" begin
        @test [family_rank(f) for f in FAMILY_ORDER] == collect(1:length(FAMILY_ORDER))
        @test family_rank(:unknown) == length(FAMILY_ORDER) + 1
    end

    @testset "colour definitions" begin
        @test color_defs(["a" => "00ff00", "b" => "112233"]) ==
              "\\definecolor{a}{HTML}{00ff00}\n\\definecolor{b}{HTML}{112233}"
        @test house_colors(["a" => "00ff00"]) == "\\definecolor{a}{HTML}{00ff00}\n"
        @test paper_color_name("IG  +LO") == "ColorIGLO"
        # the palette, plus an arm colour carried by its record's hex
        style = probe_style(["a" => ProbeLO(:welsch, :irls, "A")], Dict("a" => (color = "Colorstulo", hex = "4363d8"));
                            colors = [paper_color_name(k) => h for (k, h) in ["MSAC" => "ffe119", "IG  sel" => "3cb44b"]])
        @test paper_color_defs(style) == "\\providecolor{ColorIGsel}{HTML}{3CB44B}\n\\providecolor{ColorMSAC}{HTML}{FFE119}\n\\providecolor{Colorstulo}{HTML}{4363D8}"
        # an arm hex that contradicts the palette is refused
        @test_throws ArgumentError probe_style(["a" => ProbeLO(:welsch, :irls, "A")],
                                               Dict("a" => (color = "ColorMSAC", hex = "000000"));
                                               colors = ["ColorMSAC" => "ffe119"])
    end

    @testset "PANELS by id" begin
        @test keys(PANELS) == (:eR_mean, :mAA_mean)
        @test PANELS.eR_mean.id == "eR_mean" && PANELS[2] === PANELS.mAA_mean
    end

    @testset "arm records derive from the traits" begin
        style = probe_style(["a" => ProbeLO(:tls, :hq, "TLS-HQ", true)], Dict("a" => (color = "ColorA",)))
        @test has_arm(style, "a") && !has_arm(style, "b")
        @test arm_style(style, "a") == (family = :tls, label = "TLS-HQ", color = "ColorA", marker = "triangle", dashed = true)
        @test_throws ErrorException probe_style(["u" => ProbeLO(:unspecified, :hq, "U")], Dict("u" => (color = "C",)))
    end

    @testset "lo_us_per_call" begin
        r = (lo_ns_total = [3.0e6, 0.0], lo_calls_total = [1000, 0])
        @test lo_us_per_call(r, 1) == 3.0 && isnan(lo_us_per_call(r, 2))
    end

    @testset "budget curves → panels: pooled eligibility-ruled mAA" begin
        style = probe_style(["IG  +WIrls" => ProbeLO(:welsch, :irls, "W-IRLS")], Dict("IG  +WIrls" => (color = "ColorA",)))
        dir = mktempdir()
        lo_counts = [100, 200]; sample_counts = [100, 150, 200]
        # scene 1: 2 pairs, one failed run; scene 2: 4 pairs, all at 0°
        e1 = [0.0 0.0; NaN NaN]; e2 = zeros(4, 2)
        o1 = [0.0 9.0 0.0; 0.0 9.0 0.0]; o2 = zeros(4, 3)   # the oracle's dense batch grid
        files = [joinpath(dir, "s$i.jld2") for i in 1:2]
        for (f, e, o) in zip(files, (e1, e2), (o1, o2))
            jldsave(f; lo_counts, sample_counts, err_lo = Dict("IG  +WIrls" => e), oracle_err_batch = o)
        end
        c = budget_curves(files; arms = ["IG  +WIrls"])
        # pooled: 5 of 6 pairs at accuracy 1 → 0.8333 (the per-scene mean would be 0.75)
        @test c.budgets == lo_counts && all(isapprox.(c.maa["IG  +WIrls"], 5 / 6; atol = 1e-12))
        # the oracle is read at the LO checkpoints on its dense grid: batch 150 is skipped
        @test c.oracle_maa == [1.0, 1.0] && c.oracle_med == [0.0, 0.0]
        @test c.outcomes["IG  +WIrls"].n_failed == 1
        path = emit_budget_faceted_curves(style, joinpath(dir, "b.tex"); budgets = c.budgets, med = c.med, maa = c.maa,
                                          med_ylabel = "med", legend_name = "L", deploy = true, xlog = false,
                                          oracle_med = c.oracle_med, oracle_maa = c.oracle_maa)
        @test occursin("(100,0.8333) (200,0.8333)", read(path, String))
        @test occursin("  xmajorgrids, ymajorgrids,", read(path, String))     # the style's grid options
        nogrid = probe_style(["IG  +WIrls" => ProbeLO(:welsch, :irls, "W-IRLS")], Dict("IG  +WIrls" => (color = "ColorA",));
                             grid = "")
        p2 = emit_budget_faceted_curves(nogrid, joinpath(dir, "b2.tex"); budgets = c.budgets, med = c.med, maa = c.maa,
                                        med_ylabel = "med", legend_name = "L", deploy = true, xlog = false)
        @test !occursin("majorgrids", read(p2, String)) && !occursin(", ,", read(p2, String))
        tp = emit_budget_time_panel(style, joinpath(dir, "b_time.tex"); budgets = c.budgets, maa = c.maa,
                                    percall_us = Dict("IG  +WIrls" => 10.0), legend_name = "T",
                                    xmax = 1, xticks = "0,1", maamax = 1.0)
        @test occursin("(0.010,0.8333) (0.020,0.8333)", read(tp, String))
        # a scene file missing an arm, or an untimed arm, is refused
        @test_throws ArgumentError budget_curves(files; arms = ["IG  +WIrls", "IG  +Lift"])
        @test_throws ArgumentError emit_budget_time_panel(style, joinpath(dir, "x.tex"); budgets = c.budgets, maa = c.maa,
            percall_us = Dict{String,Float64}(), legend_name = "T", xmax = 1, xticks = "0,1", maamax = 1.0)
    end
end
