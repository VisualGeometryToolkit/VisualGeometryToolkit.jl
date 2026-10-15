# Render.miscal_summary_rows / emit_miscal_summary_table: the miscalibration
# summary table from miscalibration sidecars, the style's arms in the faceted order.

@testitem "miscal_summary_table" setup = [TuningFixtures, StyleFixtures] begin
    using Test
    using VisualGeometryEval.Render: miscal_summary_rows, emit_miscal_summary_table
    using VisualGeometryEval.Tuning: save_tuning_artifact, GridSpec, PairUnit

    @testset "miscal summary table" begin
        # Three style arms (paper-owned colours + ramp steps), one unstyled rider.
        style = probe_style(
            ["IG  +WIrls" => ProbeLO(:welsch, :irls, "W-IRLS"),
             "IG  +Lift"  => ProbeLO(:welsch, :hq, "W-HQ"),
             "IG  +GM-HQ" => ProbeLO(:gm, :hq, "GM-HQ")],
            Dict("IG  +WIrls" => (color = "ColorA", shade = 1),
                 "IG  +Lift"  => (color = "ColorB", shade = 2),
                 "IG  +GM-HQ" => (color = "ColorC", shade = 2)))
        dir = mktempdir()
        zetas = [1 / 8, 1.0, 8.0]
        mk(vals) = hcat([fill(v, 10) for v in vals]...)          # 10 pairs × 3 ζ, constant per column
        spec = fixture_spec(Dict("type" => "windowed_bootstrap"))
        save_sidecar(path; kw...) = save_tuning_artifact(path, spec; grids = Dict{Symbol,GridSpec}(), unit = PairUnit(),
                                                         code = (), kw...)
        sc1 = joinpath(dir, "a.jld2")
        save_sidecar(sc1; zetas, evaluation = Dict("IG  +WIrls" => mk([3.0, 1.0, 5.0]), "IG  +Lift" => mk([2.0, 1.5, 4.0]),
                                            "IG  +GM-HQ" => mk([2.5, 1.2, 6.0]), "MSAC +LZ" => mk([9.0, 9.0, 9.0])))
        sc2 = joinpath(dir, "b.jld2")
        save_sidecar(sc2; zetas, evaluation = Dict("IG  +WIrls" => mk([1.0, 0.5, 2.0]), "IG  +Lift" => mk([0.8, 0.6, 1.0])))
        arms, rows, zs = miscal_summary_rows(style, sc1)
        @test arms == ["IG  +WIrls", "IG  +Lift", "IG  +GM-HQ"]   # welsch facet (ramp 1, 2) before gm; the rider is not a style arm
        @test rows["IG  +WIrls"] == (3.0, 1.0, 5.0)
        @test zs == (1 / 8, 1.0, 8.0)
        @test miscal_summary_rows(style, sc1; exclude = Set(["IG  +Lift"]))[1] == ["IG  +WIrls", "IG  +GM-HQ"]
        @test_throws ErrorException miscal_summary_rows(style, sc1; stat = :bad)
        out = emit_miscal_summary_table(style, joinpath(dir, "t.tex");
                                        tracks = [("Track", [("Alpha", sc1), ("Beta", sc2), ("Missing", joinpath(dir, "none.jld2"))])],
                                        exclude = Set(["IG  +GM-HQ"]), label = a -> "L(" * a * ")")
        t = read(out, String)
        @test occursin("\\multicolumn{3}{c}{Alpha}", t) && occursin("\\multicolumn{3}{c}{Beta}", t) && !occursin("Missing", t)
        @test occursin("\\tfrac{1}{8}", t) && occursin("\\zeta{=}1", t)
        lines = split(t, '\n')
        w = only(filter(l -> startswith(l, "L(IG  +WIrls)"), lines))
        l = only(filter(l -> startswith(l, "L(IG  +Lift)"), lines))
        # Alpha: calibration best = WIrls (1.0), deflation best = Lift (2.0), inflation best = Lift (4.0)
        @test occursin("\\textbf{1.00} & 3.00 & 5.00", w) && occursin("1.50 & \\textbf{2.00} & \\textbf{4.00}", l)
        # Beta: WIrls best at calibration (0.5); Lift best at both extremes
        @test occursin("\\textbf{0.50} & 1.00 & 2.00", w) && occursin("0.60 & \\textbf{0.80} & \\textbf{1.00}", l)
        @test !occursin("GM-HQ", t)
        # mAA statistic: best is the maximum, three decimals
        out2 = emit_miscal_summary_table(style, joinpath(dir, "m.tex"); tracks = [("T", [("Alpha", sc1)])], stat = :maa)
        @test occursin("\\textbf{", read(out2, String)) && occursin("mAA", read(out2, String))
    end
end
