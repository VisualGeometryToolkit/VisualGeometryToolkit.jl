# The reproducible-figures contract (src/reproduce/reproduce_index.jl).

@testitem "reproduce_index" begin
    using Test
    using VisualGeometryToolkitEval.Reproduce

    # A miniature paper project: one paper, a macro file, a figure, a table
    # input, a pgfplots data file, a static diagram, and the generators.
    function fixture(; index, prose = "We report 0.68 mm on \\bbCaptures captures.")
        root = mktempdir()
        files = Dict(
            "papers/main.tex" => """
                \\documentclass{article}
                \\graphicspath{{images/}}
                \\input{figures/stats.tex}
                % \\includegraphics{images/commented_out.png}
                \\begin{document}
                $prose
                \\begin{figure}
                  \\includegraphics[width=\\linewidth]{panel}
                  \\includegraphics{images/diagram.pdf}
                  \\caption{x}\\label{fig:panel}
                \\end{figure}
                \\begin{table}
                  \\input{figures/table.tex}
                  \\label{tab:t}
                \\end{table}
                \\end{document}
                """,
            "papers/figures/stats.tex" => "\\newcommand{\\bbResid}{0.68}\n\\newcommand{\\bbCaptures}{56}\n\\newcommand{\\bbOne}{3}\n",
            "papers/figures/table.tex" => "\\begin{tikzpicture}\\begin{axis}\\addplot table {figures/curve.dat};\\end{axis}\\end{tikzpicture}\n",
            "papers/figures/curve.dat" => "x y\n1 2\n",
            "papers/images/panel.png" => "",
            "papers/images/diagram.pdf" => "",
            "scripts/make_stats.jl" => "# reproduce: writes papers/figures/*.tex\nwrite(\"x\")\n",
            "scripts/make_panel.jl" => "save(joinpath(dir, \"panel.png\"), fig)\n",
            "scripts/make_curve.jl" => "open(\"curve.dat\", \"w\")\n",
            "REPRODUCE.md" => index,
        )
        for (p, s) in files
            mkpath(dirname(joinpath(root, p)))
            write(joinpath(root, p), s)
        end
        return root
    end

    good_index = """
        # Reproduce

        ## Paper asset index

        | label | asset | kind | generator | data | provenance |
        |---|---|---|---|---|---|
        | macros | `papers/figures/stats.tex` | generated | `scripts/make_stats.jl` | results | |
        | `fig:panel` | `papers/images/panel.png` | generated | `scripts/make_panel.jl` | images tier | |
        | `fig:panel` | `papers/images/diagram.pdf` | static | — | — | drawn by hand in Inkscape, 2026 |
        | `tab:t` | `papers/figures/table.tex` | generated | `scripts/make_stats.jl` | results | |
        | `tab:t` | `papers/figures/curve.dat` | generated | `scripts/make_curve.jl` | results | |
        """

    @testset "paper_asset_graph" begin
        root = fixture(; index = good_index)
        g = paper_asset_graph([joinpath(root, "papers", "main.tex")]; root)
        paths = Set(a.path for a in g.assets)
        @test paths == Set(["papers/figures/stats.tex", "papers/figures/table.tex", "papers/images/panel.png",
            "papers/images/diagram.pdf", "papers/figures/curve.dat"])
        @test all(a -> a.exists, g.assets)
        panel = only(filter(a -> a.path == "papers/images/panel.png", g.assets))
        @test panel.kind === :graphics && panel.labels == ["fig:panel"]
        curve = only(filter(a -> a.path == "papers/figures/curve.dat", g.assets))
        @test curve.kind === :data && curve.labels == ["tab:t"]      # inherited through \input
        @test "fig:panel" in g.labels && "tab:t" in g.labels
        @test isempty(g.unresolved)
        @test "papers/figures/table.tex" in g.tex_files
    end

    @testset "read_reproduce_index" begin
        root = fixture(; index = good_index)
        es = read_reproduce_index(joinpath(root, "REPRODUCE.md"))
        @test length(es) == 5
        @test es[2].label == "fig:panel" && es[2].kind === :generated && es[2].generators == ["scripts/make_panel.jl"]
        @test es[3].kind === :static && occursin("Inkscape", es[3].provenance)
        @test declared_outputs(joinpath(root, "scripts", "make_stats.jl")) == ["papers/figures/*.tex"]
        @test tex_macro_values(joinpath(root, "papers", "figures", "stats.tex")) ==
            ["bbResid" => "0.68", "bbCaptures" => "56", "bbOne" => "3"]
    end

    kinds(vs) = sort!(unique(getfield.(vs, :kind)))

    @testset "a compliant project" begin
        root = fixture(; index = good_index, prose = "We report \\bbResid~mm on \\bbCaptures captures, 3 times.")
        @test isempty(reproduce_index_violations(root; papers = ["papers/main.tex"]))
    end

    @testset "macro literals in prose" begin
        root = fixture(; index = good_index)   # default prose re-types 0.68 (\bbResid); not the lone digit 3
        vs = reproduce_index_violations(root; papers = ["papers/main.tex"])
        @test kinds(vs) == [:macro_literal]
        @test only(vs).subject == "0.68" && occursin("bbResid", only(vs).message)
        @test isempty(reproduce_index_violations(root; papers = ["papers/main.tex"], check_literals = false))
        @test isempty(reproduce_index_violations(root; papers = ["papers/main.tex"], ignore_macros = ["bbResid"]))
        root2 = fixture(; index = good_index, prose = "We report \$0.68\$\\,mm. % reproduce: literal-ok")
        @test isempty(reproduce_index_violations(root2; papers = ["papers/main.tex"]))
        root3 = fixture(; index = good_index, prose = "A \$56\$~captures run, 10.68 and 0.685 do not match.")
        @test isempty(reproduce_index_violations(root3; papers = ["papers/main.tex"]))   # two digits: default skips
        @test only(reproduce_index_violations(root3; papers = ["papers/main.tex"], min_integer_digits = 2)).subject == "56"
        root4 = fixture(; index = good_index, prose = "\\begin{minipage}[t]{0.68\\linewidth}\\end{minipage}")
        @test isempty(reproduce_index_violations(root4; papers = ["papers/main.tex"]))  # a length, not prose
    end

    @testset "each violation kind" begin
        ok = "\\bbResid"
        # unindexed + unreferenced + missing generator + static without provenance + invalid kind
        bad = """
            | label | asset | kind | generator | data | provenance |
            |---|---|---|---|---|---|
            | macros | `papers/figures/stats.tex` | generated | `scripts/make_stats.jl` | | |
            | `fig:panel` | `papers/images/panel.png` | generated | `scripts/gone.jl` | | |
            | `fig:panel` | `papers/images/diagram.pdf` | static | — | — | |
            | `tab:t` | `papers/figures/table.tex` | generaated | `scripts/make_stats.jl` | | |
            | `tab:t` | `papers/images/old.png` | generated | `scripts/make_panel.jl` | | |
            """
        vs = reproduce_index_violations(fixture(; index = bad, prose = ok); papers = ["papers/main.tex"])
        @test kinds(vs) == [:generator_does_not_write, :invalid_kind, :missing_generator, :static_without_provenance,
            :unindexed_asset, :unreferenced_entry]
        @test only(filter(v -> v.kind === :unindexed_asset, vs)).subject == "papers/figures/curve.dat"
        @test only(filter(v -> v.kind === :unreferenced_entry, vs)).subject == "papers/images/old.png"
        @test only(filter(v -> v.kind === :generator_does_not_write, vs)).subject == "papers/images/old.png"

        # a generator that neither declares nor names its asset; no generator; label mismatch; glob rows
        bad2 = """
            | label | asset | kind | generator | data | provenance |
            |---|---|---|---|---|---|
            | macros | `papers/figures/*.tex` | generated | `scripts/make_stats.jl` | | |
            | `tab:t` | `papers/images/panel.png` | generated | `scripts/make_curve.jl` | | |
            | `fig:panel` | `papers/images/diagram.pdf` | static | — | — | hand-made |
            | `tab:t` | `papers/figures/curve.dat` | generated | — | | |
            """
        vs = reproduce_index_violations(fixture(; index = bad2, prose = ok); papers = ["papers/main.tex"])
        @test kinds(vs) == [:generator_does_not_write, :label_mismatch, :no_generator]
        @test only(filter(v -> v.kind === :label_mismatch, vs)).subject == "papers/images/panel.png"

        # missing asset, unresolved reference, missing index
        root = fixture(; index = good_index, prose = "\\includegraphics{images/nothere.png} \\includegraphics{\\figdir/x}")
        vs = reproduce_index_violations(root; papers = ["papers/main.tex"])
        @test kinds(vs) == [:missing_asset, :unindexed_asset, :unresolved_reference]
        @test kinds(reproduce_index_violations(root; papers = ["papers/main.tex"], index = "NOPE.md")) == [:missing_index]
        @test sprint(show, first(vs)) isa String
    end

    @testset "config column" begin
        ok = "\\bbResid"
        cfgidx(cfg) = """
            | label | asset | kind | generator | config | data | provenance |
            |---|---|---|---|---|---|---|
            | macros | `papers/figures/*.tex` | generated | `scripts/make_stats.jl` | $cfg | | |
            | `fig:panel` | `papers/images/panel.png` | generated | `scripts/make_panel.jl` | `configs/panel.toml` | | |
            | `fig:panel` | `papers/images/diagram.pdf` | static | — | | — | hand-made |
            | `tab:t` | `papers/figures/curve.dat` | generated | `scripts/make_curve.jl` | `configs/curve.toml` | | |
            """
        root = fixture(; index = cfgidx("`configs/stats.toml`"), prose = ok)
        write(joinpath(root, "scripts", "make_stats.jl"), "# reproduce: writes papers/figures/*.tex\n# reproduce: reads configs/stats.toml\n")
        write(joinpath(root, "scripts", "make_panel.jl"), "cfg = TOML.parsefile(\"configs/panel.toml\"); save(\"panel.png\", fig)\n")
        mkpath(joinpath(root, "configs"))
        foreach(f -> write(joinpath(root, "configs", f), "[experiment]\n"), ("stats.toml", "panel.toml", "curve.toml"))
        es = read_reproduce_index(joinpath(root, "REPRODUCE.md"))
        @test es[1].configs == ["configs/stats.toml"] && es[3].configs == String[]
        @test declared_inputs(joinpath(root, "scripts", "make_stats.jl")) == ["configs/stats.toml"]
        # make_curve.jl neither declares nor names its config
        vs = reproduce_index_violations(root; papers = ["papers/main.tex"])
        @test kinds(vs) == [:generator_does_not_read_config]
        @test only(vs).subject == "configs/curve.toml"
        write(joinpath(root, "scripts", "make_curve.jl"), "# reproduce: reads configs/curve.toml\nopen(\"curve.dat\", \"w\")\n")
        @test isempty(reproduce_index_violations(root; papers = ["papers/main.tex"], require_config = true))
        rm(joinpath(root, "configs", "panel.toml"))
        @test kinds(reproduce_index_violations(root; papers = ["papers/main.tex"])) == [:missing_config]
        # a generated row without a config: allowed by default, refused when required
        root2 = fixture(; index = cfgidx(""), prose = ok)
        mkpath(joinpath(root2, "configs"))
        foreach(f -> write(joinpath(root2, "configs", f), "[experiment]\n"), ("panel.toml", "curve.toml"))
        write(joinpath(root2, "scripts", "make_panel.jl"), "# reproduce: reads configs/panel.toml\nsave(\"panel.png\", fig)\n")
        write(joinpath(root2, "scripts", "make_curve.jl"), "# reproduce: reads configs/curve.toml\nopen(\"curve.dat\", \"w\")\n")
        @test isempty(reproduce_index_violations(root2; papers = ["papers/main.tex"]))
        vs = reproduce_index_violations(root2; papers = ["papers/main.tex"], require_config = true)
        @test kinds(vs) == [:no_config] && only(vs).subject == "papers/figures/*.tex"
    end
end
