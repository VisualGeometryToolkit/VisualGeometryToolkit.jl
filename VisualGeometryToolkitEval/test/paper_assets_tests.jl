# The regeneration's last stages (src/reproduce/paper_assets.jl): verify a
# regenerated asset against the submitted one, deploy it only when that
# changes nothing or with sign-off, gather the numbers file (under the same
# sign-off) and find the numbers the paper never cites; and run_isolated
# (Experiments), the in-process way to run a driver.

@testitem "paper_assets" begin
    using Test
    using VisualGeometryToolkitEval.Reproduce
    using VisualGeometryToolkitEval.Reproduce: VERIFY_REPORT
    using VisualGeometryToolkitEval.Experiments: run_isolated

    quietly(f) = redirect_stdout(f, devnull)
    refusal(f) = try
        quietly(f)
        nothing
    catch e
        e
    end

    @testset "compare_fragments" begin
        a = "\\addplot coordinates {(1.0,2.0) (3.0,4.5)};"
        @test compare_fragments(a, a).identical
        c = compare_fragments(replace(a, " (" => "  ("), a)
        @test !c.identical && c.same_numbers
        c = compare_fragments(replace(a, "4.5" => "4.25"), a)
        @test !c.same_numbers && c.max_abs_diff == 0.25
        c = compare_fragments(a * "\\addplot coordinates {(0,0)};", a)
        @test !c.same_numbers && c.blocks_new == 2 && c.blocks_old == 1 && c.max_abs_diff == Inf
        @test fragment_coordinates(a) == [[(1.0, 2.0), (3.0, 4.5)]]
        # a numbers file: its macros are its numbers
        n = "% header\n\\newcommand{\\nA}{1.5}\n"
        @test compare_fragments(replace(n, "header" => "other header"), n).same_numbers
        @test !compare_fragments(replace(n, "1.5" => "1.6"), n).same_numbers
    end

    @testset "verify and deploy: a submitted asset is replaced only with sign-off" begin
        outdir, paperdir = mktempdir(), mktempdir()
        write(joinpath(paperdir, "p.png"), UInt8[1, 2, 3])
        write(joinpath(paperdir, "f.tex"), "\\addplot coordinates {(1,2)};\n")

        # identical: verified identical, deployed as a no-op
        cp(joinpath(paperdir, "p.png"), joinpath(outdir, "p.png"))
        cp(joinpath(paperdir, "f.tex"), joinpath(outdir, "f.tex"))
        rows = quietly(() -> verify_assets(outdir, paperdir, ["p.png", "f.tex"]))
        @test all(r -> r.submitted && r.identical && r.same_numbers, rows)
        report = joinpath(outdir, VERIFY_REPORT)
        @test split(readlines(report)[1], '\t')[1:4] == ["asset", "submitted", "identical", "same_numbers"]
        @test split(readlines(report)[2], '\t')[1:3] == ["p.png", "true", "true"]
        @test isempty(quietly(() -> deploy_assets(outdir, paperdir, ["p.png", "f.tex"])))
        @test assets_deployed(outdir, paperdir, ["p.png", "f.tex"])

        # differing: refused without sign-off, nothing copied, the report rewritten by the deploy itself
        write(joinpath(outdir, "p.png"), UInt8[1, 2, 3, 4])
        write(joinpath(outdir, "f.tex"), "\\addplot coordinates {(1,2.5)};\n")
        rm(report)
        err = refusal(() -> deploy_assets(outdir, paperdir, ["p.png", "f.tex"]))
        @test err isa ErrorException && occursin(report, err.msg) && occursin("replace_submitted", err.msg)
        @test occursin("p.png", err.msg) && occursin("max |Δ| = 0.5", err.msg)
        @test isfile(report)                                                   # verified before refusing
        @test split(readlines(report)[3], '\t')[1:4] == ["f.tex", "true", "false", "false"]
        @test read(joinpath(paperdir, "p.png")) == UInt8[1, 2, 3]
        @test !assets_deployed(outdir, paperdir, ["p.png"])
        @test quietly(() -> deploy_assets(outdir, paperdir, ["p.png", "f.tex"]; replace_submitted = true)) == ["p.png", "f.tex"]
        @test assets_deployed(outdir, paperdir, ["p.png", "f.tex"])

        # per-asset sign-off: one call, one report with every row; only the named asset may replace its submitted copy
        write(joinpath(outdir, "p.png"), UInt8[5])
        write(joinpath(outdir, "f.tex"), "\\addplot coordinates {(1,3)};\n")
        err = refusal(() -> deploy_assets(outdir, paperdir, ["p.png", "f.tex"]; replace_submitted = ["p.png"]))
        @test err isa ErrorException && occursin("f.tex", err.msg) && !occursin("p.png:", err.msg)
        @test length(readlines(report)) == 3                                   # the header and both rows
        @test read(joinpath(paperdir, "p.png")) != UInt8[5]                    # refused: nothing copied
        @test quietly(() -> deploy_assets(outdir, paperdir, ["p.png", "f.tex"]; replace_submitted = Set(["p.png", "f.tex"]))) ==
              ["p.png", "f.tex"]

        # formatting only: same numbers, still not the submitted bytes, so still signed off
        write(joinpath(outdir, "f.tex"), "\\addplot coordinates {(1, 3)};\n")
        r = only(quietly(() -> verify_assets(outdir, paperdir, ["f.tex"])))
        @test !r.identical && r.same_numbers
        @test refusal(() -> deploy_assets(outdir, paperdir, ["f.tex"])) isa ErrorException

        # new: no submitted copy, deployed without sign-off; a missing render is an error
        write(joinpath(outdir, "new.png"), UInt8[9])
        @test !only(quietly(() -> verify_assets(outdir, paperdir, ["new.png"]))).submitted
        @test quietly(() -> deploy_assets(outdir, paperdir, ["new.png"])) == ["new.png"]
        @test_throws ErrorException quietly(() -> verify_assets(outdir, paperdir, ["absent.png"]))
    end

    @testset "gather_numbers: one \\newcommand per number, deployed under sign-off" begin
        dir = mktempdir()
        a, b = joinpath(dir, "run-a", "numbers.tex"), joinpath(dir, "run-b", "numbers.tex")
        mkpath.(dirname.((a, b)))
        out = joinpath(dir, "numbers-k", "numbers.tex")
        write(a, "% fragment\n\\newcommand{\\nightStarsZ}{1.5}\n\\newcommand{\\nightStarsA}{2}\n")
        write(b, "\\newcommand{\\detectorBiasM}{0.031}\n")
        @test gather_numbers([a, b], out; generator = "scripts/numbers.jl", root = dir) == out
        lines = readlines(out)
        @test lines[1] == "% generated by scripts/numbers.jl — do not edit; gathered from:"
        @test lines[2:3] == ["%   <results_root()>/run-a/numbers.tex", "%   <results_root()>/run-b/numbers.tex"]
        @test lines[4:end] == ["\\newcommand{\\detectorBiasM}{0.031}", "\\newcommand{\\nightStarsA}{2}",
                               "\\newcommand{\\nightStarsZ}{1.5}"]
        @test !occursin(dir, read(out, String))                                # no machine path
        write(b, "\\newcommand{\\nightStarsA}{3}\n")
        @test_throws ArgumentError gather_numbers([a, b], out; generator = "g", root = dir)
        write(b, "\\newcommand{\\detectorBiasM}[1]{#1 px}\n")
        @test_throws ArgumentError gather_numbers([a, b], out; generator = "g", root = dir)
        write(b, "\\def\\detectorBiasM#1{#1}\n")
        @test_throws ArgumentError gather_numbers([a, b], out; generator = "g", root = dir)

        # the numbers file is a submitted asset like a figure: a differing one is replaced only with sign-off
        latex = mkpath(joinpath(dir, "paper", "latex"))
        write(joinpath(latex, "numbers.tex"), "% generated — do not edit\n")
        gather_numbers([a], out; generator = "g", root = dir)
        @test refusal(() -> deploy_assets(dirname(out), latex, ["numbers.tex"])) isa ErrorException
        @test quietly(() -> deploy_assets(dirname(out), latex, ["numbers.tex"]; replace_submitted = true)) == ["numbers.tex"]
        write(a, "\\newcommand{\\nightStarsZ}{1.75}\n\\newcommand{\\nightStarsA}{2}\n")
        gather_numbers([a], out; generator = "g", root = dir)
        @test !only(quietly(() -> verify_assets(dirname(out), latex, ["numbers.tex"]))).same_numbers
        @test refusal(() -> deploy_assets(dirname(out), latex, ["numbers.tex"])) isa ErrorException
        @test occursin("1.5", read(joinpath(latex, "numbers.tex"), String))   # the submitted numbers untouched
    end

    @testset "uncited_macros: the files the paper reads, not every file" begin
        root = mktempdir()
        for (p, s) in (
                "paper/src/main.tex" => "\\input{src/latex/numbers}\n\\input{src/sec/x}\n% \\nB\n",
                "paper/src/latex/numbers.tex" => "\\newcommand{\\nA}{1}\n\\newcommand{\\nB}{2}\n\\newcommand{\\nC}{3}\n\\newcommand{\\nD}{4}\n",
                "paper/src/sec/x.tex" => "error \\nA~px, % \\nB\n\\nCd and \\nD{} more\n",
                "paper/src/sec/unread.tex" => "\\nB \\nC\n")   # under sec/, never \input
            mkpath(dirname(joinpath(root, p)))
            write(joinpath(root, p), s)
        end
        numbers, papers = joinpath(root, "paper/src/latex/numbers.tex"), [joinpath(root, "paper/src/main.tex")]
        kw = (; root, search_dirs = ["paper"])
        @test uncited_macros(numbers, papers; kw...) == ["nB", "nC"]          # a comment, a prefix, an unread file: uncited
        @test uncited_macros(numbers, papers; kw..., allow = Dict("nB" => "r", "nC" => "r")) == String[]
        @test uncited_macros(numbers, papers; kw..., allow = Dict("gone" => "r")) == ["nB", "nC", "stale allow: gone"]
        @test_throws ArgumentError uncited_macros(numbers, papers; root)       # without the search path the paper reads nothing
    end

    @testset "run_isolated: a script in a module of its own" begin
        dir = mktempdir()
        s = joinpath(dir, "s.jl")
        write(s, "const SEEN = copy(ARGS)\nconst E = get(ENV, \"VGTK_ISOLATED_PROBE\", \"\")\nf() = 41 + length(SEEN)\ng() = H\n")
        saved = copy(ARGS)
        m1 = run_isolated(s; args = ["--plan"], env = Dict("VGTK_ISOLATED_PROBE" => "on"), bindings = Dict(:H => 7))
        m2 = run_isolated(s; bindings = Dict(:H => 8))                       # the same constants again: no clash
        get1(m, k) = Base.invokelatest(getfield, m, k)
        @test get1(m1, :SEEN) == ["--plan"] && get1(m2, :SEEN) == String[]
        @test get1(m1, :E) == "on" && get1(m2, :E) == ""
        @test !haskey(ENV, "VGTK_ISOLATED_PROBE")
        @test Base.invokelatest(get1(m1, :f)) == 42
        @test Base.invokelatest(get1(m1, :g)) == 7 && Base.invokelatest(get1(m2, :g)) == 8
        @test ARGS == saved
        write(s, "exit(0)\n")
        @test_throws LoadError run_isolated(s; args = ["x"])           # the script's exit throws
        @test ARGS == saved
    end
end
