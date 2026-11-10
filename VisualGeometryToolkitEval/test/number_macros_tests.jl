# The one writer of a paper's number macros (src/reproduce/number_macros.jl):
# numbers typeset exactly as configured, macro names LaTeX accepts, one macro
# per table cell, a provenance comment per macro, deterministic bytes, and the
# round trip through tex_macro_values and gather_numbers; a table row compiles
# in LaTeX when pdflatex is installed.

@testitem "number_macros" begin
    using Test
    using Printf: Printf
    using VisualGeometryToolkitEval.Reproduce

    prov = (; driver = "scripts/pose_table.jl", config = "configs/pose_table.toml", run_key = "3f2a9c1e", commit = "f11598f7")

    @testset "tex_number: the number exactly as configured" begin
        # fixed decimals are Printf's %.<d>f of the same value (exact binary value, ties to even)
        for x in (0.125, 0.375, 2.5, 1.005, 0.045, 1 / 3, 123456.785, 7.0, 1e20, 42), d in 0:4
            @test tex_number(x; digits = d) == Printf.format(Printf.Format("%.$(d)f"), x)
        end
        @test tex_number(-2.675; digits = 2) == Printf.@sprintf("%.2f", -2.675)
        @test tex_number(-0.004; digits = 2) == "0.00"                  # a zero carries no sign
        # significant digits: trailing zeros dropped, never an exponent
        @test tex_number(0.5; sigdigits = 3) == "0.5"
        @test tex_number(3.14159; sigdigits = 3) == "3.14"
        @test tex_number(99.96; sigdigits = 3) == "100"
        @test tex_number(12345.0; sigdigits = 2) == "12000"
        @test tex_number(6.598534917851996e-5; sigdigits = 2) == "0.000066"
        @test tex_number(-0.0001234; sigdigits = 2) == "-0.00012"
        # integers need no precision; a non-integer does
        @test tex_number(17) == "17" && tex_number(12.0) == "12"
        @test_throws ArgumentError tex_number(1.5)
        @test_throws ArgumentError tex_number(1.5; digits = 1, sigdigits = 2)
        @test_throws ArgumentError tex_number(NaN; digits = 2)
        @test_throws ArgumentError tex_number(Inf)
        # percent, groups, units, ranges
        @test tex_number(0.256; percent = true, digits = 0) == "26\\%"
        @test tex_number(0.2563; percent = true, digits = 1) == "25.6\\%"
        @test tex_number(1234567; group_from = 4) == "1\\,234\\,567"
        @test tex_number(1234; group_from = 4) == "1\\,234" && tex_number(123; group_from = 4) == "123"
        @test tex_number(9999; group_from = 5) == "9999" && tex_number(10000; group_from = 5) == "10\\,000"
        @test tex_number(-1234.5; digits = 1, group_from = 4) == "-1\\,234.5"
        @test tex_number(1.5; digits = 1, unit = "mm") == "1.5\\,mm"
        @test tex_number(0.1, 0.25; digits = 2) == "0.10--0.25"
        @test tex_number(0.1, 0.25; digits = 2, range_separator = "\\text{--}", unit = "px") == "0.10\\text{--}0.25\\,px"
        @test_throws ArgumentError tex_number(1.5; digits = 1, unit = "50%")
    end

    @testset "number_macro_name: LaTeX-legal names, malformed ones refused" begin
        @test number_macro_name("chromeTable", "kb_start", "median") == "chromeTableKbStartMedian"
        @test number_macro_name("pose", "fx3_14mm") == "poseFxThreeOneFourMm"
        @test number_macro_name("bias", :refined, "N 12", "chi-sq.p90") == "biasRefinedNOneTwoChiSqPNineZero"
        @test number_macro_name("x", 3) == "xThree"
        @test_throws ArgumentError number_macro_name("pose2", "a")      # a prefix is letters only
        @test_throws ArgumentError number_macro_name("", "a")
        @test_throws ArgumentError number_macro_name("pose", "σ_rot")   # not a letter LaTeX names a macro by
        @test_throws ArgumentError number_macro_name("pose", "5%")
        @test_throws ArgumentError number_macro_name("end", "point")    # \endpoint: LaTeX refuses \end…
        nums = PaperNumbers(; prov...)
        for bad in ("pose2", "pose_a", "\\pose", "", "endTable", "posé")
            @test_throws ArgumentError add_number!(nums, bad, 1)
        end
        @test length(nums) == 0
    end

    @testset "add_number! and add_table_row!: one macro per number" begin
        # the driver's calls, as data, so the determinism check can replay them in another order
        scalars = [("poseFrames", 240, (;)), ("poseRotMed", 0.0412, (; digits = 3)),
            ("poseShare", 0.5, (; percent = true, digits = 0)), ("poseRange", (0.1, 0.25), (; digits = 2)),
            ("poseRtol", "10^{-6}", (;)),                                  # a typeset body, verbatim
            ("teaserBoards", 12, (; label = "fig:teaser")),                # a value a figure reads
            ("otherRunMed", 1.25, (; digits = 2, provenance = (; run_key = "77aa00bb")))]
        rows = [("heBaxter", ["frames" => 61, "recovered" => 58, "transMm" => tex_number(1.234; digits = 2),
                              "rotDeg" => tex_number(0.0456; digits = 3)], (; label = "tab:handeye")),
                ("heUr", (:frames => 12.5, :recovered => 12.0), (; digits = 1))]
        nums = PaperNumbers(; prov...)
        @test add_number!(nums, scalars[1][1], scalars[1][2]) === nums
        foreach(((n, v, kw),) -> add_number!(nums, n, v; kw...), scalars[2:end])
        @test_throws ArgumentError add_number!(nums, "poseFrames", 241)  # a name is one number
        @test_throws ArgumentError add_number!(nums, "poseX", "10^{-6}"; digits = 2)
        for body in ("1 & 2", "50%", "#1", "{1", "1}", "a\\\\b", "1\n2", "", "x\\", "\$1")
            @test_throws ArgumentError add_number!(nums, "poseBad", body)   # incl. a trailing backslash, an odd \$
        end
        @test_throws ArgumentError add_number!(nums, "poseBad", true)    # a Bool is not a number the paper prints
        @test_throws ArgumentError add_number!(nums, "poseBad", 1; provenance = (; host = "x"))
        # a `;` would make the provenance comment ambiguous
        @test_throws ArgumentError add_number!(nums, "poseBad", 1; provenance = (; run_key = "a;b"))
        @test_throws ArgumentError add_number!(nums, "poseBad", 1; label = "fig:a;b")
        @test_throws ArgumentError PaperNumbers(; prov..., config = "configs/a;b.toml")
        @test_throws ArgumentError PaperNumbers(; prov..., config = "/home/me/configs/x.toml")   # no machine path
        @test_throws ArgumentError PaperNumbers(; prov..., commit = "")
        row = add_table_row!(nums, rows[1][1], rows[1][2]; rows[1][3]...)
        @test row == "\\heBaxterFrames & \\heBaxterRecovered & \\heBaxterTransMm & \\heBaxterRotDeg"
        # a row is added whole or not at all
        n = length(nums)
        @test_throws ArgumentError add_table_row!(nums, "heUr", ["frames" => 3, "bad col" => 4])
        @test_throws ArgumentError add_table_row!(nums, "heUr", ["frames" => 3, "frames" => 4])
        @test_throws ArgumentError add_table_row!(nums, "heBaxter", ["frames" => 3])
        @test_throws ArgumentError add_table_row!(nums, "heUr", ["frames" => NaN]; digits = 1)
        # a Dict has no column order: refused, with the ordered collection to give instead
        err = try add_table_row!(nums, "heUr", Dict("frames" => 3, "recovered" => 4)); nothing catch e; e end
        @test err isa ArgumentError && occursin("ordered", err.msg)
        @test length(nums) == n
        @test add_table_row!(nums, rows[2][1], rows[2][2]; rows[2][3]...) == "\\heUrFrames & \\heUrRecovered"

        dir = mktempdir()
        frag = write_numbers(joinpath(dir, "run-a", "numbers.tex"), nums)
        lines = readlines(frag)
        @test lines[1] == "% generated by scripts/pose_table.jl — do not edit; gathered by gather_numbers"
        defs = filter(startswith("\\newcommand"), lines)
        names = [match(r"^\\newcommand\{\\([A-Za-z]+)\}", l).captures[1] for l in defs]
        @test names == sort(names) && length(names) == length(nums)    # sorted by name
        # a provenance comment before every macro
        for (i, l) in enumerate(lines)
            startswith(l, "\\newcommand") || continue
            name = match(r"^\\newcommand\{\\([A-Za-z]+)\}", l).captures[1]
            @test startswith(lines[i - 1], "% provenance \\$name: driver=scripts/pose_table.jl; config=configs/pose_table.toml; run=")
        end
        text = read(frag, String)
        @test occursin("% provenance \\otherRunMed: driver=scripts/pose_table.jl; config=configs/pose_table.toml; run=77aa00bb; commit=f11598f7\n", text)
        @test occursin("% provenance \\teaserBoards: driver=scripts/pose_table.jl; config=configs/pose_table.toml; run=3f2a9c1e; commit=f11598f7; label=fig:teaser\n", text)
        @test occursin("; label=tab:handeye\n\\newcommand{\\heBaxterRotDeg}{0.046}\n", text)

        # round trip: what was added is what tex_macro_values reads
        expected = Dict("poseFrames" => "240", "poseRotMed" => "0.041", "poseShare" => "50\\%", "poseRange" => "0.10--0.25",
            "poseRtol" => "10^{-6}", "teaserBoards" => "12", "otherRunMed" => "1.25", "heBaxterFrames" => "61",
            "heBaxterRecovered" => "58", "heBaxterTransMm" => "1.23", "heBaxterRotDeg" => "0.046",
            "heUrFrames" => "12.5", "heUrRecovered" => "12.0")
        @test Dict(tex_macro_values(frag)) == expected
        @test first.(tex_macro_values(frag)) == sort(collect(keys(expected)))
        # deterministic: the same calls in another order write the same bytes
        again = PaperNumbers(; prov...)
        for (r, cells, kw) in reverse(rows)
            add_table_row!(again, r, cells; kw...)
        end
        foreach(((n, v, kw),) -> add_number!(again, n, v; kw...), reverse(scalars))
        @test read(write_numbers(joinpath(dir, "again.tex"), again), String) == text
        # the write is whole or nothing: no temp file left beside the fragment
        @test readdir(dirname(frag)) == ["numbers.tex"]

        # gather_numbers keeps every macro and its provenance comment
        other = write_numbers(joinpath(dir, "run-b", "numbers.tex"),
            add_number!(PaperNumbers(; prov..., driver = "scripts/bias.jl"), "biasMedian", 0.031; sigdigits = 2))
        legacy = joinpath(dir, "run-c", "numbers.tex")
        mkpath(dirname(legacy))
        write(legacy, "\\newcommand{\\legacyCount}{7}\n")                   # a fragment of no provenance gathers as before
        out = gather_numbers([frag, other, legacy], joinpath(dir, "numbers-k", "numbers.tex");
            generator = "scripts/numbers.jl", root = dir)
        gathered = read(out, String)
        @test Dict(tex_macro_values(out)) == merge(expected, Dict("biasMedian" => "0.031", "legacyCount" => "7"))
        @test occursin("% provenance \\biasMedian: driver=scripts/bias.jl;", gathered)
        @test occursin("; label=fig:teaser\n\\newcommand{\\teaserBoards}{12}\n", gathered)
        @test occursin("% provenance \\heUrRecovered:", gathered) && !occursin("provenance \\legacyCount", gathered)
        @test count("% provenance", gathered) == length(expected) + 1
        @test compare_fragments(gathered, gathered).same_numbers
        # the literal check sees each cell's number (one macro per number)
        prose = joinpath(dir, "prose.tex")
        write(prose, "The Baxter arm recovered 58 of its frames, at 1.23 mm.\n")
        hits = macro_literal_violations([prose], [out]; root = dir, min_integer_digits = 2)
        @test Set(v.subject for v in hits) == Set(["58", "1.23"])
    end

    @testset "deploy: a provenance-only change deploys, a changed value needs sign-off" begin
        quietly(f) = redirect_stdout(f, devnull)
        dir = mktempdir()
        paper, results = mkpath(joinpath(dir, "paper")), mkpath(joinpath(dir, "results"))
        frag = joinpath(dir, "run", "numbers.tex")
        function regenerate(; run_key, commit, value)
            write_numbers(frag, add_number!(PaperNumbers(; prov..., run_key, commit), "poseRotMed", value; digits = 3))
            gather_numbers([frag], joinpath(results, "numbers.tex"); generator = "scripts/numbers.jl", root = dir)
        end
        regenerate(; run_key = "aaaa1111", commit = "f11598f7", value = 0.0412)
        @test quietly(() -> deploy_assets(results, paper, ["numbers.tex"])) == ["numbers.tex"]     # new: no sign-off
        # a rerun at a new run key and commit, the same numbers: deployed without sign-off, its comments updated
        regenerate(; run_key = "bbbb2222", commit = "0123abcd", value = 0.0412)
        @test quietly(() -> deploy_assets(results, paper, ["numbers.tex"])) == ["numbers.tex"]
        deployed = read(joinpath(paper, "numbers.tex"), String)
        @test occursin("run=bbbb2222; commit=0123abcd", deployed) && !occursin("aaaa1111", deployed)
        # a changed value: refused without sign-off, the submitted file untouched
        regenerate(; run_key = "cccc3333", commit = "0123abcd", value = 0.0452)
        @test_throws ErrorException quietly(() -> deploy_assets(results, paper, ["numbers.tex"]))
        @test read(joinpath(paper, "numbers.tex"), String) == deployed
        @test quietly(() -> deploy_assets(results, paper, ["numbers.tex"]; replace_submitted = true)) == ["numbers.tex"]
        @test occursin("{0.045}", read(joinpath(paper, "numbers.tex"), String))
    end

    @testset "a table row of number macros compiles in LaTeX" begin
        nums = PaperNumbers(; prov...)
        rows = [add_table_row!(nums, "tab" * name, ["frames" => f, "resid" => r, "share" => s]; digits = 2)
                for (name, f, r, s) in (("Baxter", 61.0, 1.2345, 0.5), ("Ur", 12.0, 0.25, 0.125))]
        dir = mktempdir()
        write_numbers(joinpath(dir, "numbers.tex"), nums)
        body = join((r * " \\\\" for r in rows), "\n")
        @test body == "\\tabBaxterFrames & \\tabBaxterResid & \\tabBaxterShare \\\\\n\\tabUrFrames & \\tabUrResid & \\tabUrShare \\\\"
        values = Dict(tex_macro_values(joinpath(dir, "numbers.tex")))
        @test [values["tabBaxter" * c] for c in ("Frames", "Resid", "Share")] == ["61.00", "1.23", "0.50"]
        @test [values["tabUr" * c] for c in ("Frames", "Resid", "Share")] == ["12.00", "0.25", "0.12"]
        if Sys.which("pdflatex") === nothing
            @info "pdflatex not found: the table row is checked as text only"
        else
            write(joinpath(dir, "doc.tex"), """
                \\documentclass{article}
                \\input{numbers}
                \\begin{document}
                \\begin{tabular}{lrrr}
                system & frames & resid & share \\\\
                Baxter & $(rows[1]) \\\\
                UR & $(rows[2]) \\\\
                \\end{tabular}
                \\typeout{CELL=\\tabUrResid}
                \\end{document}
                """)
            log = joinpath(dir, "latex.out")
            ok = success(pipeline(Cmd(`pdflatex -interaction=nonstopmode -halt-on-error doc.tex`; dir); stdout = log, stderr = log))
            ok || @info "pdflatex failed" log = read(log, String)
            @test ok && isfile(joinpath(dir, "doc.pdf"))
            @test occursin("CELL=0.25", read(log, String))
        end
    end
end
