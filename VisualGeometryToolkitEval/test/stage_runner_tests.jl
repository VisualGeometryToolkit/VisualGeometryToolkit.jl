# stage_runner_tests.jl — run_stages re-evaluates each stage's done() at
# EXECUTION time. The printed table is a snapshot taken before anything runs,
# so a stage whose input an earlier stage of the same run (re)writes must not
# be skipped on the strength of that snapshot.

@testitem "stage_runner" begin
    using Test
    using VisualGeometryToolkitEval.Experiments: Stage, run_stages

    @testset "run_stages: done() is re-evaluated at execution time" begin
        mktempdir() do d
            produced = joinpath(d, "input.txt")
            write(produced, "old")          # exists at plan time
            consumed = joinpath(d, "table.txt")
            write(consumed, "stale")        # newer than nothing; done() below compares mtimes
            ran = Symbol[]
            stages = [
                Stage(:produce, "rewrite the input", () -> false,
                      () -> (push!(ran, :produce); sleep(0.01); write(produced, "new"))),
                # DONE at plan time (table newer than input), pending once :produce runs
                Stage(:consume, "rebuild the table from the input",
                      () -> isfile(consumed) && mtime(consumed) >= mtime(produced),
                      () -> (push!(ran, :consume); write(consumed, "fresh"))),
            ]
            run_stages("t", stages; argv = String[])
            @test ran == [:produce, :consume]
            @test read(consumed, String) == "fresh"
        end

        # a genuinely done stage is still skipped, and --force runs everything
        mktempdir() do d
            f = joinpath(d, "out.txt"); write(f, "x")
            ran = Symbol[]
            stages = [Stage(:a, "done already", () -> isfile(f), () -> push!(ran, :a))]
            run_stages("t", stages; argv = String[])
            @test isempty(ran)
            run_stages("t", stages; argv = ["--force"])
            @test ran == [:a]
            empty!(ran)
            run_stages("t", stages; argv = ["--plan"])
            @test isempty(ran)
        end
    end
end
