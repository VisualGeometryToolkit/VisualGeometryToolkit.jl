# stage_runner_tests.jl — run_stages over stages that declare their keyed steps: every step's run is called (a produced step
# loads, a pending one is produced), the steps are listed when the stage runs (an earlier stage may have re-keyed them),
# --force unmarks them first, a check stage runs every time, and the retired done-closure form is refused by name.

@testitem "stage_runner" setup = [GitFixture] begin
    using Test
    import VisualGeometryToolkitEval as VgtkEval
    using VisualGeometryToolkitEval.Experiments: Stage, run_stages, Step, produce, produced
    const E = VgtkEval.Experiments

    repo = gitrepo()
    mkpath(joinpath(repo, "src")); write(joinpath(repo, "src", "M.jl"), "module M end\n")
    git!(repo, "add", "-A"); git!(repo, "commit", "-qm", "code")
    E.set_project_root!(repo)
    withenv("VGTK_RESULTS_ROOT" => mktempdir(), "VGTK_SCRATCH" => nothing) do
        ran = String[]
        write_n(n) = d -> (push!(ran, n); write(joinpath(d, "out.txt"), n); nothing)
        step(n) = Step("t", "a", Dict("n" => n); output = "out.txt")
        prod(s) = produce(write_n(s.record["n"]), s; driver = "t.jl", config_source = "t.toml")
        n_steps = Ref(["x", "y"])
        stages = [Stage(:a, "two steps"; steps = () -> step.(n_steps[]), run = prod),
                  Stage(:c, "a check"; check = () -> push!(ran, "check"))]

        @testset "every step produced once, then loaded; a check runs every time" begin
            run_stages("t", stages; argv = String[])
            @test ran == ["x", "y", "check"] && all(produced, step.(["x", "y"]))
            empty!(ran)
            run_stages("t", stages; argv = String[])
            @test ran == ["check"]                                    # produced: loaded, nothing recomputed
        end

        @testset "the steps are listed when the stage runs; --force unmarks them; --plan runs nothing" begin
            empty!(ran); n_steps[] = ["x", "z"]                       # a re-keyed step (an earlier stage's output changed)
            run_stages("t", stages; argv = ["--stage=a"])
            @test ran == ["z"]
            empty!(ran)
            run_stages("t", stages; argv = ["--stage=a", "--force"])
            @test ran == ["x", "z"]
            empty!(ran)
            out = mktemp() do path, io
                redirect_stdout(() -> run_stages("t", stages; argv = ["--plan"]), io)
                close(io); read(path, String)
            end
            @test isempty(ran)
            @test occursin(r"^\s+a\s+2/2 produced\s+two steps$"m, out) && occursin(r"^\s+c\s+check\s+a check$"m, out)
        end
    end
    E.set_project_root!("")

    @testset "the done-closure form is retired; a stage is steps + run, or a check" begin
        e = try Stage(:a, "d", () -> true, () -> nothing); nothing catch err; err end
        @test e isa ArgumentError && occursin("Stage(name, desc, done, run) is retired", e.msg) && occursin("steps = () ->", e.msg)
        @test_throws ArgumentError Stage(:a, "d"; steps = () -> [])                       # no run
        @test_throws ArgumentError Stage(:a, "d"; check = () -> nothing, run = identity)   # both kinds
    end
end
