# readme_tests.jl — the README's example and the documentation's first example
# run as written, in a clean environment that holds exactly what the README's
# install line adds. A reader's first contact with the package is that code;
# the General registry review of the sibling packages found README blocks with
# undefined variables and install lines that missed a package the example
# loads (#171997, #173031, #174496, #174579).
#
# The install line is a `pkg>` block (```julia-repl), so it is not run as code:
# the item checks that it names every non-standard-library package the
# examples load, and nothing else. The examples then run in a child Julia
# process in a temporary environment — this package by path, the rest of the
# install line from the registries — with none of the test run's load path.

@testitem "readme" begin
    using Test, VisualGeometryToolkitEval
    root = pkgdir(VisualGeometryToolkitEval)
    julia_blocks(text) = [m[1] for m in eachmatch(r"^```julia\n(.*?)^```"ms, text)]
    readme = read(joinpath(root, "README.md"), String)
    index = read(joinpath(root, "docs", "src", "index.md"), String)
    example = julia_blocks(readme)
    landing = julia_blocks(index)
    @test length(example) == 1
    @test landing == example                         # one example, shown twice

    # The install line adds exactly the packages the examples load.
    install = [m[1] for m in eachmatch(r"^pkg> add (.+)$"m, readme)]
    @test length(install) == 1
    packages = split(only(install))
    loaded = Set{String}()
    for code in example, m in eachmatch(r"^\s*(?:using|import)\s+([^:\n]+)"m, code)
        for part in split(replace(m[1], r"#.*" => ""), ',')   # a trailing comment is not a package
            push!(loaded, first(split(strip(part), r"[.\s]")))
        end
    end
    stdlib(name) = isdir(joinpath(Sys.STDLIB, name))
    @test Set(filter(!stdlib, loaded)) == Set(packages)
    @test occursin("pkg> add $(only(install))", index)

    env = mktempdir()
    write(joinpath(env, "readme_example.jl"), only(example))
    # The install line names this package alone (the set test above), so the
    # environment is this package, developed by path.
    script = joinpath(env, "run.jl")
    write(script, """
    using Pkg
    Pkg.develop(path = $(repr(root)); io = devnull)
    readme = Module(:ReadmeSandbox)
    Base.include(readme, $(repr(joinpath(env, "readme_example.jl"))))
    using Test
    @testset "README example" begin
        # The values the example's comments state.
        @test readme.finite_median(readme.errors) == 1.0
        @test readme.eligible_maa(readme.errors; max_error = 10.0) ≈ 0.73 atol = 0.01
        lo, hi = readme.bootstrap_ci(readme.finite_median, readme.Xoshiro(1), readme.errors)
        @test lo <= 1.0 <= hi
        @test readme.result[:x_star_fine] ≈ 2.0 rtol = 0.05
        @test readme.result[:boundary_hit] == false
    end
    """)
    # Base.julia_cmd() carries the test run's flags (--depwarn=error, coverage).
    # The load path is the project and the standard library only: no default
    # environment (@v1.12) can supply a package the install line does not add.
    cmd = `$(Base.julia_cmd()) --startup-file=no --project=$env $script`
    proc = withenv("JULIA_LOAD_PATH" => "@:@stdlib") do   # --project overrides JULIA_PROJECT
        run(ignorestatus(cmd))
    end
    @test proc.exitcode == 0
end
