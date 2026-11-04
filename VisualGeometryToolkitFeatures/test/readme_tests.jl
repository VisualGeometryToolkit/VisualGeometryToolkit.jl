# readme_tests.jl — the README's example and the documentation's first example
# run as written, in a clean environment that holds exactly what the README's
# install line adds. A reader's first contact with the package is that code,
# and the General registration review (2026-10) found that none of it ran; the
# same review asked the stack for this test (#173031, #174579).
#
# The install line is a `pkg>` block (```julia-repl), so it is not run as code:
# the item checks that it names every non-standard-library package the
# examples load, and nothing else. The example then runs in a child Julia
# process in a temporary environment — this package by path, the rest of the
# install line from the registries — with none of the test run's load path,
# and each statement the README marks `# true` is asserted there.

@testitem "README: the install line names what the examples load; the examples run in a fresh environment" begin
    using Test, VisualGeometryToolkitFeatures
    root = pkgdir(VisualGeometryToolkitFeatures)
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
    for code in example, m in eachmatch(r"^(?:using|import)\s+([^:\n]+)"m, code)
        for part in split(m[1], ',')
            push!(loaded, first(split(strip(part), r"[.\s]")))
        end
    end
    stdlib(name) = isdir(joinpath(Sys.STDLIB, name))
    @test Set(filter(!stdlib, loaded)) == Set(packages)
    @test occursin("pkg> add $(only(install))", index)

    # The README marks its claims `# true`; the child asserts the same four.
    @test count(r"#\s*true$"m, only(example)) == 4

    env = mktempdir()
    write(joinpath(env, "readme_example.jl"), only(example))
    registered = filter(!=("VisualGeometryToolkitFeatures"), String.(packages))
    script = joinpath(env, "run.jl")
    write(script, """
    using Pkg
    Pkg.develop(path = $(repr(root)); io = devnull)
    isempty($(repr(registered))) || Pkg.add($(repr(registered)); io = devnull)
    readme = Module(:ReadmeSandbox)
    Base.include(readme, $(repr(joinpath(env, "readme_example.jl"))))
    using Test
    @testset "README example" begin
        # Three blobs, one of them dark, found at its COLMAP centre; every
        # mutual nearest neighbour undoes the shuffle.
        @test length(readme.dets) == 3
        @test count(readme.is_dark, readme.dets) == 1
        @test round.(readme.dark.center; digits = 1) == [99.5, 69.5]
        @test all(readme.perm[m.second] == m.first for m in readme.matches)
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
