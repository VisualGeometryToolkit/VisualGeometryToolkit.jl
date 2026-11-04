# readme_tests.jl — the README's example and the documentation's first example
# run as written, in a clean environment that holds exactly what the README's
# install line adds. A reader's first contact with the package is that code,
# and the General registration review (2026-10) found that none of it ran; the
# same review asked the stack for this test (#173031, #174579).
#
# The install line is a `pkg>` block (```julia-repl), so it is not run as code:
# the item checks that it names every non-standard-library package the
# examples load, and nothing else. The example then runs in a child Julia
# process in a temporary environment — this package and VisualGeometryToolkitCore
# by path, the rest of the install line from the registries — with none of the test run's load path,
# and each statement the README marks `# true` is asserted there.

@testitem "README: the install line names what the examples load; the examples run in a fresh environment" begin
    using Test, VisualGeometryToolkitFeatures, VisualGeometryToolkitCore
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

    @test occursin(r"#\s*true$"m, only(example))     # the example states its claims

    # The stack packages are developed by path, VisualGeometryToolkitCore too
    # (a dependency the install line does not name): a released Core need not
    # hold what this checkout's Features calls.
    developed = [root, pkgdir(VisualGeometryToolkitCore)]
    registered = filter(!=("VisualGeometryToolkitFeatures"), String.(packages))
    env = mktempdir()
    write(joinpath(env, "readme_example.jl"), only(example))
    script = joinpath(env, "run.jl")
    # The child runs the example statement by statement, in one module, and
    # asserts every statement whose last line ends in `# true`.
    write(script, """
    using Pkg
    Pkg.develop([PackageSpec(path = p) for p in $(repr(developed))]; io = devnull)
    isempty($(repr(registered))) || Pkg.add($(repr(registered)); io = devnull)
    using Test
    readme = Module(:ReadmeSandbox)
    code = read($(repr(joinpath(env, "readme_example.jl"))), String)
    @testset "README example" begin
        pos = firstindex(code)
        nclaims = 0
        while pos <= ncodeunits(code)
            ex, next = Meta.parse(code, pos; greedy = true, raise = true)
            chunk = code[pos:prevind(code, next)]
            pos = next
            ex === nothing && continue
            value = Core.eval(readme, ex)
            if occursin(r"#\\s*true\\z", rstrip(chunk))
                nclaims += 1
                @test value === true
            end
        end
        @test nclaims == $(count(r"#\s*true$"m, only(example)))   # every claim was asserted
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
