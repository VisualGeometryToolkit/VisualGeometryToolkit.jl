# readme_tests.jl — the README's example and the documentation's first example
# run as written, in a clean environment that holds exactly what the README's
# install line adds. The General registration review (2026-10, #173033) found
# that none of the README's examples ran: undefined `src`/`dst`, `[ ... ]`
# placeholders, and an installation that never added VisualGeometryToolkitCore;
# #174579 asked for the install line to be tested too.
#
# The install line is a `pkg>` block (```julia-repl), so it is not run as code:
# the item checks that it names every non-standard-library package the
# examples load, and nothing else. The examples then run in a child Julia
# process in a temporary environment — this package and VisualGeometryToolkitCore
# by path (both are workspace members, developed together), the rest of the
# install line from the registries — with none of the test run's load path.

@testitem "README: the install line names what the examples load; the examples run in a fresh environment" begin
    using Test, VisualGeometryToolkitEstimation, VisualGeometryToolkitCore
    root = pkgdir(VisualGeometryToolkitEstimation)
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

    # The stack packages are developed by path: a released Core need not hold
    # what this checkout's Estimation calls.
    local_paths = Dict("VisualGeometryToolkitEstimation" => root,
                       "VisualGeometryToolkitCore" => pkgdir(VisualGeometryToolkitCore))
    developed = [local_paths[p] for p in packages if haskey(local_paths, p)]
    registered = filter(p -> !haskey(local_paths, p), String.(packages))
    env = mktempdir()
    write(joinpath(env, "readme_example.jl"), only(example))
    script = joinpath(env, "run.jl")
    write(script, """
    using Pkg
    Pkg.develop([PackageSpec(path = p) for p in $(repr(developed))]; io = devnull)
    isempty($(repr(registered))) || Pkg.add($(repr(registered)); io = devnull)
    readme = Module(:ReadmeSandbox)
    Base.include(readme, $(repr(joinpath(env, "readme_example.jl"))))
    using Test
    @testset "README example" begin
        # The comments' claims: the 32 true correspondences are the inliers,
        # and the estimate is the true homography up to scale.
        @test count(readme.result.inlier_mask) == 32
        @test !any(readme.result.inlier_mask[1:8])
        @test readme.H ./ readme.H[3, 3] ≈ Matrix(readme.H_true)
        @test readme.result_2px isa typeof(readme.result)
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
