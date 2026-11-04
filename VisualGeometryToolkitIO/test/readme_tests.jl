# readme_tests.jl — the README's example and the documentation's first example
# run as written, in a clean environment that holds exactly what the README's
# install line adds. A reader's first contact with the package is that code; it
# once used a `pose` it never defined, and nothing caught the UndefVarError
# (General registration review, 2026-10). The same review asked the stack for
# this test (#173031, #174579).
#
# The install line is a `pkg>` block (```julia-repl), so it is not run as code:
# the item checks that it names every non-standard-library package the
# examples load, and nothing else. The examples then run in a child Julia
# process in a temporary environment — this package by path, the rest of the
# install line from the registries — with none of the test run's load path.

@testitem "README: the install line names what the examples load; the examples run in a fresh environment" begin
    using Test, VisualGeometryToolkitIO
    root = pkgdir(VisualGeometryToolkitIO)
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

    env = mktempdir()
    write(joinpath(env, "readme_example.jl"), only(example))
    registered = filter(!=("VisualGeometryToolkitIO"), String.(packages))
    script = joinpath(env, "run.jl")
    write(script, """
    using Pkg
    Pkg.develop(path = $(repr(root)); io = devnull)
    isempty($(repr(registered))) || Pkg.add($(repr(registered)); io = devnull)
    readme = Module(:ReadmeSandbox)
    Base.include(readme, $(repr(joinpath(env, "readme_example.jl"))))
    using Test
    @testset "README example" begin
        # What the example's comments claim: the pose comes back with its
        # frames and its translation exactly, the camera equal, and the store
        # validates as a camera store.
        @test readme.X2.t == readme.X.t
        @test readme.X2.R ≈ readme.X.R
        @test readme.X2 isa readme.EuclideanMap{3, Float64, <:Any, <:Any, readme.BodyFrame{:camera_rig}, readme.CameraFrame}
        @test readme.VgtkIO.load(readme.BoundedCamera, readme.store; name = :left) == readme.camera
        @test readme.VgtkIO.validate(readme.store) == "cameras"
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
