@testitem "readme helpers: the README's examples, run as a reader runs them" begin
    # One copy of what every stack package's readme test restated (the monorepo's
    # packages, RSDR, LODR, PanoramaStitching) and of BlobBoards' and CaliBlobs'
    # install-row runners. A throwaway package with a README is run end to end:
    # its one example in an environment built from its install line alone.
    using Test
    using VisualGeometryToolkitCore: ReadmeExample, readme_examples, install_table, install_packages, loaded_packages,
        readme_environment, test_readme_row, test_readme_example
    root = mktempdir()
    mkpath(joinpath(root, "src")); mkpath(joinpath(root, "docs", "src"))
    write(joinpath(root, "Project.toml"), "name = \"TinyReadme\"\nuuid = \"5b8c6a1e-2f4d-4c3b-9e7a-1d2c3b4a5f60\"\nversion = \"0.1.0\"\n")
    write(joinpath(root, "src", "TinyReadme.jl"), "module TinyReadme\nanswer() = 42\nend\n")
    block = "```julia\n# Install row: compute\nusing TinyReadme   # the package\nusing LinearAlgebra\nx = TinyReadme.answer()\nx == 42   # true\n```\n"
    table = "| task | install | note |\n|---|---|---|\n| compute | `pkg> add TinyReadme` | n |\n| other thing | nothing | n |\n"
    page = "# TinyReadme\n\n```julia-repl\npkg> add TinyReadme\n```\n\n" * block * "\n" * table
    write(joinpath(root, "README.md"), page)
    write(joinpath(root, "docs", "src", "index.md"), page)

    ex = readme_examples(root, ["README.md"])
    @test length(ex) == 1 && only(ex) isa ReadmeExample
    @test only(ex).row == "compute" && only(ex).lang == "julia" && !only(ex).illustrative
    @test loaded_packages(only(ex).code) == Set(["TinyReadme"])        # LinearAlgebra is the standard library
    @test install_table(joinpath(root, "README.md"), ["compute"]) == Dict("compute" => "`pkg> add TinyReadme`")
    @test install_packages("`pkg> add A B`") == ["A", "B"] && isempty(install_packages("clone it"))

    test_readme_example(root; checks = "@test readme.x == 42")
    env = readme_environment(root, ["TinyReadme"])
    test_readme_row(env, "compute", [only(ex).code]; after = "@test length(mods) == 1")
end
