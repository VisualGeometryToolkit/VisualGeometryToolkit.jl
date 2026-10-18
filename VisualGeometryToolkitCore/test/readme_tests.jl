# readme_tests.jl — the README's Quick Start runs. A reviewer's first contact
# with the package is that block; it once called `has_boundary`, which is public
# but not exported, and nothing caught the UndefVarError (General registration
# review, 2026-10-17). The plotting line is dropped: no display in tests.

@testitem "README quick start" begin
    using Test, VisualGeometryToolkitCore
    readme = read(joinpath(pkgdir(VisualGeometryToolkitCore), "README.md"), String)
    i = findfirst("## Quick Start", readme)
    @test i !== nothing
    block = match(r"```julia\n(.*?)```"s, readme, last(i))
    @test block !== nothing
    code = join(filter(l -> !occursin("GLMakie", l), split(block[1], '\n')), '\n')
    m = Module(:ReadmeQuickStart)
    Core.eval(m, :(using VisualGeometryToolkitCore))
    @test (include_string(m, code, "README.md (Quick Start)"); true)
end
