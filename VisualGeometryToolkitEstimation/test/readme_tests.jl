# readme_tests.jl — every ```julia block of the README runs. The General
# registration review (2026-10, PR #173033) found that none of the README's
# examples ran: undefined `src`/`dst`, `[ ... ]` placeholders, and an
# installation that never added VisualGeometryToolkitCore. The blocks run in order
# in one fresh module, so a later block may use what an earlier one defined,
# exactly as a reader pasting them into a REPL would. Only blocks fenced
# ```julia are code to run; the installation is fenced ```julia-repl (a `pkg>`
# line) and is not run, since it would modify the test environment.

@testitem "README julia blocks" begin
    using Test, VisualGeometryToolkitEstimation
    readme = read(joinpath(pkgdir(VisualGeometryToolkitEstimation), "README.md"), String)
    blocks = [m[1] for m in eachmatch(r"^```julia\n(.*?)^```"ms, readme)]
    @test !isempty(blocks)
    sandbox = Module(:ReadmeExamples)
    for (i, code) in enumerate(blocks)
        @test (include_string(sandbox, code, "README.md (julia block $i)"); true)
    end
    # The Quick start's commented claims hold.
    @test count(Core.eval(sandbox, :result).inlier_mask) == 32
end
