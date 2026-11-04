# docs_tests.jl — the docstring gates the Documenter build cannot run. Its
# `checkdocs = :public` only sees docstrings that exist, so a public name with
# none, a docstring detached from its definition by a blank or comment line
# (Julia 1.12 makes it a bare string) and a signature line that drifted from
# the methods (a removed keyword) all go unnoticed by it. The General registration review
# asks for every public name to be documented (#173032).
#
# Every public name of PoseLib has a docstring (2026-11-04), so the gates hold
# with no allow-list. An exception, should one arise, takes the form of
# VisualGeometryToolkitCore's test/docs_tests.jl: an EXCEPTIONS dict whose
# every entry carries its reason, kept apart from a BACKLOG that only shrinks,
# and an @info that prints both.

# At this file's top level, not inside the testset: each top-level statement of
# an included file runs in the latest world, so the testset below sees these
# methods. Inside a testset, function or `let` body the include would define
# them in a world that body cannot see.
include(joinpath(@__DIR__, "docs_gates.jl"))

@testset "docs: every public name has a docstring, attached, and its signatures match the methods" begin
    @test isempty(undocumented_public(PoseLib))
    @test isempty(detached_docstrings(joinpath(pkgdir(PoseLib), "src")))
    @test isempty(signature_mismatches(PoseLib))
end
