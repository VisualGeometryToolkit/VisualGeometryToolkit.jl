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

# The scans are VisualGeometryToolkitCore's `test_docs`, the one copy every stack package calls.
using VisualGeometryToolkitCore: test_docs

@testset "docs: every public name has a docstring, attached, and its signatures match the methods" begin
    test_docs(PoseLib)
end
