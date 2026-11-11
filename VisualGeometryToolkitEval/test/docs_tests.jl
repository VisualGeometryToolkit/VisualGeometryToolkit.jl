# docs_tests.jl — the docstring gates the Documenter build cannot run. Its
# `checkdocs = :public` only sees docstrings that exist, so a public name with
# none, a docstring detached from its definition by a blank or comment line
# (Julia 1.12 makes it a bare string, which documents nothing) and a signature
# line that drifted from the methods (a removed keyword) all get through it. The
# General registration review asks for every public name to be documented
# (#173032).
#
# Two lists, kept apart, both empty at the docs review of 2026-11-04:
# - EXCEPTIONS: a name that is not documented ON PURPOSE, with its reason.
# - BACKLOG: names still to document. The list only shrinks: a name documented
#   must leave it, and a new public name without a docstring fails.

@testitem "docs" begin
    using Test, VisualGeometryToolkitEval
    using VisualGeometryToolkitCore: test_docs
    test_docs(VisualGeometryToolkitEval)
end
