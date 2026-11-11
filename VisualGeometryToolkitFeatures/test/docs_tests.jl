# docs_tests.jl — the docstring gates the Documenter build cannot run. Its
# `checkdocs = :public` only sees docstrings that exist, so a public name with
# none, a docstring detached from its definition by a blank or comment line
# (Julia 1.12 makes it a bare string) and a signature line that drifted from
# the methods (a removed keyword) all get through it. The General registration
# review asks for every public name to be documented (#173032).
#
# The scans are VisualGeometryToolkitCore's `test_docs`, the one copy every stack package calls.
# Two lists, kept apart:
# - EXCEPTIONS: a name that is not documented here ON PURPOSE, with its reason.
# - BACKLOG: names still to document. The list only shrinks: a name documented
#   must leave it, and a new public name without a docstring fails.
# The docs review of 2026-11-04 left both empty.

@testitem "docs: every public name has a docstring, attached, and its signatures match the methods" begin
    using Test, VisualGeometryToolkitFeatures
    using VisualGeometryToolkitCore: test_docs
    test_docs(VisualGeometryToolkitFeatures; dirs = ("src", "ext"))     # the extensions document methods too; every list empty: a new exception takes Core's docs_tests.jl form
end
