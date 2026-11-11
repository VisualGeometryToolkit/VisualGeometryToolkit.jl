# docs_tests.jl — the docstring gates the Documenter build cannot run. Its
# `checkdocs = :public` only sees docstrings that exist, so a public name with
# none (`Rigid3` had none), a docstring detached from its definition by a blank
# or comment line (Julia 1.12 makes it a bare string: `RansacWorkspace`'s
# constructors lost theirs that way) and a signature line that drifted from the
# methods (a removed keyword) all go unnoticed by it. The General registration
# review asks for every public name to be documented (#173032).
#
# Two lists, kept apart:
# - EXCEPTIONS: a name that is not documented here ON PURPOSE, with its reason.
# - BACKLOG: names still to document. The list only shrinks: a name documented
#   must leave it, and a new public name without a docstring fails.
# Both are empty: every public name of the package has a docstring.

@testitem "docs: every public name has a docstring, attached, and its signatures match the methods" begin
    using Test, VisualGeometryToolkitEstimation
    using VisualGeometryToolkitCore: test_docs
    test_docs(VisualGeometryToolkitEstimation)     # every list empty: a new exception takes Core's docs_tests.jl form
end
