# docs_tests.jl — the docstring gates the Documenter build cannot run. Its
# `checkdocs = :public` only sees docstrings that exist, so a public name with
# none, a docstring detached from its definition by a blank or comment line
# (Julia 1.12 makes it a bare string) and a signature line that drifted from
# the methods all get through it. The General registration review asks for every
# public name to be documented (#173032). The scans are VisualGeometryToolkitCore's
# `test_docs`, the one copy every stack package calls.
#
# Two lists, kept apart:
# - EXCEPTIONS: a name that is not documented here ON PURPOSE, with its reason.
# - BACKLOG: names still to document. The list only shrinks: a name documented
#   must leave it, and a new public name without a docstring fails.
# The docstring review of 2026-11-04 left both empty.

@testitem "docs: every public name has a docstring, attached, and its signatures match the methods" begin
    using Test, VisualGeometryToolkitIO
    using VisualGeometryToolkitCore: test_docs
    test_docs(VisualGeometryToolkitIO)     # every list empty: a new exception takes Core's docs_tests.jl form
end

# IO 0.10.3: the versions written, in kinds.md, are generated from `_written_versions` (as json-schemas.md's list is),
# and the rule of a length's persisted identity is stated once, in VisualGeometryToolkitCore's conventions/units.md.
@testitem "docs: the versions written are generated; the persisted-identity rule is Core's" begin
    import VisualGeometryToolkitIO as VgtkIO
    import VisualGeometryToolkitCore as VgtkCore
    guide = joinpath(pkgdir(VgtkIO), "docs", "src", "guide")
    kinds = read(joinpath(guide, "kinds.md"), String)
    @test occursin("_written_versions(r.kind)", kinds)
    # Core's paragraph, where Core is this monorepo's (its text ships with Core's next release; a registered Core
    # 0.10.3 holds the earlier paragraph, so IO's suite does not read it there)
    if dirname(pkgdir(VgtkCore)) == dirname(pkgdir(VgtkIO))
        units = read(joinpath(pkgdir(VgtkCore), "conventions", "units.md"), String)
        exact = something(match(r"\*\*What is exact\.\*\*(.*?)\n\n"s, units))[1]
        @test occursin("persisted", exact) && occursin("15 significant", exact)
    end
    @test occursin("conventions/units.md", read(joinpath(guide, "conventions.md"), String))
end
