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
    include(joinpath(@__DIR__, "docs_gates.jl"))
    M = VisualGeometryToolkitEstimation

    EXCEPTIONS = Dict{String, String}()
    BACKLOG = Set{String}()
    SIGNATURE_EXCEPTIONS = Set{String}()

    undocumented = Set(undocumented_public(M))
    @test isempty(setdiff(undocumented, keys(EXCEPTIONS), BACKLOG))     # a new public name is documented
    @test isempty(setdiff(BACKLOG, undocumented))                       # a documented name leaves the backlog
    @test isempty(setdiff(keys(EXCEPTIONS), undocumented))              # a stale exception is removed

    @test isempty(detached_docstrings(joinpath(pkgdir(M), "src")))

    mismatches = Set(signature_mismatches(M))
    @test isempty(setdiff(mismatches, SIGNATURE_EXCEPTIONS))
    @test isempty(setdiff(SIGNATURE_EXCEPTIONS, mismatches))           # a stale exception is removed

    # The report lists the exceptions, so they stay visible.
    @info "docs gate exceptions" EXCEPTIONS SIGNATURE_EXCEPTIONS backlog = length(BACKLOG)
end
