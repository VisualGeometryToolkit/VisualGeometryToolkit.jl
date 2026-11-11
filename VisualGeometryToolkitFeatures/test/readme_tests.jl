# readme_tests.jl — the README's example and the documentation's first example
# run as written, in a clean environment that holds exactly what the README's
# install line adds. A reader's first contact with the package is that code,
# and the General registration review (2026-10) found that none of it ran; the
# same review asked the stack for this test (#173031, #174579).
#
# The install line is a `pkg>` block (```julia-repl), so it is not run as code:
# the item checks that it names every non-standard-library package the
# examples load, and nothing else. The example then runs in a child Julia
# process in a temporary environment — this package and VisualGeometryToolkitCore
# by path, the rest of the install line from the registries — with none of the test run's load path,
# and each statement the README marks `# true` is asserted there.

@testitem "README: the install line names what the examples load; the examples run in a fresh environment" begin
    using Test
    import VisualGeometryToolkitCore as VgtkCore
    import VisualGeometryToolkitFeatures
    # The README's example, run as a reader runs it (VisualGeometryToolkitCore's
    # test_readme_example): one block shown on the landing page, its install line
    # adding exactly what it loads, run in an environment built from that line.
    VgtkCore.test_readme_example(pkgdir(VisualGeometryToolkitFeatures); developed = Dict("VisualGeometryToolkitCore" => pkgdir(VgtkCore)))
end
