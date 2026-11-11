# readme_tests.jl — the README's example and the documentation's first example
# run as written, in a clean environment that holds exactly what the README's
# install line adds. A reader's first contact with the package is that code;
# the General registry review of the sibling packages found README blocks with
# undefined variables and install lines that missed a package the example
# loads (#171997, #173031, #174496, #174579).
#
# The install line is a `pkg>` block (```julia-repl), so it is not run as code:
# the item checks that it names every non-standard-library package the
# examples load, and nothing else. The examples then run in a child Julia
# process in a temporary environment — this package by path, the rest of the
# install line from the registries — with none of the test run's load path.

@testitem "readme" begin
    using Test
    import VisualGeometryToolkitCore as VgtkCore
    import VisualGeometryToolkitEval
    # The README's example, run as a reader runs it (VisualGeometryToolkitCore's
    # test_readme_example): one block shown on the landing page, its install line
    # adding exactly what it loads, run in an environment built from that line.
    VgtkCore.test_readme_example(pkgdir(VisualGeometryToolkitEval); checks = """
    # The values the example's comments state.
    @test readme.finite_median(readme.errors) == 1.0
    @test readme.eligible_maa(readme.errors; max_error = 10.0) ≈ 0.73 atol = 0.01
    lo, hi = readme.bootstrap_ci(readme.finite_median, readme.Xoshiro(1), readme.errors)
    @test lo <= 1.0 <= hi
    @test readme.result[:x_star_fine] ≈ 2.0 rtol = 0.05
    @test readme.result[:boundary_hit] == false
    """)
end
