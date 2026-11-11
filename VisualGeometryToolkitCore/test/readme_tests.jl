# readme_tests.jl — the README's example and the documentation's first example
# run as written, in a clean environment that holds exactly what the README's
# install line adds. A reader's first contact with the package is that code; it
# once called `has_boundary`, which is public but not exported (General
# registration review, #171997), and later examples used a `camera`, an `img`
# and a `matrix` they never defined. The same review asked the stack for this
# test (#173031, #174579).
#
# The install line is a `pkg>` block (```julia-repl), so it is not run as code:
# the item checks that it names every non-standard-library package the
# examples load, and nothing else. The examples then run in a child Julia
# process in a temporary environment — this package by path, the rest of the
# install line from the registries — with none of the test run's load path.

@testitem "README: the install line names what the examples load; the examples run in a fresh environment" begin
    using Test
    import VisualGeometryToolkitCore as VgtkCore
    # The README's example, run as a reader runs it (VisualGeometryToolkitCore's
    # test_readme_example): one block shown on the landing page, its install line
    # adding exactly what it loads, run in an environment built from that line.
    VgtkCore.test_readme_example(pkgdir(VgtkCore); checks = """
    # The homography reproduces every projection, and the circle at the
    # board's centre images as an ellipse at the principal point.
    @test all(readme.to_euclidean(readme.H(p)) ≈ u for (p, u) in zip(readme.board, readme.pixels))
    @test readme.image isa readme.Ellipse
    @test readme.centroid(readme.image) ≈ readme.centroid(readme.Rect(readme.s)) atol = 1.0
    """)
end
