# readme_tests.jl — the README's example and the documentation's first example
# run as written, in a clean environment that holds exactly what the README's
# install line adds. A reader's first contact with the package is that code; a
# sibling package's example once called a name that was public but not
# exported (General registration review, #171997), and later examples used
# values they never defined. The same review asked the stack for this test
# (#173031, #174579).
#
# The install line is a `pkg>` block (```julia-repl), so it is not run as code:
# the testset checks that it names every non-standard-library package the
# examples load, and nothing else. The example then runs in a child Julia
# process in a temporary environment — this package and VisualGeometryToolkitCore
# by path (both are workspace members, developed together: a registry's Core
# need not be the one this package's compat names), the rest of the install
# line from the registries — with none of the test run's load path.

@testset "README: the install line names what the examples load; the examples run in a fresh environment" begin
    # The README's example, run as a reader runs it (VisualGeometryToolkitCore's
    # test_readme_example): one block shown on the landing page, its install line
    # adding exactly what it loads, run in an environment built from that line.
    VisualGeometryToolkitCore.test_readme_example(pkgdir(PoseLib);
        developed = Dict("VisualGeometryToolkitCore" => pkgdir(VisualGeometryToolkitCore)), checks = """
    # P3P finds the true pose among its solutions, and the refinement keeps
    # the camera at the origin, looking down the z axis.
    @test 1 <= length(readme.poses) <= 4
    @test sqrt(readme.reprojection(readme.best) / length(readme.X)) < 1e-6   # RMS, pixels
    @test maximum(abs, readme.pose.t) < 1e-8
    @test maximum(abs, readme.pose.R - [1 0 0; 0 1 0; 0 0 1]) < 1e-8
    """)
end
