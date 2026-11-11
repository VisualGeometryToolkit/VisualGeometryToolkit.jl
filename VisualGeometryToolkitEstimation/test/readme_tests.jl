# readme_tests.jl — the README's example and the documentation's first example
# run as written, in a clean environment that holds exactly what the README's
# install line adds. The General registration review (2026-10, #173033) found
# that none of the README's examples ran: undefined `src`/`dst`, `[ ... ]`
# placeholders, and an installation that never added VisualGeometryToolkitCore;
# #174579 asked for the install line to be tested too.
#
# The install line is a `pkg>` block (```julia-repl), so it is not run as code:
# the item checks that it names every non-standard-library package the
# examples load, and nothing else. The examples then run in a child Julia
# process in a temporary environment — this package and VisualGeometryToolkitCore
# by path (both are workspace members, developed together), the rest of the
# install line from the registries — with none of the test run's load path.

@testitem "README: the install line names what the examples load; the examples run in a fresh environment" begin
    using Test
    import VisualGeometryToolkitCore as VgtkCore
    import VisualGeometryToolkitEstimation
    # The README's example, run as a reader runs it (VisualGeometryToolkitCore's
    # test_readme_example): one block shown on the landing page, its install line
    # adding exactly what it loads, run in an environment built from that line.
    VgtkCore.test_readme_example(pkgdir(VisualGeometryToolkitEstimation); developed = Dict("VisualGeometryToolkitCore" => pkgdir(VgtkCore)), install_pages = (joinpath("docs", "src", "guide", "getting-started.md"),), checks = """
    # The comments' claims: the 32 true correspondences are the inliers,
    # and the estimate is the true homography up to scale. The data are
    # noise-free, so the 2 px threshold keeps the same inliers.
    @test count(readme.result.inlier_mask) == 32
    @test !any(readme.result.inlier_mask[1:8])
    @test readme.H ./ readme.H[3, 3] ≈ Matrix(readme.H_true)
    @test readme.result_2px.inlier_mask == readme.result.inlier_mask
    """)
end
