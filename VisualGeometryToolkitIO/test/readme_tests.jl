# readme_tests.jl — the README's example and the documentation's first example
# run as written, in a clean environment that holds exactly what the README's
# install line adds. A reader's first contact with the package is that code; it
# once used a `pose` it never defined, and nothing caught the UndefVarError
# (General registration review, 2026-10). The same review asked the stack for
# this test (#173031, #174579).
#
# The install line is a `pkg>` block (```julia-repl), so it is not run as code:
# the item checks that it names every non-standard-library package the
# examples load, and nothing else. The examples then run in a child Julia
# process in a temporary environment — this package and VisualGeometryToolkitCore
# by path (both are workspace members, developed together: a registry's Core
# need not be the one this package's compat names), the rest of the install
# line from the registries — with none of the test run's load path.
#
# The item is VisualGeometryToolkitCore's test/readme_tests.jl with this
# package's name and assertions; keep the two in step until the shared part
# moves into Core beside the docs gates.

@testitem "README: the install line names what the examples load; the examples run in a fresh environment" begin
    using Test
    import VisualGeometryToolkitCore as VgtkCore
    import VisualGeometryToolkitIO
    # The README's example, run as a reader runs it (VisualGeometryToolkitCore's
    # test_readme_example): one block shown on the landing page, its install line
    # adding exactly what it loads, run in an environment built from that line.
    VgtkCore.test_readme_example(pkgdir(VisualGeometryToolkitIO); developed = Dict("VisualGeometryToolkitCore" => pkgdir(VgtkCore)), checks = """
    # What the example's comments claim: the pose comes back with its
    # frames and its translation exactly, the camera equal, and the store
    # validates as a camera store.
    @test readme.X2.t == readme.X.t
    @test readme.X2.R ≈ readme.X.R
    @test readme.X2 isa readme.EuclideanMap{3, Float64, <:Any, <:Any, readme.BodyFrame{:camera_rig}, readme.CameraFrame}
    @test readme.VgtkIO.load(readme.BoundedCamera, readme.store; name = :left) == readme.camera
    @test readme.VgtkIO.validate(readme.store) == "cameras"
    """)
end
