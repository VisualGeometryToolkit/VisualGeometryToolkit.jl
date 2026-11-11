# The stack's iterative-estimator vocabulary, checked by VisualGeometryToolkitCore's
# `check_naming` on this package. The rules and the lists of banned words and
# retired names live in VisualGeometryToolkitCore (ext/VisualGeometryToolkitCoreTestExt/check_naming.jl, loaded with Test), which
# also tests the checker itself. Upstream PoseLib's C++ is not in this
# repository (the recipe in yggdrasil/ builds it); the stack-authored wrapper, in wrapper/
# and its recipe copy yggdrasil/P/PoseLib/bundled/, is inside the scan.
import VisualGeometryToolkitCore as VgtkCore
@testset "quality: naming" begin
    VgtkCore.test_naming(PoseLib)
end

# One implementation per concept (VisualGeometryToolkitCore's check_duplication); the other
# stack repositories in VGTK_STACK_PATHS are compared too.
@testset "quality: no duplication" begin
    VgtkCore.test_duplication(PoseLib)
end

@testset "quality: stack contracts" begin
    root = pkgdir(PoseLib)
    dirs = filter(isdir, [joinpath(root, "src"), joinpath(root, "ext")])
    @test isempty(reduce(vcat, [VgtkCore.private_name_violations(d, "PoseLib") for d in dirs]))
    @test isempty(VgtkCore.contract_generic_violations(PoseLib))
end
