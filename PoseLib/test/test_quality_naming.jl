# The stack's iterative-estimator vocabulary, checked by VisualGeometryToolkitCore's
# `check_naming` on this package. The rules and the lists of banned words and
# retired names live in VisualGeometryToolkitCore (ext/VisualGeometryToolkitCoreTestExt/check_naming.jl, loaded with Test), which
# also tests the checker itself. Upstream PoseLib's C++ is not in this
# repository (the recipe in yggdrasil/ builds it); the stack-authored wrapper, in wrapper/
# and its recipe copy yggdrasil/P/PoseLib/bundled/, is inside the scan.
import VisualGeometryToolkitCore as VgtkCore
@testset "quality: naming" begin
    found = VgtkCore.check_naming(pkgdir(PoseLib))
    isempty(found) || foreach(v -> println(stderr, v), found)
    @test isempty(found)
end
