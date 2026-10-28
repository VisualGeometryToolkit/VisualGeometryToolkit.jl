@testitem "pixel LM kernel: within 2e-6 px of the recorded reference LM, the same converged flags" begin
# The generic models' two-unknown pixel LM kernel (`Cameras._pixel_lm`, device-side: the one
# LM that is not NLLSsolver) against the routine it derives from — a static bounded
# Levenberg–Marquardt `lmsolve` with Marquardt's scaling — recorded on 1000 seeded problems
# (test/fixtures/pixel_lm: problems.jl, generate.jl, reference.txt). The same algorithm, NOT
# the same bits: both loops lean on `muladd` and StaticArrays' small products, whose fusion
# LLVM decides per compilation context — the reference's own result for one problem moves by
# ulps between a script and a function (2026-10-27) — so a pixel may differ in its last digits. Measured 2026-10-27
# (script context, the fixture's problems): 35 of 1000 pixels differ, by at most 5.5e-12 px,
# on Julia 1.13; 73 of 1000, at most 4.5e-8 px, on 1.12; the review's 20 000-problem probe
# (another context): 1 887 differ, at most 1.8e-6 px; the converged flags identical
# everywhere. Gated: every pixel within 2e-6 px of the reference's, and the
# converged flag (F < tol2) the same at every tolerance the projections use.
using Test, LinearAlgebra, StaticArrays, Random
import VisualGeometryToolkitCore as VgtkCore
include(joinpath(pkgdir(VgtkCore), "test", "fixtures", "pixel_lm", "problems.jl"))
lines = readlines(joinpath(pkgdir(VgtkCore), "test", "fixtures", "pixel_lm", "reference.txt"))
ps = pixel_lm_problems()
# the same problems as recorded (a different random stream would move the sum by far more than
# the last-ulp difference Julia 1.12's `randn` gives: 126791.93677324119 vs 1.13's …412)
@test isapprox(pixel_lm_problems_checksum(ps), parse(Float64, lines[1]); rtol = 1.0e-12)
ref = [Tuple(parse.(Float64, split(l))) for l in lines[2:end]]
dmax = 0.0; ndiff = 0; flags_equal = true
for (q, (rx, ry, rF)) in zip(ps, ref)
    for tol2 in (1.0e-2, 1.0e-6, 1.0e-12)
        x, ok = VgtkCore.Cameras._pixel_lm(pixel_res(q.coef...), pixel_jac(q.coef...), q.x0, q.y0, q.minx, q.miny, q.maxx, q.maxy, q.maxit, tol2)
        global dmax = max(dmax, abs(x[1] - rx), abs(x[2] - ry))
        tol2 == 1.0e-2 && (x[1] != rx || x[2] != ry) && (global ndiff += 1)
        global flags_equal &= ok == (rF < tol2)
    end
end
@info "pixel LM vs the recorded reference" ndiff dmax
@test dmax <= 2.0e-6
@test flags_equal
end
