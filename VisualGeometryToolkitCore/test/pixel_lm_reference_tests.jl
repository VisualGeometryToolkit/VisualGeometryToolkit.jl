@testitem "spline LM kernel: within 2e-6 px of the recorded reference LM, the same converged flags" begin
# The spline models' two-unknown LM kernel (`Cameras.static_lm`, device-side: the one
# LM that is not NLLSsolver), at the reference's pixel unit (s = 1, the box [min, max + 0.999]),
# against the routine it derives from — a static bounded
# Levenberg–Marquardt `lmsolve` with Marquardt's scaling — recorded on 1000 seeded problems
# (test/fixtures/pixel_lm: problems.jl, generate.jl, reference.txt). The same algorithm, NOT
# the same bits: both loops lean on `muladd` and StaticArrays' small products, whose fusion
# LLVM decides per compilation context — the reference's own result for one problem moves by
# ulps between a script and a function (2026-10-27) — so a pixel may differ in its last digits. Measured 2026-10-27
# (script context, the fixture's problems): 35 of 1000 pixels differ, by at most 5.5e-12 px,
# on Julia 1.13; 73 of 1000, at most 4.5e-8 px, on 1.12; the review's 20 000-problem probe
# (another context): 1 887 differ, at most 1.8e-6 px; the converged flags identical
# everywhere. Gated: every pixel within 2e-6 px of the reference's, and the
# converged flag at every tolerance the projections use: the reference's (F < tol2)
# where the solve ends at a stationary point; where it does not (an iteration cap of 3
# or 10 ends a solve short of its root, at a cost a lax tol2 accepts), the stack's one
# success rule fails it — the reference's acceptance alone does not succeed.
using Test, LinearAlgebra, StaticArrays, Random
import VisualGeometryToolkitCore as VgtkCore
include(joinpath(pkgdir(VgtkCore), "test", "fixtures", "pixel_lm", "problems.jl"))
lines = readlines(joinpath(pkgdir(VgtkCore), "test", "fixtures", "pixel_lm", "reference.txt"))
ps = pixel_lm_problems()
# the same problems as recorded (a different random stream would move the sum by far more than
# the last-ulp difference Julia 1.12's `randn` gives: 126791.93677324119 vs 1.13's …412)
@test isapprox(pixel_lm_problems_checksum(ps), parse(Float64, lines[1]); rtol = 1.0e-12)
ref = [Tuple(parse.(Float64, split(l))) for l in lines[2:end]]
dmax = 0.0; ndiff = 0; flags_equal = true; nflip = 0
for (q, (rx, ry, rF)) in zip(ps, ref)
    for tol2 in (1.0e-2, 1.0e-6, 1.0e-12)
        res, jac = pixel_res(q.coef...), pixel_jac(q.coef...)    # the kernel's `fj`: both at one point
        stops = VgtkCore.StaticLmStops(SVector(1.0, 1.0); acceptance = tol2, max_iter = q.maxit,
                                       cost_floor = VgtkCore.Cameras._SPLINE_REFERENCE_COST_FLOOR)   # the reference's floor
        x, ok, J, e = VgtkCore.static_lm(p -> (res(p, nothing), jac(p, nothing)), SVector(Float64(q.x0), Float64(q.y0)),
                                         SVector(q.minx, q.miny), SVector(q.maxx + 0.999, q.maxy + 0.999), stops)
        global dmax = max(dmax, abs(x[1] - rx), abs(x[2] - ry))
        tol2 == 1.0e-2 && (x[1] != rx || x[2] != ry) && (global ndiff += 1)
        # stationary at x, read independently of the kernel: the floors, the Gauss–Newton step resolved, or the cosine
        F = sum(abs2, e[1]); g = J' * e[1]
        lo, hi = SVector(q.minx, q.miny), SVector(q.maxx + 0.999, q.maxy + 0.999)
        g = SVector(ntuple(j -> ((x[j] <= lo[j] && g[j] > 0) || (x[j] >= hi[j] && g[j] < 0)) ? 0.0 : g[j], 2))   # a coordinate the box holds
        pgn = (J' * J) \ g                                       # the undamped Gauss–Newton step from x
        resolved = all(abs.(pgn) .< 1.0e-10 .* (1.0e-10 .+ abs.(x)))      # below the step tolerance (s = 1)
        stationary = F < eps(Float64) || maximum(abs, g) < eps(Float64) || resolved ||
            maximum(abs(g[j]) / (norm(J[:, j]) * sqrt(F)) for j in 1:2) <= stops.stationarity_tol
        global flags_equal &= ok == (rF < tol2 && stationary)
        (rF < tol2 && !stationary) && (global nflip += 1)
    end
end
@info "spline LM vs the recorded reference" ndiff dmax nflip
@test dmax <= 2.0e-6
@test flags_equal
end
