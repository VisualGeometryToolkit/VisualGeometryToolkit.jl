# The precompile workload (PrecompileTools, decision 13): a homography's refine (`LevenbergMarquardt` from its DLT, the
# chart LM and its rows), run once while the package image is built, so a fresh process's first refine does not
# compile it again. CPU only, no data files, no output, in Float64. Core's workload holds the closed forms and the
# minimal solves.
#
# The find (`Ransac`) is deliberately NOT here: its scoring loop (`mahalanobis_distances!`) compiled into the package
# image ran 5 % slower than the JIT copy of the same method (24.2 → 25.4 μs per 3000-point model; the same algorithm,
# laid out differently), which put a find without local optimization 3.7 % over the 0.12 runtime gate. The owner's
# decision (2026-11-24): the workload stays refine-only, the one form with which every gate holds. Left to the JIT,
# the find compiles on its first call and runs at the old speed.

@setup_workload begin
    H = SMatrix{3, 3, Float64, 9}(1.2, 0.05, 1.0e-4, -0.1, 1.1, 2.0e-4, 30.0, -20.0, 1.0)
    src = [SVector{2, Float64}(40.0 * i + 7.0 * (i % 3), 30.0 * (i % 7) + 5.0 * i) for i in 1:40]
    dst = map(src) do s
        h = H * SVector(s[1], s[2], 1.0)
        SVector(h[1] / h[3], h[2] / h[3])
    end
    @compile_workload begin
        for cs in (VgtkCore.correspondences(src, dst), src .=> dst)
            f = HomographyProblem(cs)
            H0 = solve(EstimationProblem(VgtkCore.HomographyMat, cs), VgtkCore.DLT()).u
            solve(RefineProblem(f, H0), VgtkCore.LevenbergMarquardt(); scale = 1.0)
        end
    end
end
