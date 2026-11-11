# =============================================================================
# spline_timing/bench.jl — the SplineLens timing benchmark of Core's timing gate
# (test/spline_timing_gate_tests.jl; the gate itself is test/lo_timing/gate.jl, the
# stack's one gate). A standalone script, run in its own single-threaded process
# pinned to one core by the gate:
#
#   taskset -c <core> julia -t 1 --project=<test env> test/spline_timing/bench.jl <out.toml> [reps]
#
# THE CPU HOT PATHS of the spline cameras (plan 1b, Task 4b), on the golden camera of
# test/fixtures/spline_lens_golden.jl (KB fisheye f = 300 px, 1000 × 1000, 24 × 24
# lattice); one "call" is one sweep over the arm's point set:
#   backproject            the scalar backprojection of the 289 golden lattice pixels
#   backproject-batch      the batched (KA CPU kernel) backprojection of 10⁴ pixels
#   backproject-jacobian   the bearing and its Jacobian at the 289 lattice pixels
#   project-lm             the scalar LM projection of the 289 golden rays (cold start)
#   project-forward        the same rays through the lens's forward spline
#   project-batch          the batched LM projection of 10⁴ rays, seeds 2 px off
#   noncentral-project     the non-central LM projection of the 289 lattice points
#   undistort-128          the exact undistortion of a 128 × 128 window (every pixel
#                          through the LM), KA CPU backend
# (The comparison against the OLD grid, at BASE 004a004c, is benchmark/spline_lens.jl's.)
# =============================================================================

using VisualGeometryToolkitCore
using VisualGeometryToolkitCore: CameraModel, SplineLens, ChartSpline, NoncentralSplineModel, KannalaBrandt, px, project,
    backproject, backproject_with_jacobian, undistort, calibration_matrix, Rect2, LogicalIntrinsics, CameraCalibrationMat
using VisualGeometryToolkitCore.Cameras: _spline_camera
import KernelAbstractions as KA
using StaticArrays

const OUT = ARGS[1]
const REPS = length(ARGS) >= 2 ? parse(Int, ARGS[2]) : 7
const SWEEPS = 50             # sweeps per rep (one "call" = one sweep)
include(joinpath(@__DIR__, "..", "lo_timing", "bench_record.jl"))   # calibration_spread, kernel_arm, write_bench_record: the record the gate reads

Threads.nthreads() == 1 || error("bench.jl: run single-threaded (julia -t 1); got $(Threads.nthreads()) threads")

include(joinpath(pkgdir(VisualGeometryToolkitCore), "test", "fixtures", "spline_lens_golden.jl"))
const G = GOLDEN
const SRC = CameraModel(G.f * px, SA[G.pp[1] * px, G.pp[2] * px]; lens = KannalaBrandt(G.kb...))
const CAM = CameraModel(SplineLens, SRC, Size2(G.image[1], G.image[2]); grid_size = (G.grid_size[1], G.grid_size[2]))
const CAM_FWD = let K = calibration_matrix(CAM), S = G.spline
    K′ = LogicalIntrinsics(CameraCalibrationMat(SMatrix{3, 3, Float64}(K[1, 1], 0.0, 0.0, 0.0, K[2, 2], 0.0, 499.5, 499.5, 1.0)))
    c′ = _spline_camera(K′, Rect2(0.0, 0.0, 1000.0, 1000.0), CAM.lens.backward.controls)
    C = [SVector((S.controls[1, i, j] - 499.5) / K[1, 1], (S.controls[2, i, j] - 499.5) / K[2, 2]) for i in axes(S.controls, 2), j in axes(S.controls, 3)]
    CameraModel(K′, SplineLens(c′.lens.backward; forward = ChartSpline(Rect2(S.area[1], S.area[2], S.area[3], S.area[4]), C)))
end
const NC = let b = CAM.lens.backward
    NoncentralSplineModel(CAM.intrinsics, b, ChartSpline(b.area, map(d -> 0.01 * d, b.controls)))
end
const LATTICE = [SVector(G.pixels[1, i], G.pixels[2, i]) for i in 1:G.n_lattice]
const RAYS = [SVector(G.central.backproject[1, i], G.central.backproject[2, i], G.central.backproject[3, i]) for i in 1:G.n_lattice]
const NC_X = [(r = backproject(NC, u); SVector{3, Float64}(r.origin) + 1500.0 * SVector{3, Float64}(r.direction)) for u in LATTICE]
const BPIX = [SVector(x, y) for y in range(0.0, 999.0; length = 100) for x in range(0.0, 999.0; length = 100)]
const BRAYS = [backproject(CAM, u) for u in BPIX]
const BSEEDS = [u + SVector(2.0, -1.5) for u in BPIX]
const CHECKER = [Float64(isodd(fld(x, 8) + fld(y, 8))) for y in 0:999, x in 0:999]
const WIN = Rect2(436.0, 436.0, 128.0, 128.0)
const CPU = KA.CPU()

sweep(f, xs) = (s = 0.0; for x in xs; s += f(x)[1]; end; s)
const KERNELS = [
    "backproject" => () -> sweep(u -> backproject(CAM, u), LATTICE),
    "backproject-batch" => () -> backproject(CAM, BPIX; device = CPU),
    "backproject-jacobian" => () -> sweep(u -> first(backproject_with_jacobian(CAM, u)), LATTICE),
    "project-lm" => () -> sweep(X -> project(CAM, X), RAYS),
    "project-forward" => () -> sweep(X -> project(CAM_FWD, X), RAYS),
    "project-batch" => () -> project(CAM, BRAYS, BSEEDS; device = CPU),
    "noncentral-project" => () -> sweep(X -> project(NC, X), NC_X),
    "undistort-128" => () -> undistort(CAM, CHECKER, WIN; device = CPU),
]

write_bench_record(OUT, "test/spline_timing/bench.jl", SWEEPS, REPS, [name => kernel_arm(f, SWEEPS, REPS) for (name, f) in KERNELS])
