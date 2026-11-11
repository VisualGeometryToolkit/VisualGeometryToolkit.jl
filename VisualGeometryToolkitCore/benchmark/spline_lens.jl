# The SplineLens hot paths (plan 1b, Task 4b), on the golden camera of
# test/fixtures/spline_lens_golden.jl (KB fisheye f = 300 px, 1000 × 1000, a 24 × 24
# lattice), on the host and, where CUDA is functional, on the GPU:
#   julia -t 2 --project=VisualGeometryToolkitCore/benchmark VisualGeometryToolkitCore/benchmark/spline_lens.jl [out.toml]
#
# BenchmarkTools minimum; per-call µs for a scalar path (the mean over its point set),
# ms for a batch or an image. The OLD grid's numbers (its central and non-central grid
# models and its fitted forward map, at BASE 004a004c, the same paths on the same
# machine and thread count, its own script) are recorded below; the timing gate on the
# CPU paths is test/spline_timing (LO_TIMING_GATE's mechanism).
#
# RESULTS — i9-13900K, RTX 4090, `julia -t 2`, Julia 1.13.3, every column (2026-11-10):
#   OLD = the old grid at BASE 004a004c; 4 = Task 4's head 02dfb2ed; 4b = this file's head.
#   path                                     |  OLD     |  4       |  4b      | 4b / OLD
#   cpu backproject, scalar (µs/px)          |  0.0307  |  0.0394  |  0.0308  | 1.004
#   cpu backproject, batch 1e5 (ms)          |  0.978   |  2.06    |  1.005   | 1.03
#   cpu backproject_with_jacobian (µs/px)    |  0.0500  |  0.0350  |  0.0286  | 0.57
#   cpu project, LM (µs/ray)                 |  0.637   |  0.695   |  0.374   | 0.59
#   cpu project, forward spline (µs/ray)     |  0.0647  |  0.120   |  0.0575  | 0.89
#   cpu project, batch 1e5 (ms)              |  17.1    |  19.6    |  13.5    | 0.79
#   cpu non-central project (µs/X)           |  1.023   |  0.928   |  0.649   | 0.63
#   cpu undistort 1000², LM lens (ms)        |  10.1 ⁽ˢ⁾ |  347     |  170     | exact vs sampled
#   cpu undistort 1000², forward lens (ms)   |  —       |  67.9    |  32.5    |
#   cpu parametric undistort! 1000² (ms)     |  18.9    |  17.7    |  17.7    | 0.94
#   cpu parametric batch 1024 × 64² (ms)     |  77.6    |  77.8    |  77.8    | 1.00
#   gpu backproject, batch 1e5 (ms)          |  0.0525  |  0.0858  |  0.0512  |
#   gpu project, batch 1e5 (ms)              |  0.460   |  0.489   |  0.360   |
#   gpu undistort 1000², LM lens (ms)        |  fails ⁽ᶠ⁾ |  8.67    |  4.77    |
#   gpu undistort, an Array in (ms)          |          |          |  5.83    | upload + kernel + download
#   gpu undistort 1000², forward lens (ms)   |  —       |  1.20    |  0.736   |
#   gpu parametric undistort! 1000² (ms)     |  0.199   |  0.204   |  0.204   |
#   gpu parametric batch 1024 × 64² (ms)     |  0.760   |  0.771   |  0.772   |
#   ⁽ˢ⁾ the old SAMPLED remap (a coarse inverse, interpolated); every 4b warp is exact per pixel.
#   ⁽ᶠ⁾ the old GPU undistortion does not compile (a non-isbits closure in its coarse phase).
#   The old grid's script is the same paths at BASE through its API (the central and
#   non-central grid models, the forward map from the golden spline); not kept, its numbers are.
using VisualGeometryToolkitCore, StaticArrays, BenchmarkTools, CUDA, TOML
import KernelAbstractions as KA
using LinearAlgebra: norm
using VisualGeometryToolkitCore: CameraModel, SplineLens, ChartSpline, NoncentralSplineModel, KannalaBrandt, px, project,
    backproject, backproject_with_jacobian, undistort, undistort!, image_warp, image_warp!, image_warp_batch_to!, distortion_map,
    calibration_matrix, Rect2, LogicalIntrinsics, CameraCalibrationMat
using VisualGeometryToolkitCore.Cameras: _spline_camera
using CoordinateTransformations: AffineMap

include(joinpath(pkgdir(VisualGeometryToolkitCore), "test", "fixtures", "spline_lens_golden.jl"))
const G = GOLDEN
const W, H = G.image[1], G.image[2]
const SRC = CameraModel(G.f * px, SA[G.pp[1] * px, G.pp[2] * px]; lens = KannalaBrandt(G.kb...))
const CAM = CameraModel(SplineLens, SRC, Size2(W, H); grid_size = (G.grid_size[1], G.grid_size[2]))
# the lens with the golden's fitted forward spline (the test's construction, spline_lens_tests.jl)
const CAM_FWD = let K = calibration_matrix(CAM), S = G.spline
    K′ = LogicalIntrinsics(CameraCalibrationMat(SMatrix{3, 3, Float64}(K[1, 1], 0.0, 0.0, 0.0, K[2, 2], 0.0, 499.5, 499.5, 1.0)))
    c′ = _spline_camera(K′, Rect2(0.0, 0.0, 1000.0, 1000.0), CAM.lens.backward.controls)
    C = [SVector((S.controls[1, i, j] - 499.5) / K[1, 1], (S.controls[2, i, j] - 499.5) / K[2, 2]) for i in axes(S.controls, 2), j in axes(S.controls, 3)]
    CameraModel(K′, SplineLens(c′.lens.backward; forward = ChartSpline(Rect2(S.area[1], S.area[2], S.area[3], S.area[4]), C)))
end
const NC = let b = CAM.lens.backward
    NoncentralSplineModel(CAM.intrinsics, b, ChartSpline(b.area, map(d -> 0.01 * d, b.controls)))
end

# the golden lattice: 17 × 17 pixels over the calibrated area, and their rays
const LATTICE = [SVector(G.pixels[1, i], G.pixels[2, i]) for i in 1:G.n_lattice]
const RAYS = [SVector(G.central.backproject[1, i], G.central.backproject[2, i], G.central.backproject[3, i]) for i in 1:G.n_lattice]
const NC_X = [(r = backproject(NC, u); SVector{3, Float64}(r.origin) + 1500.0 * SVector{3, Float64}(r.direction)) for u in LATTICE]
# a batch: 1e5 pixels over the area, their rays, seeds 2 px off
const BPIX = [SVector(x, y) for y in range(0.0, H - 1.0; length = 316) for x in range(0.0, W - 1.0; length = 316)]
const BRAYS = [backproject(CAM, u) for u in BPIX]
const BSEEDS = [u + SVector(2.0, -1.5) for u in BPIX]
const CHECKER = [Float64(isodd(fld(x, 8) + fld(y, 8))) for y in 0:(H - 1), x in 0:(W - 1)]
const FULL = Rect2(0.0, 0.0, Float64(W), Float64(H))
const CPU = KA.CPU()   # `device` defaults to the GPU once CUDA is loaded: the host rows name the host

us_per(b, n) = minimum(b).time / 1.0e3 / n
ms(b) = minimum(b).time / 1.0e6
sum_over(f, xs) = (s = 0.0; for x in xs; s += f(x)[1]; end; s)

R = Dict{String, Float64}()
R["cpu_backproject_scalar_us"] = us_per(@benchmark(sum_over(u -> backproject($CAM, u), $LATTICE)), length(LATTICE))
R["cpu_backproject_batch_1e5_ms"] = ms(@benchmark(backproject($CAM, $BPIX; device = $CPU)))
R["cpu_project_lm_us"] = us_per(@benchmark(sum_over(X -> project($CAM, X), $RAYS)), length(RAYS))
R["cpu_project_forward_us"] = us_per(@benchmark(sum_over(X -> project($CAM_FWD, X), $RAYS)), length(RAYS))
R["cpu_project_batch_1e5_ms"] = ms(@benchmark(project($CAM, $BRAYS, $BSEEDS; device = $CPU)))
R["cpu_backproject_with_jacobian_us"] = us_per(@benchmark(sum_over(u -> first(backproject_with_jacobian($CAM, u)), $LATTICE)), length(LATTICE))
R["cpu_noncentral_project_us"] = us_per(@benchmark(sum_over(X -> project($NC, X), $NC_X)), length(NC_X))
R["cpu_undistort_1000_ms"] = ms(@benchmark(undistort($CAM, $CHECKER, $FULL; device = $CPU); samples = 5, evals = 1, seconds = 30))
R["cpu_undistort_fwd_1000_ms"] = ms(@benchmark(undistort($CAM_FWD, $CHECKER, $FULL; device = $CPU); samples = 5, evals = 1, seconds = 30))

# the parametric lens through both warp entry points (the launch policy is every warp's)
const PDST = zeros(H, W)
R["cpu_param_undistort_1000_ms"] = ms(@benchmark(undistort!($PDST, $SRC, $CHECKER)))
const PWARPS = [image_warp(distortion_map(SRC) ∘ AffineMap(SMatrix{2, 2, Float64}(1, 0, 0, 1), SVector(100.0 + 12.0 * (k % 64), 100.0 + 12.0 * (k ÷ 64))), 64) for k in 0:1023]
const POUT = zeros(64, 64, length(PWARPS))
R["cpu_param_batch_1024x64_ms"] = ms(@benchmark(image_warp_batch_to!($POUT, $PWARPS, $CHECKER)))

if CUDA.functional()
    dpix = CuArray(BPIX); drays = CuArray(BRAYS); dseeds = CuArray(BSEEDS); dchk = CuArray(CHECKER)
    R["gpu_backproject_batch_1e5_ms"] = ms(@benchmark(CUDA.@sync backproject($CAM, $dpix)))
    R["gpu_project_batch_1e5_ms"] = ms(@benchmark(CUDA.@sync project($CAM, $drays, $dseeds)))
    R["gpu_undistort_1000_ms"] = ms(@benchmark(CUDA.@sync undistort($CAM, $dchk, $FULL)))
    R["gpu_undistort_fwd_1000_ms"] = ms(@benchmark(CUDA.@sync undistort($CAM_FWD, $dchk, $FULL)))
    # an `Array` in, the default device (the GPU): one upload, the kernel, one download
    R["gpu_undistort_1000_from_host_ms"] = ms(@benchmark(undistort($CAM, $CHECKER, $FULL)))
    dpdst = CUDA.zeros(Float64, H, W)
    R["gpu_param_undistort_1000_ms"] = ms(@benchmark(CUDA.@sync undistort!($dpdst, $SRC, $dchk)))
    dwarps = CuArray(PWARPS); dpout = CUDA.zeros(Float64, 64, 64, length(PWARPS))
    R["gpu_param_batch_1024x64_ms"] = ms(@benchmark(CUDA.@sync image_warp_batch_to!($dpout, $dwarps, $dchk)))
end

println("Julia ", VERSION, ", ", Threads.nthreads(), " threads, CUDA ", CUDA.functional())
for k in sort!(collect(keys(R)))
    println(rpad(k, 36), round(R[k]; sigdigits = 4))
end
if !isempty(ARGS)
    open(io -> TOML.print(io, Dict("julia" => string(VERSION), "threads" => Threads.nthreads(), "results" => R)), ARGS[1], "w")
end
