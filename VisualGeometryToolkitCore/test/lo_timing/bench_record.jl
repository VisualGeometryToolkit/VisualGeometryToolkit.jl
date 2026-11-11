# =============================================================================
# lo_timing/bench_record.jl — the record every timing bench writes and the gate
# (gate.jl) reads: the host, the calibration loop's spread, and per arm its µs
# per call over the reps and its bytes per call; and the seeded E scene the LO
# kernel benches time on (`essential_bench_scene`). Included by each bench —
# VisualGeometryToolkitCore's lo_timing/ and spline_timing/,
# VisualGeometryToolkitEstimation's lo_timing/ and LocalOptimizationDoneRight's
# (from `pkgdir(VisualGeometryToolkitCore)`).
# =============================================================================
using StaticArrays: SMatrix, SVector
using LinearAlgebra: normalize
using VisualGeometryToolkitCore: Point2

# THE SCENE of the LO kernel benches: `n` correspondences of a fixed pinhole pair
# (f = 800 px, principal point (640, 480), view 2 at x₂ = R·x₁ + t), 30 % gross
# outliers uniform in the 1280 × 960 frame, the inliers perturbed by 0.5 px —
# deterministic in `rng`, so no fixture file. Returns `(; u₁, u₂, K, R, t)`.
function essential_bench_scene(rng; n = 1000)
    K = SMatrix{3, 3, Float64, 9}(800, 0, 0, 0, 800, 0, 640, 480, 1)
    R = SMatrix{3, 3, Float64, 9}(exp([0 -0.1 0.05; 0.1 0 -0.02; -0.05 0.02 0]))
    t = normalize(SVector(1.0, 0.1, 0.05))
    u₁ = Point2{Float64}[]; u₂ = Point2{Float64}[]
    while length(u₁) < n
        X = SVector(4 * rand(rng) - 2, 3 * rand(rng) - 1.5, 4 + 4 * rand(rng))
        Y = R * X + t
        p1 = K * X; p2 = K * Y
        if rand(rng) < 0.3
            push!(u₁, Point2(1280 * rand(rng), 960 * rand(rng))); push!(u₂, Point2(1280 * rand(rng), 960 * rand(rng)))
        else
            push!(u₁, Point2(p1[1] / p1[3], p1[2] / p1[3]) + Point2(0.5 * randn(rng), 0.5 * randn(rng)))
            push!(u₂, Point2(p2[1] / p2[3], p2[2] / p2[3]) + Point2(0.5 * randn(rng), 0.5 * randn(rng)))
        end
    end
    return (; u₁, u₂, K, R, t)
end

# How far the host's second-fastest run of a fixed arithmetic loop is above its
# fastest, over `reps` runs: the gate's quietness test of the host.
function calibration_spread(reps)
    f() = (s = 0.0; x = 1.0; for k in 1:2_000_000; x = muladd(x, 1.0000001, 1.0e-9); s += x; end; s)
    f(); ts = Float64[]
    for _ in 1:reps
        t0 = time_ns(); v = f(); push!(ts, time_ns() - t0); v > 0 || error()
    end
    sort!(ts)
    return ts[2] / ts[1] - 1
end

# A kernel's µs per call: `sweeps` calls of `f`, timed as one.
function sweep_time(f, sweeps)
    t0 = time_ns()
    for _ in 1:sweeps
        f()
    end
    return (time_ns() - t0) / sweeps / 1.0e3
end

# A kernel arm: warmed twice, its allocation per call, then `reps` sweep times, sorted.
kernel_arm(f, sweeps, reps) = () -> (f(); f(); bytes = @allocated(f()); (sort!([sweep_time(f, sweeps) for _ in 1:reps]), Float64(bytes)))

# Writes the record to `out`: `source` (the bench's path), the host, `ncalls`
# (the calls one timing covers) and each `name => measure` arm, `measure()`
# returning its sorted µs per call over the reps and its bytes per call. Each
# arm is measured as it is written, after the host's calibration loop.
function write_bench_record(out, source, ncalls, reps, arms)
    io = IOBuffer()
    println(io, "# written by ", source)
    println(io, "[host]")
    println(io, "cpu = ", repr(Sys.cpu_info()[1].model))
    println(io, "julia = ", repr(string(VERSION)))
    println(io, "calibration_spread = ", calibration_spread(reps))
    println(io, "ncalls = ", ncalls)
    for (name, measure) in arms
        us, bytes = measure()
        println(io, "[arms.", repr(name), "]")
        println(io, "us_per_call = ", us[1])
        println(io, "reps_us_per_call = [", join(us, ", "), "]")
        println(io, "bytes_per_call = ", bytes)
    end
    return write(out, take!(io))
end
