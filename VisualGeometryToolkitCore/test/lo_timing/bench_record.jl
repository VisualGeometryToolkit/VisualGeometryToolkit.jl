# =============================================================================
# lo_timing/bench_record.jl — the record every timing bench writes and the gate
# (gate.jl) reads: the host, the calibration loop's spread, and per arm its µs
# per call over the reps and its bytes per call. Included by each bench —
# VisualGeometryToolkitCore's lo_timing/ and spline_timing/, and
# VisualGeometryToolkitEstimation's lo_timing/ (from `pkgdir(VisualGeometryToolkitCore)`).
# =============================================================================

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
