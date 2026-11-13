# =============================================================================
# lo_timing/gate.jl — THE LO timing regression gate (opt-in: LO_TIMING_GATE=1),
# one policy for the whole stack: VisualGeometryToolkitCore's test/lo_timing_gate_tests.jl
# (the LO row kernels and lift accumulations), VisualGeometryToolkitEstimation's (its
# Table 1 arms) and LocalOptimizationDoneRight's test/test_lo_timing_gate.jl
# (the external arms) include this file from `pkgdir(VisualGeometryToolkitCore)`, each
# package timing what it owns with its own `bench.jl`.
#
# A bench (`DIR/bench.jl <out.toml> <reps>`) times its arms on a fixed recorded
# call set in a fresh single-threaded process pinned to one idle core, and
# writes per arm the min µs per call over the reps, every rep, and the bytes
# allocated per call after warm-up, plus the host and a calibration loop's
# spread. The gate compares each arm with `DIR/baselines.toml`:
#
#   * TIME: the arm's min µs per call must be ≤ 1.10 × baseline.
#   * ALLOCATION: bytes per call must be ≤ max(1.10 × baseline, baseline + 1 KiB)
#     — allocation is deterministic, so this catches a new per-call allocation.
#
# AN INTENTIONAL SLOWDOWN (or a speed-up worth locking in) re-records the
# baselines IN THE SAME COMMIT: `LO_TIMING_GATE=record` rewrites
# `DIR/baselines.toml` from this host's observation.
#
# ROBUST ON A BUSY HOST: the bench runs under `taskset` on the least-loaded core
# whose SMT siblings are idle too (sampled from /proc/stat over 1 s;
# LO_TIMING_CORE overrides) of the baseline's CLASS on a hybrid CPU (a
# performance core is 1.5–2× an efficiency core: a recording takes a
# performance core and writes `core_class`, a check runs on that class only and
# skips loudly when none is idle), takes the min of 7 reps, and a run is accepted only
# when it is quiet — the calibration loop's spread < 2 % and every arm's two
# fastest reps within 3 %. Up to three runs; an arm over its bound is
# re-measured once in a fresh process before it fails. When no quiet run is
# possible the gate does NOT succeed: it records a skipped (broken) test with the
# reason. It also skips, loudly, on another CPU model or Julia version than the
# baselines were recorded on (re-record there), off Linux, and under coverage;
# the bench process always runs --check-bounds=auto (Pkg.test's
# --check-bounds=yes would time bounds checks).
# =============================================================================

using Test, TOML

const _GATE_TIME_SLACK = 1.10
const _GATE_QUIET_CALIBRATION = 0.02
const _GATE_QUIET_ARM = 0.03

# Per-core (busy, total) jiffies from /proc/stat.
function _gate_cpu_jiffies()
    lines = filter(l -> occursin(r"^cpu\d+ ", l), readlines("/proc/stat"))
    return Dict(map(lines) do l
        f = split(l); v = parse.(Int, f[2:end]); idle = v[4] + v[5]
        parse(Int, f[1][4:end]) => (sum(v) - idle, sum(v))
    end)
end
# The logical CPUs sharing core `c` (its SMT siblings, `c` included), from sysfs.
function _gate_siblings(c)
    s = _gate_cpu_list("/sys/devices/system/cpu/cpu$c/topology/thread_siblings_list")
    return isempty(s) ? [c] : s
end

# A sysfs CPU list ("0-15,20") as integers; empty when the file is absent.
function _gate_cpu_list(p)
    isfile(p) || return Int[]
    out = Int[]
    for part in split(strip(read(p, String)), ',')
        isempty(part) && continue
        lo_hi = parse.(Int, split(part, '-'))
        append!(out, first(lo_hi):last(lo_hi))
    end
    return out
end
# The class of core `c` on a hybrid CPU — "performance" (Intel's `cpu_core`) or
# "efficiency" (`cpu_atom`), whose speeds differ by 1.5–2× — and "uniform" on a
# CPU with one kind of core. A baseline is valid only on cores of its own class.
function _gate_core_class(c)
    c in _gate_cpu_list("/sys/devices/cpu_core/cpus") && return "performance"
    c in _gate_cpu_list("/sys/devices/cpu_atom/cpus") && return "efficiency"
    return "uniform"
end
# The class a recording takes: the performance cores of a hybrid CPU.
_gate_record_class() = isempty(_gate_cpu_list("/sys/devices/cpu_core/cpus")) ? "uniform" : "performance"

# The least-loaded core of class `class` (any class for `nothing`: a baseline
# recorded before the class was) whose SMT siblings are idle too (busy fraction
# over 1 s), or `nothing` when every such core is busier than 5 %.
function _gate_pick_core(class::Union{Nothing, AbstractString})
    haskey(ENV, "LO_TIMING_CORE") && return parse(Int, ENV["LO_TIMING_CORE"])
    a = _gate_cpu_jiffies(); sleep(1.0); b = _gate_cpu_jiffies()
    util = Dict(c => (b[c][1] - a[c][1]) / max(1, b[c][2] - a[c][2]) for c in keys(a) if haskey(b, c))
    load(c) = maximum(get(util, s, 1.0) for s in _gate_siblings(c))
    candidates = filter(c -> c != 0 && (class === nothing || _gate_core_class(c) == class), collect(keys(util)))
    isempty(candidates) && return nothing
    best = argmin(load, candidates)
    return load(best) < 0.05 ? best : nothing
end

function _gate_bench_once(core, DIR)
    out = tempname() * ".toml"
    cmd = `taskset -c $core $(Base.julia_cmd()) --check-bounds=auto -t 1 --startup-file=no
           --project=$(Base.active_project()) $(joinpath(DIR, "bench.jl")) $out 7`
    run(pipeline(cmd; stdout = devnull, stderr = stderr))
    return TOML.parsefile(out)
end
_gate_quiet(m) = m["host"]["calibration_spread"] < _GATE_QUIET_CALIBRATION &&
    all(a["reps_us_per_call"][2] / a["reps_us_per_call"][1] - 1 < _GATE_QUIET_ARM for a in values(m["arms"]))

# The first quiet observation within three runs, or `nothing`.
function _gate_quiet_observation(core, DIR)
    for _ in 1:3
        m = _gate_bench_once(core, DIR)
        _gate_quiet(m) && return m
    end
    return nothing
end

_gate_skip(reason) = (@warn "lo_timing_gate SKIPPED: $reason"; @test_skip reason)

"""
    run_lo_timing_gate(DIR)

The gate on the bench in `DIR` (`DIR/bench.jl`, its baselines `DIR/baselines.toml`):
the policy in this file's header, as `@test`s in the caller's testset.
"""
function run_lo_timing_gate(DIR::AbstractString)
    MODE = get(ENV, "LO_TIMING_GATE", "")
    BASELINES = joinpath(DIR, "baselines.toml")
    if MODE ∉ ("1", "record")
        @info "lo_timing_gate: opt-in, not run (LO_TIMING_GATE=1 to gate, =record to re-record the baselines)"
    elseif !Sys.islinux() || Sys.which("taskset") === nothing
        _gate_skip("needs Linux and taskset to pin the bench to one core")
    elseif Base.JLOptions().code_coverage != 0
        _gate_skip("coverage instrumentation distorts timing")
    else
        base = isfile(BASELINES) ? TOML.parsefile(BASELINES) : nothing
        host = (cpu = Sys.cpu_info()[1].model, julia = string(VERSION))
        # the class of core the timing runs on: the baseline's own (any, for a baseline
        # recorded before the class was), the performance cores for a recording
        class = MODE == "record" || base === nothing ? _gate_record_class() : get(base["host"], "core_class", nothing)
        core = _gate_pick_core(class)
        if core === nothing
            _gate_skip("no idle $(something(class, "")) core (every one > 5 % busy over 1 s, or none of that class) — the machine is too busy to time")
        elseif MODE == "record"
            m = _gate_quiet_observation(core, DIR)
            if m === nothing
                _gate_skip("could not get a quiet observation on core $core to record")
            else
                io = IOBuffer()
                println(io, "# LO timing baselines (the gate: VisualGeometryToolkitCore test/lo_timing/gate.jl): min µs per call over 7 reps")
                println(io, "# on one pinned core, and bytes allocated per call. Re-record with")
                println(io, "# LO_TIMING_GATE=record IN THE COMMIT that changes the speed on purpose.")
                println(io, "[host]")
                println(io, "cpu = ", repr(host.cpu)); println(io, "julia = ", repr(host.julia))
                println(io, "core = ", core)
                println(io, "core_class = ", repr(_gate_core_class(core)))
                for name in sort!(collect(keys(m["arms"])))
                    a = m["arms"][name]
                    println(io, "\n[arms.", repr(name), "]")
                    println(io, "us_per_call = ", round(a["us_per_call"]; digits = 1))
                    println(io, "bytes_per_call = ", round(a["bytes_per_call"]; digits = 1))
                end
                write(BASELINES, take!(io))
                @info "lo_timing_gate: baselines re-recorded on core $core" BASELINES
                @test true
            end
        elseif base === nothing
            _gate_skip("no baselines at $BASELINES — record them with LO_TIMING_GATE=record")
        elseif (base["host"]["cpu"], base["host"]["julia"]) != (host.cpu, host.julia)
            _gate_skip("baselines were recorded on $(base["host"]["cpu"]) / Julia $(base["host"]["julia"]); " *
                 "this host is $(host.cpu) / Julia $(host.julia) — re-record with LO_TIMING_GATE=record")
        else
            m = _gate_quiet_observation(core, DIR)
            if m === nothing
                _gate_skip("three runs on core $core were all noisy (calibration spread ≥ $_GATE_QUIET_CALIBRATION or an arm's " *
                     "two fastest reps ≥ $_GATE_QUIET_ARM apart) — the machine is too noisy to time")
            else
                over(m, name) = m["arms"][name]["us_per_call"] > _GATE_TIME_SLACK * base["arms"][name]["us_per_call"]
                names = sort!(collect(keys(base["arms"])))
                if any(n -> haskey(m["arms"], n) && over(m, n), names)    # re-measure once before failing
                    m2 = _gate_quiet_observation(core, DIR)
                    if m2 !== nothing
                        for n in names
                            m["arms"][n]["us_per_call"] = min(m["arms"][n]["us_per_call"], m2["arms"][n]["us_per_call"])
                        end
                    end
                end
                @testset "LO timing: $n" for n in names
                    @test haskey(m["arms"], n)
                    if haskey(m["arms"], n)
                        b = base["arms"][n]; a = m["arms"][n]
                        ratio = a["us_per_call"] / b["us_per_call"]
                        ratio > _GATE_TIME_SLACK && @error "LO arm $n is $(round(100 * (ratio - 1); digits = 1)) % slower than its baseline" observed_us = a["us_per_call"] baseline_us = b["us_per_call"]
                        @test ratio <= _GATE_TIME_SLACK
                        @test a["bytes_per_call"] <= max(_GATE_TIME_SLACK * b["bytes_per_call"], b["bytes_per_call"] + 1024)
                    end
                end
            end
        end
    end
    return nothing
end
