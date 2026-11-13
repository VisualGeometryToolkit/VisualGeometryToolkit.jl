# =============================================================================
# lo_timing/bench.jl — the LO timing benchmark the timing gate runs
# (test/lo_timing_gate_tests.jl). A standalone script, run in its own
# single-threaded process pinned to one core by the gate:
#
#   taskset -c <core> julia -t 1 --project=<test env> test/lo_timing/bench.jl <out.toml> [reps]
#
# THE CALL SET: `lo_calls_phototourism_e.bin`, 24 recorded local optimizations
# of the ACCV Table 1 timing protocol — 8 PhotoTourism british_museum pairs
# (346–1094 correspondences, the pool frame of VisualGeometryToolkitEval's PT-E-SIFT
# pool) × 3 RANSAC LO-checkpoint seeds each (the first, middle and last
# checkpoint of the Welsch-IRLS run of record), the camera matrices and the
# selection's outlier half-width (the essential class's former 20 px default,
# in the pool frame — so it carries the fixture's pixel unit). Little-endian
# binary, written once:
#
#   "LOFX0001" | npairs::Int64 | nseeds::Int64 | outlier_halfwidth::Float64 |
#   per pair: n::Int64, Kc1 (9, column-major), Kc2 (9), x1 (2n), x2 (2n),
#             nseeds seeds E (9 each, column-major)
#
# THE ARMS: the Table 1 LO strategies VisualGeometryToolkitEstimation owns, at the paper's
# equal LO budget (25 model updates) — the static Welsch and GM IRLS. (The HQ
# lifts, GNC, the Student-t + uniform IRLS and BCD, and the external TLS
# refits are timed by LocalOptimizationDoneRight's gates, which own them.)
#
# OUTPUT: per arm, the min and every rep's µs per call (each rep = the whole call
# set, `time_ns` around each call), and the bytes allocated per call after
# warm-up; plus the host and a calibration loop's spread (the noise probe).
# =============================================================================

import VisualGeometryToolkitEstimation as VgtkEstimation
using VisualGeometryToolkitEstimation, VisualGeometryToolkitCore
using StaticArrays, LinearAlgebra
using VisualGeometryToolkitEstimation: RansacWorkspace
using VisualGeometryToolkitCore: MsacScoring

const OUT = ARGS[1]
const REPS = length(ARGS) >= 2 ? parse(Int, ARGS[2]) : 7
include(joinpath(pkgdir(VisualGeometryToolkitCore), "test", "lo_timing", "bench_record.jl"))   # calibration_spread, write_bench_record: the record the gate reads

Threads.nthreads() == 1 || error("bench.jl: run single-threaded (julia -t 1); got $(Threads.nthreads()) threads")

include(joinpath(@__DIR__, "call_set.jl"))   # read_fixture, pool_correspondences, problem_of: the call set's one reader

# The Table 1 arms at the equal LO budget. The efficiency constants are the Welsch
# grid's centre (GM at the plain multiple whose half-weight depth is the Welsch
# point's, 1.5·√log2 — decision_width(GM, 1.5·√log2), stored at full precision
# when the half-weight currency was retired) — the width barely moves per-call time.
const BUDGET = 25
const GM_MULTIPLE = 1.9404025301632923
const ARMS = [
    "Welsch-IRLS" => () -> Irls{WelschShape}(; max_iter = BUDGET, efficiency_constant = 1.5),
    "GM-IRLS"     => () -> Irls{GemanMcclureShape}(; max_iter = BUDGET, efficiency_constant = GM_MULTIPLE),
]

const PAIRS, HALFWIDTH = read_fixture(CALL_SET_FIXTURE)
const PROBS = [problem_of(pr) for pr in PAIRS]
# The core's default scorer: MSAC at the essential class's threshold of record
# (configs/default.toml [pipelines.essential], 1 px), in the fixture's frame —
# the pool frame, where one pixel is the recorded half-width over the 20 px it
# was recorded at.
const POOL_PX = HALFWIDTH / 20.0
const THRESHOLD = VgtkEstimation._dflt("pipelines.essential", "scoring")["threshold"] * POOL_PX   # its scorer table (since 0.10)
const SCORINGS = [MsacScoring(EssentialProblem; threshold = THRESHOLD) for _ in PROBS]
const WORKSPACES = [RansacWorkspace(PROBS[i], SCORINGS[i]) for i in eachindex(PROBS)]
const NCALLS = call_set_size(PAIRS)


# The noise probe: a fixed arithmetic loop timed REPS times; its relative spread
# (second-fastest over fastest − 1) is the machine's jitter on this core.

# each arm compiled and its task-local buffers warmed (two runs) before it is measured: call_set.jl's lo_call_set_arm
write_bench_record(OUT, "test/lo_timing/bench.jl", NCALLS, REPS,
                   [name => lo_call_set_arm(ctor, PROBS, SCORINGS, WORKSPACES, PAIRS, REPS) for (name, ctor) in ARMS])
