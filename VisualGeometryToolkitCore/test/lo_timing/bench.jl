# =============================================================================
# lo_timing/bench.jl — the LO kernel timing benchmark VisualGeometryToolkitCore's
# timing gate runs (test/lo_timing_gate_tests.jl; the gate itself is
# test/lo_timing/gate.jl). A standalone script, run in its own single-threaded
# process pinned to one core by the gate:
#
#   taskset -c <core> julia -t 1 --project=<test env> test/lo_timing/bench.jl <out.toml> [reps]
#
# THE KERNELS every local optimization of the stack spends its time in, per
# correspondence (the "µs per call" of a kernel is one sweep over the scene):
#   E-rows, E-jacobian   the tangent-Sampson rows of E on unit bearings, and
#                        their Jacobian along the 5-dof chart (`sampson_rows!`,
#                        `sampson_jacobian_rows!` with M₁/M₂)
#   F-rows, F-jacobian   the image-plane Sampson rows of F, 7-dof chart
# (The lift accumulations moved with the lifts to LocalOptimizationDoneRight,
# whose own timing gate runs them.)
# THE SCENE: 1000 correspondences of a fixed pinhole pair (f = 800 px), 30 %
# outliers, seeded — deterministic, so no fixture file.
# =============================================================================

using VisualGeometryToolkitCore
using VisualGeometryToolkitCore: sampson_rows!, sampson_jacobian_rows!, retract, retract_directions,
    zero_tangent, backproject_with_jacobian, EssentialManifold5, FundMatManifold7, minimal_manifold
using StaticArrays, LinearAlgebra, Random

const OUT = ARGS[1]
const REPS = length(ARGS) >= 2 ? parse(Int, ARGS[2]) : 7
const SWEEPS = 200            # sweeps per rep (one "call" = one sweep)

Threads.nthreads() == 1 || error("bench.jl: run single-threaded (julia -t 1); got $(Threads.nthreads()) threads")

const N = 1000
const RNG = Xoshiro(20261007)
const K = SMatrix{3, 3, Float64, 9}(800, 0, 0, 0, 800, 0, 640, 480, 1)
const R = SMatrix{3, 3, Float64, 9}(exp([0 -0.1 0.05; 0.1 0 -0.02; -0.05 0.02 0]))
const T = normalize(SVector(1.0, 0.1, 0.05))
function scene()
    u1 = Point2{Float64}[]; u2 = Point2{Float64}[]
    while length(u1) < N
        X = SVector(4 * rand(RNG) - 2, 3 * rand(RNG) - 1.5, 4 + 4 * rand(RNG))
        Y = R * X + T
        p1 = K * X; p2 = K * Y
        if rand(RNG) < 0.3
            push!(u1, Point2(1280 * rand(RNG), 960 * rand(RNG))); push!(u2, Point2(1280 * rand(RNG), 960 * rand(RNG)))
        else
            push!(u1, Point2(p1[1] / p1[3], p1[2] / p1[3]) + Point2(0.5 * randn(RNG), 0.5 * randn(RNG)))
            push!(u2, Point2(p2[1] / p2[3], p2[2] / p2[3]) + Point2(0.5 * randn(RNG), 0.5 * randn(RNG)))
        end
    end
    return u1, u2
end
const U1, U2 = scene()
const CAM = CameraModel(K)
const FM1 = [backproject_with_jacobian(CAM, u) for u in U1]
const FM2 = [backproject_with_jacobian(CAM, u) for u in U2]
const F1 = first.(FM1); const F2 = first.(FM2); const M1 = last.(FM1); const M2 = last.(FM2)
const E = EssentialMat(R, T)
const CHE = EssentialManifold5(E)
const FMAT = FundMat(E, K, K)
const CHF = minimal_manifold(FMAT)
const SQRTW = ones(N)

const r = zeros(N)
const JE = zeros(N, 5)
const JF = zeros(N, 7)

const aE = zero_tangent(CHE); const aF = zero_tangent(CHF)
const EE = retract(CHE, aE); const dE = retract_directions(CHE, aE)
const FF = retract(CHF, aF); const dF = retract_directions(CHF, aF)

const KERNELS = [
    "E-rows"     => () -> sampson_rows!(r, EE, F1, F2, SQRTW; M₁ = M1, M₂ = M2),
    "E-jacobian" => () -> sampson_jacobian_rows!(JE, EE, dE, F1, F2, SQRTW; M₁ = M1, M₂ = M2),
    "F-rows"     => () -> sampson_rows!(r, FF, U1, U2, SQRTW),
    "F-jacobian" => () -> sampson_jacobian_rows!(JF, FF, dF, U1, U2, SQRTW),
]

function time_kernel(f)
    t0 = time_ns()
    for _ in 1:SWEEPS
        f()
    end
    return (time_ns() - t0) / SWEEPS / 1.0e3
end

function calibration_spread(reps)
    f() = (s = 0.0; x = 1.0; for k in 1:2_000_000; x = muladd(x, 1.0000001, 1.0e-9); s += x; end; s)
    f(); ts = Float64[]
    for _ in 1:reps
        t0 = time_ns(); v = f(); push!(ts, time_ns() - t0); v > 0 || error()
    end
    sort!(ts)
    return ts[2] / ts[1] - 1
end

io = IOBuffer()
println(io, "# written by test/lo_timing/bench.jl")
println(io, "[host]")
println(io, "cpu = ", repr(Sys.cpu_info()[1].model))
println(io, "julia = ", repr(string(VERSION)))
println(io, "calibration_spread = ", calibration_spread(REPS))
println(io, "ncalls = ", SWEEPS)
for (name, f) in KERNELS
    f(); f()
    bytes = @allocated f()
    us = sort!([time_kernel(f) for _ in 1:REPS])
    println(io, "[arms.", repr(name), "]")
    println(io, "us_per_call = ", us[1])
    println(io, "reps_us_per_call = [", join(us, ", "), "]")
    println(io, "bytes_per_call = ", Float64(bytes))
end
write(OUT, take!(io))
