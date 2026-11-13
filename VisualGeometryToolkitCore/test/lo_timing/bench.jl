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
#                        their Jacobian along the 5-dof chart (`weighted_whitened_residuals!`,
#                        `weighted_whitened_jacobian_rows!` over a set with M₁/M₂ columns)
#   F-rows, F-jacobian   the image-plane Sampson rows of F, 7-dof chart
# (The lift accumulations moved with the lifts to LocalOptimizationDoneRight,
# whose own timing gate runs them.)
# THE SCENE: 1000 correspondences of a fixed pinhole pair (f = 800 px), 30 %
# outliers, seeded — deterministic, so no fixture file (bench_record.jl's
# `essential_bench_scene`, which every LO kernel bench shares).
# =============================================================================

using VisualGeometryToolkitCore
using VisualGeometryToolkitCore: weighted_whitened_residuals!, weighted_whitened_jacobian_rows!, Correspondences, withattrs, retract, retract_directions,
    zero_tangent, backproject_with_jacobian, EssentialManifold5, FundMatManifold7, minimal_manifold
using StaticArrays, LinearAlgebra, Random

const OUT = ARGS[1]
const REPS = length(ARGS) >= 2 ? parse(Int, ARGS[2]) : 7
const SWEEPS = 200            # sweeps per rep (one "call" = one sweep)
include(joinpath(@__DIR__, "bench_record.jl"))   # calibration_spread, kernel_arm, write_bench_record: the record the gate reads

Threads.nthreads() == 1 || error("bench.jl: run single-threaded (julia -t 1); got $(Threads.nthreads()) threads")

const N = 1000
const SCENE = essential_bench_scene(Xoshiro(20261007); n = N)   # THE SCENE (bench_record.jl)
const U1, U2, K, R, T = SCENE.u₁, SCENE.u₂, SCENE.K, SCENE.R, SCENE.t
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

const XE = withattrs(Correspondences(F1, F2); M₁ = M1, M₂ = M2, sqrt_weights = SQRTW)
const XF = withattrs(Correspondences(U1, U2); sqrt_weights = SQRTW)
const KERNELS = [
    "E-rows"     => () -> weighted_whitened_residuals!(r, EE, XE),
    "E-jacobian" => () -> weighted_whitened_jacobian_rows!(JE, EE, dE, XE),
    "F-rows"     => () -> weighted_whitened_residuals!(r, FF, XF),
    "F-jacobian" => () -> weighted_whitened_jacobian_rows!(JF, FF, dF, XF),
]

write_bench_record(OUT, "test/lo_timing/bench.jl", SWEEPS, REPS, [name => kernel_arm(f, SWEEPS, REPS) for (name, f) in KERNELS])
