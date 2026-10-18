@testitem "lo_timing_gate" begin
# The LO timing regression gate on the kernels every local optimization of the
# stack spends its time in — the E/F Sampson rows and Jacobian rows, the HQ and
# GNC lift accumulations (test/lo_timing/bench.jl). Opt-in; policy, modes and
# robustness in test/lo_timing/gate.jl, the stack's one gate (VisualGeometryToolkitEstimation
# and LocalOptimizationDoneRight run it on their LO arms):
#
#   LO_TIMING_GATE=1      julia --project -e 'using Pkg; Pkg.test(test_args = ["lo_timing_gate"])'
#   LO_TIMING_GATE=record (same) — rewrites test/lo_timing/baselines.toml
import VisualGeometryToolkitCore
include(joinpath(pkgdir(VisualGeometryToolkitCore), "test", "lo_timing", "gate.jl"))
run_lo_timing_gate(joinpath(pkgdir(VisualGeometryToolkitCore), "test", "lo_timing"))
end
