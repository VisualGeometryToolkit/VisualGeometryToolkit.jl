@testitem "spline_timing_gate" begin
# The timing regression gate on the spline cameras' CPU hot paths — scalar and
# batched backprojection, the backward lens's Jacobian, the LM and forward
# projections, the batched and non-central projections, the exact undistortion
# (test/spline_timing/bench.jl, plan 1b Task 4b). Opt-in; policy, modes and
# robustness in test/lo_timing/gate.jl, the stack's one gate:
#
#   LO_TIMING_GATE=1      julia --project -e 'using Pkg; Pkg.test(test_args = ["spline_timing_gate"])'
#   LO_TIMING_GATE=record (same) — rewrites test/spline_timing/baselines.toml
import VisualGeometryToolkitCore
include(joinpath(pkgdir(VisualGeometryToolkitCore), "test", "lo_timing", "gate.jl"))
run_lo_timing_gate(joinpath(pkgdir(VisualGeometryToolkitCore), "test", "spline_timing"))
end
