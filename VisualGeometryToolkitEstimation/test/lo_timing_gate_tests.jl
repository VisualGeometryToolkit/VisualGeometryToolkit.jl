@testitem "lo_timing_gate" begin
# The LO timing regression gate on the Table 1 arms VisualGeometryToolkitEstimation owns
# (test/lo_timing/bench.jl). Opt-in; policy, modes and robustness in
# VisualGeometryToolkitCore's test/lo_timing/gate.jl (the stack's one gate):
#
#   LO_TIMING_GATE=1      julia --project -e 'using Pkg; Pkg.test(test_args = ["lo_timing_gate"])'
#   LO_TIMING_GATE=record (same) — rewrites test/lo_timing/baselines.toml
import VisualGeometryToolkitEstimation, VisualGeometryToolkitCore
gate = joinpath(pkgdir(VisualGeometryToolkitCore), "test", "lo_timing", "gate.jl")
if isfile(gate)
    include(gate)
    Base.invokelatest(run_lo_timing_gate, joinpath(pkgdir(VisualGeometryToolkitEstimation), "test", "lo_timing"))
elseif get(ENV, "LO_TIMING_GATE", "") in ("1", "record")
    @test_skip "the resolved VisualGeometryToolkitCore ships no test/lo_timing/gate.jl (older than the gate)"
end
end
