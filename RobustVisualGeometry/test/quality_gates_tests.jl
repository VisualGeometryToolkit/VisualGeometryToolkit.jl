@testitem "quality_gates" begin
# =============================================================================
# Quality gates — Aqua, ExplicitImports, JET (VisualGeometryCore's
# `test_quality_gates`, the stack's one statement of them)
# =============================================================================
#
# These are RATCHETS, not aspirations: every bound below is the value the
# package actually meets today, and each may only ever move DOWN. If a change
# trips one of these, the fix is the code, not the bound.
#
# Baselines (2026-10-05; first set 2026-09-03): Aqua clean (no piracy
# exemptions), 0 ambiguities involving an RVG method, 0 implicit imports, 0 stale explicit imports (no exemptions),
# 0 JET reports.
#
# `jet_bound = 0`: the last known report — `refine!(ws, ::LineFittingProblem,
# …)` union-splitting on a free struct parameter — went with the 2026-09-30
# cleanup. Never raise it.
# =============================================================================

using Test, Aqua, ExplicitImports, JET
using VisualGeometryCore
using RobustVisualGeometry

VisualGeometryCore.test_quality_gates(
    RobustVisualGeometry;
    jet_bound = 0,
)
end
