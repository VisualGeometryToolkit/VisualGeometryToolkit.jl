# =============================================================================
# Shared Interface Declarations
# =============================================================================

# `data_size(problem)` is owned by VisualGeometryToolkitCore's problem protocol
# (imported in the master module). AbstractRansacProblem and AbstractRobustProblem subtypes
# implement it; the function is shared so scoring code can call it generically.

# A non-finite cost is +Inf to the LM, as VGC's chart LM reads its cost
# (lm_core.jl `_finite_cost`): NLLSsolver accepts a trial at a step below
# `xtol` whatever its cost, and its per-iteration check `cost == cost_` is false
# for NaN — a candidate step past a wall (behind the camera, outside a lens's
# domain) would throw there. +Inf compares equal and is rejected like NaN
# everywhere else. The NLLSsolver fits of this package route each residual
# block's cost through it (`NLLSsolver.computecost(vars, block)`).
_finite_cost(c::Float64) = isfinite(c) ? c : Inf
