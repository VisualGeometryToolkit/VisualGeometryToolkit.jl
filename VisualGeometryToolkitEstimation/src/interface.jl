# =============================================================================
# Shared Interface Declarations
# =============================================================================

# `data_size(problem)` is owned by VisualGeometryToolkitCore's problem protocol
# (imported in the master module). AbstractRansacProblem and AbstractRobustProblem subtypes
# implement it; the function is shared so scoring code can call it generically.

# A non-finite cost is +Inf to the LM (VGC's chart LM reads its cost the same
# way, lm_core.jl `_finite_cost`): a trial step past a wall — behind the camera,
# outside a lens's domain — makes a residual NaN, and a registered NLLSsolver
# accepts a trial whose cost is not greater than the best, which NaN is not
# (the stack's fork rejects it itself). Read as +Inf, the trial is rejected on
# both, so the two give the same answer. The NLLSsolver fits of this package
# route each residual block's cost through it
# (`NLLSsolver.computecost(vars, block)`).
_finite_cost(c::Float64) = isfinite(c) ? c : Inf
