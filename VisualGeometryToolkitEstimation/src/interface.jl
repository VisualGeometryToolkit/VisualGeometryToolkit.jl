# =============================================================================
# Shared Interface Declarations
# =============================================================================

# `data_size(problem)` is owned by VisualGeometryToolkitCore's problem protocol
# (imported in the master module). AbstractRansacProblem and AbstractRobustProblem subtypes
# implement it; the function is shared so scoring code can call it generically.
