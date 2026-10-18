# =============================================================================
# VisualGeometryToolkitEstimation test suite — ReTestItems
# =============================================================================
#
# The stack's ReTestItems conventions (discovery of every `*_tests.jl` item and
# `*_testsetups.jl` setup, name filtering through `test_args`,
# `RETESTITEMS_NWORKERS=0` for a serial in-process run, and flags such as
# `--depwarn=error` reaching the workers) are stated once, in
# VisualGeometryToolkitCore's test/runtests.jl.
#
# RVG's one difference — workers CAPPED AT 8: discovery finds ~35 items, and
# every worker is a fresh process that loads the whole dep tree, so past ~8
# the extra process images buy nothing (wall time is floored by the slowest
# single ITEM, not by the worker count).
# =============================================================================

using ReTestItems
using VisualGeometryToolkitEstimation

runtests(
    VisualGeometryToolkitEstimation;
    nworkers = parse(Int, get(ENV, "RETESTITEMS_NWORKERS", string(clamp(Sys.CPU_THREADS ÷ 2, 1, 8)))),
    name = isempty(ARGS) ? nothing : Regex(join(ARGS, "|")),
)
