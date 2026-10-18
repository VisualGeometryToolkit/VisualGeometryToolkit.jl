# =============================================================================
# VisualGeometryToolkitCore test suite — ReTestItems
# =============================================================================
#
# Every `*_tests.jl` file under test/ is discovered automatically. Each holds
# `@testitem`s, and each item is evaluated in its own module on a pool of
# worker processes: items share no bindings and run in no fixed order, so an
# item imports and defines everything it uses. Files without the suffix
# (gte_parity/) are never discovered — an item that needs one
# includes it.
#
#   Pkg.test()                                   # every item, in parallel
#   Pkg.test(; test_args = ["lens|rig"])         # items whose name matches
#   RETESTITEMS_NWORKERS=0 → serial, in-process  # for debugging
#
# Default: one worker per two logical CPUs (each worker runs 2 threads).
#
# Workers start from `Base.julia_cmd()`, so `julia_args = ["--depwarn=error"]`,
# `--check-bounds` and coverage flags all reach them.
# =============================================================================

using ReTestItems
using VisualGeometryToolkitCore

runtests(
    VisualGeometryToolkitCore;
    nworkers = parse(Int, get(ENV, "RETESTITEMS_NWORKERS", string(max(1, Sys.CPU_THREADS ÷ 2)))),
    name = isempty(ARGS) ? nothing : Regex(join(ARGS, "|")),
)
