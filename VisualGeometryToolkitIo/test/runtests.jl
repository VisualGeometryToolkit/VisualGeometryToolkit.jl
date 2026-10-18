# =============================================================================
# VisualGeometryToolkitIo test suite — ReTestItems
# =============================================================================
#
# Every `*_tests.jl` file under test/ is discovered automatically. Each holds
# `@testitem`s, and each item is evaluated in its own module on a pool of
# worker processes: items share no bindings and run in no fixed order, so an
# item imports and defines everything it uses. The shared fixtures are
# `@testsetup` modules in the `*_testsetup.jl` files — `VGIoFixtures` (the
# prelude: the unexported `load`/`save`, units, lens types, one pose) — which
# an item names in its `setup = [...]` and sees through their exports.
#
#   Pkg.test()                                   # every item, in parallel
#   Pkg.test(; test_args = ["migrate|cameras"])       # items whose name matches
#   RETESTITEMS_NWORKERS=0 → serial, in-process  # for debugging
#
# Default: one worker per two logical CPUs (each worker runs 2 threads).
# The pycolmap item records a SKIPPED check when the pixi `py` environment is
# not installed, on a worker exactly as in-process.
# =============================================================================

using ReTestItems
using VisualGeometryToolkitIo

runtests(
    VisualGeometryToolkitIo;
    nworkers = parse(Int, get(ENV, "RETESTITEMS_NWORKERS", string(max(1, Sys.CPU_THREADS ÷ 2)))),
    name = isempty(ARGS) ? nothing : Regex(join(ARGS, "|")),
)
