# =============================================================================
# VisualGeometryToolkitEval test suite — ReTestItems
#
# Every `*_tests.jl` file under test/ is discovered automatically and is ONE
# `@testitem` named after its file, evaluated in its own module on a pool of
# worker processes. Items share no bindings and run in no fixed order, so an
# item imports and defines everything it uses; the fixtures two items share
# are `@testsetup` modules in `*_testsetup.jl`. The "suite_layout" item fails
# on any other .jl file here, which would otherwise never run.
#
#   Pkg.test()                                        # every item, in parallel
#   Pkg.test(; test_args = ["pool_io|tuning"])        # items whose name matches
#   RETESTITEMS_NWORKERS=0 → serial, in-process       # for debugging
#
# Default: one worker per two logical CPUs, capped at 8.
# =============================================================================

using ReTestItems
using VisualGeometryToolkitEval

runtests(
    VisualGeometryToolkitEval;
    nworkers = parse(Int, get(ENV, "RETESTITEMS_NWORKERS", string(clamp(Sys.CPU_THREADS ÷ 2, 1, 8)))),
    name = isempty(ARGS) ? nothing : Regex(join(ARGS, "|")),
)
