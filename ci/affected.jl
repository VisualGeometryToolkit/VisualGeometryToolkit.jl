# The packages a change affects, with their dependents in the workspace.
#
#   julia --startup-file=no ci/affected.jl <base> [<head>]     # → JSON list on stdout
#
# A path under <Package>/ affects that package; anything else (root files,
# ci/, .github/, .githooks/) affects every package. The closure follows the
# workspace's own dependency edges: [deps], [weakdeps] and the test and docs
# environments of each package, so a VisualGeometryToolkitCore change runs all six and
# a VisualGeometryToolkitEstimation change runs it and VisualGeometryToolkitEval.
# In GitHub Actions the list is also written to $GITHUB_OUTPUT as `packages`.
#
# The implementation is VisualGeometryToolkitCore's `stack_ci` (the `affected`
# command; `affected_packages`), read by path: standard libraries only, so the
# planning job needs no instantiated environment.

include(joinpath(dirname(@__DIR__), "VisualGeometryToolkitCore", "src", "quality", "stack_ci.jl"))

abspath(PROGRAM_FILE) == abspath(@__FILE__()) && exit(stack_ci(["affected"; ARGS]; root = dirname(@__DIR__)))
