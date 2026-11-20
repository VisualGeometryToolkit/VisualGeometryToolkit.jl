# Point a downstream project at this workspace's packages.
#
#   julia --startup-file=no ci/downstream.jl <project dir> [--compat] [Name=path ...]
#
# Sets `[sources] <Package> = {path = "<abs path in this workspace>"}` in the
# Project.toml in <project dir> for every workspace package it names in
# [deps], [weakdeps] or [extras], plus each extra `Name=path` pair (another
# downstream checkout, e.g. RANSACScoringDoneRight for BlobBoards). With
# --compat the stack compat entries are set to the workspace's ("0.1"), for a
# downstream repository that has not yet taken the 0.1 compat (DESIGN §6.4).
#
# It REWRITES the file through TOML.print (comments are lost): run it on a
# throwaway checkout only (the Nightly downstream job, the dry run).
#
# The implementation is VisualGeometryToolkitCore's `stack_ci` (the `downstream`
# command; `source_downstream`), read by path: standard libraries only.

include(joinpath(dirname(@__DIR__), "VisualGeometryToolkitCore", "src", "quality", "stack_ci.jl"))

exit(stack_ci(["downstream"; ARGS]; root = dirname(@__DIR__)))
