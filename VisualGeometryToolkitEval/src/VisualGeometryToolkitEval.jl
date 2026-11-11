"""
    VisualGeometryToolkitEval

The evaluation base of the visual-geometry stack: statistics ([`Stats`](@ref)),
the codified cross-validation ([`Tuning`](@ref), [`Evaluation`](@ref)), the
stage/run-record framework ([`Experiments`](@ref)), rendering
([`Render`](@ref)), pool IO ([`Pools`](@ref)), the raw dataset readers
([`Datasets`](@ref)), the published-data catalog ([`DataArtifacts`](@ref)), the
reproducible-figures contract ([`Reproduce`](@ref)) and the paper-project
conventions ([`PaperProjects`](@ref)), with the artifact formats, the defaults
of record and the driver CLI. Depends on VisualGeometryToolkitCore, VisualGeometryToolkitIO
and the core estimator VisualGeometryToolkitEstimation (its loop and problems, for the
composition evaluator) — never on a paper's scorer or optimizer package.

Each submodule's API is reached through it (`using VisualGeometryToolkitEval.Pools`);
this module exports the submodules and the CLI.
"""
module VisualGeometryToolkitEval

include("config_defaults.jl")   # the default mechanism: configs/default.toml → `_config_default`
include("cli.jl")               # cli_option, start_output_flush!; code_trees, code_tree_changes (code_identity.jl)
include("artifact_format.jl")   # the format stamp of every persisted artifact kind

import JLD2   # artifact_format.jl: every call is qualified

include("Stats.jl")
include("Tuning.jl")
include("Experiments.jl")
include("Render.jl")
include("Pools.jl")
include("Evaluation.jl")        # after Tuning, Render and Pools, which its engine reads
include("Datasets.jl")
include("DataArtifacts.jl")   # after Experiments, Tuning and Datasets: it reads all three
include("Reproduce.jl")
include("PaperProjects.jl")   # after Experiments, Reproduce and DataArtifacts: it checks through all three

export Stats, Tuning, Evaluation, Experiments, Render, Pools, Datasets, DataArtifacts, Reproduce,
    PaperProjects
export cli_option, start_output_flush!

public ARTIFACT_FORMATS, DEFAULTS_PATH
public code_trees, code_tree_changes, refuse_uncommitted_code, file_sha256   # the code identity of a run, a file's hash (src/code_identity.jl)

"""
    engine_fixtures() -> Module

The evaluation engine's shared test fixtures, the module `EngineFixtures` of
the Test extension (`using Test` loads it): `epair` (THE synthetic E pool pair,
`EPAIR_GT_VARIANCE`), `starts_pool`, `subpair` (a pair restricted to some of
its stored models), `engine_scene` (a scene of the selection stream),
`roster_tuning` and `write_roster_sidecar` (the tuning record the roster
readers are tested on). This package's engine tests and the paper projects'
engine-integration tests build their fixtures from it.
"""
function engine_fixtures end

public engine_fixtures   # the engine's shared test fixtures (the Test extension)

end # module VisualGeometryToolkitEval
