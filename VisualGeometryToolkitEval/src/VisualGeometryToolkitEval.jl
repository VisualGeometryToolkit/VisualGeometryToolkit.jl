"""
    VisualGeometryToolkitEval

The evaluation base of the visual-geometry stack: statistics ([`Stats`](@ref)),
the codified cross-validation ([`Tuning`](@ref), [`Evaluation`](@ref)), the
stage/run-record framework ([`Experiments`](@ref)), rendering
([`Render`](@ref)), pool IO ([`Pools`](@ref)), the raw dataset readers
([`Datasets`](@ref)), the published-data catalog ([`DataArtifacts`](@ref)), the
reproducible-figures contract ([`Reproduce`](@ref)) and the paper-project
conventions ([`PaperProjects`](@ref)), with the artifact formats, the defaults
of record and the driver CLI. Depends on VisualGeometryToolkitCore, VisualGeometryToolkitIo
and the core estimator VisualGeometryToolkitEstimation (its loop and problems, for the
composition evaluator) — never on a paper's scorer or optimizer package.

Each submodule's API is reached through it (`using VisualGeometryToolkitEval.Pools`);
this module exports the submodules and the CLI.
"""
module VisualGeometryToolkitEval

include("config_defaults.jl")   # THE default mechanism: configs/default.toml → `_config_default`
include("cli.jl")               # cli_option, start_output_flush!
include("artifact_format.jl")   # THE format stamp of every persisted artifact kind

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

end # module VisualGeometryToolkitEval
