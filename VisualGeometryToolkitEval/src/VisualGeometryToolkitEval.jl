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
using VisualGeometryToolkitCore: canonical_json, set_site_artifact_store!   # the content key's encoding (code_identity.jl, through cli.jl); the artifact store's site rung
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
public code_trees, code_tree_changes, refuse_uncommitted_code, file_sha256, tier_variable, tier_data_root, catalog_view_mount, inspect_input, scratch_record, tree_sha256, digest_cache_dir   # the code identity of a run, a file's hash, a tier's root (src/code_identity.jl)

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

# The site rung of VisualGeometryToolkitCore's artifact store: the `[storage] artifact_store` of the site profile
# VGTK_SITE names (this package owns the site profiles, configs/sites/).
# The store is shared by every project: the bound project's stage graph only adds its own sites directory to the
# lookup, so a graph that does not load (a project's work in progress) leaves the shipped profiles and paths.
function _site_artifact_store(site)
    g = try Experiments._project_graph() catch; nothing end
    s = Experiments._site_profile(site, g)
    return s === nothing || isempty(s.artifact_store) ? nothing : s.artifact_store
end
__init__() = (set_site_artifact_store!(_site_artifact_store); nothing)

end # module VisualGeometryToolkitEval
