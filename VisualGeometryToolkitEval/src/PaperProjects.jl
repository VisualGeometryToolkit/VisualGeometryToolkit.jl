# =============================================================================
# PaperProjects — the paper-project conventions of the stack (accv2026 is the
# worked example), as one enforcing check and one template:
#
#   paper_project_violations(dir; papers)   every rule a paper project's suite
#                                           asserts empty, in one call
#   new_paper_project(path; name, papers)   a project born compliant, on
#                                           DrWatson's initialize_project
#
# The rules themselves live with their owners — the data policy in
# Experiments (`data_policy_violations`), the asset index in Reproduce
# (`reproduce_index_violations`), the published-data catalog in DataArtifacts
# (`catalog_violations`); this module adds the layout, the configs of record and
# the provenance rules and reports all of them as one structured list.
# =============================================================================
"""
    PaperProjects

The paper-project conventions: [`paper_project_violations`](@ref) (the one
check every paper project's suite calls) and [`new_paper_project`](@ref) (the
template, on DrWatson's `initialize_project`). The conventions are stated on
the documentation page [Paper projects](@ref).
"""
module PaperProjects

using TOML: TOML
using Pkg: Pkg
using LibGit2: LibGit2
using DrWatson: initialize_project
using ..Experiments: DataPolicyAllow, data_policy_violations, contract_violations
using ..Reproduce: reproduce_index_violations, read_reproduce_index, IndexEntry, paper_figures_dir   # paper_figures_dir: Reproduce's (its deploy stages'), exported here too
using ..DataArtifacts: load_catalog, catalog_violations, artifact_parts, _PROJECT_CATALOG
import ..DataArtifacts: _is_produced, _produced_cover   # the produced kind: a derived input that names its producer
import ..Reproduce: load_stage_graph, _stage_index, _stage_configs, _config_inputs   # a producer's stage, its configs and their inputs
using ..Experiments: is_artifact_view
import ..Experiments: _PROJECT_LAYOUT   # experiments/data_policy.jl: the one project layout (the template's folders)
import ..Experiments: _is_under   # experiments/data.jl: a path under a root, both resolved (A4's links into the store)
import .._git_line                # src/code_identity.jl: one git query, `nothing` outside a work tree (L6)
using VisualGeometryToolkitCore: artifact_store, store_name_clashes, is_published_url   # the artifact store: its links' target, its name record; R3's placeholder URL

export PaperProjectViolation, PAPER_PROJECT_CHECKS, paper_project_violations,
    PAPER_PROJECT_TEMPLATE, new_paper_project, paper_figures_dir

include("paper_projects/violations.jl")
include("paper_projects/template.jl")

end # module PaperProjects
