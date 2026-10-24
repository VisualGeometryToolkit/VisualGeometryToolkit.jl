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
using ..Reproduce: reproduce_index_violations, read_reproduce_index, IndexEntry
using ..DataArtifacts: load_catalog, catalog_violations

export PaperProjectViolation, PAPER_PROJECT_CHECKS, paper_project_violations,
    PAPER_PROJECT_TEMPLATE, new_paper_project

include("paper_projects/violations.jl")
include("paper_projects/template.jl")

end # module PaperProjects
