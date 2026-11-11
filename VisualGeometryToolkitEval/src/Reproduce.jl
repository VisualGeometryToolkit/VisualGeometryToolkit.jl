# =============================================================================
# Reproduce — the reproducible-figures contract of a paper project: every
# figure, table and number macro a camera-ready paper reads is produced by
# tracked code from declared data, indexed in the project's REPRODUCE.md
# (label | asset | kind | generator | config | data | provenance), and enforced by the
# project's suite through `reproduce_index_violations`.
#
# The implementation is the dependency-free reproduce/reproduce_index.jl (then
# reproduce/number_macros.jl, the one writer of a driver's number macros, and
# reproduce/paper_assets.jl, the verify / deploy / numbers stages, each needing
# the files before it), which an environment that cannot load this package
# includes by path, in that order. reproduce/results_branches.jl (a full
# reproduction as a results branch of the paper; git and TOML) is not among them.
# reproduce/runner.jl (one command reproduces a project: its stage graph on a
# site, Slurm or local) is dependency-free on its own (stdlibs, and src/cli.jl
# included before it): a project's launcher includes the two by path before its
# environment is instantiated; reproduce/run_results_branch.jl, the runner's
# results-branch stage, is the module's only.
# =============================================================================
"""
    Reproduce

The reproducible-figures contract: [`paper_asset_graph`](@ref) (what a paper
reads), [`read_reproduce_index`](@ref) (what REPRODUCE.md declares) and
[`reproduce_index_violations`](@ref) (where the two disagree, plus generators
that do not write their asset, static assets without provenance, and generated
numbers re-typed in prose); and the regeneration's last stages:
[`verify_assets`](@ref) and [`deploy_assets`](@ref) (a submitted asset is
replaced only with sign-off), [`gather_numbers`](@ref) (the numbers file) and
[`uncited_macros`](@ref) (numbers the paper defines but never cites); and the
one writer of a driver's number macros: [`PaperNumbers`](@ref),
[`add_number!`](@ref), [`add_table_row!`](@ref), [`write_numbers`](@ref),
[`tex_number`](@ref) and [`number_macro_name`](@ref), and where its numbers
come from: [`checkout_commit`](@ref) and [`number_provenance`](@ref); and a full
reproduction as a branch of the paper for review: [`results_branch`](@ref),
[`accept_results`](@ref) and [`remove_results_worktree`](@ref); and the
runner that reproduces a project with one command: a stage graph of record
([`load_stage_graph`](@ref), [`select_stages`](@ref)) on a site
([`load_site`](@ref): Slurm or local), the inputs checked first
([`missing_inputs`](@ref)), a run manifest per launch, finished stages reused by
run key, and the results branch at the end ([`results_branch_of_run`](@ref));
[`reproduce_main`](@ref) is its command line.
"""
module Reproduce

using Dates: Dates, DateTime, UTC, now
using TOML: TOML
using VisualGeometryToolkitCore: VisualGeometryToolkitCore, sha256_hex     # results_branches.jl: a data tier's hash, the GPUs' names
using ..Tuning: _git_identity                  # results_branches.jl: a checkout's HEAD and its dirtiness, the package's one rule
import ..cli_option                             # src/cli.jl: runner.jl reads its options through it
import ..code_trees, ..code_tree_changes, ..refuse_uncommitted_code   # src/code_identity.jl: the code identity of a launch (runner.jl)
using DrWatson: gitdescribe                    # number_provenance.jl: the checkout's commit
using ..Experiments: project_root, read_provenance   # number_provenance.jl: the project a driver runs from, a run's record

include("reproduce/reproduce_index.jl")
include("reproduce/number_macros.jl")
include("reproduce/paper_assets.jl")
include("reproduce/number_provenance.jl")
include("reproduce/results_branches.jl")
include("reproduce/runner.jl")
include("reproduce/run_results_branch.jl")

export PaperAsset, PaperAssetGraph, IndexEntry, ReproduceViolation,
    paper_asset_graph, read_reproduce_index, declared_outputs, declared_inputs, tex_macro_values,
    macro_literal_violations, reproduce_index_violations,
    fragment_coordinates, compare_fragments, verify_assets, assets_deployed, deploy_assets,
    gather_numbers, uncited_macros,
    PaperNumbers, add_number!, add_table_row!, add_macro!, write_numbers, tex_number, number_macro_name,
    checkout_commit, number_provenance,
    results_branch, accept_results, remove_results_worktree,
    Site, StageSpec, StageGraph, PlannedStage, ReproducePlan, load_site, shipped_sites, load_stage_graph, select_stages,
    stage_items, missing_inputs, resolve_roots, plan_reproduction, print_plan, print_missing, run_manifests,
    slurm_job_states, slurm_queue_states, stage_markers_dir, stage_markers, stage_run_state, submit_slurm!, run_local!, reproduce_status, reproduce_main, results_branch_assets,
    results_branch_of_run

public VERIFY_REPORT, RESULTS_MANIFEST, RESULTS_COMPARISON, SITES_DIR, REPRODUCE_USAGE

end # module Reproduce
