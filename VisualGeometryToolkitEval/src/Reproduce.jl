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
[`tex_number`](@ref) and [`number_macro_name`](@ref); and a full
reproduction as a branch of the paper for review: [`results_branch`](@ref),
[`accept_results`](@ref) and [`remove_results_worktree`](@ref).
"""
module Reproduce

using Dates: Dates, DateTime, UTC, now
using TOML: TOML
using VisualGeometryToolkitCore: VisualGeometryToolkitCore, sha256_hex     # results_branches.jl: a data tier's hash, the GPUs' names
using ..Tuning: _git_identity                  # results_branches.jl: a checkout's HEAD and its dirtiness, the package's one rule

include("reproduce/reproduce_index.jl")
include("reproduce/number_macros.jl")
include("reproduce/paper_assets.jl")
include("reproduce/results_branches.jl")

export PaperAsset, PaperAssetGraph, IndexEntry, ReproduceViolation,
    paper_asset_graph, read_reproduce_index, declared_outputs, declared_inputs, tex_macro_values,
    macro_literal_violations, reproduce_index_violations,
    fragment_coordinates, compare_fragments, verify_assets, assets_deployed, deploy_assets,
    gather_numbers, uncited_macros,
    PaperNumbers, add_number!, add_table_row!, write_numbers, tex_number, number_macro_name,
    results_branch, accept_results, remove_results_worktree

public VERIFY_REPORT, RESULTS_MANIFEST, RESULTS_COMPARISON

end # module Reproduce
