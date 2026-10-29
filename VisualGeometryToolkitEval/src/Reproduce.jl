# =============================================================================
# Reproduce — the reproducible-figures contract of a paper project: every
# figure, table and number macro a camera-ready paper reads is produced by
# tracked code from declared data, indexed in the project's REPRODUCE.md
# (label | asset | kind | generator | config | data | provenance), and enforced by the
# project's suite through `reproduce_index_violations`.
#
# The implementation is the dependency-free reproduce/reproduce_index.jl (and
# reproduce/paper_assets.jl, the verify / deploy / numbers stages, which needs
# it), which an environment that cannot load this package includes by path.
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
[`uncited_macros`](@ref) (numbers the paper defines but never cites).
"""
module Reproduce

include("reproduce/reproduce_index.jl")
include("reproduce/paper_assets.jl")

export PaperAsset, PaperAssetGraph, IndexEntry, ReproduceViolation,
    paper_asset_graph, read_reproduce_index, declared_outputs, declared_inputs, tex_macro_values,
    macro_literal_violations, reproduce_index_violations,
    fragment_coordinates, compare_fragments, verify_assets, assets_deployed, deploy_assets,
    gather_numbers, uncited_macros

public VERIFY_REPORT

end # module Reproduce
