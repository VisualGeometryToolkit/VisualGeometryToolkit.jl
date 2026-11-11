# =============================================================================
# number_provenance.jl — where a driver's numbers come from, as the fields of a
# `PaperNumbers` (number_macros.jl): the checkout's commit, for numbers the
# driver computes itself, and a run's recorded provenance (its
# `_provenance.toml`, experiments/provenance.jl), for numbers read from that
# run's artifacts. BlobBoards' bmvc2026 (`figure_numbers`), CaliBlobs' icra2026
# (`checkout_commit`, `run_provenance`) and LocalOptimizationDoneRight's
# accv2026 (`scripts/paper_numbers.jl`) each restated them.
#
# Not dependency-free (DrWatson, the Experiments provenance reader), so not in
# number_macros.jl.
# =============================================================================

# A provenance field as the comment holds it: a space or a `;` (the comment's
# separators) becomes `_`.
_provenance_token(s) = replace(String(s), r"[\s;]" => "_")

"""
    checkout_commit(root = project_root()) -> String

The commit of the checkout at `root` as DrWatson describes it
(`gitdescribe(root)`: `<tag>-<n>-g<sha>` or a sha, `-dirty` on an uncommitted
tree; `"unknown"` outside a repository), a space or `;` made `_`: the
[`PaperNumbers`](@ref) `commit` of numbers a driver computes itself. Numbers
read from a run carry the run's recorded commit instead.
"""
checkout_commit(root::AbstractString = project_root()) =
    _provenance_token(something(gitdescribe(root), "unknown"))
