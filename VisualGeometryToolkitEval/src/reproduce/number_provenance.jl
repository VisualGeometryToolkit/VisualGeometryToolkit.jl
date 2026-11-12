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

"""
    provenance_token(s) -> String

A provenance field as a number macro's comment holds it (`PaperNumbers`'s `run_key`, `commit`, `config`): every
space and `;` (the comment's separators) made `_`. The one spelling [`checkout_commit`](@ref) and
[`number_provenance`](@ref) write; a project that records a field of its own uses it too.
"""
provenance_token(s) = replace(String(s), r"[\s;]" => "_")

"""
    project_relative_path(path, root = project_root()) -> String

A path as a provenance comment cites it: relative to the project `root` when absolute, as given otherwise, made a
[`provenance_token`](@ref) (`number_provenance`'s `config`).
"""
project_relative_path(path::AbstractString, root::AbstractString = project_root()) =
    provenance_token(isabspath(path) ? relpath(path, root) : path)

"""
    checkout_commit(root = project_root()) -> String

The commit of the checkout at `root` as DrWatson describes it
(`gitdescribe(root)`: `<tag>-<n>-g<sha>` or a sha, `-dirty` on an uncommitted
tree; `"unknown"` outside a repository), a space or `;` made `_`: the
[`PaperNumbers`](@ref) `commit` of numbers a driver computes itself. Numbers
read from a run carry the run's recorded commit instead
([`number_provenance`](@ref)).
"""
checkout_commit(root::AbstractString = project_root()) =
    provenance_token(something(gitdescribe(root), "unknown"))

"""
    number_provenance(run_dir; root = project_root()) -> NamedTuple

The provenance of numbers read from one run's artifacts, from its
`_provenance.toml` ([`read_provenance`](@ref)): `run_key`, the `commit` that
computed it (the recorded `gitcommit`) and the `config` it ran from (the
recorded `config_source`, made relative to `root`), each only where the run
recorded it, a space or `;` made `_`. For [`add_number!`](@ref)'s
`provenance`, or splatted into [`PaperNumbers`](@ref):
`PaperNumbers(; driver, config, number_provenance(run_dir)...)` (a field given
after the splat wins).
"""
function number_provenance(run_dir::AbstractString; root::AbstractString = project_root())
    p = read_provenance(run_dir)
    out = (;)
    haskey(p, "run_key") && (out = merge(out, (; run_key = provenance_token(p["run_key"]))))
    haskey(p, "gitcommit") && (out = merge(out, (; commit = provenance_token(p["gitcommit"]))))
    haskey(p, "config_source") && (out = merge(out, (; config = project_relative_path(String(p["config_source"]), root))))
    return out
end
