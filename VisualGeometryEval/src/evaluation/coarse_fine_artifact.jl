# =============================================================================
# THE tuning artifact of a coarse-fine search (cv_probe.jl) and its ONE gated
# reader.
#
# A tuned setting — one scalar or a joint setting of several — is selected by
# `coarse_fine_search` and deployed by reading its artifact. The writer fixes
# the layout every reader relies on; the reader is the one place the
# interior-argmin invariant is enforced: a search that hit its window's edge
# on ANY axis is never consumed.
#
# SIZE. The full errors of every evaluated cell would be the cells × units ×
# items tensor (the BlobBoards detection search: ~500 cells × 80 frames × 435
# board pairs, ~140 MB). What the selection and its bootstrap need of them is
# each cell's per-unit sufficient statistics (Tuning `unit_statistics`): for
# the mAA criterion the units' score sums and item counts — two numbers per
# unit, so the bootstrap over units is exact from them; for any other
# criterion the errors themselves (nothing smaller suffices for a median). The
# artifact stores those, never the raw tensor.
# =============================================================================

"""
    save_coarse_fine_artifact(path, spec::TuningSpec, rec; unit, code, data...) -> path

Persist one search record `rec` ([`coarse_fine_search`](@ref), one key's
NamedTuple) under its `coarse_fine` / `joint_coarse_fine` spec as a tuning
artifact ([`save_tuning_artifact`](@ref)): the grids `coarse|<axis>` and
`fine|<axis>`, and

    axes              the axis names, the search's order
    selected          axis => THE tuned value (the fine argmin)
    coarse_argmin     axis => the coarse argmin
    boundary_hit      axis => the coarse argmin within the margin of a face
                      after the extension budget (refused by the reader)
    n_extensions      axis => [lo rounds, hi rounds]
    coarse_objective, fine_objective   the objective over each product grid
    cells             every evaluated cell, one row per cell (columns: axes)
    unit_statistics   per cell, Tuning `unit_statistics` of its errors — when
                      every cell's errors are a `units × items` matrix
    incumbent, kept_incumbent, gain, incumbent_unit_statistics
                      with a declared incumbent (`selected` is it when kept)

`data` are the caller's further keys (the bootstrap of the selection, say).
"""
function save_coarse_fine_artifact(path::AbstractString, spec::TuningSpec, rec::NamedTuple;
                                   unit::SamplingUnit, code, data...)
    spec.scheme isa Union{CoarseFine,JointCoarseFine} ||
        throw(ArgumentError("save_coarse_fine_artifact: $(spec.name) is a $(nameof(typeof(spec.scheme))), not a coarse-fine spec"))
    names = String.(collect(rec.names))
    per_axis(nt) = Dict{String,Any}(n => nt[Symbol(n)] for n in names)
    grids = Dict{Symbol,GridSpec}()
    for (i, n) in enumerate(names)
        grids[Symbol("coarse|", n)] = rec.coarse[i]; grids[Symbol("fine|", n)] = rec.fine[i]
    end
    E = last.(rec.evaluated)
    stats = all(e -> e isa AbstractMatrix, E) ?
        (; unit_statistics = [unit_statistics(spec.scheme.criterion, e) for e in E]) : (;)
    return save_tuning_artifact(path, spec; grids, unit, code,
        axes = names, selected = per_axis(rec.selected), coarse_argmin = per_axis(rec.coarse_argmin),
        boundary_hit = per_axis(rec.boundary_hit),
        n_extensions = Dict{String,Any}(n => rec.n_extensions[i, :] for (i, n) in enumerate(names)),
        coarse_objective = rec.coarse_objective, fine_objective = rec.fine_objective,
        cells = [Float64(p[Symbol(n)]) for p in first.(rec.evaluated), n in names], stats...,
        (haskey(rec, :incumbent) ? (; incumbent = per_axis(rec.incumbent), kept_incumbent = rec.kept_incumbent,
            gain = Dict{String,Any}("objective" => rec.gain.objective, "p_gain" => rec.gain.p_gain),
            incumbent_unit_statistics = unit_statistics(spec.scheme.criterion, rec.incumbent_errors)) : (;))..., data...)
end

"""
    deployed_setting(path; context = path, allow_edge = false) -> NamedTuple

THE gated reader of a coarse-fine tuning artifact
([`save_coarse_fine_artifact`](@ref)): the selected value of every axis, as a
NamedTuple. HARD ERROR when the spec is not a coarse-fine one or the search
hit its window's edge on any axis ([`enforce_probe_interior`](@ref);
`allow_edge = true` downgrades to a warning for FORENSIC reads only). A tuned
setting is deployed through this, never from a literal.

Also reads the one-scalar layout written before the joint search (a CV
`"record"` and `"x_star_fine"`; BlobBoards' half-width run of record):
`(; <param_name> = x_star_fine)`.
"""
function deployed_setting(path::AbstractString; context::AbstractString = path, allow_edge::Bool = false)
    art = load_tuning_artifact(path)
    art.spec.scheme isa Union{CoarseFine,JointCoarseFine} ||
        throw(ArgumentError("deployed_setting: $path holds a $(nameof(typeof(art.spec.scheme))) spec, not a coarse-fine one"))
    if !haskey(art, "selected")
        haskey(art, "record") && haskey(art, "x_star_fine") ||
            throw(ArgumentError("deployed_setting: $path carries no selection — not a coarse-fine tuning artifact"))
        enforce_probe_interior(art["record"]; context, allow_edge)
        return NamedTuple{(Symbol(art["record"]["param_name"]),)}((Float64(art["x_star_fine"]),))
    end
    names = collect(String, art["axes"])
    for n in names
        enforce_probe_interior(Bool(art["boundary_hit"][n]); context = "$context, axis $n", allow_edge)
    end
    return NamedTuple{Tuple(Symbol.(names))}(Tuple(Float64(art["selected"][n]) for n in names))
end
