# =============================================================================
# evaluation/engine/arms.jl — composed and solo arms, their roster, and the deployed
# readers of a tuned operating point.
#
# An ARM is a selector (a sweep key `sel` whose native axis is the arm's
# selection axis) and a width parameterization (`kind`, a VisualGeometryToolkitCore
# `WidthAxis`, whose axis is its tuning axis). Rosters are paper config: a paper
# builds one `Roster` value and hands it to every reader — no registry.
#
# Grids travel as one `grids::Dict{Symbol,Vector{Float64}}` keyed by the axis's
# persisted name (VisualGeometryToolkitCore `width_axis_name`):
#
#   :outlier_halfwidth  the Student-t marginal selector's axis, and the
#                       outlier-half-width width parameter's
#   :loss_width         the threshold-score selectors' axis (MSAC, RANSAC, …:
#                       a threshold is its truncated loss's width), and the
#                       loss-width width parameter's
#   :scale_multiple     the scale-multiple width parameter's (dimensionless)
# =============================================================================

# -----------------------------------------------------------------------------
# Width parameterizations — how an arm's refinement width is supplied, by
# VisualGeometryToolkitCore's `WidthAxis`:
#
# - `ScaleMultiple()` — the user supplies the scale multiple; the loss width is
#   `scale_multiple·ŝ`, ŝ read from the consensus set at refit time
#   (CONSENSUS-ANCHORED). The value is baked into `ctor(scale_multiple)`.
# - `LossWidth()` — the loss width itself, in absolute residual units: One
#   quantity whatever the refiner. A native refiner runs its unit-multiple
#   strategy at it (`with_width(lo, fixed_width(lo, w²))` — basin-only), an
#   external refiner refines at it.
# - `OutlierHalfwidth()` — the Student-t marginal's uniform outlier half-width,
#   supplied natively (the arm re-estimates its own scale).
# -----------------------------------------------------------------------------

# -----------------------------------------------------------------------------
# Width sources — where a composed arm's refit width VALUE comes from:
#
# - `TunedWidth()` — its own value, tuned by the refiner stage at its
#   selector's tuned selection width a* (every arm of record until 2026-10-06).
# - `SelectionWidth()` — the selection's tuned width itself: the refit width IS
#   a*, never tuned, and under a ζ corruption it is ζ·a* — exactly the
#   selection's corrupted native. Only an arm whose width axis IS its selection
#   axis can take it (an outlier half-width refit of a Student-t marginal
#   selection: the strictly objective-matched arm, the LO maximizing the very
#   score that selected its seed).
# -----------------------------------------------------------------------------
"""
    WidthSource

Where a [`ComposedSpec`](@ref)'s refit width comes from: [`TunedWidth`](@ref)
or [`SelectionWidth`](@ref) (see the block comment above).
"""
abstract type WidthSource end
"""
    TunedWidth()

The arm's own refit width, tuned at its selector's tuned selection width (see [`WidthSource`](@ref)).
"""
struct TunedWidth <: WidthSource end
"""
    SelectionWidth()

The arm's refit width IS its selection's tuned width, never tuned (see [`WidthSource`](@ref)).
"""
struct SelectionWidth <: WidthSource end

"""
    ComposedSpec(base, sel, selection::WidthAxis, kind::WidthAxis, ctor; source = TunedWidth())

One composed score+LO pipeline. `sel` is the key of its selection sweep (the
selections `idxs_of[sel]` a refit seeds from); `selection` is that sweep's
native axis — VisualGeometryToolkitCore's `width_axis` of the selection scoring
(`OutlierHalfwidth()` for the Student-t marginal, `LossWidth()` for a threshold
score); `kind` is the arm's width parameterization; `ctor(value)` builds the
LO strategy at a width value (a scale-multiple arm bakes it in; the other
kinds deliver it through their port, see [`width_delivery`](@ref)); `source`
is where the width value comes from ([`WidthSource`](@ref)): a
[`SelectionWidth`](@ref) arm requires `kind == selection`, and the tuning
skips it.
"""
struct ComposedSpec
    base::String
    sel::Symbol
    selection::WidthAxis
    kind::WidthAxis
    ctor::Function
    source::WidthSource
    function ComposedSpec(base::AbstractString, sel::Symbol, selection::WidthAxis, kind::WidthAxis,
                          ctor::Function; source::WidthSource = TunedWidth())
        selection isa NativeAxis || throw(ArgumentError(
            "ComposedSpec $base: the selection axis is a native axis (OutlierHalfwidth() or " *
                "LossWidth()); got $selection"))
        source isa SelectionWidth && kind != selection && throw(ArgumentError(
            "ComposedSpec $base: a SelectionWidth() arm refits at its selection's width, so its width " *
                "axis must be the selection axis $selection; got $kind"))
        return new(String(base), sel, selection, kind, ctor, source)
    end
end

"Is `arm`'s refit width tuned (a [`TunedWidth`](@ref) arm), rather than its selection's?"
is_tuned(arm::ComposedSpec) = arm.source isa TunedWidth

"""
    operating_width(arm::ComposedSpec, selection, tuned) -> Float64

The refit width of `arm` at the tuned selection width `selection`.
By its [`WidthSource`](@ref): a [`TunedWidth`](@ref) arm's tuned value `tuned`; a
[`SelectionWidth`](@ref) arm's `selection` itself — `tuned` is `nothing` (the
tuning stage, which has none) or a recorded width, which must equal
`selection` exactly (a sidecar read back).
"""
operating_width(arm::ComposedSpec, selection::Real, tuned::Union{Nothing,Real}) =
    _operating_width(arm.source, arm, Float64(selection), tuned)
_operating_width(::TunedWidth, arm, selection, tuned::Real) = Float64(tuned)
_operating_width(::TunedWidth, arm, selection, ::Nothing) =
    throw(ArgumentError("operating_width: arm $(arm.base) has a tuned width and none was given"))
_operating_width(::SelectionWidth, arm, selection, ::Nothing) = selection
_operating_width(::SelectionWidth, arm, selection, tuned::Real) =
    Float64(tuned) == selection ? selection : throw(ArgumentError(
        "operating_width: arm $(arm.base) refits at its selection's width $selection; the record says $tuned"))

"""
    width_delivery(P::WidthAxis, lo, value, n) -> NamedTuple

How a width `value` reaches the refit of `lo` over `n` columns, as LoRefitSpec
keywords. A scale multiple is baked into the LO object; an outlier half-width
is the refit's native (`natives`); a loss width is an absolute width
(`widths`), the solo delivery bitwise whatever the refiner.
"""
width_delivery(::ScaleMultiple, lo, value, n) = (;)
width_delivery(::OutlierHalfwidth, lo, value, n) = (; natives = fill(value, n))
width_delivery(::LossWidth, lo, value, n) = (; widths = fill(Float64(value), n))

# The native a DEPLOYED refit consumes: a scale-multiple arm refits at its
# selection native (the width lives in the LO, consensus-anchored); every native
# parameterization at its tuned value.
deployed_ref(::ScaleMultiple, selection, width) = Float64(selection)
deployed_ref(::WidthAxis, selection, width) = Float64(width)

# =============================================================================
# SoloSpec — one role validated in isolation with one native.
#
# Where a ComposedSpec is a composed score+LO pipeline, a SoloSpec is one role
# — a scorer or an LO — whose single width value is supplied in ABSOLUTE
# residual units, with the other role held out of the loop entirely:
#
#   role = :refinement — the basin refiner: refits from δ-perturbed GT seeds;
#     NO scoring consensus may leak into the loss's scale. `axis` names WHAT
#     the value is, so drivers dispatch mechanically (`solo_refit_imc!`):
#       LossWidth() — the value IS the loss width: a native refiner's
#               unit-multiple strategy runs at it, an external refiner
#               refines at it.
#       OutlierHalfwidth() — a self-scaled Student-t arm: the value is the
#               outlier half-width via nat′.
#   role = :selection — the selection scorer: ctor(value) returns a concrete
#     pool scorer; the driver argmaxes the pool per pair and reads the
#     winner's pose error.
# =============================================================================

"""
    SoloSpec(base, role, axis::WidthAxis, ctor)

One role (`:selection` | `:refinement`) validated in isolation at one width
value on `axis` (see the block comment above); `ctor(value)` returns a pool
scorer (selection) or an LO strategy (refinement).
"""
struct SoloSpec
    base::String
    role::Symbol
    axis::WidthAxis
    ctor::Function
    function SoloSpec(base::AbstractString, role::Symbol, axis::WidthAxis, ctor::Function)
        role in (:selection, :refinement) ||
            throw(ArgumentError("SoloSpec $base: role must be :selection or :refinement; got :$role"))
        axis isa NativeAxis || throw(ArgumentError(
            "SoloSpec $base: a solo width is absolute (OutlierHalfwidth() or LossWidth()); got $axis"))
        return new(String(base), role, axis, ctor)
    end
end

"""
    solo_refit_imc!(ctx, spec::SoloSpec, seed_E, width, selection_halfwidth; on_fail = NaN)
        -> (imc°, lo_ns)

The width-delivery dispatch of a `:refinement` solo evaluation, one mechanical
map from `spec.axis` to the refit call, shared by the basin tuning's refits and
the ladder's solo mode so tuning and test deliver the width identically. The
seed is swept under the context's solo template (`ctx.solo_scoring`):

- `LossWidth()` — the absolute loss width `width`: a native refiner's strategy
  runs at it (`with_width(lo, fixed_width(lo, width²))`, its seed swept at
  `selection_halfwidth` — scoring only, never a width), an external refiner
  refines at it;
- `OutlierHalfwidth()` — the LO at its own native `nat′ = width`.
"""
function solo_refit_imc!(ctx, spec::SoloSpec, seed_E, width::Float64, selection_halfwidth::Float64;
                         on_fail = NaN)
    spec.role === :refinement ||
        error("solo_refit_imc!: spec $(spec.base) has role :$(spec.role), not :refinement")
    lo = spec.ctor(width)
    spec.axis isa LossWidth ?
        _width_refit_imc!(ctx, RefitFamily(lo), lo, seed_E, width, selection_halfwidth; on_fail) :
        refit_imc!(ctx, lo, seed_E, width; scoring = ctx.solo_scoring, on_fail)
end

# -----------------------------------------------------------------------------
# The roster — a paper's arms, as one value every reader takes explicitly
# -----------------------------------------------------------------------------
"""
    Roster(composed, solo, basin)

A paper's arm roster, the argument of every reader that rebuilds arms from a tuned
sidecar ([`deployed_budget_settings`](@ref), [`deployed_basin_settings`](@ref),
[`audit_operating_points`](@ref)). Each field is the paper's builder:

- `composed(problem::Symbol) -> Vector{ComposedSpec}` — the composed
  score+LO roster (scale-multiple / outlier-half-width / natively delivered
  loss-width arms) of the composed miscal and budget evaluations;
- `solo(problem::Symbol; role::Symbol) -> Vector{SoloSpec}` — the solo roster;
- `basin(problem::Symbol) -> Vector{ComposedSpec}` — the basin roster
  (loss-width and outlier-half-width arms only; a scale multiple has no
  meaning on the basin axis).
"""
struct Roster{C,S,B}
    composed::C
    solo::S
    basin::B
end

function _basin_specs(r::Roster, problem::Symbol)
    specs = r.basin(problem)
    bad = [a.base for a in specs if a.kind isa ScaleMultiple]
    isempty(bad) || error("basin roster: scale-multiple arms $(bad) — only LossWidth() / " *
                          "OutlierHalfwidth() arms have a basin width")
    return specs
end

# -----------------------------------------------------------------------------
# Deployment evaluation refits
# -----------------------------------------------------------------------------

# The composed evaluations (miscal, validation-budget) refit at a NATIVE: they
# consume the scale-multiple / outlier-half-width roster and the loss-width
# arms of an external refiner (whose native IS its loss width); a loss-width
# arm of a native refiner (an absolute width — basin-only) is refused.
_composed_kind(arm::ComposedSpec, who::String) = _composed_kind(arm.kind, arm, who)
_composed_kind(P::WidthAxis, arm, who) = P
_composed_kind(P::LossWidth, arm, who) = _composed_family(RefitFamily(arm.ctor(1.0)), P, arm, who)
_composed_family(::ExternalRefit, P, arm, who) = P
_composed_family(::NativeRefit, P, arm, who) =
    error("$who: arm $(arm.base) is LossWidth() on a native refiner (an absolute width — " *
          "basin-only); the composed evaluations refit at a native")
# The native an evaluation refit consumes: a scale-multiple arm refits at the
# (possibly corrupted) SELECTION native, mapped onto the refiner's own axis
# (`consumed_native`), so tuning and evaluation see one operating point; a
# native parameterization refits at its (possibly corrupted) tuned value.
_evaluation_native(::ScaleMultiple, lo, axis, selection, width, d_g) = consumed_native(lo, axis, selection; d_g)
_evaluation_native(::WidthAxis, lo, axis, selection, width, d_g) = Float64(width)

"""
    miscal_evaluation_refit(arm, idxs_of, tuned, zetas, grids; d_g, plugin_scale = nothing) -> LoRefitSpec

The miscalibration evaluation refit of one arm, one column per ζ in `zetas`.
Column j seeds at the
selection column nearest ζ_j·selection* and refits at the exactly-scaled
refinement native (ζ_j·width* for the native parameterizations — both formed
from the corrupted plug-in — while a scale multiple is consensus-anchored and
NOT corrupted). `tuned` is the tuned point as NATIVE VALUES,
`(; selection, width)`, its width read through [`operating_width`](@ref) (a
[`SelectionWidth`](@ref) arm: `width` is `nothing` or the selection, and
column j refits at ζ_j·selection*, the selection's own corrupted native); `grids` supplies the arm's selection grid, which the
seed columns are found in.

`plugin_scale` (the consensus-anchoring ABLATION, scale-multiple arms only):
the same operating point and seeds, the scale multiple's loss width delivered
at the FIXED external plug-in scale ζ_j·`plugin_scale` instead of the consensus
set's ŝ — `with_width(lo, fixed_width(lo, (ζ_j·plugin_scale)²))` through the
LoRefitSpec `fixed_s2s` column. A native parameterization's width already is
the corrupted plug-in; it is refused.
"""
function miscal_evaluation_refit(arm::ComposedSpec, idxs_of::Dict{Symbol,Matrix{Int32}}, tuned,
                                 zetas::Vector{Float64}, grids::AbstractDict{Symbol}; d_g::Integer,
                                 plugin_scale::Union{Nothing,Real} = nothing)
    P = _composed_kind(arm, "miscal_evaluation_refit")
    selgrid = grids[width_axis_name(arm.selection)]
    selection = Float64(tuned.selection)
    width = operating_width(arm, selection, tuned.width)
    seedcols = [nearest_column(selgrid, corrupt_native(selection, ζ)) for ζ in zetas]
    lo = arm.ctor(width)
    natives = [_evaluation_native(P, lo, arm.selection, corrupt_native(selection, ζ),
                                  corrupt_native(width, ζ), d_g) for ζ in zetas]
    LoRefitSpec("$(arm.base)|evaluation", lo, arm.selection, zetas, idxs_of[arm.sel];
                seedcols, natives, fixed_s2s = _evaluation_fixed_scale(P, arm, plugin_scale, zetas))
end

# The fixed external scale of an evaluation column: none (the arm's own interface)
# unless a plug-in scale is given, which only a scale-multiple arm can take.
_evaluation_fixed_scale(P::WidthAxis, arm, ::Nothing, zetas) = nothing
_evaluation_fixed_scale(P::ScaleMultiple, arm, plugin_scale::Real, zetas) =
    [corrupt_native(plugin_scale, ζ)^2 for ζ in zetas]
_evaluation_fixed_scale(P::WidthAxis, arm, plugin_scale::Real, zetas) =
    throw(ArgumentError("miscal_evaluation_refit: arm $(arm.base) is :$(width_axis_name(P)) — its width " *
                        "already is the ζ-corrupted plug-in; the fixed-scale evaluation is defined " *
                        "for ScaleMultiple() arms only"))

# -----------------------------------------------------------------------------
# The interior-argmin gate (read side)
#
# A tuned operating point on the boundary of its CV window means the window
# clipped the optimum; it must never be silently consumed. The tuning stage
# records each arm's clipped axes (`boundary_hits`: `:selection_lo`/`_hi`,
# `:<kind>_lo`/`_hi`) in its sidecar; the deployed readers below REFUSE any arm
# with a non-empty list.
# -----------------------------------------------------------------------------

# Per-arm clipped-axes lists of a tuned sidecar, as persisted. Every arm must
# carry one: a sidecar without is refused.
function _sidecar_boundary_hits(tk::TuningArtifact, by_base::Dict{String,ComposedSpec})
    hits = Dict{String,Vector{Symbol}}()
    for b in tk["arms"]
        haskey(by_base, b) || continue   # unknown arm → the reader's own lookup errors
        haskey(tk["boundary_hits"], b) ||
            throw(ArgumentError("$(tk.path): no boundary_hits for arm $b — regenerate it"))
        hits[b] = Symbol.(collect(tk["boundary_hits"][b]))
    end
    return hits
end

# Read-side gate of the interior-argmin invariant: any arm with a non-empty
# clipped-axes list is a hard error.
function _enforce_interior_argmin(hits, path, who::String)
    clipped = sort!([b => v for (b, v) in hits if !isempty(v)]; by = first)
    isempty(clipped) && return nothing
    detail = join(("$(b): $(join(v, ", "))" for (b, v) in clipped), "; ")
    error("$who: $path carries CLIPPED operating points ($detail) — the CV window " *
          "clipped the optimum; expand the window on the flagged side and re-run " *
          "the tuning for this arm; consuming a clipped operating point violates the " *
          "CV protocol (interior-argmin invariant)")
end

_is_basin(tk) = String(tk["mode"]) == "basin"

# -----------------------------------------------------------------------------
# The deployed readers
# -----------------------------------------------------------------------------

"""
    deployed_budget_settings(roster::Roster, path; problem = nothing)
        -> Vector{label => (lo, sel, ref, kind)}

Per-arm tuned operating points of a composed tuning sidecar, rebuilt through the roster.
The sidecar is read through VisualGeometryToolkitEval `load_tuning_artifact`, which
refuses another format; each arm is rebuilt through `roster.composed(problem)`: the LO instance (the tuned scale
multiple baked in for scale-multiple arms), the SELECTION native and the
REFINEMENT native (scale-multiple arms refit at the selection native —
consensus-anchored; native parameterizations at their tuned value) and the
arm's width parameterization `kind` — the roster's, the one source of an
arm's kind (a sidecar's own `kinds` list is not read). `problem` defaults to
the sidecar's own `"problem"`.

INTERIOR-ARGMIN INVARIANT: a HARD ERROR on any arm whose persisted
`boundary_hits` is non-empty — expand the window on the flagged side and
re-run the tuning (a [`SelectionWidth`](@ref) arm has no width of its own to
clip: its list carries the selection's axis alone, and its recorded width must
be its selection's, [`operating_width`](@ref)). A basin sidecar is refused ([`deployed_basin_settings`](@ref) reads it).
"""
function deployed_budget_settings(roster::Roster, path::AbstractString;
                                  problem::Union{Nothing,Symbol} = nothing)
    tk = load_tuning_artifact(path)
    _is_basin(tk) && error("deployed_budget_settings: $path is a basin (loss-width) sidecar — " *
                           "the composed readers refuse it; read it with deployed_basin_settings")
    prob = problem === nothing ? Symbol(tk["problem"]) : problem
    by_base = Dict(a.base => a for a in roster.composed(prob))
    _enforce_interior_argmin(_sidecar_boundary_hits(tk, by_base), path, "deployed_budget_settings")
    out = Pair{String,@NamedTuple{lo::Any, sel::Float64, ref::Float64, kind::WidthAxis}}[]
    for b in tk["arms"]
        arm = by_base[b]
        _composed_kind(arm, "deployed_budget_settings")
        selection = Float64(tk["tuned_selection"][b])
        width = operating_width(arm, selection, Float64(tk["tuned_width"][b]))
        push!(out, b => (lo = arm.ctor(width), sel = selection,
                         ref = deployed_ref(arm.kind, selection, width), kind = arm.kind))
    end
    return out
end

"""
    deployed_arm_settings(roster::Roster, path; problem = nothing) -> Vector{label => (lo, fixed_native)}

Each pipeline's DEPLOYED configuration: the LO strategy at its tuned width and its fixed native.
It is the `(lo, ref)` projection of
[`deployed_budget_settings`](@ref), under the same interior-argmin gate.
"""
deployed_arm_settings(roster::Roster, path::AbstractString; problem::Union{Nothing,Symbol} = nothing) =
    [b => (s.lo, s.ref) for (b, s) in deployed_budget_settings(roster, path; problem)]

"""
    deployed_basin_settings(roster::Roster, path)
        -> Vector{base => (spec::SoloSpec, width::Float64)}

Each basin arm's tuned width from a basin tuning sidecar (`mode = "basin"`).
The arms are rebuilt through `roster.basin(problem)` and repackaged as a refinement
`SoloSpec` whose axis is the arm's parameterization, so tuning and test
deliver the width identically via [`solo_refit_imc!`](@ref). Refuses a
non-basin sidecar. Same interior-argmin gate as the composed readers.
"""
function deployed_basin_settings(roster::Roster, path::AbstractString)
    tk = load_tuning_artifact(path)
    _is_basin(tk) ||
        error("deployed_basin_settings: $path is not a basin sidecar (mode = " *
              "$(repr(tk["mode"]))) — composed sidecars go through deployed_budget_settings")
    by_base = Dict(a.base => a for a in _basin_specs(roster, Symbol(tk["problem"])))
    _enforce_interior_argmin(_sidecar_boundary_hits(tk, by_base), path, "deployed_basin_settings")
    out = Pair{String,@NamedTuple{spec::SoloSpec, width::Float64}}[]
    for b in tk["arms"]
        arm = by_base[b]
        push!(out, b => (spec = SoloSpec(b, :refinement, arm.kind, arm.ctor),
                         width = operating_width(arm, Float64(tk["tuned_selection"][b]),
                                                 Float64(tk["tuned_width"][b]))))
    end
    return out
end

"""
    audit_operating_points(roster::Roster, path; problem = nothing) -> true

Audit the tuning sidecar at `path` through its gate-enforced reader.
A basin sidecar goes through [`deployed_basin_settings`](@ref), a composed one
through [`deployed_budget_settings`](@ref); it throws on any clipped arm (a
boundary argmin never reaches a consumer) and on any other artifact format, and
returns `true`. MANDATORY before a figure or table consumes the sidecar.
"""
function audit_operating_points(roster::Roster, path::AbstractString;
                                problem::Union{Nothing,Symbol} = nothing)
    _is_basin(load_tuning_artifact(path)) ? deployed_basin_settings(roster, path) :
        deployed_budget_settings(roster, path; problem)
    @info "audit OK (interior argmins)" path
    return true
end
