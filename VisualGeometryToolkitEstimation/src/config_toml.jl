# =============================================================================
# TOML configuration — loaders, the strategy-tag registry, and the writer
# =============================================================================
#
# The shipped configs/default.toml is the config of record (see
# config_defaults.jl). This file is the other half of the contract:
#
#   * LOADERS build config values from a TOML a caller supplies —
#     `RansacConfig(path)` and `local_optimization_from_toml(table; class)`.
#     Absent keys inherit the shipped file (the ONE default authority); present
#     keys are typed overrides; UNKNOWN keys are refused, never ignored.
#   * The WRITER (`config_dict`) is derived from the structs by walking
#     `fieldnames`, so a new field appears in the table with no edit here and
#     cannot be forgotten. `local_optimization_from_toml(config_dict(lo)) == lo`
#     is asserted by the suite.
#
# Included LAST: it dispatches on every config-carrying type in the package.
# =============================================================================

# -----------------------------------------------------------------------------
# Key discipline — shared by every section reader
# -----------------------------------------------------------------------------

# The generic loader/writer machinery is VisualGeometryToolkitCore's (hoisted from
# this fork and BlobBoards' on 2026-09-07): present-keys-only keyword
# overrides, refuse-unknown-keys with retirement, the fieldnames-derived
# table walk and its per-type hooks, the dotted default lookup, the sorted
# writer. What stays here is this package's: the Greek↔ASCII key spelling,
# the retired-key set, the tag registries and the hook methods for its types.
using VisualGeometryToolkitCore: present_kwargs, check_keys, toml_table,
    allowed_keys, convert_field
import VisualGeometryToolkitCore: toml_key, toml_skip_fields, toml_field

# RETIRED keys: knobs removed because nothing read them, which recorded run
# sidecars (immutable provenance) may still contain. Exact (section, key)
# pairs only — a typo still fails loudly; a retired key loads with a warning.
# None yet; the mechanism exists so the first retirement has a home.
const _RETIRED_KEYS = Set{Tuple{String, String}}()

# Reject unknown keys rather than ignoring them. A typo'd knob that silently
# does nothing is the failure this file exists to prevent — a config that
# looks like it set `max_trials` but spelled it `max_trial` would
# otherwise run at the default and report success.
_check_keys(d::AbstractDict, allowed, section::AbstractString) =
    check_keys(d, allowed, section; retired = _RETIRED_KEYS)

# Allowed keys are the struct's fields — derived, never hand-listed. A
# strategy's width field is spelled by its two keys, `scale_multiple` and
# `loss_width` (the strategy types' hook methods follow the tag registry below).
_allowed_keys(::Type{T}, extra::String...) where {T} = allowed_keys(T, extra...)

# The READER's half of VGC's field walk: typed per-key overrides — only the keys
# PRESENT in the table become keywords, so absent ones fall through to the
# struct's default (the shipped file), and ASCII spellings map back to the Greek
# field names (`_field_of`) — through a `present_kwargs` spec derived from
# the same `fieldnames` / `toml_skip_fields` / `toml_key` walk `toml_table`
# writes with, each value converted by its field's type (`_from_toml`, the
# one per-field hook), minus the fields in `skip` the caller supplies itself.
# Every section reader goes through it — nothing hand-lists a key.
function _table_kwargs(::Type{T}, d::AbstractDict; skip = ()) where {T}
    spec = [
        toml_key(T, f) => (v -> _from_toml(T, Val(f), v))
            for f in fieldnames(T) if !(f in skip) && !(f in toml_skip_fields(T))
    ]
    return present_kwargs(d, spec...; field_of = _field_of)
end
_from_toml(::Type{T}, ::Val{F}, v) where {T, F} = convert_field(fieldtype(T, F), v)

# -----------------------------------------------------------------------------
# [ransac] → RansacConfig
# -----------------------------------------------------------------------------

"""
    RansacConfig(path::AbstractString) -> RansacConfig

Load the `[ransac]` table of a TOML file. Keys present in the file override
the shipped defaults; absent keys inherit `configs/default.toml`. An unknown
key is an `ArgumentError`, never ignored — and so is a file with no `[ransac]`
table at all: a misspelled table (`[ransc]`, `[RANSAC]`) would otherwise load
as the shipped defaults and report success. The file's other tables (a shared
run config's `[pipelines.*]`, say) are not this loader's to judge.
"""
function RansacConfig(path::AbstractString)
    d = TOML.parsefile(path)
    haskey(d, "ransac") || throw(
        ArgumentError(
            "$path has no [ransac] table (its tables: $(join(sort!(collect(keys(d))), ", ")))"
        )
    )
    d["ransac"] isa AbstractDict || throw(ArgumentError("$path: `ransac` must be a table"))
    return _ransac_from_toml(d["ransac"])
end

function _ransac_from_toml(d::AbstractDict)
    _check_keys(d, _allowed_keys(RansacConfig), "ransac")
    return RansacConfig(; _table_kwargs(RansacConfig, d)...)
end

# -----------------------------------------------------------------------------
# Local-optimization strategies — the tag registry
# -----------------------------------------------------------------------------
#
# A strategy is spelled in TOML as an inline table:
#
#     local_optimization = { type = "irls", loss = "tls" }
#
# `type` selects the strategy from a tag registry (LO_TAGS, or a merge of it
# with a downstream package's); `loss` (Irls) selects the robust loss from
# LOSS_TAGS. A strategy with a width takes it as `scale_multiple` (the loss
# width as a multiple of the consensus scale, VGC's `ScaleMultiple`) or
# `loss_width` (an absolute width, `LossWidth`) — both is an error — and with
# neither its width follows the scorer's decision point. Every other key is a
# constructor keyword of that strategy and defaults to its
# `[local_optimization.<type>]` table. The type's owner builds it
# ([`lo_from_toml`](@ref)): a downstream package's strategies load through
# this one entry once their tags are merged in.

"""
    LOSS_TAGS

The loss-shape registry (VisualGeometryToolkitCore's `TagRegistry`, family suffix
`Shape`): TOML `loss` tag → shape type, read off the one loss-identity table
(`_LOSS_TABLE`, scoring_glue.jl).
"""
const LOSS_TAGS = TagRegistry(:loss, "Shape", (tag => K for (K, tag) in _LOSS_TABLE)...;
                              paths = [r"(^|\.)loss$"])

# `LO_TAGS` (scoring_glue.jl) is the strategy registry, derived from the one row
# per strategy (`_LO_REGISTRY`) its keyword defaults read their tag from too.

# The skeleton's per-type hooks, on OUR strategy types — one method per
# registered type rather than one on VGC's abstract `AbstractLocalOptimization`,
# which would be type piracy (VGC's generic on VGC's type; Aqua refuses it).
# The scale interface is not a config key; the Greek fields spell ASCII.
for T in values(LO_TAGS)
    @eval begin
        # The width field is written and read as its two keys
        # (`scale_multiple` | `loss_width`), never as `width`.
        toml_skip_fields(::Type{<:$T}) = (:width,)
        toml_key(::Type{<:$T}, f::Symbol) = _toml_key(f)
    end
end
const _LOSS_TAG_OF = Dict{Type, String}(shape => tag for (tag, shape) in LOSS_TAGS)

_lo_tag(lo::AbstractLocalOptimization) = _lo_tag(typeof(lo))
_loss_tag(::Type{K}) where {K <: AbstractLossShape} = _LOSS_TAG_OF[K]
# The width keys a strategy's table may carry: both for a width-carrying one
# (the least-squares arm takes neither), none for another.
_width_keys(::Type{C}) where {C} = _lo_takes_width(C) ? ("scale_multiple", "loss_width") : ()
_width_keys(::Type{<:Irls{GaussianShape}}) = ()

# The shape a FIXED-LOSS arm deploys is `_fixed_loss_shape` (scoring_glue.jl,
# beside the arms' identity, which reads it too).

# A literal `scale_multiple` is the caller's own number and claims nothing
# about provenance, so it carries no shape and is taken as given.
_resolve_scale_multiple(v::Real, ::Symbol, ::Type{<:AbstractLossShape}) = Float64(v)

# `scale_multiple = "cross_validated"` names the accv2026 operating points,
# which are LocalOptimizationDoneRight's tuning constants: its loader resolves
# them. The core has none.
_resolve_scale_multiple(v::AbstractString, ::Symbol, ::Type) = throw(ArgumentError(
    v == "cross_validated" ?
    "scale_multiple = \"cross_validated\" is resolved by LocalOptimizationDoneRight's tuning " *
        "constants: load the strategy through its tags (tags = merge(VisualGeometryToolkitEstimation.LO_TAGS, " *
        "LocalOptimizationDoneRight.LO_TAGS)), or give a number" :
    "scale_multiple must be a number, got $(repr(v))"))

function _loss_of(d::AbstractDict, section)
    haskey(d, "loss") || throw(ArgumentError("[$section] needs a `loss` (one of $(sort(collect(keys(LOSS_TAGS)))))"))
    haskey(LOSS_TAGS, d["loss"]) || throw(ArgumentError("[$section] unknown loss $(repr(d["loss"])); expected one of $(sort(collect(keys(LOSS_TAGS))))"))
    return LOSS_TAGS[d["loss"]]
end

"""
    local_optimization_from_toml(table; class::Symbol, tags = LO_TAGS) -> AbstractLocalOptimization

Build a local-optimization strategy from its TOML table (see the registry
comment above). `tags` is the strategy registry the `type` is looked up in —
[`LO_TAGS`](@ref), or its merge with a downstream package's
(`merge(LO_TAGS, LocalOptimizationDoneRight.LO_TAGS)`); the type's owner then
builds it ([`lo_from_toml`](@ref)). `class` is the pipeline's model class (a
downstream `scale_multiple = "cross_validated"` resolves through it). Unknown
`type`, `loss`, or keys are refused.
"""
function local_optimization_from_toml(d::AbstractDict; class::Symbol, section::AbstractString = "local_optimization",
                                      tags::TagRegistry = LO_TAGS)
    haskey(d, "type") || throw(ArgumentError("[$section] needs a `type` (one of $(sort(collect(keys(tags)))))"))
    T = tags[d["type"]]     # an unknown or retired tag is refused here, a retired one naming its successor
    return lo_from_toml(T, d; class, section)
end

"""
    lo_from_toml(T, table; class::Symbol, section) -> AbstractLocalOptimization

The strategy of registry entry `T` (a type or a loss-parametric family) from
its TOML table — the per-owner half of [`local_optimization_from_toml`](@ref).
The core defines it for its own strategies; a downstream package that
registers strategies (LocalOptimizationDoneRight) adds its methods.
"""
function lo_from_toml end

for T in values(LO_TAGS)
    @eval lo_from_toml(::Type{<:$T}, d::AbstractDict; class::Symbol, section::AbstractString) =
        _core_lo_from_toml($T, d; class, section)
end

# A loss-parametric family is concrete at its `loss`; a plain strategy is itself.
_core_concrete(::Type{Irls}, d, sec) = (Irls{_loss_of(d, sec)}, ("type", "loss"))
_core_concrete(::Type{T}, d, sec) where {T} = (T, ("type",))

function _core_lo_from_toml(T, d::AbstractDict; class::Symbol, section::AbstractString)
    tag = d["type"]
    sec = "$section.$tag"
    # The concrete strategy type and the keys the table may carry beyond its fields.
    C, extra = _core_concrete(T, d, sec)
    _check_keys(d, _allowed_keys(C, extra..., _width_keys(C)...), sec)
    # The ONE presence check: a key the shipped file has no default for is
    # required. (No width: follow the scorer.)
    for f in fieldnames(C)
        f in toml_skip_fields(C) && continue
        haskey(d, toml_key(C, f)) || _lo_has_default(C, f) ||
            throw(ArgumentError("[$sec] needs `$(toml_key(C, f))`"))
    end
    kw = _table_kwargs(C, d)
    haskey(d, "scale_multiple") && haskey(d, "loss_width") &&
        throw(ArgumentError("[$sec] give `scale_multiple` or `loss_width`, not both"))
    haskey(d, "loss_width") && return _lo_construct(C; kw..., loss_width = Float64(d["loss_width"]))
    haskey(d, "scale_multiple") || return _lo_construct(C; kw...)
    scale_multiple = _resolve_scale_multiple(d["scale_multiple"], class, _scale_multiple_shape(C))
    return _lo_construct(C; kw..., scale_multiple)
end

# The shape a strategy's `scale_multiple` was cross-validated for: the loss of
# the parametric family.
_scale_multiple_shape(::Type{C}) where {C <: Irls} = strategy_loss(C)

# The default strategy of a pipeline — read from the shipped file's
# [pipelines.<class>] table. This is what every find_* signature defaults to.
_dflt_lo(class::Symbol) =
    local_optimization_from_toml(_dflt("pipelines.$class", "local_optimization"); class, section = "pipelines.$class.local_optimization")

# -----------------------------------------------------------------------------
# The inverse: config → Dict, derived from fieldnames
# -----------------------------------------------------------------------------

"""
    config_dict(c) -> Dict

The TOML table for a config value (a `RansacConfig` or a local-optimization
strategy), as a plain `Dict`. Derived from the structs by walking `fieldnames`,
so it cannot fall out of sync with them; a strategy's table loads back through
[`local_optimization_from_toml`](@ref) to an equal value.
"""
config_dict(c::RansacConfig) = toml_table(c)

config_dict(::NoLocalOptimization) = Dict{String, Any}("type" => "no_local_optimization")
function config_dict(lo::Irls)
    # The field walk is VGC's `toml_table` (the scale interface skipped, Greek
    # spelled ASCII); only the
    # discriminators are this package's.
    d = toml_table(lo)
    d["type"] = _lo_tag(lo)
    # `strategy_loss` (scoring_glue.jl) is the documented way to read the
    # objective's tag out of a strategy TYPE.
    lo isa Irls && (d["loss"] = _loss_tag(strategy_loss(lo)))
    _lo_takes_width(typeof(lo)) && _width_to_toml!(d, lo.width)
    return d
end

# The width mode, written as its key: `scale_multiple`, or `loss_width` (an
# absolute width, `√s2` of `LossWidth(w) = (1, w²)`); no width writes nothing.
_width_to_toml!(d, ::Nothing) = d
_width_to_toml!(d, w::ScaleMultiple{Float64}) = (d["scale_multiple"] = w.value; d)
function _width_to_toml!(d, w::LossWidth{NTuple{2, Float64}})
    m, s2 = w.value
    d["loss_width"] = m * sqrt(s2)
    return d
end

# -----------------------------------------------------------------------------
# [mixture_em] → MixtureEmConfig — the model-update and nuisance-part tags
# -----------------------------------------------------------------------------
#
# A tagged value is spelled as its tag alone when it carries nothing, or as an
# inline table naming the tag in `type` with its fields as keys:
#
#     model_update = { type = "keep_model", tol = 1.0e-9 }      # or "weighted_refine"
#     nuisance = { scale = "estimated", inlier_prior = { type = "fixed_prior", inlier_prior = 0.5 },
#                  nu = { type = "fixed_nu", nu = 5.0 } }
#
# A field's key is its name (Greek spelled ASCII, `_toml_key`), every field is
# required, and an unknown tag or key is refused (a retired one names its
# successor). The scale's held value is VisualGeometryToolkitCore's `FixedS2(s2)`,
# keyed `s2`: a SQUARED scale. ν has a third policy, "consensus_dof": held at
# the consensus seed's posterior-predictive dof (VisualGeometryToolkitCore `ConsensusDof`).

"""
    MODEL_UPDATE_TAGS

The model-update registry of the posterior-weight EM (VisualGeometryToolkitCore's
`TagRegistry`): `"weighted_refine"` ([`WeightedRefine`](@ref)) and
`"keep_model"` ([`KeepModel`](@ref)).
"""
const MODEL_UPDATE_TAGS = TagRegistry(:model_update, "", "weighted_refine" => WeightedRefine, "keep_model" => KeepModel;
                                      retired = ("keep" => "keep_model",), paths = [r"(^|\.)model_update(\.type)?$"])
# One tag table per nuisance part: what "estimate" and "fixed" build for it.
const NUISANCE_TAGS = (
    scale = TagRegistry(:scale_policy, "", "estimated" => Estimated, "fixed_s2" => FixedS2;
                        retired = ("estimate" => "estimated", "fixed" => "fixed_s2"),
                        paths = [r"(^|\.)nuisance\.scale(\.type)?$"]),
    inlier_prior = TagRegistry(:inlier_prior_policy, "", "estimated" => Estimated, "fixed_prior" => FixedPrior;
                               retired = ("estimate" => "estimated", "fixed" => "fixed_prior"),
                               paths = [r"(^|\.)nuisance\.inlier_prior(\.type)?$"]),
    nu = TagRegistry(:nu_policy, "", "estimated" => Estimated, "fixed_nu" => FixedNu, "consensus_dof" => ConsensusDof;
                     retired = ("estimate" => "estimated", "fixed" => "fixed_nu", "partition_dof" => "consensus_dof"),
                     paths = [r"(^|\.)nuisance\.nu(\.type)?$"]),
)

function _tagged_from_toml(tags, v, section::AbstractString)
    known = sort(collect(keys(tags)))
    tag, d = v isa AbstractString ? (v, nothing) :
        v isa AbstractDict ? (haskey(v, "type") ? (v["type"], v) :
            throw(ArgumentError("[$section] needs a `type` (one of $known)"))) :
        throw(ArgumentError("[$section] must be a tag (one of $known) or a table with a `type`, got $(repr(v))"))
    T = try
        tags[tag]   # an unknown or retired tag is refused, a retired one naming its successor
    catch e
        e isa ArgumentError || rethrow()
        throw(ArgumentError("[$section] $(e.msg)"))
    end
    fkeys = String[_toml_key(f) for f in fieldnames(T) if f isa Symbol]
    if d === nothing
        isempty(fkeys) || throw(ArgumentError(
            "[$section] $(repr(tag)) needs its $(join(fkeys, ", ")): spell it { type = $(repr(tag)), … }"))
        return T()
    end
    _check_keys(d, ["type"; fkeys], section)
    for k in fkeys
        haskey(d, k) || throw(ArgumentError("[$section] needs `$k`"))
    end
    return T((d[k] for k in fkeys)...)
end

function _tagged_to_toml(tags, x)
    tag = only(k for (k, T) in tags if x isa T)
    fieldcount(typeof(x)) == 0 && return tag
    d = Dict{String, Any}("type" => tag)
    for f in fieldnames(typeof(x))
        d[_toml_key(f::Symbol)] = getfield(x, f)
    end
    return d
end

_model_update_from_toml(v, section::AbstractString) = _tagged_from_toml(MODEL_UPDATE_TAGS, v, section)

function _nuisance_from_toml(d, section::AbstractString)
    d isa AbstractDict || throw(ArgumentError("[$section] must be a table of $(join(keys(NUISANCE_TAGS), ", "))"))
    _check_keys(d, [String(k) for k in keys(NUISANCE_TAGS)], section)
    for k in keys(NUISANCE_TAGS)
        haskey(d, String(k)) || throw(ArgumentError("[$section] needs `$k`"))
    end
    return NamedTuple{keys(NUISANCE_TAGS)}(
        Tuple(_tagged_from_toml(NUISANCE_TAGS[k], d[String(k)], "$section.$k") for k in keys(NUISANCE_TAGS))
    )
end

toml_field(c::MixtureEmConfig, ::Val{:model_update}) = _tagged_to_toml(MODEL_UPDATE_TAGS, c.model_update)
toml_field(c::MixtureEmConfig, ::Val{:nuisance}) =
    Dict{String, Any}(String(k) => _tagged_to_toml(NUISANCE_TAGS[k], c.nuisance[k]) for k in keys(NUISANCE_TAGS))

_from_toml(::Type{<:MixtureEmConfig}, ::Val{:model_update}, v) = _model_update_from_toml(v, "mixture_em.model_update")
_from_toml(::Type{<:MixtureEmConfig}, ::Val{:nuisance}, v) = _nuisance_from_toml(v, "mixture_em.nuisance")
_from_toml(::Type{<:MixtureEmConfig}, ::Val{:max_iter}, v) = convert(Int, v)

"""
    mixture_em_from_toml(table; section = "mixture_em") -> MixtureEmConfig

Build a [`MixtureEmConfig`](@ref) from its TOML table: present keys override,
absent keys inherit `configs/default.toml`'s `[mixture_em]`, and an unknown
key, tag or tag field is an `ArgumentError`.
"""
function mixture_em_from_toml(d::AbstractDict; section::AbstractString = "mixture_em")
    _check_keys(d, _allowed_keys(MixtureEmConfig), section)
    return MixtureEmConfig(; _table_kwargs(MixtureEmConfig, d)...)
end

config_dict(c::MixtureEmConfig) = toml_table(c)
