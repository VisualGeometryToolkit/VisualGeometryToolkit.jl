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
# table walk and its per-type hooks (the Greek↔ASCII key spelling among
# them), the dotted default lookup, the sorted writer. What stays here is this
# package's: the retired-key set, the tag registries and the hook methods for
# its types.
using TOML: TOML
using VisualGeometryToolkitCore: present_kwargs, check_keys, toml_table,
    allowed_keys, convert_field, toml_field_name
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
# field names (`toml_field_name`) — through a `present_kwargs` spec derived from
# the same `fieldnames` / `toml_skip_fields` / `toml_key` walk `toml_table`
# writes with, each value converted by its field's type (`_from_toml`, the
# one per-field hook), minus the fields in `skip` the caller supplies itself.
# Every section reader goes through it — nothing hand-lists a key.
function _table_kwargs(::Type{T}, d::AbstractDict; skip = ()) where {T}
    spec = [
        toml_key(T, f) => (v -> _from_toml(T, Val(f), v))
            for f in fieldnames(T) if !(f in skip) && !(f in toml_skip_fields(T))
    ]
    return present_kwargs(d, spec...; field_of = toml_field_name)
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
# `[local_optimization.<type>]` table. One reader builds every registered
# strategy ([`lo_from_toml`](@ref)), its exceptions stated by its owner's hook
# methods: a downstream package's strategies load through this one entry once
# their tags are merged in.

"""
    LOSS_TAGS

The loss-shape registry (VisualGeometryToolkitCore's `TagRegistry`, family suffix
`Shape`): TOML `loss` tag → shape type, read off the one loss-identity table
(`_LOSS_TABLE`, scoring_glue.jl).
"""
const LOSS_TAGS = TagRegistry(:loss, "Shape", (tag => K for (K, tag) in _LOSS_TABLE)...;
                              paths = [r"(^|\.)loss$"])

# `LO_TAGS` (scoring_glue.jl) is the strategy registry, built by `@lo_registry`
# from the one row per strategy (`_LO_REGISTRY`), which also gives every
# strategy type its `toml_skip_fields` (the width is written and read as its
# two keys, `scale_multiple` | `loss_width`, never as `width`).

const _LOSS_TAG_OF = Dict{Type, String}(shape => tag for (tag, shape) in LOSS_TAGS)
_loss_tag(::Type{K}) where {K <: AbstractLossShape} = _LOSS_TAG_OF[K]
# The width keys a strategy's table may carry: both for a width-carrying one
# (the least-squares arm takes neither), none for another.
_width_keys(::Type{C}) where {C} = _lo_takes_width(C) ? ("scale_multiple", "loss_width") : ()
_width_keys(::Type{<:Irls{GaussianShape}}) = ()

# A strategy's TOML values convert through its hook.
_from_toml(::Type{T}, ::Val{F}, v) where {T <: AbstractLocalOptimization, F} = lo_field_from_toml(T, Val(F), v)

"""
    lo_resolve_scale_multiple(T, K, value; class::Symbol) -> Float64

A TOML `scale_multiple` of strategy `T`, read for loss shape `K`
([`lo_scale_multiple_shape`](@ref)) in pipeline model class `class`. A number
is the caller's own and claims nothing about provenance, so it is taken as
given. A string is refused here: `"cross_validated"` names the accv2026
operating points, LocalOptimizationDoneRight's tuning constants, which its
strategies resolve with a method of this hook.
"""
lo_resolve_scale_multiple(::Type, ::Type, v::Real; class::Symbol) = Float64(v)
lo_resolve_scale_multiple(::Type, ::Type, v::AbstractString; class::Symbol) = throw(ArgumentError(
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
    local_optimization_from_toml(table; class::Symbol, section = "local_optimization", tags = LO_TAGS) -> AbstractLocalOptimization

Build a local-optimization strategy from its TOML table (see the registry
comment above). `tags` is the strategy registry the `type` is looked up in —
[`LO_TAGS`](@ref), or its merge with a downstream package's
(`merge(LO_TAGS, LocalOptimizationDoneRight.LO_TAGS)`); the entry is then
built by [`lo_from_toml`](@ref). `class` is the pipeline's model class (a
downstream `scale_multiple = "cross_validated"` resolves through it). Unknown
`type`, `loss`, or keys are refused.
"""
function local_optimization_from_toml(d::AbstractDict; class::Symbol, section::AbstractString = "local_optimization",
                                      tags::TagRegistry = LO_TAGS)
    haskey(d, "type") || throw(ArgumentError("[$section] needs a `type` (one of $(sort(collect(keys(tags)))))"))
    T = tags[d["type"]]     # an unknown or retired tag is refused here, a retired one naming its successor
    return lo_from_toml(T, d; class, section, tags)
end

"""
    lo_concrete(T, table, section; tags) -> (C, extra_keys)

The concrete strategy type registry entry `T` names in its TOML `table`, and
the keys the table may carry beyond `C`'s fields and width. Default: a
loss-parametric family is concrete at its `loss` ([`LOSS_TAGS`](@ref); a
missing or unknown loss is an `ArgumentError` listing the loss tags),
`(T{K}, ("type", "loss"))`; any other strategy is itself, `(T, ("type",))`.
`tags` is the strategy registry the table was read through, for a strategy
that wraps another named by tag. `section` names the table in messages.
"""
lo_concrete(::Type{T}, d::AbstractDict, sec::AbstractString; tags::TagRegistry = LO_TAGS) where {T} =
    lo_parametric(T) ? (T{_loss_of(d, sec)}, ("type", "loss")) : (T, ("type",))

"""
    lo_from_toml(T, table; class::Symbol, section = "local_optimization", tags = LO_TAGS) -> AbstractLocalOptimization

The strategy of registry entry `T` (a type or a loss-parametric family,
registered by any package with [`@lo_registry`](@ref)) from its TOML table —
the half of [`local_optimization_from_toml`](@ref) after the tag lookup.
The table's keys are the fields (Greek spelled ASCII), the width as
`scale_multiple` or `loss_width`, and [`lo_concrete`](@ref)'s extra keys; an
unknown key is refused, a field the owner's defaults have no key for is
required, every other absent key inherits the defaults. A `scale_multiple`
goes through [`lo_resolve_scale_multiple`](@ref) at
[`lo_scale_multiple_shape`](@ref).
"""
function lo_from_toml(T::Type, d::AbstractDict; class::Symbol, section::AbstractString = "local_optimization",
                      tags::TagRegistry = LO_TAGS)
    tag = d["type"]
    sec = "$section.$tag"
    # The concrete strategy type and the keys the table may carry beyond its fields.
    C, extra = lo_concrete(T, d, sec; tags)
    _check_keys(d, _allowed_keys(C, extra..., _width_keys(C)...), sec)
    # The ONE presence check: a key the defaults have no value for is
    # required. (No width: follow the scorer.)
    defaults = lo_defaults(C)
    for f in fieldnames(C)
        f in toml_skip_fields(C) && continue
        haskey(d, toml_key(C, f)) || _lo_has_default(C, f, defaults) ||
            throw(ArgumentError("[$sec] needs `$(toml_key(C, f))`"))
    end
    kw = _table_kwargs(C, d)
    haskey(d, "scale_multiple") && haskey(d, "loss_width") &&
        throw(ArgumentError("[$sec] give `scale_multiple` or `loss_width`, not both"))
    haskey(d, "loss_width") && return lo_construct(C; defaults, kw..., loss_width = Float64(d["loss_width"]))
    haskey(d, "scale_multiple") || return lo_construct(C; defaults, kw...)
    scale_multiple = lo_resolve_scale_multiple(C, lo_scale_multiple_shape(C), d["scale_multiple"]; class)
    return lo_construct(C; defaults, kw..., scale_multiple)
end

# The terminal placement's table names its strategy in a nested `inner` table,
# read through the same registry (so a downstream inner strategy loads when
# its tags are merged in); `type` and `inner` are its only keys.
function lo_from_toml(::Type{<:TerminalLO}, d::AbstractDict; class::Symbol,
                      section::AbstractString = "local_optimization", tags::TagRegistry = LO_TAGS)
    sec = "$section.$(d["type"])"
    _check_keys(d, ["type", "inner"], sec)
    haskey(d, "inner") || throw(ArgumentError("[$sec] needs an `inner` strategy table ({ type = …, … })"))
    inner = d["inner"]
    inner isa AbstractDict ||
        throw(ArgumentError("[$sec] `inner` must be a strategy table ({ type = …, … }), got $(repr(inner))"))
    return TerminalLO(local_optimization_from_toml(inner; class, section = "$sec.inner", tags))
end

# The default strategy of a pipeline — read from the shipped file's
# [pipelines.<class>] table. This is what every find_* signature defaults to.
_dflt_lo(class::Symbol) =
    local_optimization_from_toml(_dflt("pipelines.$class", "local_optimization"); class, section = "pipelines.$class.local_optimization")

# -----------------------------------------------------------------------------
# The inverse: config → Dict, derived from fieldnames
# -----------------------------------------------------------------------------

"""
    config_dict(c) -> Dict

The TOML table for a config value (a `RansacConfig`, a `MixtureEmConfig` or a
local-optimization strategy — [`lo_config_dict`](@ref)), as a plain `Dict`.
Derived from the structs by walking `fieldnames`, so it cannot fall out of
sync with them; a strategy's table loads back through
[`local_optimization_from_toml`](@ref) to an equal value. A strategy whose
table carries a key beyond the generic writer's adds a method.
"""
config_dict(c::RansacConfig) = toml_table(c)
config_dict(lo::AbstractLocalOptimization) = lo_config_dict(lo)

"""
    lo_config_dict(lo) -> Dict{String, Any}

The TOML table of any registered strategy: VisualGeometryToolkitCore's
`toml_table` field walk (the width skipped, Greek spelled ASCII, a field's
own spelling through `toml_field`), its `type` ([`lo_tag`](@ref)), its `loss`
for a loss-parametric family, and its width as `scale_multiple` or
`loss_width` (no width writes nothing).
"""
function lo_config_dict(lo::AbstractLocalOptimization)
    T = typeof(lo)
    d = toml_table(lo)
    d["type"] = lo_tag(T)
    lo_parametric(T) && (d["loss"] = _loss_tag(strategy_loss(lo)))
    _lo_takes_width(T) && _width_to_toml!(d, lo.width)
    return d
end

# The terminal placement: its tag and the inner strategy's own table
# (`config_dict`, so a downstream inner strategy writes its own keys).
lo_config_dict(lo::TerminalLO) = Dict{String, Any}("type" => lo_tag(lo), "inner" => config_dict(lo.inner))

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
# A field's key is its name (Greek spelled ASCII, `toml_key`), every field is
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
"""
    NUISANCE_TAGS

The nuisance-policy registries of the posterior-weight EM, one
VisualGeometryToolkitCore `TagRegistry` per nuisance part, as a named tuple:
`scale` (`"estimated"`, `"fixed_s2"`), `inlier_prior` (`"estimated"`,
`"fixed_prior"`) and `nu` (`"estimated"`, `"fixed_nu"`, `"consensus_dof"`).
"""
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
    fkeys = String[toml_key(T, f) for f in fieldnames(T) if f isa Symbol]
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
        d[toml_key(typeof(x), f::Symbol)] = getfield(x, f)
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
