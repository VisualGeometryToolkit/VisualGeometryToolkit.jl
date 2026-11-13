# =============================================================================
# The core optimization strategies — Irls{K} and Triggs{K} — their loss tags,
# width modes and identity, and the strategy registry machinery every package
# registers its strategies with (moved from VisualGeometryToolkitEstimation, A2:
# PoseLib and the refine/fit verbs of every package take these types).
#
# A strategy here is CONFIGURATION: its loss shape, width mode and knobs. The
# algorithms that run it — the RANSAC local optimization, a fit's reweighting,
# a refine's LM — are the estimators' (`locally_optimize!`, a contract generic).
# Uses: the loss shapes and deploy (losses.jl), the width modes (width_axes.jl),
# the LO contract (lo_contract.jl), TagRegistry (quality/tag_registry.jl), the
# TOML hooks (base/config_toml.jl).
# =============================================================================

# -----------------------------------------------------------------------------
# Loss TOML tags — the one table mapping the canonical VisualGeometryToolkitCore
# loss shapes (robust/losses.jl) that the parametric strategy types
# (`Irls{K}`, `HQ{K}`, `Gnc{S}`) take to their TOML `loss` name
# (`LOSS_TAGS`, below). A shape's loss family and display label
# are VGC's (`loss_family`, `loss_label`, beside the shapes).
#
# Barron's shape carries its α, so `Irls{K}` refuses it at construction
# (`lo_admits_shape` below) — the Barron family runs as
# LocalOptimizationDoneRight's fixed-loss arms.
# -----------------------------------------------------------------------------
const _LOSS_TABLE = (
    (GaussianShape, "gaussian"),
    (WelschShape, "welsch"),
    (GemanMcclureShape, "geman_mcclure"),
    (CauchyShape, "cauchy"),
    (SmoothTLSShape, "smooth_tls"),
    (TLSShape, "tls"),
    (HuberShape, "huber"),
    (BarronShape, "barron"),
)

"""
    LOSS_TAGS

The loss-shape registry (a `TagRegistry`, family suffix `Shape`): TOML `loss`
tag → shape type, read off the one loss-identity table (`_LOSS_TABLE`).
"""
const LOSS_TAGS = TagRegistry(:loss, "Shape", (tag => K for (K, tag) in _LOSS_TABLE)...;
                              paths = [r"^(.*\.)?loss$"])

"""
    Irls{K<:AbstractLossShape} <: AbstractOptimizationStrategy

STATIC IRLS at a frozen loss width — direct iteratively-reweighted least
squares on loss `K`, no lift, no graduation. `Irls{K}` is an LO STRATEGY: one
instance of VisualGeometryToolkitCore's generic [`block_coordinate_descent!`](@ref)
loop, which is not itself "IRLS" (it also runs EM, the half-quadratic lifts and
GNC). Runs the posterior-weight iteration
(VisualGeometryToolkitEstimation's `posterior_irls!`) — the `PosteriorIrlsMethod` /
`PosteriorIrlsState`, same weighted `refine!` M-step (the problem's chart LM), same
best-cost tracking and stop rule — with only the per-iteration weight functor
swapped per loss:

- `Irls{WelschShape}`: `wᵢ = exp(−rᵢ²/loss_width²)`, `loss_width = efficiency_constant·ŝ`
  (the paper's Welsch-IRLS arm; 1.5 was the pre-anchoring point)
- `Irls{HuberShape}`: `wᵢ = min(1, loss_width/|rᵢ|)`, `loss_width = efficiency_constant·ŝ`
  (the elbow; 1.345 is the classical value) — the static-tail control (influence
  constant, never vanishing, past the elbow)
- `Irls{GemanMcclureShape}`: `wᵢ = (loss_width²/(loss_width²+rᵢ²))²`,
  `loss_width = efficiency_constant·ŝ` — the GM static-reweighting arm
- `Irls{GaussianShape}`: unit weights, no scale and NO efficiency constant —
  plain least squares over all points, the coverage extreme. It takes no
  `efficiency_constant`; supplying one is an `ArgumentError` (a width it would
  silently ignore)

The loss width is **frozen** at the start of the call, set by the width mode
`W` (VisualGeometryToolkitCore's `loss_at`) — no per-iteration scale refit:

- `efficiency_constant = m` (`EfficiencyConstant(m)`): `m·ŝ` at the consensus set's
  Student-t scale ŝ — `m` the plain multiple, for every loss;
- `loss_width = w` (`LossWidth(w)`): the absolute width `w`;
- neither (`nothing`, the default): the scorer's decision point
  (`decision_boundary`) for a shape with a decision width — MSAC with
  `Irls{TLSShape}` refits at MSAC's threshold —, and for one without (Huber)
  the table's `efficiency_constant(K, d)` times the consensus scale ŝ, `d` the
  scorer's codimension (a shape with neither is an `ArgumentError`).

# Fields
- `max_iter::Int=25`: max IRLS iterations per hypothesis
- `width::W`: the width mode, `nothing` | `EfficiencyConstant{Float64}` | `LossWidth{NTuple{2,Float64}}`
"""
struct Irls{K <: AbstractLossShape, W} <: AbstractOptimizationStrategy
    max_iter::Int
    width::W
end
Irls{K}(max_iter::Integer, width::W) where {K, W} = Irls{K, W}(Int(max_iter), width)
"""
    Triggs{K<:AbstractLossShape} <: AbstractOptimizationStrategy

The ROBUSTIFIED LEVENBERG–MARQUARDT strategy: from a guess θ, ONE
NLLSsolver.jl LM over all the data minimizing `Σᵢ ρ(‖rᵢ‖²)`, robustified per
datum by loss `K` through [`loss_robustifier`](@ref). `Triggs{K}` is a
strategy (configuration) type only: it selects the loss shape and its width;
how the robustified LM is formed is NLLSsolver's. There is no reweighting loop
and no inlier set inside the solve: θ is the initial guess and the loss decides
each datum's influence. A RANSAC problem's implementation (VisualGeometryToolkitEstimation)
runs it as the local optimization of every chart problem.

The loss width is `loss_width = efficiency_constant·ŝ` (the plain multiple, as for
[`Irls`](@ref)), frozen at the start of
the call: ŝ is the Student-t marginal scale of θ's residuals — the Sampson
errors of all the data — over θ's consensus partition (VisualGeometryToolkitCore's
`partition_student_t`, the scorer's `scale_s2`), the scale every
consensus-anchored strategy reads. With no `efficiency_constant` (`Triggs{K}()`)
inside a RANSAC the width follows the scorer's `decision_boundary`, as for
[`Irls`](@ref); in a refine (VisualGeometryToolkitEstimation's `refine_*`) it is
`efficiency_constant(K, d)` times the caller's `scale`: a refine reads its scale
from the caller, never from a consensus set. An absolute `loss_width` is an
`ArgumentError` from a keyword or a TOML table. An absolute width reaches it
only through `with_width`, for `fixed_width`'s freeze —
`with_width(lo, fixed_width(lo, s2))`, the loss the consensus-anchored strategy
would deploy at ŝ² = `s2` (the role-isolated controls).

`K` is any shape [`Irls`](@ref) admits — the zero-argument shape tags: Welsch
(the pipelines' default), Geman–McClure, Cauchy, smooth TLS, TLS, Huber and
Gaussian; Barron, whose shape carries its α, is refused by both.
`Triggs{GaussianShape}` is the plain chart LM over every datum: like
`Irls{GaussianShape}` it has no width and takes no `efficiency_constant`.

# Fields
- `width`: the width mode, set by `efficiency_constant = c` (`EfficiencyConstant`), or
  `nothing` (no constant stated; always for `Triggs{GaussianShape}`)
"""
struct Triggs{K <: AbstractLossShape, W} <: AbstractOptimizationStrategy
    width::W
end
Triggs{K}(width::W) where {K, W} = Triggs{K, W}(width)
# The weights a strategy's stage reports are its loss's ρ′ (robust/weights.jl): no inlier cut of their own.
weight_kind(::Type{<:Union{Irls, Triggs}}) = IrlsWeights()

"""
    lo_admits_shape(T, K) -> nothing

Construction-time admission of loss shape `K` by strategy type `T` (named in
the message): `K` must be a zero-argument shape tag, so a shape the mechanism
cannot run is refused when the strategy is built, rather than surfacing
mid-RANSAC as a `MethodError` from a VisualGeometryToolkitCore internal the
caller never wrote (`BarronShape()` for a shape that carries its α). The
admissible set is read off VisualGeometryToolkitCore's own traits, never
re-listed. The default of [`lo_admits`](@ref) for a loss-parametric family;
a family with further conditions (a half-quadratic lift, a GNC surrogate)
calls it first. An `ArgumentError` when `K` is not admitted.
"""
function lo_admits_shape(::Type{T}, ::Type{K}) where {T, K}
    applicable(K) || throw(
        ArgumentError(
            "$T: $K is not a zero-argument shape tag (a Barron " *
                "shape carries its α — LocalOptimizationDoneRight's BarronHQ or BarronAdaptive)"
        )
    )
    return nothing
end

# -----------------------------------------------------------------------------
# Identity of the loss-parametric strategies — read off the loss tag
# -----------------------------------------------------------------------------

"""
    strategy_loss(lo_or_type) -> Type{<:AbstractLossShape}

The loss-shape tag of a loss-parametric strategy (`Irls{K}`; downstream
families add methods: LocalOptimizationDoneRight's `HQ{K}`, `Gnc{S}`). Read
off the TYPE, so a wrapper recurses into its inner strategy.
"""
strategy_loss(::Type{<:Irls{K}}) where {K} = K
strategy_loss(::Type{<:Triggs{K}}) where {K} = K
strategy_loss(lo::AbstractOptimizationStrategy) = strategy_loss(typeof(lo))

# Loss-Path labels (Welsch-IRLS, Huber-IRLS, …); marker shape = optimization
# path (VGC path_marker). Sole exception: the bare-L2 arm keeps its historical
# "$\ell_2$" label and :x marker (a degenerate loss, not a loss column) and
# consumes no scale.
# (Cauchy-IRLS beside Cauchy-Triggs: one composition for both paths.)
lo_label(lo::Union{Irls, Triggs}) =
    loss_label(strategy_loss(lo)) * "-" * path_suffix(optimization_path(lo))
lo_label(::Irls{GaussianShape}) = raw"$\ell_2$"
lo_marker(::Irls{GaussianShape}) = :x
loss_family(lo::Union{Irls, Triggs}) = loss_family(strategy_loss(lo))


"""
    lo_loss(lo, cs::ConsensusSet, scoring, buf) -> DeployedLoss

The loss a loss-parametric strategy (`Irls{K}`, and downstream families with a
[`strategy_loss`](@ref)) deploys: its shape at the width its mode resolves to
(VisualGeometryToolkitCore's `loss_at`: the consensus scale times its multiple, an
absolute width, or the scorer's decision point). The
least-squares arm has no scale and no width: it deploys at `s2 = Inf`,
`efficiency_constant = 1` (its weights are unit whatever the width).
"""
lo_loss(lo::AbstractOptimizationStrategy, cs::ConsensusSet, scoring, buf) = loss_at(lo, strategy_loss(lo)(), cs, scoring, buf)
lo_loss(::Union{Irls{GaussianShape}, Triggs{GaussianShape}}, ::ConsensusSet, _scoring, _buf) =
    deploy(GaussianShape(), Inf; efficiency_constant = 1.0)

"""
    fixed_width(lo, s2) -> LossWidth

The absolute width mode of a strategy deployed at `EfficiencyConstant(m)` with the
consensus scale held at `s2`: `LossWidth(m, s2)` — so `with_width(lo, fixed_width(lo, s2))`
deploys exactly the loss the consensus-anchored strategy would at ŝ² = `s2`
(the role-isolated controls: solo CV, oracle scale, the anchoring ablation).
"""
fixed_width(lo::AbstractOptimizationStrategy, s2::Real) =
    LossWidth(_efficiency_constant_of(lo_width(lo)), Float64(s2))
_efficiency_constant_of(w::EfficiencyConstant{Float64}) = w.value
_efficiency_constant_of(w) = throw(ArgumentError("fixed_width: the strategy's width mode is $(w), not a EfficiencyConstant"))


# The width mode of `Irls{K}` and `Triggs{K}` (VisualGeometryToolkitCore's `lo_width`), its
# derived scale-interface label, and the one setter of each: the width mode is a type
# parameter, so `with_width` rebuilds the strategy at a new mode.
# (`lo_width` keeps one method per type: with one Union method the generic has three, and inference splits a
# call on an unknown strategy over all of them, which JET reads as a `nothing` reaching `strategy_loss`.)
lo_width(lo::Irls) = lo.width
lo_width(lo::Triggs) = lo.width
scale_interface(lo::Union{Irls, Triggs}) = width_scale_interface(lo.width)

"""
    lo_width_value(w) -> w

A local optimization's width mode, checked: `nothing` (follow the scorer),
`EfficiencyConstant(m)` or `LossWidth(w)` — never an outlier half-width (a
scorer's). Anything else is an `ArgumentError`.
"""
lo_width_value(::Nothing) = nothing
lo_width_value(w::EfficiencyConstant{Float64}) = w
lo_width_value(w::LossWidth{NTuple{2, Float64}}) = w
lo_width_value(w) = throw(
    ArgumentError(
        "a local optimization's width is EfficiencyConstant(m), LossWidth(w) or nothing (follow the scorer), got $w"
    )
)
with_width(lo::Irls{K}, w) where {K} = Irls{K}(lo.max_iter, lo_width_value(w))
# The robustified LM's width mode — an efficiency constant, always supplied, at the
# keyword and TOML boundary (`lo_field`); `with_width` also takes the absolute
# width `fixed_width` freezes it at, which no caller spells.
with_width(::Triggs{K}, w) where {K} = Triggs{K}(_triggs_width(Triggs{K}, w))
_triggs_width(::Type{T}, w) where {T <: Triggs} = lo_field(T, Val(:width), w)
_triggs_width(::Type{<:Triggs}, w::LossWidth{NTuple{2, Float64}}) = w
_triggs_width(::Type{T}, ::LossWidth{NTuple{2, Float64}}) where {T <: Triggs{GaussianShape}} = _no_width(T)
# The least-squares arm consumes no width (whatever its mode).
scale_interface(::Union{Irls{GaussianShape}, Triggs{GaussianShape}}) = NoScale()

# =============================================================================
# THE strategy machinery — one row of facts per strategy, one keyword
# constructor, one reader and one writer (config_toml.jl), for every package
# =============================================================================
#
# A local-optimization strategy is ONE row of a registry (`@lo_registry`):
# its TOML tag (the `[optimization_strategy.<tag>]` table of its owner's defaults
# its keyword defaults read, and the `type` a config spells it by — the tag
# rule: `snake_case` of the type's name), its optimization path, whether it is
# a loss-parametric family (`Irls{K}`, which takes a `loss` key), and — for
# the named one-off strategies — the identity VisualGeometryToolkitCore's
# `lo_label` / `loss_family` / `scale_interface` report. `nothing` in a column
# means the fact is DERIVED, not stated: a parametric family reads its label
# and loss family off its loss, a fixed-loss arm its loss family off its
# `shape` column, and a width-carrying strategy its scale interface off its
# width mode.
#
# The machinery is this package's and public; a downstream package
# (LocalOptimizationDoneRight) registers its strategies with the same macro,
# against its own defaults table, and states its exceptions as methods of the
# hooks below — never a copy of the machinery. A caller loading both merges
# the registries: `optimization_strategy_from_toml(d; class, tags = merge(OPTIMIZATION_STRATEGY_TAGS, Downstream.OPTIMIZATION_STRATEGY_TAGS))`.
#
# The keyword constructor is driven by `fieldnames` and the tag: a field's
# keyword is the field's name; its default is the strategy's table key
# (`lo_key`, ASCII-spelled); a field with NO key in the table is REQUIRED.
# The width field is set by `efficiency_constant = m` (`EfficiencyConstant(m)`) or
# `loss_width = w` (`LossWidth(w)`) — both is an `ArgumentError` — and by
# neither follows the scorer (`nothing`). An unknown keyword is an
# `ArgumentError`. The per-strategy exceptions are the hooks, each stated once:
#   `lo_admits(T, K)`                      construction-time admission, BEFORE any field (K: a family's parameter)
#   `lo_key(T, Val(f))`                    a field's default read from another key
#   `lo_field(T, Val(f), v)`               a supplied value's conversion / refusal
#   `lo_field_from_toml(T, Val(f), v)`     a TOML value's conversion (reader and defaults)
#   `lo_absent(T, Val(f); defaults)`       a field's value when the caller omits it
#   `lo_validate(lo)`                      cross-field checks on the built value
#   `lo_concrete(T, d, section; tags)`     the concrete type a TOML table names
#   `lo_efficiency_constant_shape(T)`           the shape a `efficiency_constant` is read for
#   `lo_resolve_efficiency_constant(T, K, v; class)`   a TOML `efficiency_constant` → a number

"""
    lo_tag(T) -> String
    lo_tag(lo) -> String

The TOML tag of a registered strategy type or loss-parametric family (its
`@lo_registry` row): the `type` a config spells it by and the
`[optimization_strategy.<tag>]` table its keyword defaults read.
"""
function lo_tag end
lo_tag(lo::AbstractOptimizationStrategy) = lo_tag(typeof(lo))

"""
    lo_defaults(T) -> AbstractDict

The `[optimization_strategy]` table of the defaults file of the package that
registered `T` (keyed by tag; `@lo_registry`'s `defaults`): what
[`lo_construct`](@ref) reads an omitted field from.
"""
function lo_defaults end

"""
    lo_parametric(T) -> Bool

Whether `T` is a loss-parametric family (`Irls{K}`): its TOML table names its
loss (`loss` key, [`LOSS_TAGS`](@ref)) and its type names the loss through
[`strategy_loss`](@ref). `false` unless its `@lo_registry` row says so.
"""
lo_parametric(::Type) = false

"""
    lo_admits(T, K) -> nothing

Construction-time admission of strategy type `T`, called before any field is
read. `K` is the type parameter of a loss-parametric family — the loss `K` of
`Irls{K}`, or the inner strategy of a wrapper (LocalOptimizationDoneRight's
`Gnc{S}`) — and `nothing` for any other strategy. The default admits every
non-parametric strategy and checks a family's shape with
[`lo_admits_shape`](@ref); a family with further conditions adds a method
(LocalOptimizationDoneRight's half-quadratic lift and GNC surrogate checks).
Throws `ArgumentError`.
"""
lo_admits(::Type, ::Nothing) = nothing
lo_admits(::Type{T}, ::Type{K}) where {T, K} = lo_admits_shape(T, K)
# The type parameter a loss-parametric family is built at (`K` of `Irls{K}`).
_family_parameter(::Type{T}) where {T} = first(Base.unwrap_unionall(T).parameters)

"""
    lo_key(T, Val(f)) -> String

The key of the defaults table field `f` of strategy `T` defaults from.
Default: VisualGeometryToolkitCore's `toml_key(T, f)` (Greek spelled ASCII). A
strategy whose field defaults from another key adds a method
(LocalOptimizationDoneRight's smooth-TLS lift, `max_iter_smooth_tls`).
"""
lo_key(::Type{T}, ::Val{F}) where {T, F} = toml_key(T, F)

"""
    lo_field(T, Val(f), v)

Field `f` of strategy `T` from a value the caller supplied by keyword.
Default: `convert(fieldtype(T, f), v)`; the width field goes through
[`lo_width_value`](@ref).
A downstream method is per field (`::Val{:x}`): a method on `Val{F}` for every
field of its type is ambiguous with the `Val{:width}` catch-all below.
"""
lo_field(::Type{T}, ::Val{F}, v) where {T, F} = convert(fieldtype(T, F), v)
lo_field(::Type, ::Val{:width}, w) = lo_width_value(w)
lo_field(::Type{<:Triggs}, ::Val{:width}, w::EfficiencyConstant{Float64}) = w
lo_field(::Type{T}, ::Val{:width}, w) where {T <: Triggs} = throw(ArgumentError(
    "$T: the loss width is efficiency_constant·ŝ — give `efficiency_constant`, got $w"))
# The plain LM has no width at all.
lo_field(::Type{T}, ::Val{:width}, w::EfficiencyConstant{Float64}) where {T <: Triggs{GaussianShape}} = _no_width(T)
lo_field(::Type{T}, ::Val{:width}, w) where {T <: Triggs{GaussianShape}} = w === nothing ? nothing : _no_width(T)
_no_width(T) = throw(ArgumentError("$T is the plain chart LM: it has no width, so it takes no `efficiency_constant` or `loss_width`"))

"""
    lo_field_from_toml(T, Val(f), v)

Field `f` of strategy `T` from a TOML value — a key of the caller's table or of
the defaults table. Default: VisualGeometryToolkitCore's
`convert_field(fieldtype(T, f), v)`. A field whose TOML spelling differs from
its value adds a method (LocalOptimizationDoneRight's GNC `entry`, `"r_max"`
for `NaN`).
"""
lo_field_from_toml(::Type{T}, ::Val{F}, v) where {T, F} = convert_field(fieldtype(T, F), v)

"""
    lo_absent(T, Val(f); defaults)

Field `f` of strategy `T` when the caller omits it: the strategy's
`defaults[lo_tag(T)]` key [`lo_key`](@ref)`(T, Val(f))`, converted by
[`lo_field_from_toml`](@ref); with no such key the field is required
(`UndefKeywordError`). The width field defaults to `nothing` (follow the
scorer).
A downstream method is per field (`::Val{:x}`): a method on `Val{F}` for every
field of its type is ambiguous with the `Val{:width}` catch-all below.
"""
function lo_absent(::Type{T}, ::Val{F}; defaults::AbstractDict) where {T, F}
    lo_has_default(T, F, defaults) || throw(UndefKeywordError(F))
    return lo_field_from_toml(T, Val(F), _lo_table(T, defaults)[lo_key(T, Val(F))])
end
lo_absent(::Type, ::Val{:width}; defaults::AbstractDict) = nothing

"""
    lo_validate(lo) -> lo

Cross-field checks on a freshly built strategy; returns it or throws. Default:
none.
"""
lo_validate(lo) = lo

# The strategy's table of the defaults (empty for a strategy with no fields).
_lo_table(::Type{T}, defaults::AbstractDict) where {T} = get(defaults, lo_tag(T), Dict{String, Any}())
"""
    lo_has_default(T, f::Symbol, defaults::AbstractDict) -> Bool

Whether strategy `T`'s table of `defaults` (keyed by its [`lo_tag`](@ref))
holds a default for field `f` (at its key [`lo_key`](@ref)).
"""
lo_has_default(::Type{T}, f::Symbol, defaults::AbstractDict) where {T} =
    haskey(_lo_table(T, defaults), lo_key(T, Val(f)))

"""
    lo_construct(T; defaults = lo_defaults(T), kw...) -> T

THE keyword constructor of every registered strategy (`@lo_registry` defines
`T(; kw...)` as this call): each field is its keyword or, omitted, its
`defaults` value ([`lo_absent`](@ref)); the width is spelled `efficiency_constant`
or `loss_width` (both is an `ArgumentError`; neither follows the scorer). The
hooks [`lo_admits`](@ref), [`lo_field`](@ref) and [`lo_validate`](@ref) state
the per-strategy exceptions. `defaults` is the `[optimization_strategy]` table,
keyed by tag. An unknown keyword is an `ArgumentError`.
"""
function lo_construct(::Type{T}; defaults::AbstractDict = lo_defaults(T), kw...) where {T <: AbstractOptimizationStrategy}
    lo_admits(T, lo_parametric(T) ? _family_parameter(T) : nothing)
    kw = _lo_width_keywords(T, kw)
    for f in keys(kw)
        f in fieldnames(T) || throw(
            ArgumentError(
                "$T has no keyword `$f` (its keywords: $(join(_lo_keywords(T), ", ")))"
            )
        )
    end
    vals = ntuple(fieldcount(T)) do i
        f = fieldname(T, i)
        haskey(kw, f) ? lo_field(T, Val(f), kw[f]) : lo_absent(T, Val(f); defaults)
    end
    return lo_validate(T(vals...))
end

"""
    lo_takes_width(T) -> Bool

Whether strategy type `T` takes a width (the `efficiency_constant` /
`loss_width` keys): iff it has a `width` field.
"""
lo_takes_width(::Type{T}) where {T} = :width in fieldnames(T)

# The keywords a caller spells: the fields, the width field as its two keys.
_lo_keywords(::Type{T}) where {T} =
    lo_takes_width(T) ? [filter(!=(:width), collect(fieldnames(T))); :efficiency_constant; :loss_width] :
    collect(fieldnames(T))

# `efficiency_constant = m` / `loss_width = w` → the `width` field.
function _lo_width_keywords(::Type{T}, kw) where {T}
    lo_takes_width(T) || return kw
    hm, hw = haskey(kw, :efficiency_constant), haskey(kw, :loss_width)
    hm && hw && throw(ArgumentError("$T: give `efficiency_constant` or `loss_width`, not both"))
    hm || hw || return kw
    rest = Dict{Symbol, Any}(k => v for (k, v) in kw if k ∉ (:efficiency_constant, :loss_width))
    rest[:width] = hm ? _lo_efficiency_constant(T, kw[:efficiency_constant]) : LossWidth(Float64(kw[:loss_width]))
    return rest
end
_lo_efficiency_constant(::Type, m) = EfficiencyConstant(Float64(m))
# Least squares has no efficiency constant; one would be a width it silently ignores.
_lo_efficiency_constant(::Type{<:Irls{GaussianShape}}, _) = throw(
    ArgumentError(
        "Irls{GaussianShape} is plain least squares: it has no efficiency constant, " *
            "so it takes no `efficiency_constant`"
    )
)
_lo_efficiency_constant(::Type{T}, _) where {T <: Triggs{GaussianShape}} = _no_width(T)

"""
    lo_efficiency_constant_shape(T) -> Type{<:AbstractLossShape}

The loss shape a TOML `efficiency_constant` of strategy `T` is read for (and a
named one, `"cross_validated"`, tuned for): a loss-parametric family's
[`strategy_loss`](@ref), a fixed-loss arm's `@lo_registry` `shape` column. An
`ArgumentError` for a strategy that has neither.
"""
lo_efficiency_constant_shape(::Type{T}) where {T} =
    lo_parametric(T) ? strategy_loss(T) :
    throw(ArgumentError("$(nameof(T)) takes no `efficiency_constant` (it has no width)"))

"""
    @lo_registry rows defaults = table [retired = (old => new, …)]

Register the local-optimization strategies of `rows` and return their
VisualGeometryToolkitCore `TagRegistry` (family `:optimization_strategy`, the TOML
path `local_optimization_strategy.type`): `const OPTIMIZATION_STRATEGY_TAGS = @lo_registry ROWS defaults = …`.
Each row is

    (T, tag, path, scale_interface, loss_family, label, shape, parametric)

with `nothing` for a fact that is derived rather than stated. `defaults` is an
expression evaluated, in the calling module, each time a default is read: the
`[optimization_strategy]` table of the caller's defaults file, keyed by tag.
`retired` lists retired tags with their successors; `none = true` adds the tag `"none"` → `nothing` (no local
optimization; this package's registry holds it, so a merge with a downstream one holds it once).

The expansion is one call, made at load time in the calling module. Per row it
defines [`lo_tag`](@ref), [`lo_defaults`](@ref), VisualGeometryToolkitCore's
`toml_skip_fields` and, for a stated path, `optimization_path` (the width is written as its two
keys, never as `width`); for a parametric row [`lo_parametric`](@ref); for a
stated scale interface, loss family or label the Core accessor; for a stated
`shape` [`lo_efficiency_constant_shape`](@ref) and the loss family of that shape;
and, for a strategy with fields, the keyword constructor `T(; kw...)`
(`T{K}(; kw...)` for a parametric family) as [`lo_construct`](@ref). A macro
rather than a function because it defines methods in the caller's module.
Every method it adds is on a declared contract generic.
"""
macro lo_registry(rows, options...)
    opts = Dict{Symbol, Any}()
    for o in options
        Meta.isexpr(o, :(=), 2) && o.args[1] in (:defaults, :retired, :none) ||
            throw(ArgumentError("@lo_registry rows defaults = … [retired = …] [none = true]; got `$o`"))
        opts[o.args[1]] = o.args[2]
    end
    haskey(opts, :defaults) || throw(ArgumentError("@lo_registry needs `defaults = …` (the [optimization_strategy] table)"))
    return :(_register_lo_rows($__module__, $(esc(rows)), $(QuoteNode(opts[:defaults]));
                               retired = $(esc(get(opts, :retired, :(())))), none = $(esc(get(opts, :none, false)))))
end

function _register_lo_rows(M::Module, rows, defaults; retired = (), none::Bool = false)
    E, C = @__MODULE__, VisualGeometryToolkitCore
    for (T, tag, path, iface, family, label, shape, parametric) in rows
        Core.eval(M, quote
            $E.lo_tag(::Type{<:$T}) = $tag
            $E.lo_defaults(::Type{<:$T}) = $defaults
            $C.toml_skip_fields(::Type{<:$T}) = (:width,)
        end)
        path === nothing || Core.eval(M, :($C.optimization_path(::$T) = $(QuoteNode(path))))
        parametric && Core.eval(M, :($E.lo_parametric(::Type{<:$T}) = true))
        iface === nothing || Core.eval(M, :($C.scale_interface(::$T) = $iface))
        family === nothing || Core.eval(M, :($C.loss_family(::$T) = $(QuoteNode(family))))
        label === nothing || Core.eval(M, :($C.lo_label(::$T) = $label))
        shape === nothing || Core.eval(M, quote
            $E.lo_efficiency_constant_shape(::Type{<:$T}) = $shape
            $C.loss_family(::$T) = $C.loss_family($shape)
        end)
        fieldcount(T) == 0 && continue          # no fields: the default constructor is it
        if parametric
            Core.eval(M, :((::Type{$T{P}})(; kw...) where {P} = $E.lo_construct($T{P}; kw...)))
        else
            Core.eval(M, :((::Type{$T})(; kw...) = $E.lo_construct($T; kw...)))
        end
    end
    return TagRegistry(
        :optimization_strategy, "",
        (none ? ("none" => nothing,) : ())...,   # no local optimization: `nothing` (this package's registry only)
        (tag => T for (T, tag) in rows)...;
        retired, paths = [r"^(.*\.)?(local_)?optimization_strategy(\.inner)*\.type$"]
    )
end


# -----------------------------------------------------------------------------
# The core strategies' registry, its defaults this package's configs/default.toml
# -----------------------------------------------------------------------------

#   type      tag       path     scale    family   label    shape    parametric
const _CORE_STRATEGY_ROWS = (
    (Irls,    "irls",   :irls,   nothing, nothing, nothing, nothing, true),
    (Triggs,  "triggs", :triggs, nothing, nothing, nothing, nothing, true),
)

"""
    STRATEGY_TAGS

The core optimization strategies' registry (a `TagRegistry` built by
[`@lo_registry`](@ref)): `"irls"` → [`Irls`](@ref), `"triggs"` →
[`Triggs`](@ref). A package with strategies of its own merges it with its
registry (VisualGeometryToolkitEstimation's `OPTIMIZATION_STRATEGY_TAGS`).
"""
const STRATEGY_TAGS = @lo_registry _CORE_STRATEGY_ROWS defaults = _CORE_DEFAULTS["optimization_strategy"]
