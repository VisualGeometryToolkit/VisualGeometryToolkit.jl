# =============================================================================
# Scoring glue — the RANSAC-harness side of the local-optimization contract
# =============================================================================
#
# The scoring contract, the local-optimization contract and the robust-
# statistics toolkit are VisualGeometryToolkitCore's. This file holds the two
# local-optimization strategies of the core — `NoLocalOptimization` and the
# static M-estimator `Irls{K}` — with their width modes, the weight-policy
# adapters of the posterior-weight iteration (`PosteriorLikelihood`,
# `LossPlainWeights`) and the strategy registry with its keyword constructor.
# The paper's strategies (the half-quadratic lifts, GNC, the Student-t +
# uniform IRLS, the score-matched BCD, the Barron arms) are
# LocalOptimizationDoneRight's; they build on the public pieces here.
# =============================================================================

# -----------------------------------------------------------------------------
# Local-optimization strategies (supplied to `ransac()`, dispatched in the loop)
# -----------------------------------------------------------------------------

"""
    NoLocalOptimization <: AbstractLocalOptimization

No local optimization — the candidate model is scored directly without refit.
Default for all scoring functions (via `default_local_optimization`).
"""
struct NoLocalOptimization <: AbstractLocalOptimization end

# -----------------------------------------------------------------------------
# Loss TOML tags — the one table mapping the canonical VisualGeometryToolkitCore
# loss shapes (robust/losses.jl) that the parametric strategy types
# (`Irls{K}`, `HQ{K}`, `Gnc{S}`) take to their TOML `loss` name
# (config_toml.jl's LOSS_TAGS). A shape's loss family and display label
# are VGC's (`loss_family`, `loss_label`, beside the shapes).
#
# Barron's shape carries its α, so `Irls{K}` refuses it at construction
# (`_mechanism_admits_shape` below) — the Barron family runs as
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
    Irls{K<:AbstractLossShape} <: AbstractLocalOptimization

STATIC IRLS at a frozen loss width — direct iteratively-reweighted least
squares on loss `K`, no lift, no graduation. `Irls{K}` is an LO STRATEGY: one
instance of VisualGeometryToolkitCore's generic [`block_coordinate_descent!`](@ref)
loop, which is not itself "IRLS" (it also runs EM, the half-quadratic lifts and
GNC). Runs the posterior-weight iteration
([`posterior_irls!`](@ref)) — the `PosteriorIrlsMethod` /
`PosteriorIrlsState`, same weighted `refine!` M-step (the problem's chart LM), same
best-cost tracking and stop rule — with only the per-iteration weight functor
swapped per loss:

- `Irls{WelschShape}`: `wᵢ = exp(−rᵢ²/loss_width²)`, `loss_width = scale_multiple·ŝ`
  (the paper's Welsch-IRLS arm; 1.5 was the pre-anchoring point)
- `Irls{HuberShape}`: `wᵢ = min(1, loss_width/|rᵢ|)`, `loss_width = scale_multiple·ŝ`
  (the elbow; 1.345 is the classical value) — the static-tail control (influence
  constant, never vanishing, past the elbow)
- `Irls{GemanMcclureShape}`: `wᵢ = (loss_width²/(loss_width²+rᵢ²))²`,
  `loss_width = decision_width(GM, scale_multiple·ŝ)` — the GM static-reweighting
  arm; its `scale_multiple` is in the half-weight currency
  ([`plain_scale_multiple`](@ref), the one currency table), so one CV grid
  transfers across the static and lifted mechanisms (half-weight depth of a
  Welsch point: `scale_multiple·√log2`)
- `Irls{GaussianShape}`: unit weights, no scale and NO scale multiple —
  plain least squares over all points, the coverage extreme. It takes no
  `scale_multiple`; supplying one is an `ArgumentError` (a width it would
  silently ignore)

The loss width is **frozen** at the start of the call, set by the width mode
`W` (VisualGeometryToolkitCore's `loss_at`) — no per-iteration scale refit:

- `scale_multiple = m` (`ScaleMultiple(m)`): `m·ŝ` at the consensus set's
  Student-t scale ŝ, in the loss's currency above;
- `loss_width = w` (`LossWidth(w)`): the absolute width `w`;
- neither (`nothing`, the default): the scorer's decision point
  (`inlier_threshold`) — MSAC with `Irls{TLSShape}` refits at MSAC's threshold.

# Fields
- `max_iter::Int=25`: max IRLS iterations per hypothesis
- `width::W`: the width mode, `nothing` | `ScaleMultiple{Float64}` | `LossWidth{NTuple{2,Float64}}`
"""
struct Irls{K <: AbstractLossShape, W} <: AbstractLocalOptimization
    max_iter::Int
    width::W
end
Irls{K}(max_iter::Integer, width::W) where {K, W} = Irls{K, W}(Int(max_iter), width)
# Construction-time admission: a shape the mechanism
# cannot run is refused HERE, rather than surfacing mid-RANSAC as a MethodError
# from a VGC internal the caller never wrote (`BarronShape()` for a shape that
# carries its α; `zblock(::TLSShape, …)` for one with no lift). The
# admissible set is read off VGC's own traits, never re-listed here.
# `strategy` names the strategy being built (`"Irls{…}"`) for the message.
function _mechanism_admits_shape(strategy, ::Type{K}) where {K}
    applicable(K) || throw(
        ArgumentError(
            "$strategy: $K is not a zero-argument shape tag (a Barron " *
                "shape carries its α — LocalOptimizationDoneRight's BarronHQ or BarronAdaptive)"
        )
    )
    return nothing
end
"""
    consensus_s2!(ws::RansacWorkspace, problem, scoring, model) -> Union{Float64, Nothing}

The consensus scale ŝ² of `model` under `scoring`: its residuals and score on
the workspace's buffers, then the consensus set's Student-t scale at a local
optimizer's dof (`minimal_sample_size + 1`) — the scale a consensus-anchored
loss runs at in the native loop (`scale_s2(consensus_set(…))`), floored at eps.

THE one scale source for engines that cannot run in the native loop (an
external refiner's adaptive-width arm): never a loss-width-window plug-in.
Returns `nothing` on a consensus set too thin to carry a scale
(k* ≤ m + 1) — the caller bails with no refit call.
"""
function consensus_s2!(ws::RansacWorkspace, problem, scoring, model)
    _prepare!(problem)
    buf = ws.buf
    _, k = score_model!(buf, scoring, problem, model)
    m = minimal_sample_size(problem) + 1
    k > m || return nothing                                          # thin set — no stable ŝ²
    s2 = consensus_s2(scoring, buf.residuals, consensus_order(scoring, buf.scoring_ws), k;
                      model_dof = m, d_g = codimension(problem))
    return max(s2, eps())
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
strategy_loss(lo::AbstractLocalOptimization) = strategy_loss(typeof(lo))

# Loss-Path labels (Welsch-IRLS, Huber-IRLS, …); marker shape = optimization
# path (VGC path_marker). Sole exception: the bare-L2 arm keeps its historical
# "$\ell_2$" label and :x marker (a degenerate loss, not a loss column) and
# consumes no scale.
VgtkCore.lo_label(lo::Irls) =
    loss_label(strategy_loss(lo)) * "-" * path_suffix(optimization_path(lo))
VgtkCore.lo_label(::Irls{GaussianShape}) = raw"$\ell_2$"
VgtkCore.lo_marker(::Irls{GaussianShape}) = :x
VgtkCore.loss_family(lo::Irls) = loss_family(strategy_loss(lo))

# -----------------------------------------------------------------------------
# The scale-multiple CURRENCY — ONE table, `plain_scale_multiple(shape, m)`
# -----------------------------------------------------------------------------
# A loss's `scale_multiple` is stated in the LOSS's currency, so one CV grid
# transfers across the static and lifted mechanisms: `Irls{K}` and the lifts
# (LocalOptimizationDoneRight's `HQ{K}`) read the same `scale_multiple` the
# same way. `plain_scale_multiple` converts it to the plain multiple a
# deployment's `scale_multiple` slot takes (loss width `scale_multiple·ŝ`,
# `sq_loss_width` `scale_multiple²·ŝ²`):
#
#   width multiple     loss_width = scale_multiple·ŝ        Welsch, Huber (the
#                                                            elbow), TLS and
#                                                            smooth TLS (the cut)
#   half-weight depth  w(r) = ½ at r = scale_multiple·ŝ     GM, Cauchy (VGC
#                      (`decision_width`)                   `decision_width`, linear
#                                                            in ŝ, so at ŝ = 1 it IS
#                                                            the multiple)
#
# Smooth TLS's former `scale_multiple ≤ 0` sentinel — the scorer's
# inlier/outlier crossover — is the follow-the-scorer width mode (no width), so
# a non-positive multiple is refused.

"""
    plain_scale_multiple(shape, m::Float64) -> Float64

The plain scale multiple a deployment of `shape` takes for a `scale_multiple`
`m` stated in the shape's currency: the identity for a width-multiple loss
(Welsch, Huber, TLS, smooth TLS), `decision_width(shape, m)` for a half-weight
loss (Geman–McClure, Cauchy). A non-positive multiple of smooth TLS is refused
(the retired crossover sentinel: give no width to follow the scorer).
"""
plain_scale_multiple(::AbstractLossShape, m::Float64) = m
plain_scale_multiple(shape::Union{GemanMcclureShape, CauchyShape}, m::Float64) = decision_width(shape, m)
function plain_scale_multiple(::SmoothTLSShape, m::Float64)
    m > 0 || throw(
        ArgumentError(
            "scale_multiple = $m: the crossover sentinel is retired — give no width to " *
                "follow the scorer's inlier_threshold"
        )
    )
    return m
end

# VisualGeometryToolkitCore's `deployed_scale_multiple` (the LO contract's currency
# hook, read by `loss_at` under a `ScaleMultiple` width mode).
VgtkCore.deployed_scale_multiple(::Irls, shape, m::Float64, _s2, _scoring) = plain_scale_multiple(shape, m)

"""
    lo_loss(lo, cs::ConsensusSet, scoring, buf) -> DeployedLoss

The loss a loss-parametric strategy (`Irls{K}`, and downstream families with a
[`strategy_loss`](@ref)) deploys: its shape at the width its mode resolves to
(VisualGeometryToolkitCore's `loss_at`: the consensus scale times its multiple in the
shape's currency, an absolute width, or the scorer's decision point). The
least-squares arm has no scale and no width: it deploys at `s2 = Inf`,
`scale_multiple = 1` (its weights are unit whatever the width).
"""
lo_loss(lo::AbstractLocalOptimization, cs::ConsensusSet, scoring, buf) = loss_at(lo, strategy_loss(lo)(), cs, scoring, buf)
lo_loss(::Irls{GaussianShape}, ::ConsensusSet, _scoring, _buf) = deploy(GaussianShape(), Inf; scale_multiple = 1.0)

"""
    deployed_shape(lo) -> AbstractLossShape | nothing

The loss shape a width-carrying strategy deploys, as `fixed_width` hands it to
the currency hook (`deployed_scale_multiple`): `strategy_loss(lo)()` for a
loss-parametric strategy, `nothing` (the default) for a strategy whose currency
reads no shape. A downstream fixed-loss arm adds its method.
"""
deployed_shape(::AbstractLocalOptimization) = nothing
deployed_shape(lo::Irls) = strategy_loss(lo)()

"""
    fixed_width(lo, s2) -> LossWidth

The absolute width mode of a strategy deployed at `ScaleMultiple(m)` with the
consensus scale held at `s2`: `LossWidth(m′, s2)`, `m′` the plain multiple of
`m` in the strategy's currency — so `with_width(lo, fixed_width(lo, s2))`
deploys exactly the loss the consensus-anchored strategy would at ŝ² = `s2`
(the role-isolated controls: solo CV, oracle scale, the anchoring ablation).
"""
fixed_width(lo::AbstractLocalOptimization, s2::Real) =
    LossWidth(deployed_scale_multiple(lo, deployed_shape(lo), _scale_multiple_of(lo_width(lo)), Float64(s2), nothing), Float64(s2))
_scale_multiple_of(w::ScaleMultiple{Float64}) = w.value
_scale_multiple_of(w) = throw(ArgumentError("fixed_width: the strategy's width mode is $(w), not a ScaleMultiple"))

"""
    default_local_optimization(scoring::AbstractScoring) -> AbstractLocalOptimization

Default local-optimization strategy for a scoring function: `NoLocalOptimization()`.
"""
default_local_optimization(::AbstractScoring) = NoLocalOptimization()

# -----------------------------------------------------------------------------
# PosteriorLikelihood — AbstractWeightPolicy adapter for the `block_coordinate_descent!` loop
# -----------------------------------------------------------------------------

"""
    PosteriorLikelihood(rws::PosteriorWeightState, n::Int;
                        scale_update = Estimated(), prior_update = FixedPrior(inlier_prior(rws)),
                        nu_update = Estimated()) <: NormalizedWeightPolicy

Likelihood adapter for [`block_coordinate_descent!`](@ref). Wraps a
VisualGeometryToolkitCore `PosteriorWeightState` with its data size `n` so the generic `block_coordinate_descent!`
loop can call `update_nuisance!` / `update_weights!` without problem-specific
detail.

The nuisance is a composite block of three parts, each with its own update
policy (VisualGeometryToolkitCore's): the scale ŝ² (`Estimated()` or `FixedS2(s2)`),
the inlier prior π (`Estimated()` or `FixedPrior(π)`) and the Student-t dof ν
(`Estimated()`, `FixedNu(ν)` or `ConsensusDof()`). The
defaults are the Student-t + uniform posterior of the RANSAC local
optimization: ŝ² and ν estimated, π held at the seeded state's own value
(1/2). A held part takes its value at the first nuisance update.
"""
struct PosteriorLikelihood{S, P, N} <: NormalizedWeightPolicy
    rws::PosteriorWeightState
    n::Int
    scale_update::S
    prior_update::P
    nu_update::N
end
PosteriorLikelihood(rws::PosteriorWeightState, n::Int;
                    scale_update = Estimated(), prior_update = FixedPrior(inlier_prior(rws)),
                    nu_update = Estimated()) =
    PosteriorLikelihood(rws, n, scale_update, prior_update, nu_update)

# The nuisance here is a composite block with three parts, all VGC's to move,
# each under its policy: `update_scale!(pws, n, policy)` (ŝ² from the weighted
# RSS, or held), then `update_nu!(pws, n, policy)` (ν at that new ŝ² — in that
# order, since ν reads ŝ²), then `update_inlier_prior!(pws, n, policy)` (π from
# the E-step's posteriors, which neither of the others writes). All three are
# VisualGeometryToolkitCore's toolkit functions, DIFFERENT from the verb defined here, and ARITY is the
# whole distinction: `block_coordinate_descent!` drives the 1-arg policy sweep
# `update_nuisance!(likelihood)`, which this method supplies by calling the
# parts. (Subtyping does not separate them — `PosteriorWeightState` IS a
# `NormalizedWeightPolicy`, which it subtypes for the likelihood-grade
# `is_inlier`/`inlier_mask` defaults, not to be driven by the loop.)
# The boundary is asymmetric on purpose: `update_weights!` below stays the one
# shared VGC generic that RSDR extends at 2-arg, so it needs no qualification;
# the part updates are RSDR's own, and are qualified so these cross-package
# calls read as deliberate rather than as a rename that was missed.
function update_nuisance!(lh::PosteriorLikelihood)
    update_scale!(lh.rws, lh.n, lh.scale_update)
    update_nu!(lh.rws, lh.n, lh.nu_update)
    update_inlier_prior!(lh.rws, lh.n, lh.prior_update)
    return lh
end
update_weights!(lh::PosteriorLikelihood) = update_weights!(lh.rws, lh.n)
current_s2(lh::PosteriorLikelihood) = s_hat_sq(lh.rws)   # the posterior's ŝ² (VGC `Recorded`)

# `median_seeded_posterior` is VisualGeometryToolkitCore's: a PosteriorWeightState
# constructor, so it lives beside them.

"""
    LossPlainWeights(rws::PosteriorWeightState, n::Int, loss::DeployedLoss)

Frozen-scale plain-IRLS likelihood adapter for [`Irls{K}`](@ref Irls): ONE
weight policy generic over the loss, replacing the former per-loss
Welsch/Geman–McClure/Huber/L2 quadruplet whose bodies differed only in the
weight formula — which is exactly `VisualGeometryToolkitCore.weight` on the shape.  Reuses
the `PosteriorWeightState` buffers (`sq_residuals`, `w`). The loss carries
its OWN squared width (`sq_loss_width`, Huber: elbow²), committed at construction
from the seed partition's Student-t scale and **frozen** — `update_nuisance!` is
a no-op.
"""
struct LossPlainWeights{L <: DeployedLoss} <: AbstractWeightPolicy
    rws::PosteriorWeightState
    n::Int
    loss::L        # shape + (scale_multiple, ŝ²); `sq_loss_width` = scale_multiple²·ŝ², never stored
end

update_nuisance!(::LossPlainWeights) = nothing   # scale FROZEN at the seed
current_s2(lh::LossPlainWeights) = scale_s2(lh.loss)   # the scale the loss runs at (VGC `Recorded`)
function update_weights!(lh::LossPlainWeights)
    rws = lh.rws
    # Read the shape and its width ONCE. `sq_loss_width` is
    # `scale_multiple²·ŝ²`, computed from the two stored slots, so calling it
    # per point re-multiplies every iteration for a value that is frozen at
    # construction by definition.
    shape = loss_shape(lh.loss)
    width2 = sq_loss_width(lh.loss)
    @inbounds for i in 1:lh.n
        robust_weight(rws)[i] = weight(shape, sq_residuals(rws)[i], width2)
    end
    return nothing
end
# --- VGC's weight-policy reading contract ------------------------------------
# Both policies below wrap the SAME `PosteriorWeightState` buffers, and WHICH
# quantity `robust_weight(rws)` holds depends on the policy: the mixture fills it with
# `inlier_posterior * u`, `LossPlainWeights` overwrites it with the loss's psi(r)/r.
# That is precisely why these are typed verbs and not a field access.
inlier_weight(lh::PosteriorLikelihood) = view(inlier_weight(lh.rws), 1:lh.n)
robust_weight(lh::PosteriorLikelihood) = view(robust_weight(lh.rws), 1:lh.n)

# An M-estimator: one weight does both jobs. `inlier_weight(rws)` on this state is STALE
# seed data — never read it here.
inlier_weight(lh::LossPlainWeights) = view(robust_weight(lh.rws), 1:lh.n)
robust_weight(lh::LossPlainWeights) = view(robust_weight(lh.rws), 1:lh.n)

# There is no membership to threshold, so `is_inlier` states the loss's own
# rule — but only a loss that REDESCENDS has a rule to state. VGC publishes
# that axis as `bounded(shape)`, so route through it rather than naming the
# exceptions one at a time:
#
#   bounded == true  (Welsch, GM, TLS, …) — inside the loss width frozen at
#     construction (`sq_loss_width`, from the seed partition's Student-t scale).
#   bounded == false (l2 AND Huber)       — nothing is rejected, so every
#     point counts. Their width is a placeholder that the frozen-scale
#     machinery needs only to stay generic over the loss; l2 is deployed at
#     `1.0` and Huber's elbow bounds the quadratic CORE, not the influence.
#     Thresholding r² against either would invent a consensus set in whatever
#     units the problem happens to use — 1 px² for a homography, ~1 rad² for a
#     calibrated essential.
#
# `bounded(lh.loss)` folds at compile time (the shape is a type parameter),
# so this costs nothing at runtime and — unlike a per-shape exemption method —
# cannot go stale when a shape is added.
is_inlier(lh::LossPlainWeights, i::Integer) =
    !bounded(lh.loss) || @inbounds(sq_residuals(lh.rws)[i] <= sq_loss_width(lh.loss))

# The width mode of `Irls{K}` (VisualGeometryToolkitCore's `lo_width`), its derived
# scale-interface label, and the one setter: the width mode is a type
# parameter, so `with_width` rebuilds the strategy at a new mode.
VgtkCore.lo_width(lo::Irls) = lo.width
VgtkCore.scale_interface(lo::Irls) = width_scale_interface(lo.width)

"""
    lo_width_value(w) -> w

A local optimization's width mode, checked: `nothing` (follow the scorer),
`ScaleMultiple(m)` or `LossWidth(w)` — never an outlier half-width (a
scorer's). Anything else is an `ArgumentError`.
"""
lo_width_value(::Nothing) = nothing
lo_width_value(w::ScaleMultiple{Float64}) = w
lo_width_value(w::LossWidth{NTuple{2, Float64}}) = w
lo_width_value(w) = throw(
    ArgumentError(
        "a local optimization's width is ScaleMultiple(m), LossWidth(w) or nothing (follow the scorer), got $w"
    )
)
VgtkCore.with_width(lo::Irls{K}, w) where {K} = Irls{K}(lo.max_iter, lo_width_value(w))
# The least-squares arm consumes no width (whatever its mode).
VgtkCore.scale_interface(::Irls{GaussianShape}) = NoScale()

# =============================================================================
# THE strategy registry — one row of facts per strategy, one keyword constructor
# =============================================================================
#
# Every local-optimization strategy is ONE row here: its TOML tag (the
# `configs/default.toml` `[local_optimization.<tag>]` table its keyword
# defaults read, and the `type` a config spells it by — the tag rule:
# `snake_case` of the type's name), its optimization path, whether it is a
# loss-parametric family (`Irls{K}`, which takes a `loss` key), and — for the
# named one-off strategies — the identity VGC's `lo_label` /
# `loss_family` / `scale_interface` report. `nothing` in a column means the
# fact is DERIVED, not stated: the parametric families read their label and
# loss family off the loss tag (above), a fixed-loss arm its loss family off its
# `shape` column, and a width-carrying strategy its scale interface off its
# width mode.
#
# LocalOptimizationDoneRight keeps the registry of its own strategies (its
# `LO_TAGS`); a caller loading both merges them:
# `local_optimization_from_toml(d; class, tags = merge(RVG.LO_TAGS, LODR.LO_TAGS))`.
#
#   type                 tag                      path   scale          family   label                     shape              parametric
const _LO_REGISTRY = (
    (NoLocalOptimization, "no_local_optimization", :seed, NoScale(),     :none,   raw"seed",                nothing,           false),
    (Irls,                "irls",                  :irls, nothing,       nothing, nothing,                  nothing,           true),
)

for (T, tag, path, iface, family, label, shape, _) in _LO_REGISTRY
    @eval begin
        _lo_tag(::Type{<:$T}) = $tag
        VgtkCore.optimization_path(::$T) = $(QuoteNode(path))
    end
    iface === nothing || @eval VgtkCore.scale_interface(::$T) = $iface
    family === nothing || @eval VgtkCore.loss_family(::$T) = $(QuoteNode(family))
    label === nothing || @eval VgtkCore.lo_label(::$T) = $label
end

# A strategy takes a width (`scale_multiple` / `loss_width` keys) iff it has a
# `width` field, and a `loss` key iff it is a loss-parametric family.
_lo_takes_width(::Type{T}) where {T} = :width in fieldnames(T)
const _LO_PARAMETRIC = Dict{Any, Bool}(T => par for (T, _, _, _, _, _, _, par) in _LO_REGISTRY)

"""
    LO_TAGS

The local-optimization strategy registry (VisualGeometryToolkitCore's `TagRegistry`):
TOML tag → strategy type or loss-parametric family, the tag `snake_case` of the
type's name. Retired tags name their successor.
"""
const LO_TAGS = TagRegistry(
    :local_optimization, "",
    (tag => T for (T, tag) in _LO_REGISTRY)...;
    retired = ("none" => "no_local_optimization",),
    paths = [r"(^|\.)local_optimization\.type$"]
)

# -----------------------------------------------------------------------------
# The ONE keyword constructor of every strategy, driven by `fieldnames` and the
# tag: a field's keyword is the field's name; its default is the strategy's
# `[local_optimization.<tag>]` key in `configs/default.toml` (ASCII-spelled,
# `toml_key`); a field with NO key in the file is REQUIRED (Barron's `α`).
# The width field is set by `scale_multiple = m` (`ScaleMultiple(m)`) or
# `loss_width = w` (`LossWidth(w)`) — both is an `ArgumentError` — and by
# neither follows the scorer (`nothing`). Values convert to the field's type;
# an unknown keyword is an `ArgumentError`.
#
# The per-strategy exceptions are hooks, each stated once:
#   `_lo_admits(T)`            construction-time admission, BEFORE any field
#   `_lo_key(T, Val(f))`       a field read from another key (HQ smooth-TLS)
#   `_lo_field(T, Val(f), v)`  a supplied value's conversion / refusal
#   `_lo_absent(T, Val(f))`    a field's value when the caller omits it
#   `_lo_validate(lo)`         cross-field checks on the built value
# -----------------------------------------------------------------------------
function _lo_construct(::Type{T}; kw...) where {T <: AbstractLocalOptimization}
    _lo_admits(T)
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
        haskey(kw, f) ? _lo_field(T, Val(f), kw[f]) : _lo_absent(T, Val(f))
    end
    return _lo_validate(T(vals...))
end

# The keywords a caller spells: the fields, the width field as its two keys.
_lo_keywords(::Type{T}) where {T} =
    _lo_takes_width(T) ? [filter(!=(:width), collect(fieldnames(T))); :scale_multiple; :loss_width] :
    collect(fieldnames(T))

# `scale_multiple = m` / `loss_width = w` → the `width` field.
function _lo_width_keywords(::Type{T}, kw) where {T}
    _lo_takes_width(T) || return kw
    hm, hw = haskey(kw, :scale_multiple), haskey(kw, :loss_width)
    hm && hw && throw(ArgumentError("$T: give `scale_multiple` or `loss_width`, not both"))
    hm || hw || return kw
    rest = Dict{Symbol, Any}(k => v for (k, v) in kw if k ∉ (:scale_multiple, :loss_width))
    rest[:width] = hm ? _lo_scale_multiple(T, kw[:scale_multiple]) : LossWidth(Float64(kw[:loss_width]))
    return rest
end
_lo_scale_multiple(::Type, m) = ScaleMultiple(Float64(m))
# Least squares has no scale multiple; one would be a width it silently ignores.
_lo_scale_multiple(::Type{<:Irls{GaussianShape}}, _) = throw(
    ArgumentError(
        "Irls{GaussianShape} is plain least squares: it has no scale multiple, " *
            "so it takes no `scale_multiple`"
    )
)

_lo_admits(::Type) = nothing
_lo_admits(::Type{Irls{K}}) where {K} = _mechanism_admits_shape("Irls{$K}", K)

_lo_key(T::Type, ::Val{F}) where {F} = toml_key(T, F)

_lo_field(::Type{T}, ::Val{F}, v) where {T, F} = convert(fieldtype(T, F), v)
_lo_field(::Type, ::Val{:width}, w) = lo_width_value(w)

function _lo_absent(::Type{T}, ::Val{F}) where {T, F}
    _lo_has_default(T, F) || throw(UndefKeywordError(F))
    return _from_toml(T, Val(F), _lo_table(T)[_lo_key(T, Val(F))])
end
# No width given: follow the scorer.
_lo_absent(::Type, ::Val{:width}) = nothing
# The strategy's `[local_optimization.<tag>]` table (empty for a strategy with
# no fields), and whether it defaults a field.
_lo_table(::Type{T}) where {T} = get(_TOML_DEFAULTS["local_optimization"], _lo_tag(T), Dict{String, Any}())
_lo_has_default(::Type{T}, f::Symbol) where {T} = haskey(_lo_table(T), _lo_key(T, Val(f)))

_lo_validate(lo) = lo

for (T, _, _, _, _, _, _, parametric) in _LO_REGISTRY
    T === NoLocalOptimization && continue          # no fields: the default constructor is it
    if parametric
        @eval (::Type{$T{P}})(; kw...) where {P} = _lo_construct($T{P}; kw...)
    else
        @eval (::Type{$T})(; kw...) = _lo_construct($T; kw...)
    end
end
