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

"""
    TerminalLO(inner::AbstractLocalOptimization) <: AbstractLocalOptimization
    TerminalLO(; inner)

The TERMINAL placement of a local optimization: `ransac` decides its incumbent
race on raw-hypothesis scores alone, exactly as with
[`NoLocalOptimization`](@ref) (same trials, scores and `on_improvement`
events on the same rng), and after the loop runs `inner` ONCE on the final
incumbent through the seeded path of [`locally_optimize`](@ref) — the same
scorer, inside `ransac`'s wall time. The result is that seeded refine's:

- `estimate`: the refined model, or the raw incumbent when the refined model
  fails `test_consensus` (as `locally_optimize` keeps its seed) or the
  incumbent's consensus set is too thin to refine (`lo_retcode` `:thin_partition`).
  The refined model is kept even if it scores below the raw incumbent (there is
  no `score_guard` at the terminal placement);
- `lo_retcode`: the inner strategy's (`:not_run` when the seed was kept);
- `inlier_mask`, `residuals`, `weights`, `quality`, `sigma`, `dof`, `rms`:
  the reported model re-scored under the scorer;
- `iterations` and `sample_acceptance_rate`: the raw race's.

A race that keeps no hypothesis returns its `:no_model` result unrefined, and
an incumbent the scorer cannot score as a seed (a predictive scorer with no
covariance on its consensus set) the raw race's result. `score_guard = true` is
an `ArgumentError`: the guard is a per-candidate acceptance rule and the
terminal placement has no candidate to accept. A seeded `locally_optimize`
with a `TerminalLO` runs `inner` (a seed has no race).

Every hook the loop, the composition check, the reporting and an external
refit read answers as `inner`'s (width, scale interface, loss family, path,
marker, workspace, `RefitFamily` and the external-refit bracket), except
the label, `"<inner label> (terminal)"`, so result records tell the
placements apart. TOML: `{ type = "terminal_lo", inner = { … } }`, the inner
table any registered strategy's. Counting: `CountedLO(TerminalLO(inner))` and
`TerminalLO(CountedLO(inner))` both count the one terminal call per run that
keeps a hypothesis (no LO runs inside the race). A wrapper of `TerminalLO`
must forward [`lo_placement`](@ref), or `ransac` runs it in the loop.
"""
struct TerminalLO{L <: AbstractLocalOptimization} <: AbstractLocalOptimization
    inner::L
end

"""
    InLoopPlacement()

The placement ([`lo_placement`](@ref)) of a strategy `ransac` runs on every
candidate that enters the local-optimization gate: every strategy's default.
"""
struct InLoopPlacement end

"""
    TerminalPlacement()

The placement ([`lo_placement`](@ref)) of a strategy `ransac` runs once, on the
final incumbent of a raw race: [`TerminalLO`](@ref).
"""
struct TerminalPlacement end

"""
    lo_placement(lo) -> InLoopPlacement() | TerminalPlacement()

Where `ransac` runs the strategy `lo`, dispatched statically in the loop:
[`InLoopPlacement`](@ref)`()` by default, [`TerminalPlacement`](@ref)`()` for a
[`TerminalLO`](@ref). A wrapper must forward it to the strategy it wraps (as
[`CountedLO`](@ref) does): a wrapper of a [`TerminalLO`](@ref) that does not
runs it in the loop's placement.
"""
lo_placement(::AbstractLocalOptimization) = InLoopPlacement()
lo_placement(::TerminalLO) = TerminalPlacement()

# -----------------------------------------------------------------------------
# Loss TOML tags — the one table mapping the canonical VisualGeometryToolkitCore
# loss shapes (robust/losses.jl) that the parametric strategy types
# (`Irls{K}`, `HQ{K}`, `Gnc{S}`) take to their TOML `loss` name
# (config_toml.jl's LOSS_TAGS). A shape's loss family and display label
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
"""
    RobustLm{K<:AbstractLossShape} <: AbstractLocalOptimization

The ROBUSTIFIED LEVENBERG–MARQUARDT local optimization of every chart problem
(the [`MinimalSolveProblem`](@ref) family): from the hypothesis θ, ONE chart LM
over all the data minimizing `Σᵢ ρ(‖rᵢ‖²)` — the problem's own refit rows (its
Sampson distance) on its own chart, robustified per datum by loss `K` through
VisualGeometryToolkitCore's `loss_robustifier`. NLLSsolver applies the loss with
Triggs' second-order correction (the ρ″ term of the Gauss–Newton Hessian), so
there is no reweighting loop and no inlier set inside the refit: θ is the
initial guess and the loss decides each datum's influence.

The loss width is `loss_width = scale_multiple·ŝ` (in the loss's currency,
[`plain_scale_multiple`](@ref)), frozen at the start of the call: ŝ is the
Student-t marginal scale of θ's residuals — the Sampson errors of all the data —
over θ's consensus partition (VisualGeometryToolkitCore's `partition_student_t`,
the scorer's `scale_s2`), the scale every consensus-anchored strategy reads.
`scale_multiple` is REQUIRED: the stack has no established multiple for this
strategy, so a configuration states one; an absolute `loss_width` or no width
is an `ArgumentError`.

# Fields
- `width::ScaleMultiple{Float64}`: the width mode, set by `scale_multiple = m`
"""
struct RobustLm{K <: AbstractLossShape, W} <: AbstractLocalOptimization
    width::W
end
RobustLm{K}(width::W) where {K, W} = RobustLm{K, W}(width)

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
strategy_loss(::Type{<:RobustLm{K}}) where {K} = K
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
# The robustified LM is a static loss on the IRLS path (the paper's TLS-LM is
# that path's truncated member); its label names the solver.
VgtkCore.lo_label(lo::RobustLm) = loss_label(strategy_loss(lo)) * "-LM"
VgtkCore.loss_family(lo::RobustLm) = loss_family(strategy_loss(lo))

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
VgtkCore.deployed_scale_multiple(::Union{Irls, RobustLm}, shape, m::Float64, _s2, _scoring) = plain_scale_multiple(shape, m)

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
deployed_shape(lo::Union{Irls, RobustLm}) = strategy_loss(lo)()

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
# The robustified LM's width mode — always a scale multiple, always supplied.
VgtkCore.lo_width(lo::RobustLm) = lo.width
VgtkCore.scale_interface(lo::RobustLm) = width_scale_interface(lo.width)
VgtkCore.with_width(::RobustLm{K}, w) where {K} = RobustLm{K}(lo_field(RobustLm{K}, Val(:width), w))
# The least-squares arm consumes no width (whatever its mode).
VgtkCore.scale_interface(::Irls{GaussianShape}) = NoScale()

# =============================================================================
# THE strategy machinery — one row of facts per strategy, one keyword
# constructor, one reader and one writer (config_toml.jl), for every package
# =============================================================================
#
# A local-optimization strategy is ONE row of a registry (`@lo_registry`):
# its TOML tag (the `[local_optimization.<tag>]` table of its owner's defaults
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
# the registries: `local_optimization_from_toml(d; class, tags = merge(LO_TAGS, Downstream.LO_TAGS))`.
#
# The keyword constructor is driven by `fieldnames` and the tag: a field's
# keyword is the field's name; its default is the strategy's table key
# (`lo_key`, ASCII-spelled); a field with NO key in the table is REQUIRED.
# The width field is set by `scale_multiple = m` (`ScaleMultiple(m)`) or
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
#   `lo_scale_multiple_shape(T)`           the shape a `scale_multiple` is read for
#   `lo_resolve_scale_multiple(T, K, v; class)`   a TOML `scale_multiple` → a number

"""
    lo_tag(T) -> String
    lo_tag(lo) -> String

The TOML tag of a registered strategy type or loss-parametric family (its
`@lo_registry` row): the `type` a config spells it by and the
`[local_optimization.<tag>]` table its keyword defaults read.
"""
function lo_tag end
lo_tag(lo::AbstractLocalOptimization) = lo_tag(typeof(lo))

"""
    lo_defaults(T) -> AbstractDict

The `[local_optimization]` table of the defaults file of the package that
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
lo_field(::Type{<:RobustLm}, ::Val{:width}, w::ScaleMultiple{Float64}) = w
lo_field(::Type{T}, ::Val{:width}, w) where {T <: RobustLm} = throw(ArgumentError(
    "$T: the loss width is scale_multiple·ŝ — give `scale_multiple`, got $w"))

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
    _lo_has_default(T, F, defaults) || throw(UndefKeywordError(F))
    return lo_field_from_toml(T, Val(F), _lo_table(T, defaults)[lo_key(T, Val(F))])
end
lo_absent(::Type, ::Val{:width}; defaults::AbstractDict) = nothing
lo_absent(::Type{<:RobustLm}, ::Val{:width}; defaults::AbstractDict) = throw(UndefKeywordError(:scale_multiple))

"""
    lo_validate(lo) -> lo

Cross-field checks on a freshly built strategy; returns it or throws. Default:
none.
"""
lo_validate(lo) = lo

# The strategy's table of the defaults (empty for a strategy with no fields),
# and whether it defaults a field.
_lo_table(::Type{T}, defaults::AbstractDict) where {T} = get(defaults, lo_tag(T), Dict{String, Any}())
_lo_has_default(::Type{T}, f::Symbol, defaults::AbstractDict) where {T} =
    haskey(_lo_table(T, defaults), lo_key(T, Val(f)))

"""
    lo_construct(T; defaults = lo_defaults(T), kw...) -> T

THE keyword constructor of every registered strategy (`@lo_registry` defines
`T(; kw...)` as this call): each field is its keyword or, omitted, its
`defaults` value ([`lo_absent`](@ref)); the width is spelled `scale_multiple`
or `loss_width` (both is an `ArgumentError`; neither follows the scorer). The
hooks [`lo_admits`](@ref), [`lo_field`](@ref) and [`lo_validate`](@ref) state
the per-strategy exceptions. `defaults` is the `[local_optimization]` table,
keyed by tag. An unknown keyword is an `ArgumentError`.
"""
function lo_construct(::Type{T}; defaults::AbstractDict = lo_defaults(T), kw...) where {T <: AbstractLocalOptimization}
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

# A strategy takes a width (`scale_multiple` / `loss_width` keys) iff it has a
# `width` field.
_lo_takes_width(::Type{T}) where {T} = :width in fieldnames(T)

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

"""
    lo_scale_multiple_shape(T) -> Type{<:AbstractLossShape}

The loss shape a TOML `scale_multiple` of strategy `T` is read for (and a
named one, `"cross_validated"`, tuned for): a loss-parametric family's
[`strategy_loss`](@ref), a fixed-loss arm's `@lo_registry` `shape` column. An
`ArgumentError` for a strategy that has neither.
"""
lo_scale_multiple_shape(::Type{T}) where {T} =
    lo_parametric(T) ? strategy_loss(T) :
    throw(ArgumentError("$(nameof(T)) takes no `scale_multiple` (it has no width)"))

"""
    @lo_registry rows defaults = table [retired = (old => new, …)]

Register the local-optimization strategies of `rows` and return their
VisualGeometryToolkitCore `TagRegistry` (family `:local_optimization`, the TOML
path `local_optimization.type`): `const LO_TAGS = @lo_registry ROWS defaults = …`.
Each row is

    (T, tag, path, scale_interface, loss_family, label, shape, parametric)

with `nothing` for a fact that is derived rather than stated. `defaults` is an
expression evaluated, in the calling module, each time a default is read: the
`[local_optimization]` table of the caller's defaults file, keyed by tag.
`retired` lists retired tags with their successors.

The expansion is one call, made at load time in the calling module. Per row it
defines [`lo_tag`](@ref), [`lo_defaults`](@ref), VisualGeometryToolkitCore's
`toml_skip_fields` and, for a stated path, `optimization_path` (the width is written as its two
keys, never as `width`); for a parametric row [`lo_parametric`](@ref); for a
stated scale interface, loss family or label the Core accessor; for a stated
`shape` [`lo_scale_multiple_shape`](@ref) and the loss family of that shape;
and, for a strategy with fields, the keyword constructor `T(; kw...)`
(`T{K}(; kw...)` for a parametric family) as [`lo_construct`](@ref). A macro
rather than a function because it defines methods in the caller's module.
Every method it adds is on a declared contract generic.
"""
macro lo_registry(rows, options...)
    opts = Dict{Symbol, Any}()
    for o in options
        Meta.isexpr(o, :(=), 2) && o.args[1] in (:defaults, :retired) ||
            throw(ArgumentError("@lo_registry rows defaults = … [retired = …]; got `$o`"))
        opts[o.args[1]] = o.args[2]
    end
    haskey(opts, :defaults) || throw(ArgumentError("@lo_registry needs `defaults = …` (the [local_optimization] table)"))
    return :(_register_lo_rows($__module__, $(esc(rows)), $(QuoteNode(opts[:defaults]));
                               retired = $(esc(get(opts, :retired, :(()))))))
end

function _register_lo_rows(M::Module, rows, defaults; retired = ())
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
            $E.lo_scale_multiple_shape(::Type{<:$T}) = $shape
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
        :local_optimization, "",
        (tag => T for (T, tag) in rows)...;
        retired, paths = [r"^(.*\.)?local_optimization(\.inner)*\.type$"]
    )
end

# This package's strategies.
#   type                 tag                      path   scale          family   label                     shape              parametric
const _LO_REGISTRY = (
    (NoLocalOptimization, "no_local_optimization", :seed, NoScale(),     :none,   raw"seed",                nothing,           false),
    (Irls,                "irls",                  :irls, nothing,       nothing, nothing,                  nothing,           true),
    (RobustLm,            "robust_lm",             :irls, nothing,       nothing, nothing,                  nothing,           true),
    # Every identity fact of the terminal placement is its inner strategy's
    # (local_optimization.jl, beside CountedLO).
    (TerminalLO,          "terminal_lo",           nothing, nothing,     nothing, nothing,                  nothing,           false),
)

"""
    LO_TAGS

The local-optimization strategy registry (VisualGeometryToolkitCore's `TagRegistry`,
built by [`@lo_registry`](@ref)): TOML tag → strategy type or loss-parametric
family, the tag `snake_case` of the type's name. Retired tags name their
successor.
"""
const LO_TAGS = @lo_registry _LO_REGISTRY defaults = _TOML_DEFAULTS["local_optimization"] retired = ("none" => "no_local_optimization",)
