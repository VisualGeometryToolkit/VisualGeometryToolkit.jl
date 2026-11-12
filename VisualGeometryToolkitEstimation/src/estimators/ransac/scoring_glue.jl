# =============================================================================
# Scoring glue — the RANSAC-harness side of the local-optimization contract
# =============================================================================
#
# The scoring contract, the local-optimization contract and the robust-
# statistics toolkit are VisualGeometryToolkitCore's. This file holds the two
# local-optimization strategies of the core — the static M-estimator `Irls{K}`
# and the robustified LM `Triggs{K}` (no local optimization is `nothing`) — with their width modes, the weight-policy
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
    TerminalLO(inner::AbstractOptimizationStrategy) <: AbstractOptimizationStrategy
    TerminalLO(; inner)

The TERMINAL placement of a local optimization: `ransac` decides its incumbent
race on raw-hypothesis scores alone, exactly as with
no local optimization (`nothing`; same trials, scores and `on_improvement`
events on the same rng), and after the loop runs `inner` ONCE on the final
incumbent through the seeded path of [`locally_optimize`](@ref) — the same
scorer, inside `ransac`'s wall time. The result is that seeded refine's:

- `estimate`: the refined model, or the raw incumbent when the refined model
  fails `test_consensus` (as `locally_optimize` keeps its seed) or the
  incumbent's consensus set is too thin to refine (`lo_retcode` `:thin_partition`).
  The refined model is kept even if it scores below the raw incumbent (there is
  no `score_guard` at the terminal placement);
- `lo_retcode`: the inner strategy's (`:not_run` when the seed was kept);
- `residuals`, `weights`, `quality`, `scale`, `dof`, `n`, `rms`:
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
struct TerminalLO{L <: AbstractOptimizationStrategy} <: AbstractOptimizationStrategy
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
lo_placement(::Union{Nothing, AbstractOptimizationStrategy}) = InLoopPlacement()
lo_placement(::TerminalLO) = TerminalPlacement()

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

"""
    default_local_optimization_strategy(scoring::AbstractScoring) -> AbstractOptimizationStrategy

Default local-optimization strategy for a scoring function: `nothing` (no local optimization).
"""
default_local_optimization_strategy(::AbstractScoring) = nothing

# -----------------------------------------------------------------------------
# PosteriorLikelihood — AbstractWeightPolicy adapter for the `block_coordinate_descent!` loop
# -----------------------------------------------------------------------------

"""
    PosteriorLikelihood(rws::PosteriorWeightState, n::Int;
                        scale_update = Estimated(), prior_update = FixedPrior(inlier_prior(rws)),
                        nu_update = Estimated()) <: NormalizedWeightPolicy

Likelihood adapter for [`block_coordinate_descent!`](@extref `VisualGeometryToolkitCore.block_coordinate_descent!-Tuple{AbstractWeightPolicy, AbstractBlockCoordinateMethod, Any, Any}`). Wraps a
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

Frozen-scale plain-IRLS likelihood adapter for [`Irls{K}`](@extref VisualGeometryToolkitCore.Irls): ONE
weight policy generic over the loss, replacing the former per-loss
Welsch/Geman–McClure/Huber/L2 quadruplet whose bodies differed only in the
weight formula — which is exactly `VisualGeometryToolkitCore.weight` on the shape.  Reuses
the `PosteriorWeightState` buffers (`sq_distances`, `w`). The loss carries
its OWN squared width (`sq_loss_width`, Huber: elbow²), committed at construction
from the seed partition's Student-t scale and **frozen** — `update_nuisance!` is
a no-op.
"""
struct LossPlainWeights{L <: DeployedLoss} <: AbstractWeightPolicy
    rws::PosteriorWeightState
    n::Int
    loss::L        # shape + (efficiency_constant, ŝ²); `sq_loss_width` = efficiency_constant²·ŝ², never stored
end

update_nuisance!(::LossPlainWeights) = nothing   # scale FROZEN at the seed
current_s2(lh::LossPlainWeights) = scale_s2(lh.loss)   # the scale the loss runs at (VGC `Recorded`)
function update_weights!(lh::LossPlainWeights)
    rws = lh.rws
    # Read the shape and its width ONCE. `sq_loss_width` is
    # `efficiency_constant²·ŝ²`, computed from the two stored slots, so calling it
    # per point re-multiplies every iteration for a value that is frozen at
    # construction by definition.
    shape = loss_shape(lh.loss)
    width2 = sq_loss_width(lh.loss)
    @inbounds for i in 1:lh.n
        robust_weight(rws)[i] = weight(shape, sq_distances(rws)[i], width2)
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
    !bounded(lh.loss) || @inbounds(sq_distances(lh.rws)[i] <= sq_loss_width(lh.loss))


# This package's strategies (the core ones, Irls and Triggs, are
# VisualGeometryToolkitCore's STRATEGY_TAGS).
#   type                 tag                      path   scale          family   label                     shape              parametric
const _LO_REGISTRY = (
    # Every identity fact of the terminal placement is its inner strategy's
    # (local_optimization.jl, beside CountedLO).
    (TerminalLO,          "terminal_lo",           nothing, nothing,     nothing, nothing,                  nothing,           false),
)

"""
    OPTIMIZATION_STRATEGY_TAGS

The optimization-strategy registry (VisualGeometryToolkitCore's `TagRegistry`):
TOML tag → strategy type or loss-parametric family — VisualGeometryToolkitCore's
`STRATEGY_TAGS` (`"irls"`, `"triggs"`) merged with this package's
(`"terminal_lo"`, built by `@lo_registry`) and `"none"` → `nothing` (no local
optimization). The tag is `snake_case` of the type's name; retired tags name
their successor.
"""
const OPTIMIZATION_STRATEGY_TAGS = merge(STRATEGY_TAGS, @lo_registry _LO_REGISTRY defaults = _TOML_DEFAULTS["optimization_strategy"] retired = ("no_local_optimization" => "none",) none = true)
