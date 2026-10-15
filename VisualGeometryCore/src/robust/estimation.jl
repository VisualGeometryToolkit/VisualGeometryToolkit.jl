# =============================================================================
# Robust-Estimation Interfaces — the shared type hierarchy for RANSAC scoring
# and local optimization
# =============================================================================
#
# THE join point between the scoring package (RansacScoringDoneRight: sweeps,
# marginals, GPU grids) and the estimation package (RobustVisualGeometry:
# the RANSAC loop, LO strategies, problems). Types and contracts ONLY — no
# behavior lives here. Both packages import these and implement; the RANSAC
# loop compiles against the verbs, never against concrete foreign types.
#
# The hierarchy deliberately encodes the score/refine boundary of the
# accompanying theory: whether the potential is normalizable (ρ(∞) = ∞)
# decides whether a component may be treated as *evidence* (a likelihood,
# usable for scoring and score-monotone decisions) or is merely a
# *reweighting rule* (an M-estimation weight policy for the block-coordinate-descent loop). The
# type system makes the category error unrepresentable: only
# `NormalizedWeightPolicy` unlocks likelihood-grade operations.
#
# DEPENDENCY: None (pure interfaces).
# =============================================================================

# -----------------------------------------------------------------------------
# Scoring — ranks hypotheses by objective VALUE
# -----------------------------------------------------------------------------

"""
    AbstractScoring

A hypothesis score: evaluates a model's residuals and returns a value used for
best-so-far gating, the size of the consensus set it implies (for the adaptive
stopping rule), and that set's order (for the local optimizer). The contract —
[`scoring_workspace`](@ref), [`score!`](@ref), [`inlier_count`](@ref),
[`consensus_order`](@ref), and the optional verbs with their defaults — is
stated in `robust/scoring_contract.jl`; the RANSAC loop calls only it.
"""
abstract type AbstractScoring end

"""
    ConsensusSet(; count, s2, epoch, score = NaN)

The consensus set a local optimizer starts from — the inlier set of the most
recent [`score!`](@ref) on a workspace, as [`consensus_set`](@ref) returns it:
its size k* (`count`), its scale ŝ² (`s2`, the Student-t scale of the set under
the scorer's prior, [`consensus_s2`](@ref)), the `epoch` of the score
evaluation that produced it and that evaluation's `score` (NaN when the
producer did not carry it). Construct it by keyword; the field order is not API.

A `ConsensusSet` describes ONE evaluation: it is valid only while it reflects
the MOST RECENT `score!` on the buffers it was built from. Producers stamp
`epoch` with the buffers' evaluation counter; consumers (local optimizers) must
establish that the stamp still equals the current epoch before using it, via
[`check_fresh`](@ref) or [`is_fresh`](@ref).

Read it through its accessors, never by field name:

- [`inlier_count`](@ref) — `count`, the set size k*.
- [`scale_s2`](@ref) — `s2`, the consensus scale ŝ².
- [`consensus_score`](@ref) — `score`, the evaluation's score (NaN: not carried).
- [`check_fresh`](@ref) / [`is_fresh`](@ref) — the `epoch` stamp, which is
  only ever meaningful compared against a current epoch, and so is exposed as
  that comparison rather than as a raw number.
"""
struct ConsensusSet
    count::Int
    s2::Float64
    epoch::Int
    score::Float64
end

ConsensusSet(; count::Integer, s2::Real, epoch::Integer, score::Real = NaN) =
    ConsensusSet(Int(count), Float64(s2), Int(epoch), Float64(score))

"""
    consensus_score(cs::ConsensusSet) -> Float64

The score of the evaluation that produced `cs` (the value [`score!`](@ref)
returned, so a local optimizer that ascends that score need not re-score its
seed), or NaN when the producer did not carry it.
"""
consensus_score(cs::ConsensusSet) = cs.score

"""
    inlier_count(cs::ConsensusSet) -> Int
    inlier_count(scoring, W) -> Int

The size k* of a consensus set; for a scoring and its workspace, the k* of the
most recent [`score!`](@ref) — the statistic that drives the hypergeometric
stopping rule.
"""
inlier_count(cs::ConsensusSet) = cs.count

"""
    scale_s2(x) -> Float64

The squared scale ŝ² `x` carries: a [`ConsensusSet`](@ref)'s consensus scale,
a [`DeployedLoss`](@ref)'s noise variance `s2`, a [`FixedS2`](@ref) policy's
held value. The robust layer speaks squared scale throughout.
"""
scale_s2(cs::ConsensusSet) = cs.s2

"""
    is_fresh(cs::ConsensusSet, epoch::Integer) -> Bool

Whether `cs` still reflects the most recent `score!` on the buffers that
produced it, i.e. whether its stamp equals their current `epoch`.

A stale `ConsensusSet` describes an earlier evaluation: using it yields a
plausible-looking wrong k* and ŝ², not an error — so the stamp must be checked
before the set is used. Use [`check_fresh`](@ref) to enforce it.
"""
is_fresh(cs::ConsensusSet, epoch::Integer) = cs.epoch == epoch

"""
    check_fresh(cs::ConsensusSet, epoch::Integer) -> nothing

Refuse a stale [`ConsensusSet`](@ref) with an `ArgumentError` naming both
stamps. `epoch` is the CURRENT epoch of the buffers that produced `cs`.
Returns `nothing` when the set is fresh.

This is the one place the freshness precondition lives. It is a real throw,
not an `@assert`: the failure guarded against is a silent use of an earlier
evaluation's set — wrong answers, not a crash — so the check must not be
removable.
"""
function check_fresh(cs::ConsensusSet, epoch::Integer)
    is_fresh(cs, epoch) || throw(
        ArgumentError(
            "stale ConsensusSet: built at epoch $(cs.epoch), buffers at $(epoch). " *
                "It describes an earlier score! — re-derive it with consensus_set."
        )
    )
    return nothing
end

# -----------------------------------------------------------------------------
# Local optimization — refines the selected hypothesis by DERIVATIVES
# -----------------------------------------------------------------------------

"""
    AbstractLocalOptimization

A local-optimization strategy: refines a seed model on the full measurement
set. Strategies are routed by [`RefitFamily`](@ref); in-loop IRLS-family
strategies additionally plug into the solver via a weight policy
([`AbstractWeightPolicy`](@ref)).
"""
abstract type AbstractLocalOptimization end

"""
    RefitFamily(lo::AbstractLocalOptimization) -> RefitFamily

Holy trait routing a strategy to its EXECUTION path (where the refit runs —
an implementation concern, orthogonal to the paper's two classification axes
[`optimization_path`](@ref) and [`scale_interface`](@ref)):
- `NativeRefit()`   — the in-process solver loop (weight policy / joint lift /
  continuation schedule / block-coordinate ascent all run here)
- `ExternalRefit()` — an external refiner called at an absolute width outside
  the loop ([`refine_external!`](@ref): PoseLib LM/LZ and its adaptive-width
  arm, RansacLib classical LO)

New strategies declare their family; consumers dispatch on it and never
enumerate concrete strategy types.

DESIGN RULE (RVG 2026-07-18 port): all native strategies run through the
production `locally_optimize` port (`NativeRefit`, the default); the noise scale
is ingested ONLY through the strategy's width mode ([`loss_at`](@ref)) from the
scorer's consensus set, never from a deployed plug-in or loss-width-window
bootstrap. An external arm that anchors a width (the adaptive PoseLib LM arm)
reads its consensus scale through the production scoring machinery too.
"""
abstract type RefitFamily end
struct NativeRefit <: RefitFamily end
struct ExternalRefit <: RefitFamily end

RefitFamily(::AbstractLocalOptimization) = NativeRefit()   # default

# -----------------------------------------------------------------------------
# The paper's two classification axes of a local optimizer
#   (Score Sharp, Refine Soft, Polish Hard — §3 "Wide-to-Narrow Local
#   Optimization"): the OPTIMIZATION PATH (how the arm travels) and the SCALE
#   INTERFACE (how the loss width reaches it). Loss family is the third axis
#   ([`loss_family`](@ref)). Every strategy in a figure roster declares all
#   three; registries refuse an undeclared one.
# -----------------------------------------------------------------------------

"""
    optimization_path(lo::AbstractLocalOptimization) -> Symbol

The paper's optimization-path axis — the column of the design table:

- `:irls`         — static reweighting: the weight function of a FIXED loss is
                    applied at every iteration (Welsch-IRLS, GM-IRLS, the hard
                    truncated refit TLS-LM, Stu-tU-IRLS)
- `:hq`           — half-quadratic lift: model and lifted weights optimized
                    JOINTLY, no schedule (Welsch-HQ, GM-HQ, TLS-HQ)
- `:gnc`          — explicit continuation: a scheduled sequence of progressively
                    sharper problems (Welsch-GNC, GM-GNC, TLS-GNC, and TLS-LZ —
                    Le–Zach's relaxed TLS tightened by a μ-schedule)
- `:bcd`          — block-coordinate ascent on the scoring objective itself
                    over partition, scale and model (Stu-tU-BCD)
- `:classical_lo` — classical LO-RANSAC inner refinement (TLS-LO, TLS-LO⁺)
- `:seed`         — no local optimization (the seed hypothesis is returned)

Default `:unspecified`, so registries can detect a strategy that has not
declared its identity. Drives [`lo_marker`](@ref) (marker shape = path) and
the path half of a composed [`lo_label`](@ref).
"""
optimization_path(::AbstractLocalOptimization) = :unspecified

"The display suffix of an optimization path (the `-IRLS` / `-HQ` / `-GNC` / `-BCD` / `-LO` half of a method name)."
path_suffix(path::Symbol) =
    path === :irls ? "IRLS" : path === :hq ? "HQ" : path === :gnc ? "GNC" :
    path === :bcd ? "BCD" : path === :classical_lo ? "LO" : path === :seed ? "seed" :
    error("optimization_path $path has no display suffix (declare one of :irls/:hq/:gnc/:bcd/:classical_lo/:seed)")

"The PGFPlots marker of an optimization path — the figure grammar's `shape = path` rule."
path_marker(path::Symbol) =
    path === :irls ? :o : path === :hq ? :triangle : path === :gnc ? :diamond :
    path === :bcd ? :oplus : path === :classical_lo ? Symbol("pentagon*") :
    path === :seed ? :asterisk : :o

"""
    ScaleInterface

The paper's scale-interface axis — how a local optimizer obtains the
`loss_width` it needs (§3.6 "Scale supply to local optimization"). A LABEL,
derived from the strategy, never a stored policy: a width-carrying strategy's
interface is read off its width mode ([`lo_width`](@ref)).

- [`ConsensusAnchored`](@ref) — takes `ŝ` from the consensus set the SCORE
  selected for the hypothesis being refined and sets
  `loss_width = scale_multiple·ŝ` (the [`ScaleMultiple`](@ref) width mode),
  cross-validating only the dimensionless `scale_multiple`.
- [`FixedScale`](@ref) — one ABSOLUTE loss width for every hypothesis
  (the [`LossWidth`](@ref) width mode; TLS-LM, TLS-LZ, classical TLS-LO⁺).
- [`FollowsScorer`](@ref) — the loss width follows the scorer's decision point,
  [`inlier_threshold`](@ref) (a strategy with no width of its own).
- [`SelfScaling`](@ref) — re-estimates `ŝ` at each iteration; the consensus
  set only initializes it (Stu-tU-BCD, Stu-tU-IRLS).
- [`NoScale`](@ref) — consumes no scale at all (the seed, ℓ₂).
"""
abstract type ScaleInterface end
"Scale interface: the strategy takes `ŝ` from the consensus set the score selected (see [`ScaleInterface`](@ref))."
struct ConsensusAnchored <: ScaleInterface end
"Scale interface: one absolute loss width for every hypothesis (see [`ScaleInterface`](@ref))."
struct FixedScale <: ScaleInterface end
"Scale interface: the loss width follows the scorer's [`inlier_threshold`](@ref) (see [`ScaleInterface`](@ref))."
struct FollowsScorer <: ScaleInterface end
"Scale interface: the strategy re-estimates `ŝ` at each iteration (see [`ScaleInterface`](@ref))."
struct SelfScaling <: ScaleInterface end
"Scale interface: the strategy consumes no scale at all — the seed, ℓ₂ (see [`ScaleInterface`](@ref))."
struct NoScale <: ScaleInterface end
struct UnspecifiedScale <: ScaleInterface end

"""
    scale_interface(lo::AbstractLocalOptimization) -> ScaleInterface

The strategy's [`ScaleInterface`](@ref) label. A width-carrying strategy's is
DERIVED from its width mode ([`lo_width`](@ref)): `ScaleMultiple` →
`ConsensusAnchored()`, `LossWidth` → `FixedScale()`, `nothing` →
`FollowsScorer()`. Default `UnspecifiedScale()` so registries can detect an
undeclared strategy.
"""
scale_interface(::AbstractLocalOptimization) = UnspecifiedScale()

"""
    lo_label(lo::AbstractLocalOptimization) -> String

The single source of truth for a strategy's figure/table DISPLAY NAME: its
LaTeX label string, read by every renderer instead of a hand-maintained
lookup table, so legends cannot drift. The marker companion is
[`lo_marker`](@ref). A concrete `lo_label` returns the BASE name (e.g.
`raw"Welsch-HQ"`, i.e. Loss-Path).
"""
lo_label(::AbstractLocalOptimization) = "LO"

"""
    lo_marker(lo::AbstractLocalOptimization) -> Symbol

The strategy's PGFPlots marker symbol — the figure grammar's `shape =
optimization path` rule, derived from [`optimization_path`](@ref) through
[`path_marker`](@ref). Override only for a strategy that must stand apart
from its path (ℓ₂ uses `:x`).
"""
lo_marker(lo::AbstractLocalOptimization) = path_marker(optimization_path(lo))

"""
    loss_family(lo::AbstractLocalOptimization) -> Symbol

The loss ROW of the design table an arm belongs to (`:welsch` | `:tls` |
`:gm` | `:cauchy` | …; the Student-t/uniform mixture arms report `:tls` — they
share the "uniform mixtures" column). Determines the color HUE facet in the
paper registry. Default `:unspecified` so registries can detect a strategy
that has not declared its identity.
"""
loss_family(::AbstractLocalOptimization) = :unspecified

# -----------------------------------------------------------------------------
# Weight policies — the nuisance and weight blocks of block_coordinate_descent!
# -----------------------------------------------------------------------------

"""
    AbstractWeightPolicy

A reweighting rule: the nuisance and weight blocks of
[`block_coordinate_descent!`](@ref). The contract is deliberately
minimal — a weight policy promises weights, nothing more:
- `update_nuisance!(policy)`: refresh whatever nuisance parameters the policy
  carries — ŝ², ν, the mixing weight π, the Barron shape α — at the current
  residuals (a no-op for policies that carry none).
- `update_weights!(policy)`: refresh per-residual weights from the current
  residuals and those nuisance parameters.

The two run as a pair, in that order, once per outer iteration at a fixed
model — see [`update_nuisance!`](@ref) for the contract.

A weight policy is NOT a likelihood. M-estimation losses with bounded
potentials (Welsch, Huber elbows at frozen scale, Barron shape adaptation)
have no normalizable density on an unbounded domain — their "objective" is
a loss, and only its derivative machinery (ψ(r)/r) is used here. Do not
score with a weight policy.
"""
abstract type AbstractWeightPolicy end

"""
    update_weights!(policy::AbstractWeightPolicy) -> policy

Refresh the policy's per-residual WEIGHTS from the current residuals and the
nuisance parameters [`update_nuisance!`](@ref) just set, at a fixed model: the
weight block of [`block_coordinate_descent!`](@ref), run after the nuisance
block in every iteration. It writes the weights and nothing else.
"""
function update_weights! end

"""
    update_nuisance!(policy::AbstractWeightPolicy) -> policy

Refresh the policy's NUISANCE PARAMETERS at the current residuals and a
fixed model. One half of the pair [`block_coordinate_descent!`](@ref) runs per
iteration, always in this order:

    update_model! → residuals → update_nuisance! → update_weights! → check

One sweep, at a fixed model: `update_nuisance!` may not touch the model, and
`update_weights!` reads whatever it left behind. A policy that carries no
nuisance parameters implements this as a no-op — the loop still calls it.

# Why not "scale"

The parameters this verb moves are, across the implementing policies: nothing
at all, ŝ² alone, ν together with ŝ², the pair (ŝ², π), and — on the Barron
arm — only the shape α, with no scale involved. "Scale" names only one of
those.

What they share is their ROLE, not their identity. The model (pose,
homography, essential matrix) is the parameter of interest; ŝ², ν, π and α
exist only because the likelihood cannot be written without them. That is the
definition of a nuisance parameter, and it is true of all of them.

Nor is it an EM maximization: two implementations are genuine no-ops and the
Barron α update is a direct minimization of the negative log-likelihood over a
grid. "Nuisance" describes every implementation.

See [`AbstractWeightPolicy`](@ref) for the full policy contract.
"""
function update_nuisance! end

"""
    NormalizedWeightPolicy <: AbstractWeightPolicy

A weight policy whose underlying objective IS a proper normalized
probability model (e.g. the Student-t/uniform mixture posterior, the
Gaussian). Only these support likelihood-grade operations: evaluating a
comparable objective value, likelihood-monotonicity statements, and use as
evidence in a score. The ρ(∞) = ∞ boundary of the theory, as a type.
"""
abstract type NormalizedWeightPolicy <: AbstractWeightPolicy end

# -----------------------------------------------------------------------------
# Reading a weight policy — three quantities, named apart
# -----------------------------------------------------------------------------
#
# A policy holds three different numbers per residual, and calling them all
# "weights" is what lets a posterior and an IRLS factor be confused for one
# another. They are named apart here:
#
#   inlier_weight   how much this residual is an inlier, in [0, 1]
#   robust_weight   the factor multiplying it in the weighted normal equations
#   is_inlier       the hard decision
#
# For an M-estimation loss the first two COINCIDE: psi(r)/r is the only
# weight there is. A mixture separates them, membership from reweighting, and
# `NormalizedWeightPolicy` is exactly the statement that the separation
# exists — which is why the default `is_inlier` hangs off it and not off
# `AbstractWeightPolicy`.

"""
    inlier_weight(policy) -> AbstractVector

Per-residual inlier weight in `[0, 1]`: how much each observation belongs to
the inlier component.

Its MEANING is the policy's type. Under a [`NormalizedWeightPolicy`](@ref) it
is a posterior membership probability, so it may be summed (an effective
inlier count) and thresholded. Under a plain [`AbstractWeightPolicy`](@ref) it
is a loss weight — orderable, but not a probability: summing it means
nothing and `> 0.5` is not a decision.

Returns a STORED vector or a view of one — an O(1) accessor, never a
computation: [`is_inlier`](@ref) calls it once per residual, so a freshly
broadcast result would make [`inlier_mask`](@ref) quadratic.
"""
function inlier_weight end

"""
    robust_weight(policy) -> AbstractVector

Per-residual factor in the weighted normal equations — `psi(r)/r` for an
M-estimation loss, `p_in * u` for a Student-t/uniform mixture (membership
times the component's own IRLS weight).

EQUALS [`inlier_weight`](@ref) for a loss policy, where one number does both
jobs. The two differ only where the model separates membership from
reweighting.
"""
function robust_weight end

"""
    is_inlier(policy, i::Integer) -> Bool

Whether residual `i` is an inlier UNDER THIS POLICY'S OWN RULE. Every policy
answers; they disagree about what the question means, which is why this is the
policy's method and not a caller's expression.

A [`NormalizedWeightPolicy`](@ref) takes the default below — the MAP rule
[`is_map_inlier`](@ref) on its membership. A loss policy must state its own
rule (typically a residual multiple of its scale): it has no membership to
threshold, and `0.5` on a loss weight is arbitrary.
"""
function is_inlier end

is_inlier(p::NormalizedWeightPolicy, i::Integer) = @inbounds is_map_inlier(inlier_weight(p)[i])

"""
    is_map_inlier(posterior_weight) -> Bool

The MAP inlier decision on a POSTERIOR inlier probability: inlier iff the
posterior exceeds ½, i.e. the inlier label is the more probable one. The ½ is
the definition of the MAP decision between two labels, not a tunable.

THE one statement of that rule: every posterior check in the stack calls it
(RansacScoringDoneRight's posterior weighting, RobustVisualGeometry, the
BlobBoards pipeline). It applies to a posterior membership only — never to an
M-estimation loss weight, whose `0.5` has no probabilistic meaning (see
[`is_inlier`](@ref)).
"""
@inline is_map_inlier(posterior_weight::Real) = posterior_weight > 0.5

"""
    inlier_mask(policy) -> BitVector

[`is_inlier`](@ref) over every residual. Generic on purpose: a policy defines
the rule, never the loop.
"""
function inlier_mask(p::AbstractWeightPolicy)
    n = length(inlier_weight(p))
    m = BitVector(undef, n)
    @inbounds for i in 1:n
        m[i] = is_inlier(p, i)
    end
    return m
end
