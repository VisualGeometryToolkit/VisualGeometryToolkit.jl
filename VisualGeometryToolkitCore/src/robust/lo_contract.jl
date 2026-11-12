# =============================================================================
# The local-optimization contract — what a RANSAC loop asks of a local
# optimizer, and the one resolution of an optimizer's loss width.
# =============================================================================
#
#   lo_workspace(::Type{P}, ::Type{T}, lo, n) -> L       [nothing]  built once per composed workspace
#   locally_optimize!(L, lo, problem, θ, cs::ConsensusSet, buf::ScoreBuffers, scoring) -> lo_estimate(…)
#   lo_width(lo) -> nothing | EfficiencyConstant | LossWidth  the width mode of a width-carrying optimizer
#   with_width(lo, w) -> typeof(lo) at the width mode w
#   loss_at(lo, shape, cs, scoring, buf) -> DeployedLoss the loss it runs at, resolved from its mode
#
# STATE ON ENTRY of `locally_optimize!`: `residuals(buf)` holds θ's residuals and
# `consensus_mask(buf)` is `cs`'s set; `consensus_order(scoring,
# scoring_workspace(buf))[1:inlier_count(cs)]` is the consensus; `scale_s2(cs)` is
# the consensus scale. The optimizer may overwrite `buf`'s residuals, scores
# and mask and call `score_model!`; on exit `consensus_mask(buf)` is the set the
# returned model was refit on. It never scores for acceptance — the loop
# re-scores and keeps the incumbent.
# =============================================================================

"""
    lo_workspace(::Type{P}, ::Type{T}, lo, n::Int) -> L

The local optimizer's own scratch and per-call state for problems of type `P`
with up to `n` points at storage precision `T`, built ONCE per composed
workspace and reset by each call. Default `nothing` (an optimizer that needs
none; also `nothing`, no local optimization).
"""
lo_workspace(::Type, ::Type, ::Union{Nothing, AbstractOptimizationStrategy}, ::Int) = nothing

"""
    locally_optimize!(L, lo, problem, θ, cs::ConsensusSet, buf::ScoreBuffers, scoring) -> FitResult

Refine the hypothesis `θ` of `problem` with the strategy `lo` on its workspace
`L` ([`lo_workspace`](@ref)), seeded by the consensus set `cs` of the score
that just ran on `buf`. Returns [`lo_estimate`](@ref)`(θ′, retcode; iterations,
residuals_current)`: `θ′ === θ` when nothing was refit; `residuals_current`
when `residuals(buf)` holds `θ′`'s residuals on exit (the loop then skips
recomputing them).
"""
function locally_optimize! end

# A strategy with no method for this problem type lands on this ONE refusal.
# Every argument but the strategy is untyped (or the contract's own), so every
# implemented method is strictly more specific and none is ambiguous with it.
# The problem types the strategy DOES have a method for are read off the method
# table — the implementations are the one list, so the message cannot drift.
function _lo_problem_types(lo::AbstractOptimizationStrategy)
    types = String[]
    for m in methods(locally_optimize!)
        sig = Base.unwrap_unionall(m.sig).parameters
        length(sig) == 8 || continue
        P = sig[4]
        P === Any && continue
        typeof(lo) <: sig[3] && push!(types, string(P))
    end
    return isempty(types) ? "no problem type" : join(sort!(unique!(types)), ", ")
end

# `nothing` — no local optimization: the hypothesis comes back unchanged (`:no_lo`).
locally_optimize!(_L, ::Nothing, _problem, model, ::ConsensusSet, ::ScoreBuffers, _scoring) =
    lo_estimate(model, :no_lo)

function locally_optimize!(_L, lo::AbstractOptimizationStrategy, problem, _model, _cs::ConsensusSet, _buf::ScoreBuffers, _scoring)
    throw(ArgumentError(
        "$(typeof(lo)) has no local optimization for $(typeof(problem)); it is implemented for " *
            "$(_lo_problem_types(lo))."
    ))
end

"""
    LoInfo

The metadata of one local-optimization refit ([`lo_estimate`](@ref)):
`residuals_current`, whether the optimizer left the refined model's residuals
in the scoring buffers.
"""
struct LoInfo
    residuals_current::Bool
end

"""
    lo_estimate(model, retcode; iterations = 0, residuals_current = false) -> FitResult{M, LoInfo}

The lean result of one local-optimization refit — what
[`locally_optimize!`](@ref) and a `block_coordinate_result` return. `estimate`
is the refined model; `retcode` is why the refit stopped: `:no_lo` (no LO ran),
`:converged`, `:max_iterations`, `:solve_failed` (a refit failed; the incumbent
is returned), `:thin_partition` (the consensus set has no residual dof, so no LO
ran), `:left_basin`, or another stopping criterion's reason. The refit computes
no residual summary, so `rms` is `NaN` and `n` is 0. Zero heap allocation when
`M` is isbits.
"""
lo_estimate(model::M, retcode::Symbol; iterations::Int = 0, residuals_current::Bool = false) where {M} =
    FitResult{M, LoInfo}(model, retcode, iterations, NaN, 0, LoInfo(residuals_current))

"""
    lo_width(lo) -> nothing | EfficiencyConstant{Float64} | LossWidth{NTuple{2,Float64}}

The width mode of a width-carrying local optimizer: `nothing` (follow the
scorer's [`decision_boundary`](@ref)), [`EfficiencyConstant`](@ref)`(m)` (the loss
width `m·ŝ` at the consensus scale) or [`LossWidth`](@ref) (an absolute width).
The owner of the strategy implements it; a strategy with no width mode (the
seed, a self-scaling optimizer) keeps the default `missing`.
"""
lo_width(::Union{Nothing, AbstractOptimizationStrategy}) = missing

"""
    width_scale_interface(w) -> ScaleInterface

The [`ScaleInterface`](@ref) label of a width mode: `EfficiencyConstant` →
`ConsensusAnchored()`, `LossWidth` → `FixedScale()`, `nothing` →
`FollowsScorer()` — what a width-carrying strategy's `scale_interface`
returns, so the label has one source of truth.
"""
width_scale_interface(::Nothing) = FollowsScorer()
width_scale_interface(::EfficiencyConstant) = ConsensusAnchored()
width_scale_interface(::LossWidth) = FixedScale()

"""
    loss_at(lo, shape, cs::ConsensusSet, scoring, buf::ScoreBuffers) -> DeployedLoss

THE resolution of a width-carrying local optimizer's loss, statically
dispatched on its width mode [`lo_width`](@ref):

- `nothing` (follow the scorer): for a shape with a [`decision_width`](@ref)
  method (Welsch, Geman–McClure, Cauchy, smooth TLS, TLS, Barron), `deploy(shape, w²; efficiency_constant = 1)` with
  `w = decision_width(shape, decision_boundary(scoring, buf))` — the loss's
  decision boundary at the scorer's (the identity for `TLSShape`): a threshold
  scorer's constant threshold, or the Student-t + uniform scorer's MAP crossover
  at the current consensus scale and its outlier half-width, re-estimated with
  every score; for a shape without one (Huber), `deploy(shape, ŝ²;
  efficiency_constant = efficiency_constant(shape, codimension(scoring)))` —
  the table's constant at the consensus scale; a shape with neither is an
  `ArgumentError`;
- `EfficiencyConstant(m)`: `deploy(shape, ŝ²; efficiency_constant = m)`, `ŝ² = max(scale_s2(cs), eps())` — the
  consensus-anchored width, `m` always the plain multiple (`loss_width = m·ŝ`) whatever the shape;
- `LossWidth((m₀, s2))`: `deploy(shape, max(s2, eps()); efficiency_constant = m₀)` — the absolute width.
"""
loss_at(lo::AbstractOptimizationStrategy, shape, cs::ConsensusSet, scoring, buf) =
    _loss_at(lo_width(lo), lo, shape, cs, scoring, buf)

_loss_at(::Nothing, lo, shape, cs, scoring, buf) = _follow_scorer(shape, cs, scoring, buf)

# No width stated: a shape with a decision width runs at the scorer's decision point; one without (Huber: its ψ never
# turns, so no residual flips its decision) at the table's `efficiency_constant(K, d)` times the scorer's consensus
# scale, `d` the scorer's codimension (the owner's ruling, 2026-11-12); a shape with neither is refused by name.
function _follow_scorer(shape::AbstractLossShape, cs, scoring, buf)
    if _has_decision_width(shape)
        w = decision_width(shape, decision_boundary(scoring, buf))
        return deploy(shape, w^2; efficiency_constant = 1.0)
    end
    _has_efficiency_constant(shape) || throw(ArgumentError(
        "$(nameof(typeof(shape))) has no decision width and no efficiency constant: a strategy of it must state its " *
            "width (`efficiency_constant = c`, the multiple of the consensus scale, or `loss_width = w`)"))
    return deploy(shape, max(scale_s2(cs), eps()); efficiency_constant = efficiency_constant(shape, codimension(scoring)))
end

# Whether a shape has a decision width / a table efficiency constant: read off the methods themselves, so a shape that
# gains one (Barron's decision width, losses.jl) is followed without a hand-kept list.
_has_decision_width(shape::AbstractLossShape) = hasmethod(decision_width, Tuple{typeof(shape), Float64})
_has_efficiency_constant(shape::AbstractLossShape) = hasmethod(efficiency_constant, Tuple{Type{typeof(shape)}})

function _loss_at(sm::EfficiencyConstant{Float64}, lo, shape, cs, scoring, buf)
    s2 = max(scale_s2(cs), eps())
    return deploy(shape, s2; efficiency_constant = sm.value)
end
_loss_at(lw::LossWidth{NTuple{2, Float64}}, lo, shape, cs, scoring, buf) =
    deploy(shape, max(lw.value[2], eps()); efficiency_constant = lw.value[1])
_loss_at(::Missing, lo, _shape, _cs, _scoring, _buf) =
    throw(ArgumentError("$(nameof(typeof(lo))) has no width mode (lo_width), so it has no loss to deploy"))

"""
    width_s2(lo, cs::ConsensusSet) -> Float64

The squared scale a width-carrying strategy's mode runs at — the consensus
scale under `EfficiencyConstant`, the absolute `s2` under `LossWidth`, both floored
at eps; the INITIAL estimate of a self-scaling strategy is `scale_s2(cs)`.
"""
width_s2(lo::AbstractOptimizationStrategy, cs::ConsensusSet) = _width_s2(lo_width(lo), cs)
_width_s2(::EfficiencyConstant, cs) = max(scale_s2(cs), eps())
_width_s2(lw::LossWidth{NTuple{2, Float64}}, cs) = max(lw.value[2], eps())
_width_s2(::Union{Nothing, Missing}, cs) = max(scale_s2(cs), eps())

"""
    check_composition(scoring, lo) -> nothing

Refuse, once at entry, a local optimizer that follows the scorer's decision
boundary (`lo_width(lo) === nothing`) composed with a scorer that defines none
([`has_decision_boundary`](@ref)). A strategy without a width mode is accepted.
"""
check_composition(scoring::AbstractScoring, lo::Union{Nothing, AbstractOptimizationStrategy}) =
    _check_composition(scoring, lo, lo_width(lo))
_check_composition(_scoring, _lo, _w) = nothing
_check_composition(scoring::S, lo, ::Nothing) where {S} =
    has_decision_boundary(S) ? nothing : throw(
    ArgumentError(
        "$(nameof(S)) has no decision boundary (has_decision_boundary is false) for a local optimization " *
            "with no width to follow; give it an explicit width: " *
            "with_width(lo, EfficiencyConstant(m)) or with_width(lo, LossWidth(w))"
    )
)

# =============================================================================
# The external refiners — the `ExternalRefit` member of `RefitFamily`
# =============================================================================
#
# A strategy that cannot run in the native loop (a third-party solver: PoseLib's
# truncated refiners, RansacLib's classical LO) is a pure model → model refine
# OUTSIDE the loop, at an absolute width:
#
#   external_workspace(lo, problem, frame, scoring) -> ws     once per (pair, arm)
#   prepare_external!(ws, lo, problem, θ, width, scoring) -> Bool   [true]  outside the timing bracket
#   refine_external!(ws, lo, problem, θ, width, scoring) -> θ′ | nothing   the bracketed call
#
# `width` is an absolute loss width (`LossWidth`) or a fixed noise scale
# (`FixedS2(s2)`, a role-isolated control's plug-in scale): by default a fixed
# scale is the loss width `√s2` (`LossWidth(1.0, s2)`); a refiner that scales
# its own width by a multiple (an adaptive arm) reads the scale itself.
#
# `frame` is the two-view frame data a solver needs beyond the problem: an
# object with the pair's calibrations `Kc1`, `Kc2` (3×3) and its
# correspondences `cs` in the problem's frame (a `Csponds` of `Point2{Float64}`)
# — VisualGeometryToolkitEval's `PairGeom` on a benchmark pool, or any caller's value
# with those properties.
# `scoring` is the refit's scoring template, for an arm that anchors its width
# on a consensus scale. `θ′` is the problem's model type.
# =============================================================================

"""
    external_workspace(lo, problem, frame, scoring) -> ws

The per-(pair, arm) workspace of an external refiner `lo` (see
[`refine_external!`](@ref)): whatever the solver reads beyond the problem —
from the two-view `frame` (`frame.Kc1`, `frame.Kc2`, `frame.cs`; see the
contract above) a point matrix, the solver's calibrations, a consensus
workspace — built once and reused by every refit of that pair. Default
`nothing`.
"""
external_workspace(::Union{Nothing, AbstractOptimizationStrategy}, _problem, _frame, _scoring) = nothing

"""
    prepare_external!(ws, lo, problem, θ, width::LossWidth, scoring) -> Bool

The part of an external refit that runs OUTSIDE its timing bracket, before it
(a consensus-scale bootstrap). `false` means no refit is made — the call is
neither timed nor counted, and the refit falls back. Default `true`.
"""
prepare_external!(_ws, ::Union{Nothing, AbstractOptimizationStrategy}, _problem, _θ, _width, _scoring) = true

# A fixed noise scale is, by default, the absolute loss width √s2.
refine_external!(ws, lo::Union{Nothing, AbstractOptimizationStrategy}, problem, θ, f::FixedS2, scoring) =
    refine_external!(ws, lo, problem, θ, LossWidth(1.0, scale_s2(f)), scoring)

"""
    refine_external!(ws, lo, problem, θ, width, scoring) -> θ′ | nothing

The `ExternalRefit` member of `RefitFamily`: refine the model `θ` of `problem`
with the external solver `lo` at the absolute loss width `width::LossWidth`
(`loss_width(width)`), on its workspace `ws` ([`external_workspace`](@ref)).
A fixed noise scale `FixedS2(s2)` is the loss width `√s2` unless the
refiner states otherwise.
Returns the refined model (the problem's model type) or `nothing` when the
solver produced none. A pure model → model refine: no seed sweep, no
re-score. The owner of the strategy implements it.
"""
function refine_external! end
# The refusal: a local optimizer with no external refit at an absolute width.
refine_external!(_ws, lo::Union{Nothing, AbstractOptimizationStrategy}, _problem, _θ, ::LossWidth, _scoring) = throw(ArgumentError(
    "$(nameof(typeof(lo))) has no external refit (refine_external!) — it is not an ExternalRefit strategy"))

