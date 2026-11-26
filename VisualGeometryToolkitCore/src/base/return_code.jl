# =============================================================================
# The termination record of every fit (the robust-estimation contract, §14)
# =============================================================================
#
# A fit says WHY it stopped in two parts: the standard verdict, a
# `ReturnCode.T` (SciMLBase.ReturnCode's names and success rule, matched — not a
# dependency), and the stack's detail, a `StopRule` (which tolerance, cap or
# rule fired). Both are FitResult fields (`r.retcode`, `r.stop_rule`): every
# producer records them through the one constructor, so no result is without a
# code and `successful_retcode(r)` is `successful_retcode(r.retcode)` for every result.
# A producer records the pair itself, and a reader reads it: no Symbol is ever a retcode (a persisted record writes
# the enum's own name). The types load before FitResult (base/fit_result.jl); the
# readings of a result are in base/termination_accessors.jl.
# =============================================================================

"""
    ReturnCode

Why a fit stopped: the standard verdict, modelled on `SciMLBase.ReturnCode`
(SciMLBase 3.58; the Julia ecosystem's return-code interface). The enum type is
`ReturnCode.T` and each value a module property (`ReturnCode.Success`), as with
SciMLBase's EnumX enum; the names, integer values and success rule
([`successful_retcode`](@ref VisualGeometryToolkitCore.successful_retcode)) are SciMLBase's, so a later SciMLBase bridge is
a reinterpretation. Only the values the stack produces are defined; their
meaning is SciMLBase's:

- `ReturnCode.Default` — the solving process has not been run (SciMLBase's
  default): a record of a solve that never ran, such as a RANSAC result's
  local-optimization record when no local optimization ran. Not successful.
- `ReturnCode.Success` — the solve succeeded: a tolerance or stop rule ended
  an iteration (an LM's cost-decrease or step-size tolerance, converged by
  definition; its rejection cap once its stationarity postcondition holds), a
  closed-form solve, a RANSAC that reached its
  confidence or drew every sample. Successful.
- `ReturnCode.Terminated` — a user callback ended the solve (NLLSsolver's
  callback flags). Successful, as in SciMLBase.
- `ReturnCode.MaxIters` — the iteration cap was reached with no tolerance met
  on the last iteration; for a RANSAC, the trial budget was drawn without
  reaching the confidence. Not successful.
- `ReturnCode.MaxTime` — the time cap. Not successful.
- `ReturnCode.Stalled` — an LM stopped at a point that fails the stationarity
  postcondition: the host LM's consecutive-rejection cap (a tolerance stop, the
  cost decrease or the step size, is `Success`: Core 0.11.0's owner ruling), the
  device LM's (`StaticLevenbergMarquardt`) stops short of its success rule. Not successful.
- `ReturnCode.InitialFailure` — the start was unusable: too few data or a
  degenerate configuration before any iteration, or a non-finite cost or
  gradient at the start point. Not successful.
- `ReturnCode.InternalLinearSolveFailed` — a linear system inside the fit
  (a weighted least-squares or normal-equation solve) could not be solved. Not
  successful.
- `ReturnCode.Failure` — the solve failed otherwise: a non-finite cost, step or
  gradient during it, a failed model block, or a RANSAC that found no model.
  Not successful.

Compare with [`successful_retcode`](@ref VisualGeometryToolkitCore.successful_retcode), never `== ReturnCode.Success`; the
rule that fired is the result's field `stop_rule`, a [`StopRule`](@ref VisualGeometryToolkitCore.StopRule).
"""
baremodule ReturnCode
using Base: @enum
# public, so another package's qualified `ReturnCode.Success` is a public access (each is documented below)
public T, Default, Success, Terminated, MaxIters, InitialFailure, Failure, MaxTime, Stalled, InternalLinearSolveFailed
@enum T::Int32 begin
    Default = 0
    Success = 1
    Terminated = 2
    MaxIters = 4
    InitialFailure = 8
    Failure = 10
    MaxTime = 15
    Stalled = 18
    InternalLinearSolveFailed = 20
end
end

@doc "`ReturnCode.T`: the [`ReturnCode`](@ref) enum type (SciMLBase's `ReturnCode.T`)." ReturnCode.T
for (v, doc) in (
        (:Default, "the solving process has not been run (SciMLBase's default). Not successful"),
        (:Success, "the solve succeeded (a tolerance or stop rule, a closed form, a RANSAC's confidence or every sample). Successful"),
        (:Terminated, "a callback (or LODR's lifted mixture leaving its basin) ended the solve. Successful, as in SciMLBase"),
        (:MaxIters, "the iteration cap, or a RANSAC's trial budget, with no tolerance or confidence met. Not successful"),
        (:InitialFailure, "the start was unusable: too few data, a degenerate configuration, a non-finite start point. Not successful"),
        (:Failure, "the solve failed otherwise (non-finite values, a failed model block, no RANSAC model). Not successful"),
        (:MaxTime, "the time cap. Not successful"),
        (:Stalled, "an LM stopped at a point that fails the stationarity postcondition. Not successful"),
        (:InternalLinearSolveFailed, "a linear system inside the fit could not be solved. Not successful"),
    )
    @eval @doc $("`ReturnCode.$v`: a [`ReturnCode`](@ref) — $doc.") ReturnCode.$v
end

"""
    successful_retcode(code::ReturnCode.T) -> Bool

Whether `code` is a success state — SciMLBase's rule over the values the stack
defines: `ReturnCode.Success` and `ReturnCode.Terminated`. The one success
test; a [`FitResult`](@ref)'s `successful_retcode(r)` reads it on `r.retcode`.
"""
successful_retcode(code::ReturnCode.T) = code == ReturnCode.Success || code == ReturnCode.Terminated

"""
    StopRule

Which rule ended a fit — the detail behind its [`ReturnCode`](@ref) (the
verdict), a result's `stop_rule` field:

- `NoRule` — no rule fired: a run that never iterated (a degenerate start, a
  non-finite start point), or a host-LM run NLLSsolver stopped after a cost
  disagreement with no tolerance met (`nlls_termination`, Core 0.11.1);
- `ClosedForm` — a one-shot solve (no iteration), or a selection among fixed
  candidates;
- `CostDecrease` — the cost decrease fell below its tolerance (NLLSsolver's
  relative or absolute cost stop; an IRLS outer loop's robust cost; an EM's
  negative log-likelihood under a held model);
- `IncrementSize` — the update fell below its tolerance (NLLSsolver's step stop; an
  IRLS fit's parameter change);
- `RejectionCap` — NLLSsolver's consecutive cost increases (`maxfails`);
- `ParameterChange` — the EM's `ParameterChangeStop`;
- `NoImprovement` — the EM's `NoImprovementStop` (the incumbent's cost did not
  fall);
- `ExactFit` — every residual is zero;
- `ToleranceMet` — a search met its accuracy tolerance (a spline spacing
  search);
- `LadderConverged` — a graduated-non-convexity ladder converged (LODR);
- `LeftBasin` — the iterate left the seed's basin of attraction (LODR);
- `ShapeGate` — the fitted shape failed a validity gate (BlobBoards' blob fit:
  not a peak, or an aspect above its maximum); its code is `Failure`;
- `IterationCap` — the iteration cap;
- `TimeCap` — the time cap;
- `NonFinite` — a non-finite cost, step or gradient, or a failed model block;
- `Callback` — a user callback;
- `TrialBudget` — a RANSAC drew its whole trial budget (the cap, or every
  sample of an exhaustive sampler);
- `AdaptiveCount` — a RANSAC drew its adaptive trial count (its confidence
  was reached);
- `SeedKept` — a RANSAC's score guard kept the seed: the refined model scored
  below it (recorded with the local optimization's own code).

An `@enum` (isbits): a metadata that carries it stays concretely typed.
"""
@enum StopRule::Int32 begin
    NoRule
    ClosedForm
    CostDecrease
    IncrementSize
    RejectionCap
    ParameterChange
    NoImprovement
    ExactFit
    ToleranceMet
    LadderConverged
    LeftBasin
    ShapeGate
    IterationCap
    TimeCap
    NonFinite
    Callback
    TrialBudget
    AdaptiveCount
    SeedKept
end

for (v, doc) in (
        (:NoRule, "no rule fired: the run never iterated, or the host LM stopped after a cost disagreement"),
        (:ClosedForm, "a one-shot solve, or a selection among fixed candidates"),
        (:CostDecrease, "the cost decrease fell below its tolerance"),
        (:IncrementSize, "the update fell below its tolerance"),
        (:RejectionCap, "NLLSsolver's consecutive cost increases (`maxfails`)"),
        (:ParameterChange, "the EM's `ParameterChangeStop`"),
        (:NoImprovement, "the EM's `NoImprovementStop`"),
        (:ExactFit, "every residual is zero"),
        (:ToleranceMet, "a search met its accuracy tolerance"),
        (:LadderConverged, "a graduated-non-convexity ladder converged"),
        (:LeftBasin, "the iterate left the seed's basin of attraction"),
        (:ShapeGate, "the fitted shape failed a validity gate (its code is `Failure`)"),
        (:IterationCap, "the iteration cap"),
        (:TimeCap, "the time cap"),
        (:NonFinite, "a non-finite cost, step or gradient, or a failed model block"),
        (:Callback, "a user callback"),
        (:TrialBudget, "a RANSAC drew its whole trial budget"),
        (:AdaptiveCount, "a RANSAC drew its adaptive trial count (its confidence was reached)"),
        (:SeedKept, "a RANSAC's score guard kept the seed (the refined model scored below it)"),
    )
    @eval @doc $("`$v`: a [`StopRule`](@ref) — $doc.") $v
end

# The verdicts in increasing severity, for `worst_retcode`.
_severity(c::ReturnCode.T) =
    c == ReturnCode.Default ? -1 : c == ReturnCode.Success ? 0 : c == ReturnCode.Terminated ? 1 : c == ReturnCode.Stalled ? 2 :
    c == ReturnCode.MaxIters ? 3 : c == ReturnCode.MaxTime ? 4 : c == ReturnCode.Failure ? 5 :
    c == ReturnCode.InternalLinearSolveFailed ? 6 : 7   # InitialFailure

"""
    worst_retcode(a::Tuple{ReturnCode.T, StopRule}, b::Tuple{ReturnCode.T, StopRule}, more...) -> Tuple{ReturnCode.T, StopRule}

The termination of a fit that ran several solves (stages, rounds, the cycles of
an outer loop — its own cap counted as one), from each solve's
`(code, rule)`: the most severe code, in the order `InitialFailure` >
`InternalLinearSolveFailed` > `Failure` > `MaxTime` > `MaxIters` > `Stalled` >
`Terminated` > `Success`, with that solve's rule. On a tie the LATER solve's
pair is kept (the final solve's rule is the informative one).
"""
worst_retcode(a::Tuple{ReturnCode.T, StopRule}) = a
worst_retcode(a::Tuple{ReturnCode.T, StopRule}, b::Tuple{ReturnCode.T, StopRule}) =
    _severity(b[1]) >= _severity(a[1]) ? b : a
worst_retcode(a::Tuple{ReturnCode.T, StopRule}, b::Tuple{ReturnCode.T, StopRule}, more::Tuple{ReturnCode.T, StopRule}...) =
    worst_retcode(worst_retcode(a, b), more...)

