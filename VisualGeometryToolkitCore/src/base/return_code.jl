# =============================================================================
# The termination record of every fit (the robust-estimation contract, §14)
# =============================================================================
#
# A fit says WHY it stopped in two parts: the standard verdict, a
# `ReturnCode.T` (SciMLBase.ReturnCode's names and success rule, matched — not a
# dependency), and the stack's detail, a `StopRule` (which tolerance, cap or
# rule fired). A producer records both in its metadata (`return_code`,
# `stop_rule`: the FitResult core already owns the name `retcode`), declares
# the `return_code` accessor beside its type, and `retcode(r)` reads the code. `converged(r)` is `successful_retcode(retcode(r))`.
#
# THE TRANSITION (until every producer records the code): the FitResult core keeps
# its `retcode::Symbol` field, written from the code by `retcode_symbol`, and
# `retcode(r)` returns the metadata's `ReturnCode.T` for a producer that records
# one and the core Symbol otherwise — a property of the metadata TYPE, so each is
# inferred. When the last producer records it, the core field becomes the
# `ReturnCode.T` and the metadata copy and the Symbol go (Breaking).
# =============================================================================

"""
    ReturnCode

Why a fit stopped: the standard verdict, modelled on `SciMLBase.ReturnCode`
(SciMLBase 3.58; the Julia ecosystem's return-code interface). The enum type is
`ReturnCode.T` and each value a module property (`ReturnCode.Success`), as with
SciMLBase's EnumX enum; the names, integer values and success rule
([`successful_retcode`](@ref)) are SciMLBase's, so a later SciMLBase bridge is
a reinterpretation. Only the values the stack produces are defined; their
meaning is SciMLBase's:

- `ReturnCode.Success` — the solve succeeded: a tolerance or stop rule ended
  an iteration (an LM's step-size or rejection stop once its stationarity
  postcondition holds), a closed-form solve, a RANSAC that reached its
  confidence or drew every sample. Successful.
- `ReturnCode.Terminated` — a user callback ended the solve (NLLSsolver's
  callback flags). Successful, as in SciMLBase.
- `ReturnCode.MaxIters` — the iteration cap was reached with no tolerance met
  on the last iteration; for a RANSAC, the trial budget was drawn without
  reaching the confidence. Not successful.
- `ReturnCode.MaxTime` — the time cap. Not successful.
- `ReturnCode.Stalled` — an LM stopped (step size, rejections) at a point that
  fails the stationarity postcondition. Not successful.
- `ReturnCode.InitialFailure` — the start was unusable: too few data or a
  degenerate configuration before any iteration, or a non-finite cost or
  gradient at the start point. Not successful.
- `ReturnCode.InternalLinearSolveFailed` — a linear system inside the fit
  (a weighted least-squares or normal-equation solve) could not be solved. Not
  successful.
- `ReturnCode.Failure` — the solve failed otherwise: a non-finite cost, step or
  gradient during it, a failed model block, or a RANSAC that found no model.
  Not successful.

Compare with [`successful_retcode`](@ref), never `== ReturnCode.Success`; the
rule that fired is the result's [`stop_rule`](@ref).
"""
baremodule ReturnCode
using Base: @enum
@enum T::Int32 begin
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

"""
    successful_retcode(code::ReturnCode.T) -> Bool

Whether `code` is a success state — SciMLBase's rule over the values the stack
defines: `ReturnCode.Success` and `ReturnCode.Terminated`. The one success
test; [`converged`](@ref) reads it.
"""
successful_retcode(code::ReturnCode.T) = code == ReturnCode.Success || code == ReturnCode.Terminated

"""
    StopRule

Which rule ended a fit — the detail behind its [`ReturnCode`](@ref) (the
verdict), read by [`stop_rule`](@ref):

- `NoRule` — no rule fired: a run that never iterated (a degenerate start, a
  non-finite start point);
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
  was reached).

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
end

for (v, doc) in (
        (:NoRule, "no rule fired: the run never iterated"),
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
    )
    @eval @doc $("`$v`: a [`StopRule`](@ref) — $doc.") $v
end

"""
    retcode_symbol(code::ReturnCode.T, rule::StopRule) -> Symbol

The FitResult core's `retcode::Symbol` written from a recorded code and rule —
the transition's one map, so the two never disagree: `Success` is `:solved`
for a `ClosedForm` rule and `:converged` otherwise; `Terminated` `:converged`;
`MaxIters` `:max_iterations`; `MaxTime` `:max_time`; `Stalled` `:stalled`;
`InitialFailure` `:degenerate`, or `:solve_failed` for a `NonFinite` start;
`InternalLinearSolveFailed` `:solve_failed`; `Failure` `:no_model` when a
RANSAC's trial rule ended it (`TrialBudget`, `AdaptiveCount`) and
`:solve_failed` otherwise.
"""
function retcode_symbol(code::ReturnCode.T, rule::StopRule)
    code == ReturnCode.Success && return rule == ClosedForm ? :solved : :converged
    code == ReturnCode.Terminated && return :converged
    code == ReturnCode.MaxIters && return :max_iterations
    code == ReturnCode.MaxTime && return :max_time
    code == ReturnCode.Stalled && return :stalled
    code == ReturnCode.InitialFailure && return rule == NonFinite ? :solve_failed : :degenerate
    code == ReturnCode.Failure && return (rule == TrialBudget || rule == AdaptiveCount) ? :no_model : :solve_failed
    return :solve_failed                                 # InternalLinearSolveFailed
end

# The verdicts in increasing severity, for `worst_retcode`.
_severity(c::ReturnCode.T) =
    c == ReturnCode.Success ? 0 : c == ReturnCode.Terminated ? 1 : c == ReturnCode.Stalled ? 2 :
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

"""
    return_code(metadata) -> ReturnCode.T | Nothing

The [`ReturnCode`](@ref) a fit's metadata records, or `nothing` for a metadata
that records none (the default; the transition's other producers). A metadata
type that records one declares it beside its definition,
`return_code(m::MyMetadata) = m.return_code`, with a `stop_rule::StopRule`
field (or its own [`stop_rule`](@ref) method) beside it; [`retcode`](@ref)
then reads it.
"""
return_code(_metadata) = nothing

"""
    retcode(r::FitResult) -> ReturnCode.T | Symbol

Why the fit stopped. For a producer whose metadata records the termination
([`return_code`](@ref), [`stop_rule`](@ref)), its [`ReturnCode`](@ref)
`ReturnCode.T`; during the transition, for any other producer, the FitResult
core's Symbol (`:converged`, `:max_iterations`, `:degenerate`, `:solve_failed`,
`:stalled`, `:solved`, …). Inferred per metadata type. Test success with
[`converged`](@ref), never by comparing the code; the rule that fired is
[`stop_rule`](@ref).
"""
retcode(r::FitResult) = _retcode(r, return_code(getfield(r, :metadata)))
_retcode(r::FitResult, ::Nothing) = getfield(r, :retcode)
_retcode(::FitResult, code::ReturnCode.T) = code

"""
    stop_rule(result) -> StopRule

Which rule ended the fit ([`StopRule`](@ref)), the detail behind its
[`retcode`](@ref): the metadata's `stop_rule`. A metadata that records it
otherwise adds a method on its type; a result without metadata records none
(`ArgumentError`).
"""
stop_rule(r::FitResult) = stop_rule(getfield(r, :metadata))
stop_rule(m) = m.stop_rule
stop_rule(::Nothing) = throw(ArgumentError("a result without metadata records no stop rule"))

"""
    converged(r::FitResult) -> Bool

Whether the fit succeeded — STRICT: [`successful_retcode`](@ref) of its
[`retcode`](@ref), so a run that hit its iteration (or RANSAC trial) cap, its
time cap or a stall did not converge, and a closed-form solve did. During the
transition, a producer that records no code answers `retcode(r) === :converged`.
For a RANSAC, whether it FOUND a model is [`found_model`](@ref). Also reachable
as the property `r.converged`.
"""
converged(r::FitResult) = _converged(retcode(r))
_converged(code::ReturnCode.T) = successful_retcode(code)
_converged(code::Symbol) = code === :converged

"""
    found_model(r::FitResult) -> Bool

Whether the fit returned an estimate — a RANSAC's "a model was found"
(`estimate(r) !== nothing`), whatever its [`retcode`](@ref): a search that
drew its whole trial budget without reaching its confidence still found one.
"""
found_model(r::FitResult) = getfield(r, :estimate) !== nothing
