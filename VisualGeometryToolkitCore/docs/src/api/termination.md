# Termination Record

```@meta
CurrentModule = VisualGeometryToolkitCore
```

Why a fit stopped is its termination record: the verdict, a
[`ReturnCode`](@ref) (`ReturnCode.T`, SciMLBase's names and success rule),
read by [`retcode`](@ref), and the rule that fired, a [`StopRule`](@ref), read
by [`stop_rule`](@ref). Both are in the result's core, so every producer
records them. [`converged`](@ref) is strict,
[`successful_retcode`](@ref) of the code: a cap or a stall is not converged.
Whether a RANSAC found a model is [`found_model`](@ref).

```@docs
ReturnCode
ReturnCode.T
ReturnCode.Default
ReturnCode.Success
ReturnCode.Terminated
ReturnCode.MaxIters
ReturnCode.InitialFailure
ReturnCode.Failure
ReturnCode.MaxTime
ReturnCode.Stalled
ReturnCode.InternalLinearSolveFailed
successful_retcode
StopRule
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
stop_rule
found_model
retcode_symbol
retcode_from_symbol
worst_retcode
```
