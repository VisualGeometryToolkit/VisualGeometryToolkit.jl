# Termination Record

```@meta
CurrentModule = VisualGeometryToolkitCore
```

Why a solve stopped is its termination record: the verdict, a
[`ReturnCode`](@ref) (`ReturnCode.T`, SciMLBase's names and success rule),
the result's field `r.retcode`, and the rule that fired, a [`StopRule`](@ref),
its field `r.stop_rule`. Both are in every [`FitResult`](@ref), so every producer
records them. Success is strict, [`successful_retcode`](@ref)`(r)` of the code:
a cap or a stall is not a success. Whether a RANSAC found a model is
[`found_model`](@ref).

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
found_model
retcode_symbol
retcode_from_symbol
worst_retcode
```
