# =============================================================================
# The bootstrap of a tuned argmin over the validation units
# =============================================================================

"""
    unit_statistics(criterion::TuningCriterion, errors::AbstractMatrix) -> Matrix{Float64}

What the bootstrap of a selection needs of one candidate's validation
`errors` (a `units × items` matrix, one row per sampling unit): one row per
unit, from which the criterion's objective over ANY multiset of units is
exact ([`SelectionBootstrap`](@ref)). For [`MeanAverageAccuracy`](@ref) — a
mean of per-item scores — the row is the unit's score sum and item count
(VisualGeometryToolkitCore `maa_score`), two numbers in place of `items`; for any
other criterion it is the errors themselves.
"""
unit_statistics(::TuningCriterion, errors::AbstractMatrix) = collect(Float64, errors)
unit_statistics(c::MeanAverageAccuracy, errors::AbstractMatrix) =
    hcat([sum(e -> maa_score(e; c.max_error), r; init = 0.0) for r in eachrow(errors)],
         fill(Float64(size(errors, 2)), size(errors, 1)))

"""
    stats_objective(criterion, stats) -> Real

The criterion's objective (minimised) over [`unit_statistics`](@ref) rows.
"""
function stats_objective end
stats_objective(c::TuningCriterion, S::AbstractMatrix) = tune_objective(c, vec(S))
stats_objective(::MeanAverageAccuracy, S::AbstractMatrix) = -sum(view(S, :, 1)) / sum(view(S, :, 2))

"""
    SelectionBootstrap(criterion::TuningCriterion; n_resamples)

The sampling distribution of a cross-validation's selection, a `ClosedFormKind` that draws (its solve takes an `rng`):
`solve(EstimationProblem(Hyperparameter, stats), SelectionBootstrap(criterion; n_resamples); rng)`. `stats[c]` is
candidate `c`'s [`unit_statistics`](@ref) (one row per sampling unit — a frame, a pair, a scene). Each of
`n_resamples` resamples draws the units WITH replacement (`rand(rng, 1:n, n)`, `n` the number of units) and selects
the candidate whose criterion objective over the resampled rows is smallest (the first such candidate on a tie). The
candidates are any flat set — the cells of an N-dimensional search included; the caller maps the indices to their
coordinates and reads the interval it wants (per-axis quantiles, say).

`u` is the selection per resample (`Vector{Int}`, in draw order), `n` the units, `stats.nsteps` the resamples,
`(Success, ClosedForm)`. No candidate, or candidates with different units, is an `ArgumentError`.

This is the bootstrap of the SELECTION over a fixed candidate set. The windowed budget bootstrap (Evaluation
`bootstrap_trials`) differs in what it resamples — budget draws of `n` validation units for a selection, each draw
evaluated on the test side — and is not a special case of it.
"""
struct SelectionBootstrap{C <: TuningCriterion} <: AbstractEstimationAlgorithm
    criterion::C
    n_resamples::Int
end
SelectionBootstrap(criterion::TuningCriterion; n_resamples::Integer) = SelectionBootstrap(criterion, Int(n_resamples))
estimator_kind(::SelectionBootstrap) = ClosedFormKind()
supports_solve_rng(::EstimationProblem{Hyperparameter}, ::SelectionBootstrap) = true

function __solve(prob::EstimationProblem{Hyperparameter, <:AbstractVector{<:AbstractMatrix}}, alg::SelectionBootstrap; rng)
    stats = prob.f
    isempty(stats) && throw(ArgumentError("SelectionBootstrap: no candidates"))
    n = size(first(stats), 1)
    all(S -> size(S, 1) == n, stats) ||
        throw(ArgumentError("SelectionBootstrap: every candidate needs the same units (rows)"))
    selections = map(1:alg.n_resamples) do _
        rows = rand(rng, 1:n, n)
        argmin([stats_objective(alg.criterion, S[rows, :]) for S in stats])
    end
    return FitResult(selections, ReturnCode.Success, ClosedForm, SolveStats(; nsteps = alg.n_resamples), NaN, n, nothing)
end
