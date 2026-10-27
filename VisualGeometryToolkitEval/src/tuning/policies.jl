# =============================================================================
# src/tuning/policies.jl — the strategy types of a cross-validation: an axis
# (window rule + spacing, or listed values), the expand-on-boundary rule and
# the objective (the cross-validation's sampling of units is sampling.jl). Each is a
# type whose fields are its complete parameter set; no module constant or
# buried default influences a result.
# =============================================================================

# -----------------------------------------------------------------------------
# Axes
# -----------------------------------------------------------------------------

"""
    TuningAxis

One tuning axis of a spec: [`SpacedAxis`](@ref) (a window rule and a
spacing) or [`ListedAxis`](@ref) (explicit values). `grid_spec(axis; context...)`
realizes it as the persisted [`GridSpec`](@ref).
"""
abstract type TuningAxis end

"""
    SpacedAxis(window::WindowRule, spacing::GridSpacing)

An axis whose grid is `spacing` over the window `window` evaluates to. TOML
tag `"spaced"`.
"""
struct SpacedAxis <: TuningAxis
    window::WindowRule
    spacing::GridSpacing
end

"""
    ListedAxis(values)

An axis of hand-chosen settings (a factorial tuning's axis). TOML tag
`"listed"`.
"""
struct ListedAxis <: TuningAxis
    values::Vector{Float64}
    ListedAxis(values::AbstractVector{<:Real}) = new(ListedGrid(values).values)
end

"""
    grid_spec(axis::TuningAxis; context...) -> GridSpec

The realized, persistable grid spec of `axis`: the window its rule evaluates
to under `context` ([`window`](@ref)) with its spacing, unextended; or the
listed values.
"""
grid_spec(a::SpacedAxis; context...) = SpacedGrid(window(a.window; context...)..., a.spacing)
grid_spec(a::ListedAxis; _...) = ListedGrid(a.values)

# -----------------------------------------------------------------------------
# Boundary rules
# -----------------------------------------------------------------------------

"""
    GridExtension(npts, max_rounds)

The expand-on-boundary rule of a GRID axis: on a boundary argmin, `extend` the
grid by `npts` points on the clipped side, at most `max_rounds` rounds
(a still-clipped argmin is then recorded and refused by the readers).
"""
struct GridExtension
    npts::Int
    max_rounds::Int
    function GridExtension(npts::Integer, max_rounds::Integer)
        npts >= 1 && max_rounds >= 0 ||
            throw(ArgumentError("GridExtension: need npts ≥ 1 and max_rounds ≥ 0, got ($npts, $max_rounds)"))
        return new(Int(npts), Int(max_rounds))
    end
end

# -----------------------------------------------------------------------------
# Objectives — the tuning criteria of the stack, all of them, here.
#
# `TuningCriterion`, `MedianError`, `MAA` (since folded into
# `MeanAverageAccuracy`), `tune_objective`, `_inf_failed` and
# `tuning_criterion_label` moved here from RANSACScoringDoneRight on
# 2026-10-02, completing the 2026-07-18 migration of the eval subsystem: they
# are evaluation policy, every caller was already in this package, and nothing
# in the scoring package used them (its one internal consumer, `bracket_key`,
# had no caller and was deleted). A criterion is now defined in one place,
# beside the schemes that select it and the TOML tags that name it.
# -----------------------------------------------------------------------------

# =============================================================================
# Tuning criterion — how to aggregate per-pair errors into an objective
# =============================================================================

"""
    TuningCriterion

The objective a cross-validation minimizes over the validation errors of a
candidate (`tune_objective(criterion, errors)`); each concrete criterion is
named in TOML by its tag.
"""
abstract type TuningCriterion end

"""
    MedianError()

The median validation pose error with every failure (a non-finite error)
counted as `+Inf`. TOML tag `"median_error"`.
"""
struct MedianError <: TuningCriterion end

# Failed refits (non-finite pose errors) count as ∞ — the basin-ladder
# convention: a failure is an arbitrarily bad outcome, ordered into the top
# of the pool, so medians/mAA stay defined (NaN would poison them).
_inf_failed(e) = [isfinite(x) ? x : Inf for x in e]

"""
    tune_objective(criterion::TuningCriterion, errors) -> Real

The value a cross-validation MINIMIZES for one candidate: `criterion` applied
to its validation `errors` (one vector per scene for a per-scene criterion).
"""
tune_objective(::MedianError, errors) = median(_inf_failed(errors))

"""
    tuning_criterion_label(criterion) -> String

The display label of a criterion, as persisted in result rows.
"""
tuning_criterion_label(::MedianError) = "median"

"""
    FiniteMedian()

The median over the FINITE validation errors, `+Inf` when there are none (so
an all-failed column loses every argmin). TOML tag `"finite_median"`.
"""
struct FiniteMedian <: TuningCriterion end
tune_objective(::FiniteMedian, errors) = finite_median(errors; empty = Inf)
tuning_criterion_label(::FiniteMedian) = "median"

"""
    SceneMeanMedian()

The mean over scenes of each scene's median FINITE validation error (Stats
`scene_mean_median`: scenes without a finite error are left out; `NaN` when
none is left). `errors` is one error vector per scene. TOML tag
`"scene_mean_median"`.
"""
struct SceneMeanMedian <: TuningCriterion end
tune_objective(::SceneMeanMedian, scene_errors) = scene_mean_median(scene_errors)
tuning_criterion_label(::SceneMeanMedian) = "scene_mean_median"

"""
    SceneMeanLowerMedian()

The mean over scenes of each scene's [`lower_median`](@ref) validation error
(scenes without a finite median are left out; `NaN` when none is left).
`errors` is one error vector per scene. TOML tag `"scene_mean_lower_median"`.
"""
struct SceneMeanLowerMedian <: TuningCriterion end
tune_objective(::SceneMeanLowerMedian, scene_errors) =
    scene_mean(e -> lower_median(collect(Float64, e)), scene_errors)
tuning_criterion_label(::SceneMeanLowerMedian) = "scene_mean_lower_median"

"""
    MeanAverageAccuracy(max_error)

The continuous mAA@`max_error` of the validation errors (VisualGeometryToolkitCore
`Synthetic.mean_average_accuracy`; a non-finite error is a failure that stays
in the denominator), MAXIMISED (the objective is its negative). TOML tag
`"mean_average_accuracy"`.

UNITS-NEUTRAL: `max_error` is in the errors' own unit — degrees on a pose
error, mm for BlobBoards' bmvc2026 inter-board invariance. The one mAA
criterion: a second spelling `MAA(max_error_deg)` carried the identical
statistic under a field name that asserted degrees, which a criterion handed
millimetres cannot. It is retired (2026-10-03) along with its `"maa"` tag; the
`[schemes.factorial]` default that named it now names this one. The statistic
itself is VisualGeometryToolkitCore's, the stack's one definition.
"""
struct MeanAverageAccuracy <: TuningCriterion
    max_error::Float64
    function MeanAverageAccuracy(max_error::Real)
        max_error > 0 || throw(ArgumentError("MeanAverageAccuracy: max_error = $max_error must be > 0"))
        return new(Float64(max_error))
    end
end
# Through `eligible_maa`, not VgtkCore's `mean_average_accuracy` directly: Stats
# declares the mAA rule — "every figure path computes mAA through
# `eligible_maa`" — and a tuning objective that called the statistic itself
# would be a second path whose failure/eligibility convention could drift from
# the reporting one. With no mask the two are identical
# (`_eligibility(errors, ::Nothing) = trues(...)`), so this is reuse, not a
# change of objective.
tune_objective(c::MeanAverageAccuracy, errors) =
    -eligible_maa(collect(Float64, errors); T = c.max_error)
tuning_criterion_label(::MeanAverageAccuracy) = "mAA"

# -----------------------------------------------------------------------------
# Random number generators, by name
# -----------------------------------------------------------------------------

"""
    RNG_ALGORITHMS

The RNG algorithms a spec may name (`rng = "StableRNG"`): StableRNGs'
`StableRNG` only — its streams are fixed across Julia versions, which Julia's
own `MersenneTwister`/`Xoshiro` are not, so a persisted sample or stream
reproduces on any Julia. (MersenneTwister and Xoshiro were accepted until
tuning format v4.)
"""
const RNG_ALGORITHMS = Dict{String,Type}("StableRNG" => StableRNG)

function _rng_type(name::AbstractString)
    haskey(RNG_ALGORITHMS, name) ||
        throw(ArgumentError("unknown rng $(repr(name)); known: $(sort(collect(keys(RNG_ALGORITHMS))))"))
    return RNG_ALGORITHMS[name]
end
