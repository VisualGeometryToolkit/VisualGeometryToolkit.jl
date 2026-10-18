# =============================================================================
# src/tuning/schemes.jl — the cross-validation SCHEMES: one type per family of
# tuner, whose fields are the scheme's complete parameter set. A `[tuning.<name>]`
# TOML table names its scheme with `type` and is parsed into one of these.
# =============================================================================

"""
    TuningScheme

A family of cross-validation, with every parameter it runs under:
[`CoarseFine`](@ref) and [`JointCoarseFine`](@ref) (the one coarse-fine
search of Evaluation cv_probe.jl, in one and in N dimensions),
[`WindowedBootstrap`](@ref), [`Factorial`](@ref).

A new search backend (a surrogate model over the same axes, say) is a new
`TuningScheme` whose search returns the record of Evaluation `coarse_search` /
`fine_search` — `selected`, `boundary_hit`, `evaluated` — so the artifact,
its gated reader and the bootstrap serve it unchanged.
"""
abstract type TuningScheme end

"""
    CoarseFine(coarse, extension, edge_margin, fine, criterion, cross_validation)

The coarse-fine search of ONE scalar (Evaluation cv_probe.jl): the
one-axis case of [`JointCoarseFine`](@ref), its axis spelled flat — sweep the
`coarse` axis, `extend` it by `extension.npts` points on a side whenever the
argmin lies within `edge_margin` points of that edge (at most
`extension.max_rounds` per side), then sweep `fine` over the coarse argmin's
bracketing neighbours and take the free argmin. `criterion` aggregates a
candidate's validation errors over the validation units of
`cross_validation`. TOML tag `"coarse_fine"`.
"""
struct CoarseFine <: TuningScheme
    coarse::TuningAxis
    extension::GridExtension
    edge_margin::Int
    fine::GridSpacing
    criterion::TuningCriterion
    cross_validation::CrossValidation
    function CoarseFine(coarse::TuningAxis, extension::GridExtension, edge_margin::Integer,
                        fine::GridSpacing, criterion::TuningCriterion, cross_validation::CrossValidation)
        edge_margin >= 0 || throw(ArgumentError("CoarseFine: edge_margin = $edge_margin must be ≥ 0"))
        return new(coarse, extension, Int(edge_margin), fine, criterion, cross_validation)
    end
end

"""
    CoarseFineAxis(coarse, extension, fine, expensive)

One axis of a [`JointCoarseFine`](@ref) search: its `coarse` axis (window rule
+ spacing), its expand-on-boundary rule, the `fine` spacing over the coarse
argmin's bracket, and whether it is `expensive` — a value of an expensive axis
costs a run the cheap axes reuse (a detection run, say), so the search asks
for the cells one expensive value at a time. Untagged; defaults
`[types.coarse_fine_axis]`.
"""
struct CoarseFineAxis
    coarse::TuningAxis
    extension::GridExtension
    fine::GridSpacing
    expensive::Bool
end

"""
    Incumbent(values, confidence, n_resamples, seed)

The setting in force before a search (`values`: axis name → value). The
search's selection replaces it only when a paired bootstrap of the units
(`n_resamples` draws, `StableRNG(seed)`) puts the selection's objective below
the incumbent's with probability at least `confidence`. Untagged; defaults
`[types.incumbent]`.
"""
struct Incumbent
    values::Dict{String,Float64}
    confidence::Float64
    n_resamples::Int
    seed::Int
end

"""
    JointCoarseFine(axes, edge_margin, criterion, cross_validation, incumbent = nothing)

The coarse-fine search over the product of named `axes`
([`CoarseFineAxis`](@ref)), jointly: the coarse product grid, each axis
extended on a side while the joint argmin lies within `edge_margin` points of
that face, then the product of each axis's fine bracket about the joint
argmin and its free argmin. [`CoarseFine`](@ref) is its one-axis case. An
optional [`Incumbent`](@ref) is kept unless the selection beats it (Evaluation
`fine_search`). TOML tag `"joint_coarse_fine"`.
"""
struct JointCoarseFine <: TuningScheme
    axes::Dict{String,CoarseFineAxis}
    edge_margin::Int
    criterion::TuningCriterion
    cross_validation::CrossValidation
    incumbent::Union{Nothing,Incumbent}
    function JointCoarseFine(axes::AbstractDict, edge_margin::Integer, criterion::TuningCriterion,
                             cross_validation::CrossValidation, incumbent = nothing)
        edge_margin >= 0 || throw(ArgumentError("JointCoarseFine: edge_margin = $edge_margin must be ≥ 0"))
        return new(Dict{String,CoarseFineAxis}(axes), Int(edge_margin), criterion, cross_validation, incumbent)
    end
end

"""
    WindowedBootstrap(scale_multiple_window, outlier_halfwidth, loss_width,
                      scale_multiple, criterion, cross_validation, budget_sampling)

The bootstrap tuning harness (Evaluation `WindowedGridMethod`): per draw of
`budget_sampling` from the validation units of `cross_validation`, the free argmin of `criterion` over the native-grid columns inside
the scale-multiple window at the draw's reference scale. `outlier_halfwidth`
and `loss_width` are the native grids, `scale_multiple` the standardized display
axis. TOML tag `"windowed_bootstrap"`.
"""
struct WindowedBootstrap <: TuningScheme
    scale_multiple_window::ScaleMultipleWindow
    outlier_halfwidth::TuningAxis
    loss_width::TuningAxis
    scale_multiple::TuningAxis
    criterion::TuningCriterion
    cross_validation::CrossValidation
    budget_sampling::BudgetSampling
end

"""
    Factorial(axes, criterion, cross_validation)

Every combination of the named `axes` (hand-chosen settings of a method's
configuration keys), scored by `criterion` over the validation units of
`cross_validation`. TOML tag `"factorial"`.

NOT a windowed scheme: there is no reference scale and no standardized window,
and nothing about it is extended on a boundary. Its consumers live outside
this package: a downstream product-grid spec reads its axes through it, and
a joint operating-point search is one. Deleting it on the
strength of a grep over `src/` and `test/` alone breaks them.
"""
struct Factorial <: TuningScheme
    axes::Dict{String,TuningAxis}
    criterion::TuningCriterion
    cross_validation::CrossValidation
end

"""
    TuningSpec(name, scheme)

One named cross-validation of record: the `[tuning.<name>]` table of a
config, parsed ([`tuning_spec`](@ref)). Persisted with every artifact the run
writes ([`save_tuning_artifact`](@ref)), with its [`tuning_hash`](@ref).
"""
struct TuningSpec
    name::String
    scheme::TuningScheme
end

# Structural equality for the spec types (fields compared with `==`), so a
# round trip TOML → type → TOML → type is checkable with `==`.
const _SPEC_TYPES = Union{SpacedAxis, ListedAxis, RegisteredSceneSplit, StratifiedSampling, UniformSampling,
                          CrossValidation, BudgetSampling, CoarseFine, CoarseFineAxis, Incumbent, JointCoarseFine,
                          WindowedBootstrap, Factorial, TuningSpec}
Base.:(==)(a::T, b::T) where {T<:_SPEC_TYPES} = all(getfield(a, f) == getfield(b, f) for f in fieldnames(T))
Base.hash(x::_SPEC_TYPES, h::UInt) = foldl((acc, f) -> hash(getfield(x, f), acc), fieldnames(typeof(x));
                                           init = hash(nameof(typeof(x)), h))
