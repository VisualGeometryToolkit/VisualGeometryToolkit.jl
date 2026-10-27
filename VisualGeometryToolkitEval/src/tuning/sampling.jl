# =============================================================================
# src/tuning/sampling.jl — the sampling of a cross-validation: which scenes are
# validation and which are test (a registered scene split), which units of
# those scenes a tuning reads and an evaluation reads (a unit sampling per side),
# and the random validation-budget draws. Every random choice is seeded by
# `stable_seed` (Stats) from a seed in the spec and the NAMES it applies to, so
# a draw is a function of the spec and the data's names alone.
#
#   CrossValidation(scene_split, validation, test)
#     scene_split  RegisteredSceneSplit — named lists of configs/splits.toml
#     validation   ExhaustiveSampling | StratifiedSampling | UniformSampling
#     test         ExhaustiveSampling | StratifiedSampling | UniformSampling
#   BudgetSampling(sizes, n_draws, seed, rng) — the validation-budget draws
#
# The order of every sampler is fixed: permute, keep the fraction, apply the
# cap, then sort back into the data's order LAST — so a cap never keeps a
# prefix of the data.
# =============================================================================

# -----------------------------------------------------------------------------
# The sampling unit of a dataset (a property of the data, not of the spec)
# -----------------------------------------------------------------------------

"""
    SamplingUnit

What one unit of a dataset is — what a [`UnitSampling`](@ref) draws:
[`PairUnit`](@ref) (an image pair of a two-view dataset) or
[`QueryUnit`](@ref) (a query image of an absolute-pose dataset). A dataset
declares its own with [`sampling_unit`](@ref).
"""
abstract type SamplingUnit end

"An image pair of a two-view dataset."
struct PairUnit <: SamplingUnit end

"A query image of an absolute-pose dataset."
struct QueryUnit <: SamplingUnit end

"""
    sampling_unit(::Type{D}) -> SamplingUnit

The unit a cross-validation samples from a dataset of type `D` (its
[`SamplingUnit`](@ref)). A property of the data: every dataset type defines
its own method, and there is no fallback.
"""
function sampling_unit end

# -----------------------------------------------------------------------------
# The scene split
# -----------------------------------------------------------------------------

"""
    SceneSplit

Which scenes a cross-validation tunes on and which it reports on:
[`RegisteredSceneSplit`](@ref).
"""
abstract type SceneSplit end

"""
    RegisteredSceneSplit(name, validation_scenes, test_scenes)

A scene split registered under `name` in configs/splits.toml (the one file of
record of every scene list). A config names it (`scene_split = { type =
"registered", name = "..." }`); the parser fills in the lists from the
registry, so the complete table — and with it the spec hash and every
artifact — carries the lists the run used. Both lists are nonempty, free of
duplicates and DISJOINT; they are kept sorted. TOML tag `"registered"`.
"""
struct RegisteredSceneSplit <: SceneSplit
    name::String
    validation_scenes::Vector{String}
    test_scenes::Vector{String}
    function RegisteredSceneSplit(name::AbstractString, validation_scenes::AbstractVector{<:AbstractString},
                                  test_scenes::AbstractVector{<:AbstractString})
        isempty(name) && throw(ArgumentError("RegisteredSceneSplit: empty name"))
        v, t = sort!(String.(validation_scenes)), sort!(String.(test_scenes))
        for (side, s) in (("validation", v), ("test", t))
            isempty(s) && throw(ArgumentError("RegisteredSceneSplit $(repr(name)): no $side scenes"))
            allunique(s) || throw(ArgumentError("RegisteredSceneSplit $(repr(name)): duplicate $side scenes " *
                                                "$(unique(filter(x -> count(==(x), s) > 1, s)))"))
        end
        both = intersect(v, t)
        isempty(both) ||
            throw(ArgumentError("RegisteredSceneSplit $(repr(name)): scenes $(both) are both validation and test"))
        return new(String(name), v, t)
    end
end

"The path of the scene-split registry of record, configs/splits.toml."
const SPLIT_REGISTRY_PATH = normpath(joinpath(@__DIR__, "..", "..", "configs", "splits.toml"))
"The scene-split registry of record: every scene list of the stack (configs/splits.toml, parsed at precompile)."
const SPLIT_REGISTRY = VgtkCore.parse_defaults(SPLIT_REGISTRY_PATH)

"""
    registered_split(name) -> RegisteredSceneSplit

The split `[splits.<name>]` of configs/splits.toml. An unknown name, an
unknown key or a split violating [`RegisteredSceneSplit`](@ref)'s invariants
is an `ArgumentError`.
"""
function registered_split(name::AbstractString)
    splits = get(SPLIT_REGISTRY, "splits", Dict{String,Any}())
    haskey(splits, name) ||
        throw(ArgumentError("configs/splits.toml has no [splits.$name] (has: $(sort(collect(String, keys(splits)))))"))
    d = splits[name]
    VgtkCore.check_keys(d, ("validation_scenes", "test_scenes"), "splits.$name")
    for k in ("validation_scenes", "test_scenes")
        haskey(d, k) || throw(ArgumentError("[splits.$name] has no `$k`"))
    end
    return RegisteredSceneSplit(name, d["validation_scenes"], d["test_scenes"])
end

# -----------------------------------------------------------------------------
# The unit samplings
# -----------------------------------------------------------------------------

"""
    UnitSampling

Which units of a side's scenes a cross-validation reads: [`ExhaustiveSampling`](@ref),
[`StratifiedSampling`](@ref) (per scene) or [`UniformSampling`](@ref)
(pooled over the scenes). Applied by [`sample_units`](@ref).
"""
abstract type UnitSampling end

"""
    ExhaustiveSampling()

Every unit of every scene. TOML tag `"exhaustive"` (retired: `"all"`).
"""
struct ExhaustiveSampling <: UnitSampling end

"""
    StratifiedSampling(fraction, cap_per_scene, seed, rng)

A uniform random sample of each scene (the stratum) separately, so the scene
mix is preserved: of a scene's `n` units, `max(1, round(fraction·n))` drawn
without replacement, then at most `cap_per_scene` of those (`0` = no cap),
returned in the scene's own order. Scene `s` draws from
`rng(stable_seed(seed, s))` — its NAME, so adding or reordering other scenes
leaves its sample unchanged. TOML tag `"stratified"`.
"""
struct StratifiedSampling <: UnitSampling
    fraction::Float64
    cap_per_scene::Int
    seed::Int
    rng::String
    function StratifiedSampling(fraction::Real, cap_per_scene::Integer, seed::Integer, rng::AbstractString)
        0 < fraction <= 1 && cap_per_scene >= 0 ||
            throw(ArgumentError("StratifiedSampling: need 0 < fraction ≤ 1 and cap_per_scene ≥ 0, got ($fraction, $cap_per_scene)"))
        _rng_type(rng)
        return new(Float64(fraction), Int(cap_per_scene), Int(seed), String(rng))
    end
end

"""
    UniformSampling(fraction, cap, seed, rng)

A uniform random sample of the units of all the side's scenes POOLED: of the
`N` units, `max(1, round(fraction·N))` drawn without replacement, then at most
`cap` of those (`0` = no cap), returned in the data's order (scenes by name,
each scene's units in its own order). The draw is `rng(stable_seed(seed,
"simple_random"))` over that order — the salt keeps its historical spelling, so
a sample is unchanged by the type's rename. TOML tag `"uniform"` (retired:
`"simple_random"`).
"""
struct UniformSampling <: UnitSampling
    fraction::Float64
    cap::Int
    seed::Int
    rng::String
    function UniformSampling(fraction::Real, cap::Integer, seed::Integer, rng::AbstractString)
        0 < fraction <= 1 && cap >= 0 ||
            throw(ArgumentError("UniformSampling: need 0 < fraction ≤ 1 and cap ≥ 0, got ($fraction, $cap)"))
        _rng_type(rng)
        return new(Float64(fraction), Int(cap), Int(seed), String(rng))
    end
end

# The sample of `n` units under (fraction, cap) from `rng`: permute (`shuffle`,
# in StableRNGs' guaranteed-stable API; `randperm` is not), keep the fraction,
# cap, sort LAST.
function _sample_indices(rng, n::Int, fraction::Float64, cap::Int)
    n == 0 && return Int[]
    order = shuffle(rng, 1:n)
    k = fraction >= 1 ? n : max(1, round(Int, fraction * n))
    cap > 0 && (k = min(k, cap))
    return sort!(order[1:k])
end

_scene_names(units_by_scene) = sort!([String(k) for k in keys(units_by_scene)])

"""
    sample_units(sampling::UnitSampling, units_by_scene) -> Dict{String,Vector}

The units `sampling` keeps of each scene of `units_by_scene` (an
`AbstractDict` scene name ⇒ that scene's units, in the data's order): a
`Dict` with every scene of the input, each holding its kept units in their
original order (possibly none, for a pooled sample).
"""
sample_units(::ExhaustiveSampling, units_by_scene::AbstractDict) =
    Dict{String,Vector}(String(s) => collect(u) for (s, u) in units_by_scene)

function sample_units(s::StratifiedSampling, units_by_scene::AbstractDict)
    R = _rng_type(s.rng)
    return Dict{String,Vector}(String(scene) => collect(u)[_sample_indices(R(stable_seed(s.seed, String(scene))),
                                                                           length(u), s.fraction, s.cap_per_scene)]
                               for (scene, u) in units_by_scene)
end

function sample_units(s::UniformSampling, units_by_scene::AbstractDict)
    scenes = _scene_names(units_by_scene)
    units = Dict(String(k) => collect(v) for (k, v) in units_by_scene)
    flat = [(scene, i) for scene in scenes for i in eachindex(units[scene])]
    kept = _sample_indices(_rng_type(s.rng)(stable_seed(s.seed, "simple_random")), length(flat), s.fraction, s.cap)
    out = Dict{String,Vector}(scene => similar(units[scene], 0) for scene in scenes)
    for k in kept
        scene, i = flat[k]
        push!(out[scene], units[scene][i])
    end
    return out
end

# -----------------------------------------------------------------------------
# The cross-validation
# -----------------------------------------------------------------------------

"""
    CrossValidation(scene_split::SceneSplit, validation::UnitSampling, test::UnitSampling)

The data side of a tuning: the scenes it tunes on and reports on
(`scene_split`), and the units of each side it reads (`validation`, `test`).
A driver states which samplings it supports by the signature of the method it
calls (say `cv::CrossValidation{<:SceneSplit, StratifiedSampling}`), so an
unsupported spec is a `MethodError`. Applied by [`validation_units`](@ref) and
[`test_units`](@ref).
"""
struct CrossValidation{S<:SceneSplit, V<:UnitSampling, T<:UnitSampling}
    scene_split::S
    validation::V
    test::T
end

function _side_units(scenes::Vector{String}, sampling::UnitSampling, units_by_scene::AbstractDict, side::String)
    have = Set(String(k) for k in keys(units_by_scene))
    missing_scenes = [s for s in scenes if !(s in have)]
    isempty(missing_scenes) ||
        throw(ArgumentError("$(side)_units: the data has no units for the $side scenes $(missing_scenes)"))
    kept = sample_units(sampling, Dict{String,Vector}(s => collect(units_by_scene[s]) for s in scenes))
    return [s => kept[s] for s in scenes]
end

"""
    validation_units(cv::CrossValidation, units_by_scene) -> Vector{Pair{String,Vector}}

The units a tuning reads: the validation scenes of `cv.scene_split` (sorted)
⇒ the units `cv.validation` keeps of each. `units_by_scene` maps scene names
to their units in the data's order and must hold every validation scene
(other scenes are ignored).
"""
validation_units(cv::CrossValidation, units_by_scene::AbstractDict) =
    _side_units(cv.scene_split.validation_scenes, cv.validation, units_by_scene, "validation")

"""
    test_units(cv::CrossValidation, units_by_scene) -> Vector{Pair{String,Vector}}

The units an evaluation reads: the test scenes of `cv.scene_split` (sorted) ⇒ the
units `cv.test` keeps of each (see [`validation_units`](@ref)).
"""
test_units(cv::CrossValidation, units_by_scene::AbstractDict) =
    _side_units(cv.scene_split.test_scenes, cv.test, units_by_scene, "test")

"The validation scenes of a cross-validation (sorted)."
validation_scenes(cv::CrossValidation) = cv.scene_split.validation_scenes
"The test scenes of a cross-validation (sorted)."
test_scenes(cv::CrossValidation) = cv.scene_split.test_scenes

# -----------------------------------------------------------------------------
# The validation-budget draws and their resampling
# -----------------------------------------------------------------------------

"""
    BudgetSampling(sizes, n_draws, seed, rng)

The validation-budget draws: for each budget `n ∈ sizes`, `n_draws` simple
random samples of `n` validation units ([`budget_units`](@ref)), draw `r`
from [`budget_rng`](@ref)`(b, n, r)`; the confidence interval over a budget's
draws resamples from [`resampling_rng`](@ref)`(b, n)`.
"""
struct BudgetSampling
    sizes::Vector{Int}
    n_draws::Int
    seed::Int
    rng::String
    function BudgetSampling(sizes::AbstractVector{<:Integer}, n_draws::Integer, seed::Integer, rng::AbstractString)
        !isempty(sizes) && all(>=(1), sizes) && n_draws >= 1 ||
            throw(ArgumentError("BudgetSampling: need nonempty sizes ≥ 1 and n_draws ≥ 1"))
        _rng_type(rng)
        return new(collect(Int, sizes), Int(n_draws), Int(seed), String(rng))
    end
end

"The RNG of draw `r` at budget `n`: `rng(stable_seed(seed, \"budget\", n, r))`."
budget_rng(b::BudgetSampling, n::Integer, r::Integer) =
    _rng_type(b.rng)(stable_seed(b.seed, "budget", Int(n), Int(r)))

"The RNG of the resampled confidence interval over budget `n`'s draws: `rng(stable_seed(seed, \"resampling\", n))`."
resampling_rng(b::BudgetSampling, n::Integer) = _rng_type(b.rng)(stable_seed(b.seed, "resampling", Int(n)))

"""
    budget_units(b::BudgetSampling, n_units) -> Vector{(n, r, units)}

Every draw: `min(n, n_units)` distinct indices of `1:n_units` without
replacement from [`budget_rng`](@ref)`(b, n, r)`, sorted. Callers supply the
ELIGIBLE unit count and map the indices back.
"""
budget_units(b::BudgetSampling, n_units::Integer) =
    [(n, r, _sample_indices(budget_rng(b, n, r), Int(n_units), 1.0, n)) for n in b.sizes for r in 1:b.n_draws]
