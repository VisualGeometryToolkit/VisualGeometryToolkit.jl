# =============================================================================
# Tuning — THE cross-validation specification: grid spacings and THE grid
# constructor (`grid`), window rules, the persisted grid specs and their
# extension (`extend`), the strategy types of a cross-validation (axes,
# the boundary rule, objectives), THE sampling of
# a cross-validation (the registered scene split of configs/splits.toml, the
# unit samplings, the validation-budget draws), the schemes that bundle them,
# THE `[tuning]` TOML parser/writer over this package's configs/tuning.toml of
# record, and THE tuning-artifact writer/reader (format-stamped, spec + hash +
# grid specs + sampling unit + the caller's code identity), and the
# miscalibration protocol's deployment corruption (`corrupt_native`).
#
# Every strategy is a type whose fields are its complete parameter set; every
# default lives in configs/tuning.toml; every type round-trips TOML → type →
# TOML and type → persisted table → type with `==`.
# =============================================================================
module Tuning

using Random: shuffle
using StableRNGs: StableRNG
using VisualGeometryToolkitCore: VisualGeometryToolkitCore as VGC   # the TOML key helpers (toml.jl, sampling.jl)
using SHA: sha256
using TOML: TOML
using JLD2: JLD2
using LibGit2: LibGit2
using DrWatson: isdirty                     # _git_identity (persist.jl)
using Statistics: median                      # MedianError's objective; RobustFit's prior scale
using Distributions: Gamma, fit_mle, scale, shape   # fit_prior(::InverseGamma, ::MLE)
using VisualGeometryToolkitCore: IGPrior
using VisualGeometryToolkitCore.Synthetic: maa_score   # unit_statistics of the mAA criterion
using ..Stats: finite_median, lower_median, scale_bounds, scene_mean, scene_mean_median, stable_seed,
    eligible_maa                              # THE mAA rule (MeanAverageAccuracy's objective)
import .._save_artifact, .._check_artifact_format, .._format_key   # artifact_format.jl

export
    # spacing.jl — spacings and THE grid constructor
    GridSpacing, LogPerDecade, LogPerOctave, LogCount, LinearCount, grid, npoints,
    # grids.jl — persisted grid specs and THE extension
    GridSpec, SpacedGrid, ListedGrid, extend, extension_offset, nearest_column,
    # windows.jl — the cutoff band and window rules
    ScaleMultipleWindow, WindowRule, FixedWindow, StandardizedWindow, CoverageWindow,
    CenteredWindow, window,
    # policies.jl — axes, the boundary rule, objectives
    TuningAxis, SpacedAxis, ListedAxis, grid_spec, GridExtension,
    TuningCriterion, MedianError, FiniteMedian, SceneMeanLowerMedian, SceneMeanMedian,
    MeanAverageAccuracy,
    # sampling.jl — the cross-validation's scenes and units, the budget draws
    SamplingUnit, PairUnit, QueryUnit, sampling_unit,
    SceneSplit, RegisteredSceneSplit, registered_split,
    UnitSampling, ExhaustiveSampling, StratifiedSampling, UniformSampling, sample_units,
    CrossValidation, validation_units, test_units, validation_scenes, test_scenes,
    BudgetSampling, budget_rng, budget_units, resampling_rng,
    # schemes.jl
    TuningScheme, CoarseFine, CoarseFineAxis, Incumbent, JointCoarseFine, WindowedBootstrap, Factorial, TuningSpec,
    # toml.jl — THE [tuning] parser/writer
    TUNING_TAGS, tuning_spec, tuning_specs, tuning_table, tuning_toml, tuning_hash, read_tuning,
    resolve_tuning!, layer_tuning,
    # persist.jl — THE tuning-artifact writer/reader
    TuningArtifact, save_tuning_artifact, load_tuning_artifact, code_identity,
    # bootstrap_argmin.jl — the bootstrap of a tuned argmin over the units
    unit_statistics, bootstrap_argmin,
    # corruption.jl — the miscalibration protocol's deployment corruption
    corrupt_native,
    # prior_specs.jl — the scale-prior strategy of a tuning
    PriorSpec, fit_prior

public TUNING_DEFAULTS, TUNING_DEFAULTS_PATH, RNG_ALGORITHMS, SPLIT_REGISTRY, SPLIT_REGISTRY_PATH,
    _tune_objective, _stats_objective, tuning_criterion_label
# The prior-strategy tags are public, not exported: `InverseGamma` is also
# Distributions' exported distribution.
public Jeffreys, InverseGamma, MLE, RobustFit, Unfitted, JeffreysPriorSpec, IGRobustSpec, ALL_PRIOR_SPECS,
    prior_label, family_tag, fitting_tag, ScalePriorFamily, PriorFitting

include("tuning/spacing.jl")
include("tuning/grids.jl")
include("tuning/windows.jl")
include("tuning/policies.jl")
include("tuning/sampling.jl")
include("tuning/schemes.jl")
include("tuning/toml.jl")
include("tuning/persist.jl")
include("tuning/bootstrap_argmin.jl")
include("tuning/corruption.jl")
include("tuning/prior_specs.jl")

end # module Tuning
