# =============================================================================
# Evaluation — the composition evaluator of score+LO arms (engine/: the pair
# geometry, the one refit layer, post-LO matrices, the sample-budget engine,
# live runs, arms and their roster, the selection stream; moved from
# LocalOptimizationDoneRight, K+ 2026-10-09, and written against
# VisualGeometryToolkitCore's contracts alone: any scorer, any local optimizer), and
# the bootstrap tuning harness (evaluate_val.jl: BootstrapMethod /
# FlatProtocol / WindowedGridMethod, bootstrap_trials, compute_method_rows,
# the standardized window's `window_columns`) and THE coarse-fine search
# (cv_probe.jl). Every parameter they run under is a Tuning spec. Its statistics (reference scale, mAA rule, BCa CI) are Stats'. Absorbed from RANSACScoringDoneRight 2026-07-18;
# it uses no scorer package and no optimizer package beyond the core
# (VisualGeometryToolkitCore's contracts, VisualGeometryToolkitEstimation's loop and problems).
#
# Pruned 2026-09-21: the evaluate_val / calibrate_a drivers and everything only
# they reached (the IG / IGMethodKA GPU tuner, GridMethod, the hierarchical /
# flat-resample / split protocols, the oracle and profile blocks), the scene
# IO + coarse-range sidecars (scoring_eval_harness.jl), the DrWatson cache
# helpers (wip.jl), and the Shekhovtsov-format bench IO (bench_utils.jl). None
# had a live caller or a live reader of what it wrote; see evaluate_val.jl's
# header.
# =============================================================================
module Evaluation

import VisualGeometryToolkitCore as VgtkCore
using Printf: @printf, @sprintf
using Random: AbstractRNG
using Statistics: mean, std, median
using LinearAlgebra: normalize, norm
using StaticArrays: SMatrix, SVector
using Rotations: RotMatrix, AngleAxis, rotation_angle
using StableRNGs: StableRNG

# --- The core contracts (VisualGeometryToolkitCore) and the core loop (VisualGeometryToolkitEstimation)
using VisualGeometryToolkitCore: VisualGeometryToolkitCore, AbstractLocalOptimization, NativeRefit,
    ExternalRefit, RefitFamily, EssentialMat, FundMat, HomographyMat, Csponds, Point2, sources, targets,
    converged, estimate, iterations, optimization_path, scale_interface, loss_family, lo_label, lo_marker,
    WidthAxis, ScaleMultiple, LossWidth, OutlierHalfwidth, width_axis_name, with_width, outlier_halfwidth,
    outlier_halfwidth_to_loss_width, loss_width_to_outlier_halfwidth, codimension, scoring_workspace,
    score!, external_workspace, prepare_external!, refine_external!, select_on_grid!, stage_selection,
    selection_staging, lo_workspace, ConsensusSet, ScoreBuffers, MsacScoring, FixedS2
using VisualGeometryToolkitCore.Synthetic: imc_pose_error, optimistic_pose_error
using VisualGeometryToolkitCore.Pools: PoolPair, num_points, num_models, pool_imc, gt_pose, pool_frame_calibrations
using VisualGeometryToolkitEstimation: EssentialProblem, FundMatProblem, HomographyProblem, RelposeBundleProblem,
    locally_optimize, ransac, RansacConfig, RansacWorkspace, NoLocalOptimization, fixed_width,
    locally_optimize_rescored!, AbstractRansacProblem
import VisualGeometryToolkitEstimation: reset!   # the running-best selectors' reset

import .._config_default   # configs/default.toml (config_defaults.jl)

using ..Stats: bootstrap_ci, reference_scale, eligible_maa, maa_threshold, finite_median, pair_outcomes,
    stable_seed
using ..Tuning: CoarseFine, CoarseFineAxis, Incumbent, JointCoarseFine, stats_objective, SpacedGrid, GridSpec,
    ScaleMultipleWindow, BudgetSampling, grid, extend, budget_rng, unit_statistics,
    resampling_rng, _sample_indices,
    # the criteria and their objective: Tuning owns them since 2026-10-02
    # (they came from RSDR's eval_strategy.jl — see tuning/policies.jl).
    TuningCriterion, MedianError, tune_objective,
    # the evaluator's tuned-point readers and deployment corruption
    TuningArtifact, load_tuning_artifact, corrupt_native, nearest_column,
    # the coarse-fine artifact (coarse_fine_artifact.jl)
    TuningSpec, SamplingUnit, save_tuning_artifact
using ..Pools: residual_matrix, sample_ids
using ..Render: lo_us_per_call
# extended here (method added for the harness's own method types)
import ..Tuning: tuning_criterion_label

export
    # cv_probe.jl — THE coarse-fine search (one function per stage), its
    # scalar forms, and the boundary_hit refusal at read time
    coarse_search, fine_search, coarse_fine_search, enforce_probe_interior,
    coarse_fine_cv, coarse_fine_cv_batched, probe_curve, fine_stage,
    # rigid_invariance.jl — the rigid-target invariance criterion
    rigid_invariance_errors,
    # coarse_fine_artifact.jl — THE coarse-fine tuning artifact: writer and gated reader
    save_coarse_fine_artifact, deployed_setting,
    # evaluate_val.jl — the bootstrap tuning harness
    BootstrapMethod, FlatProtocol, WindowedGridMethod, window_columns,
    tune, evaluate,
    bootstrap_trials, compute_method_rows,
    # engine/ — the composition evaluator
    # operating_point.jl
    OPERATING_SCALE_MULTIPLE,
    # geometry.jl
    problem_type, PairCtx, build_pair_ctx, model_imc, refit_seed, essential_bearings,
    perturbed_pose, perturbed_seed_E,
    # refit.jl
    refit_imc!, refit_model, guarded_refit, NativeAxis, consumed_axis, consumed_native,
    ScoreGuarded, RefitHalfwidth,
    # lo_matrix.jl
    LoRefitSpec, apply_lo_refits,
    # budget.jl
    SampleBudgetConfig, SampleBudgetMethodSpec, MsacSpec, evaluate_sample_budget, summarize_lo_timing,
    # live.jl
    live_runs, live_row,
    # arms.jl
    ComposedSpec, SoloSpec, Roster, WidthSource, TunedWidth, SelectionWidth, is_tuned, operating_width,
    width_delivery, solo_refit_imc!, miscal_evaluation_refit, deployed_budget_settings,
    deployed_arm_settings, deployed_basin_settings, audit_operating_points,
    # selection_stream.jl
    stream_selections

# Public, not exported: the evaluation's LO budget, the pair geometry, the
# budget engine's selectors, per-pair worker, schedule and result record.
public lo_iteration_budget, PairGeom,
    SampleSelector, ScoreSelector, OracleSelector, update!, checkpoint, lo_error!,
    SampleBudgetResults, evaluate_sample_budget_pair!, n_batches, batch_sample_counts,
    lo_sample_counts, n_los

include("evaluation/cv_probe.jl")
include("evaluation/rigid_invariance.jl")   # rigid_invariance_errors: the rigid-target invariance criterion
include("evaluation/coarse_fine_artifact.jl")   # save_coarse_fine_artifact, deployed_setting
include("evaluation/evaluate_val.jl")
include("evaluation/engine/operating_point.jl")  # OPERATING_SCALE_MULTIPLE, lo_iteration_budget
include("evaluation/engine/geometry.jl")         # PairGeom, PairCtx, build_pair_ctx, model_imc, problem_type
include("evaluation/engine/refit.jl")            # THE refit layer (refit_imc!), ScoreGuarded, RefitHalfwidth
include("evaluation/engine/lo_matrix.jl")        # LoRefitSpec, apply_lo_refits
include("evaluation/engine/selectors.jl")        # running-best selectors
include("evaluation/engine/budget.jl")           # the sample-budget engine
include("evaluation/engine/live.jl")             # live end-to-end RANSAC on pool pairs
include("evaluation/engine/arms.jl")             # ComposedSpec, SoloSpec, Roster, the deployed readers
include("evaluation/engine/selection_stream.jl") # stream_selections

end # module Evaluation
