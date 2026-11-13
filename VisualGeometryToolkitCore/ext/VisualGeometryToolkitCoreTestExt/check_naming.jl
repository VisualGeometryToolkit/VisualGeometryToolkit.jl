# =============================================================================
# check_naming — the stack's iterative-estimator vocabulary, checked mechanically
# =============================================================================
#
# conventions/naming.md ("Iterative-estimator vocabulary") states the rules; this file is
# their one implementation and holds the one list of banned words and retired
# names. Every stack package's quality test runs `check_naming` on itself.
#
# The checker reads its own file like any other, so the lists below are BUILT
# from fragments (`_STEM * "es"`, `"iterate" * "_lo"`): no literal in this
# file spells a banned word or a retired name. That is structural, not an
# exemption — the text scan has no skip list.
#
# Two scans, by what a file holds:
#   - Julia CODE is checked on the names it BINDS (the AST from Meta.parseall),
#     word by word, and a word that merely CONTAINS a banned word or `step`
#     (`stepsize`, `bypass`) is flagged: a use of a name owned elsewhere (Base's
#     `step(1:3)`) is not a binding, so the owner's spelling stands;
#   - the names Python, shell and TOML files bind (def/class names, parameters,
#     assignment and loop targets, variables, keys) get the same identifier
#     rules, read by line-level approximations (limits in `check_naming`);
#   - Julia comments and strings, and every line of the other in-scope files,
#     are prose, scanned word by word.
# Retired names are matched on every line of every file, code included.
# Third-party trees are declared in a root VENDORED.toml (`naming_scope`).
# =============================================================================

# -----------------------------------------------------------------------------
# The lists (built from fragments; see the header)
# -----------------------------------------------------------------------------

const _STEM = "pa" * "ss"

# In prose: whole words, case-insensitive, after splitting at `_`, `-`,
# punctuation and CamelCase, so "bypass", "compass" and "password" are other
# words. A bound name is matched on substrings instead (`_name_rules`).
const _BANNED_WORDS = Set([_STEM, _STEM * "es", _STEM * "ed", _STEM * "ing", _STEM * "through"])

# The terminology taxonomy (owner, 2026-11-13; conventions/naming.md, "Detection, correspondence,
# observation"): a detector's output is a DETECTION, a hypothesised pairing a CORRESPONDENCE (a match, an
# inlier), a detection associated with a model entity an OBSERVATION. These words are retired from every
# identifier (R14), each a substring of a bound name's word: the old pairing abbreviation, the
# synonym of an observation, the detection's two other names. "measured" is a provenance VALUE
# (`:measured`, `"measured"`: a quoted symbol or a string) and never a bound name.
const _RETIRED_TERMS = ("measure" * "ment", "cs" * "pond", "sight" * "ing", "key" * "point")
const _PROVENANCE_ONLY = "measure" * "d"

# Rule 7's trial/iteration split: a random hypothesis draw (a RANSAC or PROSAC
# draw, a random LO inner sample) is a TRIAL and is never counted or described
# as an iteration; a deterministic cycle (IRLS, LM, block coordinate descent,
# a GNC rung, Newton) is an ITERATION and is never called a trial. The one
# exception is VisualGeometryToolkitCore's `FitResult.iterations`, which counts a
# RANSAC's trials. Phrases are matched in prose like the banned words: whole
# words, case-insensitive, as a contiguous run after the same split, so they
# are listed as word tuples. Only phrases that are wrong wherever they occur
# are listed: a budget or cap of a deterministic loop is an iteration budget,
# and an ML batch sampler iterates, so neither word alone is a signal.
const _IT = "iter" * "ation"
const _RS = "ran" * "sac"
const _PS = "pro" * "sac"
const _TR = "tri" * "al"
const RETIRED_PHRASES = [
    # a random draw counted as an iteration
    (_RS, _IT), (_RS, _IT * "s"), (_RS, "iter"), (_RS, "iters"),
    (_PS, _IT), (_PS, _IT * "s"),
    ("hypotheses", "per", _IT), ("hypothesis", "per", _IT),
    # a deterministic cycle called a trial
    ("lm", _TR), ("lm", _TR * "s"),
    ("damping", _TR), ("damping", _TR * "s"),
    (_TR, "step"), (_TR, "steps"),
    (_TR, "per", _IT),
]
# In a bound name, the words that mark a random draw and those that mark an
# iteration count (`_counts_draws_as_iterations`).
const _RANDOM_DRAW_WORDS = (_RS, _PS)
const _ITERATION_WORDS = ("iter", "iters", _IT, _IT * "s")

# Every name the stack has retired. Matched exactly, `!` included, so a name
# retired in one form lists each form it took. Names renamed by the naming
# sweep are listed here as they are renamed.
const RETIRED_NAMES = String[
    # the terminology taxonomy (owner, 2026-11-13; R14): a CORRESPONDENCE is a hypothesised pairing, the old
    # abbreviation retired with every name built on it (VisualGeometryToolkitCore 0.10)
    "Cs" * "pond", "Cs" * "ponds", "cs" * "ponds", "cs" * "ponds(", "ScoredCs" * "pond", "Cs" * "pondNormalization",   # `Correspondence`, `Correspondences`, `correspondences(sources, targets)`, `ScoredCorrespondence`, `CorrespondenceNormalization`
    "bearing_cs" * "ponds", "pixel_cs" * "ponds", "pair_cs" * "ponds",   # `bearing_correspondences`, `pixel_correspondences`, `pair_correspondences`
    "eth3d_splg_cs" * "ponds",                                    # a published artifact's name (RANSACScoringDoneRight's Zenodo record): data, a foreign token below
    "Cs" * "pondSet",                                             # VisualGeometryToolkitFeatures' n-tuple of corresponding elements: `CorrespondenceSet`
    # an OBSERVATION is a detection associated with a model entity (a target point, a landmark); "measured" is provenance only
    "measurement" * "_gradient_jacobian",                         # a model's constraint Jacobian in the observation: `observation_gradient_jacobian`
    "measurement" * "_covariance",                                # `observation_covariance`
    "Target" * "Detections",                                      # per frame, per target, the observations of its points: `TargetObservations` (VisualGeometryToolkitCore 0.10)
    "Target" * "Points2D", "Target" * "Points3D", "Target" * "Point2D", "Target" * "Point3D",   # folded into `TargetObservations{F}` (a PixelFrame or a named 3D frame; weights a `weights` column)
    # the three public aliases (wave 10): one name per concept
    "Liftable" * "Problem",                                       # VisualGeometryToolkitEstimation: its one name, `MinimalSolveProblem`
    "ScalarOr" * "Quantity",                                      # VisualGeometryToolkitIO: its one name, `Number`
    "POOL" * "_RNG",                                              # VisualGeometryToolkitEval: its one name, StableRNGs' `StableRNG`
    # one χ² quantile, Distributions' (owner, 2026-11-12): the median of χ²_d is `quantile(Chisq(d), 0.5)`
    "chisq" * "_median",
    # one metadata record of the reweighted fits (the owner's ruling on the duplication audit's item 11, 2026-11-12):
    # fit_linear, fit_rotation and fit_rigid_alignment return `FitMetadata{L, W}` (VisualGeometryToolkitEstimation)
    "LinearFit" * "Metadata", "RotationFit" * "Metadata", "RigidAlignment" * "Metadata",
    # one optimization-strategy family (the owner's ruling, 2026-11-11): `find_*`/`ransac`/`locally_optimize` take
    # `local_optimization_strategy`, `refine_*`/`fit_*` `optimization_strategy`, read by `optimization_strategy_from_toml`
    "AbstractLocal" * "Optimization", "local_optimization" * "_from_toml",
    "NoLocal" * "Optimization",   # no local optimization is `nothing` (TOML "none")
    "_check_local" * "_optimization",   # refine_camera_and_poses' strategy check is the refine contract's (`bundle_loss`)
    "default_local" * "_optimization",   # the ransac keyword's default: `default_local_optimization_strategy(scoring)` (VisualGeometryToolkitEstimation)
    "_fit_local" * "_optimization",      # a fit's `optimization_strategy` default: `_fit_strategy_default(fit)` (VisualGeometryToolkitEstimation)
    # the verb states the kind (conventions/naming.md): a seeded bundle is a refine, the filtered averaging a fit
    "bundle_refine" * "_relpose",        # `refine_relative_pose(pose, K1, K2, cs)` (VisualGeometryToolkitEstimation)
    "filtered_rotation" * "_averaging",  # `fit_filtered_rotation_averaging(vg; cycle_tol, orientation_tol)` (VisualGeometryToolkitEstimation)
    "LO" * "_TAGS",                      # the strategy registry: `OPTIMIZATION_STRATEGY_TAGS` (VisualGeometryToolkitEstimation; the owner's ruling)
    # the robust-estimation contract (2026-11-11): a width multiple is the efficiency constant, always the PLAIN
    # multiple of the noise scale (the half-weight and Barron currencies are retired, their values migrated)
    "scale" * "_multiple", "Scale" * "Multiple", "plain_scale" * "_multiple", "deployed_scale" * "_multiple", "deployed" * "_shape",
    "noise" * "_sigma(", "VgtkCore" * ".noise_sigma",   # residual scales are `mad_scale(residuals)` (the call forms only: Features' Förstner keyword is another quantity)
    # the owner's one-kernel policy (2026-11-11): pointwise! is one KernelAbstractions kernel on every backend (`OnBackend`)
    "Execution" * "Policy", "THREADED" * "_MIN_LENGTH", "VgtkCore" * ".Serial", "VgtkCore" * ".Threaded", "VgtkCore" * ".Auto",
    "exec = " * "Serial()", "exec = " * "Threaded()", "exec = " * "Auto()",
    "_foreach" * "_row!",                     # every LM row fill launches `_row_kernel!` (`_launch_rows!`), the CPU's included
    # the 0.9 /simplify fixes: the spline refine's gauge by parametrization, one rows-and-Jacobian assembler
    "finite" * "_linearization",             # NLLSsolver ≥ 5.3.3 stops on a non-finite linearization itself (#68); no block wraps it
    "_Spline" * "Anchor", "_Spline" * "Pin",   # VisualGeometryToolkitEstimation: only live controls are variables, the gauges no coordinate (`restrict`)
    "_camera_poses_rows" * "_and_jacobian", "_rw_rows" * "_and_jacobian", "_spline_problem_rows" * "_and_jacobian",   # Core's `nlls_rows_and_jacobian`
    "_rect" * "_centre",                     # a rectangle's centre: `centroid(r)`
    # a value with its cost is Core's `Attributed(θ, (; cost))` (`keep_best`) (2026-11). Only the
    # forms Core's type took are listed: Eval's tuning `Incumbent` (a search's setting in force and
    # its bootstrap test) is a different concept and keeps the bare word
    "Incumbent" * "{M}", "VisualGeometryToolkitCore" * ".Incumbent", "VgtkCore" * ".Incumbent",
    "_edge" * "_weight",                      # an edge's weight is Core's `weight_of` (2026-11)
    "_rel" * "ative",                         # an edge's transform is Core's `value_of` (2026-11)
    "_bou" * "nded",                          # a stored camera's BoundedCamera is Core's `value_of` (2026-11)
    "inlier" * "_threshold",                  # the scorer's decision point is `decision_boundary(scoring, buf)` (2026-11)
    "has_inlier" * "_threshold",              # its trait is `has_decision_boundary(S)` (2026-11)
    "decision" * "_threshold",                # the residual a decision flips at is `decision_boundary` (a loss shape, the Gaussian–uniform pair, a scorer) (2026-11)
    "_grid" * "_golden_minimum",            # the one 1-D minimizer is public: `grid_brent_minimum` (2026-11)
    "grid" * "_golden_minimum",             # the one 1-D minimizer refines by Brent's method: `grid_brent_minimum` (2026-11)
    "_lens" * "_axis",                      # the forward spline's chart is about ẑ, not the principal point's ray: `forward_chart(SplineLens)` (2026-11)
    "fit" * "_height_field",                  # a one-step algorithm is no `fit_*` (the estimators guide): `HeightField(nodes, points .=> heights; smoothing)` (2026-11)
    "strong" * "est(", "VgtkFeatures" * ".strongest", "VisualGeometryToolkitFeatures" * ".strongest",   # the global top n by |response|: `spread_strongest(detections, image_size, n)` (SSC, VisualGeometryToolkitFeatures, 2026-11)
    "_imas" * "_tilts_25",                    # ASIFT's schedules by name: `tilt_schedule(:imas25)`, `tilt_schedule(:untilted)` (VisualGeometryToolkitFeatures, 2026-11)
    "Raw" * "pyDecoder",                      # the stack's one RAW decoder: `LibRawDecoder()` (VisualGeometryToolkitFeatures, 2026-11)
    "VisualGeometryToolkitFeatures" * "PythonExt",   # its extension, removed with it: the LibRaw_jll extension decodes (2026-11)
    "tps" * "_bending_coefficients",          # a wrapper of two calls: `tps_pure_bending`, then `tps_coefficients` (or `with_parameters` on the kind) (2026-11)
    "Chart" * "Placement",                    # a target's calibration chart in its frame is Core's `SimilarityMap{3}` (scale the side, the identity rotation, translation (origin, 0)): `PlanarTarget(…; placement = S₃)` (2026-11)
    "Height" * "Surface",                     # a deformed target's surface is the chart map's: `MappedSurface` (pose ∘ S₃ ∘ lift ∘ C ∘ S₂⁻¹), one root for every deformation (2026-11)
    "Implicit" * "Root",                      # a MappedSurface's root is always differentiated by the implicit function theorem: no choice to make (2026-11)
    "implicit" * "_root",                     # the per-device choice of the retired root derivative: none (a MappedSurface's root is implicit) (2026-11)
    "_height" * "_newton",                    # the height-only 1-D Newton: a MappedSurface's 1-D fast path `_height_fast` (target_surfaces.jl), the generic root's specialization (2026-11)
    "_placed" * "_surface",                   # the placement formula of a height surface: the composition `S₃ ∘ lift ∘ C` (`_target_map`) (2026-11)
    "_without" * "_plane",                    # the pure-bending projection: `tps_pure_bending` (2026-11)
    "_placed" * "_point",                     # a placed target's point: the composed map `_target_map` (2026-11)
    "image" * "_origin",                    # the pixel-frame keyword of `CameraModel(K; …)`: `pixel_frame` (2026-11)
    "_radially" * "_symmetric",             # a lens's radial symmetry is a Holy trait: `RadialSymmetry` (`RadiallySymmetric()` / `RadiallyAsymmetric()`) (2026-11)
    "_check" * "_refinable",                # refine_camera_and_poses routes by Core's `IntrinsicsRole` (a SplineLens camera refines; no refusal): `_refine_camera_and_poses(::GaugeIntrinsics, …)` (2026-11)
    "bundle" * "_adjust",                   # PoseLib's alias of its absolute-pose refinement, a second name: `refine_absolute_pose` (2026-11)
    "Tensor" * "Vignetting",                  # a tensor-PRODUCT B-spline surface, not a tensor: `TensorProductVignetting` (2026-11)
    "ShapeMatrix" * "Scale",                  # the trait dispatches on `shape_factor`: `ShapeFactorScale` (2026-11)
    "moment" * "_matrix",                   # M is a region's shape, not its second moment: `shape_matrix` (0.7; the factor is `shape_factor`)
    "field" * "_center",                    # a vignetting field's centre is Core's one `center` generic: `center(v)` (0.7)
    "transform" * "_blur",                  # a blur is the density N(0, Σ): through a linear map, its covariance's push-forward, `propagate(J, b)` (2026-11)
    # maps are the geometric objects, matrices only algebra (Core 0.6): the
    # transform class is the map's type (`EuclideanMap`, `SimilarityMap`,
    # `AffineMap`, `HomographyMap`); `to_homogeneous(map)` is a plain SMatrix
    "Hom" * "TranslationMat", "Hom" * "RotationMat", "Hom" * "EuclideanMat", "Hom" * "SimilarityMat",
    "Hom" * "ScalingMat", "Hom" * "UniformScalingMat", "Hom" * "AffineMat",
    "Abstract" * "HomMat", "Abstract" * "AffineMat", "Abstract" * "SimilarityMat",
    "Abstract" * "EuclideanMat", "Abstract" * "CalibrationMat",
    "_preserves" * "_circles", "_transform" * "_circle", "_transform" * "_ellipse",
    # the loop's former vocabulary
    "iterate" * "_lo", "iterate" * "_lo!",
    "solve" * "_step", "solve" * "_step!",
    "post" * "_step", "post" * "_step!",
    "init" * "_state",                       # the hook mutates its method's workspace: `init_state!`
    "AbstractIRLS" * "Method",
    "irls" * "_result",
    "irls" * "_framework",
    "_update" * "_best", "_update" * "_best!",
    "_newton" * "_step", "_newton" * "_step!",
    "_division_camera_" * _STEM,
    "_nu" * "_update", "_nu" * "_update!",
    "trust_region" * "_step", "trust_region" * "_step!",
    # the naming sweep (VisualGeometryToolkitCore)
    "_lift1_hq" * "_step!", "_lift2_hq" * "_step!",
    "_lift1_gnc" * "_step!", "_lift2_gnc" * "_step!",
    "_lift1_z" * "_update!", "_lift2_z" * "_update!",
    "_pose" * "_step",
    "_lo" * "_step",
    "ellipse" * "_step",
    # the naming sweep (RANSACScoringDoneRight)
    "update" * "_nu",                        # the pure ν update is `next_nu`; `update_nu!` writes the ν block
    "log" * "_step", "inv_log" * "_step",    # the τ-grid's log bin spacing: `log_spacing`, `inv_log_spacing`
    "l" * "step",                            # a test's copy of that spacing: `lspacing`
    "_m" * "step_map",                       # a test's MAP scale estimate: `_map_scale_estimate`
    # the naming sweep (VisualGeometryToolkitEstimation)
    "_Division" * uppercasefirst(_STEM),     # the positional division-camera state: `_PositionalDivisionState`
    "_" * _STEM * "_weights",                # a view's prior × posterior row weight: `_combined_weights`
    "_print_" * _STEM,                       # the verbose trace line: `_print_iteration`
    "_" * _STEM * "_dof",                    # the stacked solve's fitted dof (beta-mixing): `_fitted_dof`
    # the naming sweep (the evaluation engine: VisualGeometryToolkitEval, LocalOptimizationDoneRight)
    "Lo" * uppercasefirst(_STEM),            # one labelled LO refit of a selection sweep: `LoRefitSpec`
    "apply_lo_" * _STEM * "es",              # `apply_lo_refits`
    "_apply_" * _STEM * "_row!",             # `_apply_refit_row!`
    "two_knob_val_" * _STEM * "es",          # `product_tuning_val_refits`
    "two_knob_ext_" * _STEM * "es",          # `product_tuning_val_refits` (its `columns`)
    "two_knob_absw_ext_" * _STEM * "es",     # `product_tuning_val_refits` (its `columns`)
    "miscal_readout_" * _STEM,               # `miscal_readout_refit`
    "nn_point_" * _STEM * "es",              # `nn_point_refits`
    _STEM * "es_of",                         # stream_product_tuning's refit-list builder: `refits_of`
    "ytick" * "_step", "y" * "_step", "y" * "step",   # plot tick spacing: `ytick_spacing`, `y_spacing`
    "EXT" * "_STEP",                         # cv_probe's boundary extension: the spec's `GridExtension`
    # the naming sweep (BlobBoards)
    "_run_board_" * _STEM * "es!",           # one matching phase of find_boards: `_run_matching_phase!`
    "_find_boards_" * _STEM,                 # the greedy claim-commit round loop: `_claim_boards`
    _STEM * "ing_uids",                      # boards with a quality-gated detection: `detected_uids`
    "global" * "_step",                      # the descriptor trainer's counter: `global_iteration`
    "steps_per" * "_epoch", "total" * "_steps",   # `iterations_per_epoch`, `total_iterations`
    "last" * "_step",                        # the trace player's last tick time: `last_tick`
    "step" * "_vec",                         # the adapt scan grid's sample spacing: `spacing_vec`
    # frame changes are composition (VisualGeometryToolkitCore, 2026-09-30): a model map
    # moves by `D ∘ M ∘ inv(S)` with CoordinateTransformations' `∘`/`inv`
    "push" * "_model", "pull" * "_model",    # a normalization's sides composed onto the model map
    "re" * "frame", "re" * "frame_jacobian", # `∘` on an EpipolarMap / a TransferForm's HomographyMap; `CameraModel(model, K′)`
    # the pool frame is named as such (RANSACScoringDoneRight, 2026-10-01): the
    # stored maps, normalization, calibrations and outlier support of a pool
    "canon" * "_maps",                       # `pool_frame_maps`
    "canonical" * "_normalization",          # `pool_normalization`
    "canonical" * "_calibrations",           # `pool_frame_calibrations`
    "CANONICAL" * "_R_MAX",                  # `POOL_FRAME_R_MAX`
    "canon" * "_maps" * "_unknown",          # the retired HEB pool reader: `pool_frame_maps_unknown`
    "CANON" * "_UNKNOWN",                    # the retired HEB pool reader: `UNKNOWN_POOL_FRAME_MAP`
    "canonical" * "_essential_problem",      # VisualGeometryToolkitEstimation: `EssentialProblem(pair)`
    "canon" * "_src", "canon" * "_dst",      # StereoScoringData's maps: `source_map`, `destination_map`
    "_heb_canon" * "_maps",                  # the retired HEB pool reader: `_heb_pool_frame_maps`
    "canon" * "_sidecar",                    # the retired `migrate_heb_pool`: `pool_frame_sidecar`
    "canon" * ".jld2",                       # the April HEB sidecar suffix: `<scene>.pool_frame.jld2`
    # the lens families (VisualGeometryToolkitCore, 2026-10-08): a lens is a `Lens` (a
    # projection and a distortion in its frame) or a `LiftLens`; models are
    # structures of leaves, conventions are VisualGeometryToolkitIO's adapters
    "Brown" * "Conrady",                     # OpenCV's models: `Lens(Perspective(), Additive(Radial(…), Tangential(…), …))`
    "Rational" * "Radial",                   # the rational lift: `LiftLens(Radial(k, d))`
    "Equidistant" * "Fisheye", "Stereographic" * "Fisheye",   # `Lens(Equidistant())`, `Lens(Stereographic())`
    "Equisolid" * "Fisheye", "Orthographic" * "Fisheye",      # `Lens(Equisolid())`, `Lens(Orthographic())`
    "ThinPrism" * "Fisheye",                 # `Lens(Equidistant(), Additive(Radial(…), Tangential(…), ThinPrism(…)))`
    "Additive" * "Bundle", "Brown" * "Radial4",   # `Additive`, a leading `Radial`
    "AbstractAdditive" * "Distortion",       # `isdelta`
    "Lift" * "Family",                       # `LiftLens`
    "lens" * "_tag", "lens_from" * "_toml",  # the lens record: `lens_record`, `lens_from_record`
    "FramedDoubleSphere" * "Projection",     # `FramedLensProjection`
    # the fold is the lens model's VALID DOMAIN, not its image circle (VisualGeometryToolkitCore,
    # 2026-10-29): `valid_domain(camera)`; `image_circle` keeps only the datasheet form
    "_lens" * "_circle_radius",              # `_valid_domain_radius`
    # one per-point function per batch (VisualGeometryToolkitCore, 2026-10-08): `pointwise!`
    "Batched" * "Map", "ComposedBatched" * "Map", "FramedBatched" * "Stage",
    "Batched" * "Distort", "Batched" * "Undistort", "undistort" * "_points!",
    "apply" * "_map", "apply_with" * "_jacobian", "apply_with" * "_valid", "valid" * "_mask",
    "batched" * "_contract", "batched" * "_stack", "batched_distort" * "_stage", "batched_undistort" * "_stage",
    # the evaluation stack's retired public names (2026-10-01)
    "knob" * "_grid", "cutoff_k" * "_grid",  # the product grid's efficiency-constant axis: `grid(spec)`
    "eval" * "_cell", "cell" * "_results_root",
    "DEFAULT" * "_RESULTS_ROOT", "set_default" * "_results_root!",
    "num_lo" * "_steps",
    "Flushing" * "Logger",                   # the flush Timer: `install_flushing_logger!`
    "BUDGET" * "_LEGEND", "register_budget" * "_legend!",
    "native" * "_refit", "refit" * "_native",
    "merge_ladder" * "_shards",
    "gm_k" * "_from_width",                  # LocalOptimizationDoneRight accv2026: `welsch_half_weight_depth`
    # the tuning specification (VisualGeometryToolkitEval Tuning, 2026-10-01): one
    # grid constructor `grid(lo, hi, spacing)` over spacing TYPES, one extension
    # `extend(spec, sides; npts)`, persisted GridSpecs, every parameter in a
    # `[tuning.<name>]` spec; the two-knob tuning is the product tuning
    "log" * "_grid", "log" * "_grid_per_decade",       # `grid(lo, hi, LogCount(n))` / `LogPerDecade(p)`
    "log_k" * "_grid", "linear_k" * "_grid",           # `grid(spec)` of the efficiency-constant axis
    "extend_k" * "_grid",                              # `extend(spec, sides; npts)`
    "cutoff_a" * "_grid", "absw" * "_grid",            # `grid_spec(axis; context…)` of the spec's axes
    "cutoff_native" * "_columns",                      # the `CoverageWindow` rule + `grid`
    "grid_extension" * "_offset",                      # `extension_offset(base, sup)` on GridSpecs
    "nn_draw" * "_schedule",                           # `budget_units(::BudgetSampling, n)`
    "Common" * "CutoffSpec",                           # `EfficiencyConstantWindow`
    "PER" * "_DECADE", "EXT_NEW" * "_PTS", "EXT" * "_SPACING", "MAX" * "_EXTENSIONS",
    "PLATEAU_REL" * "_TOL", "K" * "_OP",               # spec fields; `OPERATING_EFFICIENCY_CONSTANT`
    "Knob" * "Kind", "Cutoff" * "Knob", "ARef" * "Knob", "CRef" * "Knob", "AbsWidth" * "Knob",
    "KNOB" * "_KINDS",                                 # `WidthParameterization`, `EfficiencyConstant`,
    "knob" * "_kind", "knob" * "_axis",                # `OutlierHalfwidth`, `LossWidth`,
    "knob" * "_window", "knob" * "_delivery",          # `WIDTH_PARAMETERIZATIONS`, `width_parameterization`,
                                                       # `tuning_axis`, `tuning_window`, `width_delivery`
    "sel" * "_axis", "sel" * "_grid", "sel" * "_window",   # `selection_axis`, `selection_window`
    "stream_two" * "_knob", "stream_two" * "_knob_cpu",    # `stream_product_tuning`, `stream_product_tuning_cpu`
    "tune_two_knob" * "_cpu",                          # `tune_product_cpu`
    "merge_two_knob" * "_sidecars", "audit_two" * "_knob", # `merge_product_tuning_sidecars`, `audit_product_tuning`
    "run_two" * "_knob",                               # the driver `run_product_tuning.jl`
    "two_knob_cv" * "_stage", "two_knob_merge" * "_stage",  # `product_tuning_cv_stage`, `…_merge_stage`
    "merge_two_knob" * "_arms",                        # the merge CLI
    "deployed_solo" * "_settings",                     # (historical reader; no successor)
    "_save" * "_pool", "POOL" * "_FORMAT",             # `_save_artifact(path, :pool; …)`, `ARTIFACT_FORMATS`
    "Tune" * "Config", "tune" * "_table",              # a BlobBoards project: `load_tune_config`, `tune_tables`
    "cv_polish" * "_k",                                # accv2026: `cv_polish_efficiency_constant`
    "run_sb_fair_test" * "_twoknob", "SB" * "_TWOKNOB", # accv2026: `run_sb_fair_deployed`, `SB_PRODUCT_TUNING`
    "PS" * "_KCV",                                     # accv2026: `PS_CV`
    # the codified cross-validation (VisualGeometryToolkitEval Tuning, 2026-10-01):
    # a spec's `cross_validation` = a registered scene split (configs/splits.toml)
    # + a unit sampling per side; `budget_sampling` the validation-budget draws;
    # every seed derived by `stable_seed` from the names it applies to. No scene
    # list, split, subsample fraction or seed lives in code.
    "Validation" * "Split", "validation" * "_split", "validation" * "_rows",   # `CrossValidation`, `validation_units`
    "per" * "_scene",                                  # `StratifiedSampling` (pooled: `UniformSampling`)
    "np" * "_val", "np" * "_test",                     # a sampling's `cap_per_scene` / `cap`
    "val" * "_fraction", "VAL" * "_FRACTION",          # a sampling's `fraction`
    "SUB" * "_SEED",                                   # a sampling's `seed`
    "Draw" * "Schedule", "draw" * "_schedule",         # `BudgetSampling`
    "draw" * "_rows", "draw" * "_rng",                 # `budget_units`, `budget_rng` (`resampling_rng`)
    "seed" * "_stride", "size" * "_stride",            # `stable_seed(seed, "budget", n, r)`
    "SCENE" * "_SPLITS", "print_scene" * "_env",       # `registered_split(name)` over configs/splits.toml
    "PT_VAL" * "_SCENES", "PT_TEST" * "_SCENES", "TDV_PT_VAL" * "_SCENES",
    "SCANNET_VAL" * "_SCENES", "SCANNET_TEST" * "_SCENES",
    "ETH3D_VAL" * "_SCENES", "ETH3D_TEST" * "_SCENES",
    "HEB_VAL" * "_SCENES", "HEB_TEST" * "_SCENES",
    "SEVEN_SCENES_VAL" * "_SCENES", "SEVEN_SCENES_TEST" * "_SEQUENCES",   # `registered_split(name)` lists
    "n_pairs" * "_per_scene", "OPC" * "_TUNE", "NRE" * "PORT",   # a pair-list prefix: `validation_units` / `test_units`
    "split" * "_phase",                                # BlobBoards: the within-session split (not scene-disjoint)
    # the paper constants (the evaluation engine / RansacLib baseline, 2026-10-01):
    # dataset ground-truth paths, scene lists and presets live in configs, not code
    "PRESET" * "_CAP25",
    "GT" * "_ETH3D", "GT" * "_SCANNET1500", "GT" * "_7SCENES", "GT" * "_KITTI",
    "GT_ETH" * "3D_ROMA", "GT_ETH" * "3D_SPLG",
    "GT_SCANNET" * "1500_ROMA", "GT_SCANNET" * "1500_SPLG",
    "GT_KIT" * "TI_ROMA", "GT_KIT" * "TI_SPLG",
    "KITTI_ODOMETRY" * "_SEQUENCES",
    "SEVEN_SCENES_TEST" * "_SPLITS", "SEVEN_SCENES_TRAIN" * "_SPLITS",
    "WMG" * "_NG",
    # the hand-rolled Levenberg–Marquardt cores, replaced by one LM driver
    # (NLLSsolver): the chart LM is VGC's `lmsolve_chart!`. The rule stands with ONE stated exception: the
    # spline models' device-side two-unknown LM kernel
    # (`Cameras.static_lm`, `_static_lm_increment`; NLLSsolver cannot run in a
    # KernelAbstractions kernel). No other hand-rolled LM loop is allowed.
    "_lmsolve" * "_manifold!",               # a hand-rolled copy of a static LM loop
    "_lm" * "_cholsolve!",                   # its unrolled Cholesky
    "_lmsolve" * "_p3p!",                    # VisualGeometryToolkitEstimation's P3P copy of the same loop
    "_pixel" * "_lm_increment",              # the pixel LM's Cholesky step: `_static_lm_increment` (plan 1b, the one LM `static_lm`)
    "_pixel" * "_lm",                        # the pixel LM of the old grid: `static_lm` on the area's scale-free box (plan 1b)
    "_newton" * "_solve",                    # the lens inverse's trust-region Newton (a Dual Jacobian in its loop): `_inverse_lm`, `static_lm` with the distortion's analytic Jacobian and a step bound
    "_det" * "_jacobian",                    # a ForwardDiff determinant of the distortion: `_det(_distort_jacobian(d, x))`
    # the SplineLens fold-in (plan 1b, Core 0.9): the central grid camera is a
    # CameraModel with a SplineLens, the non-central one a NoncentralSplineModel
    "Central" * "GenericModel",              # `CameraModel(K, SplineLens(…))`, from a source `CameraModel(SplineLens, source, size)`
    "Noncentral" * "GenericModel",           # `NoncentralSplineModel`
    "Spline" * "Projection",                 # a SplineLens's `forward` spline, on `equidistant_chart` about ẑ
    "nominal" * "_intrinsics",               # a SplineLens camera's K, its gauge: `calibration_matrix(cam)`, `intrinsics(cam)`
    "with" * "_nominal",                     # a re-gauge: `CameraModel(cam, K′)`
    "with" * "_domain",                      # where light falls is the bound's: `with_lens_coverage(BoundedCamera(cam, extent), region)`
    "generic" * "_lattice",                  # `spline_lattice(SplineLens, source, size)`
    "GenericModel" * "Window",               # `window(cam, rows, cols)`: a SplineLens camera (or NoncentralSplineModel) of the block
    "GenericGrid" * "Spec",                  # `SplineLensSpec`
    "central" * "_generic",                  # `CameraModelSpec(:spline; …)`, the TOML table type "spline"
    "default" * "_generic_grid",             # `default_spline_grid`
    "interpolated" * "_direction",           # the backward spline continued past its area: `cam.lens.backward(x)`, x = K⁻¹u
    "interpolated" * "_point",               # a NoncentralSplineModel's point spline: `m.points(x)`, x = K⁻¹u
    "_spline_projection" * "_fit",           # VisualGeometryToolkitEstimation: the forward spline's fit at one spacing, `_forward_fit`
    "_PixelFunction" * "Source",             # VisualGeometryToolkitEstimation: a function source; fit its `CameraModel(SplineLens, f, size)` instead
    "in_present" * "_domain",                # where a spline camera images: `all(isfinite, backproject(cam, u))`
    "spline" * "_support",                   # a point's window: `spline_weights(cam.lens.backward, x)[1:2]` (its `ix, iy`)
    "window" * "_controls", "window" * "_indices",   # a window's controls: `directions(cam)[iy - 3 + r, ix - 3 + c]`, r, c ∈ 1:4
    # the spline LM reads its residual and Jacobian from one evaluation per point (plan 1b, Task 4b)
    "_spline" * "_direction",                # the central residual's direction: `_spline_direction_differential(controls, area, x)[2]`
    "_noncentral_residual" * "_jacobian",    # `_noncentral_residual_and_jacobian` (the residual and its Jacobian, the LM's `fj`)
    "_backward" * "_ray",                    # the scalar lens backprojection `backproject(lens, x)` (`_backproject_lens` at given lattice slopes)
    "_lattice" * "_slope",                   # `_lattice_scales(T, area, gsz)`, the map's slopes in the point's float type
    "_to" * "_host",                         # VisualGeometryToolkitFeatures: `on_device(KA.CPU(), A)` (Core's one upload, base/device.jl)
    "_upload" * "_i32",                      # VisualGeometryToolkitFeatures: `ka_upload(Int32, dev, v)` (Core's one upload, base/device.jl)
    "_ka" * "_upload",                       # VisualGeometryToolkitFeatures' upload, hoisted public: `VisualGeometryToolkitCore.ka_upload`
    "in_kind" * "_of",                       # a forwarding wrapper, deleted: `on_device(get_backend(like), A)`
    "_on" * "_backend",                      # VisualGeometryToolkitFeatures: `on_device(T, dev, A)` (an array on `dev` in `T`, else a copy there)
    "_REFINE_ROBOT_WORLD" * "_MAX_MARKERS",  # VisualGeometryToolkitEstimation: the robot-world refinement's 4-marker cap, removed (2026-10-01)
    # robot-world/hand-eye calibration is named as such (VisualGeometryToolkitEstimation,
    # 2026-10-01): AX = YB, eye-to-hand; "hand_eye" is two tokens
    "fit_robot" * "_world", "refine_robot" * "_world",   # `fit_robot_world_hand_eye`, `refine_robot_world_hand_eye`
    "RobotWorld" * "Calibration", "RobotWorld" * "Metadata", "RobotWorld" * "RefinementMetadata",   # `RobotWorldHandEye…`
    "find_hand" * "eye",                     # `find_hand_eye`
    # the robust-loss vocabulary (2026-10-01): a robust loss is a LOSS, never a
    # "kernel"; one full name per width quantity (rule 10)
    "AbstractKernel" * "Shape", "Deployed" * "Kernel", "Annealed" * "Kernel",
    "Any" * "Kernel",                        # `AbstractLossShape`, `DeployedLoss`, `AnnealedLoss`, `AnyLoss`
    "kernel" * "_shape", "target" * "_kernel",   # `loss_shape`, `target_loss`
    "kernel" * "_family", "kernel" * "_short",   # `loss_family` (a method on the shape type), `loss_label`
    "KERNEL" * "_TAGS",                      # VisualGeometryToolkitEstimation: `LOSS_TAGS`
    "sq" * "_width",                         # `sq_loss_width` (and `loss_width` for the deployed width)
    "half_weight" * "_scale",                # `decision_width(shape, boundary)`
    "decision_threshold_to" * "_outlier_halfwidth",  # RANSACScoringDoneRight: `decision_width(::GauInlier, …)`
    "outlier_halfwidth_to" * "_decision_threshold",  # RANSACScoringDoneRight: `decision_boundary(::GauInlier, …)`
    "KERNELS" * ".md",                       # `LOSSES.md`
    "Inlier" * "Kernel", "AbstractLift" * "Kernel",  # aliases of `AbstractLossShape`, deleted
    "Kernel" * "Width",                      # VisualGeometryToolkitCore: `LossWidth`
    "LOSS" * "_TRIVIAL", "LOSS" * "_TRUNCATED", "LOSS" * "_HUBER", "LOSS" * "_CAUCHY",
    "LOSS" * "_TRUNCATED_CAUCHY", "LOSS" * "_TRUNCATED_LE_ZACH",   # PoseLib.jl: a typed `loss` (`TLSShape()`, `LeZachTruncation()`, …)
    # the evaluation engine: the external-solver width kind merged into `LossWidth`
    # (spelled with its call parentheses: the bare word is ordinary English)
    "Threshold" * "()",
    "PoseLibLmAdapt" * "Tau",                # `PoseLibLmAdaptLossWidth`
    "_refuse_kernel_width" * "_sidecar",     # `_refuse_basin_sidecar`
    "_loss" * "type",                        # `_poselib_loss`
    "superansac" * "_threshold", "superansac" * "_threshold_worker",   # the SupeRANSAC drivers: the `loss_width` forms
    "sr_threshold" * "_star", "tdv_sr_thr" * "_cv_path", "TDV_SR_THR" * "_DEFAULT", "SR_AUTHORS_EXAMPLE" * "_THR",
    # RANSACScoringDoneRight: the loss vocabulary and the full width names
    "Kernel" * "Scoring", "Tau" * "Scoring",             # `LossCostScoring`, `LossWidthScoring`
    "tau" * "_scoring", "grid_evaluate" * "_tau", "_grid_evaluate" * "_tau",   # `loss_width_scoring`, `grid_evaluate_loss_width`
    "Kernel" * "Quad", "Kernel" * "Spec", "kernel" * "_spec", "_kernel" * "_spec",   # `LossQuad`, `LossSpec`, `loss_spec`
    "kernel" * "_scale_bounds",                          # `loss_scale_bounds`
    "Gaussian" * "Inlier", "Welsch" * "Inlier", "Geman" * "Inlier", "SmoothTLS" * "Inlier",   # aliases of the VGC shapes, deleted
    "a_to" * "_tau", "tau_to" * "_a",                    # `outlier_halfwidth_to_loss_width`, `loss_width_to_outlier_halfwidth`
    "log_outlier" * "_width",                            # `outlier_halfwidth(scoring)`
    "log" * "2a",                                        # `log_two_outlier_halfwidth` (and `StudentTUniformScoring.outlier_halfwidth`)
    "simfitpp" * "_threshold", "simfitpp" * "_kernel",   # `simfitpp_loss_width`, `simfitpp_loss`
    "_MAGSAC" * "_K",                                    # `_MAGSAC_EFFICIENCY_CONSTANT`
    # VisualGeometryToolkitEstimation: the loss vocabulary and the full width names
    "Kernel" * "PlainWeights", "strategy" * "_kernel", "lo" * "_kernel", "_lift" * "_kernel",   # `LossPlainWeights`, `strategy_loss`, `lo_loss`, `_lift_loss`
    "_KERNEL" * "_TABLE", "_KERNEL" * "_TAG_OF", "_kernel" * "_tag", "_kernel" * "_of", "_fixed_kernel" * "_shape",
    "WELSCH_HQ_K" * "_ESSENTIAL", "WELSCH_HQ_K" * "_FUNDMAT", "WELSCH_HQ_K" * "_HOMOGRAPHY", "WELSCH_HQ_K" * "_ABSOLUTE_POSE",   # `WELSCH_HQ_EFFICIENCY_CONSTANT_*`
    "tau" * "_factor",                                   # `BarronAdaptive.truncation_multiple`
    "_crossover" * "_k", "_width" * "_k", "_resolve" * "_k", "_k" * "_currency", "_K" * "Currency", "_k" * "_shape",
    # one evaluation base (2026-10-05): the evaluation engine is VisualGeometryToolkitEval
    # and LocalOptimizationDoneRight. The pool record moved to VisualGeometryToolkitCore (`Pools.PoolPair`), hypothesis
    # drawing and the pool-pair and pool-frame problem constructors to
    # VisualGeometryToolkitEstimation, the scene-pool writer to VisualGeometryToolkitEval.
    "generate" * "_pool",                    # `draw_hypotheses(rng, problem; target, max_trials)`
    "build_stereo" * "scoringdata",          # `PoolPair(rng, problem, raw; target, max_trials)`
    "generate_scene" * "_pool",              # `write_scene_pool(path, raws, build; problem, seed)`
    "generate_all" * "_pools",               # (no successor: one driver per project)
    "pool" * "_main",                        # (no successor: the project's pool driver)
    "pool_frame_essential" * "_problem",     # `EssentialProblem(pair)`
    "_pool_frame" * "_problem",              # `FundMatProblem(pair)`, `HomographyProblem(pair)`
    "live_essential" * "_problem",           # `EssentialProblem(pair; sampler)`
    "AbstractRaw" * "Pair",                  # `StereoRawPair`
    "Pool" * "Config",                       # reader keywords + the generation verb's keywords
    "problem" * "_flag",                     # the pool file's `problem` symbol
    # --- K+ (the composable RANSAC): the scoring contract's and the
    # toolkit's renames, retired in one block. Built from fragments so this file
    # does not trip its own check.
    "Seed" * "Partition", "seed" * "_partition", "partition" * "_cardinality",
    "evaluate" * "_score!", "support" * "_count", "seed_part" * "ition_s2!",
    "Partition" * "Anchored", "Partition" * "Dof", "Estimate" * "()", "::" * "Estimate",
    "lo_scale" * "_s2", "with" * "_scale", "requires" * "_scale",
    "AbstractScoreable" * "Problem", "init" * "_score", "_score" * "_improved",
    "_wrap_solver" * "_cov", "partition_seeded" * "_posterior", "posterior" * "_scale",
    "Inverse" * "GammaPrior", "AbstractScale" * "Prior", "_log" * "Ztilde",
    "Marginal" * "Scoring", "Jeffreys" * "Marginal", "InverseGamma" * "Marginal",
    "PredictiveMarginal" * "Scoring", "StudentTUniform" * "Irls", "BarronAdaptive" * "Irls",
    "GemanMcClure" * "Shape", "GMHQ" * "Eps",
    "Factorial" * "Grid", "All" * "Units", "SimpleRandom" * "Sampling", "Product" * "GridSpec",
    "No" * "Fit", "Loss" * "Scoring",
    # K+ P3: the lift core's and the evaluation engine's retired vocabulary
    "Lift" * "Workspace", "lift" * "_workspace", "lift" * "_provider", "lift" * "_rescur",
    "Custom" * "Refit", "native" * "_axis", "MarginalPool" * "Scorer", "MsacPool" * "Scorer",
    "RansacPool" * "Scorer", "Refit" * "Marginal", "Refit" * "Jeffreys", "loss_width" * "_port",
    "Override" * "Port", "Native" * "Port", "LossWidth" * "Port", "Msac" * "Selector",
    "external_loss" * "_width", "fixed_scale" * "_s2s",
    # the units layering: FlexUnits as it ships, no stack quantity type,
    # dimension set or registry (2026-10-25)
    "Vgc" * "Quantity", "Pixel" * "Dimensions", "Pixel" * "Units",
    # the one parameter protocol: `parameter_vector`, `parameter_names`,
    # `parameter_bounds` for lenses, targets, warps and intrinsics (2026-10-25)
    "lens" * "_parameters", "target" * "_parameters", "warp" * "_parameters",
    "lens" * "_parameter_names", "target" * "_parameter_names", "warp" * "_parameter_names",
    # one camera hierarchy: `AbstractCamera` and the `is_posed` trait (2026-10-25)
    "Abstract" * "CameraModel",
    # a rig's target member is a `PosedTarget` (a posed target, as a
    # `PosedCamera` is a posed camera; `model(m)` is its target) (2026-10-26)
    "Mount" * "ed",
    # a correspondence normalization's sides are `source_map` and
    # `destination_map` (2026-10-26); format-4 pool files keep the former key,
    # spelled in fragments by VisualGeometryToolkitEval's pool IO
    "target" * "_map",
    # one placement trait for cameras and targets alike: `is_posed` (2026-10-26)
    "has" * "_extrinsics",
    # a planar target's design is moved by its `TargetCalibration` (field and
    # keyword `calibration`, and its parameter Jacobian) (2026-10-26). The target
    # record dropped its "warp" key (vgio/target/v2), but the bare word stays
    # unretired: it is also the image warp (`image_warp`) across the stack
    "Abstract" * "Warp", "warp" * "_jacobian", "warp" * " = Rigid",
    # the focal length a field of view implies is a method of the accessor,
    # `focal_length(sensor, θ; dimension)` (2026-10-27); its inverse is `angle_of_view`
    "focal_length" * "_for_fov",
    # the Sampson distance is geometric and the whitening a second layer over
    # it (2026-11-02): `sampson_residual` (signed, an SVector) and
    # `whiten(s_N, N, Σ)` / `mahalanobis_distance` for the noise
    "signed" * "_sampson", "signed_tangent" * "_sampson", "_whiten" * "_residual",
    # ONE Sampson path (2026-11-02): the residual's model Jacobian is
    # `sampson_residual_with_jacobian` over the model's
    # `observation_gradient_jacobian`, the batch rows over an observation set
    # call it per element, a frozen whitening is
    # `whitening_factor`, and the epipolar degeneracy scales are the models'
    # `constraint_rounding_scale` — the parallel epipolar rows, the whitening
    # policy types and their verb are gone
    "sampson_residual" * "_jacobian", "sampson_residual" * "_row", "tangent_sampson" * "_residual_row",
    "epipolar" * "_rounding", "whitened" * "_rows", "Whitening" * "Policy", "Sampson" * "Whitening", "No" * "Whitening",
    # the batch rows take an observation set whose per-element data are columns
    # (2026-11-02): `sampson_residuals!` / `whitened_residuals!` /
    # `sampson_jacobian_rows!` / `whitened_jacobian_rows!`; the RSS and the pool
    # matrix compose (`sampson_distances!`); the homography's and the rotation's
    # LM descend the exact rows, not a transfer whitening frozen at the anchor
    "sampson" * "_rows!", "sampson" * "_rss", "transfer" * "_whitening!", "transfer" * "_rows!", "transfer" * "_jacobian_rows!",
    # a target calibration is a CoordinateTransformations `Transformation`
    # (2026-11-03): its parameter Jacobian is CT's `transform_deriv_params`, its
    # point Jacobian CT's `transform_deriv` (conventions/jacobians.md)
    "calibration" * "_jacobian",
    # no target calibration kind has a gauge (2026-11-04): the affine print
    # error is its Iwasawa factors (the pose's directions left out of the
    # composition), the thin-plate kinds hold no plane part; `TargetManifold`
    # has no `free` keyword
    "gauge" * "_names", "gauge" * "_directions",
    # one model-to-model regression, `fit_camera(T, source, region)` (a lens
    # family, a spline lens and its forward spline), and its `CameraFitMetadata`
    # (VisualGeometryToolkitEstimation 0.6.0, 2026-11-04)
    "fit" * "_lens", "LensFit" * "Metadata",
    # the RANSAC per-point scalar is a distance under a distance's name (the 0.7
    # wave, conventions/residuals-and-distances.md): `mahalanobis_distances!(d,
    # problem, model)` over the hook `mahalanobis_distance(problem, ctx, i)`
    # (≥ 0); the signed residual is `whitened_residual(problem, ctx, i)`, its
    # rows `whitened_residuals!(r, problem, model)`; the squares are
    # `sq_distances` / `square_distances!`; the norm helper is gone
    "residuals" * "!", "whitened_constraint" * "_residual", "whitened" * "_distance",
    "sq" * "_residuals", "square" * "_residuals!",
    # VisualGeometryToolkitIO's old grid codec (plan 1b, Task 6): the SplineLens camera is a camera body, the
    # non-central grid camera the noncentral_spline/v1 body
    "_GENERIC" * "_CAMERA_SPEC",              # the non-central record's spec, its kind renamed after its `type` token: `_NONCENTRAL_SPLINE_SPEC`
    "_generic_camera" * "_v3_to_v4",          # the old grid kind's v3 step into the renamed kind: `_generic_camera_v3_to_noncentral`
    "_id_grid" * "_model",                   # a grid sensor's identity: `_id_camera`'s lens branch (`_id_lens(::SplineLens)`, its spline record)
    "_GENERIC_BODY" * "_FIELDS", "_GENERIC" * "_OPTIONAL", "_GENERIC" * "_PRESENT",   # the old grid body's spec: `_NONCENTRAL_BODY_FIELDS`, `_NONCENTRAL_BODY_OPTIONAL`
    "_RIG_GENERIC" * "_BODY",                # a rig's grid sensor body: `_NONCENTRAL_BODY` (a SplineLens sensor is a camera body)
    "_generic" * "_check",                   # `_noncentral_check` and `_lattice_problem` (the lens record's too)
    "_Generic" * "Model",                    # the old grid model union: `CameraModel{…, SplineLens}` or `NoncentralSplineModel`
    "_generic" * "_extras", "_generic" * "_extras_from",   # the nominal pinhole and domain: K, and the bound's `lens_coverage`
    "_load" * "_generic", "_typed" * "_generic",           # `load(BoundedCamera, path)` (`_load_noncentral`; a central file as a camera store)
    "_load_schoeps" * "_model",              # `_load_schoeps_camera`
    "_grid" * "_json", "_point_grid" * "_json", "_grid_from" * "_json",   # the held nodes' controls: `_controls_json`, `_controls_from_json`
    "_present" * "_count",                   # `_runs_count`
    "_refuse" * "_domain",                   # an old grid model's domain: the bound's lens coverage, any of its kinds (a camera store's form)
    # a Bayer image's planes are frame-labelled maps (VisualGeometryToolkitFeatures 0.9): a plane's sample
    # position and the cell grid's ×2 are read from `bayer_plane_map(L, c)` / `raw_frame_map(L)`, never
    # carried as bare offsets or an integer scale beside the image
    "plane" * "_offsets",                    # `bayer_plane_map(L, c)` (its translation onto the cell grid: `inv(raw_frame_map(L)) ∘ bayer_plane_map(L, c)`)
    "raw" * "_scale",                        # `raw_frame_map(L)` (PanoramaStitching's keyword: the frame's map onto the raw)
    # PanoramaStitching's 0.9 audit: its parallel feature record and its keypoint-noise and width helpers are the stack's
    # (the detections themselves, Features' `propagate(map, detection)`, the scorer's own width), never restated
    "Frame" * "Features", "keypoint" * "_covariance_scale", "keypoint" * "_uncertain", "pixel" * "_covariances",
    "bundle" * "_whitening", "scoring" * "_threshold_px",
]

# Retired names that a FOREIGN party owns — a foreign or stored format's token, or a
# foreign library's API function — each the exact string, its owner, and where it is
# accepted (`accepted_in`), never as an identifier, nor in a comment or prose elsewhere:
# - `:format_literal`, a format's token: inside a Julia string literal (a reader or
#   writer of that format names the token it parses) and anywhere in a data fixture
#   (`.yaml`, `.yml`, `.json`, an `Artifacts.toml`, whose keys are published names) —
#   its key there included (R14) — and inside a quoted TOML string value (a path or a
#   name a config quotes, `input_dir = ".../token"`), never a TOML key;
# - `:c_source`, a library's function: in a C/C++ source (`_C_SOURCE_EXTENSIONS`), the
#   wrapper that calls the library (built into a prebuilt artifact, so it is not edited
#   for a Julia rename);
# - `:cli_argument`, a foreign program's command-line value: the value of a `--flag`
#   (`--flag value`, `--flag=value`) on a command line in a shell script (`.sh`, its
#   code, not a comment), or inside a TOML string (a command line quoted as a
#   provenance record). Every hit on the line must be such a value; a TOML key, a
#   Julia file, prose and any other file still refuse it.
# (Built from fragments, as the lists above.)
const FOREIGN_FORMAT_TOKENS = [
    "Central" * "GenericModel" => (; owner = "the `type` of Schöps et al.'s camera_calibration intrinsics0.yaml, and of the old vgio grid " *
        "records (vgio/generic_camera v2/v3, a vgio/rig v6–v8 sensor's body), read through their migrations", accepted_in = :format_literal),
    "Noncentral" * "GenericModel" => (; owner = "the same, for the non-central grid model", accepted_in = :format_literal),
    "nominal" * "_intrinsics" => (; owner = "an old vgio grid record's key (vgio/generic_camera v2/v3, a vgio/rig v6–v8 sensor's body): its nominal pinhole",
        accepted_in = :format_literal),
    "image" * "_origin" => (; owner = "the legacy pixel-frame key of the OpenCV calibration JSON written before 0.9 (read by load_opencv_calibration, never written)",
        accepted_in = :format_literal),
    "bundle" * "_adjust" => (; owner = "PoseLib's C++ absolute-pose refinement, called by PoseLib.jl's C wrapper", accepted_in = :c_source),
    "inlier" * "_threshold" => (; owner = "RansacLib's and SupeRANSAC's settings field (C++ and pysuperansac), set by name in a string",
        accepted_in = :format_literal),
    "cs" * "ponds" => (; owner = "the key of the published correspondence records (the stored pool and dataset files, " *
        "RANSACScoringDoneRight's and LocalOptimizationDoneRight's Zenodo artifacts): a published name is data", accepted_in = :format_literal),
    "eth3d_splg_cs" * "ponds" => (; owner = "the name of RANSACScoringDoneRight's published ETH3D SuperPoint+LightGlue correspondence " *
        "artifact (Zenodo record 23240469): a published name is data", accepted_in = :format_literal),
    "central" * "_generic" => (; owner = "the `--model` value of Schöps et al.'s camera_calibration program (its central generic model), " *
        "in the shell drivers that run it and the TOML provenance that quotes their command line", accepted_in = :cli_argument),
]
_is_c_source(f) = any(e -> endswith(f, e), _C_SOURCE_EXTENSIONS)
_is_data_fixture(f) = any(e -> endswith(f, e), (".yaml", ".yml", ".json")) || basename(f) == "Artifacts.toml"   # an Artifacts.toml's keys are its artifacts' published names
# Whether `name` is a registered stored-format token (a data fixture's key may spell one).
_is_format_token(name) = any(t -> first(t) == name && last(t).accepted_in === :format_literal, FOREIGN_FORMAT_TOKENS)
# Whether the retired-name hit `r` on `line` (its string-literal text `strings`, a
# Julia file's) is a foreign party's name where its entry accepts it.
function _foreign_token(f, r, re, line, strings)
    k = findfirst(t -> first(t) == r, FOREIGN_FORMAT_TOKENS)
    k === nothing && return false
    accepted_in = last(FOREIGN_FORMAT_TOKENS[k]).accepted_in
    accepted_in === :c_source && return _is_c_source(f)
    accepted_in === :cli_argument && return _cli_argument(f, re, line)
    _is_data_fixture(f) && return true
    endswith(f, ".toml") && return _toml_string_value(re, line)
    return endswith(f, ".jl") && count(re, line) == count(re, strings)
end
# Whether every hit of `re` on a TOML `line` is inside a quoted string VALUE (`key = "…/token"`, or an array
# element on its own line): not in a key, a table header or a comment.
function _toml_string_value(re, line)
    occursin(re, _hash_code(line; strings = false)) && return false      # a hit outside every quoted run (a bare key)
    startswith(lstrip(line), "[") && return false                         # a table header, quoted keys included
    eq = findfirst(==('='), _hash_code(line; strings = false))
    eq === nothing && return true                                         # a value line of a multi-line array
    return all(m -> m.offset > eq, eachmatch(re, line))                   # after the `=`: the value, never a quoted key
end
# Whether every hit of `re` on `line` is a `--flag`'s value on a command line: in a
# shell script's code (comments blanked), or in a TOML file inside a quoted string.
function _cli_argument(f, re, line)
    is_shell = endswith(f, ".sh")
    is_shell || endswith(f, ".toml") || return false
    flag_value = Regex("--[A-Za-z][\\w-]*(?:\\s+|=)" * re.pattern)
    count(re, line) == count(flag_value, _hash_code(line)) || return false
    return is_shell || !occursin(re, _hash_code(line; strings = false))
end

# Rule 10: the width quantities are bound by their full names. One name per
# quantity — `efficiency_constant` (a width as a multiple of the estimated scale),
# `loss_width` (a loss's scale parameter), `outlier_halfwidth` (the Student-t
# marginal's uniform outlier half-width) — and the abbreviations they replaced
# are banned as BINDINGS, at the precision a syntactic checker can reach:
#
#   - the multi-letter abbreviations have no other stack meaning: banned as every
#     bound name, quoted symbol, TOML key, sidecar key and CLI flag, and the
#     word `tau` within any of them (`_WIDTH_ABBREVIATIONS`);
#   - a single letter (`k`, `a`, `c`, `τ`) or `thr` is also a loop index, a math
#     local, an ellipse semi-axis, a division coefficient, a k-NN count, a
#     detector threshold — so it is banned only where the binding itself says
#     it is a width (`_WIDTH_LETTERS`): a field, type parameter or constructor
#     keyword of a width-carrying type (a subtype of `_WIDTH_CARRIER_ROOTS`,
#     resolved through the package's own declarations); a name bound beside a
#     noise scale (`_NOISE_SCALE_NAMES`) in one method signature, struct or
#     NamedTuple literal; a TOML key; a CLI flag. (As a sidecar key a letter is
#     also a lens coefficient or a column name, so there only the abbreviations
#     are flagged.)
const _WIDTH_ABBREVIATIONS = Set([
    "tau", "c_ref", "a_ref", "absw", "k_op", "k_grid", "a_grid", "c_grid", "tau_grid",
])
const _WIDTH_LETTERS = Set(["k", "a", "c", "τ", "thr"])
const _NOISE_SCALE_NAMES = Set([:s2, :ŝ, :s_hat, :ŝ2, :ŝ², :s_hat_sq])
const _WIDTH_CARRIER_ROOTS = Set([
    :AbstractOptimizationStrategy, :AbstractScoring, :AbstractWeightPolicy, :AbstractLossShape,
    :ScaleInterface, :SampleSelector, :WidthParameterization,
])
# A CLI flag spelling an abbreviation or a letter (two dashes, then any of the
# names above with `_` or `-` between its words), on any line of any file.
const _WIDTH_FLAG = Regex(
    "(?<![\\p{L}\\p{N}_-])--(?:" *
        join(sort!([replace(f, "_" => "[-_]") for f in union(_WIDTH_ABBREVIATIONS, _WIDTH_LETTERS)]; by = length, rev = true), "|") *
        ")(?![\\p{L}\\p{N}_-])"
)

# An abbreviation a binding may never use: the exact name in any letter case, or
# the word `tau` inside it.
_is_width_abbreviation(name::AbstractString) =
    lowercase(name) in _WIDTH_ABBREVIATIONS || "tau" in _naming_words(name)

# The hooks of `block_coordinate_descent!` (rule 8), exactly, and its steps in
# Manopt.jl's names.
const _BCD_HOOKS = (
    :init_state!, :update_model!, :update_residuals!, :update_nuisance!, :update_weights!,
    :post_iteration!, :stopping_criterion, :stop_reason, :block_coordinate_result,
    :initialize_solver!, :step_solver!, :stop_solver!,
)

# Manopt.jl's spelling of the driver's one cycle: the one identifier besides a
# `<fit>_step` that holds the word (rule 5), kept so a Manopt bridge extends it.
const _MANOPT_CYCLE_NAMES = ("step" * "_solver!",)

# The stack's packages: a qualified definition `M.f(…)` whose `M` is one of
# these (or a module or alias of the scanned package) binds a stack name.
const _STACK_PACKAGES = Set([
    :VisualGeometryToolkitCore, :VisualGeometryToolkitEstimation, :VisualGeometryToolkitEval,
    :RANSACScoringDoneRight, :BlobBoards, :VisualGeometryToolkitFeatures, :VisualGeometryToolkitIO,
    :PoseLib, :LocalOptimizationDoneRight,
])

# -----------------------------------------------------------------------------
# Scope: the stack text of a package directory
# -----------------------------------------------------------------------------

# Paper sources (`.tex`, `.bib`, figure data) are out of scope by extension
# alone: the experiment code, scripts and docs kept beside a paper are stack text.
const _C_SOURCE_EXTENSIONS = (".cpp", ".h", ".hpp")   # the C/C++ sources scanned (a `:c_source` name is accepted in them)
const _NAMING_EXTENSIONS = (".jl", ".md", ".toml", ".yml", ".yaml", _C_SOURCE_EXTENSIONS..., ".sh", ".sbatch", ".py")
const _LICENSE_FILES = ("LICENSE", "LICENSE.md", "LICENSE.txt", "COPYING", "NOTICE")
# Third-party provenance is DECLARED here, at the repository root (see `naming_scope`).
const _VENDORED_FILE = "VENDORED.toml"
const _VENDORED_KEYS = ("path", "upstream", "license")

# `git` is `_git_command(dir)`: the git executable, or `nothing` outside a checkout.
function _package_files(dir::AbstractString, git)
    if git !== nothing
        listed = split(read(`$git -C $dir ls-files -z`, String), '\0'; keepempty = false)
        return String[f for f in listed if isfile(joinpath(dir, f))]
    end
    files = String[]
    for (root, _, names) in walkdir(dir), name in names
        push!(files, relpath(joinpath(root, name), dir))
    end
    return filter!(f -> !(".git" in splitpath(f)), files)
end

# The holders a license names: the text after `Copyright`, `(c)`/`©` and the
# years on each line that opens with the word (not the body's "the above
# copyright notice"), less a trailing "All rights reserved" and e-mail address.
function _license_holders(text::AbstractString)
    holders = String[]
    for line in eachline(IOBuffer(text))
        m = match(r"^[\s#/*;%-]*copyright\b(.*)$"i, line)
        m === nothing && continue
        h = m.captures[1]
        h = replace(h, r"\(c\)|©"i => " ", r"\ball rights reserved\b.*$"i => "", r"<[^>]*>" => "")
        h = replace(h, r"^[\s\d,.:;\-–]*" => "", r"[\s,.;:]+$" => "")
        isempty(h) || push!(holders, h)
    end
    return holders
end

# The git executable when `dir` is inside a git work tree, else `nothing`: probed
# once per scope and handed to every git query.
function _git_command(dir)
    git = Sys.which("git")
    git === nothing && return nothing
    inside = success(pipeline(`$git -C $dir rev-parse --is-inside-work-tree`; stdout = devnull, stderr = devnull))
    return inside ? git : nothing
end

# The repository's own copyright holders: those its root license names, else
# the author of its first commit (a checkout with no root license), else none.
# The root commits are listed newest first, so the first commit is the last line.
function _repo_holders(dir::AbstractString, all_files, git)
    holders = String[]
    for f in all_files
        dirname(f) == "" && f in _LICENSE_FILES && append!(holders, _license_holders(read(joinpath(dir, f), String)))
    end
    isempty(holders) || return holders
    if git !== nothing
        authors = readchomp(ignorestatus(`$git -C $dir log --max-parents=0 --format=%an HEAD`))
        isempty(authors) || push!(holders, String(last(split(authors, '\n'))))
    end
    return filter!(!isempty, holders)
end

# Why the license file `f` looks upstream, or `nothing` when it names a holder
# of the repository (the stack's own licensed subtree).
function _foreign_license(dir, f, holders)
    text = read(joinpath(dir, f), String)
    any(h -> occursin(lowercase(h), lowercase(text)), holders) && return nothing
    named = unique!(_license_holders(text))
    return string(
        "a license (", f, ") naming ", isempty(named) ? "no holder" : join(named, "; "),
        ", not the repository's holder ", isempty(holders) ? "(unknown)" : join(holders, "; ")
    )
end

# The trees declared `[[vendored]]` and the license files declared `[[template]]`
# in VENDORED.toml, each `path => reason`, and every defect of the declaration.
function _declared_provenance(dir::AbstractString)
    vendored = Dict{String, String}()
    templates = Dict{String, String}()
    errors = String[]
    file = joinpath(dir, _VENDORED_FILE)
    isfile(file) || return (; vendored, templates, errors)
    decl = try
        TOML.parsefile(file)
    catch e
        push!(errors, "$_VENDORED_FILE does not parse: " * sprint(showerror, e))
        return (; vendored, templates, errors)
    end
    for (k, v) in decl
        k in ("vendored", "template") && v isa AbstractVector && continue
        push!(errors, "$_VENDORED_FILE: `$k` is not a `[[vendored]]` or `[[template]]` entry")
    end
    for kind in ("vendored", "template"), entry in get(decl, kind, [])
        entry isa AbstractDict || continue
        what = "$_VENDORED_FILE [[$kind]] $(repr(get(entry, "path", "")))"
        lacking = filter(k -> !(get(entry, k, nothing) isa String) || isempty(entry[k]), collect(_VENDORED_KEYS))
        unknown = setdiff(keys(entry), _VENDORED_KEYS)
        isempty(lacking) || push!(errors, "$what lacks $(join(lacking, ", "))")
        isempty(unknown) || push!(errors, "$what has unknown keys $(join(sort!(collect(unknown)), ", "))")
        (isempty(lacking) && isempty(unknown)) || continue
        p = rstrip(normpath(entry["path"]), '/')
        if p == "." || startswith(p, "..") || isabspath(p)
            push!(errors, "$what is not a path below the repository root")
        elseif !ispath(joinpath(dir, p))
            push!(errors, "$what does not exist")
        elseif kind == "template" && !(basename(p) in _LICENSE_FILES)
            push!(errors, "$what is not a license file ($(join(_LICENSE_FILES, ", ")))")
        else
            reason = string("declared ", kind, " in ", _VENDORED_FILE, ": ", entry["upstream"], ", ", entry["license"])
            (kind == "vendored" ? vendored : templates)[p] = reason
        end
    end
    return (; vendored, templates, errors)
end

_is_under(f, d) = f == d || startswith(f, d * "/")

"""
    naming_scope(dir) -> (; files, excluded)

The in-scope files of the package at `dir` (relative paths), and the directory
trees excluded as third-party source, each as `tree => reason`. Scope is every
git-tracked file (every file, outside a git checkout) with a stack-text
extension, less `Manifest.toml`, `docs/build/`, the process records (plans and
release records, which name retired APIs and banned words on purpose:
`docs/superpowers/`, and the top-level `.release/`, where the release tooling
keeps its records, their first lines in its own fixed format) and the declared
vendored trees. Only the root's
`.release/` is a record; a `.release/` deeper in the tree is scanned.
Paper sources are out of scope by extension (`.tex`, `.bib` and figure data are
not scanned); the code and docs beside a paper are in scope.

Third-party provenance is declared, never guessed. A repository that carries
upstream source lists each tree in a root `VENDORED.toml`:

```toml
[[vendored]]            # third-party code: the whole tree is out of scope
path = "lib/RansacLibLO/deps/RansacLib"
upstream = "https://github.com/tsattler/RansacLib"
license = "BSD-3-Clause"

[[template]]            # a template's license file; its directory stays in scope
path = "paper/template/LICENSE"
upstream = "a conference's LaTeX template"
license = "MIT"
```

A `[[template]]` entry names one license FILE: it covers a template (a conference's
LaTeX class, say) whose directory otherwise holds stack text, so the files
beside it are scanned like any other. It excludes nothing; it only acknowledges
that the license is not the repository's.

`naming_scope` throws an `ArgumentError` naming every defect: a declared path
that does not exist, an entry lacking `path`, `upstream` or `license`, a
template entry that is not a license file, and any `LICENSE`/`COPYING`/`NOTICE`
file below the root that names a holder other than the repository's (read from
the root license's `Copyright` lines, else the author of the first commit) and
is neither inside a declared vendored tree nor declared a template.
Undeclared third-party code is declared or removed; it is never skipped.
"""
function naming_scope(dir::AbstractString)
    git = _git_command(dir)
    all_files = _package_files(dir, git)
    decl = _declared_provenance(dir)
    errors = copy(decl.errors)
    holders = _repo_holders(dir, all_files, git)
    vendored_tree(f) = findfirst(d -> _is_under(f, d), collect(keys(decl.vendored)))
    for f in all_files
        basename(f) in _LICENSE_FILES && dirname(f) != "" || continue
        (vendored_tree(f) !== nothing || haskey(decl.templates, f)) && continue
        r = _foreign_license(dir, f, holders)
        r === nothing || push!(errors, string(
            "`", dirname(f), "` carries ", r, ": declare it in ", _VENDORED_FILE,
            " ([[vendored]] for third-party code, [[template]] for a template's license beside stack text) or remove it"
        ))
    end
    isempty(errors) || throw(ArgumentError("naming_scope($(repr(dir))):\n  " * join(errors, "\n  ")))
    files = String[]
    for f in all_files
        parts = splitpath(f)
        base = parts[end]
        any(endswith(base, e) for e in _NAMING_EXTENSIONS) || continue
        (startswith(base, "Manifest") || startswith(base, "JuliaManifest")) && endswith(base, ".toml") && continue
        length(parts) >= 2 && parts[1] == "docs" && parts[2] in ("build", "superpowers") && continue
        length(parts) >= 2 && parts[1] == ".release" && continue
        any(d -> _is_under(f, d), keys(decl.vendored)) && continue
        push!(files, f)
    end
    return (; files, excluded = sort!(collect(decl.vendored); by = first))
end

# -----------------------------------------------------------------------------
# Words
# -----------------------------------------------------------------------------

# The lowercase words of a name or a run of text: split at every character that
# is not a letter or digit (`_`, `-`, `!`, …), at CamelCase humps, and between
# letters and digits.
function _naming_words(s::AbstractString)
    out = String[]
    for part in split(s, r"[^\p{L}\p{N}]+"; keepempty = false)
        for w in split(part, r"(?<=[\p{Ll}\p{N}])(?=\p{Lu})|(?<=\p{Lu})(?=\p{Lu}\p{Ll})|(?<=\p{L})(?=\p{N})|(?<=\p{N})(?=\p{L})"; keepempty = false)
            push!(out, lowercase(w))
        end
    end
    return out
end

# The first retired phrase `words` holds as a contiguous run, or `nothing`.
function _retired_phrase(words::AbstractVector{<:AbstractString})
    for p in RETIRED_PHRASES, i in 1:(length(words) - length(p) + 1)
        all(k -> words[i + k - 1] == p[k], eachindex(p)) && return p
    end
    return nothing
end

# A bound name that counts random draws as iterations (rule 7): a
# `min_`/`max_` bound spelled with `iterations` (the retired RANSAC budget's
# spelling; a deterministic loop's bound is `max_iter`), or a RANSAC/PROSAC
# word beside an iteration word. A quoted symbol is a value, not a counter:
# the stop reason `:max_iterations` of a deterministic loop stands.
function _counts_draws_as_iterations(words, kind::Symbol)
    if kind !== :symbol
        any(i -> words[i] in ("min", "max") && words[i + 1] == _IT * "s", 1:(length(words) - 1)) && return true
    end
    return any(in(_RANDOM_DRAW_WORDS), words) && any(in(_ITERATION_WORDS), words)
end

# (Julia prose, the comment and string text of a source by line, is `_julia_prose`: lexers.jl)

# -----------------------------------------------------------------------------
# Julia code: the names a file binds
# -----------------------------------------------------------------------------

struct _NamingContext
    stack::Set{Symbol}      # modules whose qualified definitions bind stack names
    external::Set{Symbol}   # names this file imports from outside the stack
    # rule 10's context, collected by the walk:
    groups::Vector{Vector{Tuple{Symbol, Int, Symbol}}}    # names bound together: one signature, struct or NamedTuple
    members::Vector{Tuple{Symbol, Vector{Tuple{Symbol, Int, Symbol}}}}   # a type's fields and parameters, a callee's keywords
    supertypes::Dict{Symbol, Symbol}                      # each declared type's supertype
    keys::Vector{Tuple{String, Int}}                      # string and symbol dict keys (sidecar keys)
end
_NamingContext(stack::Set{Symbol}, external::Set{Symbol}) = _NamingContext(
    stack, external, Vector{Tuple{Symbol, Int, Symbol}}[],
    Tuple{Symbol, Vector{Tuple{Symbol, Int, Symbol}}}[], Dict{Symbol, Symbol}(), Tuple{String, Int}[]
)

_qualifier_root(x::Symbol) = x
_qualifier_root(x::Expr) = x.head === :. && !isempty(x.args) ? _qualifier_root(x.args[1]) : nothing
_qualifier_root(x) = nothing

function _unwrap_signature(sig)
    while sig isa Expr
        if sig.head === :where
            sig = sig.args[1]
        elseif sig.head === :(::) && length(sig.args) == 2 && sig.args[1] isa Expr &&
                sig.args[1].head in (:call, :where)
            sig = sig.args[1]
        else
            break
        end
    end
    return sig
end

_is_signature(x) = (s = _unwrap_signature(x); s isa Expr && s.head === :call)

function _argument_names!(acc, a, line, kind)
    if a isa Symbol
        push!(acc, (a, line, kind))
    elseif a isa Expr
        h = a.head
        if h === :(::)
            length(a.args) == 2 && _argument_names!(acc, a.args[1], line, kind)
        elseif h === :kw || h === :(=) || h === :...
            _argument_names!(acc, a.args[1], line, kind)
        elseif h === :tuple || h === :block
            foreach(x -> _argument_names!(acc, x, line, kind), a.args)
        elseif h === :parameters
            foreach(x -> _argument_names!(acc, x, line, :keyword), a.args)
        elseif h === :macrocall
            _argument_names!(acc, a.args[end], line, kind)
        end
    end
    return acc
end

function _assigned_names!(acc, lhs, line, kind)
    if lhs isa Symbol
        push!(acc, (lhs, line, kind))
    elseif lhs isa Expr
        h = lhs.head
        if h === :(::) && length(lhs.args) == 2
            _assigned_names!(acc, lhs.args[1], line, kind)
        elseif h === :tuple || h === :parameters
            foreach(x -> _assigned_names!(acc, x, line, kind), lhs.args)
        elseif h === :curly || h === :...
            _assigned_names!(acc, lhs.args[1], line, kind)
        end
    end
    return acc
end

function _type_name!(acc, x, line)
    if x isa Symbol
        push!(acc, (x, line, :type))
    elseif x isa Expr && x.head in (:curly, :<:)
        _type_name!(acc, x.args[1], line)
    end
    return acc
end

function _field_names!(acc, body, line)
    body isa Expr || return acc
    for a in body.args
        if a isa LineNumberNode
            line = a.line
        elseif a isa Symbol
            push!(acc, (a, line, :field))
        elseif a isa Expr
            h = a.head
            if h === :(::) || h === :(=) || h === :const
                _is_signature(a.args[1]) || _assigned_names!(acc, a.args[1], line, :field)
            elseif h === :macrocall
                _field_names!(acc, Expr(:block, a.args[end]), line)
            end
        end
    end
    return acc
end

# The signature's names are also one rule-10 group, and its keywords the
# members of its callee (a constructor's keywords are its type's).
function _signature_names!(acc, sig, line, ctx::_NamingContext, kind)
    n0 = length(acc)
    _signature_bindings!(acc, sig, line, ctx, kind)
    bound = acc[(n0 + 1):end]
    push!(ctx.groups, bound)
    callee = _callee_name(_unwrap_signature(sig))
    callee === nothing || push!(ctx.members, (callee, filter(b -> b[3] === :keyword, bound)))
    return acc
end

function _callee_name(sig)
    (sig isa Expr && sig.head === :call) || return nothing
    c = sig.args[1]
    c isa Expr && c.head === :curly && (c = c.args[1])
    c isa Expr && c.head === :. && length(c.args) == 2 && c.args[2] isa QuoteNode && (c = c.args[2].value)
    return c isa Symbol ? c : nothing
end

# The last name of a (possibly qualified, possibly parametric) type expression.
function _type_symbol(x)
    x isa Expr && x.head === :curly && return _type_symbol(x.args[1])
    x isa Expr && x.head === :. && length(x.args) == 2 && x.args[2] isa QuoteNode && return _type_symbol(x.args[2].value)
    return x isa Symbol ? x : nothing
end

# `T{P, Q <: X}`: the type parameters a declaration binds.
function _type_params!(acc, x, line)
    x isa Expr && x.head === :<: && (x = x.args[1])
    x isa Expr && x.head === :curly || return acc
    for p in x.args[2:end]
        p isa Expr && p.head in (:<:, :>:) && (p = p.args[1])
        p isa Symbol && push!(acc, (p, line, :typeparam))
    end
    return acc
end

# `T <: S`: record the supertype of a declared type.
function _record_supertype!(ctx::_NamingContext, x)
    x isa Expr && x.head === :<: && length(x.args) == 2 || return
    t = _type_symbol(x.args[1])
    st = _type_symbol(x.args[2])
    t === nothing || st === nothing || (ctx.supertypes[t] = st)
    return
end

function _signature_bindings!(acc, sig, line, ctx::_NamingContext, kind)
    sig = _unwrap_signature(sig)
    if sig isa Symbol
        sig in ctx.external || push!(acc, (sig, line, kind))
        return acc
    end
    sig isa Expr || return acc
    if sig.head === :tuple                                   # anonymous function
        foreach(x -> _argument_names!(acc, x, line, :argument), sig.args)
        return acc
    end
    sig.head === :call || return acc
    callee = sig.args[1]
    callee isa Expr && callee.head === :curly && (callee = callee.args[1])
    if callee isa Symbol
        callee in ctx.external || push!(acc, (callee, line, kind))
    elseif callee isa Expr && callee.head === :. && length(callee.args) == 2 && callee.args[2] isa QuoteNode
        root = _qualifier_root(callee.args[1])
        name = callee.args[2].value
        root in ctx.stack && name isa Symbol && push!(acc, (name, line, kind))
    elseif callee isa Expr && callee.head === :(::)          # functor
        length(callee.args) == 2 && _argument_names!(acc, callee.args[1], line, :argument)
    end
    foreach(x -> _argument_names!(acc, x, line, :argument), sig.args[2:end])
    return acc
end

function _node_names!(acc, ex::Expr, line, ctx::_NamingContext)
    h = ex.head
    if h === :function || h === :macro
        _signature_names!(acc, ex.args[1], line, ctx, h === :macro ? :macro : :function)
    elseif h === :(=)
        lhs = ex.args[1]
        if _is_signature(lhs)
            _signature_names!(acc, lhs, line, ctx, :function)
        else
            _assigned_names!(acc, lhs, line, :local)
        end
    elseif h === :->
        _argument_names!(acc, ex.args[1], line, :argument)
    elseif h === :struct
        _type_name!(acc, ex.args[2], line)
        n0 = length(acc)
        _type_params!(acc, ex.args[2], line)
        _field_names!(acc, ex.args[3], line)
        members = acc[(n0 + 1):end]
        push!(ctx.groups, members)
        t = _type_symbol(ex.args[2] isa Expr && ex.args[2].head === :<: ? ex.args[2].args[1] : ex.args[2])
        t === nothing || push!(ctx.members, (t, members))
        _record_supertype!(ctx, ex.args[2])
    elseif h === :abstract || h === :primitive
        _type_name!(acc, ex.args[1], line)
        _type_params!(acc, ex.args[1], line)
        _record_supertype!(ctx, ex.args[1])
    elseif h === :module
        ex.args[2] isa Symbol && push!(acc, (ex.args[2], line, :module))
    elseif h === :local || h === :global
        foreach(a -> a isa Symbol && push!(acc, (a, line, :local)), ex.args)
    elseif h === :try
        ex.args[2] isa Symbol && push!(acc, (ex.args[2], line, :local))
    end
    return acc
end

# Every bound name, with every quoted symbol (`:sym`, a stack value), of a parsed file.
# A string or quoted-symbol literal, as the key text it spells, or `nothing`.
_key_text(x::AbstractString) = String(x)
_key_text(x::QuoteNode) = x.value isa Symbol ? String(x.value) : nothing
_key_text(x) = nothing

# The dict accessors whose second argument is a key.
const _KEY_ACCESSORS = (:get, :get!, :haskey, :getindex, :delete!, :pop!)

# The keys a sidecar (a Dict, a TOML or JSON table, a JLD2 group) is written or
# read under: the left of a `=>`, an index, and a dict accessor's key.
function _dict_key!(ctx::_NamingContext, ex::Expr, line)
    k = if ex.head === :call && length(ex.args) == 3 && ex.args[1] === :(=>)
        _key_text(ex.args[2])
    elseif ex.head === :ref && length(ex.args) == 2
        _key_text(ex.args[2])
    elseif ex.head === :call && length(ex.args) >= 3 && ex.args[1] in _KEY_ACCESSORS
        _key_text(ex.args[3])
    else
        nothing
    end
    k === nothing || push!(ctx.keys, (k, line))
    return
end

# A NamedTuple literal `(name = value, …)` or `(; name = value, …)`: its names
# are FIELDS, not locals. `(; a, b) = x` (property destructuring) has no `=`
# inside the tuple and is left to the assignment.
function _namedtuple_fields(ex::Expr)
    ex.head === :tuple || return nothing
    items = length(ex.args) == 1 && ex.args[1] isa Expr && ex.args[1].head === :parameters ? ex.args[1].args : ex.args
    isempty(items) && return nothing
    all(a -> a isa Expr && a.head in (:(=), :kw) && a.args[1] isa Symbol, items) || return nothing
    return items
end

function _walk_names!(acc, quoted, ex, line, ctx)
    if ex isa QuoteNode
        ex.value isa Symbol && push!(quoted, (ex.value, line))
        return
    end
    ex isa Expr || return
    ex.head in (:import, :using, :export, :public) && return
    fields = _namedtuple_fields(ex)
    if fields !== nothing
        group = [(a.args[1]::Symbol, line, :field) for a in fields]
        append!(acc, group)
        push!(ctx.groups, group)
        foreach(a -> _walk_names!(acc, quoted, a.args[2], line, ctx), fields)
        return
    end
    _dict_key!(ctx, ex, line)
    _node_names!(acc, ex, line, ctx)
    ex.head === :quote && length(ex.args) == 1 && ex.args[1] isa Symbol && push!(quoted, (ex.args[1], line))
    l = line
    for (k, a) in enumerate(ex.args)
        if a isa LineNumberNode
            l = a.line
        elseif !(ex.head === :. && k == 2)                  # `x.field` is a use
            _walk_names!(acc, quoted, a, l, ctx)
        end
    end
    return
end

function _visit(f, ex)
    ex isa Expr || return
    f(ex)
    foreach(a -> _visit(f, a), ex.args)
    return
end

# The modules a package defines and the aliases it gives stack modules.
function _stack_modules!(stack::Set{Symbol}, parsed)
    for ex in parsed
        _visit(ex) do e
            e.head === :module && e.args[2] isa Symbol && push!(stack, e.args[2])
        end
    end
    for _ in 1:2, ex in parsed
        _visit(ex) do e
            if e.head === :(=) && e.args[1] isa Symbol && _qualifier_root(e.args[2]) in stack
                push!(stack, e.args[1])
            elseif e.head === :as && _qualifier_root(e.args[1]) in stack && e.args[2] isa Symbol
                push!(stack, e.args[2])
            end
        end
    end
    return stack
end

# The names a file imports from outside the stack: defining a method of one
# keeps its owner's spelling (`import Base: step`).
function _external_imports(ex, stack)
    ext = Set{Symbol}()
    _visit(ex) do e
        e.head in (:import, :using) || return
        for clause in e.args
            if clause isa Expr && clause.head === :(:)
                root = _qualifier_root(clause.args[1])
                (root === :. || root in stack) && continue
                for item in clause.args[2:end]
                    item isa Expr && item.head === :. && push!(ext, item.args[end])
                    item isa Expr && item.head === :as && push!(ext, item.args[2])
                end
            elseif clause isa Expr && clause.head === :. && length(clause.args) >= 2
                root = clause.args[1]
                (root === :. || root in stack) || push!(ext, clause.args[end])
            end
        end
    end
    return ext
end

function _parse_error_line(ex)
    ex isa Expr || return nothing
    line = 0
    for a in ex.args
        a isa LineNumberNode && (line = a.line)
        a isa Expr && a.head in (:error, :incomplete) && return line
    end
    return nothing
end

# -----------------------------------------------------------------------------
# Other languages: the names a Python, shell or TOML file binds
# -----------------------------------------------------------------------------
#
# Line-level approximations, not parsers (limits in `check_naming`'s docstring).
# Each returns `(name, line, kind)` like the Julia walk.

# Python's own words: a language-owned word is never a binding.
const _PYTHON_KEYWORDS = Set(split(
    "False None True and as assert async await break class continue def del elif else except " *
    "finally for from global if import in is lambda nonlocal not or raise return try while " *
    "with yield"
))

_offset_lines(code) = pushfirst!([i + 1 for i in findall(==('\n'), code)], 1)

# Split `s` at the commas outside brackets, with each piece's byte offset.
function _toplevel_split(s::AbstractString, sep::Char = ',')
    out = Tuple{String, Int}[]
    depth = 0
    start = 1
    for (k, c) in pairs(s)
        if c in "([{"
            depth += 1
        elseif c in ")]}"
            depth -= 1
        elseif c == sep && depth == 0
            push!(out, (s[start:prevind(s, k)], start))
            start = k + 1
        end
    end
    push!(out, (s[start:end], start))
    return out
end

# The names an assignment or `for` target binds: identifiers, `*rest`,
# (nested) tuples and `self.<field>`; subscripts and other attributes bind nothing.
function _python_targets!(acc, target::AbstractString, line, kind)
    t = strip(target)
    while length(t) >= 2 && ((first(t) == '(' && last(t) == ')') || (first(t) == '[' && last(t) == ']'))
        t = strip(t[2:prevind(t, lastindex(t))])
    end
    pieces = _toplevel_split(t)
    if length(pieces) > 1
        foreach(p -> _python_targets!(acc, p[1], line, kind), pieces)
        return acc
    end
    t = lstrip(t, '*')
    if (m = match(r"^([A-Za-z_]\w*)$", t)) !== nothing
        m[1] in _PYTHON_KEYWORDS || push!(acc, (Symbol(m[1]), line, kind))
    elseif (m = match(r"^self\s*\.\s*([A-Za-z_]\w*)$", t)) !== nothing
        push!(acc, (Symbol(m[1]), line, :field))
    end
    return acc
end

# The statements of blanked Python code: `(text, byte offset)` of each run of
# lines that opens outside brackets and not after a `\` continuation, split at `;`.
function _python_statements(code::AbstractString)
    stmts = Tuple{String, Int}[]
    depth = 0
    start = 1
    cont = false
    for (k, c) in pairs(code)
        if c in "([{"
            depth += 1
        elseif c in ")]}"
            depth = max(depth - 1, 0)
        elseif c == '\n' && depth == 0 && !cont
            push!(stmts, (code[start:prevind(code, k)], start))
            start = k + 1
        end
        c == '\n' || c == ' ' || c == '\t' || (cont = c == '\\')
    end
    start <= lastindex(code) && push!(stmts, (code[start:end], start))
    out = Tuple{String, Int}[]
    for (s, o) in stmts, (p, q) in _toplevel_split(s, ';')
        isempty(strip(p)) || push!(out, (p, o + q - 1))
    end
    return out
end

function _python_names(src::AbstractString)
    acc = Tuple{Symbol, Int, Symbol}[]
    code = _python_code(src)
    starts = _offset_lines(code)
    line(o) = searchsortedlast(starts, o)
    for m in eachmatch(r"(?m)^[ \t]*(?:async[ \t]+)?def[ \t]+([A-Za-z_]\w*)[ \t]*\(", code)
        push!(acc, (Symbol(m[1]), line(m.offset), :function))
        open = m.offset + ncodeunits(m.match) - 1
        depth = 0
        close = open
        for k in open:ncodeunits(code)
            c = codeunit(code, k)
            c in (UInt8('('), UInt8('['), UInt8('{')) && (depth += 1)
            c in (UInt8(')'), UInt8(']'), UInt8('}')) && (depth -= 1)
            depth == 0 && (close = k; break)
        end
        params = String(codeunits(code)[(open + 1):(close - 1)])
        for (p, q) in _toplevel_split(params)
            pm = match(r"^\s*\**\s*([A-Za-z_]\w*)", p)
            pm === nothing || pm[1] in _PYTHON_KEYWORDS ||
                push!(acc, (Symbol(pm[1]), line(open + q + pm.offsets[1] - 1), :keyword))
        end
    end
    for m in eachmatch(r"(?m)^[ \t]*class[ \t]+([A-Za-z_]\w*)", code)
        push!(acc, (Symbol(m[1]), line(m.offset), :type))
    end
    for m in eachmatch(r"\bfor[ \t]+([^\n]+?)[ \t]+in\b", code)
        _python_targets!(acc, something(m[1]), line(m.offset), :local)
    end
    for m in eachmatch(r"\bas[ \t]+([A-Za-z_]\w*)", code)
        m[1] in _PYTHON_KEYWORDS || push!(acc, (Symbol(m[1]), line(m.offset), :local))
    end
    for m in eachmatch(r"([A-Za-z_]\w*)[ \t]*:=", code)
        push!(acc, (Symbol(m[1]), line(m.offset), :local))
    end
    # Assignments and annotations: statements that do not open with a keyword.
    for (s, o) in _python_statements(code)
        fw = match(r"^\s*([A-Za-z_]\w*)", s)
        fw !== nothing && fw[1] in _PYTHON_KEYWORDS && continue
        depth = 0
        seg = 1
        found = false
        for (k, c) in pairs(s)
            if c in "([{"
                depth += 1
            elseif c in ")]}"
                depth -= 1
            elseif c == '=' && depth == 0
                prev = k > 1 ? s[prevind(s, k)] : ' '
                nxt = k < lastindex(s) ? s[nextind(s, k)] : ' '
                (prev in "=<>!" || nxt == '=') && continue
                target = s[seg:prevind(s, k)]
                occursin(r"\blambda\b", target) && break
                target = rstrip(target, collect("+-*/%&|^@<>: \t"))
                a = findfirst(':', target)                        # `x: int = …`
                a === nothing || (target = target[1:prevind(target, a)])
                _python_targets!(acc, target, line(o + seg - 1), :local)
                seg = k + 1
                found = true
            end
        end
        if !found && (m = match(r"^\s*([A-Za-z_]\w*)[ \t]*:[ \t]*\S", s)) !== nothing
            push!(acc, (Symbol(m[1]), line(o), :field))              # `name: T`
        end
    end
    return acc
end

# Shell and sbatch: `NAME=`, `export|local|readonly|declare NAME=`, `for NAME in`
# and function definitions, on each line's code (quotes blanked, comment cut).
function _shell_names(src::AbstractString)
    acc = Tuple{Symbol, Int, Symbol}[]
    for (k, text) in enumerate(eachline(IOBuffer(src)))
        code = replace(text, r"'[^']*'" => s -> " "^length(s), r"\"(?:[^\"\\]|\\.)*\"" => s -> " "^length(s))
        cut = match(r"(?:^|[\s;])#", code)
        cut === nothing || (code = code[1:(cut.offset)])
        for m in eachmatch(r"(?:^|[;&|(`]|\b(?:then|do|else)\b)\s*(?:(?:export|local|readonly|declare|typeset)(?:\s+-\w+)*\s+)?([A-Za-z_]\w*)(?:\[[^\]]*\])?\+?=", code)
            push!(acc, (Symbol(m[1]), k, :local))
        end
        for m in eachmatch(r"\bfor\s+([A-Za-z_]\w*)\s+in\b", code)
            push!(acc, (Symbol(m[1]), k, :local))
        end
        m = match(r"^\s*(?:function\s+([A-Za-z_][\w-]*)|([A-Za-z_][\w-]*)\s*\(\s*\)\s*\{?\s*$)", code)
        m === nothing || push!(acc, (Symbol(something(m[1], m[2])), k, :function))
    end
    return acc
end

# A table whose keys NAME dependencies (a package, a conda or PyPI project): the
# key is the dependency's own name, a use, not a binding.
const _TOML_DEPENDENCY_TABLES = Set([
    "deps", "weakdeps", "extras", "compat", "sources",
    "dependencies", "dev-dependencies", "build-dependencies", "optional-dependencies",
    "pypi-dependencies", "pip",
])

# A TOML source parsed, with its lines; `nothing` when it does not parse. The one reader of the rules' TOML walks.
function _toml_doc(src::AbstractString)
    doc = try
        TOML.parse(src)
    catch
        return nothing
    end
    return doc, collect(eachline(IOBuffer(src)))
end

# Every key of a parsed TOML table, depth first in the table's order: `visit(d, k, path, table)`, with `d` the table
# holding `k`, `path` the dotted path of `d` ("" at the top) and `table` the key `d` was reached by ("" at the top). An
# array's tables are walked as tables of its key. The one walk of the rules' TOML checks.
function _walk_toml(visit, d::AbstractDict, path::AbstractString = "", table::AbstractString = "")
    for (k, v) in d
        visit(d, k, path, table)
        p = isempty(path) ? k : path * "." * k
        if v isa AbstractDict
            _walk_toml(visit, v, p, k)
        elseif v isa AbstractVector
            foreach(x -> x isa AbstractDict && _walk_toml(visit, x, p, k), v)
        end
    end
    return nothing
end

# Every key of a TOML file, at the line that spells it first (a header or a `key =`).
function _toml_names(src::AbstractString)
    acc = Tuple{Symbol, Int, Symbol}[]
    parsed = _toml_doc(src)
    parsed === nothing && return nothing
    doc, lines = parsed
    function lineof(key)
        re = Regex("(?<![\\w-])" * replace(key, r"[\\^$.|?*+()\[\]{}]" => s -> "\\" * s) * "(?![\\w-])")
        k = findfirst(l -> (h = replace(l, r"#.*$" => ""); occursin(re, first(split(h, '='))) || startswith(strip(h), "[") && occursin(re, h)), lines)
        return something(k, 1)
    end
    _walk_toml(doc) do _d, k, _path, table
        table in _TOML_DEPENDENCY_TABLES || push!(acc, (Symbol(k), lineof(k), :key))
    end
    return acc
end

# -----------------------------------------------------------------------------
# Rules on a bound name
# -----------------------------------------------------------------------------

function _is_fit_update(name::AbstractString, fits::Set{String})
    b = chopprefix(chopsuffix(name, "!"), "_")
    endswith(b, "_step") || return false
    fit = chopsuffix(b, "_step")
    return !isempty(fit) && !occursin("step", fit) && ("fit_" * fit) in fits
end

# A bound name is matched on SUBSTRINGS of its words: a joined identifier
# (`stepsize`, a `bypass_cache` flag) holds the word as surely as a separated one,
# and a name is chosen, not quoted. Prose stays whole-word (`_prose_rules!`).
function _name_rules(name::AbstractString, kind::Symbol, fits::Set{String}; julia::Bool = true)
    rules = Symbol[]
    words = _naming_words(name)
    (any(w -> occursin(_STEM, w), words) || _counts_draws_as_iterations(words, kind)) && push!(rules, :R7)
    (startswith(name, "update_") || startswith(name, "_update_")) && !endswith(name, "!") && push!(rules, :R3)
    (startswith(name, "next_") || startswith(name, "_next_")) && endswith(name, "!") && push!(rules, :R4)
    if any(w -> occursin("step", w), words)
        if "delta" in words || "increment" in words
            push!(rules, :R6)
        elseif !(kind in (:function, :symbol) && (_is_fit_update(name, fits) || name in _MANOPT_CYCLE_NAMES))
            push!(rules, :R5)
        end
    end
    # rule 14: the terminology taxonomy's retired words, in any binding; "measured" a value only
    lw = lowercase.(words)
    (any(w -> any(t -> occursin(t, w), _RETIRED_TERMS), lw) ||
     (kind !== :symbol && any(w -> occursin(_PROVENANCE_ONLY, w), lw))) && push!(rules, :R14)
    # rule 10: an abbreviation is never a binding; a letter is not a TOML key
    (_is_width_abbreviation(name) || (kind === :key && name in _WIDTH_LETTERS)) && push!(rules, :R10)
    if julia && kind === :function                          # the hooks are Julia's
        sym = Symbol(name)
        bare = chopsuffix(name, "!")
        if !(sym in _BCD_HOOKS) && any(h -> chopsuffix(String(h), "!") == bare, _BCD_HOOKS)
            push!(rules, :R8)
        end
    end
    return rules
end

# -----------------------------------------------------------------------------
# The checker
# -----------------------------------------------------------------------------

"""
    check_naming(pkgdir) -> Vector{NamingViolation}

Check the package at `pkgdir` against the stack's iterative-estimator
vocabulary (CLAUDE.md, rules 3–9). Empty means the package complies; every
stack package's quality test asserts that.

Julia code is checked on every name it BINDS — functions and methods
(including qualified methods of stack functions), types, fields, arguments,
keywords, locals, loop variables, constants, macros and modules — on every
word that CONTAINS a banned word or `step` (a joined `stepsize` included), plus
every quoted symbol (`:sym`, a name the stack spells as a value). A method of a
function owned outside the stack keeps its owner's name (`Base.step`), and a
use of such a name is not a binding. Julia comments and strings, and all other
in-scope files ([`naming_scope`](@ref)), are scanned word by word for the
banned words and the retired phrases, and every line of every file for the
retired names — except a `CHANGELOG.md`, whose job is to record what each
release removed. The lists live in this function's file and nowhere else.

Rule 7's trial/iteration split is checked both ways. In prose, a retired
phrase (`RETIRED_PHRASES`, whole words in a contiguous run) names a random
hypothesis draw as an iteration or a deterministic cycle as a trial. A bound
name (not a quoted symbol) spelled `min_iterations`/`max_iterations` (the
deterministic bound is `max_iter`; a random-draw budget is `max_trials`), or
holding a RANSAC/PROSAC word beside an iteration word, counts random draws as
iterations. A deterministic loop's stop reason `:max_iterations` is a value and
stands, and the one sanctioned crossing — VisualGeometryToolkitCore's
`FitResult.iterations`, which counts a RANSAC's trials — is a field of its own
name.

The same identifier rules (R3–R7) hold for the names other in-scope files bind,
read by line-level approximations rather than parsers:

- **Python** (`.py`): `def` and `class` names, every parameter name of a `def`
  signature (a multi-line signature included), assignment targets of a
  statement that opens outside brackets (chained, tuple, starred, annotated
  and augmented; `self.<field>` binds the field, a subscript or other attribute
  binds nothing), bare annotations (`name: T`), `for` targets (comprehensions
  included), `as` names and `:=` targets. Comments and string literals are
  blanked first, so a word in them is prose, not a name. Limits: a statement
  after `:` on the same line (`if c: x = 1`) and assignments inside a `lambda`
  are not read, and an f-string's expressions are treated as string text.
  Python's own keywords are the language's, not bindings: the identifier rules
  never see them, while the text scan still does (write `...` for an empty block).
- **Shell** (`.sh`, `.sbatch`): `NAME=` at the start of a command (after `;`,
  `&&`, `|`, `then`, `do`, `else`, or `export`/`local`/`readonly`/`declare`),
  `for NAME in`, and function definitions, on each line with quoted text
  blanked and the comment cut. Limits: a heredoc's body is read as commands,
  and a quoted string spanning lines is not blanked.
- **TOML** (`.toml`, less `Manifest.toml`): every key, parsed with `TOML`, at
  the first line that spells it. The keys of a dependency table (`[deps]`,
  `[compat]`, `[weakdeps]`, `[extras]`, `[sources]`, `[*dependencies]`,
  `[pip]`) are other projects' names, not bindings. A file that does not parse
  is reported as `:parse`.

Rule R8 (the hook names of `block_coordinate_descent!`) is Julia's alone.

Rule R11 keeps units in one place: VisualGeometryToolkitCore's units module
(`src/base/units.jl`: the stack's registry `StackUnits`, built with FlexUnits'
`RegistryTools` from exact rational scales, `parse_unit` — the stack's one
guarded `uparse` — `unit_string` and `base_quantity`) and FlexUnits itself.
Anywhere else, in Julia code (comments and strings are prose), it flags a call
to `uparse`, `qparse` or `register_unit`, a unit or registry built by hand (a
call to `Units`, `AffineTransform`, `PermanentDict`, `register_unit!` or
`registry_defaults!`), a reach into `UnitRegistry.UNITS`,
`convert(Quantity{…}, q)` (`base_quantity`), a `x isa Quantity ? ustrip(…)`
branch (`strip_or_bare`), a literal holding two or more unit names (a unit
table: `("mm", "px")`, `Dict("mm" => …)`), and, outside `test/`, a
hand-written conversion factor (`25.4`, `0.0254`, `304.8`, `0.3048`, or an
inch ratio such as `127//5000`) and a bare `ubase(…)` (a stored quantity is
`base_quantity(q)`: `ubase` keeps the exact `Rational` of `142mm`). It also keeps
the units layering. A pose or a target's points (`EuclideanMap`,
`PlanarTarget`, `PointTarget`, `TargetObservations`) built with a `ustrip(…)` or a
`ustrip.(…)` in its arguments is flagged: a physical value is built from
quantities (`t .* m`), and its plain form comes from the whole value's
`ustrip(u, X)`. Outside `test/`, a definition that names a file's
`length_unit` may not scale by a power of ten (`x * 0.001`, `x / 1000`),
because a file length is converted by `read_length` and `write_length` alone.
Limits: a component stripped into a local and handed on later is not seen, and
neither is a scale held in a named constant. What FlexUnits
genuinely lacks or what must stay bit-identical to stored values is allowed by
file, with its reason, in `_UNIT_RULE_ALLOW`.

Rule R10 binds the three width quantities by their full names
(`efficiency_constant`, `loss_width`, `outlier_halfwidth`). The multi-letter
abbreviations they replaced (`_WIDTH_ABBREVIATIONS`, and the word `tau` in any
name) are flagged wherever they are bound — every Julia binding kind, quoted
symbols, Python/shell names and TOML keys — and as dict (sidecar) keys and CLI
flags. A single letter (`k`, `a`, `c`, `τ`) or `thr` also names loop indices,
math locals and other quantities, so it is flagged only where the binding
itself says it is a width: as a TOML key or a CLI flag; as a
name bound beside a noise scale (`s2`, `ŝ`, … in one method signature, struct
or NamedTuple literal); and as a field, type parameter or constructor keyword
of a width-carrying type (one descending, through the package's own
`struct`/`abstract type` declarations, from `_WIDTH_CARRIER_ROOTS`). Limits: a
letter bound as a positional argument or local with no noise scale beside it,
or as a field of a type outside those roots, cannot be told from a loop index or
another quantity by syntax and is not flagged; neither is a quoted single
letter (`:k`) or a letter used as a dict key (a lens coefficient `"k"` is one).
Rule R12: the Student-t + uniform scorer has no threshold. That word (the
whole token, in any case and the plural too, not a compound name that holds it) within six tokens of the
scorer's type name or
its TOML tag (`_STUDENT_T_ANCHORS`) is flagged in every file but a
changelog — code, comments, docs and configuration alike. Limits: a sentence
that calls the half-width a threshold farther from the scorer's name is not
seen, and a threshold scorer's threshold written next to the Student-t's name
is a finding to reword (or to spell apart, as a test of the refusal does).
Rule R14: the terminology taxonomy. No bound name (nor quoted symbol) holds
"measurement", the retired pairing abbreviation, "sighting" or "keypoint" in
any word (a detection, a correspondence, an observation: conventions/naming.md);
"measured" is a provenance value, a quoted symbol or a string, and never a bound
name.

Rule R13: the one parameter-holding keyword is `fixed`. `free`, `fix`,
`fixed_params` and `hold` as a KEYWORD of a fit, refine or target call or
signature (typed or not, with a default or without), or as a key of a
refine/target/LM TOML table (at its line in that table), are flagged; the bare
word (comments, locals, fields, other functions) is not (see `_fixed_keyword_rules!`).
"""
function check_naming(pkgdir::AbstractString)
    (; files) = naming_scope(pkgdir)
    sources = Dict(f => read(joinpath(pkgdir, f), String) for f in files)
    jl = filter(f -> endswith(f, ".jl"), files)
    parsed = Dict(f => Meta.parseall(sources[f]; filename = f) for f in jl)

    stack = _stack_modules!(copy(_STACK_PACKAGES), values(parsed))
    names = Dict{String, Vector{Tuple{Symbol, Int, Symbol}}}()
    quoted = Dict{String, Vector{Tuple{Symbol, Int}}}()
    ctxs = Dict{String, _NamingContext}()
    for f in jl
        ctx = ctxs[f] = _NamingContext(stack, _external_imports(parsed[f], stack))
        names[f] = Tuple{Symbol, Int, Symbol}[]
        quoted[f] = Tuple{Symbol, Int}[]
        _walk_names!(names[f], quoted[f], parsed[f], 1, ctx)
    end
    # rule 10: a type carries widths when it descends from a width-carrying
    # root, through the package's own declarations
    supertypes = merge(Dict{Symbol, Symbol}(), (c.supertypes for c in values(ctxs))...)
    function is_carrier(t::Symbol)
        seen = Set{Symbol}()
        while !(t in seen)
            t in _WIDTH_CARRIER_ROOTS && return true
            push!(seen, t)
            haskey(supertypes, t) || return false
            t = supertypes[t]
        end
        return false
    end
    foreign = Dict{String, Union{Nothing, Vector{Tuple{Symbol, Int, Symbol}}}}()
    for f in files
        if endswith(f, ".py")
            foreign[f] = _python_names(sources[f])
        elseif endswith(f, ".sh") || endswith(f, ".sbatch")
            foreign[f] = _shell_names(sources[f])
        elseif endswith(f, ".toml")
            foreign[f] = _toml_names(sources[f])
        end
    end
    fits = Set{String}()
    for bound in Iterators.flatten((values(names), values(foreign)))
        bound === nothing && continue
        for (n, _, k) in bound
            k === :function && push!(fits, String(n))
        end
    end

    # A retired name stands alone: no letter, digit, `_` or `!` on either side.
    # One alternation screens each line; only a hit is resolved name by name.
    # A name is matched literally: `!`, a file suffix's `.` and a call's `()` are escaped.
    retired_word(p) = Regex("(?<![\\p{L}\\p{N}_!])" * p * "(?![\\p{L}\\p{N}_!])")
    escaped(r) = replace(r, r"([!.()])" => s"\\\1")
    retired = [(r, retired_word(escaped(r))) for r in RETIRED_NAMES]
    any_retired = retired_word("(?:" * join(escaped.(RETIRED_NAMES), "|") * ")")
    out = NamingViolation[]
    for f in files
        src = sources[f]
        if endswith(f, ".jl")
            err = _parse_error_line(parsed[f])
            err === nothing || push!(out, NamingViolation(f, err, "", :parse))
            for (n, line, kind) in names[f], rule in _name_rules(String(n), kind, fits)
                push!(out, NamingViolation(f, line, String(n), rule))
            end
            for (n, line) in quoted[f], rule in _name_rules(String(n), :symbol, fits)
                push!(out, NamingViolation(f, line, ":" * String(n), rule))
            end
            for (line, text) in _julia_prose(src)
                _prose_rules!(out, f, line, text)
            end
            _width_binding_rules!(out, f, ctxs[f], is_carrier)
        elseif haskey(foreign, f)
            bound = foreign[f]
            if bound === nothing
                push!(out, NamingViolation(f, 1, "", :parse))
            else
                for (n, line, kind) in bound, rule in _name_rules(String(n), kind, fits; julia = false)
                    rule === :R14 && _is_data_fixture(f) && _is_format_token(String(n)) && continue
                    push!(out, NamingViolation(f, line, String(n), rule))
                end
            end
        end
        _is_changelog(f) || _student_t_threshold_rules!(out, f, src)
        _fixed_keyword_rules!(out, f, src, endswith(f, ".jl") ? parsed[f] : nothing)
        literals = Ref{Union{Nothing, Dict{Int, String}}}(nothing)   # the string literals by line, lexed at the first hit
        for (k, text) in enumerate(eachline(IOBuffer(src)))
            endswith(f, ".jl") || _prose_rules!(out, f, k, text)
            for m in eachmatch(_WIDTH_FLAG, text)
                push!(out, NamingViolation(f, k, m.match, :R10))
            end
            # a changelog records what each release removed: it names the retired
            _is_changelog(f) && continue
            occursin(any_retired, text) || continue
            endswith(f, ".jl") && literals[] === nothing && (literals[] = _julia_prose(src; comments = false))
            strings = endswith(f, ".jl") ? get(literals[], k, "") : ""
            for (r, re) in retired
                occursin(r, text) && occursin(re, text) && !_foreign_token(f, r, re, text, strings) &&
                    push!(out, NamingViolation(f, k, r, :R9))
            end
        end
    end
    pkg = _package_name(pkgdir)
    for f in jl
        (pkg, f) == _UNIT_HOME && continue
        haskey(_UNIT_RULE_ALLOW, (pkg, f)) && continue
        _unit_rules!(out, f, parsed[f], startswith(f, "test/"))
    end
    unique!(v -> (v.file, v.line, v.text, v.rule), out)
    sort!(out; by = v -> (v.file, v.line, v.rule))
    return out
end

# -----------------------------------------------------------------------------
# Rule 12: the Student-t + uniform scorer has no threshold
# -----------------------------------------------------------------------------
# Its one width is `outlier_halfwidth` (the half-width of the uniform outlier
# component); a threshold is a threshold scorer's (RANSAC, MSAC, the loss-width
# families). The word `threshold` — the whole token, not a compound name that
# holds it, in any case and the plural too — within
# `_STUDENT_T_WINDOW` tokens of the scorer's type or TOML tag, in code, comments,
# docs or config alike, misnames that half-width.
const _STUDENT_T_ANCHORS = ("StudentTUniformScoring", "student_t_uniform")
const _STUDENT_T_WINDOW = 6
function _student_t_threshold_rules!(out, f, src::AbstractString)
    any(a -> occursin(a, src), _STUDENT_T_ANCHORS) || return out
    toks = collect(eachmatch(r"[\p{L}\p{N}_!]+", src))
    anchors = findall(m -> m.match in _STUDENT_T_ANCHORS, toks)
    for (i, m) in enumerate(toks)
        lowercase(m.match) in ("threshold", "thresholds") && any(j -> abs(j - i) <= _STUDENT_T_WINDOW, anchors) || continue
        push!(out, NamingViolation(f, count(==('\n'), SubString(src, 1, prevind(src, m.offset))) + 1, m.match, :R12))
    end
    return out
end

# -----------------------------------------------------------------------------
# Rule 13: the one parameter-holding keyword is `fixed`
# -----------------------------------------------------------------------------
# The owner's ruling (2026-11-11): parameters held out of a fit, refine or
# target are named by `fixed` (NLLSsolver's word; OpenCV's `CALIB_FIX_*`), the
# exceptions to "every parameter refined". `free`, `fix`, `fixed_params` and
# `hold` are refused as that KEYWORD — and only as it, never as a word:
#   - Julia: a keyword argument so named in a CALL or a SIGNATURE of a
#     parameter-selecting function (`_R13_CALLEES`: `fit_*`, `refine_*`, their
#     `_`-prefixed internals, `LensSpec` and CaliBlobs' `LensTarget`). A signature's refusal
#     slot (`free = RetiredKeyword()`) and a call inside `@test_throws` (the
#     test of the refusal) are the refusal itself, not a use. Comments
#     ("CUDA-free", "free memory", "for free"), a local `free`, a struct field
#     (a result's record of what was refined) and Base's or a package's own
#     functions are not keywords of those callees and are not seen.
#   - TOML: a key so named in a table whose path names a refine or an LM
#     (`_R13_TABLES`), or in a lens table (one with a `lens` key: a LensSpec's), where a key IS a parameter-selection
#     keyword; the same word in any other table (a scheduler's "free" slots) is
#     not seen.
# Limits: a selection keyword of a function outside `_R13_CALLEES` is not seen.
const _R13_SPELLINGS = (:free, :fix, :fixed_params, :hold)
const _R13_CALLEES = r"^_*(fit|refine)_|^_*refine$|^(LensSpec|LensTarget)$"
const _R13_TABLES = r"(^|\.)(\w*refine\w*|lm)(\.|$)"
_r13_refusal(v) = Meta.isexpr(v, :call) && (v.args[1] === :RetiredKeyword ||
    (Meta.isexpr(v.args[1], :.) && v.args[1].args[end] == QuoteNode(:RetiredKeyword)))   # qualified, as a downstream package writes it
function _r13_keywords!(acc, ex, line::Int)
    ex isa Expr || return acc
    if Meta.isexpr(ex, :macrocall) && ex.args[1] in (Symbol("@test_throws"),)
        return acc
    end
    if Meta.isexpr(ex, :call) && occursin(_R13_CALLEES, String(something(_type_symbol(ex.args[1]), Symbol(""))))   # a callee's last name (none interpolated)
        for a in ex.args[2:end]
            kws = Meta.isexpr(a, :parameters) ? a.args : (a,)
            for k in kws
                if Meta.isexpr(k, :kw, 2)
                    name = k.args[1] isa Expr && Meta.isexpr(k.args[1], :(::)) ? k.args[1].args[1] : k.args[1]
                    name in _R13_SPELLINGS && !_r13_refusal(k.args[2]) && push!(acc, (name, line))
                elseif k isa Symbol && Meta.isexpr(a, :parameters) && k in _R13_SPELLINGS
                    push!(acc, (k, line))       # `f(x; free)`
                elseif Meta.isexpr(k, :(::)) && Meta.isexpr(a, :parameters) && k.args[1] in _R13_SPELLINGS
                    push!(acc, (k.args[1], line))   # a typed keyword with no default, `f(x; free::Tuple)`
                end
            end
        end
    end
    for a in ex.args
        if a isa LineNumberNode
            line = a.line
        else
            _r13_keywords!(acc, a, line)
        end
    end
    return acc
end
function _fixed_keyword_rules!(out, f, src::AbstractString, parsed)
    if endswith(f, ".jl")
        for (name, line) in _r13_keywords!(Tuple{Symbol, Int}[], parsed, 1)
            push!(out, NamingViolation(f, line, String(name), :R13))
        end
    elseif endswith(f, ".toml")
        toml = _toml_doc(src)
        toml === nothing && return out
        doc, lines = toml
        _walk_toml(doc) do d, k, path, _table
            if Symbol(k) in _R13_SPELLINGS && (occursin(_R13_TABLES, path) || haskey(d, "lens"))
                # the key's line in ITS table: the first `k =` after the table's header (`[path]` or `[[path]]`)
                header = isempty(path) ? nothing :
                    findfirst(l -> occursin(Regex("^\\s*\\[\\[?\\s*\\Q" * path * "\\E\\s*\\]"), l), lines)
                line = something(findnext(l -> occursin(Regex("^\\s*" * k * "\\s*="), l), lines, something(header, 0) + 1), 1)
                push!(out, NamingViolation(f, line, k, :R13))
            end
        end
    end
    return out
end

# A release history (a package's or the repository's CHANGELOG.md): the one
# kind of file where a retired name is the point — it records the removal.
_is_changelog(f) = basename(f) == "CHANGELOG.md"

# -----------------------------------------------------------------------------
# Rule 11: units in one place
# -----------------------------------------------------------------------------

# The package a directory holds: its Project.toml `name` ("" without one).
function _package_name(dir::AbstractString)
    p = joinpath(dir, "Project.toml")
    isfile(p) || return ""
    return string(get(TOML.parsefile(p), "name", ""))
end

# The units module: the registry, the guarded parse and its inverse.
const _UNIT_HOME = ("VisualGeometryToolkitCore", "src/base/units.jl")

# (package, file) => why rule 11 does not apply there: what FlexUnits genuinely
# lacks, or arithmetic that must stay bit-identical to stored values.
const _UNIT_RULE_ALLOW = Dict{Tuple{String, String}, String}(
    ("VisualGeometryToolkitCore", "ext/VisualGeometryToolkitCoreTestExt/check_naming.jl") =>
        "this rule's own lists of unit names and factors",
    ("VisualGeometryToolkitCore", "test/units_tests.jl") =>
        "the units module's own tests: they exercise uparse, qparse, the registry and convert",
)

# Unit names as a file or a table spells them.
const _UNIT_NAME_STRINGS = Set([
    "mm", "cm", "m", "um", "μm", "inch", "in", "ft", "pt", "px", "pd", "dpi",
    "mm/px", "μm/px", "um/px", "px/mm",
])
const _CONVERSION_FACTORS = (25.4, 0.0254, 304.8, 0.3048)   # mm or m per inch and per foot
# the registry's exact ratios, spelled as Rational literals (inch, its inverse, foot, point)
const _RATIONAL_FACTORS = ((127, 5000), (5000, 127), (381, 1250), (127, 360000))
# the RegistryTools constructors a unit or a registry is built with
const _REGISTRY_BUILDERS = (:Units, :AffineTransform, :PermanentDict, :register_unit!, :registry_defaults!)

_unit_rule_callee(f::Symbol) = f
_unit_rule_callee(f::GlobalRef) = f.name
_unit_rule_callee(f::Expr) = f.head === :. && length(f.args) == 2 && f.args[2] isa QuoteNode ? f.args[2].value : nothing
_unit_rule_callee(_) = nothing

_is_unit_name(x) = x isa String && x in _UNIT_NAME_STRINGS
# The unit-name strings a collection literal holds directly, or as the first
# element of a pair or a tuple entry (`"mm" => mm`, `("mm", mm)`).
function _unit_name_count(ex::Expr)
    n = 0
    for a in ex.args
        if _is_unit_name(a)
            n += 1
        elseif a isa Expr && ((a.head === :call && length(a.args) == 3 && a.args[1] === :(=>)) || a.head === :tuple)
            k = a.head === :tuple ? first(a.args, 1) : a.args[2:2]
            n += count(_is_unit_name, k)
        end
    end
    return n
end
_is_quantity_type(t) = t isa Expr && t.head === :curly && _unit_rule_callee(t.args[1]) === :Quantity
_is_ustrip_call(x) = x isa Expr && x.head === :call && _unit_rule_callee(x.args[1]) === :ustrip
_is_isa_quantity(c) = c isa Expr && c.head === :call && length(c.args) == 3 && c.args[1] === :isa &&
    _unit_rule_callee(c.args[3]) === :Quantity

# A `ustrip(…)` call or a `ustrip.(…)` broadcast, qualified or not.
_is_ustrip(x) = _is_ustrip_call(x) ||
    (x isa Expr && x.head === :. && length(x.args) == 2 && x.args[2] isa Expr && x.args[2].head === :tuple &&
        _unit_rule_callee(x.args[1]) === :ustrip)
_holds_ustrip(x) = _is_ustrip(x) || (x isa Expr && any(_holds_ustrip, x.args))

# The constructors of a public physical value: a pose and the targets' points.
# A unitful value is built from quantities (`t .* m`); a plain one comes from
# the whole value's `ustrip(u, X)`, never from stripping its components.
const _PHYSICAL_CONSTRUCTORS = (:EuclideanMap, :PlanarTarget, :PointTarget, :TargetObservations)
_constructor_name(c) = c isa Expr && c.head === :curly ? _unit_rule_callee(c.args[1]) : _unit_rule_callee(c)

# A file's length unit, named in code: `d.length_unit`, `:length_unit`,
# `"length_unit"` or a binding of that name.
_names_length_unit(x) = x === :length_unit || x == "length_unit" || x == QuoteNode(:length_unit) ||
    (x isa Expr && any(_names_length_unit, x.args))

# A power-of-ten scale literal, the factor of a hand-converted file length
# (`x * 0.001`, `x / 1000`, `1.0e-6 * x`).
const _DECIMAL_SCALE_EXPONENTS = (-9, -6, -3, -2, 2, 3, 6, 9)
_is_decimal_scale(x) = (x isa Float64 || x isa Float32 || x isa Int) && !iszero(x) &&
    any(k -> abs(Float64(x)) == exp10(k), _DECIMAL_SCALE_EXPONENTS)
const _SCALING_OPERATORS = (:*, :/, :\, :.*, :./, :.\)
const _SCALING_UPDATES = (:*=, :/=, :.*=, :./=)
_is_definition(x) = x isa Expr && (x.head === :function ||
    (x.head === :(=) && length(x.args) == 2 && x.args[1] isa Expr && x.args[1].head in (:call, :where, :(::))))

function _unit_rules!(out, f, ex, in_test::Bool)
    line = Ref(0)
    flag(text) = push!(out, NamingViolation(f, line[], text, :R11))
    file_unit = Ref(false)   # inside a definition that names a file's length unit
    function walk(x)
        if x isa LineNumberNode
            line[] = x.line
        elseif x isa Float64
            !in_test && x in _CONVERSION_FACTORS && flag(string(x))
        elseif x isa Expr
            if !in_test && !file_unit[] && _is_definition(x) && _names_length_unit(x)
                file_unit[] = true
                foreach(walk, x.args)
                file_unit[] = false
                return nothing
            end
            if file_unit[] && ((x.head === :call && x.args[1] in _SCALING_OPERATORS && any(_is_decimal_scale, x.args[2:end])) ||
                    (x.head in _SCALING_UPDATES && length(x.args) == 2 && _is_decimal_scale(x.args[2])))
                flag("a file length scaled by $(first(filter(_is_decimal_scale, x.args[2:end])))")
            end
            if x.head === :call
                name = _unit_rule_callee(x.args[1])
                _constructor_name(x.args[1]) in _PHYSICAL_CONSTRUCTORS && any(_holds_ustrip, x.args[2:end]) &&
                    flag(string(_constructor_name(x.args[1]), "(… ustrip …)"))
                name in (:uparse, :qparse, :register_unit) && flag(string(name, "(…)"))
                _constructor_name(x.args[1]) in _REGISTRY_BUILDERS && flag(string(_constructor_name(x.args[1]), "(…)"))
                !in_test && name === :ubase && flag("ubase(…)")
                !in_test && name === :// && length(x.args) == 3 && (x.args[2], x.args[3]) in _RATIONAL_FACTORS &&
                    flag(string(x.args[2], "//", x.args[3]))
                name === :convert && length(x.args) >= 2 && _is_quantity_type(x.args[2]) && flag("convert(Quantity{…}, …)")
                name in (:Dict, :tuple) && _unit_name_count(x) >= 2 && flag("a table of unit names")
            elseif x.head in (:tuple, :vect)
                _unit_name_count(x) >= 2 && flag("a table of unit names")
            elseif x.head === :. && length(x.args) == 2 && _unit_rule_callee(x.args[1]) === :UnitRegistry && x.args[2] == QuoteNode(:UNITS)
                flag("UnitRegistry.UNITS")
            elseif x.head === :if && length(x.args) >= 2 && _is_isa_quantity(x.args[1]) &&
                    (_is_ustrip_call(x.args[2]) || (length(x.args) >= 3 && _is_ustrip_call(x.args[3])))
                flag("isa Quantity ? ustrip(…)")
            end
            foreach(walk, x.args)
        end
        return nothing
    end
    walk(ex)
    return out
end

# Rule 10 on what one Julia file binds beside a width's context: a letter bound
# next to a noise scale (one signature, struct or NamedTuple), a letter that is
# a field, type parameter or constructor keyword of a width-carrying type, and
# an abbreviation used as a dict (sidecar) key.
function _width_binding_rules!(out, f, ctx::_NamingContext, is_carrier)
    letter(b) = String(b[1]) in _WIDTH_LETTERS
    for group in ctx.groups
        any(b -> b[1] in _NOISE_SCALE_NAMES, group) || continue
        for b in group
            letter(b) && push!(out, NamingViolation(f, b[2], String(b[1]), :R10))
        end
    end
    for (t, members) in ctx.members
        is_carrier(t) || continue
        for b in members
            letter(b) && push!(out, NamingViolation(f, b[2], String(b[1]), :R10))
        end
    end
    for (k, line) in ctx.keys
        _is_width_abbreviation(k) && push!(out, NamingViolation(f, line, repr(k), :R10))
    end
    return out
end

# Rule 7 on one line of prose: a banned word, and a retired phrase.
function _prose_rules!(out, f, line, text)
    words = _naming_words(text)
    any(in(_BANNED_WORDS), words) && push!(out, NamingViolation(f, line, _clip(text), :R7))
    p = _retired_phrase(words)
    p === nothing || push!(out, NamingViolation(f, line, join(p, " ") * ": " * _clip(text), :R7))
    return out
end

_clip(s::AbstractString) = (t = strip(s); length(t) > 160 ? first(t, 157) * "…" : String(t))
