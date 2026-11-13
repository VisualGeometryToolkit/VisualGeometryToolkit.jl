# =============================================================================
# refine_camera_and_poses — a camera and one pose per view, from 3D → 2D
# correspondences, against the reprojection error
# =============================================================================
#
# The bundle of a calibration with a known target (or several rigid boards, each a pose: the board-pose block,
# owner decision 20, 2026-11-11): given a camera guess (its
# intrinsics and lens, a parametric `CameraModel`), a pose guess per view
# (world → camera, X_c = R·X + t) and per view the correspondences `X => u` of
# target points and the pixels they are detected at, minimize
#
#     Σ_views Σ_(X => u) ‖project(camera, pose(X)) − u‖²     (pixels)
#
# jointly over the camera's free parameters and every pose. A refine: the guess
# is required, nothing is estimated from scratch, the camera's family is kept.
#
# THE CHARTS. The camera block is VGC's `ParameterManifold` of the camera — its
# `parameter_vector` (f, aspect, skew, cx, cy, then the lens's coefficients),
# each moved by the chart that keeps it inside its `parameter_bounds` —
# restricted to the refined parameters (every parameter but the `fixed` ones and the skew by
# default), optionally with a solver box (`box`). Each pose is VGC's
# `LeftRigidManifold6` (x ↦ exp([δω]×)(R·x + t) + δt). Each chart is one
# NLLSsolver variable through VGC's NLLSsolver adapter, re-anchored at every
# accepted step; each correspondence is one residual block of 2 rows, depending
# on the camera and its view's pose (on the pose alone with nothing free in the
# camera), differentiated by NLLSsolver (ForwardDiff through the retractions and
# the lens). The driver is VGC's `nlls_options` (Marquardt's DᵀD damping); the
# stop is recorded by VGC's `nlls_termination`, under the stall postcondition on the rows and
# their Jacobian at the estimate.
#
# Hoisted from RadiometricCamera's experiments (`refine_lens_and_poses`,
# experiments/lib/lens_tools.jl: the same model and LM on its sensor-chart
# camera), 2026-10-28: the stack's pixel camera, its parameter chart and pose
# chart, one residual block per correspondence.
# =============================================================================

"""
    CameraPosesRefineMetadata

`FitResult` metadata of [`refine_camera_and_poses`](@ref): `residuals`, the
reprojection rows at the estimate (pixels, two per correspondence: views in
block order, each view's correspondences in their order); `s²`, the residual
variance `Σ‖rᵢ‖² / ν` (px²); `ν = 2n − p`, its degrees of freedom (`p` the
free camera parameters plus 6 per view); `free`, the names of the camera
parameters refined; and `spline`, `nothing` for a parametric camera, for a
SplineLens camera `(; β, λ, gcv, gcv_at_bound, effective_dof,
n_nodes_unrefined, n_dropped_window, n_dropped_unprojected, rounds,
windows_settled, forward, forward_max_px, cost_start, axis_rotation, symmetry_residual,
refined_axis_offset, refined_symmetry_residual)`:
the third-difference
penalty's normalized weight β and its `λ` (0 for none, and with `fixed = (:controls,)`;
`gcv_select`'s `β` and `λ`); the GCV curve `[(; β, λ, gcv)]` over the weights
evaluated (ascending; `nothing` unless `penalty = :gcv`), whether its minimum is
an end of the bracket (the weight GCV wants lies at or past it), and tr H, the
effective number of parameters at the chosen weight (`nothing` without GCV, and
when `max_time` cuts GCV's start solve short: then no weight is searched, and β is
that solve's, 1);
the present nodes no data row and no third-difference row reads (held at their
guesses); the correspondences detected on an absent window,
and those the camera does not project when detected within a cell of one (they
can be inliers); the solve-and-rebuild rounds (the blocks rebuilt at the
estimate until every window is its own) and whether the windows settled before
the cap (`false` when a solve was cut short by `max_time`: its windows were never
rebuilt); what became of a forward spline
(`:none` without one, `:refitted` or `:dropped`: the entry's turn of the frame
leaves it stale, so it is never kept) and the refit's held-out maximum (pixels, or `nothing`); and the joint objective
(½ the sum of squares, the penalty rows included) at the joint solve's start; then the entry's optical-axis gauge: the frame-labelled
rotation from the given camera's frame to the refined one's (`optical_axis`, the
poses turned by it) and the axis fit's symmetry residual (px); and the diagnostic of
the refined camera's own optical axis: its image's distance from the held principal
point (px) and its symmetry residual (px). For
a SplineLens camera `s²` reads the whole cost, the
third-difference rows included, and `ν = 2n + 3m − p` counts its `m`
third-difference rows as observations and `p = 3·(the nodes a row reads) − 4 +
6·views` (the gauge's 4 dimensions, the common scale and the rotation, are no
unknowns). Then the robust record: `residual_norms` (one per correspondence,
pixels), `weights` (the loss's ρ′/ρ′(0) at the estimate under a robust strategy, all
ones — lazily — under least squares), and the scale the loss ran at, `scale` and
`scale_source` (`ScaleGiven`/`ScaleUpstream`, or `NaN`/`ScaleNone` under least
squares and for a width given alone; [`noise_scale`](@extref VisualGeometryToolkitCore.noise_scale)
reads it) and `loss_width` (`NaN` under least squares). All forward as properties:
`result.residuals`, `result.spline`, ….
"""
struct CameraPosesRefineMetadata{L, W <: AbstractVector{Float64}}
    residuals::Vector{Float64}
    s²::Float64
    ν::Int
    free::Tuple{Vararg{Symbol}}
    spline::Union{Nothing, NamedTuple}
    residual_norms::Vector{Float64}
    weights::W
    scale::Float64
    scale_source::ScaleSource
    loss_width::Float64
end
CameraPosesRefineMetadata{L}(residuals, s², ν, free, spline, norms, weights::W, scale, source, loss_width) where {L, W} =
    CameraPosesRefineMetadata{L, W}(residuals, Float64(s²), Int(ν), free, spline, norms, weights, Float64(scale), source,
                                    Float64(loss_width))
# A least-squares refine: unit weights, no scale (ScaleNone) and no loss width; the norms from the rows.
CameraPosesRefineMetadata(residuals::Vector{Float64}, s²::Real, ν::Integer, free, spline) =
    CameraPosesRefineMetadata{Nothing}(residuals, s², ν, free, spline, _row_norms(residuals),
                                       unit_weights(Float64, length(residuals) ÷ 2), NaN, ScaleNone, NaN)
# The weights' kind is the strategy's: a robust loss's ρ′, or L2's unit weights.
VgtkCore.weight_kind(::Type{<:CameraPosesRefineMetadata{L}}) where {L} = VgtkCore.weight_kind(L)
# The norm of each correspondence's two residual rows (Core's `point_norms!`).
_row_norms(r) = (n = length(r) ÷ 2; point_norms!(Vector{Float64}(undef, n), r, n, 2))

# The cameras refined: a camera model, or a bounded one (refined as its model, returned bounded alike).
const _RefinedCamera = Union{CameraModel, BoundedCamera{<:CameraModel}}
_camera_model(c::CameraModel) = c
_camera_model(c::BoundedCamera) = model(c)

# One correspondence's residual block: a board point `X` (the board's frame) and its pixel `u`, projected through the
# camera, its view's pose (world → camera) and its board's pose (board → world): `project(camera, view(board(X))) − u`.
# `F = (camera, view, board)` says which of the three are variables of the block, in that order (`varind`); a held one
# is the value the block carries. A refine without boards is one held identity board (`X` in the world).
# A robust refine scales the block by `sqrt_weight` (√ρ′ at the estimate, for the stationarity and covariance rows;
# 1 otherwise) or robustifies its squared norm by `robust` (Triggs: NLLSsolver's robustifier, Core's
# `loss_robustifier`; `NoRobust()` otherwise). `M` is the camera chart's type: the problem's variable vector holds a
# Union, and `getvars` asserts each slot's concrete type (the camera
# chart, a pose chart), so the block's residual and its ForwardDiff Jacobian compile for concrete types instead of
# boxing every variable (the warm solve allocated ~29 KB a correspondence without it).
struct _BoardViewResidual{N, F, C, K, M} <: NLLSsolver.AbstractResidual
    varind::SVector{N, Int}
    camera::C                       # the held camera (read when F[1] is false)
    view::PlainEuclideanMap{3, Float64}    # the held view pose (when F[2] is false)
    board::PlainEuclideanMap{3, Float64}   # the held board pose (when F[3] is false)
    X::SVector{3, Float64}
    u::SVector{2, Float64}
    sqrt_weight::Float64
    robust::K
end
_BoardViewResidual{N, F, C}(varind, camera, view, board, X, u; chart::Type = Any) where {N, F, C} =
    _BoardViewResidual{N, F, C, NLLSsolver.NoRobust, chart}(varind, camera, view, board, X, u, 1.0, NLLSsolver.NoRobust())
NLLSsolver.robustkernel(r::_BoardViewResidual) = r.robust
_with_robust(r::_BoardViewResidual{N, F, C, <:Any, M}, sqrt_weight::Float64, robust::K) where {N, F, C, K, M} =
    _BoardViewResidual{N, F, C, K, M}(r.varind, r.camera, r.view, r.board, r.X, r.u, sqrt_weight, robust)
NLLSsolver.ndeps(::_BoardViewResidual{N}) where {N} = NLLSsolver.static(N)
NLLSsolver.nres(::_BoardViewResidual) = NLLSsolver.static(2)
NLLSsolver.varindices(r::_BoardViewResidual) = r.varind
@inline function NLLSsolver.getvars(r::_BoardViewResidual{N, F, C, K, M}, vars::Vector) where {N, F, C, K, M}
    i = 1
    cam = F[1] ? (vars[r.varind[i]]::M,) : ()
    i += F[1]
    view = F[2] ? (vars[r.varind[i]]::_CameraPosePose,) : ()
    i += F[2]
    board = F[3] ? (vars[r.varind[i]]::_CameraPosePose,) : ()
    return (cam..., view..., board...)
end
Base.eltype(::_BoardViewResidual) = Float64
function NLLSsolver.computeresidual(r::_BoardViewResidual{N, F}, vs...) where {N, F}
    i = 1
    camera = F[1] ? model(vs[i]) : r.camera
    i += F[1]
    view = F[2] ? model(vs[i]) : r.view
    i += F[2]
    board = F[3] ? model(vs[i]) : r.board
    e = VgtkCore.reprojection_error(camera, view, board(r.X), r.u)
    return r.sqrt_weight == 1.0 ? e : r.sqrt_weight * e
end
# The block's cost as the LM reads it: non-finite (a candidate update that moves a
# point off the lens's domain) is +Inf (`finite_cost`).
NLLSsolver.computecost(vars::AbstractVector, r::_BoardViewResidual) =
    finite_cost(NLLSsolver.computecost(r, NLLSsolver.getvars(r, vars)...))

"""
    refine_camera_and_poses(camera, poses, views; fixed = (), box = nothing, optimization_strategy, scale | width,
                            max_iter, ftol, xtol, max_time = nothing)   # defaults: configs/default.toml [refine_camera_and_poses]
    refine_camera_and_poses(spline_camera, poses, views; fixed = (), penalty = :gcv,
                            refit_forward = true, forward_tolerance, max_iter, ftol, xtol, max_time = nothing)
        -> FitResult{<:NamedTuple{(:camera, :poses)}, CameraPosesRefineMetadata}
    refine_camera_and_poses(camera, poses, board_poses, views; held_boards, held_views = (), fixed = (), box,
                            optimization_strategy, scale | width, max_iter, ftol, xtol, max_time = nothing)
        -> FitResult{<:NamedTuple{(:camera, :poses, :board_poses)}, CameraPosesRefineMetadata}

Refine a camera and one pose per view jointly against the reprojection error of
3D → 2D correspondences: minimize `Σ ‖project(camera, pose_v(X)) − u‖²` over
the camera's free parameters and every view's pose, from the guesses `camera`
and `poses`. A refine: both guesses are required, nothing is re-estimated from
scratch and the camera keeps its family (its lens type).

# Arguments
- `camera` — the camera guess, a `CameraModel` or a `BoundedCamera` of one
  (the estimate is then bounded alike): intrinsics and lens, the parameters
  VisualGeometryToolkitCore's `parameter_vector(camera)` lists (`f, aspect,
  skew, cx, cy`, then the lens's coefficients), in pixels. A SplineLens camera
  takes its own route (below).
- `poses` — one world → camera `EuclideanMap{3}` per view (`X_c = R·X + t`, `X`
  in the target's frame), keyed like `views`.
- `views` — per view, its correspondences `cs = [X => u, …]`: `X` a target
  point (3-vector), `u` the pixel it is detected at (2-vector). Grouped data:
  an `AbstractDict{K, <:AbstractVector{<:Cspond}}` keyed by the view, with
  `poses::AbstractDict{K}` of the same keys (block order: the sorted keys; an
  `ArgumentError` when the key sets differ), or a vector of views with a vector
  of poses of the same length. Every point must project at the guess (in front
  of the camera, inside the lens's domain): an `ArgumentError` names the first
  that does not.

# Keywords (configuration)
- `fixed` — the camera parameters HELD, by name or by group (`:k` for
  `k1…`; VisualGeometryToolkitCore's `free_parameter_names` resolves it), `()` the
  default: every parameter is refined but the gauge, the skew, which is held by
  construction (naming it is accepted); naming every parameter holds the camera
  and refines the poses alone. The
  stack's one parameter-holding keyword (NLLSsolver's word; OpenCV's
  `CALIB_FIX_*`): `free`, `fix`, `fixed_params` and `hold` are refused, naming it.
- `box` — a solver search box on the camera's parameters by name
  (VisualGeometryToolkitCore's `ParameterManifold(camera; box)`), or `nothing`;
  with every parameter `fixed` (nothing to bound) a box is an `ArgumentError`.
- `optimization_strategy` — the refine contract's strategy, as every
  `refine_*` takes it: `Triggs{Shape}` (the default, `Triggs{WelschShape}`,
  `configs/default.toml [refine_camera_and_poses]`), one robustified LM
  (Core's `loss_robustifier`), starting directly from the caller's start
  (`camera`, `poses`, board poses): it runs no least-squares stage, so gross
  outliers never pull a warm-up fit; or `nothing`, plain least squares (it
  reads no scale). TOML: `{ type = "triggs", loss = "welsch" }`. An
  `Irls{Shape}` is an `ArgumentError` naming `Triggs{Shape}` (a refine runs one
  LM, no reweighting cycles); any other strategy is one too (it needs a RANSAC
  consensus set). (The keyword was `robust`,
  then `local_optimization`; both are refused, an `ArgumentError` naming this
  one.)
- `scale` or `width` — EXACTLY ONE for a robust strategy (so for the default),
  neither for least
  squares (`ArgumentError` otherwise; Core's `scale_input`): `scale = σ`, the
  detections' noise scale (pixels, per coordinate), or an upstream result whose
  [`noise_scale`](@extref VisualGeometryToolkitCore.noise_scale) it is; the loss
  width is then the strategy's efficiency constant times σ (Core's
  `efficiency_constant(Shape, 2)` unless the strategy states one). `width = w`:
  the loss width itself (pixels). A refine estimates no scale; the metadata
  records the one it ran with (`scale`, `scale_source`, `loss_width`) and the
  weights ρ′/ρ′(0) at the estimate.
- `max_iter` — the LM's cap on ACCEPTED iterations (NLLSsolver's count);
  `ftol` — its relative cost decrease and the stall postcondition's exact-fit
  test; `xtol` — its step tolerance (chart units); `max_time` — its
  wall-clock cap in seconds (`nothing` or `Inf`: none), for a camera whose
  projection makes an iteration slow; it stops with `:max_time`.

The estimate is `(; camera, poses)`: the refined camera, and the refined poses
keyed like the input (a `Dict` by the same keys, or a vector in view order,
each with its input's frame labels). Its termination record (`retcode`, `stop_rule`) is the last solve's,
VisualGeometryToolkitCore's `nlls_termination`: `ReturnCode.Success` only at a stationary point, else `MaxIters`
(the iteration cap), `MaxTime`, `Failure`/`InitialFailure` (`NonFinite`) or
`Stalled`; `rms` is
`√(Σ‖rᵢ‖² / n)` over the `n` correspondences (pixels). The metadata
([`CameraPosesRefineMetadata`](@ref)) carries the residual rows, `s²`, `ν` and
the free parameter names. Plain least squares: the correspondences must be
inliers.

# Several boards
With `board_poses`, the target is several rigid boards, each with its own pose:
`poses` maps each view to its world → camera pose, `board_poses` each board to
its board → world pose (both `AbstractDict`s of plain `EuclideanMap{3}`), and
`views` maps `(view, board)` to the correspondences `X => u` of that board's
points (`X` in the board's frame) detected in that view. The residual is
`project(camera, pose_v(board_b(X))) − u`; each board pose is one more pose
chart of the same NLLSsolver problem, beside the camera's chart and the view
poses (the same assembler and LM, the same retcode and metadata; `ν` counts 6
per free board too). A pose the data never reach is an `ArgumentError`, and so
is a correspondence of a view or board without a pose.

The gauge: a common rigid motion of every board, undone by every view, leaves
the cost unchanged, once per connected component of the view–board graph
(views and boards linked by their correspondences). It is fixed EXACTLY, by
holding poses at their guesses (no pin, no prior): `held_boards` (by default
each component's first board, in sorted key order) and `held_views` (none by
default) are held, every other pose is a variable, and a component without a
held pose is an `ArgumentError`. Holding more than one pose of a component
constrains the solve (a held pose is known). The estimate's `board_poses`
returns a held pose as given. A parametric camera only (a SplineLens camera's
gauge is its own: an `ArgumentError`); plain poses and points.

# A SplineLens camera
Its K is a frozen gauge (skew 0) and its parameters are its backward lattice's
controls at the present nodes, so its route differs (Core's `IntrinsicsRole`
trait, `GaugeIntrinsics`, selects it):
- The variables are the present nodes' controls, any norms (a `fit_camera`
  lattice's, a quasi-interpolant's: they are kept as they are, as normalizing
  them would change the backprojection), each `c = e^ℓ·D` its direction on its
  own sphere (the reference's chart, Core's `ControlPatchManifold`) and its
  log-scale `ℓ` (3 coordinates; the controls grouped into 2 × 2 patches,
  NLLSsolver's 10-variable block cap), and the poses; only the nodes a row
  reads are variables, the others kept exactly as they are
  (`result.spline.n_nodes_unrefined` counts them).
  The solve runs in three stages,
  each from the last: the poses alone through the guess's lens, then the
  directions and the poses with every scale held exactly (the scales are no
  variable of that stage), then everything (a node's
  scale is weakly observed and nearly collinear with its direction, so a joint
  solve from far off creeps, or lets the scales absorb the poses' error). `fixed` is `()` (the default: the
  controls refined) or `(:controls,)` (the poses alone, the camera's lens held as the entry gauged it); a `box` is an
  `ArgumentError`.
- Each correspondence's rows are `K·x* − u`, `x*` the lens's projection solve
  on the correspondence's 4 × 4 window at the guess, its Jacobian analytic by
  implicit differentiation at `x*` (no automatic differentiation reaches the
  solve). With the camera held (`fixed = (:controls,)`, and the poses-alone
  stage) there is no window: each row projects through the whole camera
  (`project`), a constant. A correspondence detected on a window with an absent node is
  dropped and counted (`result.spline.n_dropped_window`), and so is one the
  camera does not project when detected within one knot cell of such a window
  (it images it onto the absent nodes; counted apart,
  `result.spline.n_dropped_unprojected`: it can be an inlier); every other one
  must project at the guess (and at each rebuild's estimate: an
  `ArgumentError` names the round). A third-difference penalty (Core's
  `difference_rows`, the P-spline fit's operator, along every row and column
  of present nodes) holds the nodes the data reach weakly or not at all. Each
  of its rows is divided by the mean control length of its own 4 nodes, so it
  is scale-free. Its weight is the stack's normalized β (`penalty_scale`):
  `λ = β · energy / N / m̄`, the data's energy per parameter read as the mean
  diagonal of its Gauss–Newton matrix in the nodes' direction coordinates
  times the controls' squared mean length (scale-free too; the residuals of
  exact data vanish, so they cannot set it), and `m̄` the mean positive
  eigenvalue of the Δ³ operator's `DᵀD`. β = 1 weighs the penalty like the
  data. β is set by `penalty`:
  - `:gcv` (the default): β minimizes the generalized cross-validation score
    `GCV = n‖r‖² / (n − tr H)²` of the refine's linearization (`n` the data
    rows): the whole refine solved ONCE at β = 1 (the bracket's top, where
    the penalty weighs like the data and no weakly held node swings), its
    Gauss–Newton step from there for every λ (`r` the data rows' residuals
    after the step, `H = J(JᵀJ + λDᵀD)⁻¹Jᵀ`, J and D the data and Δ³ rows'
    Jacobians there, over the variables' coordinates), from ONE
    generalized eigendecomposition of the pencil `(DᵀD, JᵀJ + DᵀD)` up
    to 3000 coordinates, else per λ a sparse Cholesky and a 64-probe
    Hutchinson trace from a fixed seed (deterministic; standard deviation
    ≤ √(2·tr H / 64)); minimized over `log₁₀ β ∈ [−12, 0]` by the stack's
    `gcv_select` (one β a decade, then Brent's method to its own floor;
    a minimum at a bracket end is reported, `gcv_at_bound`); then ONE final
    solve at the chosen β, from the linearization's Gauss–Newton step there
    (from the β = 1 estimate if that step stops projecting a correspondence).
  - a number β > 0: that weight; no GCV. On noisy detections a small weight
    can let the nodes the data hold weakly swing until a correspondence no
    longer projects: an `ArgumentError` says so and suggests `:gcv` or a
    larger weight.
  - `0` or `nothing`: no smoothness. Then every present node outside every
    correspondence's window is kept as it is (and counted). Without the
    Δ³ rows each node's log-scale is held by its data alone, nearly collinear
    with its own direction, so the joint solve needs ≈ 2–3× the default
    `max_iter` (≈ 210 iterations on the 3-view fixture); at the default cap it
    returns `:max_iterations`.
  The blocks are rebuilt at the estimate and solved again until every
  correspondence reads its own window (`result.spline.rounds`, at most 5; at
  the cap the rows, `rms` and the stationarity test are read on the returned
  camera's own windows, and `windows_settled` is `false`). `iterations` sums
  every solve.
- First, always, the optical axis. The camera is turned so ẑ is its optical axis and its principal point is that
  axis's pixel, and the guess poses are turned with it, so every image is kept:
  [`optical_axis`](@ref), then Core's `axis_gauged`. The axis is fitted over the camera's own bound, so the camera
  must be a `BoundedCamera`; a bare model is an `ArgumentError`. The refine runs from there. `result.spline.axis_rotation` is the
  turn, from the given camera's frame to the refined one's. The refined K is the gauged guess's.
- The gauge: a rotation of every control together with every pose, and one
  common scale of every control, leave the cost unchanged. The solve fixes it
  exactly by the parametrization: the first view's rotation and the log-scale of
  the node nearest x = 0 are no coordinate of any variable, so the LM's normal
  matrix is full rank. It holds the frame and K the entry set, and nothing moves
  the estimate after the LM. K is bit-identical to the axis-gauged guess's.
- A diagnostic only: `optical_axis` of the refined camera. `result.spline.refined_axis_offset`
  is how far its axis's image lies from the held principal point (px), and
  `result.spline.refined_symmetry_residual` is its symmetry residual. Nothing is
  re-gauged from it.
- A forward spline is dropped for the refinement and, with `refit_forward`
  (the default), refitted over `imaged_region(camera)` by
  `fit_camera(SplineLens, …; tolerance = forward_tolerance)` (pixels), which
  needs a `BoundedCamera` (an `ArgumentError` for a bare camera model). It is
  never kept, even with `fixed = (:controls,)`: the entry's turn of the frame leaves it stale.
- The model: the target is exact, so the estimate is the maximum-likelihood
  one under isotropic Gaussian detection noise (penalized by the
  smoothness prior; with `penalty = 0` the plain maximum-likelihood one). A board that departs from its
  design (an in-plane warp, an affine or smooth deformation, a height field)
  is not modelled here: that is the board-deformation solve of
  RadiometricCamera's calibration pipeline.
- `max_time` caps the whole refine (every stage and rebuild round reads what
  is left of it; the record `(ReturnCode.MaxTime, TimeCap)` when any was cut short).
- Device: the projections that build each problem run batched on the device
  (Core's `default_device()`); the LM is NLLSsolver's, on the CPU.

# Units
Plain inputs give plain poses; physical ones (unitful poses and target points,
all one form) are read in metres at entry and the poses come back physical, in
metres. The residuals are pixels either way.
"""
function refine_camera_and_poses(
        camera::_RefinedCamera, poses::AbstractDict{K, <:PlainEuclideanMap{3}},
        views::AbstractDict{K, <:AbstractVector{<:Cspond{<:StaticVector{3, <:Real}, <:StaticVector{2}}}};
        kwargs...
    ) where {K}
    ids = block_order(views, poses)
    est, rest... = _refine_camera_and_poses(camera, [poses[k] for k in ids], [views[k] for k in ids]; kwargs...)
    return _camera_poses_result((; camera = est.camera, poses = Dict{K, eltype(est.poses)}(zip(ids, est.poses))), rest...)
end
function refine_camera_and_poses(
        camera::_RefinedCamera, poses::AbstractVector{<:PlainEuclideanMap{3}},
        views::AbstractVector{<:AbstractVector{<:Cspond{<:StaticVector{3, <:Real}, <:StaticVector{2}}}};
        kwargs...
    )
    length(poses) == length(views) || throw(ArgumentError("refine_camera_and_poses: $(length(poses)) poses for $(length(views)) views"))
    return _camera_poses_result(_refine_camera_and_poses(camera, collect(poses), collect(views); kwargs...)...)
end

# Several boards: the view poses (world → camera) and the board poses (board → world) keyed, the correspondences keyed
# by (view, board). Each connected component of the view–board graph has a held pose: its gauge.
function refine_camera_and_poses(
        camera::_RefinedCamera, poses::AbstractDict{V, <:PlainEuclideanMap{3}}, board_poses::AbstractDict{B, <:PlainEuclideanMap{3}},
        views::AbstractDict{<:Tuple{V, B}, <:AbstractVector{<:Cspond{<:StaticVector{3, <:Real}, <:StaticVector{2}}}};
        held_boards = nothing, held_views = (), kwargs...
    ) where {V, B}
    VgtkCore.Cameras.IntrinsicsRole(_camera_model(camera).lens) isa VgtkCore.Cameras.FreeIntrinsics || throw(ArgumentError(
        "refine_camera_and_poses: board poses refine a parametric camera; a SplineLens camera's gauge is its own (refine it with one board)"))
    vids = sort!(collect(keys(poses))); bids = sort!(collect(keys(board_poses)))
    keys_ = sort!(collect(keys(views)))
    isempty(keys_) && throw(ArgumentError("refine_camera_and_poses: no views"))
    vi = Dict(k => i for (i, k) in enumerate(vids)); bi = Dict(k => i for (i, k) in enumerate(bids))
    for (v, b) in keys_
        haskey(vi, v) || throw(ArgumentError("refine_camera_and_poses: the correspondences of ($(repr(v)), $(repr(b))) name a view with no pose"))
        haskey(bi, b) || throw(ArgumentError("refine_camera_and_poses: the correspondences of ($(repr(v)), $(repr(b))) name a board with no pose"))
    end
    seen_v = Set(first.(keys_)); seen_b = Set(last.(keys_))
    for v in vids
        v in seen_v || throw(ArgumentError("refine_camera_and_poses: view $(repr(v)) has a pose and no correspondence"))
    end
    for b in bids
        b in seen_b || throw(ArgumentError("refine_camera_and_poses: board $(repr(b)) has a pose and no correspondence"))
    end
    # the connected components of the view–board graph (views 1:nv, boards nv+1:nv+nb), by union–find
    nv, nb = length(vids), length(bids)
    parent = collect(1:(nv + nb))
    root(i) = parent[i] == i ? i : (parent[i] = root(parent[i]))
    for (v, b) in keys_
        parent[root(vi[v])] = root(nv + bi[b])
    end
    held_b = falses(nb); held_v = falses(nv)
    if held_boards === nothing                                   # each component's first board
        firsts = Dict{Int, Int}()
        for b in 1:nb
            get!(firsts, root(nv + b), b)
        end
        foreach(b -> held_b[b] = true, values(firsts))
    else
        for b in held_boards
            haskey(bi, b) || throw(ArgumentError("refine_camera_and_poses: held board $(repr(b)) has no pose"))
            held_b[bi[b]] = true
        end
    end
    for v in held_views
        haskey(vi, v) || throw(ArgumentError("refine_camera_and_poses: held view $(repr(v)) has no pose"))
        held_v[vi[v]] = true
    end
    for c in unique(root(i) for i in 1:(nv + nb))
        any(held_v[i] for i in 1:nv if root(i) == c) || any(held_b[b] for b in 1:nb if root(nv + b) == c) || throw(ArgumentError(
            "refine_camera_and_poses: the views $(repr([vids[i] for i in 1:nv if root(i) == c])) and boards " *
            "$(repr([bids[b] for b in 1:nb if root(nv + b) == c])) hold no pose: their common motion is free (hold a board or a view)"))
    end
    obs = [(vi[v], bi[b], _correspondence(vi[v], j, c)...) for (v, b) in keys_ for (j, c) in enumerate(views[(v, b)])]
    est, rest... = _refine_bundle(camera, [poses[k] for k in vids], [board_poses[k] for k in bids], obs, held_v, held_b; kwargs...)
    return _camera_poses_result((; camera = est.camera, poses = Dict{V, eltype(est.poses)}(zip(vids, est.poses)),
                                 board_poses = Dict{B, eltype(est.boards)}(zip(bids, est.boards))), rest...)
end

# Physical inputs: the poses and the target points in metres once, the plain
# refinement, the poses read back in metres once.
function refine_camera_and_poses(
        camera::_RefinedCamera, poses::AbstractDict{K, <:_PhysicalPose3},
        views::AbstractDict{K, <:AbstractVector{<:Cspond{<:_PhysicalPoint3, <:StaticVector{2}}}}; kwargs...
    ) where {K}
    r = refine_camera_and_poses(
        camera, Dict(k => _plain_metres(p) for (k, p) in poses), Dict(k => _map_csponds(_plain_metres, identity, cs) for (k, cs) in views); kwargs...
    )
    est = estimate(r)
    return _camera_poses_result((; camera = est.camera, poses = Dict(k => _attach_metres(p) for (k, p) in est.poses)), retcode(r), stop_rule(r), iterations(r), rms(r), r.n, metadata(r))
end
function refine_camera_and_poses(
        camera::_RefinedCamera, poses::AbstractVector{<:_PhysicalPose3},
        views::AbstractVector{<:AbstractVector{<:Cspond{<:_PhysicalPoint3, <:StaticVector{2}}}}; kwargs...
    )
    r = refine_camera_and_poses(camera, map(_plain_metres, poses), [_map_csponds(_plain_metres, identity, cs) for cs in views]; kwargs...)
    est = estimate(r)
    return _camera_poses_result((; camera = est.camera, poses = map(_attach_metres, est.poses)), retcode(r), stop_rule(r), iterations(r), rms(r), r.n, metadata(r))
end

_camera_poses_result(est, code, rule, iters, rms, n, meta) = FitResult(est, code, rule, iters, rms, n, meta)

# The route, by Core's `IntrinsicsRole` trait: a free K (a parametric lens) below; a gauge K
# (a SplineLens) in refine_spline_camera_and_poses.jl.
_refine_camera_and_poses(camera::_RefinedCamera, poses::AbstractVector, views::AbstractVector; kwargs...) =
    _refine_camera_and_poses(VgtkCore.Cameras.IntrinsicsRole(_camera_model(camera).lens), camera, poses, views; kwargs...)

const _CameraPosePose = LeftRigidManifold6{Float64}

# The refine's input contract, both routes: the poses as pose charts (VGC's
# `LeftRigidManifold6`), a correspondence `X => u` as finite static vectors (an
# `ArgumentError` naming view `v`, correspondence `j` otherwise), and a pose chart's
# model as a plain map.
_pose_charts(poses) = _CameraPosePose[LeftRigidManifold6(SMatrix{3, 3, Float64, 9}(rotation(p)), SVector{3, Float64}(translation(p))) for p in poses]
function _correspondence(v::Integer, j::Integer, c)
    X = SVector{3, Float64}(first(c)[1], first(c)[2], first(c)[3]); u = SVector{2, Float64}(last(c)[1], last(c)[2])
    (all(isfinite, X) && all(isfinite, u)) || throw(ArgumentError("refine_camera_and_poses: view $v, correspondence $j: non-finite point or pixel"))
    return X, u
end
_finite_pose(P::PlainEuclideanMap{3}) = all(isfinite, rotation(P)) && all(isfinite, translation(P))
_plain_pose(P::PlainEuclideanMap{3}) = EuclideanMap(RotMatrix{3, Float64}(rotation(P)), SVector{3, Float64}(translation(P)))
# each refined pose under its input's frame labels
_relabelled(inputs::AbstractVector, outs::AbstractVector{<:PlainEuclideanMap{3}}) =
    [relabel(VgtkCore.map_frames(p0)..., _plain_pose(P)) for (p0, P) in zip(inputs, outs)]

# Without boards: every correspondence's point in the world, one held identity board.
function _refine_camera_and_poses(
        ::VgtkCore.Cameras.FreeIntrinsics, bounded::_RefinedCamera, poses::AbstractVector, views::AbstractVector; kwargs...
    )
    V = length(views)
    V >= 1 || throw(ArgumentError("refine_camera_and_poses: no views"))
    obs = [(v, 1, _correspondence(v, j, c)...) for (v, cs) in enumerate(views) for (j, c) in enumerate(cs)]
    world = EuclideanMap(RotMatrix{3, Float64}(one(SMatrix{3, 3, Float64, 9})), zero(SVector{3, Float64}))
    est, rest... = _refine_bundle(bounded, poses, [world], obs, falses(V), trues(1); kwargs...)
    return (; camera = est.camera, poses = est.poses), rest...
end


# The refine's keywords, shared by every parametrization: parsed once (the retired spellings refused; the strategy, its
# scale input and its loss: `bundle_loss`).
function _bundle_keywords(;
        fixed::Tuple{Vararg{Symbol}} = (),
        free = RetiredKeyword(), fix = RetiredKeyword(), fixed_params = RetiredKeyword(), hold = RetiredKeyword(),   # refused, naming `fixed`
        box::Union{Nothing, NamedTuple} = nothing,
        max_iter::Integer = _dflt("refine_camera_and_poses", "max_iter"),
        ftol::Real = _dflt("refine_camera_and_poses", "ftol"),
        xtol::Real = _dflt("refine_camera_and_poses", "xtol"),
        max_time::Union{Nothing, Real} = nothing,
        optimization_strategy = _refine_default(Val(:refine_camera_and_poses)),
        scale = nothing, width = nothing,
        robust = RetiredKeyword(),
        local_optimization = RetiredKeyword(),   # refused, naming its successor
    )
    refuse_retired_keyword(robust, :robust, :optimization_strategy)
    refuse_retired_keyword(local_optimization, :local_optimization, :optimization_strategy)
    _refuse_held_spellings(free, fix, fixed_params, hold)
    lo, input, loss = bundle_loss(optimization_strategy; scale, width, caller = "refine_camera_and_poses")
    return (; fixed, box, max_iter, ftol, xtol, max_time, lo, input, loss)
end

# The camera's chart: every parameter refined but the `fixed` ones (names or groups: `:k` holds k1…) and the gauge (the
# skew, held by construction): the stack's one expansion; `nothing` when every parameter is held.
function _camera_chart(camera, fixed, box)
    free = VgtkCore.free_parameter_names(parameter_names(camera), fixed)
    isempty(free) && box !== nothing &&
        throw(ArgumentError("refine_camera_and_poses: a `box` bounds the camera's refined parameters, and `fixed` holds every one"))
    return free, isempty(free) ? nothing : restrict(ParameterManifold(camera; box), free)
end

# The one bundle: a camera, view poses (world → camera) and board poses (board → world), each a variable unless held
# (`held_views`, `held_boards`), against the reprojection error of `obs`, `(view, board, X, u)` each. One NLLSsolver
# problem: the camera's chart (when anything is free), a pose chart per free view and per free board, one residual
# block per correspondence; solved by `_solve_bundle`.
function _refine_bundle(
        bounded::_RefinedCamera, poses::AbstractVector, boards::AbstractVector, obs::AbstractVector,
        held_views::AbstractVector{Bool}, held_boards::AbstractVector{Bool}; kwargs...
    )
    kw = _bundle_keywords(; kwargs...)
    camera = _camera_model(bounded)
    V, B = length(poses), length(boards)
    free, chc = _camera_chart(camera, kw.fixed, kw.box)
    nd = chc === nothing ? 0 : tangent_size(chc)
    view_charts = _pose_charts(poses); board_charts = _pose_charts(boards)
    # the variables: the camera's chart, then the free views', then the free boards'
    o = chc === nothing ? 0 : 1
    view_var = zeros(Int, V); board_var = zeros(Int, B)
    for v in 1:V
        held_views[v] || (o += 1; view_var[v] = o)
    end
    for b in 1:B
        held_boards[b] || (o += 1; board_var[b] = o)
    end
    plain(ch) = model(ch)::PlainEuclideanMap{3, Float64}
    blocks = NLLSsolver.AbstractResidual[]
    for (k, (v, b, X, u)) in enumerate(obs)
        all(isfinite, VgtkCore.reprojection_error(camera, plain(view_charts[v]), plain(board_charts[b])(X), u)) || throw(
            ArgumentError("refine_camera_and_poses: view $v, correspondence $k: the point $(X) does not project at the guess (behind the camera or outside the lens's domain)")
        )
        F = (chc !== nothing, !held_views[v], !held_boards[b])
        ind = SVector{count(F), Int}(Tuple(i for (i, f) in zip((1, view_var[v], board_var[b]), F) if f))
        push!(blocks, _BoardViewResidual{count(F), F, typeof(camera)}(ind, camera, plain(view_charts[v]), plain(board_charts[b]), X, u;
                                                                     chart = chc === nothing ? Nothing : typeof(chc)))
    end
    n = length(obs)
    p = nd + 6 * (count(!, held_views) + count(!, held_boards))
    2n > p || throw(ArgumentError("refine_camera_and_poses: underdetermined: $n correspondences ($(2n) rows) for $p unknowns"))
    o > 0 || throw(ArgumentError("refine_camera_and_poses: nothing to refine (the camera, every view and every board held)"))

    VT = chc === nothing ? _CameraPosePose : Union{typeof(chc), _CameraPosePose}
    vars0 = VT[]
    chc === nothing || push!(vars0, chc)
    foreach(v -> held_views[v] || push!(vars0, view_charts[v]), 1:V)
    foreach(b -> held_boards[b] || push!(vars0, board_charts[b]), 1:B)
    sol = _solve_bundle(blocks, vars0, p, kw)
    variables = sol.variables
    camera_out = chc === nothing ? camera : model(variables[1])
    pose_of(charts, var, i) = var[i] == 0 ? plain(charts[i]) : plain(variables[var[i]]::_CameraPosePose)
    poses_out = PlainEuclideanMap{3, Float64}[pose_of(view_charts, view_var, v) for v in 1:V]
    boards_out = PlainEuclideanMap{3, Float64}[pose_of(board_charts, board_var, b) for b in 1:B]
    all(_finite_pose, poses_out) && all(_finite_pose, boards_out) && all(isfinite, parameter_vector(camera_out)) ||
        throw(ErrorException("refine_camera_and_poses: the LM step is not finite"))
    camera_out = camera_out === camera ? bounded : VgtkCore.with_model(bounded, camera_out)
    return (; camera = camera_out, poses = _relabelled(poses, poses_out), boards = _relabelled(boards, boards_out)), sol.code, sol.rule,
        sol.iterations, sol.rms, sol.n, _bundle_metadata(sol, free, kw)
end

# The shared solve of every parametrization: the residual blocks `blocks` (2 rows each, one per correspondence, each
# answering `_with_robust`) on the variables `vars0` (`p` unknowns), under the parsed keywords `kw`. Least squares is
# the whole solve when `optimization_strategy = nothing`. A robust refine has NO least-squares stage (the owner's
# ruling, 2026-11-11): the strategy's solve starts from the caller's start, its loss at its width over the caller's
# frozen `scale` or at its absolute width: Triggs, ONE robustified LM (Core's `loss_robustifier`), the refine
# contract's one fixed-scale robust path (an Irls is refused by `bundle_loss`). A refine estimates no scale. Returns the variables at the estimate, the termination record (`code`, `rule`), the iterations, the reprojection rows,
# their sum of squares, ν, the weights, and `n` the weights' support and `rms` the weighted one (the stack's one rule,
# loop.jl: under least squares every correspondence and √(S/n)).
function _solve_bundle(blocks::AbstractVector, vars0::AbstractVector, p::Integer, kw)
    lo, loss, ftol = kw.lo, kw.loss, kw.ftol
    RT = Union{unique(typeof.(blocks))...}
    costs = convert(Vector{RT}, blocks)
    n = length(costs)
    VT = eltype(vars0)
    options = nlls_options(; max_iter = kw.max_iter, ftol, xtol = kw.xtol, max_time = kw.max_time)
    function solve(blocks, start)
        RTb = Union{unique(typeof.(blocks))...}
        pr = NLLSsolver.NLLSProblem(VT, RTb)
        foreach(x -> NLLSsolver.addvariable!(pr, x), start)
        foreach(c -> NLLSsolver.addcost!(pr, c), blocks)
        return pr, NLLSsolver.optimize!(pr, options)
    end
    stationarity_blocks = costs
    if lo === nothing
        problem, result = solve(costs, vars0)
        weights = unit_weights(Float64, n)
    else
        shape = strategy_loss(lo)(); sq = _sq_loss_width(loss)
        sqnorms(vars) = (r = first(nlls_rows_and_jacobian(costs, vars)); [r[2k - 1]^2 + r[2k]^2 for k in 1:n])
        problem, result = solve([_with_robust(c, 1.0, loss_robustifier(shape, sqrt(sq))) for c in costs], copy(vars0))
        weights = _loss_weights(shape, sq, sqnorms(problem.variables))   # ρ′/ρ′(0) (exactly as the stationarity rows read it)
        # stationarity of the robust cost: Σ ρ′ Jᵀr = 0, the rows and Jacobian scaled by √ρ′
        stationarity_blocks = [_with_robust(c, sqrt(weights[k]), NLLSsolver.NoRobust()) for (k, c) in enumerate(costs)]
    end
    r, J = nlls_rows_and_jacobian(stationarity_blocks, problem.variables)
    # the termination record: NLLSsolver's bits under the stall postcondition at the returned point
    code, rule = nlls_termination(result, () -> VgtkCore.is_stationary(J, r; tol = _CHART_LM_STATIONARITY_TOL, ftol))
    lo === nothing || (r = first(nlls_rows_and_jacobian(costs, problem.variables)))   # the reprojection rows themselves
    S = sum(abs2, r)
    support, rms = _weighted_support_rms(weights, (r[2k - 1]^2 + r[2k]^2 for k in 1:n))
    return (; variables = problem.variables, code, rule, iterations = result.niterations, residuals = r, S, ν = 2n - p, rms, n = support, weights)
end
_bundle_metadata(sol, free, kw; spline = nothing) = CameraPosesRefineMetadata{typeof(kw.lo)}(sol.residuals, sol.S / sol.ν, sol.ν, free, spline,
    _row_norms(sol.residuals), sol.weights, _recorded_scale(kw.input), VgtkCore.scale_source(kw.input),
    sqrt(_sq_loss_width(kw.loss)))

