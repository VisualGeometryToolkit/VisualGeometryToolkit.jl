# =============================================================================
# CameraPosesProblem — a camera and one pose per view, from 3D → 2D
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
# the lens). The driver is Core's one multi-block LM (the block contract of its `LevenbergMarquardt` refine, Marquardt's DᵀD damping, its
# stop settings `LmStops`); the stop is recorded under its one success rule (Core 0.11.0: a tolerance stop is converged;
# the rejection cap is judged by the stall postcondition on the rows and their Jacobian at the estimate).
#
# Hoisted from RadiometricCamera's experiments (`refine_lens_and_poses`,
# experiments/lib/lens_tools.jl: the same model and LM on its sensor-chart
# camera), 2026-10-28: the stack's pixel camera, its parameter chart and pose
# chart, one residual block per correspondence.
# =============================================================================

"""
    CameraPosesRefineMetadata

`FitResult` metadata of [`CameraPosesProblem`](@ref): `residuals`, the
reprojection rows at the estimate (pixels, two per correspondence: views in
block order, each view's correspondences in their order); `s2` (read by `s2(result)`), the residual
variance `Σ‖rᵢ‖² / dof` (px²); `dof = 2n − p`, its degrees of freedom (`dof(result)`) (`p` the
free camera parameters plus 6 per view); `free`, the names of the camera
parameters refined (`()` for a SplineLens camera, which is held); and `spline`,
`nothing` for a parametric camera, for a SplineLens camera `(; n_dropped_window,
n_dropped_unprojected)`: the correspondences detected on an absent window, and those
the camera does not project when detected within a cell of one (they can be
inliers).
For a SplineLens camera `p = 6·views` (the camera is held). Then the robust record: `residual_norms` (one per correspondence,
pixels), `weights` (the loss's ρ′/ρ′(0) at the estimate under a robust strategy, all
ones — lazily — under least squares), and the scale the loss ran at, `scale` and
`scale_source` (`ScaleGiven`/`ScaleUpstream`, or `NaN`/`ScaleNone` under least
squares and for a width given alone; [`noise_scale`](@extref VisualGeometryToolkitCore.noise_scale)
reads it) and `loss_width` (`NaN` under least squares). All forward as properties:
`result.residuals`, `result.spline`, ….
"""
struct CameraPosesRefineMetadata{L, W <: AbstractVector{Float64}} <: AbstractFitMetadata{L}
    residuals::Vector{Float64}
    s2::Float64
    dof::Int
    free::Tuple{Vararg{Symbol}}
    spline::Union{Nothing, NamedTuple}
    residual_norms::Vector{Float64}
    weights::W
    scale::Float64
    scale_source::ScaleSource
    loss_width::Float64
end
CameraPosesRefineMetadata{L}(residuals, s2, dof, free, spline, norms, weights::W, scale, source, loss_width) where {L, W} =
    CameraPosesRefineMetadata{L, W}(residuals, Float64(s2), Int(dof), free, spline, norms, weights, Float64(scale), source,
                                    Float64(loss_width))
# A least-squares refine: unit weights, no scale (ScaleNone) and no loss width; the norms from the rows.
CameraPosesRefineMetadata(residuals::Vector{Float64}, s2::Real, dof::Integer, free, spline) =
    CameraPosesRefineMetadata{Nothing}(residuals, s2, dof, free, spline, _row_norms(residuals),
                                       unit_weights(Float64, length(residuals) ÷ 2), NaN, ScaleNone, NaN)
# The norm of each correspondence's two residual rows (Core's `point_norms!`).
_row_norms(r) = (n = length(r) ÷ 2; point_norms!(Vector{Float64}(undef, n), r, n, 2))

"""
    CameraPosesProblem(correspondences; fixed = (), box = nothing, held_boards, held_views = (), residual = BearingResidual())

The `f` of the camera-and-poses refine (VisualGeometryToolkitCore's block contract, `AbstractBlockProblem`): the
3D → 2D `correspondences` of every view, and the problem's settings — `fixed` (the camera parameters held), `box` (a
solver box on the camera's parameters), the board form's `held_boards`/`held_views`, the rotation-centre form's
`residual`. A setting the form does not read is refused at the solve. Solved from a guess by Core's
`LevenbergMarquardt` (the default; `optimization_strategy`, `scale | width` and `stops` — default the
`[camera_poses_problem]` table — as every refine):

    solve(RefineProblem(CameraPosesProblem(views; fixed), (; camera, poses)), LevenbergMarquardt(); scale)
    solve(RefineProblem(CameraPosesProblem(views_by_view_and_board; held_boards), (; camera, poses, board_poses)), …)
    solve(RefineProblem(CameraPosesProblem(correspondences; fixed = held_parameters(camera)), (; camera, views)), …)   # views::RotationCenterViews

The guess is a `NamedTuple` of the result's own form; see `_camera_poses` for each form's data and record.
"""
struct CameraPosesProblem{C, K <: NamedTuple} <: AbstractBlockProblem
    correspondences::C
    settings::K
end
CameraPosesProblem(correspondences; settings...) = CameraPosesProblem(correspondences, values(settings))
const _CAMERA_POSES_STOPS = LmStops(; max_iter = _dflt("camera_poses_problem", "max_iter"), ftol = _dflt("camera_poses_problem", "ftol"),
                                    xtol = _dflt("camera_poses_problem", "xtol"))
default_stops(::CameraPosesProblem) = _CAMERA_POSES_STOPS
default_optimization_strategy(::CameraPosesProblem) = _dflt_strategy("camera_poses_problem", :camera_poses_problem)
# The block contract (Core's multi-block LM): the bundle stated ONCE at the guess by the one pipeline of every
# parametrization (`_camera_poses`) — its blocks, its variables and the assembly of its result (`block_structure`) —
# and the result assembled from that statement with Core's solution and residual summary (`block_result`): the
# pipeline never runs twice in a solve. No strategy is read when it is stated: Core robustifies the blocks.
_camera_poses_run(f::CameraPosesProblem, u0::NamedTuple{(:camera, :poses)}) =
    _camera_poses(u0.camera, u0.poses, f.correspondences; f.settings...)
_camera_poses_run(f::CameraPosesProblem, u0::NamedTuple{(:camera, :poses, :board_poses)}) =
    _camera_poses(u0.camera, u0.poses, u0.board_poses, f.correspondences; f.settings...)
function _camera_poses_run(f::CameraPosesProblem, u0::NamedTuple{(:camera, :views)})
    residual = get(f.settings, :residual, BearingResidual())
    rest = NamedTuple{Tuple(k for k in keys(f.settings) if k !== :residual)}(f.settings)
    return _camera_poses(u0.camera, u0.views, f.correspondences, residual; rest...)
end
block_structure(f::CameraPosesProblem, u0) = (s = _camera_poses_run(f, u0); (s.vars0, s.blocks, s))
# The result: the statement's assembly from Core's solved variables, its record and the residual summary (the rows, S,
# ν, the weights and their support and rms, the parsed loss the metadata records).
block_result(::CameraPosesProblem, _u0, r::FitResult, s) = _assembled(r, s, Val((:rms, :n, :residuals, :S, :ν, :weights)))

# A bundle stated at its guess, what a bundle pipeline's statement returns (the camera-poses and hand-eye bundles'):
# its residual blocks, its variables, and `finish(sol, loss)`, the assembly of its result from the solved bundle `sol`
# and the parsed loss (Core's `bundle_loss`). A wrapper of the pipeline composes its own reading of the result onto
# `finish` (`_then`). Concrete (the closures' types), so the block contract's reads and the refine are inferred.
struct _BundleStatement{B, V, F}
    blocks::B
    vars0::V
    finish::F
end
_then(g, st::_BundleStatement) = _BundleStatement(st.blocks, st.vars0, (sol, loss) -> g(st.finish(sol, loss)))
# A stated bundle's result (its `block_result`): its assembly on Core's solved variables and record and the
# residual-summary fields `keys` it reads, at the summary's parsed loss.
_assembled(r::FitResult, s::NamedTuple, ::Val{keys}) where {keys} =
    s.stated.finish(merge((; variables = r.u, code = r.retcode, rule = r.stop_rule, iterations = r.stats.nsteps),
                          NamedTuple{keys}(s)), s.loss)

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
with_robustifier(r::_BoardViewResidual{N, F, C, <:Any, M}, sqrt_weight::Float64, robust::K) where {N, F, C, K, M} =
    _BoardViewResidual{N, F, C, K, M}(r.varind, r.camera, r.view, r.board, r.X, r.u, sqrt_weight, robust)
NLLSsolver.ndeps(::_BoardViewResidual{N}) where {N} = NLLSsolver.static(N)
NLLSsolver.nres(::_BoardViewResidual) = NLLSsolver.static(2)
NLLSsolver.varindices(r::_BoardViewResidual) = r.varind
@inline function NLLSsolver.getvars(r::_BoardViewResidual{N, F, C, K, M}, vars::Vector) where {N, F, C, K, M}
    i = 1
    cam = F[1] ? (vars[r.varind[i]]::M,) : ()
    i += F[1]
    view = F[2] ? (vars[r.varind[i]]::LeftRigidManifold6{Float64},) : ()
    i += F[2]
    board = F[3] ? (vars[r.varind[i]]::LeftRigidManifold6{Float64},) : ()
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
    solve(RefineProblem(CameraPosesProblem(views; fixed = (), box = nothing), (; camera, poses)),
          LevenbergMarquardt(; optimization_strategy); scale | width, stops)   # stops default: configs/default.toml [camera_poses_problem]
    solve(RefineProblem(CameraPosesProblem(views; fixed = (:controls,)), (; camera = spline_camera, poses)), …)   # a SplineLens camera is held
        -> FitResult{<:NamedTuple{(:camera, :poses)}, CameraPosesRefineMetadata}
    solve(RefineProblem(CameraPosesProblem(views; held_boards, held_views = (), fixed = (), box), (; camera, poses, board_poses)), …)
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
  is held, its poses refined (below).
- `poses` — one world → camera `EuclideanMap{3}` per view (`X_c = R·X + t`, `X`
  in the target's frame), keyed like `views`.
- `views` — per view, its correspondences `cs = [X => u, …]`: `X` a target
  point (3-vector), `u` the pixel it is detected at (2-vector). Grouped data:
  an `AbstractDict{K, <:AbstractVector{<:Correspondence}}` keyed by the view, with
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
  and refines the poses alone. A SplineLens camera takes `(:controls,)` only
  (below). The stack's one parameter-holding keyword (NLLSsolver's word; OpenCV's
  `CALIB_FIX_*`): `free`, `fix`, `fixed_params` and `hold` are refused, naming it.
- `box` — a solver search box on the camera's parameters by name
  (VisualGeometryToolkitCore's `ParameterManifold(camera; box)`), or `nothing`;
  with every parameter `fixed` (nothing to bound) a box is an `ArgumentError`.
- `optimization_strategy` — the refine contract's strategy, as every
  `refine_*` takes it: `Triggs{Shape}` (the default, `Triggs{WelschShape}`,
  `configs/default.toml [camera_poses_problem]`), one robustified LM
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
- `stops` — the solve's `LmStops`: `max_iter`, the LM's cap on ACCEPTED iterations (NLLSsolver's count);
  `ftol` — its relative cost decrease; `xtol` — its step tolerance (chart units); `max_time` — its
  wall-clock cap in seconds (`nothing` or `Inf`: none), for a camera whose
  projection makes an iteration slow; it stops with `:max_time`.

The estimate is `(; camera, poses)`: the refined camera, and the refined poses
keyed like the input (a `Dict` by the same keys, or a vector in view order,
each with its input's frame labels). Its termination record (`retcode`, `stop_rule`) is the last solve's,
VisualGeometryToolkitCore's one LM: `ReturnCode.Success` at a tolerance stop (`ftol`, `xtol`;
Core 0.11.0: whatever the gradient there) or at the rejection cap when the estimate is stationary, else `MaxIters`
(the iteration cap), `MaxTime`, `Failure`/`InitialFailure` (`NonFinite`) or
`Stalled` (the rejection cap short of a stationary point); `rms` is
`√(Σ‖rᵢ‖² / n)` over the `n` correspondences (pixels). The metadata
([`CameraPosesRefineMetadata`](@ref)) carries the residual rows, `s2`, `dof` and
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
poses (the same assembler and LM, the same retcode and metadata; `dof` counts 6
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
controls, so its route differs (Core's `IntrinsicsRole` trait, `GaugeIntrinsics`,
selects it). **A SplineLens camera is refined only by back-projection, or held:**
its backward is fitted on back-projected rays (`CameraRegression(SplineLens, …)`,
[`PSpline`](@ref)), and here it is held while the poses are refined.
- `fixed = (:controls,)` is required: the camera held, the poses refined. The
  default `fixed = ()`, which would refine the controls against the pixel rows,
  is an `ArgumentError` naming the retired image-space refine
  (VisualGeometryToolkitEstimation 0.10.0), and so is a `penalty` (that route's
  smoothness weight); a `box` and any other `fixed` are `ArgumentError`s.
- Each correspondence's rows are `project(camera, X_c) − u` through the whole
  held camera, a constant (the parametric route's block, on its pose alone): no
  window, no node column. A correspondence detected on a window with an absent
  node is dropped and counted (`result.spline.n_dropped_window`), and so is one
  the camera does not project when detected within one knot cell of such a
  window (it images it onto the absent nodes; counted apart,
  `result.spline.n_dropped_unprojected`: it can be an inlier); every other one
  must project at the guess (an `ArgumentError` names the first that does not,
  or where the lattice folds).
- The camera comes back AS GIVEN, its forward spline included (none is refitted): its
  controls are no variable, so there is no gauge to fix and no frame to turn, and the
  poses come back in the camera's own frame. A bare model or a `BoundedCamera` alike. Each
  row is the backward's own projection solve (a forward the camera holds is not used
  in the solve: it is the backward's approximation). The keywords `refit_forward` and
  `forward_tolerance` of the retired forward refit are refused by name (0.11.0); give a
  camera its forward with [`ForwardSpline`](@ref).
- The strategy is the bundle's (`optimization_strategy`, its `scale` or `width`: the
  default Triggs{Welsch} needs one; `nothing` is least squares). `max_time` caps the
  solve (the record `(ReturnCode.MaxTime, TimeCap)`).
- Device: the projections that check the guess run batched on the device
  (Core's `default_device()`); the LM is NLLSsolver's, on the CPU.

# Units
Plain inputs give plain poses; physical ones (unitful poses and target points,
all one form) are read in metres at entry and the poses come back physical, in
metres. The residuals are pixels either way.
"""
function _camera_poses(
        camera::_RefinedCamera, poses::AbstractDict{K, <:PlainEuclideanMap{3}},
        views::AbstractDict{K, <:AbstractVector{<:Correspondence{<:StaticVector{3, <:Real}, <:StaticVector{2}}}};
        kwargs...
    ) where {K}
    ids = block_order(views, poses)
    return _then(_refine_camera_and_poses(camera, [poses[k] for k in ids], [views[k] for k in ids]; kwargs...)) do r
        est = first(r)
        FitResult((; camera = est.camera, poses = Dict{K, eltype(est.poses)}(zip(ids, est.poses))), Base.tail(r)...)
    end
end
function _camera_poses(
        camera::_RefinedCamera, poses::AbstractVector{<:PlainEuclideanMap{3}},
        views::AbstractVector{<:AbstractVector{<:Correspondence{<:StaticVector{3, <:Real}, <:StaticVector{2}}}};
        kwargs...
    )
    length(poses) == length(views) || throw(ArgumentError("CameraPosesProblem: $(length(poses)) poses for $(length(views)) views"))
    return _then(r -> FitResult(r...), _refine_camera_and_poses(camera, collect(poses), collect(views); kwargs...))
end

# Several boards: the view poses (world → camera) and the board poses (board → world) keyed, the correspondences keyed
# by (view, board). Each connected component of the view–board graph has a held pose: its gauge.
function _camera_poses(
        camera::_RefinedCamera, poses::AbstractDict{V, <:PlainEuclideanMap{3}}, board_poses::AbstractDict{B, <:PlainEuclideanMap{3}},
        views::AbstractDict{<:Tuple{V, B}, <:AbstractVector{<:Correspondence{<:StaticVector{3, <:Real}, <:StaticVector{2}}}};
        held_boards = nothing, held_views = (), kwargs...
    ) where {V, B}
    VgtkCore.Cameras.IntrinsicsRole(_camera_model(camera).lens) isa VgtkCore.Cameras.FreeIntrinsics || throw(ArgumentError(
        "CameraPosesProblem: board poses refine a parametric camera; a SplineLens camera's gauge is its own (refine it with one board)"))
    vids = sort!(collect(keys(poses))); bids = sort!(collect(keys(board_poses)))
    keys_ = sort!(collect(keys(views)))
    isempty(keys_) && throw(ArgumentError("CameraPosesProblem: no views"))
    vi = Dict(k => i for (i, k) in enumerate(vids)); bi = Dict(k => i for (i, k) in enumerate(bids))
    for (v, b) in keys_
        haskey(vi, v) || throw(ArgumentError("CameraPosesProblem: the correspondences of ($(repr(v)), $(repr(b))) name a view with no pose"))
        haskey(bi, b) || throw(ArgumentError("CameraPosesProblem: the correspondences of ($(repr(v)), $(repr(b))) name a board with no pose"))
    end
    seen_v = Set(first.(keys_)); seen_b = Set(last.(keys_))
    for v in vids
        v in seen_v || throw(ArgumentError("CameraPosesProblem: view $(repr(v)) has a pose and no correspondence"))
    end
    for b in bids
        b in seen_b || throw(ArgumentError("CameraPosesProblem: board $(repr(b)) has a pose and no correspondence"))
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
            haskey(bi, b) || throw(ArgumentError("CameraPosesProblem: held board $(repr(b)) has no pose"))
            held_b[bi[b]] = true
        end
    end
    for v in held_views
        haskey(vi, v) || throw(ArgumentError("CameraPosesProblem: held view $(repr(v)) has no pose"))
        held_v[vi[v]] = true
    end
    for c in unique(root(i) for i in 1:(nv + nb))
        any(held_v[i] for i in 1:nv if root(i) == c) || any(held_b[b] for b in 1:nb if root(nv + b) == c) || throw(ArgumentError(
            "CameraPosesProblem: the views $(repr([vids[i] for i in 1:nv if root(i) == c])) and boards " *
            "$(repr([bids[b] for b in 1:nb if root(nv + b) == c])) hold no pose: their common motion is free (hold a board or a view)"))
    end
    obs = [(vi[v], bi[b], _correspondence(vi[v], j, c)...) for (v, b) in keys_ for (j, c) in enumerate(views[(v, b)])]
    return _then(_refine_bundle(camera, [poses[k] for k in vids], [board_poses[k] for k in bids], obs, held_v, held_b; kwargs...)) do r
        est = first(r)
        FitResult((; camera = est.camera, poses = Dict{V, eltype(est.poses)}(zip(vids, est.poses)),
                     board_poses = Dict{B, eltype(est.boards)}(zip(bids, est.boards))), Base.tail(r)...)
    end
end

# Physical inputs: the poses and the target points in metres once, the plain
# refinement, the poses read back in metres once.
function _camera_poses(
        camera::_RefinedCamera, poses::AbstractDict{K, <:_PhysicalPose3},
        views::AbstractDict{K, <:AbstractVector{<:Correspondence{<:_PhysicalPoint3, <:StaticVector{2}}}}; kwargs...
    ) where {K}
    st = _camera_poses(
        camera, Dict(k => _plain_meters(p) for (k, p) in poses), Dict(k => _map_correspondences(_plain_meters, identity, cs) for (k, cs) in views); kwargs...
    )
    return _then(r -> VgtkCore.map_estimate(est -> (; camera = est.camera, poses = Dict(k => _attach_meters(p) for (k, p) in est.poses)), r), st)
end
function _camera_poses(
        camera::_RefinedCamera, poses::AbstractVector{<:_PhysicalPose3},
        views::AbstractVector{<:AbstractVector{<:Correspondence{<:_PhysicalPoint3, <:StaticVector{2}}}}; kwargs...
    )
    st = _camera_poses(camera, map(_plain_meters, poses), [_map_correspondences(_plain_meters, identity, cs) for cs in views]; kwargs...)
    return _then(r -> VgtkCore.map_estimate(est -> (; camera = est.camera, poses = map(_attach_meters, est.poses)), r), st)
end


# The route, by Core's `IntrinsicsRole` trait: a free K (a parametric lens) below; a gauge K
# (a SplineLens) in camera_poses_spline.jl.
_refine_camera_and_poses(camera::_RefinedCamera, poses::AbstractVector, views::AbstractVector; kwargs...) =
    _refine_camera_and_poses(VgtkCore.Cameras.IntrinsicsRole(_camera_model(camera).lens), camera, poses, views; kwargs...)


# The refine's input contract, both routes: the poses as pose charts (VGC's
# `LeftRigidManifold6`), a correspondence `X => u` as finite static vectors (an
# `ArgumentError` naming view `v`, correspondence `j` otherwise), and a pose chart's
# model as a plain map.
_pose_charts(poses) = LeftRigidManifold6{Float64}[LeftRigidManifold6(SMatrix{3, 3, Float64, 9}(rotation(p)), SVector{3, Float64}(translation(p))) for p in poses]
function _correspondence(v::Integer, j::Integer, c)
    X = SVector{3, Float64}(first(c)[1], first(c)[2], first(c)[3]); u = SVector{2, Float64}(last(c)[1], last(c)[2])
    (all(isfinite, X) && all(isfinite, u)) || throw(ArgumentError("CameraPosesProblem: view $v, correspondence $j: non-finite point or pixel"))
    return X, u
end
_finite_pose(P::PlainEuclideanMap{3}) = all(isfinite, rotation(P)) && all(isfinite, translation(P))
_plain_pose(P::PlainEuclideanMap{3}) = EuclideanMap(RotMatrix{3, Float64}(rotation(P)), SVector{3, Float64}(translation(P)))
# each refined pose under its input's frame labels
_relabeled(inputs::AbstractVector, outs::AbstractVector{<:PlainEuclideanMap{3}}) =
    [relabel(VgtkCore.map_frames(p0)..., _plain_pose(P)) for (p0, P) in zip(inputs, outs)]

# Without boards: every correspondence's point in the world, one held identity board (`_refine_views`).
function _refine_camera_and_poses(
        ::VgtkCore.Cameras.FreeIntrinsics, bounded::_RefinedCamera, poses::AbstractVector, views::AbstractVector; kwargs...
    )
    kw = _bundle_keywords(; kwargs...)
    camera = _camera_model(bounded)
    obs = [(v, 1, _correspondence(v, j, c)...) for (v, cs) in enumerate(views) for (j, c) in enumerate(cs)]
    return _refine_views(kw, bounded, camera, poses, obs, _camera_chart(camera, kw.fixed, kw.box))
end

# The views' poses (and the camera's `chart`, `(free, chart)`) refined against `obs`, `(view, 1, X, u)` each, every
# point in the world: one held identity board. Both routes' (the SplineLens one with nothing of the camera free).
function _refine_views(kw, bounded::_RefinedCamera, camera, poses::AbstractVector, obs::AbstractVector, chart; spline = nothing)
    V = length(poses)
    V >= 1 || throw(ArgumentError("CameraPosesProblem: no views"))
    world = EuclideanMap(RotMatrix{3, Float64}(one(SMatrix{3, 3, Float64, 9})), zero(SVector{3, Float64}))
    return _then(r -> ((; camera = first(r).camera, poses = first(r).poses), Base.tail(r)...),
                 _refine_bundle(kw, bounded, camera, chart, poses, [world], obs, falses(V), trues(1); spline))
end


# The refine's keywords, shared by every parametrization: parsed once (the retired spellings refused; the strategy, its
# scale input and its loss: `bundle_loss`).
function _bundle_keywords(;
        fixed::Tuple{Vararg{Symbol}} = (),
        free = RetiredKeyword(), fix = RetiredKeyword(), fixed_params = RetiredKeyword(), hold = RetiredKeyword(),   # refused, naming `fixed`
        box::Union{Nothing, NamedTuple} = nothing,
        robust = RetiredKeyword(),
        local_optimization = RetiredKeyword(),   # refused, naming its successor
    )
    refuse_retired_keyword(robust, :robust, :optimization_strategy)
    refuse_retired_keyword(local_optimization, :local_optimization, :optimization_strategy)
    _refuse_held_spellings(free, fix, fixed_params, hold)
    return (; fixed, box)
end

# The camera's chart: every parameter refined but the `fixed` ones (names or groups: `:k` holds k1…) and the gauge (the
# skew, held by construction): the stack's one expansion; `nothing` when every parameter is held.
function _camera_chart(camera, fixed, box)
    free = VgtkCore.free_parameter_names(parameter_names(camera), fixed)
    isempty(free) && box !== nothing &&
        throw(ArgumentError("CameraPosesProblem: a `box` bounds the camera's refined parameters, and `fixed` holds every one"))
    return free, isempty(free) ? nothing : restrict(ParameterManifold(camera; box), free)
end

# The one bundle: a camera, view poses (world → camera) and board poses (board → world), each a variable unless held
# (`held_views`, `held_boards`), against the reprojection error of `obs`, `(view, board, X, u)` each. One NLLSsolver
# problem: the camera's chart (when anything is free), a pose chart per free view and per free board, one residual
# block per correspondence: its statement (`_BundleStatement`), which Core's multi-block LM solves. The keyword form parses the bundle's keywords and the camera's
# chart; the parsed form takes them (`kw`, `(free, chart)`: `((), nothing)` holds the camera), projects through
# `camera` (the bounded camera's model, or the SplineLens route's camera without its forward) and returns `bounded`
# itself when nothing of the camera is free; `spline` is the SplineLens route's record (`CameraPosesRefineMetadata`).
function _refine_bundle(
        bounded::_RefinedCamera, poses::AbstractVector, boards::AbstractVector, obs::AbstractVector,
        held_views::AbstractVector{Bool}, held_boards::AbstractVector{Bool}; kwargs...
    )
    kw = _bundle_keywords(; kwargs...)
    camera = _camera_model(bounded)
    return _refine_bundle(kw, bounded, camera, _camera_chart(camera, kw.fixed, kw.box), poses, boards, obs, held_views, held_boards)
end
function _refine_bundle(
        kw, bounded::_RefinedCamera, camera, (free, chc), poses::AbstractVector, boards::AbstractVector, obs::AbstractVector,
        held_views::AbstractVector{Bool}, held_boards::AbstractVector{Bool}; spline = nothing
    )
    V, B = length(poses), length(boards)
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
            ArgumentError("CameraPosesProblem: view $v, correspondence $k: the point $(X) does not project at the guess (behind the camera or outside the lens's domain)")
        )
        F = (chc !== nothing, !held_views[v], !held_boards[b])
        ind = SVector{count(F), Int}(Tuple(i for (i, f) in zip((1, view_var[v], board_var[b]), F) if f))
        push!(blocks, _BoardViewResidual{count(F), F, typeof(camera)}(ind, camera, plain(view_charts[v]), plain(board_charts[b]), X, u;
                                                                     chart = chc === nothing ? Nothing : typeof(chc)))
    end
    n = length(obs)
    p = nd + 6 * (count(!, held_views) + count(!, held_boards))
    2n > p || throw(ArgumentError("CameraPosesProblem: underdetermined: $n correspondences ($(2n) rows) for $p unknowns"))
    o > 0 || throw(ArgumentError("CameraPosesProblem: nothing to refine (the camera, every view and every board held)"))

    VT = chc === nothing ? LeftRigidManifold6{Float64} : Union{typeof(chc), LeftRigidManifold6{Float64}}
    vars0 = VT[]
    chc === nothing || push!(vars0, chc)
    foreach(v -> held_views[v] || push!(vars0, view_charts[v]), 1:V)
    foreach(b -> held_boards[b] || push!(vars0, board_charts[b]), 1:B)
    return _BundleStatement(blocks, vars0, (sol, loss) -> _bundle_assembly(sol, loss, bounded, camera, chc, free, poses, boards,
                                                                          view_charts, board_charts, view_var, board_var; spline))
end
# The bundle's result from its solve `sol` at the parsed `loss`: the camera, the views' and the boards' poses read back
# (a held one as given), refused when not finite, with the record and the metadata.
function _bundle_assembly(sol, loss, bounded, camera, chc, free, poses, boards, view_charts, board_charts, view_var, board_var;
                          spline)
    plain(ch) = model(ch)::PlainEuclideanMap{3, Float64}
    variables = sol.variables
    model_out = chc === nothing ? camera : model(variables[1])
    pose_of(charts, var, i) = var[i] == 0 ? plain(charts[i]) : plain(variables[var[i]]::LeftRigidManifold6{Float64})
    poses_out = PlainEuclideanMap{3, Float64}[pose_of(view_charts, view_var, v) for v in eachindex(view_charts)]
    boards_out = PlainEuclideanMap{3, Float64}[pose_of(board_charts, board_var, b) for b in eachindex(board_charts)]
    all(_finite_pose, poses_out) && all(_finite_pose, boards_out) && all(isfinite, parameter_vector(model_out)) ||
        throw(ErrorException("CameraPosesProblem: the LM step is not finite"))
    camera_out = model_out === camera ? bounded : VgtkCore.with_model(bounded, model_out)
    return (; camera = camera_out, poses = _relabeled(poses, poses_out), boards = _relabeled(boards, boards_out)), sol.code, sol.rule,
        SolveStats(; nsteps = sol.iterations), sol.rms, sol.n, _bundle_metadata(sol, free, loss; spline)
end

# The shared solve of every parametrization (Core's multi-block LM through the block contract): the stated residual
# blocks (2 rows each, one per correspondence, each answering `with_robustifier`) on the stated variables, at the
# parsed loss (`loss`, Core's `bundle_loss`). Least squares is
# the whole solve when `optimization_strategy = nothing`. A robust refine has NO least-squares stage (the owner's
# ruling, 2026-11-11): the strategy's solve starts from the caller's start, its loss at its width over the caller's
# frozen `scale` or at its absolute width: Triggs, ONE robustified LM (Core's `loss_robustifier`), the refine
# contract's one fixed-scale robust path (an Irls is refused by `bundle_loss`). A refine estimates no scale. Returns the variables at the estimate, the termination record (`code`, `rule`), the iterations, the reprojection rows,
# their sum of squares, ν, the weights, and `n` the weights' support and `rms` the weighted one (the stack's one rule,
# loop.jl: under least squares every correspondence and √(S/n)); the metadata records them with the parsed loss.
_bundle_metadata(sol, free, loss; spline = nothing) = CameraPosesRefineMetadata{typeof(loss.strategy)}(sol.residuals, sol.S / sol.ν, sol.ν, free, spline,
    _row_norms(sol.residuals), sol.weights, VgtkCore.scale_record(loss)...)

