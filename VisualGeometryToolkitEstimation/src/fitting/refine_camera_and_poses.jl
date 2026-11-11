# =============================================================================
# refine_camera_and_poses — a camera and one pose per view, from 3D → 2D
# correspondences, against the reprojection error
# =============================================================================
#
# The bundle of a calibration with a known target: given a camera guess (its
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
# restricted to the free parameters (`free`, every parameter but the skew by
# default), optionally with a solver box (`box`). Each pose is VGC's
# `LeftRigidManifold6` (x ↦ exp([δω]×)(R·x + t) + δt). Each chart is one
# NLLSsolver variable through VGC's NLLSsolver adapter, re-anchored at every
# accepted step; each correspondence is one residual block of 2 rows, depending
# on the camera and its view's pose (on the pose alone with nothing free in the
# camera), differentiated by NLLSsolver (ForwardDiff through the retractions and
# the lens). The driver is VGC's `nlls_options` (Marquardt's DᵀD damping); the
# stop is checked by VGC's stall postcondition (`nlls_retcode`) on the rows and
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
windows_settled, forward, forward_max_px, gauge_angle, gauge_roll, cost_start)`:
the third-difference
penalty's normalized weight β and its `λ` (0 for none, and with `free = ()`;
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
(`:none` without one, `:refitted`, `:dropped`, or `:kept` with the controls
held) and the refit's held-out maximum (pixels, or `nothing`); the gauge
rotation's two angles (the axis's and the roll's, rad); and the joint objective
(½ the sum of squares, the penalty rows included) at the joint solve's start. For
a SplineLens camera `s²` reads the whole cost, the
third-difference rows included, and `ν = 2n + 3m − p` counts its `m`
third-difference rows as observations and `p = 3·(the nodes a row reads) − 4 +
6·views` (the gauge's 4 dimensions, the common scale and the rotation, are no
unknowns). All forward as properties:
`result.residuals`, `result.spline`, ….
"""
struct CameraPosesRefineMetadata
    residuals::Vector{Float64}
    s²::Float64
    ν::Int
    free::Tuple{Vararg{Symbol}}
    spline::Union{Nothing, NamedTuple}
end

# The cameras refined: a camera model, or a bounded one (refined as its model, returned bounded alike).
const _RefinedCamera = Union{CameraModel, BoundedCamera{<:CameraModel}}
_camera_model(c::CameraModel) = c
_camera_model(c::BoundedCamera) = model(c)

# One correspondence's residual block: the target point and its pixel, its
# variables (camera, pose) — or (pose,) under a held camera, which it carries.
struct _CameraPoseResidual{N, C} <: NLLSsolver.AbstractResidual
    varind::SVector{N, Int}
    camera::C                       # the held camera (read only when N == 1)
    X::SVector{3, Float64}
    u::SVector{2, Float64}
end
NLLSsolver.ndeps(::_CameraPoseResidual{N}) where {N} = NLLSsolver.static(N)
NLLSsolver.nres(::_CameraPoseResidual) = NLLSsolver.static(2)
NLLSsolver.varindices(r::_CameraPoseResidual) = r.varind
NLLSsolver.getvars(r::_CameraPoseResidual{2}, vars::Vector) = (vars[r.varind[1]], vars[r.varind[2]])
NLLSsolver.getvars(r::_CameraPoseResidual{1}, vars::Vector) = (vars[r.varind[1]],)
Base.eltype(::_CameraPoseResidual) = Float64
NLLSsolver.computeresidual(r::_CameraPoseResidual{2}, camera_chart, pose_chart) =
    VgtkCore.reprojection_error(model(camera_chart), model(pose_chart), r.X, r.u)
NLLSsolver.computeresidual(r::_CameraPoseResidual{1}, pose_chart) = VgtkCore.reprojection_error(r.camera, model(pose_chart), r.X, r.u)
# The block's cost as the LM reads it: non-finite (a candidate update that moves a
# point off the lens's domain) is +Inf (`finite_cost`).
NLLSsolver.computecost(vars::AbstractVector, r::_CameraPoseResidual) =
    finite_cost(NLLSsolver.computecost(r, NLLSsolver.getvars(r, vars)...))

"""
    refine_camera_and_poses(camera, poses, views; free, box = nothing,
                            max_iter, ftol, xtol, max_time = nothing)   # defaults: configs/default.toml [refine_camera_and_poses]
    refine_camera_and_poses(spline_camera, poses, views; free = (:controls,), penalty = :gcv,
                            refit_forward = true, forward_tolerance, max_iter, ftol, xtol, max_time = nothing)
        -> FitResult{<:NamedTuple{(:camera, :poses)}, CameraPosesRefineMetadata}

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
- `free` — the camera parameters refined, by name (every parameter but the
  skew by default); `()` holds the camera fixed and refines the poses alone.
- `box` — a solver search box on the camera's parameters by name
  (VisualGeometryToolkitCore's `ParameterManifold(camera; box)`), or `nothing`;
  with `free = ()` (nothing to bound) a box is an `ArgumentError`.
- `max_iter` — the LM's cap on ACCEPTED iterations (NLLSsolver's count);
  `ftol` — its relative cost decrease and the stall postcondition's exact-fit
  test; `xtol` — its step tolerance (chart units); `max_time` — its
  wall-clock cap in seconds (`nothing` or `Inf`: none), for a camera whose
  projection makes an iteration slow; it stops with `:max_time`.

The estimate is `(; camera, poses)`: the refined camera, and the refined poses
keyed like the input (a `Dict` by the same keys, or a vector in view order,
each with its input's frame labels). `retcode` is `:converged` only at a
stationary point (VisualGeometryToolkitCore's `nlls_retcode`), else
`:max_iterations`, `:max_time`, `:solve_failed` or `:stalled`; `rms` is
`√(Σ‖rᵢ‖² / n)` over the `n` correspondences (pixels). The metadata
([`CameraPosesRefineMetadata`](@ref)) carries the residual rows, `s²`, `ν` and
the free parameter names. Plain least squares: the correspondences must be
inliers.

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
  solve from far off creeps, or lets the scales absorb the poses' error). `free` is `(:controls,)` (the default) or
  `()` (the poses alone, the camera returned as given); a `box` is an
  `ArgumentError`.
- Each correspondence's rows are `K·x* − u`, `x*` the lens's projection solve
  on the correspondence's 4 × 4 window at the guess, its Jacobian analytic by
  implicit differentiation at `x*` (no automatic differentiation reaches the
  solve). A correspondence detected on a window with an absent node is
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
- The gauge: a rotation of every control together with every pose, and one
  common scale of every control, leave the cost unchanged; the estimate is the
  one whose lens axis `normalize(backward(0, 0))` is ẑ, whose backward
  x-derivative at x = 0 lies in the xz plane (the convention a SplineLens
  camera's K is read in), and whose refined controls' mean length is the
  guess's. During the solve the gauge is fixed exactly by the parametrization
  (the first view's rotation and the log-scale of the node nearest x = 0 are no
  coordinate of any variable), so the LM's normal matrix is full rank; after it, one
  step (the minimal rotation onto ẑ, the roll, the scale) moves the estimate
  to that canonical representative. K is bit-identical to the guess's.
- A forward spline is dropped for the refinement and, with `refit_forward`
  (the default), refitted over `imaged_region(camera)` by
  `fit_camera(SplineLens, …; tolerance = forward_tolerance)` (pixels), which
  needs a `BoundedCamera` (an `ArgumentError` for a bare camera model). With
  `free = ()` it is kept.
- The model: the target is exact, so the estimate is the maximum-likelihood
  one under isotropic Gaussian detection noise (penalized by the
  smoothness prior; with `penalty = 0` the plain maximum-likelihood one). A board that departs from its
  design (an in-plane warp, an affine or smooth deformation, a height field)
  is not modelled here: that is the board-deformation solve of
  RadiometricCamera's calibration pipeline.
- `max_time` caps the whole refine (every stage and rebuild round reads what
  is left of it; `:max_time` when any was cut short).
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
    return _camera_poses_result((; camera = est.camera, poses = Dict(k => _attach_metres(p) for (k, p) in est.poses)), retcode(r), iterations(r), rms(r), r.n, metadata(r))
end
function refine_camera_and_poses(
        camera::_RefinedCamera, poses::AbstractVector{<:_PhysicalPose3},
        views::AbstractVector{<:AbstractVector{<:Cspond{<:_PhysicalPoint3, <:StaticVector{2}}}}; kwargs...
    )
    r = refine_camera_and_poses(camera, map(_plain_metres, poses), [_map_csponds(_plain_metres, identity, cs) for cs in views]; kwargs...)
    est = estimate(r)
    return _camera_poses_result((; camera = est.camera, poses = map(_attach_metres, est.poses)), retcode(r), iterations(r), rms(r), r.n, metadata(r))
end

_camera_poses_result(est, code, iters, rms, n, meta) = FitResult(est, code, iters, rms, n, meta)

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

function _refine_camera_and_poses(
        ::VgtkCore.Cameras.FreeIntrinsics, bounded::_RefinedCamera, poses::AbstractVector, views::AbstractVector;
        free::Tuple{Vararg{Symbol}} = Tuple(p for p in parameter_names(_camera_model(bounded)) if p !== :skew),
        box::Union{Nothing, NamedTuple} = nothing,
        max_iter::Integer = _dflt("refine_camera_and_poses", "max_iter"),
        ftol::Real = _dflt("refine_camera_and_poses", "ftol"),
        xtol::Real = _dflt("refine_camera_and_poses", "xtol"),
        max_time::Union{Nothing, Real} = nothing,
    )
    camera = _camera_model(bounded)
    V = length(views)
    V >= 1 || throw(ArgumentError("refine_camera_and_poses: no views"))
    names = parameter_names(camera)
    for p in free
        p in names || throw(ArgumentError("refine_camera_and_poses: the camera has no parameter $(repr(p)); its parameters are $(join(String.(names), ", "))"))
    end
    isempty(free) && box !== nothing &&
        throw(ArgumentError("refine_camera_and_poses: a `box` bounds the camera's free parameters, and `free = ()` frees none"))
    chc = isempty(free) ? nothing : restrict(ParameterManifold(camera; box), free)
    nd = chc === nothing ? 0 : tangent_size(chc)
    pose_charts = _pose_charts(poses)
    o = chc === nothing ? 0 : 1                                              # the poses' variable offset
    N = chc === nothing ? 1 : 2
    costs = _CameraPoseResidual{N, typeof(camera)}[]
    for (v, cs) in enumerate(views), (j, c) in enumerate(cs)
        X, u = _correspondence(v, j, c)
        all(isfinite, VgtkCore.reprojection_error(camera, model(pose_charts[v]), X, u)) || throw(
            ArgumentError("refine_camera_and_poses: view $v, correspondence $j: the point $(X) does not project at the guess (behind the camera or outside the lens's domain)")
        )
        push!(costs, _CameraPoseResidual{N, typeof(camera)}(N == 2 ? SVector(1, o + v) : SVector(o + v), camera, X, u))
    end
    n = length(costs)
    p = nd + 6V
    2n > p || throw(ArgumentError("refine_camera_and_poses: underdetermined: $n correspondences ($(2n) rows) for $p unknowns"))

    problem = chc === nothing ? NLLSsolver.NLLSProblem(_CameraPosePose, eltype(costs)) :
        NLLSsolver.NLLSProblem(Union{typeof(chc), _CameraPosePose}, eltype(costs))
    chc === nothing || NLLSsolver.addvariable!(problem, chc)
    foreach(ch -> NLLSsolver.addvariable!(problem, ch), pose_charts)
    foreach(c -> NLLSsolver.addcost!(problem, c), costs)
    result = NLLSsolver.optimize!(problem, nlls_options(; max_iter, ftol, xtol, max_time))
    chc_out = chc === nothing ? nothing : problem.variables[1]
    poses_out = PlainEuclideanMap{3, Float64}[model(problem.variables[o + v]::_CameraPosePose) for v in 1:V]
    camera_out = chc_out === nothing ? camera : model(chc_out)
    all(_finite_pose, poses_out) && all(isfinite, parameter_vector(camera_out)) ||
        throw(ErrorException("refine_camera_and_poses: the LM step is not finite"))

    r, J = nlls_rows_and_jacobian(costs, problem.variables)
    code = nlls_retcode(result, J, r; tol = _CHART_LM_STATIONARITY_TOL, ftol)
    S = sum(abs2, r)
    ν = 2n - p
    out = [relabel(VgtkCore.map_frames(p0)..., _plain_pose(P)) for (p0, P) in zip(poses, poses_out)]
    camera_out = camera_out === camera ? bounded : VgtkCore.with_model(bounded, camera_out)
    return (; camera = camera_out, poses = out), code, result.niterations, sqrt(S / n), n, CameraPosesRefineMetadata(r, S / ν, ν, free, nothing)
end

