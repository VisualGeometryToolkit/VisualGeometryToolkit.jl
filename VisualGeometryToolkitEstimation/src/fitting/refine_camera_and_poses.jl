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
free camera parameters plus 6 per view); and `free`, the names of the camera
parameters refined. All forward as properties: `result.residuals`, ….
"""
struct CameraPosesRefineMetadata
    residuals::Vector{Float64}
    s²::Float64
    ν::Int
    free::Tuple{Vararg{Symbol}}
end

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
# point off the lens's domain) is +Inf (`_finite_cost`).
NLLSsolver.computecost(vars::AbstractVector, r::_CameraPoseResidual) =
    _finite_cost(NLLSsolver.computecost(r, NLLSsolver.getvars(r, vars)...))

"""
    refine_camera_and_poses(camera::CameraModel, poses, views; free, box = nothing,
                            max_iter, ftol, xtol)   # defaults: configs/default.toml [refine_camera_and_poses]
        -> FitResult{<:NamedTuple{(:camera, :poses)}, CameraPosesRefineMetadata}

Refine a camera and one pose per view jointly against the reprojection error of
3D → 2D correspondences: minimize `Σ ‖project(camera, pose_v(X)) − u‖²` over
the camera's free parameters and every view's pose, from the guesses `camera`
and `poses`. A refine: both guesses are required, nothing is re-estimated from
scratch and the camera keeps its family (its lens type).

# Arguments
- `camera::CameraModel` — the camera guess: intrinsics and lens, the
  parameters VisualGeometryToolkitCore's `parameter_vector(camera)` lists
  (`f, aspect, skew, cx, cy`, then the lens's coefficients), in pixels.
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
  (VisualGeometryToolkitCore's `ParameterManifold(camera; box)`), or `nothing`.
- `max_iter` — the LM's cap on ACCEPTED iterations (NLLSsolver's count);
  `ftol` — its relative cost decrease and the stall postcondition's exact-fit
  test; `xtol` — its step tolerance (chart units).

The estimate is `(; camera, poses)`: the refined camera, and the refined poses
keyed like the input (a `Dict` by the same keys, or a vector in view order,
each with its input's frame labels). `retcode` is `:converged` only at a
stationary point (VisualGeometryToolkitCore's `nlls_retcode`), else
`:max_iterations`, `:max_time`, `:solve_failed` or `:stalled`; `rms` is
`√(Σ‖rᵢ‖² / n)` over the `n` correspondences (pixels). The metadata
([`CameraPosesRefineMetadata`](@ref)) carries the residual rows, `s²`, `ν` and
the free parameter names. Plain least squares: the correspondences must be
inliers.

# Units
Plain inputs give plain poses; physical ones (unitful poses and target points,
all one form) are read in metres at entry and the poses come back physical, in
metres. The residuals are pixels either way.
"""
function refine_camera_and_poses(
        camera::CameraModel, poses::AbstractDict{K, <:PlainEuclideanMap{3}},
        views::AbstractDict{K, <:AbstractVector{<:Cspond{<:StaticVector{3, <:Real}, <:StaticVector{2}}}};
        kwargs...
    ) where {K}
    ids = block_order(views, poses)
    est, rest... = _refine_camera_and_poses(camera, [poses[k] for k in ids], [views[k] for k in ids]; kwargs...)
    return _camera_poses_result((; camera = est.camera, poses = Dict{K, eltype(est.poses)}(zip(ids, est.poses))), rest...)
end
function refine_camera_and_poses(
        camera::CameraModel, poses::AbstractVector{<:PlainEuclideanMap{3}},
        views::AbstractVector{<:AbstractVector{<:Cspond{<:StaticVector{3, <:Real}, <:StaticVector{2}}}};
        kwargs...
    )
    length(poses) == length(views) || throw(ArgumentError("refine_camera_and_poses: $(length(poses)) poses for $(length(views)) views"))
    return _camera_poses_result(_refine_camera_and_poses(camera, collect(poses), collect(views); kwargs...)...)
end

# Physical inputs: the poses and the target points in metres once, the plain
# refinement, the poses read back in metres once.
function refine_camera_and_poses(
        camera::CameraModel, poses::AbstractDict{K, <:_PhysicalPose3},
        views::AbstractDict{K, <:AbstractVector{<:Cspond{<:_PhysicalPoint3, <:StaticVector{2}}}}; kwargs...
    ) where {K}
    r = refine_camera_and_poses(
        camera, Dict(k => _plain_metres(p) for (k, p) in poses), Dict(k => _map_csponds(_plain_metres, identity, cs) for (k, cs) in views); kwargs...
    )
    est = estimate(r)
    return _camera_poses_result((; camera = est.camera, poses = Dict(k => _in_metres(p) for (k, p) in est.poses)), retcode(r), iterations(r), rms(r), r.n, metadata(r))
end
function refine_camera_and_poses(
        camera::CameraModel, poses::AbstractVector{<:_PhysicalPose3},
        views::AbstractVector{<:AbstractVector{<:Cspond{<:_PhysicalPoint3, <:StaticVector{2}}}}; kwargs...
    )
    r = refine_camera_and_poses(camera, map(_plain_metres, poses), [_map_csponds(_plain_metres, identity, cs) for cs in views]; kwargs...)
    est = estimate(r)
    return _camera_poses_result((; camera = est.camera, poses = map(_in_metres, est.poses)), retcode(r), iterations(r), rms(r), r.n, metadata(r))
end

_camera_poses_result(est, code, iters, rms, n, meta) = FitResult(est, code, iters, rms, n, meta)

const _CameraPosePose = LeftRigidManifold6{Float64}

# The rows of every correspondence and their dense Jacobian in the free chart
# coordinates [camera | pose₁ | pose₂ | …] at the charts' anchors — the stall
# postcondition's (ForwardDiff through the retractions, per block).
function _camera_poses_rows_and_jacobian(costs, camera_chart, pose_charts, nd::Int)
    n = length(costs)
    r = Vector{Float64}(undef, 2n)
    J = zeros(2n, nd + 6 * length(pose_charts))
    zc = zeros(nd)
    for (k, c) in enumerate(costs)
        v = nd == 0 ? c.varind[1] : c.varind[2]
        chp = pose_charts[v - (nd == 0 ? 0 : 1)]
        cam(z) = nd == 0 ? c.camera : retract(camera_chart, z[1:nd])
        f(z) = VgtkCore.reprojection_error(cam(z), retract(chp, SVector{6}(ntuple(i -> z[nd + i], 6))), c.X, c.u)
        z0 = zeros(nd + 6)
        r[(2k - 1):(2k)] = f(z0)
        Jk = ForwardDiff.jacobian(f, z0)
        o = nd + 6 * (v - (nd == 0 ? 1 : 2))
        J[(2k - 1):(2k), 1:nd] = Jk[:, 1:nd]
        J[(2k - 1):(2k), (o + 1):(o + 6)] = Jk[:, (nd + 1):(nd + 6)]
    end
    return r, J
end

function _refine_camera_and_poses(
        camera::CameraModel, poses::AbstractVector, views::AbstractVector;
        free::Tuple{Vararg{Symbol}} = Tuple(p for p in parameter_names(camera) if p !== :skew),
        box::Union{Nothing, NamedTuple} = nothing,
        max_iter::Integer = _dflt("refine_camera_and_poses", "max_iter"),
        ftol::Real = _dflt("refine_camera_and_poses", "ftol"),
        xtol::Real = _dflt("refine_camera_and_poses", "xtol"),
    )
    V = length(views)
    V >= 1 || throw(ArgumentError("refine_camera_and_poses: no views"))
    names = parameter_names(camera)
    for p in free
        p in names || throw(ArgumentError("refine_camera_and_poses: the camera has no parameter $(repr(p)); its parameters are $(join(String.(names), ", "))"))
    end
    chc = isempty(free) ? nothing : restrict(ParameterManifold(camera; box), free)
    nd = chc === nothing ? 0 : tangent_size(chc)
    pose_charts = _CameraPosePose[LeftRigidManifold6(SMatrix{3, 3, Float64, 9}(rotation(p)), SVector{3, Float64}(translation(p))) for p in poses]
    o = chc === nothing ? 0 : 1                                              # the poses' variable offset
    N = chc === nothing ? 1 : 2
    costs = _CameraPoseResidual{N, typeof(camera)}[]
    for (v, cs) in enumerate(views), (j, c) in enumerate(cs)
        X = SVector{3, Float64}(first(c)[1], first(c)[2], first(c)[3]); u = SVector{2, Float64}(last(c)[1], last(c)[2])
        (all(isfinite, X) && all(isfinite, u)) || throw(ArgumentError("refine_camera_and_poses: view $v, correspondence $j: non-finite point or pixel"))
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
    result = NLLSsolver.optimize!(problem, nlls_options(; max_iter, ftol, xtol))
    chc_out = chc === nothing ? nothing : problem.variables[1]
    poses_out = [problem.variables[o + v] for v in 1:V]
    camera_out = chc_out === nothing ? camera : model(chc_out)
    all(ch -> all(isfinite, ch.R) && all(isfinite, ch.t), poses_out) && all(isfinite, parameter_vector(camera_out)) ||
        throw(ErrorException("refine_camera_and_poses: the LM step is not finite"))

    r, J = _camera_poses_rows_and_jacobian(costs, chc_out, poses_out, nd)
    code = nlls_retcode(result, J, r; tol = _CHART_LM_STATIONARITY_TOL, ftol)
    S = sum(abs2, r)
    ν = 2n - p
    out = [relabel(_frames(p0)..., EuclideanMap(RotMatrix{3, Float64}(ch.R), SVector{3, Float64}(ch.t))) for (p0, ch) in zip(poses, poses_out)]
    return (; camera = camera_out, poses = out), code, result.niterations, sqrt(S / n), n, CameraPosesRefineMetadata(r, S / ν, ν, free)
end
