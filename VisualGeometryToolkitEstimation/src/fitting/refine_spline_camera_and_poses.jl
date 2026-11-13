# =============================================================================
# refine_camera_and_poses — the route of a SplineLens camera (gauge intrinsics)
# =============================================================================
#
# The same bundle as refine_camera_and_poses.jl — Σ ‖project(camera, X_c) − u‖² over
# the views' correspondences — for a camera whose K is a frozen gauge and whose
# parameters are its backward lattice's controls (a SplineLens). Core's
# `IntrinsicsRole` trait selects this route.
#
# THE CAMERA IS HELD. A SplineLens camera is refined only by back-projection
# (`fit_camera(SplineLens, …)`, `pspline_grid`: linear fits on the camera's rays) or
# held; its controls are never variables of an image-space solve. The image-space
# refine of the controls (one block per correspondence on its 4 × 4 window, a
# third-difference penalty, its GCV weight, window-rebuild rounds) was retired by the
# owner's ruling of 2026-11-13 (VisualGeometryToolkitEstimation 0.10.0): a call that
# would free the controls is refused by name (`_refuse_spline_image_space`).
#
# THE BLOCKS. One per correspondence, on its view's pose alone: the parametric route's
# block (`_BoardViewResidual`), the whole camera a constant through its `project`. No
# window, no node, no rounds. The poses are VGC's `LeftRigidManifold6`, every pose free
# (the camera fixes the gauge).
#
# THE FRAME. The refine runs on the camera turned onto its optical axis
# (`_refine_spline_camera_and_poses`: Core's `axis_gauged`, K's principal point that
# axis's pixel), the guess poses turned with it, so every image is kept; the camera
# comes back as the entry gauged it.
# =============================================================================

# The refusal of the retired image-space refine of a SplineLens camera's controls: `what` names the call that asked
# for it (the default `fixed = ()`, or a retired keyword of that route).
_refuse_spline_image_space(what::AbstractString) = throw(ArgumentError(
    "refine_camera_and_poses: the image-space refine of a SplineLens camera's controls is retired " *
        "(VisualGeometryToolkitEstimation 0.10.0), and $what asks for it: a SplineLens camera is refined only by " *
        "back-projection, or held. Hold it with `fixed = (:controls,)` (the poses alone), or fit its backward on " *
        "back-projected rays (`fit_camera(SplineLens, …)`, `pspline_grid`)"))

# A correspondence the camera does not project at the guess (or projects where the lattice folds), and the point.
# A status, not an exception: the refine's boundary states it (`_not_projected_message`).
struct _NotProjected
    view::Int
    correspondence::Int
    X::SVector{3, Float64}
    folds::Bool
end
function _not_projected_message(e::_NotProjected)
    what = e.folds ? "the lattice folds where the point $(e.X) projects at the guess (the backward's Jacobian is singular there)" :
        "the point $(e.X) does not project at the guess (behind the camera or outside the lens's domain)"
    return "refine_camera_and_poses: view $(e.view), correspondence $(e.correspondence): " * what
end

# The problem of a held SplineLens camera `cam` (no forward) at the guess poses: the pose variables, one block per
# correspondence on its pose (the parametric route's `_BoardViewResidual`, the whole camera a constant), and the counts
# of the correspondences dropped on an absent window and of those the camera does not project within a cell of one
# (it images them onto the absent nodes: possibly inliers), or the `_NotProjected` of the first other correspondence
# that does not project (or projects where the lattice folds). The projections run batched on the device (Core's
# `_spline_projections`).
function _spline_problem(cam::CameraModel, poses::AbstractVector, views::AbstractVector)
    l = cam.lens
    b = l.backward
    gsz = size(b.controls)
    pose_charts = _pose_charts(poses)
    cands = Tuple{Int, Int, SVector{3, Float64}, SVector{2, Float64}, Int, Int}[]
    dropped = 0
    for (v, cs) in enumerate(views), (j, c) in enumerate(cs)
        X, u = _correspondence(v, j, c)
        ox, oy, _, _ = spline_window(b.area, gsz, VgtkCore.normalized_point(cam.intrinsics, u))
        if !VgtkCore.Cameras._window_present(l.present, ox, oy)   # detected on an absent window: the lens does not image it
            dropped += 1
            continue
        end
        push!(cands, (v, j, X, u, ox, oy))
    end
    Xcs = [(P = model(pose_charts[c[1]]); rotation(P) * c[3] + translation(P)) for c in cands]
    proj = VgtkCore.Cameras._spline_projections(l, Xcs, nothing; fold = true)
    found = Tuple{SVector{3, Float64}, SVector{2, Float64}, Int}[]
    unprojected = 0
    for (q, (v, j, X, u, ox, oy)) in enumerate(cands)
        ok, g = proj.ok[q], proj.fold[q]
        if !ok && _near_absent(l.present, ox, oy, gsz)
            unprojected += 1
            continue
        end
        ok || return _NotProjected(v, j, X, false)
        # a projection where the lattice folds (the backward's Jacobian there exactly singular, or not finite): no
        # Jacobian of the point exists, so it is not linearized
        (isfinite(g) && g != 0) || return _NotProjected(v, j, X, true)
        push!(found, (X, u, v))
    end
    world = EuclideanMap(RotMatrix{3, Float64}(one(SMatrix{3, 3, Float64, 9})), zero(SVector{3, Float64}))   # the held board
    F = (false, true, false)
    costs = _BoardViewResidual{1, F, typeof(cam), NLLSsolver.NoRobust, Nothing}[
        _BoardViewResidual{1, F, typeof(cam)}(SVector(v), cam, world, world, X, u; chart = Nothing) for (X, u, v) in found
    ]
    return (; variables = pose_charts, costs, dropped, unprojected)
end

# Whether a window within one cell of window `(ix, iy)` holds an absent node.
_near_absent(::Nothing, ix, iy, gsz) = false
_near_absent(present::AbstractMatrix{Bool}, ix, iy, (gh, gw)) =
    any(!VgtkCore.Cameras._window_present(present, jx, jy) for jx in max(3, ix - 1):min(gw - 1, ix + 1), jy in max(3, iy - 1):min(gh - 1, iy + 1))

# The forward spline of a SplineLens camera, over its imaged region: `fit_forward_spline` on the camera as it is — not
# `fit_camera` itself, which gauges a gauge-K source onto its optical axis first, and the refine holds its entry's frame
# and K exactly.
function _refit_forward(camera::BoundedCamera, tolerance)
    r = fit_forward_spline(camera; tolerance)
    return estimate(r), (retcode(r), stop_rule(r)), metadata(r).max_px
end

# The SplineLens route: the camera held (`fixed = (:controls,)`), checked first, before the entry's optical-axis fit — a
# call that would free the controls (the default `fixed = ()`, or a `penalty`) is refused by name
# (`_refuse_spline_image_space`) — and least squares only (`optimization_strategy = nothing`, the default here; a robust
# strategy is refused, naming the parametric route that hosts one). The rest are the bundle's keywords, read by its one
# reader (`_bundle_keywords`: the LM's settings; least squares refuses a `scale` or `width`).
function _refine_camera_and_poses(::VgtkCore.Cameras.GaugeIntrinsics, camera::_RefinedCamera, poses::AbstractVector, views::AbstractVector;
                                  fixed::Tuple{Vararg{Symbol}} = (),
                                  free = RetiredKeyword(), fix = RetiredKeyword(), fixed_params = RetiredKeyword(), hold = RetiredKeyword(),   # refused, naming `fixed`
                                  penalty = RetiredKeyword(),   # the retired image-space refine's smoothness weight: refused by name
                                  box::Union{Nothing, NamedTuple} = nothing, optimization_strategy = nothing,
                                  refit_forward::Bool = true, forward_tolerance::Real = _dflt("refine_camera_and_poses", "forward_tolerance"),
                                  kwargs...)
    _refuse_held_spellings(free, fix, fixed_params, hold)
    penalty isa RetiredKeyword || _refuse_spline_image_space("a `penalty`")
    fixed == () && _refuse_spline_image_space("`fixed = ()` (the default: the controls free)")
    fixed == (:controls,) || throw(ArgumentError(
        "refine_camera_and_poses: a SplineLens camera's `fixed` is (:controls,) (the camera held, the poses refined), got $fixed"))
    box === nothing || throw(ArgumentError("refine_camera_and_poses: a SplineLens camera is held; a `box` has nothing to bound"))
    optimization_strategy === nothing || throw(ArgumentError(
        "refine_camera_and_poses: a SplineLens camera is refined by least squares only (optimization_strategy = nothing, " *
            "no scale or width); a robust strategy refines a parametric camera"))
    kw = _bundle_keywords(; fixed, optimization_strategy, kwargs...)
    out = _refine_spline_camera_and_poses(camera, poses, views, kw; refit_forward, forward_tolerance)
    out isa _NotProjected && throw(ArgumentError(_not_projected_message(out)))   # the public error of a point that does not project
    return out
end

# A SplineLens camera as the route holds it: its forward spline dropped (the projection is the backward's solve), the
# camera itself without one.
_without_forward(cam::CameraModel) = (l = cam.lens; l.forward === nothing ? cam : CameraModel(cam.intrinsics, SplineLens(l.backward; present = l.present)))

# The refinement: the estimate, or the `_NotProjected` of a correspondence (the public boundary,
# `_refine_camera_and_poses`, states it). ALWAYS on the camera's optical axis first: the camera turned so ẑ is its axis
# and its principal point the axis's pixel (`optical_axis`, Core's `axis_gauged`: exact), the poses turned with it, so
# every image is kept; the poses are refined from there through that held camera (`_refine_spline_axis_gauged`). A
# forward the given camera held is stale once the frame turns: refitted or dropped, never kept.
function _refine_spline_camera_and_poses(camera::_RefinedCamera, poses::AbstractVector, views::AbstractVector, kw; refit_forward, forward_tolerance)
    entry = optical_axis(camera)
    K = calibration_matrix(estimate(entry))
    R = metadata(entry).rotation
    gauged = VgtkCore.axis_gauged(camera, rotation(R), K)
    turned = [VgtkCore.axis_gauged(pose, rotation(R)) for pose in poses]
    out = _refine_spline_axis_gauged(gauged, turned, views, _camera_model(camera).lens.forward !== nothing, kw; refit_forward, forward_tolerance)
    out isa _NotProjected && return out
    est, sol, record, spline = out
    spline = (; spline..., axis_rotation = R, symmetry_residual = metadata(entry).symmetry_residual)
    return est, record..., sol.iterations, sol.rms, sol.n, _bundle_metadata(sol, (), kw; spline)
end

# The poses refined through the held, axis-gauged camera: the bundle's one solve (`_solve_bundle`) of the held
# problem (`_spline_problem`), then the forward refitted or dropped. Returns the estimate, the solve, its termination
# record and the route's metadata, or the `_NotProjected` of a correspondence.
function _refine_spline_axis_gauged(
        camera::_RefinedCamera, poses::AbstractVector, views::AbstractVector, had_forward::Bool,
        kw = _bundle_keywords(; optimization_strategy = nothing);
        refit_forward::Bool = true, forward_tolerance::Real = _dflt("refine_camera_and_poses", "forward_tolerance"),
    )
    cam = _camera_model(camera)
    forward = !had_forward ? :none : refit_forward ? :refitted : :dropped
    forward === :refitted && !(camera isa BoundedCamera) && throw(
        ArgumentError(
            "refine_camera_and_poses: the SplineLens camera holds a forward spline, which the refine drops and refits over the camera's image: " *
                "give it as a BoundedCamera (`BoundedCamera(camera, Size2(width, height))`), or `refit_forward = false`"
        )
    )
    V = length(views)
    V >= 1 || throw(ArgumentError("refine_camera_and_poses: no views"))
    back = _without_forward(cam)
    p = _spline_problem(back, poses, views)
    p isa _NotProjected && return p
    n = length(p.costs)
    2n > 6V || throw(ArgumentError("refine_camera_and_poses: underdetermined: $n correspondences ($(2n) rows) for $(6V) unknowns"))
    sol = _solve_bundle(p.costs, p.variables, 6V, kw)
    poses_out = [model(P::_CameraPosePose) for P in sol.variables]
    all(_finite_pose, poses_out) || throw(ErrorException("refine_camera_and_poses: the LM step is not finite"))
    record = (sol.code, sol.rule)
    # the camera as the entry gauged it, its stale forward dropped (`back`), then refitted on request
    forward_max_px = nothing
    if camera isa BoundedCamera
        out_camera = VgtkCore.with_model(camera, back)
        if forward === :refitted
            out_camera, fcode, forward_max_px = _refit_forward(out_camera, forward_tolerance)
            # a converged refine whose forward refit missed its tolerance reports the forward's
            VgtkCore.successful_retcode(record[1]) && (record = worst_retcode(record, fcode))
        end
    else
        out_camera = back
    end
    out = [relabel(VgtkCore.map_frames(p0)..., _plain_pose(P)) for (p0, P) in zip(poses, poses_out)]
    spline = (; n_dropped_window = p.dropped, n_dropped_unprojected = p.unprojected, forward, forward_max_px)
    return (; camera = out_camera, poses = out), sol, record, spline
end
