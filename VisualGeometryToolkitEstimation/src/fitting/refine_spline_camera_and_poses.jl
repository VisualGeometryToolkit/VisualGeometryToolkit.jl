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
# THE ROUTE. The held problem's filter (`_spline_problem`: the correspondences on an
# absent window, or not projected next to one, dropped and counted; any other that
# does not project, or projects where the lattice folds, named), then the bundle's one
# held-camera solve (`_refine_views`, `_refine_bundle`): one block per correspondence on
# its view's pose alone, the whole camera a constant through its `project`, the
# bundle's strategy included. The camera comes back AS GIVEN, its forward spline
# included (VisualGeometryToolkitEstimation 0.11.0): its controls are no variable, so
# there is no gauge to fix and no frame to turn.
# =============================================================================

# The refusal of the retired image-space refine of a SplineLens camera's controls: `what` names the call that asked
# for it (the default `fixed = ()`, or a retired keyword of that route).
_refuse_spline_image_space(what::AbstractString) = throw(ArgumentError(
    "refine_camera_and_poses: the image-space refine of a SplineLens camera's controls is retired " *
        "(VisualGeometryToolkitEstimation 0.10.0), and $what asks for it: a SplineLens camera is refined only by " *
        "back-projection, or held. Hold it with `fixed = (:controls,)` (the poses alone), or fit its backward on " *
        "back-projected rays (`fit_camera(SplineLens, …)`, `pspline_grid`)"))

# The refusal of the forward refit's keywords (`refit_forward`, `forward_tolerance`), retired in 0.11.0 with the
# refit itself: the held camera comes back as given, its forward spline kept.
_refuse_forward_refit(value::RetiredKeyword, ::Symbol) = nothing
_refuse_forward_refit(_value, name::Symbol) = throw(ArgumentError(
    "refine_camera_and_poses: the keyword `$name` is retired (VisualGeometryToolkitEstimation 0.11.0): a held SplineLens " *
        "camera comes back as given, its forward spline kept, so nothing is refitted; give a camera its forward with " *
        "`fit_forward_spline(camera; tolerance)`"))

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

# The held problem's filter for a SplineLens camera `cam` (no forward) at the guess poses: the observations the bundle
# refines, `(view, 1, X, u)` each (the one held board), and the counts of the correspondences dropped on an absent window
# and of those the camera does not project within a cell of one (it images them onto the absent nodes: possibly
# inliers), or the `_NotProjected` of the first other correspondence that does not project (or projects where the
# lattice folds). The projections run batched on the device (Core's `_spline_projections`).
function _spline_problem(cam::CameraModel, poses::AbstractVector, views::AbstractVector)
    l = cam.lens
    b = l.backward
    gsz = size(b.controls)
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
    pose_charts = _pose_charts(poses)
    Xcs = [(P = model(pose_charts[c[1]]); rotation(P) * c[3] + translation(P)) for c in cands]
    proj = VgtkCore.Cameras._spline_projections(l, Xcs, nothing; fold = true)
    obs = Tuple{Int, Int, SVector{3, Float64}, SVector{2, Float64}}[]
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
        push!(obs, (v, 1, X, u))
    end
    return (; obs, dropped, unprojected)
end

# Whether a window within one cell of window `(ix, iy)` holds an absent node.
_near_absent(::Nothing, ix, iy, gsz) = false
_near_absent(present::AbstractMatrix{Bool}, ix, iy, (gh, gw)) =
    any(!VgtkCore.Cameras._window_present(present, jx, jy) for jx in max(3, ix - 1):min(gw - 1, ix + 1), jy in max(3, iy - 1):min(gh - 1, iy + 1))

# The SplineLens route: the camera held (`fixed = (:controls,)`), checked first — a call that would free the controls
# (the default `fixed = ()`, or a `penalty`) is refused by name (`_refuse_spline_image_space`). The rest are the
# bundle's keywords, read by its one reader (`_bundle_keywords`: the strategy, its scale or width, the LM's settings).
# The camera is projected without its forward (the backward's own solve, exact) and returned as given.
function _refine_camera_and_poses(::VgtkCore.Cameras.GaugeIntrinsics, camera::_RefinedCamera, poses::AbstractVector, views::AbstractVector;
                                  fixed::Tuple{Vararg{Symbol}} = (),
                                  free = RetiredKeyword(), fix = RetiredKeyword(), fixed_params = RetiredKeyword(), hold = RetiredKeyword(),   # refused, naming `fixed`
                                  penalty = RetiredKeyword(),   # the retired image-space refine's smoothness weight: refused by name
                                  box::Union{Nothing, NamedTuple} = nothing,
                                  refit_forward = RetiredKeyword(), forward_tolerance = RetiredKeyword(),   # the retired forward refit's
                                  kwargs...)
    _refuse_held_spellings(free, fix, fixed_params, hold)
    penalty isa RetiredKeyword || _refuse_spline_image_space("a `penalty`")
    fixed == () && _refuse_spline_image_space("`fixed = ()` (the default: the controls free)")
    fixed == (:controls,) || throw(ArgumentError(
        "refine_camera_and_poses: a SplineLens camera's `fixed` is (:controls,) (the camera held, the poses refined), got $fixed"))
    box === nothing || throw(ArgumentError("refine_camera_and_poses: a SplineLens camera is held; a `box` has nothing to bound"))
    _refuse_forward_refit(refit_forward, :refit_forward)
    _refuse_forward_refit(forward_tolerance, :forward_tolerance)
    kw = _bundle_keywords(; fixed, kwargs...)
    back = _without_forward(_camera_model(camera))
    p = _spline_problem(back, poses, views)
    p isa _NotProjected && throw(ArgumentError(_not_projected_message(p)))   # the public error of a point that does not project
    return _refine_views(kw, camera, back, poses, p.obs, ((), nothing);
                         spline = (; n_dropped_window = p.dropped, n_dropped_unprojected = p.unprojected))
end

# A SplineLens camera as the route projects it: its forward spline dropped (the projection is the backward's solve), the
# camera itself without one.
_without_forward(cam::CameraModel) = (l = cam.lens; l.forward === nothing ? cam : CameraModel(cam.intrinsics, SplineLens(l.backward; present = l.present)))
