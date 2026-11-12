# =============================================================================
# RotationCenterViews — refine_camera_and_poses' views of a camera turned about
# one fixed point
# =============================================================================
#
# A camera turned on a tripod head about ONE fixed point while the target stood
# still: every view's pose is one rigid motion model, x_cam,k = R_k (X − c) + d —
# the rotation centre c (target frame), the projection centre's offset from it d
# (camera frame, the same for every view) and one rotation per view — 6 + 3N
# unknowns in place of 6N free poses (t_k = d − R_k c). It is a VIEW
# PARAMETRIZATION of the bundle composed of its existing parts: the variables are
# Core's rotation chart per view (SO3Manifold, R(ω) = exp_so3(ω)·R) and the centre
# and the offset as plain 3-vectors; each correspondence is the bundle's own
# residual block (`_BoardViewResidual`, its view `:rotation_center`: the pose
# (R_k, d − R_k c) evaluated inside the block), on the held identity board; the
# solve is the bundle's (`_solve_bundle`).
#
# THE GAUGE. With the target points fixed in the target frame and c, d and the
# R_k absolute, no continuous motion leaves the cost unchanged — PROVIDED the
# relative rotations R₁ᵀR_k do not all share one axis. If they do (the camera
# turned about one axis, or only two views), a shift δ of c along that axis, with
# d moved by R_k δ (then the same for every k), fits identically: a
# one-parameter gauge. It is not pinned: such a set is refused (`ArgumentError`),
# measured by the second singular value of the stacked rotation vectors
# vanishing to rounding. The camera is held: a free camera's gauges are not the
# views'.
# =============================================================================

const _ViewRotationChart = VgtkCore.SO3Manifold{Float64}

# The rotation-centre view's pose, evaluated inside the bundle's residual block: (R, d − R c).
@inline _rotation_center_pose(ch, c, d) = (R = model(ch); EuclideanMap(R, d - SMatrix{3, 3}(R) * c))

"""
    RotationCenterViews(rotations, center, offset)

The views of a camera turned about one fixed point (a tripod head) for
[`refine_camera_and_poses`](@ref): `rotations` maps each view `K` to its
rotation `R_k` (target → camera), `center` is the rotation centre `c` (target
frame) and `offset` the projection centre's offset from it, `d` (camera frame,
shared by every view): view `k`'s pose is `X ↦ R_k (X − c) + d`, i.e.
`(R_k, d − R_k c)`. Plain numbers (the target's length unit).
"""
struct RotationCenterViews{K}
    rotations::Dict{K, RotMatrix{3, Float64, 9}}
    center::SVector{3, Float64}
    offset::SVector{3, Float64}
end
RotationCenterViews(rotations::AbstractDict{K}, center::AbstractVector{<:Real}, offset::AbstractVector{<:Real}) where {K} =
    RotationCenterViews{K}(Dict{K, RotMatrix{3, Float64, 9}}(k => RotMatrix{3, Float64}(SMatrix{3, 3, Float64, 9}(R)) for (k, R) in rotations),
                           SVector{3, Float64}(center), SVector{3, Float64}(offset))

# The rotation-centre refusal: at least three views, and relative rotations R₁ᵀR_k spanning two axes — the second
# singular value of their stacked rotation vectors (log_so3) above rounding (√eps of the first).
function _check_rotation_center_gauge(rotations)
    length(rotations) >= 3 || throw(ArgumentError(
        "refine_camera_and_poses: rotation-centre views need at least 3 views (two views turn about one axis: the centre slides along it)"))
    R₁ = SMatrix{3, 3, Float64, 9}(first(rotations))
    ω = reduce(hcat, [SVector{3, Float64}(log_so3(R₁' * SMatrix{3, 3, Float64, 9}(R))) for R in rotations])
    σ = LinearAlgebra.svdvals(ω)
    σ[2] > sqrt(eps(Float64)) * max(σ[1], eps(Float64)) || throw(ArgumentError(
        "refine_camera_and_poses: the rotation-centre views turned about one axis (singular values of the relative rotation " *
        "vectors $(σ) rad): the centre slides along it against the offset, a gauge the data do not fix"))
    return nothing
end

# The camera parameters a rotation-centre refine would leave free (it requires none): a parametric camera's by
# `_camera_chart`; a SplineLens camera (a gauge K, its controls the parameters) is held by `fixed = (:controls,)`, its
# route's own spelling.
_held_camera_parameters(cm, kw) = _held_camera_parameters(VgtkCore.Cameras.IntrinsicsRole(cm.lens), cm, kw)
_held_camera_parameters(::VgtkCore.Cameras.FreeIntrinsics, cm, kw) = first(_camera_chart(cm, kw.fixed, kw.box))
function _held_camera_parameters(_role, cm, kw)
    kw.box === nothing || throw(ArgumentError("refine_camera_and_poses: a `box` bounds a camera's refined parameters, and a rotation-centre refine holds the camera"))
    return kw.fixed == (:controls,) ? () : (:controls,)
end

"""
    refine_camera_and_poses(camera, views::RotationCenterViews, csponds; fixed, optimization_strategy, scale | width,
                            max_iter, ftol, xtol, max_time = nothing)
        -> FitResult{<:NamedTuple{(:camera, :views)}, CameraPosesRefineMetadata}

The bundle under the rotation-centre view parametrization: the views of a camera
turned about one fixed point ([`RotationCenterViews`](@ref): per view its
rotation `R_k`, shared by all the centre `c` and the offset `d`, view `k`'s pose
`(R_k, d − R_k c)`), refined from the guess `views` against the reprojection
error of `csponds` — per view `K` its correspondences `X => u` (target point,
pixel; the key sets must match). The variables: one rotation chart per view and
the centre and the offset; the camera is HELD — `fixed` must name every
parameter of a parametric `camera` (`fixed = (:controls,)` for a SplineLens
camera, its route's spelling); a free camera is an `ArgumentError`. The
keywords, the solve, the retcode, rms, `n` and the metadata are the bundle's
(`ν = 2n − 6 − 3N`; `scale = result` records `ScaleUpstream`).

The gauge: none when the views' relative rotations span two axes. Every
relative rotation about one axis (or fewer than three views) leaves the centre
free along it against the offset, and is an `ArgumentError`. Every point must
project at the guess.

The estimate is `(; camera, views)`: the camera as given, and the refined
`RotationCenterViews` keyed like the input.
"""
function refine_camera_and_poses(
        camera::_RefinedCamera, views::RotationCenterViews{K},
        csponds::AbstractDict{K, <:AbstractVector{<:Cspond{<:StaticVector{3, <:Real}, <:StaticVector{2}}}};
        kwargs...
    ) where {K}
    ids = block_order(csponds, views.rotations)
    kw = _bundle_keywords(; kwargs...)
    cm = _camera_model(camera)
    free = _held_camera_parameters(cm, kw)
    isempty(free) || throw(ArgumentError(
        "refine_camera_and_poses: rotation-centre views refine the views of a HELD camera: `fixed` must name every camera " *
        "parameter (free: $(free))"))
    _check_rotation_center_gauge([views.rotations[k] for k in ids])
    N = length(ids)
    # the variables: the centre, the offset, then each view's rotation chart, in block order
    vars0 = Union{SVector{3, Float64}, _ViewRotationChart}[views.center, views.offset]
    foreach(k -> push!(vars0, _ViewRotationChart(SMatrix{3, 3, Float64, 9}(views.rotations[k]))), ids)
    world = EuclideanMap(RotMatrix{3, Float64}(one(SMatrix{3, 3, Float64, 9})), zero(SVector{3, Float64}))   # the held board
    F = (false, :rotation_center, false)
    blocks = NLLSsolver.AbstractResidual[]
    for (v, k) in enumerate(ids)
        pose = _rotation_center_pose(_ViewRotationChart(SMatrix{3, 3, Float64, 9}(views.rotations[k])), views.center, views.offset)
        for (j, c) in enumerate(csponds[k])
            X, u = _correspondence(v, j, c)
            all(isfinite, VgtkCore.reprojection_error(cm, pose, X, u)) || throw(ArgumentError(
                "refine_camera_and_poses: view $(repr(k)), correspondence $j: the point $(X) does not project at the guess (behind the camera or outside the lens's domain)"))
            push!(blocks, _BoardViewResidual{3, F, typeof(cm)}(SVector(2 + v, 1, 2), cm, world, world, X, u; chart = Nothing))
        end
    end
    p = 6 + 3N
    n = length(blocks)
    2n > p || throw(ArgumentError("refine_camera_and_poses: underdetermined: $n correspondences ($(2n) rows) for $p unknowns"))
    sol = _solve_bundle(blocks, vars0, p, kw)
    c, d = sol.variables[1]::SVector{3, Float64}, sol.variables[2]::SVector{3, Float64}
    Rout = Dict{K, RotMatrix{3, Float64, 9}}(k => RotMatrix{3, Float64}(SMatrix{3, 3, Float64, 9}(model(sol.variables[2 + v]::_ViewRotationChart)))
                                             for (v, k) in enumerate(ids))
    (all(isfinite, c) && all(isfinite, d) && all(R -> all(isfinite, R), values(Rout))) ||
        throw(ErrorException("refine_camera_and_poses: the LM step is not finite"))
    return FitResult((; camera, views = RotationCenterViews{K}(Rout, c, d)), sol.code, sol.iterations, sol.rms, n,
                     _bundle_metadata(sol, free, kw))
end
