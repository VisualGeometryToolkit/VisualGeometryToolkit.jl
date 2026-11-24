# =============================================================================
# RotationCenterViews — CameraPosesProblem' views of a camera turned about
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
# and the offset as plain 3-vectors; each correspondence is point i of its view's
# BearingPoseProblem (the existing bearing residual, bearings.md: the tangent-plane
# logarithm of the predicted bearing at the dot's back-projected one, whitened into
# pixels by its back-projection Jacobian M), evaluated at the view's pose
# (R_k, d − R_k c) inside the block; the solve is Core's bundle LM (`block_lm`). One
# residual for EVERY camera (2026-11-12, owner, (B)(ii)): the camera is used only to
# back-project the dots, once — never projected inside the LM, so no lens's domain or
# spline window can stop it (the pixel residual's failure on SplineLens fisheyes
# whose frame differs from the start's). Bearings take no normalization (bearings.md);
# the target points enter as given (frame in = frame out).
#
# THE GAUGE. With the target points fixed in the target frame and c, d and the
# R_k absolute, no continuous motion leaves the cost unchanged — PROVIDED the
# relative rotations R₁ᵀR_k do not all share one axis. If they do (the camera
# turned about one axis, or only two views), a shift δ of c along that axis, with
# d moved by R_k δ (then the same for every k), fits identically: a
# one-parameter gauge. It is not pinned: such a set is refused (`ArgumentError`),
# measured by the second singular value of the stacked rotation vectors
# vanishing to rounding. The camera is held: a free camera's gauges are not the
# views.
# =============================================================================

const _ViewRotationChart = VgtkCore.SO3Manifold{Float64}
# A rotation-centre problem variable: a view's rotation chart, the centre or the offset.
const _CenterVar = Union{SVector{3, Float64}, _ViewRotationChart}

"""
    rotation_center_pose(R, center, offset) -> EuclideanMap

The pose (target → camera) of one view of [`RotationCenterViews`](@ref): the
camera turned by `R` (target → camera) about the rotation centre `center`
(target frame), its projection centre at `offset` from it (camera frame), so
`X ↦ R (X − center) + offset`, i.e. `EuclideanMap(R, offset − R·center)`. The
pose the rotation-centre refine evaluates inside its bearing block, and the one
a caller builds from a result's views:
`rotation_center_pose(v.rotations[k], v.center, v.offset)` with
`v = result.u.views`. Plain numbers (the target's length unit).
"""
@inline rotation_center_pose(R, center, offset) = EuclideanMap(R, offset - SMatrix{3, 3}(R) * center)

"""
    RotationCenterViews(rotations, center, offset)

The views of a camera turned about one fixed point (a tripod head) for
[`CameraPosesProblem`](@ref): `rotations` maps each view `K` to its
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

"""
    AbstractViewResidual

The residual of [`CameraPosesProblem`](@ref) on [`RotationCenterViews`](@ref), its fourth argument:

- `BearingResidual()` (the default): each dot back-projected ONCE to its bearing, the tangent-plane angle to the
  predicted bearing whitened into pixels by the dot's back-projection Jacobian — never a projection inside the solve,
  so it converges from a rotation-centre start whatever the lens's domain;
- `ReprojectionResidual()`: that bearing refine, then a SECOND stage from its estimate on the exact pixel reprojection
  error, `project(camera, R_k (X − c) + d) − u` (Core's `reprojection_error`; a SplineLens camera projected by inverting
  its backward spline, its forward dropped, Core's implicit-function derivatives). A candidate update that leaves a
  dot's prediction undefined (off the lens's domain) costs +Inf and is rejected by the LM, never clamped. The dots of
  the second stage are the first stage's rows that project at its estimate; those that do not are
  `metadata.without_projection`. The result's views, retcode, rms and record are the second stage's.
"""
abstract type AbstractViewResidual end

"""
    BearingResidual()

The default [`AbstractViewResidual`](@ref) of [`CameraPosesProblem`](@ref) on [`RotationCenterViews`](@ref): the
bearing refine alone.
"""
struct BearingResidual <: AbstractViewResidual end

"""
    ReprojectionResidual()

The [`AbstractViewResidual`](@ref) of [`CameraPosesProblem`](@ref) on [`RotationCenterViews`](@ref) that adds the
second stage on the exact pixel reprojection error; the dots that do not project at the bearing estimate are
`metadata.without_projection`.
"""
struct ReprojectionResidual <: AbstractViewResidual end

# The rotation-centre refusal: at least three views, and relative rotations R₁ᵀR_k spanning two axes — the second
# singular value of their stacked rotation vectors (log_so3) above rounding (√eps of the first).
function _check_rotation_center_gauge(rotations)
    length(rotations) >= 3 || throw(ArgumentError(
        "CameraPosesProblem: rotation-centre views need at least 3 views (two views turn about one axis: the centre slides along it)"))
    R₁ = SMatrix{3, 3, Float64, 9}(first(rotations))
    ω = reduce(hcat, [SVector{3, Float64}(log_so3(R₁' * SMatrix{3, 3, Float64, 9}(R))) for R in rotations])
    σ = LinearAlgebra.svdvals(ω)
    σ[2] > sqrt(eps(Float64)) * max(σ[1], eps(Float64)) || throw(ArgumentError(
        "CameraPosesProblem: the rotation-centre views turned about one axis (singular values of the relative rotation " *
        "vectors $(σ) rad): the centre slides along it against the offset, a gauge the data do not fix"))
    return nothing
end

"""
    held_parameters(camera) -> Tuple{Vararg{Symbol}}

The `fixed` that holds every parameter of `camera` in a refine
([`CameraPosesProblem`](@ref)): a parametric camera's parameter names
(`parameter_names`), a SplineLens camera's `(:controls,)`, its route's one
parameter group. A refine that requires a held camera, the rotation-centre
views', takes it as `fixed = held_parameters(camera)`.

`camera` is a `CameraModel` or a `BoundedCamera` of one.
"""
function held_parameters(camera)
    cm = _camera_model(camera)
    return VgtkCore.Cameras.IntrinsicsRole(lens(cm)) isa VgtkCore.Cameras.FreeIntrinsics ? Tuple(parameter_names(cm)) : (:controls,)
end

# The camera parameters a rotation-centre refine would leave free under `kw.fixed` (it requires none): a parametric
# camera's by `_camera_chart`; a SplineLens camera's one group, `(:controls,)` (`held_parameters`), held or free whole.
_free_camera_parameters(cm, kw) = _free_camera_parameters(VgtkCore.Cameras.IntrinsicsRole(cm.lens), cm, kw)
_free_camera_parameters(::VgtkCore.Cameras.FreeIntrinsics, cm, kw) = first(_camera_chart(cm, kw.fixed, kw.box))
function _free_camera_parameters(_role, cm, kw)
    kw.box === nothing || throw(ArgumentError("CameraPosesProblem: a `box` bounds a camera's refined parameters, and a rotation-centre refine holds the camera"))
    held = held_parameters(cm)
    kw.fixed == held && return ()
    kw.fixed == () && return held          # the camera free: refused by the caller, naming what is free
    throw(ArgumentError("CameraPosesProblem: a SplineLens camera's one parameter group is its controls: `fixed = $(held)` " *
                        "holds the camera (`held_parameters(camera)`), got fixed = $(kw.fixed)"))
end

"""
    solve(RefineProblem(CameraPosesProblem(correspondences; fixed, residual = BearingResidual()), (; camera, views::RotationCenterViews)),
          LevenbergMarquardt(; optimization_strategy); scale | width, stops)
        -> FitResult{<:NamedTuple{(:camera, :views)}, RotationCenterRefineMetadata}

The bundle under the rotation-centre view parametrization: the views of a camera
turned about one fixed point ([`RotationCenterViews`](@ref): per view its
rotation `R_k`, shared by all the centre `c` and the offset `d`, view `k`'s pose
`(R_k, d − R_k c)`), refined from the guess `views` against `correspondences` — per view
`K` its correspondences `X => u` (target point, pixel; the key sets must match).
The residual is the BEARING residual for every camera: each dot back-projected
once by the camera (`backproject_with_jacobian`) to its unit bearing, the
residual the tangent-plane angle to the predicted bearing of `R_k (X − c) + d`,
whitened into pixels by the dot's back-projection Jacobian (the
`BearingPoseProblem`'s, bearings.md) — so rms, residuals and widths are pixels,
and no point is projected through the camera inside the solve. A dot the camera
does not back-project (no finite bearing), or whose back-projection Jacobian
cannot whiten the residual (not finite, not tangent, singular), makes no row and
is counted: `result.metadata.n_without_bearing` and `result.metadata.without_bearing` (its
`(view, correspondence index)` pairs). The variables: one rotation chart per view and the
centre and the offset; the camera is HELD — `fixed` must name every parameter of a
parametric `camera` (`fixed = (:controls,)` for a SplineLens camera, its route's
spelling): [`held_parameters`](@ref)`(camera)` is that `fixed`; a free camera is an `ArgumentError`. The keywords, the solve, the
retcode and its stop rule, rms, `n` and the robust record are the bundle's
(`dof = 2n − 6 − 3N`; `scale = result` records `ScaleUpstream`).

The gauge: none when the views' relative rotations span two axes. Every
relative rotation about one axis (or fewer than three views) leaves the centre
free along it against the offset, and is an `ArgumentError`.

The estimate is `(; camera, views)`: the camera as given, and the refined
`RotationCenterViews` keyed like the input.

`residual` (the fourth argument): [`BearingResidual`](@ref)`()`, the default, the bearing residual above;
[`ReprojectionResidual`](@ref)`()` that refine, then a second stage from its estimate on the exact pixel
reprojection error (the result's views, retcode and record are the second stage's).
"""
function _camera_poses(
        camera::_RefinedCamera, views::RotationCenterViews{K},
        correspondences::AbstractDict{K, <:AbstractVector{<:Correspondence{<:StaticVector{3, <:Real}, <:StaticVector{2}}}},
        residual::AbstractViewResidual = BearingResidual();
        kwargs...
    ) where {K}
    ids = block_order(correspondences, views.rotations)
    kw = _bundle_keywords(; kwargs...)
    cm = _camera_model(camera)
    free = _free_camera_parameters(cm, kw)
    isempty(free) || throw(ArgumentError(
        "CameraPosesProblem: rotation-centre views refine the views of a HELD camera: `fixed` must name every camera " *
        "parameter, `fixed = held_parameters(camera)` (free: $(free))"))
    _check_rotation_center_gauge([views.rotations[k] for k in ids])
    N = length(ids)
    vars0 = _center_variables(views, ids)
    blocks, without = _rotation_center_bearing_blocks(cm, ids, correspondences)
    p = 6 + 3N
    2length(blocks) > p || throw(ArgumentError(
        "CameraPosesProblem: underdetermined: $(length(blocks)) correspondences ($(2length(blocks)) rows) for $p unknowns"))
    sol = _block_lm(blocks, vars0, p, kw)
    c, d, Rout = _center_read_back(K, sol, ids)
    out = RotationCenterViews{K}(Rout, c, d)
    residual isa ReprojectionResidual && return _rotation_center_reprojection(camera, cm, out, ids, correspondences, without, kw)
    md = RotationCenterRefineMetadata{typeof(kw.lo)}(sol.residuals, sol.S / sol.ν, sol.ν, _row_norms(sol.residuals), sol.weights,
        VgtkCore.scale_record(kw)...,
        length(without), without)
    return FitResult((; camera, views = out), sol.code, sol.rule, SolveStats(; nsteps = sol.iterations), sol.rms, sol.n, md)
end


# The pixel stage of `ReprojectionResidual()`: from the bearing estimate `views`, one `_BoardViewResidual` per dot on the
# view's rotation chart (2 + v), the centre (1) and the offset (2), the camera held and projected exactly
# (`_without_forward` for a SplineLens camera), the bundle's solve at the same keywords.
function _rotation_center_reprojection(camera, cm, views::RotationCenterViews{K}, ids, correspondences, without_bearing, kw) where {K}
    pc = _exact_projection(cm)
    bad = Set(without_bearing)
    blocks = NLLSsolver.AbstractResidual[]
    without_projection = Tuple{K, Int}[]
    for (v, k) in enumerate(ids)
        P = rotation_center_pose(views.rotations[k], views.center, views.offset)
        for (j, c) in enumerate(correspondences[k])
            (k, j) in bad && continue
            X, u = _correspondence(v, j, c)
            if all(isfinite, VgtkCore.reprojection_error(pc, P, X, u))
                push!(blocks, _BoardViewResidual{3, _ROTATION_CENTER_VIEW, typeof(pc)}(SVector(2 + v, 1, 2), pc, _IDENTITY_POSE, _IDENTITY_POSE, X, u))
            else
                push!(without_projection, (k, j))
            end
        end
    end
    N = length(ids); p = 6 + 3N
    2length(blocks) > p || throw(ArgumentError(
        "CameraPosesProblem: underdetermined pixel stage: $(length(blocks)) correspondences project at the bearing estimate ($(2length(blocks)) rows) for $p unknowns"))
    vars0 = _center_variables(views, ids)
    sol = _block_lm(blocks, vars0, p, kw)
    c, d, Rout = _center_read_back(K, sol, ids)
    md = RotationCenterRefineMetadata{typeof(kw.lo)}(sol.residuals, sol.S / sol.ν, sol.ν, _row_norms(sol.residuals), sol.weights,
        VgtkCore.scale_record(kw)...,
        length(without_bearing), without_bearing, without_projection)
    return FitResult((; camera, views = RotationCenterViews{K}(Rout, c, d)), sol.code, sol.rule, SolveStats(; nsteps = sol.iterations), sol.rms, sol.n, md)
end
# The rotation-centre bundle's variables, in block order: the centre, the offset, then each view's rotation chart.
function _center_variables(views::RotationCenterViews, ids)
    vars0 = _CenterVar[views.center, views.offset]
    foreach(k -> push!(vars0, _ViewRotationChart(SMatrix{3, 3, Float64, 9}(views.rotations[k]))), ids)
    return vars0
end
# The estimate read back from those variables (the centre, the offset, the views' rotations), refused when not finite.
function _center_read_back(::Type{K}, sol, ids) where {K}
    c, d = sol.variables[1]::SVector{3, Float64}, sol.variables[2]::SVector{3, Float64}
    Rout = Dict{K, RotMatrix{3, Float64, 9}}(k => RotMatrix{3, Float64}(SMatrix{3, 3, Float64, 9}(model(sol.variables[2 + v]::_ViewRotationChart)))
                                             for (v, k) in enumerate(ids))
    (all(isfinite, c) && all(isfinite, d) && all(R -> all(isfinite, R), values(Rout))) ||
        throw(ErrorException("CameraPosesProblem: the LM step is not finite"))
    return c, d, Rout
end

# The camera a pixel residual projects through exactly: a SplineLens camera's backward inverted (its forward dropped), any
# other camera as it is.
_exact_projection(cm::CameraModel) = lens(cm) isa SplineLens ? _without_forward(cm) : cm
const _IDENTITY_POSE = EuclideanMap(SMatrix{3, 3, Float64, 9}(I), zero(SVector{3, Float64}))

# `_BoardViewResidual` on a rotation-centre view: its variables the view's rotation chart, the centre and the offset (no
# pose chart), the view's pose `(R, d − R c)` (`rotation_center_pose`), the camera and the board held.
const _ROTATION_CENTER_VIEW = (false, :rotation_center, false)
NLLSsolver.getvars(r::_BoardViewResidual{3, _ROTATION_CENTER_VIEW}, vars::Vector{_CenterVar}) =
    (vars[r.varind[1]]::_ViewRotationChart, vars[r.varind[2]]::SVector{3, Float64}, vars[r.varind[3]]::SVector{3, Float64})
function NLLSsolver.computeresidual(r::_BoardViewResidual{3, _ROTATION_CENTER_VIEW}, ch, c, d)
    e = VgtkCore.reprojection_error(r.camera, rotation_center_pose(model(ch), c, d), r.X, r.u)
    return r.sqrt_weight == 1.0 ? e : r.sqrt_weight * e
end

"""
    RotationCenterRefineMetadata

The metadata of [`CameraPosesProblem`](@ref) on [`RotationCenterViews`](@ref): the bundle's robust record, as
[`CameraPosesRefineMetadata`](@ref)'s and in its units, PIXELS — each dot's bearing row is its tangent residual whitened
into pixels by the back-projection Jacobian, so `residuals` (px), `s2` (px², `s2(result)`), `dof` (`dof(result)`), `residual_norms` (one per
correspondence, px), `weights`, `scale` (px) and `loss_width` (px), with `scale_source` — and the dots without a bearing
(the camera does not back-project them, or their back-projection Jacobian cannot whiten the residual), which make no
row: `n_without_bearing` and `without_bearing` (their `(view, correspondence index)` pairs), read by field.
"""
struct RotationCenterRefineMetadata{L, W <: AbstractVector{Float64}, P <: AbstractVector} <: AbstractFitMetadata{L}
    residuals::Vector{Float64}
    s2::Float64
    dof::Int
    residual_norms::Vector{Float64}
    weights::W
    scale::Float64
    scale_source::ScaleSource
    loss_width::Float64
    n_without_bearing::Int
    without_bearing::P
    without_projection::P       # `ReprojectionResidual()`'s dots that do not project at the bearing estimate (empty otherwise)
end
RotationCenterRefineMetadata{L}(residuals, s2, dof, norms, weights::W, scale, source, loss_width, n, without::P,
                                without_projection = empty(without)) where {L, W, P} =
    RotationCenterRefineMetadata{L, W, P}(residuals, Float64(s2), Int(dof), norms, weights, Float64(scale), source, Float64(loss_width), Int(n),
                                          without, convert(P, without_projection))

# The bearing block of one correspondence: point i of its view's BearingPoseProblem at the view's pose (R_k, d − R_k c),
# on the view's rotation chart (variable 2 + v), the centre (1) and the offset (2). The residual is the problem's own
# (`_bearing_residual`, eltype-generic), so NLLSsolver's ForwardDiff differentiates it as it does the bundle's blocks;
# a robust refine scales it by `sqrt_weight` or robustifies it by `robust` (`with_robustifier`).
struct _BearingCenterBlock{P, K} <: NLLSsolver.AbstractResidual
    varind::SVector{3, Int}
    problem::P
    i::Int
    sqrt_weight::Float64
    robust::K
end
_BearingCenterBlock(varind::SVector{3, Int}, problem, i::Int) =
    _BearingCenterBlock{typeof(problem), NLLSsolver.NoRobust}(varind, problem, i, 1.0, NLLSsolver.NoRobust())
NLLSsolver.robustkernel(r::_BearingCenterBlock) = r.robust
with_robustifier(r::_BearingCenterBlock{P}, sqrt_weight::Float64, robust::K) where {P, K} = _BearingCenterBlock{P, K}(r.varind, r.problem, r.i, sqrt_weight, robust)
NLLSsolver.ndeps(::_BearingCenterBlock) = NLLSsolver.static(3)
NLLSsolver.nres(::_BearingCenterBlock) = NLLSsolver.static(2)
NLLSsolver.varindices(r::_BearingCenterBlock) = r.varind
# The rotation-centre problem's variable vector is `Vector{_CenterVar}` exactly (`CameraPosesProblem`' `vars0`), and
# only it: these methods match no other problem's variables, so a call on another problem's (an abstract call site in
# the LM's generic code, which inference splits over every block type) finds no method of this block rather than an
# impossible typeassert.
NLLSsolver.getvars(r::_BearingCenterBlock, vars::AbstractVector{_CenterVar}) =
    (vars[r.varind[1]]::_ViewRotationChart, vars[r.varind[2]]::SVector{3, Float64}, vars[r.varind[3]]::SVector{3, Float64})
Base.eltype(::_BearingCenterBlock) = Float64
function NLLSsolver.computeresidual(r::_BearingCenterBlock, ch, c, d)
    P = rotation_center_pose(model(ch), c, d)
    e = _bearing_residual(r.problem, rotation(P), translation(P), r.i)
    return r.sqrt_weight == 1.0 ? e : r.sqrt_weight * e
end
NLLSsolver.computecost(vars::AbstractVector{_CenterVar}, r::_BearingCenterBlock) = finite_cost(NLLSsolver.computecost(r, NLLSsolver.getvars(r, vars)...))

# The bearing blocks of the rotation-centre views (block order `ids`): each view's dots back-projected by the camera to
# their unit bearings and Jacobians (`backproject_with_jacobian`), its BearingPoseProblem of them (the `M` column: rows
# in pixels), one block per dot; and the dots without a bearing, `(view, index)`, counted by the caller: no finite bearing,
# or a Jacobian the problem could not whiten by (`pixel_whitening_defect`: not finite, not tangent, singular), found before any
# block is built so that one bad dot never aborts the refine.
function _rotation_center_bearing_blocks(cm, ids::AbstractVector{K}, correspondences) where {K}
    blocks = NLLSsolver.AbstractResidual[]
    without = Tuple{K, Int}[]
    for (v, k) in enumerate(ids)
        X = SVector{3, Float64}[]; F = SVector{3, Float64}[]; Ms = SMatrix{3, 2, Float64, 6}[]
        for (j, c) in enumerate(correspondences[k])
            Xj, u = _correspondence(v, j, c)
            f, M = VgtkCore.backproject_with_jacobian(cm, u)
            if all(isfinite, f) && !iszero(norm(f)) && (fu = normalize(SVector{3, Float64}(f));
                                                       pixel_whitening_defect(fu, M) === nothing)
                push!(X, Xj); push!(F, fu); push!(Ms, SMatrix{3, 2, Float64, 6}(M))
            else                # no finite bearing, or a Jacobian that cannot whiten it (BearingPoseProblem's own test)
                push!(without, (k, j))
            end
        end
        isempty(X) && continue
        prob = BearingPoseProblem(VgtkCore.Correspondences(X, F; M = Ms))
        foreach(i -> push!(blocks, _BearingCenterBlock(SVector(2 + v, 1, 2), prob, i)), eachindex(X))
    end
    return blocks, without
end
