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
# and the offset as plain 3-vectors; each correspondence is point i of its view's
# BearingPoseProblem (the existing bearing residual, bearings.md: the tangent-plane
# logarithm of the predicted bearing at the dot's back-projected one, whitened into
# pixels by its back-projection Jacobian M), evaluated at the view's pose
# (R_k, d − R_k c) inside the block; the solve is the bundle's (`_solve_bundle`). One
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

# The rotation-centre view's pose, evaluated inside the bearing block: (R, d − R c).
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
    kw.fixed == (:controls,) && return ()
    kw.fixed == () && return (:controls,)          # the camera free: refused by the caller, naming what is free
    throw(ArgumentError("refine_camera_and_poses: a SplineLens camera's one parameter group is its controls: `fixed = (:controls,)` " *
                        "holds the camera, got fixed = $(kw.fixed)"))
end

"""
    refine_camera_and_poses(camera, views::RotationCenterViews, csponds; fixed, optimization_strategy, scale | width,
                            max_iter, ftol, xtol, max_time = nothing)
        -> FitResult{<:NamedTuple{(:camera, :views)}, RotationCenterRefineMetadata}

The bundle under the rotation-centre view parametrization: the views of a camera
turned about one fixed point ([`RotationCenterViews`](@ref): per view its
rotation `R_k`, shared by all the centre `c` and the offset `d`, view `k`'s pose
`(R_k, d − R_k c)`), refined from the guess `views` against `csponds` — per view
`K` its correspondences `X => u` (target point, pixel; the key sets must match).
The residual is the BEARING residual for every camera: each dot back-projected
once by the camera (`backproject_with_jacobian`) to its unit bearing, the
residual the tangent-plane angle to the predicted bearing of `R_k (X − c) + d`,
whitened into pixels by the dot's back-projection Jacobian (the
`BearingPoseProblem`'s, bearings.md) — so rms, residuals and widths are pixels,
and no point is projected through the camera inside the solve. A dot the camera
does not back-project (no finite bearing), or whose back-projection Jacobian
cannot whiten the residual (not finite, not tangent, singular), makes no row and
is counted: [`n_without_bearing`](@ref) and [`without_bearing`](@ref) (its
`(view, correspondence index)` pairs). The variables: one rotation chart per view and the
centre and the offset; the camera is HELD — `fixed` must name every parameter of a
parametric `camera` (`fixed = (:controls,)` for a SplineLens camera, its route's
spelling); a free camera is an `ArgumentError`. The keywords, the solve, the
retcode and its stop rule, rms, `n` and the robust record are the bundle's
(`ν = 2n − 6 − 3N`; `scale = result` records `ScaleUpstream`).

The gauge: none when the views' relative rotations span two axes. Every
relative rotation about one axis (or fewer than three views) leaves the centre
free along it against the offset, and is an `ArgumentError`.

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
    blocks, without = _rotation_center_bearing_blocks(cm, ids, csponds)
    p = 6 + 3N
    2length(blocks) > p || throw(ArgumentError(
        "refine_camera_and_poses: underdetermined: $(length(blocks)) correspondences ($(2length(blocks)) rows) for $p unknowns"))
    sol = _solve_bundle(blocks, vars0, p, kw)
    c, d = sol.variables[1]::SVector{3, Float64}, sol.variables[2]::SVector{3, Float64}
    Rout = Dict{K, RotMatrix{3, Float64, 9}}(k => RotMatrix{3, Float64}(SMatrix{3, 3, Float64, 9}(model(sol.variables[2 + v]::_ViewRotationChart)))
                                             for (v, k) in enumerate(ids))
    (all(isfinite, c) && all(isfinite, d) && all(R -> all(isfinite, R), values(Rout))) ||
        throw(ErrorException("refine_camera_and_poses: the LM step is not finite"))
    md = RotationCenterRefineMetadata{typeof(kw.lo)}(sol.residuals, sol.S / sol.ν, sol.ν, _row_norms(sol.residuals), sol.weights,
        kw.input === nothing ? NaN : kw.input.scale, kw.input === nothing ? ScaleNone : kw.input.source, sqrt(_sq_loss_width(kw.loss)),
        length(without), without)
    return FitResult((; camera, views = RotationCenterViews{K}(Rout, c, d)), sol.code, sol.rule, sol.iterations, sol.rms, sol.n, md)
end

"""
    RotationCenterRefineMetadata

The metadata of [`refine_camera_and_poses`](@ref) on [`RotationCenterViews`](@ref): the bundle's robust record — the
bearing rows (each dot's tangent residual whitened into pixels) `residuals`, `s²`, `ν`, `residual_norms`, `weights`,
`scale`, `scale_source` and `loss_width`, as [`CameraPosesRefineMetadata`](@ref)'s — and the dots without a bearing
(the camera does not back-project them, or their back-projection Jacobian cannot whiten the residual), which make no
row: read by [`n_without_bearing`](@ref) and [`without_bearing`](@ref) (their `(view, correspondence index)` pairs).
"""
struct RotationCenterRefineMetadata{L, W <: AbstractVector{Float64}, P <: AbstractVector}
    residuals::Vector{Float64}
    s²::Float64
    ν::Int
    residual_norms::Vector{Float64}
    weights::W
    scale::Float64
    scale_source::ScaleSource
    loss_width::Float64
    n_without_bearing::Int
    without_bearing::P
end
RotationCenterRefineMetadata{L}(residuals, s², ν, norms, weights::W, scale, source, loss_width, n, without::P) where {L, W, P} =
    RotationCenterRefineMetadata{L, W, P}(residuals, Float64(s²), Int(ν), norms, weights, Float64(scale), source, Float64(loss_width), Int(n), without)
VgtkCore.weight_kind(::Type{<:RotationCenterRefineMetadata{L}}) where {L} = VgtkCore.weight_kind(L)

"""
    n_without_bearing(r) -> Int

The number of dots a rotation-centre refine made no row of (no bearing): of its result `r` or its
[`RotationCenterRefineMetadata`](@ref).
"""
n_without_bearing(md::RotationCenterRefineMetadata) = md.n_without_bearing
n_without_bearing(r::FitResult{<:Any, <:RotationCenterRefineMetadata}) = n_without_bearing(metadata(r))

"""
    without_bearing(r) -> AbstractVector{Tuple{K, Int}}

The dots a rotation-centre refine made no row of, as `(view, correspondence index)` pairs (the view keyed as the input):
of its result `r` or its [`RotationCenterRefineMetadata`](@ref).
"""
without_bearing(md::RotationCenterRefineMetadata) = md.without_bearing
without_bearing(r::FitResult{<:Any, <:RotationCenterRefineMetadata}) = without_bearing(metadata(r))

# The bearing block of one correspondence: point i of its view's BearingPoseProblem at the view's pose (R_k, d − R_k c),
# on the view's rotation chart (variable 2 + v), the centre (1) and the offset (2). The residual is the problem's own
# (`_bearing_residual`, eltype-generic), so NLLSsolver's ForwardDiff differentiates it as it does the bundle's blocks;
# a robust refine scales it by `sqrt_weight` or robustifies it by `robust` (`_with_robust`).
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
_with_robust(r::_BearingCenterBlock{P}, sqrt_weight::Float64, robust::K) where {P, K} = _BearingCenterBlock{P, K}(r.varind, r.problem, r.i, sqrt_weight, robust)
NLLSsolver.ndeps(::_BearingCenterBlock) = NLLSsolver.static(3)
NLLSsolver.nres(::_BearingCenterBlock) = NLLSsolver.static(2)
NLLSsolver.varindices(r::_BearingCenterBlock) = r.varind
NLLSsolver.getvars(r::_BearingCenterBlock, vars::AbstractVector) =
    (vars[r.varind[1]]::_ViewRotationChart, vars[r.varind[2]]::SVector{3, Float64}, vars[r.varind[3]]::SVector{3, Float64})
Base.eltype(::_BearingCenterBlock) = Float64
function NLLSsolver.computeresidual(r::_BearingCenterBlock, ch, c, d)
    P = _rotation_center_pose(ch, c, d)
    e = _bearing_residual(r.problem, rotation(P), translation(P), r.i)
    return r.sqrt_weight == 1.0 ? e : r.sqrt_weight * e
end
NLLSsolver.computecost(vars::AbstractVector, r::_BearingCenterBlock) = finite_cost(NLLSsolver.computecost(r, NLLSsolver.getvars(r, vars)...))

# The bearing blocks of the rotation-centre views (block order `ids`): each view's dots back-projected by the camera to
# their unit bearings and Jacobians (`backproject_with_jacobian`), its BearingPoseProblem of them (the `M` column: rows
# in pixels), one block per dot; and the dots without a bearing, `(view, index)`, counted by the caller: no finite bearing,
# or a Jacobian the problem could not whiten by (`_whitening_defect`: not finite, not tangent, singular), found before any
# block is built so that one bad dot never aborts the refine.
function _rotation_center_bearing_blocks(cm, ids::AbstractVector{K}, csponds) where {K}
    blocks = NLLSsolver.AbstractResidual[]
    without = Tuple{K, Int}[]
    for (v, k) in enumerate(ids)
        X = SVector{3, Float64}[]; F = SVector{3, Float64}[]; Ms = SMatrix{3, 2, Float64, 6}[]
        for (j, c) in enumerate(csponds[k])
            Xj, u = _correspondence(v, j, c)
            f, M = VgtkCore.backproject_with_jacobian(cm, u)
            if all(isfinite, f) && !iszero(norm(f)) && (fu = normalize(SVector{3, Float64}(f));
                                                       _whitening_defect(fu, VgtkCore.s2_tangent_basis_matrix(Float64, fu), M) === nothing)
                push!(X, Xj); push!(F, fu); push!(Ms, SMatrix{3, 2, Float64, 6}(M))
            else                # no finite bearing, or a Jacobian that cannot whiten it (BearingPoseProblem's own test)
                push!(without, (k, j))
            end
        end
        isempty(X) && continue
        prob = BearingPoseProblem(VgtkCore.Csponds(X, F; M = Ms))
        foreach(i -> push!(blocks, _BearingCenterBlock(SVector(2 + v, 1, 2), prob, i)), eachindex(X))
    end
    return blocks, without
end
