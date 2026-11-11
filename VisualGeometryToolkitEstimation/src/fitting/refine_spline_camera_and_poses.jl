# =============================================================================
# refine_camera_and_poses — the route of a SplineLens camera (gauge intrinsics)
# =============================================================================
#
# The same bundle as refine_camera_and_poses.jl — Σ ‖K·x*(X_c) − u‖² over the
# views' correspondences, jointly in the camera and every pose — for a camera
# whose K is a frozen gauge and whose parameters are its backward lattice's
# controls (a SplineLens, 10²–10⁴ nodes). Core's `IntrinsicsRole` trait selects
# this route.
#
# THE VARIABLES. Each control `c = e^ℓ·D` moves by its direction on its own
# sphere (the reference's chart: Core's `ControlPatchManifold`, 2 coordinates,
# `‖D‖` kept) and by its log-scale `ℓ` (1 coordinate, Core's `EuclideanManifold`,
# 0 at the guess): any controls, unit or not, as they are (normalizing a non-unit
# lattice would change `backproject`: the sweep normalizes the weighted sum, not
# each control). The controls are grouped into 2 × 2 patches, one NLLSsolver
# variable each (the product of the two charts, 12 coordinates): NLLSsolver caps a
# block at 10 variables, and a correspondence's 4 × 4 window lies inside 3 × 3
# patches (the patch grid padded by one patch, so every window's 3 × 3 exists), its
# pose the tenth. Each pose is VGC's `LeftRigidManifold6`, as in the parametric route.
#
# THE BLOCKS. One per correspondence, 2 rows, on its 9 patches and its view's
# pose: the pixel `K·x*`, `x*` the lens's projection solve on the 4 × 4 window
# held from the guess (Core's `_solved_projection`, the one LM, on the window's
# cell, so its cubic pieces extrapolate if x* leaves the cell). Its Jacobian is
# analytic, by implicit differentiation at x* (Core's `_projection_jacobians`): no
# ForwardDiff dual enters the lens's solve. One block per third-difference row of
# the present nodes (Core's `difference_rows`, the P-spline fit's operator),
# `√λ·Δ³c / ρ`, 3 rows on the 3 patches holding its 4 nodes, `ρ` the mean length of
# those 4 controls at the current values (so the row stays within its patches and is
# exactly invariant to a common scale). A coordinate no row reads — a pad slot, an
# absent node, a present node outside every window and Δ³ row — would leave the
# Marquardt system singular: an anchor row `coordinate − guess` holds it (zero at the
# solution, so it adds nothing to the cost). With no penalty (`λ = 0`) there are no Δ³
# rows, and every node outside every window is anchored so.
#
# THE GAUGE. K is never a variable. One common scale of every control leaves every x*
# where it is (the direction normalizes the weighted sum) and every Δ³ row (each is
# divided by its own nodes' mean length): an exact symmetry of the cost, a gauge of 1
# dimension. A rotation Q of every control and every pose together — controls Q·c,
# poses (Q·R, Q·t) — leaves every x* (so the data rows) and the Δ³ rows' norms where
# they are: a gauge of 3 dimensions. The LM never runs with them open: each is held
# EXACTLY during the solve (`_SplinePin`: the step 0 on its coordinates, no cost row) —
# the first view's rotation (stages 2–3) and the log-scale of the live node nearest x = 0
# (stage 3) — so the reduced normal matrix is full rank. After the LM the estimate is
# moved, in ONE step, to the canonical representative of its orbit (a representative
# choice, not a gauge fix): the minimal rotation
# that takes the lens axis `normalize(backward(0, 0))` onto ẑ, then the roll about ẑ that
# puts the backward's x-derivative at x = 0 in the xz plane — the convention
# `_spline_gauge` reads K from (a parametric source's own) — and the common scale that
# makes the live controls' mean length the guess's. The anchored slots stay at their
# guesses.
#
# THE WEIGHT. The stack's convention (Core's `penalty_scale`): `λ = β·λ₁`, `λ₁ = penalty_scale(energy,
# N, m̄)` with the data's Gauss–Newton energy per direction coordinate (the mean diagonal
# of JᵀJ there, times the controls' squared mean length: scale-free, like the rows) and
# m̄ the mean positive eigenvalue of the Δ³ operator's DᵀD. β is a number, 0 (no
# smoothness), or chosen by generalized cross-validation (`_gcv_linearization`, `gcv_select`).
# =============================================================================

const _SplineDirections = VgtkCore.ControlPatchManifold{2, 2, 4, Float64, Nothing, VgtkCore.Cameras.AdditivePoints}
# A patch's variable: 2 × 2 controls (`[row, col]` within the patch), their directions
# on their spheres and their log-scales.
const _SplinePatch = ProductManifold{Tuple{_SplineDirections, EuclideanManifold{4, Float64}}, typeof(identity)}
_spline_patch(D::SMatrix{2, 2, SVector{3, Float64}, 4}, ℓ::SVector{4, Float64} = zero(SVector{4, Float64})) =
    ProductManifold(VgtkCore.ControlPatchManifold(D), EuclideanManifold(ℓ))
@inline _patch_direction(patch::_SplinePatch, n) = patch.charts[1].directions[n]
@inline _patch_scale(patch::_SplinePatch, n) = patch.charts[2].x[n]
# Control `n` of a patch, `e^ℓ·D`.
@inline _patch_control(patch::_SplinePatch, n) = exp(_patch_scale(patch, n)) * _patch_direction(patch, n)

# The rows' Jacobian in a patch's 12 chart coordinates from their Jacobian `Jc` in its 4
# controls (`R × 12`, control-major): the directions' through the sphere chart's
# derivative at δ = 0 (Core's `retract_jacobian`, `(t₁ t₂)` per control) scaled by
# `e^ℓ`, the log-scales' `Jc·c`.
function _in_patch_chart(Jc::SMatrix{R, 12, Float64}, patch::_SplinePatch) where {R}
    E = SVector{4, Float64}(ntuple(n -> exp(_patch_scale(patch, n)), Val(4)))
    S = SMatrix{12, 12, Float64}(LinearAlgebra.Diagonal(SVector{12, Float64}(ntuple(i -> E[(i - 1) ÷ 3 + 1], Val(12)))))
    Jd = (Jc * S) * VgtkCore.retract_jacobian(patch.charts[1])::SMatrix{12, 8, Float64, 96}
    Js = hcat(ntuple(n -> Jc[:, SVector{3}(3n - 2, 3n - 1, 3n)] * _patch_control(patch, n), Val(4))...)
    return hcat(Jd, Js)
end

# The directions-only patch: the directions-first stage's variable, every log-scale held
# EXACTLY (no coordinate of it): its controls are its directions as they are (each `‖D‖`
# its guess's control length; the guess's log-scale is 0).
@inline _patch_direction(patch::_SplineDirections, n) = patch.directions[n]
@inline _patch_control(patch::_SplineDirections, n) = patch.directions[n]
_in_patch_chart(Jc::SMatrix{R, 12, Float64}, patch::_SplineDirections) where {R} = Jc * VgtkCore.retract_jacobian(patch)::SMatrix{12, 8, Float64, 96}
const _AnyPatch = Union{_SplinePatch, _SplineDirections}
# A patch variable's coordinates: 12 (directions and log-scales), 8 (directions alone).
_patch_width(::Type{_SplinePatch}) = 12
_patch_width(::Type{_SplineDirections}) = 8
_patch_width(::Type{Nothing}) = 0                     # the poses alone: no patch

# The patch layout of a `gh × gw` lattice, `offset` nodes into its first patch (0: the
# lattice's first node is a patch's first slot): node `(i, j)` is slot `(a, b)` of patch
# `(P, Q)`, the patch grid `ph × pw` (padded so the 3 × 3 patches of every window exist).
struct _PatchLayout
    offset::Int
    ph::Int
    pw::Int
end
_PatchLayout(gsz, offset) = _PatchLayout(offset, (gsz[1] + offset) ÷ 2 + 1, (gsz[2] + offset) ÷ 2 + 1)
@inline _patch_of(L::_PatchLayout, i::Integer) = (i - 1 + L.offset) ÷ 2 + 1
@inline _slot_of(L::_PatchLayout, i::Integer) = (i - 1 + L.offset) % 2 + 1
@inline _patch_index(L::_PatchLayout, P::Integer, Q::Integer) = P + L.ph * (Q - 1)
# The patch variable and control of lattice node `i`.
@inline _node_slot(L::_PatchLayout, i::CartesianIndex{2}) =
    (_patch_index(L, _patch_of(L, i[1]), _patch_of(L, i[2])), _slot_of(L, i[1]) + 2(_slot_of(L, i[2]) - 1))
# The lattice node `(i, j)` of control `n` (column-major) of patch `(P, Q)` — off the lattice for a pad slot.
@inline _slot_node(L::_PatchLayout, P, Q, n) = (2(P - 1) + (n - 1) % 2 + 1 - L.offset, 2(Q - 1) + (n - 1) ÷ 2 + 1 - L.offset)
# The slot of node (i, j) relative to the 3 × 3 patches from (P₀, Q₀): its patch k ∈ 1:9 (column-major) and control n ∈ 1:4.
@inline function _patch_slot(L::_PatchLayout, P₀, Q₀, i, j)
    k = (_patch_of(L, i) - P₀ + 1) + 3 * (_patch_of(L, j) - Q₀)
    return k, _slot_of(L, i) + 2 * (_slot_of(L, j) - 1)
end

# The LM's setting of the lens: its box and scale (`_spline_lm_box` of the backward area).
const _SplineBox = NTuple{3, SVector{2, Float64}}

# One correspondence's block: the target point, its pixel, its window's cell and where
# its 16 controls sit. `N = 10`: its 3 × 3 patches (column-major) and its pose,
# `slots[r, c] = (k, n)` the window control `[r, c]`'s patch and control, `P` the patch
# variable's type; `N = 1`: its pose alone, the window's controls held (`held`, read
# only then; `P = Nothing`).
struct _SplineReprojection{N, P} <: NLLSsolver.AbstractResidual
    varind::SVector{N, Int}
    slots::SMatrix{4, 4, NTuple{2, Int}, 16}
    held::SMatrix{4, 4, SVector{3, Float64}, 16}
    cell::Rect2{Float64}
    box::_SplineBox
    pixel::SVector{4, Float64}           # fx, fy, cx, cy (skew 0: the gauge)
    X::SVector{3, Float64}
    u::SVector{2, Float64}
    gauge::NTuple{3, Int}                # the held gauge coordinates it reads: (its patch k ∈ 1:9 holding the held log-scale, or 0; that control n; 1 if its pose's rotation is held)
end
NLLSsolver.ndeps(::_SplineReprojection{N}) where {N} = NLLSsolver.static(N)
NLLSsolver.nres(::_SplineReprojection) = NLLSsolver.static(2)
NLLSsolver.varindices(r::_SplineReprojection) = r.varind
# The problems' variable vectors: patches and poses, or the poses alone.
const _SplineVars = Vector{Union{_SplinePatch, _CameraPosePose}}
NLLSsolver.getvars(r::_SplineReprojection{10, P}, vars::Vector{Union{P, _CameraPosePose}}) where {P} =
    (ntuple(k -> vars[r.varind[k]]::P, Val(9))..., vars[r.varind[10]]::_CameraPosePose)
NLLSsolver.getvars(r::_SplineReprojection{1}, vars::Vector{_CameraPosePose}) = (vars[r.varind[1]],)
Base.eltype(::_SplineReprojection) = Float64

_reprojection_controls(r::_SplineReprojection, patches::NTuple{9, _AnyPatch}) =
    SMatrix{4, 4, SVector{3, Float64}, 16}(ntuple(m -> (s = r.slots[m]; _patch_control(patches[s[1]], s[2])), Val(16)))

# The block's projection: the camera-frame point, the solve's point and whether it
# converged — Core's solve for differentiation (from the cell's centre).
function _spline_solve(r::_SplineReprojection, C, pose)
    P = model(pose)
    Xc = rotation(P) * r.X + translation(P)
    x, ok = VgtkCore.Cameras._solved_projection(C, r.cell, Xc, r.box...)
    return Xc, x, ok
end
_spline_pixel(r::_SplineReprojection, x) = SVector(r.pixel[1] * x[1] + r.pixel[3], r.pixel[2] * x[2] + r.pixel[4])

# The rows (pixels), NaN where the solve fails.
function _spline_rows(r::_SplineReprojection, C, pose)
    _, x, ok = _spline_solve(r, C, pose)
    return ok ? _spline_pixel(r, x) - r.u : SVector(NaN, NaN)
end
NLLSsolver.computeresidual(r::_SplineReprojection{10, <:_AnyPatch}, vars...) = _spline_rows(r, _reprojection_controls(r, ntuple(k -> vars[k], Val(9))), vars[10])
NLLSsolver.computeresidual(r::_SplineReprojection{1}, pose) = _spline_rows(r, r.held, pose)
NLLSsolver.computecost(vars::Vector{Union{P, _CameraPosePose}}, r::_SplineReprojection{10, P}) where {P} =
    finite_cost(NLLSsolver.computecost(r, NLLSsolver.getvars(r, vars)...))
NLLSsolver.computecost(vars::Vector{_CameraPosePose}, r::_SplineReprojection{1}) =
    finite_cost(NLLSsolver.computecost(r, NLLSsolver.getvars(r, vars)...))

# The rows and their Jacobian in the variables' chart coordinates at δ = 0, by
# implicit differentiation of the solve at x*: per window control
# `diag(fx, fy)·w[r, c]·(−H/‖S‖)` (zero for the controls outside the window),
# through each patch's chart; for the pose `diag(fx, fy)·∂x*/∂X_c·[−[X_c]× I₃]`
# (the left perturbation's action). A heap matrix (2 × 114), too large to unroll; its
# normal matrix goes to NLLSsolver as the lazy rank-2 `_Gram`.
function _spline_rows_and_jacobian(r::_SplineReprojection{N, P}, C, patches, pose) where {N, P}
    Xc, x, ok = _spline_solve(r, C, pose)
    np = N == 10 ? 9 * _patch_width(P) : 0
    J = fill(ok ? 0.0 : NaN, 2, np + 6)
    ok || return SVector(NaN, NaN), J
    Dt, G, _, _, w = VgtkCore.Cameras._projection_jacobians(C, r.cell, x, Xc)
    A = SMatrix{2, 2, Float64}(r.pixel[1], 0.0, 0.0, r.pixel[2])
    J[:, (np + 1):(np + 6)] = A * Dt * VgtkCore.action_jacobian(pose, r.X)
    N == 10 && _window_columns!(J, r, w, A * G, patches::NTuple{9, P})
    return _spline_pixel(r, x) - r.u, J
end
# Each patch's columns (2 × its width): the window controls it holds, `w[m]·AG` in their slots.
function _window_columns!(J, r::_SplineReprojection{10, P}, w, AG, patches::NTuple{9, P}) where {P}
    W = _patch_width(P)
    for k in 1:9
        Jc = zero(SMatrix{2, 12, Float64})
        for m in 1:16
            km, n = r.slots[m]
            km == k || continue
            Jc += hcat(ntuple(q -> q == n ? w[m] * AG : zero(SMatrix{2, 3, Float64}), Val(4))...)
        end
        iszero(Jc) || (J[:, (W * (k - 1) + 1):(W * k)] = _in_patch_chart(Jc, patches[k]))
    end
    return J
end
# The normal matrix of a block of 2 rows, `JᵀJ`, as the rank-2 form it is: its entries
# read from the 2 × n Jacobian when NLLSsolver adds its blocks into the system, never
# stored (a 114 × 114 heap matrix per block per iteration otherwise).
struct _Gram <: AbstractMatrix{Float64}
    J::Matrix{Float64}
end
Base.size(G::_Gram) = (size(G.J, 2), size(G.J, 2))
Base.@propagate_inbounds Base.getindex(G::_Gram, i::Int, j::Int) = G.J[1, i] * G.J[1, j] + G.J[2, i] * G.J[2, j]
# (Core's `finite_linearization` reads these: finite with its Jacobian, zero as a zero Jacobian's)
Base.all(::typeof(isfinite), G::_Gram) = all(isfinite, G.J)
Base.zero(G::_Gram) = _Gram(zero(G.J))

# The block's cost, gradient and normal matrix (NLLSsolver's documented extension point;
# the residual is not robustified): `(½‖r‖², Jᵀr, JᵀJ)`, the normal matrix the lazy `_Gram`,
# through Core's `finite_linearization` (a non-finite one never reaches the LM's system).
# The held gauge coordinates' columns are zeroed in the LM's gradient and normal matrix:
# no block moves them (`_SplinePin` alone gives them a unit diagonal), so their step is 0
# exactly. `computeresjac` (the Jacobian of the residual) keeps them.
function NLLSsolver.computecostgradhess(varflags, r::_SplineReprojection{10, P}, vars...) where {P <: _AnyPatch}
    res, J = NLLSsolver.computeresjac(varflags, r, vars...)
    k, n, rot = r.gauge
    k == 0 || (J[:, _patch_width(P) * (k - 1) + 8 + n] .= 0.0)
    rot == 1 && (J[:, (9 * _patch_width(P) + 1):(9 * _patch_width(P) + 3)] .= 0.0)
    return finite_linearization(0.5 * sum(abs2, res), J' * res, _Gram(J))
end
# The poses-alone block: NLLSsolver's own unrobustified form, through `finite_linearization`.
function NLLSsolver.computecostgradhess(varflags, r::_SplineReprojection{1}, pose)
    res, J = NLLSsolver.computeresjac(varflags, r, pose)
    return finite_linearization(0.5 * NLLSsolver.sqnorm(res), J' * res, J' * J)
end

# Every variable of a block is free in this refinement (`varflags` all set).
function NLLSsolver.computeresjac(varflags, r::_SplineReprojection{10, <:_AnyPatch}, vars...)
    varflags == 2^10 - 1 || throw(ArgumentError("refine_camera_and_poses: every patch and pose is free (varflags = $varflags)"))
    patches = ntuple(k -> vars[k], Val(9))
    return _spline_rows_and_jacobian(r, _reprojection_controls(r, patches), patches, vars[10])
end
function NLLSsolver.computeresjac(varflags, r::_SplineReprojection{1}, pose)
    varflags == 1 || throw(ArgumentError("refine_camera_and_poses: the pose is free (varflags = $varflags)"))
    return _spline_rows_and_jacobian(r, r.held, nothing, pose)
end

# One third-difference row of the present nodes, weighted and scale-free:
# `√λ·Σₘ sₘcₘ / ρ` (3 rows), `ρ = ¼Σₘ‖cₘ‖` the mean length of its own 4 controls at their
# current values, on the 3 patches holding its 4 nodes, `slots[m] = (k, n)` node m's
# patch and control, `coef = √λ·s` the stencil.
struct _SplineSmoothness{P} <: NLLSsolver.AbstractResidual
    varind::SVector{3, Int}
    slots::SVector{4, NTuple{2, Int}}
    coef::SVector{4, Float64}
    gauge::NTuple{2, Int}                # the held log-scale it reads: (its patch k ∈ 1:3, control n), or (0, 0)
end
NLLSsolver.ndeps(::_SplineSmoothness) = NLLSsolver.static(3)
NLLSsolver.nres(::_SplineSmoothness) = NLLSsolver.static(3)
NLLSsolver.varindices(r::_SplineSmoothness) = r.varind
NLLSsolver.getvars(r::_SplineSmoothness{P}, vars::Vector{Union{P, _CameraPosePose}}) where {P} = ntuple(k -> vars[r.varind[k]]::P, Val(3))
Base.eltype(::_SplineSmoothness) = Float64
# The row's 4 controls, its rows and their mean length `ρ`.
function _smoothness_rows(r::_SplineSmoothness, patches)
    c = ntuple(m -> (s = r.slots[m]; _patch_control(patches[s[1]], s[2])), Val(4))
    ρ = sum(norm, c) / 4
    return sum(m -> r.coef[m] * c[m], 1:4) / ρ, c, ρ
end
NLLSsolver.computeresidual(r::_SplineSmoothness, patches...) = first(_smoothness_rows(r, patches))
# The rows' Jacobian in control m: `(coefₘ·I − row·ĉₘᵀ/4) / ρ` (the denominator's
# derivative `∂ρ/∂cₘ = ĉₘᵀ/4`), through each patch's chart.
function NLLSsolver.computeresjac(varflags, r::_SplineSmoothness, patches...)
    varflags == 7 || throw(ArgumentError("refine_camera_and_poses: every patch is free (varflags = $varflags)"))
    row, c, ρ = _smoothness_rows(r, patches)
    I3 = SMatrix{3, 3, Float64}(LinearAlgebra.I)
    Jm = ntuple(m -> (r.coef[m] * I3 - row * (c[m] / norm(c[m]))' / 4) / ρ, Val(4))
    blocks = ntuple(Val(3)) do k
        Jc = zero(SMatrix{3, 12, Float64})
        for m in 1:4
            km, n = r.slots[m]
            km == k && (Jc += hcat(ntuple(q -> q == n ? Jm[m] : zero(SMatrix{3, 3, Float64}), Val(4))...))
        end
        _in_patch_chart(Jc, patches[k])
    end
    return row, hcat(blocks...)
end

# The LM's gradient and normal matrix with the held log-scale's column zeroed (as a data block's).
function NLLSsolver.computecostgradhess(varflags, r::_SplineSmoothness{P}, patches...) where {P}
    res, J = NLLSsolver.computeresjac(varflags, r, patches...)
    k, n = r.gauge
    held = _patch_width(P) * (k - 1) + 8 + n
    Jh = k == 0 ? J : J .* SVector{size(J, 2), Float64}(ntuple(j -> j == held ? 0.0 : 1.0, Val(size(J, 2))))'
    return finite_linearization(0.5 * sum(abs2, res), Jh' * res, Jh' * Jh)
end

# The anchor of a patch's unread coordinates at their guesses: per control with
# `dead[n]`, its direction (`D − D₀`, 3 rows) and, in a full patch, its log-scale (1 row):
# 16 rows (12 for a directions-only patch), zero for the others.
struct _SplineAnchor{P} <: NLLSsolver.AbstractResidual
    varind::SVector{1, Int}
    dead::SVector{4, Bool}
    guess::SMatrix{2, 2, SVector{3, Float64}, 4}
end
NLLSsolver.ndeps(::_SplineAnchor) = NLLSsolver.static(1)
NLLSsolver.nres(::_SplineAnchor{_SplinePatch}) = NLLSsolver.static(16)
NLLSsolver.nres(::_SplineAnchor{_SplineDirections}) = NLLSsolver.static(12)
NLLSsolver.varindices(r::_SplineAnchor) = r.varind
NLLSsolver.getvars(r::_SplineAnchor{P}, vars::Vector{Union{P, _CameraPosePose}}) where {P} = (vars[r.varind[1]]::P,)
Base.eltype(::_SplineAnchor) = Float64
_anchor_directions(r::_SplineAnchor, patch) = vcat(ntuple(n -> r.dead[n] ? _patch_direction(patch, n) - r.guess[n] : zero(SVector{3, Float64}), Val(4))...)
NLLSsolver.computeresidual(r::_SplineAnchor{_SplineDirections}, patch) = _anchor_directions(r, patch)
NLLSsolver.computeresidual(r::_SplineAnchor{_SplinePatch}, patch) =
    vcat(_anchor_directions(r, patch), SVector{4, Float64}(ntuple(n -> r.dead[n] ? _patch_scale(patch, n) : 0.0, Val(4))))
# the dead directions' rows of the sphere chart's derivative (12 × 8)
_anchor_jacobian(r::_SplineAnchor, ch::_SplineDirections) =
    (RJ = VgtkCore.retract_jacobian(ch)::SMatrix{12, 8, Float64, 96}; SMatrix{12, 8, Float64}(ntuple(i -> r.dead[(((i - 1) % 12) ÷ 3) + 1] ? RJ[i] : 0.0, Val(96))))
function NLLSsolver.computeresjac(varflags, r::_SplineAnchor{_SplineDirections}, patch)
    varflags == 1 || throw(ArgumentError("refine_camera_and_poses: the patch is free (varflags = $varflags)"))
    return NLLSsolver.computeresidual(r, patch), _anchor_jacobian(r, patch)
end
function NLLSsolver.computeresjac(varflags, r::_SplineAnchor{_SplinePatch}, patch)
    varflags == 1 || throw(ArgumentError("refine_camera_and_poses: the patch is free (varflags = $varflags)"))
    Js = SMatrix{4, 4, Float64}(LinearAlgebra.Diagonal(SVector{4, Float64}(map(b -> b ? 1.0 : 0.0, r.dead))))
    J = vcat(hcat(_anchor_jacobian(r, patch.charts[1]), zero(SMatrix{12, 4, Float64})), hcat(zero(SMatrix{4, 8, Float64}), Js))
    return NLLSsolver.computeresidual(r, patch), J
end
# The patch type of a problem's blocks.
_patch_type(::AbstractVector{_SplineSmoothness{P}}) where {P} = P

# The exact hold of a gauge's coordinates during the LM: on one variable, the
# coordinates `held` (a mask) get the step 0 EXACTLY — the block hands NLLSsolver a zero
# cost, a zero gradient and the identity on those coordinates of its normal matrix (and
# nothing else), so the system has a unit row and column there, decoupled from the rest:
# the LM's step there is 0 under any damping, and the cost has no row for it. The rotation
# gauge holds the first view's rotation (its pose's first 3 coordinates), the scale gauge
# one live node's log-scale. The reduced normal matrix is then full rank.
struct _SplinePin{V} <: NLLSsolver.AbstractResidual
    varind::SVector{1, Int}
    held::SVector{12, Bool}                 # the first `nvars(V)` entries are read
end
NLLSsolver.ndeps(::_SplinePin) = NLLSsolver.static(1)
NLLSsolver.nres(::_SplinePin) = NLLSsolver.static(1)
NLLSsolver.varindices(r::_SplinePin) = r.varind
NLLSsolver.getvars(r::_SplinePin, vars::AbstractVector) = (vars[r.varind[1]],)   # read for its type alone
Base.eltype(::_SplinePin) = Float64
NLLSsolver.computeresidual(::_SplinePin, v) = SVector(0.0)
NLLSsolver.computecost(::AbstractVector, ::_SplinePin) = 0.0
_pin_width(::Type{_CameraPosePose}) = 6
_pin_width(::Type{V}) where {V} = _patch_width(V)
function NLLSsolver.computecostgradhess(varflags, r::_SplinePin{V}, v) where {V}
    n = _pin_width(V)
    return 0.0, zero(SVector{n, Float64}), SMatrix{n, n, Float64}(LinearAlgebra.Diagonal(SVector{n, Float64}(ntuple(i -> r.held[i] ? 1.0 : 0.0, Val(n)))))
end


# A correspondence the camera does not project at an estimate (`msg` the refine's
# ArgumentError text), and the iterations solved before it was found. Internal: the GCV
# search scores its weight `Inf`; the refine's boundary throws it as an ArgumentError.
struct _NotProjected <: Exception
    msg::String
    iterations::Int
end

# The problem of a SplineLens camera `cam` (no forward) at the guesses, its patches laid
# `offset` nodes into the first (`_PatchLayout`): its variables (the patches, column-major
# over the patch grid, when `free`; then the poses), its data blocks, its Δ³ blocks
# (`λ = penalty·λ₁`, `λ₁ = penalty_scale(energy, N, m̄)` the unit weight: the data's
# energy per parameter is the mean diagonal of its normal matrix over the present nodes'
# direction coordinates — the weak log-scale columns, ≈ δ² of them, would lower it — times
# the present controls' squared mean length, so λ₁ is scale-free as the rows are; m̄ the
# mean positive eigenvalue of the Δ³ operator's DᵀD; none when the weight is 0),
# its anchors, the counts of the correspondences dropped on an absent window and of
# those the camera does not project within a cell of one, the layout, the live nodes
# (those a row reads), the present nodes, the patch count and the blocks' windows. A
# given `λ` is kept (a rebuild's); `at` names the point the problem is built at (an
# error's wording); `directions_only` makes each patch variable its directions alone (the
# directions-first solve: every log-scale held exactly, at the guess's 0).
function _spline_problem(
        cam::CameraModel, poses::AbstractVector, views::AbstractVector, free::Bool, penalty::Real, offset::Integer = 0;
        λ::Union{Nothing, Real} = nothing, at::AbstractString = "at the guess", directions_only::Bool = false
    )
    PT = directions_only ? _SplineDirections : _SplinePatch
    W = _patch_width(PT)
    l = cam.lens
    b = l.backward
    gsz = size(b.controls)
    L = _PatchLayout(gsz, offset)
    nP = free ? L.ph * L.pw : 0
    box = VgtkCore.Cameras._spline_lm_box(b.area)
    hx, hy = VgtkCore.Cameras._lattice_spacing(b.area, gsz)
    Kp = VgtkCore.calibration_matrix(cam)
    pixel = SVector(Kp[1, 1], Kp[2, 2], Kp[1, 3], Kp[2, 3])
    pose_charts = _CameraPosePose[LeftRigidManifold6(SMatrix{3, 3, Float64, 9}(rotation(p)), SVector{3, Float64}(translation(p))) for p in poses]
    present = l.present === nothing ? trues(gsz) : l.present
    live = falses(gsz)
    found = Tuple{SVector{3, Float64}, SVector{2, Float64}, Int, SMatrix{4, 4, CartesianIndex{2}, 16}, Rect2{Float64}}[]
    dropped = unprojected = 0
    for (v, cs) in enumerate(views), (j, c) in enumerate(cs)
        X = SVector{3, Float64}(first(c)[1], first(c)[2], first(c)[3]); u = SVector{2, Float64}(last(c)[1], last(c)[2])
        (all(isfinite, X) && all(isfinite, u)) || throw(ArgumentError("refine_camera_and_poses: view $v, correspondence $j: non-finite point or pixel"))
        ox, oy, _, _ = spline_window(b.area, gsz, VgtkCore.Cameras._normalized_point(cam.intrinsics, u))
        if !VgtkCore.Cameras._window_present(l.present, ox, oy)   # detected on an absent window: the lens does not image it
            dropped += 1
            continue
        end
        P = model(pose_charts[v])
        Xc = rotation(P) * X + translation(P)
        x, ok = VgtkCore.Cameras._project_lens(l, Xc, VgtkCore.Cameras._cold_start(l, Xc), VgtkCore.Cameras._PROJECT_MAXITER, VgtkCore.Cameras._PROJECT_TOL^2)
        # a point the camera does not project, detected within a cell of an absent window: it
        # images it onto that window (its solve cannot cross the absent nodes) — possibly an inlier
        if !ok && _near_absent(l.present, ox, oy, gsz)
            unprojected += 1
            continue
        end
        weak = at == "at the guess" ? "" :
            ": the estimate's lattice moved there, where the data hold it weakly — on noisy detections a small smoothness " *
            "weight lets those nodes swing. Use `penalty = :gcv` (the default) or a larger `penalty`"
        ok || throw(_NotProjected("refine_camera_and_poses: view $v, correspondence $j: the point $(X) does not project $at (behind the camera or outside the lens's domain)" * weak, 0))
        # a projection where the lattice folds (the backward's Jacobian there exactly singular, or not finite): no
        # Jacobian of the point exists, so it is not linearized
        _, _, Jx, _ = VgtkCore.Cameras._spline_direction_differential(b.controls, b.area, x)
        g = LinearAlgebra.det(Jx' * Jx)
        (isfinite(g) && g != 0) ||
            throw(_NotProjected("refine_camera_and_poses: view $v, correspondence $j: the lattice folds where the point $(X) projects $at (the backward's Jacobian is singular there)" * weak, 0))
        ix, iy, _, _ = spline_window(b.area, gsz, x)
        o = VgtkCore.Cameras._lattice_node(b.area, gsz, ix - 1, iy - 1)        # the window's own cell
        win = SMatrix{4, 4, CartesianIndex{2}, 16}(ntuple(m -> CartesianIndex(iy - 3 + (m - 1) % 4 + 1, ix - 3 + (m - 1) ÷ 4 + 1), Val(16)))
        foreach(i -> live[i] = true, win)
        push!(found, (X, u, v, win, Rect2{Float64}(o[1], o[2], hx, hy)))
    end
    stencils = Tuple{NTuple{4, CartesianIndex{2}}, SVector{4, Float64}}[]
    m̄ = 1.0
    if free && (λ === nothing ? penalty : λ) > 0                  # weight 0: no smoothness, no Δ³ rows
        Δ = difference_rows(present, 3)
        m̄ = mean_positive_eigenvalue(Δ)
        nodes = vec(CartesianIndices(gsz))
        for (cols, coef) in row_stencils(Δ)                  # Core's adapter: a Δ³ row's 4 nodes and stencil, one block each
            push!(stencils, (ntuple(k -> nodes[cols[k]], Val(4)), SVector{4, Float64}(ntuple(k -> coef[k], Val(4)))))
        end
        foreach(s -> foreach(i -> live[i] = true, s[1]), stencils)
    end
    # the scale gauge's held log-scale (with the scales free): the live node nearest x = 0
    scale_node = free && !directions_only ?
        argmin(i -> norm(VgtkCore.Cameras._lattice_node(b.area, gsz, i[2], i[1])), filter(i -> live[i], vec(CartesianIndices(gsz)))) : CartesianIndex(0, 0)
    # the rotation gauge's held view: the first with a data row (an unread pose would be singular)
    pinned_view = isempty(found) ? 1 : minimum(f -> f[3], found)
    N = free ? 10 : 1
    RP = free ? PT : Nothing
    costs = _SplineReprojection{N, RP}[]
    windows = SMatrix{4, 4, CartesianIndex{2}, 16}[]
    for (X, u, v, win, cell) in found
        C = SMatrix{4, 4, SVector{3, Float64}, 16}(map(i -> b.controls[i], win))
        P₀, Q₀ = _patch_of(L, win[1][1]), _patch_of(L, win[1][2])
        slots = map(i -> _patch_slot(L, P₀, Q₀, i[1], i[2]), win)
        varind = free ? SVector{10, Int}(ntuple(k -> k <= 9 ? _patch_index(L, P₀ + (k - 1) % 3, Q₀ + (k - 1) ÷ 3) : nP + v, Val(10))) : SVector(v)
        hk = findfirst(==(scale_node), win)
        gauge = ((hk === nothing ? (0, 0) : slots[hk])..., Int(free && v == pinned_view))
        push!(costs, _SplineReprojection{N, RP}(varind, slots, C, cell, box, pixel, X, u, gauge))
        push!(windows, win)
    end
    smooth = _SplineSmoothness{PT}[]
    anchors = _SplineAnchor{PT}[]
    variables = free ? Union{PT, _CameraPosePose}[] : _CameraPosePose[]   # the poses alone: their own type, the problem's
    λ₀ = λ
    λ = 0.0
    if free
        # the patches: the guess's controls (scale 0), a unit stand-in (ẑ) in the pad and absent slots
        guess = [
            SMatrix{2, 2, SVector{3, Float64}, 4}(
                ntuple(Val(4)) do n
                    i, j = _slot_node(L, P, Q, n)
                    (1 <= i <= gsz[1] && 1 <= j <= gsz[2] && present[i, j]) ? SVector{3, Float64}(b.controls[i, j]) : SVector(0.0, 0.0, 1.0)
                end
            ) for P in 1:L.ph, Q in 1:L.pw
        ]
        foreach(C -> push!(variables, directions_only ? VgtkCore.ControlPatchManifold(C) : _spline_patch(C)), vec(guess))
        append!(variables, pose_charts)
        diagonal = zeros(gsz)                                   # the data's normal matrix: its diagonal per node, the 2 direction coordinates
        for (c, win) in zip(costs, windows)
            _, J = NLLSsolver.computeresjac(2^10 - 1, c, NLLSsolver.getvars(c, variables)...)
            for m in 1:16
                k, n = c.slots[m]
                diagonal[win[m]] += sum(abs2, view(J, :, (W * (k - 1) + 2n - 1):(W * (k - 1) + 2n)))
            end
        end
        λ₁ = penalty_scale(sum(diagonal) * _mean_length(b.controls, live)^2, 2count(present), m̄)
        λ = λ₀ === nothing ? penalty * λ₁ : Float64(λ₀)
        for P in 1:L.ph, Q in 1:L.pw
            dead = SVector{4, Bool}(ntuple(Val(4)) do n
                i, j = _slot_node(L, P, Q, n)
                !(1 <= i <= gsz[1] && 1 <= j <= gsz[2] && live[i, j])
            end)
            k = _patch_index(L, P, Q)
            any(dead) && push!(anchors, _SplineAnchor{PT}(SVector(k), dead, guess[P, Q]))
        end
        for (ns, s) in stencils
            P₀, Q₀ = _patch_of(L, ns[1][1]), _patch_of(L, ns[1][2])
            varind = ns[1][1] == ns[2][1] ?                     # along a row (the column moves): 3 patches across, else down
                SVector{3, Int}(ntuple(k -> _patch_index(L, P₀, Q₀ + k - 1), Val(3))) :
                SVector{3, Int}(ntuple(k -> _patch_index(L, P₀ + k - 1, Q₀), Val(3)))
            slots = SVector{4, NTuple{2, Int}}(ntuple(Val(4)) do m      # the 3 patches in a line: k = 1 + the patch offset along it
                i, j = Tuple(ns[m])
                (1 + (_patch_of(L, i) - P₀) + (_patch_of(L, j) - Q₀), _slot_of(L, i) + 2(_slot_of(L, j) - 1))
            end)
            hk = findfirst(m -> ns[m] == scale_node, 1:4)
            push!(smooth, _SplineSmoothness{PT}(varind, slots, sqrt(λ) * s, hk === nothing ? (0, 0) : slots[hk]))
        end
    else
        append!(variables, pose_charts)
    end
    # the gauges held exactly: the first view's rotation (with the lens free), and the log-scale of the live node
    # nearest x = 0 (with the scales free); `held` their coordinates in the problem's [patches | poses] columns
    pins = Union{_SplinePin{_CameraPosePose}, _SplinePin{_SplinePatch}}[]
    held = Int[]
    if free
        push!(pins, _SplinePin{_CameraPosePose}(SVector(nP + pinned_view), SVector{12, Bool}(ntuple(i -> i <= 3, Val(12)))))
        append!(held, W * nP + 6(pinned_view - 1) .+ (1:3))
        if !directions_only
            k, n = _node_slot(L, scale_node)
            push!(pins, _SplinePin{_SplinePatch}(SVector(k), SVector{12, Bool}(ntuple(i -> i == 8 + n, Val(12)))))
            push!(held, W * (k - 1) + 8 + n)
        end
    end
    return (; variables, costs, smooth, anchors, pins, held, λ, dropped, unprojected, layout = L, live, present, nP,
        windows = [(w[1][2], w[1][1]) for w in windows])
end

# Whether a window within one cell of window `(ix, iy)` holds an absent node.
_near_absent(::Nothing, ix, iy, gsz) = false
_near_absent(present::AbstractMatrix{Bool}, ix, iy, (gh, gw)) =
    any(!VgtkCore.Cameras._window_present(present, jx, jy) for jx in max(3, ix - 1):min(gw - 1, ix + 1), jy in max(3, iy - 1):min(gh - 1, iy + 1))

# The problem's cost (½ Σ‖r‖², NLLSsolver's) at `variables` over `blocks`.
_spline_cost(variables, blocks) = sum(c -> NLLSsolver.computecost(variables, c), blocks; init = 0.0)
_spline_blocks(p) = Iterators.flatten((p.costs, p.smooth, p.anchors))

# The rows of every block (data first, in order, then the Δ³ rows, then the anchors)
# and their sparse Jacobian in the chart coordinates [patches | poses] — the stall
# postcondition's.
function _spline_problem_rows_and_jacobian(p, variables)
    nP = p.nP
    W = _patch_width(_patch_type(p.smooth))
    nc = W * nP + 6 * (length(variables) - nP)
    rows = Float64[]; Ii = Int[]; Jj = Int[]; Vv = Float64[]
    for c in _spline_blocks(p)
        rk, Jk = NLLSsolver.computeresjac(2^length(c.varind) - 1, c, NLLSsolver.getvars(c, variables)...)
        o = length(rows)
        append!(rows, rk)
        col = 0
        for v in c.varind, q in _chart_columns(W, nP, v)
            col += 1
            for i in eachindex(rk)
                Jk[i, col] == 0 && continue
                push!(Ii, o + i); push!(Jj, q); push!(Vv, Jk[i, col])
            end
        end
    end
    return rows, sparse(Ii, Jj, Vv, length(rows), nc)
end
# Variable `v`'s chart columns: the `W` of a patch (the first `nP`), then the 6 of a pose.
_chart_columns(W::Integer, nP::Integer, v::Integer) = v <= nP ? ((W * (v - 1) + 1):(W * v)) : ((W * nP + 6(v - nP) - 5):(W * nP + 6(v - nP)))

# The controls of the patches at the lattice's live nodes (the guess's elsewhere: the
# absent slots and the nodes no row reads).
function _patch_controls(b::ChartSpline, p, variables::AbstractVector)
    controls = copy(b.controls)
    for i in CartesianIndices(controls)
        p.live[i] || continue
        k, n = _node_slot(p.layout, i)
        controls[i] = _patch_control(variables[k]::_AnyPatch, n)
    end
    return controls
end

# The mean length of the controls at the nodes `live`.
_mean_length(controls::AbstractMatrix, live::AbstractMatrix{Bool}) = sum(i -> norm(controls[i]), findall(live)) / count(live)

# THE gauge rotation of a solved lattice: the minimal rotation taking its axis
# `normalize(backward(0, 0))` onto ẑ (Rotations' `rotation_between`), then the roll
# about ẑ putting its x-derivative at x = 0 in the xz plane; and the two angles.
function _gauge_rotation(controls::AbstractMatrix, area::Rect2)
    d, J = VgtkCore.Cameras._spline_direction_and_jacobian(controls, area, SVector(0.0, 0.0))
    z = SVector(0.0, 0.0, 1.0)
    θ = angle_between(d, z)
    Qa = iszero(θ) ? SMatrix{3, 3, Float64}(I) : SMatrix{3, 3, Float64}(Rotations.rotation_between(d, z))
    e = Qa * J[:, 1]
    β = atan(e[2], e[1])
    return SMatrix{3, 3, Float64}(Rotations.RotZ(-β)) * Qa, θ, β
end


# The variables moved along the gauge: the live controls' directions and the poses
# turned by `Q`, the live controls' log-scales moved by `Δℓ` (every live control scaled
# by e^Δℓ), the anchored coordinates back at their guesses bit for bit (the LM's zero
# step through the sphere chart's renormalization may move them an ulp), so the anchors
# add 0.
function _turned_variables(p, variables::_SplineVars, Q, Δℓ::Float64)
    out = copy(variables)
    for k in 1:p.nP
        ch = variables[k]::_SplinePatch
        g = p.variables[k]::_SplinePatch
        P = (k - 1) % p.layout.ph + 1; Qc = (k - 1) ÷ p.layout.ph + 1
        live = ntuple(Val(4)) do n
            i, j = _slot_node(p.layout, P, Qc, n)
            1 <= i <= size(p.live, 1) && 1 <= j <= size(p.live, 2) && p.live[i, j]
        end
        D = SMatrix{2, 2, SVector{3, Float64}, 4}(ntuple(n -> live[n] ? Q * _patch_direction(ch, n) : _patch_direction(g, n), Val(4)))
        ℓ = SVector{4, Float64}(ntuple(n -> live[n] ? _patch_scale(ch, n) + Δℓ : _patch_scale(g, n), Val(4)))
        out[k] = _spline_patch(D, ℓ)
    end
    for k in (p.nP + 1):length(variables)
        P = model(variables[k]::_CameraPosePose)
        out[k] = LeftRigidManifold6(Q * SMatrix{3, 3, Float64, 9}(rotation(P)), Q * SVector{3, Float64}(translation(P)))
    end
    return out
end

# The solved variables moved along the gauge in one step — the rotation
# (`_gauge_rotation`) and the common scale that makes the live controls' mean length
# that of the guess `b`'s at the same nodes (`_turned_variables`) — the lattice's
# controls they give, and the rotation's two angles; the poses alone: as they are.
function _gauged(p, solved::_SplineVars, b::ChartSpline)
    controls = _patch_controls(b, p, solved)
    Q, θ, β = _gauge_rotation(controls, b.area)
    vars = _turned_variables(p, solved, Q, log(_mean_length(b.controls, p.live) / _mean_length(controls, p.live)))
    return vars, _patch_controls(b, p, vars), θ, β
end
_gauged(p, solved::Vector{_CameraPosePose}, b::ChartSpline) = (solved, b.controls, 0.0, 0.0)

# The forward spline of a refined SplineLens camera, over its imaged region: Core and
# Estimation's one route (`fit_camera(SplineLens, …)`, which keeps the backward exactly).
function _refit_forward(camera::BoundedCamera, tolerance)
    fr = fit_camera(SplineLens, camera, VgtkCore.imaged_region(camera); tolerance)
    return VgtkCore.with_model(camera, estimate(fr)), retcode(fr), metadata(fr).family.forward_max_px
end

function _refine_camera_and_poses(::VgtkCore.Cameras.GaugeIntrinsics, camera::_RefinedCamera, poses::AbstractVector, views::AbstractVector; kwargs...)
    try
        return _refine_spline_camera_and_poses(camera, poses, views, 0; kwargs...)
    catch e
        e isa _NotProjected ? throw(ArgumentError(e.msg)) : rethrow()   # the public error of a point that does not project
    end
end

# What is left of the refine's time at `deadline` (`nothing`: no cap), and whether an
# NLLSsolver solve stopped on its time cap (termination bit 9).
_remaining(deadline::Float64) = isinf(deadline) ? nothing : max(0.0, deadline - time())
_timed_out(result) = result.termination & (1 << 9) != 0

# The cap on the solve-and-rebuild rounds (each round a full solve; the windows settle
# in one or two).
const _SPLINE_REFINE_ROUNDS = 5

# One NLLSsolver solve of the problem `p` from its variables: the result and the solved variables.
function _spline_solve!(p, isfree::Bool, max_iter, ftol, xtol, max_time)
    PT = _patch_type(p.smooth)
    problem = isfree ?
        NLLSsolver.NLLSProblem(Union{PT, _CameraPosePose}, Union{_SplineReprojection{10, PT}, _SplineSmoothness{PT}, _SplineAnchor{PT}, _SplinePin{_CameraPosePose}, _SplinePin{_SplinePatch}}) :
        NLLSsolver.NLLSProblem(_CameraPosePose, _SplineReprojection{1, Nothing})
    foreach(v -> NLLSsolver.addvariable!(problem, v), p.variables)
    foreach(c -> NLLSsolver.addcost!(problem, c), _spline_blocks(p))
    foreach(c -> NLLSsolver.addcost!(problem, c), p.pins)       # the gauges held exactly
    result = NLLSsolver.optimize!(problem, nlls_options(; max_iter, ftol, xtol, max_time))
    return result, collect(problem.variables)
end

# The problem rebuilt at the solved variables of `p` (the camera of its patches, its
# poses), keeping its λ; `at` names that point.
function _rebuilt_problem(p, solved::AbstractVector, cam::CameraModel, views, penalty, offset, at::AbstractString)
    b = cam.lens.backward
    est = CameraModel(cam.intrinsics, SplineLens(ChartSpline(b.area, _patch_controls(b, p, solved)); present = cam.lens.present))
    return _spline_problem(est, _solved_poses(p, solved), views, true, penalty, offset; λ = p.λ, at)
end

# The solved poses of a problem with patches, as plain maps (a rebuild's start).
_solved_poses(p, solved::AbstractVector) = [_plain_pose(solved[k]::_CameraPosePose) for k in (p.nP + 1):length(solved)]

_plain_pose(ch::_CameraPosePose) = EuclideanMap(RotMatrix{3, Float64}(rotation(model(ch))), SVector{3, Float64}(translation(model(ch))))

# The problem `p` with its Δ³ rows reweighted to `λ` (`p.λ > 0`).
_with_λ(p, λ::Float64) = merge(p, (; λ, smooth = [_SplineSmoothness{_SplinePatch}(s.varind, s.slots, sqrt(λ / p.λ) * s.coef, s.gauge) for s in p.smooth]))

# Solve `p`, then rebuild the blocks at the estimate: a point whose x* moved to another
# cell reads another window (the held one only extrapolates), so the blocks are rebuilt
# and solved again until every window is the estimate's own. At the cap the problem
# rebuilt at the estimate is the one reported (the camera's own windows). Returns the
# problem solved last, its solved variables and result, the rounds, whether the windows
# settled, the iterations, whether a solve stopped on the deadline, the joint objective
# at the start, and the problem rebuilt at the estimate (`next`: a warm start).
function _spline_rounds(p, isfree::Bool, back::CameraModel, views, penalty, offset, max_iter, ftol, xtol, deadline::Float64)
    iterations = 0
    timed_out = false
    cost_start = _spline_cost(p.variables, _spline_blocks(p))
    # at most `_SPLINE_REFINE_ROUNDS` solves, whatever the deadline (`Inf` by default)
    for rounds in 1:_SPLINE_REFINE_ROUNDS
        result, solved = _spline_solve!(p, isfree, max_iter, ftol, xtol, _remaining(deadline))
        iterations += result.niterations
        timed_out |= _timed_out(result)
        settled = true
        (isfree && !timed_out) || return (; p, solved, result, rounds, settled, iterations, timed_out, cost_start, next = p)
        q = try
            _rebuilt_problem(p, solved::_SplineVars, back, views, penalty, offset, "at the estimate of rebuild round $rounds")
        catch e
            e isa _NotProjected ? throw(_NotProjected(e.msg, e.iterations + iterations)) : rethrow()
        end
        q.windows == p.windows && return (; p, solved, result, rounds, settled, iterations, timed_out, cost_start, next = q)
        p = q
        rounds == _SPLINE_REFINE_ROUNDS &&
            return (; p, solved = collect(p.variables), result, rounds, settled = false, iterations, timed_out, cost_start, next = p)
    end
    error("unreachable: the last round returns")
end

# GCV's pencil (exact) up to this many coordinates, else per weight a sparse Cholesky and
# a Hutchinson trace from `_GCV_PROBES` Rademacher probes of a fixed seed (`gcv_curve`).
const _GCV_EXACT_MAX = 3000
const _GCV_PROBES = 64
const _GCV_SEED = 20261110

# The GCV curve of the solved problem `p` linearized at `solved`, over the weight λ: the
# Gauss–Newton step δ from there solves `min ‖J_d δ + r_d‖² + λ‖J_s δ + s‖² + ‖A δ‖²`, J_d
# the data rows' Jacobian and r_d their residuals (n rows), J_s and s the Δ³ rows' at unit
# weight (the problem's divided by √p.λ: the scale-free rows' Jacobian at the solution is
# the weight times one of the weight's own), A the anchors' (a hold, zero there), over the
# coordinates the solve moves (the pins' held ones, `p.held`, are none of them). Its GCV
# is `n‖r_d + J_dδ‖² / (n − tr H_λ)²`, `H_λ = J_d M_λ⁻¹ J_dᵀ` (the stack's `gcv_curve`, gcv.jl: the
# pencil of `(J_sᵀJ_s, J_dᵀJ_d + AᵀA + J_sᵀJ_s)`, every λ from one decomposition). Returns
# the curve `λ -> (; gcv, rss, dof, y)`, `y` the step δ.
function _gcv_linearization(p, solved::_SplineVars; exact_max::Integer = _GCV_EXACT_MAX)
    r, J = _spline_problem_rows_and_jacobian(p, solved)
    n = 2length(p.costs); m = 3length(p.smooth)
    keep = trues(size(J, 2)); keep[p.held] .= false
    Jr = J[:, keep]
    u = 1 / sqrt(p.λ)
    d = 1:n; sm = (n + 1):(n + m); an = (n + m + 1):size(J, 1)
    return gcv_curve(Jr[d, :], -r[d], u * Jr[sm, :], -u * r[sm]; hold = Jr[an, :], exact_max, probes = _GCV_PROBES, seed = _GCV_SEED)
end

# The variables `solved` of the problem `p` moved by the chart step `δ` over the
# coordinates the solve moves (`p.held` excluded, held at 0: the gauge stays exact), in
# the Jacobian's column order [patches | poses] (`_spline_problem_rows_and_jacobian`).
function _moved_along(p, solved::_SplineVars, δ::AbstractVector)
    W = _patch_width(_patch_type(p.smooth))
    full = zeros(W * p.nP + 6 * (length(solved) - p.nP))
    full[setdiff(eachindex(full), p.held)] = δ
    return [NLLSsolver.update(x, full, first(_chart_columns(W, p.nP, v))) for (v, x) in enumerate(solved)]
end

# The GCV choice of the weight β (λ = β·λ₁): ONE nonlinear solve at the start weight
# `p.λ` (the bracket's top, β = 1: the penalty weighs like the data, so no node the data
# hold weakly swings and every correspondence keeps projecting; the stages' start aid
# already solved there), the GCV curve of its linearization (`_gcv_linearization`: every
# weight from one decomposition), its minimum over log₁₀ β ∈ `_GCV_LOG10_BRACKET` by the
# stack's `gcv_select` (a scan one point a decade, then Brent's method to its own floor:
# each evaluation is closed form), then ONE final solve at the chosen weight from the
# linearization's own Gauss–Newton step there (the curve's solution `y`: the start the
# linear model predicts; from the β = 1 estimate itself the solve to a weight decades
# smaller can exhaust `max_iter`), or from the β = 1 estimate when that step stops
# projecting a correspondence. Returns the final solve's rounds, the curve `[(; β, λ,
# gcv)]` (every weight evaluated, ascending β), whether the choice is at a bracket end,
# the weight, its tr H and the iterations of both solves.
const _GCV_LOG10_BRACKET = (-12.0, 0.0)
function _gcv_search(p, λ₁::Float64, back::CameraModel, views, offset, max_iter, ftol, xtol, deadline::Float64)
    points = round(Int, _GCV_LOG10_BRACKET[2] - _GCV_LOG10_BRACKET[1]) + 1
    st₀ = _spline_rounds(p, true, back, views, 1.0, offset, max_iter, ftol, xtol, deadline)
    curve = _gcv_linearization(st₀.p, st₀.solved::_SplineVars)
    seen = Tuple{Float64, Float64}[]
    score(λ) = (e = curve(λ); push!(seen, (λ, e.gcv)); e)
    found = gcv_select(score, λ₁; log10_bracket = _GCV_LOG10_BRACKET, points)
    evaluated = [(; β = λ / λ₁, λ, gcv) for (λ, gcv) in sort(unique(seen))]
    st₀.timed_out && return st₀, evaluated, found.at_bracket_end, found.β, found.dof
    start = try
        _rebuilt_problem(_with_λ(st₀.p, found.λ), _moved_along(st₀.p, st₀.solved::_SplineVars, curve(found.λ).y), back, views, found.β, offset,
            "at the Gauss–Newton step to the GCV weight")
    catch e
        e isa _NotProjected || rethrow()
        _with_λ(st₀.next, found.λ)
    end
    st = _spline_rounds(start, true, back, views, found.β, offset, max_iter, ftol, xtol, deadline)
    return merge(st, (; iterations = st.iterations + st₀.iterations, timed_out = st.timed_out)),
        evaluated, found.at_bracket_end, found.β, found.dof
end

# The refinement with the patches laid `offset` nodes into the first (the estimate does
# not depend on it, to the solver's tolerance: a test reads both layouts).
function _refine_spline_camera_and_poses(
        camera::_RefinedCamera, poses::AbstractVector, views::AbstractVector, offset::Integer;
        free::Tuple{Vararg{Symbol}} = (:controls,),
        box::Union{Nothing, NamedTuple} = nothing,
        penalty::Union{Symbol, Real, Nothing} = :gcv,
        refit_forward::Bool = true,
        forward_tolerance::Real = _dflt("refine_camera_and_poses", "forward_tolerance"),
        max_iter::Integer = _dflt("refine_camera_and_poses", "max_iter"),
        ftol::Real = _dflt("refine_camera_and_poses", "ftol"),
        xtol::Real = _dflt("refine_camera_and_poses", "xtol"),
        max_time::Union{Nothing, Real} = nothing,
    )
    cam = _camera_model(camera)
    l = cam.lens
    (free == (:controls,) || free == ()) || throw(
        ArgumentError("refine_camera_and_poses: a SplineLens camera's `free` is (:controls,) (every present node's backward control) or () (the poses alone), got $free")
    )
    box === nothing || throw(ArgumentError("refine_camera_and_poses: a SplineLens camera's controls are unbounded; a `box` has nothing to bound"))
    check_penalty_weight(penalty, "refine_camera_and_poses: penalty"; allow_nothing = true)
    search = penalty === :gcv
    w = search ? exp10(_GCV_LOG10_BRACKET[2]) : penalty === nothing ? 0.0 : Float64(penalty::Real)   # under :gcv the scan's top β: the stages' start aid
    isfree = !isempty(free)
    held = VgtkCore.Cameras._held_nodes(l.present, size(l.backward.controls))
    forward = l.forward === nothing ? :none : !isfree ? :kept : refit_forward ? :refitted : :dropped
    forward === :refitted && !(camera isa BoundedCamera) && throw(
        ArgumentError(
            "refine_camera_and_poses: the SplineLens camera holds a forward spline, which the refine drops and refits over the camera's image: " *
                "give it as a BoundedCamera (`BoundedCamera(camera, Size2(width, height))`), or `refit_forward = false`"
        )
    )
    V = length(views)
    V >= 1 || throw(ArgumentError("refine_camera_and_poses: no views"))
    back = l.forward === nothing ? cam : CameraModel(cam.intrinsics, SplineLens(l.backward; present = l.present))   # the forward dropped
    # The solve in three stages, each from the last. A node's log-scale is weakly observed
    # (it moves x* only by the angle between its control and the ray) and nearly collinear
    # with its own direction coordinates (their data columns at a median cosine of 0.86 on
    # the 12 × 10 fixture), so a joint solve from far off creeps along that near-redundancy
    # under Marquardt's diagonal damping (158 iterations at a 0.05 rad lens error) or lets the
    # scales absorb the poses' error (a factor of 10³ in one step on a fit_camera lattice).
    # (1) The poses alone, through the guess's lens. (2) The directions and the poses, every
    # log-scale held exactly: the patches are their directions alone (14–18 iterations). (3) Everything (9 there): at the weight given, or
    # over GCV's search (stages 1–2 once, at the grid's top weight: a start aid that keeps
    # the extrapolated nodes from swinging on noisy detections, as 1e-8 lets them).
    # `max_time` caps the whole refine: every stage and round reads what is left of it.
    deadline = max_time === nothing ? Inf : time() + Float64(max_time)
    iterations = 0
    timed_out = false
    if isfree
        pre = _refine_spline_camera_and_poses(back, poses, views, offset; free = (), penalty = w, max_iter, ftol, xtol, max_time = _remaining(deadline))
        iterations += pre[3]
        timed_out |= pre[2] === :max_time
        # a stage cut short by `max_time` is not built on: the joint solve starts where it did
        if timed_out
            p = _spline_problem(back, pre[1].poses, views, true, w, offset)
        else
            p₀ = _spline_problem(back, pre[1].poses, views, true, w, offset; directions_only = true)
            r₀, s₀ = _spline_solve!(p₀, isfree, max_iter, ftol, xtol, _remaining(deadline))
            iterations += r₀.niterations
            timed_out |= _timed_out(r₀)
            p = timed_out ? _spline_problem(back, pre[1].poses, views, true, w, offset) :
                _rebuilt_problem(p₀, s₀, back, views, w, offset, "at the estimate of the directions-first stage")
        end
    else
        p = _spline_problem(back, poses, views, isfree, w, offset)
    end
    b = l.backward
    # the unknowns: the live controls' 3 coordinates (less the gauge's 4: the common scale and the rotation) and the poses'; the rows: the data's and the Δ³ rows'
    counts(q) = (m = length(q.costs); (m, (isfree ? 3count(q.live) - 4 : 0) + 6V, 2m + 3length(q.smooth)))
    n, np, nobs = counts(p)
    nobs > np || throw(ArgumentError("refine_camera_and_poses: underdetermined: $n correspondences and $(length(p.smooth)) third-difference rows ($nobs rows) for $np unknowns"))

    gcv = nothing
    gcv_at_bound = false
    effective_dof = nothing
    if search && isfree
        st, gcv, gcv_at_bound, w, effective_dof = _gcv_search(p, p.λ / w, back, views, offset, max_iter, ftol, xtol, deadline)
        its = st.iterations
    else
        st = _spline_rounds(p, isfree, back, views, w, offset, max_iter, ftol, xtol, deadline)
        its = st.iterations
    end
    iterations += its
    timed_out |= st.timed_out
    p, solved, result, rounds, settled, cost_start = st.p, st.solved, st.result, st.rounds, st.settled, st.cost_start
    n, np, nobs = counts(p)                                    # a rebuild may take back a point the guess did not project

    # the gauge, one step: the live controls and the poses turned by ONE rotation (the axis onto ẑ, the roll), the live controls scaled to the guess's mean length
    cost_before = _spline_cost(solved, _spline_blocks(p))
    vars, controls, θ, roll = _gauged(p, solved, b)
    cost_after = _spline_cost(vars, _spline_blocks(p))
    anchor_cost = _spline_cost(vars, p.anchors)
    poses_out = [vars[p.nP + v]::_CameraPosePose for v in 1:V]
    all(ch -> all(isfinite, rotation(model(ch))) && all(isfinite, translation(model(ch))), poses_out) && all(c -> all(isfinite, c), controls[held]) ||
        throw(ErrorException("refine_camera_and_poses: the LM step is not finite"))

    r, J = _spline_problem_rows_and_jacobian(p, vars)
    code = nlls_retcode(result, J, r; tol = _CHART_LM_STATIONARITY_TOL, ftol)
    timed_out && (code = :max_time)                          # a stage or round cut short by the refine's `max_time`
    rd = r[1:2n]
    ν = nobs - np
    refined = isfree ? CameraModel(cam.intrinsics, SplineLens(ChartSpline(b.area, controls); present = l.present)) : cam
    out_camera = refined === cam ? camera : camera isa BoundedCamera ? VgtkCore.with_model(camera, refined) : refined
    forward_max_px = nothing
    if forward === :refitted && out_camera isa BoundedCamera
        out_camera, fcode, forward_max_px = _refit_forward(out_camera, forward_tolerance)
        fcode === :converged || code !== :converged || (code = fcode)
    end
    out = [relabel(_frame_labels(p0)..., _plain_pose(ch)) for (p0, ch) in zip(poses, poses_out)]
    n_nodes_unrefined = isfree ? count(p.present .& .!p.live) : count(p.present)
    spline = (; β = isfree ? w : 0.0, λ = p.λ, gcv, gcv_at_bound, effective_dof, n_nodes_unrefined,
        n_dropped_window = p.dropped, n_dropped_unprojected = p.unprojected, rounds, windows_settled = settled,
        forward, forward_max_px, gauge_angle = θ, gauge_roll = roll,
        cost_start, cost_before_gauge = cost_before, cost_after_gauge = cost_after, anchor_cost)
    return (; camera = out_camera, poses = out), code, iterations, sqrt(sum(abs2, rd) / n), n, CameraPosesRefineMetadata(rd, sum(abs2, r) / ν, ν, free, spline)
end
