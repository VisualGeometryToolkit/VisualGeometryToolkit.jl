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
# `√λ·Δ³c`, 3 rows on the 3 patches holding its 4 nodes. A coordinate no row
# reads — a pad slot, an absent node, a present node outside every window and Δ³
# row, and the held log-scale (below) — would leave the Marquardt system singular:
# an anchor row `coordinate − guess` holds it (zero at the solution, so it adds
# nothing to the cost).
#
# THE GAUGE. K is never a variable. One common scale of every control leaves every
# x* where it is (the direction normalizes the weighted sum), and the Δ³ penalty
# would shrink it: the scale of one node — the live node nearest x = 0 — is held
# at its guess (the rows read its `D` alone, its `ℓ` anchored), which biases the
# 1e-8-relative penalty negligibly. A rotation Q of every control and every pose
# together — controls Q·c, poses (Q·R, Q·t) — leaves every x* (so the data rows)
# and the Δ³ rows' norms where they are: an exact symmetry of the cost, of 3
# dimensions. After the LM the estimate is moved along it by ONE rotation: the
# minimal rotation that takes the lens axis `normalize(backward(0, 0))` onto ẑ,
# then the roll about ẑ that puts the backward's x-derivative at x = 0 in the xz
# plane — the convention `_spline_gauge` reads K from (a parametric source's own).
# The anchored slots stay at their guesses.
# =============================================================================

const _SplineDirections = VgtkCore.ControlPatchManifold{2, 2, 4, Float64, Nothing, VgtkCore.Cameras.AdditivePoints}
# A patch's variable: 2 × 2 controls (`[row, col]` within the patch), their directions
# on their spheres and their log-scales.
const _SplinePatch = ProductManifold{Tuple{_SplineDirections, EuclideanManifold{4, Float64}}, typeof(identity)}
_spline_patch(D::SMatrix{2, 2, SVector{3, Float64}, 4}, ℓ::SVector{4, Float64} = zero(SVector{4, Float64})) =
    ProductManifold(VgtkCore.ControlPatchManifold(D), EuclideanManifold(ℓ))
@inline _patch_direction(patch::_SplinePatch, n) = patch.charts[1].directions[n]
@inline _patch_scale(patch::_SplinePatch, n) = patch.charts[2].x[n]
# Control `n` of a patch, `e^ℓ·D` — its direction `D` alone where its scale is held.
@inline _patch_control(patch::_SplinePatch, n, held::Bool) = held ? _patch_direction(patch, n) : exp(_patch_scale(patch, n)) * _patch_direction(patch, n)

# The rows' Jacobian in a patch's 12 chart coordinates from their Jacobian `Jc` in its 4
# controls (`R × 12`, control-major): the directions' through the sphere chart's
# derivative at δ = 0 (Core's `retract_jacobian`, `(t₁ t₂)` per control) scaled by
# `e^ℓ`, the log-scales' `Jc·c` (zero for a held scale).
function _in_patch_chart(Jc::SMatrix{R, 12, Float64}, patch::_SplinePatch, hold::Integer) where {R}
    E = SVector{4, Float64}(ntuple(n -> n == hold ? 1.0 : exp(_patch_scale(patch, n)), Val(4)))
    S = SMatrix{12, 12, Float64}(LinearAlgebra.Diagonal(SVector{12, Float64}(ntuple(i -> E[(i - 1) ÷ 3 + 1], Val(12)))))
    Jd = (Jc * S) * SMatrix{12, 8, Float64}(VgtkCore.retract_jacobian(patch.charts[1]))
    Js = hcat(ntuple(n -> n == hold ? zero(SVector{R, Float64}) : Jc[:, SVector{3}(3n - 2, 3n - 1, 3n)] * _patch_control(patch, n, false), Val(4))...)
    return hcat(Jd, Js)
end

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
# `slots[r, c] = (k, n)` the window control `[r, c]`'s patch and control, `hold` the
# patch variable and control whose scale is held (`(0, 0)` none); `N = 1`: its pose
# alone, the window's controls held (`held`, read only then).
struct _SplineReprojection{N} <: NLLSsolver.AbstractResidual
    varind::SVector{N, Int}
    slots::SMatrix{4, 4, NTuple{2, Int}, 16}
    hold::NTuple{2, Int}
    held::SMatrix{4, 4, SVector{3, Float64}, 16}
    cell::Rect2{Float64}
    box::_SplineBox
    pixel::SVector{4, Float64}           # fx, fy, cx, cy (skew 0: the gauge)
    X::SVector{3, Float64}
    u::SVector{2, Float64}
end
NLLSsolver.ndeps(::_SplineReprojection{N}) where {N} = NLLSsolver.static(N)
NLLSsolver.nres(::_SplineReprojection) = NLLSsolver.static(2)
NLLSsolver.varindices(r::_SplineReprojection) = r.varind
# The problems' variable vectors: patches and poses, or the poses alone.
const _SplineVars = Vector{Union{_SplinePatch, _CameraPosePose}}
NLLSsolver.getvars(r::_SplineReprojection{10}, vars::_SplineVars) =
    (ntuple(k -> vars[r.varind[k]]::_SplinePatch, Val(9))..., vars[r.varind[10]]::_CameraPosePose)
NLLSsolver.getvars(r::_SplineReprojection{1}, vars::Vector{_CameraPosePose}) = (vars[r.varind[1]],)
Base.eltype(::_SplineReprojection) = Float64

# The held control of patch `k` of a block (0: none).
@inline _held_slot(r, k) = r.varind[k] == r.hold[1] ? r.hold[2] : 0
_window_controls(r::_SplineReprojection, patches::NTuple{9, _SplinePatch}) =
    SMatrix{4, 4, SVector{3, Float64}, 16}(ntuple(m -> (s = r.slots[m]; _patch_control(patches[s[1]], s[2], _held_slot(r, s[1]) == s[2])), Val(16)))

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
NLLSsolver.computeresidual(r::_SplineReprojection{10}, vars...) = _spline_rows(r, _window_controls(r, ntuple(k -> vars[k], Val(9))), vars[10])
NLLSsolver.computeresidual(r::_SplineReprojection{1}, pose) = _spline_rows(r, r.held, pose)
NLLSsolver.computecost(vars::_SplineVars, r::_SplineReprojection{10}) =
    finite_cost(NLLSsolver.computecost(r, NLLSsolver.getvars(r, vars)...))
NLLSsolver.computecost(vars::Vector{_CameraPosePose}, r::_SplineReprojection{1}) =
    finite_cost(NLLSsolver.computecost(r, NLLSsolver.getvars(r, vars)...))

# The rows and their Jacobian in the variables' chart coordinates at δ = 0, by
# implicit differentiation of the solve at x*: per window control
# `diag(fx, fy)·w[r, c]·(−H/‖S‖)` (zero for the controls outside the window),
# through each patch's chart; for the pose `diag(fx, fy)·∂x*/∂X_c·[−[X_c]× I₃]`
# (the left perturbation's action). A heap matrix (2 × 114): NLLSsolver forms its
# 114 × 114 JᵀJ, too large to unroll.
function _spline_rows_and_jacobian(r::_SplineReprojection{N}, C, patches, pose) where {N}
    Xc, x, ok = _spline_solve(r, C, pose)
    np = N == 10 ? 108 : 0
    J = fill(ok ? 0.0 : NaN, 2, np + 6)
    ok || return SVector(NaN, NaN), J
    Dt, G, _, _, w = VgtkCore.Cameras._projection_jacobians(C, r.cell, x, Xc)
    A = SMatrix{2, 2, Float64}(r.pixel[1], 0.0, 0.0, r.pixel[2])
    J[:, (np + 1):(np + 6)] = A * Dt * VgtkCore.action_jacobian(pose, r.X)
    N == 10 && _window_columns!(J, r, w, A * G, patches::NTuple{9, _SplinePatch})
    return _spline_pixel(r, x) - r.u, J
end
# Each patch's 2 × 12 columns: the window controls it holds, `w[m]·AG` in their slots.
function _window_columns!(J, r::_SplineReprojection, w, AG, patches::NTuple{9, _SplinePatch})
    for k in 1:9
        Jc = zero(SMatrix{2, 12, Float64})
        for m in 1:16
            km, n = r.slots[m]
            km == k || continue
            Jc += hcat(ntuple(q -> q == n ? w[m] * AG : zero(SMatrix{2, 3, Float64}), Val(4))...)
        end
        iszero(Jc) || (J[:, (12k - 11):(12k)] = _in_patch_chart(Jc, patches[k], _held_slot(r, k)))
    end
    return J
end
# Every variable of a block is free in this refinement (`varflags` all set).
function NLLSsolver.computeresjac(varflags, r::_SplineReprojection{10}, vars...)
    varflags == 2^10 - 1 || throw(ArgumentError("refine_camera_and_poses: every patch and pose is free (varflags = $varflags)"))
    patches = ntuple(k -> vars[k], Val(9))
    return _spline_rows_and_jacobian(r, _window_controls(r, patches), patches, vars[10])
end
function NLLSsolver.computeresjac(varflags, r::_SplineReprojection{1}, pose)
    varflags == 1 || throw(ArgumentError("refine_camera_and_poses: the pose is free (varflags = $varflags)"))
    return _spline_rows_and_jacobian(r, r.held, nothing, pose)
end

# One third-difference row of the present nodes, weighted: `√λ·Σₘ sₘcₘ` (3 rows) on
# the 3 patches holding its 4 nodes, `slots[m] = (k, n)` node m's patch and control,
# `coef = √λ·s` the stencil, `hold` as a data block's.
struct _SplineSmoothness <: NLLSsolver.AbstractResidual
    varind::SVector{3, Int}
    slots::SVector{4, NTuple{2, Int}}
    hold::NTuple{2, Int}
    coef::SVector{4, Float64}
end
NLLSsolver.ndeps(::_SplineSmoothness) = NLLSsolver.static(3)
NLLSsolver.nres(::_SplineSmoothness) = NLLSsolver.static(3)
NLLSsolver.varindices(r::_SplineSmoothness) = r.varind
NLLSsolver.getvars(r::_SplineSmoothness, vars::_SplineVars) = ntuple(k -> vars[r.varind[k]]::_SplinePatch, Val(3))
Base.eltype(::_SplineSmoothness) = Float64
@inline _smooth_control(r::_SplineSmoothness, patches, m) = (s = r.slots[m]; _patch_control(patches[s[1]], s[2], _held_slot(r, s[1]) == s[2]))
NLLSsolver.computeresidual(r::_SplineSmoothness, patches...) =
    sum(m -> r.coef[m] * _smooth_control(r, patches, m), 1:4)
function NLLSsolver.computeresjac(varflags, r::_SplineSmoothness, patches...)
    varflags == 7 || throw(ArgumentError("refine_camera_and_poses: every patch is free (varflags = $varflags)"))
    I3 = SMatrix{3, 3, Float64}(LinearAlgebra.I)
    blocks = ntuple(Val(3)) do k
        Jc = zero(SMatrix{3, 12, Float64})
        for m in 1:4
            km, n = r.slots[m]
            km == k && (Jc += hcat(ntuple(q -> q == n ? r.coef[m] * I3 : zero(SMatrix{3, 3, Float64}), Val(4))...))
        end
        _in_patch_chart(Jc, patches[k], _held_slot(r, k))
    end
    return NLLSsolver.computeresidual(r, patches...), hcat(blocks...)
end

# The anchor of a patch's unread coordinates at their guesses: per control with
# `dead[n]`, its direction (`D − D₀`, 3 rows) and its log-scale (1 row); per control
# with only `scale[n]` (the held scale), its log-scale alone. 16 rows (zero for the
# others).
struct _SplineAnchor <: NLLSsolver.AbstractResidual
    varind::SVector{1, Int}
    dead::SVector{4, Bool}
    scale::SVector{4, Bool}
    guess::SMatrix{2, 2, SVector{3, Float64}, 4}
end
NLLSsolver.ndeps(::_SplineAnchor) = NLLSsolver.static(1)
NLLSsolver.nres(::_SplineAnchor) = NLLSsolver.static(16)
NLLSsolver.varindices(r::_SplineAnchor) = r.varind
NLLSsolver.getvars(r::_SplineAnchor, vars::_SplineVars) = (vars[r.varind[1]]::_SplinePatch,)
Base.eltype(::_SplineAnchor) = Float64
NLLSsolver.computeresidual(r::_SplineAnchor, patch) = vcat(
    ntuple(n -> r.dead[n] ? _patch_direction(patch, n) - r.guess[n] : zero(SVector{3, Float64}), Val(4))...,
    SVector{4, Float64}(ntuple(n -> r.scale[n] ? _patch_scale(patch, n) : 0.0, Val(4)))
)
function NLLSsolver.computeresjac(varflags, r::_SplineAnchor, patch)
    varflags == 1 || throw(ArgumentError("refine_camera_and_poses: the patch is free (varflags = $varflags)"))
    RJ = SMatrix{12, 8, Float64}(VgtkCore.retract_jacobian(patch.charts[1]))
    Jd = SMatrix{12, 8, Float64}(ntuple(i -> r.dead[(((i - 1) % 12) ÷ 3) + 1] ? RJ[i] : 0.0, Val(96)))
    Js = SMatrix{4, 4, Float64}(LinearAlgebra.Diagonal(SVector{4, Float64}(map(b -> b ? 1.0 : 0.0, r.scale))))
    J = vcat(hcat(Jd, zero(SMatrix{12, 4, Float64})), hcat(zero(SMatrix{4, 8, Float64}), Js))
    return NLLSsolver.computeresidual(r, patch), J
end

# The problem of a SplineLens camera `cam` (no forward) at the guesses, its patches laid
# `offset` nodes into the first (`_PatchLayout`): its variables (the patches, column-major
# over the patch grid, when `free`; then the poses), its data blocks, its Δ³ blocks
# (`λ = penalty · mean(diag)` of the data's normal matrix over the present nodes'
# coordinates, as the P-spline fit weighs its penalty), its anchors, the counts of the
# correspondences dropped on an absent window and of those the camera does not project
# within a cell of one, the layout, the live nodes (those a row reads), the patch count,
# the held-scale node and the blocks' windows. A given `λ` and `held_node` are kept (a
# rebuild's); `rebuild` names the estimate the problem is built at (0: the guess).
function _spline_problem(
        cam::CameraModel, poses::AbstractVector, views::AbstractVector, free::Bool, penalty::Real, offset::Integer = 0;
        λ::Union{Nothing, Real} = nothing, held_node::Union{Nothing, CartesianIndex{2}} = nothing, held_norm::Union{Nothing, Real} = nothing,
        rebuild::Integer = 0
    )
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
    where_ = rebuild == 0 ? "at the guess" : "at the estimate of rebuild round $rebuild"
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
        ok || throw(
            ArgumentError("refine_camera_and_poses: view $v, correspondence $j: the point $(X) does not project $where_ (behind the camera or outside the lens's domain)")
        )
        ix, iy, _, _ = spline_window(b.area, gsz, x)
        o = VgtkCore.Cameras._lattice_node(b.area, gsz, ix - 1, iy - 1)        # the window's own cell
        win = SMatrix{4, 4, CartesianIndex{2}, 16}(ntuple(m -> CartesianIndex(iy - 3 + (m - 1) % 4 + 1, ix - 3 + (m - 1) ÷ 4 + 1), Val(16)))
        foreach(i -> live[i] = true, win)
        push!(found, (X, u, v, win, Rect2{Float64}(o[1], o[2], hx, hy)))
    end
    stencils = Tuple{NTuple{4, CartesianIndex{2}}, SVector{4, Float64}}[]
    if free
        D = sparse(transpose(difference_rows(present, 3)))   # column m: Δ³ row m
        nodes = vec(CartesianIndices(gsz))
        for m in 1:size(D, 2)
            rg = D.colptr[m]:(D.colptr[m + 1] - 1)
            push!(stencils, (ntuple(k -> nodes[D.rowval[rg[k]]], Val(4)), SVector{4, Float64}(ntuple(k -> D.nzval[rg[k]], Val(4)))))
        end
        foreach(s -> foreach(i -> live[i] = true, s[1]), stencils)
    end
    # the held scale: the live node nearest x = 0 (the guess's choice, kept by a rebuild)
    held = !free ? CartesianIndex(0, 0) : held_node !== nothing ? held_node :
        argmin(i -> norm(VgtkCore.Cameras._lattice_node(b.area, gsz, i[2], i[1])), filter(i -> live[i], vec(CartesianIndices(gsz))))
    hold = free ? _node_slot(L, held) : (0, 0)
    held_norm = held_norm !== nothing ? Float64(held_norm) : free ? norm(b.controls[held]) : 0.0
    N = free ? 10 : 1
    costs = _SplineReprojection{N}[]
    windows = SMatrix{4, 4, CartesianIndex{2}, 16}[]
    for (X, u, v, win, cell) in found
        C = SMatrix{4, 4, SVector{3, Float64}, 16}(map(i -> b.controls[i], win))
        P₀, Q₀ = _patch_of(L, win[1][1]), _patch_of(L, win[1][2])
        slots = map(i -> _patch_slot(L, P₀, Q₀, i[1], i[2]), win)
        varind = free ? SVector{10, Int}(ntuple(k -> k <= 9 ? _patch_index(L, P₀ + (k - 1) % 3, Q₀ + (k - 1) ÷ 3) : nP + v, Val(10))) : SVector(v)
        push!(costs, _SplineReprojection{N}(varind, slots, hold, C, cell, box, pixel, X, u))
        push!(windows, win)
    end
    smooth = _SplineSmoothness[]
    anchors = _SplineAnchor[]
    variables = Union{_SplinePatch, _CameraPosePose}[]
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
        foreach(C -> push!(variables, _spline_patch(C)), vec(guess))
        append!(variables, pose_charts)
        for P in 1:L.ph, Q in 1:L.pw
            dead = SVector{4, Bool}(ntuple(Val(4)) do n
                i, j = _slot_node(L, P, Q, n)
                !(1 <= i <= gsz[1] && 1 <= j <= gsz[2] && live[i, j])
            end)
            k = _patch_index(L, P, Q)
            scale = SVector{4, Bool}(ntuple(n -> dead[n] || (k, n) == hold, Val(4)))
            any(scale) && push!(anchors, _SplineAnchor(SVector(k), dead, scale, guess[P, Q]))
        end
        diagonal = zeros(gsz)                                   # the data's normal matrix: its diagonal per node (its 3 coordinates)
        for (c, win) in zip(costs, windows)
            _, J = NLLSsolver.computeresjac(2^10 - 1, c, NLLSsolver.getvars(c, variables)...)
            for m in 1:16
                k, n = c.slots[m]
                diagonal[win[m]] += sum(abs2, view(J, :, 12(k - 1) .+ [2n - 1, 2n, 8 + n]))
            end
        end
        λ = λ₀ === nothing ? penalty * sum(diagonal) / (3count(present)) : Float64(λ₀)
        for (ns, s) in stencils
            P₀, Q₀ = _patch_of(L, ns[1][1]), _patch_of(L, ns[1][2])
            varind = ns[1][1] == ns[2][1] ?                     # along a row (the column moves): 3 patches across, else down
                SVector{3, Int}(ntuple(k -> _patch_index(L, P₀, Q₀ + k - 1), Val(3))) :
                SVector{3, Int}(ntuple(k -> _patch_index(L, P₀ + k - 1, Q₀), Val(3)))
            slots = SVector{4, NTuple{2, Int}}(ntuple(Val(4)) do m      # the 3 patches in a line: k = 1 + the patch offset along it
                i, j = Tuple(ns[m])
                (1 + (_patch_of(L, i) - P₀) + (_patch_of(L, j) - Q₀), _slot_of(L, i) + 2(_slot_of(L, j) - 1))
            end)
            push!(smooth, _SplineSmoothness(varind, slots, hold, sqrt(λ) * s))
        end
    else
        append!(variables, pose_charts)
    end
    return (; variables, costs, smooth, anchors, λ, dropped, unprojected, layout = L, live, nP, held, hold, held_norm,
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
    nc = 12nP + 6 * (length(variables) - nP)
    rows = Float64[]; Ii = Int[]; Jj = Int[]; Vv = Float64[]
    for c in _spline_blocks(p)
        rk, Jk = NLLSsolver.computeresjac(2^length(c.varind) - 1, c, NLLSsolver.getvars(c, variables)...)
        o = length(rows)
        append!(rows, rk)
        col = 0
        for v in c.varind, q in (v <= nP ? ((12v - 11):(12v)) : ((12nP + 6(v - nP) - 5):(12nP + 6(v - nP))))
            col += 1
            for i in eachindex(rk)
                Jk[i, col] == 0 && continue
                push!(Ii, o + i); push!(Jj, q); push!(Vv, Jk[i, col])
            end
        end
    end
    return rows, sparse(Ii, Jj, Vv, length(rows), nc)
end

# The controls of the patches at the lattice's live nodes (the guess's elsewhere: the
# absent slots and the nodes no row reads); the held-scale node's at the guess's norm
# to rounding (its sphere chart keeps the norm to an ulp a step).
function _patch_controls(b::ChartSpline, p, variables::_SplineVars)
    controls = copy(b.controls)
    for i in CartesianIndices(controls)
        p.live[i] || continue
        k, n = _node_slot(p.layout, i)
        c = _patch_control(variables[k]::_SplinePatch, n, (k, n) == p.hold)
        controls[i] = (k, n) == p.hold ? p.held_norm * normalize(c) : c
    end
    return controls
end

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
# turned by `Q`, the anchored coordinates back at their guesses bit for bit (the LM's
# zero step through the sphere chart's renormalization may move them an ulp), so the
# anchors add 0.
function _turned_variables(p, variables::_SplineVars, Q)
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
        ℓ = SVector{4, Float64}(ntuple(n -> live[n] && (k, n) != p.hold ? _patch_scale(ch, n) : _patch_scale(g, n), Val(4)))
        out[k] = _spline_patch(D, ℓ)
    end
    for k in (p.nP + 1):length(variables)
        P = model(variables[k]::_CameraPosePose)
        out[k] = LeftRigidManifold6(Q * SMatrix{3, 3, Float64, 9}(rotation(P)), Q * SVector{3, Float64}(translation(P)))
    end
    return out
end

# The solved variables moved along the gauge (`_gauge_rotation`, `_turned_variables`),
# the lattice's controls they give, and the gauge's two angles; the poses alone: as they are.
function _gauged(p, solved::_SplineVars, b::ChartSpline)
    Q, θ, β = _gauge_rotation(_patch_controls(b, p, solved), b.area)
    vars = _turned_variables(p, solved, Q)
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
    return _refine_spline_camera_and_poses(camera, poses, views, 0; kwargs...)
end

# The cap on the solve-and-rebuild rounds (each round a full solve; the windows settle
# in one or two).
const _SPLINE_REFINE_ROUNDS = 5

_plain_pose(ch::_CameraPosePose) = EuclideanMap(RotMatrix{3, Float64}(rotation(model(ch))), SVector{3, Float64}(translation(model(ch))))

# The refinement with the patches laid `offset` nodes into the first (the estimate does
# not depend on it, to the solver's tolerance: a test reads both layouts).
function _refine_spline_camera_and_poses(
        camera::_RefinedCamera, poses::AbstractVector, views::AbstractVector, offset::Integer;
        free::Tuple{Vararg{Symbol}} = (:controls,),
        box::Union{Nothing, NamedTuple} = nothing,
        penalty::Real = _PSPLINE_PENALTY,
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
    penalty >= 0 || throw(ArgumentError("refine_camera_and_poses: `penalty` must be ≥ 0, got $penalty"))
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
    # The poses first, alone, through the guess's lens: a node's scale is weakly observed
    # (it moves x* only by the angle between its control and the ray), so a first full step
    # from poses far off would let the scales absorb the poses' error (a factor of 10³ in one
    # step, measured on a fit_camera lattice); from refined poses the full solve is local.
    iterations = 0
    if isfree
        pre = _refine_spline_camera_and_poses(back, poses, views, offset; free = (), penalty, max_iter, ftol, xtol, max_time)
        start = pre[1].poses
        iterations += pre[3]
    else
        start = poses
    end
    p = _spline_problem(back, start, views, isfree, penalty, offset)
    b = l.backward
    # the unknowns: the live controls' 3 coordinates (less the held scale and the rotation gauge's 3) and the poses'; the rows: the data's and the Δ³ rows'
    counts(q) = (m = length(q.costs); (m, (isfree ? 3count(q.live) - 4 : 0) + 6V, 2m + 3length(q.smooth)))
    n, np, nobs = counts(p)
    nobs > np || throw(ArgumentError("refine_camera_and_poses: underdetermined: $n correspondences and $(length(p.smooth)) third-difference rows ($nobs rows) for $np unknowns"))

    # Solve, then rebuild the blocks at the estimate: a point whose x* moved to another
    # cell reads another window (the held one only extrapolates), so the blocks are
    # rebuilt and solved again until every window is the estimate's own. At the cap the
    # problem rebuilt at the estimate is the one reported (the camera's own windows).
    rounds = 0
    settled = true
    local result, solved
    while true
        rounds += 1
        problem = isfree ? NLLSsolver.NLLSProblem(Union{_SplinePatch, _CameraPosePose}, Union{_SplineReprojection{10}, _SplineSmoothness, _SplineAnchor}) :
            NLLSsolver.NLLSProblem(_CameraPosePose, _SplineReprojection{1})
        foreach(v -> NLLSsolver.addvariable!(problem, v), p.variables)
        foreach(c -> NLLSsolver.addcost!(problem, c), _spline_blocks(p))
        result = NLLSsolver.optimize!(problem, nlls_options(; max_iter, ftol, xtol, max_time))
        iterations += result.niterations
        solved = collect(problem.variables)
        isfree || break
        at = CameraModel(cam.intrinsics, SplineLens(ChartSpline(b.area, _patch_controls(b, p, solved::_SplineVars)); present = l.present))
        q = _spline_problem(at, [_plain_pose(ch) for ch in solved[(p.nP + 1):end]], views, true, penalty, offset; λ = p.λ, held_node = p.held, held_norm = p.held_norm, rebuild = rounds)
        q.windows == p.windows && break
        p = q
        if rounds == _SPLINE_REFINE_ROUNDS
            settled = false
            solved = collect(p.variables)                       # the estimate, in the blocks of its own windows
            break
        end
    end
    n, np, nobs = counts(p)                                    # a rebuild may take back a point the guess did not project

    # the gauge: the live controls and the poses turned by ONE rotation (the axis onto ẑ, the roll)
    cost_before = _spline_cost(solved, _spline_blocks(p))
    vars, controls, θ, β = _gauged(p, solved, b)
    cost_after = _spline_cost(vars, _spline_blocks(p))
    anchor_cost = _spline_cost(vars, p.anchors)
    poses_out = [vars[p.nP + v]::_CameraPosePose for v in 1:V]
    all(ch -> all(isfinite, rotation(model(ch))) && all(isfinite, translation(model(ch))), poses_out) && all(c -> all(isfinite, c), controls[held]) ||
        throw(ErrorException("refine_camera_and_poses: the LM step is not finite"))

    r, J = _spline_problem_rows_and_jacobian(p, vars)
    code = nlls_retcode(result, J, r; tol = _CHART_LM_STATIONARITY_TOL, ftol)
    rd = r[1:2n]
    ν = nobs - np
    refined = isfree ? CameraModel(cam.intrinsics, SplineLens(ChartSpline(b.area, controls); present = l.present)) : cam
    out_camera = refined === cam ? camera : camera isa BoundedCamera ? VgtkCore.with_model(camera, refined) : refined
    forward_max_px = nothing
    if forward === :refitted && out_camera isa BoundedCamera
        out_camera, fcode, forward_max_px = _refit_forward(out_camera, forward_tolerance)
        fcode === :converged || code !== :converged || (code = fcode)
    end
    out = [relabel(_frames(p0)..., _plain_pose(ch)) for (p0, ch) in zip(poses, poses_out)]
    spline = (; penalty = p.λ, n_dropped_window = p.dropped, n_dropped_unprojected = p.unprojected, rounds, windows_settled = settled,
        held_scale_node = p.held, forward, forward_max_px, gauge_angle = θ, gauge_roll = β,
        cost_before_gauge = cost_before, cost_after_gauge = cost_after, anchor_cost)
    return (; camera = out_camera, poses = out), code, iterations, sqrt(sum(abs2, rd) / n), n, CameraPosesRefineMetadata(rd, sum(abs2, r) / ν, ν, free, spline)
end
