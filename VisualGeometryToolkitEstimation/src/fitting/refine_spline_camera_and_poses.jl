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
# THE VARIABLES. Each control moves on its own sphere, the reference's chart
# (2 coordinates, the control's norm kept). Exact for unit controls, whose norm
# carries nothing: a direction `normalize(Σ wᵢcᵢ)` is unchanged by one common
# scale of the controls, which a free norm would leave the regularizer to shrink
# to zero. A lens of non-unit controls is refused (normalizing them would change
# `backproject`: the sweep normalizes the weighted sum, not each control). The
# controls are grouped into 2 × 2 patches, one NLLSsolver variable each (Core's
# `ControlPatchManifold{2, 2}`, the per-tile chart): NLLSsolver caps a block at
# 10 variables, and a correspondence's 4 × 4 window lies inside 3 × 3 patches
# (the patch grid padded by one patch, so every window's 3 × 3 exists), its pose
# the tenth. Each pose is VGC's `LeftRigidManifold6`, as in the parametric route.
#
# THE BLOCKS. One per correspondence, 2 rows, on its 9 patches and its view's
# pose: the pixel `K·x*`, `x*` the lens's projection solve (Core's
# `_spline_project`, the one LM) of `X_c` on the 4 × 4 window held from the
# guess (its cell the window's area, so its cubic pieces extrapolate if x*
# leaves the cell). Its Jacobian is analytic, by implicit differentiation at x*
# (Core's `_projection_jacobians`): no ForwardDiff dual enters the lens's solve.
# One block per third-difference row of the present nodes (Core's
# `difference_rows`, the P-spline fit's operator), `√λ·Δ³c`, 3 rows on the 3
# patches holding its 4 nodes. A patch slot no row reads — a pad slot, an absent
# node, a present node outside every window and Δ³ row — holds a coordinate no
# data constrains, which would leave the Marquardt system singular: an anchor
# row `c − c₀` holds it at its guess (zero at the solution, so it adds nothing to
# the cost).
#
# THE GAUGE. K is never a variable. A rotation Q of every control and every
# pose together — controls Q·c, poses (Q·R, Q·t) — leaves every x* (so the data
# rows) and the Δ³ rows' norms where they are: an exact symmetry of the cost.
# After the LM the estimate is moved along it by the minimal rotation that takes
# the lens axis `normalize(backward(0, 0))` back to ẑ (the controls the rows read
# and the poses; the anchored slots stay at their guesses); the rotation about ẑ
# the data leave free is the LM's (its damping keeps it near the guess's).
# =============================================================================

# A patch's variable: 2 × 2 controls, each on its sphere (`[row, col]` within the patch).
const _SplinePatch = VgtkCore.ControlPatchManifold{2, 2, 4, Float64, Nothing, VgtkCore.Cameras.AdditivePoints}
_spline_patch(C::SMatrix{2, 2, SVector{3, Float64}, 4}) = VgtkCore.ControlPatchManifold(C)
# The rows' Jacobian in a patch's chart coordinates from their Jacobian in its 4 controls
# (`rows × 12`, control-major): times the chart's derivative at δ = 0 (Core's
# `retract_jacobian`, `(t₁ t₂)` per control).
_in_patch_chart(Jc::AbstractMatrix, patch::_SplinePatch) = Jc * VgtkCore.retract_jacobian(patch)

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
# The lattice node `(i, j)` of control `n` (column-major) of patch `(P, Q)` — off the lattice for a pad slot.
@inline _slot_node(L::_PatchLayout, P, Q, n) = (2(P - 1) + (n - 1) % 2 + 1 - L.offset, 2(Q - 1) + (n - 1) ÷ 2 + 1 - L.offset)
# The slot of node (i, j) relative to the 3 × 3 patches from (P₀, Q₀): its patch k ∈ 1:9 (column-major) and control n ∈ 1:4.
@inline function _patch_slot(L::_PatchLayout, P₀, Q₀, i, j)
    k = (_patch_of(L, i) - P₀ + 1) + 3 * (_patch_of(L, j) - Q₀)
    return k, _slot_of(L, i) + 2 * (_slot_of(L, j) - 1)
end

# The LM's setting of the lens: its box and scale (`_spline_lm_box` of the backward area).
const _SplineBox = NTuple{3, SVector{2, Float64}}

# One correspondence's block: the target point, its pixel, its window's cell (the
# solve starts at its centre) and where its 16 controls sit. `N = 10`: its 3 × 3
# patches (column-major) and its pose, `slots[r, c] = (k, n)` the window control
# `[r, c]`'s patch and control; `N = 1`: its pose alone, the window's controls
# held (`held`, read only then).
struct _SplineReprojection{N} <: NLLSsolver.AbstractResidual
    varind::SVector{N, Int}
    slots::SMatrix{4, 4, NTuple{2, Int}, 16}
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

_window_controls(r::_SplineReprojection, patches::Tuple) =
    SMatrix{4, 4, SVector{3, Float64}, 16}(ntuple(m -> (s = r.slots[m]; patches[s[1]].directions[s[2]]), Val(16)))

# The block's projection: the camera-frame point, the solve's point and whether it
# converged. The solve starts at the cell's centre, never at a previous answer: the
# LM returns a start whose direction already matches to √eps (1.5e-8 rad) as it is,
# which would make the rows a step function of the variables near the guess; from
# the centre its quadratic convergence lands to rounding, so the rows are smooth.
function _spline_solve(r::_SplineReprojection, C, pose)
    P = model(pose)
    Xc = rotation(P) * r.X + translation(P)
    lb, ub, s = r.box
    x₀ = VgtkCore.Cameras._rect_centre(r.cell)
    x, ok = VgtkCore.Cameras._spline_project(C, r.cell, Xc, x₀, lb, ub, s, VgtkCore.Cameras._PROJECT_MAXITER, VgtkCore.Cameras._PROJECT_TOL^2)
    return Xc, x, ok
end
_spline_pixel(r::_SplineReprojection, x) = SVector(r.pixel[1] * x[1] + r.pixel[3], r.pixel[2] * x[2] + r.pixel[4])

# The rows (pixels), NaN where the solve fails.
function _spline_rows(r::_SplineReprojection, C, pose)
    _, x, ok = _spline_solve(r, C, pose)
    return ok ? _spline_pixel(r, x) - r.u : SVector(NaN, NaN)
end
NLLSsolver.computeresidual(r::_SplineReprojection{10}, vars...) = _spline_rows(r, _window_controls(r, vars), vars[10])
NLLSsolver.computeresidual(r::_SplineReprojection{1}, pose) = _spline_rows(r, r.held, pose)
NLLSsolver.computecost(vars::_SplineVars, r::_SplineReprojection{10}) =
    finite_cost(NLLSsolver.computecost(r, NLLSsolver.getvars(r, vars)...))
NLLSsolver.computecost(vars::Vector{_CameraPosePose}, r::_SplineReprojection{1}) =
    finite_cost(NLLSsolver.computecost(r, NLLSsolver.getvars(r, vars)...))

# The rows and their Jacobian in the variables' chart coordinates at δ = 0, by
# implicit differentiation of the solve at x*: per window control
# `diag(fx, fy)·w[r, c]·(−H/‖S‖)` (zero for the controls outside the window),
# through each patch's chart; for the pose `diag(fx, fy)·∂x*/∂X_c·[−[X_c]× I₃]`
# (the left perturbation's action). A heap matrix (2 × 78): NLLSsolver forms its
# 78 × 78 JᵀJ, too large to unroll.
function _spline_rows_and_jacobian(r::_SplineReprojection{N}, C, patches, pose) where {N}
    Xc, x, ok = _spline_solve(r, C, pose)
    np = N == 10 ? 72 : 0
    J = fill(ok ? 0.0 : NaN, 2, np + 6)
    ok || return SVector(NaN, NaN), J
    Dt, G, _, _, w = VgtkCore.Cameras._projection_jacobians(C, r.cell, x, Xc)
    A = SMatrix{2, 2, Float64}(r.pixel[1], 0.0, 0.0, r.pixel[2])
    J[:, (np + 1):(np + 6)] = A * Dt * VgtkCore.action_jacobian(pose, r.X)
    if N == 10
        AG = A * G
        Jc = zeros(2, 12, 9)                                   # per patch, the rows in its 4 controls
        for m in 1:16
            k, n = r.slots[m]
            Jc[:, (3n - 2):(3n), k] += w[m] * AG
        end
        for k in 1:9
            J[:, (8k - 7):(8k)] = _in_patch_chart(view(Jc, :, :, k), (patches::NTuple{9, _SplinePatch})[k])
        end
    end
    return _spline_pixel(r, x) - r.u, J
end
# Every variable of a block is free in this refinement (`varflags` all set).
function NLLSsolver.computeresjac(varflags, r::_SplineReprojection{10}, vars...)
    varflags == 2^10 - 1 || throw(ArgumentError("refine_camera_and_poses: every patch and pose is free (varflags = $varflags)"))
    return _spline_rows_and_jacobian(r, _window_controls(r, vars), ntuple(k -> vars[k], Val(9)), vars[10])
end
function NLLSsolver.computeresjac(varflags, r::_SplineReprojection{1}, pose)
    varflags == 1 || throw(ArgumentError("refine_camera_and_poses: the pose is free (varflags = $varflags)"))
    return _spline_rows_and_jacobian(r, r.held, nothing, pose)
end

# One third-difference row of the present nodes, weighted: `√λ·Σₘ sₘcₘ` (3 rows) on
# the 3 patches holding its 4 nodes, `slots[m] = (k, n)` node m's patch and control,
# `coef = √λ·s` the stencil.
struct _SplineSmoothness <: NLLSsolver.AbstractResidual
    varind::SVector{3, Int}
    slots::SVector{4, NTuple{2, Int}}
    coef::SVector{4, Float64}
end
NLLSsolver.ndeps(::_SplineSmoothness) = NLLSsolver.static(3)
NLLSsolver.nres(::_SplineSmoothness) = NLLSsolver.static(3)
NLLSsolver.varindices(r::_SplineSmoothness) = r.varind
NLLSsolver.getvars(r::_SplineSmoothness, vars::_SplineVars) = ntuple(k -> vars[r.varind[k]]::_SplinePatch, Val(3))
Base.eltype(::_SplineSmoothness) = Float64
@inline _smooth_control(r::_SplineSmoothness, patches, m) = (s = r.slots[m]; patches[s[1]].directions[s[2]])
NLLSsolver.computeresidual(r::_SplineSmoothness, patches...) =
    sum(m -> r.coef[m] * _smooth_control(r, patches, m), 1:4)
function NLLSsolver.computeresjac(varflags, r::_SplineSmoothness, patches...)
    varflags == 7 || throw(ArgumentError("refine_camera_and_poses: every patch is free (varflags = $varflags)"))
    Jc = zeros(3, 12, 3)
    for m in 1:4
        k, n = r.slots[m]
        Jc[:, (3n - 2):(3n), k] += r.coef[m] * I(3)
    end
    return NLLSsolver.computeresidual(r, patches...), reduce(hcat, (_in_patch_chart(view(Jc, :, :, k), patches[k]) for k in 1:3))
end

# The anchor of a patch's unread slots (`dead[n]`): `c − c₀` per such slot (3 rows
# each, zero rows for the others) at its guess `c₀`.
struct _SplineAnchor <: NLLSsolver.AbstractResidual
    varind::SVector{1, Int}
    dead::SVector{4, Bool}
    guess::SMatrix{2, 2, SVector{3, Float64}, 4}
end
NLLSsolver.ndeps(::_SplineAnchor) = NLLSsolver.static(1)
NLLSsolver.nres(::_SplineAnchor) = NLLSsolver.static(12)
NLLSsolver.varindices(r::_SplineAnchor) = r.varind
NLLSsolver.getvars(r::_SplineAnchor, vars::_SplineVars) = (vars[r.varind[1]]::_SplinePatch,)
Base.eltype(::_SplineAnchor) = Float64
NLLSsolver.computeresidual(r::_SplineAnchor, patch) =
    vcat(ntuple(n -> r.dead[n] ? patch.directions[n] - r.guess[n] : zero(SVector{3, Float64}), Val(4))...)
function NLLSsolver.computeresjac(varflags, r::_SplineAnchor, patch)
    varflags == 1 || throw(ArgumentError("refine_camera_and_poses: the patch is free (varflags = $varflags)"))
    Jc = zeros(12, 12)
    for n in 1:4
        r.dead[n] && (Jc[(3n - 2):(3n), (3n - 2):(3n)] = I(3))
    end
    return NLLSsolver.computeresidual(r, patch), _in_patch_chart(Jc, patch)
end

# The problem of a SplineLens camera `cam` (no forward) at the guesses, its patches laid
# `offset` nodes into the first (`_PatchLayout`): its variables (the patches, column-major
# over the patch grid, when `free`; then the poses), its data blocks, its Δ³ blocks
# (`λ = penalty · mean(diag)` of the data's normal matrix over the present nodes'
# coordinates, as the P-spline fit weighs its penalty), its anchors, the count of the
# correspondences dropped on an absent window, the layout, the live nodes (those a
# row reads), the patch count and the blocks' windows. A given `λ` is kept (a rebuild's).
function _spline_problem(
        cam::CameraModel, poses::AbstractVector, views::AbstractVector, free::Bool, penalty::Real, offset::Integer = 0;
        λ::Union{Nothing, Real} = nothing
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
    N = free ? 10 : 1
    live = falses(gsz)
    costs = _SplineReprojection{N}[]
    windows = SMatrix{4, 4, CartesianIndex{2}, 16}[]          # each block's window nodes
    dropped = 0
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
        # a point the guess does not project, detected within a cell of an absent window: the
        # guess images it onto that window (its solve cannot cross the absent nodes)
        if !ok && _near_absent(l.present, ox, oy, gsz)
            dropped += 1
            continue
        end
        ok || throw(
            ArgumentError("refine_camera_and_poses: view $v, correspondence $j: the point $(X) does not project at the guess (behind the camera or outside the lens's domain)")
        )
        ix, iy, _, _ = spline_window(b.area, gsz, x)
        o = VgtkCore.Cameras._lattice_node(b.area, gsz, ix - 1, iy - 1)        # the window's own cell
        cell = Rect2{Float64}(o[1], o[2], hx, hy)
        win = SMatrix{4, 4, CartesianIndex{2}, 16}(ntuple(m -> CartesianIndex(iy - 3 + (m - 1) % 4 + 1, ix - 3 + (m - 1) ÷ 4 + 1), Val(16)))
        C = SMatrix{4, 4, SVector{3, Float64}, 16}(map(i -> b.controls[i], win))
        P₀, Q₀ = _patch_of(L, iy - 2), _patch_of(L, ix - 2)
        slots = map(i -> _patch_slot(L, P₀, Q₀, i[1], i[2]), win)
        varind = free ? SVector{10, Int}(ntuple(k -> k <= 9 ? _patch_index(L, P₀ + (k - 1) % 3, Q₀ + (k - 1) ÷ 3) : nP + v, Val(10))) : SVector(v)
        foreach(i -> live[i] = true, win)
        push!(costs, _SplineReprojection{N}(varind, slots, C, cell, box, pixel, X, u))
        push!(windows, win)
    end
    smooth = _SplineSmoothness[]
    anchors = _SplineAnchor[]
    variables = Union{_SplinePatch, _CameraPosePose}[]
    λ₀ = λ
    λ = 0.0
    if free
        present = l.present === nothing ? trues(gsz) : l.present
        D = sparse(transpose(difference_rows(present, 3)))   # column m: Δ³ row m
        nodes = vec(CartesianIndices(gsz))
        stencils = map(1:size(D, 2)) do m
            rg = D.colptr[m]:(D.colptr[m + 1] - 1)
            (ntuple(k -> nodes[D.rowval[rg[k]]], Val(4)), SVector{4, Float64}(ntuple(k -> D.nzval[rg[k]], Val(4))))
        end
        foreach(s -> foreach(i -> live[i] = true, s[1]), stencils)
        # the patches: the guess's controls, a unit stand-in (ẑ) in the pad and absent slots
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
            any(dead) && push!(anchors, _SplineAnchor(SVector(_patch_index(L, P, Q)), dead, guess[P, Q]))
        end
        diagonal = zeros(gsz)                                   # the data's normal matrix: its diagonal per node (both coordinates)
        for (c, win) in zip(costs, windows)
            _, J = NLLSsolver.computeresjac(2^10 - 1, c, NLLSsolver.getvars(c, variables)...)
            for m in 1:16
                k, n = c.slots[m]
                o = 8(k - 1) + 2(n - 1)
                diagonal[win[m]] += sum(abs2, view(J, :, (o + 1):(o + 2)))
            end
        end
        λ = λ₀ === nothing ? penalty * sum(diagonal) / (2count(present)) : Float64(λ₀)
        for (ns, s) in stencils
            P₀, Q₀ = _patch_of(L, ns[1][1]), _patch_of(L, ns[1][2])
            varind = ns[1][1] == ns[2][1] ?                     # along a row (the column moves): 3 patches across, else down
                SVector{3, Int}(ntuple(k -> _patch_index(L, P₀, Q₀ + k - 1), Val(3))) :
                SVector{3, Int}(ntuple(k -> _patch_index(L, P₀ + k - 1, Q₀), Val(3)))
            slots = SVector{4, NTuple{2, Int}}(ntuple(Val(4)) do m
                i, j = Tuple(ns[m])
                kr, kc = _patch_of(L, i) - P₀, _patch_of(L, j) - Q₀
                (1 + kr + kc, _slot_of(L, i) + 2(_slot_of(L, j) - 1))
            end)
            push!(smooth, _SplineSmoothness(varind, slots, sqrt(λ) * s))
        end
    else
        append!(variables, pose_charts)
    end
    return (; variables, costs, smooth, anchors, λ, dropped, layout = L, live, nP, windows = [(w[1][2], w[1][1]) for w in windows])
end

# Whether a window within one cell of window `(ix, iy)` holds an absent node.
_near_absent(::Nothing, ix, iy, gsz) = false
_near_absent(present::AbstractMatrix{Bool}, ix, iy, (gh, gw)) =
    any(!VgtkCore.Cameras._window_present(present, jx, jy) for jx in max(3, ix - 1):min(gw - 1, ix + 1), jy in max(3, iy - 1):min(gh - 1, iy + 1))

# The problem's cost (½ Σ‖r‖², NLLSsolver's) at `variables`, and the anchors' part of it.
_spline_cost(variables, blocks) = sum(c -> NLLSsolver.computecost(variables, c), blocks; init = 0.0)
_spline_blocks(p) = Iterators.flatten((p.costs, p.smooth, p.anchors))

# The rows of every block (data first, in order, then the Δ³ rows, then the anchors)
# and their sparse Jacobian in the chart coordinates [patches | poses] — the stall
# postcondition's.
function _spline_problem_rows_and_jacobian(p, variables)
    nP = p.nP
    nc = 8nP + 6 * (length(variables) - nP)
    rows = Float64[]; Ii = Int[]; Jj = Int[]; Vv = Float64[]
    for c in _spline_blocks(p)
        rk, Jk = NLLSsolver.computeresjac(2^length(c.varind) - 1, c, NLLSsolver.getvars(c, variables)...)
        o = length(rows)
        append!(rows, rk)
        col = 0
        for v in c.varind, q in (v <= nP ? ((8v - 7):(8v)) : ((8nP + 6(v - nP) - 5):(8nP + 6(v - nP))))
            col += 1
            for i in eachindex(rk)
                push!(Ii, o + i); push!(Jj, q); push!(Vv, Jk[i, col])
            end
        end
    end
    return rows, sparse(Ii, Jj, Vv, length(rows), nc)
end

# The minimal rotation taking the unit vector `a` onto ẑ, and its angle.
function _rotation_onto_z(a::SVector{3, Float64})
    z = SVector(0.0, 0.0, 1.0)
    k = cross(a, z)
    s = norm(k)
    iszero(s) && return SMatrix{3, 3, Float64}(I), 0.0
    θ = angle_between(a, z)
    return VgtkCore.exp_so3(k * (θ / s)), θ
end

# The controls of the patches at the lattice's nodes (the guess's elsewhere: the absent slots).
function _patch_controls(b::ChartSpline, p, variables)
    controls = copy(b.controls)
    L = p.layout
    for i in CartesianIndices(controls)
        p.live[i] || continue
        controls[i] = (variables[_patch_index(L, _patch_of(L, i[1]), _patch_of(L, i[2]))]::_SplinePatch).directions[_slot_of(L, i[1]) + 2(_slot_of(L, i[2]) - 1)]
    end
    return controls
end

# The variables moved along the gauge: the live controls and the poses turned by `Q`,
# the anchored slots back at their guesses bit for bit (the LM's zero step through the
# sphere chart's renormalization may move them an ulp), so the anchors add 0.
function _turned_variables(p, variables, Q)
    out = copy(variables)
    for k in 1:p.nP
        ch = variables[k]::_SplinePatch
        P = (k - 1) % p.layout.ph + 1; Qc = (k - 1) ÷ p.layout.ph + 1
        D = SMatrix{2, 2, SVector{3, Float64}, 4}(ntuple(Val(4)) do n
            i, j = _slot_node(p.layout, P, Qc, n)
            live = 1 <= i <= size(p.live, 1) && 1 <= j <= size(p.live, 2) && p.live[i, j]
            live ? Q * ch.directions[n] : (p.variables[k]::_SplinePatch).directions[n]
        end)
        out[k] = _spline_patch(D)
    end
    for k in (p.nP + 1):length(variables)
        P = model(variables[k]::_CameraPosePose)
        out[k] = LeftRigidManifold6(Q * SMatrix{3, 3, Float64, 9}(rotation(P)), Q * SVector{3, Float64}(translation(P)))
    end
    return out
end

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
    isfree && !all(i -> abs(norm(l.backward.controls[i]) - 1) < 1.0e-12, held) && throw(
        ArgumentError(
            "refine_camera_and_poses: the SplineLens camera's controls are not unit; each control moves on its own sphere (its norm kept), " *
                "exact only for unit controls, and normalizing them would change its backprojection (the lens normalizes the weighted sum, not each control)"
        )
    )
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
    p = _spline_problem(back, poses, views, isfree, penalty, offset)
    b = l.backward
    # the unknowns: the live controls' coordinates (less the rotation gauge's 3) and the poses'; the rows: the data's and the Δ³ rows'
    counts(q) = (m = length(q.costs); (m, (isfree ? 2count(q.live) - 3 : 0) + 6V, 2m + 3length(q.smooth)))
    n, np, nobs = counts(p)
    nobs > np || throw(ArgumentError("refine_camera_and_poses: underdetermined: $n correspondences and $(length(p.smooth)) third-difference rows ($nobs rows) for $np unknowns"))

    # Solve, then rebuild the blocks at the estimate: a point whose x* moved to another
    # cell reads another window (the held one only extrapolates), so the blocks are
    # rebuilt and solved again until every window is the estimate's own.
    rounds = 0
    stable = true
    local result, solved
    while true
        rounds += 1
        problem = isfree ? NLLSsolver.NLLSProblem(Union{_SplinePatch, _CameraPosePose}, Union{_SplineReprojection{10}, _SplineSmoothness, _SplineAnchor}) :
            NLLSsolver.NLLSProblem(_CameraPosePose, _SplineReprojection{1})
        foreach(v -> NLLSsolver.addvariable!(problem, v), p.variables)
        foreach(c -> NLLSsolver.addcost!(problem, c), _spline_blocks(p))
        result = NLLSsolver.optimize!(problem, nlls_options(; max_iter, ftol, xtol, max_time))
        solved = collect(problem.variables)
        isfree || break
        at = CameraModel(cam.intrinsics, SplineLens(ChartSpline(b.area, _patch_controls(b, p, solved)); present = l.present))
        q = _spline_problem(at, [EuclideanMap(RotMatrix{3, Float64}(rotation(model(ch))), SVector{3, Float64}(translation(model(ch)))) for ch in solved[(p.nP + 1):end]],
            views, true, penalty, offset; λ = p.λ)
        q.windows == p.windows && break
        rounds == _SPLINE_REFINE_ROUNDS && (stable = false; break)
        p = q
    end
    n, np, nobs = counts(p)                                    # a rebuild may take back a point the guess did not project

    # the gauge: the live controls and the poses turned by the minimal rotation taking the axis back to ẑ
    cost_before = _spline_cost(solved, _spline_blocks(p))
    Q, θ = isfree ? _rotation_onto_z(VgtkCore.Cameras._lens_axis(ChartSpline(b.area, _patch_controls(b, p, solved)))) : (SMatrix{3, 3, Float64}(I), 0.0)
    vars = isfree ? _turned_variables(p, solved, Q) : solved
    cost_after = _spline_cost(vars, _spline_blocks(p))
    anchor_cost = _spline_cost(vars, p.anchors)
    controls = isfree ? _patch_controls(b, p, vars) : b.controls
    poses_out = [vars[p.nP + v]::_CameraPosePose for v in 1:V]
    all(ch -> all(isfinite, rotation(model(ch))) && all(isfinite, translation(model(ch))), poses_out) && all(c -> all(isfinite, c), controls[held]) ||
        throw(ErrorException("refine_camera_and_poses: the LM step is not finite"))

    r, J = _spline_problem_rows_and_jacobian(p, vars)
    code = nlls_retcode(result, J, r; tol = _CHART_LM_STATIONARITY_TOL, ftol)
    stable || code !== :converged || (code = :max_iterations)   # the windows still moving at the cap
    rd = r[1:2n]
    ν = nobs - np
    refined = isfree ? CameraModel(cam.intrinsics, SplineLens(ChartSpline(b.area, controls); present = l.present)) : cam
    out_camera = refined === cam ? camera : camera isa BoundedCamera ? VgtkCore.with_model(camera, refined) : refined
    forward_max_px = nothing
    if forward === :refitted && out_camera isa BoundedCamera
        out_camera, fcode, forward_max_px = _refit_forward(out_camera, forward_tolerance)
        fcode === :converged || code !== :converged || (code = fcode)
    end
    out = [relabel(_frames(p0)..., EuclideanMap(RotMatrix{3, Float64}(rotation(model(ch))), SVector{3, Float64}(translation(model(ch))))) for (p0, ch) in zip(poses, poses_out)]
    spline = (; penalty = p.λ, n_dropped_window = p.dropped, rounds, forward, forward_max_px, gauge_angle = θ,
        cost_before_gauge = cost_before, cost_after_gauge = cost_after, anchor_cost)
    return (; camera = out_camera, poses = out), code, result.niterations, sqrt(sum(abs2, rd) / n), n, CameraPosesRefineMetadata(rd, sum(abs2, r) / ν, ν, free, spline)
end
