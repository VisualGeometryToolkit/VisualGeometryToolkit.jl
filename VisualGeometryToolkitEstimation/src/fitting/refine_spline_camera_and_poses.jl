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
# variable each: NLLSsolver caps a block at 10 variables, and a correspondence's
# 4 × 4 window lies inside 2 × 2 to 3 × 3 patches, its pose the last. ONLY THE LIVE
# CONTROLS ARE VARIABLES: a patch is Core's `restrict` of the product of the two
# charts (12 coordinates) to the coordinates of its live nodes (those a row reads),
# so a pad slot, an absent node or a node no row reads has no coordinate and stays
# exactly as it is; a patch with no live node is no variable. Each pose is VGC's
# `LeftRigidManifold6`, as in the parametric route, restricted likewise.
#
# THE BLOCKS. One per correspondence, 2 rows, on the patches its window lies in and
# its view's pose: the pixel `K·x*`, `x*` the lens's projection solve on the 4 × 4
# window held from the guess (Core's `_solved_projection`, the one LM, on the window's
# cell, so its cubic pieces extrapolate if x* leaves the cell). Its Jacobian is
# analytic, by implicit differentiation at x* (Core's `_projection_jacobians`): no
# ForwardDiff dual enters the lens's solve. One block per third-difference row of
# the present nodes (Core's `difference_rows`, the P-spline fit's operator),
# `√λ·Δ³c / ρ`, 3 rows on the 2 or 3 patches holding its 4 nodes, `ρ` the mean length
# of those 4 controls at the current values (so the row stays within its patches and
# is exactly invariant to a common scale).
#
# THE GAUGE. K is never a variable. One common scale of every control leaves every x*
# where it is (the direction normalizes the weighted sum) and every Δ³ row (each is
# divided by its own nodes' mean length): an exact symmetry of the cost, a gauge of 1
# dimension. A rotation Q of every control and every pose together — controls Q·c,
# poses (Q·R, Q·t) — leaves every x* (so the data rows) and the Δ³ rows' norms where
# they are: a gauge of 3 dimensions. Both are fixed EXACTLY by the parametrization:
# the first view with a data row has a translation-only pose chart (its rotation is
# data, stages 2–3), and the live node nearest x = 0 has no log-scale coordinate
# (stage 3), so the reduced normal matrix is full rank. After the LM the estimate is
# moved, in ONE step, to the canonical representative of its orbit (a representative
# choice, not a gauge fix): the minimal rotation that takes the lens axis
# `normalize(backward(0, 0))` onto ẑ, then the roll about ẑ that puts the backward's
# x-derivative at x = 0 in the xz plane — the convention `_spline_gauge` reads K from
# (a parametric source's own) — and the common scale that makes the live controls'
# mean length the guess's.
#
# THE WEIGHT. The stack's convention (Core's `penalty_scale`): `λ = β·λ₁`, `λ₁ = penalty_scale(energy,
# N, m̄)` with the data's Gauss–Newton energy per direction coordinate (the mean diagonal
# of JᵀJ there, times the controls' squared mean length: scale-free, like the rows) and
# m̄ the mean positive eigenvalue of the Δ³ operator's DᵀD. β is a number, 0 (no
# smoothness), or chosen by generalized cross-validation (`_gcv_linearization`, `gcv_select`).
# =============================================================================

const _SplineDirections = VgtkCore.ControlPatchManifold{2, 2, 4, Float64, Nothing, VgtkCore.Cameras.AdditivePoints}
# A patch's full chart: 2 × 2 controls (`[row, col]` within the patch), their directions on
# their spheres (coordinates 2n − 1, 2n of control n) and their log-scales (8 + n).
const _PatchChart = ProductManifold{Tuple{_SplineDirections, EuclideanManifold{4, Float64}}, typeof(identity)}
_patch_chart(D::SMatrix{2, 2, SVector{3, Float64}, 4}, ℓ::SVector{4, Float64} = zero(SVector{4, Float64})) =
    ProductManifold(VgtkCore.ControlPatchManifold(D), EuclideanManifold(ℓ))
# A patch variable: its chart on the coordinates of its live controls; a pose variable:
# the pose's chart on its free coordinates (all 6, or the translation's 3 for the view
# holding the rotation gauge). Each a restriction to a run-time index (Core's `restrict`
# with a `Vector{Int}`): one variable type for every patch (a static size per restriction
# would make every combination of a block's patch sizes its own compiled block type).
const _SplinePatch = VgtkCore.SubsetManifold{_PatchChart, 12, Vector{Int}}
const _SplinePose = VgtkCore.SubsetManifold{_CameraPosePose, 6, Vector{Int}}
const _SplineVar = Union{_SplinePatch, _SplinePose}
@inline _patch_direction(patch::_SplinePatch, n) = patch.manifold.charts[1].directions[n]
@inline _patch_scale(patch::_SplinePatch, n) = patch.manifold.charts[2].x[n]
# Control `n` of a patch, `e^ℓ·D`.
@inline _patch_control(patch::_SplinePatch, n) = exp(_patch_scale(patch, n)) * _patch_direction(patch, n)
# Whether control `n` of a patch is a variable (its direction coordinates are).
_slot_live(patch::_SplinePatch, n) = (2n - 1) in patch.idx
_pose_variable(ch::_CameraPosePose, rotation_free::Bool) = VgtkCore.restrict(ch, rotation_free ? collect(1:6) : [4, 5, 6])

# The rows' Jacobian in a patch's 12 chart coordinates from their Jacobian `Jc[n]` (`R × 3`)
# in each of its 4 controls (zero for a control no row reads): control n's direction
# columns through its sphere chart's derivative at δ = 0 (Core's `direction_tangents`,
# `(t₁ t₂)`, scaled by `e^ℓ`), its log-scale's `Jc[n]·c`.
function _in_patch_chart(Jc::NTuple{4, SMatrix{R, 3, Float64}}, patch::_SplinePatch) where {R}
    cols = ntuple(Val(4)) do n
        iszero(Jc[n]) && return zero(SMatrix{R, 2, Float64})             # a control no row reads: no tangents formed
        D = _patch_direction(patch, n)
        t1, t2 = VgtkCore.direction_tangents(normalize(D))
        e = exp(_patch_scale(patch, n))
        hcat(Jc[n] * (e * t1), Jc[n] * (e * t2))
    end
    scales = ntuple(n -> Jc[n] * _patch_control(patch, n), Val(4))
    return hcat(cols..., scales...)
end

# The patch layout of a `gh × gw` lattice: node `(i, j)` is slot `(a, b)` of patch `(P, Q)`,
# the patch grid `ph × pw`.
struct _PatchLayout
    ph::Int
    pw::Int
end
_PatchLayout(gsz) = _PatchLayout(gsz[1] ÷ 2 + 1, gsz[2] ÷ 2 + 1)
@inline _patch_of(i::Integer) = (i - 1) ÷ 2 + 1
@inline _slot_of(i::Integer) = (i - 1) % 2 + 1
# The lattice node `(i, j)` of control `n` (column-major) of patch `(P, Q)` — off the lattice for a pad slot.
@inline _slot_node(P, Q, n) = (2(P - 1) + (n - 1) % 2 + 1, 2(Q - 1) + (n - 1) ÷ 2 + 1)
# The control of node (i, j) within its patch.
@inline _node_control(i, j) = _slot_of(i) + 2(_slot_of(j) - 1)

# The LM's setting of the lens: its box and scale (`_spline_lm_box` of the backward area).
const _SplineBox = NTuple{3, SVector{2, Float64}}
const _WindowControls = SMatrix{4, 4, SVector{3, Float64}, 16}

# One correspondence's block: the target point, its pixel, its window's cell and where
# its 16 controls sit. `N > 1`: its `N − 1` patches (column-major over the window's 2 × 2,
# 2 × 3 or 3 × 3 block of patches) and its pose, `slots[r, c] = (k, n)` the window control
# `[r, c]`'s patch and control; `N = 1`: its pose alone, the window's controls held
# (`held`). `pixel` the camera's pixel affinity (Core's `_pixel_affine`: linear part, offset).
# `solve` the block's last projection solve, by its window controls and camera-frame point:
# NLLSsolver evaluates an accepted trial point's cost and then linearizes at the same
# point, and the solve (from the cell's centre) gives the same answer bit for bit.
struct _SplineReprojection{N, H <: Union{Nothing, _WindowControls}} <: NLLSsolver.AbstractResidual
    varind::SVector{N, Int}
    slots::SMatrix{4, 4, NTuple{2, Int}, 16}
    held::H
    cell::Rect2{Float64}
    box::_SplineBox
    pixel::Tuple{SMatrix{2, 2, Float64, 4}, SVector{2, Float64}}
    X::SVector{3, Float64}
    u::SVector{2, Float64}
    solve::Base.RefValue{Tuple{_WindowControls, SVector{3, Float64}, SVector{2, Float64}, Bool}}
end
_SplineReprojection{N, H}(varind, slots, held, cell, box, pixel, X, u) where {N, H} = _SplineReprojection{N, H}(
    varind, slots, held, cell, box, pixel, X, u, Ref((_WindowControls(ntuple(_ -> SVector(NaN, NaN, NaN), Val(16))), SVector(NaN, NaN, NaN), SVector(NaN, NaN), false))
)
const _SplineDataBlock = Union{_SplineReprojection{5, Nothing}, _SplineReprojection{7, Nothing}, _SplineReprojection{10, Nothing}}
const _SplinePoseBlock = _SplineReprojection{1, _WindowControls}
NLLSsolver.ndeps(::_SplineReprojection{N}) where {N} = NLLSsolver.static(N)
NLLSsolver.nres(::_SplineReprojection) = NLLSsolver.static(2)
NLLSsolver.varindices(r::_SplineReprojection) = r.varind
NLLSsolver.getvars(r::_SplineReprojection{N, Nothing}, vars::AbstractVector) where {N} =
    ntuple(k -> vars[r.varind[k]], Val(N))
NLLSsolver.getvars(r::_SplinePoseBlock, vars::AbstractVector) = (vars[r.varind[1]]::_SplinePose,)
Base.eltype(::_SplineReprojection) = Float64

_reprojection_controls(r::_SplineReprojection, patches::Tuple) =
    _WindowControls(ntuple(m -> (s = r.slots[m]; _patch_control(patches[s[1]], s[2])), Val(16)))

# The block's projection: the camera-frame point, the solve's point and whether it
# converged — Core's solve for differentiation (from the cell's centre).
function _spline_solve(r::_SplineReprojection, C, pose::_SplinePose)
    P = model(pose)
    Xc = rotation(P) * r.X + translation(P)
    C₀, X₀, x₀, ok₀ = r.solve[]
    C₀ == C && X₀ == Xc && return Xc, x₀, ok₀                        # the point just solved
    x, ok = VgtkCore.Cameras._solved_projection(C, r.cell, Xc, r.box...)
    r.solve[] = (C, Xc, x, ok)
    return Xc, x, ok
end
_spline_pixel(r::_SplineReprojection, x) = r.pixel[1] * x + r.pixel[2]

# The rows (pixels), NaN where the solve fails.
function _spline_rows(r::_SplineReprojection, C, pose)
    _, x, ok = _spline_solve(r, C, pose)
    return ok ? _spline_pixel(r, x) - r.u : SVector(NaN, NaN)
end
NLLSsolver.computeresidual(r::_SplineReprojection{N, Nothing}, vars...) where {N} =
    _spline_rows(r, _reprojection_controls(r, ntuple(k -> vars[k], Val(N - 1))), vars[N])
NLLSsolver.computeresidual(r::_SplinePoseBlock, pose) = _spline_rows(r, r.held, pose)
NLLSsolver.computecost(vars::AbstractVector, r::_SplineReprojection) = finite_cost(NLLSsolver.computecost(r, NLLSsolver.getvars(r, vars)...))

# The rows and their Jacobian in each variable's full chart at δ = 0, by implicit
# differentiation of the solve at x*: per window control `A·w[r, c]·(−H/‖S‖)` (`A` the
# pixel affinity's linear part), through its patch's chart (`2 × 12` per patch); for the
# pose `A·∂x*/∂X_c·[−[X_c]× I₃]` (the left perturbation's action, `2 × 6`). `nothing`
# where the solve fails.
function _window_jacobians(r::_SplineReprojection{N}, C, patches, pose) where {N}
    Xc, x, ok = _spline_solve(r, C, pose)
    ok || return nothing
    Dt, G, _, _, w = VgtkCore.Cameras._projection_jacobians(C, r.cell, x, Xc)
    A = r.pixel[1]
    Jpose = A * Dt * VgtkCore.action_jacobian(pose.manifold, r.X)
    AG = A * G
    Jpatch = ntuple(Val(N - 1)) do k
        Jc = ntuple(Val(4)) do n
            J = zero(SMatrix{2, 3, Float64})
            for m in 1:16
                r.slots[m] == (k, n) && (J += w[m] * AG)
            end
            J
        end
        _in_patch_chart(Jc, patches[k])
    end
    return _spline_pixel(r, x) - r.u, Jpose, Jpatch
end

# A full-chart Jacobian `Jfull`'s columns of the variable's free coordinates, written at `col`.
function _free_columns!(J::Matrix{Float64}, col::Int, Jfull::SMatrix, idx::AbstractVector{Int})
    @inbounds for (j, q) in enumerate(idx), i in axes(J, 1)
        J[i, col + j] = Jfull[i, q]
    end
    return col + length(idx)
end

# NLLSsolver's residual Jacobian: the rows and their Jacobian over the variables' free
# coordinates (all of them: every variable of a block is free in this refinement).
function NLLSsolver.computeresjac(varflags, r::_SplineReprojection{N, Nothing}, vars...) where {N}
    varflags == 2^N - 1 || throw(ArgumentError("refine_camera_and_poses: every patch and pose is free (varflags = $varflags)"))
    patches = ntuple(k -> vars[k], Val(N - 1))
    pose = vars[N]
    width = sum(v -> length(v.idx), vars)
    lin = _window_jacobians(r, _reprojection_controls(r, patches), patches, pose)
    lin === nothing && return SVector(NaN, NaN), fill(NaN, 2, width)
    res, Jpose, Jpatch = lin
    J = Matrix{Float64}(undef, 2, width)
    col = 0
    for k in 1:(N - 1)
        col = _free_columns!(J, col, Jpatch[k], patches[k].idx)
    end
    _free_columns!(J, col, Jpose, pose.idx)
    return res, J
end
function NLLSsolver.computeresjac(varflags, r::_SplinePoseBlock, pose)
    varflags == 1 || throw(ArgumentError("refine_camera_and_poses: the pose is free (varflags = $varflags)"))
    lin = _window_jacobians(r, r.held, (), pose)
    lin === nothing && return SVector(NaN, NaN), fill(NaN, 2, length(pose.idx))
    res, Jpose, _ = lin
    J = Matrix{Float64}(undef, 2, length(pose.idx))
    _free_columns!(J, 0, Jpose, pose.idx)
    return res, J
end

# The normal matrix of a block of 2 rows, `JᵀJ`, as the rank-2 form it is: its entries
# read from the 2 × n Jacobian when NLLSsolver adds its blocks into the system, never
# stored (a 114 × 114 heap matrix per block per iteration otherwise).
struct _Gram <: AbstractMatrix{Float64}
    J::Matrix{Float64}
end
Base.size(G::_Gram) = (size(G.J, 2), size(G.J, 2))
Base.@propagate_inbounds Base.getindex(G::_Gram, i::Int, j::Int) = G.J[1, i] * G.J[1, j] + G.J[2, i] * G.J[2, j]

# The gradient of a block of 2 rows, `Jᵀr`, lazily as the Gram matrix.
struct _GramGradient <: AbstractVector{Float64}
    J::Matrix{Float64}
    r::SVector{2, Float64}
end
Base.size(g::_GramGradient) = (size(g.J, 2),)
Base.@propagate_inbounds Base.getindex(g::_GramGradient, i::Int) = g.J[1, i] * g.r[1] + g.J[2, i] * g.r[2]

# The block's cost, gradient and normal matrix (NLLSsolver's documented extension point;
# the residual is not robustified): `(½‖r‖², Jᵀr, JᵀJ)`, the gradient and the normal
# matrix lazy (`_GramGradient`, `_Gram`).
function NLLSsolver.computecostgradhess(varflags, r::_SplineReprojection, vars...)
    res, J = NLLSsolver.computeresjac(varflags, r, vars...)
    return 0.5 * sum(abs2, res), _GramGradient(J, res), _Gram(J)
end

# One third-difference row of the present nodes, weighted and scale-free:
# `√λ·Σₘ sₘcₘ / ρ` (3 rows), `ρ = ¼Σₘ‖cₘ‖` the mean length of its own 4 controls at their
# current values, on the `N` (2 or 3) patches holding its 4 nodes, `slots[m] = (k, n)` node
# m's patch and control, `coef = √λ·s` the stencil.
struct _SplineSmoothness{N} <: NLLSsolver.AbstractResidual
    varind::SVector{N, Int}
    slots::SVector{4, NTuple{2, Int}}
    coef::SVector{4, Float64}
end
const _SplineSmoothBlock = Union{_SplineSmoothness{2}, _SplineSmoothness{3}}
NLLSsolver.ndeps(::_SplineSmoothness{N}) where {N} = NLLSsolver.static(N)
NLLSsolver.nres(::_SplineSmoothness) = NLLSsolver.static(3)
NLLSsolver.varindices(r::_SplineSmoothness) = r.varind
NLLSsolver.getvars(r::_SplineSmoothness{N}, vars::AbstractVector) where {N} = ntuple(k -> vars[r.varind[k]], Val(N))
Base.eltype(::_SplineSmoothness) = Float64
# The row's 4 controls, its rows and their mean length `ρ`.
function _smoothness_rows(r::_SplineSmoothness, patches)
    c = ntuple(m -> (s = r.slots[m]; _patch_control(patches[s[1]], s[2])), Val(4))
    ρ = sum(norm, c) / 4
    return sum(m -> r.coef[m] * c[m], 1:4) / ρ, c, ρ
end
NLLSsolver.computeresidual(r::_SplineSmoothness, patches...) = first(_smoothness_rows(r, patches))
# The rows' Jacobian in control m: `(coefₘ·I − row·ĉₘᵀ/4) / ρ` (the denominator's
# derivative `∂ρ/∂cₘ = ĉₘᵀ/4`), through each patch's chart, over its free coordinates.
function NLLSsolver.computeresjac(varflags, r::_SplineSmoothness{N}, patches...) where {N}
    varflags == 2^N - 1 || throw(ArgumentError("refine_camera_and_poses: every patch is free (varflags = $varflags)"))
    row, c, ρ = _smoothness_rows(r, patches)
    I3 = SMatrix{3, 3, Float64}(LinearAlgebra.I)
    Jm = ntuple(m -> (r.coef[m] * I3 - row * (c[m] / norm(c[m]))' / 4) / ρ, Val(4))
    J = Matrix{Float64}(undef, 3, sum(v -> length(v.idx), patches))
    col = 0
    for k in 1:N
        Jc = ntuple(n -> (m = findfirst(==((k, n)), r.slots); m === nothing ? zero(SMatrix{3, 3, Float64}) : Jm[m]), Val(4))
        col = _free_columns!(J, col, _in_patch_chart(Jc, patches[k]), patches[k].idx)
    end
    return row, J
end

# A correspondence the camera does not project at a point of the refine (or projects
# where the lattice folds), and the point: `stage` (`:guess`, `:directions` the
# directions-first estimate, `:round` the estimate of rebuild round `round`, `:gauss_newton`
# the Gauss–Newton step to the GCV weight). A status, not an exception: the stages
# branch on it, and the refine's boundary states it (`_not_projected_message`).
struct _NotProjected
    view::Int
    correspondence::Int
    X::SVector{3, Float64}
    folds::Bool
    stage::Symbol
    round::Int
end
function _not_projected_message(e::_NotProjected)
    at = e.stage === :guess ? "at the guess" : e.stage === :directions ? "at the estimate of the directions-first stage" :
        e.stage === :gauss_newton ? "at the Gauss–Newton step to the GCV weight" : "at the estimate of rebuild round $(e.round)"
    weak = e.stage === :guess ? "" :
        ": the estimate's lattice moved there, where the data hold it weakly — on noisy detections a small smoothness " *
        "weight lets those nodes swing. Use `penalty = :gcv` (the default) or a larger `penalty`"
    what = e.folds ? "the lattice folds where the point $(e.X) projects $at (the backward's Jacobian is singular there)" :
        "the point $(e.X) does not project $at (behind the camera or outside the lens's domain)"
    return "refine_camera_and_poses: view $(e.view), correspondence $(e.correspondence): " * what * weak
end

# The problem of a SplineLens camera `cam` (no forward) at the guesses: its variables
# (the patches holding a live node, column-major over the patch grid, when `free`; then
# the poses), its data blocks, its Δ³ blocks (`λ = penalty·λ₁`, `λ₁ = penalty_scale(energy,
# N, m̄)` the unit weight: the data's energy per parameter is the mean diagonal of its
# normal matrix over the present nodes' direction coordinates — the weak log-scale
# columns, ≈ δ² of them, would lower it — times the present controls' squared mean
# length, so λ₁ is scale-free as the rows are; m̄ the mean positive eigenvalue of the Δ³
# operator's DᵀD; none when the weight is 0), the counts of the correspondences dropped
# on an absent window and of those the camera does not project within a cell of one,
# the live nodes (those a row reads) and each one's patch variable and control, the
# present nodes, the patch variable count, the blocks' windows and each
# correspondence's projection (NaN where none: a rebuild's warm start). A given `λ` is
# kept (a rebuild's: no unit weight is formed); `stage` and `round` name the point the
# problem is built at (`_NotProjected`, returned when a correspondence does not
# project); `directions_only` gives the patches no log-scale coordinate (the
# directions-first solve: every log-scale held exactly, at the guess's 0). The
# projections run batched on the device (Core's `_spline_projections`), from `seeds`
# (the last build's projections) or cold.
function _spline_problem(
        cam::CameraModel, poses::AbstractVector, views::AbstractVector, free::Bool, penalty::Real;
        λ::Union{Nothing, Real} = nothing, stage::Symbol = :guess, round::Int = 0, directions_only::Bool = false,
        seeds::Union{Nothing, AbstractVector} = nothing
    )
    l = cam.lens
    b = l.backward
    gsz = size(b.controls)
    L = _PatchLayout(gsz)
    box = VgtkCore.Cameras._spline_lm_box(b.area)
    hx, hy = VgtkCore.Cameras._lattice_spacing(b.area, gsz)
    aff = VgtkCore.Cameras._pixel_affine(cam.intrinsics)
    pixel = (SMatrix{2, 2, Float64, 4}(aff.linear), SVector{2, Float64}(aff.translation))
    pose_charts = _pose_charts(poses)
    present = l.present === nothing ? trues(gsz) : l.present
    # the correspondences detected on a present window, and their camera-frame points
    cands = Tuple{Int, Int, Int, SVector{3, Float64}, SVector{2, Float64}, Int, Int}[]
    dropped = 0
    k = 0
    for (v, cs) in enumerate(views), (j, c) in enumerate(cs)
        k += 1
        X, u = _correspondence(v, j, c)
        ox, oy, _, _ = spline_window(b.area, gsz, VgtkCore.Cameras._normalized_point(cam.intrinsics, u))
        if !VgtkCore.Cameras._window_present(l.present, ox, oy)   # detected on an absent window: the lens does not image it
            dropped += 1
            continue
        end
        push!(cands, (k, v, j, X, u, ox, oy))
    end
    xs = fill(SVector(NaN, NaN), k)
    Xcs = [(P = model(pose_charts[c[2]]); rotation(P) * c[4] + translation(P)) for c in cands]
    proj = VgtkCore.Cameras._spline_projections(l, Xcs, seeds === nothing ? nothing : seeds[first.(cands)]; fold = true)
    live = falses(gsz)
    found = Tuple{SVector{3, Float64}, SVector{2, Float64}, Int, SMatrix{4, 4, CartesianIndex{2}, 16}, Rect2{Float64}}[]
    unprojected = 0
    for (q, (kc, v, j, X, u, ox, oy)) in enumerate(cands)
        x, ok, g = proj.x[q], proj.ok[q], proj.fold[q]
        # a point the camera does not project, detected within a cell of an absent window: it
        # images it onto that window (its solve cannot cross the absent nodes) — possibly an inlier
        if !ok && _near_absent(l.present, ox, oy, gsz)
            unprojected += 1
            continue
        end
        ok || return _NotProjected(v, j, X, false, stage, round)
        # a projection where the lattice folds (the backward's Jacobian there exactly singular, or not finite): no
        # Jacobian of the point exists, so it is not linearized
        (isfinite(g) && g != 0) || return _NotProjected(v, j, X, true, stage, round)
        xs[kc] = x
        ix, iy, _, _ = spline_window(b.area, gsz, x)
        o = VgtkCore.Cameras._lattice_node(b.area, gsz, ix - 1, iy - 1)        # the window's own cell
        win = SMatrix{4, 4, CartesianIndex{2}, 16}(ntuple(m -> CartesianIndex(iy - 3 + (m - 1) % 4 + 1, ix - 3 + (m - 1) ÷ 4 + 1), Val(16)))
        foreach(i -> live[i] = true, win)
        push!(found, (X, u, v, win, Rect2{Float64}(o[1], o[2], hx, hy)))
    end
    stencils = Tuple{NTuple{4, CartesianIndex{2}}, SVector{4, Float64}}[]
    Δ = nothing
    if free && (λ === nothing ? penalty : λ) > 0                  # weight 0: no smoothness, no Δ³ rows
        Δ = difference_rows(present, 3)
        nodes = vec(CartesianIndices(gsz))
        for (cols, coef) in row_stencils(Δ)                  # Core's adapter: a Δ³ row's 4 nodes and stencil, one block each
            push!(stencils, (ntuple(k -> nodes[cols[k]], Val(4)), SVector{4, Float64}(ntuple(k -> coef[k], Val(4)))))
        end
        foreach(s -> foreach(i -> live[i] = true, s[1]), stencils)
    end
    windows = [(w[1][2], w[1][1]) for (_, _, _, w, _) in found]
    if !free
        variables = [_pose_variable(ch, true) for ch in pose_charts]
        costs = _SplinePoseBlock[
            _SplinePoseBlock(SVector(v), map(i -> (0, 0), win), _WindowControls(map(i -> b.controls[i], win)), cell, box, pixel, X, u)
                for (X, u, v, win, cell) in found
        ]
        return (; variables, costs, smooth = _SplineSmoothBlock[], λ = 0.0, dropped, unprojected, live, slot = nothing,
            present, nP = 0, windows, xs)
    end
    # the scale gauge's held log-scale (with the scales free): the live node nearest x = 0; the rotation gauge's
    # view: the first with a data row (an unread pose would be singular)
    scale_node = directions_only ? CartesianIndex(0, 0) :
        argmin(i -> norm(VgtkCore.Cameras._lattice_node(b.area, gsz, i[2], i[1])), filter(i -> live[i], vec(CartesianIndices(gsz))))
    pinned_view = isempty(found) ? 1 : minimum(f -> f[3], found)
    # the patch variables: the patches holding a live node, on their live controls' coordinates
    # (the guess's controls, scale 0; a unit stand-in ẑ in a pad or absent slot, no coordinate of it)
    variables = _SplineVar[]
    var_of = zeros(Int, L.ph, L.pw)
    slot = fill((0, 0), gsz)
    for Q in 1:L.pw, P in 1:L.ph
        nodes = ntuple(n -> _slot_node(P, Q, n), Val(4))
        on = ntuple(n -> 1 <= nodes[n][1] <= gsz[1] && 1 <= nodes[n][2] <= gsz[2], Val(4))
        islive = ntuple(n -> on[n] && live[nodes[n]...], Val(4))
        any(islive) || continue
        D = SMatrix{2, 2, SVector{3, Float64}, 4}(ntuple(n -> on[n] && present[nodes[n]...] ? SVector{3, Float64}(b.controls[nodes[n]...]) : SVector(0.0, 0.0, 1.0), Val(4)))
        idx = Int[]
        foreach(n -> islive[n] && append!(idx, (2n - 1, 2n)), 1:4)
        directions_only || foreach(n -> islive[n] && CartesianIndex(nodes[n]) != scale_node && push!(idx, 8 + n), 1:4)
        push!(variables, VgtkCore.restrict(_patch_chart(D), idx))
        var_of[P, Q] = length(variables)
        foreach(n -> islive[n] && (slot[nodes[n]...] = (length(variables), n)), 1:4)
    end
    nP = length(variables)
    append!(variables, [_pose_variable(ch, v != pinned_view) for (v, ch) in enumerate(pose_charts)])
    costs = _SplineDataBlock[]
    for (X, u, v, win, cell) in found
        Ps = _patch_of(win[1][1]):_patch_of(win[end][1]); Qs = _patch_of(win[1][2]):_patch_of(win[end][2])
        np = length(Ps) * length(Qs)
        varind = SVector{np + 1, Int}(ntuple(k -> k <= np ? var_of[Ps[(k - 1) % length(Ps) + 1], Qs[(k - 1) ÷ length(Ps) + 1]] : nP + v, Val(np + 1)))
        slots = map(i -> (findfirst(==(slot[i][1]), varind), slot[i][2]), win)
        push!(costs, _SplineReprojection{np + 1, Nothing}(varind, slots, nothing, cell, box, pixel, X, u))
    end
    λ₀ = λ
    λ = 0.0
    smooth = _SplineSmoothBlock[]
    if !isempty(stencils)
        if λ₀ === nothing        # the unit weight λ₁ (a rebuild keeps its λ: none formed)
            diagonal = zeros(gsz)                                   # the data's normal matrix: its diagonal per node, the 2 direction coordinates
            for (c, win) in zip(costs, (f[4] for f in found))
                vars = NLLSsolver.getvars(c, variables)
                lin = _window_jacobians(c, _reprojection_controls(c, vars[1:(end - 1)]), vars[1:(end - 1)], vars[end])
                lin === nothing && continue
                for m in 1:16
                    k, n = c.slots[m]
                    diagonal[win[m]] += sum(abs2, lin[3][k][:, SVector(2n - 1, 2n)])
                end
            end
            λ₁ = penalty_scale(sum(diagonal) * _mean_length(b.controls, live)^2, 2count(present), mean_positive_eigenvalue(Δ))
            λ = penalty * λ₁
        else
            λ = Float64(λ₀)
        end
        for (ns, s) in stencils
            vs = unique([slot[i][1] for i in ns])
            varind = SVector{length(vs), Int}(vs)
            push!(smooth, _SplineSmoothness{length(vs)}(varind, SVector{4, NTuple{2, Int}}(ntuple(m -> (findfirst(==(slot[ns[m]][1]), varind), slot[ns[m]][2]), Val(4))), sqrt(λ) * s))
        end
    end
    return (; variables, costs, smooth, λ, dropped, unprojected, live, slot, present, nP, windows, xs)
end

# Whether a window within one cell of window `(ix, iy)` holds an absent node.
_near_absent(::Nothing, ix, iy, gsz) = false
_near_absent(present::AbstractMatrix{Bool}, ix, iy, (gh, gw)) =
    any(!VgtkCore.Cameras._window_present(present, jx, jy) for jx in max(3, ix - 1):min(gw - 1, ix + 1), jy in max(3, iy - 1):min(gh - 1, iy + 1))

_spline_blocks(p) = Iterators.flatten((p.costs, p.smooth))

# The controls of the patches at the lattice's live nodes (the guess's elsewhere: the
# absent slots and the nodes no row reads).
function _patch_controls(b::ChartSpline, p, variables::AbstractVector)
    controls = copy(b.controls)
    for i in CartesianIndices(controls)
        p.live[i] || continue
        k, n = p.slot[i]
        controls[i] = _patch_control(variables[k]::_SplinePatch, n)
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
# by e^Δℓ); the other slots are no coordinate of any variable and stay as they are.
function _turned_variables(p, variables::AbstractVector, Q, Δℓ::Float64)
    out = copy(variables)
    for k in 1:p.nP
        ch = variables[k]::_SplinePatch
        live = ntuple(n -> _slot_live(ch, n), Val(4))
        D = SMatrix{2, 2, SVector{3, Float64}, 4}(ntuple(n -> live[n] ? Q * _patch_direction(ch, n) : _patch_direction(ch, n), Val(4)))
        ℓ = SVector{4, Float64}(ntuple(n -> live[n] ? _patch_scale(ch, n) + Δℓ : _patch_scale(ch, n), Val(4)))
        out[k] = VgtkCore.restrict(_patch_chart(D, ℓ), ch.idx)
    end
    for k in (p.nP + 1):length(variables)
        ch = variables[k]::_SplinePose
        P = model(ch)
        out[k] = VgtkCore.restrict(LeftRigidManifold6(Q * SMatrix{3, 3, Float64, 9}(rotation(P)), Q * SVector{3, Float64}(translation(P))), ch.idx)
    end
    return out
end

# The solved variables moved along the gauge in one step — the rotation
# (`_gauge_rotation`) and the common scale that makes the live controls' mean length
# that of the guess `b`'s at the same nodes (`_turned_variables`) — the lattice's
# controls they give, and the rotation's two angles; the poses alone: as they are.
function _gauged(p, solved::AbstractVector, b::ChartSpline)
    p.nP == 0 && return solved, b.controls, 0.0, 0.0
    controls = _patch_controls(b, p, solved)
    Q, θ, β = _gauge_rotation(controls, b.area)
    vars = _turned_variables(p, solved, Q, log(_mean_length(b.controls, p.live) / _mean_length(controls, p.live)))
    return vars, _patch_controls(b, p, vars), θ, β
end

# The forward spline of a refined SplineLens camera, over its imaged region: Core and
# Estimation's one route (`fit_camera(SplineLens, …)`, which keeps the backward exactly).
function _refit_forward(camera::BoundedCamera, tolerance)
    fr = fit_camera(SplineLens, camera, VgtkCore.imaged_region(camera); tolerance)
    return VgtkCore.with_model(camera, estimate(fr)), retcode(fr), metadata(fr).family.forward_max_px
end

function _refine_camera_and_poses(::VgtkCore.Cameras.GaugeIntrinsics, camera::_RefinedCamera, poses::AbstractVector, views::AbstractVector; kwargs...)
    out = _refine_spline_camera_and_poses(camera, poses, views; kwargs...)
    out isa _NotProjected && throw(ArgumentError(_not_projected_message(out)))   # the public error of a point that does not project
    return out
end

# What is left of the refine's time at `deadline` (`nothing`: no cap), and whether an
# NLLSsolver solve stopped on its time cap (Core's decoder of its termination bits).
_remaining(deadline::Float64) = isinf(deadline) ? nothing : max(0.0, deadline - time())
_timed_out(result) = :max_time in nlls_stop_reasons(result.termination)

# The cap on the solve-and-rebuild rounds (each round a full solve; the windows settle
# in one or two).
const _SPLINE_REFINE_ROUNDS = 5

# One NLLSsolver solve of the problem `p` from its variables: the result and the solved variables.
function _spline_solve!(p, max_iter, ftol, xtol, max_time)
    problem = p.nP > 0 ? NLLSsolver.NLLSProblem(_SplineVar, Union{_SplineDataBlock, _SplineSmoothBlock}) :
        NLLSsolver.NLLSProblem(eltype(p.variables), _SplinePoseBlock)
    foreach(v -> NLLSsolver.addvariable!(problem, v), p.variables)
    foreach(c -> NLLSsolver.addcost!(problem, c), _spline_blocks(p))
    result = NLLSsolver.optimize!(problem, nlls_options(; max_iter, ftol, xtol, max_time))
    return result, collect(problem.variables)
end

# The problem rebuilt at the solved variables of `p` (the camera of its patches, its
# poses), keeping its λ, warm-started at its projections; `stage` and `round` name that point.
function _rebuilt_problem(p, solved::AbstractVector, cam::CameraModel, views, stage::Symbol, round::Int = 0)
    b = cam.lens.backward
    est = CameraModel(cam.intrinsics, SplineLens(ChartSpline(b.area, _patch_controls(b, p, solved)); present = cam.lens.present))
    return _spline_problem(est, _solved_poses(p, solved), views, true, p.λ; λ = p.λ, stage, round, seeds = p.xs)
end

# The solved poses of a problem with patches, as plain maps (a rebuild's start).
_solved_poses(p, solved::AbstractVector) = [_plain_pose(model(solved[k]::_SplinePose)) for k in (p.nP + 1):length(solved)]

# The problem `p` with its Δ³ rows reweighted to `λ` (`p.λ > 0`).
_with_λ(p, λ::Float64) = merge(p, (; λ, smooth = _SplineSmoothBlock[typeof(s)(s.varind, s.slots, sqrt(λ / p.λ) * s.coef) for s in p.smooth]))

# Solve `p`, then rebuild the blocks at the estimate: a point whose x* moved to another
# cell reads another window (the held one only extrapolates), so the blocks are rebuilt
# and solved again until every window is the estimate's own. At the cap the problem
# rebuilt at the estimate is the one reported (the camera's own windows). Returns the
# problem solved last, its solved variables and result, the rounds, whether the windows
# settled (checked: a solve cut short by the deadline is not rebuilt, so not settled),
# the iterations, whether a solve stopped on the deadline, the joint objective at the
# start (NLLSsolver's start cost), and the problem rebuilt at the estimate (`next`: a warm
# start) — or the `_NotProjected` of a rebuild.
function _spline_rounds(p, back::CameraModel, views, max_iter, ftol, xtol, deadline::Float64)
    iterations = 0
    timed_out = false
    cost_start = NaN
    # at most `_SPLINE_REFINE_ROUNDS` solves, whatever the deadline (`Inf` by default)
    for rounds in 1:_SPLINE_REFINE_ROUNDS
        result, solved = _spline_solve!(p, max_iter, ftol, xtol, _remaining(deadline))
        rounds == 1 && (cost_start = result.startcost)
        iterations += result.niterations
        timed_out |= _timed_out(result)
        (p.nP > 0 && !timed_out) ||
            return (; p, solved, result, rounds, settled = !timed_out, iterations, timed_out, cost_start, next = p)
        q = _rebuilt_problem(p, solved, back, views, :round, rounds)
        q isa _NotProjected && return q
        q.windows == p.windows && return (; p, solved, result, rounds, settled = true, iterations, timed_out, cost_start, next = q)
        p = q
        rounds == _SPLINE_REFINE_ROUNDS &&
            return (; p, solved = collect(p.variables), result, rounds, settled = false, iterations, timed_out, cost_start, next = p)
    end
    error("unreachable: the last round returns")
end

# The GCV system of the solved problem `p` linearized at `solved`, over the weight λ: the
# Gauss–Newton step δ from there solves `min ‖J_d δ + r_d‖² + λ‖J_s δ + s‖²`, J_d the data
# rows' Jacobian and r_d their residuals, J_s and s the Δ³ rows' at unit weight (the
# problem's divided by √p.λ: the scale-free rows' Jacobian at the solution is the weight
# times one of the weight's own), over the variables' coordinates (the gauge has none).
# Returns `(J_d, −r_d, J_s, −s)`, `gcv_curve`'s rows.
function _gcv_rows(p, solved::AbstractVector)
    rd, Jd = nlls_rows_and_jacobian(p.costs, solved)
    rs, Js = nlls_rows_and_jacobian(p.smooth, solved)
    u = 1 / sqrt(p.λ)
    return Jd, -rd, u * Js, -u * rs
end
# Its GCV curve, `n‖r_d + J_dδ‖² / (n − tr H_λ)²`, `H_λ = J_d M_λ⁻¹ J_dᵀ` (the stack's
# `gcv_curve`, gcv.jl, at its defaults: the pencil of `(J_sᵀJ_s, J_dᵀJ_d + J_sᵀJ_s)`, every
# λ from one decomposition): `λ -> (; gcv, rss, dof, y)`, `y` the step δ.
_gcv_linearization(p, solved::AbstractVector) = gcv_curve(_gcv_rows(p, solved)...)

# The variables `solved` moved by the chart step `δ`, in the Jacobian's column order
# (`nlls_rows_and_jacobian`: each variable's coordinates in turn).
function _moved_along(solved::AbstractVector, δ::AbstractVector)
    offsets = cumsum([0; [length(v.idx) for v in solved]])
    return _SplineVar[NLLSsolver.update(x, δ, offsets[v] + 1) for (v, x) in enumerate(solved)]
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
# the weight, its tr H and the iterations of both solves. When `max_time` cuts the β = 1
# solve short, no weight is searched: its estimate is returned at β = 1, with no curve
# and no tr H (or the `_NotProjected` of a rebuild).
const _GCV_LOG10_BRACKET = (-12.0, 0.0)
function _gcv_search(p, λ₁::Float64, back::CameraModel, views, max_iter, ftol, xtol, deadline::Float64)
    points = round(Int, _GCV_LOG10_BRACKET[2] - _GCV_LOG10_BRACKET[1]) + 1
    st₀ = _spline_rounds(p, back, views, max_iter, ftol, xtol, deadline)
    st₀ isa _NotProjected && return st₀
    st₀.timed_out && return st₀, nothing, false, exp10(_GCV_LOG10_BRACKET[2]), nothing
    curve = _gcv_linearization(st₀.p, st₀.solved)
    seen = Tuple{Float64, Float64}[]
    score(λ) = (e = curve(λ); push!(seen, (λ, e.gcv)); e)
    found = gcv_select(score, λ₁; log10_bracket = _GCV_LOG10_BRACKET, points)
    evaluated = [(; β = λ / λ₁, λ, gcv) for (λ, gcv) in sort(unique(seen))]
    start = _rebuilt_problem(_with_λ(st₀.p, found.λ), _moved_along(st₀.solved, curve(found.λ).y), back, views, :gauss_newton)
    start isa _NotProjected && (start = _with_λ(st₀.next, found.λ))
    st = _spline_rounds(start, back, views, max_iter, ftol, xtol, deadline)
    st isa _NotProjected && return st
    return merge(st, (; iterations = st.iterations + st₀.iterations)), evaluated, found.at_bracket_end, found.β, found.dof
end

# The refinement: the estimate, or the `_NotProjected` of a correspondence (the public
# boundary, `_refine_camera_and_poses`, states it).
function _refine_spline_camera_and_poses(
        camera::_RefinedCamera, poses::AbstractVector, views::AbstractVector;
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
    # log-scale held exactly: the patches have no log-scale coordinate (14–18 iterations). (3) Everything (9 there): at the weight given, or
    # over GCV's search (stages 1–2 once, at the grid's top weight: a start aid that keeps
    # the extrapolated nodes from swinging on noisy detections, as 1e-8 lets them).
    # `max_time` caps the whole refine: every stage and round reads what is left of it.
    deadline = max_time === nothing ? Inf : time() + Float64(max_time)
    iterations = 0
    timed_out = false
    if isfree
        pre = _refine_spline_camera_and_poses(back, poses, views; free = (), penalty = w, max_iter, ftol, xtol, max_time = _remaining(deadline))
        pre isa _NotProjected && return pre
        iterations += pre[3]
        timed_out |= pre[2] === :max_time
        # a stage cut short by `max_time` is not built on: the joint solve starts where it did
        if timed_out
            p = _spline_problem(back, pre[1].poses, views, true, w)
        else
            p₀ = _spline_problem(back, pre[1].poses, views, true, w; directions_only = true)
            p₀ isa _NotProjected && return p₀
            r₀, s₀ = _spline_solve!(p₀, max_iter, ftol, xtol, _remaining(deadline))
            iterations += r₀.niterations
            timed_out |= _timed_out(r₀)
            p = timed_out ? _spline_problem(back, pre[1].poses, views, true, w) : _rebuilt_problem(p₀, s₀, back, views, :directions)
        end
    else
        p = _spline_problem(back, poses, views, isfree, w)
    end
    p isa _NotProjected && return p
    b = l.backward
    # the unknowns: the live controls' 3 coordinates (less the gauge's 4: the common scale and the rotation) and the poses'; the rows: the data's and the Δ³ rows'
    counts(q) = (m = length(q.costs); (m, (isfree ? 3count(q.live) - 4 : 0) + 6V, 2m + 3length(q.smooth)))
    n, np, nobs = counts(p)
    nobs > np || throw(ArgumentError("refine_camera_and_poses: underdetermined: $n correspondences and $(length(p.smooth)) third-difference rows ($nobs rows) for $np unknowns"))

    gcv = nothing
    gcv_at_bound = false
    effective_dof = nothing
    if search && isfree
        sg = _gcv_search(p, p.λ / w, back, views, max_iter, ftol, xtol, deadline)
        sg isa _NotProjected && return sg
        st, gcv, gcv_at_bound, w, effective_dof = sg
    else
        st = _spline_rounds(p, back, views, max_iter, ftol, xtol, deadline)
        st isa _NotProjected && return st
    end
    iterations += st.iterations
    timed_out |= st.timed_out
    p, solved, result, rounds, settled, cost_start = st.p, st.solved, st.result, st.rounds, st.settled, st.cost_start
    n, np, nobs = counts(p)                                    # a rebuild may take back a point the guess did not project

    # the gauge, one step: the live controls and the poses turned by ONE rotation (the axis onto ẑ, the roll), the live controls scaled to the guess's mean length
    vars, controls, θ, roll = _gauged(p, solved, b)
    poses_out = [model(vars[p.nP + v]::_SplinePose) for v in 1:V]
    all(P -> all(isfinite, rotation(P)) && all(isfinite, translation(P)), poses_out) && all(c -> all(isfinite, c), controls[held]) ||
        throw(ErrorException("refine_camera_and_poses: the LM step is not finite"))

    r, J = nlls_rows_and_jacobian(_spline_blocks(p), vars)
    code = nlls_retcode(result, J, r; tol = _CHART_LM_STATIONARITY_TOL, ftol)
    timed_out && (code = :max_time)                          # a stage or round cut short by the refine's `max_time`
    rd = r[1:2n]
    ν = nobs - np
    refined = isfree ? CameraModel(cam.intrinsics, SplineLens(ChartSpline(b.area, controls); present = l.present)) : cam
    out_camera = refined === cam ? camera : VgtkCore.with_model(camera, refined)
    forward_max_px = nothing
    if forward === :refitted && out_camera isa BoundedCamera
        out_camera, fcode, forward_max_px = _refit_forward(out_camera, forward_tolerance)
        fcode === :converged || code !== :converged || (code = fcode)
    end
    out = [relabel(VgtkCore.map_frames(p0)..., _plain_pose(P)) for (p0, P) in zip(poses, poses_out)]
    n_nodes_unrefined = isfree ? count(p.present .& .!p.live) : count(p.present)
    spline = (; β = isfree ? w : 0.0, λ = p.λ, gcv, gcv_at_bound, effective_dof, n_nodes_unrefined,
        n_dropped_window = p.dropped, n_dropped_unprojected = p.unprojected, rounds, windows_settled = settled,
        forward, forward_max_px, gauge_angle = θ, gauge_roll = roll, cost_start)
    return (; camera = out_camera, poses = out), code, iterations, sqrt(sum(abs2, rd) / n), n, CameraPosesRefineMetadata(rd, sum(abs2, r) / ν, ν, free, spline)
end
