# =============================================================================
# refine_robot_world_hand_eye — reprojection-error refinement of a robot-world calibration
# =============================================================================
#
# The measurement-level upgrade `fit_robot_world_hand_eye`'s header names: refine the
# M + 1 transforms of A_i X_m = Y B_i^m against the FEATURES — the marker
# points' reprojection error — rather than against per-frame PnP poses. With
# the camera and the rig poses A_i fixed, marker m's point X seen in frame i
# lands at
#
#     û = project(camera, Y⁻¹ ∘ A_i ∘ X_m (X)),
#
# and the estimate minimizes Σ‖û − u‖² over [Y | X₁ … X_M] in the calibration's
# chart — `fit_robot_world_hand_eye`'s: M + 1 `RigidExpManifold6` blocks
# (exp(δω)·R, t + δt) — with NLLSsolver's Levenberg–Marquardt: each pose one
# variable (the chart itself, an NLLSsolver variable through VGC's NLLSsolver
# extension, re-anchored at every accepted step), each point one residual block. A point depends on Y and its own X_m only, so its
# 2 × 12 Jacobian is differentiated (ForwardDiff, through the retraction of
# those two blocks) and NLLSsolver accumulates its 12 × 12 block of JᵀJ.
#
# The covariance is the stack's one model covariance, VGC's
# `implicit_covariance`, over the points' reprojection rows at the estimate
# (block form; unit isotropic pixel noise, so (JᵀJ)⁻¹), scaled by
# ŝ² = RSS/dof; the result is assembled by `fit_robot_world_hand_eye`'s one assembly
# (`_robot_world_hand_eye_result`): the joint `cov` in block order [Y | X₁ … X_M], the
# marginal `cov_Y`, the smallest eigenpair of JᵀJ and the rank rule. Unlike
# `fit_robot_world_hand_eye`'s, this one is the covariance of the estimator's own
# stationary point (an MLE under isotropic Gaussian pixel noise).
# =============================================================================

# Nothing is static in the marker count: the variables are a vector of pose
# blocks, so one compiled method serves every M.

"""
    refine_robot_world_hand_eye(guess::RobotWorldHandEyeCalibration{K}, camera::CameraModel, markers;
                       optimization_strategy, scale | width, max_iter)
        -> FitResult{RobotWorldHandEyeCalibration{K}, <:RobotWorldHandEyeMetadata{K, Nothing}}
        # defaults: configs/default.toml [refine_robot_world_hand_eye]

Refine a robot-world calibration `A_i X_m = Y B_i^m` against the reprojection
error of the markers' FEATURES: minimize `Σ‖project(camera, Y⁻¹ ∘ A_i ∘ X_m (X)) − u‖²`
over the shared `Y = T_WC` and the per-marker `X_m = T_RPm`, starting at
`guess` (e.g. [`fit_robot_world_hand_eye`](@ref)'s estimate). The camera and the rig
poses `A` are FIXED; nothing is re-estimated from scratch and the family is
unchanged — the `refine` contract.

Robot-world/hand-eye calibration is the literature's name for `AX = YB`
(Zhuang, Roth & Shah 1994; Dornaika & Horaud 1998). Ours is the EYE-TO-HAND
configuration — the camera fixed in the world, the targets on the moving rig —
with frames in the stack's `T_AB` notation: `Y = T_WC`, `X_m = T_RPm`,
`A_i = T_WR` (the rig pose at frame i) and `B_i^m = T_CPm` (marker m in the
camera at frame i). Function names do not encode frames.

# Arguments
- `guess::RobotWorldHandEyeCalibration{K}` — the starting `Y` and the keyed `X`
- `camera::CameraModel` — the (fixed) camera; `project(camera, X_C)` maps a
  camera-frame point to pixels
- `markers::AbstractDict{K, <:AbstractVector{<:Cspond}}` — per marker id, one
  element per frame it is seen in: that frame's rig pose `A_i = T_WR`
  (`unlabel`ed at the boundary) paired with the features seen with it,
  `A_i => [X => u, …]` — `X` a point in the marker's frame, `u` its measured
  pixel. The key sets must match, `keys(markers) == Set(guess.ids)`, and every
  marker must have at least one feature (`ArgumentError` otherwise): every
  marker of the guess is a parameter of this refinement and is refined.

# Keywords
- `optimization_strategy` — the refine contract's strategy, as every `refine_*`
  takes it: `Triggs{Shape}` (the default, `Triggs{WelschShape}`), ONE
  robustified LM (Core's `loss_robustifier`); `Irls{Shape}`, LMs at the loss
  weights ρ′ until the robust cost stops decreasing or its `max_iter` cycles;
  or `nothing`, plain least squares. Both robust strategies start from `guess`
  with no least-squares stage. (`local_optimization` is refused, naming this.)
- `scale` or `width` — EXACTLY ONE for a robust strategy, neither for least
  squares (`ArgumentError` otherwise): `scale = σ`, the features' noise scale in
  pixels, or an upstream result whose `noise_scale` it is (the loss width is
  then `efficiency_constant(Shape, 2) × σ`, or the strategy's own constant
  times σ); `width = w`, the loss width in pixels. A refine estimates no
  scale: the metadata records the one it ran with (`scale`, `scale_source`,
  `loss_width`) and the weights ρ′ at the estimate; `n` is their support and
  `rms` the weighted one, `√(Σ wᵢ‖rᵢ‖² / Σ wᵢ)`.
- `max_iter` — the LM's cap on ACCEPTED iterations (NLLSsolver's count; one
  may hide ~8–10 rejected candidate steps); `iterations` sums them over an
  Irls run's LMs.

Any number of markers: the LM is NLLSsolver's, one pose block per marker and one
residual block per point, with nothing compiled per marker count. Its termination
record (`retcode`, `stop_rule`; VisualGeometryToolkitCore's `nlls_termination`) is
`ReturnCode.Success` only when the estimate is stationary, `MaxIters` at the
iteration cap (an Irls run capped at its cycles), `MaxTime` at the solver's time
cap, `Failure`/`InitialFailure` (`NonFinite`) when a solve was non-finite or
failed (the estimate is then the last accepted one) and `Stalled` otherwise
(under a robust strategy the stationarity of the robust cost). Rig poses, points and pixels must
be finite, and every point must project at the guess (in front of the camera,
inside the lens's domain): an `ArgumentError` names the first that does not.

The covariance is VGC's `implicit_covariance` over the reprojection rows at
the estimate (each scaled by √ρ′ under a robust strategy) times `ŝ² = RSS/dof`
of those rows — see [`RobotWorldHandEyeMetadata`](@ref).

# Units
The guess, the rig poses and the marker points are plain or physical, all one
form (a mix is a `MethodError`). Physical ones (lengths) are read in metres
at entry — exactly, the stored value — the LM runs on plain metres, and the
calibration comes back physical (`Y`, `X` in metres), equal after
`ustrip(m, ·)` to the plain refinement's on the inputs' `ustrip(m, ·)`. The
residuals and `rms` are pixels either way; the covariance's translation
entries are plain numbers in the translation's unit, metres for physical
inputs. Plain inputs give a plain calibration in their own unit, as before.
"""
function refine_robot_world_hand_eye(
        guess::RobotWorldHandEyeCalibration{K, <:PlainEuclideanMap{3}}, camera::CameraModel,
        markers::AbstractDict{
            K, <:AbstractVector{<:Cspond{<:PlainEuclideanMap{3}, <:AbstractVector{<:Cspond{<:StaticVector{3, <:Real}, <:StaticVector{2}}}}},
        };
        max_iter::Int = _dflt("refine_robot_world_hand_eye", "max_iter"),
        optimization_strategy = _refine_default(Val(:refine_robot_world_hand_eye)),
        scale = nothing, width = nothing, local_optimization = RetiredKeyword(),
    ) where {K}
    refuse_retired_keyword(local_optimization, :local_optimization, :optimization_strategy)
    strategy = bundle_loss(optimization_strategy; scale, width, caller = "refine_robot_world_hand_eye")
    ids = block_order(markers, guess.X)             # the key sets must match: every marker of the guess is refined
    M = length(ids)
    # Flattened marker by marker in `ids` order: each element's rig pose once
    # (RA/tA, indexed by pt_frame), each of its features one point.
    RA = SMatrix{3, 3, Float64, 9}[]; tA = SVector{3, Float64}[]
    pt_frame = Int[]; pt_marker = Int[]
    Xs = SVector{3, Float64}[]; us = SVector{2, Float64}[]
    counts = _flatten_rig_observations(markers, ids, "features") do m, id, j, RAj, tAj, features
        push!(RA, RAj); push!(tA, tAj)
        f = length(RA)
        for (q, pc) in enumerate(features)
            push!(pt_frame, f); push!(pt_marker, m)
            push!(Xs, SVector{3, Float64}(first(pc)[1], first(pc)[2], first(pc)[3]))
            push!(us, SVector{2, Float64}(last(pc)[1], last(pc)[2]))
            (all(isfinite, Xs[end]) && all(isfinite, us[end])) ||
                throw(ArgumentError("marker $id, element $j, feature $q: non-finite point or pixel"))
        end
        return length(features)
    end
    n = length(us)
    N = 6 * (M + 1)
    2n > N || throw(ArgumentError("underdetermined: $n points ($(2n) residuals) for $N unknowns"))
    return _refine_robot_world_hand_eye(guess, counts, camera, RA, tA, pt_frame, pt_marker, Xs, us, max_iter, strategy)
end

# Physical inputs: the guess, the rig poses and the points stripped in metres
# once, the plain refinement, the calibration read back in metres once.
function refine_robot_world_hand_eye(
        guess::RobotWorldHandEyeCalibration{K, <:_PhysicalPose3}, camera::CameraModel,
        markers::AbstractDict{
            K, <:AbstractVector{<:Cspond{<:_PhysicalPose3, <:AbstractVector{<:Cspond{<:_PhysicalPoint3, <:StaticVector{2}}}}},
        };
        kwargs...
    ) where {K}
    features(fs) = _map_csponds(_plain_metres, identity, fs)
    plain = Dict(id => _map_csponds(_plain_metres, features, els) for (id, els) in markers)::AbstractDict{
        K, <:AbstractVector{<:Cspond{<:PlainEuclideanMap{3}, <:AbstractVector{<:Cspond{<:StaticVector{3, <:Real}, <:StaticVector{2}}}}},
    }
    return _attach_metres(refine_robot_world_hand_eye(_plain_metres(guess), camera, plain; kwargs...))
end

# One point's reprojection residual as a function of its 12 chart coordinates
# (δY, δX_m) in the product of its two blocks' charts `chY`, `chX` —
# eltype-generic, so ForwardDiff Duals flow through the retraction and the lens.
@inline function _rw_point_residual(z::SVector{12}, chY, chX, RAi, tAi, X, u, camera)
    T = eltype(z)
    Y, Xm = retract(ProductManifold(chY, chX), z)
    Pw = RAi * Xm(X) + tAi                                      # marker → rig → world
    return SVector{2, T}(VgtkCore.reprojection_error(camera, inv(Y), Pw, u))   # world → camera → pixel
end

# The NLLSsolver problem: one variable per pose — Y first, then X_m in
# `guess.ids` order — each the calibration's chart itself
# (`RigidExpManifold6`, re-anchored at every accepted step), and one residual
# block per marker point, depending on Y and its own X_m. NLLSsolver
# accumulates each point's 12 × 12 block of JᵀJ into the (M + 1) × (M + 1)
# block normal equations; nothing is static in M.
const _RwPose = RigidExpManifold6{Float64}

# A robust refine scales the block by `sqrt_weight` (an Irls cycle's √ρ′; 1 otherwise) or robustifies its squared
# norm by `robust` (Triggs: Core's `loss_robustifier`; `NoRobust()` otherwise), as the camera-and-poses bundle does.
struct _RwPointResidual{C, R} <: NLLSsolver.AbstractResidual
    varind::SVector{2, Int}           # (Y, X_m) = variables (1, m + 1)
    RA::SMatrix{3, 3, Float64, 9}     # the point's frame's rig pose
    tA::SVector{3, Float64}
    X::SVector{3, Float64}            # the point, in its marker's frame
    u::SVector{2, Float64}            # its pixel
    camera::C
    sqrt_weight::Float64
    robust::R
end
_RwPointResidual{C}(varind, RA, tA, X, u, camera) where {C} =
    _RwPointResidual{C, NLLSsolver.NoRobust}(varind, RA, tA, X, u, camera, 1.0, NLLSsolver.NoRobust())
_with_robust(r::_RwPointResidual{C}, sqrt_weight::Float64, robust::R) where {C, R} =
    _RwPointResidual{C, R}(r.varind, r.RA, r.tA, r.X, r.u, r.camera, sqrt_weight, robust)
NLLSsolver.robustkernel(r::_RwPointResidual) = r.robust
NLLSsolver.ndeps(::_RwPointResidual) = NLLSsolver.static(2)
NLLSsolver.nres(::_RwPointResidual) = NLLSsolver.static(2)
NLLSsolver.varindices(r::_RwPointResidual) = r.varind
NLLSsolver.getvars(r::_RwPointResidual, vars::Vector) = (vars[r.varind[1]]::_RwPose, vars[r.varind[2]]::_RwPose)
Base.eltype(::_RwPointResidual) = Float64

# The point's residual at the charts' origins, and its 2 × 12 Jacobian in
# (δY, δX_m) — ForwardDiff through the retraction of the two blocks.
_rw_residual(r::_RwPointResidual, chY, chX) =
    r.sqrt_weight * _rw_point_residual(zero(SVector{12, Float64}), chY, chX, r.RA, r.tA, r.X, r.u, r.camera)
_rw_jacobian(r::_RwPointResidual, chY, chX) = r.sqrt_weight * ForwardDiff.jacobian(
    z -> _rw_point_residual(z, chY, chX, r.RA, r.tA, r.X, r.u, r.camera), zero(SVector{12, Float64})
)
NLLSsolver.computeresidual(r::_RwPointResidual, Y::_RwPose, X::_RwPose) = _rw_residual(r, Y, X)
# The block's cost as the LM reads it: non-finite is +Inf (`finite_cost`).
NLLSsolver.computecost(vars::Vector{_RwPose}, r::_RwPointResidual) =
    finite_cost(NLLSsolver.computecost(r, NLLSsolver.getvars(r, vars)...))
# Every pose is free in this refinement (`optimize!` over all variables), so
# NLLSsolver asks for both blocks' columns: `varflags` = 0b11.
function NLLSsolver.computeresjac(varflags, r::_RwPointResidual, Y::_RwPose, X::_RwPose)
    varflags == 3 || throw(ArgumentError("refine_robot_world_hand_eye: every pose is free (varflags = $varflags)"))
    return _rw_residual(r, Y, X), _rw_jacobian(r, Y, X)
end

function _refine_robot_world_hand_eye(
        guess::RobotWorldHandEyeCalibration, counts, camera::C, RA, tA, fr, mk, Xs, us, max_iter, (lo, input, loss)
    ) where {C}
    n = length(us)
    costs = [_RwPointResidual{C}(SVector(1, mk[k] + 1), RA[fr[k]], tA[fr[k]], Xs[k], us[k], camera) for k in 1:n]
    poses0 = _RwPose[RigidExpManifold6(guess.Y); [RigidExpManifold6(guess.X[id]) for id in guess.ids]]
    # Every point must project at the guess: NaN is the lens's answer for a
    # ray it does not image (behind the camera, outside its domain), and one
    # NaN row would silently stall the LM.
    for (k, c) in enumerate(costs)
        all(isfinite, _rw_residual(c, poses0[1], poses0[c.varind[2]])) || throw(
            ArgumentError(
                "marker $(guess.ids[mk[k]]): point $(Xs[k]) does not project at the guess " *
                    "(behind the camera or outside the lens's domain)"
            )
        )
    end

    # The stack's one knob map (VGC `nlls_options`: one thread, so the result
    # does not depend on the thread count; no time cap) with this fit's own
    # settings: the plain λ·I damping and NLLSsolver's default tolerances
    # (relative cost decrease and step 1e-15), which it has always run with.
    options = nlls_options(; max_iter, ftol = 1.0e-15, xtol = 1.0e-15, iterator = :levenbergmarquardt)
    function solve(blocks, start)
        RT = Union{unique(typeof.(blocks))...}
        pr = NLLSsolver.NLLSProblem(_RwPose, RT)
        foreach(ch -> NLLSsolver.addvariable!(pr, ch), start)
        foreach(c -> NLLSsolver.addcost!(pr, c), blocks)
        return pr, NLLSsolver.optimize!(pr, options)
    end
    # Least squares (`optimization_strategy = nothing`), or the refine contract's robust strategy from the guess with
    # no least-squares stage: Triggs, one robustified LM; Irls, weighted LMs at the loss weights ρ′ until the robust
    # cost stops decreasing or `max_iter` cycles. The loss runs at the caller's frozen scale or width.
    stationarity_blocks = costs
    irls_capped = false             # an Irls run that reached its `max_iter` cycles is capped, whatever its last solve
    if lo === nothing
        problem, result = solve(costs, copy(poses0))
        iters = result.niterations
        weights = unit_weights(Float64, n)
    else
        shape = strategy_loss(lo)(); sq = _sq_loss_width(loss)
        sqnorms(vars) = [sum(abs2, _rw_residual(c, vars[1], vars[c.varind[2]])) for c in costs]
        if lo isa Triggs
            problem, result = solve([_with_robust(c, 1.0, loss_robustifier(shape, sqrt(sq))) for c in costs], copy(poses0))
            iters = result.niterations
        else
            weighted(q) = [_with_robust(c, sqrt(weight(shape, q[k], sq)), NLLSsolver.NoRobust()) for (k, c) in enumerate(costs)]
            q = sqnorms(poses0)
            prev = sum(x -> ρ(shape, x, sq), q)
            problem, result = solve(weighted(q), copy(poses0))
            iters = result.niterations
            irls_capped = true          # until a cycle's robust cost stops decreasing
            for _ in 2:lo.max_iter
                q = sqnorms(problem.variables)
                cost = sum(x -> ρ(shape, x, sq), q)
                cost < prev || (irls_capped = false; break)
                prev = cost
                problem, result = solve(weighted(q), collect(problem.variables))
                iters += result.niterations
            end
        end
        weights = _loss_weights(shape, sq, sqnorms(problem.variables))
        # the stationarity of the robust cost and its covariance: the rows and Jacobian scaled by √ρ′
        stationarity_blocks = [_with_robust(c, sqrt(weights[k]), NLLSsolver.NoRobust()) for (k, c) in enumerate(costs)]
    end
    charts = collect(problem.variables)   # each anchored at its estimate
    poses = map(model, charts)
    all(_finite_pose, poses) ||   # refine_camera_and_poses.jl
        throw(ErrorException("refine_robot_world_hand_eye: the LM step is not finite"))

    # the reprojection rows and their dense 2n × 6(M + 1) Jacobian at the charts' origins — the covariance's and the
    # stall postcondition's (point k's rows 2k − 1, 2k; its columns those of Y and of its X_m), √ρ′-scaled under a
    # robust strategy: the stall postcondition, the eigen-diagnostics and the covariance are the weighted problem's
    r, Js = nlls_rows_and_jacobian(stationarity_blocks, charts)
    J = Matrix(Js)
    # the termination record: an Irls run capped at its cycles, else the last solve's (NLLSsolver's bits under the
    # stall postcondition at the returned point)
    termination = irls_capped ? (ReturnCode.MaxIters, IterationCap) :
        nlls_termination(result, () -> VgtkCore.is_stationary(J, r; tol = _CHART_LM_STATIONARITY_TOL, ftol = _HAND_EYE_FTOL))
    # The covariance's blocks are the ELEMENTS — one rig pose's features, whose
    # points are contiguous (flattened element by element) — each a dense view
    # of its rows of J with measurement Jacobian −I: the same (JᵀJ)⁻¹ as per-
    # point blocks, without a static 2 × N block and an N × N static product
    # per point. Measured 2026-10-01 at M = 20 (N = 126, 6000 points): the
    # per-point form took 370 ms and 2.2 GB per call, 4× the LM itself.
    stops = [k for k in 1:n if k == n || fr[k + 1] != fr[k]]
    starts = [1; stops[1:(end - 1)] .+ 1]
    rows(e) = (2starts[e] - 1):(2stops[e])
    function covariance(s2)
        Σ̃ = implicit_covariance(e -> (view(J, rows(e), :), Diagonal(fill(-1.0, length(rows(e))))), eachindex(starts))
        return Σ̃ === nothing ? nothing : s2 .* Σ̃
    end
    r_raw = lo === nothing ? r : first(nlls_rows_and_jacobian(costs, charts))   # the reprojection rows themselves
    return _robot_world_hand_eye_result(
        _robot_world_hand_eye_calibration(poses, guess.ids), r_raw, J, 2, counts, covariance, nothing, nothing;
        termination, iterations = iters, rms_rows = n, strategy = lo, weights,
        scale = _recorded_scale(input), scale_source = VgtkCore.scale_source(input),
        loss_width = sqrt(_sq_loss_width(loss))
    )
end
