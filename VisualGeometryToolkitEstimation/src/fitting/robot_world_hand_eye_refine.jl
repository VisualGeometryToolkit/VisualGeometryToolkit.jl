# =============================================================================
# RobotWorldHandEyeProblem — the reprojection-error refine of a robot-world calibration
# =============================================================================
#
# The observation-level upgrade the robot-world fit's header names: refine the
# M + 1 transforms of A_i X_m = Y B_i^m against the FEATURES — the marker
# points' reprojection error — rather than against per-frame PnP poses. With
# the camera and the rig poses A_i fixed, marker m's point X seen in frame i
# lands at
#
#     û = project(camera, Y⁻¹ ∘ A_i ∘ X_m (X)),
#
# and the estimate minimizes Σ‖û − u‖² over [Y | X₁ … X_M] in the calibration's
# chart — the robot-world fit's: M + 1 `RigidExpManifold6` blocks
# (exp(δω)·R, t + δt) — with NLLSsolver's Levenberg–Marquardt: each pose one
# variable (the chart itself, an NLLSsolver variable through VGC's NLLSsolver
# extension, re-anchored at every accepted step), each point one residual block. A point depends on Y and its own X_m only, so its
# 2 × 12 Jacobian is differentiated (ForwardDiff, through the retraction of
# those two blocks) and NLLSsolver accumulates its 12 × 12 block of JᵀJ.
#
# The covariance is the stack's one model covariance, VGC's
# `implicit_covariance`, over the points' reprojection rows at the estimate
# (block form; unit isotropic pixel noise, so (JᵀJ)⁻¹), scaled by
# ŝ² = RSS/dof; the result is assembled by the robot-world fit's one assembly
# (`_robot_world_hand_eye_result`): the joint `cov` in block order [Y | X₁ … X_M], the
# marginal `cov_Y`, the smallest eigenpair of JᵀJ and the rank rule. Unlike
# the fit's, this one is the covariance of the estimator's own
# stationary point (an MLE under isotropic Gaussian pixel noise).
# =============================================================================

# Nothing is static in the marker count: the variables are a vector of pose
# blocks, so one compiled method serves every M.

"""
    RobotWorldHandEyeProblem(camera::CameraModel, markers)

    solve(RefineProblem(RobotWorldHandEyeProblem(camera, markers), guess::RobotWorldHandEyeCalibration{K}),
          LevenbergMarquardt(; optimization_strategy); scale | width, stops)
        -> FitResult{RobotWorldHandEyeCalibration{K}, <:RobotWorldHandEyeMetadata{K, Nothing}}
        # defaults: configs/default.toml [robot_world_hand_eye_problem] (the strategy), [camera_poses_problem] (the stops)

The reprojection refine of a robot-world calibration, a multi-block problem (VisualGeometryToolkitCore's
`AbstractBlockProblem`): the camera and the keyed markers it is refined against; the guess is the
`RefineProblem`'s `u0`.

Refine a robot-world calibration `A_i X_m = Y B_i^m` against the reprojection
error of the markers' FEATURES: minimize `Σ‖project(camera, Y⁻¹ ∘ A_i ∘ X_m (X)) − u‖²`
over the shared `Y = T_WC` and the per-marker `X_m = T_RPm`, starting at
`guess` (e.g. the robot-world fit's estimate, [`ReweightedLeastSquares`](@ref)). The camera and the rig
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
- `markers::AbstractDict{K, <:AbstractVector{<:Correspondence}}` — per marker id, one
  element per frame it is seen in: that frame's rig pose `A_i = T_WR`
  (`unlabel`ed at the boundary) paired with the features seen with it,
  `A_i => [X => u, …]` — `X` a point in the marker's frame, `u` its measured
  pixel. The key sets must match, `keys(markers) == Set(guess.ids)`, and every
  marker must have at least one feature (`ArgumentError` otherwise): every
  marker of the guess is a parameter of this refinement and is refined.

# The algorithm and the solve keywords
- `optimization_strategy` (a field of `LevenbergMarquardt`) — the refine contract's strategy:
  `Triggs{Shape}` (the class default, `Triggs{WelschShape}`), ONE
  robustified LM (Core's `loss_robustifier`) from `guess` with no
  least-squares stage; or `nothing`, plain least squares. An `Irls{Shape}` is
  an `ArgumentError` naming `Triggs{Shape}` (a refine runs one LM, no
  reweighting cycles).
- `scale` or `width` — EXACTLY ONE for a robust strategy, neither for least
  squares (`ArgumentError` otherwise): `scale = σ`, the features' noise scale in
  pixels, or an upstream result whose `noise_scale` it is (the loss width is
  then `efficiency_constant(Shape, 2) × σ`, or the strategy's own constant
  times σ); `width = w`, the loss width in pixels. A refine estimates no
  scale: the metadata records the one it ran with (`scale`, `scale_source`,
  `loss_width`) and the normalized weights ρ′/ρ′(0) at the estimate; `n` is their support and
  `rms` the weighted one, `√(Σ wᵢ‖rᵢ‖² / Σ wᵢ)`.
- `stops` — the LM's [`LmStops`](@extref VisualGeometryToolkitCore.LmStops) (`max_iter`: its cap on ACCEPTED
  iterations; NLLSsolver's count, one may hide ~8–10 rejected candidate steps); default the camera-and-poses
  bundle's (`configs/default.toml [camera_poses_problem]` `max_iter`, `ftol`, `xtol`).

Any number of markers: the LM is Core's one multi-block LM (the block contract of its `LevenbergMarquardt` refine), one pose block per marker and one
residual block per point, with nothing compiled per marker count, at the camera-and-poses bundle's stops
([`CameraPosesProblem`](@ref)'s) and the algorithm's damping (Marquardt's DᵀD by default): this refine has no
LM settings of its own. Its termination
record (`retcode`, `stop_rule`; VisualGeometryToolkitCore's one LM) is
`ReturnCode.Success` at a tolerance stop (`ftol`, `xtol`; Core 0.11.0: whatever the gradient there) or at the
rejection cap when the estimate is stationary, `MaxIters` at the
iteration cap (no time cap: it runs without `max_time`), `Failure`/`InitialFailure` (`NonFinite`) when a solve was non-finite or
failed (the estimate is then the last accepted one) and `Stalled` at the rejection cap short of a stationary point
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
struct RobotWorldHandEyeProblem{C <: CameraModel, M <: AbstractDict} <: AbstractBlockProblem
    camera::C
    markers::M
end
const _PlainRigMarkers{K} = AbstractDict{K, <:AbstractVector{<:Correspondence{<:PlainEuclideanMap{3},
    <:AbstractVector{<:Correspondence{<:StaticVector{3, <:Real}, <:StaticVector{2}}}}}}
const _PhysicalRigMarkers{K} = AbstractDict{K, <:AbstractVector{<:Correspondence{<:_PhysicalPose3,
    <:AbstractVector{<:Correspondence{<:_PhysicalPoint3, <:StaticVector{2}}}}}}
RobotWorldHandEyeProblem(camera::CameraModel, markers::Union{_PlainRigMarkers, _PhysicalRigMarkers}) =
    RobotWorldHandEyeProblem{typeof(camera), typeof(markers)}(camera, markers)

default_optimization_strategy(::RobotWorldHandEyeProblem) = _dflt_strategy("robot_world_hand_eye_problem", :robot_world_hand_eye_problem)
default_stops(::RobotWorldHandEyeProblem) = _CAMERA_POSES_STOPS   # the camera-and-poses bundle's (camera_poses.jl)

# The block contract (Core's multi-block LM): the bundle stated ONCE at the guess — one pose variable each for Y and
# every X_m, one residual block per marker point, and the assembly of its result — by the one pipeline
# (`_robot_world_hand_eye_run`, a `_BundleStatement`: `block_structure`), and the result assembled from that
# statement with Core's solution and residual summary (the rows, their Jacobian at the estimate, √ρ′-scaled under a
# robust strategy, the weights and the parsed loss: `block_result`). The statement is concrete, so the refine is
# inferred.
residual_summary(::RobotWorldHandEyeProblem) = RowsWithJacobian()
block_structure(f::RobotWorldHandEyeProblem{<:Any, <:_PlainRigMarkers}, guess) =
    (st = _robot_world_hand_eye_run(f, guess); (st.vars0, st.blocks, st))
block_result(::RobotWorldHandEyeProblem{<:Any, <:_PlainRigMarkers}, _guess, r::FitResult, s) = _robot_world_hand_eye_solved(r, s)
_robot_world_hand_eye_solved(r::FitResult, s) = _assembled(r, s, Val((:residuals, :weights, :jacobian)))

function _robot_world_hand_eye_run(f::RobotWorldHandEyeProblem{<:Any, <:_PlainRigMarkers{K}},
                                   guess::RobotWorldHandEyeCalibration{K, <:PlainEuclideanMap{3}}) where {K}
    markers = f.markers
    ids = block_order(markers, guess.X)             # the key sets must match: every marker of the guess is refined
    M = length(ids)
    # Flattened marker by marker in `ids` order: each element's rig pose once
    # (RA/tA, indexed by pt_frame), each of its features one point.
    RA = SMatrix{3, 3, Float64, 9}[]; tA = SVector{3, Float64}[]
    pt_frame = Int[]; pt_marker = Int[]
    Xs = SVector{3, Float64}[]; us = SVector{2, Float64}[]
    counts = _flatten_rig_observations(markers, ids, "features") do m, id, j, RAj, tAj, features
        push!(RA, RAj); push!(tA, tAj)
        fr = length(RA)
        for (q, pc) in enumerate(features)
            push!(pt_frame, fr); push!(pt_marker, m)
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
    return _robot_world_hand_eye_bundle(guess, counts, f.camera, RA, tA, pt_frame, pt_marker, Xs, us)
end

# Physical inputs: the guess, the rig poses and the points stripped in metres
# once, the plain refinement, the calibration read back in metres once.
function _plain_problem(f::RobotWorldHandEyeProblem{<:Any, <:_PhysicalRigMarkers{K}}) where {K}
    features(fs) = _map_correspondences(_plain_metres, identity, fs)
    plain = Dict(id => _map_correspondences(_plain_metres, features, els) for (id, els) in f.markers)::_PlainRigMarkers{K}
    return RobotWorldHandEyeProblem(f.camera, plain)
end
block_structure(f::RobotWorldHandEyeProblem{<:Any, <:_PhysicalRigMarkers}, guess) = block_structure(_plain_problem(f), _plain_metres(guess))
block_result(::RobotWorldHandEyeProblem{<:Any, <:_PhysicalRigMarkers}, _guess, r::FitResult, s) =
    _attach_metres(_robot_world_hand_eye_solved(r, s))

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

# A robust refine robustifies the block's squared norm by `robust` (Triggs: Core's `loss_robustifier`; `NoRobust()`
# otherwise), and its stationarity and covariance rows scale the block by `sqrt_weight` (√ρ′ at the estimate; 1
# otherwise), as the camera-and-poses bundle does.
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
with_robustifier(r::_RwPointResidual{C}, sqrt_weight::Float64, robust::R) where {C, R} =
    _RwPointResidual{C, R}(r.varind, r.RA, r.tA, r.X, r.u, r.camera, sqrt_weight, robust)
NLLSsolver.robustkernel(r::_RwPointResidual) = r.robust
NLLSsolver.ndeps(::_RwPointResidual) = NLLSsolver.static(2)
NLLSsolver.nres(::_RwPointResidual) = NLLSsolver.static(2)
NLLSsolver.varindices(r::_RwPointResidual) = r.varind
NLLSsolver.getvars(r::_RwPointResidual, vars::Vector) = (vars[r.varind[1]]::RigidExpManifold6{Float64}, vars[r.varind[2]]::RigidExpManifold6{Float64})
Base.eltype(::_RwPointResidual) = Float64

# The point's residual at the charts' origins, and its 2 × 12 Jacobian in
# (δY, δX_m) — ForwardDiff through the retraction of the two blocks.
_rw_residual(r::_RwPointResidual, chY, chX) =
    r.sqrt_weight * _rw_point_residual(zero(SVector{12, Float64}), chY, chX, r.RA, r.tA, r.X, r.u, r.camera)
_rw_jacobian(r::_RwPointResidual, chY, chX) = r.sqrt_weight * ForwardDiff.jacobian(
    z -> _rw_point_residual(z, chY, chX, r.RA, r.tA, r.X, r.u, r.camera), zero(SVector{12, Float64})
)
NLLSsolver.computeresidual(r::_RwPointResidual, Y::RigidExpManifold6{Float64}, X::RigidExpManifold6{Float64}) = _rw_residual(r, Y, X)
# The block's cost as the LM reads it: non-finite is +Inf (`finite_cost`).
NLLSsolver.computecost(vars::Vector{RigidExpManifold6{Float64}}, r::_RwPointResidual) =
    finite_cost(NLLSsolver.computecost(r, NLLSsolver.getvars(r, vars)...))
# Every pose is free in this refinement (`optimize!` over all variables), so
# NLLSsolver asks for both blocks' columns: `varflags` = 0b11.
function NLLSsolver.computeresjac(varflags, r::_RwPointResidual, Y::RigidExpManifold6{Float64}, X::RigidExpManifold6{Float64})
    varflags == 3 || throw(ArgumentError("RobotWorldHandEyeProblem: every pose is free (varflags = $varflags)"))
    return _rw_residual(r, Y, X), _rw_jacobian(r, Y, X)
end

function _robot_world_hand_eye_bundle(
        guess::RobotWorldHandEyeCalibration, counts, camera::C, RA, tA, fr, mk, Xs, us
    ) where {C}
    n = length(us)
    costs = [_RwPointResidual{C}(SVector(1, mk[k] + 1), RA[fr[k]], tA[fr[k]], Xs[k], us[k], camera) for k in 1:n]
    poses0 = RigidExpManifold6{Float64}[RigidExpManifold6(guess.Y); [RigidExpManifold6(guess.X[id]) for id in guess.ids]]
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

    # THE bundle LM (Core's multi-block LM) solves it at the solve's stops (default the camera-and-poses bundle's:
    # configs/default.toml [camera_poses_problem]) and the algorithm's damping: least squares
    # (`optimization_strategy = nothing`), or the refine contract's one fixed-scale robust path from the guess with no
    # least-squares stage, Triggs, ONE robustified LM at the caller's frozen scale or width (an Irls is refused by
    # `bundle_loss`). Its termination record is the solve's (NLLSsolver's bits under Core's success rule).
    return _BundleStatement(costs, poses0, (sol, parsed) -> _robot_world_hand_eye_assembly(guess, counts, fr, n, sol, parsed))
end
# The calibration's result from the bundle's solve `sol` at the parsed loss: the poses read back, refused when not
# finite, the covariance from the rows' Jacobian, the record and the scale record.
function _robot_world_hand_eye_assembly(guess::RobotWorldHandEyeCalibration, counts, fr, n, sol, parsed)
    weights = sol.weights
    charts = collect(sol.variables)   # each anchored at its estimate
    poses = map(model, charts)
    all(_finite_pose, poses) ||   # camera_poses.jl
        throw(ErrorException("RobotWorldHandEyeProblem: the LM step is not finite"))

    # the reprojection rows' dense 2n × 6(M + 1) Jacobian at the charts' origins — the covariance's (point k's rows
    # 2k − 1, 2k; its columns those of Y and of its X_m), √ρ′-scaled under a robust strategy: the eigen-diagnostics and
    # the covariance are the weighted problem's
    J = sol.jacobian   # the rows' Jacobian at the estimate, √ρ′-scaled under a robust strategy (Core's `RowsWithJacobian`)
    # The covariance's blocks are the ELEMENTS — one rig pose's features, whose
    # points are contiguous (flattened element by element) — each a dense view
    # of its rows of J with observation Jacobian −I: the same (JᵀJ)⁻¹ as per-
    # point blocks, without a static 2 × N block and an N × N static product
    # per point. Measured 2026-10-01 at M = 20 (N = 126, 6000 points): the
    # per-point form took 370 ms and 2.2 GB per call, 4× the LM itself.
    ends = [k for k in 1:n if k == n || fr[k + 1] != fr[k]]
    starts = [1; ends[1:(end - 1)] .+ 1]
    rows(e) = (2starts[e] - 1):(2ends[e])
    function covariance(s2)
        Σ̃ = implicit_covariance(e -> (view(J, rows(e), :), Diagonal(fill(-1.0, length(rows(e))))), eachindex(starts))
        return Σ̃ === nothing ? nothing : s2 .* Σ̃
    end
    r_raw = sol.residuals   # the reprojection rows themselves
    return _robot_world_hand_eye_result(
        _robot_world_hand_eye_calibration(poses, guess.ids), r_raw, J, 2, counts, covariance, nothing, nothing;
        retcode = sol.code, stop_rule = sol.rule, iterations = sol.iterations, rms_rows = n, strategy = parsed.strategy, weights,
        VgtkCore.scale_record(parsed)...
    )
end
