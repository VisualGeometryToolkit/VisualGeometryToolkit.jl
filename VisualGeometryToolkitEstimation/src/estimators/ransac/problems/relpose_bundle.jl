# =============================================================================
# RelposeBundleProblem — two-view relative pose by robust FULL-3D bundle
# adjustment (exact reprojection MLE), structure profiled via the Schur
# complement.  Exact counterpart of `EssentialProblem`'s Sampson refit: Sampson
# is the first-order, structure-marginalized approximation of the reprojection
# error this problem optimizes exactly.
#
# Reprojection is done THROUGH THE CAMERA INTRINSICS (K₁, K₂) in the observation
# frame — `π(Kᵢ·X)` — so the residual lives where the noise lives
# (the same frame the scoring's outlier half-width and σ̂² are calibrated in, and the same frame the
# MSAC LO's PoseLib relative-pose refine uses with Kc as cameras).  All geometry is carried as
# `Point2`/`Point3` (positions) and `Vec3` (the translation); `SVector`/`SMatrix`
# are reserved for the LM parameter/residual vectors and Jacobian blocks.
#
# Model & constraint are unchanged from `EssentialProblem`: epipolar relative
# pose, codimension 1.  Only the residual VALUE (exact vs first-order) and the
# refit differ — so ν, σ̂²=RSS/ν and the IG marginal scoring are unchanged.
#
# TWO DISTINCT relpose refit PATHS — not redundant (they solve different
# objectives and serve different consumers):
#   • `_refine_relative_pose` (THIS file, the refine of a `RelativePoseProblem`; NLLSsolver, the stack's LM, with
#     the structure eliminated through the Schur complement): the RelposeBundleProblem's
#     own refit — the EXACT full-3D two-view reprojection MLE (per-point 3D
#     landmarks jointly optimized then profiled out via Schur). Live consumer:
#     the `bundle=true` Schur-reprojection refit path (eval build_pair_ctx →
#     `bp = RelposeBundleProblem`, the +Lift/bundle option in refit_imc).
#   • PoseLib's C++ relative-pose refinement (the eval ExternalRefit arms,
#     LocalOptimizationDoneRight src/engines/refit.jl): POSE-ONLY refinement of the TRUNCATED Sampson
#     loss — the LO-RANSAC standard (+PoseLib(TLS)/(LZ)). First-order,
#     structure-free, hard-thresholded; a comparison baseline, not a bundle.
# Both use Kc as cameras in the same observation frame; the objective (exact
# reprojection bundle vs truncated Sampson) and the variable set (structure vs
# pose-only) are what differ.
#
# Bundle structure (camera 1 fixed at identity, camera 2 = (R, t), ‖t‖ = 1):
#   params x = [ ω(3) ⊕ τ(2) | X₁(3) … X_N(3) ]
#   the pose block is VGC's `EssentialManifold5` anchored at the seed (R₀, t₀),
#   ‖t₀‖ = 1: R(x) = RotMatrix(MRP(ω))·R₀, t(x) = normalize(t₀ + τ₁b₁ + τ₂b₂),
#   (b₁, b₂) an orthonormal basis of T_{t₀}S² frozen at the anchor — so(3) ⊕
#   T_tS², 5 dof, MINIMAL: two views fix a relative pose only up to scale, and
#   the chart carries no scale coordinate, so ‖t‖ = 1 by construction and the
#   arrowhead system has no null space. The structure blocks are free ℝ³.
#   residuals (4 per point): reproject Xᵢ into BOTH cameras
#       image 1: π(K₁ Xᵢ) − u₁ᵢ          (depends on Xᵢ only — camera 1 fixed)
#       image 2: π(K₂(R Xᵢ + t)) − u₂ᵢ   (depends on Xᵢ and pose)
#   Hessian is an ARROWHEAD: dense 5×5 pose border, block-diagonal 3×3 structure
#   spokes, pose↔structure coupling only through image 2.
#
# Schur: H̃ = H_pp − Σᵢ H_pXᵢ H_XᵢXᵢ⁻¹ H_Xᵢp (5×5, full rank); solve δpose,
# back-substitute δXᵢ. The bundle is an NLLSsolver problem — the pose and the
# N points its variables, one 4-row residual block per point with its
# closed-form 4 × 8 Jacobian — and NLLSsolver owns the LM (Marquardt's DᵀD
# damping, step accept/reject, convergence: Core's one LM, LevenbergMarquardt on the block contract) and, where the
# loaded NLLSsolver has it (upstream #60, still open: the stack's fork has it;
# Core's `has_nlls_schur()`), the Schur complement: its `MultiVariateLSschur`
# eliminates every point (`iseliminable`). Without it the same problem runs on
# NLLSsolver's sparse LDL over all 5 + 3N unknowns — the same optimum, slower.
#
# NLLSsolver re-anchors the pose variable at every accepted step (VGC's
# manifold `update` is `recenter`). The pose block is the minimal chart
# `EssentialManifold5`; a `RigidManifold6` pose (MRP ⊕ free ℝ³ t, one gauge
# direction held by the damping) reaches the same optimum (cost ≤ 5.5e-15
# relative, pose ≤ 7.4e-12°, measured 2026-10-27).
#
# Robustness: per-point weight wᵢ (the outer IRLS reweight's — the Student-t
# posterior weight under LocalOptimizationDoneRight's `StudentTUniform`) scales all four of point i's residual rows.

# -----------------------------------------------------------------------------
# The bundle as an NLLSsolver problem: the pose and the N points are its
# variables, one 4-row residual block per point.
# -----------------------------------------------------------------------------
# The pose variable: VGC's `EssentialManifold5`, an NLLSsolver variable through
# VGC's manifold protocol (`nvars` = 5, `update` = `recenter`), concrete in
# Float64 — a `Vector{Union{…}}` of the UnionAll alias would box every element.
# The pose block's dimension: `EssentialManifold5`'s tangent size.
const _POSE_DOF = 5

# A structure point, free in ℝ³: VGC's `Eliminable` around the point. `E` says
# whether NLLSsolver may eliminate it (the Schur-complement linear system,
# upstream #60, open; detected once, Core's `has_nlls_schur()`); `E = false` is
# NLLSsolver's sparse LDL over every unknown — which an NLLSsolver without #60
# takes and the fallback test forces.
const _BundlePoint{E} = Eliminable{E, SVector{3, Float64}}

# The data of the current call — the calibrations, the two views and the
# √weights — shared by every residual block. The run OWNS the three vectors
# (sized N, overwritten per call): it never keeps a reference to a caller's
# arrays after the call.
mutable struct _BundleData{T1, T2, TS}
    K1::SMatrix{3, 3, Float64, 9}
    K2::SMatrix{3, 3, Float64, 9}
    u1::T1
    u2::T2
    sqrtw::TS
end
_BundleData{Vector{SVector{2, Float64}}, Vector{SVector{2, Float64}}, Vector{Float64}}(N::Int) = _BundleData{Vector{SVector{2, Float64}}, Vector{SVector{2, Float64}}, Vector{Float64}}(
    zero(SMatrix{3, 3, Float64, 9}), zero(SMatrix{3, 3, Float64, 9}),
    Vector{SVector{2, Float64}}(undef, N), Vector{SVector{2, Float64}}(undef, N), Vector{Float64}(undef, N)
)

# Point i's four rows, √wᵢ times its two-view reprojection error
# [π(K₁X) − u₁; π(K₂(RX + t)) − u₂] — camera 1 fixed, so image 1 depends on
# the point alone.
struct _BundleRows{E, D} <: NLLSsolver.AbstractResidual
    data::D
    i::Int
    varind::SVector{2, Int}
end
NLLSsolver.ndeps(::_BundleRows) = NLLSsolver.static(2)
NLLSsolver.nres(::_BundleRows) = NLLSsolver.static(4)
NLLSsolver.varindices(r::_BundleRows) = r.varind
NLLSsolver.getvars(r::_BundleRows{E}, vars::Vector) where {E} =
    (vars[r.varind[1]]::VgtkCore.EssentialManifold{VgtkCore.S2TangentT{Float64}, Float64}, vars[r.varind[2]]::_BundlePoint{E})
Base.eltype(::_BundleRows) = Float64
function NLLSsolver.computeresidual(r::_BundleRows, pose::VgtkCore.EssentialManifold{VgtkCore.S2TangentT{Float64}, Float64}, p::_BundlePoint)
    d = r.data; i = r.i
    R = SMatrix{3, 3, Float64, 9}(pose.R); t = SVector{3, Float64}(pose.t)
    return d.sqrtw[i] * _reprojection_error(d.K1, d.K2, R, t, p.value, d.u1[i], d.u2[i])
end
# The block's cost as the LM reads it: non-finite is +Inf (`finite_cost`).
NLLSsolver.computecost(vars::Vector{Union{VgtkCore.EssentialManifold{VgtkCore.S2TangentT{Float64}, Float64}, _BundlePoint{E}}}, r::_BundleRows{E}) where {E} =
    finite_cost(NLLSsolver.computecost(r, NLLSsolver.getvars(r, vars)...))
# The closed-form 4 × 8 Jacobian, pose columns (VGC's `pose_action_jacobian`
# at the variable's own anchor, the derivative of its `update`) then point
# columns. Every variable is free in this problem (`optimize!` over all), so
# NLLSsolver asks for both blocks' columns: `varflags` = 0b11.
function NLLSsolver.computeresjac(_varflags, r::_BundleRows, pose::VgtkCore.EssentialManifold{VgtkCore.S2TangentT{Float64}, Float64}, p::_BundlePoint)
    d = r.data; i = r.i; w = d.sqrtw[i]
    R = SMatrix{3, 3, Float64, 9}(pose.R); t = SVector{3, Float64}(pose.t); X = p.value
    e = _reprojection_error(d.K1, d.K2, R, t, X, d.u1[i], d.u2[i])
    A1 = _Dproj(d.K1 * X) * d.K1                        # ∂(img1)/∂X, 2×3
    D2K2 = _Dproj(d.K2 * (R * X + t)) * d.K2            # 2×3
    A2 = D2K2 * R                                       # ∂(img2)/∂X
    P = D2K2 * pose_action_jacobian(pose, X)            # ∂(img2)/∂(ω, τ): VGC's closed form
    J = vcat(hcat(zero(SMatrix{2, _POSE_DOF, Float64}), A1), hcat(P, A2))
    return w * e, w * J
end

# The kept problem: ONE per task (and point type `E`), for the point count of
# the task's last bundle — reused by consecutive calls of the same size on that
# task and rebuilt, the old one released, when the size changes, so a sweep
# over pairs of many sizes keeps one problem per task, not one per size (a
# sweep that interleaves sizes rebuilds it on every call, and pays the
# uncached time below). Its
# residual blocks read the call's data through `data` (owned buffers), its
# variables are reset per call, and NLLSsolver's internal data (the Schur
# system's layout) is kept in Core's block-LM cache (its typed cache), built once
# for this structure. Without it a call would rebuild them
# (measured 2026-10-27: 2.0 / 1.4 / 1.4× the cached call's time at N = 100 /
# 1000 / 10000).
#
# NOT REENTRANT: a bundle started from inside the NLLSsolver `callback` of a
# bundle on the same task overwrites the running problem's data and
# variables (the internal `callback` keyword; no production caller supplies one).
struct _BundleRun{E, D, C}
    data::D
    # Core's block refine of the run's bundle (`_RelposeRun`: the pose and the N points, one `_BundleRows` block per
    # point), built once by `init` with NLLSsolver's typed cache for its structure (Schur, sparse or dense, a run-time
    # choice); `reinit!` re-seeds its variables.
    cache::C
    n::Int
end
function _BundleRun{E}(N::Int, data::D) where {E, D}
    seed = (SMatrix{3, 3, Float64, 9}(I), Vec3{Float64}(0.0, 0.0, 1.0), zero(SMatrix{3, 3, Float64, 9}), zero(SMatrix{3, 3, Float64, 9}))
    c = init(RefineProblem(_RelposeRun{E, D}(data), seed), LevenbergMarquardt(; optimization_strategy = nothing))
    return _BundleRun{E, D, typeof(c)}(data, c, N)
end
# The run's bundle as a block problem: the call's data (shared with its blocks) and, as its guess, the seed pose and the
# two inverse calibrations the points are triangulated through.
struct _RelposeRun{E, D} <: AbstractBlockProblem
    data::D
end
default_stops(::_RelposeRun) = _RELPOSE_BUNDLE_STOPS
default_optimization_strategy(::_RelposeRun) = nothing
residual_summary(::_RelposeRun) = nothing
block_result(::_RelposeRun, _u0, r::FitResult, _summary) = r
function block_variables(f::_RelposeRun{E}, (Rb, tb, iK1, iK2)) where {E}
    d = f.data
    N = length(d.u1)
    vars = Vector{Union{VgtkCore.EssentialManifold{VgtkCore.S2TangentT{Float64}, Float64}, _BundlePoint{E}}}(undef, N + 1)
    vars[1] = EssentialManifold5(Rb, tb)::VgtkCore.EssentialManifold{VgtkCore.S2TangentT{Float64}, Float64}
    @inbounds for i in 1:N
        X = _triangulate_rays(iK1, iK2, d.u1[i], d.u2[i], Rb, tb)
        vars[i + 1] = _BundlePoint{E}(SVector{3, Float64}(X[1], X[2], X[3]))
    end
    return vars
end
residual_blocks(f::_RelposeRun{E, D}, _u0) where {E, D} = [_BundleRows{E, D}(f.data, i, SVector(1, i + 1)) for i in eachindex(f.data.u1)]
# The task's slot for its one run of point type `E`.
mutable struct _BundleSlot{E}
    run::Union{Nothing, _BundleRun{E, _BundleData{Vector{SVector{2, Float64}}, Vector{SVector{2, Float64}}, Vector{Float64}}}}
end
_bundle_size(run::_BundleRun) = run.n
function _bundle_run(::Val{E}, N::Int) where {E}
    slot = task_buffers(_BundleSlot{E}, () -> _BundleSlot{E}(nothing), ())
    run = slot.run
    run !== nothing && _bundle_size(run) == N && return run
    slot.run = nothing                        # the old size's problem is released before the new is built
    new = _BundleRun{E}(N, _BundleData{Vector{SVector{2, Float64}}, Vector{SVector{2, Float64}}, Vector{Float64}}(N))
    slot.run = new
    return new
end

# -----------------------------------------------------------------------------
# Geometry helpers — projection THROUGH the intrinsics, DLT/optimal
# triangulation, cheirality.  Camera 1 = identity pose; intrinsics K₁, K₂.
# -----------------------------------------------------------------------------
# The two-view reprojection error of a point X (camera-1 frame) under the pose
# (R, t): the 4-vector [π(K₁X) − u₁; π(K₂(RX + t)) − u₂]. ONE definition for the
# scoring residual (its norm), the optimal triangulation (its GN residual) and
# the bundle rows (√w times it).
@inline function _reprojection_error(K1, K2, R, t, X, u1, u2)
    e1 = PerspectiveMap()(K1 * X); e2 = PerspectiveMap()(K2 * (R * X + t))   # π(K·Y)
    return SVector(e1[1] - u1[1], e1[2] - u1[2], e2[1] - u2[1], e2[2] - u2[2])
end
@inline _Dproj(W) = transform_deriv(PerspectiveMap(), W)   # ∂π/∂W — VGC's one definition

# Back-projection to a unit bearing through a PRECOMPUTED K⁻¹. Every caller
# hoists the inverse: these loops run per point, and `_cheiral_relpose`'s runs
# per point PER MOTION, so inverting inside would put a 3×3 inversion in the
# hottest path in this file.
@inline function _pixel_bearing(iK::SMatrix{3, 3, Float64, 9}, u)
    return normalize(iK * SVector{3, Float64}(Float64(u[1]), Float64(u[2]), one(Float64)))
end

# Ray-domain triangulation, replacing the pixel-domain DLT.
#
# WHY: the DLT builds `P = K[I|0]` — pinhole BY CONSTRUCTION — and that matrix
# does not exist for the fisheye and omnidirectional cameras this wave is for.
# VGC's `triangulate_bearings` takes bearings and needs no such matrix.
# It returns NaN depths for parallel rays rather than a plausible point.
#
# The numbers are a MIXED trade and the tail is the reason, not the median.
# Measured over 4000 samples per parallax bin: the ray form wins in the tail and
# LOSES MORE OFTEN THAN IT WINS below 0.3° (42.2% and 39.6% paired wins), while
# q99 improves 39.2 → 6.05 and 76.6 → 9.04. Above 3° the two are identical to
# three significant figures. Reporting the median alone would read this as a
# regression; reporting the tail alone would hide a real cost.
#
# NOTE THE LIMIT, which is structural and not removed by this change: this
# problem holds K MATRICES, not camera models, so the back-projection above is
# still `K⁻¹`. Ray-form triangulation inside a pinhole-by-data-structure problem
# confers no camera-generality — carrying camera models is what would, and that
# is a separate decision.
#
# PARALLEL RAYS: VGC's `triangulate_bearings` returns NaN there (nothing was
# triangulated, the honest answer for cheirality). A bundle cannot start from
# NaN — one far point, or one baseline-parallel outlier, made the whole refit
# and every residual NaN, so the bundle returned no pose even
# with that point at weight 0. The pixel DLT this replaced was finite there.
# So a parallel pair is placed FAR along the first ray (`_PARALLEL_DEPTH`
# baselines, ‖t‖ = 1): reprojection error ~ f/_PARALLEL_DEPTH px, finite, and a
# point at infinity is what parallel rays mean.
#
# `check = false`: R is the CALLER's (a seed pose, the public entry's `pose`)
# and triangulation needs no orthonormality certificate; re-checking it threw
# an uncaught `ArgumentError` for a drifted pose before the solver's `try`.
const _PARALLEL_DEPTH = 1.0e6
@inline function _triangulate_rays(iK1, iK2, u1, u2, R, t)
    f₁ = _pixel_bearing(iK1, u1)
    X, _, _ = triangulate_bearings(
        EuclideanMap(SMatrix{3, 3, Float64, 9}(R), SVector{3, Float64}(t); check = false),
        f₁, _pixel_bearing(iK2, u2)
    )
    all(isfinite, X) || (X = _PARALLEL_DEPTH * f₁)
    return Point3{Float64}(X[1], X[2], X[3])
end

# Optimal 2-view triangulation through K: minimize ‖π(K₁X)−u₁‖²+‖π(K₂(RX+t))−u₂‖²
# over X (ray-form init + Gauss-Newton).  The per-pose structure optimum the BA
# converges to — so `mahalanobis_distances!` here matches the BA's own per-point residual,
# keeping the IRLS reweight CONSISTENT with the refit.
function _triangulate_optimal(u1, u2, K1, K2, iK1, iK2, R, t; iters::Int = 6)
    X = _triangulate_rays(iK1, iK2, u1, u2, R, t)
    @inbounds for _ in 1:iters
        (X[3] <= 0) && break
        W1 = K1 * X; Y = R * X + t; (Y[3] <= 0) && break; W2 = K2 * Y
        Jm = vcat(_Dproj(W1) * K1, (_Dproj(W2) * K2) * R)               # 4×3
        e = _reprojection_error(K1, K2, R, t, X, u1, u2)
        δ = (Jm' * Jm) \ (Jm' * e)
        all(isfinite, δ) || break                  # a far (parallel-ray) point's depth is unobservable
        X = Point3(X[1] - δ[1], X[2] - δ[2], X[3] - δ[3])
        (δ[1]^2 + δ[2]^2 + δ[3]^2) < 1.0e-20 && break
    end
    return X
end

# Cheirality: of motion(E)'s four (R, t), the one with the most correspondences
# in front of both cameras — VGC's `cheiral_pose_fast` vote (ray-domain
# `test_cheirality`; a z-test assumes a forward hemisphere the ray domain does
# not have). It takes BEARINGS: the caller back-projects with its own camera.

# The bundle problem's cheirality: its K-frame points back-projected ONCE, at
# construction (two bearings per point, not two per point per motion, and none
# per model: `RelposeBundleProblem`'s `f1`/`f2`), then THE vote.
function _cheiral_relpose(p, E)
    R, t = cheiral_pose_fast(E, p.f1, p.f2)
    return R, Vec3{Float64}(t)
end

"""
    RelativePoseProblem(K1, K2, cs)

    solve(RefineProblem(RelativePoseProblem(K1, K2, cs), pose::PlainEuclideanMap{3}), LevenbergMarquardt(); stops)
        -> FitResult{Union{EuclideanMap{3}, Nothing}}   # stops default: configs/default.toml [relpose_bundle]

The full-3D two-view bundle adjustment of a relative pose, a multi-block problem (VisualGeometryToolkitCore's
`AbstractBlockProblem`): the exact reprojection MLE over the pose and one 3D landmark per correspondence (profiled out
through the Schur complement), seeded at `pose`, `u` the refined pose with unit translation.

`cs` are the `u₁ => u₂` correspondences in the frame of the intrinsics `K1`,
`K2` (reprojection is `π(Kᵢ·X)`), any `sources`/`targets` container; per-point
weights are its `weights` column when it has one (e.g. `withattrs(cs; weights
= w)`, the IRLS posterior), unit otherwise. The bundle is the weighted least squares: its
`optimization_strategy` is `nothing` (`ClassDefault()` resolves to it; a robust strategy is an `ArgumentError` —
its robustness is the weights column), and its damping Marquardt's (another is an `ArgumentError`: the kept run is
built for it).

The seed is a plain pose (a physical one is a `MethodError`): two views fix a
relative pose only up to scale, so its translation is a direction — read
unit-normalized, returned with unit norm — and no length enters or leaves. A
physical seed is stripped by the caller in any unit (`ustrip(m, pose)`).

The result: the LM's termination record and accepted steps, `n` the correspondences and `rms` the weighted
reprojection rms over them (`√(cost/n)`), no metadata. A bundle that cannot run has no pose: `u = nothing`,
`(InitialFailure, NoRule)` for fewer than 5 correspondences, `(Failure, NonFinite)` for a singular system or a
non-finite pose.
"""
struct RelativePoseProblem{K1 <: AbstractMatrix, K2 <: AbstractMatrix, C} <: AbstractBlockProblem
    K1::K1
    K2::K2
    cs::C
end
# each view's calibration in its own type (an `SMatrix` and a `CameraCalibrationMat` may meet)
RelativePoseProblem(K1::A, K2::B, cs::C) where {A <: AbstractMatrix, B <: AbstractMatrix, C <: AbstractVector{<:Pair}} =
    RelativePoseProblem{A, B, C}(K1, K2, cs)

const _RELPOSE_BUNDLE_STOPS = LmStops(; max_iter = _RELPOSE_BUNDLE_MAX_ITER, ftol = _RELPOSE_BUNDLE_FTOL)
default_stops(::RelativePoseProblem) = _RELPOSE_BUNDLE_STOPS
default_optimization_strategy(::RelativePoseProblem) = nothing

# The block contract (Core's multi-block LM): the pose (its chart `EssentialManifold5`, the seed's translation
# unit-normalized) and one free point per correspondence, triangulated in the ray domain at the seed; one 4-row block per
# correspondence, √w times its two-view reprojection error. A plain least squares: a robust strategy has no block
# robustifier. The result reads the LM's pose, its record and final cost (`rms = √(cost / n)`).
# Fewer than 5 correspondences (a relative pose has 5 dof): Core's block refine returns no pose, `(InitialFailure,
# NoRule)`, before any block is built.
data_size(f::RelativePoseProblem) = length(f.cs)
minimal_sample_size(::RelativePoseProblem) = 5
# The RANSAC refit's bundle on the problem's data (its blocks' shared `_BundleData`: the calibrations, the views, the
# √weights from the `weights` column, unit weight without one), its guess the seed pose (the translation
# unit-normalized) and the inverse calibrations.
# The data lives in the task's ONE buffer (sized to the call, grown only for more points), shared by the blocks Core's
# one-shot block refine keeps per task: the call's data is copied into it, so a kept problem's blocks read the task's
# own arrays, never a caller's, and a call allocates nothing for its data.
mutable struct _RelposeDataSlot
    data::_BundleData{Vector{SVector{2, Float64}}, Vector{SVector{2, Float64}}, Vector{Float64}}
end
function _relpose_data(N::Int)
    slot = task_buffers(_RelposeDataSlot, () -> _RelposeDataSlot(_BundleData{Vector{SVector{2, Float64}}, Vector{SVector{2, Float64}}, Vector{Float64}}(N)), ())
    d = slot.data
    length(d.u1) == N || (resize!(d.u1, N); resize!(d.u2, N); resize!(d.sqrtw, N))
    return d
end
function _relpose_run(f::RelativePoseProblem)
    N = length(f.cs)
    d = _relpose_data(N)
    d.K1 = SMatrix{3, 3, Float64, 9}(f.K1); d.K2 = SMatrix{3, 3, Float64, 9}(f.K2)
    u1 = sources(f.cs); u2 = targets(f.cs)
    for i in 1:N
        d.u1[i] = SVector{2, Float64}(u1[i][1], u1[i][2]); d.u2[i] = SVector{2, Float64}(u2[i][1], u2[i][2])
    end
    w = VgtkCore.weights_column(f.cs)
    w === nothing ? fill!(d.sqrtw, 1.0) : _relpose_sqrt_weights!(d.sqrtw, w, N)
    return _RelposeRun{has_nlls_schur(), typeof(d)}(d)
end
_relpose_guess(f::RelativePoseProblem, pose::PlainEuclideanMap{3}) =
    (SMatrix{3, 3, Float64, 9}(rotation(pose)), Vec3{Float64}(normalize(SVector{3, Float64}(translation(pose)))), _inv3(f.K1), _inv3(f.K2))
block_variables(f::RelativePoseProblem, pose::PlainEuclideanMap{3}) = block_variables(_relpose_run(f), _relpose_guess(f, pose))
residual_blocks(f::RelativePoseProblem, pose::PlainEuclideanMap{3}) = residual_blocks(_relpose_run(f), _relpose_guess(f, pose))
# A physical seed: the relative pose is up to scale and records no unit (units.md, point 3), so it is refused naming the fix.
block_variables(::RelativePoseProblem, ::_PhysicalPose3) = throw(ArgumentError(
    "RelativePoseProblem: the seed's translation is a direction (the pose is up to scale) and records no unit: give a plain seed, `ustrip(m, pose)`"))
_relpose_sqrt_weights!(s, w, N) = (VgtkCore.check_weights(view(w, 1:N)); s .= sqrt.(Float64.(view(w, 1:N))))
with_robustifier(::_BundleRows, _sqrt_weight, _robustifier) = throw(ArgumentError(
    "RelativePoseProblem: the two-view bundle is a weighted least squares (its weights the correspondences' `weights` " *
        "column): optimization_strategy = nothing"))
residual_summary(::RelativePoseProblem) = nothing
function block_result(f::RelativePoseProblem, _pose, r::FitResult, _summary)
    U = Union{typeof(EuclideanMap(one(RotMatrix{3, Float64}), zero(Vec3{Float64}); check = false)), Nothing}
    N = length(f.cs)
    r.retcode === ReturnCode.Failure && return FitResult{U}(nothing, ReturnCode.Failure, NonFinite, SolveStats(), NaN, N, nothing)
    pose = r.u[1]::VgtkCore.EssentialManifold{VgtkCore.S2TangentT{Float64}, Float64}
    R = SMatrix{3, 3, Float64, 9}(pose.R); t = Vec3{Float64}(pose.t)
    (all(isfinite, R) && all(isfinite, t)) ||
        return FitResult{U}(nothing, ReturnCode.Failure, NonFinite, SolveStats(), NaN, N, nothing)
    return FitResult{U}(EuclideanMap(RotMatrix{3, Float64}(R), t; check = false), r.retcode, r.stop_rule,
                        SolveStats(; nsteps = r.stats.nsteps), sqrt(r.metadata.cost / N), N, nothing)
end

# -----------------------------------------------------------------------------
# Core routine: robust full-3D two-view BA (camera-aware).  Seed pose (R₀,t₀) +
# intrinsics K₁,K₂ and their inverses + image points u₁,u₂ (in the K frame) +
# per-point √weights → `(; R, t, sol)`: the refined (R, t) (‖t‖=1) and Core's
# solve record `sol = (; code, rule, iterations, cost)` (Core's chart LM: its one
# success rule, a tolerance stop converged and the rejection cap judged by the stationarity postcondition; a `:stalled`
# bundle's pose is still scored by the caller, the LO's acceptance, as a converged
# one is). `nothing` on failure: a singular system (Core's `Failure`) or a
# non-finite pose. Positional and
# allocation-lean: the RANSAC refit (`refine!` below) calls it with the
# problem's stored K⁻¹ and the √weights it builds. `schur = false` takes
# NLLSsolver's sparse LDL where the Schur system (upstream #60) is loaded (the
# fallback test runs it); `callback` is NLLSsolver's per-iteration callback
# (the task's kept run is not reentrant from it: see `_BundleRun`).
# -----------------------------------------------------------------------------
function _refine_relative_pose(
        R₀::AbstractMatrix, t₀::AbstractVector,
        K1::AbstractMatrix, K2::AbstractMatrix,
        iK1::SMatrix{3, 3, Float64, 9}, iK2::SMatrix{3, 3, Float64, 9},
        u1::AbstractVector, u2::AbstractVector,
        sqrtw::AbstractVector; stops::LmStops = _RELPOSE_BUNDLE_STOPS,
        schur::Bool = has_nlls_schur(), callback::F = nothing
    ) where {F}
    # (`::F`: a Function-typed argument only handed on is not specialized,
    # which made the call below a dynamic one, boxing every argument)
    N = length(u1)
    N >= 5 || return nothing
    Rb = SMatrix{3, 3, Float64, 9}(R₀); tb = Vec3{Float64}(normalize(SVector{3, Float64}(t₀)))
    E = schur && has_nlls_schur()
    return E ? _bundle_optimize(Val(true), N, K1, K2, u1, u2, sqrtw, Rb, tb, iK1, iK2, stops, callback) :
        _bundle_optimize(Val(false), N, K1, K2, u1, u2, sqrtw, Rb, tb, iK1, iK2, stops, callback)
end

function _bundle_optimize(
        v::Val{E}, N, K1, K2, u1, u2, sqrtw, Rb, tb, iK1, iK2, stops, callback
    ) where {E}
    # a function barrier: the task's kept run is stored without its block-LM cache's concrete type (the slot holds any size's
    # run), so the per-point setup and the solve run in a method specialized on it — one dynamic call per bundle, not one
    # per point (without it a call allocated ~2N times)
    return _bundle_solve!(_bundle_run(v, N), N, K1, K2, u1, u2, sqrtw, Rb, tb, iK1, iK2, stops, callback)
end
function _bundle_solve!(
        run::_BundleRun{E}, N, K1, K2, u1, u2, sqrtw, Rb, tb, iK1, iK2, stops::LmStops, callback::F
    ) where {E, F}
    d = run.data
    d.K1 = SMatrix{3, 3, Float64, 9}(K1); d.K2 = SMatrix{3, 3, Float64, 9}(K2)
    for i in 1:N                              # the call's data copied into the run's own buffers (bounds-checked: the caller's lengths)
        d.u1[i] = SVector{2, Float64}(u1[i][1], u1[i][2]); d.u2[i] = SVector{2, Float64}(u2[i][1], u2[i][2])
        d.sqrtw[i] = sqrtw[i]
    end
    # Core's one LM at the bundle's stop settings (default [relpose_bundle] max_iter, ftol; the rest Core's [lm]), from
    # the seed pose and the points triangulated at it; a per-iteration `callback` runs on a cache of its own
    seed = (Rb, tb, iK1, iK2)
    sol = callback === nothing ? solve!(reinit!(run.cache, seed; stops)) :
        solve(RefineProblem(_RelposeRun{E, typeof(d)}(d), seed), LevenbergMarquardt(; optimization_strategy = nothing); stops, callback)
    sol.retcode === ReturnCode.Failure && return nothing                 # a singular system: no solution
    pose = sol.u[1]::VgtkCore.EssentialManifold{VgtkCore.S2TangentT{Float64}, Float64}
    R = SMatrix{3, 3, Float64, 9}(pose.R); t = Vec3{Float64}(pose.t)
    (all(isfinite, R) && all(isfinite, t)) || return nothing
    return (; R, t, sol)
end

# =============================================================================
# Problem type
# =============================================================================
# The 3×3 inverse of an intrinsics matrix, in Float64 — taken ONCE per problem
# (at construction) or per public call, never per residual evaluation.
@inline _inv3(K::AbstractMatrix) = SMatrix{3, 3, Float64, 9}(inv(SMatrix{3, 3, Float64, 9}(K)))

"""
    RelposeBundleProblem(K1, K2, cs)

Two-view relative pose scored by exact reprojection (full-3D bundle), refit by
robust Schur-reduced BA.  `K1`,`K2` are the intrinsics of the observation frame
(`π(Kᵢ·X)`), given models first; `cs` the `u₁ => u₂` correspondences in that
frame (canonical image points), any `sources`/`targets` container.
Same model/constraint as `EssentialProblem` (epipolar, codimension 1).
"""
struct RelposeBundleProblem{C, K <: AbstractMatrix} <: AbstractRansacProblem{5}
    K1::K
    K2::K
    iK1::SMatrix{3, 3, Float64, 9}   # K⁻¹, inverted once at construction
    iK2::SMatrix{3, 3, Float64, 9}
    cs::C
    f1::Vector{SVector{3, Float64}}  # the points' unit bearings through K⁻¹, back-projected once at
    f2::Vector{SVector{3, Float64}}  # construction: the per-model cheirality vote reads them (allocation-free)
end
function RelposeBundleProblem(K1::K, K2::K, cs::C) where {C, K <: AbstractMatrix}
    iK1, iK2 = _inv3(K1), _inv3(K2)
    return RelposeBundleProblem{C, K}(K1, K2, iK1, iK2, cs,
        [_pixel_bearing(iK1, u) for u in sources(cs)], [_pixel_bearing(iK2, u) for u in targets(cs)])
end

data_size(p::RelposeBundleProblem) = length(p.cs)
codimension(::Type{<:RelposeBundleProblem}) = 1
model_type(::Type{<:RelposeBundleProblem}) = EssentialMat{Float64}

# Scoring residual: exact reprojection of the OPTIMALLY-triangulated point — the
# exact codim-1 geometric error Sampson approximates to first order, and the same
# per-point objective the BA refit minimizes (so the IRLS reweight is consistent
# with the refit — same residual feeds `update_weights!` and the BA cost). The
# per-model work — the cheiral pose of θ — is RSDR's `residual_context`, taken
# once per model; `mahalanobis_distances!` is RSDR's loop over it.
struct _RelposeResidualContext
    R::SMatrix{3, 3, Float64, 9}
    t::Vec3{Float64}
end
function residual_context(p::RelposeBundleProblem, θ::EssentialMat)
    R, t = _cheiral_relpose(p, θ)
    return _RelposeResidualContext(SMatrix{3, 3, Float64, 9}(R), t)
end
@inline function mahalanobis_distance(p::RelposeBundleProblem, ctx::_RelposeResidualContext, i::Int)
    u1 = @inbounds sources(p.cs)[i]; u2 = @inbounds targets(p.cs)[i]
    X = _triangulate_optimal(u1, u2, p.K1, p.K2, p.iK1, p.iK2, ctx.R, ctx.t)
    return norm(_reprojection_error(p.K1, p.K2, ctx.R, ctx.t, X, u1, u2))
end
# The signed residual of the codim-1 constraint: the distance above with the
# side of the epipolar variety the pair lies on, the sign of the epipolar
# constraint `ũ₂ᵀ K₂⁻ᵀ [t]ₓ R K₁⁻¹ ũ₁` at the context's pose (the sign Sampson's
# signed residual carries) — so `abs` of it is the distance, bit for bit.
@inline function whitened_residual(p::RelposeBundleProblem, ctx::_RelposeResidualContext, i::Int)
    u1 = @inbounds sources(p.cs)[i]; u2 = @inbounds targets(p.cs)[i]
    x1 = p.iK1 * SVector(u1[1], u1[2], 1.0); x2 = p.iK2 * SVector(u2[1], u2[2], 1.0)
    g = dot(x2, cross(ctx.t, ctx.R * x1))
    return copysign(mahalanobis_distance(p, ctx, i), g)
end

# Robust BA refit — the problem's own LO step (a Schur-reduced full-3D bundle,
# not a chart LM: it has no `chart_dof`). `weights` are the per-point Student-t
# posterior weights from the IRLS reweight; the seed pose is the current
# essential's cheiral decomposition.
function refine!(_ws, p::RelposeBundleProblem, θ::EssentialMat, weights::AbstractVector)
    _underdetermined(p, weights) && return nothing
    u1 = sources(p.cs); u2 = targets(p.cs)
    R₀, t₀ = _cheiral_relpose(p, θ)
    sqrtw = _relpose_sqrt_weights!(Vector{Float64}(undef, length(weights)), weights, length(weights))
    res = _refine_relative_pose(R₀, t₀, p.K1, p.K2, p.iK1, p.iK2, u1, u2, sqrtw)
    res === nothing && return nothing
    return EssentialMat(res.R, res.t)
end
