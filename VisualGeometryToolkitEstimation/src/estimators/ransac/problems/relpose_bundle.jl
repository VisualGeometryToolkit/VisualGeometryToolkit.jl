# =============================================================================
# RelposeBundleProblem — two-view relative pose by robust FULL-3D bundle
# adjustment (exact reprojection MLE), structure profiled via the Schur
# complement.  Exact counterpart of `EssentialProblem`'s Sampson refit: Sampson
# is the first-order, structure-marginalized approximation of the reprojection
# error this problem optimizes exactly.
#
# Reprojection is done THROUGH THE CAMERA INTRINSICS (K₁, K₂) in the measurement
# frame — `π(Kᵢ·X)` — so the residual lives where the noise lives
# (the same frame the scoring's outlier half-width and σ̂² are calibrated in, and the same frame the
# MSAC LO `refine_relpose` uses with Kc as cameras).  All geometry is carried as
# `Point2`/`Point3` (positions) and `Vec3` (the translation); `SVector`/`SMatrix`
# are reserved for the LM parameter/residual vectors and Jacobian blocks.
#
# Model & constraint are unchanged from `EssentialProblem`: epipolar relative
# pose, codimension 1.  Only the residual VALUE (exact vs first-order) and the
# refit differ — so ν, σ̂²=RSS/ν and the IG marginal scoring are unchanged.
#
# TWO DISTINCT relpose refit PATHS — not redundant (they solve different
# objectives and serve different consumers):
#   • `bundle_refine_relpose` (THIS file; NLLSsolver, the stack's LM, with
#     the structure eliminated through the Schur complement): the RelposeBundleProblem's
#     own refit — the EXACT full-3D two-view reprojection MLE (per-point 3D
#     landmarks jointly optimized then profiled out via Schur). Live consumer:
#     the `bundle=true` Schur-reprojection refit path (eval build_pair_ctx →
#     `bp = RelposeBundleProblem`, the +Lift/bundle option in refit_imc).
#   • PoseLib's C++ `refine_relpose` (the eval ExternalRefit arms,
#     LocalOptimizationDoneRight src/engines/refit.jl): POSE-ONLY refinement of the TRUNCATED Sampson
#     loss — the LO-RANSAC standard (+PoseLib(TLS)/(LZ)). First-order,
#     structure-free, hard-thresholded; a comparison baseline, not a bundle.
# Both use Kc as cameras in the same measurement frame; the objective (exact
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
# damping, step accept/reject, convergence: VGC `nlls_options`) and, where the
# loaded NLLSsolver has it (the stack's fork, VGC `nlls_features`), the Schur
# complement: its `MultiVariateLSschur` eliminates every point
# (`iseliminable`). On a registered NLLSsolver the same problem runs on its
# sparse LDL over all 5 + 3N unknowns — the same optimum, slower.
#
# Until 2026-10-27 the LM was FastLevenbergMarquardt's, on a stacked
# fixed-pattern sparse Jacobian (22N stored entries) with this file's own
# Schur solver (`SchurSolver <: FastLevenbergMarquardt.AbstractSolver`), the
# pose block a FIXED chart anchored at the seed. NLLSsolver re-anchors the
# pose variable at every accepted step (VGC's manifold `update` is
# `recenter`); both reach the same optimum (measured 2026-10-27 on the A/B
# scenes — 800 px, σ = 0.5 px, N = 100 / 1000 / 10000 — cost to ≤ 3e-14
# relative, rotation to ≤ 6e-11°). Before 2026-10-01 J was dense, O(N²); before
# 2026-10-27 the pose block was `RigidManifold6` (MRP ⊕ free ℝ³ t, one gauge
# direction held by the damping): the minimal chart reaches its optimum (cost
# ≤ 5.5e-15 relative, pose ≤ 7.4e-12°).
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
const _BundlePose = typeof(EssentialManifold5(SMatrix{3, 3, Float64, 9}(I), SVector(0.0, 0.0, 1.0)))
# The pose block's dimension: `EssentialManifold5`'s tangent size.
const _POSE_DOF = 5

# A structure point, free in ℝ³. `E` says whether NLLSsolver may eliminate it
# (the fork's Schur-complement linear system); `_BundlePoint{false}` is the
# upstream path — NLLSsolver's sparse LDL over every unknown — which a
# registered NLLSsolver takes and the fallback test forces.
struct _BundlePoint{E}
    X::SVector{3, Float64}
end
NLLSsolver.nvars(::_BundlePoint) = NLLSsolver.static(3)
NLLSsolver.update(p::_BundlePoint{E}, δ, start = 1) where {E} =
    _BundlePoint{E}(p.X + SVector(δ[start], δ[start + 1], δ[start + 2]))

# The fork's additions, detected once (VGC `nlls_features`): the Schur
# complement (`iseliminable`, #60) and the internal-data cache (#62).
const _BUNDLE_SCHUR = nlls_features().schur
const _BUNDLE_CACHE = nlls_features().cache
if _BUNDLE_SCHUR
    NLLSsolver.iseliminable(::Type{_BundlePoint{true}}) = NLLSsolver.static(true)
end
const _BundleCacheType = _BUNDLE_CACHE ? NLLSsolver.OptimizeCache : Nothing

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
const _BundleOwnedData = _BundleData{Vector{SVector{2, Float64}}, Vector{SVector{2, Float64}}, Vector{Float64}}
_BundleOwnedData(N::Int) = _BundleOwnedData(
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
    (vars[r.varind[1]]::_BundlePose, vars[r.varind[2]]::_BundlePoint{E})
Base.eltype(::_BundleRows) = Float64
function NLLSsolver.computeresidual(r::_BundleRows, pose::_BundlePose, p::_BundlePoint)
    d = r.data; i = r.i
    R = SMatrix{3, 3, Float64, 9}(pose.R); t = SVector{3, Float64}(pose.t)
    return d.sqrtw[i] * _reprojection_error(d.K1, d.K2, R, t, p.X, d.u1[i], d.u2[i])
end
# The block's cost as the LM reads it: non-finite is +Inf (`_finite_cost`).
NLLSsolver.computecost(vars::Vector{Union{_BundlePose, _BundlePoint{E}}}, r::_BundleRows{E}) where {E} =
    _finite_cost(NLLSsolver.computecost(r, NLLSsolver.getvars(r, vars)...))
# The closed-form 4 × 8 Jacobian, pose columns (VGC's `pose_action_jacobian`
# at the variable's own anchor, the derivative of its `update`) then point
# columns. Every variable is free in this problem (`optimize!` over all), so
# NLLSsolver asks for both blocks' columns: `varflags` = 0b11.
function NLLSsolver.computeresjac(_varflags, r::_BundleRows, pose::_BundlePose, p::_BundlePoint)
    d = r.data; i = r.i; w = d.sqrtw[i]
    R = SMatrix{3, 3, Float64, 9}(pose.R); t = SVector{3, Float64}(pose.t); X = p.X
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
# system's layout) is kept in `cache`, built for the one options type the
# bundle calls with (`nlls_options`'s). Without the cache a call rebuilds them
# (measured 2026-10-27: 0.88 / 1.04 / 1.00× FastLevenbergMarquardt's time at
# N = 100 / 1000 / 10000 with it, 1.78 / 1.49 / 1.44× without).
#
# NOT REENTRANT: a bundle started from inside the NLLSsolver `callback` of a
# bundle on the same task overwrites the running problem's data and
# variables (the internal `callback` keyword; no production caller supplies one).
struct _BundleRun{E, D}
    data::D
    problem::NLLSsolver.NLLSProblem{Union{_BundlePose, _BundlePoint{E}}, _BundleRows{E, D}}
    # The fork's TYPED OptimizeCache built for this problem (VGC `nlls_features`), or `nothing`.
    # Its concrete type depends on which linear system the problem's structure selects (Schur,
    # sparse or dense), a run-time choice, so the field holds it abstractly: one dynamic call per
    # bundle, the solver's loop inside it static.
    cache::Union{Nothing, _BundleCacheType}
end
function _BundleRun{E}(N::Int, data::D) where {E, D}
    problem = NLLSsolver.NLLSProblem(Union{_BundlePose, _BundlePoint{E}}, _BundleRows{E, D})
    NLLSsolver.addvariable!(problem, EssentialManifold5(SMatrix{3, 3, Float64, 9}(I), SVector(0.0, 0.0, 1.0))::_BundlePose)
    for i in 1:N
        NLLSsolver.addvariable!(problem, _BundlePoint{E}(zero(SVector{3, Float64})))
        NLLSsolver.addcost!(problem, _BundleRows{E, D}(data, i, SVector(1, i + 1)))
    end
    # the internal data (the Schur system's layout) are built now, for this problem's structure
    cache = _BUNDLE_CACHE ? NLLSsolver.OptimizeCache(problem, nlls_options(; max_iter = 1, ftol = 1.0, xtol = 1.0), nothing) : nothing
    return _BundleRun{E, D}(data, problem, cache)
end
# The task's slot for its one run of point type `E`.
mutable struct _BundleSlot{E}
    run::Union{Nothing, _BundleRun{E, _BundleOwnedData}}
end
_bundle_size(run::_BundleRun) = length(run.problem.variables) - 1
function _bundle_run(::Val{E}, N::Int) where {E}
    slot = task_buffers(_BundleSlot{E}, () -> _BundleSlot{E}(nothing), ())
    run = slot.run
    run !== nothing && _bundle_size(run) == N && return run
    slot.run = nothing                        # the old size's problem is released before the new is built
    new = _BundleRun{E}(N, _BundleOwnedData(N))
    slot.run = new
    return new
end

# The stacked 4N × (5 + 3N) Jacobian and residual of the problem at its current
# variables (the stall postcondition's input; sparse, the arrowhead pattern).
function _bundle_jacobian(run::_BundleRun{E}) where {E}
    problem = run.problem
    N = length(problem.variables) - 1
    pose = problem.variables[1]::_BundlePose
    Is = Int[]; Js = Int[]; Vs = Float64[]
    f = zeros(4N)
    for i in 1:N
        rows = _BundleRows{E, typeof(run.data)}(run.data, i, SVector(1, i + 1))
        e, Ji = NLLSsolver.computeresjac(NLLSsolver.static(3), rows, pose, problem.variables[i + 1]::_BundlePoint{E})
        f[(4i - 3):(4i)] = e
        for a in 1:4, b in 1:(_POSE_DOF + 3)
            push!(Is, 4i - 4 + a)
            push!(Js, b <= _POSE_DOF ? b : _POSE_DOF + 3(i - 1) + b - _POSE_DOF)
            push!(Vs, Ji[a, b])
        end
    end
    return sparse(Is, Js, Vs, 4N, _POSE_DOF + 3N), f
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
@inline function _bearing(iK::SMatrix{3, 3, Float64, 9}, u)
    return normalize(iK * SVector{3, Float64}(Float64(u[1]), Float64(u[2]), one(Float64)))
end

# Ray-domain triangulation, replacing the pixel-domain DLT.
#
# WHY: the DLT builds `P = K[I|0]` — pinhole BY CONSTRUCTION — and that matrix
# does not exist for the fisheye and omnidirectional cameras this wave is for.
# VGC's `triangulate(pose, f₁, f₂)` takes bearings and needs no such matrix.
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
# PARALLEL RAYS: VGC's `triangulate` returns NaN there (nothing was
# triangulated, the honest answer for cheirality). A bundle cannot start from
# NaN — one far point, or one baseline-parallel outlier, made the whole refit
# and every residual NaN, so `bundle_refine_relpose` returned `nothing` even
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
    f₁ = _bearing(iK1, u1)
    X, _, _ = triangulate(
        EuclideanMap(SMatrix{3, 3, Float64, 9}(R), SVector{3, Float64}(t); check = false),
        f₁, _bearing(iK2, u2)
    )
    all(isfinite, X) || (X = _PARALLEL_DEPTH * f₁)
    return Point3{Float64}(X[1], X[2], X[3])
end

# Optimal 2-view triangulation through K: minimize ‖π(K₁X)−u₁‖²+‖π(K₂(RX+t))−u₂‖²
# over X (ray-form init + Gauss-Newton).  The per-pose structure optimum the BA
# converges to — so `residuals!` here matches the BA's own per-point residual,
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

# The bundle problem's cheirality: its K-frame points back-projected ONCE
# (two bearings per point, not two per point per motion) through the
# problem's stored K⁻¹, then THE vote.
function _cheiral_relpose(E, iK1, iK2, u1, u2)
    R, t = cheiral_pose_fast(E, [_bearing(iK1, u) for u in u1], [_bearing(iK2, u) for u in u2])
    return R, Vec3{Float64}(t)
end

"""
    bundle_refine_relpose(pose::PlainEuclideanMap{3}, K1, K2, cs; max_iter)   # default: configs/default.toml [relpose_bundle]
        -> Union{EuclideanMap{3}, Nothing}

Robust full-3D two-view bundle adjustment of a relative pose: the exact
reprojection MLE over the pose and one 3D landmark per correspondence (profiled
out through the Schur complement), seeded at `pose` and returning the refined
pose with unit translation, or `nothing` on failure (fewer than 5
correspondences, a solver failure).

`cs` are the `u₁ => u₂` correspondences in the frame of the intrinsics `K1`,
`K2` (reprojection is `π(Kᵢ·X)`), any `sources`/`targets` container; per-point
weights are its `weights` column when it has one (e.g. `withattrs(cs; weights
= w)`, the IRLS posterior), unit otherwise.

The seed is a plain pose (a physical one is a `MethodError`): two views fix a
relative pose only up to scale, so its translation is a direction — read
unit-normalized, returned with unit norm — and no length enters or leaves. A
physical seed is stripped by the caller in any unit (`ustrip(m, pose)`).
"""
function bundle_refine_relpose(
        pose::PlainEuclideanMap{3}, K1::AbstractMatrix, K2::AbstractMatrix,
        cs::AbstractVector{<:Pair}; max_iter::Int = _RELPOSE_BUNDLE_MAX_ITER
    )
    u1 = sources(cs); u2 = targets(cs)
    N = length(u1)
    # Per-point √weights from the `weights` column, through VGC's one fill and
    # its one contract (a non-finite or negative weight is the caller's error);
    # unit weights when there is none (per-correspondence data travels as a
    # column, never a keyword).
    w = VgtkCore.weights_column(cs)
    sqrtw = w === nothing ? ones(N) : fill_sqrt_weights!(Vector{Float64}(undef, N), w, N)
    res = _bundle_refine_relpose(
        SMatrix{3, 3, Float64, 9}(rotation(pose)), SVector{3, Float64}(translation(pose)),
        K1, K2, _inv3(K1), _inv3(K2), u1, u2, sqrtw; max_iter
    )
    res === nothing && return nothing
    return EuclideanMap(RotMatrix{3, Float64}(res.R), res.t; check = false)
end

# -----------------------------------------------------------------------------
# Core routine: robust full-3D two-view BA (camera-aware).  Seed pose (R₀,t₀) +
# intrinsics K₁,K₂ and their inverses + image points u₁,u₂ (in the K frame) +
# per-point √weights → `(; R, t, result, retcode)`: the refined (R, t) (‖t‖=1),
# NLLSsolver's result, and the retcode (VGC `nlls_retcode`, the stall
# postcondition at `stationarity_tol`) when `retcode = true` — `nothing`
# otherwise: no production caller reads it (a `:stalled` bundle's pose is still
# scored by the caller, the LO's acceptance, as a converged one is), and it
# costs a sweep over the rows. `nothing` on failure. Positional and
# allocation-lean: the RANSAC refit (`refine!` below) calls it with the
# problem's stored K⁻¹ and the √weights it builds. `schur = false` or
# `cache = false` take the upstream path where the fork is loaded (the
# fallback test runs it); `callback` is NLLSsolver's per-iteration callback
# (the task's kept run is not reentrant from it: see `_BundleRun`).
# -----------------------------------------------------------------------------
function _bundle_refine_relpose(
        R₀::AbstractMatrix, t₀::AbstractVector,
        K1::AbstractMatrix, K2::AbstractMatrix,
        iK1::SMatrix{3, 3, Float64, 9}, iK2::SMatrix{3, 3, Float64, 9},
        u1::AbstractVector, u2::AbstractVector,
        sqrtw::AbstractVector; max_iter::Int = _RELPOSE_BUNDLE_MAX_ITER,
        ftol::Float64 = _RELPOSE_BUNDLE_FTOL, retcode::Bool = false,
        stationarity_tol::Float64 = _RELPOSE_BUNDLE_STATIONARITY_TOL,
        schur::Bool = _BUNDLE_SCHUR, cache::Bool = true, callback::F = NLLSsolver.nullcallback
    ) where {F}
    # (`::F`: a Function-typed argument only handed on is not specialized,
    # which made the call below a dynamic one, boxing every argument)
    N = length(u1)
    N >= 5 || return nothing
    Rb = SMatrix{3, 3, Float64, 9}(R₀); tb = Vec3{Float64}(normalize(SVector{3, Float64}(t₀)))
    E = schur && _BUNDLE_SCHUR
    return E ? _bundle_optimize(Val(true), N, K1, K2, u1, u2, sqrtw, Rb, tb, iK1, iK2, max_iter, ftol, retcode, stationarity_tol, cache, callback) :
        _bundle_optimize(Val(false), N, K1, K2, u1, u2, sqrtw, Rb, tb, iK1, iK2, max_iter, ftol, retcode, stationarity_tol, cache, callback)
end

function _bundle_optimize(
        v::Val{E}, N, K1, K2, u1, u2, sqrtw, Rb, tb, iK1, iK2, max_iter, ftol, retcode, stationarity_tol, cache, callback
    ) where {E}
    run = _bundle_run(v, N)
    d = run.data
    d.K1 = SMatrix{3, 3, Float64, 9}(K1); d.K2 = SMatrix{3, 3, Float64, 9}(K2)
    for i in 1:N                              # the call's data copied into the run's own buffers (bounds-checked: the caller's lengths)
        d.u1[i] = SVector{2, Float64}(u1[i][1], u1[i][2]); d.u2[i] = SVector{2, Float64}(u2[i][1], u2[i][2])
        d.sqrtw[i] = sqrtw[i]
    end
    problem = run.problem
    problem.variables[1] = EssentialManifold5(Rb, tb)::_BundlePose
    @inbounds for i in 1:N
        X = _triangulate_rays(iK1, iK2, d.u1[i], d.u2[i], Rb, tb)
        problem.variables[i + 1] = _BundlePoint{E}(SVector{3, Float64}(X[1], X[2], X[3]))
    end
    # The stack's knob map (VGC `nlls_options`): Marquardt's DᵀD damping, as
    # the bundle's FastLevenbergMarquardt run had; `xtol` FLM's default.
    options = nlls_options(; max_iter, ftol, xtol = 1.0e-10)
    c = run.cache
    result = try
        if cache && c !== nothing
            NLLSsolver.optimize!(problem, options, nothing, callback, c)
        else
            NLLSsolver.optimize!(problem, options, nothing, callback)
        end
    catch
        return nothing
    end
    pose = problem.variables[1]::_BundlePose
    R = SMatrix{3, 3, Float64, 9}(pose.R); t = Vec3{Float64}(pose.t)
    (all(isfinite, R) && all(isfinite, t)) || return nothing
    code = retcode ? _bundle_retcode(run, result, stationarity_tol) : nothing
    return (; R, t, result, retcode = code)
end

# The bundle's retcode: NLLSsolver's termination bits and the stall
# postcondition on the stacked rows (`nlls_retcode`; its exact-fit test at
# FastLevenbergMarquardt's eps, as the bundle's former run had).
function _bundle_retcode(run::_BundleRun, result, tol)
    J, f = _bundle_jacobian(run)
    return nlls_retcode(result, J, f; tol, ftol = eps())
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
robust Schur-reduced BA.  `K1`,`K2` are the intrinsics of the measurement frame
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
end
RelposeBundleProblem(K1::K, K2::K, cs::C) where {C, K <: AbstractMatrix} =
    RelposeBundleProblem{C, K}(K1, K2, _inv3(K1), _inv3(K2), cs)

data_size(p::RelposeBundleProblem) = length(p.cs)
codimension(::Type{<:RelposeBundleProblem}) = 1
model_type(::Type{<:RelposeBundleProblem}) = EssentialMat{Float64}

# Scoring residual: exact reprojection of the OPTIMALLY-triangulated point — the
# exact codim-1 geometric error Sampson approximates to first order, and the same
# per-point objective the BA refit minimizes (so the IRLS reweight is consistent
# with the refit — same residual feeds `update_weights!` and the BA cost). The
# per-model work — the cheiral pose of θ — is RSDR's `residual_context`, taken
# once per model; `residuals!` is RSDR's loop over it.
struct _RelposeResidualContext
    R::SMatrix{3, 3, Float64, 9}
    t::Vec3{Float64}
end
function residual_context(p::RelposeBundleProblem, θ::EssentialMat)
    R, t = _cheiral_relpose(θ, p.iK1, p.iK2, sources(p.cs), targets(p.cs))
    return _RelposeResidualContext(SMatrix{3, 3, Float64, 9}(R), t)
end
@inline function whitened_residual(p::RelposeBundleProblem, ctx::_RelposeResidualContext, i::Int)
    u1 = @inbounds sources(p.cs)[i]; u2 = @inbounds targets(p.cs)[i]
    X = _triangulate_optimal(u1, u2, p.K1, p.K2, p.iK1, p.iK2, ctx.R, ctx.t)
    return norm(_reprojection_error(p.K1, p.K2, ctx.R, ctx.t, X, u1, u2))
end

# Robust BA refit — the problem's own LO step (a Schur-reduced full-3D bundle,
# not a chart LM: it has no `chart_dof`). `weights` are the per-point Student-t
# posterior weights from the IRLS reweight; the seed pose is the current
# essential's cheiral decomposition.
function refine!(_ws, p::RelposeBundleProblem, θ::EssentialMat, weights::AbstractVector)
    _underdetermined(p, weights) && return nothing
    u1 = sources(p.cs); u2 = targets(p.cs)
    R₀, t₀ = _cheiral_relpose(θ, p.iK1, p.iK2, u1, u2)
    sqrtw = fill_sqrt_weights!(Vector{Float64}(undef, length(weights)), weights, length(weights))
    res = _bundle_refine_relpose(R₀, t₀, p.K1, p.K2, p.iK1, p.iK2, u1, u2, sqrtw)
    res === nothing && return nothing
    return EssentialMat(res.R, res.t)
end
