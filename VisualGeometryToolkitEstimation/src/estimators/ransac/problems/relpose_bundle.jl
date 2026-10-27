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
#   • `bundle_refine_relpose` (THIS file; VGC's `lmsolve_rows!` on FastLevenbergMarquardt
#     with the Schur-complemented arrowhead structure): the RelposeBundleProblem's
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
# back-substitute δXᵢ.  Hooked into `FastLevenbergMarquardt` via `SchurSolver
# <: AbstractSolver` — FLM owns the LM outer loop (λ schedule, step
# accept/reject, convergence), we own only the damped linear solve.
#
# Until 2026-10-27 the pose block was `RigidManifold6` (MRP ⊕ free ℝ³ t): one
# gauge direction, ‖t‖, which only the Marquardt damping held, renormalized at
# exit. The minimal chart reaches the same optimum (on the A/B scenes — 800 px,
# σ = 0.5 px, N = 100 / 1000 / 10000 — cost to ≤ 5.5e-15 relative, pose to
# ≤ 7.4e-12° of the 6-dof chart's tightly converged one) with the same
# allocations. The run's length is set by its cost tolerance, not the chart:
# at FLM's default `ftol = eps` either chart ran 20–35 LM iterations after the
# cost had converged by iteration 3 (at the rounding floor every candidate step
# is rejected and FLM's per-element step test waits for λ to climb back);
# `[relpose_bundle] ftol = 1e-10`, the chart LM's, stops it in 5.
#
# The Jacobian is a `SparseArrays.SparseMatrixCSC` with the FIXED arrowhead
# pattern (`_arrowhead_jacobian`): 22N stored entries — a 4 × 3 structure block
# per point, a 2 × 5 pose block per image-2 pair — built once per call; the
# Jacobian fill writes their values in place and the Schur solver reads its
# blocks at precomputed offsets into the values (`_pose_nz`, `_point_nz`). The
# two operations FLM applies to J itself, `g = J'f` and the column scaling
# `sum!(abs2, D', J)`, are SparseArrays' O(nnz), allocation-free methods.
# Until 2026-10-01 J was FLM's DENSE 4N × (6 + 3N) matrix (the 6-dof chart's) — O(N²) to fill,
# multiply and hold. Measured against it 2026-10-01 (one full bundle solve,
# @benchmark median, the same driver, Schur solver and stall postcondition;
# two-view scenes at 800 px focal, σ = 1 px, a 1° / 0.03 seed, soft weights):
#     N            10     20     50    100    200    500   1000   2000
#     dense  ms  0.058  0.081  0.331  0.761   3.20   22.5    148    964
#     sparse ms  0.059  0.077  0.212  0.384  0.661   1.73   3.06   5.04
#     dense  MB  0.018  0.053  0.267  0.989   3.81   23.2   92.3    368
#     sparse MB  0.012  0.021  0.049  0.095  0.187  0.464  0.925   1.85
# The sparse path is never materially slower — N = 10 is the one size it
# loses, by 1 µs in 55 (minimum of repeated runs: 56.4 vs 55.3 µs), and that
# run takes 51 LM iterations to dense's 49 — so it is the ONE path. The two
# reach the same pose (the representation-independence test).
#
# Robustness: per-point weight wᵢ (the outer IRLS reweight's — the Student-t
# posterior weight under LocalOptimizationDoneRight's `StudentTUniform`) scales all four of point i's residual rows.
# =============================================================================

# -----------------------------------------------------------------------------
# Schur solver — the FastLevenbergMarquardt `AbstractSolver` hook.  Solves the
# damped arrowhead normal equations (J'J + λ·diag(D)) p = g = J'f by eliminating
# the 3×3 structure blocks. Column layout: 1:5 pose (`_POSE_DOF`), then
# 5+3i-2 : 5+3i for point i. FLM calls `init!`/`update!` once per Jacobian (the start and every
# accepted step) and `solve!` once per LM iteration (accepted or rejected), so
# the undamped blocks of J'J — the 5×5 pose block, each point's 5×3 coupling
# and 3×3 structure block — are formed in `update!` and only the damping, the
# per-point Cholesky and the 5×5 reduced solve run per iteration. Every block is
# a static array in a buffer the solver owns: nothing is allocated per call.
# -----------------------------------------------------------------------------
"""
    SchurSolver(npts) <: FastLevenbergMarquardt.AbstractSolver

The linear solver of [`bundle_refine_relpose`](@ref)'s Levenberg–Marquardt
iterations. It solves the damped normal equations of the relative pose and
`npts` structure points by eliminating each point's 3×3 block (the Schur
complement), so a solve costs one 5×5 system plus `npts` 3×3 ones and
allocates nothing.
"""
struct SchurSolver <: FastLevenbergMarquardt.AbstractSolver
    npts::Int
    Hpp::Base.RefValue{SMatrix{5, 5, Float64, 25}}       # Σᵢ J_pᵢᵀ J_pᵢ
    HpX::Vector{SMatrix{5, 3, Float64, 15}}               # J_pᵢᵀ J_Xᵢ
    HXX::Vector{SMatrix{3, 3, Float64, 9}}                # J_Xᵢᵀ J_Xᵢ (undamped)
    CX::Vector{Cholesky{Float64, SMatrix{3, 3, Float64, 9}}}   # chol(J_XᵢᵀJ_Xᵢ + λD_Xᵢ), per solve
    gX::Vector{SVector{3, Float64}}                       # the reduced right-hand sides' point parts
end
SchurSolver(npts::Int) = SchurSolver(
    npts, Ref(zero(SMatrix{5, 5, Float64, 25})),
    Vector{SMatrix{5, 3, Float64, 15}}(undef, npts), Vector{SMatrix{3, 3, Float64, 9}}(undef, npts),
    Vector{Cholesky{Float64, SMatrix{3, 3, Float64, 9}}}(undef, npts), Vector{SVector{3, Float64}}(undef, npts)
)

# FLM's workspace with the `_arrowhead_jacobian` buffer — the one J this solver reads.
const _ArrowheadLm = FastLevenbergMarquardt.LMWorkspace{<:Any, <:Any, SparseMatrixCSC{Float64, Int}}

function FastLevenbergMarquardt.init!(s::SchurSolver, ::AbstractVector, L::_ArrowheadLm)
    nz = nonzeros(L.J)
    N = s.npts
    Hpp = zero(SMatrix{5, 5, Float64, 25})
    @inbounds for i in 1:N
        # point i's 2 × 5 pose block (its image-2 rows) and 4 × 3 structure block
        Jp = SMatrix{2, 5, Float64, 10}(ntuple(q -> nz[_pose_nz(N, i, 1 + (q - 1) ÷ 2) + 1 + (q - 1) % 2], Val(10)))
        o = _point_nz(N, i, 1)
        JX = SMatrix{4, 3, Float64, 12}(ntuple(q -> nz[o + q], Val(12)))
        JX2 = SMatrix{2, 3, Float64, 6}(JX[3, 1], JX[4, 1], JX[3, 2], JX[4, 2], JX[3, 3], JX[4, 3])
        Hpp += Jp' * Jp
        s.HpX[i] = Jp' * JX2
        s.HXX[i] = JX' * JX
    end
    s.Hpp[] = Hpp
    return s
end
FastLevenbergMarquardt.update!(s::SchurSolver, f::AbstractVector, L::_ArrowheadLm) = FastLevenbergMarquardt.init!(s, f, L)

function FastLevenbergMarquardt.solve!(s::SchurSolver, λ::Real, ::AbstractVector, L)
    g, D, p = L.g, L.D, L.p
    Htil = s.Hpp[] + Diagonal(SVector{5, Float64}(ntuple(k -> λ * D[k], Val(5))))
    gtil = SVector{5, Float64}(ntuple(k -> g[k], Val(5)))
    @inbounds for i in 1:s.npts
        c = 2 + 3i
        H = s.HXX[i] + Diagonal(SVector(λ * D[c + 1], λ * D[c + 2], λ * D[c + 3]))
        C = cholesky(Symmetric(H); check = false)
        issuccess(C) || return p, 1                          # a non-PD point block → reject the step
        gXi = SVector(g[c + 1], g[c + 2], g[c + 3])
        B = s.HpX[i]
        Htil -= B * (C \ B')
        gtil -= B * (C \ gXi)
        s.CX[i] = C; s.gX[i] = gXi
    end
    F = lu(Htil; check = false)
    any(iszero, diag(F.U)) && return p, 1                    # singular reduced system → reject
    pp = F \ gtil
    all(isfinite, pp) || return p, 1
    @inbounds for k in 1:5
        p[k] = pp[k]
    end
    @inbounds for i in 1:s.npts
        c = 2 + 3i
        δX = s.CX[i] \ (s.gX[i] - s.HpX[i]' * pp)
        p[c + 1] = δX[1]; p[c + 2] = δX[2]; p[c + 3] = δX[3]
    end
    return p, 0
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

# -----------------------------------------------------------------------------
# Augmented residual (4N) + arrowhead Jacobian (4N × 5+3N), in-place.
# x = [ω(3); τ(2); X₁(3); …]; the pose block in the chart anchored at the FIXED
# base (R₀, t₀).
# -----------------------------------------------------------------------------
# The pose block's dimension: `EssentialManifold5`'s tangent size.
const _POSE_DOF = 5

# The linearization state of one relative-pose bundle call: the pose chart
# (anchored at the base pose, ‖t₀‖ = 1), the two calibrations, the two pixel
# views, the √weights and the point count.
struct _RelposeBundleLm{C, T1, T2, TS}
    ch::C
    K1::SMatrix{3, 3, Float64, 9}
    K2::SMatrix{3, 3, Float64, 9}
    u1::T1
    u2::T2
    sqrtw::TS
    N::Int
end
function _RelposeBundleLm(R₀, t₀, K1, K2, u1, u2, sqrtw, N)
    ch = EssentialManifold5(SMatrix{3, 3, Float64, 9}(R₀), normalize(SVector{3, Float64}(t₀)))
    return _RelposeBundleLm{typeof(ch), typeof(u1), typeof(u2), typeof(sqrtw)}(
        ch, SMatrix{3, 3, Float64, 9}(K1), SMatrix{3, 3, Float64, 9}(K2), u1, u2, sqrtw, N
    )
end

# √w-weighted reprojection rows in the pose chart at `x` (`residual_rows!`).
function residual_rows!(r, lm::_RelposeBundleLm, x)
    K1 = lm.K1; K2 = lm.K2; u1 = lm.u1; u2 = lm.u2; sqrtw = lm.sqrtw; N = lm.N
    T = eltype(x)
    R, t = recenter_pose(lm.ch, x)   # the pose block's chart (x[1:5]); ‖t‖ = 1
    @inbounds for i in 1:N
        c = _POSE_DOF + 3i; X = Point3{T}(x[c - 2], x[c - 1], x[c]); w = sqrtw[i]
        e = _reprojection_error(K1, K2, R, t, X, u1[i], u2[i])
        r[4i - 3] = w * e[1]; r[4i - 2] = w * e[2]
        r[4i - 1] = w * e[3]; r[4i] = w * e[4]
    end
    return r
end

# The arrowhead Jacobian's FIXED pattern, column-major (CSC): pose column
# k ∈ 1:5 holds every point's two image-2 rows (4i − 1, 4i) — 2N entries each,
# 10N in all — and point i's three columns 5 + 3(i − 1) + (1:3) its four rows
# 4i − 3 : 4i, so its 4 × 3 block is 12 CONTIGUOUS values, column-major. The
# offsets into `nonzeros(J)` of point i's entries:
@inline _pose_nz(N, i, k) = (k - 1) * 2N + 2(i - 1)          # + a, a ∈ 1:2: rows 4i − 1, 4i of pose column k
@inline _point_nz(N, i, j) = 2_POSE_DOF * N + 12(i - 1) + 4(j - 1)   # + r, r ∈ 1:4: rows 4i − 3 … 4i of point column j

# The 4N × (5 + 3N) Jacobian buffer with that pattern, values zero.
function _arrowhead_jacobian(N::Int)
    nnz = (2_POSE_DOF + 12) * N
    colptr = Vector{Int}(undef, _POSE_DOF + 1 + 3N)
    rowval = Vector{Int}(undef, nnz)
    for k in 1:_POSE_DOF
        colptr[k] = _pose_nz(N, 1, k) + 1
        for i in 1:N, a in 1:2
            rowval[_pose_nz(N, i, k) + a] = 4i - 2 + a
        end
    end
    for i in 1:N, j in 1:3
        colptr[_POSE_DOF + 3(i - 1) + j] = _point_nz(N, i, j) + 1
        for r in 1:4
            rowval[_point_nz(N, i, j) + r] = 4i - 4 + r
        end
    end
    colptr[end] = nnz + 1
    return SparseMatrixCSC(4N, _POSE_DOF + 3N, colptr, rowval, zeros(nnz))
end

# The closed-form Jacobian into the pattern's values; every stored entry is
# written, so the buffer is never zeroed. A dense buffer takes VGC's default
# instead (ForwardDiff through `residual_rows!`) — the representation-
# independence test.
function residual_jacobian_rows!(J::SparseMatrixCSC{Float64, Int}, lm::_RelposeBundleLm, x)
    K1 = lm.K1; K2 = lm.K2; u1 = lm.u1; u2 = lm.u2; sqrtw = lm.sqrtw; N = lm.N
    ch = lm.ch
    R, t = recenter_pose(ch, x)                        # the residual's pose, same chart
    v = SVector{_POSE_DOF, Float64}(x[1], x[2], x[3], x[4], x[5])
    nz = nonzeros(J)
    @inbounds for i in 1:N
        c = _POSE_DOF + 3i; w = sqrtw[i]
        X = Point3{Float64}(x[c - 2], x[c - 1], x[c])
        A1 = w .* (_Dproj(K1 * X) * K1)                                 # ∂(img1)/∂X, 2×3
        Y = R * X + t
        D2K2 = _Dproj(K2 * Y) * K2                                      # 2×3
        A2 = w .* (D2K2 * R)                                            # ∂(img2)/∂X
        P = w .* (D2K2 * pose_action_jacobian(ch, X, v))                # ∂(img2)/∂(ω, τ): VGC's closed form
        for j in 1:3
            o = _point_nz(N, i, j)
            nz[o + 1] = A1[1, j]; nz[o + 2] = A1[2, j]; nz[o + 3] = A2[1, j]; nz[o + 4] = A2[2, j]
        end
        for k in 1:_POSE_DOF
            o = _pose_nz(N, i, k)
            nz[o + 1] = P[1, k]; nz[o + 2] = P[2, k]
        end
    end
    return J
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
    R, t = res
    return EuclideanMap(RotMatrix{3, Float64}(R), t; check = false)
end

# -----------------------------------------------------------------------------
# Core routine: robust full-3D two-view BA (camera-aware).  Seed pose (R₀,t₀) +
# intrinsics K₁,K₂ and their inverses + image points u₁,u₂ (in the K frame) +
# per-point √weights → refined (R, t) (‖t‖=1).  `nothing` on failure.
# Positional and allocation-lean: the RANSAC refit (`_refine!` below) calls it
# with the problem's stored K⁻¹ and the √weights it builds.
# -----------------------------------------------------------------------------
function _bundle_refine_relpose(
        R₀::AbstractMatrix, t₀::AbstractVector,
        K1::AbstractMatrix, K2::AbstractMatrix,
        iK1::SMatrix{3, 3, Float64, 9}, iK2::SMatrix{3, 3, Float64, 9},
        u1::AbstractVector, u2::AbstractVector,
        sqrtw::AbstractVector; max_iter::Int = _RELPOSE_BUNDLE_MAX_ITER,
        stationarity_tol::Float64 = _RELPOSE_BUNDLE_STATIONARITY_TOL,
        ftol::Float64 = _RELPOSE_BUNDLE_FTOL
    )
    N = length(u1)
    N >= 5 || return nothing
    Rb = SMatrix{3, 3, Float64, 9}(R₀); tb = Vec3{Float64}(normalize(SVector{3, Float64}(t₀)))
    Xi = Vector{Float64}(undef, 3N)
    @inbounds for i in 1:N
        X = _triangulate_rays(iK1, iK2, u1[i], u2[i], Rb, tb)
        Xi[3i - 2] = X[1]; Xi[3i - 1] = X[2]; Xi[3i] = X[3]
    end
    # FLM's buffers (the 4N × (5 + 3N) Jacobian in its fixed sparse pattern: see
    # the header) and the Schur solver's blocks, allocated once per call and
    # reused by every LM iteration; the run is VGC's one LM call site. Its
    # retcode is not read: a `:stalled` bundle's pose is still scored by the
    # caller (the LO's acceptance), as a converged one is.
    buffers = FastLevenbergMarquardt.LMWorkspace(vcat(zeros(_POSE_DOF), Xi), zeros(4N), _arrowhead_jacobian(N))
    lm = _RelposeBundleLm(Rb, tb, K1, K2, u1, u2, sqrtw, N)
    out = try
        lmsolve_rows!(buffers, SchurSolver(N), lm; maxit = max_iter, stationarity_tol, ftol)
    catch
        return nothing
    end
    x = out[1]
    any(isnan, x) && return nothing
    # POSE BLOCK via the EssentialManifold5 retraction (so(3) ⊕ T_tS²), the
    # chart the residual and the arrowhead Jacobian's pose block step in: the
    # translation is unit by construction (the retraction normalizes it — an
    # invariant of the chart, checked by Core's `test_manifold_contract`),
    # nothing to renormalize. The N point blocks are free-R³ updates Xᵢ + δ (no
    # product-manifold covariance object: no consumer needs per-landmark
    # covariance).
    R, t = recenter_pose(lm.ch, x)
    return SMatrix{3, 3, Float64, 9}(R), Vec3{Float64}(t)
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
    R, t = res
    return EssentialMat(R, t)
end
