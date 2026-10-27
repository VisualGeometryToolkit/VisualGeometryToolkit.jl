@testitem "relpose_bundle" setup = [PoseScenes, EpipolarScenes] begin
    import VisualGeometryToolkitCore as VgtkCore
    import VisualGeometryToolkitEstimation as VgtkEstimation
# =============================================================================
# RelposeBundleProblem — camera-aware full-3D two-view bundle adjustment with
# Schur-reduced FLM solve.  Validates: (1) SchurSolver step == dense damped
# solve; (2) the analytic arrowhead Jacobian (fixed-pattern sparse) ==
# ForwardDiff (K≠I); (3) representation independence — the sparse Schur run and
# a dense run (ForwardDiff Jacobian, FLM's own Cholesky) reach the same pose —
# and 0 allocations per LM iteration on the sparse path; (4) noiseless recovery
# is machine-exact (canonical frame); (5) noisy refinement improves a perturbed
# seed; (6) optimal-triangulation `residuals!` ≤ DLT reprojection; (7) the
# 5-dof chart reaches the retired 6-dof chart's optimum, stationary, ‖t‖ = 1.
# =============================================================================
using Test, LinearAlgebra, Random, StaticArrays, Rotations
using StableRNGs
using VisualGeometryToolkitCore: VisualGeometryToolkitCore
using VisualGeometryToolkitEstimation
using VisualGeometryToolkitEstimation: SchurSolver, bundle_refine_relpose, _RelposeBundleLm,
    residual_rows!, residual_jacobian_rows!, _triangulate_optimal, _reprojection_error,
    _arrowhead_jacobian, recenter_pose, _POSE_DOF
using VisualGeometryToolkitCore: EssentialManifold5, is_stationary
using VisualGeometryToolkitCore: lmsolve_rows!, lm_retcode
using VisualGeometryToolkitCore: CameraModel, Point2, Point3, Vec3, calibration_matrix, csponds, Csponds, project,
    rotation, translation, EssentialMat
using VisualGeometryToolkitCore.Synthetic: pose_error
using VisualGeometryToolkitEstimation: RelposeBundleProblem, _triangulate_rays
using VisualGeometryToolkitCore: cheiral_pose_fast   # its owner (moved from RVG)

# An INDEPENDENT reference, not production code: the pixel-domain DLT
# triangulation through K₁, K₂ (P₁ = K₁[I|0], P₂ = K₂[R|t]; the null vector of
# the 4×4 system uᵢ × (Pᵢ X̃) = 0). Nothing in the stack triangulates this way
# any more — the bundle works in the ray domain, VGC's `triangulate` is the ray
# midpoint — so it is kept here as the oracle the optimal triangulation must
# beat in reprojection, and as the tests' seed structure.
function _dlt_reference_triangulation(u1, u2, K1, K2, R, t)
    P1 = hcat(K1, SVector{3, Float64}(0, 0, 0))               # 3×4
    P2 = hcat(K2 * R, K2 * SVector{3, Float64}(t))            # 3×4
    A = vcat(
        (u1[1] .* P1[3, :] .- P1[1, :])', (u1[2] .* P1[3, :] .- P1[2, :])',
        (u2[1] .* P2[3, :] .- P2[1, :])', (u2[2] .* P2[3, :] .- P2[2, :])'
    )
    h = svd(Matrix(A)).V[:, 4]
    return Point3(h[1] / h[4], h[2] / h[4], h[3] / h[4])
end
import FastLevenbergMarquardt as FLM

# canonical-scale intrinsics (image-size-normalized: fx≈1, cx≈0.5) — the regime
# the eval pipeline runs in (NOT raw pixels, which are badly LM-conditioned).
# The bundle takes K matrices (a declared pinhole island); they are the cameras'.
const CAM1 = CameraModel(; focal = (1.1, 1.05), pp = (0.5, 0.5))
const CAM2 = CameraModel(; focal = (1.2, 1.15), pp = (0.48, 0.52))
const KC1 = SMatrix{3, 3, Float64, 9}(calibration_matrix(CAM1))
const KC2 = SMatrix{3, 3, Float64, 9}(calibration_matrix(CAM2))

# rotation and translation-direction errors, in degrees
_pose_errs(R, t, Rgt, tgt) = rad2deg.(pose_error(R, t, Rgt, tgt))

# Synthetic two-view scene, projected through the two cameras.
function _scene(rng, N; σ = 0.0)
    Rgt = RotMatrix(RotXYZ(0.1, -0.15, 0.08)); tgt = normalize(SVector(0.7, 0.2, 0.6))
    u1, u2 = stereo_pixel_pairs(
        rng, CAM1, CAM2, Rgt, tgt; n = N, σ₁ = σ, σ₂ = σ,
        point = rng -> Point3(2randn(rng), 2randn(rng), 5 + 2rand(rng))
    )
    return Rgt, tgt, u1, u2
end

# The seed a refinement starts from: the truth's rotation composed with
# RotXYZ(δ...), its translation direction moved by s·N(0, I₃).
_seed(rng, Rgt, tgt; δ, s) = (
    SMatrix{3, 3, Float64, 9}(Rgt * RotXYZ(δ...)),
    Vec3{Float64}(normalize(tgt + s * SVector(randn(rng), randn(rng), randn(rng)))),
)

@testset "RelposeBundle: SchurSolver == dense damped solve" begin
    rng = StableRNG(1); N = 7
    Rgt, tgt, u1, u2 = _scene(rng, N; σ = 0.003)
    R0, t0 = _seed(rng, Rgt, tgt; δ = (0.03, -0.02, 0.04), s = 0.08)
    Xi = reduce(vcat, [_dlt_reference_triangulation(u1[i], u2[i], KC1, KC2, R0, t0) for i in 1:N])
    x = vcat(0.02 * randn(rng, _POSE_DOF), Xi .+ 0.01 * randn(rng, 3N))
    J = _arrowhead_jacobian(N); residual_jacobian_rows!(J, _RelposeBundleLm(R0, t0, KC1, KC2, u1, u2, ones(N), N), x)
    Jd = Matrix(J)                                             # the dense reference
    r = zeros(4N); residual_rows!(r, _RelposeBundleLm(R0, t0, KC1, KC2, u1, u2, ones(N), N), x)
    g = Jd' * r; D = diag(Jd' * Jd) .+ 1.0e-9; λ = 0.41
    L = FLM.LMWorkspace(copy(x), copy(r), J)                  # FLM's workspace: the J, g, D and p the solver reads
    copyto!(L.g, g); copyto!(L.D, D)
    s = FLM.init!(SchurSolver(N), r, L)                       # FLM forms the blocks once per Jacobian
    FLM.solve!(s, λ, r, L)
    @test isapprox(L.p, (Jd' * Jd + λ * Diagonal(D)) \ g; rtol = 1.0e-8)
    # the blocks are reused across damping values (FLM re-solves per LM iteration)
    FLM.solve!(s, 10λ, r, L)
    @test isapprox(L.p, (Jd' * Jd + 10λ * Diagonal(D)) \ g; rtol = 1.0e-8)
    @test (@allocated FLM.solve!(s, λ, r, L)) == 0
    @test (@allocated FLM.init!(s, r, L)) == 0
end

@testset "RelposeBundle: analytic Jacobian == ForwardDiff (K≠I)" begin
    rng = StableRNG(2); N = 9
    Rgt, tgt, u1, u2 = _scene(rng, N; σ = 0.003)
    R0, t0 = SMatrix{3, 3, Float64, 9}(Rgt), Vec3{Float64}(tgt)
    Xi = reduce(vcat, [_dlt_reference_triangulation(u1[i], u2[i], KC1, KC2, R0, t0) for i in 1:N])
    x = vcat(0.03 * randn(rng, _POSE_DOF), Xi .+ 0.01 * randn(rng, 3N))
    lm = _RelposeBundleLm(R0, t0, KC1, KC2, u1, u2, ones(N), N)
    @test VgtkCore.check_rows_jacobian(lm, x, 4N; rtol = 1.0e-6, atol = 1.0e-9, jacobian = _arrowhead_jacobian(N))
end

@testset "RelposeBundle: sparse and dense Jacobians reach the same pose; 0 allocations" begin
    # The same LM (VGC's `lmsolve_rows!`) on the same rows in two
    # representations: the fixed-pattern sparse J with the Schur solver (what
    # the bundle runs), and a dense J — VGC's ForwardDiff default — with FLM's
    # own Cholesky on the full (5 + 3N)² normal equations. Both are run to a
    # tight step tolerance, so the comparison is the representation, not where
    # each LM stops.
    rng = StableRNG(11); N = 25
    Rgt, tgt, u1, u2 = _scene(rng, N; σ = 0.002)
    R0, t0 = _seed(rng, Rgt, tgt; δ = (0.03, -0.02, 0.04), s = 0.08)
    iK1 = inv(KC1); iK2 = inv(KC2)
    Xi = reduce(vcat, [Vector(_triangulate_rays(iK1, iK2, u1[i], u2[i], R0, t0)) for i in 1:N])
    sw = sqrt.(0.5 .+ 0.5 .* rand(rng, N))
    lm = _RelposeBundleLm(R0, t0, KC1, KC2, u1, u2, sw, N)
    tol::Float64 = VgtkEstimation._dflt("relpose_bundle", "stationarity_tol")
    function bundle_pose(J, solver)
        buffers = FLM.LMWorkspace(vcat(zeros(_POSE_DOF), Xi), zeros(4N), J)
        x, F, code, _ = lmsolve_rows!(buffers, solver, lm; maxit = 500, xtol = 1.0e-15, stationarity_tol = tol)
        R, t = recenter_pose(lm.ch, x)
        return SMatrix{3, 3, Float64, 9}(R), SVector{3, Float64}(t), F, lm_retcode(code)
    end
    Rs, ts, Fs, cs = bundle_pose(_arrowhead_jacobian(N), SchurSolver(N))
    Rd, td, Fd, cd = bundle_pose(zeros(4N, _POSE_DOF + 3N), FLM.CholeskySolver(_POSE_DOF + 3N))
    @test cs === :converged && cd === :converged
    @test maximum(abs, Rs - Rd) < 1.0e-8 && maximum(abs, ts - td) < 1.0e-8
    @test isapprox(Fs, Fd; rtol = 1.0e-10)
    # The sparse run allocates nothing: FLM's iterations, the Jacobian fill,
    # the Schur solve and the stall postcondition reuse the buffers.
    # (a function barrier: the test item's locals are globals)
    x0 = vcat(zeros(_POSE_DOF), Xi)
    buffers = FLM.LMWorkspace(copy(x0), zeros(4N), _arrowhead_jacobian(N)); solver = SchurSolver(N)
    rerun!(b, s, lm, x0, tol, k) = (copyto!(b.x, x0); lmsolve_rows!(b, s, lm; maxit = k, stationarity_tol = tol))
    rerun!(buffers, solver, lm, x0, tol, 100)
    @test (@allocated rerun!(buffers, solver, lm, x0, tol, 100)) == 0
    @test (@allocated rerun!(buffers, solver, lm, x0, tol, 1)) == 0
end

# The A/B harness's scene (2026-10-27, nlls-schur): pixel cameras at ~800 px focal,
# σ = 0.5 px, N points in a 8 × 6 × (6…12) box, the seed 1° (rotation) and 2°
# (translation direction) off; the seed structure is the bundle's own ray
# triangulation.
const KP1 = SMatrix{3, 3, Float64, 9}(800.0, 0, 0, 0, 800.0, 0, 640, 360, 1)
const KP2 = SMatrix{3, 3, Float64, 9}(820.0, 0, 0, 0, 815.0, 0, 630, 370, 1)
function _harness_scene(N; seed = 1, σ = 0.5)
    rng = StableRNG(seed)
    Rgt = SMatrix{3, 3, Float64, 9}(RotMatrix(AngleAxis(deg2rad(8.0), normalize(SVector(0.1, 1.0, 0.05))...)))
    tgt = normalize(SVector(-1.0, 0.05, 0.1))
    u1 = SVector{2, Float64}[]; u2 = SVector{2, Float64}[]
    while length(u1) < N
        X = SVector(8 * (rand(rng) - 0.5), 6 * (rand(rng) - 0.5), 6 + 6 * rand(rng))
        Y = Rgt * X + tgt
        Y[3] <= 0.5 && continue
        w1 = KP1 * X; w2 = KP2 * Y
        p1 = SVector(w1[1], w1[2]) / w1[3]; p2 = SVector(w2[1], w2[2]) / w2[3]
        (0 <= p1[1] <= 1280 && 0 <= p1[2] <= 720 && 0 <= p2[1] <= 1280 && 0 <= p2[2] <= 720) || continue
        push!(u1, p1 + σ * SVector(randn(rng), randn(rng)))
        push!(u2, p2 + σ * SVector(randn(rng), randn(rng)))
    end
    a = normalize(SVector(randn(rng), randn(rng), randn(rng)))
    R0 = SMatrix{3, 3, Float64, 9}(RotMatrix(AngleAxis(deg2rad(1.0), a...))) * Rgt
    p = normalize(cross(tgt, SVector(randn(rng), randn(rng), randn(rng))))
    t0 = normalize(SVector{3, Float64}(RotMatrix(AngleAxis(deg2rad(2.0), p...)) * tgt))
    return R0, t0, u1, u2
end

# FLM's solver hook around the Schur solver, recording ‖f‖² at every LM
# iteration (FLM has no callback): the cost trajectory of a production run.
struct _CostRecorder <: FLM.AbstractSolver
    s::SchurSolver
    F::Vector{Float64}
end
FLM.init!(r::_CostRecorder, f::AbstractVector, L) = (FLM.init!(r.s, f, L); r)
FLM.update!(r::_CostRecorder, f::AbstractVector, L) = (FLM.update!(r.s, f, L); r)
FLM.solve!(r::_CostRecorder, λ::Real, f::AbstractVector, L) = (push!(r.F, sum(abs2, f)); FLM.solve!(r.s, λ, f, L))

# `_bundle_refine_relpose`'s run, step for step (unit weights), with the cost
# trajectory: (x, F, FLM code, LM iterations, the first iteration within 1e-6
# of the final cost, the linearization state). The seed is read from the pose
# exactly as the public entry reads it.
function _recorded_bundle(seed, u1, u2, tol)
    N = length(u1)
    R0 = SMatrix{3, 3, Float64, 9}(rotation(seed)); t0 = Vec3{Float64}(normalize(SVector{3, Float64}(translation(seed))))
    iK1 = SMatrix{3, 3, Float64, 9}(inv(KP1)); iK2 = SMatrix{3, 3, Float64, 9}(inv(KP2))
    Xi = reduce(vcat, [Vector(_triangulate_rays(iK1, iK2, u1[i], u2[i], R0, t0)) for i in 1:N])
    lm = _RelposeBundleLm(R0, t0, KP1, KP2, u1, u2, ones(N), N)
    buffers = FLM.LMWorkspace(vcat(zeros(_POSE_DOF), Xi), zeros(4N), _arrowhead_jacobian(N))
    rec = _CostRecorder(SchurSolver(N), Float64[])
    x, F, code, iters = lmsolve_rows!(
        buffers, rec, lm; maxit = VgtkEstimation._dflt("relpose_bundle", "max_iter"), stationarity_tol = tol,
        ftol = VgtkEstimation._dflt("relpose_bundle", "ftol")
    )
    k1e6 = findfirst(Fk -> abs(Fk - F) <= 1.0e-6 * F, vcat(rec.F, F)) - 1   # rec.F[1] is the start
    return copy(x), F, code, iters, k1e6, lm
end

@testset "RelposeBundle: the 5-dof chart reaches the 6-dof chart's optimum" begin
    # Until 2026-10-27 the pose block was RigidManifold6 (MRP ⊕ free ℝ³ t: one
    # gauge direction, ‖t‖, held by the damping, renormalized at exit). Its
    # optimum on these scenes, recorded at 750e174 with that chart run TIGHTLY
    # (xtol = 1e-15, maxit = 500; its default stop is within 3.5e-12° in
    # rotation and 2.9e-11° in translation direction of it), is the baseline:
    # the cost agrees to 1e-12 relative and the pose to 1e-6°.
    #
    # WHY 1e-6°: the bundle stops at `[relpose_bundle] ftol = 1e-10` (FLM code
    # 1), one step short of the cost's rounding floor; measured there (Julia
    # 1.13): cost ≤ 4.6e-14 relative, rotation ≤ 1.8e-8°, translation direction
    # ≤ 1.6e-7°. The statistical pose uncertainty on these scenes is ~3e-2° /
    # 8e-3° / 3e-3° (N = 100 / 1000 / 10000; the A/B harness's covariance
    # table), so 1e-6° is a numerical-agreement bound orders below what the
    # data determines — not a precision claim. No `ftol` both stops early and
    # agrees to 1e-9°: even 1e-12 lands ~4e-9° away, the rounding-floor jitter,
    # where the endpoint depends on the compilation context (Pkg.test's
    # --check-bounds=yes, how StaticArrays' `muladd`s fuse) and not on the chart.
    BASELINE = Dict(
        100 => (
            F = 21.272658798665393,
            R = (0.9903552915256028, 0.00779945170937784, -0.13833135978609992, -0.006002995750992926, 0.9998922085940856, 0.01339907589556886, 0.13842095429974424, -0.01243944314970475, 0.9902953598118462),
            t = (-0.9932975970140219, 0.042228515232460316, 0.10759477806767324),
        ),
        1000 => (
            F = 256.86901574388264,
            R = (0.9902541174104681, 0.00790241076589221, -0.1390479588332823, -0.006022098717210892, 0.9998847224576579, 0.013938296983948823, 0.13914207587446445, -0.012965095443524534, 0.990187552447255),
            t = (-0.9938998312778352, 0.04609623159296084, 0.10019113143796067),
        ),
        10000 => (
            F = 2515.6274355985856,
            R = (0.9903655419139288, 0.007860131171017515, -0.13825451792800467, -0.005947941616479485, 0.9998809342080509, 0.014238658566100775, 0.1383499742683659, -0.013279147006093481, 0.9902943748576648),
            t = (-0.9938488000490688, 0.04911359029957447, 0.09925934660731761),
        ),
    )
    tol::Float64 = VgtkEstimation._dflt("relpose_bundle", "stationarity_tol")
    rotdeg(Ra, Rb) = rad2deg(rotation_angle(RotMatrix{3, Float64}(Ra' * Rb)))
    dirdeg(a, b) = rad2deg(atan(norm(cross(a, b)), dot(a, b)))
    iterations_to_stop = Dict{Int, Tuple{Int, Int}}()
    for N in (100, 1000, 10000)
        R0, t0, u1, u2 = _harness_scene(N)
        b = BASELINE[N]
        seed = rigid(R0, t0)
        res = bundle_refine_relpose(seed, KP1, KP2, csponds(u1, u2))
        R = SMatrix{3, 3, Float64, 9}(rotation(res)); t = SVector{3, Float64}(translation(res))
        @test rotdeg(R, SMatrix{3, 3, Float64, 9}(b.R)) <= 1.0e-6
        @test dirdeg(t, SVector(b.t)) <= 1.0e-6
        @test abs(norm(t) - 1) <= 4eps()                     # the chart keeps ‖t‖ = 1; nothing renormalizes it
        x, F, code, iters, k1e6, lm = _recorded_bundle(seed, u1, u2, tol)
        Rr, tr = recenter_pose(lm.ch, x)
        # The recorded run is the public entry's code compiled in another
        # context (StaticArrays' `muladd`s may fuse differently), so it agrees
        # with the public entry to the same numerical bound, not bit for bit.
        @test rotdeg(SMatrix{3, 3, Float64, 9}(Rr), R) <= 1.0e-6
        @test dirdeg(SVector{3, Float64}(tr), t) <= 1.0e-6
        @test abs(F - b.F) <= 1.0e-12 * b.F
        # converged at exit (FLM's cost-tolerance stop), and stationary
        @test lm_retcode(code) === :converged
        J = _arrowhead_jacobian(N); residual_jacobian_rows!(J, lm, x)
        f = zeros(4N); residual_rows!(f, lm, x)
        @test is_stationary(J, f; tol, ftol = eps())
        iterations_to_stop[N] = (iters, k1e6)
    end
    # The run stops when the cost has converged: within 1e-6 of its final value
    # by iteration 3, stopped at iteration 5 (FLM code 1) on every scene. At
    # FLM's default `ftol = eps` it ran 21–34 (Julia 1.13: 34 / 21 / 26; the
    # 6-dof chart 20 / 23 / 28): at the rounding floor every candidate step is
    # rejected and the step-size test waits for λ to climb back
    # (configs/default.toml [relpose_bundle]). The count itself (5 at every N,
    # Julia 1.12 and 1.13) is not tested: it can move with the compilation
    # context.
    for N in (100, 1000, 10000)
        iters, k1e6 = iterations_to_stop[N]
        @test k1e6 <= 4
        @test iters <= k1e6 + 3
    end
end

@testset "RelposeBundle: allocations per call do not grow with N" begin
    # The public entry, Base.@allocations (BenchmarkTools is not a test
    # dependency). ALLOCS_BOUND is the 6-dof chart's count at 750e174
    # (BenchmarkTools: 61 at N = 100, 62 at 1000 and 10000); the 5-dof chart's,
    # with the [relpose_bundle] settings read once, is 53 / 54 / 54
    # (BenchmarkTools and Base.@allocations alike), with 6% fewer bytes.
    ALLOCS_BOUND = 62
    counts = map((100, 1000, 10000)) do N
        R0, t0, u1, u2 = _harness_scene(N)
        cs = csponds(u1, u2); seed = rigid(R0, t0)
        bundle_refine_relpose(seed, KP1, KP2, cs)
        Base.@allocations bundle_refine_relpose(seed, KP1, KP2, cs)
    end
    @test all(<=(ALLOCS_BOUND), counts)
end

@testset "RelposeBundle: noiseless recovery is exact" begin
    rng = StableRNG(3); N = 30
    Rgt, tgt, u1, u2 = _scene(rng, N; σ = 0.0)
    R0, t0 = _seed(rng, Rgt, tgt; δ = (0.04, -0.03, 0.05), s = 0.1)
    res = bundle_refine_relpose(rigid(R0, t0), KC1, KC2, csponds(u1, u2); max_iter = 200)
    @test res !== nothing
    R, t = rotation(res), translation(res); Rerr, terr = _pose_errs(R, t, Rgt, tgt)
    @test Rerr < 1.0e-6
    @test terr < 1.0e-6
end

@testset "RelposeBundle: noisy refinement beats the seed" begin
    rng = StableRNG(4); N = 60
    Rgt, tgt, u1, u2 = _scene(rng, N; σ = 0.002)
    R0, t0 = _seed(rng, Rgt, tgt; δ = (0.05, -0.04, 0.06), s = 0.12)
    seedR, seedt = _pose_errs(R0, t0, Rgt, tgt)
    res = bundle_refine_relpose(rigid(R0, t0), KC1, KC2, csponds(u1, u2))
    R, t = rotation(res), translation(res)
    finR, fint = _pose_errs(R, t, Rgt, tgt)
    @test finR < seedR
    @test fint < seedt
end

@testset "RelposeBundle: optimal triangulation ≤ DLT reprojection" begin
    rng = StableRNG(5); N = 40
    Rgt, tgt, u1, u2 = _scene(rng, N; σ = 0.002)
    R0, t0 = _seed(rng, Rgt, tgt; δ = (0.04, -0.03, 0.05), s = 0.1)
    res = bundle_refine_relpose(rigid(R0, t0), KC1, KC2, csponds(u1, u2))
    R, t = SMatrix{3, 3, Float64, 9}(rotation(res)), Vec3{Float64}(translation(res))
    # The inverses are hoisted here exactly as production hoists them:
    # `_triangulate_optimal` takes K⁻¹ rather than inverting per call, because
    # its callers are per-point loops. Supplying them keeps the test on the
    # same signature the code actually uses.
    iKC1 = SMatrix{3, 3, Float64, 9}(inv(KC1)); iKC2 = SMatrix{3, 3, Float64, 9}(inv(KC2))
    reproj(tri) = [
        norm(_reprojection_error(KC1, KC2, R, t, tri(u1[i], u2[i], KC1, KC2, R, t), u1[i], u2[i]))
            for i in 1:N
    ]
    opt(a, b, K1, K2, Rr, tt) = _triangulate_optimal(a, b, K1, K2, iKC1, iKC2, Rr, tt)
    rgold = reproj(opt); rdlt = reproj(_dlt_reference_triangulation)
    @test all(isfinite, rgold)
    @test all(rgold .<= rdlt .+ 1.0e-9)
end

@testset "RelposeBundle: parallel rays stay finite (RB1)" begin
    # VGC's `triangulate` is NaN for exactly parallel rays; the bundle used to
    # start from that NaN and return `nothing` even with the point at weight 0.
    I3 = SMatrix{3, 3, Float64, 9}(I)
    u = Point2{Float64}(0.3, -0.2)
    X = _triangulate_rays(I3, I3, u, u, I3, Vec3{Float64}(1.0, 0.0, 0.0))   # identical rays, R = I
    @test all(isfinite, X)
    @test X[3] > 0                                   # far along the first ray, in front

    rng = StableRNG(6); N = 30
    Rgt, tgt, u1, u2 = _scene(rng, N; σ = 0.002)
    # A point at infinity: parallel rays in the two views (up to rounding).
    d = normalize(SVector(0.1, -0.05, 1.0))
    push!(u1, SVector{2, Float64}(project(CAM1, d))); push!(u2, SVector{2, Float64}(project(CAM2, Rgt * d)))
    R0, t0 = _seed(rng, Rgt, tgt; δ = (0.03, -0.02, 0.04), s = 0.08)
    for w_far in (0.0, 1.0)
        w = vcat(ones(N), w_far)
        res = bundle_refine_relpose(rigid(R0, t0), KC1, KC2, Csponds(u1, u2; weights = w))
        @test res !== nothing
        @test all(isfinite, SMatrix{3, 3, Float64, 9}(rotation(res)))
    end
    p = RelposeBundleProblem(KC1, KC2, csponds(u1, u2))
    θ = EssentialMat(Rgt, tgt)
    r = zeros(N + 1)
    VgtkEstimation.residuals!(r, p, θ)
    @test all(isfinite, r)
end

@testset "RelposeBundle: caller input validation (RB2, RB3, RB4)" begin
    rng = StableRNG(7); N = 20
    Rgt, tgt, u1, u2 = _scene(rng, N; σ = 0.002)
    t0 = Vec3{Float64}(tgt)
    # RB2: a slightly drifted (non-orthonormal) seed, built with check = false,
    # was re-checked inside the triangulation and threw before the solver's
    # `try`. The bundle takes the caller's pose as given.
    Rdrift = 1.001 * SMatrix{3, 3, Float64, 9}(Rgt)
    @test bundle_refine_relpose(rigid(Rdrift, t0), KC1, KC2, csponds(u1, u2)) !== nothing
    # RB3: a negative (or non-finite) weight is named, not a `sqrt` DomainError.
    w = ones(N); w[4] = -0.5
    @test_throws ArgumentError bundle_refine_relpose(rigid(Rgt, t0), KC1, KC2, Csponds(u1, u2; weights = w))
    w[4] = NaN
    @test_throws ArgumentError bundle_refine_relpose(rigid(Rgt, t0), KC1, KC2, Csponds(u1, u2; weights = w))
    # RB4: bearings of different lengths are refused, not read out of bounds.
    f = [normalize(SVector(randn(rng), randn(rng), 3.0)) for _ in 1:6]
    E = EssentialMat(Rgt, tgt)
    @test_throws DimensionMismatch cheiral_pose_fast(E, f, f[1:5])
    Rv, tv = cheiral_pose_fast(E, f, f)
    @test Rv isa SMatrix{3, 3, Float64, 9}
end
end
