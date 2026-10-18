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
# seed; (6) optimal-triangulation `residuals!` ≤ DLT reprojection.
# =============================================================================
using Test, LinearAlgebra, Random, StaticArrays, Rotations
using VisualGeometryToolkitCore: VisualGeometryToolkitCore
using VisualGeometryToolkitEstimation
using VisualGeometryToolkitEstimation: SchurSolver, bundle_refine_relpose, _RelposeBundleLm,
    residual_rows!, residual_jacobian_rows!, _triangulate_optimal, _reprojection_error,
    _arrowhead_jacobian, recenter_pose, RigidManifold6
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
    rng = MersenneTwister(1); N = 7
    Rgt, tgt, u1, u2 = _scene(rng, N; σ = 0.003)
    R0, t0 = _seed(rng, Rgt, tgt; δ = (0.03, -0.02, 0.04), s = 0.08)
    Xi = reduce(vcat, [_dlt_reference_triangulation(u1[i], u2[i], KC1, KC2, R0, t0) for i in 1:N])
    x = vcat(0.02 * randn(rng, 6), Xi .+ 0.01 * randn(rng, 3N))
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
    rng = MersenneTwister(2); N = 9
    Rgt, tgt, u1, u2 = _scene(rng, N; σ = 0.003)
    R0, t0 = SMatrix{3, 3, Float64, 9}(Rgt), Vec3{Float64}(tgt)
    Xi = reduce(vcat, [_dlt_reference_triangulation(u1[i], u2[i], KC1, KC2, R0, t0) for i in 1:N])
    x = vcat(0.03 * randn(rng, 6), Xi .+ 0.01 * randn(rng, 3N))
    lm = _RelposeBundleLm(R0, t0, KC1, KC2, u1, u2, ones(N), N)
    @test VgtkCore.check_rows_jacobian(lm, x, 4N; rtol = 1.0e-6, atol = 1.0e-9, jacobian = _arrowhead_jacobian(N))
end

@testset "RelposeBundle: sparse and dense Jacobians reach the same pose; 0 allocations" begin
    # The same LM (VGC's `lmsolve_rows!`) on the same rows in two
    # representations: the fixed-pattern sparse J with the Schur solver (what
    # the bundle runs), and a dense J — VGC's ForwardDiff default — with FLM's
    # own Cholesky on the full (6 + 3N)² normal equations. Both are run to a
    # tight step tolerance, so the comparison is the representation, not where
    # each LM stops in the gauge-damped tail. The pose is numerically determined
    # only to ~1e-8 there: perturbing the seed by 1e-14 moves the SAME (sparse)
    # solver's endpoint by up to ~5e-9 in t, so the two representations agree
    # to 1e-8 and the costs to rtol 1e-10.
    rng = MersenneTwister(11); N = 25
    Rgt, tgt, u1, u2 = _scene(rng, N; σ = 0.002)
    R0, t0 = _seed(rng, Rgt, tgt; δ = (0.03, -0.02, 0.04), s = 0.08)
    iK1 = inv(KC1); iK2 = inv(KC2)
    Xi = reduce(vcat, [Vector(_triangulate_rays(iK1, iK2, u1[i], u2[i], R0, t0)) for i in 1:N])
    sw = sqrt.(0.5 .+ 0.5 .* rand(rng, N))
    lm = _RelposeBundleLm(R0, t0, KC1, KC2, u1, u2, sw, N)
    tol::Float64 = VgtkEstimation._dflt("relpose_bundle", "stationarity_tol")
    function bundle_pose(J, solver)
        buffers = FLM.LMWorkspace(vcat(zeros(6), Xi), zeros(4N), J)
        x, F, code, _ = lmsolve_rows!(buffers, solver, lm; maxit = 500, xtol = 1.0e-15, stationarity_tol = tol)
        R, t = recenter_pose(RigidManifold6(R0, t0), x)
        return SMatrix{3, 3, Float64, 9}(R), normalize(SVector{3, Float64}(t)), F, lm_retcode(code)
    end
    Rs, ts, Fs, cs = bundle_pose(_arrowhead_jacobian(N), SchurSolver(N))
    Rd, td, Fd, cd = bundle_pose(zeros(4N, 6 + 3N), FLM.CholeskySolver(6 + 3N))
    @test cs === :converged && cd === :converged
    @test maximum(abs, Rs - Rd) < 1.0e-8 && maximum(abs, ts - td) < 1.0e-8
    @test isapprox(Fs, Fd; rtol = 1.0e-10)
    # The sparse run allocates nothing: FLM's iterations, the Jacobian fill,
    # the Schur solve and the stall postcondition reuse the buffers.
    # (a function barrier: the test item's locals are globals)
    x0 = vcat(zeros(6), Xi)
    buffers = FLM.LMWorkspace(copy(x0), zeros(4N), _arrowhead_jacobian(N)); solver = SchurSolver(N)
    rerun!(b, s, lm, x0, tol, k) = (copyto!(b.x, x0); lmsolve_rows!(b, s, lm; maxit = k, stationarity_tol = tol))
    rerun!(buffers, solver, lm, x0, tol, 100)
    @test (@allocated rerun!(buffers, solver, lm, x0, tol, 100)) == 0
    @test (@allocated rerun!(buffers, solver, lm, x0, tol, 1)) == 0
end

@testset "RelposeBundle: noiseless recovery is exact" begin
    rng = MersenneTwister(3); N = 30
    Rgt, tgt, u1, u2 = _scene(rng, N; σ = 0.0)
    R0, t0 = _seed(rng, Rgt, tgt; δ = (0.04, -0.03, 0.05), s = 0.1)
    res = bundle_refine_relpose(rigid(R0, t0), KC1, KC2, csponds(u1, u2); max_iter = 200)
    @test res !== nothing
    R, t = rotation(res), translation(res); Rerr, terr = _pose_errs(R, t, Rgt, tgt)
    @test Rerr < 1.0e-6
    @test terr < 1.0e-6
end

@testset "RelposeBundle: noisy refinement beats the seed" begin
    rng = MersenneTwister(4); N = 60
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
    rng = MersenneTwister(5); N = 40
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

    rng = MersenneTwister(6); N = 30
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
    rng = MersenneTwister(7); N = 20
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
