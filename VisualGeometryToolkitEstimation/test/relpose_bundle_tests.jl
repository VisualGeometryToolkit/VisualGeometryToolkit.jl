@testitem "relpose_bundle" setup = [PoseScenes, EpipolarScenes] begin
    import VisualGeometryToolkitCore as VgtkCore
    import VisualGeometryToolkitEstimation as VgtkEstimation
# =============================================================================
# RelposeBundleProblem — camera-aware full-3D two-view bundle adjustment on
# NLLSsolver, structure eliminated by the Schur complement. Validates: (1) the
# closed-form block Jacobian == ForwardDiff (K≠I); (2) the fork's Schur system
# and the upstream sparse path (the fallback a registered NLLSsolver takes)
# reach the same pose; (3) allocations per call flat in N (the kept problem);
# (4) noiseless recovery is machine-exact (canonical frame); (5) noisy
# refinement improves a perturbed seed; (6) optimal-triangulation `residuals!`
# ≤ DLT reprojection; (7) the 5-dof chart reaches the retired 6-dof chart's
# optimum, stationary, ‖t‖ = 1.
# =============================================================================
using Test, LinearAlgebra, Random, StaticArrays, Rotations
using StableRNGs
using VisualGeometryToolkitCore: VisualGeometryToolkitCore
using VisualGeometryToolkitEstimation
using VisualGeometryToolkitEstimation: bundle_refine_relpose, _bundle_refine_relpose, _BundleData, _BundleRows,
    _BundlePoint, _BundlePose, _triangulate_optimal, _reprojection_error, _POSE_DOF, _inv3
using VisualGeometryToolkitCore: EssentialManifold5, recenter_pose
import NLLSsolver
using ForwardDiff: ForwardDiff
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

@testset "RelposeBundle: closed-form block Jacobian == ForwardDiff (K≠I)" begin
    # Point i's 4 × 8 block — pose columns along the chart the variable's
    # `update` steps in (VGC `recenter_pose`), then the point's — against
    # ForwardDiff of the same rows; non-unit weights.
    rng = StableRNG(2); N = 9
    Rgt, tgt, u1, u2 = _scene(rng, N; σ = 0.003)
    R0, t0 = _seed(rng, Rgt, tgt; δ = (0.03, -0.02, 0.04), s = 0.08)
    sw = sqrt.(0.5 .+ 0.5 .* rand(rng, N))
    data = _BundleData(KC1, KC2, u1, u2, sw)
    pose = EssentialManifold5(R0, t0)::_BundlePose
    for i in 1:N
        X = SVector{3, Float64}(_dlt_reference_triangulation(u1[i], u2[i], KC1, KC2, R0, t0)) + 0.01 * randn(rng, SVector{3, Float64})
        rows = _BundleRows{true, typeof(data)}(data, i, SVector(1, i + 1))
        e, J = NLLSsolver.computeresjac(NLLSsolver.static(3), rows, pose, _BundlePoint{true}(X))
        f(δ) = (Rt = recenter_pose(pose, SVector{5}(δ[1], δ[2], δ[3], δ[4], δ[5]));
            sw[i] * _reprojection_error(KC1, KC2, Rt[1], Rt[2], X + SVector(δ[6], δ[7], δ[8]), u1[i], u2[i]))
        @test e ≈ f(zeros(8)) rtol = 1.0e-12
        @test isapprox(J, ForwardDiff.jacobian(f, zeros(8)); rtol = 1.0e-9, atol = 1.0e-12)
    end
end

@testset "RelposeBundle: the Schur system and the upstream path reach the same pose (fallback)" begin
    # The fork's Schur-complement system with its OptimizeCache (what a
    # checkout runs) and the upstream path — NLLSsolver's sparse LDL over all
    # 5 + 3N unknowns, no cache: what a registered NLLSsolver runs, forced here
    # through `schur = false, cache = false` — on the same rows, run to a tight
    # tolerance so the comparison is the linear system, not where each LM
    # stops. Measured 2026-10-27 (Julia 1.13): rotation entries within 2.8e-11,
    # translation within 4.6e-11, cost 6.4e-15 relative (14 and 12 iterations).
    rng = StableRNG(11); N = 25
    Rgt, tgt, u1, u2 = _scene(rng, N; σ = 0.002)
    R0, t0 = _seed(rng, Rgt, tgt; δ = (0.03, -0.02, 0.04), s = 0.08)
    iK1 = _inv3(KC1); iK2 = _inv3(KC2)
    sw = sqrt.(0.5 .+ 0.5 .* rand(rng, N))
    run(; kw...) = _bundle_refine_relpose(R0, t0, KC1, KC2, iK1, iK2, u1, u2, sw; max_iter = 500, ftol = 1.0e-15, retcode = true, kw...)
    fast = run(); slow = run(; schur = false, cache = false)
    # With the fork loaded (a checkout) `fast` is the Schur system and the
    # cache; on a registered NLLSsolver both runs take the upstream path and
    # agree trivially — the comparison holds either way.
    @test fast.retcode === :converged && slow.retcode === :converged
    @test maximum(abs, fast.R - slow.R) < 1.0e-9 && maximum(abs, fast.t - slow.t) < 1.0e-9
    @test isapprox(fast.result.bestcost, slow.result.bestcost; rtol = 1.0e-10)
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

# NLLSsolver's per-iteration callback, recording ‖f‖² after every iteration:
# the cost trajectory of a production run.
struct _CostLog
    F::Vector{Float64}
end
(cb::_CostLog)(cost, problem, data, iteratedata) = (push!(cb.F, 2cost); (cost, 0))

# `_bundle_refine_relpose`'s run (unit weights), with the cost trajectory:
# (R, t, F, retcode, NLLSsolver iterations, the first iteration within 1e-6 of
# the final cost). The seed is read from the pose exactly as the public entry
# reads it.
function _recorded_bundle(seed, u1, u2, tol)
    N = length(u1)
    R0 = SMatrix{3, 3, Float64, 9}(rotation(seed)); t0 = Vec3{Float64}(normalize(SVector{3, Float64}(translation(seed))))
    log = _CostLog(Float64[])
    out = _bundle_refine_relpose(
        R0, t0, KP1, KP2, _inv3(KP1), _inv3(KP2), u1, u2, ones(N);
        retcode = true, stationarity_tol = tol, callback = log
    )
    F = 2 * out.result.bestcost
    k1e6 = findfirst(Fk -> abs(Fk - F) <= 1.0e-6 * F, vcat(2 * out.result.startcost, log.F)) - 1   # [1] is the start
    return out.R, out.t, F, out.retcode, out.result.niterations, k1e6
end

@testset "RelposeBundle: the 5-dof chart reaches the 6-dof chart's optimum" begin
    # Until 2026-10-27 the pose block was RigidManifold6 (MRP ⊕ free ℝ³ t: one
    # gauge direction, ‖t‖, held by the damping, renormalized at exit). Its
    # optimum on these scenes, recorded at 750e174 with that chart run TIGHTLY
    # (xtol = 1e-15, maxit = 500; its default stop is within 3.5e-12° in
    # rotation and 2.9e-11° in translation direction of it), is the baseline:
    # the cost agrees to 1e-12 relative and the pose to 1e-6°.
    #
    # WHY 1e-6°: the bundle stops at `[relpose_bundle] ftol = 1e-10`
    # (NLLSsolver's relative cost-decrease stop, until 2026-10-27
    # FastLevenbergMarquardt's code 1), one step short of the cost's rounding
    # floor; measured there (Julia 1.13, NLLSsolver): cost ≤ 6.2e-14 relative,
    # rotation ≤ 2.1e-8°, translation direction ≤ 1.8e-7° (FastLevenbergMarquardt:
    # 4.6e-14, 1.8e-8°, 1.6e-7°). The statistical pose uncertainty on these scenes is ~3e-2° /
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
        Rr, tr, F, retcode, iters, k1e6 = _recorded_bundle(seed, u1, u2, tol)
        # The recorded run is the public entry's code compiled in another
        # context (StaticArrays' `muladd`s may fuse differently), so it agrees
        # with the public entry to the same numerical bound, not bit for bit.
        @test rotdeg(SMatrix{3, 3, Float64, 9}(Rr), R) <= 1.0e-6
        @test dirdeg(SVector{3, Float64}(tr), t) <= 1.0e-6
        @test abs(F - b.F) <= 1.0e-12 * b.F
        # converged at exit: NLLSsolver's cost-tolerance stop, and stationary
        # (VGC `nlls_retcode`: the stall postcondition on the stacked rows)
        @test retcode === :converged
        iterations_to_stop[N] = (iters, k1e6)
    end
    # The run stops when the cost has converged: within 1e-6 of its final value
    # by iteration 3, stopped at iteration 5 (NLLSsolver's relative
    # cost-decrease stop) on every scene. At NLLSsolver's default tolerance
    # (1e-15) it runs 8 / 8 / 7 — every one a full linearization — and at
    # FastLevenbergMarquardt's eps the former driver ran 21–34
    # (configs/default.toml [relpose_bundle]). The count itself is not tested:
    # it can move with the compilation context.
    for N in (100, 1000, 10000)
        iters, k1e6 = iterations_to_stop[N]
        @test k1e6 <= 4
        @test iters <= k1e6 + 3
    end
end

@testset "RelposeBundle: allocations per call do not grow with N" begin
    # The public entry, Base.@allocations (BenchmarkTools is not a test
    # dependency), on the second call of each size: the NLLSsolver problem and
    # its internal data are kept per (task, N), so a call allocates the seed
    # structure's √weights and a few small solver objects, flat in N. Measured
    # 2026-10-27 (NLLSsolver, the fork's Schur system and TYPED cache, stack
    # b0f800c): 26 / 28 / 28 at N = 100 / 1000 / 10000 on Julia 1.13 (the
    # untyped cache: 31 / 34 / 34). The bound is the fork's: on a registered
    # NLLSsolver (no cache) the assertion is skipped. Until then FastLevenbergMarquardt's
    # call built its buffers per call: 53 / 54 / 54 (the bound was 62).
    ALLOCS_BOUND = 28
    counts = map((100, 1000, 10000)) do N
        R0, t0, u1, u2 = _harness_scene(N)
        cs = csponds(u1, u2); seed = rigid(R0, t0)
        bundle_refine_relpose(seed, KP1, KP2, cs)
        Base.@allocations bundle_refine_relpose(seed, KP1, KP2, cs)
    end
    if VgtkCore.nlls_features().cache
        @test all(<=(ALLOCS_BOUND), counts)
    else
        @test_skip all(<=(ALLOCS_BOUND), counts)
    end
end

# The bundle runs a task keeps: the kept problems found in its task-local
# storage, however they are stored (a slot holding one, or a Dict of them).
function _kept_bundle_runs()
    runs = Any[]
    for v in values(task_local_storage()), x in (v isa AbstractDict ? values(v) : (v,))
        if x isa VisualGeometryToolkitEstimation._BundleRun
            push!(runs, x)
        elseif hasproperty(x, :run) && getproperty(x, :run) isa VisualGeometryToolkitEstimation._BundleRun
            push!(runs, getproperty(x, :run))
        end
    end
    return runs
end

@testset "RelposeBundle: one kept problem per task, whatever the sizes, holding no caller array" begin
    # An eval sweep calls the bundle with a different N per pair: the task
    # keeps ONE problem (the last size's), not one per size, and the run owns
    # its data — the caller's arrays are not referenced after the call.
    fetch(@async begin                          # a fresh task: empty task-local storage
        for N in (100, 120, 100, 150)
            R0, t0, u1, u2 = _harness_scene(N)
            cs = csponds(u1, u2)
            @test bundle_refine_relpose(rigid(R0, t0), KP1, KP2, cs) !== nothing
            runs = _kept_bundle_runs()
            @test length(runs) == 1
            d = only(runs).data
            @test d.u1 !== VgtkCore.sources(cs) && d.u2 !== VgtkCore.targets(cs) && d.u1 !== u1 && d.u2 !== u2
        end
    end)
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
