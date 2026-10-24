@testitem "refine_robot_world_hand_eye" setup = [PoseScenes] begin
# refine_robot_world_hand_eye: the reprojection-error refinement of a robot-world
# calibration A_i X_m = Y B_i^m against the markers' FEATURES, camera and rig
# poses fixed. Exact recovery, the argument contract, degeneracy REPORTED, and
# the Monte Carlo hard gate: chart NEES of Y, each X_m and the joint vector,
# held-out reprojection, the number of frames, the field of view, the number
# of markers and a poor seed — all against VGC's harness (z = 4 gates) — and
# many markers (M = 6, 10, 20: no marker cap).

using Test
using StableRNGs
using LinearAlgebra: I, norm, normalize, diag
using StaticArrays
using Rotations: Rotations, AngleAxis, RotMatrix, RotationVec, RotZ
using ForwardDiff: ForwardDiff
using VisualGeometryToolkitCore: EuclideanMap, CameraModel, project
using VisualGeometryToolkitCore: RigidExpManifold6, local_coordinates   # the decoupled pose chart (public)
using VisualGeometryToolkitCore: estimate, retcode, rotation, translation   # VGC accessors (public)
using VisualGeometryToolkitCore.Synthetic: run_calibration, is_calibrated   # VGC's MC calibration harness (public)
using VisualGeometryToolkitEstimation: refine_robot_world_hand_eye, RobotWorldHandEyeCalibration
import VisualGeometryToolkitEstimation as RVG

# A marker: a 5×5 grid of points (spacing 3 cm) in its own z = 0 plane.
_grid(s = 0.03) = [SVector(s * (i - 3), s * (j - 3), 0.0) for i in 1:5 for j in 1:5]

# The camera Y = T_WC, rig poses A_i = Y ∘ C_i placing the rig ~depth in front
# of the camera (C_i = rig in camera, rotations up to `ang`), M markers X_m on
# the rig — in a row 8 cm apart (`layout = :row`), or, for many markers, on a
# ⌈√M⌉-column grid 15 cm apart (`layout = :grid`) that keeps every marker in
# view. f and the image set the FOV; the rig's lateral jitter scales with it
# so every point stays in the image. Pixel noise σ, isotropic.
function _rw_marker_origin(rng, m, M, layout)
    layout === :row && return SVector(0.08 * (m - (M + 1) / 2), 0.03 * randn(rng), 0.0)
    nc = ceil(Int, sqrt(M)); nr = cld(M, nc); r, c = divrem(m - 1, nc)
    return SVector(0.15 * (c - (nc - 1) / 2), 0.15 * (r - (nr - 1) / 2), 0.0)
end
function rw_scene(rng; N = 12, M = 3, f = 600.0, W = 800, H = 600, depth = 1.2, σ = 0.5, ang = 0.35, layout = :row)
    camera = CameraModel(; focal = f, pp = (W / 2, H / 2))
    Y = rigid(AngleAxis(π * rand(rng), normalize(randn(rng, 3))...), 0.5 .* randn(rng, SVector{3, Float64}))
    X = [rigid(AngleAxis(0.3 * rand(rng), normalize(randn(rng, 3))...), _rw_marker_origin(rng, m, M, layout)) for m in 1:M]
    lat = 0.35 * depth * (W / 2) / f
    A = map(1:N) do _
        C = rigid(AngleAxis(ang * rand(rng), normalize(randn(rng, 3))...), SVector(lat * (2rand(rng) - 1), lat * (H / W) * (2rand(rng) - 1), depth * (0.85 + 0.3rand(rng))))
        Y ∘ C
    end
    obs = Tuple{Int, Int, Vector{Pair{SVector{3, Float64}, SVector{2, Float64}}}}[]
    for i in 1:N, m in 1:M
        B = marker_pose(Y, A[i], X[m])
        cs = Pair{SVector{3, Float64}, SVector{2, Float64}}[]
        for P in _grid()
            Pc = B(P)
            Pc[3] > 0.1 || continue
            u0 = SVector{2, Float64}(project(camera, Pc)...)
            (0 < u0[1] < W && 0 < u0[2] < H) || continue
            push!(cs, P => u0 + σ * randn(rng, SVector{2, Float64}))
        end
        isempty(cs) || push!(obs, (i, m, cs))
    end
    return (; camera, Y, X, A, obs)
end
# The refinement's input: per marker, one `A_i => [X => u, …]` element per frame
# it is seen in (frame order), over the frames in `frames`.
rw_markers(sc; frames = 1:length(sc.A)) = Dict(
    m => [sc.A[i] => cs for (i, mm, cs) in sc.obs if mm == m && i in frames]
        for m in unique(o[2] for o in sc.obs if o[1] in frames)
)
# A seed: the truth perturbed by `rot` rad and `tr` (length) per block.
function rw_seed(rng, sc; rot = 0.01, tr = 0.01)
    pert(T) = perturb_pose(rng, T; σ_R = rot, σ_t = tr)
    return RobotWorldHandEyeCalibration(pert(sc.Y), Dict(m => pert(sc.X[m]) for m in eachindex(sc.X)))
end
# The chart error (δω, δt) with truth = exp(δω)·R̂, t̂ + δt — the reported chart.
_rwδ(gt, h) = local_coordinates(RigidExpManifold6(h), gt)
# block: :Y, a marker index, or :joint (dof 6(M + 1)).
function rw_trial(block; N = 12, M = 3, f = 600.0, σ = 0.5, rot = 0.01, tr = 0.01, layout = :row)
    return function (rng)
        sc = rw_scene(rng; N, M, f, σ, layout)
        res = refine_robot_world_hand_eye(rw_seed(rng, sc; rot, tr), sc.camera, rw_markers(sc))
        retcode(res) === :converged || return nothing
        cal = estimate(res)
        δs = vcat([_rwδ(sc.Y, cal.Y)], [_rwδ(sc.X[m], cal.X[m]) for m in 1:M])
        block === :joint && return Vector(reduce(vcat, δs)), Matrix(res.cov)
        k = block === :Y ? 0 : block
        rows = (6k + 1):(6k + 6)
        return Vector(δs[k + 1]), Matrix(res.cov)[rows, rows]
    end
end
# Held-out reprojection: a NEW rig pose (not in the fit), one marker point;
# predicted pixel covariance ŝ²·I₂ + G·Σ·Gᵀ with G = ∂û/∂(δY, δX_m) at the
# estimate and Σ the (Y, X_m) sub-block of the reported covariance, χ²₂.
function rw_heldout_trial(; N = 12, M = 3, f = 600.0, σ = 0.5)
    return function (rng)
        sc = rw_scene(rng; N = N + 1, M, f, σ)
        held = filter(o -> o[1] == N + 1, sc.obs)
        isempty(held) && return nothing
        res = refine_robot_world_hand_eye(rw_seed(rng, sc), sc.camera, rw_markers(sc; frames = 1:N))
        retcode(res) === :converged || return nothing
        cal = estimate(res)
        i, m, cs = held[rand(rng, 1:length(held))]
        P, u = cs[rand(rng, 1:length(cs))]
        RA, tA = SMatrix{3, 3, Float64, 9}(rotation(sc.A[i])), SVector{3, Float64}(translation(sc.A[i]))
        g(z) = RVG._rw_point_residual(z, RigidExpManifold6(cal.Y), RigidExpManifold6(cal.X[m]), RA, tA, P, u, sc.camera)
        G = ForwardDiff.jacobian(g, zero(SVector{12, Float64}))
        idx = vcat(1:6, (6m + 1):(6m + 6))
        Σ = SMatrix{12, 12}(res.cov[idx, idx])
        return Vector(g(zero(SVector{12, Float64}))), Matrix(res.s2 * SMatrix{2, 2}(1.0, 0.0, 0.0, 1.0) + G * Σ * G')
    end
end

@testset "refine_robot_world_hand_eye" begin
    @testset "noiseless recovery is exact from a perturbed seed" begin
        rng = StableRNG(1)
        sc = rw_scene(rng; σ = 0.0)
        res = refine_robot_world_hand_eye(rw_seed(rng, sc; rot = 0.02, tr = 0.02), sc.camera, rw_markers(sc))
        @test estimate(res) isa RobotWorldHandEyeCalibration{Int}
        @test retcode(res) === :converged
        @test res.rms < 1.0e-9
        @test norm(_rwδ(sc.Y, estimate(res).Y)) < 1.0e-9
        @test all(norm(_rwδ(sc.X[m], estimate(res).X[m])) < 1.0e-9 for m in 1:3)
        @test estimate(res).ids == [1, 2, 3]
        @test res.dof == 2 * res.n - 24
        @test sum(length, values(res.residuals)) == res.n          # per marker, every point
    end

    @testset "the argument contract" begin
        sc = rw_scene(StableRNG(2); M = 2)
        seed = rw_seed(StableRNG(3), sc)
        mk = rw_markers(sc)
        @test_throws ArgumentError refine_robot_world_hand_eye(seed, sc.camera, Dict(7 => mk[1], 1 => mk[1], 2 => mk[2]))   # unknown marker
        @test_throws ArgumentError refine_robot_world_hand_eye(seed, sc.camera, Dict(1 => mk[1]))                 # key sets must match: marker 2 missing
        tiny = Dict(1 => [first(mk[1][1]) => last(mk[1][1])[1:1]], 2 => [first(mk[2][1]) => last(mk[2][1])[1:1]])
        @test_throws ArgumentError refine_robot_world_hand_eye(seed, sc.camera, tiny)                             # 4 residuals, 18 unknowns
        # A marker with no features is refused, not refined into an Inf covariance.
        @test_throws ArgumentError refine_robot_world_hand_eye(seed, sc.camera, Dict(1 => mk[1], 2 => empty(mk[2])))
        @test_throws ArgumentError refine_robot_world_hand_eye(seed, sc.camera, Dict(1 => mk[1], 2 => [first(e) => empty(last(e)) for e in mk[2]]))
        # Non-finite inputs and points the camera does not image at the guess are refused.
        bad(f) = Dict(1 => [mk[1][1:(end - 1)]; [first(mk[1][end]) => f(last(mk[1][end]))]], 2 => mk[2])
        nan_px = bad(cs -> [first(cs[1]) => SVector(NaN, last(cs[1])[2]); cs[2:end]])
        nan_pt = bad(cs -> [SVector(NaN, 0.0, 0.0) => last(cs[1]); cs[2:end]])
        behind = bad(cs -> [SVector(0.0, 0.0, -10.0) => last(cs[1]); cs[2:end]])   # 10 m behind the marker: behind the camera
        nan_rig = Dict(1 => [mk[1][1:(end - 1)]; [rigid(rotation(first(mk[1][end])), SVector(NaN, 0.0, 0.0)) => last(mk[1][end])]], 2 => mk[2])
        for m in (nan_px, nan_pt, behind, nan_rig)
            @test_throws ArgumentError refine_robot_world_hand_eye(seed, sc.camera, m)
        end
        @test retcode(refine_robot_world_hand_eye(seed, sc.camera, mk)) === :converged        # the untouched input is fine
    end

    @testset "degeneracy is REPORTED (pure-z gauge family)" begin
        # Rig rotations all about the world z axis with zero translation: the
        # gauge (Y, X_m) → (G∘Y, G∘X_m), G = (RotZ(φ), s·ẑ) commutes with every
        # A_i and leaves every reprojection unchanged. JᵀJ is singular, and the
        # fit says so: min_eigval collapses and the covariance is unbounded.
        rng = StableRNG(7)
        camera = CameraModel(; focal = 600.0, pp = (400.0, 300.0))
        Y = rigid(RotMatrix{3}(1.0I), SVector(0.0, 0.0, -1.2))
        X = [rigid(AngleAxis(0.3 * rand(rng), normalize(randn(rng, 3))...), SVector(0.08 * (m - 1.5), 0.03, 0.0)) for m in 1:2]
        A = [rigid(RotMatrix(RotZ(θ)), SVector(0.0, 0.0, 0.0)) for θ in range(0.3, 2.0; length = 8)]
        markers = Dict(m => [A[i] => [P => SVector{2, Float64}(project(camera, marker_pose(Y, A[i], X[m])(P))...) for P in _grid()] for i in 1:8] for m in 1:2)
        rd = refine_robot_world_hand_eye(RobotWorldHandEyeCalibration(Y, Dict(1 => X[1], 2 => X[2])), camera, markers)
        sc = rw_scene(StableRNG(8); M = 2, σ = 0.0)
        rh = refine_robot_world_hand_eye(rw_seed(StableRNG(9), sc; rot = 0.0, tr = 0.0), sc.camera, rw_markers(sc))
        @test abs(rd.min_eigval) < 1.0e-8 * rh.min_eigval
        @test !all(isfinite, rd.cov)
    end

    @testset "Monte Carlo hard gate (VGC harness, z = 4)" begin
        # The reported covariance is s²·implicit_covariance at the estimate —
        # the covariance of the estimator's own stationary point — in the
        # left-perturbation chart, block order [Y | X₁ … X_M]. Measured
        # 2026-09-22: Y 5.92/6, X₁₋₃ 5.91–6.03/6, joint 24.1/24, held-out
        # reprojection 1.96/2; N = 4 24.1, N = 30 24.5, f = 2000 24.0, f = 250
        # 24.2, M = 1 12.0/12, M = 4 30.2/30, a 0.1 rad / 10 cm seed 23.9 —
        # every median ratio in [0.96, 1.04], no failed trial.
        # Every cell asserts nfail == 0: a trial that does not converge (or
        # returns nothing) is COUNTED, never silently dropped from the gate.
        gate(trial, seed, n, d) = (r = run_calibration(trial, StableRNG(seed), n, d); is_calibrated(r) && r.nfail == 0)
        @test gate(rw_trial(:Y), 1101, 300, 6)
        for m in 1:3
            @test gate(rw_trial(m), 1101 + m, 300, 6)
        end
        @test gate(rw_trial(:joint), 1105, 300, 24)
        @test gate(rw_heldout_trial(), 1111, 400, 2)
        @test gate(rw_trial(:joint; N = 4), 1112, 300, 24)
        @test gate(rw_trial(:joint; N = 30), 1113, 200, 24)
        @test gate(rw_trial(:joint; f = 2000.0), 1114, 300, 24)
        @test gate(rw_trial(:joint; f = 250.0), 1115, 300, 24)
        @test gate(rw_trial(:joint; M = 1), 1116, 300, 12)
        @test gate(rw_trial(:joint; M = 4), 1117, 300, 30)
        rP = run_calibration(rw_trial(:joint; rot = 0.1, tr = 0.1), StableRNG(1118), 300, 24)
        @test is_calibrated(rP) && rP.nfail == 0          # the poor seed converges every time
    end

    @testset "many markers: M = $M (no marker cap)" for (M, seed) in ((6, 1201), (10, 1202), (20, 1203))
        # Exact recovery from a POOR seed (0.1 rad / 10 cm per block), and the
        # joint NEES gate (dof 6(M + 1)) from the usual seed and from the poor
        # one, every trial converged; 100 trials (the NEES mean of a χ²_d with
        # d ≥ 42 is then within ~2% of d by its own spread). Measured
        # 2026-10-01, mean NEES / d (usual seed, poor seed): M = 6 0.976,
        # 1.005; M = 10 1.018, 1.003; M = 20 0.989, 0.993 — no failed trial.
        rng = StableRNG(seed)
        sc = rw_scene(rng; M, σ = 0.0, layout = :grid)
        @test length(rw_markers(sc)) == M                 # every marker is seen
        res = refine_robot_world_hand_eye(rw_seed(rng, sc; rot = 0.1, tr = 0.1), sc.camera, rw_markers(sc))
        @test retcode(res) === :converged
        @test res.rms < 1.0e-9
        @test norm(_rwδ(sc.Y, estimate(res).Y)) < 1.0e-9
        @test all(norm(_rwδ(sc.X[m], estimate(res).X[m])) < 1.0e-9 for m in 1:M)
        @test res.dof == 2 * res.n - 6 * (M + 1)
        d = 6 * (M + 1)
        r = run_calibration(rw_trial(:joint; M, layout = :grid), StableRNG(seed + 10), 100, d)
        @test is_calibrated(r) && r.nfail == 0
        rP = run_calibration(rw_trial(:joint; M, layout = :grid, rot = 0.1, tr = 0.1), StableRNG(seed + 20), 100, d)
        @test is_calibrated(rP) && rP.nfail == 0
    end
end
end
