@testitem "hand_eye" setup = [PoseScenes] begin
    using VisualGeometryToolkitCore: solve, EstimationProblem, successful_retcode
    using VisualGeometryToolkitEstimation: Ransac
    using VisualGeometryToolkitCore: inlier_mask
    import VisualGeometryToolkitEstimation as VgtkEstimation
using Test
using LinearAlgebra: norm, I
using Random: Random
using StableRNGs
using StaticArrays: SVector
using Rotations: RotMatrix, RotationVec
using VisualGeometryToolkitCore: EuclideanMap, log_so3, Vec3
# Unlabeled rigid motions (the hand-eye currency; the chains `marker_pose(Ygt, A[i], Xgt)` below need frame-free
# composition).
using VisualGeometryToolkitCore: rotation, translation
using VisualGeometryToolkitEstimation: HandEyeProblem, AllMotionPairs

# Synthesize an AX=YB dataset: a robot-world scene (Y*, one X*, N rig poses
# A_i), its observations B_i = Y⁻¹ A_i X, then return (A, B). Optionally
# noise + outliers.
function _synth_dataset(rng; N = 60, σ_R = 0.001, σ_t = 0.001, n_outliers = 0)
    sc = robot_world_hand_eye_scene(rng; N, Y = (ang_max = π, t_scale = 2.0), X = (ang_max = π, t_scale = 0.1), A = (ang_max = π / 3, t_scale = 0.5))
    Xgt, Ygt, A = only(sc.X), sc.Y, sc.A
    B = [marker_pose(Ygt, A[i], Xgt) for i in 1:N]

    A_obs = [perturb_pose(rng, A[i]; σ_R, σ_t) for i in 1:N]
    B_obs = [perturb_pose(rng, B[i]; σ_R, σ_t) for i in 1:N]

    # Replace n_outliers frames with completely random B poses
    outlier_indices = Random.randperm(rng, N)[1:n_outliers]
    for k in outlier_indices
        B_obs[k] = rand_pose(rng; ang_max = π, t_scale = 5.0)
    end

    return (A_obs, B_obs, Xgt, Ygt, outlier_indices)
end

@testset "Hand-Eye Calibration (AX = XB)" begin
    rng = StableRNG(20260520)

    @testset "noiseless recovery" begin
        A, B, Xgt, _, _ = _synth_dataset(rng; N = 30, σ_R = 0.0, σ_t = 0.0, n_outliers = 0)
        problem = HandEyeProblem(A .=> B)
        @test problem isa HandEyeProblem
        @test VgtkEstimation.data_size(problem) == 29   # N-1 consecutive

        result = solve(EstimationProblem(HandEyeProblem(A .=> B)), Ransac(); rng = StableRNG(247))
        X_est = result.u
        @test norm(log_so3(rotation(X_est)' * rotation(Xgt))) < 1.0e-4
        @test norm(translation(X_est) - translation(Xgt)) < 1.0e-4
    end

    @testset "moderate noise (σ_R = 1 mrad, σ_t = 1 mm)" begin
        A, B, Xgt, _, _ = _synth_dataset(rng; N = 80, σ_R = 1.0e-3, σ_t = 1.0e-3, n_outliers = 0)
        result = solve(EstimationProblem(HandEyeProblem(A .=> B)), Ransac(); rng = StableRNG(248))
        X_est = result.u
        @test norm(log_so3(rotation(X_est)' * rotation(Xgt))) < 0.01            # < 0.6°
        @test norm(translation(X_est) - translation(Xgt)) < 0.01            # < 1 cm
    end

    @testset "robust to gross outliers" begin
        A, B, Xgt, _, outlier_idx =
            _synth_dataset(rng; N = 120, σ_R = 1.0e-3, σ_t = 1.0e-3, n_outliers = 15)
        result = solve(EstimationProblem(HandEyeProblem(A .=> B)), Ransac(); rng = StableRNG(249))
        X_est = result.u
        # Should still recover X* despite ~12% outlier rate.
        @test norm(log_so3(rotation(X_est)' * rotation(Xgt))) < 0.05            # < 3°
        @test norm(translation(X_est) - translation(Xgt)) < 0.05
    end

    @testset "all-pairs motion mode" begin
        A, B, Xgt, _, _ = _synth_dataset(rng; N = 20, σ_R = 1.0e-3, σ_t = 1.0e-3, n_outliers = 0)
        result = solve(EstimationProblem(HandEyeProblem(A .=> B; motion_pairs = AllMotionPairs())), Ransac(); rng = StableRNG(250))
        X_est = result.u
        @test norm(log_so3(rotation(X_est)' * rotation(Xgt))) < 0.02
        @test norm(translation(X_est) - translation(Xgt)) < 0.02
        # the frame pairs are types, a TOML value their tag (the registry): the default is the file's, consecutive
        using VisualGeometryToolkitEstimation: MOTION_PAIRS_TAGS, ConsecutiveMotionPairs
        @test MOTION_PAIRS_TAGS["consecutive"] === ConsecutiveMotionPairs && MOTION_PAIRS_TAGS["all"] === AllMotionPairs
        @test_throws ArgumentError MOTION_PAIRS_TAGS["other"]
        n_consecutive = length(HandEyeProblem(A .=> B; motion_pairs = ConsecutiveMotionPairs()).motions)
        @test length(HandEyeProblem(A .=> B).motions) == n_consecutive <= 19   # N − 1 pairs, less those below min_rel_angle
        @test n_consecutive < length(HandEyeProblem(A .=> B; motion_pairs = AllMotionPairs()).motions) <= 190   # of N(N − 1)/2
    end

    @testset "argument validation" begin
        A = [rand_pose(rng) for _ in 1:5]
        @test_throws ArgumentError HandEyeProblem(A[1:2] .=> A[1:2])   # too few frames
    end

    @testset "construction and the sample gate share min_rel_angle" begin
        # Every consecutive motion rotates 3°: below the default gate, no
        # motion succeeds at the constructor's rotation gate, so the problem is refused at
        # construction (it used to filter at a hard-coded 1°, build, and burn
        # every RANSAC trial to :no_model). A gate lowered by keyword admits it.
        r7 = StableRNG(7)
        sc = robot_world_hand_eye_scene(r7; N = 1, Y = (ang_max = π, t_scale = 2.0), X = (ang_max = π, t_scale = 0.1), A = (ang_max = π, t_scale = 0.5))
        Xgt, Ygt, A = only(sc.X), sc.Y, sc.A
        for _ in 2:30
            ax = randn(r7, 3); ax ./= norm(ax)
            Δ = EuclideanMap(RotMatrix{3, Float64}(RotMatrix(RotationVec((deg2rad(3) .* ax)...))), Vec3{Float64}((0.2 .* randn(r7, 3))...))
            push!(A, A[end] ∘ Δ)
        end
        B = [marker_pose(Ygt, a, Xgt) for a in A]
        @test_throws ArgumentError HandEyeProblem(A .=> B)
        @test_throws ArgumentError solve(EstimationProblem(HandEyeProblem(A .=> B)), Ransac())
        p2 = HandEyeProblem(A .=> B; min_rel_angle = deg2rad(2))
        @test VgtkEstimation.data_size(p2) == 29
        # The inner constructor is the gate: raw motions below the angle throw.
        @test_throws ArgumentError HandEyeProblem{typeof(p2._sampler)}(p2.motions, p2.char_length, deg2rad(5), p2._sampler)
        r = solve(EstimationProblem(HandEyeProblem(A .=> B; min_rel_angle = deg2rad(2))), Ransac(); rng = StableRNG(251))
        @test successful_retcode(r)
        @test norm(log_so3(rotation(r.u)' * rotation(Xgt))) < 1.0e-4
        @test norm(translation(r.u) - translation(Xgt)) < 1.0e-3
    end

    @testset "default threshold is unit-free (metres vs millimetres)" begin
        # The file's value is a fraction of the characteristic length L, so the
        # same scene in m and in mm scores identically: same inliers, the
        # estimate scaled by 1000. (As an absolute value it kept 3 of 79
        # motions in mm.)
        A, B, Xgt, _, _ = _synth_dataset(StableRNG(252); N = 80, σ_R = 1.0e-3, σ_t = 1.0e-3, n_outliers = 8)
        mm(p) = EuclideanMap(RotMatrix{3, Float64}(rotation(p)), Vec3{Float64}((1000 .* translation(p))...))
        r_m = solve(EstimationProblem(HandEyeProblem(A .=> B)), Ransac(); rng = StableRNG(253))
        r_mm = solve(EstimationProblem(HandEyeProblem(mm.(A) .=> mm.(B))), Ransac(); rng = StableRNG(253))
        @test sum(inlier_mask(r_m)) >= 50
        @test inlier_mask(r_m) == inlier_mask(r_mm)
        @test norm(log_so3(rotation(r_mm.u)' * rotation(r_m.u))) < 1.0e-6
        @test norm(translation(r_mm.u) ./ 1000 - translation(r_m.u)) < 1.0e-6
    end
end
end
