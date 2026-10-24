@testitem "robot_world_hand_eye" setup = [PoseScenes] begin
# fit_robot_world_hand_eye: staged linear robot-world calibration A_i X_m = Y B_i^m.
# Ladder per house conventions + two rungs the older hand-eye tests lack:
# Monte-Carlo covariance calibration, and a degeneracy REGRESSION — the
# 2026-05 dataset pathology (unobservable direction invisible to residuals)
# must surface in the reported eigen-diagnostics, not vanish.

using Test
using LinearAlgebra: norm, I, eigen, Symmetric, dot, cross, qr
using Random: Random
using StableRNGs
using StaticArrays
using Rotations: Rotations, RotMatrix, RotationVec, RotZ
using VisualGeometryToolkitCore: EuclideanMap, log_so3
using VisualGeometryToolkitCore: RigidExpManifold6, local_coordinates   # the decoupled pose chart (public)
using VisualGeometryToolkitCore: estimate, rotation, translation   # VGC accessors (public)
using VisualGeometryToolkitEstimation: fit_robot_world_hand_eye, RobotWorldHandEyeCalibration
using VisualGeometryToolkitCore.Synthetic: run_calibration, is_calibrated   # VGC's MC calibration harness (public)

# A_i X_m = Y B_i^m  ⇒  B = Y⁻¹ ∘ A ∘ X. `markers[m]` holds marker m's
# correspondences A_i => B_i^m, one per frame it is seen in (frame order).
function _synth(rng; N = 25, M = 4, σ_R = 0.0, σ_t = 0.0, drop = 0.0)
    (; Y, X, A) = robot_world_hand_eye_scene(rng; N, M, Y = (t_scale = 0.5,), X = (ang_max = 0.5, t_scale = 0.3), A = (t_scale = 1.0,))
    markers = Dict(m => Pair{typeof(Y), typeof(Y)}[] for m in 1:M)
    for i in 1:N, m in 1:M
        rand(rng) < drop && continue
        B = marker_pose(Y, A[i], X[m])
        push!(markers[m], A[i] => (σ_R == 0 && σ_t == 0 ? B : perturb_pose(rng, B; σ_R, σ_t)))
    end
    return markers, Y, X
end
_nobs(markers) = sum(length, values(markers))


@testset "fit_robot_world_hand_eye" begin

    @testset "noiseless recovery is exact" begin
        rng = StableRNG(1)
        markers, Ygt, Xgt = _synth(rng; N = 12, M = 3)
        est = fit_robot_world_hand_eye(markers)
        @test estimate(est) isa RobotWorldHandEyeCalibration{Int}
        @test estimate(est).ids == [1, 2, 3]
        @test norm(log_so3(rotation(estimate(est).Y)' * rotation(Ygt))) < 1.0e-6
        @test norm(translation(estimate(est).Y) - translation(Ygt)) < 1.0e-6
        for m in 1:3
            @test norm(log_so3(rotation(estimate(est).X[m])' * rotation(Xgt[m]))) < 1.0e-6
            @test norm(translation(estimate(est).X[m]) - translation(Xgt[m])) < 1.0e-6
        end
        @test est.s2 < 1.0e-12
    end

    @testset "moderate noise (1 mrad / 1 mm)" begin
        rng = StableRNG(2)
        markers, Ygt, Xgt = _synth(rng; N = 40, M = 5, σ_R = 1.0e-3, σ_t = 1.0e-3, drop = 0.2)
        est = fit_robot_world_hand_eye(markers)
        @test norm(log_so3(rotation(estimate(est).Y)' * rotation(Ygt))) < 0.01            # < 0.6°
        @test norm(translation(estimate(est).Y) - translation(Ygt)) < 0.01
        for m in 1:5
            @test norm(log_so3(rotation(estimate(est).X[m])' * rotation(Xgt[m]))) < 0.01
        end
        # profiled scales make the variance factor ≈ 1
        @test 0.3 < est.s2 < 3.0
        @test est.dof == 6 * _nobs(markers) - 6 * 6
        @test all(length(est.residuals[m]) == length(markers[m]) for m in 1:5)   # per marker, input order
    end

    @testset "covariance calibration (Monte Carlo, harness gates: Y, each X, joint)" begin
        # VGC's Monte Carlo harness (`run_calibration`, z = 4 gates on the NEES
        # mean and the 50/90/95 % coverages). The reported covariance is the
        # first-order covariance of the staged LINEAR estimator itself (the delta
        # method through its stage conditions), in the left-perturbation chart
        # (δω = Log(R_true·R̂ᵀ) in rad, δt = t_true − t̂), block order [Y | X₁ … X_M].
        _δ(gt, h) = local_coordinates(RigidExpManifold6(h), gt)   # (log(R_gt·R̂ᵀ), t_gt − t̂): the reported chart
        M = 3
        trial(block) = function (rng)
            markers, Ygt, Xgt = _synth(rng; N = 15, M = M, σ_R = 2.0e-3, σ_t = 2.0e-3)
            est = fit_robot_world_hand_eye(markers)
            cal = estimate(est)
            δs = vcat([_δ(Ygt, cal.Y)], [_δ(Xgt[m], cal.X[m]) for m in 1:M])
            block === :joint && return Vector(reduce(vcat, δs)), Matrix(est.cov)
            k = block === :Y ? 0 : block
            rows = (6k + 1):(6k + 6)
            return Vector(δs[k + 1]), Matrix(est.cov)[rows, rows]
        end
        # Every block and the joint vector are calibrated. Until 2026-09-30 the
        # reported covariance was the Gauss–Newton ŝ²(JᵀJ)⁺ of the pose
        # residual — the covariance of the residual's stationary point, which
        # the staged solve is not — and Y scattered wider than it claimed
        # (median NEES ratio 1.26, 95 % coverage 0.85; 1.48 at N = 40), pinned
        # here as a known miss.
        rY = run_calibration(trial(:Y), StableRNG(4), 400, 6)
        @test is_calibrated(rY) && rY.nfail == 0
        for m in 1:M
            rX = run_calibration(trial(m), StableRNG(40 + m), 400, 6)
            @test is_calibrated(rX) && rX.nfail == 0
        end
        rJ = run_calibration(trial(:joint), StableRNG(44), 400, 6 * (M + 1))
        @test is_calibrated(rJ) && rJ.nfail == 0
        # More poses: the regime where the Gauss–Newton miss grew (1.48 at N = 40).
        trial40 = function (rng)
            markers, Ygt, Xgt = _synth(rng; N = 40, M = M, σ_R = 2.0e-3, σ_t = 2.0e-3)
            est = fit_robot_world_hand_eye(markers)
            return Vector(_δ(Ygt, estimate(est).Y)), Matrix(est.cov)[1:6, 1:6]
        end
        @test is_calibrated(run_calibration(trial40, StableRNG(45), 300, 6))
    end

    @testset "degeneracy is REPORTED (pure-z gauge family)" begin
        rng = StableRNG(5)
        # Rig rotations all about z with zero translation ⇒ the centralizer of
        # {A_i} is the TWO-parameter gauge family (Y, X_m) → (G∘Y, G∘X_m) with
        # G = (RotZ(φ), s·ẑ): a common yaw AND a common shift along z — the
        # "yaw ⊕ t_Y" pathology of 2026-05. JᵀJ must be singular and the fit
        # must say so via min_eigval.
        Y = rand_pose(rng; t_scale = 0.5)
        X = [rand_pose(rng; ang_max = 0.3, t_scale = 0.3) for _ in 1:2]
        A = [
            EuclideanMap(RotMatrix(RotZ(θ)), SVector(0.0, 0.0, 0.0); check = false)
                for θ in range(0.3, 2.0; length = 8)
        ]
        markers = Dict(m => [A[i] => marker_pose(Y, A[i], X[m]) for i in 1:8] for m in 1:2)
        est = fit_robot_world_hand_eye(markers; σ_rot = 1.0, σ_trans = 1.0)
        λmin = est.min_eigval
        # healthy geometry for comparison
        markersh, _, _ = _synth(StableRNG(6); N = 12, M = 2)
        esth = fit_robot_world_hand_eye(markersh; σ_rot = 1.0, σ_trans = 1.0)
        @test λmin < 1.0e-8 * esth.min_eigval    # collapsed vs healthy
        # The null space is 2-D, so `min_eigvec` is an ARBITRARY unit vector in
        # it (its yaw/z mix is decided by FP round-off in `eigen`; an earlier
        # `norm(v[1:3]) > 0.1` check was flaky for exactly that reason — it saw
        # 0.35 and 0.08 across otherwise identical runs). The invariant claim is
        # that the reported direction lies IN the analytic gauge subspace. In
        # the reported chart (exp(δω)·R, t + δt — the one `cov` is in) at δ = 0:
        #   yaw generator per block: δ_rot = (0, 0, 1), δ_t = ẑ × t₀
        #   z-shift generator per block: δ_rot = 0,     δ_t = ẑ
        ẑ = SVector(0.0, 0.0, 1.0)
        _g_yaw(t) = vcat(SVector(0.0, 0.0, 1.0), cross(ẑ, SVector{3, Float64}(t)))
        _g_z() = vcat(SVector(0.0, 0.0, 0.0), ẑ)
        Ŷ, X̂ = estimate(est).Y, estimate(est).X
        g_yaw = vcat(_g_yaw(translation(Ŷ)), _g_yaw(translation(X̂[1])), _g_yaw(translation(X̂[2])))
        g_z = vcat(_g_z(), _g_z(), _g_z())
        Q = Matrix(qr(hcat(Vector(g_yaw), Vector(g_z))).Q)[:, 1:2]
        v = est.min_eigvec
        @test norm(v) ≈ 1.0
        @test norm(v - Q * (Q' * v)) < 1.0e-6          # null direction ∈ span(yaw, z-shift)
        # ... and the covariance is unbounded, not a confident finite cov_Y
        # (pinv's cutoff used to zero the gauge directions), as in refine_robot_world_hand_eye.
        @test !all(isfinite, est.cov) && !all(isfinite, est.cov_Y)
        @test all(isfinite, esth.cov)
    end

    @testset "argument validation" begin
        rng = StableRNG(7)
        markers, _, _ = _synth(rng; N = 6, M = 2)
        @test_throws ArgumentError fit_robot_world_hand_eye(Dict(1 => markers[1][1:2], 2 => markers[2][1:1]))   # underdetermined
        @test_throws ArgumentError fit_robot_world_hand_eye(Dict(1 => markers[1], 2 => empty(markers[2])))      # a marker with no observations
        # Marker ids are the caller's own keys (here Strings), X is keyed by them,
        # and the covariance block order is the sorted keys.
        named = Dict("b" => markers[1], "a" => markers[2])
        estn = fit_robot_world_hand_eye(named)
        @test estimate(estn).ids == ["a", "b"]
        Xb, X1 = estimate(estn).X["b"], estimate(fit_robot_world_hand_eye(markers)).X[1]
        # matrices, not the angle: acos resolves no better than ~3e-8 rad near zero
        @test norm(Matrix(rotation(Xb)) - Matrix(rotation(X1))) < 1.0e-12 && norm(translation(Xb) - translation(X1)) < 1.0e-12
    end
end
end
