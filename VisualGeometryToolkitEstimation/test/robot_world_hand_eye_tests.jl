@testitem "robot-world fit" setup = [PoseScenes] begin
# ReweightedLeastSquares on a RobotWorldHandEyeCalibration: the staged linear robot-world calibration A_i X_m = Y B_i^m.
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
using VisualGeometryToolkitCore: rotation, translation   # VGC accessors (public)
using VisualGeometryToolkitEstimation: ReweightedLeastSquares, RobotWorldHandEyeCalibration
using VisualGeometryToolkitCore: ClassDefault, EstimationProblem, solve
include(joinpath(pkgdir(VisualGeometryToolkitEstimation), "test", "fixtures", "reweighted_fits.jl"))
import VisualGeometryToolkitCore as VgtkCore
import VisualGeometryToolkitEstimation as VgtkEstimation
using VisualGeometryToolkitCore.Synthetic: run_calibration, is_calibrated   # VGC's MC calibration harness (public)

# The markers of A_i X_m = Y B_i^m (`rwhe_synth`): the suite's fixture file, shared with the golden records
include(joinpath(pkgdir(VgtkEstimation), "test", "fixtures", "robot_world_scenes.jl"))
_nobs(markers) = sum(length, values(markers))


@testset "robot-world fit" begin

    @testset "noiseless recovery is exact" begin
        rng = StableRNG(1)
        markers, Ygt, Xgt = rwhe_synth(rng; N = 12, M = 3)
        est = rwhe_fit(markers)
        @test est.u isa RobotWorldHandEyeCalibration{Int}
        @test est.u.ids == [1, 2, 3]
        @test norm(log_so3(rotation(est.u.Y)' * rotation(Ygt))) < 1.0e-6
        @test norm(translation(est.u.Y) - translation(Ygt)) < 1.0e-6
        for m in 1:3
            @test norm(log_so3(rotation(est.u.X[m])' * rotation(Xgt[m]))) < 1.0e-6
            @test norm(translation(est.u.X[m]) - translation(Xgt[m])) < 1.0e-6
        end
        @test VgtkCore.s2(est) < 1.0e-12
    end

    @testset "noiseless recovery far from the origin (the translations' normalized solve)" begin
        # the rig base 10 km from the world origin: A′ = T∘A, Y′ = T∘Y, B unchanged. The joint
        # translation solve runs about each side's centroid at one Hartley scale, so the
        # offset leaves the recovery at the noiseless test's tolerance.
        rng = StableRNG(1)
        markers, Ygt, Xgt = rwhe_synth(rng; N = 12, M = 3)
        off = SVector(1.0e4, -7.0e3, 2.5e3)
        far = Dict(m => [EuclideanMap(rotation(a), translation(a) + off) => b for (a, b) in cs] for (m, cs) in markers)
        est = rwhe_fit(far)
        @test norm(log_so3(rotation(est.u.Y)' * rotation(Ygt))) < 1.0e-6
        @test norm(translation(est.u.Y) - (translation(Ygt) + off)) < 1.0e-6
        for m in 1:3
            @test norm(translation(est.u.X[m]) - translation(Xgt[m])) < 1.0e-6
        end
    end

    @testset "moderate noise (1 mrad / 1 mm)" begin
        rng = StableRNG(2)
        markers, Ygt, Xgt = rwhe_synth(rng; N = 40, M = 5, σ_R = 1.0e-3, σ_t = 1.0e-3, drop = 0.2)
        est = rwhe_fit(markers)
        @test norm(log_so3(rotation(est.u.Y)' * rotation(Ygt))) < 0.01            # < 0.6°
        @test norm(translation(est.u.Y) - translation(Ygt)) < 0.01
        for m in 1:5
            @test norm(log_so3(rotation(est.u.X[m])' * rotation(Xgt[m]))) < 0.01
        end
        # profiled scales make the variance factor ≈ 1
        @test 0.3 < VgtkCore.s2(est) < 3.0
        @test VgtkCore.dof(est) == 6 * _nobs(markers) - 6 * 6
        @test all(length(est.metadata.residuals[m]) == length(markers[m]) for m in 1:5)   # per marker, input order
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
            markers, Ygt, Xgt = rwhe_synth(rng; N = 15, M = M, σ_R = 2.0e-3, σ_t = 2.0e-3)
            est = rwhe_fit(markers)
            cal = est.u
            δs = vcat([_δ(Ygt, cal.Y)], [_δ(Xgt[m], cal.X[m]) for m in 1:M])
            block === :joint && return Vector(reduce(vcat, δs)), Matrix(est.metadata.cov)
            k = block === :Y ? 0 : block
            rows = (6k + 1):(6k + 6)
            return Vector(δs[k + 1]), Matrix(est.metadata.cov)[rows, rows]
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
            markers, Ygt, Xgt = rwhe_synth(rng; N = 40, M = M, σ_R = 2.0e-3, σ_t = 2.0e-3)
            est = rwhe_fit(markers)
            return Vector(_δ(Ygt, est.u.Y)), Matrix(est.metadata.cov)[1:6, 1:6]
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
        est = rwhe_fit(markers; σ_rot = 1.0, σ_trans = 1.0)
        λmin = est.metadata.min_eigval
        # healthy geometry for comparison
        markersh, _, _ = rwhe_synth(StableRNG(6); N = 12, M = 2)
        esth = rwhe_fit(markersh; σ_rot = 1.0, σ_trans = 1.0)
        @test λmin < 1.0e-8 * esth.metadata.min_eigval    # collapsed vs healthy
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
        Ŷ, X̂ = est.u.Y, est.u.X
        g_yaw = vcat(_g_yaw(translation(Ŷ)), _g_yaw(translation(X̂[1])), _g_yaw(translation(X̂[2])))
        g_z = vcat(_g_z(), _g_z(), _g_z())
        Q = Matrix(qr(hcat(Vector(g_yaw), Vector(g_z))).Q)[:, 1:2]
        v = est.metadata.min_eigvec
        @test norm(v) ≈ 1.0
        @test norm(v - Q * (Q' * v)) < 1.0e-6          # null direction ∈ span(yaw, z-shift)
        # ... and the covariance is unbounded, not a confident finite cov_Y
        # (pinv's cutoff used to zero the gauge directions), as in the reprojection refine.
        @test !all(isfinite, est.metadata.cov) && !all(isfinite, est.metadata.cov_Y)
        @test all(isfinite, esth.metadata.cov)
    end

    @testset "a robust result whose weights leave no degrees of freedom: dof 0, no variance factor, no covariance" begin
        # 6Σw − N ≤ 0 (a redescending loss that zeroes most rows): until 0.11.0 recorded a negative dof and a
        # variance factor RSS/eps ≈ 1e15 that scaled the covariance, silently
        VE = VgtkEstimation
        rng = StableRNG(2)
        Id = EuclideanMap(RotMatrix{3}(1.0I), SVector(0.0, 0.0, 0.0))
        cal = RobotWorldHandEyeCalibration(Id, Dict(1 => Id))
        n = 4; N = 12
        r = randn(rng, 6n); J = randn(rng, 6n, N)
        res = VE._robot_world_hand_eye_result(cal, r, J, 6, [n], s2 -> s2 * Matrix(1.0I, N, N), 0.01, 0.01;
                                              retcode = VgtkCore.ReturnCode.Success, stop_rule = VgtkCore.NoRule, iterations = 3, rms_rows = 6n,
                                              strategy = VgtkCore.Irls{VgtkCore.WelschShape}(), weights = [1.0, 0.5, 0.0, 0.0])
        md = res.metadata
        @test md.dof == 0 && isnan(md.s2) && all(isinf, md.cov)
        # with dof left, the weighted variance factor as before
        ok = VE._robot_world_hand_eye_result(cal, r, J, 6, [n], s2 -> s2 * Matrix(1.0I, N, N), 0.01, 0.01;
                                             retcode = VgtkCore.ReturnCode.Success, stop_rule = VgtkCore.NoRule, iterations = 3, rms_rows = 6n,
                                             strategy = VgtkCore.Irls{VgtkCore.WelschShape}(), weights = [1.0, 1.0, 1.0, 0.5])
        @test VgtkCore.dof(ok) == 9 && isfinite(VgtkCore.s2(ok))
    end

    @testset "argument validation" begin
        rng = StableRNG(7)
        markers, _, _ = rwhe_synth(rng; N = 6, M = 2)
        @test_throws ArgumentError rwhe_fit(Dict(1 => markers[1][1:2], 2 => markers[2][1:1]))   # underdetermined
        @test_throws ArgumentError rwhe_fit(Dict(1 => markers[1], 2 => empty(markers[2])))      # a marker with no observations
        # Marker ids are the caller's own keys (here Strings), X is keyed by them,
        # and the covariance block order is the sorted keys.
        named = Dict("b" => markers[1], "a" => markers[2])
        estn = rwhe_fit(named)
        @test estn.u.ids == ["a", "b"]
        Xb, X1 = estn.u.X["b"], (rwhe_fit(markers)).u.X[1]
        # matrices, not the angle: acos resolves no better than ~3e-8 rad near zero
        @test norm(Matrix(rotation(Xb)) - Matrix(rotation(X1))) < 1.0e-12 && norm(translation(Xb) - translation(X1)) < 1.0e-12
    end
end
end
