# refine_camera_and_poses — a camera and one pose per view from 3D → 2D
# correspondences.
#
# Correctness: from exact pixels, a perturbed camera (intrinsics and lens) and
# perturbed poses are recovered to the LM's tolerance — a pinhole camera, and a
# Kannala–Brandt fisheye whose rays reach past 90°. The Jacobian the stall
# postcondition reads is in the charts' tangent coordinates: its pose block is
# the projection's Jacobian times the left-perturbation action
# (`Jπ·[−[X_c]×  I₃]`), its camera block the retraction's central difference.
# The held camera, the dict and vector forms, lengths, and the refusals.
#
# A SplineLens camera (its K a frozen gauge, its parameters the backward controls
# of its present nodes): from exact pixels of a KB-derived spline camera, its
# perturbed controls and perturbed poses are refined, K bit-identical, the lens
# axis kept on ẑ by the rotation gauge, the block Jacobian (implicit
# differentiation of the projection solve) against finite differences of the
# residual, the poses alone, a forward spline refitted, a masked lattice.

@testsetup module CameraPoseScenes
    using StaticArrays: SVector
    using LinearAlgebra: normalize
    using Rotations: RotMatrix, RotXYZ
    using VisualGeometryToolkitCore: EuclideanMap, exp_so3, project, CameraModel, Pinhole, KannalaBrandt, parameter_vector
    export views_of, truth_poses, perturbed_pose, pinhole_camera, fisheye_camera

    pinhole_camera() = CameraModel(; focal = 800.0, pp = (640.0, 480.0), lens = Pinhole())
    fisheye_camera() = CameraModel(; focal = 420.0, pp = (640.0, 480.0), lens = KannalaBrandt(-0.02, 0.004, -0.001, 0.0002))
    truth_poses(V) = [EuclideanMap(RotMatrix(RotXYZ(0.3 - 0.1v, -0.2 + 0.07v, 0.1v)), SVector(0.1 - 0.05v, -0.05, 2.0 + 0.1v)) for v in 1:V]
    # the truth moved by ≈ 0.05 rad and ≈ 0.08 in translation, differently per view
    perturbed_pose(E, v) = EuclideanMap(
        RotMatrix{3, Float64}(exp_so3(SVector(0.03, -0.02v / 4, 0.03)) * E.R), E.t + SVector(0.05, -0.04, 0.06 - 0.01v)
    )
    # per view, scene points whose camera-frame directions spiral out to `θmax` off the axis, and their exact pixels
    function views_of(camera, poses, θmax; n = 60)
        map(enumerate(poses)) do (v, E)
            map(1:n) do k
                θ = θmax * (k / n); φ = 2.399963 * k + v
                d = SVector(sin(θ) * cos(φ), sin(θ) * sin(φ), cos(θ))
                Xc = (2.0 + 1.5 * ((7k + v) % 10) / 10) * d
                X = E.R' * (Xc - E.t)
                X => SVector{2, Float64}(project(camera, Xc))
            end
        end
    end
end

@testitem "refine_camera_and_poses: a pinhole camera and its poses recovered" setup = [CameraPoseScenes] begin
    using StaticArrays: SVector
    using VisualGeometryToolkitCore: CameraModel, Pinhole, parameter_vector, parameter_names, with_parameters, estimate, retcode, rms
    using VisualGeometryToolkitEstimation: refine_camera_and_poses
    cam = pinhole_camera(); Es = truth_poses(5)
    views = views_of(cam, Es, 0.6)
    θ = parameter_vector(cam)
    cam0 = with_parameters(cam, θ .* SVector(1.04, 1.01, 1.0, 1.0, 1.0) .+ SVector(0.0, 0.0, 0.0, 6.0, -5.0))   # f, aspect, skew, cx, cy
    res = refine_camera_and_poses(cam0, [perturbed_pose(E, v) for (v, E) in enumerate(Es)], views)
    @test retcode(res) === :converged
    est = estimate(res)
    @test maximum(abs.(parameter_vector(est.camera) - θ) ./ max.(abs.(θ), 1.0)) < 1.0e-9
    @test maximum(maximum(abs, P.R - E.R) for (P, E) in zip(est.poses, Es)) < 1.0e-9
    @test maximum(maximum(abs, P.t - E.t) for (P, E) in zip(est.poses, Es)) < 1.0e-9
    @test rms(res) < 1.0e-6 && length(res.residuals) == 2 * 5 * 60 && res.free == (:f, :aspect, :cx, :cy) && res.ν == 600 - 4 - 30
    @test parameter_vector(est.camera)[3] == 0.0                                       # the skew is held by default
end

@testitem "refine_camera_and_poses: a fisheye past 90°, camera and poses recovered" setup = [CameraPoseScenes] begin
    using StaticArrays: SVector
    using LinearAlgebra: norm
    using VisualGeometryToolkitCore: parameter_vector, with_parameters, estimate, retcode, rms, KannalaBrandt, lens
    using VisualGeometryToolkitEstimation: refine_camera_and_poses
    cam = fisheye_camera(); Es = truth_poses(4)
    views = views_of(cam, Es, deg2rad(100))                                      # rays up to 100° off the axis
    θ = parameter_vector(cam)
    δ = SVector(1.03, 1.0, 1.0, 1.0, 1.0, 0.8, 1.2, 0.9, 1.1)                    # f, aspect, skew, cx, cy, k1 … k4
    cam0 = with_parameters(cam, θ .* δ .+ SVector(0.0, 0.0, 0.0, -4.0, 3.0, 0.0, 0.0, 0.0, 0.0))
    res = refine_camera_and_poses(cam0, [perturbed_pose(E, v) for (v, E) in enumerate(Es)], views)
    @test retcode(res) === :converged
    est = estimate(res)
    @test lens(est.camera) isa KannalaBrandt                                    # the family is kept
    @test maximum(abs.(parameter_vector(est.camera) - θ) ./ max.(abs.(θ), 1.0e-2)) < 1.0e-7
    @test maximum(maximum(abs, P.R - E.R) for (P, E) in zip(est.poses, Es)) < 1.0e-9
    @test maximum(norm(P.t - E.t) for (P, E) in zip(est.poses, Es)) < 1.0e-9
    @test rms(res) < 1.0e-6
end

@testitem "refine_camera_and_poses: the Jacobian is in the charts' tangent coordinates" setup = [CameraPoseScenes] begin
    using StaticArrays: SVector, SMatrix
    using LinearAlgebra: norm
    import ForwardDiff
    using VisualGeometryToolkitCore: LeftRigidManifold6, ParameterManifold, restrict, retract, project, action_jacobian, reprojection_error, model
    using VisualGeometryToolkitEstimation: VisualGeometryToolkitEstimation as VE
    cam = fisheye_camera(); Es = truth_poses(2)
    views = views_of(cam, Es, deg2rad(95); n = 8)
    poses0 = [perturbed_pose(E, v) for (v, E) in enumerate(Es)]
    free = (:f, :aspect, :cx, :cy, :k1, :k2, :k3, :k4)
    chc = restrict(ParameterManifold(cam), free)
    chs = [LeftRigidManifold6(P) for P in poses0]
    world = model(LeftRigidManifold6(SMatrix{3, 3, Float64}(1.0, 0, 0, 0, 1.0, 0, 0, 0, 1.0), SVector(0.0, 0.0, 0.0)))   # the held identity board
    costs = [VE._BoardViewResidual{2, (true, true, false), typeof(cam)}(SVector(1, 1 + v), cam, world, world, SVector{3, Float64}(first(c)), last(c))
             for v in 1:2 for c in views[v]]
    using VisualGeometryToolkitCore: nlls_rows_and_jacobian
    r, J = nlls_rows_and_jacobian(costs, [chc; chs])                 # Core's assembler of the stall postcondition's rows
    @test size(J) == (32, 8 + 12)
    for (k, c) in enumerate(costs)
        v = c.varind[2] - 1
        P = model(chs[v]); Xc = P.R * c.X + P.t
        @test r[(2k - 1):(2k)] ≈ reprojection_error(cam, P, c.X, c.u)
        # the pose block: Jπ · [−[X_c]× I₃], the left perturbation's action (zero outside the view's columns)
        Jπ = ForwardDiff.jacobian(x -> project(cam, x), Xc)
        @test J[(2k - 1):(2k), (8 + 6v - 5):(8 + 6v)] ≈ Jπ * action_jacobian(LeftRigidManifold6(P), c.X) rtol = 1.0e-10
        @test iszero(J[(2k - 1):(2k), (8 + 6(3 - v) - 5):(8 + 6(3 - v))])
        # the camera block: the central difference of the retraction
        h = 1.0e-6
        fd = reduce(hcat, [begin
            e = SVector{8}(ntuple(i -> i == j ? h : 0.0, 8))
            (reprojection_error(retract(chc, e), P, c.X, c.u) - reprojection_error(retract(chc, -e), P, c.X, c.u)) / 2h
        end for j in 1:8])
        @test J[(2k - 1):(2k), 1:8] ≈ fd rtol = 1.0e-6
    end
end

@testitem "refine_camera_and_poses: a held camera, dict and vector forms, lengths, refusals" setup = [CameraPoseScenes] begin
    using StaticArrays: SVector
    using VisualGeometryToolkitCore: parameter_vector, with_parameters, estimate, retcode, EuclideanMap, m, ustrip, with_unit, Csponds
    using VisualGeometryToolkitEstimation: refine_camera_and_poses
    cam = pinhole_camera(); Es = truth_poses(3)
    views = views_of(cam, Es, 0.5)
    poses0 = [perturbed_pose(E, v) for (v, E) in enumerate(Es)]
    # the camera held: the poses alone (the camera comes back as given)
    held = refine_camera_and_poses(cam, poses0, views; free = ())
    @test retcode(held) === :converged && estimate(held).camera === cam && held.free == ()
    @test maximum(maximum(abs, P.t - E.t) for (P, E) in zip(estimate(held).poses, Es)) < 1.0e-9
    # a dict keyed by view: the same estimate, keyed back; a Csponds view is one too
    cam0 = with_parameters(cam, parameter_vector(cam) .* SVector(1.02, 1.0, 1.0, 1.0, 1.0))
    rv = refine_camera_and_poses(cam0, poses0, views)
    rd = refine_camera_and_poses(cam0, Dict(:c => poses0[3], :a => poses0[1], :b => poses0[2]),
                                 Dict(:a => views[1], :b => views[2], :c => Csponds(first.(views[3]), last.(views[3]))))
    @test parameter_vector(estimate(rd).camera) ≈ parameter_vector(estimate(rv).camera) rtol = 1.0e-12
    @test estimate(rd).poses[:b].t ≈ estimate(rv).poses[2].t rtol = 1.0e-12
    # lengths in, lengths out (metres), the same refinement
    rp = refine_camera_and_poses(cam0, [with_unit(P, m) for P in poses0], [[X .* m => u for (X, u) in cs] for cs in views])
    @test ustrip(m, estimate(rp).poses[1]).t ≈ estimate(rv).poses[1].t rtol = 1.0e-12
    # refusals
    @test_throws ArgumentError refine_camera_and_poses(cam0, Dict(:a => poses0[1]), Dict(:b => views[1]))
    @test_throws ArgumentError refine_camera_and_poses(cam0, poses0[1:2], views)
    @test_throws ArgumentError refine_camera_and_poses(cam0, poses0, views; free = (:f, :k1))
    @test_throws ArgumentError refine_camera_and_poses(cam0, [EuclideanMap(Es[1].R, SVector(0.0, 0.0, -50.0)); poses0[2:3]], views)   # behind the camera
    @test_throws ArgumentError refine_camera_and_poses(cam0, poses0[1:1], [views[1][1:2]])   # underdetermined
    @test_throws ArgumentError refine_camera_and_poses(cam0, poses0, views; free = (), box = (; f = (1.0, 2.0)))   # a box with nothing to bound
    # the wall-clock cap reaches the LM: a zero cap stops after the first iteration, :max_time; Inf is no cap
    rt = refine_camera_and_poses(cam0, poses0, views; max_time = 0.0)
    @test retcode(rt) === :max_time && rt.iterations <= 1
    @test estimate(refine_camera_and_poses(cam0, poses0, views; max_time = Inf)).camera == estimate(rv).camera
end

@testsetup module SplineCameraPoseScenes
    using StaticArrays: SVector
    using LinearAlgebra: normalize, norm
    using VisualGeometryToolkitCore: CameraModel, SplineLens, ChartSpline, KannalaBrandt, Size2, project, reprojection_error, backproject, angle_between
    using Test: @test, @test_throws
    using Random: Xoshiro, randn
    using VisualGeometryToolkitCore: estimate, retcode, rms
    using VisualGeometryToolkitEstimation: refine_camera_and_poses, optical_axis
    using VisualGeometryToolkitCore: axis_gauged, calibration_matrix, metadata, rotation, BoundedCamera, Size2, model
    using Rotations: RotMatrix
    export spline_truth, perturbed_controls, spline_views, spline_rms, ray_error, check_gcv_choice, entry_gauge, truth_in_frame, bounded, model

    # a SplineLens camera is refined as a bounded camera (2026-11-11, owner: its optical axis is fitted over the camera's
    # own bound, never a default region): the fixture's whole 1280 × 960 sensor
    bounded(c) = BoundedCamera(c, Size2(1280, 960))

    # the camera and the rotation the refine's entry gives a guess: its optical axis (`optical_axis`), then Core's
    # `axis_gauged` — the frame and the K the refine then holds exactly
    function entry_gauge(cam)
        b = cam isa BoundedCamera ? cam : bounded(cam)
        r = optical_axis(b)
        K = calibration_matrix(estimate(r))
        R = rotation(metadata(r).rotation)
        return axis_gauged(model(b), R, K), R
    end
    # the truth and its poses in an estimate's frame (2026-11-11: the refine holds its guess's axis-gauged frame, no
    # longer the truth's canonical one): turned by the rotation Q taking the truth's first pose onto the estimate's — the
    # view whose rotation the joint solve holds; a turn of the frame changes no image
    function truth_in_frame(truth, Es, est)
        Q = RotMatrix{3, Float64}(est.poses[1].R * Es[1].R')
        K = calibration_matrix(model(est.camera))
        return axis_gauged(truth, Q, K), [axis_gauged(E, Q) for E in Es]
    end

    # the truth: a Kannala–Brandt fisheye's SplineLens camera (unit controls, 12 × 10 nodes over 1280 × 960)
    spline_truth() = CameraModel(SplineLens, CameraModel(; focal = 420.0, pp = (640.0, 480.0), lens = KannalaBrandt(-0.02, 0.004, -0.001, 0.0002)),
                                 Size2(1280, 960); grid_size = (12, 10))
    # every present control turned by ≈ `ϵ` rad off its own direction, differently per node (its norm kept)
    function perturbed_controls(cam, ϵ)
        b = cam.lens.backward
        C = map(CartesianIndices(b.controls)) do i
            c = b.controls[i]; k = i[1] + 13 * i[2]
            isfinite(c[1]) ? norm(c) * normalize(c / norm(c) + ϵ * SVector(sin(1.3k), cos(0.7k), sin(2.1k + 0.5))) : c
        end
        return CameraModel(cam.intrinsics, SplineLens(ChartSpline(b.area, C); present = cam.lens.present))
    end
    # per view, scene points spiralling out to `θmax` off the axis, and their exact pixels through `camera`
    function spline_views(camera, poses, θmax; n = 120)
        map(enumerate(poses)) do (v, E)
            map(1:n) do k
                θ = θmax * sqrt(k / n); φ = 2.399963 * k + v
                d = SVector(sin(θ) * cos(φ), sin(θ) * sin(φ), cos(θ))
                Xc = (2.0 + 1.5 * ((7k + v) % 10) / 10) * d
                X = E.R' * (Xc - E.t)
                X => SVector{2, Float64}(project(camera, Xc))
            end
        end
    end
    # the largest angle between two cameras' rays at the detections
    ray_error(a, b, views) = maximum(angle_between(backproject(a, last(x)), backproject(b, last(x))) for cs in views for x in cs)
    spline_rms(camera, poses, views) =
        sqrt(sum(sum(abs2, reprojection_error(camera, P, first(c), last(c))) for (P, cs) in zip(poses, views) for c in cs) / sum(length, views))
    # The views of the poses `E` (their perturbed starts `p0`), `n` detections each with known noise (σ = 0.3 px), of a
    # lattice rougher than the KB one (every control turned 0.02 rad, so the penalty's bias and the noise trade within
    # the grid), refined with `penalty = :gcv`: the search's bookkeeping, GCV's weight within 0.1 decade (the old search's
    # final bracket) of `old_log10β`, the weight the per-weight nonlinear search chose (before 2026-11-11), and within a
    # decade of the weight of `grid` (log₁₀ β) whose estimate predicts the clean pixels best (the reprojection rms against them: gauge-free),
    # predicting them as well to 2%. A fixed small weight on the noisy detections: the call returns within 3×
    # `weak_seconds` (an error naming the cause and the remedy, or a finite estimate), its time measured warm on one core (raptorlake,
    # 2026-11-11): its solves are bounded (the LM never sees a non-finite linearization, the rebuild rounds are capped),
    # so no machine spins in it (on dgx01 NLLSsolver 5.3.2's damping loop once did, > 60 min).
    function check_gcv_choice(E, p0, n, grid, old_log10β, weak_seconds)
        rough = perturbed_controls(spline_truth(), 0.02)
        clean = spline_views(rough, E, 1.1; n)
        rng = Xoshiro(20261110)
        noisy = [[first(c) => last(c) + 0.3 * SVector(randn(rng), randn(rng)) for c in cs] for cs in clean]
        g0 = perturbed_controls(rough, 0.01)
        prediction(r) = spline_rms(model(estimate(r).camera), estimate(r).poses, clean)
        rg = refine_camera_and_poses(bounded(g0), p0, noisy)
        @test retcode(rg) === :converged && !rg.spline.gcv_at_bound
        @test rg.spline.β ≈ first(sort(rg.spline.gcv; by = e -> e.gcv)).β rtol = 1.0e-12    # the least of the weights evaluated
        @test all(e -> isfinite(e.gcv), rg.spline.gcv) && issorted(rg.spline.gcv; by = e -> e.β)
        @test abs(log10(rg.spline.β) - old_log10β) <= 0.1
        errors = [prediction(refine_camera_and_poses(bounded(g0), p0, noisy; penalty = 10.0^k)) for k in grid]
        @test abs(log10(rg.spline.β) - grid[argmin(errors)]) <= 1
        @test prediction(rg) <= 1.02 * minimum(errors)
        # (the weak weight's solve is ill-posed: whether a correspondence stops projecting on its way depends on the
        # last bits of the LM's path, so the call either names the cause and the remedy or returns a finite estimate)
        t = @elapsed weak = try
            refine_camera_and_poses(bounded(g0), p0, noisy; penalty = 1.0e-9)
        catch e
            e
        end
        @test weak isa ArgumentError ? occursin("penalty = :gcv", weak.msg) : isfinite(rms(weak))
        # (2026-11-11: the refine now also fits the guess's optical axis at its entry and the refined camera's after it, a
        # diagnostic; those two fits run whatever the weight, so the bound adds them, timed warm here)
        t_axis = (optical_axis(bounded(g0)); @elapsed optical_axis(bounded(g0)))
        @test t <= 3 * (weak_seconds + 2 * t_axis)
        return nothing
    end
end

# A robust refine (`local_optimization`): the RANSAC local optimization's own strategies (Irls{Shape}, Triggs{Shape} with a
# efficiency_constant of the caller's `scale`, or an absolute loss_width), read by its TOML reader, from the caller's start
# (no least-squares stage); a refine estimates no scale; plain least squares unchanged; the refusals.
@testitem "refine_camera_and_poses: local_optimization = Irls{Huber} / Triggs{Welsch} from the caller's start at the caller's scale, no least-squares stage; least squares unchanged" setup = [CameraPoseScenes] begin
    using StaticArrays: SVector
    using LinearAlgebra: normalize
    using Random: Xoshiro
    using VisualGeometryToolkitCore: parameter_vector, with_parameters, estimate, retcode, WelschShape, HuberShape
    using VisualGeometryToolkitEstimation: refine_camera_and_poses, Irls, Triggs, NoLocalOptimization, local_optimization_from_toml, config_dict
    using VisualGeometryToolkitCore: EuclideanMap, exp_so3
    using Rotations: RotMatrix
    cam = pinhole_camera(); Es = truth_poses(5); views = views_of(cam, Es, 0.6)
    θ = parameter_vector(cam)
    cam0 = with_parameters(cam, θ .* SVector(1.04, 1.01, 1.0, 1.0, 1.0) .+ SVector(0.0, 0.0, 0.0, 6.0, -5.0))
    poses0 = [perturbed_pose(E, v) for (v, E) in enumerate(Es)]
    rng = Xoshiro(5)
    # σ = 0.2 px detections, 8 % of them moved 20–60 px in a random direction
    noisy = [[X => u + 0.2 * SVector(randn(rng), randn(rng)) +
                    (rand(rng) < 0.08 ? (20 + 40rand(rng)) * normalize(SVector(randn(rng), randn(rng))) : zero(SVector{2}))
              for (X, u) in cs] for cs in views]
    err(r) = maximum(abs.(parameter_vector(estimate(r).camera) - θ)[[1, 4, 5]])      # f, cx, cy, pixels
    ls = refine_camera_and_poses(cam0, poses0, noisy)
    @test err(ls) > 3.0 && ls.scale === nothing && ls.loss_width === nothing && all(==(1.0), ls.weights)
    # least squares is the default: the same estimate, bit for bit
    ls2 = refine_camera_and_poses(cam0, poses0, noisy; local_optimization = nothing)
    @test parameter_vector(estimate(ls2).camera) == parameter_vector(estimate(ls).camera) && ls2.residuals == ls.residuals
    @test length(ls.residual_norms) == 300 && ls.residual_norms[7] == hypot(ls.residuals[13], ls.residuals[14])
    # A robust refine runs NO least-squares stage (the owner's ruling) and estimates no scale: its width is the
    # strategy's efficiency_constant times the caller's `scale` (the detections' σ). From this far start (every residual
    # 20–80 px, median 30): the truth, to the noise; the outliers' weights low (Irls{Huber} 0.175 px; Triggs{Welsch}
    # 0.30 px, 0.20 with a least-squares stage: one Welsch LM from a start 25 px off)
    for lo in (Irls{HuberShape}(efficiency_constant = 1.345), Triggs{WelschShape}(efficiency_constant = 2.98))
        @test_throws r"estimates none: supply `scale`" refine_camera_and_poses(cam0, poses0, noisy; local_optimization = lo)
        rs = refine_camera_and_poses(cam0, poses0, noisy; local_optimization = lo, scale = 0.2)
        @test retcode(rs) === :converged && err(rs) < 0.5 && rs.scale == 0.2
        @test rs.loss_width ≈ (lo isa Irls ? 1.345 : 2.98) * 0.2
        @test count(<(0.5), rs.weights) >= 20 && length(rs.weights) == 300
    end
    # From a start near the data (f 0.5 %, 1 px, poses 2 mrad / 5 mm off: residuals median 3.3 px, a caller's linear
    # calibration's), at the caller's σ (Irls{Huber} 0.175 px, Triggs{Welsch} 0.202 px)
    near(E, v) = EuclideanMap(RotMatrix{3, Float64}(exp_so3(SVector(0.002, -0.0015v / 4, 0.002)) * E.R),
                              E.t + SVector(0.004, -0.003, 0.005 - 0.001v))
    cam1 = with_parameters(cam, θ .* SVector(1.005, 1.0, 1.0, 1.0, 1.0) .+ SVector(0.0, 0.0, 0.0, 1.0, -1.0))
    poses1 = [near(E, v) for (v, E) in enumerate(Es)]
    for lo in (Irls{HuberShape}(efficiency_constant = 1.345), Triggs{WelschShape}(efficiency_constant = 2.98))
        r = refine_camera_and_poses(cam1, poses1, noisy; local_optimization = lo, scale = 0.2)
        @test retcode(r) === :converged && err(r) < 0.4 && r.scale == 0.2
    end
    # an absolute width needs no scale (Irls; Triggs takes an efficiency constant only): Huber at loss_width 1.345 · 0.2 px
    rw = refine_camera_and_poses(cam1, poses1, noisy; local_optimization = Irls{HuberShape}(loss_width = 1.345 * 0.2))
    @test retcode(rw) === :converged && err(rw) < 0.4 && rw.scale === nothing && rw.loss_width ≈ 1.345 * 0.2
    # the strategy from the LO's own TOML reader, and back
    t = Dict("type" => "triggs", "loss" => "welsch", "efficiency_constant" => 2.98)
    lo = local_optimization_from_toml(t; class = :refine_camera_and_poses, section = "refine_camera_and_poses.local_optimization")
    @test lo == Triggs{WelschShape}(efficiency_constant = 2.98) && config_dict(lo) == t
    # refusals: a strategy anchored on a RANSAC consensus set, one following a scorer's threshold, a scale without a
    # loss, a scorer (a refine takes none)
    @test_throws r"RANSAC consensus" refine_camera_and_poses(cam0, poses0, noisy; local_optimization = NoLocalOptimization())
    @test_throws r"inlier threshold" refine_camera_and_poses(cam0, poses0, noisy; local_optimization = Irls{WelschShape}())
    @test_throws r"reads none" refine_camera_and_poses(cam0, poses0, noisy; scale = 0.2)
    @test_throws MethodError refine_camera_and_poses(cam1, poses1, noisy; local_optimization = lo, scale = 0.2, scoring = nothing)
    @test_throws r"positive finite" refine_camera_and_poses(cam0, poses0, noisy; local_optimization = lo, scale = -1.0)
    # the retired keyword `robust`, whatever its value: refused, naming `local_optimization`
    @test_throws r"`robust` is retired: it is `local_optimization`" refine_camera_and_poses(cam1, poses1, noisy; robust = nothing)
    @test_throws r"`robust` is retired" refine_camera_and_poses(cam1, poses1, noisy; robust = lo)
    @test_throws r"`robust` is retired" refine_camera_and_poses(cam1, poses1, noisy; robust = lo, scale = 0.2)
end

# Several boards (owner decision 20): the board poses a block of the same bundle, each board → world a variable unless
# held; each connected component of the view–board graph fixes its gauge by a held pose, exactly (no pin, no prior).
@testsetup module BoardScenes
    using StaticArrays: SVector
    using Rotations: RotMatrix, RotXYZ
    using Random: Xoshiro, randn
    using VisualGeometryToolkitCore: EuclideanMap, exp_so3, project, rotation, translation
    export board_points, truth_boards, view_poses, board_views, nudged
    # a 9 × 7 grid of points on the board's plane (z = 0), 4 cm pitch
    board_points() = [SVector(0.04 * (i - 5), 0.04 * (j - 4), 0.0) for i in 1:9 for j in 1:7]
    # three boards around the world origin, tilted differently (board → world)
    truth_boards() = Dict(:A => EuclideanMap(RotMatrix(RotXYZ(0.0, 0.0, 0.0)), SVector(0.0, 0.0, 0.0)),
                          :B => EuclideanMap(RotMatrix(RotXYZ(0.25, -0.3, 0.1)), SVector(0.42, 0.05, 0.08)),
                          :C => EuclideanMap(RotMatrix(RotXYZ(-0.2, 0.35, -0.15)), SVector(-0.38, -0.06, 0.12)))
    # six cameras looking at the boards from 1.2–1.6 m (world → camera)
    view_poses() = Dict(v => EuclideanMap(RotMatrix(RotXYZ(0.15 * sin(v), 0.2 * cos(v), 0.05v)), SVector(0.08 * (v - 3.5), 0.03 * (-1)^v, 1.2 + 0.08v))
                        for v in 1:6)
    # which boards each view sees: every board twice or more, the graph connected
    const SEES = Dict(1 => (:A, :B), 2 => (:A, :C), 3 => (:B, :C), 4 => (:A, :B, :C), 5 => (:B,), 6 => (:C, :A))
    function board_views(camera, views, boards; σ = 0.0, rng = Xoshiro(1))
        Dict((v, b) => [X => SVector{2, Float64}(project(camera, views[v](boards[b](X)))) + σ * SVector(randn(rng), randn(rng))
                        for X in board_points()] for v in keys(views) for b in SEES[v])
    end
    # a pose moved by ≈ 0.04 rad and ≈ 3 cm, differently per key
    nudged(E, k) = EuclideanMap(RotMatrix{3, Float64}(exp_so3(SVector(0.02, -0.015, 0.03) .* (1 + 0.1k)) * rotation(E)),
                                translation(E) + SVector(0.03, -0.02, 0.025) .* (1 - 0.07k))
end

@testitem "refine_camera_and_poses: board poses, recovered from exact and noisy pixels, the gauge held exactly" setup = [CameraPoseScenes, BoardScenes] begin
    using StaticArrays: SVector
    using LinearAlgebra: norm
    using Random: Xoshiro
    using VisualGeometryToolkitCore: parameter_vector, with_parameters, estimate, retcode, rms, rotation, translation, project
    using VisualGeometryToolkitEstimation: refine_camera_and_poses
    cam = pinhole_camera(); boards = truth_boards(); views = view_poses()
    θ = parameter_vector(cam)
    cam0 = with_parameters(cam, θ .* SVector(1.03, 1.0, 1.0, 1.0, 1.0) .+ SVector(0.0, 0.0, 0.0, 5.0, -4.0))
    views0 = Dict(v => nudged(E, v) for (v, E) in views)
    keyn = Dict(:A => 1, :B => 2, :C => 3)
    boards0 = Dict(b => b === :A ? E : nudged(E, 7 + keyn[b]) for (b, E) in boards)     # board A at the truth: the held gauge
    Δ(P, E) = max(maximum(abs, rotation(P) - rotation(E)), norm(translation(P) - translation(E)))   # ≈ rad, m
    # exact pixels: everything recovered to the LM's tolerance, board A (each component's first board) held exactly
    res = refine_camera_and_poses(cam0, views0, boards0, board_views(cam, views, boards))
    est = estimate(res)
    @test retcode(res) === :converged && rms(res) < 1.0e-9
    @test maximum(abs.(parameter_vector(est.camera) - θ) ./ max.(abs.(θ), 1.0)) < 1.0e-9
    @test est.board_poses[:A] === boards0[:A]
    @test maximum(Δ(est.board_poses[b], boards[b]) for b in (:B, :C)) < 1.0e-9
    @test maximum(Δ(est.poses[v], views[v]) for v in 1:6) < 1.0e-9
    @test res.ν == 2 * 63 * 12 - (4 + 6 * 6 + 6 * 2)
    # noisy pixels (σ = 0.2 px per coordinate): from the perturbed start, the optimum the truth start reaches (the
    # estimator's own, recovered), its rms at the noise (σ√2 per correspondence), the camera and poses to the noise
    # (measured: f, cx, cy within 4.2 px, the boards within 1.5 mrad / mm, the views within 1 cm: the boards are near
    # fronto-parallel, so the views' depth trades with f)
    noisy = board_views(cam, views, boards; σ = 0.2, rng = Xoshiro(7))
    rn = refine_camera_and_poses(cam0, views0, boards0, noisy)
    rt = refine_camera_and_poses(cam, views, boards, noisy)
    en, et = estimate(rn), estimate(rt)
    @test retcode(rn) === :converged && retcode(rt) === :converged
    @test parameter_vector(en.camera) ≈ parameter_vector(et.camera) rtol = 1.0e-8
    @test maximum(Δ(en.poses[v], et.poses[v]) for v in 1:6) < 1.0e-6
    @test maximum(Δ(en.board_poses[b], et.board_poses[b]) for b in (:A, :B, :C)) < 1.0e-6
    @test 0.24 < rms(rn) < 0.32
    @test maximum(abs.(parameter_vector(en.camera) - θ)[[1, 4, 5]]) < 8.0          # f, cx, cy: 1 % of f
    @test maximum(Δ(en.board_poses[b], boards[b]) for b in (:B, :C)) < 5.0e-3
    @test maximum(Δ(en.poses[v], views[v]) for v in 1:6) < 2.0e-2
    # the gauge held by a view instead: the same solution up to the gauge (every view ∘ board the same map)
    rv = refine_camera_and_poses(cam0, views0, boards0, noisy; held_boards = (), held_views = (1,))
    ev = estimate(rv)
    @test ev.poses[1] == views0[1] && retcode(rv) === :converged
    @test rms(rv) ≈ rms(rn) rtol = 1.0e-8
    for v in 1:6, b in (:A, :B, :C)
        @test Δ(ev.poses[v] ∘ ev.board_poses[b], en.poses[v] ∘ en.board_poses[b]) < 1.0e-6
    end
    # a second component (view 7 sees board D alone) is held by its own first board
    vd = merge(views, Dict(7 => views[2])); bd = merge(boards, Dict(:D => nudged(boards[:B], 1)))
    obs = merge(board_views(cam, views, boards), Dict((7, :D) => [X => SVector{2, Float64}(project(cam, vd[7](bd[:D](X))))
                                                                   for X in board_points()]))
    r2 = refine_camera_and_poses(cam0, merge(views0, Dict(7 => nudged(views[2], 3))), merge(boards0, Dict(:D => bd[:D])), obs)
    @test retcode(r2) === :converged && estimate(r2).board_poses[:D] === bd[:D] && rms(r2) < 1.0e-9
    # refusals: a component with no held pose, a key without a pose, a pose without correspondences
    @test_throws r"hold no pose" refine_camera_and_poses(cam0, views0, boards0, noisy; held_boards = ())
    @test_throws r"no pose" refine_camera_and_poses(cam0, views0, Dict(:A => boards[:A], :B => boards[:B]), noisy)
    @test_throws r"no correspondence" refine_camera_and_poses(cam0, merge(views0, Dict(9 => views[1])), boards0, noisy)
end

@testitem "refine_camera_and_poses: a SplineLens camera's controls and poses refined from its optical axis, K frozen" setup = [CameraPoseScenes, SplineCameraPoseScenes] begin
    using StaticArrays: SVector
    using LinearAlgebra: normalize, norm
    using VisualGeometryToolkitCore: VisualGeometryToolkitCore as VC, estimate, retcode, rms, calibration_matrix, backproject, lens, SplineLens,
        parameter_vector, angle_between, rotation
    using VisualGeometryToolkitEstimation: VisualGeometryToolkitEstimation as VE, refine_camera_and_poses
    truth = spline_truth(); Es = truth_poses(3)
    views = spline_views(truth, Es, 1.1)
    cam0 = perturbed_controls(truth, 0.01); poses0 = [perturbed_pose(E, v) for (v, E) in enumerate(Es)]
    res = refine_camera_and_poses(bounded(cam0), poses0, views)
    est = estimate(res)
    @test retcode(res) === :converged
    # K bit-identical to the entry's: the guess on its optical axis (2026-11-11: no longer the guess's own K — the entry
    # gauge moves its principal point to the axis's pixel; held exactly from there)
    g0, R0 = entry_gauge(cam0)
    @test calibration_matrix(model(est.camera)) == calibration_matrix(g0)
    @test lens(model(est.camera)) isa SplineLens && lens(model(est.camera)).forward === nothing
    @test rms(res) < 1.0e-3 < spline_rms(cam0, poses0, views)                  # the guess's error is ≈ 3 px
    @test rms(res) ≈ spline_rms(model(est.camera), est.poses, views) rtol = 1.0e-6     # the rows are the refined camera's own
    # the entry's gauge recorded, and the diagnostic of the refined camera's own optical axis (2026-11-11, option A: the
    # end step that turned the estimate's ray at x = 0 onto ẑ, then its roll, and rescaled the controls to the guess's mean
    # length, is deleted — the entry's frame and K are held, nothing re-gauges the estimate)
    @test rotation(res.spline.axis_rotation) == R0 && isfinite(res.spline.symmetry_residual)
    # measured 2026-11-11: the axis's image 0.11 px from the held principal point; the symmetry residual 1.22 px for the
    # perturbed guess, 0.56 px refined (the truth's own 12 × 10 lattice: 0.15 px; the diagnostic samples the whole image,
    # past the detections, where the penalty holds the lattice)
    @test res.spline.refined_axis_offset < 0.5
    @test res.spline.refined_symmetry_residual < res.spline.symmetry_residual
    @test res.spline.n_nodes_unrefined == 0                                   # every node a data or Δ³ row reads
    # GCV chose the weight (the default): the curve over the decade grid and its golden refinement, tr H in (0, 2n)
    @test res.spline.gcv isa Vector && length(res.spline.gcv) >= 13 && res.spline.λ > 0
    @test res.spline.gcv_at_bound == (res.spline.β in (1.0e-12, 1.0))
    @test 0 < res.spline.effective_dof < 2res.n
    # backprojection against the truth at the detections, in the estimate's frame (the guess's axis-gauged one): better
    # than the guess by > 10³, and the poses the truth's
    tf, Ef = truth_in_frame(truth, Es, est)
    @test ray_error(model(est.camera), tf, views) < ray_error(cam0, truth, views) / 1000
    @test maximum(maximum(abs, P.R - E.R) for (P, E) in zip(est.poses, Ef)) < 1.0e-6
    @test res.n == 360 && res.spline.n_dropped_window == 0 && res.spline.n_dropped_unprojected == 0 && res.free == (:controls,)
    # the joint solve lowers its objective (λ fixed across the rounds), and the rows' rms is the penalty's bias against
    # exact data: at a 1e-14 relative weight it falls toward the solver's floor
    @test 0.5 * res.s² * res.ν <= res.spline.cost_start
    @test rms(refine_camera_and_poses(bounded(cam0), poses0, views; penalty = 1.0e-14)) < 1.0e-6
    # a lens five times as far off (every control turned ≈ 0.05 rad): converges within the per-solve cap (the
    # directions-first stage; a joint solve from there crept for 158 iterations)
    far = perturbed_controls(truth, 0.05)
    rf = refine_camera_and_poses(bounded(far), poses0, views)
    @test retcode(rf) === :converged && rf.spline.windows_settled
    # `max_time` caps the whole refine, all stages and rounds: with a zero cap the first solve stops after its first
    # iteration, and only the joint solve's one capped iteration follows. GCV's start solve (at β = 1) cut short: no
    # weight is searched, and the metadata is that estimate's — β = 1, its λ, no curve and no tr H — and its windows,
    # never rebuilt, are not settled
    rt = refine_camera_and_poses(bounded(far), poses0, views; max_time = 0.0)
    @test retcode(rt) === :max_time && rt.iterations <= 2 && rt.spline.rounds == 1
    @test rt.spline.β == 1.0 && rt.spline.gcv === nothing && rt.spline.effective_dof === nothing && !rt.spline.gcv_at_bound
    @test !rt.spline.windows_settled
    rp = refine_camera_and_poses(bounded(far), poses0, views; penalty = 1.0e-8, max_time = 0.0)
    @test retcode(rp) === :max_time && !rp.spline.windows_settled && rp.spline.β == 1.0e-8
    @test ray_error(model(estimate(rf).camera), first(truth_in_frame(truth, Es, estimate(rf))), views) < ray_error(far, truth, views) / 1000
end

@testitem "refine_camera_and_poses: the SplineLens cameras fit_camera and the constructors make, non-unit controls" setup = [CameraPoseScenes, SplineCameraPoseScenes] begin
    using StaticArrays: SVector
    using LinearAlgebra: norm
    using VisualGeometryToolkitCore: CameraModel, SplineLens, ChartSpline, KannalaBrandt, BoundedCamera, Size2, backproject, imaged_region,
        estimate, retcode, rms, lens, calibration_matrix
    using VisualGeometryToolkitEstimation: refine_camera_and_poses, fit_camera
    kb = CameraModel(; focal = 420.0, pp = (640.0, 480.0), lens = KannalaBrandt(-0.02, 0.004, -0.001, 0.0002))
    bk = BoundedCamera(kb, Size2(1280, 960))
    fitted = estimate(fit_camera(SplineLens, bk, imaged_region(bk); tolerance = 1.0e-2, forward = false))   # the P-spline's controls, 1.8e-2 off unit
    quasi = CameraModel(SplineLens, u -> backproject(kb, u), Size2(1280, 960); init = :quasi_interpolate)          # 7.7e-3 off unit
    t = spline_truth(); bt = lens(t).backward
    scaled = CameraModel(t.intrinsics, SplineLens(ChartSpline(bt.area, map(c -> 2.0 * c, bt.controls))))        # every norm 2
    Es = truth_poses(3); poses0 = [perturbed_pose(E, v) for (v, E) in enumerate(Es)]
    for truth in (fitted, quasi, scaled)
        @test maximum(c -> abs(norm(c) - 1), lens(truth).backward.controls) > 1.0e-3
        views = spline_views(truth, Es, 1.1)
        cam0 = perturbed_controls(truth, 0.01)
        res = refine_camera_and_poses(bounded(cam0), poses0, views)
        est = estimate(res)
        @test retcode(res) === :converged
        @test calibration_matrix(model(est.camera)) == calibration_matrix(first(entry_gauge(cam0)))   # K bit-identical to the entry's
        @test rms(res) < 1.0e-3 < spline_rms(cam0, poses0, views)
        @test ray_error(model(est.camera), first(truth_in_frame(truth, Es, est)), views) < ray_error(cam0, truth, views) / 1000
        # (the mean control length is no longer rescaled to the guess's after the solve, 2026-11-11: the scale gauge is
        # held during the solve by the node nearest x = 0's log-scale, and the images do not read it)
    end
    # a lens five times as far off (every control turned ≈ 0.05 rad): the fit_camera lattice still converges
    views = spline_views(fitted, Es, 1.1)
    cam0 = perturbed_controls(fitted, 0.05)
    res = refine_camera_and_poses(bounded(cam0), poses0, views)
    @test retcode(res) === :converged
    @test ray_error(model(estimate(res).camera), first(truth_in_frame(fitted, Es, estimate(res))), views) < ray_error(cam0, fitted, views) / 1000
end

@testitem "refine_camera_and_poses: a SplineLens block's analytic Jacobian matches finite differences of its residual" setup = [CameraPoseScenes, SplineCameraPoseScenes] begin
    using StaticArrays: SVector
    using LinearAlgebra: norm
    import NLLSsolver
    using VisualGeometryToolkitEstimation: VisualGeometryToolkitEstimation as VE
    truth = spline_truth(); Es = truth_poses(2)
    views = spline_views(truth, Es, 1.1; n = 12)
    cam0 = perturbed_controls(truth, 0.01); poses0 = [perturbed_pose(E, v) for (v, E) in enumerate(Es)]
    h = 1.0e-6
    # the full patches (directions and log-scales) and the directions-first stage's directions-only patches
    for directions_only in (false, true)
        p = VE._spline_problem(cam0, poses0, views, true, 1.0e-8; directions_only)
        @test !isempty(p.costs) && !isempty(p.smooth)
        # every block arity the layout makes: a window over 2 × 2, 2 × 3 (3 × 2) and 3 × 3 patches, a Δ³ row over 2 and 3
        @test Set(length(c.varind) for c in p.costs) == Set((5, 7, 10)) && Set(length(c.varind) for c in p.smooth) == Set((2, 3))
        for c in [p.costs; p.smooth]                                           # the data and Δ³ blocks
            vars = NLLSsolver.getvars(c, p.variables)
            J = NLLSsolver.computeresjac(2^length(c.varind) - 1, c, vars...)[2]
            col = 0
            for (k, v) in enumerate(vars), j in 1:NLLSsolver.dynamic(NLLSsolver.nvars(v))
                col += 1
                e = zeros(NLLSsolver.dynamic(NLLSsolver.nvars(v))); e[j] = h
                f(δ) = NLLSsolver.computeresidual(c, ntuple(i -> i == k ? NLLSsolver.update(v, δ) : vars[i], length(vars))...)
                fd = (f(e) - f(-e)) / 2h
                @test norm(J[:, col] - fd) <= 1.0e-6 * max(norm(fd), maximum(abs, J))
            end
            @test col == size(J, 2)
        end
    end
end

@testitem "refine_camera_and_poses: a SplineLens camera's poses alone, a forward refitted, a masked lattice, refusals" setup = [CameraPoseScenes, SplineCameraPoseScenes] begin
    using StaticArrays: SVector
    using VisualGeometryToolkitCore: VisualGeometryToolkitCore as VC, CameraModel, SplineLens, ChartSpline, BoundedCamera, Size2, estimate, retcode,
        lens, model, parameter_vector, with_present, spline_window, imaged_region, calibration_matrix
    using LinearAlgebra: norm
    using VisualGeometryToolkitEstimation: VisualGeometryToolkitEstimation, refine_camera_and_poses, fit_camera
    truth = spline_truth(); Es = truth_poses(3)
    views = spline_views(truth, Es, 1.1)
    poses0 = [perturbed_pose(E, v) for (v, E) in enumerate(Es)]
    # the poses alone: the camera comes back as the entry gauged it — the truth on its optical axis, its controls and K
    # == (2026-11-11: no longer the given camera itself) — and the poses recovered in that frame
    held = refine_camera_and_poses(bounded(truth), poses0, views; free = ())
    gt, Rt = entry_gauge(truth)
    @test retcode(held) === :converged && held.free == ()
    @test parameter_vector(model(estimate(held).camera)) == parameter_vector(gt) && calibration_matrix(model(estimate(held).camera)) == calibration_matrix(gt)
    @test maximum(maximum(abs, P.t - VC.axis_gauged(E, Rt).t) for (P, E) in zip(estimate(held).poses, Es)) < 1.0e-6
    # a camera holding a forward spline: refined without it, the forward refitted over the bounded camera's imaged region
    cam0 = perturbed_controls(truth, 0.01)
    bc = BoundedCamera(estimate(fit_camera(SplineLens, BoundedCamera(cam0, Size2(1280, 960)), imaged_region(BoundedCamera(cam0, Size2(1280, 960)));
        tolerance = 1.0e-3)), Size2(1280, 960))
    @test lens(model(bc)).forward isa ChartSpline{2}
    rf = refine_camera_and_poses(bc, poses0, views; forward_tolerance = 1.0e-3)
    @test retcode(rf) === :converged && estimate(rf).camera isa BoundedCamera
    @test lens(model(estimate(rf).camera)).forward isa ChartSpline{2} && rf.spline.forward === :refitted
    @test rf.spline.forward_max_px <= 1.0e-3
    @test calibration_matrix(model(estimate(rf).camera)) == calibration_matrix(first(entry_gauge(bc)))   # the entry's K, held
    rd = refine_camera_and_poses(bc, poses0, views; refit_forward = false)
    @test lens(model(estimate(rd).camera)).forward === nothing && rd.spline.forward === :dropped
    @test_throws r"BoundedCamera" refine_camera_and_poses(model(bc), poses0, views)   # a refit needs the image
    # a masked lattice: the detections on an absent window are dropped and counted
    present = trues(10, 12); present[1:3, 9:12] .= false
    masked = with_present(cam0, present)
    b = lens(masked).backward
    absent(u) = (w = spline_window(b.area, size(b.controls), VC.Cameras._normalized_point(masked.intrinsics, u)); !all(present[(w[2] - 2):(w[2] + 1), (w[1] - 2):(w[1] + 1)]))
    n_absent = count(absent(last(x)) for cs in views for x in cs)
    @test n_absent > 0
    rm = refine_camera_and_poses(bounded(masked), poses0, views)
    @test rm.spline.n_dropped_window == n_absent
    @test rm.n == 360 - n_absent - rm.spline.n_dropped_unprojected
    @test rm.spline.n_dropped_unprojected <= 10        # those the camera images onto an absent window, within a cell of one (the guess's taken back)
    @test lens(model(estimate(rm).camera)).present == present
    # a present node no window and no third-difference row reads is no variable: it comes back as its guess, bit for bit
    island = trues(10, 12); island[1:4, 1:4] .= false; island[1, 1] = true           # node (1, 1) alone in its corner
    ri = refine_camera_and_poses(bounded(with_present(cam0, island)), poses0, views)
    @test ri.spline.n_nodes_unrefined == 1
    @test lens(model(estimate(ri).camera)).backward.controls[1, 1] == last(entry_gauge(with_present(cam0, island))) * lens(cam0).backward.controls[1, 1]   # turned by the entry
    # a rim fragment: two present nodes (1, 1) and (1, 2) that no window and no third-difference row reads are held
    # at their guesses and counted
    rim = trues(10, 12); rim[1:4, 1:4] .= false; rim[1, 1] = rim[1, 2] = true
    rr = refine_camera_and_poses(bounded(with_present(cam0, rim)), poses0, views; penalty = 1.0e-8)
    @test rr.spline.n_nodes_unrefined == 2
    Rr = last(entry_gauge(with_present(cam0, rim)))
    @test lens(model(estimate(rr).camera)).backward.controls[1, 1:2] == map(c -> Rr * c, lens(cam0).backward.controls[1, 1:2])
    # (the mean length over the live nodes is no longer rescaled to the guess's after the solve, 2026-11-11: the end
    # step is deleted; the scale is held during the solve by one node's log-scale)
    # refusals: a box, a partial `free`, a penalty that is neither :gcv nor a weight ≥ 0
    @test_throws ArgumentError refine_camera_and_poses(bounded(truth), poses0, views; penalty = :aic)
    @test_throws ArgumentError refine_camera_and_poses(bounded(truth), poses0, views; penalty = -1.0)
    @test_throws ArgumentError refine_camera_and_poses(bounded(truth), poses0, views; box = (; c_1_1_x = (0.0, 1.0)))
    @test_throws ArgumentError refine_camera_and_poses(bounded(truth), poses0, views; free = (:c_1_1_x,))
    # a lattice that folds where a point projects (every control ẑ: the backward's Jacobian exactly 0 there) is not
    # linearized: the error names the fold
    # (2026-11-11: the refine's entry fits the optical axis first, and a lattice imaging ẑ everywhere has no field to
    # fit — refused there; the fold refusal is the axis-gauged refine's own, checked on it directly)
    flat = CameraModel(truth.intrinsics, SplineLens(ChartSpline(b.area, fill(SVector(0.0, 0.0, 1.0), size(b.controls)))))
    P = poses0[1]
    one_point = [[P.R' * (SVector(0.0, 0.0, 2.0) - P.t) => SVector(640.0, 480.0)]]
    @test_throws r"empty fit domain" refine_camera_and_poses(bounded(flat), [P], one_point)
    np = VisualGeometryToolkitEstimation._refine_spline_axis_gauged(flat, [P], one_point, false)
    @test occursin(r"lattice folds where the point .* projects at the guess", VisualGeometryToolkitEstimation._not_projected_message(np))
end

@testitem "refine_camera_and_poses: a SplineLens camera's smoothness is scale-free; the penalty fixed, off, or chosen by GCV" setup = [CameraPoseScenes, SplineCameraPoseScenes] begin
    using StaticArrays: SVector
    using LinearAlgebra: norm, eigvals, Symmetric, tr
    using VisualGeometryToolkitCore: VisualGeometryToolkitCore as VC, CameraModel, SplineLens, ChartSpline, estimate, retcode, rms, lens,
        parameter_vector, spline_window, angle_between, backproject, calibration_matrix
    using VisualGeometryToolkitEstimation: VisualGeometryToolkitEstimation as VE, refine_camera_and_poses
    truth = spline_truth(); Es = truth_poses(3)
    poses0 = [perturbed_pose(E, v) for (v, E) in enumerate(Es)]
    cam0 = perturbed_controls(truth, 0.01)
    # every control scaled by s: the objective (the data rows and the third-difference rows, each divided by the mean
    # length of its own nodes) is unchanged to rounding, and so is λ from a relative weight
    small = spline_views(truth, Es[1:2], 1.1; n = 24)
    b = lens(cam0).backward
    scaled = CameraModel(cam0.intrinsics, SplineLens(ChartSpline(b.area, map(c -> 3.7 * c, b.controls))))
    using VisualGeometryToolkitCore: nlls_rows_and_jacobian, rotation, model
    cost(q, vars = q.variables) = 0.5 * sum(abs2, first(nlls_rows_and_jacobian(VE._spline_blocks(q), vars)))
    p = VE._spline_problem(cam0, poses0[1:2], small, true, 1.0e-8)
    q = VE._spline_problem(scaled, poses0[1:2], small, true, 1.0e-8; λ = p.λ)
    @test !isempty(p.smooth) && cost(merge(p, (; costs = empty(p.costs)))) > 0
    @test cost(q) ≈ cost(p) rtol = 1.0e-12
    @test VE._spline_problem(scaled, poses0[1:2], small, true, 1.0e-8).λ ≈ p.λ rtol = 1.0e-12
    # the gauge fixed exactly by the parametrization: at the refined estimate the variables' coordinates are the live
    # controls' 3 less the gauge's 4 (the pinned view's rotation and the scale node's log-scale are no coordinate) and
    # the poses', and the normal matrix over them is full rank; the gauge is a symmetry of the cost (the turned and
    # scaled variables cost the same, to rounding). Measured: λmin/λmax 4.9e-13
    views = spline_views(truth, Es, 1.1)
    r7 = estimate(refine_camera_and_poses(bounded(cam0), poses0, views; penalty = 3.0e-7))
    pf = VE._spline_problem(model(r7.camera), r7.poses, views, true, 3.0e-7)
    _, Jf = nlls_rows_and_jacobian(VE._spline_blocks(pf), pf.variables)
    @test size(Jf, 2) == 3count(pf.live) - 4 + 6 * 3
    ev = eigvals(Symmetric(Matrix(Jf' * Jf)))
    @test count(>(eps() * length(ev) * maximum(ev)), ev) == length(ev)                     # full rank
    # (the rotation: the camera and the poses turned together by Core's axis_gauged, every image kept)
    Q = VC.RotMatrix{3, Float64}(VC.exp_so3(SVector(0.02, -0.01, 0.03)))
    K7 = calibration_matrix(model(r7.camera))
    pq = VE._spline_problem(VC.axis_gauged(model(r7.camera), Q, K7), [VC.axis_gauged(P, Q) for P in r7.poses], views, true, 3.0e-7; λ = pf.λ)
    @test cost(pq) ≈ cost(pf) rtol = 1.0e-8 atol = 1.0e-18
    # the gauge held in the LM: an NLLSsolver solve of the directions-first (stage 2) and the joint (stage 3) problem
    # leaves the pinned view's rotation to the SO(3) re-projection's ulps and the scale node's log-scale == its start,
    # while the rest moves; the slots no row reads are no coordinate and stay bit for bit
    for directions_only in (true, false)
        ph = VE._spline_problem(cam0, poses0, views, true, 3.0e-7; directions_only)
        _, sh = VE._spline_solve!(ph, 10, 1.0e-12, 1.0e-12, nothing)
        v = findfirst(k -> sh[k] isa VE._SplinePose && length(sh[k].idx) == 3, eachindex(sh))
        @test count(k -> sh[k] isa VE._SplinePose && length(sh[k].idx) == 3, eachindex(sh)) == 1 && ph.variables[v].idx == [4, 5, 6]
        R0 = rotation(model(ph.variables[v])); R1 = rotation(model(sh[v]))
        @test maximum(abs, R1 - R0) <= 4 * eps()
        @test maximum(abs, VC.translation(model(sh[v])) - VC.translation(model(ph.variables[v]))) > 1.0e-6   # its translation moved
        held = [(k, n) for k in 1:ph.nP for n in 1:4 if VE._slot_live(ph.variables[k], n) && !(8 + n in ph.variables[k].idx)]
        @test length(held) == (directions_only ? count(ph.live) : 1)
        @test all(((k, n),) -> VE._patch_scale(sh[k], n) == VE._patch_scale(ph.variables[k], n), held)
        dead = [(k, n) for k in 1:ph.nP for n in 1:4 if !VE._slot_live(ph.variables[k], n)]
        @test all(((k, n),) -> VE._patch_direction(sh[k], n) == VE._patch_direction(ph.variables[k], n), dead)
        directions_only || @test any(k -> any(n -> VE._patch_scale(sh[k], n) != 0.0, 1:4), 1:ph.nP)
    end

    # tr H on the 3-view fixture: the exact trace and the 64-probe Hutchinson estimate (the switch lowered to 0) agree
    # within 3σ, σ ≤ √(2 tr H / 64); tr H well below the 720 data rows
    using VisualGeometryToolkitEstimation: gcv_curve
    r8 = estimate(refine_camera_and_poses(bounded(cam0), poses0, views; penalty = 1.0e-8))
    pt = VE._spline_problem(model(r8.camera), r8.poses, views, true, 1.0e-8)            # at the refined estimate
    solved = pt.variables
    # (the linearized curve at the problem's own weight: the pencil, exact, and the per-weight sparse Cholesky with the probes)
    rows = VE._gcv_rows(pt, solved)
    ex = gcv_curve(rows...; exact_max = typemax(Int))(pt.λ); hu = gcv_curve(rows...; exact_max = 0)(pt.λ)
    @test VE._gcv_linearization(pt, solved)(pt.λ).dof == ex.dof                     # the refine's: the pencil, at 3000 and below
    @test 0 < ex.dof < 0.6 * 720
    @test abs(hu.dof - ex.dof) <= 3 * sqrt(2 * ex.dof / 64)
    @test gcv_curve(rows...; exact_max = 0)(pt.λ).dof == hu.dof   # the probes' seed is fixed
    @test hu.rss ≈ ex.rss rtol = 1.0e-8
    # the curve is the Gauss–Newton step's: δ = argmin ‖J_d δ + r_d‖² + λ‖J_s δ + s‖² over the variables' coordinates
    # (the problem's own rows; its Δ³ rows already carry √λ), solved directly
    rr, Jt = nlls_rows_and_jacobian(VE._spline_blocks(pt), solved); nd = 2length(pt.costs)
    δ = -(Matrix(Jt' * Jt) \ (Jt' * rr))
    @test ex.rss ≈ sum(abs2, rr[1:nd] + Jt[1:nd, :] * δ) rtol = 1.0e-8
    @test ex.dof ≈ tr(Matrix(Jt[1:nd, :] * (Matrix(Jt' * Jt) \ Matrix(Jt[1:nd, :]')))) rtol = 1.0e-8
    @test ex.y ≈ δ rtol = 1.0e-6

    # a rebuild keeps its λ: no unit weight is formed (the data's Gauss–Newton diagonal and the Δ³ operator's mean
    # eigenvalue, a sparse QR), so it costs a fraction of a first build
    VE._spline_problem(cam0, poses0, views, true, 1.0e-8; λ = pt.λ)
    a_fresh = @allocated VE._spline_problem(cam0, poses0, views, true, 1.0e-8)
    a_given = @allocated VE._spline_problem(cam0, poses0, views, true, 1.0e-8; λ = pt.λ)
    @test a_given < 0.5 * a_fresh

    # a numeric penalty: GCV off, the weight as given, the same result every time
    r1 = refine_camera_and_poses(bounded(cam0), poses0, views; penalty = 1.0e-8)
    r2 = refine_camera_and_poses(bounded(cam0), poses0, views; penalty = 1.0e-8)
    @test retcode(r1) === :converged && r1.spline.β == 1.0e-8 && r1.spline.gcv === nothing && !r1.spline.gcv_at_bound
    @test parameter_vector(model(estimate(r1).camera)) == parameter_vector(model(estimate(r2).camera))
    # penalty = 0: no smoothness; the present nodes no correspondence's window reads are held at their guesses and counted
    # (without the Δ³ rows each log-scale is held by its data alone, nearly collinear with its direction: the joint
    # solve takes ≈ 210 iterations, past the default per-solve cap)
    @test retcode(refine_camera_and_poses(bounded(cam0), poses0, views; penalty = 0)) === :max_iterations   # the default cap: visible
    r0 = refine_camera_and_poses(bounded(cam0), poses0, views; penalty = 0, max_iter = 300)
    @test retcode(r0) === :converged && r0.spline.β == 0 && r0.spline.λ == 0 && r0.spline.gcv === nothing
    # (the nodes no window reads, and their held controls, are the entry-gauged guess's, 2026-11-11: the refine starts
    # from the guess on its optical axis — its controls turned, its principal point moved)
    g0 = first(entry_gauge(cam0))
    read = falses(size(b.controls))
    for cs in views, x in cs
        w = spline_window(lens(g0).backward.area, size(b.controls), VC.Cameras._normalized_point(g0.intrinsics, last(x)))
        read[(w[2] - 2):(w[2] + 1), (w[1] - 2):(w[1] + 1)] .= true
    end
    @test count(!, read) > 0 && r0.spline.n_nodes_unrefined == count(!, read)
    @test lens(model(estimate(r0).camera)).backward.controls[.!read] == lens(g0).backward.controls[.!read]
    @test rms(r0) < 1.0e-6                                                        # exact data, no penalty bias

    # detections with known noise (σ = 0.3 px), 3 views × 60: GCV's choice (`check_gcv_choice`, the scenes' setup).
    # Measured: GCV β = 10^-2.225 (the per-weight nonlinear search's 10^-2.24), prediction 0.2388 px (0.2389); the grid 0.2573, 0.2385, 0.3726 px at β = 1e-3 … 1e-1 (at 1e-5
    # and below the noisy estimate stops projecting a correspondence: GCV = Inf there). The same claim at 6 views × 200
    # is the benchmark-size item below (VGTK_BENCHMARK_TESTS=1).
    check_gcv_choice(Es, poses0, 60, -3:-1, -2.24, 0.96)
end

# Benchmark size, opt-in (VGTK_BENCHMARK_TESTS=1; otherwise the item is skipped): 7.8 s and 8.3 GB of allocation
# warm on one core (2026-11-10, raptorlake), where the default item's 3 × 60 case above takes ≈ 3 s and 2.7 GB.
@testitem "refine_camera_and_poses: GCV's weight on 6 views × 200 noisy detections (benchmark size, VGTK_BENCHMARK_TESTS=1)" setup = [CameraPoseScenes, SplineCameraPoseScenes] skip = get(ENV, "VGTK_BENCHMARK_TESTS", "") != "1" begin
    # Measured: GCV β = 10^-2.99 (the per-weight nonlinear search's 10^-3.06), prediction 0.1211 px (0.1214); the grid 0.1288, 0.1211, 0.1416 px at β = 1e-4 … 1e-2
    E6 = truth_poses(6)
    check_gcv_choice(E6, [perturbed_pose(E, v) for (v, E) in enumerate(E6)], 200, -4:-2, -3.06, 2.35)
end

@testitem "gcv_curve, gcv_select: the stack's one GCV of a penalty weight" begin
    using LinearAlgebra: I, Symmetric, eigvals, tr
    using Random: Xoshiro, randn
    using SparseArrays: sparse
    using VisualGeometryToolkitCore: difference_rows
    using VisualGeometryToolkitEstimation: gcv_curve, gcv_select, gcv_score, check_penalty_weight
    # the closed-form curve equals the direct solve at every λ: rss of the penalized fit and tr H
    rng = Xoshiro(3)
    J = randn(rng, 40, 8); r = randn(rng, 40); Dm = Matrix(difference_rows((1, 8), 3)); c = zeros(size(Dm, 1))
    curve = gcv_curve(J' * J, J' * r, sum(abs2, r), Dm' * Dm, Dm' * c, 40)
    for λ in (1.0e-3, 0.1, 10.0)
        A = J' * J + λ * Dm' * Dm
        y = A \ (J' * r)
        H = J * (A \ J')
        e = curve(λ)
        @test e.rss ≈ sum(abs2, r - J * y) rtol = 1.0e-8
        @test e.dof ≈ tr(H) rtol = 1.0e-8
        @test e.gcv ≈ 40 * e.rss / (40 - e.dof)^2 rtol = 1.0e-12
    end
    # with a target c, normal-equation and rows forms; the rows form past `exact_max` (a sparse Cholesky per λ, its
    # symbolic analysis once, a Hutchinson trace) gives the same rss and tr H within 3σ
    c2 = randn(rng, size(Dm, 1))
    ch = gcv_curve(J' * J, J' * r, sum(abs2, r), Dm' * Dm, Dm' * c2, 40)
    cr = gcv_curve(J, r, Dm, c2); cs = gcv_curve(J, r, Dm, c2; exact_max = 0, probes = 4000)
    for λ in (1.0e-3, 0.1, 10.0)
        A = J' * J + λ * Dm' * Dm
        y = A \ (J' * r + λ * Dm' * c2)
        e = ch(λ)
        @test e.rss ≈ sum(abs2, r - J * y) rtol = 1.0e-8
        @test e.dof ≈ tr(J * (A \ J')) rtol = 1.0e-8
        @test cr(λ).rss ≈ e.rss rtol = 1.0e-10
        @test cr(λ).dof ≈ e.dof rtol = 1.0e-10
        @test cs(λ).rss ≈ e.rss rtol = 1.0e-8
        @test e.y ≈ y rtol = 1.0e-8                                                     # the solution at λ, every form
        @test cr(λ).y ≈ y rtol = 1.0e-8
        @test cs(λ).y ≈ y rtol = 1.0e-8
        @test abs(cs(λ).dof - e.dof) <= 3 * sqrt(2 * e.dof / 4000)
    end
    # a parameter no row reads (a zero column of J and of D) is dropped by every form, the pencil and the per-λ
    # Cholesky alike: 0 in y, the rest as without it
    Ju = hcat(J, zeros(40)); Du = hcat(Dm, zeros(size(Dm, 1)))
    for exact_max in (3000, 0)
        cu = gcv_curve(Ju, r, Du, c2; exact_max, probes = 4000)
        @test cu(0.1).y[end] == 0.0
        @test cu(0.1).y[1:8] ≈ cr(0.1).y rtol = 1.0e-8
        @test cu(0.1).rss ≈ cr(0.1).rss rtol = 1.0e-8
    end
    # a direction neither term reads but the parameters share (two equal columns, the penalty blind to their difference):
    # a gauge, refused — never ridged
    Jg = hcat(J[:, 1:7], J[:, 7]); Dg = hcat(Dm[:, 1:6], zeros(size(Dm, 1), 2))
    @test_throws r"not positive definite" gcv_curve(Jg, r, Dg, c)
    # the search, a coarse scan and Brent's method: a minimum inside the bracket to Brent's floor (no tolerance to
    # choose), in a handful of evaluations past the scan; one at an end reported there
    calls = Ref(0)
    s = gcv_select(λ -> (calls[] += 1; (; gcv = (log10(λ) + 2.3)^2, rss = 0.0, dof = 0.0)), 1.0; log10_bracket = (-8.0, 4.0), points = 25)
    @test abs(log10(s.β) + 2.3) < 1.0e-7 && !s.at_bracket_end && s.λ == s.β && calls[] <= 25 + 12
    e = gcv_select(λ -> (; gcv = log10(λ), rss = 0.0, dof = 0.0), 1.0; log10_bracket = (-8.0, 4.0), points = 25)
    @test e.at_bracket_end && e.β == 1.0e-8
    # a minimum inside the scan's first interval, sampled lowest at the scan's end: refined there and interior
    s1 = gcv_select(λ -> (; gcv = (log10(λ) - 3.8)^2, rss = 0.0, dof = 0.0), 1.0; log10_bracket = (-8.0, 4.0), points = 25)
    @test abs(log10(s1.β) - 3.8) < 1.0e-7 && !s1.at_bracket_end
    # a weight the caller cannot evaluate (Inf) is never the minimum, even beside the basin
    s2 = gcv_select(λ -> (; gcv = log10(λ) < -3 ? Inf : (log10(λ) + 2.3)^2, rss = 0.0, dof = 0.0), 1.0; log10_bracket = (-8.0, 4.0), points = 25)
    @test abs(log10(s2.β) + 2.3) < 1.0e-7 && isfinite(s2.gcv)
    @test gcv_score(40, 2.0, 8.0) == 40 * 2.0 / 32^2
    @test check_penalty_weight(:gcv, "w") === :gcv && check_penalty_weight(0.5, "w") == 0.5
    @test_throws ArgumentError check_penalty_weight(nothing, "w")                              # RC's options: no `nothing`
    @test check_penalty_weight(nothing, "w"; allow_nothing = true) === nothing
    @test_throws ArgumentError check_penalty_weight(-1.0, "w")
    @test_throws ArgumentError check_penalty_weight(:aic, "w")
end
