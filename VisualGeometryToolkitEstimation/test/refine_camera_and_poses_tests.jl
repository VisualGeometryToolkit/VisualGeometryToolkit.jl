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
    costs = [VE._CameraPoseResidual{2, typeof(cam)}(SVector(1, 1 + v), cam, SVector{3, Float64}(first(c)), last(c)) for v in 1:2 for c in views[v]]
    r, J = VE._camera_poses_rows_and_jacobian(costs, chc, chs, 8)
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
    using LinearAlgebra: normalize
    using VisualGeometryToolkitCore: CameraModel, SplineLens, ChartSpline, KannalaBrandt, project, reprojection_error
    export spline_truth, perturbed_controls, spline_views, spline_rms

    # the truth: a Kannala–Brandt fisheye's SplineLens camera (unit controls, 12 × 10 nodes over 1280 × 960)
    spline_truth() = CameraModel(SplineLens, CameraModel(; focal = 420.0, pp = (640.0, 480.0), lens = KannalaBrandt(-0.02, 0.004, -0.001, 0.0002)),
                                 (1280, 960); grid_size = (12, 10))
    # every present control turned by ≈ `ϵ` rad off its own direction, differently per node (kept unit)
    function perturbed_controls(cam, ϵ)
        b = cam.lens.backward
        C = map(CartesianIndices(b.controls)) do i
            c = b.controls[i]; k = i[1] + 13 * i[2]
            isfinite(c[1]) ? normalize(c + ϵ * SVector(sin(1.3k), cos(0.7k), sin(2.1k + 0.5))) : c
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
    spline_rms(camera, poses, views) =
        sqrt(sum(sum(abs2, reprojection_error(camera, P, first(c), last(c))) for (P, cs) in zip(poses, views) for c in cs) / sum(length, views))
end

@testitem "refine_camera_and_poses: a SplineLens camera's controls and poses refined, K frozen, the axis on ẑ" setup = [CameraPoseScenes, SplineCameraPoseScenes] begin
    using StaticArrays: SVector
    using LinearAlgebra: normalize
    using VisualGeometryToolkitCore: VisualGeometryToolkitCore as VC, estimate, retcode, rms, calibration_matrix, backproject, lens, SplineLens,
        parameter_vector, angle_between
    using VisualGeometryToolkitEstimation: VisualGeometryToolkitEstimation as VE, refine_camera_and_poses
    truth = spline_truth(); Es = truth_poses(3)
    views = spline_views(truth, Es, 1.1)
    cam0 = perturbed_controls(truth, 0.01); poses0 = [perturbed_pose(E, v) for (v, E) in enumerate(Es)]
    res = refine_camera_and_poses(cam0, poses0, views)
    est = estimate(res)
    @test retcode(res) === :converged
    @test est.camera.intrinsics === cam0.intrinsics && calibration_matrix(est.camera) == calibration_matrix(cam0)   # K bit-identical
    @test lens(est.camera) isa SplineLens && lens(est.camera).forward === nothing
    @test rms(res) < 1.0e-3 < spline_rms(cam0, poses0, views)                  # the guess's error is ≈ 3 px
    @test rms(res) ≈ spline_rms(est.camera, est.poses, views) rtol = 1.0e-6     # the rows are the refined camera's own
    # the axis: normalize(backward(0, 0)) is ẑ to rounding, and the gauge rotation left the cost where it was
    @test VC.Cameras._lens_axis(lens(est.camera).backward) ≈ SVector(0.0, 0.0, 1.0) atol = 1.0e-12
    @test angle_between(VC.Cameras._lens_axis(lens(est.camera).backward), SVector(0.0, 0.0, 1.0)) < 1.0e-9
    @test res.spline.cost_after_gauge ≈ res.spline.cost_before_gauge rtol = 1.0e-10
    # backprojection against the truth, read in the refined frame (the truth up to the rotation about ẑ the data
    # leave free: Q = R_est R_trueᵀ, the same for every view), at the detections: better than the guess by > 10×
    Q = est.poses[1].R * Es[1].R'
    @test maximum(maximum(abs, P.R * E.R' - Q) for (P, E) in zip(est.poses, Es)) < 1.0e-6
    err(c) = maximum(angle_between(backproject(c, last(x)), Q * backproject(truth, last(x))) for cs in views for x in cs)
    err0 = maximum(angle_between(backproject(cam0, last(x)), backproject(truth, last(x))) for cs in views for x in cs)
    @test err(est.camera) < err0 / 10
    @test res.n == 360 && res.spline.n_dropped_window == 0 && res.free == (:controls,)
    @test res.spline.anchor_cost == 0.0
    # the patch layout is no part of the estimate: the lattice laid one node into the patches agrees to the solver's tolerance
    e1 = first(VE._refine_spline_camera_and_poses(cam0, poses0, views, 1))
    @test maximum(angle_between(backproject(e1.camera, last(x)), backproject(est.camera, last(x))) for cs in views for x in cs) < 1.0e-8
    @test maximum(maximum(abs, P.R - E.R) for (P, E) in zip(e1.poses, est.poses)) < 1.0e-8
    @test maximum(abs, parameter_vector(e1.camera) - parameter_vector(est.camera)) < 1.0e-5
end

@testitem "refine_camera_and_poses: a SplineLens block's analytic Jacobian matches finite differences of its residual" setup = [CameraPoseScenes, SplineCameraPoseScenes] begin
    using StaticArrays: SVector
    using LinearAlgebra: norm
    import NLLSsolver
    using VisualGeometryToolkitEstimation: VisualGeometryToolkitEstimation as VE
    truth = spline_truth(); Es = truth_poses(2)
    views = spline_views(truth, Es, 1.1; n = 12)
    cam0 = perturbed_controls(truth, 0.01); poses0 = [perturbed_pose(E, v) for (v, E) in enumerate(Es)]
    p = VE._spline_problem(cam0, poses0, views, true, 1.0e-8)
    h = 1.0e-6
    @test !isempty(p.costs) && !isempty(p.smooth) && !isempty(p.anchors)
    for c in [p.costs; p.smooth; p.anchors]                                    # the data, Δ³ and anchor blocks
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

@testitem "refine_camera_and_poses: a SplineLens camera's poses alone, a forward refitted, a masked lattice, refusals" setup = [CameraPoseScenes, SplineCameraPoseScenes] begin
    using StaticArrays: SVector
    using VisualGeometryToolkitCore: VisualGeometryToolkitCore as VC, CameraModel, SplineLens, ChartSpline, BoundedCamera, Size2, estimate, retcode,
        lens, model, parameter_vector, with_present, spline_window, imaged_region
    using VisualGeometryToolkitEstimation: refine_camera_and_poses, fit_camera
    truth = spline_truth(); Es = truth_poses(3)
    views = spline_views(truth, Es, 1.1)
    poses0 = [perturbed_pose(E, v) for (v, E) in enumerate(Es)]
    # the poses alone: the camera comes back as given, its controls ==, the poses recovered
    held = refine_camera_and_poses(truth, poses0, views; free = ())
    @test retcode(held) === :converged && estimate(held).camera === truth && held.free == ()
    @test parameter_vector(estimate(held).camera) == parameter_vector(truth)
    @test maximum(maximum(abs, P.t - E.t) for (P, E) in zip(estimate(held).poses, Es)) < 1.0e-6
    # a camera holding a forward spline: refined without it, the forward refitted over the bounded camera's imaged region
    cam0 = perturbed_controls(truth, 0.01)
    bc = BoundedCamera(estimate(fit_camera(SplineLens, BoundedCamera(cam0, Size2(1280, 960)), imaged_region(BoundedCamera(cam0, Size2(1280, 960)));
        tolerance = 1.0e-3)), Size2(1280, 960))
    @test lens(model(bc)).forward isa ChartSpline{2}
    rf = refine_camera_and_poses(bc, poses0, views; forward_tolerance = 1.0e-3)
    @test retcode(rf) === :converged && estimate(rf).camera isa BoundedCamera
    @test lens(model(estimate(rf).camera)).forward isa ChartSpline{2} && rf.spline.forward === :refitted
    @test rf.spline.forward_max_px <= 1.0e-3
    @test model(estimate(rf).camera).intrinsics === model(bc).intrinsics
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
    rm = refine_camera_and_poses(masked, poses0, views)
    @test rm.spline.n_dropped_window >= n_absent && rm.n == 360 - rm.spline.n_dropped_window
    @test rm.spline.n_dropped_window <= n_absent + 10        # and the few the guess images onto an absent window, within a cell of one
    @test lens(estimate(rm).camera).present == present
    # a present node no window and no third-difference row reads is anchored: it comes back as its guess, and the anchors add 0
    island = trues(10, 12); island[1:4, 1:4] .= false; island[1, 1] = true           # node (1, 1) alone in its corner
    ri = refine_camera_and_poses(with_present(cam0, island), poses0, views)
    @test lens(estimate(ri).camera).backward.controls[1, 1] == lens(cam0).backward.controls[1, 1]
    @test ri.spline.anchor_cost == 0.0
    @test ri.spline.cost_after_gauge ≈ ri.spline.cost_before_gauge rtol = 1.0e-10
    # refusals: non-unit controls (a per-node sphere chart would fix each norm, a blend weight), a box, a partial `free`
    bt = truth.lens.backward
    loose = CameraModel(truth.intrinsics, SplineLens(ChartSpline(bt.area, map(c -> 2.0 * c, bt.controls))))
    @test_throws r"unit" refine_camera_and_poses(loose, poses0, views)
    @test_throws ArgumentError refine_camera_and_poses(truth, poses0, views; box = (; c_1_1_x = (0.0, 1.0)))
    @test_throws ArgumentError refine_camera_and_poses(truth, poses0, views; free = (:c_1_1_x,))
end
