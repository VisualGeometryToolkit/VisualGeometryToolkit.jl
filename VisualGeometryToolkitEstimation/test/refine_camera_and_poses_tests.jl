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
    using LinearAlgebra: normalize, norm
    using VisualGeometryToolkitCore: CameraModel, SplineLens, ChartSpline, KannalaBrandt, project, reprojection_error, backproject, angle_between
    export spline_truth, perturbed_controls, spline_views, spline_rms, ray_error

    # the truth: a Kannala–Brandt fisheye's SplineLens camera (unit controls, 12 × 10 nodes over 1280 × 960)
    spline_truth() = CameraModel(SplineLens, CameraModel(; focal = 420.0, pp = (640.0, 480.0), lens = KannalaBrandt(-0.02, 0.004, -0.001, 0.0002)),
                                 (1280, 960); grid_size = (12, 10))
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
end

@testitem "refine_camera_and_poses: a SplineLens camera's controls and poses refined, K frozen, the axis on ẑ" setup = [CameraPoseScenes, SplineCameraPoseScenes] begin
    using StaticArrays: SVector
    using LinearAlgebra: normalize, norm
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
    # the gauge, one rotation: the axis normalize(backward(0, 0)) on ẑ and the roll (the x-derivative at x = 0 in
    # the xz plane) to rounding, the cost where it was
    b = lens(est.camera).backward
    d, J = VC.Cameras._spline_direction_and_jacobian(b.controls, b.area, SVector(0.0, 0.0))
    @test angle_between(d, SVector(0.0, 0.0, 1.0)) < 1.0e-9 && abs(atan(J[2, 1], J[1, 1])) < 1.0e-9
    @test d / norm(d) ≈ SVector(0.0, 0.0, 1.0) atol = 1.0e-12
    # (GCV's weight leaves a cost of ≈ 1e-10 px² against exact data, so the rows' rounding in the pixel, ≈ 1e-13 px, is
    # an absolute floor: measured 1.8e-19)
    @test res.spline.cost_after_gauge ≈ res.spline.cost_before_gauge rtol = 1.0e-10 atol = 1.0e-15
    @test res.spline.anchor_cost == 0.0
    # the scale gauge: every control rescaled after the solve so the mean control length is the guess's
    @test sum(norm, b.controls) / length(b.controls) ≈ sum(norm, lens(cam0).backward.controls) / length(b.controls) rtol = 1.0e-14
    @test res.spline.n_nodes_unrefined == 0                                   # every node a data or Δ³ row reads
    # GCV chose the weight (the default): the curve over the decade grid and its golden refinement, tr H in (0, 2n)
    @test res.spline.gcv isa Vector && length(res.spline.gcv) >= 13 && res.spline.λ > 0
    @test res.spline.gcv_at_bound == (res.spline.penalty in (1.0e-12, 1.0))
    @test 0 < res.spline.effective_dof < 2res.n
    # backprojection against the truth at the detections (the truth's own frame: the gauge is the truth's): better
    # than the guess by > 10³, and the poses the truth's
    @test ray_error(est.camera, truth, views) < ray_error(cam0, truth, views) / 1000
    @test maximum(maximum(abs, P.R - E.R) for (P, E) in zip(est.poses, Es)) < 1.0e-6
    @test res.n == 360 && res.spline.n_dropped_window == 0 && res.spline.n_dropped_unprojected == 0 && res.free == (:controls,)
    # the joint solve lowers its objective (λ fixed across the rounds), and the rows' rms is the penalty's bias against
    # exact data: at a 1e-14 relative weight it falls toward the solver's floor
    @test res.spline.cost_after_gauge <= res.spline.cost_start
    @test rms(refine_camera_and_poses(cam0, poses0, views; penalty = 1.0e-14)) < 1.0e-6
    # a lens five times as far off (every control turned ≈ 0.05 rad): converges within the per-solve cap (the
    # directions-first stage; a joint solve from there crept for 158 iterations)
    far = perturbed_controls(truth, 0.05)
    rf = refine_camera_and_poses(far, poses0, views)
    @test retcode(rf) === :converged && rf.spline.windows_settled
    # `max_time` caps the whole refine, all stages and rounds: with a zero cap the first solve stops after its first
    # iteration, and only the joint solve's one capped iteration follows
    rt = refine_camera_and_poses(far, poses0, views; max_time = 0.0)
    @test retcode(rt) === :max_time && rt.iterations <= 2 && rt.spline.rounds == 1
    @test ray_error(estimate(rf).camera, truth, views) < ray_error(far, truth, views) / 1000
    # the patch layout is no part of the estimate: the lattice laid one node into the patches agrees to the solver's tolerance
    e1 = first(VE._refine_spline_camera_and_poses(cam0, poses0, views, 1))
    @test ray_error(e1.camera, est.camera, views) < 1.0e-9
    @test maximum(maximum(abs, P.R - E.R) for (P, E) in zip(e1.poses, est.poses)) < 1.0e-9
    @test maximum(abs, parameter_vector(e1.camera) - parameter_vector(est.camera)) < 1.0e-6
end

@testitem "refine_camera_and_poses: the SplineLens cameras fit_camera and the constructors make, non-unit controls" setup = [CameraPoseScenes, SplineCameraPoseScenes] begin
    using StaticArrays: SVector
    using LinearAlgebra: norm
    using VisualGeometryToolkitCore: CameraModel, SplineLens, ChartSpline, KannalaBrandt, BoundedCamera, Size2, backproject, imaged_region,
        estimate, retcode, rms, lens
    using VisualGeometryToolkitEstimation: refine_camera_and_poses, fit_camera
    kb = CameraModel(; focal = 420.0, pp = (640.0, 480.0), lens = KannalaBrandt(-0.02, 0.004, -0.001, 0.0002))
    bk = BoundedCamera(kb, Size2(1280, 960))
    fitted = estimate(fit_camera(SplineLens, bk, imaged_region(bk); tolerance = 1.0e-2, forward = false))   # the P-spline's controls, 1.8e-2 off unit
    quasi = CameraModel(SplineLens, u -> backproject(kb, u), (1280, 960); init = :quasi_interpolate)          # 7.7e-3 off unit
    t = spline_truth(); bt = lens(t).backward
    scaled = CameraModel(t.intrinsics, SplineLens(ChartSpline(bt.area, map(c -> 2.0 * c, bt.controls))))        # every norm 2
    Es = truth_poses(3); poses0 = [perturbed_pose(E, v) for (v, E) in enumerate(Es)]
    for truth in (fitted, quasi, scaled)
        @test maximum(c -> abs(norm(c) - 1), lens(truth).backward.controls) > 1.0e-3
        views = spline_views(truth, Es, 1.1)
        cam0 = perturbed_controls(truth, 0.01)
        res = refine_camera_and_poses(cam0, poses0, views)
        est = estimate(res)
        @test retcode(res) === :converged
        @test est.camera.intrinsics === cam0.intrinsics                                      # K bit-identical
        @test rms(res) < 1.0e-3 < spline_rms(cam0, poses0, views)
        @test ray_error(est.camera, truth, views) < ray_error(cam0, truth, views) / 1000
        mean_length(c) = sum(norm, lens(c).backward.controls) / length(lens(c).backward.controls)
        @test mean_length(est.camera) ≈ mean_length(cam0) rtol = 1.0e-14                 # the scale gauge
    end
    # a lens five times as far off (every control turned ≈ 0.05 rad): the fit_camera lattice still converges
    views = spline_views(fitted, Es, 1.1)
    cam0 = perturbed_controls(fitted, 0.05)
    res = refine_camera_and_poses(cam0, poses0, views)
    @test retcode(res) === :converged
    @test ray_error(estimate(res).camera, fitted, views) < ray_error(cam0, fitted, views) / 1000
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
        @test !isempty(p.costs) && !isempty(p.smooth) && !isempty(p.anchors)
        for c in [p.costs; p.smooth; p.anchors]                                # the data, Δ³ and anchor blocks
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
    @test rm.spline.n_dropped_window == n_absent
    @test rm.n == 360 - n_absent - rm.spline.n_dropped_unprojected
    @test rm.spline.n_dropped_unprojected <= 10        # those the camera images onto an absent window, within a cell of one (the guess's taken back)
    @test lens(estimate(rm).camera).present == present
    # a present node no window and no third-difference row reads is anchored: it comes back as its guess, and the anchors add 0
    island = trues(10, 12); island[1:4, 1:4] .= false; island[1, 1] = true           # node (1, 1) alone in its corner
    ri = refine_camera_and_poses(with_present(cam0, island), poses0, views)
    @test lens(estimate(ri).camera).backward.controls[1, 1] == lens(cam0).backward.controls[1, 1]
    @test ri.spline.anchor_cost == 0.0
    @test ri.spline.cost_after_gauge ≈ ri.spline.cost_before_gauge rtol = 1.0e-10 atol = 1.0e-15
    # a rim fragment: two present nodes (1, 1) and (1, 2) that no window and no third-difference row reads are held
    # at their guesses and counted
    rim = trues(10, 12); rim[1:4, 1:4] .= false; rim[1, 1] = rim[1, 2] = true
    rr = refine_camera_and_poses(with_present(cam0, rim), poses0, views; penalty = 1.0e-8)
    @test rr.spline.n_nodes_unrefined == 2
    @test lens(estimate(rr).camera).backward.controls[1, 1:2] == lens(cam0).backward.controls[1, 1:2]
    # refusals: a box, a partial `free`, a penalty that is neither :gcv nor a weight ≥ 0
    @test_throws ArgumentError refine_camera_and_poses(truth, poses0, views; penalty = :aic)
    @test_throws ArgumentError refine_camera_and_poses(truth, poses0, views; penalty = -1.0)
    @test_throws ArgumentError refine_camera_and_poses(truth, poses0, views; box = (; c_1_1_x = (0.0, 1.0)))
    @test_throws ArgumentError refine_camera_and_poses(truth, poses0, views; free = (:c_1_1_x,))
end

@testitem "refine_camera_and_poses: a SplineLens camera's smoothness is scale-free; the penalty fixed, off, or chosen by GCV" setup = [CameraPoseScenes, SplineCameraPoseScenes] begin
    using StaticArrays: SVector
    using LinearAlgebra: norm, eigvals, Symmetric
    using Random: Xoshiro, randn
    using VisualGeometryToolkitCore: VisualGeometryToolkitCore as VC, CameraModel, SplineLens, ChartSpline, estimate, retcode, rms, lens,
        parameter_vector, spline_window, angle_between, backproject
    using VisualGeometryToolkitEstimation: VisualGeometryToolkitEstimation as VE, refine_camera_and_poses
    truth = spline_truth(); Es = truth_poses(3)
    poses0 = [perturbed_pose(E, v) for (v, E) in enumerate(Es)]
    cam0 = perturbed_controls(truth, 0.01)
    # every control scaled by s: the objective (the data rows and the third-difference rows, each divided by the mean
    # length of its own nodes) is unchanged to rounding, and so is λ from a relative weight
    small = spline_views(truth, Es[1:2], 1.1; n = 24)
    b = lens(cam0).backward
    scaled = CameraModel(cam0.intrinsics, SplineLens(ChartSpline(b.area, map(c -> 3.7 * c, b.controls))))
    p = VE._spline_problem(cam0, poses0[1:2], small, true, 1.0e-8)
    q = VE._spline_problem(scaled, poses0[1:2], small, true, 1.0e-8; λ = p.λ)
    @test !isempty(p.smooth) && VE._spline_cost(p.variables, p.smooth) > 0
    @test VE._spline_cost(q.variables, VE._spline_blocks(q)) ≈ VE._spline_cost(p.variables, VE._spline_blocks(p)) rtol = 1.0e-12
    @test VE._spline_problem(scaled, poses0[1:2], small, true, 1.0e-8).λ ≈ p.λ rtol = 1.0e-12
    # tr H: the exact trace and the 64-probe Hutchinson estimate (the switch lowered to 0) agree within 3σ, σ ≤ √(2 tr H / 64)
    # the gauge held exactly: at the refined estimate, the normal matrix of every row (data, Δ³, anchors) over the
    # coordinates the solve moves (the pins' held ones deleted) is full rank, while the unreduced one is singular along
    # the common scale (the rotation's 3 directions are held off zero only by the Δ³ rows' Gauss–Newton term).
    # Measured: reduced λmin/λmax 4.9e-13; unreduced 4e-20, then 5e-14 (×3)
    r7 = estimate(refine_camera_and_poses(cam0, poses0, spline_views(truth, Es, 1.1); penalty = 3.0e-7))
    pf = VE._spline_problem(r7.camera, r7.poses, spline_views(truth, Es, 1.1), true, 3.0e-7)
    _, Jf = VE._spline_problem_rows_and_jacobian(pf, pf.variables)
    @test length(pf.held) == 4 && length(pf.pins) == 2
    kept = setdiff(axes(Jf, 2), pf.held)
    ev = eigvals(Symmetric(Matrix(Jf[:, kept]' * Jf[:, kept]))); ev0 = sort(eigvals(Symmetric(Matrix(Jf' * Jf))))
    @test minimum(ev) / maximum(ev) > 1.0e-13 && ev0[1] / ev0[end] < 1.0e-17

    # tr H on the 3-view fixture: the exact trace and the 64-probe Hutchinson estimate (the switch lowered to 0) agree
    # within 3σ, σ ≤ √(2 tr H / 64); tr H well below the 720 data rows
    views = spline_views(truth, Es, 1.1)
    r8 = estimate(refine_camera_and_poses(cam0, poses0, views; penalty = 1.0e-8))
    pt = VE._spline_problem(r8.camera, r8.poses, views, true, 1.0e-8)            # at the refined estimate
    solved = pt.variables
    ex = VE._gcv(pt, solved; exact_max = typemax(Int)); hu = VE._gcv(pt, solved; exact_max = 0)
    @test 0 < ex.trace < 0.6 * ex.n && ex.n == 720
    @test abs(hu.trace - ex.trace) <= 3 * sqrt(2 * ex.trace / 64)
    @test VE._gcv(pt, solved; exact_max = 0).trace == hu.trace                    # the probes' seed is fixed
    @test ex.rss ≈ 2 * VE._spline_cost(solved, pt.costs) rtol = 1.0e-12

    # a numeric penalty: GCV off, the weight as given, the same result every time
    r1 = refine_camera_and_poses(cam0, poses0, views; penalty = 1.0e-8)
    r2 = refine_camera_and_poses(cam0, poses0, views; penalty = 1.0e-8)
    @test retcode(r1) === :converged && r1.spline.penalty == 1.0e-8 && r1.spline.gcv === nothing && !r1.spline.gcv_at_bound
    @test parameter_vector(estimate(r1).camera) == parameter_vector(estimate(r2).camera)
    # penalty = 0: no smoothness; the present nodes no correspondence's window reads are held at their guesses and counted
    # (without the Δ³ rows each log-scale is held by its data alone, nearly collinear with its direction: the joint
    # solve takes ≈ 210 iterations, past the default per-solve cap)
    r0 = refine_camera_and_poses(cam0, poses0, views; penalty = 0, max_iter = 300)
    @test retcode(r0) === :converged && r0.spline.penalty == 0 && r0.spline.λ == 0 && r0.spline.gcv === nothing
    read = falses(size(b.controls))
    for cs in views, x in cs
        w = spline_window(b.area, size(b.controls), VC.Cameras._normalized_point(cam0.intrinsics, last(x)))
        read[(w[2] - 2):(w[2] + 1), (w[1] - 2):(w[1] + 1)] .= true
    end
    @test count(!, read) > 0 && r0.spline.n_nodes_unrefined == count(!, read)
    @test lens(estimate(r0).camera).backward.controls[.!read] == b.controls[.!read]
    @test rms(r0) < 1.0e-6                                                        # exact data, no penalty bias

    # detections with known noise (σ = 0.3 px) of a lattice rougher than the KB one (every control turned 0.02 rad, so
    # the penalty's bias and the noise trade within the grid), 6 views × 200: GCV's weight lies within a decade of the
    # grid weight whose estimate predicts the clean pixels best (the reprojection rms against them: gauge-free), and
    # predicts them as well to 2%. Measured: GCV β = 10^-3.06, prediction 0.1214 px; the grid 0.1288, 0.1211, 0.1416 px
    # at β = 1e-4 … 1e-2 (at 1e-5 and below the noisy estimate stops projecting a correspondence: GCV = Inf there).
    rough = perturbed_controls(truth, 0.02); E6 = truth_poses(6)
    clean = spline_views(rough, E6, 1.1; n = 200)
    rng = Xoshiro(20261110)
    noisy = [[first(c) => last(c) + 0.3 * SVector(randn(rng), randn(rng)) for c in cs] for cs in clean]
    g0 = perturbed_controls(rough, 0.01); p6 = [perturbed_pose(E, v) for (v, E) in enumerate(E6)]
    prediction(r) = spline_rms(estimate(r).camera, estimate(r).poses, clean)
    rg = refine_camera_and_poses(g0, p6, noisy)
    @test retcode(rg) === :converged && !rg.spline.gcv_at_bound
    @test rg.spline.penalty == first(sort(filter(e -> isfinite(e.gcv), rg.spline.gcv); by = e -> e.gcv)).penalty
    @test any(e -> e.gcv == Inf, rg.spline.gcv)                                   # the descent stopped at a weight that fails
    grid = -4:-2
    errors = [prediction(refine_camera_and_poses(g0, p6, noisy; penalty = 10.0^k)) for k in grid]
    @test abs(log10(rg.spline.penalty) - grid[argmin(errors)]) <= 1
    @test prediction(rg) <= 1.02 * minimum(errors)
    # a fixed small weight on the noisy detections: the error names the cause and the remedy
    @test_throws r"penalty = :gcv" refine_camera_and_poses(g0, p6, noisy; penalty = 1.0e-9)
end

@testitem "gcv_curve, gcv_select, penalty_scale, mean_positive_eigenvalue: the stack's one GCV of a penalty weight" begin
    using LinearAlgebra: I, Symmetric, eigvals, tr
    using Random: Xoshiro, randn
    using SparseArrays: sparse
    using VisualGeometryToolkitCore: difference_rows
    using VisualGeometryToolkitEstimation: gcv_curve, gcv_select, penalty_scale, mean_positive_eigenvalue, check_penalty_weight
    # m̄ from the operator (‖D‖²_F / rank) is the mean positive eigenvalue of DᵀD, sparse or dense
    D = difference_rows(trues(10, 12), 3)
    λs = eigvals(Symmetric(Matrix(D' * D))); pos = λs[λs .> 1.0e-10 * maximum(λs)]
    @test mean_positive_eigenvalue(D) ≈ sum(pos) / length(pos) rtol = 1.0e-12
    @test mean_positive_eigenvalue(Matrix(D)) ≈ mean_positive_eigenvalue(D) rtol = 1.0e-12
    @test penalty_scale(12.0, 3, 2.0) == 2.0
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
    # the search: a minimum inside the bracket refined to xtol; one at an end reported, not refined
    s = gcv_select(λ -> (; gcv = (log10(λ) + 2.3)^2, rss = 0.0, dof = 0.0), 1.0; log10_bracket = (-8.0, 4.0), points = 25, xtol = 1.0e-3)
    @test abs(log10(s.beta) + 2.3) < 2.0e-3 && !s.at_bracket_end
    e = gcv_select(λ -> (; gcv = log10(λ), rss = 0.0, dof = 0.0), 1.0; log10_bracket = (-8.0, 4.0), points = 25)
    @test e.at_bracket_end && e.beta == 1.0e-8
    @test check_penalty_weight(:gcv, "w") === :gcv && check_penalty_weight(0.5, "w") == 0.5
    @test_throws ArgumentError check_penalty_weight(-1.0, "w")
    @test_throws ArgumentError check_penalty_weight(:aic, "w")
end
