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
