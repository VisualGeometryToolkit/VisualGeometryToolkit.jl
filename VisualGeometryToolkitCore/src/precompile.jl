# The precompile workload (PrecompileTools): the camera models of every lens family through the calls every consumer makes first,
# run once while the package image is built, so that their native code is cached in it and a fresh process (a test worker, a
# downstream package's first call) does not compile it again. CPU only, no data files, no output, in Float64.

@setup_workload begin
    K = LogicalIntrinsics(CameraCalibrationMat(SMatrix{3, 3, Float64}(400.0, 0, 0, 0, 395.0, 0, 320.0, 240.0, 1)))
    X = SVector(0.2, -0.1, 1.0)
    u = SVector(300.0, 250.0)
    @compile_workload begin
        lenses = (Lens(Perspective()), Division(-0.1667, 0.0291), KannalaBrandt(-0.02, 0.003, -0.0004, 0.0001),
                  Lens(Perspective(), Additive(Radial((-0.2, 0.05, -0.01)), Tangential(0.002, -0.001))),
                  Lens(Perspective(), (Tilt(0.01, -0.02) ∘ Additive(Radial((-0.2, 0.05, -0.01), (0.03, -0.004, 0.001)),
                                                                   Tangential(0.002, -0.001), ThinPrism((-5.5e-3, 7.0e-4), (2.2e-3, -1.2e-3))))),
                  Lens(Equidistant()), Lens(DoubleSphere(-0.2, 0.6)), Lens(Unified(0.8)), Lens(ExtendedUnified(0.6, 1.1)),
                  Lens(Stereographic()))
        for lens in lenses
            cam = CameraModel(K, lens)
            project(cam, X)
            backproject(cam, u)
            backproject_with_jacobian(cam, u)
            θ = parameter_vector(cam)
            with_parameters(cam, θ)
            ch = minimal_manifold(cam)
            action_jacobian(ch, X)
            retract(ch, zero(θ))
        end
        # a SplineLens camera resampled from a parametric one: evaluate, project, back-project with its Jacobian
        g = CameraModel(SplineLens, CameraModel(K, KannalaBrandt(-0.02, 0.003, 0.0, 0.0)), Size2(640, 480); grid_size = (16, 12))
        backproject(g, u)
        backproject_with_jacobian(g, u)
        project(g, X)
        # two frames of a pinhole camera turning about its centre: the solid angle they share
        frame(R) = BoundedCamera(PosedCamera(CameraModel(K, Lens(Perspective())), relabel(CameraFrame, WorldFrame, RotationMap(R))), Size2(640, 480))
        view_overlap(frame(one(RotMatrix{3, Float64})), frame(RotMatrix(RotationVec(0.0, 0.3, 0.0))); boundary_points = 16)
        # the estimation interface's common entries: the DLTs over the two correspondence containers, the minimal
        # solves of a homography and a fundamental matrix, Kabsch and the total-least-squares line
        src = [SVector(Float64(i), Float64(i^2 % 7)) for i in 1:10]
        dst = [SVector(1.1 * p[1] + 0.2 * p[2] + 3.0, -0.1 * p[1] + 0.9 * p[2] - 1.0) for p in src]
        for pairs in (Correspondences(src, dst), correspondences(src, dst))
            solve(EstimationProblem(HomographyMat, pairs), DLT())
            solve(EstimationProblem(FundMat, pairs), DLT())
            solve(MinimalProblem(HomographyMat, pairs, 1:4), Homography4pt())
            solve(MinimalProblem(FundMat, pairs, 1:7), Fundamental7pt())
        end
        vs = [SVector(1.0, 0.0, 0.0), SVector(0.0, 1.0, 0.0), SVector(0.0, 0.0, 1.0)]
        solve(EstimationProblem(RotMatrix{3}, correspondences(vs, [SVector(v[2], -v[1], v[3]) for v in vs])), Kabsch())
        solve(EstimationProblem(Line2D, src), TotalLeastSquares())
    end
end
