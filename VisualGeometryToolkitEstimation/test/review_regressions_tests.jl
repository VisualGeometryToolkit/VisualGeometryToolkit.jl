@testitem "review regressions: the cache, the context and the too-few paths" setup = [PoseScenes] begin
    using Test, LinearAlgebra, StaticArrays, StableRNGs
    using Rotations: RotMatrix
    import VisualGeometryToolkitCore as VgtkCore
    import VisualGeometryToolkitEstimation as VgtkEstimation
    using VisualGeometryToolkitCore: solve, init, solve!, reinit!, EstimationProblem, RefineProblem, LevenbergMarquardt,
        HomographyMat, MsacScoring, Irls, TLSShape, CameraModel, project, m, successful_retcode, default_optimization_strategy,
        withattrs, Correspondences, Extrinsics3, Vec3, CameraCalibrationMat, EuclideanMap
    using VisualGeometryToolkitEstimation: Ransac, LocalOptimization, HomographyProblem, HandEyeProblem, AbsolutePoseProblem,
        BearingPoseProblem, LineFittingProblem, RelativePoseProblem
    using VisualGeometryToolkitCore: HyperRectangle, Vec2, Point2

    rng = StableRNG(5)
    H = @SMatrix [1.1 0.02 3.0; -0.01 0.95 -2.0; 1.0e-4 2.0e-5 1.0]
    src = [SVector(640rand(rng), 480rand(rng)) for _ in 1:40]
    dst = [(q = H * SVector(p[1], p[2], 1.0); SVector(q[1] / q[3], q[2] / q[3]) .+ 0.2 .* randn(rng, SVector{2})) for p in src]
    cs = src .=> dst

    # CR1: a cache pointed at fewer data than a minimal sample is the no-model result of no trial, and back again
    c = init(EstimationProblem(HomographyProblem(cs)), Ransac(); rng = StableRNG(1))
    @test successful_retcode(solve!(c))
    reinit!(c; f = HomographyProblem(cs[1:3]))
    few = solve!(c)
    @test few.u === nothing && few.stats.nsteps == 0
    reinit!(c; f = HomographyProblem(cs))
    @test successful_retcode(solve!(c))

    # CR5: PROSAC on a `:score` column builds on fewer data than a minimal sample (nothing draws from it)
    scored = withattrs(Correspondences(src[1:3], dst[1:3]); score = [3.0, 2.0, 1.0])
    r5 = solve(EstimationProblem(HomographyProblem(scored)), Ransac(); rng = StableRNG(2))
    @test r5.u === nothing && r5.stats.nsteps == 0

    # CR2: reinit! with new data re-resolves the class scorer, whose width reads the data (hand-eye's length)
    sc = robot_world_hand_eye_scene(StableRNG(29); N = 30, Y = (ang_max = π, t_scale = 2.0), X = (ang_max = π, t_scale = 0.1),
                                    A = (ang_max = π / 3, t_scale = 0.5))
    X, Y, A = only(sc.X), sc.Y, sc.A
    B = [marker_pose(Y, a, X) for a in A]
    scale(E, k) = EuclideanMap(VgtkCore.rotation(E), k * VgtkCore.translation(E))
    he1 = HandEyeProblem(A .=> B)
    he2 = HandEyeProblem(scale.(A, 10.0) .=> scale.(B, 10.0))
    ch = init(EstimationProblem(he1), Ransac(); rng = StableRNG(3))
    reinit!(ch; f = he2)
    @test ch.cache.scoring == init(EstimationProblem(he2), Ransac(); rng = StableRNG(3)).cache.scoring
    @test ch.cache.scoring != init(EstimationProblem(he1), Ransac(); rng = StableRNG(3)).cache.scoring

    # CR3: reinit! of a physical problem with its extent keeps the caller's context (metres), as init took it
    cam = CameraModel(; focal = 800.0, pp = (480.0, 360.0))
    Xw = [SVector(0.4 * (rand(rng) - 0.5), 0.4 * (rand(rng) - 0.5), 2.0 + 0.3 * rand(rng)) for _ in 1:20]
    u = [project(cam, x) for x in Xw]
    phys = AbsolutePoseProblem(cam, [x .* m for x in Xw] .=> u)
    extent = HyperRectangle(Vec2(-1.0, -1.0) .* m, Vec2(2.0, 2.0) .* m)
    cp = init(EstimationProblem(phys, (; extent)), Ransac(); rng = StableRNG(4))
    @test reinit!(cp; f = phys) === cp
    @test cp.prob.p == (; extent) && cp.cache.problem isa VgtkEstimation.BoundedProblem

    # CR4: a stray context on a LevenbergMarquardt refine is refused, never ignored
    H0 = solve(EstimationProblem(HomographyMat, cs), VgtkCore.DLT()).u
    @test_throws MethodError solve(RefineProblem(HomographyProblem(cs), H0, (; extent = nothing)), LevenbergMarquardt(); scale = 1.0)

    # CR6: the two views' calibrations in their own types
    K1 = @SMatrix [800.0 0 320; 0 800.0 240; 0 0 1]
    K2 = CameraCalibrationMat(810.0, Point2(330.0, 250.0))
    @test RelativePoseProblem(K1, K2, cs) isa RelativePoseProblem

    # CR7: a problem without a class has no refine strategy of record: Core's refusal, not a TypeError
    pts = [Point2{Float64}(x, 2x + 1) for x in 1.0:10.0]
    @test_throws "has no class of record" default_optimization_strategy(LineFittingProblem(pts))

    # CR8: the bearing pose on fewer data than a minimal sample hands the guess back, even one at a ray origin
    Xb = [SA[0.0, 0.0, 0.0], SA[-0.3, 0.1, 2.5]]
    pose = Extrinsics3(RotMatrix{3, Float64}(I), Vec3{Float64}(0.0, 0.0, 0.0))
    rb = solve(RefineProblem(BearingPoseProblem(Correspondences(Xb, [normalize(SA[0.1, 0.2, 1.0]), normalize(Xb[2])])), pose),
               LevenbergMarquardt(); scale = 1.0)
    @test rb.retcode == VgtkCore.ReturnCode.InitialFailure && rb.stop_rule == VgtkCore.NoRule

    # CR9: a local optimization writes into its cache's result buffers
    lo = LocalOptimization(Irls{TLSShape}(); scoring = MsacScoring(HomographyProblem; threshold = 3.0))
    cl = init(RefineProblem(HomographyProblem(cs), H0), lo)
    rl = solve!(cl)
    @test rl.resid === cl.cache.result.residuals
end
