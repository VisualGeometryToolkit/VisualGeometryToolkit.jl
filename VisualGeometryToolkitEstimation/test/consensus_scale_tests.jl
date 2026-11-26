@testitem "ConsensusScale: the consensus scale of a model under a scorer" begin
    using Test, Random, StableRNGs, StaticArrays
    using VisualGeometryToolkitCore: solve, init, solve!, reinit!, EstimationProblem, SquaredScale, MsacScoring, ReturnCode,
        ClosedForm, NoRule, HomographyMat, correspondences, successful_retcode, ClosedFormKind, estimator_kind
    import VisualGeometryToolkitCore as VgtkCore
    using VisualGeometryToolkitEstimation: ConsensusScale, HomographyProblem, RansacWorkspace
    import VisualGeometryToolkitEstimation as VgtkEstimation

    rng = StableRNG(7)
    H = SMatrix{3, 3, Float64}(1.02, 0.01, 1.0e-5, -0.02, 0.98, 2.0e-5, 3.0, -2.0, 1.0)
    apply(x) = (y = H * SVector(x[1], x[2], 1.0); SVector(y[1] / y[3], y[2] / y[3]))
    src = [SVector(640 * rand(rng), 480 * rand(rng)) for _ in 1:120]
    tgt = [i <= 90 ? apply(x) + 0.5 * SVector(randn(rng), randn(rng)) : SVector(640 * rand(rng), 480 * rand(rng))
           for (i, x) in enumerate(src)]
    problem = HomographyProblem(correspondences(src, tgt))
    model = HomographyMat{Float64}(Tuple(H))
    sc = MsacScoring(problem; threshold = 3.0)

    @test estimator_kind(ConsensusScale(sc)) === ClosedFormKind()
    r = solve(EstimationProblem(SquaredScale, problem, (; model)), ConsensusScale(sc))
    @test successful_retcode(r) && r.retcode === ReturnCode.Success && r.stop_rule === ClosedForm
    @test r.u isa Float64 && r.u > 0 && r.n > 0
    # the scale the native loop anchors at: the consensus set's Student-t scale at a local optimizer's dof
    buf = RansacWorkspace(problem, sc).buf
    _, k = VgtkEstimation.score_model!(buf, sc, problem, model)
    m = VgtkCore.minimal_sample_size(problem) + 1
    @test r.n == k
    @test r.u === max(VgtkCore.consensus_s2(sc, buf.residuals, VgtkCore.consensus_order(sc, buf.scoring_ws), k;
                                            model_dof = m, d_g = VgtkCore.codimension(problem)), eps())

    # the cache: another model, another width of the same scorer type, the workspace kept
    c = init(EstimationProblem(SquaredScale, problem, (; model)), ConsensusScale(sc))
    @test solve!(c).u === r.u
    sc2 = MsacScoring(problem; threshold = 6.0)
    r2 = solve!(reinit!(c; alg = ConsensusScale(sc2)))
    @test r2.u === solve(EstimationProblem(SquaredScale, problem, (; model)), ConsensusScale(sc2)).u
    allocs(c, model) = @allocations solve!(reinit!(c; p = (; model)))   # its reuse allocates nothing (a function barrier)
    allocs(c, model)
    @test allocs(c, model) == 0
    @test (@inferred solve!(c)).u === r2.u                              # one result type for a scale or none

    # a consensus too thin to carry a scale: degenerate, no model
    far = HomographyMat{Float64}(Tuple(SMatrix{3, 3, Float64}(1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 500.0, 500.0, 1.0)))
    rt = solve(EstimationProblem(SquaredScale, problem, (; model = far)), ConsensusScale(sc))
    @test rt.u === nothing && rt.retcode === ReturnCode.InitialFailure && rt.stop_rule === NoRule && rt.n == 0
end
