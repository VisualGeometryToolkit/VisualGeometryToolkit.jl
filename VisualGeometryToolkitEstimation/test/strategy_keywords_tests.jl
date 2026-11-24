# =============================================================================
# One optimization-strategy family, two keywords (the owner's ruling, 2026-11-11):
# a RANSAC algorithm (`Ransac`, `LocalOptimization`) takes `local_optimization_strategy`,
# a refine (`LevenbergMarquardt`) and a fit (`ReweightedLeastSquares`) `optimization_strategy`;
# a retired keyword is refused naming its successor; a fit takes Irls{K} or `nothing`, never a Triggs.
# =============================================================================

@testitem "strategy keywords: the new names, the retired ones refused" begin
    using VisualGeometryToolkitCore: solve, EstimationProblem, RefineProblem, LevenbergMarquardt
    using VisualGeometryToolkitEstimation: Ransac, HomographyProblem
    using VisualGeometryToolkitEstimation: ReweightedLeastSquares, LinearModel, optimization_strategy_from_toml
    using VisualGeometryToolkitCore: Irls, Triggs
    using VisualGeometryToolkitCore: TLSShape, WelschShape, GaussianShape, HuberShape, Point2
    using StableRNGs: StableRNG
    using StaticArrays: SVector

    H = [1.1 0.05 3.0; -0.02 0.95 -2.0; 1.0e-4 -2.0e-4 1.0]
    src = [Point2(10.0i, 7.0j) for i in 1:6 for j in 1:6]
    dst = map(p -> (h = H * SVector(p[1], p[2], 1.0); Point2(h[1] / h[3], h[2] / h[3])), src)
    pairs = src .=> dst

    r = solve(EstimationProblem(HomographyProblem(pairs)), Ransac(; local_optimization_strategy = Irls{TLSShape}()); rng = StableRNG(1))
    @test r.u !== nothing
    @test solve(EstimationProblem(HomographyProblem(pairs)), Ransac(; local_optimization_strategy = nothing); rng = StableRNG(1)).u !== nothing
    @test_throws r"`local_optimization` is retired: give `local_optimization_strategy`" solve(EstimationProblem(HomographyProblem(pairs)), Ransac(; local_optimization = Irls{TLSShape}()); rng = StableRNG(1))

    @test solve(RefineProblem(HomographyProblem(pairs), r.u), LevenbergMarquardt(; optimization_strategy = Triggs{TLSShape}()); width = 2.0).u !== nothing
    # a strategy is the refine algorithm's field, never a solve keyword (D5: an unknown solve keyword is a MethodError)
    @test_throws MethodError solve(RefineProblem(HomographyProblem(pairs), r.u), LevenbergMarquardt(); local_optimization_strategy = Irls{TLSShape}())
    @test_throws MethodError LevenbergMarquardt(; local_optimization = Irls{TLSShape}())

    A = [1.0 0.0; 0.0 1.0; 1.0 1.0; 2.0 1.0]; y = A * [0.5, -1.0]
    lin(alg; kw...) = solve(EstimationProblem(LinearModel, (A, y)), alg; kw...)
    @test lin(ReweightedLeastSquares(; optimization_strategy = Irls{HuberShape}()); scale = 0.01).u ≈ [0.5, -1.0]
    @test lin(ReweightedLeastSquares(; optimization_strategy = nothing)).u ≈ [0.5, -1.0]        # plain least squares
    @test_throws r"needs a starting guess" lin(ReweightedLeastSquares(; optimization_strategy = Triggs{WelschShape}(efficiency_constant = 2.5)); scale = 0.01)
    @test_throws MethodError ReweightedLeastSquares(; local_optimization = Irls{HuberShape}())

    @test optimization_strategy_from_toml(Dict("type" => "none"); class = :homography) === nothing
end

@testitem "strategy keywords: a strategy that defines no lo_workspace composes (no method ambiguity)" begin
    # Irls and Triggs define their own methods and so mask an ambiguity between Core's
    # default `lo_workspace` and this package's: a downstream strategy (LocalOptimizationDoneRight's
    # HQ, Gnc, …) defines none and must still build a workspace.
    import VisualGeometryToolkitCore as VgtkCore
    using VisualGeometryToolkitEstimation: HomographyProblem, RansacWorkspace, lo_workspace, LoBuffers
    using VisualGeometryToolkitCore: MsacScoring, AbstractOptimizationStrategy, Point2
    using Test

    struct _BareStrategy <: AbstractOptimizationStrategy end
    src = [Point2(10.0i, 7.0j) for i in 1:4 for j in 1:4]
    problem = HomographyProblem(src .=> src)
    sc = MsacScoring(problem; threshold = 2.0)
    for lo in (_BareStrategy(), nothing)
        @test RansacWorkspace(problem, sc, lo) isa RansacWorkspace
        @test lo_workspace(typeof(problem), Float32, lo, 16) isa LoBuffers
    end
    amb = filter(p -> p[1].name in (:lo_workspace, :locally_optimize!, :lo_width) ||
                      p[2].name in (:lo_workspace, :locally_optimize!, :lo_width),
                 Test.detect_ambiguities(VgtkCore, VisualGeometryToolkitEstimation))
    @test isempty(amb)
end
