# =============================================================================
# One optimization-strategy family, two keywords (the owner's ruling, 2026-11-11):
# find_*/ransac/locally_optimize take `local_optimization_strategy`, refine_*
# and fit_* `optimization_strategy`; a retired keyword is refused naming its
# successor; a fit takes Irls{K} or `nothing`, never a Triggs.
# =============================================================================

@testitem "strategy keywords: the new names, the retired ones refused" begin
    using VisualGeometryToolkitEstimation: find_homography, refine_homography, fit_linear, Irls, Triggs,
        optimization_strategy_from_toml
    using VisualGeometryToolkitCore: TLSShape, WelschShape, GaussianShape, HuberShape, estimate, Point2
    using StableRNGs: StableRNG
    using StaticArrays: SVector

    H = [1.1 0.05 3.0; -0.02 0.95 -2.0; 1.0e-4 -2.0e-4 1.0]
    src = [Point2(10.0i, 7.0j) for i in 1:6 for j in 1:6]
    dst = map(p -> (h = H * SVector(p[1], p[2], 1.0); Point2(h[1] / h[3], h[2] / h[3])), src)
    pairs = src .=> dst

    r = find_homography(pairs; local_optimization_strategy = Irls{TLSShape}(), rng = StableRNG(1))
    @test estimate(r) !== nothing
    @test estimate(find_homography(pairs; local_optimization_strategy = nothing, rng = StableRNG(1))) !== nothing
    @test_throws r"`local_optimization` is retired: give `local_optimization_strategy`" find_homography(
        pairs; local_optimization = Irls{TLSShape}(), rng = StableRNG(1))

    @test estimate(refine_homography(estimate(r), pairs; optimization_strategy = Irls{TLSShape}())) !== nothing
    @test_throws r"give `optimization_strategy`" refine_homography(estimate(r), pairs; local_optimization_strategy = Irls{TLSShape}())
    @test_throws r"give `optimization_strategy`" refine_homography(estimate(r), pairs; local_optimization = Irls{TLSShape}())

    A = [1.0 0.0; 0.0 1.0; 1.0 1.0; 2.0 1.0]; y = A * [0.5, -1.0]
    @test estimate(fit_linear(A, y; optimization_strategy = Irls{HuberShape}())) ≈ [0.5, -1.0]
    @test estimate(fit_linear(A, y; optimization_strategy = nothing)) ≈ [0.5, -1.0]        # plain least squares
    @test_throws r"needs a starting guess" fit_linear(A, y; optimization_strategy = Triggs{WelschShape}(efficiency_constant = 2.5))
    @test_throws r"give `optimization_strategy`" fit_linear(A, y; local_optimization = Irls{HuberShape}())

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
