@testitem "LikelihoodRatioSelection: the most likely of several seeded local optimizations" begin
    using Test, StableRNGs, StaticArrays
    using VisualGeometryToolkitCore: solve, RefineProblem, MsacScoring, Irls, TLSShape, HomographyMat, HomographyMap, PerspectiveMap, correspondences,
        found_model, RefineKind, estimator_kind
    using VisualGeometryToolkitEstimation: LikelihoodRatioSelection, LocalOptimization, HomographyProblem

    rng = StableRNG(5)
    H = SMatrix{3, 3, Float64}(1.02, 0.01, 1.0e-5, -0.02, 0.98, 2.0e-5, 3.0, -2.0, 1.0)
    apply = PerspectiveMap() ∘ HomographyMap(H)
    src = [SVector(640 * rand(rng), 480 * rand(rng)) for _ in 1:100]
    tgt = [i <= 80 ? apply(x) + 0.5 * SVector(randn(rng), randn(rng)) : SVector(640 * rand(rng), 480 * rand(rng))
           for (i, x) in enumerate(src)]
    f = HomographyProblem(correspondences(src, tgt))
    lo = LocalOptimization(Irls{TLSShape}(); scoring = MsacScoring(f; threshold = 3.0))
    good = HomographyMat{Float64}(Tuple(H))
    bad = HomographyMat{Float64}(Tuple(SMatrix{3, 3, Float64}(1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 60.0, -40.0, 1.0)))

    @test estimator_kind(LikelihoodRatioSelection(lo)) === RefineKind()
    rg = solve(RefineProblem(f, good), lo)
    for (starts, kept) in (((bad, good), 2), ((good, bad), 1))
        r = solve(RefineProblem(f, starts), LikelihoodRatioSelection(lo))
        @test r.metadata.selected == kept
        @test r.u == rg.u                                          # the good start's refine is kept, in either order
        @test length(r.metadata.candidates) == 2 && r.metadata.candidates[kept].u == rg.u
    end
    # a tie keeps the first: two identical starts
    @test solve(RefineProblem(f, (good, good)), LikelihoodRatioSelection(lo)).metadata.selected == 1
    @test_throws ArgumentError solve(RefineProblem(f, ()), LikelihoodRatioSelection(lo))
end
