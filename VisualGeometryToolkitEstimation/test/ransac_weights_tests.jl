# =============================================================================
# The find_* results under the robust contract (§2, §12): the scorer's scale
# (`scale`/`scale_source`, read by noise_scale/scale_source), its weights and
# their kind (`weights`, `weight_kind` from RansacMetadata{T, S}'s scorer type
# S, `inlier_mask` cut by the kind), and `n`/`rms` from the weights.
# =============================================================================

@testitem "ransac_weights: a threshold scorer's result is binary, its n and rms the consensus set's" setup = [HomographyScenes] begin
    using VisualGeometryToolkitCore: solve, EstimationProblem, RefineProblem, successful_retcode
    using VisualGeometryToolkitEstimation: Ransac, LocalOptimization
    using Test, StableRNGs, StaticArrays
    using VisualGeometryToolkitCore: correspondences, rms, weights, MsacScoring, RansacScoring, inlier_mask, weight_kind, noise_scale, scale_source, BinaryWeights, ScaleEstimated, ScaleNone
    using VisualGeometryToolkitCore: TLSShape
    using VisualGeometryToolkitEstimation: RansacEstimate, RansacMetadata, HomographyProblem, inlier_ratio
    using VisualGeometryToolkitCore: Irls

    src, dst = homography_pairs(StableRNG(5), H_TRUE; n_in = 100, n_out = 30, σ = 0.5)
    cs = correspondences(src, dst)
    for sc in (MsacScoring(HomographyProblem; threshold = 3.0), RansacScoring(HomographyProblem; threshold = 3.0))
        r = solve(EstimationProblem(HomographyProblem(cs)), Ransac(; scoring = sc); rng = StableRNG(9))
        @test successful_retcode(r)
        @test r isa RansacEstimate{<:Any, Float32, typeof(sc)}
        @test r.metadata isa RansacMetadata{Float32, typeof(sc)}
        @test @inferred(weight_kind(r)) === BinaryWeights()
        w = @inferred weights(r)
        @test w isa Vector{Float32} && all(x -> x == 0 || x == 1, w)
        mask = @inferred inlier_mask(r)
        @test mask isa BitVector && mask == (w .> 0)
        @test inlier_mask(r, 0.5) == mask
        # n is the support, rms the weighted rms: the consensus set's size and rms
        @test r.n == count(mask) == count(>(0), w)
        @test r.rms ≈ sqrt(sum(abs2, Float64.(r.resid[mask])) / r.n) rtol = 1.0e-12
        @test inlier_ratio(r) == count(mask) / length(w)
        # the scorer originates the scale
        @test scale_source(r) === ScaleEstimated
        @test @inferred(noise_scale(r)) isa Float64 && noise_scale(r) > 0
        @test 0.2 < noise_scale(r) < 1.5
        # the retired result properties
        @test !hasproperty(r, :sigma) && !hasproperty(r, :inlier_mask)
        @test_throws FieldError r.sigma   # Julia 1.12+: a missing field is a FieldError (asserting ErrorException is deprecated)
    end

    # LocalOptimization keeps the scorer: its result is of the scorer's kind too
    sc = MsacScoring(HomographyProblem; threshold = 3.0)
    r = solve(EstimationProblem(HomographyProblem(cs)), Ransac(; scoring = sc); rng = StableRNG(9))
    lo = solve(RefineProblem(HomographyProblem(cs), r.u), LocalOptimization(Irls{TLSShape}(); scoring = sc))
    @test lo.metadata isa RansacMetadata{Float32, typeof(sc)}
    @test weight_kind(lo) === BinaryWeights() && lo.n == count(inlier_mask(lo))
end

@testitem "ransac_weights: the Student-t scorer's result is posterior, its n the support and rms weighted" setup = [HomographyScenes] begin
    using VisualGeometryToolkitCore: solve, EstimationProblem, successful_retcode
    using VisualGeometryToolkitEstimation: Ransac
    using Test, StableRNGs, StaticArrays
    using VisualGeometryToolkitCore: correspondences, rms, weights, is_map_inlier, StudentTUniformScoring, JeffreysPrior, FlatLabelPrior, inlier_mask, weight_kind, noise_scale, scale_source, PosteriorWeights, ScaleEstimated
    using VisualGeometryToolkitEstimation: RansacEstimate, RansacMetadata, HomographyProblem

    src, dst = homography_pairs(StableRNG(5), H_TRUE; n_in = 100, n_out = 30, σ = 0.5)
    sc = StudentTUniformScoring(HomographyProblem, 20.0; scale_prior = JeffreysPrior(), label_prior = FlatLabelPrior())
    r = solve(EstimationProblem(HomographyProblem(correspondences(src, dst))), Ransac(; scoring = sc); rng = StableRNG(9))
    @test successful_retcode(r)
    @test r.metadata isa RansacMetadata{Float32, typeof(sc)}
    @test @inferred(weight_kind(r)) === PosteriorWeights()
    w = weights(r)
    @test all(x -> 0 <= x <= 1, w) && any(x -> 0 < x < 1, w)
    @test all(x -> x == 0 || x >= eps(Float32), w)                  # below eps(T): an exact 0
    @test @inferred(inlier_mask(r)) == is_map_inlier.(w)            # the MAP rule
    @test count(inlier_mask(r)[1:100]) >= 95 && count(inlier_mask(r)[101:end]) <= 3
    # n is the support count(w > 0); rms is √(Σwr²/Σw)
    @test r.n == count(>(0), w)
    wf, rf = Float64.(w), Float64.(r.resid)
    @test r.rms ≈ sqrt(sum(wf .* rf .^ 2) / sum(wf)) rtol = 1.0e-12
    @test scale_source(r) === ScaleEstimated && 0.2 < noise_scale(r) < 1.5
end

@testitem "ransac_weights: every find_* result type, and the no-model result" begin
    using VisualGeometryToolkitCore: solve, EstimationProblem
    using VisualGeometryToolkitEstimation: Ransac, HomographyProblem, FundMatProblem, EssentialProblem, AbsolutePoseProblem, RadialHomographyProblem, RadialCameraProblem, RotationProblem, EllipseProblem
    using Test, StaticArrays, LinearAlgebra
    using VisualGeometryToolkitCore: Point2, CameraModel, Correspondences, correspondences, weights, MsacScoring, inlier_mask, weight_kind, noise_scale, scale_source, BinaryWeights, ScaleNone, ScaleEstimated, ReturnCode, NoRule
    using VisualGeometryToolkitEstimation: RansacEstimate, RansacMetadata, CircleProblem

    # a few distinct points, fewer than every class's minimal sample: each search returns its no-model result
    ring = [Point2(300 + 120cospi(0.29k), 240 + 90sinpi(0.29k)) for k in 1:6]
    pts(n) = ring[1:n]
    cs(n) = correspondences(ring[1:n], reverse(ring)[1:n])
    cam = CameraModel(; focal = 800.0, pp = (480.0, 360.0))
    c = Point2(500.0, 400.0)
    X = [SA[0.1, 0.2, 2.0], SA[-0.3, 0.1, 2.5]]
    u = [SA[400.0, 300.0], SA[500.0, 350.0]]
    results = (solve(EstimationProblem(HomographyProblem(cs(3))), Ransac()), solve(EstimationProblem(FundMatProblem(cs(6))), Ransac()), solve(EstimationProblem(EssentialProblem(cam, cam, cs(4))), Ransac()),
        solve(EstimationProblem(RotationProblem(cam, cam, cs(1))), Ransac()),
        solve(EstimationProblem(CircleProblem(pts(2))), Ransac()), solve(EstimationProblem(EllipseProblem(pts(4))), Ransac()), solve(EstimationProblem(RadialHomographyProblem(cs(4); center = c)), Ransac()),
        solve(EstimationProblem(RadialCameraProblem(cs(4); center = c)), Ransac()), solve(EstimationProblem(AbsolutePoseProblem(cam, X .=> u)), Ransac()))
    for r in results
        @test r isa RansacEstimate
        @test r.metadata isa RansacMetadata{Float32, <:MsacScoring}      # every class's scorer of record
        @test @inferred(weight_kind(r)) === BinaryWeights()
        @test @inferred(weights(r)) == zeros(Float32, length(weights(r)))
        @test !any(@inferred inlier_mask(r)) && r.n == 0 && isnan(r.rms)
        @test scale_source(r) === ScaleNone
        @test_throws ArgumentError noise_scale(r)
    end
    # the constructor is concrete and inferred
    sc = MsacScoring(CircleProblem; threshold = 1.0)
    m = @inferred RansacMetadata{Float32, typeof(sc)}(Float32[1, 0], 1.0, 0.5,
        ScaleEstimated, 3, 1.0, ReturnCode.Default, NoRule)
    @test isconcretetype(typeof(m)) && isbitstype(typeof(m.scale_source))
end

@testitem "ransac_weights: PooledScale pools the results' scales by their degrees of freedom" setup = [HomographyScenes] begin
    using VisualGeometryToolkitCore: dof
    using VisualGeometryToolkitCore: solve, EstimationProblem
    using VisualGeometryToolkitEstimation: Ransac
    using VisualGeometryToolkitCore: retcode_symbol
    using Test, StableRNGs
    import VisualGeometryToolkitCore as VgtkCore
    using VisualGeometryToolkitCore: correspondences, noise_scale, scale_source, MsacScoring, ScaleEstimated, ScaleNone, ScaleUpstream, scale_input
    using VisualGeometryToolkitEstimation: HomographyProblem
    using VisualGeometryToolkitEstimation: PooledScaleMetadata, PooledScale
    using VisualGeometryToolkitCore: NoiseScale, successful_retcode
    pool(results) = solve(EstimationProblem(NoiseScale, results), PooledScale())
    pool(s, ν) = solve(EstimationProblem(NoiseScale, s => ν), PooledScale())

    # three views of one noise model at σ = 0.5 px: the pool is √(Σ νᵢ ŝᵢ² / Σ νᵢ), between the views' scales
    rs = [solve(EstimationProblem(HomographyProblem(correspondences(homography_pairs(StableRNG(s), H_TRUE; n_in = n, n_out = 20, σ = 0.5)...))), Ransac(; scoring = MsacScoring(HomographyProblem; threshold = 3.0)); rng = StableRNG(s + 1)) for (s, n) in ((1, 60), (2, 120), (3, 240))]
    @test all(successful_retcode, rs)
    p = pool(rs)
    ν = [dof(r) for r in rs]; s = [noise_scale(r) for r in rs]
    @test p.retcode == VgtkCore.ReturnCode.Success && p.stop_rule == VgtkCore.ClosedForm && p.metadata isa PooledScaleMetadata
    VgtkCore.test_termination_record(p)
    # type-stable: the pool, its metadata and its scale
    @test @inferred(pool(rs)) isa VgtkCore.FitResult{Float64, PooledScaleMetadata}
    @test @inferred(PooledScaleMetadata(0.5, ScaleEstimated, 10, 2, 0)) isa PooledScaleMetadata
    @test @inferred(noise_scale(p)) isa Float64
    @test p.u ≈ sqrt(sum(ν .* s .^ 2) / sum(ν)) rtol = 1.0e-14
    @test noise_scale(p) == p.u && scale_source(p) === ScaleEstimated
    @test minimum(s) <= noise_scale(p) <= maximum(s)
    @test VgtkCore.dof(p) == sum(ν) && p.metadata.pooled == 3 && p.metadata.left_out == 0
    # the hand-off: a downstream stage given the pool records ScaleUpstream
    @test scale_input(p, nothing).source === ScaleUpstream && scale_input(p, nothing).scale == noise_scale(p)
    # one view's pool is its own scale
    @test noise_scale(pool(rs[1:1])) ≈ noise_scale(rs[1]) rtol = 1.0e-14
    # a failed search (no scale) is left out and counted; none left: degenerate, no scale handed on
    failed = solve(EstimationProblem(HomographyProblem(correspondences(homography_pairs(StableRNG(7), H_TRUE; n_in = 3, n_out = 0, σ = 0.5)...))), Ransac(; scoring = MsacScoring(HomographyProblem; threshold = 3.0)); rng = StableRNG(8))
    @test scale_source(failed) === ScaleNone
    q = pool([rs; failed])
    @test noise_scale(q) ≈ noise_scale(p) rtol = 1.0e-14
    @test q.metadata.left_out == 1
    d = pool([failed])
    @test retcode_symbol(d) === :degenerate && isnan(d.u) && scale_source(d) === ScaleNone
    @test_throws ArgumentError noise_scale(d)
    # the same pool from the recorded numbers (each result's scale and dof, as a record on disk holds them): the one body,
    # bit-identical; a NaN scale (none recorded) or a dof ≤ 0 is left out and counted
    @test @inferred(pool(s, ν)) isa VgtkCore.FitResult{Float64, PooledScaleMetadata}
    @test pool(s, ν).u === p.u && pool(s, ν).metadata == p.metadata
    r = pool([s; NaN; 0.7], [ν; 12; 0])
    @test r.u === p.u && r.metadata.left_out == 2 && r.metadata.pooled == 3
    @test retcode_symbol(pool([NaN], [5])) === :degenerate
    @test_throws DimensionMismatch pool(s, ν[1:2])
end
