# =============================================================================
# The find_* results under the robust contract (§2, §12): the scorer's scale
# (`scale`/`scale_source`, read by noise_scale/scale_source), its weights and
# their kind (`weights`, `weight_kind` from RansacMetadata{T, S}'s scorer type
# S, `inlier_mask` cut by the kind), and `n`/`rms` from the weights.
# =============================================================================

@testitem "ransac_weights: a threshold scorer's result is binary, its n and rms the consensus set's" setup = [HomographyScenes] begin
    using Test, StableRNGs, StaticArrays
    using VisualGeometryToolkitCore: csponds, estimate, converged, rms, weights, metadata, MsacScoring, RansacScoring,
        inlier_mask, weight_kind, noise_scale, scale_source, BinaryWeights, ScaleEstimated, ScaleNone
    using VisualGeometryToolkitCore: TLSShape
    using VisualGeometryToolkitEstimation: RansacEstimate, RansacMetadata, HomographyProblem, inlier_ratio,
        find_homography, locally_optimize, Irls

    src, dst = homography_pairs(StableRNG(5), H_TRUE; n_in = 100, n_out = 30, σ = 0.5)
    cs = csponds(src, dst)
    for sc in (MsacScoring(HomographyProblem; threshold = 3.0), RansacScoring(HomographyProblem; threshold = 3.0))
        r = find_homography(cs; scoring = sc, rng = StableRNG(9))
        @test converged(r)
        @test r isa RansacEstimate{<:Any, Float32, typeof(sc)}
        @test metadata(r) isa RansacMetadata{Float32, typeof(sc)}
        @test @inferred(weight_kind(r)) === BinaryWeights()
        w = @inferred weights(r)
        @test w isa Vector{Float32} && all(x -> x == 0 || x == 1, w)
        mask = @inferred inlier_mask(r)
        @test mask isa BitVector && mask == (w .> 0)
        @test inlier_mask(r, 0.5) == mask
        # n is the support, rms the weighted rms: the consensus set's size and rms
        @test r.n == count(mask) == count(>(0), w)
        @test rms(r) ≈ sqrt(sum(abs2, Float64.(r.residuals[mask])) / r.n) rtol = 1.0e-12
        @test inlier_ratio(r) == count(mask) / length(w)
        # the scorer originates the scale
        @test scale_source(r) === ScaleEstimated
        @test @inferred(noise_scale(r)) === r.scale
        @test 0.2 < noise_scale(r) < 1.5
        # the retired result properties
        @test !hasproperty(r, :sigma) && !hasproperty(r, :inlier_mask)
        @test_throws ErrorException r.sigma
    end

    # locally_optimize keeps the scorer: its result is of the scorer's kind too
    sc = MsacScoring(HomographyProblem; threshold = 3.0)
    r = find_homography(cs; scoring = sc, rng = StableRNG(9))
    lo = locally_optimize(HomographyProblem(cs), estimate(r); scoring = sc, local_optimization = Irls{TLSShape}())
    @test metadata(lo) isa RansacMetadata{Float32, typeof(sc)}
    @test weight_kind(lo) === BinaryWeights() && lo.n == count(inlier_mask(lo))
end

@testitem "ransac_weights: the Student-t scorer's result is posterior, its n the support and rms weighted" setup = [HomographyScenes] begin
    using Test, StableRNGs, StaticArrays
    using VisualGeometryToolkitCore: csponds, converged, rms, weights, metadata, is_map_inlier, StudentTUniformScoring,
        JeffreysPrior, FlatLabelPrior, inlier_mask, weight_kind, noise_scale, scale_source, PosteriorWeights, ScaleEstimated
    using VisualGeometryToolkitEstimation: RansacEstimate, RansacMetadata, HomographyProblem, find_homography

    src, dst = homography_pairs(StableRNG(5), H_TRUE; n_in = 100, n_out = 30, σ = 0.5)
    sc = StudentTUniformScoring(HomographyProblem, 20.0; scale_prior = JeffreysPrior(), label_prior = FlatLabelPrior())
    r = find_homography(csponds(src, dst); scoring = sc, rng = StableRNG(9))
    @test converged(r)
    @test metadata(r) isa RansacMetadata{Float32, typeof(sc)}
    @test @inferred(weight_kind(r)) === PosteriorWeights()
    w = weights(r)
    @test all(x -> 0 <= x <= 1, w) && any(x -> 0 < x < 1, w)
    @test all(x -> x == 0 || x >= eps(Float32), w)                  # below eps(T): an exact 0
    @test @inferred(inlier_mask(r)) == is_map_inlier.(w)            # the MAP rule
    @test count(inlier_mask(r)[1:100]) >= 95 && count(inlier_mask(r)[101:end]) <= 3
    # n is the support count(w > 0); rms is √(Σwr²/Σw)
    @test r.n == count(>(0), w)
    wf, rf = Float64.(w), Float64.(r.residuals)
    @test rms(r) ≈ sqrt(sum(wf .* rf .^ 2) / sum(wf)) rtol = 1.0e-12
    @test scale_source(r) === ScaleEstimated && 0.2 < noise_scale(r) < 1.5
end

@testitem "ransac_weights: every find_* result type, and the no-model result" begin
    using Test, StaticArrays, LinearAlgebra
    using VisualGeometryToolkitCore: Point2, CameraModel, Csponds, csponds, weights, metadata, MsacScoring,
        inlier_mask, weight_kind, noise_scale, scale_source, BinaryWeights, ScaleNone, ScaleEstimated
    using VisualGeometryToolkitEstimation: RansacEstimate, RansacMetadata, CircleProblem,
        find_homography, find_fundmat, find_essential, find_rotation, find_circle, find_ellipse,
        find_radial_homography, find_radial_camera, find_absolute_pose

    pts(n) = [Point2(100.0 + 10i, 50.0 + 7i^2) for i in 1:n]
    cs(n) = csponds(pts(n), pts(n) .+ Ref(Point2(1.0, 2.0)))
    cam = CameraModel(; focal = 800.0, pp = (480.0, 360.0))
    c = Point2(500.0, 400.0)
    X = [SA[0.1, 0.2, 2.0], SA[-0.3, 0.1, 2.5]]
    u = [SA[400.0, 300.0], SA[500.0, 350.0]]
    results = (find_homography(cs(3)), find_fundmat(cs(6)), find_essential(cam, cam, cs(4)),
        find_rotation(cam, cam, cs(1)), find_rotation(Csponds([SA[0.0, 0.0, 1.0]], [SA[0.0, 0.0, 1.0]])),
        find_circle(pts(2)), find_ellipse(pts(4)), find_radial_homography(cs(4); center = c),
        find_radial_camera(cs(4); center = c), find_absolute_pose(cam, X .=> u))
    for r in results
        @test r isa RansacEstimate
        @test metadata(r) isa RansacMetadata{Float32, <:MsacScoring}      # every class's scorer of record
        @test @inferred(weight_kind(r)) === BinaryWeights()
        @test @inferred(weights(r)) == zeros(Float32, length(weights(r)))
        @test !any(@inferred inlier_mask(r)) && r.n == 0 && isnan(r.rms)
        @test scale_source(r) === ScaleNone && isnan(r.scale)
        @test_throws ArgumentError noise_scale(r)
    end
    # the constructor is concrete and inferred
    sc = MsacScoring(CircleProblem; threshold = 1.0)
    m = @inferred RansacMetadata{Float32, typeof(sc)}(Float32[1, 2], Float32[1, 0], 1.0, 0.5,
        ScaleEstimated, 3, 1.0, :not_run)
    @test isconcretetype(typeof(m)) && isbitstype(typeof(m.scale_source))
end
