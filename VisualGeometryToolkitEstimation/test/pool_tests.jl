@testitem "draw_hypotheses and PoolPair" begin
    import VisualGeometryToolkitCore as VgtkCore
using Test
using LinearAlgebra
using Random
using Statistics: median
using StaticArrays
using Rotations: RotXYZ
using VisualGeometryToolkitCore
using VisualGeometryToolkitCore: MsacScoring
using VisualGeometryToolkitCore: CameraModel, EssentialMat, FundMat, HomographyMat, HomographyMap, Point2, Size2,
    GnomonicFrame, UnlabeledFrame, csponds, sources, targets, sampson_distance, COINCIDENT_PX
using VisualGeometryToolkitCore.Pools: PoolPair, StereoRawPair, pool_pixel_scale, pool_frame_calibrations,
    pool_normalization, num_models, num_points
using VisualGeometryToolkitEstimation
using VisualGeometryToolkitEstimation: draw_hypotheses, Hypotheses, InlierVariance, TrackVariance, UniformSampler,
    RansacConfig, residuals!

# A raw pair of the pool interface (pixels).
struct TestRawPair <: StereoRawPair
    csponds; snn; K1; K2; size1; size2; R_gt; t_gt; gt_inliers
end

const K1 = SA[800.0 0 320; 0 820.0 240; 0 0 1]
const K2 = SA[760.0 0 330; 0 770.0 250; 0 0 1]
const R_GT = SMatrix{3, 3, Float64, 9}(RotXYZ(0.03, -0.12, 0.05))
const T_GT = SA[1.0, 0.1, 0.05]

_px(K, X) = (p = K * X; Point2{Float64}(p[1] / p[3], p[2] / p[3]))

# `n` pairs (fraction `out` gross outliers): a planar scene (Z = 4 in view 1) or
# a box of depths; returns the raw pair and noise-free track points.
function raw_scene(rng; n = 200, out = 0.3, planar = false, σ = 0.5)
    u1 = Point2{Float64}[]; u2 = Point2{Float64}[]; gt = Bool[]
    tA = SVector{2, Float64}[]; tB = SVector{2, Float64}[]
    for _ in 1:n
        X = SA[2 * (rand(rng) - 0.5), 1.5 * (rand(rng) - 0.5), planar ? 4.0 : 3.0 + 2 * rand(rng)]
        Y = R_GT * X + T_GT
        push!(u1, _px(K1, X))
        push!(tA, _px(K1, X)); push!(tB, _px(K2, Y))
        if rand(rng) < out
            push!(u2, Point2{Float64}(640 * rand(rng), 480 * rand(rng))); push!(gt, false)
        else
            q = _px(K2, Y)
            push!(u2, Point2{Float64}(q[1] + σ * randn(rng), q[2] + σ * randn(rng))); push!(gt, true)
        end
    end
    raw = TestRawPair(csponds(u1, u2), rand(rng, Float32, n), K1, K2, Size2(640.0, 480.0), Size2(660.0, 500.0),
        R_GT, T_GT, BitVector(gt))
    return raw, tA, tB
end

@testset "draw_hypotheses is the RANSAC trial" begin
    raw, _, _ = raw_scene(Xoshiro(1); planar = true)
    p = HomographyProblem(raw.csponds; sampler = UniformSampler(200))
    seen = Any[]
    r = ransac(p, MsacScoring(p; threshold = 3.0); rng = Xoshiro(5), config = RansacConfig(max_trials = 400),
        on_hypothesis = (t, m) -> push!(seen, m))
    h = draw_hypotheses(Xoshiro(5), p; target = 10^6, max_trials = 400)
    @test h isa Hypotheses{HomographyMat{Float64}}
    # ransac stops adaptively; its hypotheses are a prefix of the same stream
    @test length(seen) <= length(h.models)
    @test all(isequal(SMatrix{3, 3}(seen[k]), SMatrix{3, 3}(h.models[k])) for k in eachindex(seen))
    @test h.trials == 400
    @test size(h.sample_indices) == (4, length(h.models))
    # stopping rule: `target` accepted samples
    h2 = draw_hypotheses(Xoshiro(5), p; target = 25, max_trials = 400)
    @test h2.samples_accepted == 25 && h2.trials < 400
    @test h2.models == h.models[1:length(h2.models)]      # one stream, cut at the target
    @test draw_hypotheses(Xoshiro(5), p; target = 25, max_trials = 400).sample_indices == h2.sample_indices
    @test h2.n_solutions >= length(h2.models)
    @test_throws ArgumentError draw_hypotheses(Xoshiro(5), p; target = -1, max_trials = 10)
end

@testset "PoolPair: E, F, H into the pool frame" begin
    for kind in (:E, :F, :H)
        raw, tA, tB = raw_scene(Xoshiro(2); planar = kind === :H, σ = 0.3)
        p, var = if kind === :E
            EssentialProblem(CameraModel(K1), CameraModel(K2), raw.csponds), TrackVariance(tA, tB)
        elseif kind === :F
            FundMatProblem(raw.csponds), InlierVariance(; min_inliers = 8)
        else
            C1 = HomographyMap{UnlabeledFrame{2}, GnomonicFrame{:src}}(SMatrix{3, 3, Float64, 9}(inv(K1)))
            C2 = HomographyMap{UnlabeledFrame{2}, GnomonicFrame{:dst}}(SMatrix{3, 3, Float64, 9}(inv(K2)))
            HomographyProblem(raw.csponds; src_map = C1, dst_map = C2,
                model_frames = (GnomonicFrame{:src}, GnomonicFrame{:dst})), InlierVariance(; min_inliers = 5)
        end
        pair, R = PoolPair(Xoshiro(3), p, raw; target = 60, max_trials = 5000, variance = var)
        @test pair isa PoolPair{kind === :H ? HomographyMat{Float64} : FundMat{Float64}}
        @test size(R) == (num_points(pair), num_models(pair)) && eltype(R) === Float32
        @test size(pair.sample_indices, 2) == num_models(pair) > 0
        @test pair.gt_inliers == raw.gt_inliers
        @test isfinite(pair.gt_variance) && pair.gt_variance > 0
        @test all(<(180.0), pair.errors_r)
        # The stored csponds are the raw pixels through the pool normalization.
        N = pool_normalization(raw)
        @test sources(pair.csponds) ≈ VgtkCore.source_map(N).(sources(raw.csponds))
        # The best model is near the GT, and its residuals are pool-frame small on GT inliers.
        j = argmin(max.(pair.errors_r, kind === :H ? 0.0 : pair.errors_t))
        @test pair.errors_r[j] < 2.0
        @test median(R[pair.gt_inliers, j]) < 5 * pool_pixel_scale(pair)
        # The residual matrix is the record's own pool-frame problem's.
        pp = kind === :E ? EssentialProblem(pair) : kind === :F ? FundMatProblem(pair) : HomographyProblem(pair)
        if kind === :E
            @test pp.coincident_tol == COINCIDENT_PX * pool_pixel_scale(pair)
            Kc1, Kc2 = pool_frame_calibrations(pair)
            r = residuals!(zeros(num_points(pair)), pp, EssentialMat(pair.models[j], Kc1, Kc2))
            @test R[:, j] == Float32.(abs.(r))
        else
            @test R[:, j] == [Float32(sampson_distance(pair.models[j], SVector{2, Float64}(a), SVector{2, Float64}(b)))
                              for (a, b) in zip(sources(pair.csponds), targets(pair.csponds))]
        end
        ctor = kind === :E ? EssentialProblem : kind === :F ? FundMatProblem : HomographyProblem
        # E: always 0.1 px in pool units; H and F: their gate reads none
        @test_throws ArgumentError ctor(pair; coincident_tol = 1.0)
    end
end

@testset "PoolPair refusals" begin
    raw, _, _ = raw_scene(Xoshiro(4); planar = true)
    # a mapped homography whose maps name no calibrated model frame
    p = HomographyProblem(raw.csponds; src_map = SMatrix{3, 3, Float64, 9}(inv(K1)), dst_map = SMatrix{3, 3, Float64, 9}(inv(K2)))
    @test_throws ArgumentError PoolPair(Xoshiro(1), p, raw; target = 5, max_trials = 100, variance = InlierVariance(; min_inliers = 5))
    # a problem on other data
    q = HomographyProblem(raw.csponds[1:100])
    @test_throws DimensionMismatch PoolPair(Xoshiro(1), q, raw; target = 5, max_trials = 100, variance = InlierVariance(; min_inliers = 5))
    # a homography has no oracle model, so no track variance
    tv = TrackVariance([SVector(1.0, 2.0)], [SVector(3.0, 4.0)])
    @test_throws ArgumentError PoolPair(Xoshiro(1), HomographyProblem(raw.csponds), raw; target = 5, max_trials = 100, variance = tv)
    # too few GT inliers: σ̂² is NaN, the mask is kept
    pair, _ = PoolPair(Xoshiro(1), HomographyProblem(raw.csponds), raw; target = 5, max_trials = 100,
        variance = InlierVariance(; min_inliers = 10^6))
    @test isnan(pair.gt_variance) && pair.gt_inliers == raw.gt_inliers
end
end
