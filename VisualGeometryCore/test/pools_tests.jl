# Pools: the pool record PoolPair (stored in NormalizedImageFrame) with its
# accessors, and the raw-pair interface StereoRawPair (src/pools/).

@testitem "pools" begin
    using Test
    using VisualGeometryCore
    using VisualGeometryCore.Pools
    using VisualGeometryCore.Pools: rig
    using VisualGeometryCore: CspondNormalization, FramedAffine, similarity_scale, source_map, target_map
    using StaticArrays, LinearAlgebra, Random
    using Rotations: RotY, RotZ

    rng = Xoshiro(7)
    sz1 = Size2(width = 1920.0, height = 1080.0)
    sz2 = Size2(width = 1600.0, height = 1200.0)
    K1 = SMatrix{3, 3, Float64}(1500.0, 0, 0, 0, 1510.0, 0, 960.0, 540.0, 1.0)
    K2 = SMatrix{3, 3, Float64}(1300.0, 0, 0, 0, 1300.0, 0, 800.0, 600.0, 1.0)
    R = SMatrix{3, 3, Float64}(RotZ(0.1) * RotY(-0.2))
    t = SVector(1.0, 0.1, -0.2)
    n, m = 12, 5
    px1 = [SVector(rand(rng) * 1920, rand(rng) * 1080) for _ in 1:n]
    px2 = [SVector(rand(rng) * 1600, rand(rng) * 1200) for _ in 1:n]
    N = pool_normalization(sz1, sz2)
    S, D = source_map(N), target_map(N)
    cs = csponds(S.(px1), D.(px2))                       # the stored pool-frame csponds
    models = [FundMat(SMatrix{3, 3, Float64}(randn(rng, 9)...)) for _ in 1:m]
    idx = rand(rng, 1:n, 7, m)
    er, et = rand(rng, m) .* 10, rand(rng, m) .* 10
    gt = BitVector(rand(rng, Bool, n))
    snn = rand(rng, Float32, n)
    args = (cs, K1, K2, R, t, models, idx, er, et, gt, 0.25)
    pair = PoolPair(args..., N, snn)

    @testset "the record and its maps" begin
        @test pair isa PoolPair{eltype(models)}
        @test pair.source_map == to_homogeneous(S) && pair.target_map == to_homogeneous(D)
        @test PoolPair(args..., pair.source_map, pair.target_map, snn).source_map == pair.source_map
        @test PoolPair(args..., N, nothing).snn === nothing
        # the keyword form: every field named, the maps as both sides or as N
        kw = (; csponds = cs, K1, K2, R_gt = R, t_gt = t, models, sample_indices = idx,
              errors_r = er, errors_t = et, gt_inliers = gt, gt_variance = 0.25)
        kpair = PoolPair(; kw..., normalization = N, snn)
        @test kpair isa PoolPair{eltype(models)}
        @test kpair.source_map == pair.source_map && kpair.target_map == pair.target_map && kpair.snn == snn
        @test PoolPair{eltype(models)}(; kw..., source_map = pair.source_map, target_map = pair.target_map).snn === nothing
        @test_throws ArgumentError PoolPair(; kw...)
        @test_throws ArgumentError PoolPair(; kw..., source_map = pair.source_map)
        @test_throws ArgumentError PoolPair(; kw..., normalization = N, source_map = pair.source_map)
        # the stored csponds are concretely typed; a vector of pairs is converted
        @test isconcretetype(fieldtype(PoolPair{eltype(models)}, :csponds))
        @test (@inferred (p -> p.csponds)(pair)) == cs
        @test PoolPair(; kw..., csponds = collect(cs), normalization = N).csponds == cs
        @test num_points(pair) == n
        @test num_models(pair) == m
        # the pool frame from the stored maps reproduces the size construction
        Np = pool_normalization(pair)
        @test Np isa CspondNormalization
        @test to_homogeneous(source_map(Np)) == pair.source_map
        @test to_homogeneous(target_map(Np)) == pair.target_map
    end

    @testset "a record without its maps is refused" begin
        @test_throws ArgumentError PoolPair(args...)
        @test_throws ArgumentError PoolPair{eltype(models)}(args...)
        bad = SMatrix{3, 3, Float64}(1, 0, 0.1, 0, 1, 0, 0, 0, 1)   # projective last row
        @test_throws ArgumentError PoolPair(args..., bad, pair.target_map, nothing)
        @test_throws DimensionMismatch PoolPair(args..., N, rand(Float32, n + 1))
        @test_throws DimensionMismatch PoolPair(cs, K1, K2, R, t, models, idx[:, 1:(m - 1)], er, et, gt, 0.25, N, nothing)
        @test_throws DimensionMismatch PoolPair(cs, K1, K2, R, t, models, idx, er, et, gt[1:(n - 1)], 0.25, N, nothing)
    end

    @testset "accessors" begin
        @test calibration_matrices(pair) == (K1, K2)
        @test gt_pose(pair) == (R, t)
        @test pool_imc(pair, 2) == max(er[2], et[2])
        Hpair = PoolPair(cs, K1, K2, R, t, [HomographyMat(SMatrix{3, 3, Float64}(I)) for _ in 1:m],
            idx, er, et, gt, 0.25, N, nothing)
        @test pool_imc(Hpair, 2) == er[2]               # a homography's translation is not scored
        Kc1, Kc2 = pool_frame_calibrations(pair)
        @test Kc1 == pair.source_map * K1 && Kc2 == pair.target_map * K2
        r = rig(pair)
        @test r isa StereoRig
        @test calibration_matrix(source(r)) ≈ K1
        @test calibration_matrix(target(r)) ≈ K2
        @test extrinsics(r).R ≈ R && extrinsics(r).t ≈ t
        @test ids(r) == (:cam1, :cam2)
    end

    @testset "frame-typed maps, the pixel scale, and the way back to pixels" begin
        Sf, Df = pool_frame_maps(pair)
        @test Sf isa FramedAffine{ImageFrame{:cam1}, NormalizedImageFrame{:cam1}, Float64}
        @test Df isa FramedAffine{ImageFrame{:cam2}, NormalizedImageFrame{:cam2}, Float64}
        # the same map as the size factory's NormalizedImageFrame map
        @test Sf.A == FramedAffine(ImageFrame{:cam1}, NormalizedImageFrame{:cam1}, sz1).A
        @test Sf.t == FramedAffine(ImageFrame{:cam1}, NormalizedImageFrame{:cam1}, sz1).t
        @test Sf(px1[3]) == S(px1[3])
        # pool units per pixel: the denser view's inverse half-diagonal
        @test pool_pixel_scale(pair) == max(similarity_scale(Sf), similarity_scale(Df))
        @test pool_pixel_scale(pair) ≈ 1 / (hypot(1600, 1200) / 2)
        back = pixel_csponds(pair)
        @test length(back) == n
        @test all(i -> sources(back)[i] ≈ px1[i] && targets(back)[i] ≈ px2[i], 1:n)
    end

    # A minimal reader: the field-backed defaults, and the pool frame of a raw pair.
    struct _TestRawPair <: StereoRawPair
        csponds
        snn::Vector{Float32}
        K1::SMatrix{3, 3, Float64, 9}
        K2::SMatrix{3, 3, Float64, 9}
        size1::Size2{Float64}
        size2::Size2{Float64}
        R_gt::SMatrix{3, 3, Float64, 9}
        t_gt::SVector{3, Float64}
        gt_inliers::BitVector
    end

    @testset "the raw-pair interface" begin
        raw = _TestRawPair(csponds(px1, px2), snn, K1, K2, sz1, sz2, R, t, gt)
        @test pair_csponds(raw) === raw.csponds
        @test pair_snn(raw) === snn
        @test (pair_K1(raw), pair_K2(raw)) == (K1, K2)
        @test (pair_size1(raw), pair_size2(raw)) == (sz1, sz2)
        @test (pair_R_gt(raw), pair_t_gt(raw)) == (R, t)
        @test pair_gt_inliers(raw) === gt
        @test pair_scene(raw) === nothing
        Nr = pool_normalization(raw)
        @test to_homogeneous(source_map(Nr)) == to_homogeneous(S)
        @test to_homogeneous(target_map(Nr)) == to_homogeneous(D)
    end
end
