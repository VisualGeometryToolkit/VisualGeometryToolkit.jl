# =============================================================================
# raw_readers_tests.jl — the raw dataset readers (src/datasets): THE SNN filter
# (`snn < threshold`), THE seeded per-scene pair subsample, THE row-major 3×4
# [R | t] parser, and the HEB reader end to end on a file written here.
# =============================================================================

@testitem "raw_readers" begin
    using Test, Random, StaticArrays, LinearAlgebra
    using StableRNGs: StableRNG
    using Rotations: RotationVec
    using HDF5: h5open
    using VisualGeometryEval.Stats: stable_seed
    using VisualGeometryEval.Datasets: load_heb_pairs, HebPair
    using VisualGeometryCore.Pools: StereoRawPair, pair_csponds, pair_snn, pair_K1, pair_R_gt, pair_t_gt,
        pair_gt_inliers, pair_scene
    const _D = VisualGeometryEval.Datasets
    rng = MersenneTwister(20261001)

    @testset "SNN filter: strict `<`, csponds and SNN of the kept columns" begin
        m = 100 .* rand(rng, 4, 6)
        snn = Float32[0.1, 0.7, 0.69, 0.9, 0.0, 0.7000001]
        cs, kept_snn, kept = _D._snn_filtered_matches(m, snn, 0.7)
        @test kept == [1, 3, 5]                     # a match AT the threshold is dropped
        @test kept_snn == snn[kept]
        @test cs.first == [SVector{2,Float64}(m[1, i], m[2, i]) for i in kept]
        @test cs.second == [SVector{2,Float64}(m[3, i], m[4, i]) for i in kept]
        @test isempty(_D._snn_filtered_matches(m, snn, 0.0)[3])
    end

    @testset "seeded pair subsample: the overdraw, order kept, all keys under the cap" begin
        keys = ["k$(lpad(i, 4, '0'))" for i in 1:500]
        sub = _D._seeded_key_subsample(keys, "scene"; max_pairs_per_scene = 100, subsample_overdraw = 2,
                                       subsample_seed = 7)
        @test length(sub) == 200 && issorted(sub) && sub ⊆ keys
        # the draw: a StableRNG shuffle keyed by the scene NAME (Julia-version stable)
        order = shuffle(StableRNG(stable_seed(7, "scene")), 1:500)[1:200]
        @test sub == keys[sort!(order)]
        @test _D._seeded_key_subsample(keys, "scene"; max_pairs_per_scene = 500, subsample_overdraw = 2,
                                       subsample_seed = 7) === keys
    end

    @testset "row-major [R | t]" begin
        R = SMatrix{3,3,Float64,9}(RotationVec(0.1, -0.3, 0.2)); t = SVector(1.0, -2.0, 0.5)
        stream = vec(permutedims(hcat(R, t)))        # R₁₁ R₁₂ R₁₃ t₁ R₂₁ …
        @test _D._row_major_pose(stream) == (R, t)
        # HEB: HDF5.jl returns the row-major buffer transposed (4×3 = [Rᵀ; tᵀ])
        @test _D._row_major_pose(vec(Matrix(vcat(R', t')))) == (R, t)
        @test_throws ArgumentError _D._row_major_pose(stream[1:11])
    end

    @testset "HEB reader: a scene file written here" begin
        R = SMatrix{3,3,Float64,9}(RotationVec(0.05, 0.2, -0.1)); t = SVector(0.3, -0.1, 1.0)
        K1 = SMatrix{3,3,Float64,9}(800.0, 0, 0, 0, 810.0, 0, 320.0, 240.0, 1)
        K2 = SMatrix{3,3,Float64,9}(700.0, 0, 0, 0, 700.0, 0, 300.0, 200.0, 1)
        path = joinpath(mktempdir(), "Alamo_homographies.h5")
        N = 40
        h5open(path, "w") do f
            for (j, sub) in enumerate(("0", "1", "2"))
                c = zeros(10, N)
                c[1:4, :] .= 100 .* rand(rng, 4, N)
                c[9, :] .= range(0.05, 0.95; length = N)          # SNN: half below 0.5
                c[10, :] .= isodd.(1:N)                           # GT flags
                f["corr_a_b_c_d_e_f_$sub"] = c
                f["pose_a_b_c_d_e_f_$sub"] = Matrix(vcat(R', t'))  # row-major [R | t] as h5py writes it
            end
            f["corr_x_y_z_d_e_f_0"] = zeros(10, N)               # no pose / K: never a pair
            f["K_a_b_c"] = Matrix(K1'); f["K_d_e_f"] = Matrix(K2')
            f["size_a_b_c"] = [640, 480]; f["size_d_e_f"] = [600, 400]
        end
        pairs = load_heb_pairs(path; min_csponds = 5, max_pairs_per_scene = 2, subsample_overdraw = 2,
                               subsample_seed = 42, snn_threshold = 0.5)
        @test length(pairs) == 2 && all(p -> p isa HebPair && p isa StereoRawPair, pairs)
        p = first(pairs)
        @test pair_R_gt(p) ≈ R && pair_t_gt(p) == t && pair_K1(p) == K1
        @test all(<(0.5f0), pair_snn(p)) && length(pair_csponds(p)) == length(pair_snn(p)) == 20
        @test pair_gt_inliers(p) == BitVector(isodd.(1:20))
        @test pair_scene(p) === nothing
        @test p.size1.width == 640.0 && p.size2.height == 400.0
        @test isempty(load_heb_pairs(path; min_csponds = 100, max_pairs_per_scene = 2, subsample_overdraw = 2,
                                     subsample_seed = 42, snn_threshold = 0.5))
    end

    @testset "RoMa: the retired Uncertain struct reads as its point" begin
        # The consolidated RoMa files store their endpoints as VisualGeometryCore's
        # former `Uncertain` struct (fields `value`, `matrix`); `Uncertain` is now an
        # alias of `Attributed`, so the reader maps the stored name to
        # `_LegacyUncertain` and keeps the point, in Float64, as `centroid` did.
        up = _D._LEGACY_TYPEMAP["VisualGeometryCore.Uncertain"]
        @test up isa _D.JLD2.Upgrade && up.target === _D._LegacyUncertain
        p32 = SVector{2, Float32}(445.39493f0, 3579.3875f0)
        lu = _D.JLD2.rconvert(_D._LegacyUncertain, (value = p32, matrix = nothing))
        @test _D._legacy_p2(lu) === SVector{2, Float64}(p32)
        @test _D._legacy_p2(p32) === SVector{2, Float64}(p32)       # an SP+LG point reads the same
    end
end
