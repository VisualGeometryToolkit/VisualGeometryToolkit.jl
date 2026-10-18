# =============================================================================
# stereo_datasets_tests.jl — the depth-validated stereo datasets and the
# consolidated-csponds producer (src/datasets/stereo_datasets.jl, eth3d.jl,
# kitti.jl, scannet1500.jl, seven_scenes.jl, sparse_matches.jl,
# consolidate.jl): the interface is whole, every reader against the layout it
# reads (written here), and `consolidate_csponds` end to end on a two-frame
# 7-Scenes-shaped tree whose geometry is known, read back by `load_roma_pairs`.
# =============================================================================

@testitem "stereo_datasets" begin
    import VisualGeometryToolkitCore as VgtkCore
    import VisualGeometryToolkitEval as VgtkEval
    using Test, Random, StaticArrays, LinearAlgebra
    using Rotations: RotationVec
    using HDF5: h5open, attrs
    using PNGFiles: PNGFiles
    using VisualGeometryToolkitEval.Datasets: AbstractStereoDataset, ETH3D, ScanNet1500, KITTI, SevenScenes,
        StereoPair, GTConfig, stereo_pairs, load_validation_data, gt_config, consolidate_csponds,
        load_sparse_matches, load_roma_pairs, CONSOLIDATED_FORMAT
    import VisualGeometryToolkitCore
    const _D = VgtkEval.Datasets
    rng = MersenneTwister(20261009)

    @testset "the interface: every dataset can drive consolidate_csponds" begin
        for T in (ScanNet1500, ETH3D, KITTI, SevenScenes)
            @test hasmethod(stereo_pairs, Tuple{T})
            @test hasmethod(load_validation_data, Tuple{T, NamedTuple, Any})
            @test any(m -> hasmethod(gt_config, Tuple{T, Val{m}}), (:roma, :splg))
        end
        @test length(methods(stereo_pairs)) == 4
        # every (dataset, matcher) table of configs/default.toml
        for (ds, m) in ((ScanNet1500(""), :roma), (ScanNet1500(""), :splg), (ETH3D(scenes_root = ""), :roma),
                        (ETH3D(scenes_root = ""), :splg), (SevenScenes(root = ""), :roma),
                        (KITTI(root = ""), :roma), (KITTI(root = ""), :splg))
            g = gt_config(ds, m)
            @test g isa GTConfig && g.rel_tol > 0
        end
        @test gt_config(ETH3D(scenes_root = ""), :splg).metric_tol == 68.6 * VgtkCore.mm
        @test_throws MethodError gt_config(SevenScenes(root = ""), :splg)   # not calibrated
        @test ETH3D(scenes_root = "").min_covisibility == 200             # the published inputs' value
        @test length(_D._seven_scenes_sequences(:test)) == 18 && length(_D._seven_scenes_sequences(:train)) == 28
        @test_throws ArgumentError _D._seven_scenes_sequences(:val)
        @test_throws ArgumentError SevenScenes(root = mktempdir(), scene = "kitchen") |> stereo_pairs
    end

    @testset "frame pairs: (i, i + interval), i = 1, 1 + stride, …" begin
        @test _D._consecutive_pose_pairs(5; interval = 1, stride = 1) == [(1, 2), (2, 3), (3, 4), (4, 5)]
        @test _D._consecutive_pose_pairs(120; interval = 50, stride = 10) == [(i, i + 50) for i in 1:10:70]
        @test isempty(_D._consecutive_pose_pairs(50; interval = 50, stride = 10))
        @test_throws ArgumentError _D._consecutive_pose_pairs(5; interval = 0, stride = 1)
    end

    @testset "nearest resampling: outer corners aligned, round half up" begin
        A = reshape(collect(1:6), 2, 3)
        @test _D._nearest_resize(A, (2, 3)) == A
        @test _D._nearest_resize(A, (4, 6)) == A[[1, 1, 2, 2], [1, 1, 2, 2, 3, 3]]
        # ScanNet's 480×640 → 968×1296: rows 1 and 968 map onto the first and last row
        r = _D._nearest_resize(reshape(collect(1:480), 480, 1), (968, 1))
        @test r[1] == 1 && r[end] == 480 && issorted(vec(r))
    end

    @testset "KITTI calib, poses and velodyne; ETH3D depth; depth PNGs" begin
        R = SMatrix{3,3,Float64,9}(RotationVec(0.1, -0.3, 0.2)); t = SVector(1.0, -2.0, 0.5)
        stream = vec(permutedims(hcat(R, t)))                    # R₁₁ R₁₂ R₁₃ t₁ R₂₁ …
        K = @SMatrix [700.0 0 600; 0 700 180; 0 0 1]
        mktempdir() do d
            open(joinpath(d, "calib.txt"), "w") do io
                println(io, "P0: ", join(vec(permutedims(hcat(K, zeros(3)))), " "))
                println(io, "Tr: ", join(stream, " "))
            end
            @test _D._parse_kitti_calib(d) == K
            Tr = _D._parse_kitti_tr(d)
            @test Tr(SVector(0.0, 0.0, 0.0)) ≈ t && Tr(SVector(1.0, 0.0, 0.0)) ≈ R[:, 1] + t
            write(joinpath(d, "poses.txt"), join(stream, " ") * "\n" * join(stream, " ") * "\n")
            T = _D._parse_kitti_poses(joinpath(d, "poses.txt"))
            @test length(T) == 2 && T[1] == [R t; 0 0 0 1]
            open(io -> println(io, "P1: 1 2 3"), joinpath(d, "bad", "calib.txt") |> (p -> (mkpath(dirname(p)); p)), "w")
            @test_throws ErrorException _D._parse_kitti_calib(joinpath(d, "bad"))

            xyzi = rand(rng, Float32, 4, 37)
            write(joinpath(d, "v.bin"), xyzi)
            @test _D._load_kitti_velodyne(joinpath(d, "v.bin")) ==
                  [SVector{3,Float64}(xyzi[1, i], xyzi[2, i], xyzi[3, i]) for i in 1:37]
            write(joinpath(d, "bad.bin"), rand(rng, Float32, 5))
            @test_throws "not divisible" _D._load_kitti_velodyne(joinpath(d, "bad.bin"))

            # returns straight ahead (velodyne x forward = camera z) at 10 m and 12 m
            # land on the principal point; one behind the scanner is culled
            cam = VgtkCore.CameraModel(VgtkCore.CameraCalibrationMat(K); image_origin = :opencv)
            velo_to_cam = VgtkCore.EuclideanMap(SMatrix{3,3,Float64,9}(0, 0, 1, -1, 0, 0, 0, -1, 0),
                                                          zero(SVector{3,Float64}))
            depth = _D._kitti_velo_to_depth([VgtkCore.Point3{Float64}(12.0, 0.0, 0.0),
                                            VgtkCore.Point3{Float64}(10.0, 0.0, 0.0),
                                            VgtkCore.Point3{Float64}(-10.0, 0.0, 0.0)],
                                           cam, velo_to_cam, VgtkCore.Size2(1241, 376))
            @test size(depth) == (376, 1241) && depth[181, 601] == 10.0f0      # the nearest return wins
            @test count(!isnan, depth) == 1

            dm = rand(rng, Float32, 3, 5); dm[2, 4] = Inf32; dm[1, 1] = -Inf32
            write(joinpath(d, "depth"), permutedims(dm))                      # row-major on disk
            got = _D._load_eth3d_depth(joinpath(d, "depth"), 5, 3)
            @test isnan(got[2, 4]) && isnan(got[1, 1]) && got[isfinite.(dm)] == dm[isfinite.(dm)]

            raw = UInt16[0 1500; 65535 2000]
            PNGFiles.save(joinpath(d, "d.png"), raw)                            # a 16-bit grey PNG
            s = _D._decode_depth_png(joinpath(d, "d.png"), UInt16(0))           # ScanNet: 0 invalid
            @test isnan(s[1, 1]) && s[1, 2] == 1.5f0 && s[2, 1] == 65.535f0
            k = _D._decode_depth_png(joinpath(d, "d.png"), typemax(UInt16))     # 7-Scenes: 65535 invalid
            @test k[1, 1] == 0.0f0 && isnan(k[2, 1]) && k[2, 2] == 2.0f0
        end
    end

    @testset "sparse-match files: RoMa v2 and SP+LG by the matcher attribute" begin
        mktempdir() do d
            a, b = 100 .* rand(rng, Float32, 2, 4), 100 .* rand(rng, Float32, 2, 4)
            c = rand(rng, Float32, 4)
            h5open(joinpath(d, "splg.h5"), "w") do f
                f["kpts_A"] = a; f["kpts_B"] = b; f["scores"] = c
                attrs(f)["matcher"] = "superpoint+lightglue"
            end
            h5open(joinpath(d, "roma.h5"), "w") do f            # RoMa's files predate the attribute
                f["kpts_A"] = a; f["kpts_B"] = b; f["confidence"] = c; f["precision_A"] = rand(rng, Float32, 2, 2, 4)
            end
            h5open(joinpath(d, "other.h5"), "w") do f
                f["kpts_A"] = a; f["kpts_B"] = b; attrs(f)["matcher"] = "disk"
            end
            for name in ("splg.h5", "roma.h5")
                m = load_sparse_matches(joinpath(d, name))
                @test m.confidence == c
                @test [p.first for p in m.csponds] == [SVector{2,Float32}(a[:, i]) for i in 1:4]
                @test [p.second for p in m.csponds] == [SVector{2,Float32}(b[:, i]) for i in 1:4]
            end
            @test_throws "unknown matcher" load_sparse_matches(joinpath(d, "other.h5"))
            @test _D._pair_file_id((scene = "chess/seq-03", source_name = "frame-000000.color.png",
                                    target_name = "frame-000001.color.png")) ==
                  "chess__seq-03__frame-000000.color__frame-000001.color"
        end
    end

    @testset "consolidate_csponds end to end: a plane 2 m away, the camera moved 10 cm" begin
        root, matches, out = mktempdir(), mktempdir(), mktempdir()
        seq = joinpath(root, "chess", "seq-03"); mkpath(seq)
        f, cx, cy = 585.0, 320.0, 240.0
        poses = (Matrix{Float64}(I, 4, 4), [1.0 0 0 0.1; 0 1 0 0; 0 0 1 0; 0 0 0 1])   # camera to world
        plane = fill(UInt16(2000), 480, 640)                                   # 2 m everywhere
        for (k, T) in enumerate(poses)
            stem = "frame-$(lpad(k - 1, 6, '0'))"
            write(joinpath(seq, "$stem.color.png"), "")                        # only the name is read
            open(io -> foreach(r -> println(io, join(T[r, :], " ")), 1:4), joinpath(seq, "$stem.pose.txt"), "w")
            PNGFiles.save(joinpath(seq, "$stem.depth.png"), plane)
        end
        ds = SevenScenes(root = root, scene = "chess", interval = 1, stride = 1)
        ps = stereo_pairs(ds)
        @test length(ps) == 1 && only(ps).scene == "chess/seq-03"
        # a point at depth Z moves by −f·0.1/Z = −29.25 px; the last match is 40 px off
        src = Float32[200 300 400 350; 150 250 300 200]
        tgt = copy(src); tgt[1, :] .-= 29.25f0; tgt[1, 4] -= 40
        h5open(joinpath(matches, "$(_D._pair_file_id(only(ps))).h5"), "w") do h
            h["kpts_A"] = src; h["kpts_B"] = tgt; h["scores"] = Float32[0.9, 0.8, 0.7, 0.6]
            attrs(h)["matcher"] = "superpoint+lightglue"
        end
        written = consolidate_csponds(ds, :roma; match_dir = matches, out_dir = out)
        @test written == [joinpath(out, "chess__seq-03.jld2")]
        @test _D.JLD2.load(only(written), "format") == CONSOLIDATED_FORMAT
        @test _D.JLD2.load(only(written), "gt_method") === :depth_warp_seven_scenes_kinect
        rp = only(load_roma_pairs(only(written)))
        @test rp.gt_inliers == [true, true, true, false]
        @test rp.snn ≈ 1 .- Float32[0.9, 0.8, 0.7, 0.6]
        @test rp.scene == "chess/seq-03" && rp.src_id == "frame-000000.color"
        @test rp.R_gt ≈ I && rp.t_gt ≈ [-0.1, 0, 0]                         # source camera → target camera
        @test rp.size1 == VgtkCore.Size2(640, 480)
        @test [p.first for p in rp.csponds] == [SVector{2,Float64}(src[:, i]) for i in 1:4]
    end
end
