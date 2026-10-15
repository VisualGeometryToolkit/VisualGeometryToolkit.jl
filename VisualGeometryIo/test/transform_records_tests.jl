# =============================================================================
# transforms/v3 ↔ Vector of labeled VGC poses
# =============================================================================

@testitem "transforms/v3: labeled poses round trip" setup = [VGIoFixtures] begin
    using VisualGeometryCore: relabel, TargetFrame, CameraFrame, BodyFrame, UnlabeledFrame
    pose = _mkpose()
    labeled = relabel(TargetFrame{:b76e}, CameraFrame, pose)
    @test frame_from_name("camera_optical") === CameraFrame
    mktempdir() do dir
        path = joinpath(dir, "poses.json")
        X2 = relabel(TargetFrame{:a}, BodyFrame{:b}, qpose(SVector(1.0, 0.0, 0.0, 0.0), SVector(1.0, 2.0, 3.0)))
        save(path, [labeled, X2]; provenance = (image = "DSC04520.JPG",))
        back = load(Vector{EuclideanMap}, path)
        # EXACT by type: labels, translation and (quaternion-born) rotation bit for bit
        @test back == [labeled, X2]
        @test typeof(back[1]) == typeof(labeled) && typeof(back[2]) == typeof(X2)
        d = JSON3.read(read(path, String))
        @test d.schema == "vgio/transforms/v3" && d.length_unit == "mm"
        @test d.rotation_encoding == "quaternion_wxyz_hamilton" && d.transform_convention == "x_to = R * x_from + t"
        @test (d.transforms[1].from, d.transforms[1].to) == ("target:b76e", "camera_optical")
        @test d.transforms[1].rotation[1] >= 0
        @test d.provenance.image == "DSC04520.JPG"
        # a file in metres reads into millimetres
        body = JSON3.read(read(path, String), Dict{String, Any})
        body["length_unit"] = "m"
        for r in body["transforms"]
            r["translation"] = r["translation"] ./ 1000
        end
        open(io -> JSON3.write(io, body), joinpath(dir, "metres.json"), "w")
        bm = load(Vector{EuclideanMap}, joinpath(dir, "metres.json"))
        @test bm[1].t ≈ labeled.t atol = 1.0e-9
        @test typeof(bm[1]) == typeof(labeled)
        save(joinpath(dir, "empty.json"), EuclideanMap[])
        @test isempty(load(Vector{EuclideanMap}, joinpath(dir, "empty.json")))
        # unlabeled or half-labeled poses have no frames to name: refused
        @test_throws ArgumentError save(joinpath(dir, "free.json"), [pose])
        @test_throws ArgumentError save(joinpath(dir, "half.json"), [relabel(TargetFrame{:a}, UnlabeledFrame{3}, pose)])
        @test !isfile(joinpath(dir, "free.json"))
        # not poses at all: refused
        @test_throws ArgumentError save(joinpath(dir, "num.json"), [1.0, 2.0])
        # a px translation is not metric: the quantity boundary refuses it
        @test_throws Exception VisualGeometryIo.pose_from_json(JSON3.read("""{"rotation": [1,0,0,0], "translation": [1,2,3]}"""), "px")
    end
end

# Rotations are quaternions on disk and in memory: a pose is READ as the stored
# quaternion itself (a `QuatRotation`, not renormalized), so a file-born value
# round-trips `==` and file → value → file is byte-identical; a `RotMatrix`
# value is converted on write and reads back as that quaternion.
@testitem "quaternion encoding: rotations read as the stored quaternion" setup = [VGIoFixtures] begin
    using Random: Xoshiro
    using VisualGeometryCore: relabel, BodyFrame, WorldFrame
    rng = Xoshiro(7)
    mktempdir() do dir
        p1, p2 = joinpath(dir, "1.json"), joinpath(dir, "2.json")
        for _ in 1:200
            v = randn(rng, 4)
            q = canonical_qvec(SVector{4, Float64}(v ./ norm(v)))
            # a RotMatrix value: converted once, read back as a QuatRotation-valued map
            Rm = RotMatrix{3, Float64}(QuatRotation(q[1], q[2], q[3], q[4]))
            X = relabel(BodyFrame{:a}, WorldFrame, EuclideanMap(Rm, SVector(1.0, 2.0, 3.0)))
            save(p1, [X])
            Y = only(load(Vector{EuclideanMap}, p1))
            @test Y.R isa QuatRotation{Float64} && Y.t == X.t
            @test maximum(abs, Matrix(Y.R) - Matrix(X.R)) <= 4 * eps()
            # a file-born value: exact, and the file a fixed point
            save(p2, [Y])
            @test read(p2, String) == read(p1, String)
            Z = only(load(Vector{EuclideanMap}, p2))
            @test Z == Y && Z.R.q == Y.R.q && typeof(Z) == typeof(Y)
        end
        # the stored quaternion is not renormalized: what the file says is the value
        body = JSON3.read(read(p1, String), Dict{String, Any})
        qs = body["transforms"][1]["rotation"]
        qs[1] = nextfloat(Float64(qs[1]))
        open(io -> JSON3.write(io, body), p1, "w")
        W = only(load(Vector{EuclideanMap}, p1))
        @test collect(VisualGeometryIo._qvec(W.R)) == Float64.(qs)
        # …but it must be a unit quaternion
        qs[1] = 2.0
        open(io -> JSON3.write(io, body), p1, "w")
        @test_throws ArgumentError load(Vector{EuclideanMap}, p1)
    end
end

# Uncertainty mirrors the type (VisualGeometryCore conventions/persisted-geometry.md):
# plain ↔ no covariance, Uncertain ↔ covariance, exact ↔ zeros; the covariance is
# the pose's [δω; δt] in rad and the file's unit, moved with the unit on read.
@testitem "transforms/v3: uncertain poses, typed frames" setup = [VGIoFixtures] begin
    using VisualGeometryCore: Uncertain, CovMat, geom, BodyFrame, CameraFrame, relabel
    using Statistics: cov
    using LinearAlgebra: Diagonal
    pose = relabel(BodyFrame{:camera_rig}, CameraFrame, _mkpose())
    Σ = CovMat(SMatrix{6, 6, Float64}(Diagonal([1.0e-6, 2.0e-6, 3.0e-6, 0.1, 0.2, 0.3])))
    mktempdir() do dir
        u = Uncertain(pose, Σ)
        e = Uncertain(pose, CovMat(zero(SMatrix{6, 6, Float64})))
        path = joinpath(dir, "x.json")
        save(path, [pose, u, e])
        d = JSON3.read(read(path, String))
        @test !haskey(d.transforms[1], :covariance) && haskey(d.transforms[2], :covariance)   # written iff Uncertain
        back = load(Vector{EuclideanMap}, path)
        @test back == [pose, u, e]
        @test back[1] isa EuclideanMap                        # plain in, plain out
        @test back[2] isa Uncertain && cov(back[2]) == Σ && geom(back[2]) == pose
        @test iszero(cov(back[3]))                            # exact stays exact
        # a file in metres: the translation block of the covariance moves with the unit
        body = JSON3.read(read(path, String), Dict{String, Any})
        body["length_unit"] = "m"
        r = body["transforms"][2]
        r["translation"] = r["translation"] ./ 1000
        S = [Float64(r["covariance"][i][j]) * (i <= 3 && j <= 3 ? 1.0 : i > 3 && j > 3 ? 1.0e-6 : 1.0e-3) for i in 1:6, j in 1:6]
        r["covariance"] = [S[i, :] for i in 1:6]
        body["transforms"] = [r]
        open(io -> JSON3.write(io, body), joinpath(dir, "m.json"), "w")
        um = only(load(Vector{EuclideanMap}, joinpath(dir, "m.json")))
        @test cov(um) ≈ Σ rtol = 1.0e-12
        # an asymmetric covariance is refused by the validator
        r["covariance"][1][2] = 1.0e-7
        open(io -> JSON3.write(io, body), joinpath(dir, "asym.json"), "w")
        @test_throws ArgumentError load(Vector{EuclideanMap}, joinpath(dir, "asym.json"))
        # ROS PoseWithCovariance orders translation first: a block swap, and back
        Σr = ros_pose_covariance(Σ)
        @test Σr[1:3, 1:3] == SMatrix{6, 6, Float64}(Σ)[4:6, 4:6] && Σr[4:6, 4:6] == SMatrix{6, 6, Float64}(Σ)[1:3, 1:3]
        @test pose_covariance_from_ros(Σr) == Σ
        @test pose_covariance_from_ros(vec(permutedims(Matrix(Σr)))) == Σ          # the message's row-major float64[36]
    end
end
