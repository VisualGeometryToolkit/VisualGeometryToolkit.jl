# The covariance storage on the wire (the 2026-11 versions: target_detections v2, target_points2d / target_points3d v2,
# target_poses v2, trajectory v4, transforms v4, target v5, rig v8): a covariance is written in the storage its value holds
# — rows (full), {"diagonal": […]}, {"isotropic": λ} — and read back as that storage; an isotropic one over mixed length
# dimensions (a pose) is written as its diagonal; the validator refuses a malformed one; and every file of the version each
# kind replaced reads as its current counterpart.
@testitem "covariance storage: the wire form is the held matrix" setup = [VGIoFixtures] begin
    using LinearAlgebra: Diagonal
    using StaticArrays: SDiagonal
    import Rotations
    using VisualGeometryToolkitCore: TargetDetections, Uncertain, CovMat, ScaledIdentity, EuclideanMap, WorldFrame, ViewFrame,
        relabel, geom, csponds, cov
    h = repeat("ab", 32)
    mktempdir() do dir
        # pixels: the three storages round-trip as themselves, px² never converted
        for (Σ, wire) in ((CovMat(SMatrix{2, 2, Float64}(0.25, 0.01, 0.01, 0.5)), "rows"),
                          (CovMat(SDiagonal(0.25, 0.5)), "diagonal"),
                          (CovMat(ScaledIdentity{2}(0.25)), "isotropic"))
            x = TargetDetections{PixelFrame{:colmap}}(
                Dict(1 => Dict(:b => csponds([2, 7], [Uncertain(SVector(1.0, 2.0), Σ), Uncertain(SVector(3.0, 4.0), Σ)])));
                sensors = "cam1", images = Dict(1 => "a.png"), targets = [h])
            p = joinpath(dir, "td_$(wire).json")
            save(p, x)
            c = JSON.parse(read(p, String))["frames"][1]["detections"][1]["covariance"][1]
            @test wire == "rows" ? c isa AbstractVector : (c isa AbstractDict && only(keys(c)) == wire)
            back = load(TargetDetections, p)
            @test back == x && typeof(parent(cov(back[1][:b][1].second))) == typeof(parent(Σ))
        end
        # a pose: rad² and the file's length unit squared; full and diagonal kept, an isotropic one written as its diagonal
        X = relabel(WorldFrame, ViewFrame{:a}, EuclideanMap(Rotations.RotXYZ(0.1, 0.2, 0.3), SVector(1.0, 2.0, 3.0) .* m))
        for (Σ, back_storage, diag_mm) in (
                (CovMat(SDiagonal(1.0e-4, 2.0e-4, 3.0e-4, 1.0e-6, 4.0e-6, 9.0e-6)), Diagonal, [1.0e-4, 2.0e-4, 3.0e-4, 1.0, 4.0, 9.0]),
                (CovMat(ScaledIdentity{6}(1.0e-6)), Diagonal, [1.0e-6, 1.0e-6, 1.0e-6, 1.0, 1.0, 1.0]),
                (CovMat(SMatrix{6, 6, Float64}(Diagonal(fill(1.0e-6, 6)))), SMatrix, nothing))
            p = joinpath(dir, "x.json")
            save(p, [Uncertain(X, Σ)])
            c = JSON.parse(read(p, String))["transforms"][1]["covariance"]
            diag_mm === nothing ? (@test c isa AbstractVector) : (@test c["diagonal"] == diag_mm)
            b = only(load(Vector{EuclideanMap}, p))
            @test parent(cov(b)) isa back_storage && SMatrix(cov(b)) ≈ SMatrix(Σ)
        end
    end
end

@testitem "covariance storage: the validator refuses a malformed covariance" setup = [VGIoFixtures] begin
    using VisualGeometryToolkitCore: TargetDetections, Uncertain, CovMat, ScaledIdentity, csponds
    h = repeat("ab", 32)
    mktempdir() do dir
        x = TargetDetections{PixelFrame{:colmap}}(
            Dict(1 => Dict(:b => csponds([2], [Uncertain(SVector(1.0, 2.0), CovMat(ScaledIdentity{2}(0.25)))])));
            sensors = "cam1", images = Dict(1 => "a.png"), targets = [h])
        p = joinpath(dir, "td.json")
        save(p, x)
        for bad in (Dict("diagonal" => [0.25]), Dict("isotropic" => [0.25]), Dict("scale" => 0.25), Dict("diagonal" => [1.0, 2.0], "isotropic" => 1.0))
            t = JSON.parse(read(p, String); dicttype = Dict{String, Any})
            t["frames"][1]["detections"][1]["covariance"][1] = bad
            q = joinpath(dir, "bad.json")
            write(q, JSON.json(t))
            @test_throws ArgumentError load(TargetDetections, q)
        end
    end
end

@testitem "covariance storage: a file of each replaced version reads as its current counterpart" setup = [VGIoFixtures] begin
    using VisualGeometryToolkitCore: Rig, AbstractTarget, Trajectory, TargetPoses, TargetPoints2D, TargetPoints3D
    import VisualGeometryToolkitIO as VgtkIO
    old(f) = joinpath(@__DIR__, "fixtures", "old", f)
    cur(f) = joinpath(@__DIR__, "fixtures", "units", f)
    for (o, c, T) in (("rig_v7.json", "rig.json", Rig), ("target_v4.json", "target_planar.json", AbstractTarget),
                      ("target_points2d_v1.json", "target_points2d.json", TargetPoints2D),
                      ("target_points3d_v1.json", "target_points3d.json", TargetPoints3D),
                      ("target_poses_v1.json", "target_poses.json", TargetPoses), ("trajectory_v3.json", "trajectory.json", Trajectory),
                      ("transforms_v3.json", "transforms.json", Vector{EuclideanMap}))
        @test VgtkIO._same_value(load(T, old(o)), load(T, cur(c)))
    end
end
