@testitem "paper_support" begin   # assert_active_project
    # bmvc2026's and icra2026's `_assert_project`: DrWatson's projectdir() is the
    # paper project only when it is the active project, so a driver asserts it.
    using Test
    using VisualGeometryToolkitEval.Experiments: assert_active_project
    using VisualGeometryToolkitCore: thrown_message
    here = dirname(Base.active_project())
    @test assert_active_project("Project.toml") === nothing
    msg = try assert_active_project(joinpath("src", "NoSuchPaper.jl")); "" catch e; sprint(showerror, e) end
    @test occursin("NoSuchPaper.jl", msg) && occursin(here, msg) && occursin("--project", msg)
end

@testitem "paper_support: station_cells" begin
    # The floorplan fan's cells, restated by bmvc2026 and icra2026: one row per
    # (frame, marker) attempt → per (station, distance) the count placed, the
    # rate, the median distance error and median |angle error|.
    using Test, DataFrames
    using VisualGeometryToolkitEval.Render: station_cells
    df = DataFrame(rig_distances_bin = [1000, 1000, 1000, 1000, 2000],
                   rig_angles = deg2rad.([-20.0, -20.0, -10.0, -10.0, 0.0]),
                   error_distances = [1.0, 3.0, missing, 2.0, 5.0],
                   abs_error_angles_deg = [0.1, -0.3, 0.2, 0.4, 1.0])
    cells = station_cells(df)
    @test length(cells) == 2 * 17                         # every station of -80:10:80 at both distances
    e = only(filter(c -> c.dist == 2000 && c.angle == 40, cells))
    @test e.n == 0 && e.rate == 0.0 && isnan(e.med)       # an empty station is a zero, not missing
    c = only(filter(c -> c.dist == 1000 && c.angle == -20, cells))
    @test c.n == 2 && c.rate == 1.0 && c.med == 2.0 && c.medr ≈ 0.2
    d = only(filter(c -> c.dist == 1000 && c.angle == -10, cells))
    @test d.n == 1 && d.rate == 0.5 && d.med == 2.0
    @test only(filter(c -> c.dist == 2000 && c.n > 0, cells)).angle == 0
end

@testitem "paper_support: save_hand_eye, load_hand_eye" begin
    # A hand-eye calibration as its two records (bmvc2026 writes them, icra2026
    # reads them): X (body:camera_rig → camera_optical) and the marker mount.
    using Test, StaticArrays, Rotations
    using VisualGeometryToolkitEval.Experiments: save_hand_eye, load_hand_eye, hand_eye_mount_file
    using VisualGeometryToolkitCore: EuclideanMap, BodyFrame, ViewFrame, Rig, ids, pose, geom, unlabel
    dir = mktempdir()
    X = EuclideanMap(RotMatrix(RotXYZ(0.1, -0.2, 0.3)), SVector(10.0, -5.0, 120.0))
    Y = Dict("a" => EuclideanMap(RotMatrix(RotZ(0.5)), SVector(1.0, 2.0, 3.0)),
             "b" => EuclideanMap(RotMatrix(RotY(-0.2)), SVector(-4.0, 0.0, 9.5)))
    file = joinpath(dir, "hand_eye.json")
    @test save_hand_eye(file, X, Y) == (file, hand_eye_mount_file(file))
    @test isfile(hand_eye_mount_file(file)) && endswith(hand_eye_mount_file(file), "hand_eye_board_rig.json")
    X2, mount = load_hand_eye(file)
    @test geom(X2) isa EuclideanMap{3, <:Any, <:Any, <:Any, BodyFrame{:camera_rig}, ViewFrame{:optical}}
    @test unlabel(geom(X2)).t ≈ X.t && unlabel(geom(X2)).R ≈ X.R
    @test mount isa Rig{BodyFrame{:board_rig}} && Set(String.(ids(mount))) == Set(["a", "b"])
    @test unlabel(geom(pose(mount, :a))).t ≈ Y["a"].t
end

@testitem "paper_support: load_hand_eye reads the 2026-09 one-file record; save_hand_eye writes the pair" begin
    # icra2026's motion-capture runs of 2026-09 hold the hand-eye record as ONE
    # vgio/transforms/v2 file (the camera mount and the marker mount together,
    # no mount file). VisualGeometryToolkitIO reads it as the current transforms
    # record; the mount is its target:<id> → body:board_rig poses.
    using Test, StaticArrays, Rotations
    import VisualGeometryToolkitIO as VgtkIO
    using VisualGeometryToolkitEval.Experiments: save_hand_eye, load_hand_eye, hand_eye_mount_file
    using VisualGeometryToolkitCore: EuclideanMap, BodyFrame, ViewFrame, Rig, ids, pose, geom, unlabel
    src = joinpath(pkgdir(VgtkIO), "test", "fixtures", "old", "hand_eye_transforms_v2_printed_board.json")
    raw = VgtkIO.read_record(src)                    # its numbers (the record's order and values)
    q(r) = (v = Float64.(r["rotation"]); RotMatrix(QuatRotation(v[1], v[2], v[3], v[4])))
    t(r) = SVector{3, Float64}(r["translation"]...)
    dir = mktempdir()
    file = joinpath(dir, "hand_eye.json")
    cp(src, file)
    X, mount = load_hand_eye(file)
    @test geom(X) isa EuclideanMap{3, <:Any, <:Any, <:Any, BodyFrame{:camera_rig}, ViewFrame{:optical}}
    @test unlabel(geom(X)).t ≈ t(raw["transforms"][1]) rtol = 1.0e-15
    @test unlabel(geom(X)).R ≈ q(raw["transforms"][1]) atol = 1.0e-15
    @test mount isa Rig{BodyFrame{:board_rig}} && ids(mount) == [:aprilgrid]
    @test unlabel(geom(pose(mount, :aprilgrid))).t ≈ t(raw["transforms"][2]) rtol = 1.0e-15
    out = joinpath(dir, "out", "hand_eye.json")
    mkpath(dirname(out))
    save_hand_eye(out, unlabel(geom(X)), Dict("aprilgrid" => unlabel(geom(pose(mount, :aprilgrid)))))
    @test VgtkIO.read_record(out)["schema"] == "vgio/transforms/v4"          # only the current records
    @test startswith(VgtkIO.read_record(hand_eye_mount_file(out))["schema"], "vgio/rig/")
    X2, mount2 = load_hand_eye(out)
    @test unlabel(geom(X2)).t ≈ unlabel(geom(X)).t && unlabel(geom(pose(mount2, :aprilgrid))).t ≈ unlabel(geom(pose(mount, :aprilgrid))).t
end
