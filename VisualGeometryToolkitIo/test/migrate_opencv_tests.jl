# =============================================================================
# migrate_opencv: a third-party OpenCV calibration → native v2 files
# =============================================================================

@testitem "migrate_opencv: camera + image-named poses, verified, original untouched" setup = [VGIoFixtures] begin
    import VisualGeometryToolkitCore as VgtkCore
    import VisualGeometryToolkitIo as VgtkIo
    using VisualGeometryToolkitIo: migrate_opencv, load_opencv_calibration, validate
    using Rotations: RotationVec

    mktempdir() do dir
        rvec = [0.12, -0.34, 0.56]
        body = Dict(
            :fx => 2487.06, :fy => 2480.08, :cx => 2175.69, :cy => 1448.16,
            :k1 => -2.7068, :k2 => 4.0035, :p1 => 0.0028, :p2 => 0.0017, :k3 => 1.5827,
            :rms => 0.2737,
            :poses => [Dict(:rvec => rvec, :tvec => [0.17, 0.0122, 1.3969]), nothing, Dict(:rvec => -rvec, :tvec => [0.1, 0.2, 1.5])],
        )
        src = joinpath(dir, "calibration_opencv_default.json")
        write(src, JSON3.write(body))
        original = read(src, String)
        names = ["IMG_0001.png", "IMG_0002.png", "IMG_0003.png"]
        out = mkpath(joinpath(dir, "v2"))

        # no tvec unit declared and none stated: refused, never defaulted
        @test_throws ArgumentError migrate_opencv(src; images = names, image_size = (4240, 2832), out)
        # images must name every pose, uniquely
        @test_throws ArgumentError migrate_opencv(src; images = names[1:2], image_size = (4240, 2832), file_unit = m, out)
        @test_throws ArgumentError migrate_opencv(src; images = ["a", "a", "b"], image_size = (4240, 2832), file_unit = m, out)
        @test_throws UndefKeywordError migrate_opencv(src; image_size = (4240, 2832), file_unit = m, out)
        @test_throws UndefKeywordError migrate_opencv(src; images = names, file_unit = m, out)

        dry = migrate_opencv(src; images = names, image_size = (4240, 2832), file_unit = m, out, dry_run = true)
        @test dry.cameras === nothing && isempty(readdir(out)) && (dry.n_posed, dry.n_unposed) == (2, 1)
        r = migrate_opencv(src; images = names, image_size = (4240, 2832), file_unit = m, out)
        @test read(src, String) == original                                          # the original is never touched
        @test validate(r.cameras) == "cameras" && validate(r.poses) == "transforms"

        cal = load_opencv_calibration(src; file_unit = m)                                # OpenCV's (0, 0) convention, by the user's rule
        store = load(Dict{Symbol, VgtkCore.BoundedCamera}, r.cameras)
        cam = only(values(store))
        @test only(keys(store)) === :calibration_opencv_default && extent(cam) == Size2(4240, 2832)
        @test VgtkCore.model(cam) == cal.camera                            # K canonical (cx + 0.5), lens as the file names it
        @test VgtkCore.model(cam).intrinsics.K[1, 3] == 2175.69 + 0.5
        cd_ = JSON3.read(read(r.cameras, String))
        @test VgtkIo._plain(cd_.cameras[1].lens) == lens_record(cal.camera.lens)
        @test cd_.provenance.reprojection_error == 0.2737 && cd_.provenance.error_statistic == "rms"

        poses = load(Vector{EuclideanMap}, r.poses)
        @test [VgtkIo._pose_frames(p) for p in poses] == [
            (VgtkCore.TargetFrame{:calibration_target}, VgtkCore.ViewFrame{Symbol("IMG_0001.png")}),
            (VgtkCore.TargetFrame{:calibration_target}, VgtkCore.ViewFrame{Symbol("IMG_0003.png")}),
        ]
        # the OpenCV read's values: the translation bit for bit (metres → mm by the reader:
        # 0.0122 m is 12.200000000000001 mm), the rotation to the quaternion encoding's precision
        @test poses[1].t == cal.poses[1].t && maximum(abs, Matrix(poses[1].R) - Matrix(cal.poses[1].R)) <= 8 * eps()
        @test poses[1].t ≈ SVector(170.0, 12.2, 1396.9)
        @test poses[2].t == cal.poses[3].t && maximum(abs, Matrix(poses[2].R) - Matrix(cal.poses[3].R)) <= 8 * eps()

        # nothing is overwritten
        @test_throws ArgumentError migrate_opencv(src; images = names, image_size = (4240, 2832), file_unit = m, out)

        # a file that declares its tvec unit supplies it, and a contradicting caller is refused
        declared = joinpath(dir, "calibration_opencv_rational.json")
        write(declared, JSON3.write(merge(body, Dict(:tvec_unit => "m"))))
        o2 = mkpath(joinpath(dir, "v2b"))
        r2 = migrate_opencv(declared; images = names, image_size = (4240, 2832), out = o2)
        @test load(Vector{EuclideanMap}, r2.poses)[1].t == cal.poses[1].t                    # tvec_unit "m" honoured
        @test_throws ArgumentError migrate_opencv(declared; images = names, image_size = (4240, 2832), file_unit = mm, out = mkpath(joinpath(dir, "v2c")))
        # the reader holds a declared tvec_unit too
        @test_throws ArgumentError load_opencv_calibration(declared; file_unit = mm)

        # the frame override, for a calibration VERIFIED to be in another frame
        o3 = mkpath(joinpath(dir, "v2d"))
        r3 = migrate_opencv(PixelFrame{:julia}, src; images = names, image_size = (4240, 2832), file_unit = m, out = o3)
        @test load(CameraModel, r3.cameras).intrinsics.K[1, 3] == 2175.69 - 0.5

        # the command line: a dry run unless --apply; the images from a directory's sorted names
        imgs = mkpath(joinpath(dir, "imgs"))
        foreach(n -> touch(joinpath(imgs, n)), reverse(names))
        touch(joinpath(imgs, "notes.txt"))
        o4 = mkpath(joinpath(dir, "v2e"))
        io = IOBuffer()
        args = ["opencv", "--images", imgs, "--image-size", "4240x2832", "--file-unit", "m", "--out", o4, src]
        @test VgtkIo._migrate_cli(args; io) == 0
        @test occursin("2 posed, 1 unposed", String(take!(io))) && isempty(readdir(o4))
        @test VgtkIo._migrate_cli([args[1]; "--apply"; args[2:end]]; io) == 0
        @test sort(readdir(o4)) == ["calibration_opencv_default_cameras.json", "calibration_opencv_default_poses.json"]
        @test [VgtkIo.frame_name(last(VgtkIo._pose_frames(p))) for p in load(Vector{EuclideanMap}, joinpath(o4, "calibration_opencv_default_poses.json"))] == ["view:IMG_0001.png", "view:IMG_0003.png"]
        @test VgtkIo._migrate_cli(["opencv", src]; io) == 2                               # --images and --image-size are required
        @test VgtkIo._migrate_cli(["opencv", "--images", imgs, "--image-size", "4240x2832", "--out", mkpath(joinpath(dir, "v2f")), src]; io) == 1   # no unit
        # the command converts third-party calibrations only: anything else is a usage error
        @test VgtkIo._migrate_cli(["--help"]; io) == 0
        @test VgtkIo._migrate_cli([dir]; io) == 2
    end
end
