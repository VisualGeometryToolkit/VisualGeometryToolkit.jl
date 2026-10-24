# =============================================================================
# migrate_matlab: a third-party MATLAB calibration → native v2 files
# =============================================================================

@testitem "migrate_matlab: radial and fisheye, verified, original untouched" setup = [VGIoFixtures] begin
    import VisualGeometryToolkitCore as VgtkCore
    import VisualGeometryToolkitIO as VgtkIO
    using VisualGeometryToolkitIO: migrate_matlab, load_matlab_calibration, validate
    using VisualGeometryToolkitCore: Scaramuzza

    mktempdir() do dir
        rvec = [0.12, -0.34, 0.56]
        radial = joinpath(dir, "calibration_matlab_3radial2tangential.json")
        write(
            radial, JSON3.write(
                Dict(
                    :fx => 2472.59, :fy => 2470.3, :cx => 2155.82, :cy => 1409.01,
                    :r1 => -0.1642, :r2 => 0.0662, :r3 => -0.0155, :t1 => -0.000515, :t2 => 0.00092,
                    :mean_error => 0.4036,
                    :poses => [Dict(:rvec => rvec, :tvec => [-708.695, 148.6439, 1212.2104]), Dict(:rvec => -rvec, :tvec => [1.0, 2.0, 900.0])],
                )
            )
        )
        a0, S = 2470.92, [1.0009 -0.0043; 0.004 1.0]
        fisheye = joinpath(dir, "calibration_matlab_fisheye.json")
        write(
            fisheye, JSON3.write(
                Dict(
                    :model => "fisheye", :mapping_coeffs => [a0, -6.73e-5, -1.03e-9, 2.32e-12],
                    :distortion_center => [2148.0, 1414.32],
                    :stretch_matrix => [[S[1, 1], S[1, 2]], [S[2, 1], S[2, 2]]],
                    :mean_error => 0.4955, :poses => [Dict(:rvec => rvec, :tvec => [1.0, 2.0, 3.0]), nothing],
                )
            )
        )
        names = ["DSC0001.JPG", "DSC0002.JPG"]
        originals = Dict(p => read(p, String) for p in (radial, fisheye))

        # the toolbox declares no unit: it must be stated — never defaulted
        @test_throws ArgumentError migrate_matlab(radial; images = names, image_size = (4240, 2832), out = mkpath(joinpath(dir, "x")))

        for (src, lens_type, n_posed) in ((radial, Lens{Perspective}, 2), (fisheye, Scaramuzza, 1))
            out = mkpath(joinpath(dir, "v2_$(splitext(basename(src))[1])"))
            r = migrate_matlab(src; images = names, image_size = (4240, 2832), file_unit = mm, target = "aprilgrid_matlab", out)
            @test read(src, String) == originals[src]                                # never touched
            @test validate(r.cameras) == "cameras" && validate(r.poses) == "transforms"
            @test (r.n_posed, r.n_unposed) == (n_posed, 2 - n_posed)
            cal = load_matlab_calibration(src; file_unit = mm, target = "aprilgrid_matlab")   # MATLAB's (1, 1) frame by default
            cam = only(values(load(Dict{Symbol, VgtkCore.BoundedCamera}, r.cameras)))
            @test VgtkCore.model(cam) == cal.camera && VgtkCore.model(cam).lens isa lens_type && extent(cam) == Size2(4240, 2832)
            d = JSON3.read(read(r.cameras, String))
            @test d.provenance.source == "matlab" && d.provenance.error_statistic == "mean"
            @test d.provenance.reprojection_error == cal.reprojection_error
            poses = load(Vector{EuclideanMap}, r.poses)
            posed = [(names[i], p) for (i, p) in enumerate(cal.poses) if p !== nothing]
            @test length(poses) == length(posed)
            for (rec, (n, p)) in zip(poses, posed)
                @test VgtkIO._pose_frames(rec) == (VgtkCore.TargetFrame{:aprilgrid_matlab}, VgtkCore.ViewFrame{Symbol(n)})
                # the reader's values (fisheye twist and all): the translation bit for bit, the rotation to the encoding's precision
                @test rec.t == p.t && maximum(abs, Matrix(rec.R) - Matrix(p.R)) <= 8 * eps()
            end
        end
        # MATLAB's documented frame is the default: the principal point moves by −0.5 into the canonical frame
        @test load(CameraModel, joinpath(dir, "v2_calibration_matlab_3radial2tangential", "calibration_matlab_3radial2tangential_cameras.json")).intrinsics.K[1, 3] == 2155.82 - 0.5
        # …and the override is positional, for a file verified to be in another frame
        o = mkpath(joinpath(dir, "v2_opencv_frame"))
        r = migrate_matlab(PixelFrame{:opencv}, radial; images = names, image_size = (4240, 2832), file_unit = mm, out = o)
        @test load(CameraModel, r.cameras).intrinsics.K[1, 3] == 2155.82 + 0.5

        # the command line: `matlab`, a dry run unless --apply
        lst = joinpath(dir, "images.txt")
        write(lst, join(names, "\n"))
        o2 = mkpath(joinpath(dir, "v2_cli"))
        io = IOBuffer()
        args = ["matlab", "--images", lst, "--image-size", "4240x2832", "--file-unit", "mm", "--out", o2, radial]
        @test VgtkIO._migrate_cli(args; io) == 0
        @test occursin("2 posed, 0 unposed", String(take!(io))) && isempty(readdir(o2))
        @test VgtkIO._migrate_cli([args[1]; "--apply"; args[2:end]]; io) == 0
        @test sort(readdir(o2)) == ["calibration_matlab_3radial2tangential_cameras.json", "calibration_matlab_3radial2tangential_poses.json"]
        @test load(CameraModel, joinpath(o2, "calibration_matlab_3radial2tangential_cameras.json")).intrinsics.K[1, 3] == 2155.82 - 0.5
    end
end
