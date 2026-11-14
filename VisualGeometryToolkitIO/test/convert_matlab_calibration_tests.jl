# =============================================================================
# convert_matlab_calibration: a third-party MATLAB calibration → native v2 files
# =============================================================================

@testitem "convert_matlab_calibration: radial and fisheye, verified, original untouched" setup = [VGIoFixtures] begin
    import VisualGeometryToolkitCore as VgtkCore
    import VisualGeometryToolkitIO as VgtkIO
    using VisualGeometryToolkitIO: convert_matlab_calibration, load_matlab_calibration, validate
    using VisualGeometryToolkitCore: Scaramuzza

    mktempdir() do dir
        rvec = [0.12, -0.34, 0.56]
        radial = joinpath(dir, "calibration_matlab_3radial2tangential.json")
        write(
            radial, JSON.json(
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
            fisheye, JSON.json(
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
        @test_throws ArgumentError convert_matlab_calibration(radial; images = names, image_size = (4240, 2832), out = mkpath(joinpath(dir, "x")))

        for (src, lens_type, n_posed) in ((radial, Lens{Perspective}, 2), (fisheye, Scaramuzza, 1))
            out = mkpath(joinpath(dir, "v2_$(splitext(basename(src))[1])"))
            r = convert_matlab_calibration(src; images = names, image_size = (4240, 2832), file_unit = mm, target = "aprilgrid_matlab", out)
            @test read(src, String) == originals[src]                                # never touched
            @test validate(r.cameras) == "cameras" && validate(r.poses) == "transforms"
            @test (r.n_posed, r.n_unposed) == (n_posed, 2 - n_posed)
            cal = load_matlab_calibration(src; file_unit = mm, target = "aprilgrid_matlab")   # MATLAB's (1, 1) frame by default
            cam = only(values(load(Dict{Symbol, VgtkCore.BoundedCamera}, r.cameras)))
            @test VgtkCore.model(cam) == cal.camera && VgtkCore.model(cam).lens isa lens_type && extent(cam) == Size2(4240, 2832)
            d = parse_json(read(r.cameras, String))
            @test d.provenance.source == "matlab" && d.provenance.error_statistic == "mean"
            @test d.provenance.reprojection_error == cal.reprojection_error
            poses = load(Vector{EuclideanMap}, r.poses)
            posed = [(names[i], p) for (i, p) in enumerate(cal.poses) if p !== nothing]
            @test length(poses) == length(posed)
            for (rec, (n, p)) in zip(poses, posed)
                @test VgtkCore.map_frames(rec) == (VgtkCore.TargetFrame{:aprilgrid_matlab}, VgtkCore.ViewFrame{Symbol(n)})
                # the reader's values: the translation bit for bit (a fisheye's twisted, computed one
                # to the mm file's 2 ulp), the rotation to the encoding's precision
                @test VgtkIO._same_pose(rec, p) && maximum(abs, Matrix(rec.R) - Matrix(p.R)) <= 8 * eps()
                lens_type === Scaramuzza || @test rec.t == p.t
            end
        end
        # MATLAB's documented frame is the default: the principal point moves by −0.5 into the canonical frame
        @test load(CameraModel, joinpath(dir, "v2_calibration_matlab_3radial2tangential", "calibration_matlab_3radial2tangential_cameras.json")).intrinsics.K[1, 3] == 2155.82 - 0.5
        # …and the override is positional, for a file verified to be in another frame
        o = mkpath(joinpath(dir, "v2_opencv_frame"))
        r = convert_matlab_calibration(PixelFrame{:opencv}, radial; images = names, image_size = (4240, 2832), file_unit = mm, out = o)
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

# A fisheye's twisted translations are computed, full-precision metre values; for about one in eight no mm number reads
# back to the value, and the writer then writes its persisted form (IO 0.10.1), which reads back up to ~22 ulp off
# with the value's persisted identity. The conversion's comparison accepts that identity (IO 0.10.3).
@testitem "convert_matlab_calibration: a fisheye of many poses and a stretch reads back to its persisted identity" setup = [VGIoFixtures] begin
    import VisualGeometryToolkitIO as VgtkIO
    using VisualGeometryToolkitIO: convert_matlab_calibration, load_matlab_calibration
    using Random: Xoshiro
    rng = Xoshiro(3)
    poses = [Dict(:rvec => (rand(rng, 3) .- 0.5) .* 2, :tvec => round.((rand(rng, 3) .- 0.5) .* 2000; digits = 4)) for _ in 1:50]
    mktempdir() do dir
        src = joinpath(dir, "calibration_matlab_fisheye.json")
        write(
            src, JSON.json(
                Dict(
                    :model => "fisheye", :mapping_coeffs => [2470.92, -6.73e-5, -1.03e-9, 2.32e-12],
                    :distortion_center => [2148.0, 1414.32], :stretch_matrix => [[1.0009, -0.0043], [0.004, 1.0]],
                    :mean_error => 0.4955, :poses => poses,
                )
            )
        )
        names = ["DSC$(lpad(i, 4, '0')).JPG" for i in 1:50]
        r = convert_matlab_calibration(src; images = names, image_size = (4240, 2832), file_unit = mm, out = mkpath(joinpath(dir, "v2")))
        @test r.n_posed == 50
        cal = load_matlab_calibration(src; file_unit = mm)
        back = load(Vector{EuclideanMap}, r.poses)
        @test all(((a, b),) -> VgtkIO._same_pose(a, b), zip(back, cal.poses))
        metres(p) = [Float64(ustrip(m, t)) for t in p.t]
        # the identity of every translation read back is its value's
        @test all(((a, b),) -> VgtkIO._persisted.(metres(a)) == VgtkIO._persisted.(metres(b)), zip(back, cal.poses))
        # some translation is NOT within 2 ulp (the case the comparison must accept)
        @test any(((a, b),) -> any(abs.(metres(a) .- metres(b)) .> 2 .* eps.(abs.(metres(a)))), zip(back, cal.poses))
    end
end
