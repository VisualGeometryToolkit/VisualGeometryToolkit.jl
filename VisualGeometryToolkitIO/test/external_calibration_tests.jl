# The OpenCV and MATLAB calibration exports: read into the stack's conventions
# (COLMAP pixel frame, millimetres, poses labeled `TargetFrame{target} → CameraFrame`), with the two
# things the files never declare — pixel convention and length unit — stated
# by the caller.

@testitem "external calibration exports (OpenCV, MATLAB)" setup = [VGIoFixtures] begin
    import VisualGeometryToolkitCore as VgtkCore
    using Rotations: RotationVec, RotZ
    import VisualGeometryToolkitCore
    using VisualGeometryToolkitIO: ExternalCalibration, load_opencv_calibration, load_matlab_calibration,
        save_opencv_calibration
    using VisualGeometryToolkitCore: CameraModel, KannalaBrandt, TargetFrame, CameraFrame, unlabel
    using VisualGeometryToolkitIO: _pose_frames
    using Rotations: QuatRotation

    rvec = [0.12, -0.34, 0.56]
    R = RotMatrix{3, Float64}(RotationVec(rvec...))
    mktempdir() do dir
        cv = joinpath(dir, "calibration_opencv_thin_prism.json")
        write(
            cv, JSON.json(
                Dict(
                    :fx => 2487.06, :fy => 2480.08, :cx => 2175.69, :cy => 1448.16,
                    :k1 => -2.7068, :k2 => 4.0035, :p1 => 0.0028, :p2 => 0.0017, :k3 => 1.5827,
                    :k4 => -2.541, :k5 => 3.5018, :k6 => 2.406,
                    :s1 => -3.14e-5, :s2 => 7.36e-5, :s3 => -3.22e-3, :s4 => 2.75e-4,
                    :rms => 0.2737,
                    :poses => [Dict(:rvec => rvec, :tvec => [0.17, 0.0122, 1.3969]), nothing],
                )
            )
        )
        cal = load_opencv_calibration(cv; file_unit = m, target = "aprilgrid")
        @test cal isa ExternalCalibration{<:CameraModel} && cal.source === :opencv
        @test cal.error_statistic === :rms && cal.reprojection_error == 0.2737
        K = cal.camera.intrinsics.K
        @test (K[1, 1], K[2, 2]) == (2487.06, 2480.08)
        @test K[1, 3] ≈ 2175.69 + 0.5 atol = 1.0e-12       # OpenCV's (0, 0) frame → COLMAP
        @test K[2, 3] ≈ 1448.16 + 0.5 atol = 1.0e-12
        lens = cal.camera.lens
        c = parameters(lens)
        @test (c.k1, c.d3, c.p2, c.sy1) == (-2.7068, 2.406, 0.0017, -3.22e-3)
        @test lens === opencv_lens([-2.7068, 4.0035, 0.0028, 0.0017, 1.5827, -2.541, 3.5018, 2.406, -3.14e-5, 7.36e-5, -3.22e-3, 2.75e-4])
        @test length(cal.poses) == 2 && cal.poses[2] === nothing   # a null pose keeps its slot
        rec = cal.poses[1]
        @test _pose_frames(rec) == (TargetFrame{:aprilgrid}, CameraFrame)   # a labeled VGC pose, target → camera
        @test rec.t ≈ [170.0, 12.2, 1396.9]                         # metres → millimetres
        @test Matrix(rec.R) ≈ Matrix(R) atol = 1.0e-12
        # the default model: no k4–k6, s1–s4 in the file → OpenCV's 5-coefficient model
        cv0 = joinpath(dir, "calibration_opencv_default.json")
        write(
            cv0, JSON.json(
                Dict(
                    :fx => 2466.9, :fy => 2464.4, :cx => 2155.5, :cy => 1408.2,
                    :k1 => -0.1653, :k2 => 0.0679, :p1 => -0.00046, :p2 => 0.00097, :k3 => -0.0167,
                    :rms => 0.4471, :poses => [],
                )
            )
        )
        cal0 = load_opencv_calibration(cv0; file_unit = m)
        @test cal0.camera.lens === opencv_lens([-0.1653, 0.0679, -0.00046, 0.00097, -0.0167])
        @test parameters(cal0.camera.lens).k3 == -0.0167 && isempty(cal0.poses)
        @test cal0.poses isa Vector{Union{Nothing, EuclideanMap{3}}}
        @test cal.images === nothing && cal0.images === nothing           # a third-party export names no images

        # MATLAB, radial + tangential: OpenCV's polynomial under other names
        ml = joinpath(dir, "calibration_matlab_3radial2tangential.json")
        write(
            ml, JSON.json(
                Dict(
                    :fx => 2472.59, :fy => 2470.3, :cx => 2155.82, :cy => 1409.01,
                    :r1 => -0.1642, :r2 => 0.0662, :r3 => -0.0155, :t1 => -0.000515, :t2 => 0.00092,
                    :mean_error => 0.4036,
                    :poses => [Dict(:rvec => rvec, :tvec => [-708.695, 148.6439, 1212.2104])],
                )
            )
        )
        calm = load_matlab_calibration(ml; file_unit = mm)                # MATLAB's documented (1, 1) frame by default
        @test calm isa ExternalCalibration{<:CameraModel} && calm.source === :matlab
        @test calm.error_statistic === :mean && calm.reprojection_error == 0.4036
        Km = calm.camera.intrinsics.K
        @test Km[1, 3] ≈ 2155.82 - 0.5 atol = 1.0e-12        # MATLAB's (1, 1) frame → COLMAP
        @test Km[2, 3] ≈ 1409.01 - 0.5 atol = 1.0e-12
        lm = parameters(calm.camera.lens)
        @test (lm.k1, lm.k2, lm.k3, lm.p1, lm.p2) == (-0.1642, 0.0662, -0.0155, -0.000515, 0.00092)
        @test calm.camera.lens === opencv_lens([-0.1642, 0.0662, -0.000515, 0.00092, -0.0155])
        @test calm.poses[1].t == [-708.695, 148.6439, 1212.2104]   # already millimetres
        @test _pose_frames(calm.poses[1]) == (TargetFrame{:calibration_target}, CameraFrame)
        # the file's frame defaults to MATLAB's documented one and is overridden
        # positionally, for a file verified to be in another; the unit has no default
        @test load_matlab_calibration(PixelFrame{:julia}, ml; file_unit = mm).camera == calm.camera
        @test_throws UndefKeywordError load_matlab_calibration(ml)
        @test load_matlab_calibration(PixelFrame{:opencv}, ml; file_unit = mm).camera.intrinsics.K[1, 3] ≈ 2155.82 + 0.5 atol = 1.0e-12

        # MATLAB fisheye: Scaramuzza's lift — a0 is the focal, the stretch matrix and
        # the distortion centre are K, the polynomial is normalized by a0
        a0, a2, a3, a4 = 2470.92, -6.73e-5, -1.03e-9, 2.32e-12
        S = [1.0009 -0.0043; 0.004 1.0]
        fe = joinpath(dir, "calibration_matlab_fisheye.json")
        write(
            fe, JSON.json(
                Dict(
                    :model => "fisheye", :mapping_coeffs => [a0, a2, a3, a4],
                    :distortion_center => [2148.0, 1414.32],
                    :stretch_matrix => [[S[1, 1], S[1, 2]], [S[2, 1], S[2, 2]]],
                    :mean_error => 0.4955, :poses => [Dict(:rvec => rvec, :tvec => [1.0, 2.0, 3.0])],
                )
            )
        )
        calf = load_matlab_calibration(fe; file_unit = mm)
        @test calf isa ExternalCalibration{<:CameraModel} && calf.camera.lens isa Scaramuzza
        cf = parameters(calf.camera.lens)
        @test cf.d1 ≈ a2 * a0 && cf.d2 ≈ a3 * a0^2 && cf.d3 ≈ a4 * a0^3
        # K is upper triangular: the stretch's rotation part (S = U·Q) went into the camera frame
        φ = atan(S[2, 1], S[2, 2])
        Rz = RotMatrix{3, Float64}(RotZ(φ))
        Kf = calf.camera.intrinsics.K
        @test Kf[2, 1] == 0
        @test Kf[1, 1] ≈ a0 * (S[1, 1] * cos(φ) - S[1, 2] * sin(φ))
        @test Kf[1, 2] ≈ a0 * (S[1, 1] * sin(φ) + S[1, 2] * cos(φ))
        @test Kf[2, 2] ≈ a0 * hypot(S[2, 1], S[2, 2])
        @test Kf[1, 3] ≈ 2148.0 - 0.5 atol = 1.0e-12
        @test Kf[2, 3] ≈ 1414.32 - 0.5 atol = 1.0e-12
        # the model evaluated the way MATLAB states it, independently of VGC's solver:
        # for a ray (X, Y, Z) in MATLAB's camera frame the sensor-plane radius ρ solves
        # ρ·Z = R·(a0 + a2ρ² + a3ρ³ + a4ρ⁴) (bisected on the branch before the fold,
        # ≈ 5066 px), the sensor point is ρ·(X, Y)/R, the pixel stretch·sensor + centre;
        # VGC's camera sees that ray in its own frame, rotated by φ about the axis
        for X in (SVector(0.3, -0.2, 1.0), SVector(-0.8, 0.6, 1.2), SVector(0.05, 0.9, 0.7))
            Rxy = hypot(X[1], X[2])                                             # (not `R`: the rotation above)
            f(ρ) = ρ * X[3] - Rxy * (a0 + a2 * ρ^2 + a3 * ρ^3 + a4 * ρ^4)
            lo, hi = 0.0, 5000.0
            for _ in 1:200
                mid = (lo + hi) / 2
                f(mid) < 0 ? (lo = mid) : (hi = mid)
            end
            ρ = (lo + hi) / 2
            px_matlab = S * (ρ / Rxy .* SVector(X[1], X[2])) .+ SVector(2148.0, 1414.32)
            u = VgtkCore.project(calf.camera, Rz * X)
            @test SVector(u[1], u[2]) ≈ px_matlab .- 0.5 atol = 1.0e-6         # MATLAB's (1, 1) frame → COLMAP
        end
        # and the poses carry the same frame rotation
        @test length(calf.poses) == 1
        @test calf.poses[1].t ≈ Rz * SVector(1.0, 2.0, 3.0)
        @test Matrix(calf.poses[1].R) ≈ Matrix(Rz * R) atol = 1.0e-12
        @test calf.error_statistic === :mean

        # a mirrored MATLAB solution (a0 < 0, target behind the camera) is refused as no camera
        mirrored = joinpath(dir, "calibration_matlab_fisheye_mirrored.json")
        write(
            mirrored, JSON.json(
                Dict(
                    :model => "fisheye", :mapping_coeffs => [-a0, a2, a3, a4],
                    :distortion_center => [2148.0, 1414.32],
                    :stretch_matrix => [[S[1, 1], S[1, 2]], [S[2, 1], S[2, 2]]],
                    :mean_error => 4.68, :poses => [Dict(:rvec => rvec, :tvec => [1.0, 2.0, -3.0])],
                )
            )
        )
        errm = try
            load_matlab_calibration(mirrored; file_unit = mm)
        catch e
            e
        end
        @test errm isa ArgumentError && occursin("mirrored", errm.msg)

        # refusals: a missing required key names the file and the key; a
        # non-length unit is refused before any pose is scaled
        bad = joinpath(dir, "bad.json")
        write(bad, JSON.json(Dict(:fx => 1.0)))
        @test_throws ArgumentError load_opencv_calibration(bad; file_unit = m)
        @test_throws ArgumentError load_matlab_calibration(bad; file_unit = mm)
        @test_throws ArgumentError load_opencv_calibration(cv; file_unit = px)
        err = try
            load_opencv_calibration(bad; file_unit = m)
        catch e
            e
        end
        @test occursin("bad.json", err.msg) && occursin("`k1`", err.msg)
    end

    # =========================================================================
    # save_opencv_calibration — the writer is the reader's inverse
    # =========================================================================
    @testset "save_opencv_calibration round-trips through load_opencv_calibration" begin
        dir = mktempdir()
        cv = joinpath(dir, "cv.json")
        # Twelve coefficients, a pose, and a NULL pose — the reader keeps the
        # positional join, so the writer must keep the null in place.
        write(
            cv, JSON.json(
                Dict(
                    :fx => 2467.46, :fy => 2464.95, :cx => 2158.78, :cy => 1420.35, :rms => 0.4453,
                    :k1 => 0.3227, :k2 => -0.462, :p1 => 0.001024, :p2 => 0.001357,
                    :k3 => -0.1251, :k4 => 0.491, :k5 => -0.4617, :k6 => -0.1989,
                    :s1 => -0.000553, :s2 => 6.807e-6, :s3 => -0.002196, :s4 => 0.00011991,
                    :poses => [
                        Dict(:rvec => [-1.7919, 2.0021, 0.15627], :tvec => [0.16809, 0.005368, 1.39719]),
                        nothing,
                        Dict(:rvec => [0.1, -0.2, 0.3], :tvec => [-0.05, 0.02, 2.0]),
                    ],
                )
            )
        )
        a = load_opencv_calibration(cv; file_unit = m)
        names = ["IMG_0001.png", "IMG_0002.png", "IMG_0003.png"]
        out = joinpath(dir, "roundtrip.json")
        @test save_opencv_calibration(out, a; to_unit = m, images = names) == out
        b = load_opencv_calibration(out; file_unit = m)
        # the join with the images is in the file now, not the caller's
        @test b.images == names
        @test parse_json(read(out, String)).images == names
        # …and a calibration read back writes them again without being told
        @test save_opencv_calibration(joinpath(dir, "again.json"), b; to_unit = m) == joinpath(dir, "again.json")
        @test load_opencv_calibration(joinpath(dir, "again.json"); file_unit = m).images == names
        # every pose needs its image name: none, too few, or an empty one is refused
        @test_throws ArgumentError save_opencv_calibration(joinpath(dir, "n0.json"), a; to_unit = m)
        @test_throws ArgumentError save_opencv_calibration(joinpath(dir, "n1.json"), a; to_unit = m, images = names[1:2])
        @test_throws ArgumentError save_opencv_calibration(joinpath(dir, "n2.json"), a; to_unit = m, images = ["a", "", "c"])
        # a file naming the wrong number of images is refused on read
        bad_names = replace(read(out, String), "\"IMG_0003.png\"" => "")
        bad_names = replace(bad_names, r",\s*\]" => "]")
        write(joinpath(dir, "badnames.json"), bad_names)
        @test_throws ArgumentError load_opencv_calibration(joinpath(dir, "badnames.json"); file_unit = m)

        Ka = SMatrix{3, 3, Float64}(a.camera.intrinsics.K)
        Kb = SMatrix{3, 3, Float64}(b.camera.intrinsics.K)
        @test Ka ≈ Kb atol = 1.0e-12
        la = collect(Float64, VgtkCore.lens_parameters(VgtkCore.lens(a.camera)))
        lb = collect(Float64, VgtkCore.lens_parameters(VgtkCore.lens(b.camera)))
        @test la == lb                                   # all twelve, exactly
        @test length(lb) == 12
        @test VgtkCore.lens(a.camera) === VgtkCore.lens(b.camera)
        # the writer names the coefficients of the lens's own OpenCV vector, no more
        @test !haskey(parse_json(read(out, String)), :tilt_x)
        # a tilted sensor round-trips, and its tilt is never dropped: an unknown key is refused
        tilted = merge(Dict(pairs(parse_json(read(cv, String)))), Dict(:tilt_x => 0.01, :tilt_y => -0.02))
        write(joinpath(dir, "tilt.json"), JSON.json(tilted))
        ct = load_opencv_calibration(joinpath(dir, "tilt.json"); file_unit = m)
        @test opencv_coefficients(ct.camera.lens)[13:14] == [0.01, -0.02]
        save_opencv_calibration(joinpath(dir, "tilt_again.json"), ct; to_unit = m, images = names)
        @test load_opencv_calibration(joinpath(dir, "tilt_again.json"); file_unit = m).camera.lens === ct.camera.lens
        write(joinpath(dir, "extra.json"), JSON.json(merge(tilted, Dict(:tilt_z => 0.0))))
        @test_throws ArgumentError load_opencv_calibration(joinpath(dir, "extra.json"); file_unit = m)
        @test a.reprojection_error == b.reprojection_error
        # The null stays null, in its own position.
        @test [p === nothing for p in a.poses] == [p === nothing for p in b.poses]
        for (x, y) in zip(a.poses, b.poses)
            x === nothing && continue
            @test Matrix(x.R) ≈ Matrix(y.R) atol = 1.0e-12
            @test x.t ≈ y.t atol = 1.0e-9
        end

        # The unit is the writer's to apply, and it is recorded: written in
        # millimetres, the same poses come back the same once read as mm.
        mmfile = joinpath(dir, "mm.json")
        save_opencv_calibration(mmfile, a; to_unit = mm, images = names)
        c = load_opencv_calibration(mmfile; file_unit = mm)
        for (x, y) in zip(a.poses, c.poses)
            x === nothing && continue
            @test x.t ≈ y.t atol = 1.0e-9
        end
        declared = parse_json(read(mmfile, String))
        @test declared.unit == "mm" && declared.image_origin == "opencv"

        # A file that DECLARES its unit and origin is held to them. A metre
        # file read as millimetres would be a thousand times too near and still
        # look plausible; it is refused instead, naming the file, the key and
        # both values. So is a pixel convention other than the declared one.
        wrong_unit = try
            load_opencv_calibration(out; file_unit = mm)
        catch e
            e
        end
        @test wrong_unit isa ArgumentError
        @test occursin("roundtrip.json", wrong_unit.msg) && occursin("`unit`", wrong_unit.msg)
        @test occursin("\"m\"", wrong_unit.msg) && occursin("\"mm\"", wrong_unit.msg)
        wrong_origin = try
            load_opencv_calibration(PixelFrame{:julia}, mmfile; file_unit = mm)
        catch e
            e
        end
        @test wrong_origin isa ArgumentError && occursin("`image_origin`", wrong_origin.msg)
        @test occursin("\"opencv\"", wrong_origin.msg) && occursin("\"julia\"", wrong_origin.msg)
        # The matching declaration reads, in any frame the file was written in:
        # a `:julia` file read as `:julia` is the same camera as the `:opencv` one.
        jlfile = joinpath(dir, "julia.json")
        save_opencv_calibration(PixelFrame{:julia}, jlfile, a; to_unit = mm, images = names)
        @test parse_json(read(jlfile, String)).image_origin == "julia"
        jl = load_opencv_calibration(PixelFrame{:julia}, jlfile; file_unit = mm)
        @test SMatrix{3, 3, Float64}(jl.camera.intrinsics.K) ≈ Ka atol = 1.0e-9
        @test_throws ArgumentError load_opencv_calibration(jlfile; file_unit = mm)   # default :opencv ≠ declared
        # A file that declares nothing — every third-party export — is read as
        # the caller says, whatever that is: the unit is then the caller's to know.
        @test norm(load_opencv_calibration(cv; file_unit = mm).poses[1].t) ≈ norm(a.poses[1].t) / 1000
        # A declaration that is not a name is refused, not coerced.
        odd = joinpath(dir, "odd.json")
        write(odd, replace(read(mmfile, String), "\"unit\": \"mm\"" => "\"unit\": 1"))
        @test occursin("\"unit\": 1", read(odd, String))
        @test_throws ArgumentError load_opencv_calibration(odd; file_unit = mm)

        # A POSE AS A PIPELINE PRODUCES IT, not as the reader returns it: an
        # unlabeled `EuclideanMap` writes the same numbers a labeled one does.
        em = [p === nothing ? nothing : unlabel(p) for p in a.poses]
        emfile = joinpath(dir, "euclideanmap.json")
        save_opencv_calibration(emfile, a.camera, em; rms = 0.4453, to_unit = mm, images = names)
        d = load_opencv_calibration(emfile; file_unit = mm)
        @test [p === nothing for p in d.poses] == [p === nothing for p in a.poses]
        for (x, y) in zip(a.poses, d.poses)
            x === nothing && continue
            @test Matrix(x.R) ≈ Matrix(y.R) atol = 1.0e-12
            @test x.t ≈ y.t atol = 1.0e-9
        end

        # A lens this shape cannot describe is refused, not silently zeroed.
        fisheye = ExternalCalibration(
            :ours, CameraModel(a.camera.intrinsics.K; lens = KannalaBrandt(-0.3)),
            a.poses, 0.1, :rms, "",
        )
        @test_throws ArgumentError save_opencv_calibration(joinpath(dir, "no.json"), fisheye; to_unit = m, images = names)
        # So is a skewed K: the shape has fx, fy, cx, cy and nothing else.
        skewed = VgtkCore.CameraCalibrationMat(
            SMatrix{3, 3, Float64}(Ka[1, 1], 0, 0, 1.5, Ka[2, 2], 0, Ka[1, 3], Ka[2, 3], 1)
        )
        @test_throws ArgumentError save_opencv_calibration(
            joinpath(dir, "skew.json"), CameraModel(skewed; lens = a.camera.lens), a.poses;
            rms = 0.1, to_unit = m, images = names
        )
    end
end
