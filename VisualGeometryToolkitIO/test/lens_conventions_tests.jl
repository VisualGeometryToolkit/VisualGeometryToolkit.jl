# The lens conventions against their reference implementations: the pixels
# OpenCV (cv2.projectPoints, cv2.fisheye.projectPoints) and COLMAP
# (pycolmap.Camera.img_from_cam) computed on fixed rays — frozen in
# test/oracles/lens_conventions.json by test/oracles/generate.py — must be what
# the adapters' lenses compute. Different evaluation orders differ in the last
# bits, so the bar is 1e-9 px on pixel coordinates of order 10³.

@testitem "lens conventions: OpenCV, COLMAP, Kalibr" setup = [VGIoFixtures] begin
    import VisualGeometryToolkitCore as VgtkCore
    using VisualGeometryToolkitIO: opencv_lens, opencv_coefficients, kalibr_lens, ColmapCamera, camera_model, colmap_camera
    using VisualGeometryToolkitCore: project, Lens, Perspective, Equidistant, DoubleSphere, Unified, ExtendedUnified,
        FieldOfView, Radial, Additive, Tangential

    oracle = parse_json(read(joinpath(@__DIR__, "oracles", "lens_conventions.json"), String))
    pixel(K, lens, X) = (x = project(lens, SVector{3, Float64}(X...)); SVector(K.fx * x[1] + K.cx, K.fy * x[2] + K.cy))
    worst = Dict{String, Float64}()
    for c in oracle.cases
        name = "$(c.convention)/$(c.model)"
        if c.convention == "opencv"
            lens = c.model == "fisheye" ? KannalaBrandt(Float64.(c.coeffs)...) : opencv_lens(Float64.(c.coeffs))
            got = [pixel(c.K, lens, X) for X in c.rays]
            # the coefficient vector round-trips through the lens
            c.model == "fisheye" || @test opencv_coefficients(lens) == Float64.(c.coeffs)
        else
            row = ColmapCamera(1, Symbol(c.model), 1280, 960, Float64.(c.params))
            cam = camera_model(row)
            got = [VgtkCore.project(cam, SVector{3, Float64}(X...)) for X in c.rays]
        end
        want = [SVector{2, Float64}(p...) for p in c.pixels]
        worst[name] = maximum(norm.(got .- want))
        @test worst[name] <= 1.0e-9
    end
    @test length(worst) == 22

    # Kalibr: each camchain pairing is the VGC lens of the model Kalibr's formulas define
    for (cm, intr, dm, d, want) in (
            ("pinhole", [800.0, 810.0, 320.0, 240.0], "radtan", [-0.28, 0.07, 1.0e-3, -5.0e-4], opencv_lens([-0.28, 0.07, 1.0e-3, -5.0e-4])),
            ("pinhole", [800.0, 810.0, 320.0, 240.0], "equidistant", [-0.04, 0.01, -0.008, 0.0015], KannalaBrandt(-0.04, 0.01, -0.008, 0.0015)),
            ("pinhole", [800.0, 810.0, 320.0, 240.0], "fov", [0.9], Lens(Perspective(), FieldOfView(0.9))),
            ("pinhole", [800.0, 810.0, 320.0, 240.0], "none", Float64[], Pinhole()),
            ("omni", [0.8, 800.0, 810.0, 320.0, 240.0], "radtan", [-0.1, 0.01, 1.0e-3, 0.0], Lens(Unified(0.8), Additive(Radial((-0.1, 0.01)), Tangential(1.0e-3, 0.0)))),
            ("ds", [-0.2, 0.59, 800.0, 810.0, 320.0, 240.0], "none", Float64[], Lens(DoubleSphere(-0.2, 0.59))),
            ("eucm", [0.6, 1.1, 800.0, 810.0, 320.0, 240.0], "none", Float64[], Lens(ExtendedUnified(0.6, 1.1))),
        )
        fx, fy, cx, cy, lens = kalibr_lens(cm, intr, dm, d)
        @test (fx, fy, cx, cy) == (intr[end - 3], intr[end - 2], intr[end - 1], intr[end])
        @test lens === want
    end
    @test_throws ArgumentError kalibr_lens("ds", [-0.2, 0.59, 800.0, 810.0, 320.0, 240.0], "radtan", [0.0, 0.0, 0.0, 0.0])
    @test_throws ArgumentError kalibr_lens("pinhole", [800.0, 810.0, 320.0], "none", Float64[])
    # Kalibr's radtan, written out as its RadialTangentialDistortion evaluates it
    k1, k2, r1, r2 = -0.28, 0.07, 1.0e-3, -5.0e-4
    _, _, _, _, rt = kalibr_lens("pinhole", [1.0, 1.0, 0.0, 0.0], "radtan", [k1, k2, r1, r2])
    for (x, y) in ((0.3, -0.2), (-0.1, 0.4), (0.55, 0.35))
        mx2, my2, mxy = x^2, y^2, x * y
        rho2 = mx2 + my2
        rad = k1 * rho2 + k2 * rho2^2
        want = SVector(x + x * rad + 2r1 * mxy + r2 * (rho2 + 2mx2), y + y * rad + 2r2 * mxy + r1 * (rho2 + 2my2))
        @test project(rt, SVector(x, y, 1.0)) ≈ want atol = 1.0e-15
    end
    # OpenCV vectors of other lengths, and lenses OpenCV does not compute, are refused
    @test_throws ArgumentError opencv_lens(zeros(6))
    @test_throws ArgumentError opencv_coefficients(KannalaBrandt(0.1))
    @test_throws ArgumentError opencv_coefficients(Lens(Perspective(), Radial((0.1, 0.2, 0.3, 0.4))))
end

@testitem "OpenCV's camera matrix ↔ a camera in the COLMAP frame" setup = [VGIoFixtures] begin
    import VisualGeometryToolkitCore as VgtkCore
    using VisualGeometryToolkitIO: opencv_camera, opencv_parameters, opencv_lens
    using VisualGeometryToolkitCore: CameraModel, CameraCalibrationMat, LogicalIntrinsics, PixelFrame, project, KannalaBrandt
    using Random: MersenneTwister

    K = [1210.5 0.0 1011.25; 0.0 1208.75 763.5; 0.0 0.0 1.0]
    dist = [-0.15, -0.05, 0.002, 0.006, 0.024, 0.19, 0.026, -0.0013, -0.0016, -0.00097, -0.00076, 0.0021, -0.0099, -0.0062]
    cam = opencv_camera(K, dist)
    @test cam.intrinsics isa LogicalIntrinsics && VgtkCore.lens(cam) == opencv_lens(dist)
    # OpenCV's pixel centre (0, 0) is COLMAP's (½, ½): the principal point moves by half a pixel, the focal lengths stay
    Kc = cam.intrinsics.K
    @test (Kc[1, 1], Kc[2, 2], Kc[1, 3], Kc[2, 3]) == (1210.5, 1208.75, 1011.75, 764.0)
    # the same camera as Core's own origin-shifted construction: equal projections
    ref = CameraModel(CameraCalibrationMat(SMatrix{3, 3, Float64}(K)); lens = opencv_lens(dist), image_origin = :opencv)
    X = SVector(0.3, -0.2, 1.0)
    @test project(cam, X) ≈ project(ref, X) atol = 1e-12
    # another declared frame of the file: MATLAB's (1, 1) centre
    @test opencv_camera(PixelFrame{:julia}, K, dist).intrinsics.K[1, 3] == 1010.75
    # the round trip is bit-exact: K and dist back out, in OpenCV's frame or another
    p = opencv_parameters(cam)
    @test p.K == K && p.dist == dist
    @test opencv_parameters(PixelFrame{:julia}, opencv_camera(PixelFrame{:julia}, K, dist)).K == K
    # the principal point: exact unless c + ½ needs one bit more than its binade holds — c ∈ [2ᵏ − ½, 2ᵏ) with
    # its last mantissa bit set — and then within one ulp of c (the documented exception)
    lossy(c) = (k = exponent(c + 0.5); c >= 2.0^k - 0.5 && c < 2.0^k && isodd(reinterpret(UInt64, c)))
    rtc(c) = opencv_parameters(opencv_camera([1000.0 0.0 c; 0.0 1000.0 c; 0.0 0.0 1.0], dist)).K[1, 3]
    @test rtc(511.5) === 511.5 && rtc(2047.5) === 2047.5 && rtc(511.75) === 511.75 && rtc(1023.5) === 1023.5   # exact halves
    for c in (511.75000000000006, 2047.5000000000002, prevfloat(512.0), prevfloat(1024.0), prevfloat(4096.0))
        @test lossy(c) && rtc(c) != c && abs(rtc(c) - c) == eps(c)                     # the boundary cases: one ulp
    end
    for c in (nextfloat(511.75, 2), nextfloat(2047.5, 2), nextfloat(512.0), 0.5, 1.0, 1000.25, 3071.9999999999995)
        @test !lossy(c) && rtc(c) === c
    end
    @test count(lossy, (511.75000000000006, 511.5, 2047.5000000000002, 2047.5)) == 2
    # every other entry, any c, is the documented guarantee
    rng = MersenneTwister(7)
    for _ in 1:1000
        Kr = [100 + 4000rand(rng) 0.0 1 + 6000rand(rng); 0.0 100 + 4000rand(rng) 1 + 4000rand(rng); 0.0 0.0 1.0]
        dr = 0.1 .* randn(rng, rand(rng, (4, 5, 8, 12, 14)))
        q = opencv_parameters(opencv_camera(Kr, dr))
        @test q.dist == dr && all(q.K[i] == Kr[i] for i in (1, 2, 3, 4, 5, 6, 9))
        @test all(c -> lossy(c[2]) ? abs(c[1] - c[2]) == eps(c[2]) : c[1] === c[2], ((q.K[1, 3], Kr[1, 3]), (q.K[2, 3], Kr[2, 3])))
    end
    # another frame: the shift is −½ and the round trip exact at the boundary cases too
    for c in (511.75000000000006, 2047.5000000000002, prevfloat(512.0))
        Kj = [1000.0 0.0 c; 0.0 1000.0 c; 0.0 0.0 1.0]
        @test opencv_parameters(PixelFrame{:julia}, opencv_camera(PixelFrame{:julia}, Kj, dist)).K == Kj
    end
    # refused: a skewed or non-pinhole K, a lens OpenCV does not compute, a camera without a calibration matrix
    @test_throws ArgumentError opencv_camera([1000.0 1.0 500.0; 0.0 1000.0 400.0; 0.0 0.0 1.0], dist)
    @test_throws ArgumentError opencv_camera([1000.0 0.0 500.0; 0.0 1000.0 400.0; 0.0 0.0 2.0], dist)
    @test_throws ArgumentError opencv_parameters(CameraModel(LogicalIntrinsics(CameraCalibrationMat(SMatrix{3, 3, Float64}(K))), KannalaBrandt(0.1, 0.0, 0.0, 0.0)))
    @test_throws ArgumentError opencv_parameters(CameraModel(LogicalIntrinsics(CameraCalibrationMat(SMatrix{3, 3, Float64}(1000.0, 0, 0, 2.0, 1000.0, 0, 500.0, 400.0, 1))), opencv_lens(dist)))
end
