# The lens conventions against their reference implementations: the pixels
# OpenCV (cv2.projectPoints, cv2.fisheye.projectPoints) and COLMAP
# (pycolmap.Camera.img_from_cam) computed on fixed rays — frozen in
# test/oracles/lens_conventions.json by test/oracles/generate.py — must be what
# the adapters' lenses compute. Different evaluation orders differ in the last
# bits, so the bar is 1e-9 px on pixel coordinates of order 10³.

@testitem "lens conventions: OpenCV, COLMAP, Kalibr" setup = [VGIoFixtures] begin
    using VisualGeometryIo: opencv_lens, opencv_coefficients, kalibr_lens, ColmapCamera, camera_model, colmap_camera
    using VisualGeometryCore: project, Lens, Perspective, Equidistant, DoubleSphere, Unified, ExtendedUnified,
        FieldOfView, Radial, Additive, Tangential

    oracle = JSON3.read(read(joinpath(@__DIR__, "oracles", "lens_conventions.json"), String))
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
            got = [VisualGeometryCore.project(cam, SVector{3, Float64}(X...)) for X in c.rays]
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
