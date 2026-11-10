@testitem "SplineLens: K is a frozen gauge; radial-only methods refuse it" begin
    using Test
    using StaticArrays
    using LinearAlgebra: norm
    using Rotations: RotXYZ, RotMatrix
    using VisualGeometryToolkitCore
    using VisualGeometryToolkitCore: CameraModel, SplineLens, ChartSpline, KannalaBrandt, Lens, Perspective, Equidistant,
        Radial, Tangential, ThinPrism, Tilt, Additive, FieldOfView, px, calibration_matrix, with_present, valid_domain, Rect2,
        parameter_names, parameter_vector, parameter_bounds, with_parameters, monotone_radius,
        horizon_radius, fov_radius, angle_of_view, BoundedCamera, Size2, RadialCamera, PosedCamera, EuclideanMap,
        backproject, ParameterManifold, minimal_manifold
    using GeometryBasics: origin, widths

    # =============================================================================
    # A SplineLens camera's K is the gauge pinhole at the optical-axis pixel, never a
    # parameter: the camera's parameter protocol reads the lens's controls alone. The
    # radial camera is exact only for a radially symmetric lens, so it refuses a grid
    # and any decentering piece.
    # =============================================================================

    include(joinpath(pkgdir(VisualGeometryToolkitCore), "test", "fixtures", "spline_lens_golden.jl"))
    G = GOLDEN
    src = CameraModel(G.f * px, SA[G.pp[1] * px, G.pp[2] * px]; lens = KannalaBrandt(G.kb...))
    image = (G.image[1], G.image[2])
    cam = CameraModel(SplineLens, src, image; grid_size = (G.grid_size[1], G.grid_size[2]))
    masked = with_present(cam, G.masked.present)
    gh, gw = size(cam.lens.backward.controls)

    @testset "the parameters are the lens's controls" begin
        names = parameter_names(cam)
        @test length(names) == 3 * gh * gw
        @test !any(in(names), (:f, :aspect, :skew, :cx, :cy))
        @test names == parameter_names(cam.lens)
        @test first(names) === :c_1_1_x && names[4] === :c_2_1_x          # column-major over [row, col], x y z per node
        θ = parameter_vector(cam)
        @test θ == parameter_vector(cam.lens) && length(θ) == length(names)
        @test θ[1:3] == cam.lens.backward.controls[1, 1]
        lo, hi = parameter_bounds(cam)
        @test all(==(-Inf), lo) && all(==(Inf), hi) && length(lo) == length(θ)
        # a masked lens holds coordinates for its present nodes only
        @test length(parameter_vector(masked)) == 3 * count(G.masked.present)
        @test all(isfinite, parameter_vector(masked))
        # a parametric camera keeps its intrinsics first
        @test parameter_names(src)[1:5] == (:f, :aspect, :skew, :cx, :cy)
    end

    @testset "with_parameters keeps K identical" begin
        θ = parameter_vector(cam)
        @test with_parameters(cam, θ).lens.backward.controls == cam.lens.backward.controls
        θ2 = copy(θ); θ2[3 * (gh * (gw ÷ 2) + gh ÷ 2) + 1] += 1.0e-3     # move one interior control
        c2 = with_parameters(cam, θ2)
        @test calibration_matrix(c2) === calibration_matrix(cam)
        @test c2.intrinsics === cam.intrinsics
        @test parameter_vector(c2) == θ2
        @test c2.lens.backward.area == cam.lens.backward.area
        m2 = with_parameters(masked, parameter_vector(masked) .* 1.0)
        @test m2.lens.present == masked.lens.present
        @test isequal(m2.lens.backward.controls, masked.lens.backward.controls)   # absent controls kept (NaN)
        @test_throws ArgumentError with_parameters(cam, θ[1:(end - 1)])
        # a lens with a forward spline refuses new controls (its forward would be stale)
        forward = ChartSpline(Rect2(-1.0, -1.0, 2.0, 2.0), fill(SVector(0.0, 0.0), 4, 4))
        with_forward = CameraModel(cam.intrinsics, SplineLens(cam.lens.backward; forward))
        @test_throws ArgumentError with_parameters(with_forward, θ)
        # the static parameter chart refuses a heap vector of controls (an ArgumentError, not an n × n static matrix)
        small = CameraModel(SplineLens, src, image; grid_size = (5, 4))
        @test_throws ArgumentError ParameterManifold(small)
        @test_throws ArgumentError minimal_manifold(cam)
        @test_throws ArgumentError ParameterManifold(cam.lens)
    end

    @testset "horizon from the grid" begin
        f = calibration_matrix(cam)[1, 1]
        pp = SVector(calibration_matrix(cam)[1, 3], calibration_matrix(cam)[2, 3])
        d = valid_domain(cam)
        inscribed = minimum(vcat(pp - origin(d), origin(d) + widths(d) - pp))
        @test isfinite(horizon_radius(cam))
        @test horizon_radius(src) > inscribed            # the grid's clip binds, not the KB source's fold
        expect = min(horizon_radius(src), inscribed)
        @test abs(horizon_radius(cam) - expect) <= 0.01 * expect
        # the golden mask drops the 3 outermost node rings: the support is the valid-domain rectangle
        dm = valid_domain(masked)
        @test horizon_radius(masked) ≈ minimum(vcat(pp - origin(dm), origin(dm) + widths(dm) - pp)) rtol = 1.0e-9
        # one absent interior node off the axis poisons the 16 cells of its windows: the block of
        # nodes c0 − 2 … c0 + 2 (rows likewise), [o + (c0 − 4)h, o + c0·h] in x
        b = cam.lens.backward
        o, w = origin(b.area), widths(b.area)
        h = (w[1] / (gw - 3), w[2] / (gh - 3))
        x0 = 2 + (0 - o[1]) / h[1]; y0 = 2 + (0 - o[2]) / h[2]             # x = 0 at node coordinates (x0, y0)
        c0 = ceil(Int, x0) + 4; r0 = round(Int, y0)
        @test 4 <= c0 <= gw - 3 && 4 <= r0 <= gh - 3                     # interior: its windows are all on the lattice
        lo_x = o[1] + (c0 - 4) * h[1]
        @test lo_x > 0 && o[2] + (r0 - 4) * h[2] < 0 < o[2] + r0 * h[2]  # the block lies right of x = 0, across y = 0
        one_off = trues(gh, gw); one_off[r0, c0] = false
        @test monotone_radius(with_present(cam, one_off).lens) ≈ min(lo_x, monotone_radius(cam.lens)) rtol = 1.0e-12
        @test monotone_radius(with_present(cam, one_off).lens) < monotone_radius(cam.lens)
        # an absent node whose windows cover x = 0: the lattice no longer images ẑ there — the gauge's own fallback
        # condition (`_imaged_z`) — so the radius about the principal point is refused, not 0
        centre_off = trues(gh, gw); centre_off[round(Int, y0), round(Int, x0)] = false
        @test_throws ArgumentError monotone_radius(with_present(cam, centre_off).lens)
        for θ in (0.1, 0.5, 1.0)
            @test fov_radius(cam, θ) == f * tan(θ)
        end
    end

    @testset "angle of view" begin
        a = angle_of_view(BoundedCamera(cam, Size2(1000, 1000)))
        b = angle_of_view(BoundedCamera(src, Size2(1000, 1000)))
        @test abs(a - b) <= deg2rad(0.5)
    end

    @testset "RadialCamera refuses a lens that is not radially symmetric" begin
        X = EuclideanMap(RotMatrix{3, Float64}(RotXYZ(0.1, -0.05, 0.02)), SVector(0.1, -0.2, 2.0))
        K = calibration_matrix(src)
        bc = CameraModel(K; lens = Lens(Perspective(), Additive(Radial((-0.2, 0.05)), Tangential(1.0e-3, -5.0e-4))))
        kb_tan = CameraModel(K; lens = Lens(Equidistant(), Additive(Radial((0.01, -0.002)), Tangential(1.0e-3, 2.0e-4))))
        prism = CameraModel(K; lens = Lens(Perspective(), Additive(Radial((-0.1,)), ThinPrism((1.0e-3,), (2.0e-4,)))))
        tilted = CameraModel(K; lens = Lens(Perspective(), Tilt(0.02, -0.01) ∘ Radial((-0.1,))))
        tangential_alone = CameraModel(K; lens = Lens(Perspective(), Tangential(1.0e-3, 0.0)))
        nested = CameraModel(K; lens = Lens(Perspective(), FieldOfView(0.1) ∘ Additive(Radial((-0.1,)), Tangential(1.0e-3, 0.0))))
        for c in (cam, bc, kb_tan, prism, tilted, tangential_alone, nested)
            @test_throws ArgumentError RadialCamera(PosedCamera(c, X))
        end
        @test RadialCamera(PosedCamera(src, X)) isa RadialCamera
        @test RadialCamera(PosedCamera(CameraModel(K; lens = Lens(Perspective(), Radial((-0.2, 0.05)))), X)) isa RadialCamera
        @test RadialCamera(PosedCamera(CameraModel(K; lens = Lens(Perspective(), FieldOfView(0.1) ∘ Radial((-0.1,)))), X)) isa RadialCamera
    end
end

@testitem "SplineLens gauge: a lattice whose centre window holds an absent node gauges by its present nodes" begin
    using Test
    using StaticArrays
    using VisualGeometryToolkitCore
    using VisualGeometryToolkitCore: CameraModel, SplineLens, ChartSpline, KannalaBrandt, LogicalIntrinsics, px, calibration_matrix,
        backproject
    import VisualGeometryToolkitCore.Cameras as VgtkCore

    # Any skew-0 K is an exact gauge of a SplineLens camera (a pixel's lattice
    # coordinate does not depend on it). A lattice whose window at the calibrated
    # area's centre holds an absent node has no local scale there, so the gauge
    # takes the paraxial Jacobian discretely, from the present nodes nearest the
    # centre; with no two adjacent present nodes it has none, and refuses.
    W, H = 640, 480
    src = CameraModel(400.0px, SA[321.5px, 238.25px]; lens = KannalaBrandt(-0.02, 0.003, 0.0, 0.0))
    cam = CameraModel(SplineLens, src, (W, H); grid_size = (12, 10))
    C = cam.lens.backward.controls
    gh, gw = size(C)
    rect = VgtkCore._bounds_rect(0, 0, W - 1, H - 1)
    nan_absent(P) = map((v, p) -> p ? v : SVector(NaN, NaN, NaN), C, P)
    gauge(P) = VgtkCore._spline_gauge(ChartSpline(rect, nan_absent(P)), rect)
    camera(K, P) = VgtkCore._spline_camera(K, rect, nan_absent(P); present = P)
    pixels = [SVector(x, y) for y in 0.5:7.0:(H - 0.5) for x in 0.5:7.0:(W - 0.5)]
    same_rays(a, b) = all(pixels) do u
        p, q = backproject(a, u), backproject(b, u)
        all(isnan, p) ? all(isnan, q) : (all(isfinite, q) && maximum(abs.(p - q)) <= 1e-12)
    end

    @testset "a hole at the centre: the gauge is finite, skew 0, and backproject is any skew-0 K's" begin
        P = trues(gh, gw)
        P[(gh ÷ 2):(gh ÷ 2 + 1), (gw ÷ 2):(gw ÷ 2 + 1)] .= false
        K = gauge(P)
        Kc = K.K
        @test all(isfinite, Kc) && iszero(Kc[1, 2]) && Kc[1, 1] > 0 && Kc[2, 2] > 0
        @test Kc[1, 3] == W / 2 && Kc[2, 3] == H / 2                     # the calibrated area's centre
        a, b = camera(K, P), camera(src.intrinsics, P)
        @test count(u -> all(isfinite, backproject(a, u)), pixels) > length(pixels) ÷ 2
        @test same_rays(a, b)
    end

    @testset "every window holds an absent node: the gauge is finite, backproject NaN for every K alike" begin
        P = [!(gy % 3 == 0 && gx % 3 == 0) for gy in 1:gh, gx in 1:gw]   # 4 consecutive rows and columns meet a multiple of 3
        K = gauge(P)
        Kc = K.K
        @test all(isfinite, Kc) && iszero(Kc[1, 2]) && Kc[1, 1] > 0 && Kc[2, 2] > 0
        a, b = camera(K, P), camera(src.intrinsics, P)
        @test all(u -> all(isnan, backproject(a, u)), pixels) && same_rays(a, b)
    end

    @testset "no two adjacent present nodes: refused, naming the cause and the remedy" begin
        P = [iseven(gx + gy) for gy in 1:gh, gx in 1:gw]                 # a checkerboard
        e = try
            gauge(P); nothing
        catch err
            err
        end
        @test e isa ArgumentError && occursin("adjacent nodes", e.msg) && occursin("supply K", e.msg)
    end
end

@testitem "CameraModel(SplineLens, controls, size): the camera of a given pixel lattice" begin
    using Test
    using StaticArrays
    using VisualGeometryToolkitCore
    using VisualGeometryToolkitCore: CameraModel, SplineLens, KannalaBrandt, LogicalIntrinsics, CameraCalibrationMat, px, backproject

    # The reference's form of a grid camera (Schöps' intrinsics0.yaml, a stored
    # grid record): direction controls over a calibrated pixel area. Its K is the
    # given one or the gauge; either way each pixel backprojects as the lattice does.
    src = CameraModel(400.0px, SA[321.5px, 238.25px]; lens = KannalaBrandt(-0.02, 0.003, 0.0, 0.0))
    area = (8, 6, 631, 473)
    cam = CameraModel(SplineLens, src, (640, 480); grid_size = (12, 10), calib_area = area)
    C = cam.lens.backward.controls
    given = CameraModel(SplineLens, C, (640, 480); calib_area = area, intrinsics = cam.intrinsics)
    @test given.intrinsics.K == cam.intrinsics.K && given.lens.backward.area === cam.lens.backward.area && given.lens.backward.controls == C
    gauged = CameraModel(SplineLens, C, (640, 480); calib_area = area)
    @test iszero(gauged.intrinsics.K[1, 2]) && gauged.intrinsics.K != cam.intrinsics.K
    pixels = [SVector(x, y) for y in 6.5:9.0:473.5 for x in 8.5:9.0:631.5]   # inside the calibrated area
    @test all(u -> maximum(abs.(backproject(gauged, u) - backproject(cam, u))) <= 1.0e-12, pixels)
    # a mask, and the reference reader's normalization
    P = trues(size(C)); P[1, 1] = false
    m = CameraModel(SplineLens, 2 .* C, (640, 480); calib_area = area, intrinsics = cam.intrinsics, present = P, unit_controls = true)
    @test m.lens.present == P && all(v -> abs(sqrt(sum(abs2, v)) - 1) < 1.0e-12, m.lens.backward.controls[P])
    # the gauge reads no absent node: finite placeholders there gauge as NaN does (a hole at the centre: the node tier)
    H = trues(size(C)); H[5:6, 6:7] .= false
    nan_C = map((v, p) -> p ? v : SVector(NaN, NaN, NaN), C, H)
    a, b = CameraModel(SplineLens, C, (640, 480); calib_area = area, present = H), CameraModel(SplineLens, nan_C, (640, 480); calib_area = area, present = H)
    @test a.intrinsics.K == b.intrinsics.K && all(v -> all(isnan, v), a.lens.backward.controls[.!H])
    # a skewed or flipped K has no SplineLens camera, given or a source's (one check, the one constructor's)
    flipped = CameraModel(-400.0px, SA[321.5px, 238.25px]; lens = KannalaBrandt(-0.02, 0.003, 0.0, 0.0))
    @test_throws ArgumentError CameraModel(SplineLens, flipped, (640, 480); grid_size = (12, 10))
    skewed = LogicalIntrinsics(CameraCalibrationMat(SMatrix{3, 3, Float64}(400.0, 0, 0, 0.5, 400.0, 0, 320.0, 240.0, 1)))
    @test_throws ArgumentError CameraModel(SplineLens, C, (640, 480); calib_area = area, intrinsics = skewed)
    @test_throws ArgumentError CameraModel(SplineLens, C, (640, 480); calib_area = (0, 0, 700, 479))   # outside the image
end

@testitem "SplineLens: every camera of the lens has a skew-0 gauge K, through one check" begin
    using Test
    using StaticArrays
    using VisualGeometryToolkitCore
    using VisualGeometryToolkitCore: CameraModel, SplineLens, NoncentralSplineModel, ChartSpline, KannalaBrandt, px,
        calibration_matrix, LogicalIntrinsics, CameraCalibrationMat

    src = CameraModel(300.0px, SA[500.0px, 500.0px]; lens = KannalaBrandt(-0.02))
    cam = CameraModel(SplineLens, src, (1000, 1000); grid_size = (12, 12))
    K = calibration_matrix(cam)
    skewed = LogicalIntrinsics(CameraCalibrationMat(SMatrix{3, 3, Float64}(K[1, 1], 0.0, 0.0, 0.25, K[2, 2], 0.0, K[1, 3], K[2, 3], 1.0)))
    message(f) = (e = try f(); nothing catch e; e end; e isa ArgumentError ? e.msg : e)
    # the plain constructor, as a reader builds it
    m_plain = message(() -> CameraModel(skewed, cam.lens))
    @test m_plain isa String && occursin("skew", m_plain)
    # the controls constructor and the non-central model: the same check, the same message
    m_controls = message(() -> CameraModel(SplineLens, cam.lens.backward.controls, (1000, 1000); intrinsics = skewed))
    d = cam.lens.backward
    m_noncentral = message(() -> NoncentralSplineModel(skewed, d, ChartSpline(d.area, fill(SVector(0.0, 0.0, 0.0), size(d.controls)))))
    @test m_controls == m_plain
    @test m_noncentral == m_plain
    # a skew-0 K builds
    @test CameraModel(cam.intrinsics, cam.lens) isa CameraModel
end

@testitem "SplineLens: CameraModel(cam, K′) re-gauges to any skew-0 K′, a principal-point move included" begin
    using Test
    using StaticArrays
    using LinearAlgebra: norm
    using VisualGeometryToolkitCore
    using VisualGeometryToolkitCore: CameraModel, SplineLens, ChartSpline, KannalaBrandt, px, calibration_matrix,
        backproject, project, CameraCalibrationMat, Rect2, with_present

    src = CameraModel(300.0px, SA[500.0px, 500.0px]; lens = KannalaBrandt(-0.02))
    cam = CameraModel(SplineLens, src, (1000, 1000); grid_size = (16, 14))
    P = trues(size(cam.lens.backward.controls)); P[1, 1] = P[end, 3] = false
    masked = with_present(cam, P)
    K = calibration_matrix(cam)
    Kat(fx, fy, cx, cy) = CameraCalibrationMat(SMatrix{3, 3, Float64}(fx, 0.0, 0.0, 0.0, fy, 0.0, cx, cy, 1.0))
    moved = Kat(0.8 * K[1, 1], 0.9 * K[2, 2], K[1, 3] + 37.25, K[2, 3] - 21.5)        # a principal-point move and a scale
    pixels = [SVector(x, y) for x in 0.5:23.0:999.5 for y in 0.5:19.0:999.5]
    rays = [backproject(cam, u) for u in pixels[1:7:end]]
    same(a, b; tol) = all(isnan, b) ? all(isnan, a) : (all(isfinite, a) && norm(a - b) <= tol)
    for c in (cam, masked)
        c′ = CameraModel(c, moved)
        @test calibration_matrix(c′) == moved
        @test all(u -> same(backproject(c′, u), backproject(c, u); tol = 1.0e-12), pixels)
        @test count(u -> all(isfinite, backproject(c, u)), pixels) > length(pixels) ÷ 2
        # the projection solve finds the same pixel under either gauge, to its tolerance
        @test all(X -> same(project(c′, X), project(c, X); tol = 1.0e-9), rays)
    end
    # a forward's output moves by the scale (its chart, about the principal point's ray, stays): every
    # projection the same; a principal-point move would change that chart, so a lens with a forward refuses it
    fwd = ChartSpline(Rect2(-1.2, -1.2, 2.4, 2.4), [SVector(0.03 * gx - 0.4, -0.02 * gy + 0.3) for gy in 1:6, gx in 1:7])
    cf = CameraModel(cam.intrinsics, SplineLens(cam.lens.backward; forward = fwd))
    cf′ = CameraModel(cf, Kat(0.8 * K[1, 1], 0.9 * K[2, 2], K[1, 3], K[2, 3]))
    @test cf′.lens.chart == cf.lens.chart
    @test count(X -> all(isfinite, project(cf, X)), rays) > length(rays) ÷ 2
    @test all(X -> same(project(cf′, X), project(cf, X); tol = 1.0e-12), rays)
    @test_throws ArgumentError CameraModel(cf, moved)
    # a parametric lens's principal point is data: still refused
    @test_throws ArgumentError CameraModel(src, Kat(K[1, 1], K[2, 2], K[1, 3] + 1.0, K[2, 3]))
end

@testitem "SplineLens: the off-axis gauge refuses the methods that read the principal point as the optical axis" begin
    using Test
    using StaticArrays
    using VisualGeometryToolkitCore
    using VisualGeometryToolkitCore: CameraModel, SplineLens, KannalaBrandt, px, mm, μm, calibration_matrix, backproject,
        undistort, valid_rect, undistorted_rect, angle_of_view, horizon_radius, monotone_radius, image_circle, distortion_map,
        BoundedCamera, Size2, Sensor, CameraCalibrationMat
    import KernelAbstractions as KA

    src = CameraModel(300.0px, SA[500.0px, 500.0px]; lens = KannalaBrandt(-0.02))
    # Review Focus 1: ẑ (the pixel (500, 500)) is outside the calibrated area, so the gauge's principal point is
    # the area's centre, which images a ray off ẑ
    off = CameraModel(SplineLens, u -> backproject(src, u), (1000, 1000); calib_area = (600, 600, 999, 999), init = :quasi_interpolate)
    on = CameraModel(SplineLens, src, (1000, 1000); grid_size = (16, 16))
    img = zeros(64, 64)
    sensor = Sensor(Size2(width = 1000px, height = 1000px), 3.45μm)
    refusals(c) = (
        () -> undistort(c, img; device = KA.CPU()), () -> distortion_map(c), () -> valid_rect(c, (1000, 1000)),
        () -> undistorted_rect(c, (1000, 1000)), () -> angle_of_view(BoundedCamera(c, Size2(1000, 1000))),
        () -> horizon_radius(c), () -> monotone_radius(c.lens), () -> image_circle(c, sensor, 3.0mm),
    )
    for f in refusals(off)
        e = try f(); nothing catch e; e end
        @test e isa ArgumentError && occursin("does not image ẑ", e.msg)
    end
    # the refusal is the gauge's own fallback condition, not a tolerance on the lens axis: a coarse lattice whose
    # approximation error puts its axis off ẑ (IO's 8 × 7 test camera, 4e-4 rad; a 4 × 4 lattice, 1e-2 rad) images ẑ
    # inside its area, and answers
    io = CameraModel(; focal = 31.7, pp = (32.25, 23.875), lens = KannalaBrandt(-4 / (3π^2)))
    coarse = (CameraModel(SplineLens, io, (64, 48); grid_size = (8, 7)), CameraModel(SplineLens, io, (64, 48); grid_size = (4, 4)))
    @test all(c -> VisualGeometryToolkitCore.Cameras._lens_axis(c.lens.backward)[3] < cos(1.0e-4), coarse)   # off ẑ by approximation
    for c in (on, coarse...)
        @test isfinite(horizon_radius(c)) && monotone_radius(c.lens) > 0
        @test size(undistort(c, img; device = KA.CPU())) == size(img)
        @test distortion_map(c) !== nothing
    end
    @test isfinite(angle_of_view(BoundedCamera(on, Size2(1000, 1000))))
    # a principal point moved by CameraModel(camera, K′) is a gauge choice, not the fallback: the lattice images ẑ
    K = calibration_matrix(on)
    moved = CameraModel(on, CameraCalibrationMat(SMatrix{3, 3, Float64}(K[1, 1], 0.0, 0.0, 0.0, K[2, 2], 0.0, K[1, 3] + 40.0, K[2, 3], 1.0)))
    @test isfinite(horizon_radius(moved))
end

@testitem "the pixel-frame change of K: one Core function" begin
    using Test
    using StaticArrays
    using VisualGeometryToolkitCore
    using VisualGeometryToolkitCore: CameraModel, CameraCalibrationMat, PixelFrame, calibration_matrix
    using VisualGeometryToolkitCore.Cameras: reframed_calibration
    K = CameraCalibrationMat(SMatrix{3, 3, Float64}(800.0, 0.0, 0.0, 0.5, 790.0, 0.0, 320.0, 240.0, 1.0))
    R = reframed_calibration(PixelFrame{:opencv}, PixelFrame{:colmap}, K)
    @test (R[1, 1], R[2, 2], R[1, 2]) == (K[1, 1], K[2, 2], K[1, 2]) && (R[1, 3], R[2, 3]) == (320.5, 240.5)
    @test calibration_matrix(CameraModel(K; image_origin = :opencv)) == SMatrix{3, 3, Float64}(R)
    @test reframed_calibration(PixelFrame{:colmap}, PixelFrame{:opencv}, R) == K
    # the warp's device form is `Adapt.adapt` itself: no forwarding wrapper
    @test !isdefined(VisualGeometryToolkitCore, :_device_warp)
end
