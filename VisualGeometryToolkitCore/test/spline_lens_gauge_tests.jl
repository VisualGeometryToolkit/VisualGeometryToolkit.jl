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
        # an absent node whose windows cover x = 0: no disc
        centre_off = trues(gh, gw); centre_off[round(Int, y0), round(Int, x0)] = false
        @test monotone_radius(with_present(cam, centre_off).lens) == 0
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
