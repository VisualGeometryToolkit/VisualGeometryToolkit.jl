# =============================================================================
# Every lens family, central and non-central, in the camera store and in a rig
# =============================================================================
#
# IO 0.10: the store (vgio/cameras/v6.0) holds any bounded camera, as a rig's
# sensors do (vgio/rig/v10.0), through the one camera body. Each camera reads
# back exactly by type — a parametric lens `===`, a spline lattice and a
# non-central model value for value — in the store and posed in a rig, bounded by
# its extent or by its physical sensor, in mm and in m.

@testitem "every lens family, central and non-central: the store and a rig round-trip exactly" setup = [VGIoFixtures] begin
    import VisualGeometryToolkitCore as VgtkCore
    import VisualGeometryToolkitIO as VgtkIO
    using VisualGeometryToolkitCore: Rig, PosedCamera, ViewFrame, Sensor, relabel, model, sensor, μm, with_present, lens_coverage, Circle, Point2

    K = CameraCalibrationMat(SMatrix{3, 3, Float64}(900.0, 0, 0, 0, 905.0, 0, 320.5, 240.25, 1))
    lenses = [
        Pinhole(), Lens(Equidistant()),
        Lens(Perspective(), Additive(Radial((-0.15, 0.045)), Tangential(-3.5e-4, 8.4e-4))),
        opencv_lens([-0.1, 0.05, 1.0e-4, -2.0e-4, 0.01, 0.0, 0.0, 0.002]),
        KannalaBrandt(0.1, -0.02, 0.003, -0.0004), Scaramuzza(-0.1664, -0.0063, 0.035),
        Lens(DoubleSphere(0.8838, 0.9)), Lens(VgtkCore.ExtendedUnified(0.6, 1.1)), Lens(VgtkCore.Unified(0.7)),
        LiftLens(Radial((0.01, -0.002), (-0.3, 0.05, -0.004))), Division(-0.28, -0.09),
        Lens(Equidistant(), Additive(Radial((0.1, -0.02, 0.003, -0.0004)), Tangential(1.0e-4, -2.0e-4), ThinPrism((2.0e-4,), (-1.0e-4,)))),
        Lens(Perspective(), VgtkCore.FieldOfView(0.9)),
        Lens(Equidistant(), VgtkCore.Tilt(0.01, -0.02) ∘ Additive(Tangential(3.0e-4, -2.0e-4), ThinPrism((1.0e-3,), (-5.0e-4,))) ∘ Radial((0.1, -0.02))),
    ]
    ext = Size2(640, 480)
    central = [CameraModel(K; lens = l, pixel_frame = :colmap) for l in lenses]
    # the grid cameras: a SplineLens camera with absent nodes, and a non-central one on its lattice
    gc = CameraModel(SplineLens, CameraModel(; focal = 31.7, pp = (32.25, 23.875), lens = KannalaBrandt(-0.02)), Size2(64, 48); grid_size = (6, 5))
    P = trues(size(gc.lens.backward.controls)); P[1, 1] = P[end, end] = false
    gc = with_present(gc, P)
    d = gc.lens.backward
    gn = with_present(NoncentralSplineModel(gc.intrinsics, d, ChartSpline(d.area, [SVector(1.0e-3 * gx, -2.0e-3 * gy, 5.0e-4) for gy in 1:5, gx in 1:6])), P)
    grid = [(gc, Size2(64, 48)), (gn, Size2(64, 48))]
    cams = vcat([(c, ext) for c in central], grid)
    bounded(i, c, e) = isodd(i) ? BoundedCamera(c, e) : BoundedCamera(c, Sensor(e, 3.45μm); lens_coverage = Circle(Point2(e.width / 2, e.height / 2), e.height / 2))
    same(a, b) = typeof(a) == typeof(b) && VgtkIO.same_value(model(a), model(b); length_ulps = 0) && extent(a) == extent(b) &&
        sensor(a) == sensor(b) && lens_coverage(a) == lens_coverage(b)
    mktempdir() do dir
        names = [Symbol("cam_", lpad(i, 2, '0')) for i in eachindex(cams)]
        store = Dict(n => bounded(i, c, e) for (i, (n, (c, e))) in enumerate(zip(names, cams)))
        for u in (mm, m)
            p = joinpath(dir, "store_$(unit_string(u)).json")
            save(p, store; length_unit = u)
            back = load(Dict{Symbol, BoundedCamera}, p)
            @test keys(back) == keys(store) && all(n -> same(back[n], store[n]), names)
            @test all(n -> model(back[n]).lens === model(store[n]).lens, names[1:length(central)])   # parametric: bit for bit
            d = parse_json(read(p, String))
            @test d.schema == "vgio/cameras/v6.0" && d.units.length == unit_string(u)
            @test count(c -> get(c, :type, "") == "noncentral_spline", d.cameras) == 1
            q = joinpath(dir, "again.json"); save(q, back; length_unit = u)
            @test read(q, String) == read(p, String)
        end
        # the same cameras posed in a rig: one sensor each, its extrinsics a length
        # (a pose as a reader returns one: its stored quaternion, its translation lengths; a file-born value reads back `==`)
        X(i) = qpose(canonical_qvec(SVector(1.0, 0.01i, -0.02i, 0.005i) / norm(SVector(1.0, 0.01i, -0.02i, 0.005i))), SVector(10.0i, -2.5i, 0.125i))
        ids = Tuple(Symbol("s", i) for i in eachindex(cams))
        sensors = Tuple(BoundedCamera(PosedCamera(c, relabel(ViewFrame{:s1}, ViewFrame{ids[i]}, X(i - 1))),
                                      isodd(i) ? e : Sensor(e, 3.45μm)) for (i, (c, e)) in enumerate(cams))
        rig = Rig{ViewFrame{:s1}}(sensors)
        for u in (mm, m)
            p = joinpath(dir, "rig_$(unit_string(u)).json")
            save(p, rig; length_unit = u)
            back = load(Rig, p)
            @test VgtkIO.same_value(back, rig)
            @test all(i -> same(back[i], rig[i]), eachindex(cams))
            @test VgtkIO.content_id(back) == VgtkIO.content_id(rig) == VgtkIO.content_id(p)
            d = parse_json(read(p, String))
            @test d.schema == "vgio/rig/v10.0" && length(d.sensors) == length(cams)
            @test d.sensors[end].camera.type == "noncentral_spline" && haskey(d.sensors[end - 1].camera.lens, :spline)
        end
    end
end
