# =============================================================================
# The spline lens record, the non-central record (vgio/generic_camera/v4), and
# the old grid files read through their migrations
# =============================================================================
#
# A camera with a `SplineLens` is stored as any camera: a camera body whose lens
# is the `{"spline": …}` record, in the K-normalized plane. The non-central grid
# camera is the generic_camera/v4 record. An old central grid file
# (vgio/generic_camera/v2, v3) reads as the camera store of its one camera, a
# non-central one through its v3 → v4 step; both backproject as the old models
# did (GOLDEN_IO, recorded with the old reader), and neither is rewritten.

@testsetup module SplineRecordFixtures
    using StaticArrays: SVector
    using VisualGeometryToolkitCore: CameraModel, SplineLens, ChartSpline, KannalaBrandt, with_present
    export spline_camera, corner_cut, with_forward, rays_match, golden_pixels

    # a SplineLens camera from a parametric one, over a 64 × 48 image
    spline_camera() = CameraModel(
        SplineLens, CameraModel(; focal = 31.7, pp = (32.25, 23.875), lens = KannalaBrandt(-4 / (3π^2))), (64, 48); grid_size = (8, 7),
    )
    # its lattice with the two corner nodes of the first row absent
    corner_cut(cam) = (P = trues(size(cam.lens.backward.controls)); P[1, 1] = P[1, end] = false; with_present(cam, P))
    # the same lens with a (made-up) forward spline: its record and its read-back are the thing tested
    function with_forward(cam)
        l = cam.lens
        fwd = ChartSpline(VisualGeometryToolkitCore.Rect2{Float64}(-0.5, -0.4, 1.0, 0.8), [SVector(0.01 * gx - 0.1, 0.02 * gy - 0.3) for gy in 1:5, gx in 1:6])
        return CameraModel(cam.intrinsics, SplineLens(l.backward; forward = fwd, present = l.present))
    end
    import VisualGeometryToolkitCore
    # the golden pixels, a 2 × N matrix, as points
    golden_pixels(g) = [SVector(g.pixels[1, j], g.pixels[2, j]) for j in axes(g.pixels, 2)]
    # NaN exactly where the reference is NaN, else within `tol`
    rays_match(a, b; tol = 1.0e-12) = all(isnan, b) ? all(isnan, a) : (all(isfinite, a) && maximum(abs.(a - b)) <= tol)
end

@testitem "old grid files load and backproject as the old models (GOLDEN_IO)" setup = [VGIoFixtures, SplineRecordFixtures] begin
    using VisualGeometryToolkitCore: backproject, model, lens_coverage, NoncentralSplineModel
    include(joinpath(@__DIR__, "fixtures", "spline_lens_golden_io.jl"))
    fixtures = joinpath(@__DIR__, "fixtures")

    # the non-central fixtures: a v2 file and its v3 rewrite (every node held)
    for (file, g) in (("old/generic_camera_v2.json", GOLDEN_IO.v2_noncentral), ("units/generic_camera.json", GOLDEN_IO.units_noncentral))
        bc = load(BoundedCamera, joinpath(fixtures, file))
        @test model(bc) isa NoncentralSplineModel && extent(bc) == Size2(g.size[1], g.size[2])
        @test lens_coverage(bc) == VisualGeometryToolkitCore.Rect2{Float64}(0, 0, g.size[1], g.size[2])
        R = [backproject(model(bc), p) for p in golden_pixels(g)]
        @test all(j -> rays_match(collect(R[j].direction), g.directions[:, j]), eachindex(R))
        @test all(j -> rays_match(collect(R[j].origin), g.origins[:, j]), eachindex(R))
        @test count(j -> all(isfinite, g.directions[:, j]), eachindex(R)) > 250      # most of the lattice is imaged
    end

    # the masked central v3 fixture: a camera with a spline lens, its nodes and mask the old model's
    g = GOLDEN_IO.units_masked_central
    p = joinpath(fixtures, "units", "generic_camera_masked.json")
    bc = load(BoundedCamera, p)
    cam = model(bc)
    @test cam isa CameraModel && cam.lens isa SplineLens && extent(bc) == Size2(g.size[1], g.size[2])
    @test iszero(cam.intrinsics.K[1, 2])                                        # the gauge: skew 0
    P = cam.lens.present
    @test P == g.present
    C = cam.lens.backward.controls
    @test all(I -> P[I] ? collect(C[I]) == g.grid[:, I[1], I[2]] : all(isnan, C[I]), CartesianIndices(P))
    # every 4 × 4 window holds an absent node: NaN at every golden pixel, as the old model
    @test all(isnan, g.backproject)
    @test all(u -> all(isnan, backproject(cam, u)), golden_pixels(g))
    # the fixture's domain is the camera's lens coverage, which a bare CameraModel would drop
    @test lens_coverage(bc) == VisualGeometryToolkitCore.Ellipse(VisualGeometryToolkitCore.Point2(32.0, 24.0), 30.0, 22.0, 0.0)
    @test_throws ArgumentError load(CameraModel, p)
    # a store of its one camera; the old file is read, never rewritten
    @test only(keys(load(Dict{Symbol, BoundedCamera}, p))) === :camera
    @test_throws ArgumentError VisualGeometryToolkitIO.vgio_migrate(p; dry_run = true)
    @test_throws ArgumentError VisualGeometryToolkitIO.vgio_migrate(joinpath(fixtures, "old", "generic_camera_v2.json"); dry_run = true)
end

@testitem "an old central record's domain becomes the lens coverage; its nominal pinhole the K" setup = [VGIoFixtures] begin
    using VisualGeometryToolkitCore: model, lens_coverage, Ellipse, Point2, backproject
    src = joinpath(@__DIR__, "fixtures", "units", "generic_camera_masked.json")
    base = JSON.parse(read(src, String))
    ell = Any[32.25, 23.875, 33.2, 30.5, 0.125]
    nominal = Any[31.7, 31.9, 32.25, 23.875, 0.0]
    mktempdir() do dir
        write_with(f) = (t = deepcopy(base); f(t); q = joinpath(dir, "v3.json"); write(q, JSON.json(t)); q)
        p = write_with(t -> (t["domain"] = Dict("ellipse" => ell); t["nominal_intrinsics"] = nominal))
        bc = load(BoundedCamera, p)
        @test lens_coverage(bc) == Ellipse(Point2(32.25, 23.875), 33.2, 30.5, 0.125)
        K = model(bc).intrinsics.K
        @test (K[1, 1], K[2, 2], K[1, 3], K[2, 3], K[1, 2]) == Tuple(Float64.(nominal))
        # a bare CameraModel would drop the domain: refused, naming the call that keeps it
        e = try load(CameraModel, p); nothing catch e; e end
        @test e isa ArgumentError && occursin("BoundedCamera", e.msg)
        # the K is a gauge: the camera of the stored pinhole and the camera of the lattice's gauge are one camera
        q = write_with(t -> nothing)
        a, b = model(bc), model(load(BoundedCamera, q))
        @test a.intrinsics.K != b.intrinsics.K && isequal(a.lens.present, b.lens.present)
        pixels = [SVector(x, y) for x in 0.5:3.0:63.5 for y in 0.5:3.0:47.5]
        @test all(u -> isequal(isnan.(backproject(a, u)), isnan.(backproject(b, u))), pixels)
        # a skewed nominal pinhole has no SplineLens camera
        r = write_with(t -> (t["nominal_intrinsics"] = Any[31.7, 31.9, 32.25, 23.875, 0.5]))
        e = try load(BoundedCamera, r); nothing catch e; e end
        @test e isa ArgumentError && occursin("skew", e.msg)
        # a v2 central record (every node) reads too, as a v2 record never held `present`
        v2 = write_with(t -> (t["schema"] = "vgio/generic_camera/v2"))
        e = try load(BoundedCamera, v2); nothing catch e; e end
        @test e isa ArgumentError && occursin("present", e.msg)
    end
end

@testitem "the spline lens record: a masked lens with a forward spline round-trips bit for bit" setup = [VGIoFixtures, SplineRecordFixtures] begin
    using VisualGeometryToolkitCore: model, lens_coverage, Ellipse, Point2, backproject
    cam = with_forward(corner_cut(spline_camera()))
    ell = Ellipse(Point2(32.25, 23.875), 33.2, 30.5, 0.125)
    mktempdir() do dir
        p = joinpath(dir, "cams.json")
        save(p, Dict(:grid => BoundedCamera(cam, Size2(64, 48); lens_coverage = ell)))
        d = parse_json(read(p, String))
        lens = only(d.cameras).lens
        @test String.(collect(keys(lens))) == ["spline"]
        sp = lens.spline
        @test sort!(collect(String.(keys(sp)))) == ["area", "controls", "forward", "grid_size", "present"]
        @test sp.grid_size == [8, 7] && length(sp.controls) == 8 * 7 - 2 && sp.present[1] == [0, [[1, 6]]]
        @test sort!(collect(String.(keys(sp.forward)))) == ["area", "controls", "grid_size"]
        bc = load(BoundedCamera, p)
        back = model(bc)
        b, a = back.lens, cam.lens
        @test b.backward.area === a.backward.area && isequal(b.backward.controls, a.backward.controls)   # NaN in the absent slots
        @test b.present == a.present && b.forward.area === a.forward.area && b.forward.controls == a.forward.controls
        @test back.intrinsics.K == cam.intrinsics.K && lens_coverage(bc) == ell
        q = joinpath(dir, "again.json"); save(q, Dict(:grid => bc))
        @test read(q, String) == read(p, String)                       # the file a lens writes is its own
        # no mask, no forward: neither key
        save(q, Dict(:grid => BoundedCamera(spline_camera(), Size2(64, 48))))
        sp0 = only(parse_json(read(q, String)).cameras).lens.spline
        @test !haskey(sp0, :present) && !haskey(sp0, :forward)
        # a malformed record is refused by name
        text = read(p, String)
        for (why, f!) in (
                "entries" => t -> pop!(t["cameras"][1]["lens"]["spline"]["controls"]),
                "apart" => t -> (t["cameras"][1]["lens"]["spline"]["present"][1][2] = Any[Any[1, 3], Any[4, 6]]),
                "4 × 4" => t -> (t["cameras"][1]["lens"]["spline"]["forward"]["grid_size"] = Any[3, 5]),
                "positive widths" => t -> (t["cameras"][1]["lens"]["spline"]["area"][3] = 0.0),
                "alone" => t -> (t["cameras"][1]["lens"]["radial"] = Any[]),
            )
            t = JSON.parse(text); f!(t)
            bad = joinpath(dir, "bad.json"); write(bad, JSON.json(t))
            e = try load(BoundedCamera, bad); nothing catch e; e end
            @test e isa ArgumentError && occursin(why, e.msg)
        end
    end
end

@testitem "Review Focus 4: a SplineLens camera in another pixel frame moves only its K" setup = [VGIoFixtures, SplineRecordFixtures] begin
    import VisualGeometryToolkitIO as VgtkIO
    using VisualGeometryToolkitCore: backproject
    cam = corner_cut(spline_camera())
    ext = Size2(64, 48)
    a = VgtkIO.camera_json(cam, ext)
    o = VgtkIO.camera_json(cam, ext, PixelFrame{:opencv})
    @test o.lens == a.lens                                              # the lens record is frame-free (x = K⁻¹u)
    @test (o.K.cx, o.K.cy) == (a.K.cx - 0.5, a.K.cy - 0.5) && (o.K.fx, o.K.fy) == (a.K.fx, a.K.fy)
    back, e = VgtkIO.camera_from_json(JSON.parse(JSON.json(o)), PixelFrame{:opencv})
    @test e == ext && isequal(back.lens.backward.controls, cam.lens.backward.controls) && back.lens.backward.area === cam.lens.backward.area
    pixels = [SVector(x, y) for x in 0.25:1.5:63.75 for y in 0.25:1.5:47.75]
    same_ray(r, s) = all(isnan, s) ? all(isnan, r) : maximum(abs.(r - s)) <= 1.0e-12
    @test all(u -> same_ray(backproject(back, u), backproject(cam, u)), pixels)
    @test count(u -> all(isfinite, backproject(cam, u)), pixels) > length(pixels) ÷ 2
end

@testitem "exporters refuse a SplineLens camera through one message" setup = [VGIoFixtures, SplineRecordFixtures] begin
    import VisualGeometryToolkitIO as VgtkIO
    cam = spline_camera()
    bc = BoundedCamera(cam, Size2(64, 48))
    msg = "a SplineLens camera has no OpenCV/COLMAP form; store it as a vgio camera record (VisualGeometryToolkitIO.save), " *
        "or fit a parametric lens to it first (VisualGeometryToolkitEstimation.fit_camera)"
    refused(f) = (e = try f(); nothing catch e; e end; e isa ArgumentError && e.msg == msg)
    @test refused(() -> VgtkIO.opencv_parameters(cam))
    @test refused(() -> VgtkIO.colmap_cameras(Dict(:c => bc)))
    pose = EuclideanMap(one(RotMatrix{3, Float64}), SVector(0.0, 0.0, 1.0) .* mm)
    mktempdir() do dir
        @test refused(() -> VgtkIO.save_opencv_calibration(joinpath(dir, "cal.yml"), cam, [pose]; rms = 0.1, to_unit = mm, images = ["a.png"]))
    end
end

@testitem "generic_camera/v4: the non-central record round-trips; the central type is refused" setup = [VGIoFixtures, SplineRecordFixtures] begin
    import VisualGeometryToolkitIO as VgtkIO
    using VisualGeometryToolkitCore: model, lens_coverage, Circle, Point2, backproject, Sensor, sensor, μm, with_present
    c = corner_cut(spline_camera())
    d = c.lens.backward
    pts = [SVector(1e-3 * gx, -2e-3 * gy, 0.0) for gy in 1:7, gx in 1:8]
    nc = with_present(NoncentralSplineModel(c.intrinsics, d, ChartSpline(d.area, pts)), c.lens.present)   # NaN in both splines' absent slots
    circle = Circle(Point2(32.0, 24.0), 30.0)
    mktempdir() do dir
        p = joinpath(dir, "nc.json")
        bc = BoundedCamera(nc, Sensor(Size2(width = 64px, height = 48px), 3.45μm); lens_coverage = circle)
        save(p, bc)
        j = parse_json(read(p, String))
        @test j.schema == "vgio/generic_camera/v4" && j.type == "NoncentralSplineModel" && j.length_unit == "mm"
        @test length(j.spline.directions) == length(j.spline.points) == 8 * 7 - 2 && haskey(j, :pitch)
        back = load(BoundedCamera, p)
        @test VgtkIO.same_value(model(back), nc) && lens_coverage(back) == circle && sensor(back) !== nothing
        q = joinpath(dir, "again.json"); save(q, back)
        @test read(q, String) == read(p, String)
        @test_throws ArgumentError load(CameraModel, p)                  # not a CameraModel
        pixels = [SVector(x, y) for x in 0.5:4.0:63.5 for y in 0.5:4.0:47.5]
        @test all(u -> isequal(backproject(model(back), u), backproject(nc, u)), pixels)
        text = read(p, String)
        for (why, f!) in (
                "load(BoundedCamera, path)" => t -> (t["type"] = "CentralGenericModel"),
                "entries" => t -> pop!(t["spline"]["points"]),
                "skew" => t -> (t["K"]["skew"] = 0.25),
                "schema" => t -> (t["schema"] = "vgio/generic_camera/v5"),
            )
            t = JSON.parse(text); f!(t)
            bad = joinpath(dir, "bad.json"); write(bad, JSON.json(t))
            e = try load(BoundedCamera, bad); nothing catch e; e end
            @test e isa ArgumentError && occursin(why, e.msg)
        end
        # a central v3 record has no v4 form: its step names the read that takes it
        old = JSON.parse(read(joinpath(@__DIR__, "fixtures", "units", "generic_camera_masked.json"), String))
        e = try VgtkIO.migrate_record(old); nothing catch e; e end
        @test e isa ArgumentError && occursin("load(BoundedCamera, path)", e.msg)
        # a non-central v3 record with a domain: the v4 record's lens coverage
        v3 = JSON.parse(read(joinpath(@__DIR__, "fixtures", "units", "generic_camera.json"), String))
        v3["domain"] = Dict("ellipse" => Any[32.0, 24.0, 30.0, 25.0, 0.0])
        t = VgtkIO.migrate_record(v3)
        @test t["schema"] == "vgio/generic_camera/v4" && t["lens_coverage"] == Dict("ellipse" => Any[32.0, 24.0, 30.0, 25.0, 0.0])
        @test t["spline"]["points"] == v3["point_grid"]                  # the points as the file holds them
    end
end

@testitem "rig/v9: spline and non-central sensors round-trip; their identity" setup = [VGIoFixtures, SplineRecordFixtures] begin
    import VisualGeometryToolkitIO as VgtkIO
    using VisualGeometryToolkitCore: Rig, PosedCamera, ViewFrame, relabel, model, lens_coverage, Circle, Point2, with_present
    using Rotations: RotXYZ
    c = corner_cut(spline_camera())
    d = c.lens.backward
    nc = with_present(NoncentralSplineModel(c.intrinsics, d, ChartSpline(d.area, [SVector(1e-3 * gx, -2e-3 * gy, 0.0) for gy in 1:7, gx in 1:8])), c.lens.present)
    X(t) = EuclideanMap(RotMatrix{3, Float64}(RotXYZ(0.0, 0.0, 0.0)), [t * mm, 0.0mm, 0.0mm])
    circle = Circle(Point2(32.0, 24.0), 30.0)
    rig = Rig{ViewFrame{:g}}((
        BoundedCamera(PosedCamera(c, relabel(ViewFrame{:g}, ViewFrame{:a}, X(0.0))), Size2(64, 48); lens_coverage = circle),
        BoundedCamera(PosedCamera(nc, relabel(ViewFrame{:g}, ViewFrame{:b}, X(10.0))), Size2(64, 48)),
    ))
    mktempdir() do dir
        p = joinpath(dir, "rig.json"); save(p, rig)
        j = parse_json(read(p, String))
        @test j.schema == "vgio/rig/v9" && haskey(j.sensors[1].camera, :lens) && j.sensors[2].camera.type == "NoncentralSplineModel"
        back = load(Rig, p)
        @test isequal(model(back[1]).lens.backward.controls, c.lens.backward.controls) && lens_coverage(back[1]) == circle
        @test VgtkIO.same_value(model(back[2]), nc)
        @test VgtkIO.content_id(back) == VgtkIO.content_id(rig) == VgtkIO.content_id(p)
        cams = [x["camera"] for x in VgtkIO._identity_tree(rig)["sensors"]]
        @test sort!(collect(keys(cams[1]))) == ["extent", "intrinsics", "lens", "lens_coverage"]
        @test sort!(collect(keys(cams[2]))) == ["extent", "intrinsics", "noncentral_spline"]
        @test sort!(collect(keys(cams[1]["lens"]["spline"]))) == ["area", "controls", "grid_size", "present"]
    end
end

@testitem "rig/v4 → v9: a non-central grid sensor backprojects as the reference's grid" setup = [VGIoFixtures, SchoepsReference] begin
    using VisualGeometryToolkitCore: Rig, model, backproject, NoncentralSplineModel
    # fixtures/v4/rig_canonical_m.json's grid sensor, against an independent
    # evaluation of the reference's B-spline surfaces on its grids as stored (the
    # rig's metres: the point grid as the file holds it)
    path = joinpath(@__DIR__, "fixtures", "v4", "rig_canonical_m.json")
    raw = JSON.parse(read(path, String))
    i = findfirst(s -> haskey(s["camera"], "type"), raw["sensors"])
    b = raw["sensors"][i]["camera"]
    gw, gh = b["grid_width"], b["grid_height"]
    grid(tag) = [Float64.(b[tag][y * gw + x + 1]) for x in 0:(gw - 1), y in 0:(gh - 1)]   # [x+1, y+1]
    D, Pg = grid("direction_grid"), grid("point_grid")
    rig = load(Rig, path)
    cam = model(rig[i])
    @test cam isa NoncentralSplineModel
    lims = (; gw, gh, cminx = b["calibration_min_x"], cminy = b["calibration_min_y"], cmaxx = b["calibration_max_x"], cmaxy = b["calibration_max_y"])
    worst_d = worst_o = 0.0
    for px in range(lims.cminx, lims.cmaxx + 0.9; length = 9), py in range(lims.cminy, lims.cmaxy + 0.9; length = 7)
        gx, gy = schoeps_grid_coordinate(px, py; lims...)
        d = schoeps_surface(D, gx, gy); d = d ./ norm(d)
        o = schoeps_surface(Pg, gx, gy)
        r = backproject(cam, SVector(px, py))
        worst_d = max(worst_d, norm(collect(r.direction) - d))
        δ = collect(r.origin) - o
        worst_o = max(worst_o, norm(δ - (δ' * d) * d))
    end
    @test worst_d < 1.0e-12 && worst_o < 1.0e-12
end
