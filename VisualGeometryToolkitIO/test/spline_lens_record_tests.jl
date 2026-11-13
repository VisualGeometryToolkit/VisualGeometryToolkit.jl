# =============================================================================
# The spline lens record, the non-central record (vgio/noncentral_spline/v1), and
# the old grid files read through their migrations
# =============================================================================
#
# A camera with a `SplineLens` is stored as any camera: a camera body whose lens
# is the `{"spline": …}` record, in the K-normalized plane. The non-central grid
# camera is the noncentral_spline/v1 record. An old central grid file
# (vgio/generic_camera/v2, v3) reads as the camera store of its one camera, a
# non-central one through its step to the noncentral_spline/v1 record; both
# backproject as the old models did (GOLDEN_IO, recorded with the old reader),
# and neither is rewritten: `vgio_migrate` writes a central one's camera store,
# a non-central one's record, to a new file beside it.

@testsetup module SplineRecordFixtures
    using StaticArrays: SVector
    using VisualGeometryToolkitCore: CameraModel, SplineLens, ChartSpline, KannalaBrandt, Size2, with_present
    export spline_camera, corner_cut, with_forward, rays_match, golden_pixels

    # a SplineLens camera from a parametric one, over a 64 × 48 image
    spline_camera() = CameraModel(
        SplineLens, CameraModel(; focal = 31.7, pp = (32.25, 23.875), lens = KannalaBrandt(-4 / (3π^2))), Size2(64, 48); grid_size = (8, 7),
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
        # the far-edge probes (x = W or y = H) were NaN under the old reader's half-open box; the lattice's box is
        # closed on both sides now (Core's SplineLens far edge), so they are imaged, and every other probe is the old one
        far = [g.pixels[1, j] == g.size[1] || g.pixels[2, j] == g.size[2] for j in eachindex(R)]
        @test all(j -> far[j] || rays_match(collect(R[j].direction), g.directions[:, j]), eachindex(R))
        @test all(j -> far[j] || rays_match(collect(R[j].origin), g.origins[:, j]), eachindex(R))
        @test any(far) && all(j -> !far[j] || (all(isfinite, R[j].direction) && all(isfinite, R[j].origin)), eachindex(R))
        @test count(j -> all(isfinite, g.directions[:, j]), eachindex(R)) > 250      # most of the lattice is imaged
    end

    # the masked central v3 fixture: every 4 × 4 window holds an absent node and the file stores no K, so the
    # lattice has no scale to gauge K by — refused, saying to supply K; with a stored K (its nominal pinhole) it
    # loads: a camera with a spline lens, its nodes and mask the old model's
    g = GOLDEN_IO.units_masked_central
    p0 = joinpath(fixtures, "units", "generic_camera_masked.json")
    e = try load(BoundedCamera, p0); nothing catch e; e end
    @test e isa ArgumentError && occursin("no complete 4 × 4 window", e.msg) && occursin("supply K", e.msg)
    tmp = mktempdir()
    p = joinpath(tmp, "generic_camera_masked.json")
    t = JSON.parse(read(p0, String)); t["nominal_intrinsics"] = Any[640.0, 640.0, 32.0, 24.0, 0.0]
    write(p, JSON.json(t))
    bc = load(BoundedCamera, p)
    cam = model(bc)
    @test cam.intrinsics.K == model(only(values(load(Dict{Symbol, BoundedCamera}, joinpath(fixtures, "units", "spline_camera_masked.json"))))).intrinsics.K   # the stored fixture's K
    # the refusal names the reader's keyword; given K, the file loads as it is (never edited), the same rays as the
    # stored-K form; a record that holds K refuses the keyword
    @test occursin("load(BoundedCamera, path; intrinsics = K)", e.msg)
    Kg = cam.intrinsics.K
    for given in (load(BoundedCamera, p0; intrinsics = Kg), only(values(load(Dict{Symbol, BoundedCamera}, p0; intrinsics = Kg))))
        @test model(given).intrinsics.K == Kg && isequal(model(given).lens.backward.controls, cam.lens.backward.controls)
        @test all(u -> isequal(backproject(model(given), u), backproject(cam, u)), [SVector(x, y) for x in 0.5:3.0:63.5 for y in 0.5:3.0:47.5])
    end
    for q in (p, joinpath(fixtures, "units", "spline_camera_masked.json"))
        e2 = try load(BoundedCamera, q; intrinsics = Kg); nothing catch e2; e2 end
        @test e2 isa ArgumentError && occursin("intrinsics", e2.msg)
    end
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
    # a store of its one camera, named after the file
    @test only(keys(load(Dict{Symbol, BoundedCamera}, p))) === :generic_camera_masked
    # vgio_migrate writes a NEW file beside it, <stem>.camera.json, the vgio/cameras/v5 record of the same camera;
    # the input stays byte for byte, and an existing output is never overwritten
    before = read(p)
    r = VisualGeometryToolkitIO.vgio_migrate(p)
    out = joinpath(tmp, "generic_camera_masked.camera.json")
    @test r.out == out && r.from == "vgio/generic_camera/v3" && r.to == "vgio/cameras/v5"
    @test read(p) == before
    @test VisualGeometryToolkitIO._peek_schema(out) == "vgio/cameras/v5"
    back = load(Dict{Symbol, BoundedCamera}, out)
    @test only(keys(back)) === :generic_camera_masked
    @test VisualGeometryToolkitIO.same_value(back, load(Dict{Symbol, BoundedCamera}, p))
    e = try VisualGeometryToolkitIO.vgio_migrate(p); nothing catch e; e end
    @test e isa ArgumentError && occursin(out, e.msg) && read(p) == before
    e = try VisualGeometryToolkitIO.vgio_migrate(p; out = p); nothing catch e; e end
    @test e isa ArgumentError && occursin("never rewritten", e.msg) && read(p) == before
    # without a stored K the camera is refused, so nothing is written
    e = try VisualGeometryToolkitIO.vgio_migrate(p0; out = joinpath(tmp, "nok.camera.json")); nothing catch e; e end
    @test e isa ArgumentError && occursin("supply K", e.msg) && !ispath(joinpath(tmp, "nok.camera.json"))
    # a non-central old file: the same rule, its vgio/noncentral_spline/v1 record at <stem>.noncentral.json
    mktempdir() do dir
        for f in ("old/generic_camera_v2.json", "units/generic_camera.json")
            q = joinpath(dir, basename(f)); cp(joinpath(fixtures, f), q)
            raw = read(q)
            r = VisualGeometryToolkitIO.vgio_migrate(q)
            o = joinpath(dir, first(splitext(basename(f))) * ".noncentral.json")
            @test r.out == o && r.to == "vgio/noncentral_spline/v1" && VisualGeometryToolkitIO._peek_schema(o) == "vgio/noncentral_spline/v1"
            @test read(q) == raw
            @test VisualGeometryToolkitIO.same_value(load(BoundedCamera, o), load(BoundedCamera, q))
            e = try VisualGeometryToolkitIO.vgio_migrate(q); nothing catch e; e end
            @test e isa ArgumentError && occursin(o, e.msg)
            e = try VisualGeometryToolkitIO.vgio_migrate(q; out = q); nothing catch e; e end
            @test e isa ArgumentError && occursin("never rewritten", e.msg) && read(q) == raw
        end
    end
    # a tree: the same, file by file; the inputs untouched
    mktempdir() do dir
        cp(p, joinpath(dir, "grid.json"))
        nc = joinpath(fixtures, "old", "generic_camera_v2.json")
        cp(nc, joinpath(dir, "nc.json"))
        io = IOBuffer()
        VisualGeometryToolkitIO.vgio_migrate_tree(dir; io)
        log = String(take!(io))
        @test occursin("grid.camera.json", log) && occursin("nc.noncentral.json", log)
        @test sort(readdir(dir)) == ["grid.camera.json", "grid.json", "nc.json", "nc.noncentral.json"]
        @test read(joinpath(dir, "grid.json")) == before && read(joinpath(dir, "nc.json")) == read(nc)
        @test only(keys(load(Dict{Symbol, BoundedCamera}, joinpath(dir, "grid.camera.json")))) === :grid
    end
    # in place, twice: an old grid file is never rewritten, and the second run finds its output current
    mktempdir() do dir
        cp(p, joinpath(dir, "grid.json"))
        cp(joinpath(fixtures, "old", "generic_camera_v2.json"), joinpath(dir, "nc.json"))
        for run in 1:2
            io = IOBuffer()
            VisualGeometryToolkitIO.vgio_migrate_tree(dir; in_place = true, io)
            log = String(take!(io))
            @test (run, sort(readdir(dir))) == (run, ["grid.camera.json", "grid.json", "nc.json", "nc.noncentral.json"])
            run == 2 && @test occursin("already migrated", log) && !occursin("FAILED", log)
        end
        @test read(joinpath(dir, "grid.json")) == before
        @test only(keys(load(Dict{Symbol, BoundedCamera}, joinpath(dir, "grid.camera.json")))) === :grid
    end
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
        # the K is a gauge: the camera of a stored pinhole and the camera of the lattice's gauge are one camera, on a
        # lattice holding every node (the masked fixture images nothing): units/generic_camera.json's directions as a
        # central v3 record
        full = JSON.parse(read(joinpath(@__DIR__, "fixtures", "units", "generic_camera.json"), String))
        full["type"] = "CentralGenericModel"
        delete!(full, "point_grid"); delete!(full, "length_unit")
        write_full(f) = (t = deepcopy(full); f(t); q = joinpath(dir, "full.json"); write(q, JSON.json(t)); q)
        a = model(load(BoundedCamera, write_full(t -> nothing)))
        b = model(load(BoundedCamera, write_full(t -> (t["nominal_intrinsics"] = Any[61.7, 58.9, 30.25, 22.5, 0.0]))))
        @test a.intrinsics.K != b.intrinsics.K && b.intrinsics.K[1, 1] == 61.7
        pixels = [SVector(x, y) for x in 0.5:3.0:63.5 for y in 0.5:3.0:47.5]
        same_ray(r, q) = all(isnan, q) ? all(isnan, r) : maximum(abs.(r - q)) <= 1.0e-12
        @test all(u -> same_ray(backproject(a, u), backproject(b, u)), pixels)
        @test count(u -> all(isfinite, backproject(a, u)), pixels) > length(pixels) ÷ 2
        # a well-formed central v2 record (every node, no `present`) reads as its v3 rewrite does
        c = model(load(BoundedCamera, write_full(t -> (t["schema"] = "vgio/generic_camera/v2"))))
        @test c.intrinsics.K == a.intrinsics.K && c.lens.present === nothing && c.lens.backward.controls == a.lens.backward.controls
        @test all(u -> isequal(backproject(c, u), backproject(a, u)), pixels)
        # a skewed nominal pinhole has no SplineLens camera
        r = write_with(t -> (t["nominal_intrinsics"] = Any[31.7, 31.9, 32.25, 23.875, 0.5]))
        e = try load(BoundedCamera, r); nothing catch e; e end
        @test e isa ArgumentError && occursin("skew", e.msg)
        # a "v2" central record holding `present` was not written by v2 (every v2 record held every node): refused
        v2 = write_with(t -> (t["schema"] = "vgio/generic_camera/v2"))
        e = try load(BoundedCamera, v2); nothing catch e; e end
        @test e isa ArgumentError && occursin("present", e.msg)
    end
end

@testitem "the spline lens record: a masked lens with a forward spline round-trips bit for bit" setup = [VGIoFixtures, SplineRecordFixtures] begin
    using VisualGeometryToolkitCore: model, lens_coverage, Ellipse, Point2, backproject, ChartSpline, CameraModel
    using StaticArrays: SVector, SMatrix
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
        # re-gauged to a moved principal point and a new scale, the forward held (its chart is about ẑ): stored and
        # read back, every projection the same as the camera's own, and its Float32 form keeps a forward
        K = cam.intrinsics.K
        K′ = VisualGeometryToolkitCore.CameraCalibrationMat(SMatrix{3, 3, Float64}(0.9 * K[1, 1], 0.0, 0.0, 0.0, 1.1 * K[2, 2], 0.0, K[1, 3] + 3.25, K[2, 3] - 1.5, 1.0))
        moved = CameraModel(cam, K′)
        r = joinpath(dir, "moved.json"); save(r, Dict(:grid => BoundedCamera(moved, Size2(64, 48))))
        mb = model(load(BoundedCamera, r))
        rays = [backproject(cam, SVector(x, y)) for x in 0.5:5.0:63.5 for y in 0.5:5.0:47.5]
        @test count(X -> all(isfinite, VisualGeometryToolkitCore.project(cam, X)), rays) > 10
        @test all(X -> rays_match(VisualGeometryToolkitCore.project(mb, X), VisualGeometryToolkitCore.project(cam, X)), rays)
        @test all(X -> isequal(VisualGeometryToolkitCore.project(mb, X), VisualGeometryToolkitCore.project(moved, X)), rays)
        @test VisualGeometryToolkitCore.convert_eltype(Float32, mb).lens.forward isa ChartSpline
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
        "or approximate it by a parametric lens first, `VisualGeometryToolkitEstimation.approximate_camera(KannalaBrandt{4}, camera, region)` " *
        "(any lens family; the region of its image to fit over)"   # the route Estimation's approximate_camera tests follow
    refused(f) = (e = try f(); nothing catch e; e end; e isa ArgumentError && e.msg == msg)
    @test refused(() -> VgtkIO.opencv_parameters(cam))
    @test refused(() -> VgtkIO.colmap_cameras(Dict(:c => bc)))
    pose = EuclideanMap(one(RotMatrix{3, Float64}), SVector(0.0, 0.0, 1.0) .* mm)
    mktempdir() do dir
        @test refused(() -> VgtkIO.save_opencv_calibration(joinpath(dir, "cal.yml"), cam, [pose]; rms = 0.1, to_unit = mm, images = ["a.png"]))
    end
end

@testitem "noncentral_spline/v1: the non-central record round-trips; the central type is refused" setup = [VGIoFixtures, SplineRecordFixtures] begin
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
        @test j.schema == "vgio/noncentral_spline/v1" && j.type == "noncentral_spline" && j.length_unit == "mm"   # a token of the format
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
                "schema" => t -> (t["schema"] = "vgio/noncentral_spline/v2"),
            )
            t = JSON.parse(text); f!(t)
            bad = joinpath(dir, "bad.json"); write(bad, JSON.json(t))
            e = try load(BoundedCamera, bad); nothing catch e; e end
            @test e isa ArgumentError && occursin(why, e.msg)
        end
        # the old kind's unpublished v4 has no step: refused
        t = JSON.parse(text); t["schema"] = "vgio/generic_camera/v4"
        bad = joinpath(dir, "v4.json"); write(bad, JSON.json(t))
        @test_throws ArgumentError load(BoundedCamera, bad)
        # a central v3 record has no non-central form: its step names the read that takes it
        old = JSON.parse(read(joinpath(@__DIR__, "fixtures", "units", "generic_camera_masked.json"), String))
        e = try VgtkIO.migrate_record(old); nothing catch e; e end
        @test e isa ArgumentError && occursin("load(BoundedCamera, path)", e.msg)
        # a non-central v3 record with a domain: the noncentral_spline/v1 record's lens coverage
        v3 = JSON.parse(read(joinpath(@__DIR__, "fixtures", "units", "generic_camera.json"), String))
        v3["domain"] = Dict("ellipse" => Any[32.0, 24.0, 30.0, 25.0, 0.0])
        t = VgtkIO.migrate_record(v3)
        @test t["schema"] == "vgio/noncentral_spline/v1" && t["lens_coverage"] == Dict("ellipse" => Any[32.0, 24.0, 30.0, 25.0, 0.0])
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
        @test j.schema == "vgio/rig/v9" && haskey(j.sensors[1].camera, :lens) && j.sensors[2].camera.type == "noncentral_spline"
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

@testitem "a spline lens camera's K is a skew-0 gauge in a camera store and a rig sensor: Core's one check" setup = [VGIoFixtures, SplineRecordFixtures] begin
    using VisualGeometryToolkitCore: Rig, PosedCamera, ViewFrame, relabel
    using Rotations: RotXYZ
    cam = spline_camera()
    X = EuclideanMap(RotMatrix{3, Float64}(RotXYZ(0.0, 0.0, 0.0)), [0.0mm, 0.0mm, 0.0mm])
    skewed(f) = t -> (f(t)["K"]["skew"] = 0.25)
    mktempdir() do dir
        refusal(p) = (e = try load(p[1], p[2]); nothing catch e; e end; e isa ArgumentError && occursin("skew 0", e.msg))
        p = joinpath(dir, "cams.json"); save(p, Dict(:grid => BoundedCamera(cam, Size2(64, 48))))
        r = joinpath(dir, "rig.json"); save(r, Rig{ViewFrame{:g}}((BoundedCamera(PosedCamera(cam, relabel(ViewFrame{:g}, ViewFrame{:a}, X)), Size2(64, 48)),)))
        for (path, T, f) in ((p, Dict{Symbol, BoundedCamera}, t -> t["cameras"][1]), (r, Rig, t -> t["sensors"][1]["camera"]))
            @test load(T, path) !== nothing                             # as written: a skew-0 gauge
            t = JSON.parse(read(path, String)); skewed(f)(t)
            bad = joinpath(dir, "bad.json"); write(bad, JSON.json(t))
            @test refusal((T, bad))
        end
    end
end

# The public camera bodies a package's own record embeds: a BoundedCamera's body (a CameraModel's of any lens, or a
# non-central grid camera's) with its sensor's pitch and its lens coverage, and a non-central model's bare body, each
# read back exactly through a parsed JSON round trip, in another pixel frame and length unit too.
@testitem "camera_json / camera_from_json: a bounded camera's body carries its lens coverage; the non-central body" setup = [VGIoFixtures, SplineRecordFixtures] begin
    import VisualGeometryToolkitIO as VgtkIO
    using VisualGeometryToolkitCore: model, lens_coverage, Circle, Ellipse, Point2, Rect2, Sensor, sensor, μm, with_present, extent
    c = corner_cut(spline_camera())
    d = c.lens.backward
    pts = [SVector(1e-3 * gx, -2e-3 * gy, 0.0) for gy in 1:7, gx in 1:8]
    nc = with_present(NoncentralSplineModel(c.intrinsics, d, ChartSpline(d.area, pts)), c.lens.present)
    K = CameraCalibrationMat(SMatrix{3, 3, Float64}(500.0, 0, 0, 0, 500.0, 0, 32.0, 24.0, 1))
    ext = Size2(64, 48)
    circle = Circle(Point2(32.0, 24.0), 30.0)
    ellipse = Ellipse(Point2(31.5, 24.5), 30.0, 26.0, 0.1)
    parsed(x) = JSON.parse(JSON.json(x))
    same(a, b) = typeof(a) == typeof(b) && VgtkIO.same_value(model(a), model(b)) && a.bound == b.bound && lens_coverage(a) == lens_coverage(b)
    cams = (BoundedCamera(CameraModel(K), ext; lens_coverage = circle),
            BoundedCamera(c, Sensor(ext, 3.45μm); lens_coverage = ellipse),
            BoundedCamera(nc, ext; lens_coverage = circle),
            BoundedCamera(nc, Sensor(ext, 3.45μm)))                     # the whole sensor: no coverage written
    for bc in cams, frame in (VgtkIO.CANONICAL_PIXEL_FRAME, PixelFrame{:opencv}), u in (mm, μm)
        body = VgtkIO.camera_json(bc, frame; length_unit = u)
        @test haskey(body, :lens_coverage) == (lens_coverage(bc) != Rect2{Float64}(0, 0, 64, 48))
        @test haskey(body, :pitch) == (sensor(bc) !== nothing)
        back = VgtkIO.camera_from_json(BoundedCamera, parsed(body), frame; length_unit = u)
        @test back isa BoundedCamera && same(back, bc)
    end
    # the non-central model's bare body, and the (model, extent) read of either kind
    for frame in (VgtkIO.CANONICAL_PIXEL_FRAME, PixelFrame{:opencv})
        body = VgtkIO.camera_json(nc, ext, frame; length_unit = μm)
        m, e = VgtkIO.camera_from_json(parsed(body), frame; length_unit = μm)
        @test m isa NoncentralSplineModel && e == ext && VgtkIO.same_value(m, nc)
    end
    # a body's own check: a malformed lens coverage is refused by name
    bad = parsed(merge(VgtkIO.camera_json(cams[1]), (lens_coverage = (circle = [32.0, 24.0, -1.0],),)))
    @test_throws ArgumentError VgtkIO.camera_from_json(BoundedCamera, bad, VgtkIO.CANONICAL_PIXEL_FRAME)
end

@testitem "lattice_json / lattice_from_json: the lattice record alone (an aperture spline of radii), held nodes, lengths" setup = [VGIoFixtures, SplineRecordFixtures] begin
    # The ONE lattice codec the spline lens record writes its splines with, public for a package's own spline
    # (RadiometricCamera's aperture spline: scalar radii, lengths, over the lens's lattice).
    import VisualGeometryToolkitIO as VgtkIO
    using VisualGeometryToolkitCore: ChartSpline, Rect2
    area = Rect2{Float64}(-0.7, -0.55, 1.4, 1.1)
    radii = [SVector(VgtkIO.metres(round(0.5 + 0.01 * gx + 0.001 * gy; digits = 3), mm)) for gy in 1:6, gx in 1:7]   # metres, mm-born
    r = ChartSpline(area, radii)
    P = trues(6, 7); P[1, 1] = P[1, 2] = P[6, 7] = false
    for (present, unit) in ((nothing, nothing), (P, mm), (P, m), (nothing, mm))
        b = VgtkIO.lattice_json(r; name = :radius, present, length_unit = unit)
        @test haskey(b, :present) == (present !== nothing) && b.grid_size == (width = 7, height = 6)
        @test length(b.controls.radius) == (present === nothing ? 42 : 39)
        unit === mm && @test b.controls.radius[1] == [round(0.5 + 0.01 * (present === nothing ? 1 : 3) + 0.001; digits = 3)]   # the file's mm
        for rec in (JSON.parse(JSON.json(b)), b)                               # a parsed record, and the NamedTuple itself
            back = VgtkIO.lattice_from_json(rec, Val(1); name = :radius, length_unit = unit)
            @test back.present == present && back.spline.area === area
            @test all(I -> present !== nothing && !present[I] ? all(isnan, back.spline.controls[I]) : back.spline.controls[I] == radii[I],
                      CartesianIndices(radii))                                 # bit for bit; NaN in the absent nodes
        end
    end
    # the spline lens record's splines ARE lattice records
    cam = corner_cut(spline_camera())
    sp = VgtkIO.camera_json(cam, Size2(64, 48)).lens.spline
    asdict(x) = JSON.parse(JSON.json(x); dicttype = Dict{String, Any})
    spd = asdict(sp); delete!(spd, "forward")
    @test asdict(VgtkIO.lattice_json(cam.lens.backward; name = :direction, cam.lens.present)) == spd
    # refused by name: a missing key, an unknown key, a lattice under 4 × 4, a control count, a control's length,
    # runs outside the lattice, a mask of another size
    good = JSON.parse(JSON.json(VgtkIO.lattice_json(r; name = :radius, present = P)))
    for (why, f!) in (
            "missing" => t -> delete!(t, "area"), "unknown key" => t -> (t["forward"] = Dict{String, Any}()),
            "4 × 4" => t -> (t["grid_size"]["width"] = 3), "entries" => t -> pop!(t["controls"]["radius"]),
            "expected 1 entries" => t -> (t["controls"]["radius"][1] = Any[1.0, 2.0]), "row" => t -> push!(t["present"], Any[9, Any[Any[1, 2]]]),
        )
        t = deepcopy(good); f!(t)
        e = try VgtkIO.lattice_from_json(t, Val(1); name = :radius); nothing catch err; err end
        @test e isa ArgumentError && occursin("a lattice record", e.msg) && occursin(why, e.msg)
    end
    @test_throws ArgumentError VgtkIO.lattice_json(r; name = :radius, present = trues(7, 6))
end
