# =============================================================================
# THE region body (core_json.jl): Core's 2-D regions, tagged
# =============================================================================
# {"rect": [x, y, w, h]} | {"ellipse": [cx, cy, a, b, θ]} | {"circle": [cx, cy, r]}
#   | {"intersection": [<region>, <region>]} | {"whole_plane": {}}
# pixels in the file's units.pixel; a camera's lens_coverage is the body restricted to its leaves.

@testitem "region body: every leaf, the whole plane and nested intersections round-trip exactly, by type" begin
    using Test
    using JSON
    import VisualGeometryToolkitIO as VgtkIO
    using VisualGeometryToolkitCore: Rect2, Ellipse, Circle, Point2, RegionIntersection, WholePlane, PixelFrame

    rect = Rect2{Float64}(10.25, 20.5, 1900.0, 1180.75)
    ell = Ellipse(Point2(960.5, 600.25), 700.0, 650.5, 0.375)
    circ = Circle(Point2(961.0, 599.5), 640.25)
    w = WholePlane()
    regions = Any[
        rect, ell, circ, w,
        RegionIntersection(rect, ell),
        RegionIntersection(circ, rect),                                     # a circle stays a circle (∩ would make it an ellipse)
        RegionIntersection(rect, Rect2{Float64}(0.0, 0.0, 100.0, 50.0)),     # two rectangles stay two (∩ would fold them)
        RegionIntersection(ell, w),                                         # the whole plane kept (∩ would drop it)
        RegionIntersection(RegionIntersection(rect, ell), circ),            # depth 2, left
        RegionIntersection(rect, RegionIntersection(ell, RegionIntersection(circ, w))),   # depth 3, right, every kind
        RegionIntersection(RegionIntersection(RegionIntersection(rect, ell), RegionIntersection(circ, rect)),
                           RegionIntersection(w, RegionIntersection(ell, RegionIntersection(rect, circ)))),   # depth 4, mixed
    ]
    for frame in (PixelFrame{:colmap}, PixelFrame{:opencv}, PixelFrame{:julia})
        s = VgtkIO.VgioStyle(; pixel_frame = frame)
        for r in regions
            text = JSON.json(JSON.lower(s, r, (; region = true)))
            back = JSON.lift(s, typeof(r), JSON.parse(text))
            @test typeof(back) === typeof(r)
            @test back === r
            # an abstract target: a Union of region types, and the leaf read as itself
            @test JSON.lift(s, Union{Rect2, Ellipse, Circle, RegionIntersection, WholePlane}, JSON.parse(text)) === r
        end
    end

    s = VgtkIO.VgioStyle()
    # the shapes, as stored (colmap: Core's canonical frame, the numbers as they are)
    @test JSON.json(JSON.lower(s, RegionIntersection(rect, ell))) ==
          """{"intersection":[{"rect":[10.25,20.5,1900.0,1180.75]},{"ellipse":[960.5,600.25,700.0,650.5,0.375]}]}"""
    @test JSON.json(JSON.lower(s, w)) == """{"whole_plane":{}}"""
    @test JSON.json(JSON.lower(s, circ, (; region = true))) == """{"circle":[961.0,599.5,640.25]}"""
    # a leaf's two-argument lower stays its positional record (and so does a tag set without `region`)
    @test JSON.lower(s, rect) == [10.25, 20.5, 1900.0, 1180.75] == JSON.lower(s, rect, (;))
    # embedded in an owner's record (RC's initialization region), through JSON.json on the style
    rec = JSON.parse(JSON.json((initial_region = RegionIntersection(rect, ell), coverage = w); style = s))
    @test JSON.lift(s, RegionIntersection, rec["initial_region"]) === RegionIntersection(rect, ell)
    @test JSON.lift(s, WholePlane, rec["coverage"]) === w
    # pixels in the file's frame: an OpenCV file holds the colmap numbers less half a pixel
    so = VgtkIO.VgioStyle(; pixel_frame = PixelFrame{:opencv})
    @test JSON.lower(so, RegionIntersection(rect, circ)).intersection[1].rect == [9.75, 20.0, 1900.0, 1180.75]
    @test JSON.lower(so, RegionIntersection(rect, circ)).intersection[2].circle == [960.5, 599.0, 640.25]
    # a region of another type than asked is refused, naming what it holds
    @test_throws "the record holds" JSON.lift(s, Rect2, JSON.parse("""{"ellipse": [1, 2, 3, 2, 0]}"""))
    @test_throws ArgumentError JSON.lift(s, RegionIntersection, JSON.parse("""{"whole_plane": {}}"""))
end

@testitem "region body: refusals (an intersection of other than two, an unknown key, a bad leaf)" begin
    using Test
    using JSON
    import VisualGeometryToolkitIO as VgtkIO
    using VisualGeometryToolkitCore: RegionIntersection, WholePlane, Rect2

    s = VgtkIO.VgioStyle()
    check(text) = VgtkIO._validate(VgtkIO._REGION, JSON.parse(text), "f.json", "region")
    bad = [
        """{"intersection": []}""" => "an intersection is two regions",
        """{"intersection": [{"rect": [0, 0, 1, 1]}]}""" => "got 1",
        """{"intersection": [{"rect": [0, 0, 1, 1]}, {"whole_plane": {}}, {"circle": [0, 0, 1]}]}""" => "got 3",
        """{"intersection": {"rect": [0, 0, 1, 1]}}""" => "an intersection is two regions",
        """{"polygon": [[0, 0], [1, 0], [0, 1]]}""" => "region.polygon: not a region",
        """{}""" => "a region is an object of ONE key",
        """{"rect": [0, 0, 1, 1], "circle": [0, 0, 1]}""" => "a region is an object of ONE key",
        """[0, 0, 1, 1]""" => "a region is an object of ONE key",
        """{"rect": [0, 0, 0, 1]}""" => "the rect's width is positive",
        """{"rect": [0, 0, -1, -1]}""" => "width and height are positive",
        """{"rect": [0, 0, 1]}""" => "expected 4 entries",
        """{"circle": [0, 0, -2]}""" => "the circle's r is positive",
        """{"circle": [0, "0", 2]}""" => "region.circle[2]: expected a finite number",
        """{"ellipse": [0, 0, 2, 3, 0]}""" => "major semi-axis first",
        """{"ellipse": [0, 0, 3, 0, 0]}""" => "the ellipse's b is positive",
        """{"whole_plane": {"x": 1}}""" => "region.whole_plane.x: unknown key",
        """{"whole_plane": []}""" => "expected an object",
        # deep: the place named through the nesting
        """{"intersection": [{"rect": [0, 0, 1, 1]}, {"intersection": [{"whole_plane": {}}, {"circle": [0, 0, 0]}]}]}""" =>
            "region.intersection[2].intersection[2].circle: the circle's r is positive",
        """{"intersection": [{"intersection": [{"rect": [0, 0, 1, 1]}, {"bogus": 1}]}, {"whole_plane": {}}]}""" =>
            "region.intersection[1].intersection[2].bogus: not a region",
    ]
    for (text, msg) in bad
        @test_throws msg check(text)
        # the public lift checks a parsed record the same way (a record an owner reads)
        text == "[0, 0, 1, 1]" || @test_throws ArgumentError JSON.lift(s, Union{Rect2, RegionIntersection, WholePlane}, JSON.parse(text))
    end
    # a leaf-only place (a camera's lens coverage) refuses an intersection and the whole plane by name
    leaf(text) = VgtkIO._validate(VgtkIO._REGION_LEAF, JSON.parse(text), "f.json", "lens_coverage")
    @test_throws "lens_coverage.intersection: not a region here" leaf("""{"intersection": [{"rect": [0, 0, 1, 1]}, {"circle": [0, 0, 1]}]}""")
    @test_throws "lens_coverage.whole_plane: not a region here" leaf("""{"whole_plane": {}}""")
    @test leaf("""{"circle": [0, 0, 1]}""") === nothing
end

@testitem "region body: a camera's lens coverage is its leaves; an intersection there is refused" begin
    using Test
    using JSON
    import VisualGeometryToolkitCore as VgtkCore
    import VisualGeometryToolkitIO as VgtkIO
    using VisualGeometryToolkitCore: BoundedCamera, CameraModel, KannalaBrandt, Size2, Point2, Circle, lens_coverage, px

    m = CameraModel(1754.7px, VgtkCore.SA[2133.6px, 1422.5px]; lens = KannalaBrandt(0.2468, 0.1505, -0.0214, 0.1308))
    bc = BoundedCamera(m, Size2(4256, 2848); lens_coverage = Circle(Point2(2131.1, 1407.9), 1680.1))
    dir = mktempdir()
    p = joinpath(dir, "cam.json")
    VgtkIO.save(p, Dict(:cam => bc))
    @test only(JSON.parsefile(p)["cameras"])["lens_coverage"] == Dict("circle" => [2131.1, 1407.9, 1680.1])
    @test lens_coverage(VgtkIO.load(BoundedCamera, p)) == lens_coverage(bc)
    t = read(p, String)
    q = joinpath(dir, "cam_intersection.json")
    write(q, replace(t, r"\"lens_coverage\": \{[^}]*\}"s =>
        "\"lens_coverage\": {\"intersection\": [{\"rect\": [0, 0, 4256, 2848]}, {\"circle\": [2131.1, 1407.9, 1680.1]}]}"))
    @test occursin("intersection", read(q, String))
    @test_throws "lens_coverage.intersection: not a region here" VgtkIO.validate(q)
    @test_throws ArgumentError VgtkIO.load(BoundedCamera, q)
    write(q, replace(t, r"\"lens_coverage\": \{[^}]*\}"s => "\"lens_coverage\": {\"circle\": [2131.1, 1407.9, 0.0]}"))
    @test_throws "the circle's r is positive" VgtkIO.validate(q)
end

@testitem "region body: JSON Schema validation of region documents (Python jsonschema)" setup = [VGIoSchemaCases] begin
    using Test
    using JSON
    import VisualGeometryToolkitIO as VgtkIO
    import VisualGeometryToolkitCore as VgtkCore
    using VisualGeometryToolkitCore: Rect2, Ellipse, Circle, Point2, RegionIntersection, WholePlane, BoundedCamera, CameraModel,
        KannalaBrandt, Size2, px

    python = pixi_python("jsonschema")
    if python === nothing
        @test_skip "jsonschema is not installed in the pixi py environment (pixi install -e py)"
    else
        mktempdir() do dir
            # the published schemas, and a schema of the region body alone (its $defs as any kind holds them)
            root = joinpath(dir, "schemas")
            cp(schema_dir(), root)
            defs = VgtkIO.JSON.Object{String, Any}()
            ref = VgtkIO._json_schema(VgtkIO._REGION, defs)
            id = VgtkIO._JSON_SCHEMA_ID_BASE * "test/region.json"
            mkpath(joinpath(root, "test"))
            write(joinpath(root, "test", "region.json"), JSON.json(VgtkIO._ordered_json(VgtkIO.JSON.Object{String, Any}(
                "\$schema" => VgtkIO._JSON_SCHEMA_DIALECT, "\$id" => id, "\$ref" => ref["\$ref"], "\$defs" => defs))))
            @test haskey(defs, "region") && haskey(defs, "region_leaf")

            s = VgtkIO.VgioStyle()
            rect, ell, circ = Rect2{Float64}(0.0, 0.0, 640.0, 480.0), Ellipse(Point2(320.0, 240.0), 300.0, 250.0, 0.25), Circle(Point2(320.0, 240.0), 260.0)
            valid = Any[rect, ell, circ, WholePlane(), RegionIntersection(rect, ell),
                        RegionIntersection(rect, RegionIntersection(ell, RegionIntersection(circ, WholePlane())))]
            invalid = [
                """{"intersection": []}""", """{"intersection": [{"rect": [0, 0, 1, 1]}]}""",
                """{"intersection": [{"rect": [0, 0, 1, 1]}, {"whole_plane": {}}, {"circle": [0, 0, 1]}]}""",
                """{"polygon": [0, 0, 1]}""", """{}""", """{"rect": [0, 0, 1, 1], "circle": [0, 0, 1]}""",
                """{"rect": [0, 0, 0, 1]}""", """{"rect": [0, 0, 1]}""", """{"circle": [0, 0, -2]}""", """{"circle": [0, "0", 2]}""",
                """{"ellipse": [0, 0, 3, 0, 0]}""", """{"whole_plane": {"x": 1}}""",
                """{"intersection": [{"rect": [0, 0, 1, 1]}, {"intersection": [{"whole_plane": {}}, {"circle": [0, 0, 0]}]}]}""",
                """{"intersection": [{"intersection": [{"rect": [0, 0, 1, 1]}, {"bogus": 1}]}, {"whole_plane": {}}]}""",
            ]
            refused(doc, spec = VgtkIO._REGION) = try
                VgtkIO._validate(spec, doc, "case", ""); false
            catch e
                e isa ArgumentError || rethrow(); true
            end
            cases = Dict{String, Any}[]
            expect = Dict{String, Bool}()
            add!(name, sid, doc) = (push!(cases, Dict("name" => name, "schema_id" => sid, "doc" => doc)); expect[name] = !refused(doc))
            for (i, r) in enumerate(valid)
                add!("valid $(i)", id, JSON.parse(JSON.json(JSON.lower(s, r, (; region = true)))))
            end
            for (i, t) in enumerate(invalid)
                add!("invalid $(i)", id, JSON.parse(t))
            end
            @test count(values(expect)) == length(valid)      # the validator's verdicts: the valid valid, the invalid refused
            # a camera store whose lens coverage is a leaf, and one whose coverage is an intersection (refused there)
            m = CameraModel(1754.7px, VgtkCore.SA[2133.6px, 1422.5px]; lens = KannalaBrandt(0.2468, 0.1505, -0.0214, 0.1308))
            p = joinpath(dir, "cam.json")
            VgtkIO.save(p, Dict(:cam => BoundedCamera(m, Size2(4256, 2848); lens_coverage = ell)))
            cam = plain_doc(JSON.parsefile(p))
            cam_id = VgtkIO.json_schema("cameras")["\$id"]
            push!(cases, Dict("name" => "camera, an ellipse coverage", "schema_id" => cam_id, "doc" => cam)); expect["camera, an ellipse coverage"] = true
            bad = deepcopy(cam)
            bad["cameras"][1]["lens_coverage"] = Dict("intersection" => Any[Dict("rect" => [0, 0, 10, 10]), Dict("circle" => [5, 5, 4])])
            push!(cases, Dict("name" => "camera, an intersection coverage", "schema_id" => cam_id, "doc" => bad)); expect["camera, an intersection coverage"] = false
            q = joinpath(dir, "bad_cam.json"); write(q, JSON.json(bad))
            @test_throws ArgumentError VgtkIO.validate(q)

            cases_file = joinpath(dir, "cases.json")
            write(cases_file, JSON.json(cases))
            out = IOBuffer()
            ok = success(pipeline(`$python $(joinpath(@__DIR__, "python", "check_json_schemas.py")) $root $cases_file`; stdout = out))
            @test ok
            if ok
                results = JSON.parse(String(take!(out)))
                @test length(results) == length(cases)
                @test ["$(r["name"]): the schema says valid = $(r["valid"]) ($(r["error"]))" for r in results if r["valid"] != expect[r["name"]]] == String[]
            end
            # the one check JSON Schema cannot state (the ellipse's major semi-axis first) is the validator's alone
            @test refused(JSON.parse("""{"ellipse": [0, 0, 2, 3, 0]}"""))
            @test occursin("a ≥ b", JSON.json(defs["region_leaf"]))
        end
    end
end
