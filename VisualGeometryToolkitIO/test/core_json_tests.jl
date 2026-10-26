# =============================================================================
# The quantity JSON format: numbers in the declared unit (one length unit per
# role), the {value, unit} object refused, and the codecs of VisualGeometryToolkitCore's
# sizes, feature frames and detections (moved from VisualGeometryToolkitFeatures)
# =============================================================================

@testitem "quantity JSON: write_quantity and read_quantity" setup = [VGIoFixtures] begin
    using VisualGeometryToolkitIO: UnitContract, MM_PX, write_quantity, read_quantity
    using VisualGeometryToolkitCore: dpi, inch, μm
    # a length in the world unit, a pixel count as a px number, a density in the density unit
    @test write_quantity(2.5mm, MM_PX) === 2.5
    @test write_quantity(2.5mm, UnitContract(world = m)) === 0.0025
    @test write_quantity(640px, MM_PX) === 640
    @test write_quantity(640.5px, MM_PX) === 640.5
    @test write_quantity(300dpi, MM_PX) === 300
    @test write_quantity(72.5dpi, MM_PX) === 72.5
    @test write_quantity(0.5mm / px, MM_PX) === 0.5          # a pitch is a length (px is dimensionless)
    @test_throws ArgumentError write_quantity(1.0 * parse_unit("s"), MM_PX)  # no declared unit for a time
    @test_throws ArgumentError write_quantity(1.0mm^2, MM_PX)
    @test read_quantity(2.5, :length, MM_PX) == 2.5mm
    @test read_quantity(0.0025, :length, UnitContract(world = m)) == 2.5mm
    @test read_quantity(640, :pixel, MM_PX) == 640px
    @test read_quantity(300, :density, MM_PX) == 300dpi
    px_per_mm = parse_unit("px/mm"; compound = true)
    @test read_quantity(12, :density, UnitContract(density = px_per_mm)) == 12 * px_per_mm
    @test write_quantity(300dpi, UnitContract(density = px_per_mm)) === ustrip(px_per_mm, 300dpi)
    # the retired {value, unit} object is refused, naming the migration
    for x in (Dict("value" => 2.5, "unit" => "mm"), JSON.parse("""{"value": 1.3, "unit": "cm"}"""), (value = 2.5, unit = "mm"))
        e = try read_quantity(x, :length, MM_PX); nothing catch e; e end
        @test e isa ArgumentError && occursin("vgio_migrate", e.msg) && occursin("declared unit", e.msg)
    end
    @test_throws ArgumentError read_quantity(Dict("value" => 640, "unit" => "px"), :pixel, MM_PX)
    @test_throws ArgumentError read_quantity(2.5, :angle, MM_PX)
    @test_throws ArgumentError read_quantity("2.5", :length, MM_PX)
    @test_throws ArgumentError read_quantity(true, :pixel, MM_PX)
    # a round trip through every kind and both world units
    for c in (MM_PX, UnitContract(world = m), UnitContract(world = inch)), (q, kind) in
            ((62.8mm, :length), (5.86μm, :length), (1.0inch, :length), (640px, :pixel), (480.25px, :pixel), (300dpi, :density))
        @test read_quantity(write_quantity(q, c), kind, c) == q
    end
end

@testitem "quantity JSON: UnitContract density and its declaration" setup = [VGIoFixtures] begin
    using VisualGeometryToolkitIO: UnitContract, MM_PX, json_value, from_json, save_units_sidecar, load_units_sidecar
    using VisualGeometryToolkitCore: dpi
    px_per_mm = parse_unit("px/mm"; compound = true)
    @test MM_PX.density == dpi && UnitContract(m).density == dpi
    @test UnitContract(world = m, density = px_per_mm).density == px_per_mm
    @test_throws ArgumentError UnitContract(density = mm)      # a density is px per length
    @test_throws ArgumentError UnitContract(density = px)
    @test UnitContract(world = m) == UnitContract(m) && UnitContract(world = mm, density = dpi) == MM_PX
    # records declare it inline, by registered unit names
    @test json_value(MM_PX) == (world = "mm", density = "dpi")
    @test JSON.json(json_value(UnitContract(world = m))) == """{"world":"m","density":"dpi"}"""
    @test_throws ArgumentError json_value(UnitContract(density = px_per_mm))   # no registered name
    @test from_json(UnitContract, JSON.parse("""{"world": "m", "density": "dpi"}""")) == UnitContract(world = m)
    @test from_json(UnitContract, Dict("world" => "mm")) == MM_PX       # a declaration without a density: dpi
    @test from_json(UnitContract, (world = "m",)) == UnitContract(world = m)
    @test_throws ArgumentError from_json(UnitContract, Dict("world" => "px"))
    @test_throws ArgumentError from_json(UnitContract, Dict("world" => "mm", "density" => "mm"))
    # the COLMAP sidecar: a default density writes nothing new; a declared one reads
    mktempdir() do dir
        save_units_sidecar(dir, MM_PX)
        @test read(joinpath(dir, "vgio_units.json"), String) ==
            "{\n    \"schema\": \"vgio/units/v2\",\n    \"world\": \"mm\"\n}"
        @test load_units_sidecar(dir) == MM_PX
        write(joinpath(dir, "vgio_units.json"), """{"schema": "vgio/units/v2", "world": "m", "density": "dpi"}""")
        @test load_units_sidecar(dir) == UnitContract(world = m)
        write(joinpath(dir, "vgio_units.json"), """{"schema": "vgio/units/v2", "world": "m", "density": "mm"}""")
        @test_throws ArgumentError load_units_sidecar(dir)
    end
end

@testitem "quantity JSON: Size2 and the feature frames round trip" setup = [VGIoFixtures] begin
    using VisualGeometryToolkitIO: UnitContract, MM_PX, json_value, from_json
    using VisualGeometryToolkitCore: Disc, OrientedDisc, EllipticFrame, AffineFrame, Point2, AbstractDetection
    rt(x, c; kind) = from_json(typeof(x).name.wrapper, JSON.parse(JSON.json(json_value(x, c; kind))), c; kind)
    for c in (MM_PX, UnitContract(world = m))
        for (x, kind) in (
                (Size2(width = 640px, height = 480px), :pixel), (Size2(width = 210.0mm, height = 297.0mm), :length),
                (Size2(width = 62.8mm, height = 1.25mm), :length), (Size2(width = 640, height = 480), :plain),
                (Size2(width = 640.5, height = 480.0), :plain),
                (Disc(10.5, 20.25, 3.0), :plain), (Disc(Point2(10.5px, 20.0px), 3.0px), :pixel),
                (Disc(Point2(12.7mm, 0.3mm), 1.25mm), :length),
                (OrientedDisc(1.0, 2.0, 3.0, 0.5), :plain),
                (EllipticFrame([10.5, 20.25], [2.0 0.3; 0.3 1.5]), :plain),
                (AffineFrame([1.0, 2.0], [1.2 0.1; -0.2 0.9]), :plain),
            )
            @test rt(x, c; kind) == x
        end
    end
    # the numbers are in the declared unit
    @test JSON.json(json_value(Size2(width = 640px, height = 480px), MM_PX; kind = :pixel)) == """{"width":640,"height":480}"""
    @test JSON.json(json_value(Size2(width = 210.0mm, height = 297.0mm), MM_PX; kind = :length)) == """{"width":210.0,"height":297.0}"""
    @test JSON.json(json_value(Size2(width = 210.0mm, height = 297.0mm), UnitContract(world = m); kind = :length)) ==
        """{"width":0.21,"height":0.297}"""
    @test JSON.json(json_value(Disc(Point2(1.0mm, 2.0mm), 3.0mm), MM_PX; kind = :length)) == """{"center":[1.0,2.0],"σ":3.0}"""
    @test JSON.json(json_value(EllipticFrame([10.5, 20.25], [2.0 0.3; 0.3 1.5]), MM_PX; kind = :plain)) ==
        """{"center":[10.5,20.25],"S":[2.0,0.3,0.3,1.5]}"""
    @test JSON.json(json_value(AffineFrame([1.0, 2.0], [1.2 0.1; -0.2 0.9]), MM_PX; kind = :plain)) ==
        """{"center":[1.0,2.0],"A":[1.2,-0.2,0.1,0.9]}"""
    @test JSON.json(json_value(OrientedDisc(1.0, 2.0, 3.0, 0.5), MM_PX; kind = :plain)) == """{"center":[1.0,2.0],"σ":3.0,"θ":0.5}"""
    # the declared kind is the value's: a size half in pixels and half in
    # millimetres, or pixels written as lengths, or quantities written plain, is refused
    @test_throws ArgumentError json_value(Size2(width = 640px, height = 3.0mm), MM_PX; kind = :pixel)
    @test_throws ArgumentError json_value(Size2(width = 640px, height = 480px), MM_PX; kind = :length)
    @test_throws ArgumentError json_value(Size2(width = 640px, height = 480px), MM_PX; kind = :plain)
    @test_throws ArgumentError json_value(Size2(width = 640, height = 480), MM_PX; kind = :pixel)
    @test_throws ArgumentError json_value(Disc(Point2(1.0mm, 2.0mm), 3.0mm), MM_PX; kind = :pixel)
    @test_throws ArgumentError json_value(OrientedDisc(1.0, 2.0, 3.0, 0.5), MM_PX; kind = :pixel)
    @test_throws ArgumentError from_json(OrientedDisc, Dict("center" => [1, 2], "σ" => 3, "θ" => 0), MM_PX; kind = :length)
    # `kind` is required: no default hands a length or a pixel count back as a bare number
    for (x, d) in (
            (Size2(width = 640px, height = 480px), Dict("width" => 640, "height" => 480)),
            (Disc(Point2(10.5px, 20.0px), 3.0px), Dict("center" => [10.5, 20], "σ" => 3)),
        )
        @test_throws UndefKeywordError json_value(x, MM_PX)
        @test_throws UndefKeywordError from_json(typeof(x).name.wrapper, d, MM_PX)
    end
    @test_throws UndefKeywordError from_json(AbstractDetection, Dict("frame" => Dict("center" => [1, 2], "σ" => 3)), MM_PX)
    @test_throws ArgumentError from_json(Size2, Dict("width" => 1, "height" => 2), MM_PX; kind = :angle)
    # a parsed object, a Dict or a NamedTuple
    @test from_json(Size2, Dict{String, Any}("width" => 640, "height" => 480), MM_PX; kind = :plain) == Size2(width = 640, height = 480)
    @test from_json(Size2, (width = 640, height = 480), MM_PX; kind = :pixel) == Size2(width = 640px, height = 480px)
end

@testitem "quantity JSON: detections round trip, every frame type, all metadata" setup = [VGIoFixtures] begin
    using VisualGeometryToolkitIO: MM_PX, json_value, from_json
    using VisualGeometryToolkitCore: Disc, OrientedDisc, EllipticFrame, AffineFrame, Point2, Attributed,
        AbstractDetection, Uncertain, CovMat
    using Statistics: cov
    rt(d; kind = :plain) = from_json(AbstractDetection, JSON.parse(JSON.json(json_value(d, MM_PX; kind))), MM_PX; kind)
    meta = (response = 0.25, edge_score = 0.0, laplacian_scale_score = 1.0, warp_idx = 7)
    frames = (
        Disc(10.5, 20.25, 3.0), OrientedDisc(10.5, 20.25, 3.0, 0.7),
        EllipticFrame([10.5, 20.25], [2.0 0.3; 0.3 1.5]), AffineFrame([10.5, 20.25], [1.2 0.1; -0.2 0.9]),
    )
    for f in frames
        d = Attributed(f, meta)
        r = rt(d)
        @test getfield(r, :value) == f
        @test getfield(r, :attributes) == meta
        @test typeof(getfield(r, :attributes)) == typeof(meta)   # 0.0 and 1.0 stay Float64
    end
    # documented fields first, the rest by name, whatever the input's key order
    e = Attributed(frames[1], (zeta = 1, response = 2.0, alpha = "a"))
    @test getfield(rt(e), :attributes) == (response = 2.0, alpha = "a", zeta = 1)
    # a pixel-quantity frame, by its declared kind
    p = Attributed(Disc(Point2(10.5px, 20.0px), 3.0px), (response = 1.0,))
    @test getfield(rt(p; kind = :pixel), :value) == getfield(p, :value)
    # an uncertain frame keeps its covariance; a NaN one is written null and read back NaN
    Σ = CovMat{3, Float64, 9}(Tuple([0.01 0.001 0.0; 0.001 0.02 0.0; 0.0 0.0 0.005]))
    u = Attributed(Uncertain(frames[1], Σ), meta)
    @test JSON.json(json_value(u, MM_PX; kind = :plain)) ==
        """{"frame":{"center":[10.5,20.25],"σ":3.0},"cov":[0.01,0.001,0.0,0.001,0.02,0.0,0.0,0.0,0.005],"response":0.25,"edge_score":0.0,"laplacian_scale_score":1.0,"warp_idx":7}"""
    ru = rt(u)
    @test ru isa AbstractDetection && cov(getfield(ru, :value)) == cov(getfield(u, :value))
    n = Attributed(Uncertain(frames[1], CovMat{3, Float64, 9}(ntuple(_ -> NaN, 9))), meta)
    @test occursin("\"cov\":null", JSON.json(json_value(n, MM_PX; kind = :plain)))
    @test all(isnan, cov(getfield(rt(n), :value)))
    # the wire format of a plain detection is the one stored files hold
    @test JSON.json(json_value(Attributed(Disc(10.5, 20.25, 3.0), meta), MM_PX; kind = :plain)) ==
        """{"frame":{"center":[10.5,20.25],"σ":3.0},"response":0.25,"edge_score":0.0,"laplacian_scale_score":1.0,"warp_idx":7}"""
    # metadata has no declared kind, so a quantity there is refused, naming the field
    err = try json_value(Attributed(frames[1], (response = 1.0, pitch = 2.5mm)), MM_PX; kind = :plain) catch x; x end
    @test err isa ArgumentError && occursin("pitch", err.msg)
end

@testitem "quantity JSON: detections and frames as numbers (Features' golden strings); the {value, unit} form refused" setup = [VGIoFixtures] begin
    # The plain-number strings below were written by VisualGeometryToolkitFeatures'
    # JSON.lower hooks (its golden-bytes test, pinned to the writer at ddc1bbe),
    # and the value beside it is what Features' JSON.lift read back from it
    # before the hooks moved here (recorded 2026-10-25 at 61389c3). Its strings
    # in the {value, unit} form are not read (the form is retired): refused.
    using VisualGeometryToolkitIO: UnitContract, MM_PX, json_value, from_json, read_quantity
    using VisualGeometryToolkitCore: Disc, OrientedDisc, EllipticFrame, AffineFrame, Point2, Attributed,
        AbstractDetection, Uncertain, CovMat, dpi, pt, inch, μm
    using Statistics: cov
    cm = parse_unit("cm")
    meta = (response = 0.25, edge_score = 0.0, laplacian_scale_score = -1.0, warp_idx = 7)
    quantities = [
        ("""{"value":2.5,"unit":"mm"}""", :length, 2.5mm),
        ("""{"value":2.0,"unit":"mm"}""", :length, 2.0mm),
        ("""{"value":0.1,"unit":"mm"}""", :length, 0.1mm),
        ("""{"value":13.0,"unit":"mm"}""", :length, 1.3cm),
        ("""{"value":62.79999999999999,"unit":"mm"}""", :length, 0.0628m),
        ("""{"value":49.999194444444434,"unit":"mm"}""", :length, 141.73pt),
        ("""{"value":25.4,"unit":"mm"}""", :length, 1.0inch),
        ("""{"value":0.00586,"unit":"mm"}""", :length, 5.86μm),
        ("""{"value":300,"unit":"dpi"}""", :density, 300dpi),
        ("""{"value":600,"unit":"dpi"}""", :density, 600dpi),
        ("""{"value":72.5,"unit":"dpi"}""", :density, 72.5dpi),
    ]
    for (s, kind, q) in quantities, c in (MM_PX, UnitContract(world = m))
        @test_throws ArgumentError read_quantity(JSON.parse(s), kind, c)
    end
    values = [
        (Size2, """{"width":{"value":640,"unit":"px"},"height":{"value":480,"unit":"px"}}""", Size2(width = 640px, height = 480px), :pixel),
        (Size2, """{"width":{"value":640.5,"unit":"px"},"height":{"value":480.25,"unit":"px"}}""", Size2(width = 640.5px, height = 480.25px), :pixel),
        (Size2, """{"width":{"value":210.0,"unit":"mm"},"height":{"value":297.0,"unit":"mm"}}""", Size2(width = 210.0mm, height = 297.0mm), :length),
        (Size2, """{"width":{"value":62.79999999999999,"unit":"mm"},"height":{"value":1.3,"unit":"mm"}}""", Size2(width = 62.8mm, height = 1.3mm), :length),
        (Size2, """{"width":640,"height":480}""", Size2(width = 640, height = 480), :plain),
        (Size2, """{"width":640.5,"height":480.0}""", Size2(width = 640.5, height = 480.0), :plain),
        (Disc, """{"center":[10.5,20.25],"σ":3.0}""", Disc(10.5, 20.25, 3.0), :plain),
        (Disc, """{"center":[{"value":10.5,"unit":"px"},{"value":20,"unit":"px"}],"σ":{"value":3,"unit":"px"}}""", Disc(Point2(10.5px, 20.0px), 3.0px), :pixel),
        (Disc, """{"center":[{"value":10.25,"unit":"px"},{"value":20.75,"unit":"px"}],"σ":{"value":1.6,"unit":"px"}}""", Disc(Point2(10.25px, 20.75px), 1.6px), :pixel),
        (Disc, """{"center":[{"value":1.0,"unit":"mm"},{"value":2.0,"unit":"mm"}],"σ":{"value":3.0,"unit":"mm"}}""", Disc(Point2(1.0mm, 2.0mm), 3.0mm), :length),
        (Disc, """{"center":[{"value":12.7,"unit":"mm"},{"value":0.3,"unit":"mm"}],"σ":{"value":1.25,"unit":"mm"}}""", Disc(Point2(12.7mm, 0.3mm), 1.25mm), :length),
        (OrientedDisc, """{"center":[1.0,2.0],"σ":3.0,"θ":0.5}""", OrientedDisc(1.0, 2.0, 3.0, 0.5), :plain),
        (EllipticFrame, """{"center":[10.5,20.25],"S":[2.0,0.3,0.3,1.5]}""", EllipticFrame([10.5, 20.25], [2.0 0.3; 0.3 1.5]), :plain),
        (AffineFrame, """{"center":[1.0,2.0],"A":[1.2,-0.2,0.1,0.9]}""", AffineFrame([1.0, 2.0], [1.2 0.1; -0.2 0.9]), :plain),
    ]
    for (T, s, x, kind) in values
        if occursin("\"value\"", s)
            @test_throws ArgumentError from_json(T, JSON.parse(s), MM_PX; kind)
        else
            @test (s, from_json(T, JSON.parse(s), MM_PX; kind)) == (s, x)
        end
    end
    detections = [
        ("""{"frame":{"center":[10.5,20.25],"σ":3.0},"response":0.25,"edge_score":0.0,"laplacian_scale_score":-1.0,"warp_idx":7}""", Disc(10.5, 20.25, 3.0), :plain),
        ("""{"frame":{"center":[{"value":10.5,"unit":"px"},{"value":20,"unit":"px"}],"σ":{"value":3,"unit":"px"}},"response":0.25,"edge_score":0.0,"laplacian_scale_score":-1.0,"warp_idx":7}""", Disc(Point2(10.5px, 20.0px), 3.0px), :pixel),
        ("""{"frame":{"center":[1.0,2.0],"σ":3.0,"θ":0.5},"response":0.25,"edge_score":0.0,"laplacian_scale_score":-1.0,"warp_idx":7}""", OrientedDisc(1.0, 2.0, 3.0, 0.5), :plain),
        ("""{"frame":{"center":[10.5,20.25],"S":[2.0,0.3,0.3,1.5]},"response":0.25,"edge_score":0.0,"laplacian_scale_score":-1.0,"warp_idx":7}""", EllipticFrame([10.5, 20.25], [2.0 0.3; 0.3 1.5]), :plain),
        ("""{"frame":{"center":[1.0,2.0],"A":[1.2,-0.2,0.1,0.9]},"response":0.25,"edge_score":0.0,"laplacian_scale_score":-1.0,"warp_idx":7}""", AffineFrame([1.0, 2.0], [1.2 0.1; -0.2 0.9]), :plain),
    ]
    for (s, f, kind) in detections
        if occursin("\"value\"", s)
            @test_throws ArgumentError from_json(AbstractDetection, JSON.parse(s), MM_PX; kind)
        else
            r = from_json(AbstractDetection, JSON.parse(s), MM_PX; kind)
            @test (s, getfield(r, :value), getfield(r, :attributes)) == (s, f, meta)
        end
    end
    s = """{"frame":{"center":[10.5,20.25],"σ":3.0},"cov":[0.01,0.001,0.0,0.001,0.02,0.0,0.0,0.0,0.005],"response":0.25,"edge_score":0.0,"laplacian_scale_score":-1.0,"warp_idx":7}"""
    r = from_json(AbstractDetection, JSON.parse(s), MM_PX; kind = :plain)
    @test r == Attributed(Uncertain(Disc(10.5, 20.25, 3.0), CovMat{3, Float64, 9}(Tuple([0.01 0.001 0.0; 0.001 0.02 0.0; 0.0 0.0 0.005]))), meta)
    # metadata is kept as it was read (it has no declared kind): an object there stays a parsed object
    s = """{"frame":{"center":[1.0,2.0],"σ":3.0},"response":1.0,"image_size":{"width":{"value":640,"unit":"px"},"height":{"value":480,"unit":"px"}},"pitch":{"value":2.5,"unit":"mm"}}"""
    a = getfield(from_json(AbstractDetection, JSON.parse(s), MM_PX; kind = :plain), :attributes)
    @test keys(a) == (:response, :image_size, :pitch) && a.response === 1.0
    @test a.pitch == Dict("value" => 2.5, "unit" => "mm")
    @test a.image_size == Dict("width" => Dict("value" => 640, "unit" => "px"), "height" => Dict("value" => 480, "unit" => "px"))
end

@testitem "quantity JSON: length roles — mm and pt in one record, each decimal-exact" setup = [VGIoFixtures] begin
    using VisualGeometryToolkitIO: UnitContract, MM_PX, write_quantity, read_quantity, json_value, from_json, role_unit
    using VisualGeometryToolkitCore: dpi, pt, inch, Disc, Point2
    import VisualGeometryToolkitIO as VgtkIO
    # a board-like record: its canvas in mm, its pattern in pt, its density in dpi
    c = UnitContract(world = mm, density = dpi, lengths = (pattern = pt,))
    @test role_unit(c, :world) == mm && role_unit(c, :pattern) == pt
    @test_throws ArgumentError role_unit(c, :ruler)
    @test_throws ArgumentError UnitContract(lengths = (world = pt,))       # world is the contract's own
    @test_throws ArgumentError UnitContract(lengths = (pattern = dpi,))    # a role is a length
    canvas = Size2(width = VgtkIO.read_length(150, mm), height = VgtkIO.read_length(77, mm))
    min_scale = VgtkIO.read_length(2.12598425197, pt)         # file-born pt numbers (BlobBoards' v4 pattern config)
    max_scale = VgtkIO.read_length(21.2595, pt)
    grid = VgtkIO.read_length(0.01, pt)
    blob = Disc(Point2(VgtkIO.read_length(17.5, pt), VgtkIO.read_length(125.25, pt)), min_scale)
    record = (
        units = json_value(c),
        canvas_size = json_value(canvas, c; kind = :length),
        print_density = write_quantity(1200dpi, c),
        frame_width = write_quantity(VgtkIO.read_length(11.25, mm), c),
        pattern = (min_scale = write_quantity(min_scale, c; role = :pattern), max_scale = write_quantity(max_scale, c; role = :pattern),
                   layout_grid = write_quantity(grid, c; role = :pattern)),
        blob = json_value(blob, c; kind = :length, role = :pattern),
    )
    text = JSON.json(record)
    # every number is the file's own decimal, in its role's unit
    @test occursin("\"units\":{\"world\":\"mm\",\"density\":\"dpi\",\"lengths\":{\"pattern\":\"pt\"}}", text)
    @test occursin("\"canvas_size\":{\"width\":150.0,\"height\":77.0}", text)
    @test occursin("\"min_scale\":2.12598425197", text) && occursin("\"max_scale\":21.2595", text) && occursin("\"layout_grid\":0.01", text)
    @test occursin("\"center\":[17.5,125.25]", text) && occursin("\"print_density\":1200", text) && occursin("\"frame_width\":11.25", text)
    # and read back, under the declaration the record carries, to the identical quantities
    d = JSON.parse(text)
    c2 = from_json(UnitContract, d["units"])
    @test c2 == c
    @test from_json(Size2, d["canvas_size"], c2; kind = :length) == canvas
    @test read_quantity(d["print_density"], :density, c2) == 1200dpi
    @test read_quantity(d["frame_width"], :length, c2) === VgtkIO.read_length(11.25, mm)
    @test read_quantity(d["pattern"]["min_scale"], :length, c2; role = :pattern) === min_scale
    @test read_quantity(d["pattern"]["max_scale"], :length, c2; role = :pattern) === max_scale
    @test read_quantity(d["pattern"]["layout_grid"], :length, c2; role = :pattern) === grid
    @test from_json(Disc, d["blob"], c2; kind = :length, role = :pattern) == blob
    # a pt length in mm has no decimal-exact number: the role is what keeps it exact
    @test write_quantity(min_scale, c) != 2.12598425197
    # roles are for lengths; an undeclared role is refused
    @test_throws ArgumentError write_quantity(1200dpi, c; role = :pattern)
    @test_throws ArgumentError read_quantity(640, :pixel, c; role = :pattern)
    @test_throws ArgumentError read_quantity(1.0, :length, MM_PX; role = :pattern)
    # a declaration without roles is the contract without them; the COLMAP sidecar holds none
    @test json_value(MM_PX) == (world = "mm", density = "dpi")
    @test from_json(UnitContract, JSON.parse("{\"world\": \"mm\"}")) == MM_PX
    @test_throws ArgumentError VgtkIO.save_units_sidecar(mktempdir(), c)
end
