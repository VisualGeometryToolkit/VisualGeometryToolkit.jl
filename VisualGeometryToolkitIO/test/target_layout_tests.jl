# =============================================================================
# vgio/target/v4 and vgio/rig/v7: a planar target stated by its LAYOUT (VGC's
# AprilGrid, SchoepsPattern), the v3/v6 files still read, and the tools' own
# layout files (Schöps' pattern YAML, Kalibr's aprilgrid YAML) read as the
# same targets VGC builds from the same parameters.
# =============================================================================

@testsetup module LayoutFixtures
    using StaticArrays: SVector
    using VisualGeometryToolkitCore: mm, m, AprilGrid, SchoepsPattern, PlanarTarget, Canvas, Size2, Aspect, Skew
    export layouts_fixture, april_grid, schoeps_pattern, layout_targets

    layouts_fixture(name) = joinpath(@__DIR__, "fixtures", "layouts", name)
    april_grid() = AprilGrid(; tag_rows = 6, tag_cols = 5, tag_size = 88mm, tag_spacing = 26.4mm, apriltag_family = "tag36h11", first_id = 30)
    schoeps_pattern() = SchoepsPattern(;
        squares_x = 12, squares_y = 9, square_length = 25mm, star_segments = 12, apriltag_family = "tag25h9",
        apriltags = [(; x = 2, y = 1, width = 2, height = 2, id = 4), (; x = 7, y = 5, width = 3, height = 2, id = 11)],
    )
    # each layout bare, and with a calibration and a page
    layout_targets() = [
        PlanarTarget(april_grid()),
        PlanarTarget(april_grid(); calibration = Aspect(0.01) ∘ Skew(-0.002), canvas = Canvas(Size2(width = 700mm, height = 800mm), SVector(-20mm, -15mm))),
        PlanarTarget(schoeps_pattern()),
        PlanarTarget(schoeps_pattern(); canvas = Canvas(Size2(width = 310mm, height = 235mm), SVector(-5mm, -5mm))),
    ]
end

@testitem "target/v4: a layout target is written by its layout and reads back the same target" setup = [VGIoFixtures, LayoutFixtures] begin
    import VisualGeometryToolkitIO as VgtkIO
    using VisualGeometryToolkitCore: PlanarTarget, AbstractTarget, Rig, PosedTarget, BodyFrame, layout, design, points, calibration, canvas
    using Rotations: RotZ
    mktempdir() do dir
        for (i, t) in enumerate(layout_targets()), u in (mm, m)
            p = joinpath(dir, "t$(i)_$(u).json")
            save(p, t; length_unit = u)
            d = parse_json(read(p, String))
            @test d.schema == "vgio/target/v5" && haskey(d.target, :layout) && !haskey(d.target, :nominal) && !haskey(d.target, :reference)
            back = load(AbstractTarget, p)
            @test back == t && layout(back) == layout(t)
            @test points(back) == points(t) && all(map(===, points(back), points(t)))   # bit for bit: the layout's lengths are decimal-exact
            @test load(PlanarTarget, p) == t
        end
        # the record as written: the tag (the tag rule's snake_case of the type), every key, lengths in the file's unit
        p = joinpath(dir, "grid.json")
        save(p, PlanarTarget(april_grid()))
        l = parse_json(read(p, String)).target.layout
        @test l.type == "april_grid" && l.tag_rows == 6 && l.tag_cols == 5 && l.tag_size == 88.0 && l.tag_spacing == 26.4 &&
            l.apriltag_family == "tag36h11" && l.first_id == 30
        save(p, PlanarTarget(schoeps_pattern()); length_unit = m)
        l = parse_json(read(p, String)).target.layout
        @test l.type == "schoeps_pattern" && l.squares_x == 12 && l.squares_y == 9 && l.square_length == 0.025 && l.star_segments == 12 &&
            l.apriltag_family == "tag25h9" && [(t.x, t.y, t.width, t.height, t.id) for t in l.apriltags] == [(2, 1, 2, 2, 4), (7, 5, 3, 2, 11)]
        # in a rig (rig/v7): the layout travels, and the identity is the points', as a nominal target's
        X = EuclideanMap(RotMatrix(RotZ(0.3)), SVector(0.1, -0.2, 1.5) .* m)
        lt = PlanarTarget(april_grid())
        nt = PlanarTarget(design(lt))                               # the same points, no layout
        @test layout(nt) === nothing && points(nt) == points(lt)
        rig(t) = Rig{BodyFrame{:board_rig}}(Dict(:grid => PosedTarget(t, X)))
        rp = joinpath(dir, "rig.json")
        save(rp, rig(lt))
        @test parse_json(read(rp, String)).schema == "vgio/rig/v8"
        back = load(Rig, rp)
        @test layout(back[:grid].target) == april_grid() && back[:grid].target == lt
        @test content_id(rig(lt)) == content_id(rig(nt)) == content_id(rp)
        np = joinpath(dir, "rig_nominal.json")
        save(np, rig(nt))
        @test content_id(np) == content_id(rp)
    end
end

@testitem "target/v4: layout refusals" setup = [VGIoFixtures, LayoutFixtures] begin
    import VisualGeometryToolkitIO as VgtkIO
    using VisualGeometryToolkitCore: PlanarTarget, AbstractTarget, AprilGrid
    msg(f) = (e = try f(); nothing catch e; e end; e isa ArgumentError ? e.msg : "")
    mktempdir() do dir
        p = joinpath(dir, "grid.json")
        save(p, PlanarTarget(april_grid()))
        good = JSON.parse(read(p, String); dicttype = Dict{String, Any})
        edited(f) = (t = deepcopy(good); f(t); q = joinpath(dir, "edited.json"); write(q, JSON.json(t, 2)); q)
        # exactly one of reference, nominal and layout
        both = edited(t -> t["target"]["nominal"] = Any[Any[0.0, 0.0]])
        @test occursin("exactly one of", msg(() -> load(AbstractTarget, both)))
        @test occursin("exactly one of", msg(() -> load(AbstractTarget, edited(t -> t["target"]["reference"] = "b76e"))))
        @test occursin("exactly one of", msg(() -> load(AbstractTarget, edited(t -> delete!(t["target"], "layout")))))
        # an unknown layout type lists the known ones
        m1 = msg(() -> load(AbstractTarget, edited(t -> t["target"]["layout"]["type"] = "aprilgrid")))
        @test occursin("unknown layout type \"aprilgrid\"", m1) && occursin("april_grid", m1) && occursin("schoeps_pattern", m1)
        # a key missing, a key of the other layout, a bad value
        @test occursin("states its `first_id`", msg(() -> load(AbstractTarget, edited(t -> delete!(t["target"]["layout"], "first_id")))))
        @test occursin("has no `squares_x`", msg(() -> load(AbstractTarget, edited(t -> t["target"]["layout"]["squares_x"] = 3))))
        @test occursin("positive integer", msg(() -> load(AbstractTarget, edited(t -> t["target"]["layout"]["tag_rows"] = 0))))
        @test occursin("not positive", msg(() -> load(AbstractTarget, edited(t -> t["target"]["layout"]["tag_size"] = -1.0))))   # VGC's
        # a points target holds no layout
        pts = edited(t -> (t["target"]["type"] = "points"; t["target"]["points"] = Any[Any[0.0, 0.0, 0.0]]; delete!(t["target"], "calibration")))
        @test occursin("points target has `points`", msg(() -> load(AbstractTarget, pts)))
        # units: a plain layout is refused on write (attach the unit), a file unit that is not a length on read
        @test occursin("plain numbers", msg(() -> save(joinpath(dir, "plain.json"), PlanarTarget(AprilGrid(; tag_rows = 2, tag_cols = 2, tag_size = 0.09, tag_spacing = 0.027)))))
        @test !isfile(joinpath(dir, "plain.json"))
        @test occursin("length unit", msg(() -> load(AbstractTarget, edited(t -> t["length_unit"] = "px"))))
        # a v3 record holds no layout: refused by its step, naming the version that does
        @test occursin("target v4", msg(() -> load(AbstractTarget, edited(t -> t["schema"] = "vgio/target/v3"))))
    end
end

@testitem "target/v3 and rig/v6 files still load, and migrate to v4 / v7" setup = [VGIoFixtures, LayoutFixtures] begin
    import VisualGeometryToolkitIO as VgtkIO
    using VisualGeometryToolkitCore: AbstractTarget, Rig, Attributed
    units(name) = joinpath(@__DIR__, "fixtures", "units", name)
    mktempdir() do dir
        for (name, old, new, T) in (("target_planar.json", "vgio/target/v3", "vgio/target/v5", AbstractTarget),
                                    ("target_points.json", "vgio/target/v3", "vgio/target/v5", AbstractTarget),
                                    ("rig_targets.json", "vgio/rig/v6", "vgio/rig/v8", Rig), ("rig.json", "vgio/rig/v6", "vgio/rig/v8", Rig))
            src = joinpath(dir, name)
            write(src, replace(read(units(name), String), "\"schema\": \"$(new)\"" => "\"schema\": \"$(old)\""))
            @test occursin(old, read(src, String))
            @test VgtkIO._same_value(load(T, src), load(T, units(name)))     # read through its step, silently
            r = vgio_migrate(src)
            @test r.from == old && r.to == new
            @test read(r.out, String) == read(units(name), String)            # the current file, byte for byte
            T === Rig && @test r.old_ids == [r.new_id] && r.new_id == content_id(units(name))   # the identity does not move
        end
        # the identity fixture is a rig/v6 file: it reads, and names the rig it always did (pinned in target_record_tests)
        every_kind = joinpath(@__DIR__, "fixtures", "identity", "rig_every_kind.json")
        @test parse_json(read(every_kind, String)).schema == "vgio/rig/v6"
        @test content_id(every_kind) == content_id(load(Rig, every_kind))
    end
end

@testitem "layout YAML: Schöps' pattern and Kalibr's aprilgrid read as VGC's targets of the same parameters" setup = [VGIoFixtures, LayoutFixtures] begin
    import VisualGeometryToolkitIO as VgtkIO
    using VisualGeometryToolkitCore: PlanarTarget, AbstractTarget, AprilGrid, SchoepsPattern, Canvas, layout, points, canvas, board
    msg(f) = (e = try f(); nothing catch e; e end; e isa ArgumentError ? e.msg : "")
    rl = VgtkIO.read_length
    # Schöps: the lab's 1 m, 30 × 30 pattern (create_calibration_pattern.py --num_star_segments 16 --apriltag_index 9)
    sp = layouts_fixture("pattern_1m_30x30_segments_16_apriltag_9.yaml")
    t = load(PlanarTarget, sp)
    want = SchoepsPattern(; squares_x = 30, squares_y = 30, square_length = rl(0.03306666666666667, m), star_segments = 16,
        apriltags = [(; x = 13, y = 13, width = 4, height = 4, id = 9)])
    @test layout(t) == want && points(t) == points(PlanarTarget(want)) && length(points(t)) == 816
    # the page is the canvas: its lower-left corner at (−pattern_start_x, pattern_end_y − height) in the design chart; no board
    W, H = rl(999.9999999999999, mm), rl(999.9999999999999, mm)
    @test canvas(t) == Canvas(Size2(width = W, height = H), SVector(-rl(4.0, mm), rl(995.9999999999998, mm) - H)) && board(t) === nothing
    @test t == PlanarTarget(want; canvas = canvas(t)) && load(AbstractTarget, sp) == t
    # Kalibr: tagSpacing is the gap over the tag size; VGC's AprilGrid takes the gap
    kp = layouts_fixture("aprilgrid_8x8_90mm.yaml")
    k = load(PlanarTarget, kp)
    @test layout(k) == AprilGrid(; tag_rows = 8, tag_cols = 8, tag_size = rl(0.09, m), tag_spacing = rl(27.0, mm))   # 0.09 · 0.3, decimal-exact
    @test k == PlanarTarget(layout(k)) && length(points(k)) == 256 && layout(k).apriltag_family == "tag36h11" && layout(k).first_id == 0
    @test isapprox(ustrip(mm, layout(k).tag_spacing), 27.0; atol = 1.0e-12)
    # either, saved, is a layout record that reads back the same target
    mktempdir() do dir
        for x in (t, k)
            p = joinpath(dir, "t.json")
            save(p, x)
            @test haskey(parse_json(read(p, String)).target, :layout) && load(AbstractTarget, p) == x
        end
        # refusals: another Kalibr target, a key the reader does not know, a page that disagrees with the pattern, neither format
        y(name, text) = (p = joinpath(dir, name); write(p, text); p)
        @test occursin("only an `aprilgrid`", msg(() -> load(PlanarTarget, y("cb.yaml", "target_type: 'checkerboard'\ntargetCols: 6\n"))))
        @test occursin("`codeOffset`", msg(() -> load(PlanarTarget, y("k.yaml", read(kp, String) * "codeOffset: 5\n"))))
        @test occursin("not an integer", msg(() -> load(PlanarTarget, y("k2.yaml", replace(read(kp, String), "tagCols: 8" => "tagCols: 8.5")))))
        @test occursin("spans", msg(() -> load(PlanarTarget, y("s.yaml", replace(read(sp, String), "pattern_end_x_mm: 995.9999999999998" => "pattern_end_x_mm: 990.0")))))
        @test occursin("no `squares_y`", msg(() -> load(PlanarTarget, y("s2.yaml", replace(read(sp, String), "squares_y: 30\n" => "")))))
        @test occursin("neither a Kalibr", msg(() -> load(PlanarTarget, y("x.yaml", "foo: 1\n"))))
    end
end

@testitem "layout YAML: comments, a byte-order mark, column-0 list items, the Kalibr gap decimal-exact" setup = [VGIoFixtures, LayoutFixtures] begin
    import VisualGeometryToolkitIO as VgtkIO
    using VisualGeometryToolkitCore: PlanarTarget, AprilGrid, SchoepsPattern, Rig, PosedTarget, BodyFrame, layout, points
    using Rotations: RotZ
    rl = VgtkIO.read_length
    # Kalibr's stock april_6x6.yaml, comments and all
    kalibr = """
    target_type: 'aprilgrid' #gridtype
    tagCols: 6               #number of apriltags
    tagRows: 6               #number of apriltags
    tagSize: 0.088           #size of apriltag, edge to edge [m]
    tagSpacing: 0.3          #ratio of space between tags to tagSize
                             #example: tagSize=2m, spacing=0.5m --> tagSpacing=0.25[-]
    """
    # Schöps' pattern as PyYAML's default dump writes it: list items at column 0
    schoeps(items) = """
    num_star_segments: 12
    squares_x: 12
    squares_y: 9
    square_length_in_meters: 0.025
    apriltags:
    $(items)
    """
    column0 = "- tag_x: 2\n  tag_y: 1\n  width: 2\n  height: 2\n  index: 4\n- tag_x: 7\n  tag_y: 5\n  width: 3\n  height: 2\n  index: 11"
    indented = "  - tag_x: 2\n    tag_y: 1\n    width: 2\n    height: 2\n    index: 4\n  - tag_x: 7\n    tag_y: 5\n    width: 3\n    height: 2\n    index: 11"
    want_pattern = SchoepsPattern(;
        squares_x = 12, squares_y = 9, square_length = rl(0.025, m), star_segments = 12,
        apriltags = [(; x = 2, y = 1, width = 2, height = 2, id = 4), (; x = 7, y = 5, width = 3, height = 2, id = 11)],
    )
    mktempdir() do dir
        y(name, text) = (p = joinpath(dir, name); write(p, text); p)
        bom(name, text) = (p = joinpath(dir, name); write(p, vcat(UInt8[0xef, 0xbb, 0xbf], codeunits(text))); p)
        # comments: Kalibr's stock file reads
        k = load(PlanarTarget, y("april_6x6.yaml", kalibr))
        @test layout(k) isa AprilGrid && layout(k).tag_rows == 6 && layout(k).tag_cols == 6 && layout(k).tag_size == rl(0.088, m)
        @test layout(k) == layout(load(PlanarTarget, y("q.yaml", "target_type: \"aprilgrid\"  # a comment\ntagCols: 6\ntagRows: 6\ntagSize: 0.088\ntagSpacing: 0.3\n")))
        # a quoted '#' is not a comment
        @test occursin("\"april #grid\"", let e = try load(PlanarTarget, y("h.yaml", "target_type: 'april #grid'\n")); nothing catch e; e end; e.msg end)
        # the gap is the product of the decimals, rounded once: 26.4 mm, as read from a file
        @test layout(k).tag_spacing === rl(26.4, mm) && layout(k).tag_spacing !== rl(0.088, m) * 0.3
        p = joinpath(dir, "k.json")
        save(p, k)
        @test parse_json(read(p, String)).target.layout.tag_spacing == 26.4
        from_mm = PlanarTarget(AprilGrid(; tag_rows = 6, tag_cols = 6, tag_size = rl(88.0, mm), tag_spacing = rl(26.4, mm)))
        @test k == from_mm && all(map(===, points(k), points(from_mm)))
        X = EuclideanMap(RotMatrix(RotZ(0.3)), SVector(0.1, -0.2, 1.5) .* m)
        rig(t) = Rig{BodyFrame{:board_rig}}(Dict(:grid => PosedTarget(t, X)))
        @test content_id(rig(k)) == content_id(rig(from_mm))
        # a UTF-8 byte-order mark is skipped, by both layout readers
        @test load(PlanarTarget, bom("bom_k.yaml", kalibr)) == k
        @test layout(load(PlanarTarget, bom("bom_s.yaml", schoeps(indented)))) == want_pattern
        # list items at column 0 under their key, as the indented form
        @test layout(load(PlanarTarget, y("s0.yaml", schoeps(column0)))) == want_pattern
        @test layout(load(PlanarTarget, y("s2.yaml", schoeps(indented)))) == want_pattern
    end
end
