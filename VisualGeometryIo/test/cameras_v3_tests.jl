# =============================================================================
# vgio/cameras/v3
# =============================================================================

@testitem "vgio/cameras/v3" setup = [VGIoFixtures] begin
    using VisualGeometryCore: BoundedCamera, Attributed, model
    # A lens VGIo has no serialization triple for — the extension point's guard rail.
    struct _UnregisteredLens <: AbstractLensModel end

    raw_cx, raw_cy = 2139.6593652825422, 1399.8009044208025      # OpenCV-frame K
    K = CameraCalibrationMat(
        SMatrix{3, 3, Float64}(
            2461.4310993500767, 0, 0,
            0, 2460.563045263503, 0,
            raw_cx, raw_cy, 1
        )
    )
    bc = Lens(Perspective(), Additive(Radial((-0.15680625941667098, 0.04565665542192153)), Tangential(-0.0003527415347489064, 0.0008423616851487834)))
    tpf = Lens(Equidistant(), Additive(Radial((0.1, -0.02, 0.003, -0.0004)), Tangential(1.0e-4, -2.0e-4), ThinPrism((2.0e-4,), (-1.0e-4,))))
    every_lens = [
        Pinhole(), Lens(Equidistant()), bc,
        opencv_lens([-0.1, 0.05, 1.0e-4, -2.0e-4, 0.01, 0.0, 0.0, 0.002, -5.5e-4, 7.0e-6, -2.2e-3, 1.2e-4, 1.0e-3, -2.0e-3]),
        KannalaBrandt(0.1, -0.02, 0.003, -0.0004),
        Scaramuzza(-0.1664, -0.0063, 0.035),
        Lens(DoubleSphere(0.8838, 0.9)),
        LiftLens(Radial((0.01, -0.002), (-0.3, 0.05, -0.004))),
        Division(-0.28, -0.09),
        tpf,
        Lens(VisualGeometryCore.Unified(0.7), Additive(Radial((-0.1, 0.01)), Tangential(1.0e-3, 0.0))),
    ]

    @testset "round trip is exact for every lens type" begin
        names = [Symbol("cam_", lpad(i, 2, '0')) for i in eachindex(every_lens)]
        recs = Dict(
            n => BoundedCamera(CameraModel(K; lens = l, image_origin = :colmap), Size2(4240, 2832))
                for (n, l) in zip(names, every_lens)
        )
        mktempdir() do dir
            path = joinpath(dir, "cameras.json")
            save(path, recs; provenance = (reprojection_error = 0.0354, n_calibration_images = 56))
            back = load(Dict{Symbol, BoundedCamera}, path)
            d = JSON3.read(read(path, String))
            @test back == recs                         # exact round trip by type
            @test back isa Dict{Symbol} && all(c -> c isa BoundedCamera, values(back))
            for n in names
                a, b = recs[n], back[n]
                @test extent(b) == Size2(4240, 2832)   # via VGC's bounds trait
                # `==`, not `≈`: JSON prints Float64 in its shortest
                # round-trippable form, so anything less than exact is a bug.
                @test model(a).intrinsics.K == model(b).intrinsics.K
                @test model(a).lens === model(b).lens
            end
            @test [c.name for c in d.cameras] == String.(names)       # written sorted by name
            @test d.provenance.reprojection_error == 0.0354
            @test model(back[:cam_03]).lens == bc
            # the lens is VGC's record: the structure, the arities as array lengths
            for (c, l) in zip(d.cameras, every_lens)
                @test VisualGeometryIo._plain(c.lens) == lens_record(l)
            end
            @test VisualGeometryIo._plain(d.cameras[9].lens) == Dict("lift" => Dict("radial" => Any[Any[], Any[-0.28, -0.09]]))
            @test VisualGeometryIo._plain(d.cameras[5].lens) == Dict("equidistant" => Any[], "radial" => Any[Any[0.1, -0.02, 0.003, -0.0004], Any[]])
            # one pixel frame for the file; nothing restated per camera
            @test d.pixel_frame == "colmap" && !haskey(d.cameras[1], :pixel_frame)
            @test !haskey(d.cameras[1], :size) && !haskey(d.cameras[1].K, :unit)
        end
    end

    @testset "pixel_frame carries the half pixel, exactly once" begin
        # PixelFrame{:opencv} puts the top-left pixel CENTER at (0,0); VGC's
        # canonical PixelFrame{:colmap} puts it at (0.5, 0.5). Reference values
        # cross-checked against cameras_vgio/cameras.bin and poses/bb undist_K
        # in the BlobBoards dataset, which are derived by an independent path.
        m = CameraModel(K; lens = bc, image_origin = :opencv)
        @test m.intrinsics.K[1, 3] == raw_cx + 0.5
        @test m.intrinsics.K[2, 3] == raw_cy + 0.5

        mktempdir() do dir
            path = joinpath(dir, "cameras.json")
            save(path, Dict(:c => BoundedCamera(m, Size2(4240, 2832))))
            @test JSON3.read(read(path, String)).pixel_frame == "colmap"
            back = load(Dict{Symbol, BoundedCamera}, path)
            # reloading must NOT shift again
            @test model(back[:c]).intrinsics.K == m.intrinsics.K
        end

        # A FOREIGN file may legitimately declare julia or opencv — an OpenCV
        # calibration puts the top-left pixel CENTRE at (0,0) — and the reader
        # must convert it. This package never WRITES such a file, so this
        # hand-edited one is what exercises the conversion that is the
        # declaration's whole purpose. Signs: opencv→colmap is +0.5, julia→colmap is -0.5.
        mktempdir() do dir
            canonical = CameraModel(K; lens = bc, image_origin = :colmap)
            path = joinpath(dir, "cameras.json")
            save(path, Dict(:c => BoundedCamera(canonical, Size2(4240, 2832))))
            top = Dict{Symbol, Any}(pairs(JSON3.read(read(path, String))))
            cx0 = canonical.intrinsics.K[1, 3]
            cy0 = canonical.intrinsics.K[2, 3]
            for (frame, shift) in (("colmap", 0.0), ("opencv", 0.5), ("julia", -0.5))
                q = joinpath(dir, "$(frame).json")
                write(q, JSON3.write(merge(top, Dict(:pixel_frame => frame))))
                got = load(Dict{Symbol, BoundedCamera}, q)[:c]
                @test model(got).intrinsics.K[1, 3] ≈ cx0 + shift
                @test model(got).intrinsics.K[2, 3] ≈ cy0 + shift
                # ...and re-saving states the canonical frame, not the file's
                r2 = joinpath(dir, "$(frame)_again.json")
                save(r2, Dict(:c => got))
                @test JSON3.read(read(r2, String)).pixel_frame == "colmap"
            end
        end

        # ...and a stored camera has NO frame of its own to contradict the
        # model with: K is in CANONICAL_PIXEL_FRAME by construction.
        @test !hasproperty(BoundedCamera(m, Size2(4240, 2832)), :pixel_frame)

        # The registered set lives in VGC, not here: this package asks rather
        # than keeping a list, so the refusal quotes VGC's own valid options.
        @test VisualGeometryIo.pixel_frame_name(VisualGeometryIo.CANONICAL_PIXEL_FRAME) == "colmap"
        @test VisualGeometryIo.pixel_frame_from_name("opencv") === PixelFrame{:opencv}
        @test VisualGeometryIo.pixel_origin(PixelFrame{:julia}) === :julia
        err = try
            VisualGeometryIo.pixel_frame_from_name("weird"); nothing
        catch e
            e
        end
        @test err isa ArgumentError
        @test occursin("colmap", err.msg) && occursin("julia", err.msg)
    end

    @testset "corrupt records are refused, not repaired" begin
        rec = BoundedCamera(CameraModel(K; lens = bc, image_origin = :colmap), Size2(4240, 2832))
        mktempdir() do dir
            path = joinpath(dir, "cameras.json")
            save(path, Dict(:c => rec))
            d = JSON3.read(read(path, String))
            tomutable(x::AbstractDict) = Dict{Symbol, Any}(Symbol(k) => tomutable(v) for (k, v) in x)
            tomutable(x::AbstractVector) = Any[tomutable(v) for v in x]
            tomutable(x) = x
            function refused(mutate, key)
                dd = tomutable(d)
                mutate(dd)
                q = joinpath(dir, "bad.json")
                write(q, JSON3.write(dd))
                e = try
                    load(Dict{Symbol, BoundedCamera}, q); nothing
                catch e
                    e
                end
                return e isa ArgumentError && occursin(key, e.msg) && occursin("bad.json", e.msg)
            end
            lens!(f) = dd -> f(dd[:cameras][1][:lens])
            # the lens record: exactly VGC's grammar, refused by name otherwise
            @test refused(lens!(l -> (l[:spline] = [1.0])), "spline")
            @test refused(lens!(l -> delete!(l, :perspective)), "projection")
            @test refused(lens!(l -> (l[:add][2][:tangential] = [0.1])), "tangential")
            @test refused(lens!(l -> (l[:add][1][:radial] = [[0.1]])), "radial")
            @test refused(lens!(l -> (l[:add][1][:radial] = [["0.1"], []])), "radial")
            @test refused(dd -> (dd[:cameras][1][:lens] = Dict(:model => "brown_conrady", :params => Dict(:k1 => 0.1))), "retired v1/v2 lens record")   # a v1/v2 lens record is refused by name
            @test refused(dd -> (dd[:pixel_frame] = "weird"), "pixel_frame")
            @test refused(dd -> delete!(dd, :pixel_frame), "pixel_frame")
            @test refused(dd -> (dd[:cameras][1][:width] = 0), "width")
            @test refused(dd -> (dd[:cameras][1][:unit] = "px"), "unit")               # no key the table does not name
            @test refused(dd -> (dd[:schema] = "vgio/cameras/v4"), "schema")
            @test refused(dd -> (dd[:schema] = "vgio/cameras/v2"), "lens records")     # v2 is retired, refused by name
            @test refused(dd -> (dd[:schema] = "vgio/cameras/v1"), "lens records")
        end

        mktempdir() do dir
            @test_throws ArgumentError save(joinpath(dir, "e.json"), Dict(Symbol("") => rec))       # a name is non-empty
            @test_throws ArgumentError save(joinpath(dir, "z.json"), Dict(:c => BoundedCamera(model(rec), Size2(0, 100))))   # extent positive
            @test_throws ArgumentError save(joinpath(dir, "p.json"), Dict(:c => model(rec)))       # a stored camera is bounded
        end

        # typed single reads — record or model. A one-camera store resolves
        # unnamed; a multi-camera store REFUSES the unnamed form (naming the
        # choices) instead of guessing the first.
        mktempdir() do dir
            one = joinpath(dir, "one.json")
            save(one, Dict(:c => rec))
            @test load(CameraModel, one) == model(rec)
            @test load(BoundedCamera, one) == rec
            @test load(CameraModel, one; name = "c") == model(rec)
            @test load(CameraModel, one; name = :c) == model(rec)
            @test_throws ArgumentError load(CameraModel, one; name = "nope")

            rec2 = BoundedCamera(model(rec), Size2(640, 480))
            two = joinpath(dir, "two.json")
            save(two, Dict(:c => rec, :other => rec2))
            err = try
                load(CameraModel, two); nothing
            catch e
                e
            end
            @test err isa ArgumentError
            @test occursin("other", err.msg)          # the names are the message
            @test load(CameraModel, two; name = "other") == model(rec2)
            @test load(BoundedCamera, two; name = "other") == rec2
            # intent assertion: asking for transforms from a cameras file errors
            @test_throws ArgumentError load(Vector{EuclideanMap}, one)
        end
    end

    @testset "export to COLMAP: exact where expressible, loud where not" begin
        # The store is strictly wider than the export: every lens a COLMAP model
        # computes survives exactly — written as that model, read back as the same
        # lens — and anything else throws rather than being approximated.
        expressible = [
            Pinhole() => :PINHOLE, bc => :OPENCV, every_lens[4] => nothing,
            Lens(Perspective(), Additive(Radial((-0.1, 0.02, 0.003), (0.01, 0.0, 0.0)), Tangential(1.0e-4, 2.0e-4))) => :FULL_OPENCV,
            KannalaBrandt(0.1, -0.02, 0.003, -4.0e-4) => :OPENCV_FISHEYE, tpf => :THIN_PRISM_FISHEYE,
            Lens(Equidistant()) => :FISHEYE, Division(-0.3) => :DIVISION,
            Lens(Perspective(), VisualGeometryCore.FieldOfView(0.9)) => :FOV,
            Lens(Equidistant(), Chain(Radial((0.1, -0.02, 0.003, 0.0, 0.0, 1.0e-4)), Additive(Tangential(3.0e-4, -2.0e-4), ThinPrism((1.0e-3, 0.0), (-5.0e-4, 2.0e-5))))) => :RAD_TAN_THIN_PRISM_FISHEYE,
            Lens(DoubleSphere(0.5, 0.5)) => nothing, Division(-0.3, 0.01) => nothing,
        ]
        for (l, want) in expressible
            @test colmap_model(l) === want
        end
        ok = Dict(Symbol("cam_", lpad(i, 2, '0')) => BoundedCamera(CameraModel(K; lens = l, image_origin = :colmap), Size2(4240, 2832)) for (i, (l, m)) in enumerate(expressible) if m !== nothing)
        cams, ids = colmap_cameras(ok)
        @test ids[:cam_02] == 2 && ids[:cam_01] == 1                                    # ids in name order
        mktempdir() do dir
            save(dir, ColmapModel(cams, ColmapImage[], VisualGeometryIo.ColmapPoint3D[], MM_PX); format = :text)
            back = load(ColmapModel, dir)
            for (n, j) in ids
                m = camera_model(back.cameras[j])
                @test m.intrinsics.K == model(ok[n]).intrinsics.K
                @test m.lens === model(ok[n]).lens
            end
            # names are NOT recoverable from the COLMAP side — the documented loss
            @test !hasproperty(back.cameras[1], :name)
        end
        # the partial-export filter of `colmap_cameras`' docstring
        mixed = Dict(:a => ok[:cam_01], :ds => BoundedCamera(CameraModel(K; lens = Lens(DoubleSphere(0.5, 0.5)), image_origin = :colmap), Size2(4240, 2832)))
        @test length(filter(p -> colmap_model(model(last(p)).lens) !== nothing, mixed)) == 1
        @test_throws ArgumentError colmap_cameras(mixed)
    end

    @testset "the calibration block: an attribute of the value, closed, read back exactly" begin
        m = CameraModel(K; lens = bc, image_origin = :opencv)
        full = (;
            tool = "OpenCV calibrateCamera", source = "sony-fx3_tamron-20mm-f-2.8-f-8.json", date = "2026-01-19",
            reprojection_rms = 0.15948540185086135, n_images = 36, n_points = 2016, note = "chessboard 7x8, 50 mm squares",
        )
        part = (; tool = "OpenCV calibrateCamera", reprojection_rms = 0.2737)
        bcam = BoundedCamera(m, Size2(4240, 2832))
        recs = Dict(:full => Attributed(bcam, (; calibration = full)), :part => Attributed(bcam, (; calibration = part)), :none => bcam)
        @test recs[:full].calibration.tool == "OpenCV calibrateCamera"            # the attribute forwards
        mktempdir() do dir
            path = joinpath(dir, "cameras.json")
            save(path, recs)
            back = load(Dict{Symbol, BoundedCamera}, path)
            @test back == recs                                                     # exact by type
            @test back[:full] isa Attributed && back[:none] isa BoundedCamera
            @test back[:full].calibration === full && back[:part].calibration === part
            d = JSON3.read(read(path, String))
            # what is unknown is absent, in declaration order; no block at all when there is none
            @test collect(keys(d.cameras[1].calibration)) == [:tool, :source, :date, :reprojection_rms, :n_images, :n_points, :note]
            @test !haskey(d.cameras[2], :calibration)
            @test collect(keys(d.cameras[3].calibration)) == [:tool, :reprojection_rms]
            @test load(BoundedCamera, path; name = "full") == bcam                 # the single-camera views drop it
            # write → read → write is byte-identical
            again = joinpath(dir, "again.json")
            save(again, back)
            @test read(again, String) == read(path, String)

            tomutable(x::AbstractDict) = Dict{Symbol, Any}(Symbol(k) => tomutable(v) for (k, v) in x)
            tomutable(x::AbstractVector) = Any[tomutable(v) for v in x]
            tomutable(x) = x
            function refused(mutate, key)
                dd = tomutable(d)
                mutate(dd[:cameras][1][:calibration])
                q = joinpath(dir, "bad.json")
                write(q, JSON3.write(dd))
                e = try
                    load(Dict{Symbol, BoundedCamera}, q); nothing
                catch e
                    e
                end
                return e isa ArgumentError && occursin(key, e.msg)
            end
            @test refused(c -> (c[:operator] = "x"), "operator")                       # a closed set of keys
            @test refused(empty!, "at least one")
            @test refused(c -> (c[:source] = "/raid/data/calib.json"), "machine path")
            @test refused(c -> (c[:source] = "C:\\calib.json"), "machine path")
            @test refused(c -> (c[:date] = "19.01.2026"), "date")
            @test refused(c -> (c[:date] = "2026-13-01"), "date")
            @test refused(c -> (c[:reprojection_rms] = -0.1), "reprojection_rms")
            @test refused(c -> (c[:reprojection_rms] = "0.1"), "reprojection_rms")
            @test refused(c -> (c[:n_images] = 0), "n_images")
            @test refused(c -> (c[:note] = ""), "note")
            # the writer holds the value to the file's rules
            for bad in ((;), (; source = "/home/someone/calib.json"), (; date = "2026-1-19"), (; reprojection_rms = -1.0),
                    (; n_points = 0), (; tool = ""), (; operator = "x"))
                @test_throws ArgumentError save(joinpath(dir, "w.json"), Dict(:c => Attributed(bcam, (; calibration = bad))))
            end
        end
    end

    @testset "a lens the store cannot hold fails loudly at write time" begin
        unregistered = BoundedCamera(CameraModel(K; lens = _UnregisteredLens(), image_origin = :colmap), Size2(10, 10))
        mktempdir() do dir
            @test_throws ArgumentError save(joinpath(dir, "u.json"), Dict(:u => unregistered))
            @test !isfile(joinpath(dir, "u.json"))                                     # nothing half-written
        end
    end
end
