# A kind of another package (kind_spec.jl): the public hook an owner states its kind's table with — `KindSpec`, the
# combinators, `write_kind` / `read_kind`, `json_schema(kind)`, `schema_versions(kind)`, `migrate_file`, `migrate_body`,
# `body_spec` — on a toy kind defined here, outside the package; and the kinds that moved to their owners in IO 0.11
# (vignetting, environment_light, radial_profile: VisualRadiometryToolkitIO's; panorama: PanoramaStitching's), each
# refused by name with its owner.

@testitem "kind spec: a toy kind's table checks what is written and what is read" setup = [VGIoFixtures] begin
    import VisualGeometryToolkitIO as VgtkIO
    using VisualGeometryToolkitCore: FitResult, ReturnCode, StopRule, Trajectory, CameraFrame, WorldFrame
    # the table: every combinator, a body of this package's (a camera, a trajectory, a fit), a cross-field check
    tone(self) = ["flat" => VgtkIO.object_spec(:type => VgtkIO.token_spec("flat"), :level => VgtkIO.real_spec()),
                  "log" => VgtkIO.object_spec(:type => VgtkIO.token_spec("log"), :of => self)]
    TONE = VgtkIO.tagged_spec(tone, "a tone"; def = "tone", unknown = t -> "$(repr(t)) is no tone (the toy's own message)")
    WINDOW = VgtkIO.if_key_spec(:radius, VgtkIO.object_spec(:center => VgtkIO.body_spec(PixelFrame), :radius => VgtkIO.real_spec()),
                                VgtkIO.object_spec(:center => VgtkIO.body_spec(PixelFrame)))
    SWEEP = VgtkIO.KindSpec(
        "toy/sweep/v2.0",
        [:camera => VgtkIO.body_spec(BoundedCamera), :rotations => VgtkIO.body_spec(Trajectory),
         :gains => VgtkIO.array_spec(VgtkIO.array_spec(VgtkIO.real_spec(); length = 3)), :frames => VgtkIO.integer_spec(; positive = true),
         :mode => VgtkIO.named_spec(in(("still", "video")), "a capture mode (still, video)"), :tone => TONE, :window => WINDOW],
        [:fit => VgtkIO.body_spec(FitResult), :offset => VgtkIO.integer_spec()];
        package = "Toy",
        check = r -> length(r[:gains]) == r[:frames] ? nothing : "gains: one row per frame ($(r[:frames]))",
        rule = "one row of gains per frame", schema_keywords = [Dict("properties" => Dict("gains" => Dict("minItems" => 1)))],
    )
    K = CameraCalibrationMat(SMatrix{3, 3, Float64}(1210.5, 0, 0, 0, 1209.25, 0, 320.375, 240.625, 1))
    cam = BoundedCamera(CameraModel(K; lens = KannalaBrandt(0.1, -0.02, 0.003, -0.0004), pixel_frame = :colmap), Size2(640, 480))
    z = [0.0, 0.0, 0.0] .* m
    rots = Trajectory(CameraFrame, WorldFrame, Dict(1 => EuclideanMap(QuatRotation(1.0, 0.0, 0.0, 0.0, false), z)); images = Dict(1 => "a.png"))
    fit = FitResult(nothing, ReturnCode.Success, first(instances(StopRule)), 12, 0.41, 31580, nothing)
    style = VgtkIO.VgioStyle(; pixel_frame = PixelFrame{:opencv})
    @test VgtkIO.pixel_frame(style) === PixelFrame{:opencv}
    @test VgtkIO.pixel_frame_offset(PixelFrame{:colmap}, PixelFrame{:opencv}) == SVector(-0.5, -0.5)
    @test VgtkIO.pixel_frame_offset(PixelFrame{:opencv}, PixelFrame{:colmap}) == SVector(0.5, 0.5)
    body(; kw...) = merge((camera = VgtkIO.camera_json(cam, VgtkIO.pixel_frame(style)), rotations = rots, gains = [[1.0, 1.0, 1.0]], frames = 1,
                           mode = "still", tone = (type = "log", of = (type = "flat", level = 0.5)), window = (center = [3.5, 4.0],),
                           fit = VgtkIO.fit_record_json(fit)), (; kw...))
    mktempdir() do dir
        p = joinpath(dir, "sweep.json")
        @test VgtkIO.write_kind(p, SWEEP, body(); style, provenance = (run = "a",)) === nothing
        d = JSON.parse(read(p, String))
        # the envelope is this package's, the body's keys in their order, the provenance last
        @test collect(keys(d)) == ["schema", "generator", "units", "conventions", "camera", "rotations", "gains", "frames", "mode", "tone",
                                   "window", "fit", "provenance"]
        @test d["schema"] == "toy/sweep/v2.0" && d["units"]["pixel"] == "opencv" && d["generator"]["name"] == "VisualGeometryToolkitIO"
        r = VgtkIO.read_kind(p, SWEEP)
        s = VgtkIO.VgioStyle(r)
        @test VgtkIO.pixel_frame(s) === PixelFrame{:opencv} && r.frames == 1 && r.tone.of.level == 0.5
        @test VgtkIO.camera_from_json(BoundedCamera, r.camera, VgtkIO.pixel_frame(s)) isa BoundedCamera
        @test JSON.lift(s, Trajectory, r.rotations) isa Trajectory
        back = VgtkIO.fit_record_from_json(r.fit)
        @test back isa FitResult{Nothing, Nothing} && VgtkIO.fit_record_json(back) == VgtkIO.fit_record_json(fit)
        # the writer checks the very bytes: a record its reader would refuse is not written, the refusal names the key
        q = joinpath(dir, "refused.json")
        refusal(f) = try f(); "" catch e; e isa ArgumentError ? e.msg : rethrow() end
        @test occursin("frames", refusal(() -> VgtkIO.write_kind(q, SWEEP, body(; frames = 0); style)))
        @test occursin("mode", refusal(() -> VgtkIO.write_kind(q, SWEEP, body(; mode = "burst"); style)))
        @test occursin("one row per frame", refusal(() -> VgtkIO.write_kind(q, SWEEP, body(; frames = 2); style)))
        @test occursin("gains[1]", refusal(() -> VgtkIO.write_kind(q, SWEEP, body(; gains = [[1.0, 1.0]]); style)))
        @test occursin("the toy's own message", refusal(() -> VgtkIO.write_kind(q, SWEEP, body(; tone = (type = "gamma",)); style)))
        @test occursin("window", refusal(() -> VgtkIO.write_kind(q, SWEEP, body(; window = (center = [1.0, 2.0], radius = "wide")); style)))
        @test occursin("unknown key", refusal(() -> VgtkIO.write_kind(q, SWEEP, merge(body(), (extra = 1,)); style)))
        @test occursin("envelope", refusal(() -> VgtkIO.write_kind(q, SWEEP, merge(body(), (units = 1,)); style)))
        @test !isfile(q)
        # the reader: a missing key, an unknown key at its own minor (tolerated at a higher minor), another schema
        edited(f) = VgtkIO.edited_copy(f, p, q)
        @test occursin("mode", refusal(() -> VgtkIO.read_kind(edited(t -> delete!(t, "mode")), SWEEP)))
        @test occursin("unknown key", refusal(() -> VgtkIO.read_kind(edited(t -> t["extra"] = 1), SWEEP)))
        @test VgtkIO.read_kind(edited(t -> (t["extra"] = 1; t["schema"] = "toy/sweep/v2.1")), SWEEP).extra == 1
        @test occursin("later major", refusal(() -> VgtkIO.read_kind(edited(t -> t["schema"] = "toy/sweep/v3.0"), SWEEP)))
        @test occursin("retired toy/sweep/v1", refusal(() -> VgtkIO.read_kind(edited(t -> t["schema"] = "toy/sweep/v1"), SWEEP)))   # no version step
        @test occursin("expected schema toy/sweep/v2.0", refusal(() -> VgtkIO.read_kind(edited(t -> t["schema"] = "vgio/trajectory/v5.0"), SWEEP)))
        @test occursin("payload", refusal(() -> VgtkIO.read_kind(edited(t -> t["payload"] = Dict("file" => "a.bin", "sha256" => "0"^64)), SWEEP)))
    end
    # the kind's own statement is checked where it is made
    @test_throws ArgumentError VgtkIO.KindSpec("toy/sweep/v2", [:frames => VgtkIO.integer_spec()]; package = "Toy")        # the minor stated
    @test_throws ArgumentError VgtkIO.KindSpec("Toy/Sweep/v2.0", [:frames => VgtkIO.integer_spec()]; package = "Toy")
    @test_throws ArgumentError VgtkIO.KindSpec("vgio/trajectory/v9.0", [:frames => VgtkIO.integer_spec()]; package = "Toy")  # this package's own
    @test_throws ArgumentError VgtkIO.tagged_spec(tone, "a tone"; def = "")
    # the JSON Schema is generated from the table: the owner's `$id`, the recursive body one definition, the check's keywords
    js = VgtkIO.json_schema(SWEEP; id_base = "https://example.org/schemas/")
    @test js["\$id"] == "https://example.org/schemas/toy/sweep/v2.0.json" && js["title"] == "toy/sweep/v2.0"
    @test occursin("as Toy writes and reads it", js["description"]) && occursin("one row of gains per frame", js["description"])
    @test js["properties"]["schema"]["pattern"] == "^toy/sweep/v2(\\.[0-9]+)?\$"
    @test js["properties"]["tone"] == Dict("\$ref" => "#/\$defs/tone") && length(js["\$defs"]["tone"]["oneOf"]) == 2
    @test js["\$defs"]["tone"]["oneOf"][2]["properties"]["of"] == Dict("\$ref" => "#/\$defs/tone")
    @test js["properties"]["mode"]["enum"] == ["still", "video"] && js["properties"]["frames"]["minimum"] == 1
    @test "fit" ∉ js["required"] && "tone" in js["required"] && js["additionalProperties"] == false
    @test VgtkIO.json_schema(SWEEP)["\$id"] == "https://visualgeometrytoolkit.github.io/schemas/toy/sweep/v2.0.json"
end

# A toy kind whose file is one value, read and written by the owner's methods of this package's two verbs, with one
# earlier version (v1 named the key `values` and declared its pixel frame at the top: IO 0.9's envelope).
@testsetup module ToyLevelsKind
    import VisualGeometryToolkitIO as VgtkIO
    export ToyLevels, LEVELS
    struct ToyLevels
        levels::Vector{Float64}
    end
    Base.:(==)(a::ToyLevels, b::ToyLevels) = a.levels == b.levels
    const LEVELS = VgtkIO.KindSpec("toylevels/levels/v2.0", [:levels => VgtkIO.array_spec(VgtkIO.real_spec())]; package = "ToyLevels")
    VgtkIO.save(path::AbstractString, x::ToyLevels; provenance::NamedTuple = (;)) = VgtkIO.write_kind(path, LEVELS, (levels = x.levels,); provenance)
    VgtkIO.load(::Type{ToyLevels}, path::AbstractString) = ToyLevels(Float64[Float64(v) for v in VgtkIO.read_kind(path, LEVELS).levels])
    function VgtkIO.migrate_schema(::Val{Symbol("toylevels/levels/v1")}, t::AbstractDict)
        n = VgtkIO.migrate_envelope(t, "toylevels/levels/v2.0")
        n["levels"] = pop!(n, "values")
        return n
    end
    VgtkIO.migrate_schema(::Val{Symbol("toylevels/levels/v1")}, src::AbstractString, dst::AbstractString) =
        VgtkIO.migrate_file(ToyLevels, LEVELS, src, dst; what = "the levels", length_unit = false)
end

@testitem "kind spec: an owner's version steps — read on load, rewritten by vgio_migrate through migrate_file" setup = [VGIoFixtures, ToyLevelsKind] begin
    import VisualGeometryToolkitIO as VgtkIO
    using VisualGeometryToolkitCore: Trajectory
    @test VgtkIO.schema_versions(LEVELS) == ["toylevels/levels/v1", "toylevels/levels/v2.0"]
    mktempdir() do dir
        old = joinpath(dir, "old.json")
        write(old, JSON.json(Dict("schema" => "toylevels/levels/v1", "pixel_frame" => "colmap", "values" => [0.25, 0.5],
                                  "provenance" => Dict("run" => "a"))))
        @test VgtkIO.load(ToyLevels, old) == ToyLevels([0.25, 0.5])                 # through its version step, in memory
        @test VgtkIO.read_kind(old, LEVELS).schema == "toylevels/levels/v2.0"
        out = joinpath(dir, "new.json")
        r = VgtkIO.vgio_migrate(old; out)
        @test r.from == "toylevels/levels/v1" && r.to == "toylevels/levels/v2.0"
        d = JSON.parse(read(out, String))
        @test d["schema"] == "toylevels/levels/v2.0" && d["levels"] == [0.25, 0.5] && d["provenance"] == Dict("run" => "a")
        @test VgtkIO.load(ToyLevels, out) == ToyLevels([0.25, 0.5])
        # a record the version step cannot carry is refused naming the file and its schema
        bad = joinpath(dir, "bad.json")
        write(bad, JSON.json(Dict("schema" => "toylevels/levels/v1", "pixel_frame" => "colmap")))
        @test_throws ArgumentError VgtkIO.load(ToyLevels, bad)
    end
    # a body IO 0.9 wrote, embedded in an owner's record: its 0.10 form
    cam9 = Dict{String, Any}("width" => 640, "height" => 480, "K" => Dict("fx" => 1.0), "lens" => Dict{String, Any}("type" => "pinhole"))
    cam10 = VgtkIO.migrate_body(BoundedCamera, cam9)
    @test cam10["image_size"] == Dict("width" => 640, "height" => 480) && !haskey(cam10, "width")
    @test VgtkIO.migrate_body(BoundedCamera, cam10) == cam10                        # already the 0.10 body
    tr9 = Dict{String, Any}("from" => "camera_optical", "to" => "world", "frames" => Any[Dict{String, Any}("t" => 1, "pose" => Dict{String, Any}())])
    tr10 = VgtkIO.migrate_body(Trajectory, tr9)
    @test tr10["from"] == "camera" && tr10["to"] == "world" && tr10["frames"][1]["frame_index"] == 1 && !haskey(tr10["frames"][1], "t")
end

@testitem "moved kinds: vignetting, environment_light, radial_profile and panorama are refused by name with their owner" setup = [VGIoFixtures] begin
    import VisualGeometryToolkitIO as VgtkIO
    using VisualGeometryToolkitCore: Trajectory
    using VisualGeometryToolkitIO: validate
    owners = ("vignetting" => "VisualRadiometryToolkitIO", "environment_light" => "VisualRadiometryToolkitIO",
              "radial_profile" => "VisualRadiometryToolkitIO", "panorama" => "PanoramaStitching")
    message(f) = try f(); "" catch e; e isa ArgumentError ? e.msg : rethrow() end
    named(msg, kind, owner) = occursin("the $(kind) kind", msg) && occursin("$(owner)'s", msg)
    mktempdir() do dir
        for (kind, owner) in owners, version in ("v1", "v2.0")
            p = joinpath(dir, "$(kind).json")
            write(p, JSON.json(Dict("schema" => "vgio/$(kind)/$(version)")))
            @test named(message(() -> validate(p)), kind, owner)                      # the validator
            @test named(message(() -> VgtkIO.vgio_migrate(p; out = joinpath(dir, "out.json"))), kind, owner)   # the migrator
            @test named(message(() -> load(Trajectory, p)), kind, owner)             # a reader of another kind
            @test !isfile(joinpath(dir, "out.json"))
        end
        # a manifest binds a moved kind's file by its hash, as a file of an owner that is not loaded
        t = joinpath(dir, "bound"); mkpath(t)
        write(joinpath(t, "vignetting.json"), JSON.json(Dict("schema" => "vgio/vignetting/v2.0")))
        VgtkIO.save_manifest(t, ["vignetting.json"]; result_kind = "image")
        @test VgtkIO.load_manifest(t) !== nothing
    end
    for (kind, owner) in owners
        @test named(message(() -> VgtkIO.json_schema(kind)), kind, owner)             # by kind name
        @test named(message(() -> VgtkIO.schema_versions(kind)), kind, owner)
        @test !haskey(VgtkIO._SCHEMA_VERSION, kind)
        @test all(r -> r.kind != kind, VgtkIO.schema_versions())
    end
    # the panorama's two functions had names of their own: each refused by name, neither public
    @test named(message(() -> VgtkIO.save_panorama("p.json")), "panorama", "PanoramaStitching")
    @test named(message(() -> VgtkIO.load_panorama("p.json")), "panorama", "PanoramaStitching")
    @test !Base.ispublic(VgtkIO, :save_panorama) && !Base.ispublic(VgtkIO, :load_panorama)
    # an unknown kind is still an unknown kind
    @test occursin("no vgio kind", message(() -> VgtkIO.json_schema("vignette")))
end
