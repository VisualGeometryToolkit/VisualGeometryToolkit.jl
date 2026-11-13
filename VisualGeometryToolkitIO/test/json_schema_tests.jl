# =============================================================================
# The published JSON Schemas (json_schema.jl) and the content identity, checked outside Julia
# =============================================================================
#
# - The committed schemas (docs/src/schemas/vgio/<kind>/v<M>.<m>.json) are the
#   generator's output, byte for byte.
# - Every valid document — every fixture the validator reads, as its current
#   record, and a written file of every kind no fixture holds — validates
#   against its kind's schema in Python's `jsonschema` (draft 2020-12); every
#   deletion case the validator refuses (each required key deleted in turn,
#   VGIoSchemaCases.required_paths) and an unknown key, at the top and nested,
#   is refused by the schema too. JSONSchema.jl implements drafts 4–7 (no
#   `prefixItems`, no `$defs`), so the documents are validated in Python.
# - A content id is recomputed from a stored rig and target file by an
#   independent implementation of the "Content identity" specification
#   (test/python/content_id.py: its own JCS writer).
#
# The Python items run when the pixi `py` environment is installed
# (`pixi install -e py`, which brings jsonschema) and are recorded as SKIPPED
# otherwise, never silently absent (test/pycolmap_tests.jl's pattern).

@testitem "json schemas: the committed files are the generator's" setup = [VGIoFixtures, VGIoSchemaCases] begin
    import VisualGeometryToolkitIO as VgtkIO
    mktempdir() do dir
        VgtkIO._write_json_schemas(dir)
        made = sort!([relpath(joinpath(r, f), dir) for (r, _, fs) in walkdir(dir) for f in fs])
        committed = isdir(schema_dir()) ? sort!([relpath(joinpath(r, f), schema_dir()) for (r, _, fs) in walkdir(schema_dir()) for f in fs]) : String[]
        # (regenerate: VisualGeometryToolkitIO._write_json_schemas("VisualGeometryToolkitIO/docs/src/schemas"),
        # after deleting the directory so a retired version's file goes too)
        @test committed == made
        for f in made
            @test isfile(joinpath(schema_dir(), f)) && read(joinpath(schema_dir(), f), String) == read(joinpath(dir, f), String)
        end
        @test length(made) == length(VgtkIO._SCHEMA_VERSION)
    end
    # each kind's schema states its own identity and dialect
    for kind in keys(VgtkIO._SCHEMA_VERSION)
        s = VgtkIO.json_schema(kind)
        @test s["\$schema"] == "https://json-schema.org/draft/2020-12/schema"
        @test s["\$id"] == "https://visualgeometrytoolkit.github.io/schemas/$(VgtkIO._schema(kind)).json"
        @test s["title"] == VgtkIO._schema(kind)
    end
    @test_throws ArgumentError VgtkIO.json_schema("no_such_kind")
    # a predicate that IS a set is published as one, the set asked of the code that holds it
    s = VgtkIO.json_schema("transforms")
    @test s["properties"]["units"]["properties"]["pixel"]["enum"] == ["colmap", "julia", "opencv"]
    @test "mm" in s["properties"]["units"]["properties"]["length"]["enum"] && !("px" in s["properties"]["units"]["properties"]["length"]["enum"])
    @test s["properties"]["units"]["properties"]["density"]["enum"] == ["dpi"]
    @test haskey(s["properties"]["transforms"]["items"]["properties"]["from"], "pattern")
    q = s["properties"]["transforms"]["items"]["properties"]["rotation"]
    @test [p["title"] for p in q["prefixItems"]] == ["w", "x", "y", "z"]   # a positional primitive, field by field
end

@testitem "json schemas: every valid document validates, every deletion case is refused (Python jsonschema)" setup = [VGIoFixtures, VGIoSchemaCases] begin
    import VisualGeometryToolkitIO as VgtkIO
    import VisualGeometryToolkitCore as VgtkCore
    using VisualGeometryToolkitIO: validate, read_record
    using VisualGeometryToolkitCore: TensorProductVignetting, RadialVignetting, LogVignetting, with_control_points,
        EnvironmentLight, SphericalHarmonicMap, with_unit, GaussianBlur, Rect, TargetHomographies, HomographyMap,
        HomographyMat, sign_normalize, TargetPlaneFrame, relabel, PointTarget
    using StaticArrays: SDiagonal
    using Statistics: cov
    python = pixi_python("jsonschema")
    if python === nothing
        @test_skip "jsonschema is not installed in the pixi py environment (pixi install -e py)"
    else
        mktempdir() do dir
            # THE VALID DOCUMENTS: every fixture the validator reads (as the current record its version
            # steps give), and a written file of the kinds no fixture holds
            docs = Pair{String, Tuple{String, Dict{String, Any}}}[]   # name => (kind, current record)
            function collect_doc!(name, path)
                kind = try
                    validate(path)
                catch e
                    e isa ArgumentError || rethrow()
                    println(stderr, "json schemas: $(name) is not a readable vgio file, left out: ", first(e.msg, 300))
                    return nothing
                end
                file = isdir(path) ? joinpath(path, VgtkIO.UNITS_SIDECAR) : path
                push!(docs, name => (kind, plain_doc(read_record(file))))
                return nothing
            end
            fixtures = joinpath(@__DIR__, "fixtures")
            for (r, _, fs) in walkdir(fixtures), f in fs
                endswith(f, ".json") && collect_doc!(relpath(joinpath(r, f), fixtures), joinpath(r, f))
            end
            # the kinds the fixtures do not hold, written by their writers (each on its own: one failing
            # writer is named, and the coverage test below says which kind it leaves out)
            failed = String[]                                  # writers that failed (shown by the test below)
            function written(f, name)
                try
                    f(joinpath(dir, name))
                catch e
                    push!(failed, "$(name): $(first(sprint(showerror, e), 400))")
                    return nothing
                end
                return collect_doc!("written/$(name)", joinpath(dir, name))
            end
            w(i, j) = 0.5 + 0.0371 * i + 0.0193 * j * j
            tv = with_control_points(TensorProductVignetting((5, 4), Rect(0.0, 0.0, 640.0, 480.0); degree = (2, 3)), [w(i, j) for i in 1:5, j in 1:4])
            rv = with_control_points(RadialVignetting(7, SVector(320.3, 241.7), 401.25), [w(i, 1) for i in 1:7])
            written(p -> save(p, tv), "vignetting_tensor.json")
            written(p -> save(p, LogVignetting(rv)), "vignetting_log.json")
            written(p -> save(p, with_unit(EnvironmentLight(SphericalHarmonicMap([w(i, j) for i in 1:3, j in 1:9], 2), SVector(0.1, -0.2, 0.3), 1.5), m)),
                    "environment_light.json")
            written("blur.json") do p
                camera = GaussianBlur(0.6, 0.45)
                save(p, Dict(:camera => GaussianBlur(relabel(PixelFrame{:colmap}, cov(camera))),
                             :print => with_unit(GaussianBlur(relabel(TargetPlaneFrame, cov(GaussianBlur(SDiagonal(7.84e-10, 1.8225e-10))))), m)))
            end
            written(p -> (mkpath(p); VgtkIO.save_units_sidecar(p, MM_PX)), "units_sidecar")
            written(p -> save(p, PointTarget([mm_lengths(SVector(1.0, 2.0, 3.0)), mm_lengths(SVector(4.0, 5.0, 6.5))])), "target.json")
            written("target_homographies.json") do p
                H = HomographyMap(sign_normalize(HomographyMat{Float64}(Tuple(SMatrix{3, 3, Float64}([800.0 0 300; 0 800 200; 0 0 1]')))))
                tid = VgtkIO.content_id(joinpath(dir, "target.json"))
                save(p, TargetHomographies{PixelFrame{:colmap}}(Dict(1 => Dict(Symbol(tid) => H)); images = Dict(1 => "a.png"), sensors = "cam", targets = [tid]))
            end
            written("manifest/calibration.json") do p
                d = mkpath(dirname(p))
                cp(joinpath(dir, "target.json"), joinpath(d, "target.json"))
                VgtkIO.save_manifest(d, ["target.json"]; result_kind = "image")
            end
            @test failed == String[]
            # every kind has a valid document
            @test sort!(unique(first.(last.(docs)))) == sort!(collect(keys(VgtkIO._SCHEMA_VERSION)))

            # THE CASES: each valid document (must validate), each deletion the validator refuses and an
            # unknown key at the top and in the first nested object (each must be refused)
            cases = Dict{String, Any}[]
            expect = Dict{String, Bool}()
            refused(kind, doc) = try
                VgtkIO._validate(VgtkIO._spec(kind), doc, "case", "")
                false
            catch e
                e isa ArgumentError || rethrow()
                true
            end
            add!(name, kind, doc, valid) = (push!(cases, Dict("name" => name, "schema_id" => VgtkIO.json_schema(kind)["\$id"], "doc" => doc));
                                            expect[name] = valid)
            n_deleted = 0
            for (name, (kind, doc)) in docs
                @test !refused(kind, doc)
                add!(name, kind, doc, true)
                paths = required_paths(VgtkIO._spec(kind), doc)
                @test length(paths) >= 4            # the envelope's keys at least
                for p in paths
                    bad = without(doc, p)
                    @test refused(kind, bad)        # the validator's verdict, which the schema must share
                    add!("$(name) without $(join(p, '.'))", kind, bad, false)
                    n_deleted += 1
                end
                add!("$(name) with an unknown key", kind, merge(doc, Dict("bogus" => 1)), false)
                deep = [p for p in paths if length(p) >= 2 && getpath(doc, p[1:(end - 1)]) isa AbstractDict]
                isempty(deep) || add!("$(name) with an unknown nested key", kind, with_key(doc, [first(deep)[1:(end - 1)]; "bogus"], 1), false)
            end
            @test n_deleted > 100
            cases_file = joinpath(dir, "cases.json")
            write(cases_file, JSON.json(cases))
            out = IOBuffer()
            ok = success(pipeline(`$python $(joinpath(@__DIR__, "python", "check_json_schemas.py")) $(schema_dir()) $cases_file`; stdout = out))
            @test ok
            if ok
                for r in JSON.parse(String(take!(out)))
                    agrees = r["valid"] == expect[r["name"]]
                    agrees || println(stderr, "json schemas: the schema and the validator disagree on ", r["name"], ": schema valid = ", r["valid"], " (", r["error"], ")")
                    @test agrees
                end
            end
        end
    end
end

@testitem "content identity: recomputed in Python from a stored rig and target file" setup = [VGIoFixtures, VGIoSchemaCases] begin
    import VisualGeometryToolkitIO as VgtkIO
    using VisualGeometryToolkitIO: content_id
    using VisualGeometryToolkitCore: Rig, AbstractTarget
    python = joinpath(pkgdir(VgtkIO), ".pixi", "envs", "py", "bin", "python")
    if !isfile(python)
        @test_skip "the pixi py environment is not installed (pixi install -e py)"
    else
        script = joinpath(@__DIR__, "python", "content_id.py")
        python_id(path) = strip(read(`$python $script $path`, String))
        fixtures = joinpath(@__DIR__, "fixtures")
        mktempdir() do dir
            # every target kind (the identity fixture: thin-plate and height-field values with 17 digits,
            # compositions, covariances, boards) and the sensor rigs and single targets the fixtures hold, each
            # stored as the current record in decimal units, and its pixels in another frame
            files = String[]
            for (f, T) in (("identity/rig_every_kind.json", Rig), ("units/rig.json", Rig), ("units/rig_pitch.json", Rig),
                           ("units/rig_targets.json", Rig), ("units/spline_camera_masked.json", Rig),
                           ("units/target_planar.json", AbstractTarget), ("units/target_points.json", AbstractTarget))
                src = joinpath(fixtures, f)
                x = try
                    load(T, src)
                catch e
                    e isa ArgumentError || rethrow()
                    println(stderr, "content identity: $(f) does not read, left out: ", first(e.msg, 300))
                    continue
                end
                for u in (mm, m)
                    p = joinpath(dir, "$(replace(f, '/' => '_'))_$(u).json")
                    save(p, x; length_unit = u)
                    push!(files, p)
                end
            end
            @test length(files) >= 4
            for p in files
                id = content_id(p)
                @test python_id(p) == id
            end
        end
    end
end
