# =============================================================================
# IO 0.10: every kind's IO 0.9 version read through its step and rewritten by
# vgio_migrate; the frozen IO 0.9 identity; the one re-id, and a tree's
# observation files re-pointed through the persisted identity map
# =============================================================================

@testsetup module V010Fixtures
    import VisualGeometryToolkitIO as VgtkIO
    using JSON: JSON
    export old09, tree09, write09, strip_time, ID09

    old09(name) = joinpath(@__DIR__, "fixtures", "old", name)
    tree09(p) = JSON.parse(read(p, String); dicttype = Dict{String, Any})
    write09(p, t) = (write(p, JSON.json(t, 4)); p)
    # a 0.9 capture file without its zone-less time_origin (and its frames' times),
    # which the 0.10 step refuses (`_time_origin_v010`)
    function strip_time(src, dir)
        t = tree09(src)
        delete!(t, "time_origin")
        foreach(f -> delete!(f, "time"), t["frames"])
        return write09(joinpath(dir, basename(src)), t)
    end
    # the identities IO 0.9 pinned (its target_record_tests.jl), of the fixtures copied here
    const ID09 = Dict(
        "rig_v6_every_kind.json" => "f34bee5347b45f2a9d6bca72741e8d2bf47dd2fa7e1b3d12ddb237694b8ef865",
        "rig_v9_targets.json" => "980c47d473610daa038b633cf5d4d133931699a9cafe1ee210f507b9b1c793b3",
        "rig_v9.json" => "5cdb12271e474c04c68ac5ea539f509a720449fcf42b79d8cc4a506cddcf9fe0",
    )
end

@testitem "migrate v0.10: the frozen IO 0.9 identity reproduces the ids IO 0.9 pinned" setup = [VGIoFixtures, V010Fixtures] begin
    import VisualGeometryToolkitIO as VgtkIO
    using VisualGeometryToolkitCore: Rig
    for (name, id) in ID09
        p = old09(name)
        rig = load(Rig, p)
        @test VgtkIO._v09_id(rig) == id
        # the re-id: the 0.10 identity differs, and the file's migration maps the 0.9 one to it
        @test content_id(rig) != id
        s = VgtkIO._peek_schema(p)
        @test id in VgtkIO._old_rig_ids(s, VgtkIO._raw_tree(p), rig)
        @test (id => content_id(rig)) in VgtkIO._old_id_map(p)
    end
    # 0.9's canonical number printing, not JCS
    @test VgtkIO._canonical_json_v09 isa Function
    @test sprint(VgtkIO._canonical_json_v09, Dict("b" => 1.0e-5, "a" => 2.0)) == "{\"a\":2,\"b\":1.0e-5}"
    @test VgtkIO.canonical_json(Dict("b" => 1.0e-5, "a" => 2.0)) == "{\"a\":2,\"b\":0.00001}"
end

@testitem "migrate v0.10: every IO 0.9 file reads through its step and vgio_migrate rewrites it" setup = [VGIoFixtures, V010Fixtures] begin
    import VisualGeometryToolkitIO as VgtkIO
    using VisualGeometryToolkitCore: Rig, AbstractTarget, Trajectory, TargetPoses, TargetObservations, CaptureSettings,
        AbstractVignetting, EnvironmentLight
    files = [
        "rig_v9.json" => Rig, "rig_v9_pitch.json" => Rig, "rig_v9_targets.json" => Rig, "rig_v6_every_kind.json" => Rig,
        "target_v5_planar.json" => AbstractTarget, "target_v5_points.json" => AbstractTarget,
        "transforms_v4.json" => Vector{EuclideanMap}, "trajectory_v4.json" => Trajectory, "target_poses_v2.json" => TargetPoses,
        "target_detections_v2.json" => TargetObservations,
        "capture_settings_v1.json" => CaptureSettings, "cameras_v5_spline_masked.json" => Dict{Symbol, BoundedCamera},
        "noncentral_spline_v1.json" => Dict{Symbol, BoundedCamera},
        "vignetting_v1_linear.json" => AbstractVignetting, "vignetting_v1_chart.json" => AbstractVignetting,
        "environment_light_v1.json" => EnvironmentLight,
    ]
    mktempdir() do dir
        for (name, T) in files
            src = old09(name)
            haskey(tree09(src), "time_origin") && (src = strip_time(src, dir))
            value = load(T, src)
            out = joinpath(dir, "migrated_" * name)
            r = vgio_migrate(src; out)
            @test r.to == VgtkIO._peek_schema(out)
            @test VgtkIO._is_current(r.to)
            @test VgtkIO.same_value(load(T, out), value)
            VgtkIO.validate(out)
        end
    end
end

@testitem "migrate v0.10: a time_origin without its UTC offset is refused by name" setup = [VGIoFixtures, V010Fixtures] begin
    using VisualGeometryToolkitCore: Trajectory, TargetPoses
    for (name, T) in ("trajectory_v4.json" => Trajectory, "target_poses_v2.json" => TargetPoses)
        @test_throws r"time_origin .* states no UTC offset" load(T, old09(name))
    end
    mktempdir() do dir
        t = tree09(old09("trajectory_v4.json"))
        t["time_origin"] = "2026-05-27T00:00:00+02:00"      # a hand-edited offset is carried over
        p = write09(joinpath(dir, "t.json"), t)
        @test load(Trajectory, p).time_origin == "2026-05-27T00:00:00+02:00"
    end
end

@testitem "migrate v0.10: the observation kinds fold into observations v3.0" setup = [VGIoFixtures, V010Fixtures] begin
    import VisualGeometryToolkitIO as VgtkIO
    using VisualGeometryToolkitCore: TargetObservations, Uncertain, sources, targets, geom
    # detections: each set's per-point covariances onto its points, its weights column
    d = load(TargetObservations, old09("target_detections_v2.json"))
    cs = d[1][:a1f2]
    @test collect(sources(cs)) == [2, 4] && cs.attrs.weights == [1.0, 0.5]
    @test all(p -> p isa Uncertain, targets(cs))
    @test d.targets == ["980c47d473610daa038b633cf5d4d133931699a9cafe1ee210f507b9b1c793b3"]
    t = VgtkIO.migrate_record(tree09(old09("target_detections_v2.json")))
    @test t["schema"] == "vgio/observations/v3.0" && t["entity"] == "target_point"
    @test t["frames"][2]["observations"][1]["points"][1] == Dict("xy" => [1.5, 2.5], "covariance" => [[0.1, 0.0], [0.0, 0.2]])
    # points2d: a point's weight is its set's weights column; one point type per file
    p2 = tree09(old09("target_points2d_v2.json"))
    @test_throws r"plain and Uncertain points: frame 1 target a1f2 index 2 is plain, frame 3 target b0c1 index 1 states a covariance" VgtkIO.migrate_record(p2)
    for f in p2["frames"], p in f["points"]
        delete!(p, "covariance")
    end
    t2 = VgtkIO.migrate_record(p2)
    @test t2["frames"][2]["observations"] == Any[
        Dict("target" => "a1f2", "index" => [4], "points" => [Dict("xy" => [100.0, 50.0])], "weights" => [1.0]),
        Dict("target" => "b0c1", "index" => [1], "points" => [Dict("xy" => [1.5, 2.5])], "weights" => [0.5]),
    ]
    @test t2["units"]["pixel"] == "julia" && t2["conventions"]["index_base"] == 1
    # a (target, index) observed twice in a frame is refused naming it
    push!(p2["frames"][2]["points"], Dict{String, Any}("target" => "b0c1", "index" => 1, "xy" => [3.0, 4.0], "weight" => 1.0))
    @test_throws r"frame 3 observes point 1 of target b0c1 twice" VgtkIO.migrate_record(p2)
    # 3D points: the frame name, no weights; this file mixes plain and Uncertain points (refused), its plain form reads
    mktempdir() do dir
        p3 = strip_time(old09("target_points3d_v2.json"), dir)
        @test_throws r"plain and Uncertain points" load(TargetObservations, p3)
        t3 = tree09(p3)
        for f in t3["frames"], p in f["points"]
            delete!(p, "covariance")
        end
        write09(p3, t3)
        o3 = load(TargetObservations, p3)
        @test isempty(o3.targets)
        @test VgtkIO.migrate_record(t3)["frame"] == t3["frame"]
        r = vgio_migrate(p3; out = joinpath(dir, "o3.json"))
        @test r.to == "vgio/observations/v3.0" && VgtkIO.same_value(load(TargetObservations, joinpath(dir, "o3.json")), o3)
    end
end

@testitem "migrate v0.10: vignetting v1 with and without its chart (the review's I1)" setup = [VGIoFixtures, V010Fixtures] begin
    import VisualGeometryToolkitIO as VgtkIO
    using VisualGeometryToolkitCore: AbstractVignetting, RadialVignetting, LogVignetting, radial_chart, log_spline
    v = load(AbstractVignetting, old09("vignetting_v1_linear.json"))
    @test v isa RadialVignetting && radial_chart(v) === :linear
    w = load(AbstractVignetting, old09("vignetting_v1_chart.json"))
    @test w isa LogVignetting && radial_chart(log_spline(w)) === :projective
    t = VgtkIO.migrate_record(tree09(old09("vignetting_v1_linear.json")))
    @test t["vignetting"]["chart"] == "linear" && t["units"]["pixel"] == "colmap"
end

@testitem "migrate v0.10: environment light, capture settings, units sidecar, manifest" setup = [VGIoFixtures, V010Fixtures] begin
    import VisualGeometryToolkitIO as VgtkIO
    using VisualGeometryToolkitCore: EnvironmentLight, CaptureSettings, max_degree
    using VisualGeometryToolkitIO: UnitContract
    @test max_degree(load(EnvironmentLight, old09("environment_light_v1.json"))) == 1
    t = VgtkIO.migrate_record(tree09(old09("environment_light_v1.json")))
    @test t["map"]["max_degree"] == 1 && !haskey(t["map"], "l_max") && t["units"]["length"] == "mm"
    # capture settings: the one device-identity encoding, the sources, the EXIF claim
    c = VgtkIO.migrate_record(tree09(old09("capture_settings_v1.json")))
    @test c["device"] == Dict("body" => Dict("make" => "Sony", "model" => "ILME-FX3"),
                              "lens" => Dict("make" => "SIGMA", "model" => "14mm F1.8 DG HSM | Art"))
    @test c["identity_sources"] == Dict("body" => "exif", "lens" => "config") && c["exif_lens"] == Dict("model" => "DT 14mm F2.8 SAM")
    @test [f["frame_index"] for f in c["frames"]] == [1, 4]
    cs = load(CaptureSettings, old09("capture_settings_v1.json"))
    @test cs.identity_sources == (body = "exif", lens = "config")
    # a units sidecar: world → units.length, density dpi as v2 read it; "unknown" kept
    @test VgtkIO.load_units_sidecar(old09("units_v2")) == UnitContract(mm, VisualGeometryToolkitCore.dpi)
    @test VgtkIO.load_units_sidecar(old09("units_v2_unknown")) isa VgtkIO.UnknownScale
    mktempdir() do dir
        d = mkpath(joinpath(dir, "model"))
        cp(old09("units_v2/vgio_units.json"), joinpath(d, "vgio_units.json"))
        vgio_migrate(joinpath(d, "vgio_units.json"); out = joinpath(d, "vgio_units.json"))
        @test VgtkIO._peek_schema(joinpath(d, "vgio_units.json")) == "vgio/units/v3.0"
        @test VgtkIO.load_units_sidecar(d) == UnitContract(mm, VisualGeometryToolkitCore.dpi)
    end
    # the manifest: result_kind, a file's schema, a non-numeric metric under provenance.metrics
    mktempdir() do dir
        cp(old09("target_v5_planar.json"), joinpath(dir, "target.json"))
        m = Dict{String, Any}(
            "schema" => "vgio/calibration/v2", "kind" => "metric",
            "files" => Any[Dict{String, Any}("path" => "target.json", "sha256" => open(VisualGeometryToolkitCore.sha256_hex, joinpath(dir, "target.json")),
                                             "kind" => "vgio/target/v5", "role" => "authority", "derived_from" => Any[])],
            "metrics" => Dict{String, Any}("reprojection_rms" => 0.21, "solver" => "lm", "per_view" => Any[0.1, 0.3]),
        )
        p = write09(joinpath(dir, "calibration.json"), m)
        t = VgtkIO.migrate_record(tree09(p))
        @test t["result_kind"] == "metric" && only(t["files"])["schema"] == "vgio/target/v5"
        @test t["metrics"] == Dict("reprojection_rms" => 0.21)
        @test t["provenance"]["metrics"] == Dict("solver" => "lm", "per_view" => [0.1, 0.3])
        r = vgio_migrate(p; out = joinpath(dir, "calibration.migrated.json"))
        @test r.to == "vgio/calibration/v3.0"
        @test VgtkIO.validate(joinpath(dir, "calibration.migrated.json")) == "calibration"
    end
end

@testitem "migrate v0.10: a tree's rigs re-identified, its observations re-pointed through the persisted map" setup = [VGIoFixtures, V010Fixtures] begin
    import VisualGeometryToolkitIO as VgtkIO
    using VisualGeometryToolkitCore: Rig, TargetObservations
    mktempdir() do dir
        cp(old09("rig_v9_targets.json"), joinpath(dir, "targets.json"))
        p2 = tree09(old09("target_points2d_v2.json"))
        for f in p2["frames"], p in f["points"]
            delete!(p, "covariance")
        end
        mkpath(joinpath(dir, "capture"))
        write09(joinpath(dir, "capture", "points.json"), p2)
        cp(old09("target_detections_v2.json"), joinpath(dir, "capture", "detections.json"))
        log = IOBuffer()
        rs = vgio_migrate_tree(dir; in_place = true, io = log)
        new_id = content_id(joinpath(dir, "targets.json"))
        @test new_id != ID09["rig_v9_targets.json"]
        ids, defs = VgtkIO._read_ids(joinpath(dir, "vgio_migrate_ids.json"))
        @test ids[ID09["rig_v9_targets.json"]] == new_id && defs[new_id] == "targets.json"
        for f in ("points.json", "detections.json")
            p = joinpath(dir, "capture", f)
            @test VgtkIO._peek_schema(p) == "vgio/observations/v3.0"
            @test load(TargetObservations, p).targets == [new_id]
        end
        @test count(r -> haskey(r, :repointed), rs) == 2
        # idempotent: a rerun finds every file current
        log2 = IOBuffer()
        @test isempty(vgio_migrate_tree(dir; in_place = true, io = log2))
        @test all(l -> startswith(l, "current"), split(strip(String(take!(log2))), '\n'))
        # an observation file kept outside the tree: re-pointed later from the persisted map (the command's `repoint`)
        mktempdir() do out
            q = cp(old09("target_detections_v2.json"), joinpath(out, "late.json"))
            buf = IOBuffer()
            @test VgtkIO._migrate_cli(["repoint", "--ids", joinpath(dir, "vgio_migrate_ids.json"), "--in-place", "--apply", q]; io = buf) == 0
            @test load(TargetObservations, q).targets == [new_id]
        end
    end
end

@testitem "migrate v0.10: a tree's manifest binds the files the migration rewrote, and loads" setup = [VGIoFixtures, V010Fixtures] begin
    import VisualGeometryToolkitIO as VgtkIO
    using VisualGeometryToolkitCore: sha256_hex
    mktempdir() do dir
        cp(old09("rig_v9_targets.json"), joinpath(dir, "targets.json"))
        cp(old09("target_points2d_v2_plain.json"), joinpath(dir, "points.json"))
        cp(old09("cameras_v4.json"), joinpath(dir, "camera.json"))
        cp(old09("generic_camera_v2.json"), joinpath(dir, "grid.json"))
        entry(p; role = "authority", from = Any[]) = Dict{String, Any}("path" => p, "sha256" => open(sha256_hex, joinpath(dir, p)),
            "kind" => VgtkIO._peek_schema(joinpath(dir, p)), "role" => role, "derived_from" => from)
        m = Dict{String, Any}("schema" => "vgio/calibration/v2", "kind" => "metric",
            "files" => Any[entry("camera.json"), entry("grid.json"), entry("targets.json"), entry("points.json"; role = "derived", from = Any["grid.json", "targets.json"])],
            "metrics" => Dict{String, Any}("reprojection_rms" => 0.21))
        write09(joinpath(dir, "calibration.json"), m)
        # a dry run says what it would rebind and writes nothing
        log = IOBuffer()
        vgio_migrate_tree(dir; in_place = true, dry_run = true, io = log)
        @test occursin("would rebind  $(joinpath(dir, "calibration.json"))", String(take!(log)))
        @test VgtkIO._peek_schema(joinpath(dir, "calibration.json")) == "vgio/calibration/v2"
        rs = vgio_migrate_tree(dir; in_place = true, io = log)
        r = only(x for x in rs if haskey(x, :rebound))
        @test r.rebound == 4 && r.to == "vgio/calibration/v3.0"
        man = VgtkIO.load_manifest(dir)                                   # every hash, schema and target checked
        byname = Dict(f.path => f for f in man.files)
        @test sort!(collect(keys(byname))) == ["camera.json", "grid.camera.json", "points.json", "targets.json"]   # the old grid's new store
        @test byname["points.json"].derived_from == ["grid.camera.json", "targets.json"]
        @test all(f -> VgtkIO._is_current(f.schema), man.files) && byname["points.json"].schema == "vgio/observations/v3.0"
        @test man.result_kind == "metric"
        @test haskey(VgtkIO._read_rewritten(joinpath(dir, "vgio_migrate_ids.json")), "grid.json")
        # a rerun: every file current, the manifest left as it is
        before = read(joinpath(dir, "calibration.json"), String)
        @test isempty(vgio_migrate_tree(dir; in_place = true, io = log))
        @test occursin("binds no rewritten file", String(take!(log))) && read(joinpath(dir, "calibration.json"), String) == before
    end
end
