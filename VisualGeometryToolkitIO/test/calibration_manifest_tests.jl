# =============================================================================
# vgio/calibration/v3.0 — the manifest binding a result's record files (any owner's) by SHA-256
# =============================================================================

@testsetup module ManifestFixtures
    using JSON: JSON
    using VisualGeometryToolkitCore: TargetHomographies, HomographyMap, HomographyMat, PixelFrame, sign_normalize
    using StaticArrays: SMatrix
    using VisualGeometryToolkitIO: load, save, content_id

    export fixture, metric_dir, projective_dir, rewrite, retargeted

    retargeted(th, targets) = TargetHomographies{PixelFrame{:colmap}}(th.frames; images = th.images, sensors = th.sensors, targets)

    fixture(name) = joinpath(@__DIR__, "fixtures", "units", name)

    # a metric result: the targets rig, the trajectory (authority), the observations, and target_poses
    # derived from the trajectory and the rig
    function metric_dir(dir)
        for f in ("rig_targets.json", "trajectory.json", "target_poses.json", "observations2d.json")
            cp(fixture(f), joinpath(dir, f))
        end
        return [
            "rig_targets.json", "trajectory.json", "observations2d.json",
            (path = "target_poses.json", role = "derived", derived_from = ["trajectory.json", "rig_targets.json"]),
        ]
    end

    # a projective result: one single target and the homographies against it
    function projective_dir(dir; name = nothing, reference = nothing)
        if reference === nothing
            cp(fixture("target_planar.json"), joinpath(dir, "target_planar.json"); force = true)
        else   # the planar target as a referenced design
            rewrite(joinpath(dir, "target_planar.json"), d -> (delete!(d["target"], "nominal"); d["target"]["reference"] = reference); src = fixture("target_planar.json"))
        end
        name === nothing && (name = content_id(joinpath(dir, "target_planar.json")))
        H = HomographyMap(sign_normalize(HomographyMat{Float64}(Tuple(SMatrix{3, 3, Float64}([800.0 0 300; 0 800 200; 0 0 1]')))))
        th = TargetHomographies{PixelFrame{:colmap}}(
            Dict(1 => Dict(Symbol(name) => H)); images = Dict(1 => "a.png"), sensors = "cam", targets = [content_id(joinpath(dir, "target_planar.json"))],
        )
        save(joinpath(dir, "target_homographies.json"), th)
        return ["target_planar.json", "target_homographies.json"]
    end

    rewrite(path, f; src = path) = (d = JSON.parse(read(src, String)); f(d); write(path, JSON.json(d)); path)

    # a reader of another owner's records, as an owner declares one (IO's extension point `validate_record`)
    import VisualGeometryToolkitIO as VgtkIO
    VgtkIO.validate_record(::Val{:manifesttestowner}, path::String, ::String) =
        (VgtkIO.read_record(path).value > 0 || throw(ArgumentError("$(path): value must be positive")); nothing)
end

@testitem "calibration/v3.0: a metric manifest round trips, hashes and roles intact" setup = [ManifestFixtures] begin
    import VisualGeometryToolkitIO as VgtkIO
    import VisualGeometryToolkitCore as VgtkCore
    using JSON: JSON
    mktempdir() do dir
        files = metric_dir(dir)
        model = VgtkCore.CameraModelSpec(Dict("lens" => "division", "coefficients" => 3, "intrinsics" => "metric"))
        p = VgtkIO.save_manifest(dir, files; result_kind = "metric", model, metrics = (rms_px = 0.4, n_frames = 3), provenance = (; tool = "test"))
        @test p == joinpath(dir, "calibration.json")
        @test VgtkIO.validate(p) == "calibration"
        m = VgtkIO.load_manifest(dir)
        @test m.result_kind == "metric" && m.model == model && m.dir == dir
        @test [f.path for f in m.files] == ["rig_targets.json", "trajectory.json", "observations2d.json", "target_poses.json"]
        poses = m.files[4]
        @test poses.role == "derived" && poses.derived_from == ["trajectory.json", "rig_targets.json"] && poses.schema == "vgio/target_poses/v3.0"
        @test [f.schema for f in m.files[1:3]] == ["vgio/rig/v10.0", "vgio/trajectory/v5.0", "vgio/observations/v3.0"]
        d = JSON.parsefile(p)
        @test d["schema"] == "vgio/calibration/v3.0" && d["result_kind"] == "metric" && !haskey(d, "kind")
        @test haskey(d, "generator") && haskey(d, "units") && haskey(d, "conventions") && all(f -> !haskey(f, "kind"), d["files"])
        @test m.files[2].role == "authority" && isempty(m.files[2].derived_from)
        @test all(f.sha256 == VgtkCore.sha256_hex(read(joinpath(dir, f.path), String)) for f in m.files)
        @test m.metrics["rms_px"] == 0.4 && m.provenance["tool"] == "test"
        # written again, the same bytes
        before = read(p, String)
        VgtkIO.save_manifest(dir, files; result_kind = "metric", model, metrics = (rms_px = 0.4, n_frames = 3), provenance = (; tool = "test"))
        @test read(p, String) == before
        # optional parts left out read back as nothing
        VgtkIO.save_manifest(dir, files[[1, 3]]; result_kind = "image")   # no placement in space: no trajectory
        m2 = VgtkIO.load_manifest(dir)
        @test m2.result_kind == "image" && m2.model === nothing && m2.metrics === nothing && m2.provenance === nothing
    end
end

@testitem "calibration/v3.0: a projective manifest with a single target round trips" setup = [ManifestFixtures] begin
    import VisualGeometryToolkitIO as VgtkIO
    import VisualGeometryToolkitCore as VgtkCore
    mktempdir() do dir
        files = projective_dir(dir)
        model = VgtkCore.CameraModelSpec(Dict("lens" => "division", "coefficients" => 3, "intrinsics" => "projective"))
        VgtkIO.save_manifest(dir, files; result_kind = "projective", model)
        m = VgtkIO.load_manifest(dir)
        @test m.result_kind == "projective" && m.model == model && length(m.files) == 2
        # a metric kind with a projective model is inconsistent
        @test_throws "calibrates \"projective\" intrinsics" VgtkIO.save_manifest(dir, files; result_kind = "metric", model)
    end
end

@testitem "calibration/v3.0: a tampered or missing file is refused naming the file" setup = [ManifestFixtures] begin
    import VisualGeometryToolkitIO as VgtkIO
    mktempdir() do dir
        VgtkIO.save_manifest(dir, metric_dir(dir); result_kind = "metric")
        write(joinpath(dir, "trajectory.json"), read(joinpath(dir, "trajectory.json"), String) * "\n")
        err = try
            VgtkIO.load_manifest(dir); nothing
        catch e
            e
        end
        @test err isa ArgumentError && occursin("trajectory.json", err.msg) && occursin("changed since the manifest was written", err.msg)
        cp(fixture("trajectory.json"), joinpath(dir, "trajectory.json"); force = true)
        @test VgtkIO.load_manifest(dir).result_kind == "metric"
        rm(joinpath(dir, "target_poses.json"))
        @test_throws "target_poses.json, which is not a file" VgtkIO.load_manifest(dir)
    end
end

@testitem "calibration/v3.0: a manifest edited by hand is refused" setup = [ManifestFixtures] begin
    import VisualGeometryToolkitIO as VgtkIO
    mktempdir() do dir
        VgtkIO.save_manifest(dir, metric_dir(dir); result_kind = "metric")
        p = joinpath(dir, "calibration.json")
        original = read(p, String)
        edit(f) = (write(p, original); rewrite(p, f))
        # a recorded schema that the file does not hold
        edit(d -> d["files"][2]["schema"] = "vgio/target_poses/v3.0")
        @test_throws "holds a \"vgio/trajectory/v5.0\" record" VgtkIO.load_manifest(dir)
        # the 0.9 key names: `kind` for the result's kind and a file's schema
        edit(d -> (d["kind"] = d["result_kind"]; delete!(d, "result_kind")))
        @test_throws "result_kind" VgtkIO.load_manifest(dir)
        edit(d -> (d["files"][1]["kind"] = d["files"][1]["schema"]; delete!(d["files"][1], "schema")))
        @test_throws "schema" VgtkIO.load_manifest(dir)
        # the metrics are a closed object of numbers; the model VGC's closed table
        edit(d -> d["metrics"] = Dict("rms_px" => "small"))
        @test_throws "metrics.rms_px" VgtkIO.load_manifest(dir)
        edit(d -> d["model"] = Dict("lens" => "division", "coefficients" => 3, "intrinsics" => "metric", "note" => "x"))
        @test_throws "model" VgtkIO.load_manifest(dir)
        # a record schema of no owner's form
        edit(d -> d["files"][1]["schema"] = "rig.json")
        @test_throws "a record schema" VgtkIO.load_manifest(dir)
        # a derived entry without derived_from (missing key, and empty list)
        edit(d -> delete!(d["files"][4], "derived_from"))
        @test_throws "derived_from" VgtkIO.load_manifest(dir)
        edit(d -> d["files"][4]["derived_from"] = [])
        @test_throws "a derived file says which listed files it is derived from" VgtkIO.load_manifest(dir)
        # an authority with a derived_from; a derived_from outside the manifest; itself; a cycle
        edit(d -> d["files"][1]["derived_from"] = ["trajectory.json"])
        @test_throws "an authority is derived from nothing" VgtkIO.load_manifest(dir)
        edit(d -> d["files"][4]["derived_from"] = ["nope.json"])
        @test_throws "which the manifest does not list" VgtkIO.load_manifest(dir)
        edit(d -> d["files"][4]["derived_from"] = ["target_poses.json"])
        @test_throws "not derived from itself" VgtkIO.load_manifest(dir)
        edit(d -> (d["files"][2]["role"] = "derived"; d["files"][2]["derived_from"] = ["target_poses.json"]))
        @test_throws "derived from itself through" VgtkIO.load_manifest(dir)
        # the kind, a path leaving the directory, a duplicate path, an unknown key, an unknown schema token
        edit(d -> d["result_kind"] = "euclidean")
        @test_throws "result_kind" VgtkIO.load_manifest(dir)
        edit(d -> d["files"][1]["path"] = "../rig_targets.json")
        @test_throws "relative path inside" VgtkIO.load_manifest(dir)
        edit(d -> d["files"][2]["path"] = "rig_targets.json")
        @test_throws "file paths must be unique" VgtkIO.load_manifest(dir)
        edit(d -> d["files"][2]["path"] = "./rig_targets.json")     # the same file spelled another way
        @test_throws "file paths must be unique" VgtkIO.load_manifest(dir)
        # an image result lists no trajectory or target_poses record
        edit(d -> d["result_kind"] = "image")
        @test_throws "a result with no placement in space" VgtkIO.load_manifest(dir)
        edit(d -> d["extra"] = 1)
        @test_throws "unknown key" VgtkIO.load_manifest(dir)
        edit(d -> d["schema"] = "vgio/calibration/v0")
        @test_throws ArgumentError VgtkIO.load_manifest(dir)
        # a model VGC does not read
        edit(d -> d["model"] = Dict("lens" => "no_such_lens"))
        @test_throws "model" VgtkIO.load_manifest(dir)
        # no manifest at all
        mktempdir() do empty
            @test_throws "holds no calibration.json" VgtkIO.load_manifest(empty)
        end
    end
end

@testitem "calibration/v3.0: save_manifest refuses what load_manifest would" setup = [ManifestFixtures] begin
    import VisualGeometryToolkitIO as VgtkIO
    mktempdir() do dir
        files = metric_dir(dir)
        p = joinpath(dir, "calibration.json")
        @test_throws "a derived file says which listed files" VgtkIO.save_manifest(dir, [files[1], (path = "target_poses.json", role = "derived")]; result_kind = "metric")
        @test_throws "a derived file says which listed files" VgtkIO.save_manifest(dir, [files[1], (path = "target_poses.json", role = "derived", derived_from = String[])]; result_kind = "metric")
        @test_throws "which the manifest does not list" VgtkIO.save_manifest(dir, [files[1], (path = "target_poses.json", role = "derived", derived_from = ["x.json"])]; result_kind = "metric")
        @test_throws "a role" VgtkIO.save_manifest(dir, [(path = "trajectory.json", role = "author")]; result_kind = "metric")
        @test_throws "is not a file" VgtkIO.save_manifest(dir, ["missing.json"]; result_kind = "metric")
        @test_throws "a calibration result kind" VgtkIO.save_manifest(dir, files; result_kind = "euclidean")
        @test_throws "metrics" VgtkIO.save_manifest(dir, files; result_kind = "metric", metrics = (note = "not a number",))
        @test_throws ArgumentError VgtkIO.save_manifest(dir, [(path = "trajectory.json", rol = "authority")]; result_kind = "metric")
        @test_throws "a result with no placement in space" VgtkIO.save_manifest(dir, files; result_kind = "image")
        @test_throws "file paths must be unique" VgtkIO.save_manifest(dir, [files[1], "./" * files[1]]; result_kind = "metric")
        write(joinpath(dir, "notes.txt"), "{\"a\": 1}")
        @test_throws "not a record of any owner" VgtkIO.save_manifest(dir, ["notes.txt"]; result_kind = "metric")
        @test !isfile(p)                                       # nothing was written by any refusal
    end
end

@testitem "calibration/v3.0: observations must name targets the manifest's target and rig files hold" setup = [ManifestFixtures] begin
    import VisualGeometryToolkitIO as VgtkIO
    import VisualGeometryToolkitCore
    using VisualGeometryToolkitCore: PixelFrame
    using StaticArrays: SVector
    using VisualGeometryToolkitCore: TargetHomographies, TargetObservations, correspondences
    using VisualGeometryToolkitIO: load, save, content_id
    mktempdir() do dir
        files = metric_dir(dir)
        # the rig the observations reference is not listed: refused, naming the observation file
        err = try
            VgtkIO.save_manifest(dir, filter(f -> f != "rig_targets.json" && f isa String, files); result_kind = "metric"); nothing
        catch e
            e
        end
        @test err isa ArgumentError && occursin("observations2d.json", err.msg) && occursin("names the targets identity", err.msg)
        # a target name that is not in the rig it references
        obs = load(TargetObservations, joinpath(dir, "observations2d.json"))
        frames = Dict(t => Dict(Symbol("zz", id) => cs for (id, cs) in sets) for (t, sets) in getfield(obs, :frames))
        bad = TargetObservations{PixelFrame{:julia}}(frames; stamps = obs.stamps, targets = obs.targets)
        save(joinpath(dir, "observations2d.json"), bad)
        err = try
            VgtkIO.save_manifest(dir, files; result_kind = "metric"); nothing
        catch e
            e
        end
        @test err isa ArgumentError && occursin("observations2d.json", err.msg) && occursin("\"zz", err.msg)
    end
    # homographies against a single target: an unreferenced one is measured under its content id,
    # a referenced one under its reference; any other name is refused naming the file
    mktempdir() do dir
        files = projective_dir(dir)
        VgtkIO.save_manifest(dir, files; result_kind = "image")
        @test VgtkIO.load_manifest(dir).result_kind == "image"
        wrong = projective_dir(dir; name = "board")
        err = try
            VgtkIO.save_manifest(dir, wrong; result_kind = "image"); nothing
        catch e
            e
        end
        @test err isa ArgumentError && occursin("target_homographies.json", err.msg) && occursin("frame 1", err.msg) && occursin("\"board\"", err.msg)
        files = projective_dir(dir; reference = "uid7", name = "uid7")
        VgtkIO.save_manifest(dir, files; result_kind = "image")
        @test VgtkIO.load_manifest(dir).result_kind == "image"
        for bad in ("board", content_id(joinpath(dir, "target_planar.json")))   # a referenced target is not named by its content id
            files = projective_dir(dir; reference = "uid7", name = bad)
            err = try
                VgtkIO.save_manifest(dir, files; result_kind = "image"); nothing
            catch e
                e
            end
            @test err isa ArgumentError && occursin("target_homographies.json", err.msg) && occursin("uid7", err.msg)
        end
        files = projective_dir(dir)
        th = load(TargetHomographies, joinpath(dir, "target_homographies.json"))
        stray = retargeted(th, ["0"^64])
        save(joinpath(dir, "target_homographies.json"), stray)
        err = try
            VgtkIO.save_manifest(dir, files; result_kind = "image"); nothing
        catch e
            e
        end
        @test err isa ArgumentError && occursin("target_homographies.json", err.msg) && occursin("0"^64, err.msg)
        # a rig's homographies against a name the rig does not hold
        cp(fixture("rig_targets.json"), joinpath(dir, "rig_targets.json"))
        rig = (content_id(joinpath(dir, "rig_targets.json")))
        mis = retargeted(th, [rig])
        save(joinpath(dir, "target_homographies.json"), mis)
        err = try
            VgtkIO.save_manifest(dir, ["rig_targets.json", "target_homographies.json"]; result_kind = "image"); nothing
        catch e
            e
        end
        @test err isa ArgumentError && occursin("target_homographies.json", err.msg) && occursin("frame 1", err.msg)
        # observations of the single target (the file names it by content id, its points by its name)
        tid = content_id(joinpath(dir, "target_planar.json"))
        observed(name) = TargetObservations{PixelFrame{:colmap}}(
            Dict(1 => Dict(Symbol(name) => correspondences([1, 2], [SVector(1.0, 2.0), SVector(3.0, 4.0)])));
            images = Dict(1 => "a.png"), sensors = "cam", targets = [tid],
        )
        save(joinpath(dir, "observations_cam.json"), observed(tid))
        VgtkIO.save_manifest(dir, ["target_planar.json", "observations_cam.json"]; result_kind = "image")
        @test VgtkIO.load_manifest(dir).result_kind == "image"
        save(joinpath(dir, "observations_cam.json"), observed("board"))
        err = try
            VgtkIO.save_manifest(dir, ["target_planar.json", "observations_cam.json"]; result_kind = "image"); nothing
        catch e
            e
        end
        @test err isa ArgumentError && occursin("observations_cam.json", err.msg) && occursin("\"board\"", err.msg)
        # the load side refuses a tree that became inconsistent after the manifest was written (hash updated by hand)
        save(joinpath(dir, "target_homographies.json"), th)
        VgtkIO.save_manifest(dir, ["target_planar.json", "target_homographies.json"]; result_kind = "image")
        @test VgtkIO.load_manifest(dir).result_kind == "image"
    end
end

@testitem "calibration/v3.0: the load side runs the target resolution too" setup = [ManifestFixtures] begin
    import VisualGeometryToolkitIO as VgtkIO
    import VisualGeometryToolkitCore as VgtkCore
    using VisualGeometryToolkitCore: TargetHomographies
    using VisualGeometryToolkitIO: load, save
    mktempdir() do dir
        files = projective_dir(dir)
        VgtkIO.save_manifest(dir, files; result_kind = "image")
        # a manifest written by hand over an observation file whose target is unknown: the hash is right, the resolution is not
        th = load(TargetHomographies, joinpath(dir, "target_homographies.json"))
        stray = retargeted(th, ["1"^64])
        save(joinpath(dir, "target_homographies.json"), stray)
        p = joinpath(dir, "calibration.json")
        rewrite(p, d -> d["files"][2]["sha256"] = VgtkCore.sha256_hex(read(joinpath(dir, "target_homographies.json"), String)))
        @test_throws "target_homographies.json: names the targets identity" VgtkIO.load_manifest(dir)
    end
end

@testitem "calibration/v3.0: a listed file that is a symlink leaving the directory is refused" setup = [ManifestFixtures] begin
    import VisualGeometryToolkitIO as VgtkIO
    mktempdir() do outside
        mktempdir() do dir
            cp(fixture("rig_targets.json"), joinpath(outside, "rig_targets.json"))
            symlink(joinpath(outside, "rig_targets.json"), joinpath(dir, "rig_targets.json"))
            @test_throws "resolves outside" VgtkIO.save_manifest(dir, ["rig_targets.json"]; result_kind = "metric")
            # a symlink that stays inside is fine
            cp(fixture("rig_targets.json"), joinpath(dir, "real.json"))
            symlink(joinpath(dir, "real.json"), joinpath(dir, "link.json"))
            @test isfile(VgtkIO.save_manifest(dir, ["link.json"]; result_kind = "metric"))
        end
    end
end

@testitem "calibration/v3.0: a v1 manifest's model `free` reads as its complement `fixed`; a save writes `fixed`" setup = [ManifestFixtures] begin
    import VisualGeometryToolkitCore as VgtkCore
    import VisualGeometryToolkitIO as VgtkIO
    using JSON: JSON
    mktempdir() do dir
        files = metric_dir(dir)
        new = VgtkCore.CameraModelSpec(Dict("lens" => "kannala_brandt", "coefficients" => 4, "fixed" => ["k3", "k4"], "intrinsics" => "metric"))
        p = VgtkIO.save_manifest(dir, files; result_kind = "metric", model = new)
        saved = JSON.parse(read(p, String))
        @test saved["schema"] == "vgio/calibration/v3.0"
        @test saved["model"]["fixed"] == ["k3", "k4"] && !haskey(saved["model"], "free")    # a save writes `fixed` only
        # the same manifest as v1 wrote it (no envelope, `kind` for the result's and each file's): the coefficients MOVED, k1 and k2
        rewrite(p, d -> begin
            d["schema"] = "vgio/calibration/v1"
            foreach(k -> delete!(d, k), ("generator", "units", "conventions"))
            d["kind"] = pop!(d, "result_kind")
            foreach(f -> (f["kind"] = pop!(f, "schema")), d["files"])
            delete!(d["model"], "fixed"); d["model"]["free"] = ["k1", "k2"]
        end)
        before = read(p, String)
        m = VgtkIO.load_manifest(dir)
        @test m.model == new                                                                 # the same LensSpec
        @test VgtkCore.free_names(m.model.model) == (:k1, :k2)
        @test read(p, String) == before                                                      # never rewritten on disk
        # a v1 model with every coefficient moving (no "free") is unchanged
        rewrite(p, d -> delete!(d["model"], "free"))
        @test VgtkCore.free_names(VgtkIO.load_manifest(dir).model.model) == (:k1, :k2, :k3, :k4)
    end
end

@testitem "calibration/v3.0: a record of any owner is bound by its hash, and checked by its owner's reader" setup = [ManifestFixtures] begin
    import VisualGeometryToolkitIO as VgtkIO
    # an owner with a reader (ManifestFixtures' `validate_record` method for `manifesttestowner`) runs on every
    # listed file of its kinds
    mktempdir() do dir
        files = metric_dir(dir)
        VgtkIO.write_record(joinpath(dir, "rc_state.json"), (schema = "rc/calibration_state/v1", note = "an rc record"))
        VgtkIO.write_record(joinpath(dir, "owned.json"), (schema = "manifesttestowner/thing/v2.1", value = 3))
        all_files = [files; "rc_state.json"; (path = "owned.json", role = "derived", derived_from = ["rc_state.json"])]
        VgtkIO.save_manifest(dir, all_files; result_kind = "metric")
        m = VgtkIO.load_manifest(dir)
        @test [f.schema for f in m.files[5:6]] == ["rc/calibration_state/v1", "manifesttestowner/thing/v2.1"]
        # an owner with no reader: its file is bound by its hash alone, which is checked
        write(joinpath(dir, "rc_state.json"), read(joinpath(dir, "rc_state.json"), String) * " ")
        @test_throws "changed since the manifest was written" VgtkIO.load_manifest(dir)
        # an owner's reader refusing its file: refused, naming the file
        VgtkIO.write_record(joinpath(dir, "rc_state.json"), (schema = "rc/calibration_state/v1", note = "an rc record"))
        VgtkIO.write_record(joinpath(dir, "owned.json"), (schema = "manifesttestowner/thing/v2.1", value = -1))
        @test_throws "value must be positive" VgtkIO.save_manifest(dir, all_files; result_kind = "metric")
    end
end
