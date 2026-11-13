# =============================================================================
# The validator: every required key of every kind, refused by name
# =============================================================================
#
# A valid file of every kind is written by its own writer; then, driven by the
# kind's spec table (validate.jl) through the one deletion-case generator
# (`VGIoSchemaCases.required_paths`, which the JSON Schema tests share), every
# required key the file holds — the envelope's included — is deleted in turn
# and the result must be refused, by `validate` and so by the kind's reader,
# which runs the same table, naming the key. An unknown key, a wrong
# declaration token and a retired or later schema are refused too.

@testitem "v2 validation: every required key, refused by name" setup = [VGIoFixtures, VGIoSchemaCases, EnvelopeCases] begin
    using VisualGeometryToolkitIO: validate
    using VisualGeometryToolkitCore: Rig, PosedTarget, PlanarTarget, PointTarget, Attributed, BoundedCamera, PosedCamera, TargetFrame,
        ViewFrame, BodyFrame, WorldFrame, CameraFrame, relabel, Trajectory, TargetPoses, TargetObservations, correspondences, withattrs
    using Rotations: RotXYZ
    import VisualGeometryToolkitCore
    const VgtkIO = VisualGeometryToolkitIO

    function refused(doc, dir, name; units = false)
        target = units ? mkpath(joinpath(dir, "sidecar_$(name)")) : joinpath(dir, "bad_$(name).json")
        write(units ? joinpath(target, VgtkIO.UNITS_SIDECAR) : target, JSON.json(doc))
        return try
            validate(target)
            nothing
        catch e
            e isa ArgumentError ? e.msg : rethrow()
        end
    end

    mktempdir() do dir
        # a file of every kind (EnvelopeCases), and the shapes those leave out: a camera store and a rig with a
        # non-central camera, a rig of targets, a target with its calibration record, 3D observations with
        # covariances, the units sidecar and the correspondences export
        files = Dict{String, String}(k => first(v) for (k, v) in write_every_kind(dir))
        K = CameraCalibrationMat(SMatrix{3, 3, Float64}(900.0, 0, 0, 0, 900.0, 0, 2120.0, 1416.0, 1))
        pose(z) = EuclideanMap(RotMatrix{3, Float64}(RotXYZ(0.1, -0.2, 0.3)), [1.0mm, 2.0mm, z])
        pts = [mm_lengths(SVector(10.0i, 5.0i)) for i in 1:4]
        gc = CameraModel(SplineLens, CameraModel(K; lens = KannalaBrandt(-0.02), pixel_frame = :colmap), Size2(64, 48); grid_size = (5, 4))
        gd = gc.lens.backward
        gn = BoundedCamera(NoncentralSplineModel(gc.intrinsics, gd, ChartSpline(gd.area, [SVector(0.1gx, -0.1gy, 0.05) for gy in 1:4, gx in 1:5])), Size2(64, 48))
        ident = EuclideanMap(one(RotMatrix{3, Float64}), [0.0mm, 0.0mm, 0.0mm])
        targets = Rig{TargetFrame{:a}}(Dict(:a => PosedTarget(PlanarTarget(pts), ident), :b => PosedTarget(PlanarTarget(:b), pose(100.0mm))))
        at!(name, base) = (files[name] = joinpath(dir, base))
        files["units"] = mkpath(joinpath(dir, "model"))
        VgtkIO.save_units_sidecar(files["units"], MM_PX)
        save(at!("cameras_noncentral", "cameras_noncentral.json"), Dict(:g => gn, :s => BoundedCamera(gc, Size2(64, 48))))
        save(at!("rig_noncentral", "rig_noncentral.json"), Rig{ViewFrame{:g}}((BoundedCamera(PosedCamera(VisualGeometryToolkitCore.model(gn), relabel(ViewFrame{:g}, ViewFrame{:g}, ident)), Size2(64, 48)),)))
        save(at!("rig_targets", "targets.json"), targets)
        save(at!("target_record", "target_record.json"), Attributed(PlanarTarget(pts), (; calibration_record = (; tool = "t"))))
        save(at!("observations3d", "observations3d.json"), TargetObservations{WorldFrame}(
            Dict(1 => Dict(:rig => correspondences([1], [VisualGeometryToolkitCore.Uncertain(mm_lengths(SVector(1.0, 2.0, 3.0)), VisualGeometryToolkitCore.CovMat(SMatrix{3, 3, Float64}(1.0e-8, 0, 0, 0, 1.0e-8, 0, 0, 0, 1.0e-8)))])));
            sensors = "mocap", images = Dict(1 => "a.png")))
        obs = TargetObservations{PixelFrame{:colmap}}(Dict(1 => Dict(:a => withattrs(correspondences([2], [SVector(640.0, 480.0)]); weights = [1.0])));
            images = Dict(1 => "a.png"), sensors = "cam", targets = [content_id(targets)])
        save(at!("observations", "observations.json"), obs)
        save_correspondences(at!("correspondences", "correspondences.json"), obs, files["rig_targets"])
        kind_of(name) = name == "correspondences" ? nothing : name in ("cameras_noncentral",) ? "cameras" :
            startswith(name, "rig") ? "rig" : startswith(name, "observations") ? "observations" : name == "target_record" ? "target" : name
        spec(name) = name == "correspondences" ? VgtkIO._CORRESPONDENCES_SPEC : VgtkIO._spec(kind_of(name))
        for (name, path) in sort!(collect(files))
            units = name == "units"
            file = units ? joinpath(path, VgtkIO.UNITS_SIDECAR) : path
            doc = plain_doc(parse_json(read(file, String)))
            name == "correspondences" || @test validate(path) == kind_of(name)
            paths = required_paths(spec(name), doc)
            @test length(paths) >= 6                                    # the envelope's four at least, and the body's
            @test all(p -> p in paths, (["schema"], ["generator"], ["units"], ["conventions"], ["units", "pixel"], ["conventions", "index_base"]))
            name == "correspondences" && continue                       # an export: no reader, its spec is the writer's check
            for p in paths
                msg = refused(without(doc, p), dir, "$(name)_$(length(p))_$(p[end])"; units)
                @test msg !== nothing && occursin(String(p[end] isa Integer ? p[end - 1] : p[end]), msg)
            end
            # no key the table does not name — at the top and in the first nested object
            @test occursin("bogus", something(refused(merge(doc, Dict("bogus" => 1)), dir, "$(name)_bogus"; units), ""))
            deep = [p for p in paths if length(p) >= 2 && getpath(doc, p[1:(end - 1)]) isa AbstractDict && p[1] ∉ ("units", "conventions", "generator")]
            if !isempty(deep)
                q = first(deep)[1:(end - 1)]
                @test occursin("bogus", something(refused(with_key(doc, [q; "bogus"], 1), dir, "$(name)_deepbogus"; units), ""))
            end
            # an earlier major is read through its version step where a published file holds it, and refused by name
            # otherwise; a current record relabeled as the major before is either read by that step or refused naming the
            # file (never a crash), and any v1 with no step is refused by name (migrate_*_tests.jl hold each step's claims)
            M = VgtkIO._SCHEMA_VERSION[kind_of(name)][1]
            prev = refused(with_key(doc, ["schema"], "vgio/$(kind_of(name))/v$(M - 1)"), dir, "$(name)_prev"; units)
            @test prev === nothing || occursin("$(name)_prev", prev)                   # the file it read, named
            later = refused(with_key(doc, ["schema"], "vgio/$(kind_of(name))/v$(M + 1).0"), dir, "$(name)_later"; units)
            @test occursin("later major", something(later, ""))
        end

        # a declaration is a TOKEN: anything else is refused, never reinterpreted
        tr = plain_doc(parse_json(read(files["transforms"], String)))
        @test occursin("rotation_encoding", something(refused(with_key(tr, ["conventions", "rotation_encoding"], "quaternion_xyzw"), dir, "tok1"), ""))
        @test occursin("transform_convention", something(refused(with_key(tr, ["conventions", "transform_convention"], "x_from = R x_to + t"), dir, "tok2"), ""))
        @test occursin("units.length", something(refused(with_key(tr, ["units", "length"], "px"), dir, "tok3"), ""))
        ob = plain_doc(parse_json(read(files["observations"], String)))
        @test occursin("index_base", something(refused(with_key(ob, ["conventions", "index_base"], 0), dir, "tok4"), ""))
        @test occursin("targets", something(refused(with_key(ob, ["targets"], Any[]), dir, "tok5"), ""))
        # the retired IO 0.9 declarations are refused at the top level, not read
        for (k, v) in (("length_unit", "mm"), ("pixel_frame", "colmap"), ("point_index_base", 1), ("rotation_encoding", "quaternion_wxyz_hamilton"))
            @test occursin(k, something(refused(merge(ob, Dict(k => v)), dir, "retired_$(k)"), ""))
        end
        nc = plain_doc(parse_json(read(files["cameras_noncentral"], String)))
        g = findfirst(c -> haskey(c, "type"), nc["cameras"])
        @test occursin("spline lens", something(refused(with_key(nc, ["cameras", g, "type"], "CentralGenericModel"), dir, "tok7"), ""))
        @test occursin("units.pixel", something(refused(with_key(nc, ["units", "pixel"], "lower_left"), dir, "tok8"), ""))
        # the non-central camera's points are declared in a unit, its lattice whole
        @test refused(without(nc, ["units", "length"]), dir, "gn_unit") !== nothing
        @test refused(without(nc, ["cameras", g, "spline", "controls", "point"]), dir, "gn_grid") !== nothing
        @test refused(with_key(nc, ["cameras", g, "spline", "grid_size"], Any[5, 4]), dir, "gn_size") !== nothing      # a Size2 object
        @test refused(with_key(nc, ["cameras", g, "image_size"], Any[64, 48]), dir, "gn_image_size") !== nothing

        # a frame name is the grammar's: a free-form one, or an IO 0.9 one, is refused
        @test occursin("frame name", something(refused(with_key(tr, ["transforms", 1, "from"], "printed_board_a"), dir, "free_from"), ""))
        @test refused(with_key(tr, ["transforms", 1, "to"], "camera_optical"), dir, "old_camera") !== nothing
        tj = plain_doc(parse_json(read(files["trajectory"], String)))
        @test occursin("frame name", something(refused(with_key(tj, ["to"], "mocap_world"), dir, "free_to"), ""))
        # a time is seconds since a stated origin, which states its UTC offset; the join key is `frame_index`
        @test occursin("time_origin", something(refused(without(tj, ["time_origin"]), dir, "no_origin"), ""))
        @test occursin("time_origin", something(refused(with_key(tj, ["time_origin"], "2026-05-27T00:00:00"), dir, "no_offset"), ""))
        @test occursin("frames[1].t", something(refused(with_key(tj, ["frames", 1, "t"], 1), dir, "old_t"), ""))
        # a covariance is symmetric and of the value's size
        bad = Any[Any[i == j ? 1.0 : 0.0 for j in 1:6] for i in 1:6]; bad[1][2] = 0.5
        @test occursin("symmetric", something(refused(with_key(tr, ["transforms", 1, "covariance"], bad), dir, "asym"), ""))
        @test refused(with_key(tr, ["transforms", 1, "covariance"], Any[Any[1.0, 0.0], Any[0.0, 1.0]]), dir, "cov2") !== nothing
        # the reserved `payload` (a binary sidecar of bulk arrays) is refused in a kind that does not implement it
        @test occursin("payload", something(refused(merge(tj, Dict("payload" => Dict("file" => "x.arrow", "sha256" => "0"))), dir, "payload"), ""))
        # the readers run the same table: a deletion refused by `validate` is refused by `load`
        cs = plain_doc(parse_json(read(files["cameras"], String)))
        @test_throws ArgumentError load(Dict{Symbol, BoundedCamera}, (write(joinpath(dir, "c1.json"), JSON.json(without(cs, ["cameras", 1, "K", "skew"]))); joinpath(dir, "c1.json")))
        @test_throws ArgumentError load(Trajectory, (write(joinpath(dir, "t1.json"), JSON.json(without(tj, ["to"]))); joinpath(dir, "t1.json")))
        # a writer cannot emit what its reader refuses
        @test_throws ArgumentError save(joinpath(dir, "empty_targets.json"), TargetObservations{PixelFrame{:colmap}}(getfield(obs, :frames); stamps = obs.stamps))
        @test !isfile(joinpath(dir, "empty_targets.json"))
    end
end
