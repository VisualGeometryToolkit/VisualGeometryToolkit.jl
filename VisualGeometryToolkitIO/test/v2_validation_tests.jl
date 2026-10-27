# =============================================================================
# The validator: every required key of every kind, refused by name
# =============================================================================
#
# A valid file of every kind is written by its own writer; then, driven by the
# kind's spec table (validate.jl), every required key the file holds is deleted
# in turn and the result must be refused — by `validate` and so by the kind's
# reader, which runs the same table — naming the key. An unknown key, a wrong
# declaration token and a retired schema are refused too.

@testitem "v2 validation: every required key, refused by name" setup = [VGIoFixtures] begin
    using VisualGeometryToolkitIO: validate
    using VisualGeometryToolkitCore: Rig, PosedTarget, PlanarTarget, PointTarget, Attributed, BoundedCamera, PosedCamera, TargetFrame, ViewFrame, BodyFrame, WorldFrame,
        CameraFrame, CentralGenericModel, NoncentralGenericModel, Vec3, relabel, Trajectory, TargetPoses, TargetPoints2D, TargetPoints3D
    using Rotations: RotXYZ, RotationVec
    import VisualGeometryToolkitCore
    const VgtkIO = VisualGeometryToolkitIO

    tomutable(x::AbstractDict) = Dict{Symbol, Any}(Symbol(k) => tomutable(v) for (k, v) in x)
    tomutable(x::AbstractVector) = Any[tomutable(v) for v in x]
    tomutable(x) = x

    # The paths of the required keys a document holds, following its spec (the
    # first element of every array).
    function required_paths(spec, doc)
        out = Vector{Vector{Any}}()
        walk!(out, spec, doc, Any[])
        return out
    end
    walk!(out, s::VgtkIO._Checked, x, at) = walk!(out, s.inner, x, at)
    function walk!(out, s::VgtkIO._Obj, x, at)
        x isa AbstractDict || return nothing
        for (k, sk) in s.required
            haskey(x, k) || continue
            push!(out, [at; k])
            walk!(out, sk, x[k], [at; k])
        end
        for (k, sk) in s.optional
            (k !== :provenance && haskey(x, k)) && walk!(out, sk, x[k], [at; k])
        end
        return nothing
    end
    walk!(out, s::VgtkIO._Vec, x, at) = (x isa AbstractVector && !isempty(x)) ? walk!(out, s.elem, x[1], [at; 1]) : nothing
    walk!(out, s::VgtkIO._OrNull, x, at) = x === nothing ? nothing : walk!(out, s.inner, x, at)
    walk!(out, s::VgtkIO._Either, x, at) = foreach(p -> haskey(x, p.first) && walk!(out, p.second, x, at), s.alts)
    walk!(out, s, x, at) = nothing

    getpath(x, path) = foldl((v, k) -> v[k], path; init = x)
    without(doc, path) = (d = deepcopy(doc); delete!(getpath(d, path[1:(end - 1)]), path[end]); d)
    with(doc, path, v) = (d = deepcopy(doc); getpath(d, path[1:(end - 1)])[path[end]] = v; d)

    function refusal(doc, dir, name; units = false)
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
        K = CameraCalibrationMat(SMatrix{3, 3, Float64}(900.0, 0, 0, 0, 900.0, 0, 2120.0, 1416.0, 1))
        bc = opencv_lens([-0.1, 0.05, 1.0e-4, -2.0e-4])
        pose(z) = EuclideanMap(RotMatrix{3, Float64}(RotXYZ(0.1, -0.2, 0.3)), [1.0mm, 2.0mm, z])
        pts = [mm_lengths(SVector(10.0i, 5.0i)) for i in 1:4]
        dirs = [Vec3{Float64}(0.01 * gx, 0.02 * gy, 1.0) for gy in 1:4, gx in 1:5]
        gc = CentralGenericModel(64, 48, 0, 0, 63, 47, dirs)
        gn = NoncentralGenericModel(64, 48, 0, 0, 63, 47, dirs, [Vec3{Float64}(0.1gx, -0.1gy, 0.05) for gy in 1:4, gx in 1:5])
        ident = EuclideanMap(one(RotMatrix{3, Float64}), [0.0mm, 0.0mm, 0.0mm])
        targets = Rig{TargetFrame{:a}}(Dict(:a => PosedTarget(PlanarTarget(pts), ident), :b => PosedTarget(PlanarTarget(:b), pose(100.0mm))))

        files = Dict{String, String}()
        at!(name, base) = (files[name] = joinpath(dir, base))
        files["units"] = mkpath(joinpath(dir, "model"))
        VgtkIO.save_units_sidecar(files["units"], MM_PX)
        save(at!("transforms", "transforms.json"), [relabel(TargetFrame{:a}, BodyFrame{:b}, pose(3.0mm))])
        save(at!("cameras", "cameras.json"), Dict(:c => Attributed(BoundedCamera(CameraModel(K; lens = bc, image_origin = :colmap), Size2(4240, 2832)), (; calibration = (; tool = "t")))))
        save(at!("generic_camera", "central.json"), gc)
        save(at!("generic_noncentral", "noncentral.json"), gn)
        save(
            at!("rig", "rig.json"),
            Rig{ViewFrame{:a}}(
                (
                    BoundedCamera(PosedCamera(CameraModel(K; lens = bc, image_origin = :colmap), relabel(ViewFrame{:a}, ViewFrame{:a}, ident)), Size2(4240, 2832)),
                    BoundedCamera(PosedCamera(CameraModel(K; lens = bc, image_origin = :colmap), relabel(ViewFrame{:a}, ViewFrame{:b}, pose(5.0mm))), Size2(4240, 2832)),
                )
            )
        )
        save(at!("rig_generic", "rig_generic.json"), Rig{ViewFrame{:g}}((PosedCamera(gn, relabel(ViewFrame{:g}, ViewFrame{:g}, ident)),)))
        save(at!("rig_targets", "targets.json"), targets)
        save(at!("target", "target.json"), Attributed(PointTarget([mm_lengths(SVector(1.0, 2.0, 3.0))]), (; calibration_record = (; tool = "t"))))
        save(
            at!("trajectory", "trajectory.json"),
            Trajectory(BodyFrame{:rig}, WorldFrame, Dict(1 => pose(900.0mm)); images = Dict(1 => "a.png"), times = Dict(1 => 0.5), time_origin = "2026-05-27T00:00:00"),
        )
        save(
            at!("target_poses", "target_poses.json"),
            TargetPoses{CameraFrame}(Dict(1 => Dict(:a => pose(900.0mm))); images = Dict(1 => "a.png"), sensors = "cam", times = Dict(1 => 0.5), time_origin = "2026-05-27T00:00:00"),
        )
        save(
            at!("target_points3d", "target_points3d.json"),
            TargetPoints3D{WorldFrame}(Dict(1 => [(target = :rig, index = 1, xyz = mm_lengths(SVector(1.0, 2.0, 3.0)))]); sensors = "mocap", images = Dict(1 => "a.png")),
        )
        obs = TargetPoints2D{PixelFrame{:colmap}}(Dict(1 => [(; target = :a, index = 2, xy = SVector(640.0, 480.0), weight = 1.0)]);
            images = Dict(1 => "a.png"), sensors = "cam", targets = [content_id(targets)])
        save(at!("target_points2d", "target_points2d.json"), obs)
        save_correspondences(at!("correspondences", "correspondences.json"), obs, files["rig_targets"])
        specs = Dict(
            "units" => VgtkIO._UNITS_SPEC, "transforms" => VgtkIO._TRANSFORMS_SPEC, "cameras" => VgtkIO._CAMERAS_SPEC,
            "generic_camera" => VgtkIO._GENERIC_CAMERA_SPEC, "generic_noncentral" => VgtkIO._GENERIC_CAMERA_SPEC,
            "rig" => VgtkIO._RIG_SPEC, "rig_generic" => VgtkIO._RIG_SPEC, "rig_targets" => VgtkIO._RIG_SPEC, "target" => VgtkIO._TARGET_SPEC,
            "trajectory" => VgtkIO._TRAJECTORY_SPEC, "target_points2d" => VgtkIO._TARGET_POINTS2D_SPEC,
            "target_poses" => VgtkIO._TARGET_POSES_SPEC, "target_points3d" => VgtkIO._TARGET_POINTS3D_SPEC,
            "correspondences" => VgtkIO._CORRESPONDENCES_SPEC,
        )
        for (name, path) in sort!(collect(files))
            units = name == "units"
            file = units ? joinpath(path, VgtkIO.UNITS_SIDECAR) : path
            doc = tomutable(parse_json(read(file, String)))
            @test validate(path) == get(Dict("generic_noncentral" => "generic_camera", "rig_generic" => "rig", "rig_targets" => "rig"), name, name)
            paths = required_paths(specs[name], doc)
            @test length(paths) >= 2
            for p in paths
                msg = refusal(without(doc, p), dir, "$(name)_$(length(p))"; units)
                @test msg !== nothing && occursin(String(p[end]), msg)
            end
            # no key the table does not name — at the top and in the first nested object
            @test occursin("bogus", something(refusal(merge(doc, Dict(:bogus => 1)), dir, "$(name)_bogus"; units), ""))
            deep = [p for p in paths if length(p) >= 2 && getpath(doc, p[1:(end - 1)]) isa AbstractDict]
            if !isempty(deep)
                q = first(deep)[1:(end - 1)]
                @test occursin("bogus", something(refusal(with(doc, [q; :bogus], 1), dir, "$(name)_deepbogus"; units), ""))
            end
            # a retired schema is refused by name: a lens kind's v1 and v2 for
            # their lens records (and rig/v3 with its successor), a typed-frame
            # kind's v1 and v2 for their free-form frame names, any other kind's
            # earlier version as retired; a kind born at v1 has nothing retired
            v1 = replace(doc[:schema], r"/v[2345]$" => "/v1")
            if name in ("target_poses", "target_points2d", "target_points3d")
                @test endswith(doc[:schema], "/v1")
            elseif name == "target"
                # the version before is migrated, not read: the refusal names the migration
                @test doc[:schema] == "vgio/target/v2"
                @test occursin("vgio_migrate", something(refusal(merge(doc, Dict(:schema => v1)), dir, "$(name)_v1"; units), ""))
            elseif name in ("transforms", "trajectory")
                @test occursin("free strings", something(refusal(merge(doc, Dict(:schema => replace(doc[:schema], "/v3" => "/v2"))), dir, "$(name)_v2"; units), ""))
                @test occursin("free strings", something(refusal(merge(doc, Dict(:schema => v1)), dir, "$(name)_v1"; units), ""))
            elseif startswith(name, "rig")
                @test occursin("lens records", something(refusal(merge(doc, Dict(:schema => v1)), dir, "$(name)_v1"; units), ""))
                @test occursin("lens records", something(refusal(merge(doc, Dict(:schema => "vgio/rig/v2")), dir, "$(name)_v2"; units), ""))
                @test occursin("rig/v5", something(refusal(merge(doc, Dict(:schema => "vgio/rig/v3")), dir, "$(name)_v3"; units), ""))
                @test occursin("vgio_migrate", something(refusal(merge(doc, Dict(:schema => "vgio/rig/v4")), dir, "$(name)_v4"; units), ""))
            elseif endswith(doc[:schema], "/v3")
                @test occursin("lens records", something(refusal(merge(doc, Dict(:schema => v1)), dir, "$(name)_v1"; units), ""))
                @test occursin("lens records", something(refusal(merge(doc, Dict(:schema => replace(doc[:schema], "/v3" => "/v2"))), dir, "$(name)_v2"; units), ""))
            else
                @test occursin("retired $(v1)", something(refusal(merge(doc, Dict(:schema => v1)), dir, "$(name)_v1"; units), ""))
            end
        end

        # a declaration is a TOKEN: anything else is refused, never reinterpreted
        tr = tomutable(parse_json(read(files["transforms"], String)))
        @test occursin("rotation_encoding", something(refusal(merge(tr, Dict(:rotation_encoding => "quaternion_xyzw")), dir, "tok1"), ""))
        @test occursin("transform_convention", something(refusal(merge(tr, Dict(:transform_convention => "x_from = R x_to + t")), dir, "tok2"), ""))
        @test occursin("length_unit", something(refusal(merge(tr, Dict(:length_unit => "px")), dir, "tok3"), ""))
        ob = tomutable(parse_json(read(files["target_points2d"], String)))
        @test occursin("point_index_base", something(refusal(merge(ob, Dict(:point_index_base => 0)), dir, "tok4"), ""))
        @test occursin("targets", something(refusal(merge(ob, Dict(:targets => Any[])), dir, "tok5"), ""))
        ge = tomutable(parse_json(read(files["generic_camera"], String)))
        @test occursin("grid_layout", something(refusal(merge(ge, Dict(:grid_layout => "column_major")), dir, "tok7"), ""))
        @test occursin("pixel_frame", something(refusal(merge(ge, Dict(:pixel_frame => "opencv")), dir, "tok8"), ""))
        # a non-central model's point grid is declared in a unit, and a central one has neither
        gn_doc = tomutable(parse_json(read(files["generic_noncentral"], String)))
        @test refusal(without(gn_doc, [:length_unit]), dir, "gn_unit") !== nothing
        @test refusal(without(gn_doc, [:point_grid]), dir, "gn_grid") !== nothing
        @test refusal(merge(ge, Dict(:length_unit => "mm")), dir, "gc_unit") !== nothing

        # a frame name is the grammar's: a free-form one is refused
        @test occursin("frame name", something(refusal(with(tr, [:transforms, 1, :from], "printed_board_a"), dir, "free_from"), ""))
        tj = tomutable(parse_json(read(files["trajectory"], String)))
        @test occursin("frame name", something(refusal(merge(tj, Dict(:to => "mocap_world")), dir, "free_to"), ""))
        # a time is seconds since a stated origin
        @test occursin("time_origin", something(refusal(without(tj, [:time_origin]), dir, "no_origin"), ""))
        # a covariance is symmetric and of the value's size
        bad = Any[Any[i == j ? 1.0 : 0.0 for j in 1:6] for i in 1:6]; bad[1][2] = 0.5
        @test occursin("symmetric", something(refusal(with(tr, [:transforms, 1, :covariance], bad), dir, "asym"), ""))
        @test refusal(with(tr, [:transforms, 1, :covariance], Any[Any[1.0, 0.0], Any[0.0, 1.0]]), dir, "cov2") !== nothing
        # the reserved `payload` (an Arrow sidecar of bulk arrays) is refused until a kind implements it
        @test occursin("payload", something(refusal(merge(tj, Dict(:payload => Dict(:file => "x.arrow", :sha256 => "0"))), dir, "payload"), ""))
        # the readers run the same table: a deletion refused by `validate` is refused by `load`
        @test_throws ArgumentError load(Dict{Symbol, BoundedCamera}, (write(joinpath(dir, "c1.json"), JSON.json(without(tomutable(parse_json(read(files["cameras"], String))), [:cameras, 1, :K, :skew]))); joinpath(dir, "c1.json")))
        @test_throws ArgumentError load(Trajectory, (write(joinpath(dir, "t1.json"), JSON.json(without(tomutable(parse_json(read(files["trajectory"], String))), [:to]))); joinpath(dir, "t1.json")))
        # a writer cannot emit what its reader refuses
        @test_throws ArgumentError save(joinpath(dir, "empty_targets.json"), TargetPoints2D{PixelFrame{:colmap}}(obs.frames; obs.images, obs.sensors))
        @test !isfile(joinpath(dir, "empty_targets.json"))
    end
end
