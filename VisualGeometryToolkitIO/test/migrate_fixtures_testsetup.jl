# The `MigrateFixtures` test setup (moved out of migrate_versions_tests.jl: ReTestItems reads setups from *_testsetup.jl files,
# so an item of another file finds it when that file runs alone).

@testsetup module MigrateFixtures
    import VisualGeometryToolkitIO
    import VisualGeometryToolkitIO as VgtkIO
    export units_fixture, as_v1_target, as_v1_points2d, as_v4_rig, old_copy, as_v4_observations, V4_ID, ID09, current_id,
        current_text, v09_tree

    # the IO 0.9 fixtures (test/fixtures/old/, the 0.9 current versions the
    # suite's units fixtures held), by their 0.9 names
    const _OLD09 = Dict("rig.json" => "rig_v9.json", "rig_targets.json" => "rig_v9_targets.json", "rig_pitch.json" => "rig_v9_pitch.json",
        "target_planar.json" => "target_v5_planar.json", "target_points.json" => "target_v5_points.json",
        "target_points2d.json" => "target_points2d_v2_plain.json", "target_points3d.json" => "target_points3d_v2_plain.json",
        "target_poses.json" => "target_poses_v2.json", "trajectory.json" => "trajectory_v4.json", "transforms.json" => "transforms_v4.json")
    units_fixture(name) = joinpath(@__DIR__, "fixtures", "old", _OLD09[name])

    # The fixtures ARE the migration of the files they were (git history before
    # the target record v2): the v1/v4 text is the current (v4/v7) text with the renamed
    # keys put back — `"calibration": {` was `"warp": {` (rigid only),
    # `"reference"` was `"design"`, the target file's `"calibration_record"`
    # was `"calibration"`.
    as_v4_rig(text) = replace(text, "\"schema\": \"vgio/rig/v9\"" => "\"schema\": \"vgio/rig/v4\"",
        "\"calibration\": {" => "\"warp\": {", "\"reference\":" => "\"design\":")
    as_v1_target(text) = replace(replace(text, "\"schema\": \"vgio/target/v5\"" => "\"schema\": \"vgio/target/v1\"",
        "\"calibration\": {" => "\"warp\": {", "\"reference\":" => "\"design\":"), "\"calibration_record\":" => "\"calibration\":")
    # target_points2d.json as vgio/target_points2d/v1 held it: the same record (covariances as rows), the earlier schema token
    as_v1_points2d(text) = replace(text, "\"schema\": \"vgio/target_points2d/v2\"" => "\"schema\": \"vgio/target_points2d/v1\"")
    # rig_targets.json's identity as vgio/rig/v4 defined it, which the
    # observations named before the migration, and its identity now
    const V4_ID = "a19adf2a1b6cb5d352dac3d6f1018f844cf9cf8f413ddf480539652e80957a88"
    # …its identity in IO 0.9 (the one target_points2d.json names), and its identity now (IO 0.10's re-id)
    const ID09 = "980c47d473610daa038b633cf5d4d133931699a9cafe1ee210f507b9b1c793b3"
    current_id() = VisualGeometryToolkitIO.content_id(units_fixture("rig_targets.json"))
    # the current file of a 0.9 fixture: its migration's text (a migrated old version must give the same bytes)
    function current_text(name)
        return mktempdir() do dir
            VgtkIO.vgio_migrate(units_fixture(name); out = joinpath(dir, name))
            read(joinpath(dir, name), String)
        end
    end
    # a value's IO 0.9 rig record of version `schema`, in `unit` (the shape the old versions' tests edit)
    function v09_tree(rig, schema; unit = VgtkIO.mm)
        d = VgtkIO._rig_record_v09(rig, (x, n) -> x, false; style = VgtkIO.VgioStyle(; length_unit = unit))
        merge!(d, Dict{String, Any}("schema" => schema, "length_unit" => VgtkIO.unit_string(unit),
            "rotation_encoding" => VgtkIO._ROTATION_ENCODING, "transform_convention" => VgtkIO._TRANSFORM_CONVENTION))
        haskey(d, "sensors") && (d["pixel_frame"] = "colmap")
        return d
    end
    # the observation fixtures as they were (target_points2d naming the v4 id;
    # target_points3d names no targets)
    function as_v4_observations(dir)
        write(joinpath(dir, "target_points2d.json"), replace(read(units_fixture("target_points2d.json"), String), ID09 => V4_ID))
        cp(units_fixture("target_points3d.json"), joinpath(dir, "target_points3d.json"))
        return joinpath(dir, "target_points2d.json")
    end
    function old_copy(dir, name, f)
        p = joinpath(dir, name)
        write(p, f(read(units_fixture(name), String)))
        return p
    end
end
