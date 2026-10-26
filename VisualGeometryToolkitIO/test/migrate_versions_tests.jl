# =============================================================================
# vgio_migrate: target/v1 → v2 and rig/v4 → v5 by the guarded pattern, the
# measurement files re-pointed, the command, and the extension point
# =============================================================================

@testsetup module MigrateFixtures
    import VisualGeometryToolkitIO
    export units_fixture, as_v1_target, as_v4_rig, old_copy, as_v4_measurements, V4_ID, current_id

    units_fixture(name) = joinpath(@__DIR__, "fixtures", "units", name)

    # The fixtures ARE the migration of the files they were (git history before
    # the target record v2): the v1/v4 text is the v2/v5 text with the renamed
    # keys put back — `"calibration": {` was `"warp": {` (rigid only),
    # `"reference"` was `"design"`, the target file's `"calibration_record"`
    # was `"calibration"`.
    as_v4_rig(text) = replace(text, "\"schema\": \"vgio/rig/v5\"" => "\"schema\": \"vgio/rig/v4\"",
        "\"calibration\": {" => "\"warp\": {", "\"reference\":" => "\"design\":")
    as_v1_target(text) = replace(replace(text, "\"schema\": \"vgio/target/v2\"" => "\"schema\": \"vgio/target/v1\"",
        "\"calibration\": {" => "\"warp\": {", "\"reference\":" => "\"design\":"), "\"calibration_record\":" => "\"calibration\":")
    # rig_targets.json's identity as vgio/rig/v4 defined it, which the
    # measurements named before the migration, and its identity now
    const V4_ID = "a19adf2a1b6cb5d352dac3d6f1018f844cf9cf8f413ddf480539652e80957a88"
    current_id() = VisualGeometryToolkitIO.content_id(units_fixture("rig_targets.json"))
    # the measurement fixtures as they were (target_points2d naming the v4 id;
    # target_points3d names no targets)
    function as_v4_measurements(dir)
        write(joinpath(dir, "target_points2d.json"), replace(read(units_fixture("target_points2d.json"), String), current_id() => V4_ID))
        cp(units_fixture("target_points3d.json"), joinpath(dir, "target_points3d.json"))
        return joinpath(dir, "target_points2d.json")
    end
    function old_copy(dir, name, f)
        p = joinpath(dir, name)
        write(p, f(read(units_fixture(name), String)))
        return p
    end
end

@testitem "vgio_migrate: a v1 target reads back as the original, number for number" setup = [VGIoFixtures, MigrateFixtures] begin
    import VisualGeometryToolkitIO as VgtkIO
    using VisualGeometryToolkitCore: AbstractTarget, Attributed
    mktempdir() do dir
        for name in ("target_planar.json", "target_points.json")
            src = old_copy(dir, name, as_v1_target)
            before = read(src, String)
            @test occursin("vgio/target/v1", before) && occursin("\"warp\"", before) == (name == "target_planar.json")
            # no reader: refused, naming the migration
            e = try load(AbstractTarget, src); nothing catch e; e end
            @test e isa ArgumentError && occursin("vgio_migrate", e.msg)
            # a dry run verifies and writes nothing
            r = vgio_migrate(src; dry_run = true)
            @test r.out === nothing && r.from == "vgio/target/v1" && r.to == "vgio/target/v2"
            @test !isfile(VgtkIO._migrated_path(src))
            # beside the original, which is untouched
            r = vgio_migrate(src)
            @test r.out == VgtkIO._migrated_path(src) && read(src, String) == before
            @test read(r.out, String) == read(units_fixture(name), String)     # the committed migration, byte for byte
            a, b = load(AbstractTarget, r.out), load(AbstractTarget, units_fixture(name))
            @test a isa Attributed ? (getfield(a, :value) == getfield(b, :value) && getfield(a, :attributes) == getfield(b, :attributes)) : a == b
            @test_throws ArgumentError vgio_migrate(src)                       # an existing out is refused
            # in place: replaced once verified
            vgio_migrate(src; out = src)
            @test read(src, String) == read(units_fixture(name), String)
            # a current file has no migration
            @test_throws ArgumentError vgio_migrate(src; out = joinpath(dir, "again.json"))
        end
        t = load(AbstractTarget, units_fixture("target_planar.json"))
        @test t isa Attributed && t.calibration_record == (; tool = "board", n_images = 12)   # the v1 block, renamed
    end
end

@testitem "vgio_migrate: a number that cannot come back refuses the migration" setup = [VGIoFixtures, MigrateFixtures] begin
    import VisualGeometryToolkitIO as VgtkIO
    # 287.15788802842013 mm has no metre value that writes back to it: the
    # migration cannot reproduce the file's numbers, so it writes nothing.
    mktempdir() do dir
        src = old_copy(dir, "target_planar.json", t -> replace(as_v1_target(t), "1234.5" => "287.15788802842013"))
        e = try vgio_migrate(src); nothing catch e; e end
        @test e isa ArgumentError && occursin("does not read back", e.msg) && occursin("nothing was written", e.msg)
        @test !isfile(VgtkIO._migrated_path(src))
    end
end

@testitem "vgio_migrate: a v4 rig, its content identity, and the measurements re-pointed" setup = [VGIoFixtures, MigrateFixtures] begin
    import VisualGeometryToolkitIO as VgtkIO
    using VisualGeometryToolkitCore: Rig, TargetPoints2D
    mktempdir() do dir
        rig = old_copy(dir, "rig_targets.json", as_v4_rig)
        sensors = old_copy(dir, "rig.json", as_v4_rig)
        obs_new = load(TargetPoints2D, units_fixture("target_points2d.json"))
        # the measurements as they were: naming the v4 rig by its v4 identity
        v4_id = V4_ID
        obs = joinpath(dir, "target_points2d.json")
        write(obs, replace(read(units_fixture("target_points2d.json"), String), only(obs_new.targets) => v4_id))
        e = try load(Rig, rig); nothing catch e; e end
        @test e isa ArgumentError && occursin("vgio/rig/v4", e.msg) && occursin("vgio_migrate", e.msg)
        r = vgio_migrate(rig)
        @test r.to == "vgio/rig/v5" && r.old_id == v4_id && r.old_ids == [v4_id] && r.new_id == content_id(r.out) == only(obs_new.targets)   # an mm file: one v4 id
        @test read(r.out, String) == read(units_fixture("rig_targets.json"), String)
        rs = vgio_migrate(sensors; out = sensors)
        @test read(sensors, String) == read(units_fixture("rig.json"), String)
        @test load(Rig, sensors).members == load(Rig, units_fixture("rig.json")).members
        # re-pointing: guarded, the measurements otherwise unchanged
        ids = Dict(r.old_id => r.new_id)
        d = repoint_targets(obs, ids, [r.out]; dry_run = true)
        @test d.out === nothing && d.repointed == 1 && d.unchecked == 0
        d = repoint_targets(obs, ids, [r.out])
        @test read(d.out, String) == read(units_fixture("target_points2d.json"), String)
        @test load(TargetPoints2D, d.out).frames == load(TargetPoints2D, obs).frames
        # refused: a new identity none of the rigs has; a rig that does not hold the measured targets
        @test_throws ArgumentError repoint_targets(obs, Dict(v4_id => "0"^64), [r.out]; out = joinpath(dir, "x.json"))
        @test_throws ArgumentError repoint_targets(obs, Dict(v4_id => rs.new_id), [sensors]; out = joinpath(dir, "y.json"))
        @test !isfile(joinpath(dir, "x.json")) && !isfile(joinpath(dir, "y.json"))
    end
end

@testitem "vgio_migrate: the command, and a tree" setup = [VGIoFixtures, MigrateFixtures] begin
    import VisualGeometryToolkitIO as VgtkIO
    mktempdir() do dir
        tree = mkpath(joinpath(dir, "tree"))
        for name in ("rig.json", "rig_targets.json")
            old_copy(tree, name, as_v4_rig)
        end
        for name in ("target_planar.json", "target_points.json")
            old_copy(tree, name, as_v1_target)
        end
        as_v4_measurements(tree)
        before = Dict(f => read(joinpath(tree, f), String) for f in readdir(tree))
        io = IOBuffer()
        @test VgtkIO._migrate_cli(["tree", tree]; io) == 0                      # a dry run
        @test occursin("verified", String(take!(io))) && Dict(f => read(joinpath(tree, f), String) for f in readdir(tree)) == before
        @test VgtkIO._migrate_cli(["rig", joinpath(tree, "target_planar.json")]; io) == 1      # not a rig
        @test occursin("FAILED", String(take!(io)))
        @test VgtkIO._migrate_cli(["tree", tree, "--in-place", "--apply"]; io) == 0
        out = String(take!(io))
        @test occursin("re-pointed", out) && occursin(V4_ID, out)
        for f in readdir(tree)
            f == "vgio_migrate_ids.json" && continue
            @test read(joinpath(tree, f), String) == read(units_fixture(f), String)
        end
        @test VgtkIO._read_ids(joinpath(tree, "vgio_migrate_ids.json"))[V4_ID] == current_id()   # the map, persisted
        @test VgtkIO._migrate_cli(["tree", tree, "--in-place", "--apply"]; io) == 0   # current: nothing to do
        out = String(take!(io))
        @test !occursin("migrated", out) && !occursin("re-pointed", out)
        @test occursin("current   $(joinpath(tree, "rig.json"))", out) && occursin("current   $(joinpath(tree, "target_points2d.json")): names current rigs", out)
        @test occursin("current   $(joinpath(tree, "target_points3d.json")): names no targets", out)
        @test VgtkIO._migrate_cli(["target"]; io) == 2
    end
end

@testitem "vgio_migrate: a schema owner adds its migration (migrate_schema)" setup = [VGIoFixtures] begin
    import VisualGeometryToolkitIO as VgtkIO
    # a downstream owner's schema, migrated by vgio_migrate without VGIo knowing it
    VgtkIO.migrate_schema(::Val{Symbol("toy/thing/v1")}, src::AbstractString, dst::AbstractString) =
        (write(dst, replace(read(src, String), "toy/thing/v1" => "toy/thing/v2")); (; to = "toy/thing/v2"))
    @test VgtkIO.migrate_schema in VgtkIO.CONTRACT_GENERICS
    mktempdir() do dir
        p = joinpath(dir, "thing.json")
        write(p, """{"schema": "toy/thing/v1"}""")
        r = vgio_migrate(p)
        @test r.to == "toy/thing/v2" && occursin("toy/thing/v2", read(r.out, String))
        write(p, """{"schema": "toy/other/v1"}""")
        @test_throws ArgumentError vgio_migrate(p; out = joinpath(dir, "o.json"))   # no migration for it
    end
end

@testitem "vgio_migrate: no stale identity is ever kept; an interrupted tree resumes from its persisted map" setup = [VGIoFixtures, MigrateFixtures] begin
    import VisualGeometryToolkitIO as VgtkIO
    using VisualGeometryToolkitCore: TargetPoints2D
    mktempdir() do dir
        tree = mkpath(joinpath(dir, "tree"))
        old_copy(tree, "rig_targets.json", as_v4_rig)
        obs = as_v4_measurements(tree)
        # repoint_targets: an identity neither mapped nor a given rig's is refused, naming the file and the id
        rig = vgio_migrate(joinpath(tree, "rig_targets.json"); out = joinpath(dir, "rig_v5.json"))
        e = try repoint_targets(obs, Dict{String, String}(), [rig.out]; out = joinpath(dir, "x.json")); nothing catch e; e end
        @test e isa ArgumentError && occursin(obs, e.msg) && occursin(V4_ID, e.msg) && !isfile(joinpath(dir, "x.json"))
        rm(rig.out)
        # a tree whose run was interrupted after its rigs were migrated in place: the
        # measurements were not seen (here: not yet .json), the map was persisted
        mv(obs, obs * ".later")
        io = IOBuffer()
        vgio_migrate_tree(tree; in_place = true, io)
        @test VgtkIO._read_ids(joinpath(tree, "vgio_migrate_ids.json")) == Dict(V4_ID => current_id())
        mv(obs * ".later", obs)
        # the rerun finds every rig current, and re-points from the persisted map
        out = (vgio_migrate_tree(tree; in_place = true, io); String(take!(io)))
        @test occursin("current   $(joinpath(tree, "rig_targets.json"))", out) && occursin("re-pointed  $(obs)", out)
        @test only(load(TargetPoints2D, obs).targets) == current_id()
        # a measurement file naming an identity no rig of the tree has: FAILED, nothing written for it, the run an error
        stray = joinpath(tree, "stray.json")
        write(stray, replace(read(units_fixture("target_points2d.json"), String), current_id() => "0"^64))
        before = read(stray, String)
        e = try vgio_migrate_tree(tree; in_place = true, io); nothing catch e; e end
        out = String(take!(io))
        @test e isa ArgumentError && occursin(stray, e.msg) && occursin("FAILED    $(stray)", out) && occursin("0"^64, out)
        @test read(stray, String) == before
    end
end

@testitem "vgio_migrate: staged beside the file, renamed in place; a refusal leaves the original" setup = [VGIoFixtures, MigrateFixtures] begin
    import VisualGeometryToolkitIO as VgtkIO
    mktempdir() do dir
        # a migration that cannot reproduce the file's numbers: refused, the original byte for byte, nothing left beside it
        src = old_copy(dir, "target_planar.json", t -> replace(as_v1_target(t), "1234.5" => "287.15788802842013"))
        before = read(src, String)
        @test_throws ArgumentError vgio_migrate(src; out = src)
        @test read(src, String) == before && readdir(dir) == ["target_planar.json"]
        # a migration in place: the file replaced, no staging directory left
        src = old_copy(dir, "target_points.json", as_v1_target)
        vgio_migrate(src; out = src)
        @test read(src, String) == read(units_fixture("target_points.json"), String)
        @test sort(readdir(dir)) == ["target_planar.json", "target_points.json"]
        # beside it: the original untouched; a tree without in_place reruns (the migrated file is current)
        tree = mkpath(joinpath(dir, "tree"))
        o = old_copy(tree, "target_planar.json", as_v1_target)
        io = IOBuffer()
        vgio_migrate_tree(tree; io)
        vgio_migrate_tree(tree; io)
        out = String(take!(io))
        @test occursin("already migrated to $(VgtkIO._migrated_path(o))", out)
        @test read(o, String) == as_v1_target(read(units_fixture("target_planar.json"), String))
    end
end
