# =============================================================================
# vgio_migrate: target/v1 → v2 and rig/v4 → v5 by the guarded pattern, the
# measurement files re-pointed, the command, and the extension point
# =============================================================================

@testsetup module MigrateFixtures
    export units_fixture, as_v1_target, as_v4_rig, old_copy

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
        v4_id = "a19adf2a1b6cb5d352dac3d6f1018f844cf9cf8f413ddf480539652e80957a88"
        obs = joinpath(dir, "target_points2d.json")
        write(obs, replace(read(units_fixture("target_points2d.json"), String), only(obs_new.targets) => v4_id))
        e = try load(Rig, rig); nothing catch e; e end
        @test e isa ArgumentError && occursin("vgio/rig/v4", e.msg) && occursin("vgio_migrate", e.msg)
        r = vgio_migrate(rig)
        @test r.to == "vgio/rig/v5" && r.old_id == v4_id && r.new_id == content_id(r.out) == only(obs_new.targets)
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
        write(joinpath(tree, "target_points2d.json"), replace(read(units_fixture("target_points2d.json"), String),
            "eb9ad42d8920caa208a5ce708d0b1c7954c541cd65a857cc9ec97b540a8c00fe" => "a19adf2a1b6cb5d352dac3d6f1018f844cf9cf8f413ddf480539652e80957a88"))
        before = Dict(f => read(joinpath(tree, f), String) for f in readdir(tree))
        io = IOBuffer()
        @test VgtkIO._migrate_cli(["tree", tree]; io) == 0                      # a dry run
        @test occursin("verified", String(take!(io))) && Dict(f => read(joinpath(tree, f), String) for f in readdir(tree)) == before
        @test VgtkIO._migrate_cli(["rig", joinpath(tree, "target_planar.json")]; io) == 1      # not a rig
        @test occursin("FAILED", String(take!(io)))
        @test VgtkIO._migrate_cli(["tree", tree, "--in-place", "--apply"]; io) == 0
        out = String(take!(io))
        @test occursin("re-pointed", out) && occursin("a19adf2a", out)
        for f in readdir(tree)
            @test read(joinpath(tree, f), String) == read(units_fixture(f), String)
        end
        @test VgtkIO._migrate_cli(["tree", tree, "--in-place", "--apply"]; io) == 0   # current: nothing to do
        @test !occursin("migrated", String(take!(io)))
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
