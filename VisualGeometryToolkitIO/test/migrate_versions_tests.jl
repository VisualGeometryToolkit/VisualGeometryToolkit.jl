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

@testitem "vgio_migrate: a full-precision length migrates, as the same metre value within 2 ulp" setup = [VGIoFixtures, MigrateFixtures] begin
    import VisualGeometryToolkitIO as VgtkIO
    # 287.15788802842013 mm has no metre value that writes back to its own
    # text: the migration compares values, so it migrates, the file's number
    # moved by at most 2 ulp and read as the identical metre value.
    x = 287.15788802842013
    mktempdir() do dir
        src = old_copy(dir, "target_planar.json", t -> replace(as_v1_target(t), "1234.5" => repr(x)))
        r = vgio_migrate(src)
        c = JSON.parsefile(r.out)["target"]["nominal"][4][1]
        @test c !== x && abs(c - x) <= 2 * eps(x)
        @test VgtkIO.read_length(c, mm) === VgtkIO.read_length(x, mm)
    end
end

@testitem "vgio_migrate: a few hundred random full-precision lengths and covariances all migrate" setup = [VGIoFixtures, MigrateFixtures] begin
    import VisualGeometryToolkitIO as VgtkIO
    using Random: Xoshiro
    rng = Xoshiro(20261027)
    n = 300
    pts = [[(rand(rng) - 0.5) * 10.0^rand(rng, 0:3) for _ in 1:3] for _ in 1:n]
    covs = [[[i == j ? rand(rng) * 10.0^rand(rng, -3:1) : 0.0 for j in 1:3] for i in 1:3] for _ in 1:n]
    num(x) = repr(x)                                          # full precision: the shortest text that is x
    vec_(v) = "[" * join(num.(v), ", ") * "]"
    text = """{"schema": "vgio/target/v1", "length_unit": "mm", "target": {"type": "points",
        "points": [$(join(vec_.(pts), ", "))],
        "point_covariances": [$(join(["[" * join(vec_.(c), ", ") * "]" for c in covs], ", "))]}}"""
    mktempdir() do dir
        src = joinpath(dir, "target.json")
        write(src, text)
        r = vgio_migrate(src)
        d = JSON.parsefile(r.out)["target"]
        moved = 0
        for (i, p) in enumerate(pts), k in 1:3
            c = d["points"][i][k]
            moved += c !== p[k]
            @test abs(c - p[k]) <= 2 * eps(abs(p[k])) && VgtkIO.read_length(c, mm) === VgtkIO.read_length(p[k], mm)
            c2 = d["point_covariances"][i][k][k]
            moved += c2 !== covs[i][k][k]
            @test abs(c2 - covs[i][k][k]) <= 2 * eps(covs[i][k][k]) && VgtkIO._read_power(c2, mm, 2) === VgtkIO._read_power(covs[i][k][k], mm, 2)
        end
        @test moved > 0                                           # the numbers the text check refused are in the set
    end
end

@testitem "_repoint_targets: a target_points3d file of full-precision lengths re-points, its values kept" setup = [VGIoFixtures, MigrateFixtures] begin
    import VisualGeometryToolkitIO as VgtkIO
    xs = (287.15788802842013, -1319.0000000000002, 0.30000000000000004)
    mktempdir() do dir
        obs = joinpath(dir, "points3d.json")
        write(obs, """{"schema": "vgio/target_points3d/v1", "length_unit": "mm", "frame": "world", "point_index_base": 1,
            "targets": ["$(V4_ID)"],
            "frames": [{"t": 1, "sensor": "mocap", "points": [{"target": "a1f2", "index": 2, "xyz": [$(join(repr.(xs), ", "))]}]}]}""")
        r = VgtkIO._repoint_targets(obs, Dict(V4_ID => current_id()), [units_fixture("rig_targets.json")])
        d = JSON.parsefile(r.out)
        @test r.repointed == 1 && d["targets"] == [current_id()]
        c = d["frames"][1]["points"][1]["xyz"]
        @test any(i -> c[i] !== xs[i], 1:3)
        @test all(i -> abs(c[i] - xs[i]) <= 2 * eps(abs(xs[i])) && VgtkIO.read_length(c[i], mm) === VgtkIO.read_length(xs[i], mm), 1:3)
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
        d = VgtkIO._repoint_targets(obs, ids, [r.out]; dry_run = true)
        @test d.out === nothing && d.repointed == 1 && d.unchecked == 0
        d = VgtkIO._repoint_targets(obs, ids, [r.out])
        @test read(d.out, String) == read(units_fixture("target_points2d.json"), String)
        @test load(TargetPoints2D, d.out).frames == load(TargetPoints2D, obs).frames
        # refused: a new identity none of the rigs has; a rig that does not hold the measured targets
        @test_throws ArgumentError VgtkIO._repoint_targets(obs, Dict(v4_id => "0"^64), [r.out]; out = joinpath(dir, "x.json"))
        @test_throws ArgumentError VgtkIO._repoint_targets(obs, Dict(v4_id => rs.new_id), [sensors]; out = joinpath(dir, "y.json"))
        @test !isfile(joinpath(dir, "x.json")) && !isfile(joinpath(dir, "y.json"))
    end
end

@testitem "vgio_migrate: a v4 rig in metres keeps both v4 identities, as v4 computed them" setup = [VGIoFixtures, MigrateFixtures] begin
    import VisualGeometryToolkitIO as VgtkIO
    # fixtures/v4/rig_metres.json: units/rig.json as a v4 file in metres, plus
    # a non-central generic sensor, with 6-digit metre values whose millimetre
    # number v4 computed differently from the decimal shift (0.0139 m was
    # 13.899999999999999 mm). Its identities were recorded by v4's own code:
    # `content_id(path)` and `content_id(load(Rig, path))`, run with
    # VisualGeometryToolkitIO at main ddc1bbe (FlexUnits 0.6.12).
    path_id = "1d3efbb1c7c7ceba0be3d5df04979f744829c8c12ecb41340d08d52391b8898e"
    value_id = "e20468bffbcb556f221c68cae447edc7b2bc368f33dabb1009c36a4c7a929d14"
    mktempdir() do dir
        src = joinpath(dir, "rig.json")
        cp(joinpath(@__DIR__, "fixtures", "v4", "rig_metres.json"), src)
        r = vgio_migrate(src)
        @test r.old_ids == [path_id, value_id] && r.new_id == content_id(r.out)
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
        # re-pointing: an identity neither mapped nor a given rig's is refused, naming the file and the id
        rig = vgio_migrate(joinpath(tree, "rig_targets.json"); out = joinpath(dir, "rig_v5.json"))
        e = try VgtkIO._repoint_targets(obs, Dict{String, String}(), [rig.out]; out = joinpath(dir, "x.json")); nothing catch e; e end
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

@testitem "vgio_migrate_tree: every refusal is logged and listed; the other files are migrated" setup = [VGIoFixtures, MigrateFixtures] begin
    import VisualGeometryToolkitIO as VgtkIO
    mktempdir() do dir
        # sorted first, so a refusal that aborted the walk would leave the good file unmigrated
        bad = joinpath(dir, "a_bad.json")                    # the current reader refuses it (an asymmetric covariance)
        write(bad, """{"schema": "vgio/target/v1", "length_unit": "mm", "target": {"type": "points", "points": [[0.0, 0.0, 0.0]],
            "point_covariances": [[[1.0, 0.5, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]]}}""")
        taken = joinpath(dir, "b_taken.json")                # its .migrated.json exists and is not a current file
        write(taken, as_v1_target(read(units_fixture("target_points.json"), String)))
        write(VgtkIO._migrated_path(taken), "not a vgio file")
        stray = joinpath(dir, "c_stray.json")                # names an identity no rig of the tree has
        write(stray, replace(read(units_fixture("target_points2d.json"), String), current_id() => "0"^64))
        good = old_copy(dir, "target_planar.json", as_v1_target)
        io = IOBuffer()
        e = try vgio_migrate_tree(dir; io); nothing catch e; e end
        out = String(take!(io))
        @test e isa ArgumentError && occursin("3 file(s) refused", e.msg)
        for f in (bad, taken, stray)
            @test occursin("FAILED    $(f): ", out) && occursin(f, e.msg)
        end
        @test read(VgtkIO._migrated_path(good), String) == read(units_fixture("target_planar.json"), String)
        @test !isfile(VgtkIO._migrated_path(bad)) && !isfile(VgtkIO._migrated_path(stray))
        @test read(VgtkIO._migrated_path(taken), String) == "not a vgio file"
    end
end

@testitem "vgio_migrate: staged beside the file, renamed in place; a refusal leaves the original" setup = [VGIoFixtures, MigrateFixtures] begin
    import VisualGeometryToolkitIO as VgtkIO
    mktempdir() do dir
        # a migration the current reader refuses (an asymmetric point covariance): the original byte for byte, nothing left beside it
        src = joinpath(dir, "target_planar.json")
        write(src, """{"schema": "vgio/target/v1", "length_unit": "mm", "target": {"type": "points", "points": [[0.0, 0.0, 0.0]],
            "point_covariances": [[[1.0, 0.5, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]]}}""")
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

@testitem "vgio_migrate: a tree logs what it skips — a staging directory, an unreadable file, an unknown version" setup = [VGIoFixtures, MigrateFixtures] begin
    import VisualGeometryToolkitIO as VgtkIO
    mktempdir() do dir
        tree = mkpath(joinpath(dir, "tree"))
        stage = mkpath(joinpath(tree, ".vgio_migrate_left"))           # left by an interrupted migration
        inside = old_copy(stage, "target_planar.json", as_v1_target)
        before = read(inside, String)
        write(joinpath(tree, "notes.json"), "{\"no\": \"schema\"}")
        write(joinpath(tree, "broken.json"), "{ not json")
        write(joinpath(tree, "old_rig.json"), "{\"schema\": \"vgio/rig/v3\"}")
        io = IOBuffer()
        vgio_migrate_tree(tree; in_place = true, io)
        out = String(take!(io))
        @test occursin("SKIPPED   $(stage): a staging directory", out)
        @test read(inside, String) == before                            # never walked
        @test occursin("SKIPPED   $(joinpath(tree, "notes.json")): no readable schema", out)
        @test occursin("SKIPPED   $(joinpath(tree, "broken.json")): no readable schema", out)
        @test occursin("SKIPPED   $(joinpath(tree, "old_rig.json")): vgio/rig/v3 has no migration", out)
    end
end

@testitem "write_staged: renamed into place once f returns; a failure leaves the destinations as they were" setup = [VGIoFixtures] begin
    import VisualGeometryToolkitIO as VgtkIO
    using VisualGeometryToolkitIO: write_staged
    staging(d) = filter(f -> startswith(f, ".vgio_migrate_"), readdir(d))
    mktempdir() do dir
        # success: two files, in a directory created for them, no staging left
        out = joinpath(dir, "new", "deeper")
        a, b = joinpath(out, "a.txt"), joinpath(out, "b.txt")
        seen = String[]
        r = write_staged(a => "a.txt", b => "b.txt"; dry_run = false) do sa, sb
            push!(seen, sa, sb)
            write(sa, "A"); write(sb, "B")
            :ok
        end
        @test r === :ok && read(a, String) == "A" && read(b, String) == "B"
        @test all(p -> dirname(p) != out && !ispath(p), seen) && isempty(staging(out)) && sort(readdir(out)) == ["a.txt", "b.txt"]
        # a dry run creates nothing anywhere
        dry = joinpath(dir, "dry")
        @test write_staged(joinpath(dry, "x.txt") => "x.txt"; dry_run = true) do sx
            write(sx, "X"); isfile(sx)
        end
        @test !ispath(dry)
        # failure in f: the error propagates; an existing destination is untouched, nothing staged remains
        e = try
            write_staged(a => "a.txt"; dry_run = false) do sa
                write(sa, "changed"); error("refused")
            end
            nothing
        catch e
            e
        end
        @test e isa ErrorException && e.msg == "refused" && read(a, String) == "A" && isempty(staging(out))
        # failure in f, the destination directory new: it is removed again (with what mkpath created)
        fresh = joinpath(dir, "fresh", "sub")
        @test_throws ErrorException write_staged(joinpath(fresh, "f.txt") => "f.txt"; dry_run = false) do sf
            write(sf, "F"); error("no")
        end
        @test !ispath(joinpath(dir, "fresh"))
        # a failing rename after an earlier one: the new file already renamed is removed again
        out2 = mkpath(joinpath(dir, "two"))
        blocker = joinpath(out2, "d.txt")
        mkpath(joinpath(blocker, "occupied"))          # a non-empty directory: renaming a file onto it fails
        @test_throws Exception write_staged(joinpath(out2, "c.txt") => "c.txt", blocker => "d.txt"; dry_run = false) do sc, sd
            write(sc, "C"); write(sd, "D")
        end
        @test !ispath(joinpath(out2, "c.txt")) && isdir(blocker) && isempty(staging(out2))
        # destinations in two directories: refused before anything is staged or written
        other = mkpath(joinpath(dir, "other"))
        called = Ref(false)
        e = try
            write_staged(joinpath(out2, "e.txt") => "e.txt", joinpath(other, "f.txt") => "f.txt"; dry_run = false) do se, sf
                called[] = true; write(se, "E"); write(sf, "F")
            end
            nothing
        catch e
            e
        end
        @test e isa ArgumentError && occursin("one directory", e.msg) && !called[]
        @test !ispath(joinpath(out2, "e.txt")) && isempty(readdir(other)) && isempty(staging(out2))
    end
end
