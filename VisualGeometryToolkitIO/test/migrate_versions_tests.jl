# =============================================================================
# vgio_migrate: target/v1 → v3 and rig/v4 → v6 (and v2 → v3, v5 → v6) by the guarded pattern, the
# measurement files re-pointed, the command, and the extension point
# =============================================================================

@testsetup module MigrateFixtures
    import VisualGeometryToolkitIO
    export units_fixture, as_v1_target, as_v4_rig, old_copy, as_v4_measurements, V4_ID, current_id

    units_fixture(name) = joinpath(@__DIR__, "fixtures", "units", name)

    # The fixtures ARE the migration of the files they were (git history before
    # the target record v2): the v1/v4 text is the current (v3/v6) text with the renamed
    # keys put back — `"calibration": {` was `"warp": {` (rigid only),
    # `"reference"` was `"design"`, the target file's `"calibration_record"`
    # was `"calibration"`.
    as_v4_rig(text) = replace(text, "\"schema\": \"vgio/rig/v6\"" => "\"schema\": \"vgio/rig/v4\"",
        "\"calibration\": {" => "\"warp\": {", "\"reference\":" => "\"design\":")
    as_v1_target(text) = replace(replace(text, "\"schema\": \"vgio/target/v3\"" => "\"schema\": \"vgio/target/v1\"",
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
            @test r.out === nothing && r.from == "vgio/target/v1" && r.to == "vgio/target/v3"
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

@testitem "the migration compares typed records: a length within its guarantee, every other number bit for bit" setup = [VGIoFixtures, MigrateFixtures] begin
    import VisualGeometryToolkitIO as VgtkIO
    using VisualGeometryToolkitCore: Rig
    # the old file and a "new" file in mm that differs in one number by one ulp: a length is accepted (the
    # conversion's 2-ulp guarantee), a quaternion component or a focal length is refused — the former file-tree
    # comparison gave every number under a declared unit the length tolerance
    text = read(units_fixture("rig.json"), String)
    q, t, f = 0.9233805168766387, "251.9", "\"fx\": 900.0"
    @test occursin(repr(q), text) && occursin(t, text) && occursin(f, text)
    mktempdir() do dir
        rd(s) = (p = joinpath(dir, "r$(hash(s)).json"); write(p, s); load(Rig, p))
        old = rd(text)
        @test VgtkIO._same_value(old, rd(text))
        @test VgtkIO._same_value(old, rd(replace(text, t => repr(nextfloat(251.9)))))                     # a length: accepted
        @test !VgtkIO._same_value(old, rd(replace(text, repr(q) => repr(nextfloat(q)))))                 # a quaternion: refused
        @test !VgtkIO._same_value(old, rd(replace(text, f => "\"fx\": $(repr(nextfloat(900.0)))")))     # a focal length: refused
        @test !VgtkIO._same_value(old, rd(replace(text, t => "252.0")))                                  # a length beyond 2 ulp: refused
    end
    # a covariance entry by its parameters' length powers: a pose's rotation block bit for bit, its translation block within 2 ulp
    using VisualGeometryToolkitCore: EuclideanMap, Uncertain, CovMat
    using LinearAlgebra: I, Diagonal
    P = EuclideanMap(RotMatrix{3}(Matrix(1.0I, 3, 3)), SVector(1.0, 2.0, 3.0) .* VisualGeometryToolkitCore.m)
    Σ = Matrix(Diagonal(fill(1.0e-4, 6)))
    U(Σ) = Uncertain(P, CovMat{6, Float64, 36}(Tuple(Σ)))
    Σt = copy(Σ); Σt[5, 5] = nextfloat(Σt[5, 5])
    Σr = copy(Σ); Σr[2, 2] = nextfloat(Σr[2, 2])
    @test VgtkIO._same_value(U(Σ), U(Σt)) && !VgtkIO._same_value(U(Σ), U(Σr))
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
        @test r.to == "vgio/rig/v6" && r.old_id == v4_id && r.old_ids == [v4_id] && r.new_id == content_id(r.out) == only(obs_new.targets)   # an mm file: one v4 id
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

@testitem "vgio_migrate: a v4 rig's composed sensor lenses read in application order, as v4 read them" setup = [VGIoFixtures, MigrateFixtures] begin
    # v4 (like v5) listed a lens's "chain" in APPLICATION order: [s₁, s₂] is s₁
    # then s₂, the lens s₂ ∘ s₁. The migration to v6 (CT's order, the outer
    # first) must give the camera v4 read.
    import VisualGeometryToolkitIO as VgtkIO
    using VisualGeometryToolkitCore: Rig, BoundedCamera, PosedCamera, Tilt, ViewFrame, BodyFrame, relabel, model, project, backproject
    using JSON: JSON
    K = SMatrix{3, 3}(800.0, 0.0, 0.0, 0.0, 810.0, 0.0, 640.0, 480.0, 1.0)
    add14 = Additive(Radial((-0.1, 0.05, 0.01), (0.002, -5.5e-4, 7.0e-6)), Tangential(1.0e-4, -2.0e-4), ThinPrism((-2.2e-3, 1.2e-4), (1.0e-3, -2.0e-3)))
    tilt = Tilt(0.01, -0.02)
    rad, aria_add = Radial((0.1, -0.02, 0.003)), Additive(Tangential(3.0e-4, -2.0e-4), ThinPrism((1.0e-3,), (-5.0e-4,)))
    # the v4 stage lists, in application order, and the lenses v4 read from them (built by hand: s₂ ∘ s₁)
    stages = Dict(:opencv => (Perspective(), [add14, tilt]), :aria => (Equidistant(), [rad, aria_add]))
    expected = Dict(n => Lens(p, s[2] ∘ s[1]) for (n, (p, s)) in stages)
    I3 = EuclideanMap(one(RotMatrix{3, Float64}), [0.0mm, 0.0mm, 0.0mm])
    cam(n) = BoundedCamera(PosedCamera(CameraModel(K; lens = expected[n], image_origin = :colmap), relabel(BodyFrame{:camera_rig}, ViewFrame{n}, I3)), Size2(1280, 960))
    rig = Rig{BodyFrame{:camera_rig}}((cam(:opencv), cam(:aria)))
    mktempdir() do dir
        v6 = joinpath(dir, "rig_v6.json")
        save(v6, rig)
        d = JSON.parse(read(v6, String); dicttype = Dict{String, Any})
        d["schema"] = "vgio/rig/v4"
        # a stage's own term record: a one-stage lens's record without its projection
        stage_record(p, s) = (pk = only(keys(lens_record(Lens(p)))); Dict{String, Any}(k => v for (k, v) in lens_record(Lens(p, s)) if k != pk))
        for x in d["sensors"]
            p, ss = stages[Symbol(x["id"])]
            delete!(x["camera"]["lens"], "compose")
            x["camera"]["lens"]["chain"] = Any[stage_record(p, s) for s in ss]   # v4's key, application order
        end
        @test [only(keys(s)) for s in d["sensors"][1]["camera"]["lens"]["chain"]] == ["add", "tilt"]
        @test [only(keys(s)) for s in d["sensors"][2]["camera"]["lens"]["chain"]] == ["radial", "add"]
        v4 = joinpath(dir, "rig_v4.json")
        write(v4, JSON.json(d, 2))
        r = VgtkIO.vgio_migrate(v4; out = joinpath(dir, "rig_migrated.json"))
        @test r.to == "vgio/rig/v6" && read(r.out) == read(v6)
        @test length(r.old_ids) == 1 && r.new_id == VgtkIO.content_id(rig)   # an mm file: v4's two identities agree
        back = load(Rig, r.out)
        for (i, n) in enumerate((:opencv, :aria))
            got, want = model(back.members[i]), CameraModel(K; lens = expected[n], image_origin = :colmap)
            @test got.lens == expected[n]
            for x in -0.4:0.2:0.4, y in -0.3:0.15:0.3
                X = SVector(x, y, 1.0)
                @test project(got, X) == project(want, X)
                u = project(want, X)
                @test backproject(got, u) == backproject(want, u)
            end
        end
    end
end

@testitem "vgio_migrate: a nested lens chain is relisted at every depth" setup = [VGIoFixtures, MigrateFixtures] begin
    # a v3/v5 "chain" may hold a "chain" as a stage (Core reads a nested
    # composition): [{chain: [A₁, A₂]}, T] in application order is T ∘ (A₂ ∘ A₁).
    # Relisting only the outer list left the inner "chain", which the current
    # reader refuses.
    import VisualGeometryToolkitIO as VgtkIO
    using VisualGeometryToolkitCore: Rig, BoundedCamera, PosedCamera, Tilt, ViewFrame, BodyFrame, relabel, model
    using JSON: JSON
    K = SMatrix{3, 3}(800.0, 0.0, 0.0, 0.0, 810.0, 0.0, 640.0, 480.0, 1.0)
    A1, A2, T1 = Additive(Radial((-0.1, 0.05)), Tangential(1.0e-4, -2.0e-4)), Additive(ThinPrism((1.0e-3,), (-5.0e-4,))), Tilt(0.01, -0.02)
    L = Lens(Perspective(), T1 ∘ (A2 ∘ A1))
    term(x) = Dict{String, Any}(k => v for (k, v) in lens_record(Lens(Perspective(), x)) if k != "perspective")
    v5_lens = Dict{String, Any}("perspective" => Any[], "chain" => Any[Dict{String, Any}("chain" => Any[term(A1), term(A2)]), term(T1)])
    I3 = EuclideanMap(one(RotMatrix{3, Float64}), [0.0mm, 0.0mm, 0.0mm])
    cam = BoundedCamera(PosedCamera(CameraModel(K; lens = L, image_origin = :colmap), relabel(BodyFrame{:camera_rig}, ViewFrame{:c}, I3)), Size2(1280, 960))
    rig = Rig{BodyFrame{:camera_rig}}((cam,))
    @test VgtkIO._lens_v3_to_v4(v5_lens) == Dict{String, Any}("perspective" => Any[], "compose" => Any[term(T1), Dict{String, Any}("compose" => Any[term(A2), term(A1)])])
    @test VgtkIO._lens_v4_to_v3(VgtkIO._lens_v3_to_v4(v5_lens)) == v5_lens
    mktempdir() do dir
        v6 = joinpath(dir, "rig_v6.json")
        save(v6, rig)
        d = JSON.parse(read(v6, String); dicttype = Dict{String, Any})
        d["schema"] = "vgio/rig/v5"
        d["sensors"][1]["camera"]["lens"] = v5_lens
        v5 = joinpath(dir, "rig_v5.json")
        write(v5, JSON.json(d, 2))
        r = VgtkIO.vgio_migrate(v5; out = joinpath(dir, "rig_migrated.json"))
        @test model(load(Rig, r.out).members[1]).lens == L
        # the identity tree lists the stages in application order whatever the record's nesting: v5's id is the value's
        @test r.old_id == r.new_id == VgtkIO.content_id(rig)
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

@testitem "vgio_migrate: a v4 rig or v1 target held rigid warps only — any other is refused, never migrated" setup = [VGIoFixtures, MigrateFixtures] begin
    import VisualGeometryToolkitIO as VgtkIO
    # v4's writer refused a non-rigid warp and its reader knew none, so v4 has
    # no identity for one: a "v4" file holding an affine warp is not v4's
    mktempdir() do dir
        d = JSON.parse(read(joinpath(@__DIR__, "fixtures", "v4", "rig_metres.json"), String); dicttype = Dict{String, Any})
        d["targets"][1]["target"]["warp"] = Dict{String, Any}("type" => "affine", "parameters" => Any[1.0, 0.0, 0.0, 1.0, 0.001, 0.0])
        src = joinpath(dir, "rig.json")
        write(src, JSON.json(d, 2))
        e = try vgio_migrate(src); nothing catch e; e end
        @test e isa ArgumentError && occursin("rigid warp only", e.msg) && occursin("affine", e.msg)
        @test !isfile(VgtkIO._migrated_path(src))
        t = Dict{String, Any}("schema" => "vgio/target/v1", "length_unit" => "mm",
            "target" => Dict{String, Any}("type" => "planar", "design" => "b76e", "warp" => Dict{String, Any}("type" => "thin_plate_spline")))
        tsrc = joinpath(dir, "target.json")
        write(tsrc, JSON.json(t, 2))
        e = try vgio_migrate(tsrc); nothing catch e; e end
        @test e isa ArgumentError && occursin("rigid warp only", e.msg)
        # the rigid fixture still migrates
        cp(joinpath(@__DIR__, "fixtures", "v4", "rig_metres.json"), src; force = true)
        @test vgio_migrate(src).to == "vgio/rig/v6"
    end
end

@testitem "vgio_migrate: v4's value identity where v4's reader changed a number — a w < 0 quaternion, non-unit directions" setup = [VGIoFixtures, MigrateFixtures] begin
    import VisualGeometryToolkitIO as VgtkIO
    # fixtures/v4/rig_canonical_m.json: rig_metres.json with sensor `cam`'s quaternion negated (w < 0, the same
    # rotation) and three of the generic sensor's direction controls scaled by 1.1 (v4's reader renormalized them)
    # and one by 1 + 1e-13 (unit to 1e-12: v4 kept it); rig_canonical_mm.json the same numbers declared in mm (v4
    # converted an mm file through the same FlexUnits arithmetic). The identities were recorded by v4's own code:
    # `content_id(path)` and `content_id(load(Rig, path))`, run with VisualGeometryToolkitIO at main ddc1bbe
    # (FlexUnits 0.6.12) — the same run reproduced rig_metres.json's recorded pair.
    ids = Dict(
        "rig_canonical_m.json" => ("c94b8452e03947e1802da593e27069d3b2626039de9e9c9acf2fcac82f5f23b6",
            "fa7ea9e17d2a21ac0d78b774c1f27a476b67402fbd728347b85a6e5a9834a6f5"),
        "rig_canonical_mm.json" => ("0cdbe8e109ede4618e44ba18b93f3fdb6bd8df767d3006887ec73fe7edff502e",
            "b92649c972750df333f7621cf68c90cd76f4ca8b98d96887d5ae5258ab8351c6"),
    )
    mktempdir() do dir
        for (name, (path_id, value_id)) in ids
            src = joinpath(dir, name)
            cp(joinpath(@__DIR__, "fixtures", "v4", name), src)
            r = vgio_migrate(src)
            @test r.old_ids == [path_id, value_id] && r.new_id == content_id(r.out)
        end
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

@testitem "vgio_migrate: rig/v5 → v6 reverses a sensor lens's chain and a target's composition, and maps the id" setup = [VGIoFixtures] begin
    # A v5 rig is the v6 text with every composed record's members in
    # application order (CoordinateTransformations' order reversed) under v5's
    # keys: a sensor's lens "chain" (v6: "compose") and a target's
    # "calibration_chain" "members" with its covariance (v6: "composition",
    # "compose").
    # The target's v5 calibration is a chain [affine, height_field]: the affine
    # kind is retired, so it migrates to its factors — HeightField ∘ EuclideanMap
    # ∘ Scale ∘ Aspect ∘ Skew, the same map — its covariance through the
    # factorization's Jacobian.
    import VisualGeometryToolkitIO as VgtkIO
    using VisualGeometryToolkitCore: Rig, BoundedCamera, CameraModel, PosedCamera, PosedTarget, PlanarTarget, HeightField,
        Lens, Perspective, Additive, Radial, Tangential, Tilt, Size2, ViewFrame, BodyFrame, Uncertain, CovMat, grid_nodes, relabel, mm,
        calibration, parameter_vector, Scale, Aspect, Skew, geom
    using LinearAlgebra: Diagonal
    import LinearAlgebra
    using JSON: JSON
    K = SMatrix{3, 3}(800.0, 0.0, 0.0, 0.0, 810.0, 0.0, 640.0, 480.0, 1.0)
    lens = Lens(Perspective(), Tilt(0.01, -0.02) ∘ Additive(Radial((-0.12, 0.03)), Tangential(1.0e-3, -5.0e-4)))
    I3 = EuclideanMap(one(RotMatrix{3, Float64}), [0.0mm, 0.0mm, 0.0mm])
    cam = BoundedCamera(PosedCamera(CameraModel(K; lens, image_origin = :colmap), relabel(BodyFrame{:camera_rig}, ViewFrame{:cam}, I3)), Size2(1280, 960))
    nodes = [VgtkIO.read_length.(p, mm) for p in grid_nodes(3; lo = 0.0, hi = 100.0)]
    fb(x) = VgtkIO.read_length(VgtkIO.write_length(x * VisualGeometryToolkitCore.m, mm), mm)    # file-born
    H = HeightField(nodes, fb.(parameter_vector(HeightField(nodes, VgtkIO.read_length.([0.1, 0.0, -0.1, 0.2, 0.0, 0.0, 0.0, 0.3, 0.0], mm)))))
    pts = [SVector(0.0, 0.0) .* mm, SVector(100.0, 50.0) .* mm]
    rig = Rig{BodyFrame{:camera_rig}}((cam,), Dict(:t => PosedTarget(PlanarTarget(pts; calibration = H), I3)))
    A, b = (1.001, 0.002, -0.003, 0.999), (0.5, -0.25)       # a11, a12, a21, a22 (row by row); b in the file's mm
    Σold = Diagonal([1.0e-8 * i for i in 1:15])              # over [affine (6); height field (9)], the file's units
    mktempdir() do dir
        v6 = joinpath(dir, "rig.json")
        VgtkIO.save(v6, rig)
        d = JSON.parse(read(v6, String); dicttype = Dict{String, Any})
        @test [only(keys(x)) for x in d["sensors"][1]["camera"]["lens"]["compose"]] == ["tilt", "add"]        # the outer first
        # the v5 file: v5's keys, in application order; the calibration a chain [affine, height_field]
        d["schema"] = "vgio/rig/v5"
        l = d["sensors"][1]["camera"]["lens"]
        l["chain"] = reverse(pop!(l, "compose"))
        hrec = d["targets"][1]["target"]["calibration"]
        d["targets"][1]["target"]["calibration"] = Dict{String, Any}("type" => "calibration_chain",
            "members" => Any[Dict{String, Any}("type" => "affine", "parameters" => Any[A..., b...]), hrec],
            "covariance" => Any[Any[Σold[i, j] for j in 1:15] for i in 1:15])
        v5 = joinpath(dir, "rig_v5.json")
        write(v5, JSON.json(d, 2))
        e = try VgtkIO.load(Rig, v5); nothing catch e; e end
        @test e isa ArgumentError && occursin("vgio_migrate", e.msg)
        r = VgtkIO.vgio_migrate(v5; out = joinpath(dir, "rig_migrated.json"))
        @test r.to == "vgio/rig/v6"
        # the value changed kind (affine → its factors), so the identity moves: v5's is reported as the old one
        @test r.new_id == VgtkIO.content_id(r.out) && r.old_id != r.new_id && r.old_ids == [r.old_id]
        back = VgtkIO.load(Rig, r.out)
        @test VisualGeometryToolkitCore.model(only(back.members.sensors)).lens == lens          # the sensor's lens, in CT's order
        cu = calibration(back[:t].target)
        c = geom(cu)
        ms = VisualGeometryToolkitCore.members(c)
        @test ms[1] == H && ms[2] isa EuclideanMap && ms[3] isa Scale && ms[4] isa Aspect && ms[5] isa Skew
        # the same map: H(A·p + b), in metres
        Am, bm = SMatrix{2, 2}(A[1], A[3], A[2], A[4]), SVector(b...) ./ 1000
        for q in (SVector(0.0, 0.0), SVector(0.1, 0.05), SVector(-0.02, 0.08))
            @test ustrip.(VisualGeometryToolkitCore.m, c(q .* VisualGeometryToolkitCore.m)) ≈ ustrip.(VisualGeometryToolkitCore.m, H((Am * q + bm) .* VisualGeometryToolkitCore.m)) atol = 1.0e-14
        end
        # the covariance, over [height field (9); ϑ, tx, ty, log s, a₁, u₁₂] in the file's units: J Σ Jᵀ, J by central differences
        f(x) = (θ = VisualGeometryToolkitCore.iwasawa_parameters(SMatrix{2, 2}(x[1], x[3], x[2], x[4])); [θ[1], x[5], x[6], θ[2], θ[3], θ[4]])
        x0 = [A..., b...]
        Jfd = reduce(hcat, [(h = zeros(6); h[k] = 1.0e-6; (f(x0 .+ h) .- f(x0 .- h)) ./ 2.0e-6) for k in 1:6])
        @test VgtkIO._member_v2_to_v3(Dict{String, Any}("type" => "affine", "parameters" => Any[x0...])).J ≈ Jfd rtol = 1.0e-8
        Jt = zeros(15, 15)
        Jt[1:9, 7:15] = Matrix{Float64}(LinearAlgebra.I, 9, 9)
        Jt[10:15, 1:6] = Jfd
        rows = JSON.parse(read(r.out, String))["targets"][1]["target"]["calibration"]["covariance"]
        @test [rows[i][j] for i in 1:15, j in 1:15] ≈ Jt * Σold * Jt' rtol = 1.0e-6
    end
end

@testitem "vgio_migrate: an affine calibration is its Iwasawa factors — the same map, det A > 0 required" setup = [VGIoFixtures] begin
    import VisualGeometryToolkitCore
    using VisualGeometryToolkitCore: Scale, Aspect, Skew
    for A in ((1.001, 0.002, -0.003, 0.999), (0.7, -1.3, 2.1, 0.4), (-1.2, 0.3, -0.1, -0.9), (2.0, 0.0, 0.0, 0.5))
        M = SMatrix{2, 2}(A[1], A[3], A[2], A[4])
        ϑ, k, a, u = VisualGeometryToolkitCore.iwasawa_parameters(M)
        F = SMatrix(RotMatrix{2, Float64}(ϑ)) * exp(k) * SMatrix{2, 2}(exp(a), 0.0, 0.0, exp(-a)) * SMatrix{2, 2}(1.0, 0.0, u, 1.0)
        @test F ≈ M rtol = 1.0e-14
        c = EuclideanMap(RotMatrix{2, Float64}(ϑ), SVector(0.0, 0.0); check = false) ∘ Scale(k) ∘ Aspect(a) ∘ Skew(u)
        @test c(SVector(0.3, -0.7)) ≈ M * SVector(0.3, -0.7) rtol = 1.0e-14
    end
    e = try VisualGeometryToolkitCore.iwasawa_parameters(SMatrix{2, 2}(1.0, 0.0, 0.0, -1.0)); nothing catch e; e end   # a reflection
    @test e isa ArgumentError && occursin("det A", e.msg)
end

@testitem "vgio_migrate: a v5 rig of every target kind and its v6 migration have one content_id, the id v5 had" setup = [VGIoFixtures, TargetRecordFixtures] begin
    # persisted-geometry.md: "An identity does not change when the file format
    # does". The identity tree lists a composition's members (and a lens's
    # stages) in application order, the order v5's record held, whatever order
    # the record lists them in.
    import VisualGeometryToolkitIO as VgtkIO
    using VisualGeometryToolkitCore: Rig
    using JSON: JSON
    # identity_rig() without its affine target (`b_affine` holds the factors,
    # which v5 never wrote; the affine kind's migration and the old id it
    # reports are checked against the chain-equivalence goldens): every other
    # kind, its v5 form — the current record with v5's names and order — of
    # the same identity, pinned.
    using VisualGeometryToolkitCore: TargetFrame, ids
    v5_ids = Dict(false => "531c4aac5d154481e35f144c48fe7450711e35f88497def8c70ca5190fa58ae0",
        true => "5c82c4d9d13a2abdbcb6af39163b6a6372f22eb5d0ceb5c2c697f22e04cb87c4")
    mktempdir() do dir
        for boards in (false, true), length_unit in (mm, m)
            full = identity_rig(; boards)
            rig = Rig{TargetFrame{:a_rigid}}(Dict(id => full[id] for id in ids(full) if id != :b_affine))
            v6 = joinpath(dir, "rig_v6.json")
            VgtkIO.save(v6, rig; length_unit)
            d = JSON.parse(read(v6, String); dicttype = Dict{String, Any})
            d["schema"] = "vgio/rig/v5"
            for x in d["targets"]
                c = get(x["target"], "calibration", nothing)
                c isa AbstractDict && (x["target"]["calibration"] = VgtkIO._composition_v3_to_v2(c))
            end
            v5 = joinpath(dir, "rig_v5.json")
            write(v5, JSON.json(d, 2))
            r = VgtkIO.vgio_migrate(v5; out = joinpath(dir, "rig_migrated_$(boards)_$(length_unit).json"))
            @test read(r.out) == read(v6)
            @test r.old_id == r.new_id == VgtkIO.content_id(r.out) == VgtkIO.content_id(rig) == v5_ids[boards]
        end
    end
end

@testitem "vgio_migrate: a v2 chain with no affine member permutes its covariance exactly, Inf and -0.0 included" begin
    using Test
    import VisualGeometryToolkitIO as VgtkIO
    # every member carried over as it is: a pure reorder of the covariance's
    # blocks, never J Σ Jᵀ (0·Inf would be NaN, and -0.0 + 0.0 is +0.0)
    nodes = Any[Any[0.0, 0.0], Any[1.0, 0.0], Any[0.0, 1.0], Any[1.0, 1.0]]
    tps = Dict{String, Any}("type" => "thin_plate_spline", "nodes" => nodes, "parameters" => Any[zeros(8)...])
    hf = Dict{String, Any}("type" => "height_field", "nodes" => nodes, "parameters" => Any[zeros(4)...])
    n = 12
    Σ = [i == j ? Float64(i) : 0.01 * (i + j) for i in 1:n, j in 1:n]
    Σ[1, 1] = Inf
    Σ[2, 9] = Σ[9, 2] = -0.0
    Σ[10, 3] = Σ[3, 10] = -0.0
    chain = Dict{String, Any}("type" => "calibration_chain", "members" => Any[tps, hf],   # application order: inner first
        "covariance" => Any[Any[Σ[i, j] for j in 1:n] for i in 1:n])
    out = VgtkIO._calibration_v2_to_v3(chain)
    @test [r["type"] for r in out["compose"]] == ["height_field", "thin_plate_spline"]
    perm = [9:12; 1:8]                                   # the height field's block first, as it is listed
    S = out["covariance"]
    @test all(isequal(S[i][j], Σ[perm[i], perm[j]]) for i in 1:n, j in 1:n)   # bit for bit: Inf stays Inf, -0.0 keeps its sign
end
