# =============================================================================
# Lengths in files: quantities in memory, the declared unit on disk
# =============================================================================
#
# The fixtures under test/fixtures/units/ were written by the writer of
# VisualGeometryToolkitIO before lengths became quantities (in-memory plain
# millimetres, commit ddc1bbe). Read into metres and written again, every one
# must come back byte for byte: a file of an existing schema version does not
# change because memory moved to metres.

@testitem "units: every fixture of a physical kind reads and rewrites byte for byte" setup = [VGIoFixtures] begin
    import VisualGeometryToolkitCore as VgtkCore
    import VisualGeometryToolkitIO as VgtkIO
    using VisualGeometryToolkitCore: Rig, AbstractTarget, Trajectory, TargetPoses, TargetPoints2D, TargetPoints3D,
        NoncentralGenericModel, has_units, geom, pose, translation
    using Statistics: cov

    fixtures = joinpath(@__DIR__, "fixtures", "units")
    kinds = Dict(
        "transforms.json" => Vector{EuclideanMap}, "rig.json" => Rig, "rig_targets.json" => Rig,
        "rig_pitch.json" => Rig,
        "target_points.json" => AbstractTarget, "target_planar.json" => AbstractTarget,
        "trajectory.json" => Trajectory, "target_poses.json" => TargetPoses,
        "target_points3d.json" => TargetPoints3D, "generic_camera.json" => NoncentralGenericModel,
        "target_points2d.json" => TargetPoints2D,
    )
    # every fixture is covered: the table above and correspondences.json (an export, below)
    @test sort(readdir(fixtures)) == sort([collect(keys(kinds)); "correspondences.json"])
    mktempdir() do dir
        for (name, T) in kinds
            src = joinpath(fixtures, name)
            value = load(T, src)
            out = joinpath(dir, name)
            save(out, value)
            @test read(out, String) == read(src, String)
        end
        # the export, regenerated from the fixtures it was written from
        obs = load(TargetPoints2D, joinpath(fixtures, "target_points2d.json"))
        boards = load(Rig, joinpath(fixtures, "rig_targets.json"))
        out = joinpath(dir, "correspondences.json")
        save_correspondences(PixelFrame{:opencv}, out, obs, boards)
        @test read(out, String) == read(joinpath(fixtures, "correspondences.json"), String)
        # the content identity of a rig is unchanged with it
        @test content_id(boards) == content_id(joinpath(fixtures, "rig_targets.json")) == only(obs.targets)
    end

    # what was read is physical, in metres — each the metre value nearest the
    # file's decimal (13.9 mm is 0.0139 m, not 0.013900000000000001)
    poses = load(Vector{EuclideanMap}, joinpath(fixtures, "transforms.json"))
    @test all(has_units, poses)
    @test map(q -> ustrip(m, q), translation(geom(poses[1]))) === SVector(0.2519, -0.1444, 0.9041)
    @test ustrip(m, translation(geom(poses[3]))[3]) === 0.0139
    Σ = cov(poses[2])                                       # rad², m·rad, m²
    @test Σ[4, 4] === 2.5e-7 && Σ[1, 4] === 3.5e-8 && Σ[1, 1] === 1.0e-6
    rig = load(Rig, joinpath(fixtures, "rig.json"))
    @test has_units(rig) && has_units(rig[:markers].target)
    @test ustrip(m, geom(VgtkCore.points(rig[:a1f2].target)[3])[1]) === -0.0139
    @test cov(VgtkCore.points(rig[:markers].target)[1])[3, 3] === 1.23e-8
    @test signbit(ustrip(m, geom(VgtkCore.points(rig[:markers].target)[2])[2]))    # -0.0 keeps its sign
    gn = load(NoncentralGenericModel, joinpath(fixtures, "generic_camera.json"))
    @test gn.points[2, 3][2] === -0.0005                    # plain metres
end

@testitem "units: a writer refuses plain numbers where a length is meant" setup = [VGIoFixtures] begin
    import VisualGeometryToolkitIO as VgtkIO
    using VisualGeometryToolkitCore: Rig, PosedTarget, PlanarTarget, PointTarget, Trajectory, TargetPoses, TargetPoints3D,
        TargetPoints2D, BoundedCamera, PosedCamera, StereoRig, WorldFrame, BodyFrame, CameraFrame, TargetFrame, relabel, has_units

    refused(f) = try
        f(); false
    catch e
        e isa ArgumentError && occursin("holds plain numbers", e.msg) && occursin("t .* m", e.msg)
    end
    plain = EuclideanMap(one(RotMatrix{3, Float64}), SVector(1.0, 2.0, 3.0))
    pts = [SVector(0.0, 0.0), SVector(10.0, 0.0)]
    K = SMatrix{3, 3, Float64}(900.0, 0, 0, 0, 900.0, 0, 640.0, 480.0, 1)
    mktempdir() do dir
        p = joinpath(dir, "x.json")
        @test refused(() -> save(p, [relabel(WorldFrame, CameraFrame, plain)]))
        @test refused(() -> save(p, Trajectory(BodyFrame{:a}, WorldFrame, Dict(1 => plain))))
        @test refused(() -> save(p, TargetPoses{CameraFrame}(Dict(1 => Dict(:a => plain)); images = Dict(1 => "a"), sensors = "c")))
        @test refused(() -> save(p, Rig{WorldFrame}(Dict(:a => PosedTarget(PlanarTarget(pts), plain)))))
        @test refused(() -> save(p, StereoRig(BoundedCamera(PosedCamera(K, one(RotMatrix{3, Float64}), zero(SVector{3, Float64})), Size2(1280, 960)),
            BoundedCamera(PosedCamera(K, one(RotMatrix{3, Float64}), SVector(120.0, 0.0, 0.0)), Size2(1280, 960)))))
        @test refused(() -> save(p, TargetPoints3D{WorldFrame}(Dict(1 => [(target = :a, index = 1, xyz = SVector(1.0, 2.0, 3.0))]); sensors = "m")))
        @test refused(() -> save(p, PointTarget([SVector(1.0, 2.0, 3.0)])))
        @test refused(() -> save(p, PlanarTarget(pts)))
        @test refused(() -> VgtkIO.pose_json(plain))
        @test refused(() -> save_tum_trajectory(joinpath(dir, "t.txt"), Trajectory(BodyFrame{:a}, WorldFrame, Dict(1 => plain)); stamp = :index))
        @test refused(() -> colmap_image(plain, 1, 1, "a.png"))
        @test refused(() -> save_opencv_calibration(p, CameraModel(CameraCalibrationMat(K)), [plain]; rms = 0.1, to_unit = mm, images = ["a"]))
        # the correspondences export reads its object points from the targets
        targets = Rig{TargetFrame{:a}}(Dict(:a => PosedTarget(PlanarTarget(pts), EuclideanMap(one(RotMatrix{3, Float64}), zero(SVector{3, Float64})))))
        obs = TargetPoints2D{PixelFrame{:colmap}}(Dict(1 => [(; target = :a, index = 1, xy = SVector(1.0, 2.0), weight = 1.0)]); images = Dict(1 => "a.png"), sensors = "cam")
        @test refused(() -> save_correspondences(p, obs, targets))
        # a resolver of a referenced design hands back lengths
        save(p, Rig{WorldFrame}(Dict(:b => PosedTarget(PlanarTarget(:b), EuclideanMap(one(RotMatrix{3, Float64}), SVector(1.0, 2.0, 3.0) .* mm)))))
        @test refused(() -> load(Rig, p; resolve = _ -> pts))
        @test has_units(load(Rig, p; resolve = _ -> [x .* mm for x in pts])[:b].target)
    end
end

@testitem "units: a file in m or inch reads to the metres of its mm twin" setup = [VGIoFixtures] begin
    import VisualGeometryToolkitIO as VgtkIO
    using VisualGeometryToolkitCore: TargetPoints3D, geom, translation, inch
    using Statistics: cov

    fixtures = joinpath(@__DIR__, "fixtures", "units")
    raw = JSON.parse(read(joinpath(fixtures, "transforms.json"), String); dicttype = Dict{String, Any})
    mm_poses = load(Vector{EuclideanMap}, joinpath(fixtures, "transforms.json"))
    metres(X) = map(q -> ustrip(m, q), translation(geom(X)))
    mktempdir() do dir
        # metres, typed by hand: the mm decimals with the point moved — the same metre values, exactly
        dm = deepcopy(raw)
        dm["length_unit"] = "m"
        dm["transforms"][1]["translation"] = [0.2519, -0.1444, 0.9041]
        dm["transforms"][2]["translation"] = [0.0, 0.0628, 1.2122104]
        dm["transforms"][3]["translation"] = [-0.708695, 0.1486439, 0.0139]
        c = dm["transforms"][2]["covariance"]
        for i in 1:6, j in 1:6
            n = (i > 3) + (j > 3)
            c[i][j] = VgtkIO._shift_exponent(Float64(c[i][j]), -3n)   # the decimal point moved, as a person would
        end
        pm = joinpath(dir, "m.json"); write(pm, JSON.json(dm))
        m_poses = load(Vector{EuclideanMap}, pm)
        for (a, b) in zip(mm_poses, m_poses)
            @test metres(a) === metres(b)
        end
        @test cov(m_poses[2]) == cov(mm_poses[2])
        # inches, typed by hand (1 inch = 25.4 mm): within the scale's one rounding
        di = deepcopy(raw)
        di["length_unit"] = "inch"
        di["transforms"][1]["translation"] = [1.0, 2.0, -10.0]
        dmm = deepcopy(raw)
        dmm["transforms"][1]["translation"] = [25.4, 50.8, -254.0]
        pi_, pmm = joinpath(dir, "inch.json"), joinpath(dir, "mm.json")
        write(pi_, JSON.json(di)); write(pmm, JSON.json(dmm))
        a, b = metres(load(Vector{EuclideanMap}, pmm)[1]), metres(load(Vector{EuclideanMap}, pi_)[1])
        @test all(abs.(a .- b) .<= eps.(abs.(a)))
        # the writer's unit: any length unit, declared; a power of ten reads back
        # to the identical quantities, an inch to within the scale's rounding
        for u in (m, inch, mm)
            p = joinpath(dir, "w.json")
            save(p, mm_poses; length_unit = u)
            @test parse_json(read(p, String)).length_unit == unit_string(u)
            back = load(Vector{EuclideanMap}, p)
            if u === inch
                @test all(all(abs.(metres(x) .- metres(y)) .<= eps.(abs.(metres(y)))) for (x, y) in zip(back, mm_poses))
                @test cov(back[2]) ≈ cov(mm_poses[2]) rtol = 4 * eps()
            else
                @test all(metres(x) === metres(y) for (x, y) in zip(back, mm_poses))
                @test cov(back[2]) == cov(mm_poses[2])
            end
        end
        obs = load(TargetPoints3D, joinpath(fixtures, "target_points3d.json"))
        p = joinpath(dir, "p3.json")
        save(p, obs; length_unit = m)
        @test load(TargetPoints3D, p) == obs
    end
end

@testitem "units: negative, ±0, integers, NaN and ±Inf" setup = [VGIoFixtures] begin
    import VisualGeometryToolkitIO as VgtkIO
    using VisualGeometryToolkitCore: inch, ft, pt, cm, μm, WorldFrame, CameraFrame, relabel, translation
    # read_length / write_length are total: every Float64 comes back as itself
    for u in (mm, cm, μm, m, inch, ft, pt)
        xs = (-62.8, -0.0, 0.0, 5, -3, 1.0e-310, 1.0e300, NaN, Inf, -Inf)
        @test all(isequal(VgtkIO.write_length(VgtkIO.read_length(x, u), u), Float64(x)) for x in xs)
        @test all(signbit(ustrip(m, VgtkIO.read_length(x, u))) == signbit(Float64(x)) for x in xs)
    end
    @test VgtkIO.read_length(5, mm) === VgtkIO.read_length(5.0, mm)    # a JSON integer
    # a vgio file holds finite numbers only: a non-finite one is refused by its field, reading and writing
    mktempdir() do dir
        X = relabel(WorldFrame, CameraFrame, EuclideanMap(one(RotMatrix{3, Float64}), SVector(1.0, NaN, 3.0) .* mm))
        msg(f) = try
            f(); ""
        catch e
            e isa ArgumentError ? e.msg : rethrow()
        end
        p = joinpath(dir, "nan.json")
        @test occursin("transforms[1].translation[2]: expected a finite number", msg(() -> save(p, [X])))
        @test !isfile(p)
        good = joinpath(dir, "good.json")
        save(good, [relabel(WorldFrame, CameraFrame, EuclideanMap(one(RotMatrix{3, Float64}), SVector(1.0, 2.0, 3.0) .* mm))])
        for (text, field) in (("NaN", "translation[2]"), ("Infinity", "translation[2]"), ("-Infinity", "translation[2]"), ("1e400", "translation[2]"))
            write(p, replace(read(good, String), r"2\.0" => text; count = 1))
            @test occursin(field, msg(() -> load(Vector{EuclideanMap}, p)))
            @test occursin(field, msg(() -> VgtkIO.validate(p)))
        end
        # integers in a hand-written file are lengths like any other number
        write(p, replace(read(good, String), "2.0" => "-2"; count = 1))
        @test ustrip(m, translation(only(load(Vector{EuclideanMap}, p)))[2]) === -0.002
    end
end

@testitem "units: a full-precision value moves at most once" setup = [VGIoFixtures] begin
    import VisualGeometryToolkitIO as VgtkIO
    using Random: Xoshiro
    using VisualGeometryToolkitCore: inch
    # A 17-digit millimetre number need not survive into metres (the two grids
    # differ; about a quarter of such numbers share a metre value). Its first
    # rewrite moves it by at most 2 ulp, and every rewrite after is exact.
    rng = Xoshiro(7)
    for u in (mm, inch)
        xs = [(rand(rng) - 0.5) * 10.0^rand(rng, -1:4) for _ in 1:20_000]
        t1 = [VgtkIO.write_length(VgtkIO.read_length(x, u), u) for x in xs]
        @test all(abs(a - x) <= 2 * eps(abs(x)) for (a, x) in zip(t1, xs))
        @test all(VgtkIO.write_length(VgtkIO.read_length(a, u), u) === a for a in t1)
    end
    # a computed metre value is written as a number that reads back to it, whenever one exists
    n_exact = count(1:20_000) do _
        y = (rand(rng) - 0.5) * 10.0^rand(rng, -4:1)
        ustrip(m, VgtkIO.read_length(VgtkIO.write_length(VgtkIO._metre_quantity(y), mm), mm)) === y
    end
    @test n_exact > 17_000
end

@testitem "units: a rig's identity survives a load, full-precision numbers included" setup = [VGIoFixtures] begin
    using VisualGeometryToolkitCore: Rig, TargetPoints2D
    # 287.15788802842013 mm has no metre value that writes back to it (it comes
    # back 287.1578880284201): a file of computed target poses names the same targets
    # loaded or by path.
    fixtures = joinpath(@__DIR__, "fixtures", "units")
    mktempdir() do dir
        for name in ("rig.json", "rig_targets.json")
            p = joinpath(dir, name)
            write(p, replace(read(joinpath(fixtures, name), String), "251.9" => "287.15788802842013"))
            @test occursin("287.15788802842013", read(p, String))
            @test content_id(p) == content_id(load(Rig, p))
        end
        # a measurement file naming the targets by path resolves against the loaded rig
        p = joinpath(dir, "rig_targets.json")
        id = content_id(p)
        obs_src = joinpath(fixtures, "target_points2d.json")
        old_id = only(load(TargetPoints2D, obs_src).targets)
        obs_path = joinpath(dir, "target_points2d.json")
        write(obs_path, replace(read(obs_src, String), old_id => id))
        obs = load(TargetPoints2D, obs_path)
        out = joinpath(dir, "correspondences.json")
        @test save_correspondences(PixelFrame{:opencv}, out, obs, load(Rig, p)) == out
        @test save_correspondences(PixelFrame{:opencv}, out, obs, p) == out
        # a file whose numbers round-trip keeps the identity it always had
        @test content_id(joinpath(fixtures, "rig_targets.json")) == old_id
    end
end

@testitem "units: a loaded non-central grid model refuses a plain camera, naming the fix" setup = [VGIoFixtures] begin
    using VisualGeometryToolkitCore: PosedCamera, NoncentralGenericModel, has_units, project, with_unit
    # The grid comes back in metres; the pre-0.2 habit `EuclideanMap(R, t_mm)`
    # would compare it with scene points in mm (offsets 1000× too small).
    gn = load(NoncentralGenericModel, joinpath(@__DIR__, "fixtures", "units", "generic_camera.json"))
    R = one(RotMatrix{3, Float64})
    t_mm = SVector(10.0, -5.0, 20.0)
    err = try PosedCamera(gn, EuclideanMap(R, t_mm)); nothing catch e; e end
    @test err isa ArgumentError
    @test occursin("t .* m", sprint(showerror, err)) && occursin("with_unit(extrinsics, mm)", sprint(showerror, err))
    # the fix: the extrinsics in mm, stated
    cam = PosedCamera(gn, with_unit(EuclideanMap(R, t_mm), mm))
    @test has_units(cam)
    X = SVector(0.0, 0.0, 1000.0) .* mm
    @test project(cam, X) == project(ustrip(m, cam), ustrip.(m, X))
end

@testitem "units: a rig built in code, its file and the rig loaded from it have one identity" setup = [VGIoFixtures] begin
    import VisualGeometryToolkitIO as VgtkIO
    using Random: Xoshiro
    using VisualGeometryToolkitCore: Rig, PosedTarget, PlanarTarget, TargetFrame
    # `v * mm` is `v * 0.001`, an ulp from the nearest decimal of `v` for about a quarter of
    # values, while a file holds (and reads) the decimal: the identity is of the persisted form.
    R = one(RotMatrix{3, Float64})
    rng = Xoshiro(3)
    n_fail = Ref(0)
    mktempdir() do dir
        p = joinpath(dir, "rig.json")
        for _ in 1:10_000
            t = [round(rand(rng) * 1000 - 500; sigdigits = rand(rng, 15:17)) * mm for _ in 1:3]
            board = [round(rand(rng) * 100; sigdigits = rand(rng, 15:17)) * mm for _ in 1:2]
            rig = Rig{TargetFrame{:a}}(
                Dict(
                    :a => PosedTarget(PlanarTarget([board, [0.0mm, 0.0mm]]), EuclideanMap(R, [0.0mm, 0.0mm, 0.0mm])),
                    :b => PosedTarget(PlanarTarget([board, [1.0mm, 2.0mm]]), EuclideanMap(R, t)),
                )
            )
            save(p, rig)
            n_fail[] += !(content_id(rig) == content_id(p) == content_id(load(Rig, p)))
        end
    end
    @test n_fail[] == 0
    # the persisted form is idempotent, and the same in a file of any decimal unit
    for _ in 1:10_000
        y = (rand(rng) - 0.5) * 10.0^rand(rng, -4:1)
        c = VgtkIO._persisted(y)
        @test VgtkIO._persisted(c) === c
        @test abs(c - y) <= 2 * eps(abs(y))
    end
end
