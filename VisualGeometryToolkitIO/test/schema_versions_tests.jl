# =============================================================================
# Every earlier schema version a published file holds is READ: the reader runs
# the version steps in memory (`migrate_record`), the same steps `vgio_migrate`
# writes a file with, and the writers write only the current version
# =============================================================================

@testsetup module OldVersions
    import VisualGeometryToolkitIO as VgtkIO
    using JSON: JSON
    using VisualGeometryToolkitCore: AbstractTarget, Rig, BoundedCamera
    export old_fixture, old_versions, json_tree, write_tree

    old_fixture(name) = joinpath(@__DIR__, "fixtures", "old", name)
    # the IO 0.9 fixtures (test/fixtures/old/, the 0.9 current versions the
    # suite's units fixtures held), by their 0.9 names
    const _OLD09 = Dict("rig.json" => "rig_v9.json", "rig_targets.json" => "rig_v9_targets.json", "rig_pitch.json" => "rig_v9_pitch.json",
        "target_planar.json" => "target_v5_planar.json", "target_points.json" => "target_v5_points.json",
        "target_points2d.json" => "target_points2d_v2.json", "target_points3d.json" => "target_points3d_v2.json",
        "target_poses.json" => "target_poses_v2.json", "trajectory.json" => "trajectory_v4.json", "transforms.json" => "transforms_v4.json")
    units_fixture(name) = joinpath(@__DIR__, "fixtures", "old", _OLD09[name])
    json_tree(p) = JSON.parse(read(p, String); dicttype = Dict{String, Any})
    write_tree(p, t) = (write(p, JSON.json(t, 4)); p)

    # One file of every supported old version, written into `dir`: the cameras
    # fixtures as they are; the target and rig ones from the current fixtures
    # by the inverse renames (v1/v4: v2/v5's keys put back; v2/v5: every
    # composition back in application order under its old keys) — the shape
    # the restored migrations' own tests build them in.
    function old_versions(dir)
        out = Pair{String, Any}[]
        for v in ("v1", "v2", "v3", "v4")
            push!(out, old_fixture("cameras_$(v).json") => Dict{Symbol, BoundedCamera})
        end
        # (generic_camera v2/v3: read through their steps, never rewritten — spline_lens_record_tests.jl)
        for name in ("target_planar.json", "target_points.json")
            t = json_tree(units_fixture(name))
            v2 = copy(t)
            v2["schema"] = "vgio/target/v2"
            v2["target"] = VgtkIO._body_v3_to_v2(t["target"])
            push!(out, write_tree(joinpath(dir, "v2_" * name), v2) => AbstractTarget)
            v1 = Dict{String, Any}(k == "calibration_record" ? "calibration" => v : k => v for (k, v) in v2)
            v1["schema"] = "vgio/target/v1"
            v1["target"] = VgtkIO._body_v2_to_v1(v2["target"])
            push!(out, write_tree(joinpath(dir, "v1_" * name), v1) => AbstractTarget)
        end
        for name in ("rig.json", "rig_targets.json")
            t = json_tree(units_fixture(name))
            v5 = VgtkIO._rig_v6_to_v5(t)
            v5["schema"] = "vgio/rig/v5"
            push!(out, write_tree(joinpath(dir, "v5_" * name), v5) => Rig)
            v4 = VgtkIO._map_rig_targets(VgtkIO._body_v2_to_v1, v5)
            v4["schema"] = "vgio/rig/v4"
            push!(out, write_tree(joinpath(dir, "v4_" * name), v4) => Rig)
        end
        return out
    end
end

@testitem "schema versions: every supported old version loads as its migrated file does" setup = [VGIoFixtures, OldVersions] begin
    import VisualGeometryToolkitIO as VgtkIO
    using VisualGeometryToolkitCore: Rig
    mktempdir() do dir
        for (src, T) in old_versions(dir)
            s = VgtkIO._peek_schema(src)
            kind = String(match(r"^vgio/([a-z_0-9]+)/v\d+$", s)[1])
            @test s != VgtkIO._schema(kind)                                  # an old version
            a = load(T, src)                                                 # read through the version steps
            r = vgio_migrate(src; out = joinpath(dir, "migrated_" * basename(src)))
            @test r.from == s && r.to == VgtkIO._schema(kind) && VgtkIO._peek_schema(r.out) == r.to
            b = load(T, r.out)
            @test VgtkIO.same_value(a, b)                                   # exact, a length within its 2-ulp guarantee
            # vgio_migrate's file is the current writer's file of the value the loader returns
            p = joinpath(dir, "saved_" * basename(src))
            save(p, a; VgtkIO._resave_kwargs(VgtkIO.migrate_record(json_tree(src)))...)
            @test read(p, String) == read(r.out, String)
            if T === Rig
                @test VgtkIO.content_id(src) == VgtkIO.content_id(r.out) == VgtkIO.content_id(a) == r.new_id
            end
        end
    end
end

@testitem "schema versions: a written file is always the current version" setup = [VGIoFixtures, OldVersions] begin
    import VisualGeometryToolkitIO as VgtkIO
    mktempdir() do dir
        for (src, T) in old_versions(dir)
            kind = String(match(r"^vgio/([a-z_0-9]+)/v\d+$", VgtkIO._peek_schema(src))[1])
            p = joinpath(dir, "w_" * basename(src))
            save(p, load(T, src))
            @test VgtkIO._peek_schema(p) == VgtkIO._schema(kind) && VgtkIO.validate(p) == kind
        end
    end
end

@testitem "schema versions: an old file loads silently; @debug names the version migrated" setup = [VGIoFixtures, OldVersions] begin
    using Base.CoreLogging: Debug
    using VisualGeometryToolkitCore: BoundedCamera
    p = old_fixture("cameras_v1.json")
    @test_logs load(Dict{Symbol, BoundedCamera}, p)                          # nothing at the default level
    @test_logs (:debug, r"vgio/cameras/v1") min_level = Debug match_mode = :any load(Dict{Symbol, BoundedCamera}, p)
end

@testitem "cameras v1, v2 and v3: the same camera in every version loads equal" setup = [VGIoFixtures, OldVersions] begin
    import VisualGeometryToolkitIO as VgtkIO
    using VisualGeometryToolkitCore: BoundedCamera, Tilt, model, extent, lens_coverage
    import VisualGeometryToolkitCore as VgtkCore
    v1, v2, v3, v4 = (load(Dict{Symbol, BoundedCamera}, old_fixture("cameras_$(v).json")) for v in ("v1", "v2", "v3", "v4"))
    # v4 → v5 moves no number: v4's cameras are v3's, every one with the whole sensor covered
    @test v4 == v3 && all(c -> lens_coverage(c) == VgtkCore.Rect2{Float64}(0, 0, extent(c)...), values(v4))
    # the Brown–Conrady lens of v1/v2 is the OpenCV lens of its shortest coefficient vector, as BlobBoards' cameras v3 of these files held it
    @test v1[:radtan] == v2[:radtan] == v3[:radtan]
    @test model(v1[:radtan]).lens == Lens(Perspective(), Additive(Radial((-0.15680625941667098, 0.04565665542192153)), Tangential(-0.0003527415347489064, 0.0008423616851487834)))
    @test model(v1[:radtan]).intrinsics.K[1, 3] == 641.375                    # colmap, as the file
    @test model(v2[:radtan]).intrinsics.K[1, 3] == 641.375                    # v2 declared opencv: moved into the canonical frame
    # v3's "chain" is in application order: [radial, tilt] is Tilt ∘ Radial
    @test model(v3[:tilted]).lens == Lens(Perspective(), Tilt(0.003, -0.001) ∘ Radial((0.01, -0.002)))
    # v1's metadata keys are the provenance
    @test VgtkIO.migrate_record(json_tree(old_fixture("cameras_v1.json")))["provenance"] ==
        Dict{String, Any}("converted_from" => "pinhole_radtan", "camera_id" => "test camera", "lens_id" => "test lens")
end

@testitem "cameras v1/v2: every lens model reads as the lens VGC 0.x computed" setup = [VGIoFixtures, OldVersions] begin
    import VisualGeometryToolkitIO as VgtkIO
    import VisualGeometryToolkitCore as VgtkCore
    using VisualGeometryToolkitCore: Vec3
    using LinearAlgebra: norm
    # one v1 store per lens: the lens read back through the version steps
    function v1_lens(model, params)
        cam = Dict{String, Any}("schema" => "vgio/camera/v1", "name" => "c", "pixel_chart" => "colmap",
            "size" => Dict{String, Any}("unit" => "px", "width" => 100, "height" => 80),
            "intrinsics" => Dict{String, Any}("unit" => "px", "fx" => 50.0, "fy" => 50.0, "cx" => 50.0, "cy" => 40.0, "skew" => 0.0),
            "lens" => Dict{String, Any}("model" => model, "params" => Any[params...]))
        t = VgtkIO.migrate_record(Dict{String, Any}("schema" => "vgio/cameras/v1", "cameras" => Any[cam]))
        @test t["schema"] == "vgio/cameras/v5"
        return lens_from_record(t["cameras"][1]["lens"])
    end
    X = Vec3(0.3, -0.2, 1.0)
    r, θ = norm(X[1:2]), atan(norm(X[1:2]), X[3])
    xd = SVector(0.31, -0.17)
    ρ = norm(xd)
    ray(L) = (v = VgtkCore.backproject(L, xd); v[1:2] ./ v[3])
    @test v1_lens("Pinhole", ()) == Pinhole()
    # v1's model names, retired VGC 0.x type names, as the reader spells them
    BC, RR, EQ = VgtkIO._V1_BROWN_CONRADY, VgtkIO._V1_RATIONAL_RADIAL, VgtkIO._V1_EQUIDISTANT
    # VGC 0.x's equidistant fisheye: r_d = θ
    @test VgtkCore.project(v1_lens(EQ, ()), X) ≈ θ .* X[1:2] ./ r
    # its Brown–Conrady lens: fields k1…k6, p1, p2, s1…s4 — OpenCV's model ([k1, k2, p1, p2, k3, k4, k5, k6, s1…s4])
    bc = (0.1, -0.02, 0.003, 0.01, -0.002, 0.0004, 1.0e-3, -2.0e-3, 1.0e-4, 2.0e-4, -3.0e-4, 4.0e-4)
    @test v1_lens(BC, bc) == VgtkIO.opencv_lens(collect((bc[1], bc[2], bc[7], bc[8], bc[3], bc[4], bc[5], bc[6], bc[9:12]...)))
    @test v1_lens(BC, bc[1:8]) == VgtkIO.opencv_lens(collect((bc[1], bc[2], bc[7], bc[8], bc[3], bc[4], bc[5], bc[6])))   # 8: s = 0
    @test v1_lens(BC, (0.1, -0.02, 0, 0, 0, 0, 1.0e-3, -2.0e-3)) == VgtkIO.opencv_lens([0.1, -0.02, 1.0e-3, -2.0e-3])
    # … and its map, written out (VGC 0.x's: the radial factor rational in r², k4–k6 the denominator)
    let (k1, k2, k3, k4, k5, k6, p1, p2, s1, s2, s3, s4) = bc, x = X[1] / X[3], y = X[2] / X[3], r2 = x^2 + y^2
        f = (1 + k1 * r2 + k2 * r2^2 + k3 * r2^3) / (1 + k4 * r2 + k5 * r2^2 + k6 * r2^3)
        xd = x * f + 2p1 * x * y + p2 * (r2 + 2x^2) + s1 * r2 + s2 * r2^2
        yd = y * f + p1 * (r2 + 2y^2) + 2p2 * x * y + s3 * r2 + s4 * r2^2
        @test VgtkCore.project(v1_lens(BC, bc), X) ≈ [xd, yd]
    end
    # its KannalaBrandt: θ_d = θ(1 + k1θ² + k2θ⁴ + k3θ⁶ + k4θ⁸)
    k = (0.03, -0.004, 0.0005, -2.0e-5)
    @test VgtkCore.project(v1_lens("KannalaBrandt", k), X) ≈ θ * (1 + k[1]θ^2 + k[2]θ^4 + k[3]θ^6 + k[4]θ^8) .* X[1:2] ./ r
    # its DoubleSphere (Usenko et al. 2018): u = X / (α d₂ + (1 − α)(ξ d₁ + Z))
    ξ, α = 0.2, 0.6
    d1 = norm(X)
    z = ξ * d1 + X[3]
    d2 = sqrt(X[1]^2 + X[2]^2 + z^2)
    @test VgtkCore.project(v1_lens("DoubleSphere", (ξ, α)), X) ≈ X[1:2] ./ (α * d2 + (1 - α) * z)
    # its Scaramuzza: the lift (x_d, 1 + a2ρ² + a3ρ³ + a4ρ⁴)
    a = (-0.4, 0.05, 0.02)
    @test ray(v1_lens("Scaramuzza", a)) ≈ xd ./ (1 + a[1]ρ^2 + a[2]ρ^3 + a[3]ρ^4)
    # its rational-radial lens ([P, Q, num…, den…]): x_u = x_d·P(ρ²)/Q(ρ²); P = 0 is the division model
    @test ray(v1_lens(RR, (1, 2, 0.1, -0.3, 0.02))) ≈ xd .* (1 + 0.1ρ^2) ./ (1 - 0.3ρ^2 + 0.02ρ^4)
    @test v1_lens(RR, (0, 1, -0.3)) == Division(-0.3)
    # refused: a model v1 never held, a wrong arity, degrees that do not match
    for (model, params) in (("Fisheye624", (0.1,)), ("KannalaBrandt", (0.1, 0.2)), (BC, (0.1,)), (RR, (1, 1, 0.1)))
        @test_throws ArgumentError v1_lens(model, params)
    end
end

@testitem "cameras v1: cameras in different pixel charts each read in their own" setup = [VGIoFixtures, OldVersions] begin
    import VisualGeometryToolkitIO as VgtkIO
    using VisualGeometryToolkitCore: BoundedCamera, model
    t = json_tree(old_fixture("cameras_v1.json"))
    c2 = deepcopy(t["cameras"][1])
    c2["name"], c2["pixel_chart"] = "radtan_opencv", "opencv"
    c2["intrinsics"]["cx"], c2["intrinsics"]["cy"] = 640.875, 477.625
    push!(t["cameras"], c2)
    mktempdir() do dir
        s = load(Dict{Symbol, BoundedCamera}, write_tree(joinpath(dir, "mixed.json"), t))
        @test model(s[:radtan]) == model(s[:radtan_opencv])
        bad = deepcopy(t)
        bad["cameras"][1]["intrinsics"]["unit"] = "mm"
        e = try load(Dict{Symbol, BoundedCamera}, write_tree(joinpath(dir, "bad.json"), bad)); nothing catch e; e end
        @test e isa ArgumentError && occursin("bad.json", e.msg) && occursin("px", e.msg)
        # a value v1 never wrote (a degree that is not an integer) is refused naming the file, not a bare Julia error
        odd = deepcopy(t)
        odd["cameras"][1]["lens"] = Dict{String, Any}("model" => VgtkIO._V1_RATIONAL_RADIAL, "params" => Any[1.5, 1, 0.1, 0.2])
        e = try load(Dict{Symbol, BoundedCamera}, write_tree(joinpath(dir, "odd.json"), odd)); nothing catch e; e end
        @test e isa ArgumentError && occursin("odd.json", e.msg) && occursin("vgio/cameras/v1", e.msg)
    end
end

@testitem "schema versions: observations naming an old rig's identity resolve against its file" setup = [VGIoFixtures, OldVersions, MigrateFixtures] begin
    import VisualGeometryToolkitIO as VgtkIO
    using VisualGeometryToolkitCore: TargetObservations, PixelFrame
    mktempdir() do dir
        rig = old_copy(dir, "rig_targets.json", as_v4_rig)
        obs = load(TargetObservations, as_v4_observations(dir))                       # naming the v4 identity
        @test only(obs.targets) == V4_ID != VgtkIO.content_id(rig) == current_id()
        a = joinpath(dir, "a.json")
        save_correspondences(a, obs, rig)
        b = joinpath(dir, "b.json")
        save_correspondences(b, load(TargetObservations, units_fixture("target_points2d.json")), units_fixture("rig_targets.json"))
        @test read(a, String) == read(b, String)
        @test_throws ArgumentError save_correspondences(joinpath(dir, "c.json"), obs, units_fixture("rig_targets.json"))   # the 0.9 file was never named by the v4 identity
    end
end

@testitem "schema versions: a version no published file held, or one that lacks what the current one needs, is refused" setup = [VGIoFixtures, OldVersions] begin
    import VisualGeometryToolkitIO as VgtkIO
    using VisualGeometryToolkitCore: Rig, EuclideanMap
    mktempdir() do dir
        refusal(T, schema, file) = (p = joinpath(dir, file);
            t = JSON.parse(read(old_fixture(file), String); dicttype = Dict{String, Any});
            t["schema"] = schema; write(p, JSON.json(t)); e = try load(T, p); nothing catch e; e end; e isa ArgumentError ? e.msg : "")
        @test occursin("free strings", refusal(Vector{EuclideanMap}, "vgio/transforms/v2", "transforms_v4.json"))   # its frames' names are not the grammar's
        @test occursin("rig/v10.0", refusal(Rig, "vgio/rig/v3", "rig_v9.json"))                                # a retired kind
        @test occursin("re-export", refusal(Rig, "vgio/rig/v2", "rig_v9.json"))
        @test occursin("re-export", refusal(Dict{Symbol, VisualGeometryToolkitCore.GaussianBlur}, "vgio/blur/v1", "rig_v9.json"))   # blur v1: no published file                                # no migration
    end
end

@testitem "migrate_record: an owner's version step is run on load, until no step is left" setup = [VGIoFixtures] begin
    import VisualGeometryToolkitIO as VgtkIO
    VgtkIO.migrate_schema(::Val{Symbol("toy/record/v1")}, t::AbstractDict) =
        merge(t, Dict{String, Any}("schema" => "toy/record/v2", "b" => t["a"]))
    VgtkIO.migrate_schema(::Val{Symbol("toy/record/v2")}, t::AbstractDict) =
        Dict{String, Any}("schema" => "toy/record/v3", "c" => 2 * t["b"])
    @test VgtkIO.migrate_record(Dict{String, Any}("schema" => "toy/record/v1", "a" => 1)) == Dict{String, Any}("schema" => "toy/record/v3", "c" => 2)
    @test VgtkIO.migrate_record(Dict{String, Any}("schema" => "toy/record/v3", "c" => 2)) == Dict{String, Any}("schema" => "toy/record/v3", "c" => 2)
    # a step must move to a later version of its kind: no loop, no other kind
    VgtkIO.migrate_schema(::Val{Symbol("toy/stuck/v1")}, t::AbstractDict) = t
    @test_throws ArgumentError VgtkIO.migrate_record(Dict{String, Any}("schema" => "toy/stuck/v1"))
    VgtkIO.migrate_schema(::Val{Symbol("toy/back/v2")}, t::AbstractDict) = Dict{String, Any}("schema" => "toy/back/v1")
    @test_throws ArgumentError VgtkIO.migrate_record(Dict{String, Any}("schema" => "toy/back/v2"))
    VgtkIO.migrate_schema(::Val{Symbol("toy/away/v1")}, t::AbstractDict) = Dict{String, Any}("schema" => "toy/other/v2")
    @test_throws ArgumentError VgtkIO.migrate_record(Dict{String, Any}("schema" => "toy/away/v1"))
end
