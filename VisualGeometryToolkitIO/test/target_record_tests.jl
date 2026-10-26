# =============================================================================
# vgio/target/v2 and vgio/rig/v5: a planar target's calibration record — every
# TargetCalibration kind, with and without its covariance, round-tripped
# =============================================================================

@testsetup module TargetRecordFixtures
    using VisualGeometryToolkitCore: Rigid, Affine, ThinPlateSpline, HeightField, CalibrationChain, Uncertain, CovMat,
        grid_nodes, parameter_vector, geom, reference, calibration
    using Statistics: cov
    using StaticArrays: SVector, SMatrix
    using LinearAlgebra: Diagonal
    import VisualGeometryToolkitIO as VgtkIO
    using VisualGeometryToolkitCore: mm

    export kinds, uncertain, design_points, mmq, same_target, same_calibration

    # `==` of an `Uncertain` is `===` (VGC defines no `==` for `Attributed`), which a
    # thin-plate kind's solver matrices make identity: compare the kind and the
    # covariance instead.
    same_calibration(a, b) = typeof(a) == typeof(b) && (a isa Uncertain ? (geom(a) == geom(b) && cov(a) == cov(b)) : a == b)
    same_target(a, b) = reference(a) === reference(b) && isequal(a.nominal, b.nominal) && same_calibration(calibration(a), calibration(b))

    # one millimetre number as a reader returns it (lengths.jl: decimal-exact)
    mmq(x) = VgtkIO.read_length(x, mm)
    design_points() = [SVector(mmq(0.0), mmq(0.0)), SVector(mmq(62.8), mmq(0.0)), SVector(mmq(-13.9), mmq(40.25)), SVector(mmq(30.5), mmq(77.0))]

    # every kind, physical, its numbers file-born (each length the metre value
    # nearest a millimetre decimal, so the record writes it back exactly)
    function kinds()
        nodes = [SVector(mmq(p[1]), mmq(p[2])) for p in grid_nodes(3; lo = 0.0, hi = 100.0)]
        A = Affine(SMatrix{2, 2}(1.001, -0.003, 0.002, 0.999), [mmq(0.5), mmq(-0.25)])
        T = ThinPlateSpline(nodes, mmq.([0.01, 0.0, -0.02, 0.03, 0.0, 0.0, 0.01, -0.01, 0.0]), mmq.([-0.02, 0.01, 0.0, 0.0, 0.04, 0.0, 0.0, 0.02, -0.03]))
        H = HeightField(nodes, mmq.([0.1, 0.0, -0.1, 0.2, 0.0, 0.0, 0.0, 0.3, 0.0]))
        return (rigid = Rigid(), affine = A, thin_plate_spline = T, height_field = H,
            calibration_chain = CalibrationChain(A, T, H), chain_in_plane = CalibrationChain(A, T))
    end

    # an Uncertain one: a covariance over its parameters (lengths² in m²), its
    # entries file-born decimals (i·10⁻⁶ on the diagonal, a correlation in the
    # first row and column) so the record writes them back exactly
    function uncertain(c)
        n = length(parameter_vector(c))
        Σ = Matrix(Diagonal([parse(Float64, "$(i)e-6") for i in 1:n]))
        if n >= 2
            Σ[1, 2] = Σ[2, 1] = 2.5e-7
            Σ[1, n] = Σ[n, 1] = -1.5e-7
        end
        return Uncertain(c, CovMat(SMatrix{n, n, Float64}(Σ)))
    end
end

@testitem "target/v2: every calibration kind round-trips, with and without its covariance" setup = [VGIoFixtures, TargetRecordFixtures] begin
    using VisualGeometryToolkitCore: PlanarTarget, AbstractTarget, Attributed, Uncertain, calibration, points, geom,
        is_resolved, reference, parameter_names
    using Statistics: cov
    mktempdir() do dir
        p = joinpath(dir, "target.json")
        for (name, c) in pairs(kinds()), cc in (c, uncertain(c)), length_unit in (mm, m)
            t = PlanarTarget(design_points(); calibration = cc)
            save(p, t; length_unit)
            d = parse_json(read(p, String))
            @test d.schema == "vgio/target/v2" && d.target.calibration.type == (name == :chain_in_plane ? "calibration_chain" : String(name))
            @test haskey(d.target.calibration, :covariance) == (cc isa Uncertain)
            back = load(AbstractTarget, p)
            @test same_target(back, t)                        # exact, by type: values, units, the covariance
            @test points(back) == points(t)
            # an unresolved target (a referenced design, no points) carries its calibration too
            r = PlanarTarget(:b76e; calibration = cc)
            save(p, r; length_unit)
            br = load(AbstractTarget, p)
            @test same_target(br, r) && !is_resolved(br) && reference(br) === :b76e
            @test same_target(load(AbstractTarget, p; resolve = _ -> design_points()), PlanarTarget(:b76e; nominal = design_points(), calibration = cc))
        end
    end
end

@testitem "target/v2: the record as written — tags, parameter order, units" setup = [VGIoFixtures, TargetRecordFixtures] begin
    using VisualGeometryToolkitCore: PlanarTarget, AbstractTarget, Rigid, Affine, CalibrationChain, parameter_vector
    k = kinds()
    mktempdir() do dir
        p = joinpath(dir, "target.json")
        # affine: A plain, b a length in the file's unit, in parameter_names order
        save(p, PlanarTarget(design_points(); calibration = k.affine))
        @test parse_json(read(p, String)).target.calibration == Dict(:type => "affine", :parameters => [1.001, 0.002, -0.003, 0.999, 0.5, -0.25])
        save(p, PlanarTarget(design_points(); calibration = k.affine); length_unit = m)
        @test parse_json(read(p, String)).target.calibration.parameters == [1.001, 0.002, -0.003, 0.999, 0.0005, -0.00025]
        # the thin-plate kinds: nodes and every parameter lengths
        save(p, PlanarTarget(design_points(); calibration = k.thin_plate_spline))
        c = parse_json(read(p, String)).target.calibration
        @test c.type == "thin_plate_spline" && length(c.nodes) == 9 && c.nodes[9] == [100.0, 100.0] && length(c.parameters) == 18
        @test c.parameters[1:3] == [0.01, 0.0, -0.02] && c.parameters[10:12] == [-0.02, 0.01, 0.0]    # fx…, then fy…
        save(p, PlanarTarget(design_points(); calibration = k.height_field))
        c = parse_json(read(p, String)).target.calibration
        @test c.type == "height_field" && length(c.parameters) == 9 && c.parameters[8] == 0.3
        # a chain: its members in order, the covariance once, over every member's parameters
        save(p, PlanarTarget(design_points(); calibration = uncertain(k.calibration_chain)))
        c = parse_json(read(p, String)).target.calibration
        @test [x.type for x in c.members] == ["affine", "thin_plate_spline", "height_field"]
        @test all(x -> !haskey(x, :covariance), c.members) && size(reduce(hcat, c.covariance)) == (33, 33)
        @test c.covariance[1][1] == 1.0e-6 && c.covariance[5][5] == 5.0 && c.covariance[1][33] == -1.5e-4   # A plain; bx² in mm²; a11·h9 in mm
        # rigid
        save(p, PlanarTarget(design_points()))
        @test parse_json(read(p, String)).target.calibration == Dict(:type => "rigid")
    end
end

@testitem "target/v2: refusals — plain calibrations, malformed records, the v1 keys" setup = [VGIoFixtures, TargetRecordFixtures] begin
    using VisualGeometryToolkitCore: PlanarTarget, AbstractTarget, Affine, HeightField, grid_nodes
    mktempdir() do dir
        p = joinpath(dir, "target.json")
        # a calibration with lengths in plain numbers is refused at save, as plain points are
        @test_throws ArgumentError save(p, PlanarTarget(:b76e; calibration = Affine([1.0 0.0; 0.0 1.0], [0.1, 0.0])))
        @test_throws ArgumentError save(p, PlanarTarget(:b76e; calibration = HeightField(grid_nodes(2; lo = 0.0, hi = 1.0))))
        @test !isfile(p)
        save(p, PlanarTarget(design_points(); calibration = kinds().calibration_chain))
        good = JSON.parse(read(p, String); dicttype = Dict{String, Any})
        function refused(f, key)
            dd = deepcopy(good)
            f(dd["target"]["calibration"])
            q = joinpath(dir, "bad.json")
            write(q, JSON.json(dd))
            e = try load(AbstractTarget, q); nothing catch e; e end
            return e isa ArgumentError && occursin(key, e.msg)
        end
        @test refused(c -> (c["type"] = "spline"), "unknown calibration type")
        @test refused(c -> (c["members"] = c["members"][1:1]), "at least two members")
        @test refused(c -> push!(c["members"], Dict("type" => "calibration_chain", "members" => [])), "not a chain")
        @test refused(c -> pop!(c["members"][1]["parameters"]), "states 6 parameters")
        @test refused(c -> pop!(c["members"][2]["parameters"]), "states 18 parameters")
        @test refused(c -> delete!(c["members"][3], "nodes"), "states its `nodes`")
        @test refused(c -> (c["members"][1]["nodes"] = [[0, 0]]), "has no `nodes`")
        @test refused(c -> (c["members"][1]["covariance"] = [[1.0]]), "carries no `covariance`")
        @test refused(c -> (c["covariance"] = [[1.0]]), "33×33")
        @test refused(c -> (c["members"] = [Dict("type" => "rigid"), Dict("type" => "rigid", "parameters" => [])]), "has no `parameters`")
        # the v1 keys and version name the migration
        dd = deepcopy(good)
        dd["target"]["warp"] = pop!(dd["target"], "calibration")
        write(p, JSON.json(dd))
        e = try load(AbstractTarget, p); nothing catch e; e end
        @test e isa ArgumentError && occursin("`warp`", e.msg) && occursin("vgio_migrate", e.msg)
        dd = deepcopy(good)
        dd["schema"] = "vgio/target/v1"
        write(p, JSON.json(dd))
        e = try load(AbstractTarget, p); nothing catch e; e end
        @test e isa ArgumentError && occursin("vgio/target/v1", e.msg) && occursin("vgio_migrate", e.msg)
    end
end

@testitem "rig/v5: a non-rigid posed target, and its content identity" setup = [VGIoFixtures, TargetRecordFixtures] begin
    using VisualGeometryToolkitCore: Rig, PosedTarget, PlanarTarget, TargetFrame, WorldFrame, Uncertain, CovMat, pose, points,
        calibration, is_planar, ids, geom
    using LinearAlgebra: LinearAlgebra
    k = kinds()
    X = EuclideanMap(one(RotMatrix{3, Float64}), [mmq(10.0), mmq(-5.0), mmq(900.0)])
    ident = EuclideanMap(one(RotMatrix{3, Float64}), [mmq(0.0), mmq(0.0), mmq(0.0)])
    rig = Rig{TargetFrame{:a}}(
        Dict(
            :a => PosedTarget(PlanarTarget(design_points(); calibration = uncertain(k.calibration_chain)), ident),
            :b => PosedTarget(PlanarTarget(:b; calibration = k.height_field), Uncertain(X, CovMat(SMatrix{6, 6, Float64}(LinearAlgebra.I) * 1.0e-6))),
            :c => PosedTarget(PlanarTarget(design_points(); calibration = k.affine), X),
        )
    )
    mktempdir() do dir
        p = joinpath(dir, "rig.json")
        save(p, rig)
        @test parse_json(read(p, String)).schema == "vgio/rig/v5"
        back = load(Rig, p)
        @test ids(back) == ids(rig) && all(id -> same_target(back[id].target, rig[id].target), ids(rig))
        @test pose(back, :b) isa Uncertain && geom(pose(back, :b)).t == X.t && pose(back, :c).t == X.t
        @test !is_planar(back[:a].target) && eltype(points(back[:a].target)) <: SVector{3}
        @test calibration(back[:c].target) == k.affine
        # the identity does not change with the file's unit, or a read
        pm = joinpath(dir, "rig_m.json")
        save(pm, rig; length_unit = m)
        @test content_id(p) == content_id(pm) == content_id(rig) == content_id(back)
        # the calibration is identity
        rig2 = Rig{TargetFrame{:a}}(Dict(:a => rig[:a], :b => rig[:b], :c => PosedTarget(PlanarTarget(design_points(); calibration = k.rigid), X)))
        @test content_id(rig2) != content_id(rig)
    end
end

@testitem "rig/v5: content_id is pinned on a fixed canonical tree" setup = [VGIoFixtures] begin
    # The identity measurement files name a rig by is the SHA-256 of a FIXED
    # canonical tree (content_id's docstring): changing what the tree holds
    # re-ids every rig and orphans every measurement file. If this fails, the
    # change must say why and migrate the measurement files that name the old ids.
    using VisualGeometryToolkitCore: Rig, TargetPoints2D
    fixtures = joinpath(@__DIR__, "fixtures", "units")
    id = content_id(joinpath(fixtures, "rig_targets.json"))
    @test id == "eb9ad42d8920caa208a5ce708d0b1c7954c541cd65a857cc9ec97b540a8c00fe"
    @test only(load(TargetPoints2D, joinpath(fixtures, "target_points2d.json")).targets) == id
    @test content_id(joinpath(fixtures, "rig.json")) == "2f461cfef0e8e465a9e4b107cb88c10783583b37ffeffe6842f76fec0a465bd0"
end

@testitem "target/v2: a planar target's page — present and absent" setup = [VGIoFixtures, TargetRecordFixtures] begin
    import VisualGeometryToolkitIO as VgtkIO
    using VisualGeometryToolkitCore: PlanarTarget, AbstractTarget
    # The record of the sheet a target is printed on (VGC's PlanarTarget gains a
    # `Page` on the primitives branch): its size in the Size2 form and its
    # lower-left corner `origin` in the design chart, lengths in the file's
    # unit, exact; absent when the target has none.
    page = (size = Size2(width = mmq(150.0), height = mmq(77.25)), origin = SVector(mmq(-13.9), mmq(0.0)))
    for u in (mm, m)
        j = VgtkIO._page_json(page, u)
        @test j.page.width == (u == mm ? 150.0 : 0.15) && j.page.height == (u == mm ? 77.25 : 0.07725)
        @test j.page.origin == (u == mm ? [-13.9, 0.0] : [-0.0139, 0.0])
        back = VgtkIO._page_from_json(parse_json(JSON.json(j.page)), u)
        @test back.size == page.size && back.origin == page.origin
    end
    @test VgtkIO._page_json(nothing, mm) == (;)
    @test VgtkIO._target_page(PlanarTarget(design_points())) === nothing    # (until VGC's PlanarTarget holds a page)
    @test_throws ArgumentError VgtkIO._page_json((size = Size2(width = 1.0, height = 1.0), origin = SVector(0.0, 0.0)), mm)   # plain numbers
    mktempdir() do dir
        p = joinpath(dir, "t.json")
        save(p, PlanarTarget(design_points()))
        d = JSON.parse(read(p, String); dicttype = Dict{String, Any})
        @test !haskey(d["target"], "page")                           # absent when the target has none
        q = joinpath(dir, "e.json")
        refused(f) = (dd = deepcopy(d); f(dd); write(q, JSON.json(dd)); e = try load(AbstractTarget, q); nothing catch e; e end; e)
        e = refused(dd -> (dd["target"]["page"] = Dict("width" => 150, "height" => 0, "origin" => [0, 0])))
        @test e isa ArgumentError && occursin("are > 0", e.msg)
        e = refused(dd -> (dd["target"]["page"] = Dict("width" => 150, "height" => 77)))
        @test e isa ArgumentError && occursin("origin", e.msg)
        # validated, then read into the target (refused while VGC's PlanarTarget holds no page)
        e = refused(dd -> (dd["target"]["page"] = Dict("width" => 150, "height" => 77, "origin" => [0, 0])))
        @test e isa ArgumentError && (hasfield(PlanarTarget, :page) || occursin("page", e.msg))
    end
end
