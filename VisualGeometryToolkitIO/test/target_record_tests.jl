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
    using Rotations: QuatRotation, RotMatrix
    import VisualGeometryToolkitIO as VgtkIO
    import VisualGeometryToolkitCore as VgtkCore
    using VisualGeometryToolkitCore: mm

    export kinds, uncertain, design_points, mmq, identity_rig

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

    # A rig of targets of every kind — rigid, affine, thin-plate, height field,
    # a chain, Uncertain calibrations, a point target with covariances, an
    # Uncertain pose — its numbers file-born: content_id's pin.
    # rotation = :quat builds the poses as the reader returns them (QuatRotation);
    # :matrix as RotMatrix; :quat_of_matrix as QuatRotation(RotMatrix(...)).
    function identity_rig(; boards::Bool = false, rotation::Symbol = :quat)
        k = kinds()
        bd = boards ? VgtkCore.Board(VgtkCore.Size2(width = mmq(150.0), height = mmq(77.25)), SVector(mmq(-13.9), mmq(0.0))) : nothing
        rot(q) = rotation === :quat ? q : rotation === :matrix ? RotMatrix{3, Float64}(q) : QuatRotation(RotMatrix{3, Float64}(q))
        X(t) = VgtkCore.EuclideanMap(rot(QuatRotation(0.9128709291752769, 0.18257418583505536, 0.36514837167011072, 0.0, false)), SVector(mmq(t[1]), mmq(t[2]), mmq(t[3])))
        I6 = VgtkCore.CovMat(SMatrix{6, 6, Float64}(Diagonal([1.0e-6, 1.0e-6, 1.0e-6, 2.5e-7, 2.5e-7, 4.0e-7])))
        ident = VgtkCore.EuclideanMap(rot(QuatRotation(1.0, 0.0, 0.0, 0.0, false)), SVector(mmq(0.0), mmq(0.0), mmq(0.0)))
        pt = VgtkCore.PointTarget([VgtkCore.Uncertain(SVector(mmq(0.0), mmq(1.5), mmq(-2.0)), VgtkCore.CovMat(SMatrix{3, 3, Float64}(Diagonal([1.0e-8, 1.0e-8, 4.0e-8]))))])
        members = Dict(
            :a_rigid => VgtkCore.PosedTarget(VgtkCore.PlanarTarget(nothing, design_points(), Rigid(), bd), ident),
            :b_affine => VgtkCore.PosedTarget(VgtkCore.PlanarTarget(design_points(); calibration = uncertain(k.affine)), X((10.0, 0.0, 900.0))),
            :c_tps => VgtkCore.PosedTarget(VgtkCore.PlanarTarget(:c_tps; calibration = k.thin_plate_spline), X((-200.0, 15.5, 850.0))),
            :d_height => VgtkCore.PosedTarget(VgtkCore.PlanarTarget(design_points(); calibration = k.height_field), VgtkCore.Uncertain(X((0.0, 120.0, 910.0)), I6)),
            :e_chain => VgtkCore.PosedTarget(VgtkCore.PlanarTarget(nothing, design_points(), uncertain(k.calibration_chain), bd), X((300.0, -40.25, 1000.0))),
            :f_points => VgtkCore.PosedTarget(pt, X((5.0, 5.0, 700.0))),
        )
        return VgtkCore.Rig{VgtkCore.TargetFrame{:a_rigid}}(members)
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
        is_resolved, reference, parameter_names, Rigid
    using Statistics: cov
    mktempdir() do dir
        p = joinpath(dir, "target.json")
        for (name, c) in pairs(kinds()), cc in (c isa Rigid ? (c,) : (c, uncertain(c))), length_unit in (mm, m)
            t = PlanarTarget(design_points(); calibration = cc)
            save(p, t; length_unit)
            d = parse_json(read(p, String))
            @test d.schema == "vgio/target/v2" && d.target.calibration.type == (name == :chain_in_plane ? "calibration_chain" : String(name))
            @test haskey(d.target.calibration, :covariance) == (cc isa Uncertain)
            back = load(AbstractTarget, p)
            @test back == t                        # exact, by type: values, units, the covariance
            @test points(back) == points(t)
            # an unresolved target (a referenced design, no points) carries its calibration too
            r = PlanarTarget(:b76e; calibration = cc)
            save(p, r; length_unit)
            br = load(AbstractTarget, p)
            @test br == r && !is_resolved(br) && reference(br) === :b76e
            @test load(AbstractTarget, p; resolve = _ -> design_points()) == PlanarTarget(:b76e; nominal = design_points(), calibration = cc)
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
    struct AnotherKind <: VisualGeometryToolkitCore.TargetCalibration end
    VisualGeometryToolkitCore.parameter_vector(::AnotherKind) = SVector{0, Float64}()
    VisualGeometryToolkitCore.parameter_names(::AnotherKind) = ()
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
        # a rigid calibration has no parameters: no covariance, in the record or in memory
        save(p, PlanarTarget(design_points()))
        rd = JSON.parse(read(p, String); dicttype = Dict{String, Any})
        rd["target"]["calibration"]["covariance"] = []
        write(p, JSON.json(rd))
        e = try load(AbstractTarget, p); nothing catch e; e end
        @test e isa ArgumentError && occursin("no parameters, so no `covariance`", e.msg)
        e = try VisualGeometryToolkitCore.Uncertain(VisualGeometryToolkitCore.Rigid(), VisualGeometryToolkitCore.CovMat(SMatrix{0, 0, Float64}())); nothing catch e; e end
        @test e isa ArgumentError && occursin("no parameters, so no covariance", e.msg)
        # the provenance attribute's previous name is refused naming its successor, nothing written
        q = joinpath(dir, "old_attribute.json")
        e = try save(q, VisualGeometryToolkitCore.Attributed(PlanarTarget(design_points()), (; calibration = (; tool = "t")))); nothing catch e; e end
        @test e isa ArgumentError && occursin("calibration_record", e.msg) && occursin("vgio_migrate", e.msg) && !isfile(q)
        # a downstream calibration kind (not in the registry) is refused at save, naming it
        e = try save(q, PlanarTarget(design_points(); calibration = AnotherKind())); nothing catch e; e end
        @test e isa ArgumentError && occursin("AnotherKind", e.msg) && !isfile(q)
        # a v1 key is an unknown key of the current record; the v1 version names the migration
        dd = deepcopy(good)
        dd["target"]["warp"] = pop!(dd["target"], "calibration")
        write(p, JSON.json(dd))
        e = try load(AbstractTarget, p); nothing catch e; e end
        @test e isa ArgumentError && occursin("target.warp: unknown key", e.msg)
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
        @test ids(back) == ids(rig) && all(id -> back[id].target == rig[id].target, ids(rig))
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

@testitem "rig/v5: content_id is pinned on a fixed canonical tree" setup = [VGIoFixtures, TargetRecordFixtures] begin
    # The identity measurement files name a rig by is the SHA-256 of a FIXED
    # identity tree (content_id's docstring), built from the value under its own
    # names. Changing what the tree holds re-ids every rig and orphans every
    # measurement file: if a literal below fails, the change must say why and
    # migrate the measurement files that name the old ids.
    import VisualGeometryToolkitIO as VgtkIO
    using VisualGeometryToolkitCore: Rig, TargetPoints2D
    fixtures = joinpath(@__DIR__, "fixtures", "units")
    id = content_id(joinpath(fixtures, "rig_targets.json"))
    @test id == "980c47d473610daa038b633cf5d4d133931699a9cafe1ee210f507b9b1c793b3"
    @test only(load(TargetPoints2D, joinpath(fixtures, "target_points2d.json")).targets) == id
    @test content_id(joinpath(fixtures, "rig.json")) == "5cdb12271e474c04c68ac5ea539f509a720449fcf42b79d8cc4a506cddcf9fe0"   # sensors
    # a rig of every target kind, Uncertain calibrations and poses, a point target
    rig = identity_rig()
    @test content_id(rig) == "eac2ee11c635ca73bdbb4dcbb2087784d3d364331a77c5f712b6a8c6c1622a99"   # poses as the reader returns them
    # The identity is of the PERSISTED rotation, the quaternion the writer
    # stores (one conversion, `_qvec`, shared by the writer and the tree): a
    # rig held with RotMatrix poses, the same rig built from
    # QuatRotation(RotMatrix(...)), the files written from either and the rig
    # loaded back all have one identity.
    rm_rig, qm_rig = identity_rig(; rotation = :matrix), identity_rig(; rotation = :quat_of_matrix)
    mktempdir() do dir
        a, b = joinpath(dir, "a.json"), joinpath(dir, "b.json")
        save(a, rm_rig); save(b, qm_rig)
        @test content_id(rm_rig) == content_id(qm_rig) == content_id(a) == content_id(b) == content_id(load(Rig, a)) == content_id(load(Rig, b))
    end
    # a rig whose targets carry boards: the board is part of the identity
    brig = identity_rig(; boards = true)
    @test content_id(brig) == "da15d512e982ab0247a51d71c37da499eaefcf5a4bd4b9276b0d87cc61e95176"
    @test content_id(brig) != content_id(rig)
    tb = VgtkIO._identity_tree(brig)
    @test only(x for x in tb["targets"] if x["id"] == "a_rigid")["target"]["board"] ==
        Dict("size" => [0.15, 0.07725], "origin" => [-0.0139, 0.0])
    # the identity is the value's: every unit, and a file read back, agree
    mktempdir() do dir
        for r in (rig, brig), u in (mm, m)
            p = joinpath(dir, "rig_$(u).json")
            save(p, r; length_unit = u)
            @test content_id(p) == content_id(r) == content_id(load(Rig, p))
        end
    end
    # its names are the identity's own, not the record's
    t = VgtkIO._identity_tree(rig)
    @test sort!(collect(keys(t))) == ["frame", "targets"]
    e = only(x for x in t["targets"] if x["id"] == "e_chain")
    @test e["target"]["kind"] == "planar" && e["target"]["calibration"]["kind"] == "calibration_chain" &&
        haskey(e["target"]["calibration"], "covariance") && [x["kind"] for x in e["target"]["calibration"]["members"]] == ["affine", "thin_plate_spline", "height_field"]
    # a generic sensor with a nominal pinhole and a domain: both enter under the identity's own names
    using VisualGeometryToolkitCore: PosedCamera, ViewFrame, relabel, CentralGenericModel, NoncentralGenericModel, KannalaBrandt,
        EuclideanMap, Vec3, CameraModel, Ellipse
    using Rotations: RotXYZ
    src = CameraModel(; focal = 31.7, pp = (32.25, 23.875), lens = KannalaBrandt(-4 / (3π^2)))
    gc = CentralGenericModel(src, (64, 48); grid_size = (8, 7), init = :quasi_interpolate)
    P = [Vec3{Float64}(1e-3 * gx, -2e-3 * gy, 0.0) for gy in 1:7, gx in 1:8]
    nc = NoncentralGenericModel(64, 48, 0, 0, 63, 47, gc.grid, P; unit_controls = false, gc.nominal, gc.domain)
    Xl = relabel(ViewFrame{:g}, ViewFrame{:g}, EuclideanMap(RotMatrix{3, Float64}(RotXYZ(0.0, 0.0, 0.0)), [0.0mm, 0.0mm, 0.0mm]))
    grig = Rig{ViewFrame{:g}}((PosedCamera(gc, Xl),))
    @test content_id(grig) == "dbbdd3f0f0d26b39fa6aaa377e062b33a4d4545fcefa1e0f3c04adf004599426"
    @test content_id(Rig{ViewFrame{:g}}((PosedCamera(nc, Xl),))) == "5349a44b940aa6ac1937013f31d794bfcfa926a4eec95dc3a22887b82d9c4773"
    cam = only(VgtkIO._identity_tree(grig)["sensors"])["camera"]
    K, ell = VisualGeometryToolkitCore.nominal_intrinsics(gc).K, gc.domain
    @test cam["nominal_intrinsics"] == [K[1, 1], K[2, 2], K[1, 3], K[2, 3], K[1, 2]]
    @test cam["domain"] == Dict("ellipse" => [ell.center[1], ell.center[2], ell.a, ell.b, ell.θ])
    @test sort!(collect(keys(cam))) == ["calibrated_area", "directions", "domain", "extent", "grid", "grid_model", "nominal_intrinsics"]
end

@testitem "target/v2: a planar target's board — present and absent" setup = [VGIoFixtures, TargetRecordFixtures] begin
    import VisualGeometryToolkitIO as VgtkIO
    using VisualGeometryToolkitCore: PlanarTarget, AbstractTarget, Board, board, Rig, PosedTarget, TargetFrame, Uncertain
    import VisualGeometryToolkitCore as VgtkCore
    using VisualGeometryToolkitCore: inch
    # The board a target is printed on (VGC's Board: the board proper, inside the
    # sheet's margin): its size in the Size2 form and its lower-left corner in
    # the design chart, lengths in the file's unit, exact; absent when none.
    bd = Board(Size2(width = mmq(150.0), height = mmq(77.25)), SVector(mmq(-13.9), mmq(0.0)))
    k = kinds()
    # An inch file holds every number the value has only when the value is
    # inch-born (lengths.jl: a metre value the inch grid skips is written within
    # an ulp), as an mm file does for mm-born values: each case is born in its unit.
    iq(x) = VgtkIO.read_length(x, inch)
    bd_inch = Board(Size2(width = iq(5.9), height = iq(3.0)), SVector(iq(-0.5), iq(0.0)))
    pts_inch = [SVector(iq(0.0), iq(0.0)), SVector(iq(2.5), iq(0.0)), SVector(iq(-0.5), iq(1.625)), SVector(iq(1.2), iq(3.0))]
    mktempdir() do dir
        p = joinpath(dir, "t.json")
        # (the calibrations' numbers are mm-born, so the inch file carries a rigid one)
        cases = [[(c, u, bd, design_points()) for c in (k.rigid, k.affine, uncertain(k.calibration_chain)) for u in (mm, m)];
                 [(k.rigid, inch, bd_inch, pts_inch)]]
        for (c, u, bd, pts) in cases
            t = PlanarTarget(nothing, pts, c, bd)
            save(p, t; length_unit = u)
            d = parse_json(read(p, String))
            @test haskey(d.target, :board) && sort(collect(keys(d.target.board))) == [:origin, :size]
            u == mm && @test d.target.board.size.width == 150.0 && d.target.board.size.height == 77.25 && d.target.board.origin == [-13.9, 0.0]
            back = load(AbstractTarget, p)
            @test back == t && board(back) == bd                                   # the Board included, bit for bit
            @test board(back).origin === bd.origin && board(back).size.width === bd.size.width
            # a referenced design keeps its board too
            r = PlanarTarget(:b76e, nothing, c, bd)
            save(p, r; length_unit = u)
            @test load(AbstractTarget, p) == r
        end
        # absent when the target has none
        save(p, PlanarTarget(design_points()))
        @test !haskey(parse_json(read(p, String)).target, :board) && board(load(AbstractTarget, p)) === nothing
        # malformed boards are refused
        q = joinpath(dir, "e.json")
        dd = JSON.parse(read(p, String); dicttype = Dict{String, Any})
        refused(f) = (x = deepcopy(dd); f(x); write(q, JSON.json(x)); e = try load(AbstractTarget, q); nothing catch e; e end; e)
        e = refused(x -> (x["target"]["board"] = Dict("size" => Dict("width" => 150, "height" => 0), "origin" => [0, 0])))
        @test e isa ArgumentError && occursin("are > 0", e.msg)
        e = refused(x -> (x["target"]["board"] = Dict("size" => Dict("width" => 150, "height" => 77))))
        @test e isa ArgumentError && occursin("origin", e.msg)
        # a point off the board: Core's check, at read
        e = refused(x -> (x["target"]["board"] = Dict("size" => Dict("width" => 10, "height" => 10), "origin" => [0, 0])))
        @test e isa ArgumentError && occursin("off the board", e.msg)
        # a unitless board (on a reference-only, rigid target: no points to give it a form) is refused at save
        e = try save(q, PlanarTarget(:b76e, nothing, k.rigid, Board(Size2(width = 150.0, height = 77.0)))); nothing catch e; e end
        @test e isa ArgumentError && occursin("board", e.msg) && occursin("plain numbers", e.msg)
        # in a rig, round trip and identity
        rig = Rig{TargetFrame{:a}}(Dict(:a => PosedTarget(PlanarTarget(nothing, pts_inch, k.rigid, bd_inch),
            VgtkCore.EuclideanMap(QuatRotation(1.0, 0.0, 0.0, 0.0, false), SVector(mmq(0.0), mmq(0.0), mmq(0.0))))))
        rp = joinpath(dir, "rig.json")
        save(rp, rig; length_unit = inch)
        @test load(Rig, rp)[:a].target == rig[:a].target && content_id(rp) == content_id(rig)
    end
end

@testitem "rig/v5, target/v2: a code-built board size round-trips bit for bit" setup = [VGIoFixtures] begin
    # Board sizes built in code (`297mm`, `0.142m`), written in mm and in m and
    # read back: every one is the identical quantity. The size is written by
    # `write_length` and read through the same decimal shift (`read_length`).
    using VisualGeometryToolkitCore: PlanarTarget, AbstractTarget, Board, board, Rig, PosedTarget, TargetFrame
    rig_of(bd) = Rig{TargetFrame{:a}}(Dict(:a => PosedTarget(
        PlanarTarget([SVector(1.0mm, 1.0mm), SVector(50.0mm, 1.0mm), SVector(1.0mm, 50.0mm)]; board = bd),
        EuclideanMap(QuatRotation(1.0, 0.0, 0.0, 0.0, false), SVector(0.0m, 0.0m, 0.0m)),
    )))
    roundtrip(rig, dir, u) = (p = joinpath(dir, "rig.json"); save(p, rig; length_unit = u); load(Rig, p))
    mktempdir() do dir
        for (w, h) in ((297mm, 210mm), (100mm, 62.8mm), (0.142m, 0.1m)), u in (mm, m)
            rig = rig_of(Board(Size2(width = w, height = h)))
            back = roundtrip(rig, dir, u)
            @test back.members == rig.members                  # (a Rig has no ==: its members are the value)
            @test board(back[:a].target).size.width === w && board(back[:a].target).size.height === h
        end
        # `142mm` is FlexUnits' multiply, 142 × 0.001 = 0.14200000000000002 m —
        # one ulp off the decimal-nearest 0.142 — and no millimetre file number
        # reads back to it (lengths.jl). Fixed upstream by FlexUnits PR
        # Deduction42/FlexUnits.jl#131 (a correctly rounded decimal scale, so
        # `142mm` is the decimal-nearest value); the day that fix is adopted
        # this broken test succeeds, Test reports it, and it is promoted to @test.
        rig = rig_of(Board(Size2(width = 142mm, height = 100mm)))
        @test_broken roundtrip(rig, dir, mm).members == rig.members   # FlexUnits #131
        @test roundtrip(rig, dir, m).members == rig.members            # a metre file holds it
    end
end

@testitem "rig/v5, target/v2: a planar target's canvas and its texture — present and absent" setup = [VGIoFixtures, TargetRecordFixtures] begin
    using VisualGeometryToolkitCore: PlanarTarget, AbstractTarget, Board, Canvas, CanvasTexture, canvas, board, Rig, PosedTarget,
        TargetFrame, Rigid
    using VisualGeometryToolkitIO: content_id
    bd = Board(Size2(width = mmq(150.0), height = mmq(77.25)), SVector(mmq(-13.9), mmq(0.0)))
    sha = "9f86d081884c7d659a2feaa0c55ad015a3bf4f1b2b0b822cd15d6c15b0f00a08"
    tex = CanvasTexture("textures/board.png", sha, Size2(width = mmq(190.0), height = mmq(97.25)), SVector(mmq(-23.9), mmq(-10.0)))
    cv_tex = Canvas(Size2(width = mmq(210.0), height = mmq(120.5)), SVector(mmq(-30.0), mmq(-20.125)); texture = tex)
    cv_bare = Canvas(Size2(width = mmq(210.0), height = mmq(120.5)), SVector(mmq(-30.0), mmq(-20.125)))
    ident = EuclideanMap(QuatRotation(1.0, 0.0, 0.0, 0.0, false), SVector(mmq(0.0), mmq(0.0), mmq(0.0)))   # the gauge target
    mktempdir() do dir
        p = joinpath(dir, "t.json"); r = joinpath(dir, "rig.json")
        for cv in (cv_tex, cv_bare, nothing), u in (mm, m)
            t = PlanarTarget(nothing, design_points(), Rigid(), bd, cv)
            save(p, t; length_unit = u)
            d = parse_json(read(p, String))
            @test haskey(d.target, :canvas) == (cv !== nothing)
            if cv !== nothing && u == mm
                @test d.target.canvas.size.width == 210.0 && d.target.canvas.origin == [-30.0, -20.125]
                cv.texture === nothing || @test d.target.canvas.texture.path == "textures/board.png" && d.target.canvas.texture.sha256 == sha &&
                    d.target.canvas.texture.size.height == 97.25
            end
            back = load(AbstractTarget, p)
            @test back == t && canvas(back) == cv                          # bit for bit, the texture included
            cv === nothing || @test canvas(back).size.width === cv.size.width && canvas(back).origin === cv.origin
            # a referenced design keeps its canvas; and inside a rig, with its identity
            ref = PlanarTarget(:b76e, nothing, Rigid(), bd, cv)
            save(p, ref; length_unit = u)
            @test load(AbstractTarget, p) == ref
            rig = Rig{TargetFrame{:a}}(Dict(:a => PosedTarget(t, ident)))
            save(r, rig; length_unit = u)
            @test load(Rig, r).members == rig.members
            @test content_id(r) == content_id(rig)
        end
        # the canvas is part of the identity; without one the identity is the board-only one
        rig_of(cv) = Rig{TargetFrame{:a}}(Dict(:a => PosedTarget(PlanarTarget(nothing, design_points(), Rigid(), bd, cv), ident)))
        @test length(unique(content_id.([rig_of(cv_tex), rig_of(cv_bare), rig_of(nothing)]))) == 3
        # malformed canvases are refused by the validator, a board off the page by Core
        save(p, PlanarTarget(nothing, design_points(), Rigid(), bd, cv_tex))
        dd = JSON.parse(read(p, String); dicttype = Dict{String, Any})
        q = joinpath(dir, "e.json")
        refused(f) = (x = deepcopy(dd); f(x); write(q, JSON.json(x)); e = try load(AbstractTarget, q); nothing catch e; e end; e)
        @test refused(x -> (x["target"]["canvas"]["texture"]["sha256"] = "ABC")) isa ArgumentError
        @test refused(x -> (x["target"]["canvas"]["texture"]["path"] = "/abs/board.png")) isa ArgumentError
        @test refused(x -> (x["target"]["canvas"]["size"]["width"] = -1.0)) isa ArgumentError
        @test refused(x -> (x["target"]["canvas"]["extra"] = 1)) isa ArgumentError
        e = refused(x -> (x["target"]["canvas"]["size"] = Dict("width" => 50.0, "height" => 50.0)))
        @test e isa ArgumentError && occursin("off the canvas", e.msg)
        pt = deepcopy(dd); pt["target"] = Dict("type" => "points", "points" => [[0, 0, 0]], "canvas" => dd["target"]["canvas"])
        write(q, JSON.json(pt)); @test (try load(AbstractTarget, q); nothing catch e; e end) isa ArgumentError
    end
end
