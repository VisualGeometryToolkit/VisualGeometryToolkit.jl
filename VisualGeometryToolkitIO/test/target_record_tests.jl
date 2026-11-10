# =============================================================================
# vgio/target/v3 and vgio/rig/v6: a planar target's calibration record — every
# TargetCalibration kind and composition, with and without its covariance,
# round-tripped
# =============================================================================

@testsetup module TargetRecordFixtures
    using VisualGeometryToolkitCore: Rigid, ThinPlateSpline, HeightField, Scale, Aspect, Skew, Uncertain, CovMat,
        grid_nodes, parameter_vector, geom, reference, calibration
    using Statistics: cov
    using StaticArrays: SVector, SMatrix
    using LinearAlgebra: Diagonal
    using Rotations: QuatRotation, RotMatrix, Angle2d
    import VisualGeometryToolkitIO as VgtkIO
    import VisualGeometryToolkitCore as VgtkCore
    using VisualGeometryToolkitCore: mm, m

    export kinds, uncertain, design_points, mmq, identity_rig, fileborn

    # one millimetre number as a reader returns it (lengths.jl: decimal-exact)
    mmq(x) = VgtkIO.read_length(x, mm)
    design_points() = [SVector(mmq(0.0), mmq(0.0)), SVector(mmq(62.8), mmq(0.0)), SVector(mmq(-13.9), mmq(40.25)), SVector(mmq(30.5), mmq(77.0))]
    # a plain metre value as a millimetre file holds it and reads it back
    fileborn(x::Real) = VgtkIO.read_length(VgtkIO.write_length(x * m, mm), mm)

    # every kind, physical, its numbers file-born (each length the metre value
    # nearest a millimetre decimal, so the record writes it back exactly). The
    # thin-plate kinds are pure bending (VGC projects their values: a spline's
    # affine displacement, a field's plane, dropped), so theirs are LITERALS:
    # pure-bending values, which the constructor keeps bit for bit. They are
    # never computed here. The projection is a BLAS product whose last ulp
    # depends on the OpenBLAS kernel the CPU selects (Haswell/Zen vs
    # Sandybridge, Prescott or GitHub's AVX-512 runners), so a computed value,
    # and the content_id pinned on it, would differ by machine. (The literals
    # are the file-born projections, on the Haswell kernel, of
    # fx = [0.01, 0, -0.02, 0.03, 0, 0, 0.01, -0.01, 0] mm,
    # fy = [-0.02, 0.01, 0, 0, 0.04, 0, 0, 0.02, -0.03] mm and
    # h = [0.1, 0, -0.1, 0.2, 0, 0, 0, 0.3, 0] mm.)
    function kinds()
        nodes = [SVector(mmq(p[1]), mmq(p[2])) for p in grid_nodes(3; lo = 0.0, hi = 100.0)]
        E = VgtkCore.EuclideanMap(Angle2d{Float64}(0.01), SVector(mmq(0.5), mmq(-0.25)); check = false)   # the member stores its angle
        S, A, U = Scale(0.001), Aspect(0.002), Skew(-0.003)
        T = ThinPlateSpline(nodes,
            [1.5764702259321875e-5, 1.5158418539816482e-5, 4.552134820311087e-6, 2.9552134820311098e-5, 8.945851100805704e-6,
                1.833956738130031e-5, 3.33956738130032e-6, -7.266716338205075e-6, 1.2126999942289532e-5] .* m,
            [6.897225551491389e-5, 0.00010177539737466663, 9.457853923441935e-5, 9.238168109417197e-5, 0.0001351848229539247,
                9.798796481367743e-5, 9.579110667343005e-5, 0.00011859424853318279, 7.139739039293552e-5] .* m)
        H = HeightField(nodes, [0.0002615206031300322, 0.00020545776593497856, 0.00014939492873992493, 0.00037970911471519355,
            0.0002236462775201399, 0.0002675834403250863, 0.0001978976263003549, 0.0005418347891053013, 0.0002857719519102476] .* m)
        right(cs...) = foldr(∘, cs)   # right-associated, as a target stores a composition
        return (rigid = Rigid(), euclidean = E, scale = S, aspect = A, skew = U, thin_plate_spline = T, height_field = H,
            print_error = A ∘ U, affine_factors = right(E, S, A, U), calibration_chain = H ∘ T, chain_affine = right(H, A, U),
            canonical = right(H, T, E, S, A, U))
    end

    # A rig of targets of every kind — rigid, affine, thin-plate, height field,
    # a composition, Uncertain calibrations, a point target with covariances, an
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
            :b_affine => VgtkCore.PosedTarget(VgtkCore.PlanarTarget(design_points(); calibration = uncertain(k.affine_factors)), X((10.0, 0.0, 900.0))),
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

@testitem "target/v3: every calibration kind round-trips, with and without its covariance" setup = [VGIoFixtures, TargetRecordFixtures] begin
    using VisualGeometryToolkitCore: PlanarTarget, AbstractTarget, Attributed, Uncertain, calibration, points, geom,
        is_resolved, reference, parameter_names, Rigid, members
    using Statistics: cov
    mktempdir() do dir
        p = joinpath(dir, "target.json")
        for (name, c) in pairs(kinds()), cc in (c isa Rigid ? (c,) : (c, uncertain(c))), length_unit in (mm, m)
            t = PlanarTarget(design_points(); calibration = cc)
            save(p, t; length_unit)
            d = parse_json(read(p, String))
            composed = length(members(c)) > 1
            @test d.schema == "vgio/target/v5" && d.target.calibration.type == (composed ? "composition" : String(name))
            composed && @test length(d.target.calibration.compose) == length(members(c))   # flat, outer first
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

@testitem "target/v3: the record as written — tags, parameter order, units" setup = [VGIoFixtures, TargetRecordFixtures] begin
    import VisualGeometryToolkitIO as VgtkIO
    using VisualGeometryToolkitCore: PlanarTarget, AbstractTarget, Rigid, parameter_vector, with_parameters, calibration
    using Random: Random
    k = kinds()
    mktempdir() do dir
        p = joinpath(dir, "target.json")
        mmnum(x) = VgtkIO.write_length(x * m, mm)
        # the affine factors: a euclidean's angle plain (radians) and its
        # translation lengths in the file's unit; scale, aspect and skew one plain number each
        save(p, PlanarTarget(design_points(); calibration = k.euclidean))
        c = parse_json(read(p, String)).target.calibration
        @test c.type == "euclidean" && c.parameters[2:3] == [0.5, -0.25] && c.parameters[1] === 0.01
        save(p, PlanarTarget(design_points(); calibration = k.euclidean); length_unit = m)
        @test parse_json(read(p, String)).target.calibration.parameters[2:3] == [0.0005, -0.00025]
        # the angle is the value (VGC stores the rigid member's rotation as an Angle2d):
        # the record holds the stored angle and value → file → value is exact
        rng = Random.Xoshiro(20261104)
        E0 = VisualGeometryToolkitCore.EuclideanMap(RotMatrix{2, Float64}(0.0), SVector(mmq(0.0), mmq(0.0)); check = false)
        for _ in 1:1000
            ϑ = π * (2 * rand(rng) - 1)
            E = with_parameters(E0, [ϑ, 0.5e-3, -0.25e-3])
            t = PlanarTarget(design_points(); calibration = E)
            save(p, t)
            @test parse_json(read(p, String)).target.calibration.parameters[1] === ϑ
            back = load(AbstractTarget, p)
            @test back == t && parameter_vector(calibration(back)) == parameter_vector(E)
        end
        # a matrix-built member is converted once, where the target stores it, and then round-trips exactly
        for ϑ in range(-3.1, 3.1; length = 301)
            E = VisualGeometryToolkitCore.EuclideanMap(RotMatrix{2, Float64}(ϑ), SVector(mmq(0.5), mmq(-0.25)); check = false)
            t = PlanarTarget(design_points(); calibration = E)
            save(p, t)
            @test load(AbstractTarget, p) == t
        end
        for (kind, x) in ((:scale, 0.001), (:aspect, 0.002), (:skew, -0.003))
            save(p, PlanarTarget(design_points(); calibration = k[kind]); length_unit = m)   # no lengths: the same in any unit
            @test parse_json(read(p, String)).target.calibration == Dict(:type => String(kind), :parameters => [x])
        end
        # the thin-plate kinds: nodes and every parameter lengths, pure bending
        save(p, PlanarTarget(design_points(); calibration = k.thin_plate_spline))
        c = parse_json(read(p, String)).target.calibration
        @test c.type == "thin_plate_spline" && length(c.nodes) == 9 && c.nodes[9] == [100.0, 100.0] && length(c.parameters) == 18
        @test c.parameters == mmnum.(parameter_vector(k.thin_plate_spline))    # fx…, then fy…
        save(p, PlanarTarget(design_points(); calibration = k.height_field))
        c = parse_json(read(p, String)).target.calibration
        @test c.type == "height_field" && c.parameters == mmnum.(parameter_vector(k.height_field))
        # a composition: its members under "compose", flat, in CT's order (the
        # outer map, the height field, first), the covariance once, over every member's parameters in that order
        save(p, PlanarTarget(design_points(); calibration = uncertain(k.calibration_chain)))
        c = parse_json(read(p, String)).target.calibration
        @test c.type == "composition" && !haskey(c, :members)
        @test [x.type for x in c.compose] == ["height_field", "thin_plate_spline"]
        @test all(x -> !haskey(x, :covariance), c.compose) && size(reduce(hcat, c.covariance)) == (27, 27)
        @test c.covariance[1][1] == 1.0 && c.covariance[5][5] == 5.0 && c.covariance[1][27] == -0.15   # every entry lengths²: mm²
        save(p, PlanarTarget(design_points(); calibration = uncertain(k.chain_affine)))
        c = parse_json(read(p, String)).target.calibration
        @test [x.type for x in c.compose] == ["height_field", "aspect", "skew"]
        @test c.covariance[1][1] == 1.0 && c.covariance[10][10] == 1.0e-5 && c.covariance[1][11] == -1.5e-4   # h1² in mm²; a1 plain; h1·u12 in mm
        save(p, PlanarTarget(design_points(); calibration = k.canonical))
        @test [x.type for x in parse_json(read(p, String)).target.calibration.compose] ==
            ["height_field", "thin_plate_spline", "euclidean", "scale", "aspect", "skew"]
        # rigid
        save(p, PlanarTarget(design_points()))
        @test parse_json(read(p, String)).target.calibration == Dict(:type => "rigid")
    end
end

@testitem "target/v3: refusals — plain calibrations, malformed records, the v1 keys" setup = [VGIoFixtures, TargetRecordFixtures] begin
    using VisualGeometryToolkitCore: PlanarTarget, AbstractTarget, HeightField, Aspect, Skew, Scale, grid_nodes, EuclideanMap
    mktempdir() do dir
        p = joinpath(dir, "target.json")
        # a calibration with lengths in plain numbers is refused at save, as plain points are
        @test_throws ArgumentError save(p, PlanarTarget(:b76e; calibration = EuclideanMap(RotMatrix{2, Float64}(0.1), SVector(0.1, 0.0))))
        @test_throws ArgumentError save(p, PlanarTarget(:b76e; calibration = HeightField(grid_nodes(2; lo = 0.0, hi = 1.0))))
        @test !isfile(p)
        # ... one without lengths has no form to refuse: the dimensionless factors are written
        save(p, PlanarTarget(:b76e; calibration = Scale(0.01) ∘ Aspect(0.001) ∘ Skew(0.002)))
        @test load(AbstractTarget, p) == PlanarTarget(:b76e; calibration = Scale(0.01) ∘ Aspect(0.001) ∘ Skew(0.002))
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
        order = "in the canonical order HeightField ∘ ThinPlateSpline ∘ EuclideanMap ∘ Scale ∘ Aspect ∘ Skew"
        @test refused(c -> (c["type"] = "spline"), "unknown calibration type")
        @test refused(c -> (c["compose"] = c["compose"][1:1]), order)                            # one member
        @test refused(c -> push!(c["compose"], Dict("type" => "rigid")), order)                  # a Rigid member
        @test refused(c -> push!(c["compose"], deepcopy(c["compose"][2])), order)                # a kind twice
        @test refused(c -> (c["compose"] = reverse(c["compose"])), "the outer map first")       # application order
        # the pure-bending rule: a spline's values whose interpolant has an affine term are refused, not moved
        @test refused(c -> (c["compose"][2]["parameters"] = c["compose"][2]["parameters"] .+ 0.5), "pure bending")
        @test refused(c -> (c["compose"][1]["parameters"] = c["compose"][1]["parameters"] .+ 0.5), "a plane of heights to the")
        # the retired affine kind, by name, naming its factors and the migration
        aff(c) = (c["compose"] = [c["compose"][1], Dict("type" => "affine", "parameters" => [1.0, 0.0, 0.0, 1.0, 0.0, 0.0])])
        @test refused(aff, "EuclideanMap ∘ Scale ∘ Aspect ∘ Skew") && refused(aff, "vgio_migrate")
        @test refused(c -> push!(c["compose"], Dict("type" => "composition", "compose" => [])), "not a composition")
        @test refused(c -> pop!(c["compose"][1]["parameters"]), "states 9 parameters")
        @test refused(c -> pop!(c["compose"][2]["parameters"]), "states 18 parameters")
        @test refused(c -> delete!(c["compose"][1], "nodes"), "states its `nodes`")
        @test refused(c -> (c["parameters"] = [1.0]), "has no `parameters`")
        @test refused(c -> (c["compose"][1]["covariance"] = [[1.0]]), "carries no `covariance`")
        @test refused(c -> (c["covariance"] = [[1.0]]), "27×27")
        @test refused(c -> (c["compose"] = [Dict("type" => "rigid"), Dict("type" => "rigid", "parameters" => [])]), "has no `parameters`")
        @test refused(c -> delete!(c, "compose"), "states its members, `compose`")
        # the retired record — "calibration_chain", its "members" in application
        # order — in a current file is refused by name, naming the migration
        old_form(c) = (c["type"] = "calibration_chain"; c["members"] = reverse(pop!(c, "compose")))
        @test refused(old_form, "retired") && refused(old_form, "vgio_migrate") && refused(old_form, "\"compose\"")
        @test refused(c -> (c["members"] = pop!(c, "compose")), "vgio_migrate")             # the old key under the new type
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
        # a v2 file (a composition in application order) is read through its migration, as the v3 record
        dd = deepcopy(good)
        dd["schema"] = "vgio/target/v2"
        dd["target"]["calibration"] = VisualGeometryToolkitIO._composition_v3_to_v2(dd["target"]["calibration"])
        write(p, JSON.json(dd))
        q = joinpath(dir, "good.json")
        write(q, JSON.json(good))
        @test VisualGeometryToolkitIO.same_value(load(AbstractTarget, p), load(AbstractTarget, q))
        # a v1 key is an unknown key of the current record; a current record labeled v1 is refused by the v1 step
        dd = deepcopy(good)
        dd["target"]["warp"] = pop!(dd["target"], "calibration")
        write(p, JSON.json(dd))
        e = try load(AbstractTarget, p); nothing catch e; e end
        @test e isa ArgumentError && occursin("target.warp: unknown key", e.msg)
        dd = deepcopy(good)
        dd["schema"] = "vgio/target/v1"
        write(p, JSON.json(dd))
        e = try load(AbstractTarget, p); nothing catch e; e end
        @test e isa ArgumentError && occursin("vgio/target/v1", e.msg) && occursin("was not written by them", e.msg)
    end
end

@testitem "target/v3: a record's parameter counts are VGC's length_mask, for every kind and composition" setup = [VGIoFixtures, TargetRecordFixtures] begin
    using VisualGeometryToolkitCore: length_mask, parameter_vector
    import VisualGeometryToolkitIO as VgtkIO
    for (name, c) in pairs(kinds())
        rec = parse_json(JSON.json(VgtkIO._target_calibration_json(VgtkIO.VgioStyle(; length_unit = mm), c)))   # the parametric record
        @test VgtkIO._n_parameters(rec) == length(length_mask(c)) == length(parameter_vector(c))
    end
    @test_throws ArgumentError VgtkIO._record_parameter_count("affine", ())        # the retired kind: VGC's registry names its successor
    @test_throws ArgumentError VgtkIO._record_parameter_count("bogus", ())
end

@testitem "target/v3: a thin-plate kind pure bending in the unit it was built in reads back ==" setup = [VGIoFixtures, TargetRecordFixtures] begin
    # "No affine term" depends on the nodes' unit (scaling them moves a piston),
    # and VGC's with_unit does not re-project: a field built from millimetre
    # numbers and given the unit is pure bending in mm, not in metres. The
    # reader checks the unit the record was written in (or metres); the writer
    # refuses a field that would not read back.
    using VisualGeometryToolkitCore: PlanarTarget, AbstractTarget, HeightField, ThinPlateSpline, grid_nodes, with_unit, cm, parameter_vector
    nodes_mm = grid_nodes(3; lo = 0.0, hi = 100.0)
    nodes_m = [p .* m for p in grid_nodes(3; lo = 0.0, hi = 0.1)]
    # pure bending in millimetre numbers (projected there), given the unit —
    # `with_unit(plain, mm)` to an ulp; built here file-born (each metre value
    # one a millimetre file writes back exactly), so the round trip is exact
    h_mm = parameter_vector(HeightField(nodes_mm, [0.3, -0.1, 0.4, 0.0, 0.2, -0.5, 0.1, 0.0, 0.25]))
    f_mm = parameter_vector(ThinPlateSpline(nodes_mm, fill(0.0, 9), [0.3, -0.1, 0.4, 0.0, 0.2, -0.5, 0.1, 0.0, 0.25]))
    born(ns) = [mmq.(p) for p in ns]
    hmm = HeightField(born(nodes_mm), fileborn.(h_mm ./ 1000); project = false)
    tmm = ThinPlateSpline(born(nodes_mm), fileborn.(f_mm[1:9] ./ 1000), fileborn.(f_mm[10:18] ./ 1000); project = false)
    @test parameter_vector(hmm) ≈ parameter_vector(with_unit(HeightField(nodes_mm, h_mm), mm)) rtol = 1.0e-14
    hm = HeightField(nodes_m, fileborn.(parameter_vector(HeightField(nodes_m, [3.0, -1.0, 4.0, 0.0, 2.0, -5.0, 1.0, 0.0, 2.5] .* 1.0e-4 .* m))))
    design = [SVector(mmq(10.0 * i), mmq(10.0 * j)) for i in 0:9 for j in 0:9]
    mktempdir() do dir
        p = joinpath(dir, "target.json")
        function roundtrip(c, u)
            t = PlanarTarget(design; calibration = c)
            save(p, t; length_unit = u)
            return load(AbstractTarget, p) == t
        end
        @test parameter_vector(HeightField(hmm.system.nodes .* m, parameter_vector(hmm) .* m)) != parameter_vector(hmm)   # not pure in metres
        @test roundtrip(hmm, mm)                   # built in mm: the mm file reads back
        @test roundtrip(tmm, mm)
        @test roundtrip(hm, m) && roundtrip(hm, mm) # built in m: either file reads back
        # in a third unit neither form holds: the writer refuses, naming why, and writes nothing
        rm(p; force = true)
        for (c, u) in ((hmm, m), (tmm, cm))
            e = try save(p, PlanarTarget(design; calibration = c); length_unit = u); nothing catch e; e end
            @test e isa ArgumentError && occursin("would not read back", e.msg)
        end
        @test !isfile(p)
    end
end

@testitem "rig/v6: a non-rigid posed target, and its content identity" setup = [VGIoFixtures, TargetRecordFixtures] begin
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
            :c => PosedTarget(PlanarTarget(design_points(); calibration = k.affine_factors), X),
        )
    )
    mktempdir() do dir
        p = joinpath(dir, "rig.json")
        save(p, rig)
        @test parse_json(read(p, String)).schema == "vgio/rig/v9"
        back = load(Rig, p)
        @test ids(back) == ids(rig) && all(id -> back[id].target == rig[id].target, ids(rig))
        @test pose(back, :b) isa Uncertain && geom(pose(back, :b)).t == X.t && pose(back, :c).t == X.t
        @test !is_planar(back[:a].target) && eltype(points(back[:a].target)) <: SVector{3}
        @test calibration(back[:c].target) == calibration(rig[:c].target)          # (stored right-associated)
        # the identity does not change with the file's unit, or a read
        pm = joinpath(dir, "rig_m.json")
        save(pm, rig; length_unit = m)
        @test content_id(p) == content_id(pm) == content_id(rig) == content_id(back)
        # the calibration is identity
        rig2 = Rig{TargetFrame{:a}}(Dict(:a => rig[:a], :b => rig[:b], :c => PosedTarget(PlanarTarget(design_points(); calibration = k.rigid), X)))
        @test content_id(rig2) != content_id(rig)
    end
end

@testitem "rig/v6: content_id is pinned on a fixed canonical tree" setup = [VGIoFixtures, TargetRecordFixtures] begin
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
    # (re-pinned 2026-11-03: the rig's composed target, `e_chain`, was an Affine →
    # ThinPlateSpline → HeightField chain, retired with the in-plane compositions;
    # it is HeightField ∘ ThinPlateSpline now. No measurement file names this rig.
    # Re-pinned 2026-11-04 to the id rig/v5 gave it: the identity lists the
    # composition's members in application order, independent of the record.
    # Re-pinned again 2026-11-04 because the VALUE changed with VGC 0.5's
    # gauge-free print error: `b_affine` holds EuclideanMap ∘ Scale ∘ Aspect ∘
    # Skew instead of the retired AffineMap, and `c_tps`, `d_height` and
    # `e_chain`'s thin-plate kinds hold pure bending, their values projected
    # (no affine term in the interpolant); `a_rigid` and `f_points` are as
    # they were. No measurement file names this rig.)
    @test content_id(rig) == "e7d6b28fde6a1ac9d1a13ee041897919192f13b9379c6aa9719e35fec5c356d7"   # poses as the reader returns them
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
    @test content_id(brig) == "f34bee5347b45f2a9d6bca72741e8d2bf47dd2fa7e1b3d12ddb237694b8ef865"   # (re-pinned with `rig`'s)
    @test content_id(brig) != content_id(rig)
    # the same identity from a checked-in file of that rig (every target kind,
    # thin-plate and height-field values with 17 significant digits): a file's
    # identity hashes its numbers as read (the reader does not re-project), so
    # it is the same on every CPU, OpenBLAS kernel, thread count and Julia version
    every_kind = joinpath(@__DIR__, "fixtures", "identity", "rig_every_kind.json")
    @test content_id(every_kind) == content_id(load(Rig, every_kind)) == "f34bee5347b45f2a9d6bca72741e8d2bf47dd2fa7e1b3d12ddb237694b8ef865"
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
        haskey(e["target"]["calibration"], "covariance") && [x["kind"] for x in e["target"]["calibration"]["members"]] == ["thin_plate_spline", "height_field"]   # application order, as v5
    # grid sensors — a SplineLens camera with an image circle, a non-central grid camera — enter under the identity's own names
    using VisualGeometryToolkitCore: PosedCamera, ViewFrame, relabel, KannalaBrandt, EuclideanMap, CameraModel, Ellipse, Point2,
        SplineLens, NoncentralSplineModel, ChartSpline, BoundedCamera, Size2
    using Rotations: RotXYZ
    src = CameraModel(; focal = 31.7, pp = (32.25, 23.875), lens = KannalaBrandt(-4 / (3π^2)))
    circle = Ellipse(Point2(32.25, 23.875), 33.2, 33.2, 0.0)   # the image circle, supplied (a lens model's fold is not one)
    gc = CameraModel(SplineLens, src, (64, 48); grid_size = (8, 7), init = :quasi_interpolate)
    gd = gc.lens.backward
    nc = NoncentralSplineModel(gc.intrinsics, gd, ChartSpline(gd.area, [SVector(1e-3 * gx, -2e-3 * gy, 0.0) for gy in 1:7, gx in 1:8]))
    Xl = relabel(ViewFrame{:g}, ViewFrame{:g}, EuclideanMap(RotMatrix{3, Float64}(RotXYZ(0.0, 0.0, 0.0)), [0.0mm, 0.0mm, 0.0mm]))
    grig = Rig{ViewFrame{:g}}((BoundedCamera(PosedCamera(gc, Xl), Size2(64, 48); lens_coverage = circle),))
    nrig = Rig{ViewFrame{:g}}((BoundedCamera(PosedCamera(nc, Xl), Size2(64, 48); lens_coverage = circle),))
    @test content_id(grig) == "ce2186d905e16b001fd7cd378e37b26fbab24d2901e825c28db088d6a5eeea6a"
    @test content_id(nrig) == "9acbebe8c01572b2fb57d0c4976ca33fc3d158a37b55998d875a0a6cab5d2409"
    cam = only(VgtkIO._identity_tree(grig)["sensors"])["camera"]
    K = gc.intrinsics.K
    @test cam["intrinsics"] == [K[1, 1], K[2, 2], K[1, 3], K[2, 3], K[1, 2]]
    @test cam["lens_coverage"] == Dict("ellipse" => [circle.center[1], circle.center[2], circle.a, circle.b, circle.θ])
    @test sort!(collect(keys(cam))) == ["extent", "intrinsics", "lens", "lens_coverage"]
    @test sort!(collect(keys(only(VgtkIO._identity_tree(nrig)["sensors"])["camera"]))) == ["extent", "intrinsics", "lens_coverage", "noncentral_spline"]
    # the same camera in a rig/v8 file (its old grid body, with the source's pinhole and the circle as its domain):
    # the identity v8 minted for it, pinned before the spline lens (9cfb569c…), is the old identity its migration reports
    mktempdir() do dir
        p = joinpath(dir, "grig.json"); save(p, grig)
        t = JSON.parse(read(p, String))
        Kv = Any[K[1, 1], K[2, 2], K[1, 3], K[2, 3], K[1, 2]]
        t["schema"] = "vgio/rig/v8"
        t["sensors"][1]["camera"] = Dict(
            "type" => "CentralGenericModel", "width" => 64, "height" => 48,
            "calibration_min_x" => 0, "calibration_min_y" => 0, "calibration_max_x" => 63, "calibration_max_y" => 47,
            "grid_width" => 8, "grid_height" => 7, "grid_layout" => "row_major",
            "direction_grid" => [collect(gd.controls[gy, gx]) for gy in 1:7 for gx in 1:8],
            "nominal_intrinsics" => Kv, "domain" => Dict("ellipse" => Any[circle.center[1], circle.center[2], circle.a, circle.b, circle.θ]),
        )
        v8 = joinpath(dir, "grig_v8.json"); write(v8, JSON.json(t, 4))
        r = VgtkIO.vgio_migrate(v8)
        @test r.old_ids == ["9cfb569ce7b14e9e31416b3e70e13af89819a36e5c64f4a8899720579c577d51"] && r.new_id == content_id(grig)
        # a grid sensor that stores no K: the reader supplies it, keyed by sensor id; the sensor's rays are the same
        # camera's; a sensor that holds its K, or an unknown id, refuses one
        delete!(t["sensors"][1]["camera"], "nominal_intrinsics")
        sid = Symbol(t["sensors"][1]["id"])
        nok = joinpath(dir, "grig_v8_nok.json"); write(nok, JSON.json(t, 4))
        before = read(nok)
        given = load(Rig, nok; intrinsics = Dict(sid => K))
        gm = VisualGeometryToolkitCore.model(given.members[1])
        @test gm.intrinsics.K == K && read(nok) == before
        @test all(u -> isapprox(VisualGeometryToolkitCore.backproject(gm, u), VisualGeometryToolkitCore.backproject(gc, u); atol = 1.0e-12),
            [SVector(x, y) for x in 0.5:5.0:63.5 for y in 0.5:5.0:47.5])
        for (q, Ks) in ((v8, Dict(sid => K)), (nok, Dict(:nope => K)))
            e = try load(Rig, q; intrinsics = Ks); nothing catch e; e end
            @test e isa ArgumentError && (occursin("holds its K", e.msg) || occursin("no sensor", e.msg))
        end
    end
end

@testitem "target/v3: a planar target's board — present and absent" setup = [VGIoFixtures, TargetRecordFixtures] begin
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
        cases = [[(c, u, bd, design_points()) for c in (k.rigid, k.affine_factors, uncertain(k.calibration_chain)) for u in (mm, m)];
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

@testitem "rig/v6, target/v3: a code-built board size round-trips bit for bit" setup = [VGIoFixtures] begin
    # Board sizes built in code (`297mm`, `0.142m`), written in mm and in m and
    # read back: every one is the identical quantity. The size is written by
    # `write_length` and read through the same decimal shift (`read_length`).
    using VisualGeometryToolkitCore: PlanarTarget, AbstractTarget, Board, board, Rig, PosedTarget, TargetFrame, base_quantity
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
            # the board stores the Float64 base form (`297mm` is an exact Rational until stored)
            @test board(back[:a].target).size.width === base_quantity(w) && board(back[:a].target).size.height === base_quantity(h)
        end
        # `142mm` is exact: VisualGeometryToolkitCore's registry scales are rationals
        # (mm = 1//1000), so the stored length is the correctly rounded 71//500 m,
        # the decimal-nearest 0.142, and a millimetre file holds it. (With
        # FlexUnits' Float64 scales it was 142 × 0.001 = 0.14200000000000002 m,
        # which no millimetre file number reads back to.)
        rig = rig_of(Board(Size2(width = 142mm, height = 100mm)))
        @test board(rig[:a].target).size.width === 0.142m
        @test roundtrip(rig, dir, mm).members == rig.members
        @test roundtrip(rig, dir, m).members == rig.members            # a metre file holds it
    end
end

@testitem "rig/v6, target/v3: a planar target's canvas and its texture — present and absent" setup = [VGIoFixtures, TargetRecordFixtures] begin
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

@testitem "target/v3: a placement — the calibration in plain chart numbers, the placement in the file's unit" setup = [VGIoFixtures, TargetRecordFixtures] begin
    import VisualGeometryToolkitIO as VgtkIO
    using VisualGeometryToolkitCore: PlanarTarget, AbstractTarget, SimilarityMap, HeightField, ThinPlateSpline, Aspect, Skew, Rigid,
        Uncertain, CovMat, Rig, PosedTarget, TargetFrame, grid_nodes, calibration, parameter_vector, geom, placement
    using Statistics: cov
    using LinearAlgebra: Diagonal
    # VGC's placement (a SimilarityMap{3}): the calibration acts in the board's dimensionless
    # chart ([0, 1]², RadiometricCamera's page), plain numbers whatever the
    # points' form; the placement (origin, side) carries the lengths.
    chart = grid_nodes(3; lo = 0.0, hi = 1.0)
    H = HeightField(chart, [0.01, 0.0, -0.01, 0.02, 0.0, 0.0, 0.0, 0.03, 0.0])          # projected: pure bending, plain
    c = Uncertain(foldr(∘, (H, Aspect(0.002), Skew(-0.001))), CovMat(SMatrix{11, 11, Float64}(Diagonal([1.0e-6 * i for i in 1:11]))))
    placed_at(o, side) = SimilarityMap(side, one(RotMatrix{3, Float64}), SVector(o[1], o[2], 0.0))   # plain metres: the target is physical
    P = placed_at(SVector(-0.0139, 0.0), 0.15)
    t = PlanarTarget(design_points(); calibration = c, placement = P)
    mktempdir() do dir
        p = joinpath(dir, "t.json")
        for u in (mm, m)
            save(p, t; length_unit = u)
            d = parse_json(read(p, String))
            # the placement in the file's unit, exactly; the calibration's numbers as held, in any unit
            @test d.target.placement.origin == (u == mm ? [-13.9, 0.0] : [-0.0139, 0.0]) && d.target.placement.side == (u == mm ? 150.0 : 0.15)
            @test d.target.calibration.compose[1].nodes == [[q[1], q[2]] for q in chart]
            @test d.target.calibration.compose[1].parameters == collect(parameter_vector(H))
            @test d.target.calibration.covariance == [[Float64(cov(c)[i, j]) for j in 1:11] for i in 1:11]   # never scaled by the unit
            back = load(AbstractTarget, p)
            @test back == t && placement(back) == P
            @test calibration(back) == calibration(t) && parameter_vector(geom(calibration(back))) == parameter_vector(geom(c))
        end
        # a placement on a points target is refused
        q = joinpath(dir, "points.json")
        save(q, VisualGeometryToolkitCore.PointTarget([SVector(mmq(0.0), mmq(1.0), mmq(2.0))]))
        dd = JSON.parse(read(q, String); dicttype = Dict{String, Any})
        dd["target"]["placement"] = Dict("origin" => [0.0, 0.0], "side" => 1.0)
        write(q, JSON.json(dd))
        e = try load(AbstractTarget, q); nothing catch e; e end
        @test e isa ArgumentError && occursin("no `placement`", e.msg)
        # and a placement's side must be positive
        save(p, t)
        dd = JSON.parse(read(p, String); dicttype = Dict{String, Any})
        dd["target"]["placement"]["side"] = 0.0
        write(q, JSON.json(dd))
        e = try load(AbstractTarget, q); nothing catch e; e end
        @test e isa ArgumentError && occursin("side is > 0", e.msg)
        # a placed target without lengths is refused when written: its placement's numbers are metres only on a
        # physical target
        plain = PlanarTarget(:page; calibration = c, placement = P)
        e = try save(p, plain); nothing catch e; e end
        @test e isa ArgumentError && occursin("placed target", e.msg)
    end
    # the identity holds the placement (origin and side, in metres): placed and unplaced differ, and so do two placements
    X = EuclideanMap(one(RotMatrix{3, Float64}), [mmq(0.0), mmq(0.0), mmq(0.0)])
    rig(tt) = Rig{TargetFrame{:a}}(Dict(:a => PosedTarget(tt, X)))
    placed = PlanarTarget(design_points(); calibration = Rigid(), placement = P)
    unplaced = PlanarTarget(design_points(); calibration = Rigid())
    moved = PlanarTarget(design_points(); calibration = Rigid(), placement = placed_at(SVector(-0.0139, 0.0), 0.151))
    ids = VgtkIO.content_id.(rig.((placed, unplaced, moved)))
    @test allunique(ids)
    @test VgtkIO._identity_tree(rig(placed))["targets"][1]["target"]["placement"] == Dict("origin" => [-0.0139, 0.0], "side" => 0.15)
    mktempdir() do dir
        p = joinpath(dir, "rig.json")
        save(p, rig(t); length_unit = m)
        @test VgtkIO.content_id(p) == VgtkIO.content_id(rig(t)) == VgtkIO.content_id(load(Rig, p))
    end
end
