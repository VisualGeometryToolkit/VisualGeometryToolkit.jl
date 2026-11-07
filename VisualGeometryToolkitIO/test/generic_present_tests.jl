# =============================================================================
# vgio/generic_camera/v3: the nodes a grid model holds
# =============================================================================
#
# A v3 record lists the held nodes as runs per lattice row (`present`) and its
# grids over them only; a v2 record, every node held, reads as every node
# present; a rig sensor's body carries `present` only for a masked model.

@testitem "generic_camera v3: present nodes round trip, every case" setup = [VGIoFixtures] begin
    import VisualGeometryToolkitIO as VgtkIO
    using VisualGeometryToolkitCore: Rig, PosedCamera, ViewFrame, Vec3, relabel, model, present_nodes, CentralGenericModel,
        NoncentralGenericModel, backproject, in_present_domain
    using Rotations: RotXYZ

    fixtures = joinpath(@__DIR__, "fixtures")
    nanslot(v) = all(isnan, v)

    # the masked fixture: a row without nodes, a hole (a row of two runs), rows cut at both ends
    pm = joinpath(fixtures, "units", "generic_camera_masked.json")
    gm = load(CentralGenericModel, pm)
    P = present_nodes(gm)
    @test !all(P) && !any(P[1, :]) && P[4, :] == Bool[1, 1, 1, 0, 1, 1, 1] && P[2, :] == Bool[0, 0, 1, 1, 1, 1, 0]
    @test all(nanslot, gm.grid[.!P]) && !any(nanslot, gm.grid[P])     # an absent node has no value
    d = parse_json(read(pm, String))
    @test d.schema == "vgio/generic_camera/v3" && length(d.direction_grid) == count(P)
    @test d.present[3] == [3, [[0, 2], [4, 6]]]                       # 0-based, inclusive
    mktempdir() do dir
        out = joinpath(dir, "m.json"); save(out, gm)
        @test read(out, String) == read(pm, String)                   # the file a masked model writes is its own

        # v2 (every node) reads as the all-present model: the same numbers as its v3 rewrite
        old = joinpath(fixtures, "old", "generic_camera_v2.json")
        a = load(NoncentralGenericModel, old)
        b = load(NoncentralGenericModel, joinpath(fixtures, "units", "generic_camera.json"))
        @test a.present === nothing && b.present === nothing && all(present_nodes(a))
        @test a.directions == b.directions && a.points == b.points
        mig = VgtkIO.migrate_record(JSON.parse(read(old, String)))
        @test mig["schema"] == "vgio/generic_camera/v3" && !haskey(mig, "present")         # every node: no `present` to write
        @test !haskey(parse_json(read(joinpath(fixtures, "units", "generic_camera.json"), String)), :present)

        # a masked non-central model: both grids over the held nodes
        gh, gw = size(gm.grid)
        pts = [Vec3{Float64}(0.001gx, -0.001gy, 0.0) for gy in 1:gh, gx in 1:gw]
        gn = NoncentralGenericModel(64, 48, 0, 0, 63, 47, gm.grid, pts; unit_controls = false, present = P)
        pn = joinpath(dir, "n.json"); save(pn, gn)
        bn = load(NoncentralGenericModel, pn)
        @test present_nodes(bn) == P && bn.directions[P] == gn.directions[P] && bn.points[P] == gn.points[P]
        @test length(parse_json(read(pn, String)).point_grid) == count(P)
        # (the fixture's mask leaves no 4 × 4 window whole: a corner-cut lens evaluates where its windows are whole, as read back)
        Pc = trues(gh, gw); Pc[1, 1] = false
        full_dirs = map(v -> all(isnan, v) ? Vec3{Float64}(0, 0, 1) : v, gm.grid)
        gc = NoncentralGenericModel(64, 48, 0, 0, 63, 47, full_dirs, pts; unit_controls = false, present = Pc)
        pc = joinpath(dir, "c.json"); save(pc, gc); bc = load(NoncentralGenericModel, pc)
        inside = [SVector(x, y) for x in 0.5:2.0:63.5 for y in 0.5:2.0:47.5 if in_present_domain(gc, SVector(x, y))]
        @test !isempty(inside) && all(c -> backproject(bc, c).direction == backproject(gc, c).direction, inside)
        @test !in_present_domain(bc, SVector(0.5, 0.5)) && all(isnan, backproject(bc, SVector(0.5, 0.5)).direction)

        # a rig sensor's body: `present` for a masked model only; an unmasked one writes the rig/v6 body it always did
        Xl = relabel(ViewFrame{:g}, ViewFrame{:g}, EuclideanMap(RotMatrix{3, Float64}(RotXYZ(0.0, 0.0, 0.0)), [0.0mm, 0.0mm, 0.0mm]))
        full = CentralGenericModel(64, 48, 0, 0, 63, 47, map(v -> nanslot(v) ? Vec3{Float64}(0, 0, 1) : v, gm.grid);
                                   unit_controls = false)
        for (g, masked) in ((gm, true), (full, false))
            pr = joinpath(dir, "rig_$(masked).json"); save(pr, Rig{ViewFrame{:g}}((PosedCamera(g, Xl),)))
            body = parse_json(read(pr, String)).sensors[1].camera
            @test haskey(body, :present) == masked
            back = model(load(Rig, pr)[1])
            @test present_nodes(back) == present_nodes(g) && back.grid[present_nodes(g)] == g.grid[present_nodes(g)]
        end
        # the identity of an unmasked model's rig is its identity before the mask; a mask changes it
        rid(g) = VgtkIO.content_id(Rig{ViewFrame{:g}}((PosedCamera(g, Xl),)))
        @test rid(full) != rid(CentralGenericModel(64, 48, 0, 0, 63, 47, full.grid; unit_controls = false, present = P))
    end
end

@testitem "generic_camera v3: a malformed `present` is refused by name" setup = [VGIoFixtures] begin
    import VisualGeometryToolkitIO as VgtkIO
    using VisualGeometryToolkitCore: CentralGenericModel
    pm = joinpath(@__DIR__, "fixtures", "units", "generic_camera_masked.json")
    base = JSON.parse(read(pm, String))
    refused(f, why) = mktempdir() do dir
        t = deepcopy(base); f(t)
        p = joinpath(dir, "bad.json"); write(p, JSON.json(t))
        try
            load(CentralGenericModel, p); false
        catch e
            e isa ArgumentError && occursin(why, e.msg)
        end
    end
    @test refused(t -> delete!(t, "present"), "grid_width × grid_height")                      # every node then: too few controls
    @test refused(t -> (t["present"][1][2] = Any[Any[2, 3], Any[3, 5]]), "apart")              # overlapping runs
    @test refused(t -> (t["present"][1][2] = Any[Any[2, 3], Any[4, 5]]), "apart")              # adjacent: not one maximal run
    @test refused(t -> (t["present"][2][1] = 1), "increasing")                                  # a row twice
    @test refused(t -> (t["present"][end][1] = 6), "grid_height")
    @test refused(t -> (t["present"][1][2] = Any[Any[2, 7]]), "grid_width")
    @test refused(t -> pop!(t["direction_grid"]), "direction_grid")                             # one node short
    @test refused(t -> (t["present"][1][2] = Any[Any[3, 2]]), "col_lo ≤ col_hi")
    @test refused(t -> (t["schema"] = "vgio/generic_camera/v2"), "holds every node")            # v2 wrote no `present`
end
