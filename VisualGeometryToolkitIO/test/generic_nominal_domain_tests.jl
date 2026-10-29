# =============================================================================
# vgio/generic_camera/v2 and rig/v5: a generic model's nominal pinhole and domain
# =============================================================================
#
# A generic model built from a source camera carries the source's pinhole
# (`nominal_intrinsics`) and the image circle it is given (`domain`). Both
# travel in the model's body, optional and additive in the unreleased schemas,
# bit for bit; absent, the body and the identity are the bare grid's.

@testitem "generic models: nominal pinhole and domain round trip" setup = [VGIoFixtures] begin
    using VisualGeometryToolkitCore: Rig, PosedCamera, ViewFrame, Vec3, relabel, model, LogicalIntrinsics, CameraCalibrationMat, CameraModel,
        nominal_intrinsics, Ellipse, Point2, WholePlane, KannalaBrandt, Rect2, CentralGenericModel, NoncentralGenericModel
    using VisualGeometryToolkitIO: content_id
    using StaticArrays: SMatrix
    using Rotations: RotXYZ

    src = CameraModel(; focal = 31.7, pp = (32.25, 23.875), lens = KannalaBrandt(-4 / (3π^2)))
    circle = Ellipse(Point2(32.25, 23.875), 33.2, 33.2, 0.0)   # the image circle, supplied (a lens model's fold is not one)
    gc = CentralGenericModel(src, (64, 48); grid_size = (8, 7), init = :quasi_interpolate, domain = circle)
    @test nominal_intrinsics(gc) isa LogicalIntrinsics && gc.domain isa Ellipse
    P = [Vec3{Float64}(1e-3 * gx, -2e-3 * gy, 0.0) for gy in 1:7, gx in 1:8]
    nc = NoncentralGenericModel(64, 48, 0, 0, 63, 47, gc.grid, P; unit_controls = false, gc.nominal, gc.domain)
    bare = CentralGenericModel(64, 48, 0, 0, 63, 47, gc.grid; unit_controls = false)
    same(a, b) = a.grid == b.grid && nominal_intrinsics(a).K == nominal_intrinsics(b).K && a.domain == b.domain
    mktempdir() do dir
        p = joinpath(dir, "g.json"); save(p, gc)
        d = parse_json(read(p, String))
        @test length(d.nominal_intrinsics) == 5 && length(d.domain.ellipse) == 5
        back = load(CentralGenericModel, p)
        @test same(back, gc)                                               # bit for bit: ==, not ≈
        pn = joinpath(dir, "n.json"); save(pn, nc)
        bn = load(NoncentralGenericModel, pn)
        @test bn.directions == nc.directions && bn.points == nc.points && bn.domain == nc.domain &&
              nominal_intrinsics(bn).K == nominal_intrinsics(nc).K
        pb = joinpath(dir, "b.json"); save(pb, bare)
        @test !occursin("nominal_intrinsics", read(pb, String)) && !occursin("domain", read(pb, String))
        bb = load(CentralGenericModel, pb)
        @test nominal_intrinsics(bb) === nothing && bb.domain === WholePlane()
        # malformed values are refused by the reader (they would not read back as the file's numbers, or give an
        # infinite K⁻¹): an ellipse stored minor axis first, a semi-axis ≤ 0, a focal length 0; the valid file re-saves
        # bit for bit
        text = read(p, String)
        for (desc, f!) in (
                "a < b" => t -> (t[:domain][:ellipse][3] = t[:domain][:ellipse][4] / 2),
                "b ≤ 0" => t -> (t[:domain][:ellipse][4] = 0.0),
                "fx = 0" => t -> (t[:nominal_intrinsics][1] = 0.0),
            )
            t = parse_json(text); f!(t)
            bad = joinpath(dir, "bad.json"); write(bad, JSON.json(t))
            @test_throws ArgumentError load(CentralGenericModel, bad)
        end
        p2 = joinpath(dir, "g2.json"); save(p2, back)
        @test read(p2, String) == text
        # a domain of another region type is refused by the writer
        odd = CentralGenericModel(64, 48, 0, 0, 63, 47, gc.grid; unit_controls = false, domain = Rect2{Float64}(0, 0, 64, 48) ∩ gc.domain)
        @test_throws ArgumentError save(joinpath(dir, "odd.json"), odd)
        # in a rig: the body carries them, the identity covers them
        Xl = relabel(ViewFrame{:g}, ViewFrame{:g}, EuclideanMap(RotMatrix{3, Float64}(RotXYZ(0.0, 0.0, 0.0)), [0.0mm, 0.0mm, 0.0mm]))
        rig = Rig{ViewFrame{:g}}((PosedCamera(gc, Xl),))
        rb = Rig{ViewFrame{:g}}((PosedCamera(bare, Xl),))
        pr = joinpath(dir, "rig.json"); save(pr, rig)
        back_rig = load(Rig, pr)
        @test same(model(back_rig[1]), gc)
        @test content_id(back_rig) == content_id(rig) == content_id(pr)
        @test content_id(rig) != content_id(rb)
    end
end
