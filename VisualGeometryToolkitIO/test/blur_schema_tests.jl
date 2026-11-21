# vgio/blur/v2.0 (blur_schema.jl): named GaussianBlurs in the storage they hold, each on its plane (a pixel frame, a
# view's image frame, a target's plane), the body nested in another record, and the validator's refusals.
@testitem "blur: named blurs round trip, each on its plane in the storage it holds; the body nests in another record" setup = [VGIoFixtures] begin
    using StaticArrays: @SMatrix, SDiagonal
    using VisualGeometryToolkitCore: GaussianBlur, CovMat, relabel, ImageFrame
    using Statistics: cov
    mktempdir() do dir
        # blurs, each on the plane it lives on, in the storage it holds
        on(F, Σ) = GaussianBlur(relabel(F, CovMat(Σ)))
        B = Dict(:R => on(PixelFrame{:colmap}, VisualGeometryToolkitCore.ScaledIdentity{2}(0.36)), :G => on(PixelFrame{:opencv}, SDiagonal(0.25, 0.49)),
                 :B => on(ImageFrame{:cam1}, @SMatrix [0.5 0.1; 0.1 0.4]))
        p = joinpath(dir, "blur.json")
        save(p, B)
        bb = load(Dict{Symbol, GaussianBlur}, p)
        @test all(typeof(cov(bb[k])) == typeof(cov(B[k])) && cov(bb[k]) == cov(B[k]) for k in keys(B))   # the frame is in the type
        db = JSON.parse(read(p, String))
        @test db["schema"] == "vgio/blur/v2.0"
        t = Dict(b["name"] => b for b in db["blurs"])
        @test [b["name"] for b in db["blurs"]] == ["B", "G", "R"]                            # sorted
        @test t["R"]["pixel_frame"] == "colmap" && collect(keys(t["R"]["covariance"])) == ["isotropic"] && t["R"]["covariance"]["isotropic"] == 0.36
        @test t["G"]["pixel_frame"] == "opencv" && t["G"]["covariance"]["diagonal"] == [0.25, 0.49]
        @test t["B"]["frame"] == "image:cam1" && t["B"]["covariance"] == [[0.5, 0.1], [0.1, 0.4]] && !haskey(t["B"], "pixel_frame")
        @test !any(haskey(b, "unit") for b in db["blurs"])                                   # no per-blur unit
        # the same body nested in another record (RadiometricCamera's per-channel in-focus blur): lowered and lifted
        # through the one pair of methods the file uses, byte for byte the file's `blurs`
        s = VisualGeometryToolkitIO.VgioStyle(; length_unit = mm)
        nested = JSON.parse(JSON.json((camera = (in_focus_blur = B,),); style = s))["camera"]["in_focus_blur"]
        @test nested == db["blurs"] && [b["name"] for b in nested] == ["B", "G", "R"]
        lifted = JSON.lift(s, Dict{Symbol, GaussianBlur}, JSON.parse(JSON.json(nested)))
        @test lifted isa Dict{Symbol, GaussianBlur} && all(cov(lifted[k]) == cov(B[k]) && typeof(cov(lifted[k])) == typeof(cov(B[k])) for k in keys(B))
        @test_throws ArgumentError JSON.lift(s, Dict{Symbol, GaussianBlur}, JSON.parse(JSON.json([nested[1], nested[1]])))   # a repeated name
    end
end

@testitem "blur: a blur names its plane — a sensor's in px², a print's in length², both exact" setup = [VGIoFixtures] begin
    using StaticArrays: @SMatrix, SDiagonal
    using VisualGeometryToolkitCore: GaussianBlur, has_units, with_unit, μm, CovMat, relabel, TargetPlaneFrame, ScaledIdentity
    using Statistics: cov
    on(F, Σ) = GaussianBlur(relabel(F, CovMat(Σ)))
    print_blur = with_unit(on(TargetPlaneFrame{:board}, SDiagonal(7.84e-10, 1.8225e-10)), m)          # Σ in m², decimal (σ = 0.028, 0.0135 mm)
    print_full = with_unit(on(TargetPlaneFrame{:board}, @SMatrix([7.84e-10 1.0e-11; 1.0e-11 1.8225e-10])), m)
    print_iso = with_unit(on(TargetPlaneFrame{:board}, ScaledIdentity{2}(4.0e-10)), m)
    camera = on(PixelFrame{:colmap}, SDiagonal(0.6^2, 0.45^2))
    B = Dict(:print => print_blur, :print_full => print_full, :print_iso => print_iso, :camera => camera)
    mktempdir() do dir
        p = joinpath(dir, "blur.json")
        save(p, B)
        bb = load(Dict{Symbol, GaussianBlur}, p)
        for k in keys(B)                                                                   # exact: storage, frame, type, every entry
            @test typeof(bb[k]) == typeof(B[k]) && has_units(bb[k]) == has_units(B[k]) && Tuple(cov(bb[k])) == Tuple(cov(B[k]))
        end
        d = JSON.parse(read(p, String))
        @test d["units"]["length"] == "mm"
        t = Dict(b["name"] => b for b in d["blurs"])
        @test t["camera"]["pixel_frame"] == "colmap" && t["camera"]["covariance"]["diagonal"] == [0.6^2, 0.45^2]   # px² as they are
        @test t["print"]["frame"] == "target:board" && t["print"]["covariance"]["diagonal"] == [7.84e-4, 1.8225e-4]  # m² → mm², a shift
        @test t["print_full"]["covariance"][1][2] == 1.0e-5 && t["print_iso"]["covariance"]["isotropic"] == 4.0e-4
        save(p, B; length_unit = μm)
        d = JSON.parse(read(p, String))
        t = Dict(b["name"] => b for b in d["blurs"])
        bb = load(Dict{Symbol, GaussianBlur}, p)
        @test d["units"]["length"] == unit_string(μm) && t["print"]["covariance"]["diagonal"] == [784.0, 182.25]
        @test t["camera"]["covariance"]["diagonal"] == [0.6^2, 0.45^2] && all(k -> Tuple(cov(bb[k])) == Tuple(cov(B[k])), keys(B))
        # the writer: a blur in px quantities on a pixel plane is px²; an unlabeled blur names no plane; a plain blur on a
        # target's plane holds lengths without units; a length is not a variance
        pxq = on(PixelFrame{:colmap}, SDiagonal(0.25px^2 |> VisualGeometryToolkitCore.base_quantity, 0.5px^2 |> VisualGeometryToolkitCore.base_quantity))
        save(p, Dict(:q => pxq))
        @test JSON.parse(read(p, String))["blurs"][1]["covariance"]["diagonal"] == [0.25, 0.5]
        @test cov(load(Dict{Symbol, GaussianBlur}, p)[:q]) == SDiagonal(0.25, 0.5)
        q = joinpath(dir, "refused.json")
        @test_throws ArgumentError save(q, Dict(:u => GaussianBlur(0.6)))                                    # unlabeled
        @test_throws ArgumentError save(q, Dict(:t => on(TargetPlaneFrame{:board}, SDiagonal(0.25, 0.5))))           # plain on a print
        @test_throws ArgumentError save(q, Dict(:t => with_unit(on(TargetPlaneFrame, SDiagonal(7.84e-10, 1.8225e-10)), m)))   # the bare plane names no target
        @test_throws ArgumentError save(q, Dict(:t => on(TargetPlaneFrame{:board}, SDiagonal(1.0m, 2.0m))))          # a length, not an area
        @test_throws ArgumentError save(q, Dict(:c => on(PixelFrame{:colmap}, SDiagonal(VisualGeometryToolkitCore.base_quantity(1.0m^2), VisualGeometryToolkitCore.base_quantity(2.0m^2)))))   # an area on a sensor
        @test !isfile(q)
        # on read: a plane of the kind's, stated once
        save(p, B)
        function refused(f)
            edited = JSON.parse(read(p, String); dicttype = Dict{String, Any}); f(edited["blurs"][1])
            r = joinpath(dir, "bad.json"); write(r, JSON.json(edited))
            return try load(Dict{Symbol, GaussianBlur}, r); false catch e; e isa ArgumentError end
        end
        @test !refused(identity)
        @test refused(b -> delete!(b, "pixel_frame"))                                        # no plane
        @test refused(b -> b["frame"] = "target:board")                                      # two planes
        @test refused(b -> b["pixel_frame"] = "lower_left")                                  # not a registered pixel frame
        @test refused(b -> (delete!(b, "pixel_frame"); b["frame"] = "view:a1f2"))             # not a blur's plane (a 3-D frame)
        @test refused(b -> (delete!(b, "pixel_frame"); b["frame"] = "image:"))
        @test refused(b -> b["unit"] = "px^2")                                               # IO 0.9's per-blur unit
    end
end

@testitem "blur: the body nests in another record (VgioStyle), an IO 0.9 blur body too" setup = [VGIoFixtures] begin
    import VisualGeometryToolkitIO as VgtkIO
    using StaticArrays: @SMatrix, SDiagonal
    using VisualGeometryToolkitCore: with_unit, GaussianBlur, has_units, CovMat, relabel, TargetPlaneFrame
    using Statistics: cov
    using VisualGeometryToolkitIO: VgioStyle
    s = VgioStyle(; length_unit = mm)
    roundtrip(T, x) = VgtkIO.nested_roundtrip(T, x; style = s)
    g = GaussianBlur(relabel(PixelFrame{:colmap}, CovMat(@SMatrix [0.5 0.1; 0.1 0.4])))
    @test cov(roundtrip(GaussianBlur, g)) == cov(g) && keys(JSON.lower(s, g)) == (:pixel_frame, :covariance) && JSON.lower(s, g).pixel_frame == "colmap"
    p = with_unit(GaussianBlur(relabel(TargetPlaneFrame{:board}, CovMat(SDiagonal(4.0e-10, 9.0e-10)))), m)      # a print's blur, m² (σ = 0.02, 0.03 mm)
    q = roundtrip(GaussianBlur, p)
    @test has_units(q) && typeof(q) == typeof(p) && Tuple(cov(q)) == Tuple(cov(p))
    @test JSON.lower(s, p).frame == "target:board" && JSON.lower(s, p).covariance["diagonal"] ≈ [0.02^2, 0.03^2]   # mm² in an mm record
    # an IO 0.9 blur body (`{"unit", "covariance"}`), as another package's record of an earlier version embeds it: read as
    # the unlabeled blur it held; its unit "px^2" or a length squared, told apart by NAME (to FlexUnits px, px³ and pd²
    # are all the same unitless scale-1 unit, so a dimension check accepts them all); a length is not an area
    old(u, c) = JSON.parse(JSON.json(Dict("unit" => u, "covariance" => c)))
    b9 = JSON.lift(s, GaussianBlur, old("px^2", Dict("diagonal" => [0.25, 0.5])))
    @test !has_units(b9) && Tuple(cov(b9)) == (0.25, 0.0, 0.0, 0.5)
    m9 = JSON.lift(s, GaussianBlur, old("mm^2", Dict("isotropic" => 4.0e-4)))
    @test has_units(m9) && Tuple(cov(ustrip(m, m9))) == (4.0e-10, 0.0, 0.0, 4.0e-10)
    refused_unit(u) = try JSON.lift(s, GaussianBlur, old(u, Dict("isotropic" => 1.0))); false catch e; e isa ArgumentError end
    @test refused_unit("px")
    @test refused_unit("px^3")
    @test refused_unit("pd^2")
    @test refused_unit("mm")
    @test refused_unit("mm^3")
    @test refused_unit("mm*px")
    @test refused_unit("furlong^2")
end

@testitem "blur: the validator" setup = [VGIoFixtures] begin
    import VisualGeometryToolkitIO as VgtkIO
    using VisualGeometryToolkitCore: GaussianBlur, CovMat, relabel
    mktempdir() do dir
        edited(src, f) = VgtkIO.edited_copy(f, src)
        b = joinpath(dir, "b.json")
        on(F, σ) = GaussianBlur(relabel(F, CovMat(VisualGeometryToolkitCore.ScaledIdentity{2}(σ^2))))
        save(b, Dict(:R => on(PixelFrame{:colmap}, 0.5), :G => on(PixelFrame{:colmap}, 0.5)))
        @test_throws ArgumentError load(Dict{Symbol, GaussianBlur}, edited(b, t -> t["blurs"][2]["name"] = t["blurs"][1]["name"]))   # names unique
    end
end

@testitem "covariance storage: a point target's structured point covariances" setup = [VGIoFixtures] begin
    using StaticArrays: SDiagonal
    using VisualGeometryToolkitCore: PointTarget, AbstractTarget, Uncertain, CovMat, ScaledIdentity, points, cov
    t = PointTarget([Uncertain(SVector(0.1, 0.2, 0.0) .* m, CovMat(SDiagonal(1.0e-8, 4.0e-8, 9.0e-8))),
                     Uncertain(SVector(0.3, 0.2, 0.0) .* m, CovMat(ScaledIdentity{3}(1.0e-8)))])
    mktempdir() do dir
        p = joinpath(dir, "t.json")
        save(p, t)
        b = load(AbstractTarget, p)
        @test [typeof(parent(cov(q))) for q in points(b)] == [typeof(parent(cov(q))) for q in points(t)]
        @test all(SMatrix(cov(q)) ≈ SMatrix(cov(r)) for (q, r) in zip(points(b), points(t)))
        @test haskey(JSON.parse(read(p, String))["target"]["points"][1], "covariance")   # the point vocabulary: per point
    end
end
