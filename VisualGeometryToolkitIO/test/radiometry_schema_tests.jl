# The radiometric kinds (radiometry_schema.jl): vgio/vignetting/v2.0 (VGC's tensor-product, radial and log fields, rebuilt
# bit for bit; positions in any registered pixel frame, a radial profile's chart always stated), vgio/environment_light/v2.0
# (an EnvironmentLight of a SphericalHarmonicMap, `max_degree`; a plain light refused; the basis convention pinned),
# vgio/blur/v2.0 (named GaussianBlurs in the storage they hold, each on its plane: a pixel frame, a view's image frame,
# a target's plane), and the validator's refusals.
@testitem "radiometry kinds: vignetting, environment light, blurs round trip" setup = [VGIoFixtures] begin
    using StaticArrays: @SMatrix, SDiagonal
    using VisualGeometryToolkitCore: AbstractVignetting, TensorProductVignetting, RadialVignetting, LogVignetting, with_control_points,
        control_points, knot_vectors, EnvironmentLight, SphericalHarmonicMap, coefficients, center, radius, with_unit, max_degree,
        GaussianBlur, Rect, has_units, CovMat, relabel, ImageFrame
    using Statistics: cov
    import FlexUnits
    w(i, j) = 0.5 + 0.0371 * i + 0.0193 * j * j                       # fixed control points, not ones (every entry distinct)
    mktempdir() do dir
        tv = with_control_points(TensorProductVignetting((5, 4), Rect(0.0, 0.0, 640.0, 480.0); degree = (2, 3)), [w(i, j) for i in 1:5, j in 1:4])
        rv = with_control_points(RadialVignetting(7, SVector(320.3, 241.7), 401.25), [w(i, 1) for i in 1:7])
        pv = with_control_points(RadialVignetting(8, SVector(320.3, 241.7), 401.25; chart = :projective), [w(i, 1) for i in 1:8])
        for v in (tv, rv, pv, LogVignetting(rv), LogVignetting(tv), LogVignetting(pv))
            p = joinpath(dir, "v.json")
            save(p, v)
            b = load(AbstractVignetting, p)
            @test typeof(b) == typeof(v) && control_points(b) == control_points(v) && knot_vectors(b) == knot_vectors(v)
            us = (SVector(0.0, 0.0), SVector(100.5, 200.25), SVector(639.0, 479.0))
            @test [b(u) for u in us] == [v(u) for u in us]
            save(joinpath(dir, "v2.json"), b)                                       # what is read writes back byte for byte
            @test read(joinpath(dir, "v2.json")) == read(p)
        end
        d = JSON.parse(read(joinpath(dir, "v.json"), String))
        @test d["schema"] == "vgio/vignetting/v2.0" && d["units"]["pixel"] == "colmap"
        # a radial profile states its chart, ALWAYS: a declaration has no default
        for (v, body, chart) in ((pv, x -> x, "projective"), (rv, x -> x, "linear"), (LogVignetting(rv), x -> x["spline"], "linear"))
            save(joinpath(dir, "r.json"), v)
            b = body(JSON.parse(read(joinpath(dir, "r.json"), String))["vignetting"])
            @test b["chart"] == chart && collect(keys(b)) == ["type", "degree", "center", "radius", "chart", "control_points"]
            @test load(AbstractVignetting, joinpath(dir, "r.json")) == v
        end
        save(joinpath(dir, "r.json"), rv)
        edited(f) = (t = JSON.parse(read(joinpath(dir, "r.json"), String)); f(t); q = joinpath(dir, "bad.json"); write(q, JSON.json(t)); q)
        @test_throws ArgumentError load(AbstractVignetting, edited(t -> delete!(t["vignetting"], "chart")))   # v2.0: required
        @test_throws ArgumentError load(AbstractVignetting, edited(t -> t["vignetting"]["chart"] = "polar"))
        L = with_unit(EnvironmentLight(SphericalHarmonicMap([w(i, j) for i in 1:3, j in 1:9], 2), SVector(0.1, -0.2, 0.3), 1.5), m)
        p = joinpath(dir, "light.json")
        save(p, L)
        b = load(EnvironmentLight, p)
        @test coefficients(b) == coefficients(L) && center(b) == center(L) && radius(b) == radius(L) && max_degree(b) == max_degree(L)
        dl = JSON.parse(read(p, String))
        @test dl["schema"] == "vgio/environment_light/v2.0" && dl["units"]["radiance"] == "relative" && dl["units"]["length"] == "mm"
        @test dl["map"]["basis"] == "real_orthonormal_no_condon_shortley" && dl["map"]["max_degree"] == 2 && !haskey(dl["map"], "l_max")
        @test dl["radius"] == 1500.0                                                        # in units.length
        save(joinpath(dir, "light_m.json"), L; length_unit = m)
        @test center(load(EnvironmentLight, joinpath(dir, "light_m.json"))) == center(L)
        @test_throws ArgumentError save(joinpath(dir, "plain.json"), FlexUnits.ustrip(m, L))     # a plain light: lengths without units
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

@testitem "radiometry kinds: a vignetting file in any registered pixel frame reads to the same field" setup = [VGIoFixtures] begin
    using VisualGeometryToolkitCore: AbstractVignetting, TensorProductVignetting, RadialVignetting, LogVignetting, with_control_points,
        control_points, knot_vectors, Rect
    using VisualGeometryToolkitIO: VgioStyle
    w(i, j) = 0.5 + 0.0371 * i + 0.0193 * j * j
    # positions exact in every frame (the frames differ by 0.5 px)
    tv = with_control_points(TensorProductVignetting((5, 4), Rect(0.0, 0.0, 640.0, 480.0); degree = (2, 3)), [w(i, j) for i in 1:5, j in 1:4])
    rv = with_control_points(RadialVignetting(7, SVector(320.25, 241.75), 401.25), [w(i, 1) for i in 1:7])
    mktempdir() do dir
        for v in (tv, rv, LogVignetting(rv)), (frame, shift) in ((:opencv, -0.5), (:julia, 0.5))
            p = joinpath(dir, "v.json")
            save(p, v)                                      # the writer's frame: colmap
            t = JSON.parse(read(p, String))
            t["units"]["pixel"] = String(frame)
            t["vignetting"] = JSON.parse(JSON.json(JSON.lower(VgioStyle(; pixel_frame = PixelFrame{frame}), v)))
            q = joinpath(dir, "v_$(frame).json")
            write(q, JSON.json(t))
            @test VisualGeometryToolkitIO.validate(q) == "vignetting"
            b = load(AbstractVignetting, q)
            @test typeof(b) == typeof(v) && control_points(b) == control_points(v)
            @test all(isapprox.(knot_vectors(b), knot_vectors(v); atol = 1.0e-12))
            us = (SVector(0.0, 0.0), SVector(100.5, 200.25), SVector(639.0, 479.0))
            @test all(isapprox(b(u), v(u); atol = 1.0e-12) for u in us)
            # the numbers ARE the frame's: the radial centre moved by the frames' offset
            if v isa RadialVignetting
                @test t["vignetting"]["center"] == [320.25 + shift, 241.75 + shift] && b == v
            end
        end
        # an unregistered pixel frame is refused
        save(joinpath(dir, "v.json"), rv)
        t = JSON.parse(read(joinpath(dir, "v.json"), String)); t["units"]["pixel"] = "lower_left"
        write(joinpath(dir, "bad.json"), JSON.json(t))
        @test_throws ArgumentError load(AbstractVignetting, joinpath(dir, "bad.json"))
    end
end

@testitem "radiometry kinds: a blur names its plane — a sensor's in px², a print's in length², both exact" setup = [VGIoFixtures] begin
    using StaticArrays: @SMatrix, SDiagonal
    using VisualGeometryToolkitCore: GaussianBlur, has_units, with_unit, μm, CovMat, relabel, TargetPlaneFrame, ScaledIdentity
    using Statistics: cov
    on(F, Σ) = GaussianBlur(relabel(F, CovMat(Σ)))
    print_blur = with_unit(on(TargetPlaneFrame, SDiagonal(7.84e-10, 1.8225e-10)), m)          # Σ in m², decimal (σ = 0.028, 0.0135 mm)
    print_full = with_unit(on(TargetPlaneFrame, @SMatrix([7.84e-10 1.0e-11; 1.0e-11 1.8225e-10])), m)
    print_iso = with_unit(on(TargetPlaneFrame, ScaledIdentity{2}(4.0e-10)), m)
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
        @test t["print"]["frame"] == "target_plane" && t["print"]["covariance"]["diagonal"] == [7.84e-4, 1.8225e-4]  # m² → mm², a shift
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
        @test_throws ArgumentError save(q, Dict(:t => on(TargetPlaneFrame, SDiagonal(0.25, 0.5))))           # plain on a print
        @test_throws ArgumentError save(q, Dict(:t => on(TargetPlaneFrame, SDiagonal(1.0m, 2.0m))))          # a length, not an area
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
        @test refused(b -> b["frame"] = "target_plane")                                      # two planes
        @test refused(b -> b["pixel_frame"] = "lower_left")                                  # not a registered pixel frame
        @test refused(b -> (delete!(b, "pixel_frame"); b["frame"] = "target:a1f2"))           # not a blur's plane
        @test refused(b -> (delete!(b, "pixel_frame"); b["frame"] = "image:"))
        @test refused(b -> b["unit"] = "px^2")                                               # IO 0.9's per-blur unit
    end
end

@testitem "radiometry kinds: the bodies nest in another record (VgioStyle), an IO 0.9 blur body too" setup = [VGIoFixtures] begin
    using StaticArrays: @SMatrix, SDiagonal
    using VisualGeometryToolkitCore: AbstractVignetting, RadialVignetting, LogVignetting, with_control_points, control_points,
        EnvironmentLight, SphericalHarmonicMap, coefficients, center, radius, with_unit, GaussianBlur, has_units, CovMat, relabel,
        TargetPlaneFrame
    using Statistics: cov
    using VisualGeometryToolkitIO: VgioStyle, read_length
    import FlexUnits
    s = VgioStyle(; length_unit = mm)
    roundtrip(T, x; kw...) = JSON.lift(s, T, JSON.parse(JSON.json((body = x,); style = s)).body; kw...)
    v = LogVignetting(with_control_points(RadialVignetting(6, SVector(320.0, 240.0), 400.0), [0.5 + 0.01i for i in 1:6]))
    @test control_points(roundtrip(AbstractVignetting, v)) == control_points(v)
    L = with_unit(EnvironmentLight(SphericalHarmonicMap(ones(3, 4), 1), SVector(0.1, -0.2, 0.3), 1.5), m)
    b = roundtrip(EnvironmentLight, L)
    @test center(b) == center(L) && radius(b) == radius(L) && coefficients(b) == coefficients(L)
    @test JSON.lower(s, L).radius == 1500.0                                       # in the style's unit
    g = GaussianBlur(relabel(PixelFrame{:colmap}, CovMat(@SMatrix [0.5 0.1; 0.1 0.4])))
    @test cov(roundtrip(GaussianBlur, g)) == cov(g) && keys(JSON.lower(s, g)) == (:pixel_frame, :covariance) && JSON.lower(s, g).pixel_frame == "colmap"
    p = with_unit(GaussianBlur(relabel(TargetPlaneFrame, CovMat(SDiagonal(4.0e-10, 9.0e-10)))), m)      # a print's blur, m² (σ = 0.02, 0.03 mm)
    q = roundtrip(GaussianBlur, p)
    @test has_units(q) && typeof(q) == typeof(p) && Tuple(cov(q)) == Tuple(cov(p))
    @test JSON.lower(s, p).frame == "target_plane" && JSON.lower(s, p).covariance["diagonal"] ≈ [0.02^2, 0.03^2]   # mm² in an mm record
    # an IO 0.9 blur body (`{"unit", "covariance"}`), as another package's record of an earlier version embeds it: read as
    # the unlabeled blur it held; its unit "px^2" or a length squared, told apart by NAME (FlexUnits gives px, px³ and
    # pd² one dimension, none, and one scale, so they would pass a dimension check); a length is not an area
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

@testitem "radiometry kinds: the validator" setup = [VGIoFixtures] begin
    using VisualGeometryToolkitCore: RadialVignetting, AbstractVignetting, EnvironmentLight, SphericalHarmonicMap, with_unit, GaussianBlur,
        CovMat, relabel
    mktempdir() do dir
        edited(src, f) = (t = JSON.parse(read(src, String); dicttype = Dict{String, Any}); f(t); q = joinpath(dir, "bad.json"); write(q, JSON.json(t)); q)
        p = joinpath(dir, "v.json")
        save(p, RadialVignetting(5, SVector(10.0, 20.0), 30.0))
        @test_throws ArgumentError load(AbstractVignetting, edited(p, t -> t["vignetting"]["type"] = "polynomial"))
        # a retired tag: the registry's message, naming its successor
        err = try load(AbstractVignetting, edited(p, t -> t["vignetting"]["type"] = "tensor")); nothing catch e; e end
        @test err isa ArgumentError && occursin("tensor_product", err.msg) && occursin("renamed", err.msg)
        @test_throws ArgumentError load(AbstractVignetting, edited(p, t -> t["pixel_frame"] = "opencv"))     # IO 0.9's declaration: unknown key
        @test_throws ArgumentError load(AbstractVignetting, edited(p, t -> delete!(t["units"], "pixel")))    # the envelope's
        l = joinpath(dir, "l.json")
        save(l, with_unit(EnvironmentLight(SphericalHarmonicMap(ones(3, 4), 1), SVector(0.0, 0.0, 0.0), 1.0), m))
        @test_throws ArgumentError load(EnvironmentLight, edited(l, t -> t["map"]["basis"] = "complex_condon_shortley"))
        err = try load(EnvironmentLight, edited(l, t -> t["map"]["coefficients"][1] = [1.0, 2.0])); nothing catch e; e end   # (max_degree+1)² per channel
        @test err isa ArgumentError && occursin("max_degree", err.msg)
        @test_throws ArgumentError load(EnvironmentLight, edited(l, t -> t["map"]["l_max"] = pop!(t["map"], "max_degree")))   # IO 0.9's key
        @test_throws ArgumentError load(EnvironmentLight, edited(l, t -> t["units"]["radiance"] = "W/m^2/sr"))
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
