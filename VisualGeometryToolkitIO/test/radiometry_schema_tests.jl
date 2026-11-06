# The radiometric kinds (radiometry_schema.jl): vgio/vignetting/v1 (VGC's tensor-product, radial and log fields, rebuilt
# bit for bit), vgio/environment_light/v1 (an EnvironmentLight of a SphericalHarmonicMap; a plain light refused; the basis
# convention pinned), vgio/blur/v1 (named GaussianBlurs in the storage they hold), and the validator's refusals.
@testitem "radiometry kinds: vignetting, environment light, blurs round trip" setup = [VGIoFixtures] begin
    using StaticArrays: @SMatrix
    using VisualGeometryToolkitCore: AbstractVignetting, TensorProductVignetting, RadialVignetting, LogVignetting, with_control_points,
        control_points, knot_vectors, EnvironmentLight, SphericalHarmonicMap, coefficients, center, radius, with_unit,
        GaussianBlur, Rect, has_units
    using Statistics: cov
    import FlexUnits
    w(i, j) = 0.5 + 0.0371 * i + 0.0193 * j * j                       # fixed control points, not ones (every entry distinct)
    mktempdir() do dir
        tv = with_control_points(TensorProductVignetting((5, 4), Rect(0.0, 0.0, 640.0, 480.0); degree = (2, 3)), [w(i, j) for i in 1:5, j in 1:4])
        rv = with_control_points(RadialVignetting(7, SVector(320.3, 241.7), 401.25), [w(i, 1) for i in 1:7])
        for v in (tv, rv, LogVignetting(rv), LogVignetting(tv))
            p = joinpath(dir, "v.json")
            save(p, v)
            b = load(AbstractVignetting, p)
            @test typeof(b) == typeof(v) && control_points(b) == control_points(v) && knot_vectors(b) == knot_vectors(v)
            us = (SVector(0.0, 0.0), SVector(100.5, 200.25), SVector(639.0, 479.0))
            @test [b(u) for u in us] == [v(u) for u in us]
        end
        @test JSON.parse(read(joinpath(dir, "v.json"), String))["pixel_frame"] == "colmap"
        L = with_unit(EnvironmentLight(SphericalHarmonicMap([w(i, j) for i in 1:3, j in 1:9], 2), SVector(0.1, -0.2, 0.3), 1.5), m)
        p = joinpath(dir, "light.json")
        save(p, L)
        b = load(EnvironmentLight, p)
        @test coefficients(b) == coefficients(L) && center(b) == center(L) && radius(b) == radius(L)
        @test JSON.parse(read(p, String))["map"]["basis"] == "real_orthonormal_no_condon_shortley"
        @test_throws ArgumentError save(joinpath(dir, "plain.json"), FlexUnits.ustrip(m, L))     # a plain light: lengths without units
        B = Dict(:R => GaussianBlur(0.6), :G => GaussianBlur(0.5, 0.7), :B => GaussianBlur(@SMatrix [0.5 0.1; 0.1 0.4]))
        p = joinpath(dir, "blur.json")
        save(p, B)
        bb = load(Dict{Symbol, GaussianBlur}, p)
        @test all(typeof(cov(bb[k])) == typeof(cov(B[k])) && cov(bb[k]) == cov(B[k]) for k in keys(B))
        bR = JSON.parse(read(p, String))["blurs"][3]                                          # :R, sorted last: round, one number
        @test bR["unit"] == "px^2" && collect(keys(bR["covariance"])) == ["isotropic"] && bR["covariance"]["isotropic"] ≈ 0.36
    end
end

@testitem "radiometry kinds: a blur states its unit — a camera's in px², a print's in mm², both exact" setup = [VGIoFixtures] begin
    using StaticArrays: @SMatrix, SDiagonal
    using VisualGeometryToolkitCore: GaussianBlur, has_units, with_unit, μm
    using Statistics: cov
    print_blur = with_unit(GaussianBlur(SDiagonal(7.84e-10, 1.8225e-10)), m)               # Σ in m², decimal (σ = 0.028, 0.0135 mm)
    print_full = with_unit(GaussianBlur(@SMatrix([7.84e-10 1.0e-11; 1.0e-11 1.8225e-10])), m)
    camera = GaussianBlur(0.6, 0.45)
    B = Dict(:print => print_blur, :print_full => print_full, :camera => camera)
    mktempdir() do dir
        p = joinpath(dir, "blur.json")
        save(p, B)
        bb = load(Dict{Symbol, GaussianBlur}, p)
        for k in keys(B)                                                                   # exact: storage, type, every entry
            @test typeof(bb[k]) == typeof(B[k]) && has_units(bb[k]) == has_units(B[k]) && Tuple(cov(bb[k])) == Tuple(cov(B[k]))
        end
        t = Dict(b["name"] => b for b in JSON.parse(read(p, String))["blurs"])
        @test t["camera"]["unit"] == "px^2" && t["camera"]["covariance"]["diagonal"] == [0.6^2, 0.45^2]   # px² as they are
        @test t["print"]["unit"] == "mm^2" && t["print_full"]["unit"] == "mm^2" && t["print_full"]["covariance"][1][2] == 1.0e-5   # m² → mm², a shift
        save(p, B; length_unit = μm)
        t = Dict(b["name"] => b for b in JSON.parse(read(p, String))["blurs"])
        bb = load(Dict{Symbol, GaussianBlur}, p)
        @test t["print"]["unit"] == "μm^2" && t["camera"]["unit"] == "px^2" && all(k -> Tuple(cov(bb[k])) == Tuple(cov(B[k])), keys(B))
        # only "px^2" or a length squared: FlexUnits equates every dimensionless unit of scale 1, so px, px³ and pd²
        # are told apart by name; a length is not an area
        function refused(bad)
            edited = JSON.parse(read(p, String); dicttype = Dict{String, Any}); edited["blurs"][1]["unit"] = bad
            q = joinpath(dir, "bad.json"); write(q, JSON.json(edited))
            return try load(Dict{Symbol, GaussianBlur}, q); false catch e; e isa ArgumentError end
        end
        @test refused("px")
        @test refused("px^3")
        @test refused("pd^2")
        @test refused("mm")
        @test refused("mm^3")
        @test refused("mm*px")
        @test refused("furlong^2")
        # the writer: a blur in px quantities is px²; a plain blur on a target's frame holds lengths without units
        pxq = GaussianBlur(SDiagonal(0.25px^2 |> VisualGeometryToolkitCore.base_quantity, 0.5px^2 |> VisualGeometryToolkitCore.base_quantity))
        save(p, Dict(:q => pxq))
        @test JSON.parse(read(p, String))["blurs"][1]["unit"] == "px^2" && cov(load(Dict{Symbol, GaussianBlur}, p)[:q]) == SDiagonal(0.25, 0.5)
        tf = GaussianBlur(VisualGeometryToolkitCore.relabel(VisualGeometryToolkitCore.TargetPlaneFrame, cov(camera)))
        save(p, Dict(:c => GaussianBlur(VisualGeometryToolkitCore.relabel(VisualGeometryToolkitCore.PixelFrame{:colmap}, cov(camera)))))   # a pixel frame: px²
        @test JSON.parse(read(p, String))["blurs"][1]["unit"] == "px^2"
        @test_throws ArgumentError save(p, Dict(:t => tf))
        @test_throws ArgumentError save(p, Dict(:t => GaussianBlur(SDiagonal(1.0m, 2.0m))))   # a length is not a variance
    end
end

@testitem "radiometry kinds: the bodies nest in another record (VgioStyle)" setup = [VGIoFixtures] begin
    using StaticArrays: @SMatrix, SDiagonal
    using VisualGeometryToolkitCore: AbstractVignetting, RadialVignetting, LogVignetting, with_control_points, control_points,
        EnvironmentLight, SphericalHarmonicMap, coefficients, center, radius, with_unit, GaussianBlur, has_units
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
    g = GaussianBlur(@SMatrix [0.5 0.1; 0.1 0.4])
    @test cov(roundtrip(GaussianBlur, g)) == cov(g) && keys(JSON.lower(s, g)) == (:unit, :covariance) && JSON.lower(s, g).unit == "px^2"
    p = with_unit(GaussianBlur(SDiagonal(4.0e-10, 9.0e-10)), m)      # a print's blur, m² (σ = 0.02, 0.03 mm)
    q = roundtrip(GaussianBlur, p)                                   # no keyword: the block states mm²
    @test has_units(q) && typeof(q) == typeof(p) && Tuple(cov(q)) == Tuple(cov(p))
    @test JSON.lower(s, p).unit == "mm^2" && JSON.lower(s, p).covariance["diagonal"] ≈ [0.02^2, 0.03^2]   # mm² in an mm record
end

@testitem "radiometry kinds: the validator" setup = [VGIoFixtures] begin
    using VisualGeometryToolkitCore: RadialVignetting, AbstractVignetting, EnvironmentLight, SphericalHarmonicMap, with_unit, GaussianBlur
    mktempdir() do dir
        edited(src, f) = (t = JSON.parse(read(src, String); dicttype = Dict{String, Any}); f(t); q = joinpath(dir, "bad.json"); write(q, JSON.json(t)); q)
        p = joinpath(dir, "v.json")
        save(p, RadialVignetting(5, SVector(10.0, 20.0), 30.0))
        @test_throws ArgumentError load(AbstractVignetting, edited(p, t -> t["vignetting"]["type"] = "polynomial"))
        @test_throws ArgumentError load(AbstractVignetting, edited(p, t -> t["pixel_frame"] = "opencv"))     # COLMAP pixels only
        l = joinpath(dir, "l.json")
        save(l, with_unit(EnvironmentLight(SphericalHarmonicMap(ones(3, 4), 1), SVector(0.0, 0.0, 0.0), 1.0), m))
        @test_throws ArgumentError load(EnvironmentLight, edited(l, t -> t["map"]["basis"] = "complex_condon_shortley"))
        @test_throws ArgumentError load(EnvironmentLight, edited(l, t -> t["map"]["coefficients"][1] = [1.0, 2.0]))   # (l_max+1)² per channel
        b = joinpath(dir, "b.json")
        save(b, Dict(:R => GaussianBlur(0.5), :G => GaussianBlur(0.5)))
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
    end
end
