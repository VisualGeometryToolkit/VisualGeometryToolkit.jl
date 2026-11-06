# The radiometric kinds (radiometry_schema.jl): vgio/vignetting/v1 (VGC's tensor-product, radial and log fields, rebuilt
# bit for bit), vgio/environment_light/v1 (an EnvironmentLight of a SphericalHarmonicMap; a plain light refused; the basis
# convention pinned), vgio/blur/v1 (named GaussianBlurs in the storage they hold), and the validator's refusals.
@testitem "radiometry kinds: vignetting, environment light, blurs round trip" setup = [VGIoFixtures] begin
    using StaticArrays: @SMatrix
    using VisualGeometryToolkitCore: AbstractVignetting, TensorProductVignetting, RadialVignetting, LogVignetting, with_control_points,
        control_points, knot_vectors, EnvironmentLight, SphericalHarmonicMap, coefficients, center, radius, with_unit,
        GaussianBlur, moment_matrix, Rect
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
        @test all(typeof(moment_matrix(bb[k])) == typeof(moment_matrix(B[k])) && moment_matrix(bb[k]) == moment_matrix(B[k]) for k in keys(B))
        mR = JSON.parse(read(p, String))["blurs"][3]["moments"]                              # :R, sorted last: round, one number
        @test collect(keys(mR)) == ["isotropic"] && mR["isotropic"] ≈ 0.36
    end
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
