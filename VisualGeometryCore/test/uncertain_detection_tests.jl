# A detection whose frame carries its uncertainty — `Attributed(Uncertain(frame,
# Σ), metadata)`, what VisualGeometryFeatures' `asift_detect(…; uncertainty =
# FeatureCovariance())` returns — is an `AbstractDetection`, and every method on
# that alias reads its geometry through `feature_frame`, so it behaves as the
# bare detection does.
@testitem "AbstractDetection: a detection of an Uncertain frame" begin
    using Test
    using StaticArrays: SMatrix
    using GeometryBasics: Circle
    using VisualGeometryCore
    using VisualGeometryCore: Attributed, Uncertain, CovMat, PrecMat, Point2, Disc, EllipticFrame,
        AbstractDetection, FeatureFrame, feature_frame, characteristic_scale, centroid, geom, cov

    meta = (response = 0.5, laplacian_scale_score = 1.0)
    Σ3 = CovMat{3, Float64}(0.04, 0.0, 0.0, 0.0, 0.04, 0.0, 0.0, 0.0, 0.01)
    disc = Disc(Point2(10.0, 20.0), 2.0)
    S = SMatrix{2, 2}(4.0, 1.0, 1.0, 9.0)
    ell = EllipticFrame(Point2(30.0, 40.0), S)

    plain = Attributed(disc, meta)
    unc = Attributed(Uncertain(disc, Σ3), meta)
    unc_prec = Attributed(Uncertain(disc, inv(Σ3)), meta)
    unc_ell = Attributed(Uncertain(ell, CovMat{5}(σ = ntuple(_ -> 0.3, 5))), meta)

    # the alias: both kinds, either statistical form, and nothing else
    @test plain isa AbstractDetection
    @test unc isa AbstractDetection && unc_prec isa AbstractDetection && unc_ell isa AbstractDetection
    @test Uncertain(disc, Σ3) isa AbstractDetection          # an uncertain frame, no metadata: its own detection
    @test !(Attributed(Point2(1.0, 2.0), meta) isa AbstractDetection)
    @test !(Attributed(Uncertain(Point2(1.0, 2.0), CovMat{2}(σ = 1.0)), meta) isa AbstractDetection)
    @test !(Attributed(Attributed(disc, 1.0), meta) isa AbstractDetection)   # only the statistical layer unwraps

    # feature_frame: the bare frame of either kind, inferred
    @test @inferred(feature_frame(plain)) === disc
    @test @inferred(feature_frame(unc)) === disc
    @test @inferred(feature_frame(unc_prec)) === disc
    @test @inferred(feature_frame(unc_ell)) === ell

    # Circle: the frame's circle, identical for both kinds, inferred
    @test @inferred(Circle(plain, 3.0)) == @inferred(Circle(unc, 3.0)) == Circle(Point2(10.0, 20.0), 6.0)
    @test Circle(unc) == Circle(plain)
    c = @inferred Circle(unc_ell, 2.0)
    @test c.center == Point2(30.0, 40.0) && c.r ≈ 2.0 * characteristic_scale(ell)

    # property forwarding reaches the frame through the Uncertain layer; the
    # metadata and the covariance stay where they are
    @test unc.center == disc.center && unc.σ == disc.σ && unc.response == 0.5
    @test @inferred((d -> d.center)(unc)) == disc.center
    @test unc.value === Uncertain(disc, Σ3) && cov(unc.value) === Σ3
    @test centroid(unc) == centroid(plain) == disc.center
end

@testitem "AbstractDetection: Makie conversion of an uncertain detection" begin
    using Test
    using Makie
    using GeometryBasics: Circle
    using VisualGeometryCore
    using VisualGeometryCore: Attributed, Uncertain, CovMat, Point2, Disc

    meta = (response = 0.5,)
    Σ3 = CovMat{3, Float64}(0.04, 0.0, 0.0, 0.0, 0.04, 0.0, 0.0, 0.0, 0.01)
    discs = [Disc(Point2(10.0i, 20.0), 2.0) for i in 1:3]
    plain = [Attributed(d, meta) for d in discs]
    unc = [Attributed(Uncertain(d, Σ3), meta) for d in discs]

    # the blobs-only conversion: the same circles from either kind
    sp = Makie.convert_arguments(Makie.Plot{Makie.plot}, unc; extent = 3.0)
    sp0 = Makie.convert_arguments(Makie.Plot{Makie.plot}, plain; extent = 3.0)
    @test sp.args[1] == sp0.args[1] == Circle.(discs, Ref(3.0))
end
