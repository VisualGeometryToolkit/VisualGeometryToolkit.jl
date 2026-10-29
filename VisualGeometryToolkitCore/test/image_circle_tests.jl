# valid_domain and image_circle (cameras/image_circle.jl): where the lens MODEL
# is valid (view_angle_limit through the radial projection and K; the whole
# plane when the lens has no limit of its own — view_angle_limit reports only a
# chart's or the sphere's edge, a cap), and where light falls, from a datasheet
# diameter through the sensor's pitch. The same Ellipse primitive either way,
# never the same region by construction.
@testitem "valid_domain and image_circle" begin
    using Test
    using VisualGeometryToolkitCore
    using VisualGeometryToolkitCore: view_angle_limit, projected_radius, Pinhole, KannalaBrandt, Division,
        RegionIntersection, WholePlane, AbstractLensModel, boundary_point
    using FlexUnits: DimensionError
    using StaticArrays

    mk(K, l) = CameraModel(CameraCalibrationMat(K); lens = l)
    mk(fx, fy, cx, cy, l) = mk(@SMatrix([fx 0.0 cx; 0.0 fy cy; 0.0 0.0 1.0]), l)
    imx250 = CMOS_SENSORS["Sony"]["IMX250"]              # 2448 × 2048 @ 3.45 μm
    d780 = CMOS_SENSORS["Nikon"]["D780"]                 # 6048 × 4024 full frame
    # a 180° fisheye: r = θ(1 + kθ²) folds at θ = 1/√(−3k) = 90°, where r = (2/3)θ = π/3
    k180 = -4 / (3π^2)
    fold = KannalaBrandt(k180)

    # user-defined lenses: only `project` on rays — the generic scan finds their limit
    struct FoldingLens <: AbstractLensModel end           # r = θ(1 − 0.1θ²), folds at θ = √(10/3)
    struct MonotoneLens <: AbstractLensModel end          # r = θ, monotone to the sphere's edge
    radial_lens(X, r) = (θ = atan(hypot(X[1], X[2]), X[3]); ψ = atan(X[2], X[1]); ρ = r(θ); SVector(ρ * cos(ψ), ρ * sin(ψ)))
    VisualGeometryToolkitCore.project(::FoldingLens, X::StaticVector{3}) = radial_lens(X, θ -> θ * (1 - 0.1θ^2))
    VisualGeometryToolkitCore.project(::MonotoneLens, X::StaticVector{3}) = radial_lens(X, θ -> θ)

    @testset "a fold, isotropic K: radius f·r(θ_max) about the principal point" begin
        c = mk(800.0, 800.0, 1224.0, 1024.0, fold)
        θ = view_angle_limit(lens(c))
        @test π / 2 * (1 - 2.0e-9) < θ < π / 2             # just inside the fold
        q = valid_domain(c)
        @test q isa Ellipse{Float64}
        @test q.a ≈ 800 * π / 3 rtol = 1.0e-8
        @test q.b ≈ q.a rtol = 1.0e-12
        @test centroid(q) ≈ Point2(1224.0, 1024.0) atol = 1.0e-9
        @test Point2(1224.0 + q.a - 0.5, 1024.0) ∈ q
        @test Point2(1224.0 + q.a + 0.5, 1024.0) ∉ q
        @test Point2(1224.0, 1024.0 - q.a + 0.5) ∈ q
        @test valid_domain(BoundedCamera(c, imx250)) == q   # through the wrappers
        @test valid_domain(PosedCamera(c, EuclideanMap(one(SMatrix{3, 3, Float64}), SVector(0.0, 0.0, 1.0)))) == q
        # the image circle is never predicted from the lens: no one-argument form
        @test !hasmethod(image_circle, Tuple{CameraModel})
        @test !hasmethod(image_circle, Tuple{BoundedCamera})
        @test !hasmethod(imaged_region, Tuple{BoundedCamera})
    end

    @testset "anisotropic K: the ellipse with semi-axes fx·ρ, fy·ρ" begin
        l = KannalaBrandt(-0.1)                          # folds at θ = √(1/0.3)
        ρ = projected_radius(l, view_angle_limit(l))
        q = valid_domain(mk(900.0, 600.0, 640.0, 480.0, l))
        @test q.a ≈ 900ρ rtol = 1.0e-8
        @test q.b ≈ 600ρ rtol = 1.0e-8
        @test mod(q.θ, π) ≈ 0 atol = 1.0e-9
        @test centroid(q) ≈ Point2(640.0, 480.0) atol = 1.0e-9
    end

    @testset "skewed K: the conic is the image of the disc through K" begin
        K = @SMatrix [900.0 50.0 640.0; 0.0 600.0 480.0; 0.0 0.0 1.0]
        l = KannalaBrandt(-0.1)
        ρ = projected_radius(l, view_angle_limit(l)) * (1 - 1.0e-9)
        q = valid_domain(mk(K, l))
        Ki = inv(K)
        rs = [(x = Ki * SVector(boundary_point(q, t)..., 1.0); hypot(x[1], x[2])) for t in range(0, 2π; length = 50)]
        @test all(r -> isapprox(r, ρ; rtol = 1.0e-12), rs)
        @test centroid(q) ≈ Point2(640.0, 480.0) atol = 1.0e-9
    end

    @testset "no limit of its own: the whole plane" begin
        for l in (Pinhole(), Lens(Equidistant()), Lens(Perspective(), Radial((0.05,))), Division(-0.3), KannalaBrandt(0.01))
            @test valid_domain(mk(800.0, 800.0, 640.0, 480.0, l)) === WholePlane()
        end
        @test valid_domain(mk(800.0, 800.0, 640.0, 480.0, MonotoneLens())) === WholePlane()
        # a limit of its own (a fold: perspective, a division lift, Kannala–Brandt, a user lens): a conic
        for l in (Lens(Perspective(), Radial((-0.2,))), Division(-0.3, 0.05), KannalaBrandt(-0.1), FoldingLens())
            @test valid_domain(mk(800.0, 800.0, 640.0, 480.0, l)) isa Ellipse
        end
        # the double sphere's fold is its own limit even where it reaches past
        # the sphere's edge π − 10⁻³: ξ = 1, α = ½ images all but the backward
        # axis, its rim the disc of normalized radius 2
        ds = Lens(DoubleSphere(1.0, 0.5))
        @test π - 1.0e-3 < view_angle_limit(ds) < π
        qd = valid_domain(mk(800.0, 800.0, 640.0, 480.0, ds))
        @test qd isa Ellipse
        @test qd.a ≈ 800 * 2 rtol = 1.0e-5
        qu = valid_domain(mk(800.0, 800.0, 640.0, 480.0, FoldingLens()))
        @test qu.a ≈ 800 * sqrt(10 / 3) * (2 / 3) rtol = 1.0e-3   # the scan's fold, to its grid
        # the whole plane restricts nothing
        r = Rect(0, 0, 1280, 960)
        @test (r ∩ valid_domain(mk(800.0, 800.0, 640.0, 480.0, Pinhole()))) === r
        @test r ∈ valid_domain(mk(800.0, 800.0, 640.0, 480.0, Pinhole()))
    end

    @testset "not radially symmetric: strictly inscribed" begin
        rim_min(c, q, n) = (l = lens(c); θ = view_angle_limit(l);
            minimum(ψ -> hypot((project(c, SVector(sin(θ) * cos(ψ), sin(θ) * sin(ψ), cos(θ))) .- centroid(q))...), range(0, 2π; length = n)))
        for l in (
                Lens(Equidistant(), Chain(Tilt(0.05, 0.02), Radial((-0.05,)))),          # the review's case
                Lens(Perspective(), Additive(Radial((-0.3, 0.04)), Tangential(1.0e-3, -5.0e-4))),
            )
            c = mk(800.0, 800.0, 640.0, 480.0, l)
            q = valid_domain(c)
            @test q.a ≈ q.b rtol = 1.0e-12
            m = rim_min(c, q, 20001)
            @test m >= q.a                                  # every point inside is imaged
            @test m ≈ q.a rtol = 1.0e-7
        end
        # the review's pin: q.a below 1286.438031 px, the rim's minimum over 20001 azimuths
        qt = valid_domain(mk(800.0, 800.0, 640.0, 480.0, Lens(Equidistant(), Chain(Tilt(0.05, 0.02), Radial((-0.05,))))))
        @test qt.a < 1286.438031
        @test qt.a ≈ 1286.438031 rtol = 1.0e-7
    end

    @testset "the datasheet route, through the pitch" begin
        c = mk(800.0, 800.0, 1224.0, 1024.0, fold)
        q = image_circle(c, imx250, 17.3mm)
        @test q.a ≈ 2507.246376811594 rtol = 1.0e-12   # 17.3 mm / (2 · 3.45 μm)
        @test q.b == q.a
        @test centroid(q) == Point2(1224.0, 1024.0)
        @test image_circle(BoundedCamera(c, imx250), 17.3mm) == q
        @test image_circle(c, imx250, 1.73cm).a ≈ q.a
        # an anisotropic pitch: an ellipse
        s = Sensor(Size2(width = 2000px, height = 1500px), Size2(width = 3.0μm / px, height = 4.0μm / px))
        qa = image_circle(c, s, 6.0mm)
        @test qa.a ≈ 1000.0 && qa.b ≈ 750.0
        # the valid domain matches the datasheet circle only where the lens was built so
        f = 2507.246376811594 / (π / 3)
        qf = valid_domain(mk(f, f, 1224.0, 1024.0, fold))
        @test qf.a ≈ q.a rtol = 1.0e-8
        @test qf.b ≈ q.b rtol = 1.0e-8
        @test centroid(qf) ≈ centroid(q) atol = 1.0e-9
        # an undistorted equidistant lens: valid on the whole plane, its circle from the datasheet
        @test valid_domain(mk(800.0, 800.0, 1224.0, 1024.0, Lens(Equidistant()))) === WholePlane()
        @test image_circle(mk(800.0, 800.0, 1224.0, 1024.0, Lens(Equidistant())), imx250, 17.3mm) == q
        # refusals: a non-length diameter, a camera with no pitch
        @test_throws DimensionError image_circle(c, imx250, 17.3px)
        @test_throws DimensionError image_circle(c, imx250, 17.3dpi)
        @test_throws ArgumentError image_circle(BoundedCamera(c, Size2(2448, 2048)), 17.3mm)
    end

    @testset "is the model valid over the whole sensor? Rect(extent) ∈ valid_domain" begin
        kb = BoundedCamera(mk(800.0, 800.0, 640.0, 480.0, KannalaBrandt(-0.1)), Size2(1280, 960))
        q = valid_domain(kb)
        @test q.a > hypot(640, 480)                      # the fold lies beyond the sensor
        @test Rect(extent(kb)) ∈ q                       # a conic, covering: valid everywhere on the sensor
        fish = BoundedCamera(mk(800.0, 800.0, 3024.0, 2012.0, fold), d780)
        qf = valid_domain(fish)
        @test qf.a ≈ 800π / 3 rtol = 1.0e-8               # 838 px, inside the 2012 px half-height
        @test Rect(extent(fish)) ∉ qf                    # a circular fisheye on full frame: folds inside the extent
        domain = Rect(extent(fish)) ∩ qf
        @test domain isa RegionIntersection
        @test Point2(3024.0, 2012.0) ∈ domain && Point2(10.0, 10.0) ∉ domain
        @test Rect(extent(fish)) ∉ domain && Rect(3000, 2000, 10, 10) ∈ domain
    end
end
