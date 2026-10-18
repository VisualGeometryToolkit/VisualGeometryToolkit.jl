# NormalizedImageFrame{id} and its map: the image centred and divided by the
# half-diagonal, minted by the FramedAffine factory; covariances, scales,
# intrinsics and models carried through it.

@testitem "normalized_image_frame" begin
    import VisualGeometryToolkitCore as VgtkCore
    using Test
    using VisualGeometryToolkitCore
    using VisualGeometryToolkitCore: FramedAffine, frame_dependence, GeometryDefined, similarity_scale, calibration_stage, DistortedFrame
    using StaticArrays, LinearAlgebra, Random
    using Statistics: cov

    sz = Size2(width = 1920, height = 1200)
    s = hypot(1920, 1200) / 2

    @testset "the map: centre to the origin, corners to unit radius" begin
        m = FramedAffine(ImageFrame{:v}, NormalizedImageFrame{:v}, sz)
        @test m isa FramedAffine{ImageFrame{:v}, NormalizedImageFrame{:v}, Float64}
        @test m(SVector(960.0, 600.0)) == SVector(0.0, 0.0)
        for corner in (SVector(0.0, 0.0), SVector(1920.0, 0.0), SVector(0.0, 1200.0), SVector(1920.0, 1200.0))
            @test norm(m(corner)) ≈ 1 atol = 1.0e-15
        end
        @test m.A == SMatrix{2, 2}(1 / s, 0.0, 0.0, 1 / s)
        x = SVector(123.25, 987.5)
        @test inv(m)(m(x)) ≈ x rtol = 1.0e-15
        @test inv(m) isa FramedAffine{NormalizedImageFrame{:v}, ImageFrame{:v}, Float64}
        # pixel counts with units are the same map
        @test FramedAffine(ImageFrame{:v}, NormalizedImageFrame{:v}, Size2(width = 1920px, height = 1200px)).t == m.t
        # the view ids must agree; the frame is geometry-defined
        @test_throws MethodError FramedAffine(ImageFrame{:a}, NormalizedImageFrame{:b}, sz)
        @test frame_dependence(NormalizedImageFrame{:v}) === GeometryDefined()
    end

    @testset "every pixel convention lands on the same normalized point" begin
        m = FramedAffine(ImageFrame{:v}, NormalizedImageFrame{:v}, sz)
        x = SVector(700.5, 311.25)                       # a :colmap point
        for O in (:colmap, :opencv, :julia)
            mO = FramedAffine(PixelFrame{O}, NormalizedImageFrame{:v}, sz)
            @test mO(EuclideanMap(PixelFrame{:colmap}, PixelFrame{O})(x)) ≈ m(x) atol = 1.0e-15
        end
        # the first pixel's centre, written in each convention
        @test FramedAffine(PixelFrame{:julia}, NormalizedImageFrame{:v}, sz)(SVector(1.0, 1.0)) ≈
            FramedAffine(PixelFrame{:opencv}, NormalizedImageFrame{:v}, sz)(SVector(0.0, 0.0)) atol = 1.0e-15
    end

    @testset "propagate through the map: exact, checked, labeled" begin
        m = FramedAffine(ImageFrame{:v}, NormalizedImageFrame{:v}, sz)
        Σ = relabel(ImageFrame{:v}, CovMat{2, Float64}(4.0, 1.0, 1.0, 9.0))
        Σn = propagate(m, Σ)
        @test Σn isa CovMat{2, Float64, 4, NormalizedImageFrame{:v}}
        @test SMatrix{2, 2}(Σn) ≈ SMatrix{2, 2}(Σ) / s^2 rtol = 1.0e-14          # A = I/s
        @test SMatrix{2, 2}(propagate(inv(m), Σn)) ≈ SMatrix{2, 2}(Σ) rtol = 1.0e-14
        # precision pulls back: the law of inv(Σ) is the inverse of the law of Σ
        Λn = propagate(m, inv(Σ))
        @test Λn isa PrecMat{2, Float64, 4, NormalizedImageFrame{:v}}
        @test SMatrix{2, 2}(Λn) ≈ SMatrix{2, 2}(inv(Σn)) rtol = 1.0e-12
        # a labeled mismatch is refused; an unlabeled form is the wildcard
        @test_throws MethodError propagate(m, relabel(ImageFrame{:w}, CovMat{2, Float64}(4.0, 1.0, 1.0, 9.0)))
        @test propagate(m, CovMat{2, Float64}(4.0, 1.0, 1.0, 9.0)) isa CovMat{2, Float64, 4, NormalizedImageFrame{:v}}
        # an Uncertain moves its value and its form together
        x = SVector(400.0, 800.0)
        u = propagate(m, Uncertain(x, Σ))
        @test geom(u) ≈ m(x)
        @test SMatrix{2, 2}(cov(u)) ≈ SMatrix{2, 2}(Σn) rtol = 1.0e-14
        # a general (sheared) framed affine takes the same law, J = A
        g = FramedAffine{ImageFrame{:v}, ImageFrame{:v}, Float64}(SMatrix{2, 2}(1.3, 0.2, -0.4, 0.8), SVector(3.0, -2.0))
        @test SMatrix{2, 2}(propagate(g, Σ)) ≈ g.A * SMatrix{2, 2}(Σ) * g.A' rtol = 1.0e-14
        # Monte Carlo: map pixel samples, the sample covariance meets the law
        rng = MersenneTwister(3)
        Lc = cholesky(Symmetric(Matrix(SMatrix{2, 2}(Σ)))).L
        ys = reduce(hcat, [g(x + SVector{2}(Lc * randn(rng, 2))) for _ in 1:40_000])
        @test cov(ys; dims = 2) ≈ SMatrix{2, 2}(propagate(g, Σ)) rtol = 0.03
    end

    @testset "similarity_scale: the map's length ratio, a similarity only" begin
        m = FramedAffine(ImageFrame{:v}, NormalizedImageFrame{:v}, sz)
        @test similarity_scale(m) ≈ 1 / s rtol = 1.0e-15
        @test similarity_scale(inv(m)) ≈ s rtol = 1.0e-15
        # a pixel threshold: two points τ px apart land τ·σ apart
        τ = 2.5; x = SVector(300.0, 700.0); d = SVector(0.6, 0.8)
        @test norm(m(x + τ * d) - m(x)) ≈ τ * similarity_scale(m) rtol = 1.0e-13
        # a rotated similarity reports its scale; shear or anisotropy has none
        θ = 0.3; R = SMatrix{2, 2}(cos(θ), sin(θ), -sin(θ), cos(θ))
        @test similarity_scale(FramedAffine{ImageFrame{:v}, ImageFrame{:v}, Float64}(2.5 * R, SVector(1.0, 2.0))) ≈ 2.5 rtol = 1.0e-14
        @test_throws ArgumentError similarity_scale(FramedAffine{ImageFrame{:v}, ImageFrame{:v}, Float64}(SMatrix{2, 2}(1.0, 0.0, 0.1, 1.0), SVector(0.0, 0.0)))
        @test_throws ArgumentError similarity_scale(FramedAffine{ImageFrame{:v}, ImageFrame{:v}, Float64}(SMatrix{2, 2}(2.0, 0.0, 0.0, 1.0), SVector(0.0, 0.0)))
    end

    @testset "K through the map: affines fuse, K de-normalizes, the lens is untouched" begin
        m = FramedAffine(ImageFrame{:v}, NormalizedImageFrame{:v}, sz)
        Kpx = CameraCalibrationMat(SMatrix{3, 3}(1400.0, 0.0, 0.0, 0.5, 1398.0, 0.0, 955.0, 610.0, 1.0))
        cal = calibration_stage(LogicalIntrinsics(Kpx); id = :v)          # DistortedFrame{:v} → ImageFrame{:v}
        cn = m ∘ cal
        @test cn isa FramedAffine{DistortedFrame{:v}, NormalizedImageFrame{:v}}   # one fused affine, not a chain
        x = SVector(0.1, -0.2)
        @test cn(x) ≈ m(cal(x)) rtol = 1.0e-15
        Kn = CameraCalibrationMat(cn)
        @test Kn[1, 1] ≈ 1400 / s rtol = 1.0e-15
        @test Kn[1, 3] ≈ (955 - 960) / s rtol = 1.0e-12
        # a normalized K's stage lands on the normalized frame (the default lands on
        # ImageFrame, and the junction refuses it)
        stn = calibration_stage(DistortedFrame{:v}, NormalizedImageFrame{:v}, LogicalIntrinsics(Kn))
        @test_throws MethodError inv(m) ∘ calibration_stage(LogicalIntrinsics(Kn); id = :v)
        back = inv(m) ∘ stn
        @test back isa FramedAffine{DistortedFrame{:v}, ImageFrame{:v}}
        @test SMatrix{3, 3}(CameraCalibrationMat(back)) ≈ SMatrix{3, 3}(Kpx) rtol = 1.0e-14
        # the junction is still checked: another view's normalization refuses
        @test_throws MethodError FramedAffine(ImageFrame{:w}, NormalizedImageFrame{:w}, sz) ∘ cal
        # a rotated stage is no pinhole K
        θ = 0.3; Rθ = SMatrix{2, 2}(cos(θ), sin(θ), -sin(θ), cos(θ))
        @test_throws ArgumentError CameraCalibrationMat(FramedAffine{DistortedFrame{:v}, ImageFrame{:v}, Float64}(1000 * Rθ, SVector(0.0, 0.0)))
        # the lens is untouched: its coefficients live in K⁻¹·pixel, which the
        # normalization leaves where it was — the normalized camera, de-normalized,
        # IS the pixel camera
        for lens in (Division(-0.2), Lens(Perspective(), Additive(Radial((-0.1, 0.02)), Tangential(0.001, -0.0005))))
            cpx = CameraModel(LogicalIntrinsics(Kpx); lens)
            cn_ = CameraModel(LogicalIntrinsics(Kn); lens)
            for X in (SVector(0.3, -0.2, 2.0), SVector(-0.5, 0.4, 1.5), SVector(0.0, 0.0, 3.0))
                @test inv(m)(project(cn_, X)) ≈ project(cpx, X) rtol = 1.0e-12
            end
        end
    end

    @testset "models through the map: composition (one law owner)" begin
        m = FramedAffine(ImageFrame{:v}, NormalizedImageFrame{:v}, sz)
        Mm = HomographyMap(m)                                   # the framed affine as a framed collineation
        _apply_h(H, X) = (PerspectiveMap() ∘ HomographyMap(H))(X)
        H = HomographyMat{Float64}(Tuple(SMatrix{3, 3}(900.0, 20.0, 1.0e-3, -15.0, 880.0, 2.0e-3, 950.0, 610.0, 1.0)))
        Hn = HomographyMat(to_homogeneous(Mm ∘ HomographyMap(H)))
        @test Hn isa HomographyMat
        Xs = (SVector(0.3, -0.2), SVector(-0.4, 0.25), SVector(0.0, 0.1))
        for X in Xs
            @test _apply_h(Hn, X) ≈ m(_apply_h(H, X)) rtol = 1.0e-13        # the map after the model
        end
        @test SMatrix{3, 3}(to_homogeneous(inv(Mm) ∘ HomographyMap(Hn))) ≈ SMatrix{3, 3}(H) rtol = 1.0e-14
        # a radial camera: its centre moves by the map, every pixel stays on its radial line
        P = RadialCamera(H, Point2(955.0, 605.0))
        Pn = Mm ∘ P
        @test Pn isa RadialCamera
        @test SVector(Pn.center...) ≈ m(SVector(P.center...)) rtol = 1.0e-15
        for X in Xs
            g, Gx, _ = VgtkCore.constraint_with_jacobians(Pn, X, m(_apply_h(H, X)))
            @test abs(g) / norm(Gx) < 1.0e-12
        end
        Pb = inv(Mm) ∘ Pn
        @test SVector(Pb.center...) ≈ SVector(P.center...) rtol = 1.0e-14
        @test abs(abs(dot(vec(Pb.P), vec(P.P))) / (norm(Pb.P) * norm(P.P)) - 1) < 1.0e-14   # up to scale
    end
end
