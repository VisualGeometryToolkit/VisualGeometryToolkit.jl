# Maps are the geometric objects: a map's TYPE is its transform class, and it acts on a primitive in that class
# (transforms/ct_contravariant.jl, homography_map.jl). One item per row of the primitive table — random maps of
# the class, the result's type, and agreement with the primitive's sampled boundary points pushed through the
# map's point action — then SimilarityMap's exact algebra, the symmetric conic storage and its class checks, and
# the calibration matrix's structural storage.

@testsetup module MapClassFixtures
using StaticArrays, Rotations, LinearAlgebra, Random
using VisualGeometryToolkitCore
using VisualGeometryToolkitCore: boundary_point
export random_similarity, random_euclidean, random_affine, random_homography, on_conic, circle, ellipse, points_on

random_euclidean(rng) = EuclideanMap(RotMatrix{2}(2π * rand(rng)), SVector(10randn(rng), 10randn(rng)))
random_similarity(rng) = SimilarityMap(0.5 + 2rand(rng), RotMatrix{2}(2π * rand(rng)), SVector(10randn(rng), 10randn(rng)))
function random_affine(rng)
    A = SMatrix{2, 2}(1.0I) + 0.4 * @SMatrix(randn(rng, 2, 2))
    return CoordinateTransformations.AffineMap(A, SVector(10randn(rng), 10randn(rng)))
end
using CoordinateTransformations: CoordinateTransformations
random_homography(rng) = HomographyMap(SMatrix{3, 3}(1.0I) + SMatrix{3, 3}(0.2randn(rng), 0.2randn(rng), 1.0e-3randn(rng),
    0.2randn(rng), 0.2randn(rng), 1.0e-3randn(rng), 5randn(rng), 5randn(rng), 0.0))

circle(rng) = Circle(Point2(5randn(rng), 5randn(rng)), 1 + 5rand(rng))
ellipse(rng) = Ellipse(Point2(5randn(rng), 5randn(rng)), 3 + 4rand(rng), 1 + 2rand(rng), π * rand(rng))
points_on(shape, n = 16) = [SVector{2}(boundary_point(shape, 2π * k / n)) for k in 0:(n - 1)]

# x̃ᵀQx̃ relative to the conic's and the point's scale: 0 on the curve
on_conic(Q, x) = (xh = SVector(x[1], x[2], 1.0); abs(dot(xh, SMatrix{3, 3}(Q) * xh)) / (norm(SMatrix{3, 3}(Q)) * dot(xh, xh)) < 1.0e-11)
end

@testitem "map classes: Euclidean keeps every primitive's type" setup = [MapClassFixtures] begin
    using Random, StaticArrays, LinearAlgebra, GeometryBasics
    using VisualGeometryToolkitCore
    using VisualGeometryToolkitCore: Disc, OrientedDisc, EllipticFrame, AffineFrame, moment_matrix, Line2D
    rng = Xoshiro(1)
    for _ in 1:50
        E = random_euclidean(rng)
        c, e = circle(rng), ellipse(rng)
        c′, e′ = E(c), E(e)
        @test c′ isa Circle && e′ isa Ellipse
        @test all(x -> on_conic(HomCircleMat(c′), E(x)), points_on(c))
        @test all(x -> on_conic(HomEllipseMat(e′), E(x)), points_on(e))
        # a line: points on it stay on it (the offset moves by n′·t, R n = n′)
        l = Line2D(randn(rng), randn(rng), randn(rng))
        n = SVector(l.coeffs[1], l.coeffs[2]); p0 = -l.coeffs[3] * n / dot(n, n); d = SVector(-n[2], n[1])
        @test E(l) isa Line2D
        @test all(x -> (y = E(x); abs(dot(E(l).coeffs, SVector(y[1], y[2], 1.0))) < 1.0e-9 * norm(E(l).coeffs) * (1 + norm(y))), (p0, p0 + 3d))
        @test E(l).coeffs ≈ HomographyMap(E)(l).coeffs
        # feature frames keep their type
        p = Point2(randn(rng), randn(rng))
        @test E(Disc(p, 2.0)) isa Disc && E(OrientedDisc(p, 2.0, 0.3)) isa OrientedDisc
        @test E(EllipticFrame(p, SMatrix{2, 2}(2.0, 0.1, 0.1, 1.0))) isa EllipticFrame && E(AffineFrame(p, SMatrix{2, 2}(1.0I))) isa AffineFrame
    end
    # a plane: points on it stay on it
    using Rotations: RotXYZ
    using VisualGeometryToolkitCore: Plane3D, signed_distance
    E3 = EuclideanMap(RotXYZ(0.3, -0.2, 0.5), SVector(1.0, -2.0, 3.0))
    P = Plane3D(Point3(0.5, 0.2, 1.0), Vec3(0.2, 0.3, 1.0))
    @test abs(signed_distance(E3(P), Point3(E3(SVector(0.5, 0.2, 1.0))...))) < 1.0e-12
end

@testitem "map classes: similarity keeps every primitive's type" setup = [MapClassFixtures] begin
    using Random, StaticArrays, LinearAlgebra, GeometryBasics, Rotations
    using CoordinateTransformations: LinearMap, transform_deriv
    using VisualGeometryToolkitCore
    using VisualGeometryToolkitCore: Disc, OrientedDisc, EllipticFrame, AffineFrame, moment_matrix, Line2D, scale, centroid
    rng = Xoshiro(2)
    for _ in 1:50
        S = random_similarity(rng)
        maps = (S, LinearMap(RotMatrix{2}(2π * rand(rng))), LinearMap((0.5 + 2rand(rng)) * I))
        for M in maps
            c, e = circle(rng), ellipse(rng)
            c′, e′ = M(c), M(e)
            @test c′ isa Circle && e′ isa Ellipse
            @test all(x -> on_conic(HomCircleMat(c′), M(x)), points_on(c))
            @test all(x -> on_conic(HomEllipseMat(e′), M(x)), points_on(e))
        end
        # a line: two points on it stay on it
        l = Line2D(randn(rng), randn(rng), randn(rng))
        l′ = S(l)
        @test l′ isa Line2D
        n = SVector(l.coeffs[1], l.coeffs[2]); p0 = -l.coeffs[3] * n / dot(n, n); d = SVector(-n[2], n[1])
        for x in (p0, p0 + 3d)
            y = S(x)
            @test abs(dot(l′.coeffs, SVector(y[1], y[2], 1.0))) / norm(l′.coeffs) < 1.0e-9 * (1 + norm(y))
        end
        # feature frames: the centre mapped, the moment matrix carried as J Σ Jᵀ (J = s R)
        J = transform_deriv(S, SVector(0.0, 0.0))
        p = Point2(randn(rng), randn(rng))
        d = S(Disc(p, 2.0))
        @test d isa Disc && centroid(d) ≈ S(SVector(p...)) && scale(d) ≈ scale(S) * 2.0
        o = S(OrientedDisc(p, 2.0, 0.3))
        @test o isa OrientedDisc && scale(o) ≈ scale(S) * 2.0
        Σ = SMatrix{2, 2}(3.0, 0.5, 0.5, 1.0)
        f = S(EllipticFrame(p, Σ, Val(:moment_matrix)))
        @test f isa EllipticFrame && moment_matrix(f) ≈ J * Σ * J'
        A = SMatrix{2, 2}(2.0, 0.3, -0.4, 1.0)
        a = S(AffineFrame(p, A))
        @test a isa AffineFrame && moment_matrix(a) ≈ J * (A * A') * J'
    end
end

@testitem "map classes: affine gives an ellipse's matrix" setup = [MapClassFixtures] begin
    using Random, StaticArrays, LinearAlgebra, GeometryBasics
    using CoordinateTransformations: LinearMap
    using VisualGeometryToolkitCore
    using VisualGeometryToolkitCore: Line2D
    rng = Xoshiro(3)
    for _ in 1:50
        am = random_affine(rng)
        lm = LinearMap(am.linear)
        for M in (am, lm)
            c, e = circle(rng), ellipse(rng)
            Qc, Qe = M(c), M(e)
            @test Qc isa HomEllipseMat && Qe isa HomEllipseMat
            @test all(x -> on_conic(Qc, M(x)), points_on(c))
            @test all(x -> on_conic(Qe, M(x)), points_on(e))
        end
        @test am(Line2D(1.0, 2.0, 3.0)) isa Line2D
    end
end

@testitem "map classes: a homography gives a Conic; feature frames refuse it" setup = [MapClassFixtures] begin
    using Random, StaticArrays, LinearAlgebra, GeometryBasics
    using VisualGeometryToolkitCore
    using VisualGeometryToolkitCore: Disc, Line2D
    using CoordinateTransformations: PerspectiveMap
    rng = Xoshiro(4)
    for _ in 1:50
        H = random_homography(rng)
        PH = PerspectiveMap() ∘ H
        c, e = circle(rng), ellipse(rng)
        Qc, Qe = H(c), H(e)
        @test Qc isa Conic && Qe isa Conic
        @test all(x -> on_conic(Qc, PH(x)), points_on(c))
        @test all(x -> on_conic(Qe, PH(x)), points_on(e))
        @test H(Line2D(1.0, 2.0, 3.0)) isa Line2D
    end
    # every structured map lifted to a HomographyMap is projective in meaning
    @test HomographyMap(random_similarity(rng))(circle(rng)) isa Conic
    @test HomographyMap(random_affine(rng))(ellipse(rng)) isa Conic
    @test_throws MethodError random_homography(rng)(Disc(Point2(0.0, 0.0), 1.0))
end

@testitem "SimilarityMap: exact class under ∘ and inv, frames as EuclideanMap" setup = [MapClassFixtures] begin
    using Random, StaticArrays, LinearAlgebra, Rotations
    using CoordinateTransformations: LinearMap, Translation
    using VisualGeometryToolkitCore
    using VisualGeometryToolkitCore: scale, rotation, translation, relabel, unlabel, m
    rng = Xoshiro(5)
    for _ in 1:50
        A, B = random_similarity(rng), random_similarity(rng)
        C = A ∘ B
        @test C isa SimilarityMap{2}
        @test scale(C) == scale(A) * scale(B)
        # the parts' own products, to rounding: StaticArrays' muladds contract
        # differently with the call site (Julia 1.13 inlined differs by an ulp)
        @test rotation(C) ≈ rotation(A) * rotation(B) rtol = 4eps()
        @test translation(C) ≈ scale(A) * (rotation(A) * translation(B)) + translation(A) rtol = 4eps()
        x = SVector(randn(rng), randn(rng))
        @test C(x) ≈ A(B(x))
        Ai = inv(A)
        @test Ai isa SimilarityMap{2} && scale(Ai) == inv(scale(A)) && rotation(Ai) == rotation(A)'
        @test Ai(A(x)) ≈ x
        E = random_euclidean(rng)
        @test E ∘ A isa SimilarityMap{2} && A ∘ E isa SimilarityMap{2}
        @test (E ∘ A)(x) ≈ E(A(x)) && (A ∘ E)(x) ≈ A(E(x))
        @test to_homogeneous(C) ≈ to_homogeneous(A) * to_homogeneous(B)
    end
    E = random_euclidean(rng)
    @test scale(SimilarityMap(E)) == 1 && SimilarityMap(E)(SVector(1.0, 2.0)) == E(SVector(1.0, 2.0))
    @test SimilarityMap(LinearMap(RotMatrix{2}(0.3))) isa SimilarityMap{2}
    Sneg = SimilarityMap(LinearMap(-2.0I))
    @test scale(Sneg) == 2.0 && Sneg(SVector(1.0, 3.0)) ≈ SVector(-2.0, -6.0)
    @test SimilarityMap(Translation(SVector(1.0, 2.0)))(SVector(0.0, 0.0)) == SVector(1.0, 2.0)
    # plain numbers only; a positive scale
    @test_throws ArgumentError SimilarityMap(2.0, RotMatrix{2}(0.1), SVector(1.0, 2.0) .* m)
    @test_throws ArgumentError SimilarityMap(EuclideanMap(RotMatrix{2}(0.1), SVector(1.0, 2.0) .* m))
    @test_throws ArgumentError SimilarityMap(-1.0, RotMatrix{2}(0.1), SVector(1.0, 2.0))
    # frame tags: inv swaps, ∘ checks the junction, relabel/unlabel
    S = relabel(TargetPlaneFrame, PixelFrame{:colmap}, random_similarity(rng))
    @test inv(S) isa SimilarityMap{2, Float64, Float64, <:Any, <:Any, PixelFrame{:colmap}, TargetPlaneFrame}
    @test inv(S) ∘ S isa SimilarityMap{2, Float64, Float64, <:Any, <:Any, TargetPlaneFrame, TargetPlaneFrame}
    @test_throws MethodError S ∘ S
    @test unlabel(S) ∘ unlabel(S) isa SimilarityMap
    @test HomographyMap(S) isa HomographyMap{3, Float64, <:Any, TargetPlaneFrame, PixelFrame{:colmap}}
end

@testitem "conic storage: exactly symmetric, class-checked" setup = [MapClassFixtures] begin
    using Random, StaticArrays, LinearAlgebra, GeometryBasics
    using VisualGeometryToolkitCore
    rng = Xoshiro(6)
    # 1000 random homographies on circles, against an independently computed dense
    # H⁻ᵀ·C·H⁻¹ (symmetry by storage makes `Q == Q'` true of every stored conic, so
    # only the values can tell a congruence of the upper triangle from a wrong one)
    @test all(1:1000) do _
        H, c = random_homography(rng), circle(rng)
        Hi = inv(SMatrix{3, 3}(to_homogeneous(H)))
        C = SMatrix{3, 3}(HomCircleMat(c))
        dense = Hi' * C * Hi
        isapprox(SMatrix{3, 3}(H(c)), (dense + dense') / 2; rtol = 1.0e-10)
    end
    # a near-symmetric matrix: its off-diagonal pair averaged, the stored Q₁₂ = (Q₁₂ + Q₂₁)/2 of the input
    e = Ellipse(Point2(1.0, 2.0), 3.0, 1.0, 0.4)
    Q = SMatrix{3, 3}(HomEllipseMat(e))
    Qδ = Q + SMatrix{3, 3}(0, 1.0e-12, 0, 0, 0, 0, 0, 0, 0)   # δ in Q₂₁ only
    @test HomEllipseMat(Qδ)[1, 2] == (Qδ[1, 2] + Qδ[2, 1]) / 2 != Qδ[1, 2]
    @test HomEllipseMat(Qδ)[1, 1] == Q[1, 1]                  # the diagonal as given
    @test_throws ArgumentError HomEllipseMat(Q + SMatrix{3, 3}(0, 0.1, 0, 0, 0, 0, 0, 0, 0))
    @test_throws ArgumentError Conic(Q + SMatrix{3, 3}(0, 0.1, 0, 0, 0, 0, 0, 0, 0))
    # the six coefficients: (Q₁₁, Q₁₂, Q₂₂, Q₁₃, Q₂₃, Q₃₃)
    @test SMatrix{3, 3}(HomEllipseMat((Q[1, 1], Q[1, 2], Q[2, 2], Q[1, 3], Q[2, 3], Q[3, 3]))) == Q
    # class refusals: a hyperbola, an imaginary ellipse, an ellipse as a circle
    hyperbola = SMatrix{3, 3}(1.0, 0, 0, 0, -1.0, 0, 0, 0, -1.0)
    imaginary = SMatrix{3, 3}(1.0, 0, 0, 0, 1.0, 0, 0, 0, 1.0)
    @test_throws ArgumentError HomEllipseMat(hyperbola)
    @test_throws ArgumentError HomEllipseMat(imaginary)
    @test_throws ArgumentError HomCircleMat(HomEllipseMat(e))
    @test HomEllipseMat(hyperbola; check = false) isa HomEllipseMat   # the escape for a guaranteed class
    @test Conic(hyperbola) isa Conic && classify_conic(Conic(hyperbola)) isa VisualGeometryToolkitCore.HyperbolaTrait
    @test HomCircleMat(Circle(Point2(1.0, 2.0), 3.0)) isa HomCircleMat
    @test -HomEllipseMat(e) isa SMatrix                               # arithmetic escapes the class
    @test isbitstype(HomEllipseMat{Float64}) && sizeof(HomEllipseMat{Float64}) == 48
end

@testitem "CameraCalibrationMat: five parameters, exact structure" begin
    using StaticArrays, LinearAlgebra, ForwardDiff
    using VisualGeometryToolkitCore
    Kf = SMatrix{3, 3}(800.0, 0, 0, 0.3, 790.0, 0, 320.0, 240.0, 1)
    K = CameraCalibrationMat(Kf)
    @test isbitstype(typeof(K)) && sizeof(K) == 5 * sizeof(Float64)
    @test SMatrix(K) == Kf && (K[2, 1], K[3, 1], K[3, 2], K[3, 3]) === (0.0, 0.0, 0.0, 1.0)
    # the structure is always checked (the storage drops it); check = false skips the focals only
    @test_throws ArgumentError CameraCalibrationMat(Kf')
    @test_throws ArgumentError CameraCalibrationMat(Kf'; check = false)
    @test_throws ArgumentError CameraCalibrationMat(SMatrix{3, 3}(800.0, 0, 0, 0, 790.0, 0, 320.0, 240.0, 2.0); check = false)
    @test_throws ArgumentError CameraCalibrationMat(SMatrix{3, 3}(-800.0, 0, 0, 0, 790.0, 0, 320.0, 240.0, 1))
    @test CameraCalibrationMat(SMatrix{3, 3}(-800.0, 0, 0, 0, 790.0, 0, 320.0, 240.0, 1); check = false)[1, 1] == -800.0
    # inv: analytic, exact structural zeros, a plain SMatrix
    Ki = inv(K)
    @test Ki isa SMatrix{3, 3, Float64, 9} && (Ki[2, 1], Ki[3, 1], Ki[3, 2], Ki[3, 3]) === (0.0, 0.0, 0.0, 1.0)
    @test Kf * Ki ≈ I(3) atol = 1.0e-14
    # eltype-generic
    @test CameraCalibrationMat(Float32.(Kf)) isa CameraCalibrationMat{Float32}
    @test inv(CameraCalibrationMat(Float32.(Kf))) isa SMatrix{3, 3, Float32, 9}
    J = ForwardDiff.jacobian(p -> vec(SMatrix(inv(CameraCalibrationMat(SMatrix{3, 3}(p[1], 0, 0, 0, p[2], 0, p[3], p[4], 1))))),
        [800.0, 790.0, 320.0, 240.0])
    @test all(isfinite, J) && iszero(J[2, :])                         # K⁻¹[2, 1] is exactly 0 in every direction
end

# Review finding 1: the class was decided in the conic's own eltype, so a Float32
# conic at pixel coordinates lost γ to cancellation (51% of 1–10 px blobs in a
# 4000×3000 image refused, 56% of direct_ellipse fits nothing).
@testitem "conic class: Float32 ellipses at pixel coordinates" begin
    using Random, StaticArrays, GeometryBasics
    using VisualGeometryToolkitCore
    using VisualGeometryToolkitCore: boundary_point
    e = Ellipse(Point2f(1000, 800), 3.0f0, 1.0f0, 0.3f0)
    @test classify_conic(HomEllipseMat(e)) isa EllipseTrait
    @test Ellipse(Conic(e)) isa Ellipse{Float32}
    @test isapprox(Ellipse(Conic(e)).a, 3; rtol = 0.02)
    @test Ellipse(HomographyMap(SMatrix{3, 3, Float32}(1, 0, 0, 0, 1, 0, 0.5, -0.25, 1))(e)) isa Ellipse{Float32}
    @test Circle(Conic(Circle(Point2f(1000, 800), 2.0f0))) isa Circle
    ring(x) = [Point2f(boundary_point(x, θ)) for θ in range(0, 2π; length = 25)[1:24]]
    @test direct_ellipse(ring(e)) isa Ellipse{Float32}
    rng = Xoshiro(7)
    refused = fitted_nothing = 0
    for _ in 1:2000
        a = Float32(1 + 9rand(rng))
        b = Float32(1 + (a - 1) * rand(rng))
        x = Ellipse(Point2f(4000rand(rng), 3000rand(rng)), a, b, Float32(π * rand(rng)))
        direct_ellipse(ring(x)) === nothing && (fitted_nothing += 1)   # classified where it is solved
        # below a 2 px minor axis a Float32 coefficient set itself no longer resolves γ at 4000 px
        b >= 2 && !is_ellipse(Conic(x)) && (refused += 1)
    end
    @test fitted_nothing == 0
    @test refused == 0
end

# Review finding 7: the symmetry and class checks squared raw entries in the target eltype.
@testitem "conic checks: exact types, scale and the input's precision" begin
    using Random, StaticArrays, LinearAlgebra, GeometryBasics
    using VisualGeometryToolkitCore
    @test HomEllipseMat([1 0 0; 0 1 0; 0 0 -1]) isa HomEllipseMat{Int}                  # was InexactError
    @test HomEllipseMat(Rational{Int}[1 0 0; 0 1 0; 0 0 -1]) isa HomEllipseMat{Rational{Int}}   # was MethodError
    unit = Ellipse(HomEllipseMat([1 0 0; 0 1 0; 0 0 -1]))
    @test unit isa Ellipse{Float64} && unit.center == Point2(0.0, 0.0) && (unit.a, unit.b) == (1.0, 1.0)
    Q = SMatrix{3, 3}(HomEllipseMat(Ellipse(Point2(3.0, 2.0), 2.0, 1.0, 0.4)))
    @test classify_conic(Conic(1.0e160 * Q)) isa EllipseTrait                           # was a non-finite refusal
    @test classify_conic(Conic(1.0e-170 * Q)) isa EllipseTrait                          # was Degenerate
    @test classify_conic(Conic(Float32(1.0e20) * SMatrix{3, 3, Float32}(Q))) isa EllipseTrait
    @test HomCircleMat(1.0e160 * SMatrix{3, 3}(HomCircleMat(Circle(Point2(1.0, 2.0), 3.0)))) isa HomCircleMat
    @test_throws ArgumentError HomCircleMat(1.0e160 * Q)                                # isotropy is relative too
    # Float32 data read into a Conic{Float64}: its asymmetry is Float32's, and so is the tolerance
    rng = Xoshiro(3)
    @test all(1:500) do _
        H = SMatrix{3, 3, Float32}(I) + 0.3f0 * @SMatrix(randn(rng, Float32, 3, 3))
        C = H' * SMatrix{3, 3, Float32}(Q) * H
        Conic{Float64}(C) isa Conic{Float64}
    end
    @test_throws ArgumentError Conic{Float64}(Q + SMatrix{3, 3}(0, 1.0e-3, 0, 0, 0, 0, 0, 0, 0))
    # one circle predicate: Circle(Q) of an ellipse's or a conic's matrix is the HomCircleMat class check
    big = Circle(Point2(1.0e5, 2.0), 1.0e3)
    @test Circle(HomCircleMat(big)).r ≈ 1.0e3 && Circle(HomEllipseMat(Ellipse(big))).r ≈ 1.0e3
    @test_throws ArgumentError Circle(HomEllipseMat(Ellipse(Point2(1.0, 2.0), 3.0, 1.0, 0.0)))
    @test_throws ArgumentError Circle(Conic(Ellipse(Point2(1.0, 2.0), 3.0, 1.0, 0.0)))
end

# Review findings 3 and 15: StaticArrays' generic copy/convert/zero rebuilt the
# structured matrices through their checked constructors; HomographyMat lost
# its same-type promotion.
@testitem "structured matrices: copy, convert, padding, promotion" begin
    using StaticArrays, LinearAlgebra, GeometryBasics
    using VisualGeometryToolkitCore
    Q = HomEllipseMat(Ellipse(Point2f(1000, 800), 3.0f0, 1.0f0, 0.3f0))
    @test copy(Q) === Q
    v = HomEllipseMat{Float64}[]
    push!(v, HomEllipseMat(Ellipse(Point2f(1000, 800), 300.0f0, 0.05f0, 0.3f0)))   # converted, not re-checked
    @test v[1] isa HomEllipseMat{Float64}
    hyperbola = SMatrix{3, 3}(1.0, 0, 0, 0, -1.0, 0, 0, 0, -1.0)
    Qh = HomEllipseMat(hyperbola; check = false)
    @test copy(Qh) === Qh && convert(HomEllipseMat{Float32}, Qh) isa HomEllipseMat{Float32}
    @test_throws ArgumentError convert(HomEllipseMat{Float64}, Conic(hyperbola))   # another class: checked
    Kf = SMatrix{3, 3}(800.0, 0, 0, 0.3, 790.0, 0, 320.0, 240.0, 1)
    K = CameraCalibrationMat(Kf)
    fm = FixedModels{2, CameraCalibrationMat{Float64}}(i -> K, 1)
    @test length(fm) == 1 && fm[1] === K
    Kn = CameraCalibrationMat(SMatrix{3, 3}(-800.0, 0, 0, 0, 790.0, 0, 320.0, 240.0, 1); check = false)
    @test copy(Kn) === Kn
    @test convert(CameraCalibrationMat{Float32}, Kn)[1, 1] == -800.0f0
    @test CameraModel(Kn; image_origin = :opencv) isa CameraModel
    # a non-finite focal with check = false (a fit that ended non-finite reports it, VisualGeometryToolkitEstimation's fit_camera)
    @test CameraCalibrationMat(SMatrix{3, 3}(Inf, 0, 0, 0, Inf, 0, 320.0, 240.0, 1); check = false)[1, 1] == Inf
    # a 3-D affine map is not a calibration matrix
    @test_throws ArgumentError CameraModel(AffineMap(SMatrix{3, 3}(1.0I), SVector(1.0, 2.0, 3.0)))
    # finding 14: K·v is StaticArrays' product (the vector's type kept, bit for bit SMatrix(K)·v)
    p = Point3(1.0, 2.0, 3.0)
    @test K * p isa Point3{Float64} && K * p == SMatrix(K) * p
    # finding 15
    H32, H64 = HomographyMat(SMatrix{3, 3, Float32}(I)), HomographyMat(SMatrix{3, 3, Float64}(I))
    @test promote_type(typeof(H32), typeof(H64)) == HomographyMat{Float64}
    @test [H32, H64] isa Vector{HomographyMat{Float64}}
end

# Review findings 2, 8, 9 and 11: the similarity class composes exactly and
# junction-checked with every similarity-class partner; its line action
# promotes over every part; a physical map is refused by name.
@testitem "SimilarityMap: the whole class under ∘, lines, units, propagate" setup = [MapClassFixtures] begin
    using StaticArrays, LinearAlgebra, Rotations, ForwardDiff, GeometryBasics
    using CoordinateTransformations: LinearMap, Translation, AffineMap
    using VisualGeometryToolkitCore
    using VisualGeometryToolkitCore: relabel, scale, m, similarity_scale
    R, t = RotMatrix{2}(0.3), SVector(1.0, -2.0)
    S = SimilarityMap(2.0, R, t)
    x = SVector(0.7, -1.1)
    for B in (LinearMap(RotMatrix{2}(0.2)), LinearMap(2.0I), RotationMap(RotMatrix{2}(0.2)), Translation(SVector(1.0, 2.0)))
        @test S ∘ B isa SimilarityMap{2} && (S ∘ B)(x) ≈ S(B(x))
        @test B ∘ S isa SimilarityMap{2} && (B ∘ S)(x) ≈ B(S(x))
    end
    # a labeled junction mismatch is refused, as for an EuclideanMap
    Rm = relabel(TargetPlaneFrame, PixelFrame{:julia}, RotationMap(R))
    Sc = relabel(PixelFrame{:julia}, PixelFrame{:julia}, S)
    @test_throws MethodError Rm ∘ Sc
    @test Sc ∘ Rm isa SimilarityMap{2, Float64, Float64, <:Any, <:Any, TargetPlaneFrame, PixelFrame{:julia}}
    # the model-matrix and epipolar bridges take the stack's affine-class maps
    @test S ∘ HomographyMat(SMatrix{3, 3}(1.0I)) isa HomographyMat
    @test S ∘ FundMat(SMatrix{3, 3}(0.0, 1, 0, -1, 0, 0, 0, 0, 0)) isa FundMat
    @test EpipolarMap(FundMat(SMatrix{3, 3}(0.0, 1, 0, -1, 0, 0, 0, 0, 0))) ∘ S isa EpipolarMap
    # to_homogeneous of a CoordinateTransformations composition (the product)
    A = AffineMap(@SMatrix([1.0 2; 3 4]), SVector(1.0, 2))
    @test to_homogeneous(LinearMap(@SMatrix([1.0 2; 0 1])) ∘ A) ≈ to_homogeneous(LinearMap(@SMatrix([1.0 2; 0 1]))) * to_homogeneous(A)
    # the line action differentiates through s and t and promotes over every part
    @test ForwardDiff.derivative(s -> SimilarityMap(s, R, t)(Line2D(1.0, 2.0, -3.0)).coeffs[3], 2.0) isa Float64
    @test ForwardDiff.derivative(a -> EuclideanMap(R, SVector(a, 2.0))(Line2D(1.0, 2.0, -3.0)).coeffs[3], 1.0) isa Float64
    @test SimilarityMap(2.0f0, RotMatrix{2, Float32}(0.3f0), SVector(1000.1, -2.0))(Line2D(1.0f0, 2.0f0, -3.0f0)) isa Line2D{Float64}
    # the storage-boundary convert labels an unlabeled map; propagate and similarity_scale
    Sl = relabel(TargetPlaneFrame, PixelFrame{:colmap}, S)
    slot = typeof(Sl)[]
    push!(slot, S)
    @test slot[1] isa typeof(Sl)
    Σ = CovMat(SMatrix{2, 2}(1.0, 0.1, 0.1, 2.0))
    J = 2.0 * SMatrix{2, 2}(R)
    @test SMatrix{2, 2}(propagate(S, Σ)) ≈ J * SMatrix{2, 2}(Σ) * J'
    @test SMatrix{2, 2}(propagate(S, inv(Σ))) ≈ inv(SMatrix{2, 2}(propagate(S, Σ)))
    @test similarity_scale(S) == 2.0
    # units: one refusal naming the fix, wherever a physical map meets a plain-only operation
    Eu = EuclideanMap(RotMatrix{2}(0.1), SVector(1.0, 2.0) .* m)
    H = HomographyMap(SMatrix{3, 3}(1.0I))
    for f in (() -> SimilarityMap(2.0, R, Any[1.0m, 2.0m]), () -> Eu ∘ H, () -> H ∘ Eu, () -> HomographyMap(Eu),
            () -> Eu ∘ S, () -> SimilarityMap(Eu), () -> Eu(Circle(Point2(1.0, 2.0), 1.0)))
        err = try
            f()
            nothing
        catch e
            e
        end
        @test err isa ArgumentError && occursin("ustrip(m, E)", err.msg)
    end
end

# Review finding 4: the affine-class actions read ONE checked, static inverse.
@testitem "affine maps: checked static inverse, class agreement" begin
    using StaticArrays, LinearAlgebra, GeometryBasics
    using CoordinateTransformations: LinearMap, AffineMap
    using VisualGeometryToolkitCore
    singular = AffineMap(SMatrix{2, 2}(1.0, 2.0, 2.0, 4.0), SVector(1.0, 2.0))
    c = Circle(Point2(1.0, 2.0), 3.0)
    @test_throws ArgumentError singular(c)                         # was a HomEllipseMat of NaNs
    @test_throws ArgumentError singular(Line2D(1.0, 2.0, -3.0))    # was 'Normal vector cannot be zero'
    @test_throws ArgumentError HomographyMap(singular)              # was an Inf inverse stored
    @test_throws ArgumentError LinearMap(Diagonal(SVector(0.0, 1.0)))(Line2D(1.0, 2.0, -3.0))   # was SingularException
    # so ill-conditioned the image reads as a parabola: refused, never a HomEllipseMat classify_conic disowns
    @test_throws ArgumentError LinearMap(Diagonal(SVector(1.0e7, 1.0)))(c)
    Q = LinearMap(Diagonal(SVector(1.0e3, 1.0)))(c)
    @test Q isa HomEllipseMat && classify_conic(Q) isa EllipseTrait
    # a Matrix-backed map is made static first: no allocation per primitive
    am = AffineMap([1.3 -0.4; 0.2 0.9], [3.7, -1.3])
    @test am(c) isa HomEllipseMat{Float64}
    am(c)
    @test (@allocated am(c)) == 0
end

# Review finding 6: an unsized or 3-D Translation is not a planar matrix leaf.
@testitem "fuse: an unsized Translation stays a lazy link" begin
    using StaticArrays
    using CoordinateTransformations: Translation, PerspectiveMap
    using VisualGeometryToolkitCore
    using VisualGeometryToolkitCore: fuse
    H = HomographyMap(SMatrix{3, 3}(1.0, 0.1, 0.001, 0.2, 1.0, 0.002, 3.0, 4.0, 1.0))
    chain = Translation([1.0, 2.0]) ∘ PerspectiveMap() ∘ H
    x = SVector(0.3, 0.4)
    @test fuse(chain)(x) ≈ chain(x)
    @test image_warp(chain, 4) isa BoundedMap   # was a MethodError on _as_homography
end
