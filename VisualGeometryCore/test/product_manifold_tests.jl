# EuclideanManifold and ProductManifold: a plain vector as a chart, a model of
# independent parts, and the radial homography's sl(3) ⊕ k chart — whose
# directions are the radial transfer LM rows' Jacobian columns.
@testitem "product_manifold" begin
    using Test
    using LinearAlgebra
    using StaticArrays
    using Random
    using Rotations
    using ForwardDiff: ForwardDiff
    using FiniteDiff: FiniteDiff
    using VisualGeometryCore
    using VisualGeometryCore: EuclideanManifold, ProductManifold, RadialHomography, retract, recenter,
        retract_jacobian, local_coordinates, tangent_size, model, minimal_manifold, transform_covariance,
        radial_transfer, radial_transfer_rows!, radial_transfer_jacobian_rows!

    @testset "EuclideanManifold" begin
        ch = EuclideanManifold(SVector(1.0, -2.0, 3.0))
        v = SVector(0.1, 0.2, -0.3)
        @test tangent_size(ch) == 3 && model(ch) == SVector(1.0, -2.0, 3.0)
        @test retract(ch, v) == model(ch) + v
        @test retract_jacobian(ch, v) == SMatrix{3, 3, Float64}(I)
        @test local_coordinates(ch, retract(ch, v)) ≈ v atol = 1.0e-15
        @test model(recenter(ch, v)) == retract(ch, v)
    end

    @testset "ProductManifold: a pose and a point" begin
        pose = RigidManifold6(RotMatrix{3, Float64}(RotXYZ(0.3, -0.2, 0.5)), SVector(1.0, -2.0, 3.0))
        pt = EuclideanManifold(SVector(0.4, -0.7, 2.5))
        ch = ProductManifold(pose, pt)
        @test tangent_size(ch) == 9
        v = SVector(0.05, -0.02, 0.03, 0.4, -0.1, 0.25, 0.01, 0.02, -0.03)
        P, X = retract(ch, v)
        @test P == retract(pose, v[SOneTo(6)]) && X == retract(pt, SVector(v[7], v[8], v[9]))
        J = retract_jacobian(ch, v)
        @test size(J) == (15, 9)
        @test J[1:12, 1:6] ≈ retract_jacobian(pose, v[SOneTo(6)]) && J[13:15, 7:9] == I && iszero(J[1:12, 7:9]) && iszero(J[13:15, 1:6])
        @test local_coordinates(ch, (P, X)) ≈ v atol = 1.0e-12
        P2, X2 = model(recenter(ch, v))
        @test P2.t ≈ P.t && X2 == X
        amb = EuclideanManifold(ch)                                        # the ambient chart: [vec R; t; X]
        pose0, X0 = model(ch)
        @test amb isa EuclideanManifold{15}
        @test model(amb) == vcat(vec(SMatrix{3, 3}(pose0.R)), pose0.t, X0)
        @test size(transform_covariance(ch, amb, Matrix(1.0I, 9, 9))) == (15, 15)
        # the parts are static slices of v's eltype: the retraction infers,
        # ForwardDiff Duals and a dynamically sized v included
        @test @inferred(VisualGeometryCore.manifold_parts(ch, Vector(v))) == (v[SOneTo(6)], SVector(v[7], v[8], v[9]))
        @test ForwardDiff.jacobian(w -> retract(ch, w)[2], Vector(v)) == hcat(zeros(3, 6), I(3))
        # a part with no coordinates is an empty vector of v's eltype
        @test VisualGeometryCore.manifold_parts(ProductManifold(pt, EuclideanManifold(SVector{0, Float64}())), SVector(1.0, 2.0, 3.0))[2] isa SVector{0, Float64}
    end

    @testset "minimal_manifold(::RadialHomography): sl(3) ⊕ k, the radial rows' chart" begin
        H = HomographyMat(SMatrix{3, 3, Float64}(1.0, 0.01, 1.0e-5, -0.02, 0.98, -2.0e-5, 30.0, -20.0, 1.0))
        c = Point2(640.0, 480.0)
        m = RadialHomography{Float64}(H, -1.1e-7, c)
        ch = minimal_manifold(m)
        @test tangent_size(ch) == 9
        v = SVector(1.0e-3, -2.0e-3, 5.0e-4, 1.0e-3, -1.0e-4, 2.0e-4, 1.0e-6, -2.0e-6, 3.0e-8)
        mv = retract(ch, v)
        @test mv isa RadialHomography{Float64} && mv.center == c && mv.k == m.k + v[9]
        J = retract_jacobian(ch, v)
        @test J ≈ ForwardDiff.jacobian(w -> (r = retract(ch, w); vcat(SVector{9}(Tuple(r.H)), SVector(r.k))), v) atol = 1.0e-15
        @test local_coordinates(ch, mv) ≈ v atol = 1.0e-13
        # the radial LM rows along this chart: dH from its first part, k its last column
        rng = Xoshiro(7)
        n = 30
        u₁ = [SVector(1200rand(rng), 900rand(rng)) for _ in 1:n]
        u₂ = [radial_transfer(m, s) + 0.5 * SVector(randn(rng), randn(rng)) for s in u₁]
        sw = ones(2n)
        dH = ntuple(q -> SMatrix{3, 3}(J[1:9, q]), 8)
        Jr = zeros(2n, 9)
        radial_transfer_jacobian_rows!(Jr, mv, dH, u₁, sw)
        rows(w) = (r = zeros(2n); radial_transfer_rows!(r, retract(ch, w), u₁, u₂, sw); r)
        # k's coordinate is ~1e-7 against sl(3)'s ~1e-3, and FiniteDiff's step is one
        # scalar: differentiate in k/|k| so that step fits every coordinate
        S = SVector(ntuple(q -> q == 9 ? abs(m.k) : 1.0, 9))
        fd = FiniteDiff.finite_difference_jacobian(z -> rows(S .* z), v ./ S, Val(:central)) ./ transpose(S)
        @test Jr ≈ fd rtol = 1.0e-5 atol = 1.0e-6 * maximum(abs, Jr)
    end
end
