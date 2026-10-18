# A lens is a map and so is its inverse: `transform_deriv` on both, `inv`
# between them, and the chain rule through compositions. Checked against
# central differences for every lens family and as mutual inverses.
@testitem "lens_jacobians" begin
    using Test, LinearAlgebra
    using VisualGeometryToolkitCore
    using FiniteDiff: FiniteDiff
    using StaticArrays: SVector, SMatrix
    using CoordinateTransformations: AffineMap, Translation
    using VisualGeometryToolkitCore: Pinhole, Division, Scaramuzza, InverseLens, undistort_normalized,
        distort_normalized, transform_deriv

    const _LENSES = [
        ("pinhole", Pinhole()),
        ("division-1", Division(-0.1667)),
        ("division-2", Division(-0.1667, 0.0291)),
        ("brown-conrady", Lens(Perspective(), Additive(Radial((-0.157, 0.046)), Tangential(-3.5e-4, 8.4e-4)))),
        ("brown-conrady thin prism", Lens(Perspective(), Additive(Radial((-0.157, 0.046, 0.004), (0.02, -0.001, 0.0003)), Tangential(-3.5e-4, 8.4e-4), ThinPrism((-5.5e-3, 7.0e-4), (2.2e-3, -1.2e-3))))),
        ("scaramuzza", Scaramuzza(-0.1664, -0.0063, 0.035)),
    ]

    @testset "inv(lens) is the back-projection as a map ($name)" for (name, lens) in _LENSES
        U = inv(lens)
        @test inv(U) === lens || (lens isa Pinhole && U isa Pinhole)
        for xd in (SVector(0.0, 0.0), SVector(0.05, -0.02), SVector(-0.4, 0.3), SVector(0.7, 0.55), SVector(0.0, 0.85))
            @test U(xd) ≈ undistort_normalized(lens, xd) atol = 1.0e-15
            J = transform_deriv(U, xd)
            @test J isa SMatrix{2, 2, Float64, 4}
            @test isapprox(Matrix(J), Matrix(FiniteDiff.finite_difference_jacobian(U, xd, Val(:central))); rtol = 1.0e-6, atol = 1.0e-9)
            # mutual inverses: forward Jacobian at the undistorted point
            xu = U(xd)
            @test isapprox(Matrix(transform_deriv(lens, xu) * J), Matrix(I, 2, 2); atol = 1.0e-8)
            @test isapprox(Matrix(transform_deriv(lens, xu)), Matrix(FiniteDiff.finite_difference_jacobian(lens, xu, Val(:central))); rtol = 1.0e-6, atol = 1.0e-9)
            # a shape carried by J is what the finite map does to it
            @test isapprox(J * SVector(1.0, 2.0) * 1.0e-3, U(xd + SVector(1.0, 2.0) * 1.0e-3) - xu; rtol = 1.0e-3)
        end
        @test transform_deriv(U, SVector(0.0f0, 0.1f0)) isa SMatrix{2, 2, Float32, 4}
    end

    @testset "compositions take the chain rule" begin
        lens = Division(-0.1667, 0.0291)
        K = AffineMap(SMatrix{2, 2, Float64}(2464.0, 0.0, 0.0, 2464.0), SVector(2136.2, 1410.1))
        U = Translation(SVector(0.5, 0.5)) ∘ K ∘ inv(lens) ∘ inv(K) ∘ Translation(SVector(-0.5, -0.5))   # pixel undistortion
        x = SVector(3800.0, 400.0)
        J = transform_deriv(U, x)
        @test isapprox(Matrix(J), Matrix(FiniteDiff.finite_difference_jacobian(U, x, Val(:central))); rtol = 1.0e-6)
        @test isapprox(SVector{2, Float64}(inv(U)(U(x))), x; atol = 1.0e-6)                               # inv composes too
        @test isapprox(Matrix(transform_deriv(inv(U), U(x)) * J), Matrix(I, 2, 2); atol = 1.0e-8)
    end
end
