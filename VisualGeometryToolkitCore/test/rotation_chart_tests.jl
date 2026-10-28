# =============================================================================
# RotationChart / SO3Manifold (geometry/manifolds/rigid.jl): the manifold
# contract, local_coordinates ∘ retract, the closed-form differentials against
# ForwardDiff, the rotation block of the rigid charts, Float32 and Dual steps.
# =============================================================================

@testitem "RotationChart: the 3-dof member of the rigid family" begin
    using Test, StaticArrays, LinearAlgebra, Random, Rotations
    import ForwardDiff
    using VisualGeometryToolkitCore: RotationChart, SO3Manifold, MRPParam, ExpParam, LeftRigidManifold6, RigidManifold6,
        retract, recenter, local_coordinates, retract_jacobian, action_jacobian, tangent_size, zero_tangent, model,
        minimal_manifold, ambient_coordinates, test_manifold_contract

    R0 = SMatrix{3, 3, Float64, 9}(RotXYZ(0.3, -0.2, 0.5))
    mrp = RotationChart{MRPParam, RotMatrix{3, Float64, 9}}(RotMatrix{3}(R0))
    for ch in (SO3Manifold(R0), mrp, SO3Manifold(SMatrix{3, 3, Float32}(R0)))
        test_manifold_contract(ch; atol = eltype(ch.R) == Float32 ? 1.0e-5 : 1.0e-12)
    end
    @test minimal_manifold(RotMatrix{3}(R0)) isa SO3Manifold{Float64}
    rng = Xoshiro(4)
    for ch in (SO3Manifold(R0), mrp)
        @test tangent_size(ch) == 3 && model(ch) ≈ R0
        for _ in 1:50
            ω = 0.3 * randn(rng, SVector{3, Float64})
            @test local_coordinates(ch, retract(ch, ω)) ≈ ω rtol = 1.0e-10           # the inverse retraction
            x = randn(rng, SVector{3, Float64})
            @test action_jacobian(ch, x, ω) ≈ ForwardDiff.jacobian(w -> retract(ch, w) * x, ω) rtol = 1.0e-10
            @test retract_jacobian(ch, ω) ≈ ForwardDiff.jacobian(w -> ambient_coordinates(retract(ch, w)), ω) rtol = 1.0e-10
        end
        @test action_jacobian(ch, SVector(0.2, 0.1, 1.0)) ≈ ForwardDiff.jacobian(w -> retract(ch, w) * SVector(0.2, 0.1, 1.0), zero_tangent(ch))
    end
    # the rotation block of the rigid charts, one formula: same step, same rotation
    ω = SVector(0.02, -0.01, 0.03)
    @test retract(SO3Manifold(R0), ω) ≈ retract(LeftRigidManifold6(R0, zeros(3)), vcat(ω, zeros(3))).R
    @test retract(mrp, ω) ≈ retract(RigidManifold6(RotMatrix{3}(R0), zeros(3)), vcat(ω, zeros(3))).R
    # a Dual step gives a Dual chart, a Float32 chart stays Float32
    d = ForwardDiff.Dual(0.01, 1.0)
    @test eltype(recenter(SO3Manifold(R0), SVector(d, 0.0, 0.0)).R) <: ForwardDiff.Dual
    @test recenter(SO3Manifold(SMatrix{3, 3, Float32}(R0)), SVector{3, Float32}(0.01, 0, 0)) isa SO3Manifold{Float32}
end
