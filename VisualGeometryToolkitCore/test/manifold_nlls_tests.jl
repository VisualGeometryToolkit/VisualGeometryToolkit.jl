@testitem "Manifolds as NLLSsolver variables, and is_stationary" begin
# Every Manifold is an NLLSsolver variable (the NLLSsolver extension): its size
# static for every manifold family, its re-anchoring update, a pose fit through
# it — tangent_part, the slice both the update and a product manifold cut — and
# is_stationary, the LM stall postcondition for a solver other than FLM.

using Test
using LinearAlgebra: I, norm
using StaticArrays
using NLLSsolver: NLLSsolver
using ForwardDiff: ForwardDiff
using Rotations: RotMatrix, RotXYZ
using VisualGeometryToolkitCore: DoubleSphere, EuclideanManifold, EuclideanMap, FundMatManifold7, HomographyManifold8,
    LeftRigidManifold6, Lens, LensManifold, Pinhole, PlanarTarget, ProductManifold, RigidExpManifold6, RigidManifold6,
    SphereManifold, TargetManifold, is_stationary, local_coordinates, manifold_parts, model, recenter, restrict, retract,
    tangent_part, tangent_size

pose(ω, t) = EuclideanMap(RotMatrix{3}(RotXYZ(ω...)), SVector{3, Float64}(t))

# One point's residual T(x) − y, through the variable's pose.
struct _PointResidual <: NLLSsolver.AbstractResidual
    x::SVector{3, Float64}
    y::SVector{3, Float64}
end
NLLSsolver.ndeps(::_PointResidual) = NLLSsolver.static(1)
NLLSsolver.nres(::_PointResidual) = NLLSsolver.static(3)
NLLSsolver.varindices(::_PointResidual) = SVector(1)
NLLSsolver.getvars(::_PointResidual, vars::Vector) = (vars[1],)
NLLSsolver.computeresidual(r::_PointResidual, ch::RigidExpManifold6) = model(ch)(r.x) - r.y
# The Jacobian in the manifold's coordinates, ForwardDiff through `retract` at
# the chart origin (NLLSsolver's own autodiff would differentiate `update`, whose
# `recenter` keeps the manifold's storage type).
function NLLSsolver.computeresjac(_, r::_PointResidual, ch::RigidExpManifold6)
    J = ForwardDiff.jacobian(δ -> retract(ch, δ)(r.x) - r.y, zero(SVector{6, Float64}))
    return NLLSsolver.computeresidual(r, ch), J
end
Base.eltype(::_PointResidual) = Float64

nvars_of(ch) = NLLSsolver.nvars(ch)

@testset "Manifolds as NLLSsolver variables, and is_stationary" begin
    R0 = SMatrix{3, 3, Float64, 9}(RotXYZ(0.3, -0.2, 0.5))
    t0 = SVector(1.0, -2.0, 3.0)
    ch = RigidExpManifold6(R0, t0)

    @testset "the size is static for every manifold family" begin
        # A variable's size must be a compile-time constant of its type: it
        # selects NLLSsolver's static, allocation-free path.
        for (m, D) in (
                (ch, 6), (LeftRigidManifold6(R0, t0), 6), (RigidManifold6(R0, t0), 6),
                (HomographyManifold8(R0), 8), (FundMatManifold7(R0 + 0.1I), 7),
                (SphereManifold(SVector(0.0, 0.6, 0.8)), 2), (EuclideanManifold(t0), 3),
                (LensManifold(Lens(DoubleSphere(-0.2, 0.6))), 2), (LensManifold(Pinhole()), 0),
                (TargetManifold(PlanarTarget(:board)), 0),
                (ProductManifold(ch, EuclideanManifold(t0)), 9),
                (restrict(ch, SVector(true, true, true, false, false, true)), 4),
            )
            @test only(Base.return_types(nvars_of, (typeof(m),))) === typeof(NLLSsolver.static(D))
            @test tangent_size(m) == D
        end
    end

    @testset "tangent_part, the update" begin
        δ = SVector(0.01, -0.02, 0.03, 0.1, 0.2, -0.3)
        v = vcat(SVector(9.0, 9.0), δ, SVector(7.0))
        @test tangent_part(ch, v, 2) === δ
        @test tangent_part(ch, δ) === δ
        @test eltype(tangent_part(ch, ForwardDiff.Dual.(v), 2)) <: ForwardDiff.Dual   # Duals flow through
        @test manifold_parts(ProductManifold(ch, EuclideanManifold(t0)), vcat(δ, t0)) == (δ, t0)
        @test NLLSsolver.update(ch, δ) == recenter(ch, δ)
        @test NLLSsolver.update(ch, v, 3) == recenter(ch, δ)        # read from `start`
        @test typeof(NLLSsolver.update(ch, δ)) === typeof(ch)        # the type the variable vector holds
    end

    @testset "a pose fit on NLLSsolver, the manifold as its variable" begin
        truth = pose((0.3, -0.1, 0.2), (0.5, -0.2, 1.0))
        xs = [SVector(sin(i), cos(2i), 0.1i) for i in 1:12]
        problem = NLLSsolver.NLLSProblem(RigidExpManifold6{Float64}, _PointResidual)
        NLLSsolver.addvariable!(problem, RigidExpManifold6(pose((0.0, 0.0, 0.0), (0.0, 0.0, 0.0))))
        for x in xs
            NLLSsolver.addcost!(problem, _PointResidual(x, truth(x)))
        end
        result = NLLSsolver.optimize!(problem, NLLSsolver.NLLSOptions(; numthreads = NLLSsolver.static(1)))
        @test result.bestcost < 1.0e-20
        @test norm(local_coordinates(problem.variables[1], truth)) < 1.0e-9
    end

    @testset "is_stationary" begin
        J = [1.0 0.0; 0.0 2.0; 1.0 1.0]
        y = [1.0, 2.0, 4.0]
        x̂ = J \ y
        @test is_stationary(J, J * x̂ - y; tol = 1.0e-10, ftol = eps())          # the least-squares solution
        @test !is_stationary(J, J * (x̂ .+ 0.1) - y; tol = 1.0e-3, ftol = eps())  # away from it
        @test is_stationary(J, zeros(3); tol = 1.0e-10, ftol = eps())            # an exact fit
        @test is_stationary(hcat(J, zeros(3)), J * x̂ - y; tol = 1.0e-10, ftol = eps())   # a zero column is skipped
        @test !is_stationary(J, [NaN, 0.0, 1.0]; tol = 1.0e-3, ftol = eps())     # a NaN fails
    end
end
end
