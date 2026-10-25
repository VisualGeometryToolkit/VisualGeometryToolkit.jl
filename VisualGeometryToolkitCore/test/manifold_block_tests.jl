@testitem "ManifoldBlock and is_stationary" begin
# ManifoldBlock: a manifold as an NLLSsolver variable block (the NLLSsolver
# extension) — its size, its re-anchoring update, a pose fit through it — and
# is_stationary, the LM stall postcondition for a solver other than FLM.

using Test
using LinearAlgebra: norm
using StaticArrays
using NLLSsolver: NLLSsolver
using ForwardDiff: ForwardDiff
using Rotations: RotMatrix, RotXYZ
using VisualGeometryToolkitCore: EuclideanMap, ManifoldBlock, RigidExpManifold6, is_stationary, local_coordinates,
    model, recenter, retract

pose(ω, t) = EuclideanMap(RotMatrix{3}(RotXYZ(ω...)), SVector{3, Float64}(t))

# One point's residual T(x) − y, through the block's pose.
struct _PointResidual <: NLLSsolver.AbstractResidual
    x::SVector{3, Float64}
    y::SVector{3, Float64}
end
NLLSsolver.ndeps(::_PointResidual) = NLLSsolver.static(1)
NLLSsolver.nres(::_PointResidual) = NLLSsolver.static(3)
NLLSsolver.varindices(::_PointResidual) = SVector(1)
NLLSsolver.getvars(::_PointResidual, vars::Vector) = (vars[1],)
NLLSsolver.computeresidual(r::_PointResidual, b::ManifoldBlock) = model(b.manifold)(r.x) - r.y
# The Jacobian in the block's coordinates, ForwardDiff through `retract` at the
# chart origin (NLLSsolver's own autodiff would differentiate `update`, whose
# `recenter` stores the manifold's own eltype).
function NLLSsolver.computeresjac(_, r::_PointResidual, b::ManifoldBlock)
    J = ForwardDiff.jacobian(δ -> retract(b.manifold, δ)(r.x) - r.y, zero(SVector{6, Float64}))
    return NLLSsolver.computeresidual(r, b), J
end
Base.eltype(::_PointResidual) = Float64

@testset "ManifoldBlock and is_stationary" begin
    @testset "the block: size, properties, update" begin
        ch = RigidExpManifold6(pose((0.1, -0.2, 0.3), (1.0, 2.0, 3.0)))
        b = ManifoldBlock(ch)
        @test b isa ManifoldBlock{6, typeof(ch)}
        @test b.manifold === ch
        @test :manifold in propertynames(b)
        @test NLLSsolver.nvars(b) == 6
        δ = SVector(0.01, -0.02, 0.03, 0.1, 0.2, -0.3)
        @test NLLSsolver.update(b, δ).manifold == recenter(ch, δ)
        @test NLLSsolver.update(b, vcat(SVector(9.0, 9.0), δ), 3).manifold == recenter(ch, δ)   # read from `start`
        @test typeof(NLLSsolver.update(b, δ)) === typeof(b)                                    # one concrete block type
    end

    @testset "a pose fit on NLLSsolver through the block" begin
        truth = pose((0.3, -0.1, 0.2), (0.5, -0.2, 1.0))
        xs = [SVector(sin(i), cos(2i), 0.1i) for i in 1:12]
        problem = NLLSsolver.NLLSProblem(ManifoldBlock{6, RigidExpManifold6{Float64}}, _PointResidual)
        NLLSsolver.addvariable!(problem, ManifoldBlock(RigidExpManifold6(pose((0.0, 0.0, 0.0), (0.0, 0.0, 0.0)))))
        for x in xs
            NLLSsolver.addcost!(problem, _PointResidual(x, truth(x)))
        end
        result = NLLSsolver.optimize!(problem, NLLSsolver.NLLSOptions(; numthreads = NLLSsolver.static(1)))
        @test result.bestcost < 1.0e-20
        @test norm(local_coordinates(RigidExpManifold6(model(problem.variables[1].manifold)), truth)) < 1.0e-9
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
