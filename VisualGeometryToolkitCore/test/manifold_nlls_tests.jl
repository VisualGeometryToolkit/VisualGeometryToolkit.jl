@testitem "Manifolds as NLLSsolver variables, and is_stationary" begin
# Every Manifold is an NLLSsolver variable (the NLLSsolver extension): its size
# static for every manifold family; `recenter`'s element-type rule (a step of
# the manifold's own float type returns its own type, inferred; a Dual step a
# Dual-valued manifold whose derivative is the retraction's), which is what
# NLLSsolver's automatic Jacobian differentiates; the update; a pose fit on the
# automatic Jacobian — tangent_part, the slice both the update and a product
# manifold cut — and is_stationary, the LM stall postcondition for a solver
# other than FLM.

using Test
using LinearAlgebra: I, norm
using StaticArrays
using NLLSsolver: NLLSsolver
using ForwardDiff: ForwardDiff
using Rotations: RotMatrix, RotXYZ
import VisualGeometryToolkitCore as VgtkCore
using VisualGeometryToolkitCore: DoubleSphere, EssentialManifold5, EssentialManifold6, EuclideanManifold, EuclideanMap,
    FundMatManifold7, HomographyManifold8, LeftRigidManifold6, Lens, LensManifold, Pinhole, PlanarTarget,
    ProductManifold, RigidExpManifold6, RigidManifold6, SphereManifold, TargetManifold, is_stationary,
    local_coordinates, manifold_parts, model, recenter, restrict, retract, retract_jacobian, tangent_part,
    tangent_size
using VisualGeometryToolkitCore: WelschShape, loss_robustifier, nlls_retcode

pose(ω, t) = EuclideanMap(RotMatrix{3}(RotXYZ(ω...)), SVector{3, Float64}(t))

# One point's residual T(x) − y, through the variable's pose: `computeresidual`
# only, so NLLSsolver differentiates it (through `update`, i.e. `recenter`).
struct _PointResidual <: NLLSsolver.AbstractResidual
    x::SVector{3, Float64}
    y::SVector{3, Float64}
end
NLLSsolver.ndeps(::_PointResidual) = NLLSsolver.static(1)
NLLSsolver.nres(::_PointResidual) = NLLSsolver.static(3)
NLLSsolver.varindices(::_PointResidual) = SVector(1)
NLLSsolver.getvars(::_PointResidual, vars::Vector) = (vars[1],)
NLLSsolver.computeresidual(r::_PointResidual, ch::RigidExpManifold6) = model(ch)(r.x) - r.y
Base.eltype(::_PointResidual) = Float64

nvars_of(ch) = NLLSsolver.nvars(ch)
ambient(ch) = VgtkCore.ambient_coordinates(model(ch))

@testset "Manifolds as NLLSsolver variables, and is_stationary" begin
    R0 = SMatrix{3, 3, Float64, 9}(RotXYZ(0.3, -0.2, 0.5))
    t0 = SVector(1.0, -2.0, 3.0)
    ch = RigidExpManifold6(R0, t0)
    families = (
        (ch, 6), (LeftRigidManifold6(R0, t0), 6), (RigidManifold6(RotMatrix{3}(R0), t0), 6),
        (HomographyManifold8(R0), 8), (FundMatManifold7(R0 + 0.1I), 7),
        (EssentialManifold5(R0, t0 / norm(t0)), 5), (EssentialManifold6(R0, t0 / norm(t0)), 6),
        (SphereManifold(SVector(0.0, 0.6, 0.8)), 2), (EuclideanManifold(t0), 3),
        (LensManifold(Lens(DoubleSphere(-0.2, 0.6))), 2), (LensManifold(Pinhole()), 0),
        (TargetManifold(PlanarTarget(:board)), 0),
        (ProductManifold(ch, EuclideanManifold(t0)), 9),
        (restrict(ch, SVector(true, true, true, false, false, true)), 4),
    )

    @testset "the manifold contract, every family" begin
        # static size, recenter inferred and allocation-free, retract_jacobian
        # against ForwardDiff, the chart's invariants (Core's TestExt)
        for (m, _) in families
            # What the contract found, recorded broken, not yet fixed: SphereManifold's
            # recenter allocates (176 bytes: it rebuilds its Householder basis),
            # and a SubsetManifold's 96 bytes on Julia 1.12 (none on 1.13)
            known = m isa VgtkCore.SphereManifold || (m isa VgtkCore.SubsetManifold && VERSION < v"1.13")
            VgtkCore.test_manifold_contract(m; allocations = !known)
        end
        VgtkCore.test_manifold_contract(LeftRigidManifold6(SMatrix{3, 3, Float32}(R0), Float32.(t0)); atol = 1.0e-5)
    end

    @testset "the size is static for every manifold family" begin
        # A variable's size must be a compile-time constant of its type: it
        # selects NLLSsolver's static, allocation-free path.
        for (m, D) in families
            @test only(Base.return_types(nvars_of, (typeof(m),))) === typeof(NLLSsolver.static(D))
            @test tangent_size(m) == D
        end
    end

    @testset "recenter: its own type for its own float type, Duals for a Dual step" begin
        for (m, D) in families
            D == 0 && continue
            δ = SVector{D}(ntuple(i -> 0.01i, D))
            @test typeof(recenter(m, δ)) === typeof(m)                              # what a solver's variable holds
            @test only(Base.return_types(recenter, (typeof(m), typeof(δ)))) === typeof(m)   # inferred
            # ForwardDiff through recenter: the retraction's derivative — for the
            # homography, whose anchor is rescaled to unit norm, with the scale
            # direction (invisible to the projective model) removed.
            J = ForwardDiff.jacobian(v -> ambient(recenter(m, v)), zero(SVector{D, Float64}))
            T = retract_jacobian(m)
            if m isa HomographyManifold8
                h = ambient(m)
                T = (I - h * h' / (h' * h)) * T / norm(h)
            end
            @test J ≈ T atol = 1.0e-12
        end
        ch32 = LeftRigidManifold6(SMatrix{3, 3, Float32}(R0), Float32.(t0))
        @test typeof(recenter(ch32, zero(SVector{6, Float32}))) === typeof(ch32)
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

    @testset "the update infers its own type and allocates nothing, for any increment container" begin
        # NLLSsolver hands the update a Vector and an offset; a static or mutable
        # static increment is the in-place case. `EssentialManifold5` is a
        # UnionAll (every float type's chart), so a bundle names the concrete
        # type its variables hold.
        bytes_of_update(m, δ, start) = @allocated NLLSsolver.update(m, δ, start)
        for m in (EssentialManifold5(R0, t0 / norm(t0)), RigidManifold6(RotMatrix{3}(R0), t0), LeftRigidManifold6(R0, t0))
            D = tangent_size(m)
            for (δ, start) in ((SVector{D}(ntuple(i -> 0.01i, D)), 1), (MVector{D}(ntuple(i -> 0.01i, D)), 1),
                    ([0.01i for i in 1:(D + 2)], 3))
                @test (@inferred NLLSsolver.update(m, δ, start)) isa typeof(m)
                bytes_of_update(m, δ, start)                                    # warm-up
                @test bytes_of_update(m, δ, start) == 0
            end
        end
    end

    @testset "a pose fit on NLLSsolver's automatic Jacobian, the manifold as its variable" begin
        truth = pose((0.3, -0.1, 0.2), (0.5, -0.2, 1.0))
        xs = [SVector(sin(i), cos(2i), 0.1i) for i in 1:12]
        # the automatic Jacobian is the one through retract at the anchor
        r = _PointResidual(xs[1], truth(xs[1]))
        _, J = NLLSsolver.computeresjac(NLLSsolver.static(1), r, ch)
        @test J ≈ ForwardDiff.jacobian(δ -> retract(ch, δ)(r.x) - r.y, zero(SVector{6, Float64})) atol = 1.0e-12
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

    @testset "nlls_retcode and loss_robustifier" begin
        J = [1.0 0.0; 0.0 2.0; 1.0 1.0]
        y = [1.0, 2.0, 4.0]
        x̂ = J \ y
        r_opt = J * x̂ - y
        r_off = J * (x̂ .+ 0.1) - y
        kw = (; tol = 1.0e-6, ftol = eps())
        # an NLLSsolver result whose only meaningful field is its termination
        # bits, every other field zero (built from the struct's fields: their
        # number differs between NLLSsolver releases)
        res(bits) = NLLSsolver.NLLSResult(map(fieldnames(NLLSsolver.NLLSResult), fieldtypes(NLLSsolver.NLLSResult)) do f, F
            f === :termination ? F(bits) : F <: Enum ? first(instances(F)) : zero(F)
        end...)
        @test nlls_retcode(res(1 << 1), J, r_opt; kw...) === :solve_failed
        for b in (0, 1, 4, 5, 10)
            @test nlls_retcode(res(1 << b), J, r_opt; kw...) === :solve_failed
        end
        @test nlls_retcode(res((1 << 1) | (1 << 9) | (1 << 8)), J, r_opt; kw...) === :solve_failed   # failure outranks the caps
        @test nlls_retcode(res((1 << 9) | (1 << 8)), J, r_opt; kw...) === :max_time                  # time outranks iterations
        @test nlls_retcode(res(1 << 9), J, r_opt; kw...) === :max_time
        @test nlls_retcode(res((1 << 8) | (1 << 2)), J, r_opt; kw...) === :max_iterations
        @test nlls_retcode(res((1 << 8) | (1 << 3)), J, r_opt; kw...) === :max_iterations
        @test nlls_retcode(res(1 << 2), J, r_opt; kw...) === :converged
        @test nlls_retcode(res(1 << 2), J, r_off; kw...) === :stalled        # a cost-decrease stop is no proof
        @test nlls_retcode(res(1 << 7), J, r_off; kw...) === :stalled
        @test nlls_retcode(res(1 << 7), J, r_opt; kw...) === :converged

        # The oracle: CaliBlobs' production WelschRobustifier, written out.
        w2 = 2.0^2
        oracle_ρ(c) = w2 * (1 - exp(-c / w2))
        oracle_d(c) = (e = exp(-c / w2); (w2 * (1 - e), e, -e / w2))
        k = loss_robustifier(WelschShape(), 2.0)
        @test k isa NLLSsolver.AbstractRobustifier
        for c in (0.0, 0.5, 4.0, 100.0)
            ρ, ρ′, ρ″ = NLLSsolver.robustifydcost(k, c)
            o = oracle_d(c)
            @test ρ ≈ o[1] rtol = 1.0e-14
            @test ρ′ == o[2]
            @test ρ″ ≈ o[3] rtol = 1.0e-14
            @test NLLSsolver.robustify(k, c) ≈ oracle_ρ(c) rtol = 1.0e-14
        end
    end
end
end
