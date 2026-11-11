# =============================================================================
# Triggs = NLLSsolver's robustified Hessian, where ρ″ ≤ 0 (owner ruling)
# =============================================================================
#
# With s = ‖r‖², D = 1 + 2sρ″/ρ′ (Triggs et al., "Bundle Adjustment — A Modern
# Synthesis", §4.3). A Triggs refine is NLLSsolver.jl's robustified LM as it
# stands, and the stack hand-codes no robustifier Hessian of its own: the datum
# Hessian is NLLSsolver's (residual.jl), H = ρ′JᵀJ + max(2ρ″, −ρ′/(3s))·ggᵀ,
# g = Jᵀr — in Triggs' α form, D floored at 2/3 (a correction weight of 1/3).
# (Ceres instead drops the correction wherever ρ″ ≤ 0; the owner chose
# NLLSsolver's clamp.) These items PIN NLLSsolver's behaviour at the two points
# where D ≤ 0 — Huber's linear region (D = 0 exactly) and a redescending
# shape's tail (Welsch, Geman–McClure past s = c²/2, D < 0) — so a change
# upstream shows here instead of as a silent move of every Triggs result.

@testitem "triggs_correction" begin
    using VisualGeometryToolkitCore: loss_robustifier, HuberShape, WelschShape, GemanMcclureShape
    using NLLSsolver
    using StaticArrays
    using LinearAlgebra

    D(k, s) = ((_, ρ1, ρ2) = NLLSsolver.robustifydcost(k, s); 1 + 2s * ρ2 / ρ1)

    # One datum r(x) = A·x − y, so J = A and the datum Hessian is NLLSsolver's own.
    struct _Datum{K} <: NLLSsolver.AbstractResidual
        A::SMatrix{2, 2, Float64, 4}
        y::SVector{2, Float64}
        kernel::K
    end
    NLLSsolver.ndeps(::_Datum) = NLLSsolver.static(1)
    NLLSsolver.nres(::_Datum) = NLLSsolver.static(2)
    NLLSsolver.varindices(::_Datum) = SVector(1)
    NLLSsolver.getvars(::_Datum, vars::Vector) = (vars[1]::SVector{2, Float64},)
    NLLSsolver.robustkernel(d::_Datum) = d.kernel
    Base.eltype(::_Datum) = Float64
    NLLSsolver.computeresidual(d::_Datum, x) = d.A * x - d.y
    NLLSsolver.computeresjac(_flags, d::_Datum, x) = (d.A * x - d.y, d.A)

    function datum_hessian(k, s)
        A = @SMatrix [2.0 0.3; -0.4 1.5]
        r = SVector(sqrt(s), 0.0)                   # ‖r‖² = s at x = 0
        d = _Datum(A, -r, k)
        _, g, H = NLLSsolver.computerescostgradhess(NLLSsolver.static(2), d, k, (zero(SVector{2, Float64}),))
        return A, r, g, H
    end
    # H = ρ′·(AᵀA − κ·ĝĝᵀ s) with ĝ = Aᵀr/s: κ is the correction's weight on ggᵀ/(ρ′s).
    function correction_weight(k, s)
        A, r, g, H = datum_hessian(k, s)
        _, ρ1, _ = NLLSsolver.robustifydcost(k, s)
        u = A' * r
        return dot(u, (ρ1 * A' * A - H) * u) / (ρ1 * dot(u, u)^2 / s)
    end

    @testset "Huber's linear region: D = 0 exactly" begin
        k = loss_robustifier(HuberShape(), 1.0)
        for s in (1.5, 4.0, 25.0)
            @test D(k, s) ≈ 0 atol = 1.0e-12
            # NLLSsolver clamps the correction at −ρ′/(3s) (D floored at 2/3)
            @test correction_weight(k, s) ≈ 1 / 3 rtol = 1.0e-10
        end
        # the quadratic region: ρ″ = 0, D = 1, no correction
        @test D(k, 0.5) == 1
        @test correction_weight(k, 0.5) ≈ 0 atol = 1.0e-12
    end

    @testset "$(nameof(typeof(shape))): D < 0 past s = c²/2" for shape in (WelschShape(), GemanMcclureShape())
        k = loss_robustifier(shape, 1.0)
        @test D(k, 0.1) > 0
        for s in (0.75, 2.0, 6.0)
            @test D(k, s) < 0
            @test correction_weight(k, s) ≈ 1 / 3 rtol = 1.0e-10
        end
    end
end
