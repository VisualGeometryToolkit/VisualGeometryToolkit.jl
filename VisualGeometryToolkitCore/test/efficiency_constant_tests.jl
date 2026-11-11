# =============================================================================
# efficiency_constant's table, re-derived: the provenance of the solved
# constants lives here, not in a scratch script.
# =============================================================================
#
# The efficiency of a norm-based M-estimator of a d-dimensional location under
# N(0, I_d), r = ‖x − θ‖ ~ χ_d, from the estimating equation Σ ψ(rᵢ)·uᵢ = 0:
#
#     eff(c, d) = (E[ψ′(r)] + (d − 1)·E[ψ(r)/r])² / (d · E[ψ(r)²])
#
# (least squares, ψ = r, gives 1; at d = 1 it is the classical (E ψ′)²/E ψ²).
# Each shape's constant solves eff(c, d) = 0.95 by bisection on adaptive
# Gauss–Kronrod quadrature, with a breakpoint at c so Huber's kink is not
# integrated across (without it Huber's d = 1 comes out 1.3371, not 1.345).
# ψ is the stack's own, r·weight(shape, r², c²), so a change of a shape's
# parameterisation fails here.

@testitem "efficiency_constant" begin
    using VisualGeometryToolkitCore: efficiency_constant, ψ,
        HuberShape, CauchyShape, WelschShape, GemanMcclureShape
    using QuadGK: quadgk
    using Distributions: Chi, pdf
    using ForwardDiff: derivative

    function efficiency(shape, c, d)
        c2 = c^2
        f(r) = pdf(Chi(d), r)
        E(g) = first(quadgk(r -> g(r) * f(r), 0.0, c, Inf; rtol = 1.0e-12, order = 15))
        dψ = E(r -> derivative(t -> ψ(shape, t, c2), r))
        ψr = d == 1 ? 0.0 : E(r -> ψ(shape, r, c2) / r)
        ψ2 = E(r -> ψ(shape, r, c2)^2)
        return (dψ + (d - 1) * ψr)^2 / (d * ψ2)
    end
    function solve_constant(shape, d; target = 0.95)
        lo, hi = 0.5, 10.0
        for _ in 1:50
            mid = (lo + hi) / 2
            efficiency(shape, mid, d) < target ? (lo = mid) : (hi = mid)
        end
        return (lo + hi) / 2
    end

    @testset "$(nameof(typeof(shape))): eff(c, d) = 0.95 to 4 decimals, d = 1…3" for shape in
            (HuberShape(), CauchyShape(), WelschShape(), GemanMcclureShape())
        for d in 1:3
            @test round(solve_constant(shape, d); digits = 4) == efficiency_constant(typeof(shape), d)
        end
    end
end
