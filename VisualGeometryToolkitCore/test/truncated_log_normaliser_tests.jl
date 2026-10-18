# `truncated_log_normaliser(shape, sq_loss_width, depth, r_max)`: the log
# normaliser of the density a loss shape induces at depth Δ on [−r_max, r_max],
# log ∫ exp(−Δ·ρ(shape, r²/w², 1)) dr — the one quadrature behind every density
# built from a loss (RansacScoringDoneRight's scale marginals call it).
#
# Pinned against a densely split reference quadrature in ρ's scaled spelling
# (rtol 1e-14), down to a well far narrower than the domain.

@testitem "truncated_log_normaliser" begin
    using Test
    using QuadGK: quadgk
    import VisualGeometryToolkitCore as VGC
    using VisualGeometryToolkitCore: WelschShape, GemanMcclureShape, SmoothTLSShape, truncated_log_normaliser

    for shape in (WelschShape(), GemanMcclureShape(), SmoothTLSShape()),
        sq_loss_width in (1e-12, 1e-4, 0.01, 1.0), Δ in (0.25, 0.5, 2.0, 8.0), r_max in (1.0, 3.0)
        c = sqrt(sq_loss_width)
        knots = [0.0; [c * 2.0^k for k in -3:60 if c * 2.0^k < r_max]; r_max]
        Z, _ = quadgk(r -> exp(-(Δ / sq_loss_width) * VGC.ρ(shape, r^2, sq_loss_width)), knots...; rtol = 1e-14)
        @test truncated_log_normaliser(shape, sq_loss_width, Δ, r_max) ≈ log(2Z) rtol = 1e-12
    end
end
