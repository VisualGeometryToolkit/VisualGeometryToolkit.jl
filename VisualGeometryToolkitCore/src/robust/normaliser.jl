# =============================================================================
# The truncated normaliser of a loss shape (moved from RANSACScoringDoneRight):
# the one quadrature behind every density built from a loss.
# =============================================================================

"""
    truncated_log_normaliser(shape, sq_loss_width, depth, r_max) -> Float64

`log Z̃ = log ∫_{−r_max}^{r_max} exp(−depth·ρ(shape, r²/sq_loss_width, 1)) dr`
— the normaliser of the density a loss shape induces on the truncated domain
`[−r_max, r_max]`, as `log 2∫₀^{r_max}` (the integrand is even), by adaptive
Gauss–Kronrod (QuadGK) at rtol 1e-12. The integrand is a well of width
`c = √sq_loss_width` on a flat floor, so for `c ≪ r_max` an unsplit rule never
samples the well: the domain is split at `c·4ᵏ < r_max`, which holds the result
to ≲1e-12 relative over every width and depth in use.
"""
function truncated_log_normaliser(shape::AbstractLossShape, sq_loss_width::Float64, depth::Float64, r_max::Float64)
    c = sqrt(sq_loss_width)
    knots = [0.0; [c * 4.0^k for k in 0:60 if c * 4.0^k < r_max]; r_max]
    Z, _ = quadgk(
        r -> exp(-depth * ρ(shape, r * r / sq_loss_width, 1.0)), knots...;
        rtol = 1.0e-12
    )
    return log(2.0 * Z)
end
