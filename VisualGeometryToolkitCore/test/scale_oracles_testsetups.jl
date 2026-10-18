# Shared test fixtures of the robust-statistics toolkit's tests. ReTestItems
# discovers `@testsetup` modules in `*_testsetups.jl` files; an item takes one
# with `setup = [Name]`.

# Independent oracles for the scale estimators, and Student-t residuals.
@testsetup module ScaleOracles
    using Random
    using Distributions: Gamma, InverseGamma, MvTDist, MvNormal, logpdf
    using LinearAlgebra: I
    using FiniteDiff: FiniteDiff
    export map_scale_oracle, t_sq_residuals, student_t_mixture_oracle

    """
        map_scale_oracle(prior, rss, dof) -> ŝ²

    The inverse-gamma MAP squared scale, found numerically rather than from a
    closed form: `dof` Gaussian components with residual sum of squares `rss`
    under the prior `IG(α₀, β₀)` on ŝ² have posterior `IG(α₀ + dof/2, β₀ + rss/2)`
    (Distributions' density), whose mode in the scale-invariant coordinate
    `t = log ŝ²` is maximised by golden section.
    """
    function map_scale_oracle(prior, rss, dof)
        post = InverseGamma(prior.alpha0 + dof / 2, prior.beta0 + rss / 2)
        f(t) = logpdf(post, exp(t)) + t                  # density of t = log ŝ²
        lo, hi = -700.0, 700.0
        g = (sqrt(5) - 1) / 2
        a = hi - g * (hi - lo); b = lo + g * (hi - lo); fa = f(a); fb = f(b)
        while hi - lo > 1e-13 * max(1.0, abs(lo))
            if fa < fb
                lo = a; a = b; fa = fb; b = lo + g * (hi - lo); fb = f(b)
            else
                hi = b; b = a; fb = fa; a = hi - g * (hi - lo); fa = f(a)
            end
        end
        return exp((lo + hi) / 2)
    end

    """
        student_t_mixture_oracle(sq_r, ν, s2, log_two_outlier_halfwidth, dg, πin) -> (; p, w, nll)

    The Student-t + uniform mixture at a residual of squared norm `sq_r`, from its
    definition in probability space: the inlier density is Distributions'
    `dg`-variate Student-t (ν = Inf: its Gaussian) at scale `s2·I`, the outlier
    `(2a)^-dg`. `p` the inlier responsibility, `w = p·(−2 ∂ log t/∂sq_r)` the
    IRLS weight (the derivative by finite difference), `nll` the mixture's
    negative log-density.
    """
    function student_t_mixture_oracle(sq_r, ν, s2, log_two_outlier_halfwidth, dg, πin)
        Σ = s2 * Matrix(1.0I, dg, dg)
        D = isinf(ν) ? MvNormal(zeros(dg), Σ) : MvTDist(ν, zeros(dg), Σ)
        logt(x) = logpdf(D, [sqrt(x); zeros(dg - 1)])
        t = exp(logt(sq_r))
        pout = exp(-dg * log_two_outlier_halfwidth)
        p = πin * t / (πin * t + (1 - πin) * pout)
        h = 1e-6 * (s2 + sq_r)
        # central where sq_r − h stays in the domain, forward at sq_r = 0; the step's
        # floor is h, on logt's scale s2 (FiniteDiff's default floor is absolute)
        dlogt = FiniteDiff.finite_difference_derivative(logt, sq_r, sq_r > 0 ? Val(:central) : Val(:forward); absstep = h)
        return (p = p, w = -2 * p * dlogt, nll = -log(πin * t + (1 - πin) * pout))
    end

    """
        t_sq_residuals(rng, n, d, ν, s) -> Vector{Float64}

    Squared norms of `n` `d`-variate Student-t residuals with `ν` dof and scale
    `s`: `s²·‖g‖²/(χ²_ν/ν)`.
    """
    t_sq_residuals(rng::AbstractRNG, n, d, ν, s) =
        [s^2 * sum(abs2, randn(rng, d)) / rand(rng, Gamma(ν / 2, 2 / ν)) for _ in 1:n]
end
