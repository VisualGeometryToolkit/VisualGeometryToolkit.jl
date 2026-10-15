# `next_nu`, the ν M-step: Newton on Q(ν) = Σ wᵢ log t_d(rᵢ; ν, ŝ²).
#
# Its curvature's data term was −z(ν+d+1+z)/(2ν²(1+z)²), the right derivative
# only at d = 1: the derivative of its own gradient is −z(ν+2d+d·z)/(2ν²(1+z)²).
# At d = 2 the old form had the wrong sign where the truth is strongly concave,
# so Newton climbed away from the maximum. Q is also genuinely CONVEX in ν away
# from the optimum (t₄ data, ν ≳ 6, every d), where a Newton step is downhill;
# there the update steps geometrically along the gradient instead.
#
# Pinned against Distributions' d-variate Student-t log-density (independent of
# the package): the gradient and curvature by finite differences, at d = 1, 2, 6.

@testitem "next_nu" setup = [ScaleOracles] begin
    using Test, Random
    using SpecialFunctions: trigamma
    using Distributions: MvTDist, logpdf
    using LinearAlgebra: I
    using FiniteDiff: FiniteDiff
    using VisualGeometryCore
    using VisualGeometryCore: next_nu, _nu_derivatives
    const _t_sq = t_sq_residuals

    # Q(ν) = Σ pᵢ log t_d(rᵢ; ν, ŝ²·I): the density depends on rᵢ through ‖rᵢ‖²
    # only, so each squared norm is placed on the first axis.
    function _Q(ν, sq, s2, p, d)
        D = MvTDist(ν, zeros(d), s2 * Matrix(1.0I, d, d)); r = zeros(d)
        return sum(eachindex(sq)) do i
            r[1] = sqrt(sq[i])
            p[i] * logpdf(D, r)
        end
    end

    @testset "next_nu: the d-variate curvature and an ascent safeguard" begin
        @testset "gradient and curvature == finite differences of Q (d = $d)" for d in (1, 2, 6)
            rng = MersenneTwister(7 + d)
            n = 2000
            sq = _t_sq(rng, n, d, 4.0, 1.3)
            p = 0.5 .+ 0.5 .* rand(rng, n)                  # E-step weights, fixed
            s2 = 1.3^2
            for ν in (3.0, 4.0, 10.0, 50.0)
                dQ, d2Q, sum_w = _nu_derivatives(ν, sq, s2, p, n, d)
                @test sum_w ≈ sum(p)
                @test dQ ≈ FiniteDiff.finite_difference_derivative(ν′ -> _Q(ν′, sq, s2, p, d), ν) rtol = 1e-6
                # Q is a 2000-term sum, so rounding (≈ eps·|Q|/h²) dominates at
                # the Hessian's default step eps^¼ ≈ 1.2e-4: measured max relative
                # error over these 12 (d, ν), on the closed-form Q with t₄ draws,
                # 6.8e-5. relstep = 1e-3 balances it against the O(h²) truncation:
                # 1.1e-5 (3e-4: 1.6e-5, 3e-3: 9.1e-5), equal to the hand second
                # difference at H = 1e-3·ν it replaces. A nested central
                # derivative is worse here (up to 2.4e-2).
                fd2 = FiniteDiff.finite_difference_hessian(x -> _Q(x[1], sq, s2, p, d), [ν]; relstep = 1e-3)[1]
                @test d2Q ≈ fd2 rtol = 1e-4 atol = 1e-6 * abs(d2Q) + 1e-8
            end
        end

        @testset "d = 2, t₄: concave at the truth; the old term was not" begin
            rng = MersenneTwister(3)
            n = 2000; d = 2
            sq = _t_sq(rng, n, d, 4.0, 1.0); p = ones(n)
            _, d2Q, _ = _nu_derivatives(4.0, sq, 1.0, p, n, d)
            @test d2Q < 0
            # the retired data term, at the same point
            old = sum(sq) do r2
                z = r2 / 4.0
                0.5z / (4.0 * (1 + z)) - 0.5z * (4.0 + d + 1 + z) / (1 + z)^2 / 16.0
            end + n * (0.25 * (trigamma(3.0) - trigamma(2.0)) + 0.5 * d / 16.0)
            @test old > 0 > d2Q                              # Newton would have climbed away
        end

        @testset "a convex point steps uphill, geometrically (d = $d)" for d in (1, 2, 6)
            rng = MersenneTwister(11 + d)
            n = 5000
            sq = _t_sq(rng, n, d, 4.0, 1.3); p = ones(n); s2 = 1.3^2
            dQ, d2Q, _ = _nu_derivatives(10.0, sq, s2, p, n, d)
            @test d2Q > 0 && dQ < 0                          # convex, optimum below
            ν1 = next_nu(10.0, sq, s2, p, n; d_g = d, max_newton = 1)
            @test ν1 == 5.0                                   # ν/2, not the downhill Newton step
            @test _Q(ν1, sq, s2, p, d) > _Q(10.0, sq, s2, p, d)
        end

        @testset "Q never decreases; ν is recovered (d = $d)" for d in (1, 2, 6)
            rng = MersenneTwister(11 + d)
            n = 20000
            sq = _t_sq(rng, n, d, 4.0, 1.3); p = ones(n); s2 = 1.3^2
            for ν0 in exp.(range(log(2.0), log(1e4); length = 40))
                ν1 = next_nu(ν0, sq, s2, p, n; d_g = d)
                @test 2.0 <= ν1 <= 1e4
                @test _Q(ν1, sq, s2, p, d) >= _Q(ν0, sq, s2, p, d) - 1e-9 * abs(_Q(ν0, sq, s2, p, d))
            end
            # at the true scale, repeated M-steps settle on the MLE, near ν₀ = 4
            ν = foldl((ν, _) -> next_nu(ν, sq, s2, p, n; d_g = d), 1:20; init = 30.0)
            dQ, d2Q, _ = _nu_derivatives(ν, sq, s2, p, n, d)
            @test abs(dQ / n) < 1e-6 && d2Q < 0              # a maximum
            @test 3.5 < ν < 4.6
        end
    end
end
