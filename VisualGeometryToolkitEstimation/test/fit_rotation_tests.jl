# fit_rotation — the robust rotation between direction pairs: exact on clean
# pairs, and with 20% gross outliers recovered, at the inliers' GIVEN scale, to
# the noise's accuracy where the plain Kabsch is pulled degrees away.

@testitem "fit_rotation: exact on clean pairs" begin
    using StaticArrays: SVector
    using LinearAlgebra: normalize
    using Rotations: RotMatrix, RotXYZ, rotation_angle
    using VisualGeometryToolkitCore: Csponds, estimate, retcode, rms, HuberShape, WelschShape
    using VisualGeometryToolkitEstimation: fit_rotation, Irls
    R = RotMatrix(RotXYZ(0.4, -1.1, 2.3))
    as = [normalize(SVector(sin(3k), cos(5k), sin(7k + 1))) for k in 1:30]
    cs = Csponds(as, [R * a for a in as])
    for K in (HuberShape, WelschShape)
        res = fit_rotation(cs; optimization_strategy = Irls{K}(), scale = 1.0e-3)
        @test retcode(res) === :converged
        @test rotation_angle(estimate(res) * R') < 1.0e-14
        @test rms(res) < 1.0e-14
    end
    # directions of any length are directions
    @test rotation_angle(estimate(fit_rotation(Csponds(3 .* as, [R * a for a in as]); scale = 1.0e-3)) * R') < 1.0e-14
    @test rotation_angle(estimate(fit_rotation(cs; optimization_strategy = nothing)) * R') < 1.0e-14
    @test_throws ArgumentError fit_rotation(Csponds(as[1:1], as[1:1]); scale = 1.0e-3)
end

@testitem "fit_rotation: 20% outliers" begin
    using StaticArrays: SVector
    using LinearAlgebra: normalize, norm, cross
    using Rotations: RotMatrix, RotXYZ, rotation_angle
    using StableRNGs: StableRNG
    using VisualGeometryToolkitCore: Csponds, withattrs, estimate, retcode, kabsch, HuberShape, WelschShape, TLSShape,
        weight, angle_between, efficiency_constant
    using VisualGeometryToolkitEstimation: fit_rotation, Irls
    rng = StableRNG(7)
    R = RotMatrix(RotXYZ(0.2, 0.3, -0.5))
    n = 200
    σ = 1.0e-3                                                     # rad of direction noise per tangent component
    as = [normalize(randn(rng, SVector{3, Float64})) for _ in 1:n]
    function jitter(b)
        t = normalize(cross(b, randn(rng, SVector{3, Float64})))   # a random tangent direction
        return normalize(b + σ * randn(rng) * t + σ * randn(rng) * normalize(cross(b, t)))
    end
    bs = [jitter(R * a) for a in as]
    out = 1:5:n                                                    # 20%: a random direction instead
    for i in out
        bs[i] = normalize(randn(rng, SVector{3, Float64}))
    end
    cs = Csponds(as, bs)
    plain = rotation_angle(kabsch(cs) * R')
    @test plain > 10σ                                              # the outliers pull the least-squares rotation away
    # The documented tolerance: within 3σ/√n_inliers·√2 ≈ the noise floor of
    # the inlier fit, here bounded by 5e-4 rad (σ = 1e-3, 160 inliers).
    for K in (HuberShape, WelschShape)
        res = fit_rotation(cs; optimization_strategy = Irls{K}(), scale = σ)
        @test retcode(res) === :converged
        @test rotation_angle(estimate(res) * R') < 5.0e-4
        @test rotation_angle(estimate(res) * R') < plain / 10
    end
    # the outliers end down-weighted (Welsch's weights vanish on them)
    resw = fit_rotation(cs; optimization_strategy = Irls{WelschShape}(), scale = σ)
    w = resw.weights
    @test maximum(w[out]) < 1.0e-3 && minimum(w[setdiff(1:n, out)]) > 0.3
    @test resw.loss_width ≈ efficiency_constant(WelschShape, 2) * σ
    # prior weights multiply: a zeroed outlier set gives the inlier answer
    prior = ones(n); prior[out] .= 0
    resp = fit_rotation(withattrs(cs; weights = prior); optimization_strategy = Irls{HuberShape}(), scale = σ)
    @test rotation_angle(estimate(resp) * R') < 5.0e-4
    @test resp.n == n - length(out)
    # RC's width (Huber at 3 × 1.4826 × median of the angles), as an absolute width
    @test rotation_angle(estimate(fit_rotation(cs; width = 3 * 1.4826 * 1.1774 * σ)) * R') < 1.0e-3
    # the metadata is at the RETURNED rotation, also when the fit stops at max_iter: the residuals and the weights
    # recomputed from the estimate agree with the result's, at the one fixed width
    for k in (1, 2, 50)
        res = fit_rotation(cs; optimization_strategy = Irls{WelschShape}(max_iter = k), scale = σ)
        Rr = estimate(res)
        r = [angle_between(Rr * a, b) for (a, b) in zip(as, bs)]
        c² = (efficiency_constant(WelschShape, 2) * σ)^2
        @test res.residuals ≈ r rtol = 1.0e-9                  # (the fit normalizes the directions again: ulps)
        @test res.weights ≈ [weight(WelschShape(), x^2, c²) for x in r] rtol = 1.0e-9
    end
    # a loss that is no M-estimator (TLS, a hard cutoff): no efficiency default, so its constant (or a width) is
    # required — an ArgumentError naming both. A hard cutoff at a FIXED width must contain the least-squares seed's
    # residuals (here 0.05 rad off): at 3σ it rejects every pair (:degenerate), at 100σ it keeps the inliers and drops
    # every random direction
    e = try fit_rotation(cs; optimization_strategy = Irls{TLSShape}(), scale = σ); nothing catch e; e end
    @test e isa ArgumentError && occursin("width", e.msg) && occursin("TLSShape", e.msg)
    @test retcode(fit_rotation(cs; optimization_strategy = Irls{TLSShape}(scale_multiple = 3.0), scale = σ)) === :degenerate
    rest = fit_rotation(cs; optimization_strategy = Irls{TLSShape}(scale_multiple = 100.0), scale = σ)
    @test retcode(rest) === :converged && rotation_angle(estimate(rest) * R') < 5.0e-4
    @test all(iszero, rest.weights[out])
end
