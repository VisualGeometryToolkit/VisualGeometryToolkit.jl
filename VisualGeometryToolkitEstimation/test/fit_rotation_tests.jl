# fit_rotation — the robust rotation between direction pairs: exact on clean
# pairs, and with 20% gross outliers recovered to the noise's accuracy where
# the plain Kabsch is pulled degrees away.

@testitem "fit_rotation: exact on clean pairs" begin
    using StaticArrays: SVector
    using LinearAlgebra: normalize
    using Rotations: RotMatrix, RotXYZ, rotation_angle
    using VisualGeometryToolkitCore: Csponds, estimate, retcode, rms, HuberShape, WelschShape
    using VisualGeometryToolkitEstimation: fit_rotation
    R = RotMatrix(RotXYZ(0.4, -1.1, 2.3))
    as = [normalize(SVector(sin(3k), cos(5k), sin(7k + 1))) for k in 1:30]
    cs = Csponds(as, [R * a for a in as])
    for loss in (HuberShape(), WelschShape())
        res = fit_rotation(cs; loss)
        @test retcode(res) === :converged
        @test rotation_angle(estimate(res) * R') < 1.0e-14
        @test rms(res) < 1.0e-14
    end
    # directions of any length are directions
    @test rotation_angle(estimate(fit_rotation(Csponds(3 .* as, [R * a for a in as]))) * R') < 1.0e-14
    @test_throws ArgumentError fit_rotation(Csponds(as[1:1], as[1:1]))
end

@testitem "fit_rotation: 20% outliers" begin
    using StaticArrays: SVector
    using LinearAlgebra: normalize, norm, cross
    using Rotations: RotMatrix, RotXYZ, rotation_angle
    using StableRNGs: StableRNG
    using VisualGeometryToolkitCore: Csponds, withattrs, estimate, retcode, kabsch, HuberShape, WelschShape, TLSShape
    using VisualGeometryToolkitEstimation: fit_rotation
    rng = StableRNG(7)
    R = RotMatrix(RotXYZ(0.2, 0.3, -0.5))
    n = 200
    σ = 1.0e-3                                                     # rad of direction noise
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
    for loss in (HuberShape(), WelschShape())
        res = fit_rotation(cs; loss)
        @test retcode(res) === :converged
        @test rotation_angle(estimate(res) * R') < 5.0e-4
        @test rotation_angle(estimate(res) * R') < plain / 10
    end
    # the outliers end down-weighted (Welsch's weights vanish on them)
    resw = fit_rotation(cs; loss = WelschShape())
    w = resw.weights
    @test maximum(w[out]) < 1.0e-3 && minimum(w[setdiff(1:n, out)]) > 0.3
    @test resw.sigma < 2σ
    # prior weights multiply: a zeroed outlier set gives the inlier answer
    prior = ones(n); prior[out] .= 0
    resp = fit_rotation(withattrs(cs; weights = prior); loss = HuberShape())
    @test rotation_angle(estimate(resp) * R') < 5.0e-4
    # RC's width (Huber at 3 × 1.4826 × median-of-angles) as a scale_multiple
    @test rotation_angle(estimate(fit_rotation(cs; loss = HuberShape(), scale_multiple = 3 * 1.4826 * 1.1774)) * R') < 1.0e-3
    # the metadata is at the RETURNED rotation, also when the fit stops at max_iter: the residuals, σ̂ and the weights
    # recomputed from the estimate agree with the result's (they came from the previous iterate before)
    using VisualGeometryToolkitCore: chisq_median, deploy, sq_loss_width, weight, angle_between
    for k in (1, 2, 50)
        res = fit_rotation(cs; loss = WelschShape(), max_iter = k)
        Rr = estimate(res)
        r = [angle_between(Rr * a, b) for (a, b) in zip(as, bs)]
        σr = sum(sort(r)[n ÷ 2:(n ÷ 2 + 1)]) / 2 / sqrt(chisq_median(2))
        c² = sq_loss_width(deploy(WelschShape(), σr^2; codim = 2))
        @test res.residuals ≈ r rtol = 1.0e-9                  # (the fit normalizes the directions again: ulps)
        @test res.sigma ≈ σr rtol = 1.0e-9
        @test res.weights ≈ [weight(WelschShape(), x^2, c²) for x in r] rtol = 1.0e-9
    end
    # a loss that is no M-estimator (TLS, a hard cutoff): no efficiency default, so its width multiple is required —
    # an ArgumentError naming `scale_multiple` without one; with one, the fit runs and rejects the outliers
    e = try fit_rotation(cs; loss = TLSShape()); nothing catch e; e end
    @test e isa ArgumentError && occursin("scale_multiple", e.msg) && occursin("TLSShape", e.msg)
    rest = fit_rotation(cs; loss = TLSShape(), scale_multiple = 3.0)
    @test retcode(rest) === :converged && rotation_angle(estimate(rest) * R') < 5.0e-4
    @test all(iszero, rest.weights[out])
end
