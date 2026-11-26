# ReweightedLeastSquares on a rotation — the robust rotation between direction pairs: exact on clean
# pairs, and with 20% gross outliers recovered, at the inliers' GIVEN scale, to
# the noise's accuracy where the plain Kabsch is pulled degrees away.

@testitem "rotation fit: exact on clean pairs" begin
    import VisualGeometryToolkitCore as VgtkCore
    
    using StaticArrays: SVector
    using LinearAlgebra: normalize
    using Rotations: RotMatrix, RotXYZ, rotation_angle
    using VisualGeometryToolkitCore: Correspondences, HuberShape, WelschShape, ClassDefault, EstimationProblem, solve
    using VisualGeometryToolkitEstimation: ReweightedLeastSquares
    include(joinpath(pkgdir(VisualGeometryToolkitEstimation), "test", "fixtures", "reweighted_fits.jl"))
    using VisualGeometryToolkitCore: Irls
    R = RotMatrix(RotXYZ(0.4, -1.1, 2.3))
    as = [normalize(SVector(sin(3k), cos(5k), sin(7k + 1))) for k in 1:30]
    cs = Correspondences(as, [R * a for a in as])
    for K in (HuberShape, WelschShape)
        res = rotation_fit(cs; optimization_strategy = Irls{K}(), scale = 1.0e-3)
        @test (res.retcode, res.stop_rule) === (VgtkCore.ReturnCode.Success, VgtkCore.IncrementSize)
        @test rotation_angle(res.u * R') < 1.0e-14
        @test res.rms < 1.0e-14
    end
    # directions of any length are directions
    @test rotation_angle((rotation_fit(Correspondences(3 .* as, [R * a for a in as]); scale = 1.0e-3).u) * R') < 1.0e-14
    @test rotation_angle((rotation_fit(cs; optimization_strategy = nothing).u) * R') < 1.0e-14
    @test_throws ArgumentError rotation_fit(Correspondences(as[1:1], as[1:1]); scale = 1.0e-3)
end

@testitem "rotation fit: 20% outliers" begin
    import VisualGeometryToolkitCore as VgtkCore
    
    using StaticArrays: SVector
    using LinearAlgebra: normalize, norm, cross
    using Rotations: RotMatrix, RotXYZ, rotation_angle
    using StableRNGs: StableRNG
    using VisualGeometryToolkitCore: Correspondences, withattrs, HuberShape, WelschShape, TLSShape,
        weight, angle_between, efficiency_constant, ClassDefault, EstimationProblem, solve, Kabsch, weights
    using VisualGeometryToolkitEstimation: ReweightedLeastSquares
    include(joinpath(pkgdir(VisualGeometryToolkitEstimation), "test", "fixtures", "reweighted_fits.jl"))
    using VisualGeometryToolkitCore: Irls
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
    cs = Correspondences(as, bs)
    plain = rotation_angle(solve(EstimationProblem(RotMatrix{3}, cs), Kabsch()).u * R')
    @test plain > 10σ                                              # the outliers pull the least-squares rotation away
    # The documented tolerance: within 3σ/√n_inliers·√2 ≈ the noise floor of
    # the inlier fit, here bounded by 5e-4 rad (σ = 1e-3, 160 inliers).
    for K in (HuberShape, WelschShape)
        res = rotation_fit(cs; optimization_strategy = Irls{K}(), scale = σ)
        @test (res.retcode, res.stop_rule) === (VgtkCore.ReturnCode.Success, VgtkCore.IncrementSize)
        @test rotation_angle(res.u * R') < 5.0e-4
        @test rotation_angle(res.u * R') < plain / 10
    end
    # the outliers end down-weighted (Welsch's weights vanish on them)
    resw = rotation_fit(cs; optimization_strategy = Irls{WelschShape}(), scale = σ)
    w = weights(resw)
    @test maximum(w[out]) < 1.0e-3 && minimum(w[setdiff(1:n, out)]) > 0.3
    @test resw.metadata.loss_width ≈ efficiency_constant(WelschShape, 2) * σ
    # prior weights multiply: a zeroed outlier set gives the inlier answer
    prior = ones(n); prior[out] .= 0
    resp = rotation_fit(withattrs(cs; weights = prior); optimization_strategy = Irls{HuberShape}(), scale = σ)
    @test rotation_angle(resp.u * R') < 5.0e-4
    @test resp.n == n - length(out)
    # RC's width (Huber at 3 × 1.4826 × median of the angles), as an absolute width
    @test rotation_angle((rotation_fit(cs; width = 3 * 1.4826 * 1.1774 * σ).u) * R') < 1.0e-3
    # the metadata is at the RETURNED rotation, also when the fit stops at max_iter: the residuals and the weights
    # recomputed from the estimate agree with the result's, at the one fixed width
    for k in (1, 2, 50)
        res = rotation_fit(cs; optimization_strategy = Irls{WelschShape}(max_iter = k), scale = σ)
        Rr = res.u
        r = [angle_between(Rr * a, b) for (a, b) in zip(as, bs)]
        c² = (efficiency_constant(WelschShape, 2) * σ)^2
        @test res.metadata.residuals ≈ r rtol = 1.0e-9                  # (the fit normalizes the directions again: ulps)
        @test weights(res) ≈ [weight(WelschShape(), x^2, c²) for x in r] rtol = 1.0e-9
    end
    # a loss that is no M-estimator (TLS, a hard cutoff): no efficiency default, so its constant (or a width) is
    # required — an ArgumentError naming both. A hard cutoff at a FIXED width must contain the least-squares seed's
    # residuals (here 0.05 rad off): at 3σ it rejects every pair (an unsolvable weighted solve), at 100σ it keeps the inliers and drops
    # every random direction
    e = try rotation_fit(cs; optimization_strategy = Irls{TLSShape}(), scale = σ); nothing catch e; e end
    @test e isa ArgumentError && occursin("width", e.msg) && occursin("TLSShape", e.msg)
    # (a weighted solve left with no row inside its loop: SciMLBase's InternalLinearSolveFailed, `:solve_failed`)
    @test (let r = rotation_fit(cs; optimization_strategy = Irls{TLSShape}(efficiency_constant = 3.0), scale = σ); (r.retcode, r.stop_rule) end) === (VgtkCore.ReturnCode.InternalLinearSolveFailed, VgtkCore.NoRule)
    rest = rotation_fit(cs; optimization_strategy = Irls{TLSShape}(efficiency_constant = 100.0), scale = σ)
    @test (rest.retcode, rest.stop_rule) === (VgtkCore.ReturnCode.Success, VgtkCore.IncrementSize) && rotation_angle(rest.u * R') < 5.0e-4
    @test all(iszero, weights(rest)[out])
end
