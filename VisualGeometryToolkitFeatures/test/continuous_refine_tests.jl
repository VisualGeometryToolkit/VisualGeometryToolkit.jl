# The continuous DoG-extremum refinement (src/asift/continuous_refine.jl; research path, off by default).

@testitem "continuous refinement: Core's N = 3 device LM on the spline response converges, beats today's bias, gates" setup = [UncFixtures] begin
    import VisualGeometryToolkitFeatures as VgtkFeatures
    import VisualGeometryToolkitCore as VgtkCore
    using VisualGeometryToolkitFeatures: AsiftPlan
    using StaticArrays: SVector
    using LinearAlgebra: eigvals, Symmetric
    import KernelAbstractions as KA
    # one blob at a near-half-sample phase of octave 0 (σ_b = 3.36: the jump of today's step is largest there)
    H = W = 256
    cx, cy = 127.45, 128.2                                  # 0-indexed centre
    img = blob_image(H, W, [(cx + 0.5, cy + 0.5, 3.36, 0.3)])
    plan = AsiftPlan(W, H; device = KA.CPU(), tilts = [(1.0, 0.0)], upsample = true)
    ds = detect_on(plan, img)
    d = argmin(d -> hypot(d.center[1] - 0.5 - cx, d.center[2] - 0.5 - cy) + abs(log(d.σ / 3.36)), ds)   # the detection itself
    e = extremum_of(plan, d)
    @test e !== nothing && e.o == 0
    vA = SVector(e.X + e.off[2], e.Y + e.off[3], e.sl + e.off[1])
    cfg = VgtkFeatures._continuous_refinement(:spline)
    ext, wins, _ = VgtkFeatures._refine_dog_extrema(KA.CPU(), plan.dogs[e.oi], [(vA..., 1)]; seed_sigma = plan.seed_sigma, S = plan.nb_scales, cfg)
    r = ext[1]
    @test VgtkCore.successful_retcode(r.retcode)
    truth = SVector(cx + 1, cy + 1)                          # octave 0: level px = image px, 1-based
    errA = hypot(vA[1] - truth[1], vA[2] - truth[2]); errC = hypot(r.v[1] - truth[1], r.v[2] - truth[2])
    @test errC < errA && errC < 2.0e-3
    # the point is a stationary point of the model: its gradient vanishes there
    D, g, _ = VgtkFeatures._dog_response(wins[1], r.v)
    @test maximum(abs, g) < 1.0e-6 * abs(D)
    # a start 2.5 level px off: the extremum lies outside its box (half-width 1), so the solve ends on the box —
    # LeftBasin, a failure, and the start returned as the last valid estimate
    v2 = SVector(vA[1] + 2.5, vA[2], vA[3])
    ext2, _, _ = VgtkFeatures._refine_dog_extrema(KA.CPU(), plan.dogs[e.oi], [(v2..., 1)]; seed_sigma = plan.seed_sigma, S = plan.nb_scales, cfg)
    @test !VgtkCore.successful_retcode(ext2[1].retcode) && ext2[1].stop_rule == VgtkCore.LeftBasin && ext2[1].v == v2
    # the recipe (ContinuousRefinement(): the matched spatial gradient, the scale row from the unfiltered spline):
    # converged, no farther from the truth than A
    extS, winsS, _ = VgtkFeatures._refine_dog_extrema(KA.CPU(), plan.dogs[e.oi], [(vA..., 1)]; seed_sigma = plan.seed_sigma, S = plan.nb_scales, cfg = VgtkFeatures.ContinuousRefinement())
    @test VgtkCore.successful_retcode(extS[1].retcode)
    @test hypot(extS[1].v[1] - truth[1], extS[1].v[2] - truth[2]) < errA
    ΣS = VgtkFeatures._continuous_covariance(plan, e.oi, 1, winsS[1], extS[1], VgtkFeatures.ContinuousRefinement())
    @test all(isfinite, ΣS) && all(>(0), eigvals(Symmetric(Matrix(ΣS))))
    # the window's field count must match the refinement's response
    @test_throws ArgumentError VgtkFeatures._continuous_covariance(plan, e.oi, 1, wins[1], r, VgtkFeatures.ContinuousRefinement())
    # the covariance at C's point is finite and positive definite
    Σ = VgtkFeatures._continuous_covariance(plan, e.oi, 1, wins[1], r, cfg)
    @test all(isfinite, Σ) && all(>(0), eigvals(Symmetric(Matrix(Σ))))
    # the level's extent bounds the solve as the detector's keep test does (1 ≤ x ≤ W): from a start 0.6 px left of
    # the extremum the refinement reaches it inside the level, but with the level's last column at x = 128 (the
    # extremum at 128.45 beyond it) it ends on that edge — LeftBasin, the start returned — never outside the level
    v3 = SVector(vA[1] - 0.6, vA[2], vA[3])
    S = plan.nb_scales
    inside = VgtkFeatures._refine_dog_extremum(wins[1], v3, 3.36, S, cfg, (W, H))
    @test VgtkCore.successful_retcode(inside[4]) && abs(inside[1][1] - truth[1]) < 2.0e-3
    cut = VgtkFeatures._refine_dog_extremum(wins[1], v3, 3.36, S, cfg, (128, H))
    @test cut[5] == VgtkCore.LeftBasin && cut[1] == v3
end

@testitem "continuous refinement: CPU = GPU on one DoG — the same §14 records, points within the derived tolerance" setup = [UncFixtures] begin
    import VisualGeometryToolkitFeatures as VgtkFeatures
    using VisualGeometryToolkitFeatures: AsiftPlan
    import KernelAbstractions as KA
    using CUDA
    if CUDA.functional()
        H, W = 256, 384
        blobs = [(40.0 + 37.3k, 60.0 + 23.9 * (k % 5), 1.6 + 0.9k, iseven(k) ? 0.3 : -0.3) for k in 1:8]
        img = blob_image(H, W, blobs)
        plan = AsiftPlan(W, H; device = CUDA.CUDABackend(), tilts = [(1.0, 0.0)], upsample = true)
        ds = detect_on(plan, img)
        es = filter(!isnothing, [extremum_of(plan, d) for d in ds])
        @test !isempty(es)
        tol = VgtkFeatures._cr("cpu_gpu_tolerance")                 # derived in configs/continuous_refinement.toml
        for response in (:spline, :farid_simoncelli_spatial), oi in unique(e.oi for e in es)
            st = [(e.X + e.off[2], e.Y + e.off[3], e.sl + e.off[1], 1) for e in es if e.oi == oi]
            kw = (; seed_sigma = plan.seed_sigma, S = plan.nb_scales, cfg = VgtkFeatures._continuous_refinement(response))
            eg, _, _ = VgtkFeatures._refine_dog_extrema(CUDA.CUDABackend(), plan.dogs[oi], st; kw...)
            ec, _, _ = VgtkFeatures._refine_dog_extrema(KA.CPU(), Array(plan.dogs[oi]), st; kw...)
            # the same verdict on both backends; the stop RULE may differ between two converged solves at a tie of their
            # stops (measured: 2 of 386 dense-scene detections, both converged), so it is not compared
            @test all(a.retcode == b.retcode for (a, b) in zip(eg, ec))
            @test maximum(maximum(abs, a.v - b.v) for (a, b) in zip(eg, ec)) <= tol
        end
    else
        @test_skip CUDA.functional()                                  # no functional GPU here: recorded as Broken
    end
end

@testitem "asift_detect(…; refinement): opt-in, the same candidates refined, covariances at the refined points" begin
    using VisualGeometryToolkitFeatures: asift_detect, FeatureCovariance, ContinuousRefinement
    using VisualGeometryToolkitCore: cov, centroid, characteristic_scale, feature_frame
    using LinearAlgebra: eigvals, Symmetric
    import KernelAbstractions as KA
    H = W = 192
    for tilts in ([(1.0, 0.0)], [(1.0, 0.0), (2.0, 0.0)])
        worstA = 0.0; worstC = 0.0
        for fx in (0.1, 0.3, 0.45), σ in (2.5, 3.3, 6.0)
            cx, cy = 96 + fx, 96.2                           # COLMAP
            img = Float32[0.5 - 0.3exp(-((j - 0.5 - cx)^2 + (i - 0.5 - cy)^2) / (2σ^2)) for i in 1:H, j in 1:W]
            kw = (; device = KA.CPU(), tilts, uncertainty = FeatureCovariance())
            dA = asift_detect(img; kw...)
            dC = asift_detect(img; kw..., refinement = ContinuousRefinement())
            @test length(dA) == length(dC)                    # the same candidates (a failed refinement keeps today's point)
            @test all(d -> all(isfinite, cov(d)) && all(>(0), eigvals(Symmetric(Matrix(cov(d))))), dC)
            near(ds) = minimum(d -> hypot(centroid(d)[1] - cx, centroid(d)[2] - cy), filter(d -> 0.5σ < characteristic_scale(feature_frame(d)) < 2σ, ds))
            worstA = max(worstA, near(dA)); worstC = max(worstC, near(dC))
        end
        @test worstC < worstA && worstC < 0.01
    end
    # the default is today's detector: no keyword, no change
    img = Float32[0.5 - 0.3exp(-((j - 96.8)^2 + (i - 96.7)^2) / 18) for i in 1:H, j in 1:W]
    d0 = asift_detect(img; device = KA.CPU(), tilts = [(1.0, 0.0)]); d1 = asift_detect(img; device = KA.CPU(), tilts = [(1.0, 0.0)], refinement = nothing)
    @test map(centroid, d0) == map(centroid, d1) && map(d -> characteristic_scale(feature_frame(d)), d0) == map(d -> characteristic_scale(feature_frame(d)), d1)
end
