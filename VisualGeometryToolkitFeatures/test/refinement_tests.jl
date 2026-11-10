# ASIFT's sub-pixel refinement (src/asift/driver.jl `_extract_at!`): every detection returned is refined, with a finite
# covariance; an extremum whose refinement fails is dropped (counted in the plan), never returned on the octave's grid.

@testitem "asift: every detection refined, finite covariance; a failed refinement dropped and counted" begin
    using VisualGeometryToolkitFeatures: asift_detect, asift_detect!, AsiftPlan, FeatureCovariance, asift_detections
    using VisualGeometryToolkitCore: cov
    import KernelAbstractions as KA
    using Random: Xoshiro
    # a textured scene: blobs of many scales and contrasts on smooth noise
    rng = Xoshiro(9)
    H, W = 384, 512
    img = fill(0.5f0, H, W)
    for _ in 1:120
        cx, cy, σ, a = W * rand(rng), H * rand(rng), 1.5 + 14rand(rng), 0.4f0 * (rand(rng) - 0.5f0)
        r = ceil(Int, 4σ)
        for j in max(1, floor(Int, cx) - r):min(W, floor(Int, cx) + r), i in max(1, floor(Int, cy) - r):min(H, floor(Int, cy) + r)
            img[i, j] += a * exp(-Float32((j - 0.5 - cx)^2 + (i - 0.5 - cy)^2) / Float32(2σ^2))
        end
    end
    img .+= 0.01f0 .* randn(rng, Float32, H, W)
    ds = asift_detect(img; device = KA.CPU(), uncertainty = FeatureCovariance(), intensity_thr = 0.01)
    @test length(ds) > 200
    @test all(d -> all(isfinite, cov(d)), ds)
    # the plan counts the extrema its refinement dropped
    plan = AsiftPlan(W, H; device = KA.CPU())
    r = asift_detect!(plan, permutedims(img); nms = :none, intensity_thr = 0.01, uncertainty = FeatureCovariance())
    @test all(c -> all(isfinite, c), r.cov)
    @test plan.refine_dropped[] > 0
end

@testitem "asift: sub-pixel localization of a blob across scales, offsets and tilts" begin
    using VisualGeometryToolkitFeatures: asift_detect, FeatureCovariance
    using VisualGeometryToolkitCore: cov, centroid, characteristic_scale, feature_frame
    import KernelAbstractions as KA
    H = W = 160
    worst = Dict{Tuple{Float64, Bool}, Float64}()
    for σ in (2.0, 3.0, 5.0, 8.0), fx in (0.0, 0.17, 0.33, 0.5), fy in (0.0, 0.29, 0.5), tilted in (false, true)
        cx, cy = 80 + fx, 80 + fy                          # COLMAP: pixel (i, j) centred at (j − ½, i − ½)
        img = Float32[0.5 - 0.4exp(-((j - 0.5 - cx)^2 + (i - 0.5 - cy)^2) / (2σ^2)) for i in 1:H, j in 1:W]
        ds = asift_detect(img; device = KA.CPU(), uncertainty = FeatureCovariance(),
                          tilts = tilted ? [(1.0, 0.0), (2.62, 0.0), (2.62, π / 2)] : [(1.0, 0.0)])
        @test all(d -> all(isfinite, cov(d)), ds)
        # the detection at the blob: nearest the centre among those of about its scale
        near = filter(d -> 0.5σ < characteristic_scale(feature_frame(d)) < 2σ, ds)
        @test !isempty(near)
        e = minimum(d -> hypot(centroid(d)[1] - cx, centroid(d)[2] - cy), near)
        worst[(σ, tilted)] = max(get(worst, (σ, tilted), 0.0), e)
    end
    @test all(e -> e < 0.1, values(worst))
end

@testitem "AsiftPlan: an out-of-memory error halves the resident warps; any other error is raised" begin
    import VisualGeometryToolkitFeatures as VF
    import KernelAbstractions as KA
    @test VF._out_of_memory(OutOfMemoryError()) && !VF._out_of_memory(ArgumentError("x"))
    # an explicit chunk is honoured: the buffers of 3 warps, as asked
    r = VF._pyramid_buffers(KA.CPU(), 64, 48, 70, 60, [(128, 96), (64, 48)], 3, 3, true)
    @test r[5] == 3 && size(r[2]) == (64, 48, 3) && size(r[3][1]) == (128, 96, 6, 3)
end
