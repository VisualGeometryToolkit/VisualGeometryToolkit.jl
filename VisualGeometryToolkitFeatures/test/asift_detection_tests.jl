# ASIFT's detection options: a result independent of scheduling and pool order, the detection mask, the untilted
# schedule. Shared fixtures: vgf_testsetup.jl.

@testsetup module TextureFixture
using Colors: Gray
using FixedPointNumbers: N0f8
using Random: Xoshiro
export TEXTURE

# 400 blobs of random size and contrast on a mid-grey field, 8-bit: thousands of extrema over the 25 warps, many of
# them overlapping, with quantized responses (the cross-warp NMS's tie case)
const TEXTURE = let rng = Xoshiro(11), H = 120, W = 176
    img = fill(0.5f0, H, W)
    for _ in 1:400
        cy, cx = H * rand(rng), W * rand(rng)
        s = 1.5f0 + 5 * rand(rng); a = 0.5f0 * (rand(rng) - 0.5f0)
        for j in 1:W, i in 1:H
            img[i, j] += a * exp(-((i - cy)^2 + (j - cx)^2) / (2s^2))
        end
    end
    Gray{N0f8}.(clamp.(img, 0, 1))
end
end

@testitem "ASIFT: the same image, the same keypoints in the same order, under any scheduling and pool order" setup = [Fixtures, TextureFixture] begin
    import VisualGeometryToolkitFeatures as VgtkFeatures
    using VisualGeometryToolkitFeatures: spread_strongest
    using Random: Xoshiro, randperm
    using CUDA: CUDA
    # The pool fills through an atomic counter, so its order is the order the threads reach it: it varies run to run on
    # many threads and on a GPU, and the cross-warp NMS breaks its ties by pool index. The pool is put in canonical order
    # (position, scale, response, then every other field) before the NMS, so the result is a function of the pool's
    # contents alone. On one thread (the suite's workers) or many (an interactive session), the same keypoints come back.
    H, W = size(TEXTURE)
    fields(f) = (f.x, f.y, f.σ, f.intensity, f.s_xx, f.s_xy, f.s_yy, f.warp_idx)
    detect(dev; kw...) = asift_detect(TEXTURE; device = dev, raw = true, uncertainty = FeatureCovariance(), kw...)
    for nms in (:none, :mxm, :ellipse)
        ref = detect(KA.CPU(); nms)
        @test length(ref.features) > 40
        @test issorted(ref.features; by = fields)
        @test isequal(detect(KA.CPU(); nms), ref)
        nested = Vector{Any}(undef, 2)
        Threads.@threads for i in 1:2           # launched dynamically, inside a threaded region
            nested[i] = detect(KA.CPU(); nms)
        end
        @test all(isequal(ref), nested)
        @test all(isequal(ref), fetch.([Threads.@spawn detect(KA.CPU(); nms) for _ in 1:3]))
        if CUDA.functional()
            g = detect(CUDA.CUDABackend(); nms)
            @test issorted(g.features; by = fields)
            @test all(_ -> isequal(detect(CUDA.CUDABackend(); nms), g), 1:3)
        end
    end
    # The NMS decides on the pool's contents, not its order: any permutation of the pool keeps the same survivors, in
    # the same order, and they are exactly what the in-pipeline NMS keeps (no tie left to the order).
    pool = asift_detect(TEXTURE; device = KA.CPU(), nms = :none, raw = true)
    rng = Xoshiro(3)
    for mode in (:mxm, :ellipse)
        ref = VgtkFeatures.cross_warp_nms(pool, W, H; device = KA.CPU(), mode)
        @test ref == asift_detect(TEXTURE; device = KA.CPU(), nms = mode, raw = true)
        @test all(_ -> VgtkFeatures.cross_warp_nms(pool[randperm(rng, length(pool))], W, H; device = KA.CPU(), mode) == ref, 1:3)
    end
    # and the spread selection after it: the same detections, run to run
    dets = asift_detect(TEXTURE; device = KA.CPU())
    kept = spread_strongest(dets, (H, W), 100; device = KA.CPU())
    @test isequal(spread_strongest(asift_detect(TEXTURE; device = KA.CPU()), (H, W), 100; device = KA.CPU()), kept)
end

@testitem "ASIFT detection mask: no detection within three scales of a masked pixel, on every path" setup = [Fixtures] begin
    using VisualGeometryToolkitCore: centroid, characteristic_scale, feature_frame
    using VisualGeometryToolkitFeatures: StabilizedImage
    # IMG holds three blobs; masking a disc over the first removes it and nothing far from the mask
    H, W = size(IMG)
    cy, cx = BLOBS[1][1], BLOBS[1][2]
    mask = [hypot(i - cy, j - cx) < 6 for i in 1:H, j in 1:W]
    # the masked pixels' distance to a detection's centre, in COLMAP coordinates
    gap(c) = minimum(hypot(c[1] - (j - 0.5), c[2] - (i - 0.5)) for i in 1:H, j in 1:W if mask[i, j])
    # the rule reads the distance at the pixel holding the centre, so it is the exact gap within a pixel's reach
    reach(d) = 3characteristic_scale(feature_frame(d))
    near(d, (y, x)) = hypot(centroid(d)[1] - (x - 0.5), centroid(d)[2] - (y - 0.5)) < 2
    all_ = asift_detect(IMG; device = KA.CPU())
    @test any(d -> near(d, (cy, cx)), all_)
    for nms in (:mxm, :none)
        m = asift_detect(IMG; device = KA.CPU(), nms, mask)
        unmasked = asift_detect(IMG; device = KA.CPU(), nms)
        @test !isempty(m) && all(d -> gap(centroid(d)) > reach(d) - 1, m)
        @test !any(d -> near(d, (cy, cx)), m)
        @test all(b -> any(d -> near(d, (b[1], b[2])), m), BLOBS[2:3])   # the unmasked blobs stay
        # what the mask keeps is the unmasked result, in its order, less the detections the mask reaches
        @test m == filter(in(Set(m)), unmasked)
        @test all(d -> gap(centroid(d)) <= reach(d) + 1, setdiff(unmasked, m))
        @test length(m) < length(unmasked)
    end
    # the raw features and the covariances take the same mask
    r = asift_detect(IMG; device = KA.CPU(), raw = true, uncertainty = FeatureCovariance(), mask)
    full = asift_detect(IMG; device = KA.CPU(), raw = true, uncertainty = FeatureCovariance())
    keep = [f in r.features for f in full.features]
    @test r.features == full.features[keep] && r.cov == full.cov[keep]
    @test asift_detect(IMG; device = KA.CPU(), mask = falses(H, W)) == all_
    @test_throws DimensionMismatch asift_detect(IMG; device = KA.CPU(), mask = falses(W, H))
    # a StabilizedImage: its saturation is masked by the same rule, and a caller's mask joins it
    img = Float64.(IMG) .* 50
    sat = [hypot(i - BLOBS[3][1], j - BLOBS[3][2]) < 4 for i in 1:H, j in 1:W]
    s = StabilizedImage(img, sat, 50.0)
    open_ = StabilizedImage(img, falses(H, W), 50.0)
    @test asift_detect(s; device = KA.CPU(), noise_sigmas = 1) == asift_detect(open_; device = KA.CPU(), noise_sigmas = 1, mask = sat)
    both = asift_detect(s; device = KA.CPU(), noise_sigmas = 1, mask)
    @test both == asift_detect(open_; device = KA.CPU(), noise_sigmas = 1, mask = mask .| sat)
    @test !any(d -> near(d, (cy, cx)) || near(d, (BLOBS[3][1], BLOBS[3][2])), both)
    @test any(d -> near(d, (BLOBS[2][1], BLOBS[2][2])), both)
end

@testitem "ASIFT untilted: tilts = :untilted is the identity warp alone" setup = [Fixtures, TextureFixture] begin
    import VisualGeometryToolkitFeatures as VgtkFeatures
    # the untilted view (t = 1) localizes best; restricting ASIFT to it is a schedule of its own
    u = asift_detect(TEXTURE; device = KA.CPU(), tilts = :untilted, nms = :none, raw = true)
    @test !isempty(u) && all(f -> f.warp_idx == 1, u)
    @test u == asift_detect(TEXTURE; device = KA.CPU(), tilts = [(1.0, 0.0)], nms = :none, raw = true)
    # exactly the identity warp's detections of the full schedule
    full = asift_detect(TEXTURE; device = KA.CPU(), tilts = :imas25, nms = :none, raw = true)
    @test any(f -> f.warp_idx > 1, full)
    @test u == filter(f -> f.warp_idx == 1, full)
    @test full == asift_detect(TEXTURE; device = KA.CPU(), nms = :none, raw = true)   # :imas25 is the default
    @test length(VgtkFeatures.tilt_schedule(:imas25)) == 25 && VgtkFeatures.tilt_schedule(:untilted) == [(1.0, 0.0)]
    @test AsiftPlan(64, 48; device = KA.CPU(), tilts = :untilted).nw == 1
    @test_throws ArgumentError asift_detect(TEXTURE; device = KA.CPU(), tilts = :sideways)
end
