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
        m = asift_detect(IMG; device = KA.CPU(), nms, exclusion_mask = mask)
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
    r = asift_detect(IMG; device = KA.CPU(), raw = true, uncertainty = FeatureCovariance(), exclusion_mask = mask)
    full = asift_detect(IMG; device = KA.CPU(), raw = true, uncertainty = FeatureCovariance())
    keep = [f in r.features for f in full.features]
    @test r.features == full.features[keep] && r.cov == full.cov[keep]
    @test asift_detect(IMG; device = KA.CPU(), exclusion_mask = falses(H, W)) == all_
    @test_throws DimensionMismatch asift_detect(IMG; device = KA.CPU(), exclusion_mask = falses(W, H))
    # a StabilizedImage: its saturation is masked by the same rule, and a caller's mask joins it
    img = Float64.(IMG) .* 50
    sat = [hypot(i - BLOBS[3][1], j - BLOBS[3][2]) < 4 for i in 1:H, j in 1:W]
    s = StabilizedImage(img, sat, 50.0)
    open_ = StabilizedImage(img, falses(H, W), 50.0)
    @test asift_detect(s; device = KA.CPU(), noise_sigmas = 1) == asift_detect(open_; device = KA.CPU(), noise_sigmas = 1, exclusion_mask = sat)
    both = asift_detect(s; device = KA.CPU(), noise_sigmas = 1, exclusion_mask = mask)
    @test both == asift_detect(open_; device = KA.CPU(), noise_sigmas = 1, exclusion_mask = mask .| sat)
    @test !any(d -> near(d, (cy, cx)) || near(d, (BLOBS[3][1], BLOBS[3][2])), both)
    @test any(d -> near(d, (BLOBS[2][1], BLOBS[2][2])), both)
end

@testitem "ASIFT exclusion mask: the retired keyword `mask` is refused, naming `exclusion_mask`" setup = [Fixtures] begin
    using VisualGeometryToolkitFeatures: StabilizedImage
    H, W = size(IMG)
    refused(f) = try
        f(); false
    catch e
        e isa ArgumentError && occursin("exclusion_mask", e.msg)
    end
    @test refused(() -> asift_detect(IMG; device = KA.CPU(), mask = falses(H, W)))
    @test refused(() -> asift_detect(StabilizedImage(Float64.(IMG), falses(H, W), 1.0); device = KA.CPU(), mask = falses(H, W)))
    @test asift_detect(IMG; device = KA.CPU(), exclusion_mask = falses(H, W)) == asift_detect(IMG; device = KA.CPU())
end

@testitem "ASIFT detection mask: the clearance is the exact Euclidean distance to the nearest masked pixel" begin
    using VisualGeometryToolkitCore: centroid, characteristic_scale
    using VisualGeometryToolkitFeatures: ImasFeatureKA
    using Random: Xoshiro
    const VGF = VisualGeometryToolkitFeatures
    # the rule, by brute force: the pixel holding the centre is clear when its distance to every masked pixel,
    # √(Δi² + Δj²) in pixels, exceeds three scales
    function clear(f, mask)
        fr = VGF._asift_frame(f)
        H, W = size(mask)
        c = centroid(fr)
        i, j = clamp(floor(Int, c[2]) + 1, 1, H), clamp(floor(Int, c[1]) + 1, 1, W)
        d = minimum((sqrt(Float64((k - i)^2 + (l - j)^2)) for k in 1:H, l in 1:W if mask[k, l]); init = Inf)
        return d > VGF._MASK_REACH * characteristic_scale(fr)
    end
    rng = Xoshiro(7)
    for (H, W, density) in ((40, 60, 0.002), (37, 23, 0.05), (50, 50, 0.3), (16, 16, 1.0))
        mask = rand(rng, H, W) .< density
        # centres over the frame and a little past it, scales from sub-pixel to wider than the frame, isotropic and
        # sheared shapes; a scale on a pixel distance exactly (a tie: not clear)
        feats = [begin
                     σ = rand(rng) < 0.1 ? Float32(rand(rng, 1:6) / 3) : Float32(exp(rand(rng) * 3 - 1.5))
                     a, b = Float32(1 + rand(rng)), Float32(rand(rng) - 0.5)
                     ImasFeatureKA(Float32(rand(rng) * (W + 4) - 2), Float32(rand(rng) * (H + 4) - 2), σ, 1.0f0,
                                   σ^2 * a, σ^2 * b, σ^2 / a + σ^2 * b^2 / a, Int32(0))
                 end for _ in 1:400]
        @test VGF._clear_of_mask(feats, mask) == [clear(f, mask) for f in feats]
    end
end

@testitem "ASIFT pool: the canonical order sorts the columns lexicographically, the covariances alike" begin
    using Random: Xoshiro
    const VGF = VisualGeometryToolkitFeatures
    rng = Xoshiro(5)
    for n in (1, 2, 7, 500), with_cov in (false, true)
        # few distinct values, so that columns tie in their leading rows (and some in every row); signed zeros
        fh = Float32.(rand(rng, -2:2, 9, n)) ./ 2; fh[rand(rng, 1:9n, n ÷ 3 + 1)] .= -0.0f0
        ch = with_cov ? Float32.(rand(rng, 0:1, 6, n)) : nothing
        cols = [with_cov ? (fh[:, i]..., ch[:, i]...) : (fh[:, i]...,) for i in 1:n]
        VGF._canonical_order!(fh, ch)
        @test [with_cov ? (fh[:, i]..., ch[:, i]...) : (fh[:, i]...,) for i in 1:n] == sort(cols)
    end
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
