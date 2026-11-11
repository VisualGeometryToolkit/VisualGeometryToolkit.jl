# variance_stabilize (src/variance_stabilization.jl): the generalized Anscombe transform of a green-cell image under
# Poisson–Gaussian noise, and ASIFT on it with its contrast threshold in noise σ, saturated pixels masked.

@testsetup module NoisyMosaic
using VisualGeometryToolkitFeatures: RawMosaic
using Random: AbstractRNG
export noisy_mosaic, poisson, RGGB, BLACK, WHITE

const RGGB = UInt8[0 1; 3 2]
const BLACK = (512.0, 500.0, 520.0, 508.0)     # per colour index: the two greens (1, 3) differ
const WHITE = 16383.0

# A Poisson draw: Knuth's product for a small mean, the normal of the same mean and variance above it (the transform
# reads only the first two moments).
function poisson(rng::AbstractRNG, λ::Real)
    λ > 50 && return max(0.0, round(λ + sqrt(λ) * randn(rng)))
    L = exp(-λ); k = 0; p = 1.0
    while true
        p *= rand(rng)
        p <= L && return Float64(k)
        k += 1
    end
end

# A mosaic whose photosite (i, j) records `electrons(i, j)` mean electrons at `electrons_per_dn` electrons per code,
# with Gaussian read noise `read_dn` codes, rounded to codes and clipped at the white level.
function noisy_mosaic(rng, electrons, H, W; electrons_per_dn, read_dn)
    raw = Matrix{UInt16}(undef, H, W)
    for j in 1:W, i in 1:H
        c = RGGB[mod1(i, 2), mod1(j, 2)]
        v = BLACK[c + 1] + poisson(rng, electrons(i, j)) / electrons_per_dn + read_dn * randn(rng)
        raw[i, j] = round(UInt16, clamp(v, 0, WHITE))
    end
    return RawMosaic(raw, RGGB, BLACK, WHITE, (0, 0), (H, W), (; model = "synthetic"))
end
end

@testitem "variance_stabilize: unit noise at every level, from the photosites' own black levels" setup = [NoisyMosaic] begin
    using VisualGeometryToolkitFeatures: LinearRaw, variance_stabilize
    using Random: Xoshiro
    using Statistics: std, mean
    rng = Xoshiro(1)
    K, σr = 2.5, 3.0
    read_noise = sqrt(σr^2 + 1 / 12)                   # the codes' rounding is read noise too
    levels = (0.0, 2.0, 10.0, 50.0, 300.0, 3000.0, 20_000.0, 38_000.0)
    for gain in (1.0, 2.0), mode in (:green_cells, :bin2)
        m = noisy_mosaic(rng, (i, j) -> levels[cld(j, 128)], 128, 128 * length(levels); electrons_per_dn = K, read_dn = σr)
        L = LinearRaw(m; mode, gain)
        @test sort(collect(L.geometry.green_dn)) == [WHITE - BLACK[4], WHITE - BLACK[2]]   # the two greens (colour indices 1 and 3)
        s = variance_stabilize(L; electrons_per_dn = K, read_noise)
        @test size(s.image) == (64, 64 * length(levels)) && s.saturated == L.saturated
        for (b, λ) in enumerate(levels)
            σ = std(s.image[:, (64(b - 1) + 1):(64b)])
            @test (gain, mode, λ, isapprox(σ, 1; atol = 0.06)) == (gain, mode, λ, true)
        end
        # the transform is increasing, and the white level is its value at 1 × gain
        @test issorted([mean(s.image[:, (64(b - 1) + 1):(64b)]) for b in eachindex(levels)])
        @test s.white ≈ variance_stabilize(LinearRaw(noisy_mosaic(rng, (i, j) -> 0.0, 2, 2; electrons_per_dn = K, read_dn = 0.0); mode, gain);
                                           electrons_per_dn = K, read_noise).white
    end
    @test_throws ArgumentError variance_stabilize(LinearRaw(noisy_mosaic(rng, (i, j) -> 1.0, 4, 4; electrons_per_dn = K, read_dn = σr); mode = :bin2);
                                                  electrons_per_dn = 0, read_noise)
    @test_throws ArgumentError variance_stabilize(LinearRaw(noisy_mosaic(rng, (i, j) -> 1.0, 4, 4; electrons_per_dn = K, read_dn = σr); mode = :bin2);
                                                  electrons_per_dn = K, read_noise = -1)
end

@testitem "asift_detect on a stabilized image: the threshold in noise σ of the noisiest DoG slice" setup = [NoisyMosaic] begin
    using VisualGeometryToolkitFeatures: LinearRaw, variance_stabilize, asift_detect, asift_detect!, AsiftPlan, StabilizedImage
    import VisualGeometryToolkitFeatures as VF
    import KernelAbstractions as KA
    using Random: Xoshiro
    using Statistics: std
    # the unit-noise intensity threshold against the DoG of white noise through the same pyramid: the finest octave's
    # noisiest searched slice (the interior ones the extremum search reads, 2 … NS − 1), measured, is the derived σ (the
    # impulse response's energy), within the sampling error
    thr1 = VF._noise_intensity_thr(1.6, true)
    n = 512
    plan = AsiftPlan(n, n; device = KA.CPU(), tilts = [(1.0, 0.0)])
    asift_detect!(plan, randn(Xoshiro(2), Float32, n, n); intensity_thr = 1.0e9)
    dog = Array(plan.dogs[1])[33:(2n - 32), 33:(2n - 32), :, 1]
    measured = maximum(std(dog[px:2:end, py:2:end, sl]) for px in 1:2, py in 1:2, sl in 2:(size(dog, 3) - 1))
    @test isapprox(thr1 / plan.nb_scales, measured; rtol = 0.03)
    # on pure noise of σ 1, a threshold of k σ admits fewer extrema as k grows, and few at 5σ
    rng = Xoshiro(3)
    img = 100 .+ randn(rng, 256, 256)
    s = StabilizedImage(img, falses(256, 256), 200.0)
    counts = [length(asift_detect(s; noise_sigmas = k, device = KA.CPU(), tilts = [(1.0, 0.0)])) for k in (1, 3, 5)]
    @test counts[1] > counts[2] > counts[3]
    @test counts[3] <= 5
    # the same detections as the [0, 1] image at the equivalent intensity threshold (the pool before the cross-warp NMS)
    a = asift_detect(s; noise_sigmas = 2, nms = :none, device = KA.CPU(), tilts = [(1.0, 0.0)])
    b = asift_detect(Float32.(img ./ 200); nms = :none, intensity_thr = 2thr1 / 200, device = KA.CPU(), tilts = [(1.0, 0.0)])
    @test sort(map(d -> (d.center[1], d.center[2]), a)) == sort(map(d -> (d.center[1], d.center[2]), b))
    @test_throws ArgumentError asift_detect(s; intensity_thr = 0.04)
    # the raw features too: the masking (here none) works on either
    @test asift_detections(asift_detect(s; noise_sigmas = 2, nms = :none, device = KA.CPU(), tilts = :untilted, raw = true)) == a
    @test_throws ArgumentError asift_detect(s; noise_sigmas = 0)
end

@testitem "asift_detect on a stabilized image: saturated pixels give no detection" setup = [NoisyMosaic] begin
    using VisualGeometryToolkitFeatures: LinearRaw, variance_stabilize, asift_detect
    using VisualGeometryToolkitCore: centroid, characteristic_scale, feature_frame
    import KernelAbstractions as KA
    using Random: Xoshiro
    rng = Xoshiro(4)
    K = 2.5
    # a clipped bright window (a disc beyond the white level) and dark blobs elsewhere, on a mid-grey wall
    H, W = 256, 512
    blobs = [(64.0 + 128k, 192.0) for k in 0:3]
    function scene(i, j)
        x, y = j - 0.5, i - 0.5
        hypot(x - 128, y - 64) < 30 && return 60_000.0               # the window: past the white level
        v = 8000.0
        for (bx, by) in blobs
            v -= 6000 * exp(-((x - bx)^2 + (y - by)^2) / (2 * 8.0^2))
        end
        return v
    end
    m = noisy_mosaic(rng, scene, H, W; electrons_per_dn = K, read_dn = 3.0)
    L = LinearRaw(m; mode = :green_cells)
    @test count(L.saturated) > 100
    s = variance_stabilize(L; electrons_per_dn = K, read_noise = 3.0)
    ds = asift_detect(s; device = KA.CPU(), tilts = [(1.0, 0.0)])
    # the cell grid is half the raw's: the window at (64, 32) r 15 cells, the blobs at y = 96 cells
    near_window(d) = hypot(centroid(d)[1] - 64, centroid(d)[2] - 32) < 15 + 3characteristic_scale(feature_frame(d))
    @test !isempty(ds) && !any(near_window, ds)
    @test all(any(d -> hypot(centroid(d)[1] - bx / 2, centroid(d)[2] - by / 2) < 3, ds) for (bx, by) in blobs)
    # the same image with the saturated pixels unflagged detects at the window's rim
    u = typeof(s)(s.image, falses(size(s.saturated)), s.white)
    @test any(near_window, asift_detect(u; device = KA.CPU(), tilts = [(1.0, 0.0)]))
end

@testitem "StabilizedImage holds a Float64 image and a BitMatrix mask without copying them" setup = [NoisyMosaic] begin
    using VisualGeometryToolkitFeatures: LinearRaw, StabilizedImage, variance_stabilize
    using Random: Xoshiro
    img = rand(Xoshiro(2), 300, 400); sat = falses(300, 400)
    s = StabilizedImage(img, sat, 2.0)
    @test s.image === img && s.saturated === sat
    @test @allocated(StabilizedImage(img, sat, 2.0)) < sizeof(img) ÷ 10
    # another element type is converted, once
    s32 = StabilizedImage(Float32.(img), Matrix(sat), 2.0)
    @test s32.image isa Matrix{Float64} && s32.image == Float64.(Float32.(img)) && s32.saturated isa BitMatrix
    # the transform's image is the one full-frame array variance_stabilize allocates
    L = LinearRaw(noisy_mosaic(Xoshiro(3), (i, j) -> 100.0, 600, 800; electrons_per_dn = 2.5, read_dn = 3.0); mode = :green_cells)
    variance_stabilize(L; electrons_per_dn = 2.5, read_noise = 3.0)
    @test @allocated(variance_stabilize(L; electrons_per_dn = 2.5, read_noise = 3.0)) < 1.5 * sizeof(Float64) * 300 * 400
end
