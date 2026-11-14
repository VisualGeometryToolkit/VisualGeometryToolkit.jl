# detect_sfop (SFOP scale selection for circular points on gaussian_scale_space) and the seeded detect_foerstner
# (the Förstner window estimation at given windows), on synthetic dots of known centre.

@testsetup module SfopFixture
using Random: Xoshiro, rand, randn
export dot_grid

# An n×n grid of discs of radius `radius` at pitch 30 px (sub-pixel phases), blurred by a Gaussian PSF of 1.5 px,
# background 0.0015, plateau 1.4 clipped at 1 (the Thorlabs dots as the chrome photographs show them), one empty
# tile of margin; noise σn grey levels of an 8-bit image. Returns the COLMAP centres and the image.
function dot_grid(n, radius; σn = 0.5, seed = 1, ss = 4)
    rng = Xoshiro(seed)
    truth = [(30tx + 15 + rand(rng) - 0.5, 30ty + 15 + rand(rng) - 0.5) for tx in 1:n for ty in 1:n]
    H = W = 30(n + 2)
    disc = zeros(H, W)
    for r in 1:H, c in 1:W
        tx, ty = fld(c - 1, 30), fld(r - 1, 30)
        (1 <= tx <= n && 1 <= ty <= n) || continue
        a, b = truth[(tx - 1) * n + ty]
        disc[r, c] = sum((c - 1 + (i - 0.5) / ss - a)^2 + (r - 1 + (j - 0.5) / ss - b)^2 < radius^2 for i in 1:ss, j in 1:ss) / ss^2
    end
    k = [exp(-x^2 / (2 * 1.5^2)) for x in -6:6]; k ./= sum(k)
    blur1(a, d) = [sum(k[t + 7] * (d == 1 ? a[clamp(i + t, 1, H), j] : a[i, clamp(j + t, 1, W)]) for t in -6:6) for i in 1:H, j in 1:W]
    clean = min.(0.0015 .+ (1.4 - 0.0015) .* blur1(blur1(disc, 1), 2), 1.0)
    return truth, Float32.(clean .+ (σn / 255) .* randn(rng, H, W))
end
end

@testitem "detect_sfop: every dot, the selected scale, no point on pure noise" setup = [SfopFixture] begin
    import KernelAbstractions as KA
    using LinearAlgebra: norm
    using Random: Xoshiro, randn
    truth, img = dot_grid(5, 2.2)
    seeds = detect_sfop(img; device = KA.CPU(), noise_sigma = 0.5)
    near = [argmin(k -> norm(seeds[k].center .- t), eachindex(seeds)) for t in truth]
    @test allunique(near)
    # The schedule (detect_sfop's defaults): S = 3 levels per octave, the first octave 2× upsampled, its first level the
    # input's own blur (seed σ = 2·0.5 in its pixels), maxima searched on levels 2 … S + 2 of each of 3 octaves, each
    # level l at the integration scale σ = 3·(level l + 1's blur). The ladder of (σ, octave pixel in input pixels):
    space = gaussian_scale_space(img; device = KA.CPU(), seed_sigma = 1.0, nb_octaves = 3)
    S = space.nb_scales
    ladder = [(3 * VisualGeometryToolkitFeatures.level_sigma(space, o, l + 1), exp2(space.first_octave + o - 1))
              for o in 1:3 for l in 2:(S + 2)]
    level_ratio = exp2(1 / S)                                 # one level; σ is Float32 arithmetic (relative rounding ≤ 1e-6)
    within(σ, s) = s / level_ratio * (1 - 1.0e-6) <= σ <= s * level_ratio * (1 + 1.0e-6)   # a sub-level interpolation stays within a level
    # The σ selected: the largest ladder σ whose window — the integration taps' support, ⌈3.3σ⌉ pixels of its octave —
    # still holds one dot, i.e. ends before the neighbour's blurred edge at pitch − radius − 3·PSF (30 − 2.2 − 4.5 px);
    # the precision peaks there, within one level (the peak can fall between levels)
    reach = 30 - 2.2 - 3 * 1.5
    σ_one = maximum(s for (s, sf) in ladder if ceil(3.3 * s / sf) * sf <= reach)
    @test all(within(seeds[k].σ, σ_one) for k in near)
    # Stage 1's position: a maximum kept unrefined sits on its octave's pixel grid, ≤ half that octave's pixel off per
    # axis, plus the noise term 3/√w (w the estimate's precision, px⁻²). The octave: the coarsest whose searched levels
    # reach the detection's σ.
    pixel(σ) = maximum(sf for (s, sf) in ladder if within(σ, s))
    @test all(all(abs.(seeds[k].center .- t) .<= 0.5 * pixel(seeds[k].σ) + 3 / sqrt(seeds[k].precision)) for (k, t) in zip(near, truth))
    # every seed (the dots among them) succeeds in the detector's own precision test T_w = 1/precision_threshold² (0.5 px)
    @test all(d -> d.precision >= 1 / 0.5^2, seeds)
    # the λ₂ test at significance 0.999 (no empirical factor): nothing on pure noise
    for σn in (0.5, 2.0)
        noise = Float32.(0.3 .+ (σn / 255) .* randn(Xoshiro(5), 400, 400))
        @test isempty(detect_sfop(noise; device = KA.CPU(), noise_sigma = σn))
    end
    @test_throws ArgumentError detect_sfop(img; significance_level = 1.0)
    @test_throws ArgumentError detect_sfop(img; noise_sigma = 0)
end

@testitem "detect_foerstner at seeds: the window estimation of detect_foerstner, bit for bit" setup = [SfopFixture] begin
    import KernelAbstractions as KA
    using VisualGeometryToolkitCore: Disc, Point2
    truth, img = dot_grid(4, 2.2)
    for noise in (nothing, 0.5), σi in (2.0, 4.0)
        found = detect_foerstner(Float32, img; device = KA.CPU(), noise_sigma = 0.5, noise, integration_sigma = σi)
        # a seed at each window centre with the scale σi: the same windows, so the same rows
        seeds = [Detection(Disc(d.window, σi), (; precision = 1.0)) for d in found]
        again = detect_foerstner(Float32, img, seeds; device = KA.CPU(), noise)
        @test length(again) == length(found)
        @test all(isequal(a, b) for (a, b) in zip(again, found))
    end
end

@testitem "detect_sfop → detect_foerstner: today's accuracy on the dots" setup = [SfopFixture] begin
    import KernelAbstractions as KA
    using LinearAlgebra: norm
    truth, img = dot_grid(6, 2.2; seed = 3)
    points = detect_foerstner(Float32, img, detect_sfop(img; device = KA.CPU(), noise_sigma = 0.5); device = KA.CPU(), noise = 0.5)
    today = detect_foerstner(Float32, img; device = KA.CPU(), noise_sigma = 0.5, integration_sigma = 4.0)
    err(ps) = [minimum(norm(d.center .- t) for d in ps if d.label === :circular) for t in truth]
    @test maximum(err(points)) < 0.05                    # pixel locking of a 4×-supersampled render
    @test sqrt(sum(abs2, err(points)) / length(truth)) <= 1.05 * sqrt(sum(abs2, err(today)) / length(truth))
end

@testitem "detect_sfop's maxima: no edge test, a maximum beside a failed window kept" begin
    import KernelAbstractions as KA
    VGF = VisualGeometryToolkitFeatures
    # one octave's precision cube (−1/w): a dot, a peak of w at pixel (20, 20) on level 3, and one neighbour whose window
    # failed the λ₂ test (−∞). The extraction's quadratic interpolation fails there, so the maximum is kept on its pixel
    # and level; the edge test (ASIFT's, meant off in the precision mode) must not drop it either: its curvature across
    # the −∞ neighbour is −∞, and tr²/det = ∞/∞ was NaN, never below the limit
    S = 3
    cube = [-1.0f0 - 0.01f0 * Float32((i - 20)^2 + (j - 20)^2 + (l - 3)^2) for i in 1:40, j in 1:40, l in 1:(S + 3), _ in 1:1]
    found(c) = (pool = zeros(Float32, 5, 16); count = zeros(Int32, 2);
                VGF._sfop_maxima!(pool, count, KA.CPU(), c, 1.0f0, Int32(0), Int32(S)); pool[:, 1:count[1]])
    @test found(cube)[1:2, :] == reshape(Float32[19, 19], 2, 1)    # 0-based octave pixels
    beside = copy(cube)
    beside[21, 20, 3, 1] = -Inf32
    @test found(beside)[1:2, :] == reshape(Float32[19, 19], 2, 1)
    diagonal = copy(cube)
    diagonal[21, 21, 3, 1] = -Inf32                                # the cross term −∞: det < 0, a saddle to the edge test
    @test found(diagonal)[1:2, :] == reshape(Float32[19, 19], 2, 1)
end

@testitem "detect_sfop's detections: each keeps its own precision when two octaves find the same point" begin
    import KernelAbstractions as KA
    VGF = VisualGeometryToolkitFeatures
    # two maxima at the same input position and scale, one per octave (adjacent octaves share a level), w = 4 and 2:
    # the NMS keeps the stronger, and its precision is its own w (a table keyed by position and scale alone gave the
    # other octave's)
    raw = Float32[50 50; 50 50; 3 3; -1/4 -1/2; 1 1]
    rows = VGF._sfop_detections(KA.CPU(), (; nb_scales = 3), raw, [1, 2], 100, 100, 0.0)
    @test length(rows) == 1 && only(rows).precision == 4.0
    flipped = VGF._sfop_detections(KA.CPU(), (; nb_scales = 3), raw[:, [2, 1]], [1, 2], 100, 100, 0.0)
    @test length(flipped) == 1 && only(flipped).precision == 4.0
end

@testitem "detect_sfop: the default octaves are what the image holds" setup = [SfopFixture] begin
    import KernelAbstractions as KA
    using Random: Xoshiro, randn
    # a 40 × 40 image holds sift_nb_octaves(40, 40; upsample = true) = 2 octaves: the default 3 is clamped to them
    small = Float32.(0.3 .+ (0.5 / 255) .* randn(Xoshiro(2), 40, 40))
    @test VisualGeometryToolkitFeatures.sift_nb_octaves(40, 40; upsample = true) == 2
    @test isempty(detect_sfop(small; device = KA.CPU(), noise_sigma = 0.5))
    @test isequal(collect(detect_sfop(small; device = KA.CPU(), noise_sigma = 0.5)),
                  collect(detect_sfop(small; device = KA.CPU(), noise_sigma = 0.5, nb_octaves = 2)))
    # an image that holds more keeps 3; more than the image holds, asked for, is refused
    truth, img = dot_grid(5, 2.2)
    @test isequal(collect(detect_sfop(img; device = KA.CPU(), noise_sigma = 0.5)),
                  collect(detect_sfop(img; device = KA.CPU(), noise_sigma = 0.5, nb_octaves = 3)))
    @test_throws ArgumentError detect_sfop(small; device = KA.CPU(), nb_octaves = 3)
end

@testitem "detect_foerstner at seeds: its own keywords and seeds with a scale" setup = [SfopFixture] begin
    import KernelAbstractions as KA
    using VisualGeometryToolkitCore: Disc, Point2, PointFrame
    truth, img = dot_grid(2, 2.2)
    seeds = [Detection(Disc(Point2(t...), 2.0), (; precision = 1.0)) for t in truth]
    @test length(detect_foerstner(Float32, img, seeds; device = KA.CPU(), noise = 0.5, derivative_sigma = 1.0,
                                  derivative = :gaussian2d, significance_level = 0.99)) == length(seeds)
    # the image-wide search's keywords are not read at given windows: refused, by name
    err(f) = try
        f(); nothing
    catch e
        e
    end
    e = err(() -> detect_foerstner(Float32, img, seeds; device = KA.CPU(), integration_sigma = 3.0, method = :koethe))
    @test e isa ArgumentError && occursin("integration_sigma", e.msg) && occursin("method", e.msg)
    for kw in (:integration, :noise_sigma, :precision_threshold, :roundness_threshold)
        e = err(() -> detect_foerstner(img, seeds; device = KA.CPU(), (kw => 1)))
        @test e isa ArgumentError && occursin(string(kw), e.msg)
    end
    # a seed without a scale (a point) has no window: a MethodError of detect_foerstner itself, not of a verb inside it
    points = [Detection(PointFrame(Point2(t...)), (; precision = 1.0)) for t in truth]
    e = err(() -> detect_foerstner(Float32, img, points))         # no keywords: the MethodError names the function
    @test e isa MethodError && e.f === detect_foerstner
end
