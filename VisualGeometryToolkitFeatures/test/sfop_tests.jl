# detect_sfop (SFOP scale selection for circular points on gaussian_scale_space) and the seeded detect_foerstner
# (the Förstner window estimation at given windows), on synthetic dots of known centre.

@testsetup module SfopFixture
using Random: Xoshiro, rand, randn
export dot_grid, signed_grid, sharp_disc

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

# An n×n grid at pitch 30 px of bright dots (light, +0.35 on a 0.5 background) and dark holes (−0.35), alternating
# like a checkerboard, radius 2.5, the same PSF and margin as `dot_grid`. Returns the light and the dark centres.
function signed_grid(n; σn = 0.5, seed = 2, ss = 4, radius = 2.5)
    rng = Xoshiro(seed)
    cells = [(tx, ty, 30tx + 15 + rand(rng) - 0.5, 30ty + 15 + rand(rng) - 0.5) for tx in 1:n for ty in 1:n]
    H = W = 30(n + 2)
    a = fill(0.5, H, W)
    for r in 1:H, c in 1:W, (tx, ty, x, y) in cells
        (fld(c - 1, 30), fld(r - 1, 30)) == (tx, ty) || continue
        cover = sum((c - 1 + (i - 0.5) / ss - x)^2 + (r - 1 + (j - 0.5) / ss - y)^2 < radius^2 for i in 1:ss, j in 1:ss) / ss^2
        a[r, c] += (iseven(tx + ty) ? 0.35 : -0.35) * cover
    end
    k = [exp(-x^2 / (2 * 1.5^2)) for x in -6:6]; k ./= sum(k)
    blur1(b, d) = [sum(k[t + 7] * (d == 1 ? b[clamp(i + t, 1, H), j] : b[i, clamp(j + t, 1, W)]) for t in -6:6) for i in 1:H, j in 1:W]
    img = Float32.(blur1(blur1(a, 1), 2) .+ (σn / 255) .* randn(rng, H, W))
    light = [(x, y) for (tx, ty, x, y) in cells if iseven(tx + ty)]
    dark = [(x, y) for (tx, ty, x, y) in cells if isodd(tx + ty)]
    return light, dark, img
end

# One sharp bright disc of radius `radius` at COLMAP (x, y) on a 96 × 96 image (4×4 supersampled, no PSF).
function sharp_disc(x, y, radius; σn = 0.5, seed = 1, ss = 4)
    img = [sum(hypot(c - 1 + (i - 0.5) / ss - x, r - 1 + (j - 0.5) / ss - y) < radius for i in 1:ss, j in 1:ss) / ss^2
           for r in 1:96, c in 1:96]
    return Float32.(0.1 .+ 0.6 .* img .+ (σn / 255) .* randn(Xoshiro(seed), 96, 96))
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
    # an uncertain frame with a scale is a seed too, read through its frame (`Disc` of the detection): the plain seeds' windows
    using VisualGeometryToolkitCore: EllipticFrame, Uncertain, CovMat
    plain = detect_foerstner(Float32, img, seeds; device = KA.CPU(), noise = 0.5)
    ellipses = [Detection(Uncertain(EllipticFrame(Disc(Point2(t...), 2.0)), CovMat{5}(σ = 0.1)), (; precision = 1.0)) for t in truth]
    @test [d.window for d in detect_foerstner(Float32, img, ellipses; device = KA.CPU(), noise = 0.5)] == [d.window for d in plain]
    # a seed without a scale (a point) has no window: an ArgumentError naming its frame, not an error of a verb inside the call
    points = [Detection(PointFrame(Point2(t...)), (; precision = 1.0)) for t in truth]
    e = err(() -> detect_foerstner(Float32, img, points))
    @test e isa ArgumentError && occursin("PointFrame", e.msg) && occursin("scale", e.msg)
    # a vector of an abstract element type holds seeds too: empty (no rows), or scaled frames of several kinds
    @test isempty(detect_foerstner(Float32, img, Detection[]; device = KA.CPU(), noise = 0.5))
    mixed = Detection[seeds[1:2]; ellipses[3:end]]
    @test [d.window for d in detect_foerstner(Float32, img, mixed; device = KA.CPU(), noise = 0.5)] == [d.window for d in plain]
    e = err(() -> detect_foerstner(Float32, img, Detection[seeds[1:2]; points[3:end]]))
    @test e isa ArgumentError && occursin("PointFrame", e.msg)
end

@testitem "detect_sfop and the seeded detect_foerstner: polarity keeps each sign's own blobs" setup = [SfopFixture] begin
    import KernelAbstractions as KA
    using LinearAlgebra: norm
    light, dark, img = signed_grid(4)
    near(ds, ts) = all(any(norm(d.center .- t) < 1.5 for t in ts) for d in ds)
    found(ds, ts) = all(any(norm(d.center .- t) < 1.5 for d in ds) for t in ts)
    both = detect_sfop(img; device = KA.CPU(), noise_sigma = 0.5)
    # :none is the default, bit for bit, and finds both signs (the structure tensor is sign-invariant)
    @test isequal(collect(both), collect(detect_sfop(img; device = KA.CPU(), noise_sigma = 0.5, polarity = :none)))
    @test found(both, light) && found(both, dark)
    l = detect_sfop(img; device = KA.CPU(), noise_sigma = 0.5, polarity = :light)
    d = detect_sfop(img; device = KA.CPU(), noise_sigma = 0.5, polarity = :dark)
    @test near(l, light) && found(l, light)
    @test near(d, dark) && found(d, dark)
    # the seeded windows: each polarity keeps its own seeds, :none every one; the sign test needs the pixel noise
    rows(pol) = detect_foerstner(Float32, img, both; device = KA.CPU(), noise = 0.5, polarity = pol)
    @test isequal(collect(rows(:none)), collect(detect_foerstner(Float32, img, both; device = KA.CPU(), noise = 0.5)))
    @test near(rows(:light), light) && found(rows(:light), light)
    @test near(rows(:dark), dark) && found(rows(:dark), dark)
    @test_throws ArgumentError detect_foerstner(Float32, img, both; device = KA.CPU(), polarity = :dark)   # no noise known
    @test_throws ArgumentError detect_foerstner(Float32, img; device = KA.CPU(), noise = 0.5, polarity = :dark)   # image-wide
    @test_throws ArgumentError detect_sfop(img; device = KA.CPU(), polarity = :bright)
    @test_throws ArgumentError detect_foerstner(Float32, img, both; device = KA.CPU(), noise = 0.5, polarity = :bright)
end

@testitem "detect_sfop: polarity drops the windows between bright dots (no sign significant there)" setup = [SfopFixture] begin
    import KernelAbstractions as KA
    using LinearAlgebra: norm
    # between the dots of a 3×3 grid SFOP finds windows on the dots' tails (w just past the precision test); their
    # Laplacian's sign is the noise's, so the one-sided sign test keeps them for neither polarity: :light is the dots
    truth, img = dot_grid(3, 2.2)
    all_ = detect_sfop(img; device = KA.CPU(), noise_sigma = 0.5)
    isdot(d) = any(norm(d.center .- t) < 1.5 for t in truth)
    @test count(!isdot, all_) > 0
    l = detect_sfop(img; device = KA.CPU(), noise_sigma = 0.5, polarity = :light)
    @test length(l) == length(truth) && all(isdot, l)
    @test isempty(detect_sfop(img; device = KA.CPU(), noise_sigma = 0.5, polarity = :dark))
end

@testitem "the seeded detect_foerstner: one :circular row per blob (a sharp dot SFOP finds at two scales)" setup = [SfopFixture] begin
    import KernelAbstractions as KA
    VGF = VisualGeometryToolkitFeatures
    # a sharp disc of radius 1 px: w peaks at σ ≈ 3.3 and again at σ ≈ 9.5, more than a half-octave band apart, so the
    # NMS keeps both seeds; nothing acts on the seeds, and the seeded estimate keeps one :circular row, at the dot
    img = sharp_disc(47.0, 48.6, 1.0)
    seeds = detect_sfop(img; device = KA.CPU(), noise_sigma = 0.5)
    @test length(seeds) == 2
    points = filter(d -> d.label === :circular, detect_foerstner(Float32, img, seeds; device = KA.CPU(), noise = 0.5))
    @test length(points) == 1 && hypot((only(points).center .- (47.0, 48.6))...) < 0.5
    # the rule itself: two :circular rows closer than max(σ)/√2 are one blob, the smaller covariance trace kept (a tie
    # to the earlier row); farther apart both kept; another label never drops a row
    one(xs, σs, spreads; labels = fill(:circular, length(xs))) =
        VGF._one_per_blob_mask([VGF.Point2(x, 50.0) for x in xs], σs, labels, spreads)
    @test one([50.0, 54.0], [3.0, 6.0], [1.0, 2.0]) == [true, false]      # 4 < 6/√2 ≈ 4.24
    @test one([50.0, 54.0], [3.0, 6.0], [2.0, 1.0]) == [false, true]
    @test one([50.0, 56.0], [3.0, 3.0], [1.0, 2.0]) == [true, true]       # 6 > 3/√2
    @test one([50.0, 51.0], [3.0, 3.0], [1.0, 1.0]) == [true, false]
    @test one([50.0, 51.0], [3.0, 3.0], [1.0, 2.0]; labels = [:corner, :circular]) == [true, true]
    # only a KEPT row drops another: B (dropped by A) does not take C with it (4 < 6/√2 from B, but B is gone and C
    # lies 8 px from A, past max(3, 6)/√2)
    @test one([50.0, 54.0, 58.0], [3.0, 6.0, 6.0], [1.0, 2.0, 3.0]) == [true, false, true]
end

@testitem "detect_sfop and the seeded detect_foerstner on a Float16 image: its values in Float32, within its rounding" setup = [SfopFixture] begin
    import KernelAbstractions as KA
    using LinearAlgebra: norm, tr
    using Statistics: cov
    using Colors: Gray
    truth, img = dot_grid(5, 2.2; seed = 3)
    h = Float16.(img)
    # the Float16 storage is the only difference: bit for bit the same as its values in Float32, as Gray{Float16} too
    seeds = detect_sfop(h; device = KA.CPU(), noise_sigma = 0.5)
    @test isequal(collect(seeds), collect(detect_sfop(Float32.(h); device = KA.CPU(), noise_sigma = 0.5)))
    @test isequal(collect(seeds), collect(detect_sfop(Gray.(h); device = KA.CPU(), noise_sigma = 0.5)))
    @test isequal(collect(detect_sfop(h; device = KA.CPU(), noise_sigma = 0.5, polarity = :light)),
                  collect(detect_sfop(Float32.(h); device = KA.CPU(), noise_sigma = 0.5, polarity = :light)))
    rows16 = detect_foerstner(Float32, h, seeds; device = KA.CPU(), noise = 0.5)
    @test isequal(collect(rows16), collect(detect_foerstner(Float32, Float32.(h), seeds; device = KA.CPU(), noise = 0.5)))
    # against the Float32 image: the rounding to Float16 is noise of σ ≤ eps(Float16(1))/√12 on [0, 1] (the values are
    # below 1); each circular point moves by at most 4 of its standard deviations under that noise (propagated)
    rows32 = detect_foerstner(Float32, img, seeds; device = KA.CPU(), noise = 0.5)
    σr = 255 * eps(Float16(1)) / sqrt(12)
    under = detect_foerstner(Float32, img, seeds; device = KA.CPU(), noise = σr)
    moves = [norm(a.center - b.center) / sqrt(tr(cov(c.value))) for (a, b, c) in zip(rows16, rows32, under) if c.label === :circular]
    @test length(moves) >= length(truth) && maximum(moves) < 4
end

@testitem "detect_sfop on a LinearRaw: the λ₂ test under the frame's sensor model, no blob from shot noise" setup = [NoisyMosaic] begin
    import KernelAbstractions as KA
    import VisualGeometryToolkitCore
    using VisualGeometryToolkitCore: LinearRaw
    using LinearAlgebra: norm
    using Random: Xoshiro
    # three bright dots on a dark left half (300 e⁻), a flat bright right half (30 000 e⁻, shot noise ≈ 1.1 grey levels)
    dots = ((40.3, 38.7), (70.6, 101.2), (110.2, 60.4))
    scene(i, j) = (c = ((j + 1) ÷ 2, (i + 1) ÷ 2);
                   j > 280 ? 30000.0 : any(hypot(c[1] - a, c[2] - b) < 3 for (a, b) in dots) ? 6000.0 : 300.0)
    model = pg_model(2.5, 3.0)
    L = LinearRaw(noisy_mosaic(Xoshiro(4), scene, 280, 560; electrons_per_dn = 2.5, read_dn = 3.0); mode = :green_cells, sensor_model = model)
    # a window wholly on the flat bright half (the integration taps ⌈3.3σ⌉ and the gradient's, ≈ 4.5σ, past the edge at
    # cell 140): the step edge's own windows are structure, which the polarity test removes (below), not noise
    bright(d) = d.center[1] - 4.5 * d.σ > 141
    # one noise level for the whole frame, the dark half's: the λ₂ test takes the bright half's shot noise for blobs
    @test count(bright, detect_sfop(L; device = KA.CPU(), noise_sigma = 0.3)) > 0
    # each pixel's variance (the sensor model's law): nothing on the bright half; with the polarity, the dots alone
    rows = detect_sfop(L; device = KA.CPU(), sensor_model = model)
    @test !any(bright, rows)
    l = detect_sfop(L; device = KA.CPU(), sensor_model = model, polarity = :light)
    # every dot seeded (a weak coarse window on two dots' tails may seed too: its seeded row is no :circular point, below)
    @test all(any(norm(d.center .- t) < 1.5 for d in l) for t in dots)
    # the seeded windows under the same model: each a dot, once
    pts = detect_foerstner(Float32, L, l; sensor_model = model, device = KA.CPU(), polarity = :light)
    @test length(pts) == length(dots) && all(d -> d.label === :circular, pts)
    # no gain, no variances: noise_sigma's test, as on the matrix of the values
    @test isequal(collect(detect_sfop(L; device = KA.CPU())), collect(detect_sfop(L; device = KA.CPU(), sensor_model = VisualGeometryToolkitCore.SensorModel())))
end

# GPU: polarity, the single-detection rule and a Float16 image as on the CPU. Runs only with a functional CUDA device.
@testitem "detect_sfop polarity, single detections and Float16: CUDA matches the CPU" setup = [SfopFixture] begin
    import KernelAbstractions as KA
    using LinearAlgebra: norm
    cuda = try
        m = Base.require(Main, :CUDA)
        (m !== nothing && Base.invokelatest(getproperty(m, :functional))) ? m : nothing
    catch
        nothing
    end
    if cuda === nothing
        @test_skip cuda !== nothing
    else
        backend = Base.invokelatest(getproperty(cuda, :CUDABackend))
        light, dark, img = signed_grid(4)
        for im in (img, Float16.(img), sharp_disc(47.0, 48.6, 1.0)), pol in (:none, :light, :dark)
            c = detect_sfop(im; device = KA.CPU(), noise_sigma = 0.5, polarity = pol)
            g = Base.invokelatest(detect_sfop, im; device = backend, noise_sigma = 0.5, polarity = pol)
            @test length(c) == length(g)
            @test maximum(norm(a.center - b.center) for (a, b) in zip(c, g); init = 0.0) < 1.0e-3
            seeded(dev) = Base.invokelatest(detect_foerstner, Float32, im, c; device = dev, noise = 0.5, polarity = pol)
            @test length(seeded(KA.CPU())) == length(seeded(backend))
        end
    end
end
