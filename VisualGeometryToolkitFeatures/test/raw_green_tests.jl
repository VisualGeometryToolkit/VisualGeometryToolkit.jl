# detect_sfop and the seeded detect_foerstner on a RawMosaic: the bilinear green image on the raw grid (raw_green.jl),
# its photosite noise, and dots rendered on a Bayer mosaic of known centres against the green cells' path.

@testsetup module BayerDots
using Random: Xoshiro, rand
using LinearAlgebra: norm, tr
using Statistics: mean, std, cov
import KernelAbstractions as KA
using VisualGeometryToolkitCore: LinearRaw
using VisualGeometryToolkitFeatures: detect_sfop, detect_foerstner
export dot_scene, dot_centres, dot_study

# Dots of radius `radius` (raw px) at COLMAP `centres` on an H × W raw grid, in mean electrons per photosite: each
# photosite's coverage by the disc (4 × 4 supersampled: the photosite's box), blurred by a Gaussian PSF of `psf` raw px,
# between `background` and `plateau`. Every photosite records the same scene (a grey target).
function dot_scene(H, W, centres, radius; psf = 1.0, background = 600.0, plateau = 6000.0, ss = 4)
    disc = zeros(H, W)
    for (a, b) in centres
        for c in max(1, floor(Int, a - radius) - 1):min(W, ceil(Int, a + radius) + 2),
                r in max(1, floor(Int, b - radius) - 1):min(H, ceil(Int, b + radius) + 2)
            disc[r, c] += sum(hypot(c - 1 + (i - 0.5) / ss - a, r - 1 + (j - 0.5) / ss - b) < radius for i in 1:ss, j in 1:ss) / ss^2
        end
    end
    t = ceil(Int, 4psf)
    k = [exp(-x^2 / (2psf^2)) for x in -t:t]; k ./= sum(k)
    blur(a, d) = [sum(k[s + t + 1] * (d == 1 ? a[clamp(i + s, 1, H), j] : a[i, clamp(j + s, 1, W)]) for s in -t:t) for i in 1:H, j in 1:W]
    return background .+ (plateau - background) .* blur(blur(disc, 1), 2)
end

# An n × n grid of dot centres at pitch 30 raw px, one tile of margin: random sub-pixel phases, or `phase` in every tile.
dot_centres(n; same_phase = false, phase = (0.31, 0.77), seed = 1) = (rng = Xoshiro(seed);
    [same_phase ? (30tx + 15 + phase[1], 30ty + 15 + phase[2]) : (30tx + 15 + rand(rng) - 0.5, 30ty + 15 + rand(rng) - 0.5)
     for ty in 1:n for tx in 1:n])

# SFOP then the seeded Förstner on `m` under `model`, on the raw grid (`path = :raw`) or the green cells (`:cells`,
# positions and covariances mapped to raw pixels, x ↦ 2x): each centre's nearest :circular row within 1 px — the
# error's mean (`bias`), RMS about the truth, spread about the mean and the predicted σ (√(tr Σ/2), RMS over the rows),
# the NEES about the mean (one geometry per tile: the noise's spread, not the estimator's bias) and the χ²₂ 95 %
# coverage — and the :circular rows farther than 2 px from every centre (`wrong`).
function dot_study(m, model, centres; path = :raw, polarity = :none, T = Float64, device = KA.CPU())
    img, f = path === :raw ? (m, 1.0) : (LinearRaw(m; mode = :green_cells, sensor_model = model), 2.0)
    seeds = detect_sfop(img; sensor_model = model, device, polarity)
    rows = filter(d -> d.label === :circular, detect_foerstner(T, img, seeds; sensor_model = model, device, polarity))
    pos = [f .* d.center for d in rows]
    Σs = [f^2 .* Matrix(cov(d.value)) for d in rows]
    errs, S = Vector{Float64}[], Matrix{Float64}[]
    for c in centres
        isempty(pos) && break
        i = argmin([norm(p .- c) for p in pos])
        norm(pos[i] .- c) < 1 || continue
        push!(errs, collect(pos[i] .- c)); push!(S, Σs[i])
    end
    wrong = count(p -> minimum(norm(p .- c) for c in centres) > 2, pos)
    E = reduce(hcat, errs); μ = vec(mean(E; dims = 2))
    q = [(x = E[:, k] .- μ; x' * (S[k] \ x)) for k in axes(E, 2)]
    return (; found = size(E, 2), wrong, bias = μ, rms = sqrt(mean(sum(abs2, E; dims = 1))), spread = sqrt(sum(abs2, std(E; dims = 2)) / 2),
            predicted = sqrt(mean(tr.(S)) / 2), nees = mean(q), cov95 = mean(q .< 5.991464547107979))
end
end

@testitem "the bilinear green image: a constant field, a ramp, a saturated green censored, a defect missing (its 3×3 fallback), a defect cluster without a value" setup = [NoisyMosaic] begin
    import KernelAbstractions as KA
    using VisualGeometryToolkitCore: RawMosaic, SensorModel, sensor_fixture, parse_unit, linearview, value_variance
    const VGF = VisualGeometryToolkitFeatures
    mosaic(raw) = RawMosaic(raw, RGGB, (0.0, 0.0, 0.0, 0.0), 65535.0, (0, 0), size(raw), (; model = "synthetic"))
    green(i, j) = RGGB[mod1(i, 2), mod1(j, 2)] in (0x01, 0x03)
    H, W = 20, 24
    # a constant green field: a constant image, no ripple (red and blue photosites never read)
    raw = [green(i, j) ? UInt16(30000) : UInt16(rand(100:60000)) for i in 1:H, j in 1:W]
    g = VGF._bilinear_green(Float64, KA.CPU(), mosaic(raw), SensorModel())
    @test all(==(30000 / 65535), g.value) && count(>(0), g.photosite) == 0 && all(g.support .>= 1) && !any(g.saturated)
    # a ramp in the greens' codes: reproduced at every pixel whose four greens lie in the image
    ramp(i, j) = 1000.0 + 7i + 11j
    g = VGF._bilinear_green(Float64, KA.CPU(), mosaic([UInt16(ramp(i, j)) for i in 1:H, j in 1:W]), SensorModel())
    @test all(isapprox(g.value[i, j], ramp(i, j) / 65535; rtol = 1.0e-14) for i in 2:(H - 1), j in 2:(W - 1))
    @test g.value[1, 1] ≈ (ramp(1, 2) + ramp(2, 1)) / 2 / 65535        # a corner red: its two greens, renormalised
    # a saturated green (8, 7): CENSORED, a sample at 1, flagged with every pixel that reads it; a defect green (8, 5):
    # MISSING, its value the mean of its four diagonal greens; a defect cluster — the green (14, 15) and its four diagonal
    # greens — leaves (14, 15) alone without a value (its diagonal greens fall back on theirs)
    raw = fill(UInt16(20000), H, W)
    raw[8, 7] = 65535
    raw[7, 6] = raw[9, 6] = 10000
    raw[8, 5] = 50000                                                  # a defect's value is never read
    s = parse_unit("s")
    cluster = ((14, 15), (13, 14), (15, 14), (13, 16), (15, 16))
    defects = (x = [5; [c for (_, c) in cluster]], y = [8; [r for (r, _) in cluster]], onset = fill(0.001s, 6), size = (H, W))
    model = sensor_fixture(; read_noise = (3.0, 3.0, 3.0, 3.0), defects).model
    g = VGF._bilinear_green(Float64, KA.CPU(), mosaic(raw), model)
    @test g.value[8, 7] == 1 && g.saturated[8, 7] && g.saturated[8, 6] && g.saturated[8, 8] && !g.saturated[8, 9]
    @test g.value[8, 6] ≈ (10000 + 10000 + 65535) / 3 / 65535         # (8, 5) dropped, (8, 7) censored at 1
    @test g.value[8, 5] ≈ (20000 + 10000 + 20000 + 10000) / 4 / 65535 # the 3×3 fallback: (7, 4), (9, 4), (7, 6), (9, 6)
    @test g.support[8, 5] == (0x04 | VGF._SUPPORT_3X3) && g.support[8, 6] == 0x03 && g.support[8, 9] == 0x01
    @test isnan(g.value[14, 15]) && g.support[14, 15] == 0 && count(isnan, g.value) == 1
    @test g.support[13, 14] == (0x03 | VGF._SUPPORT_3X3)               # a cluster's green: three usable diagonal greens
    # a window of the mosaic (an odd offset: another colour phase): the defects are the visible raw's photosites
    w = VGF._bilinear_green(Float64, KA.CPU(), view(mosaic(raw), 3:H, 2:W), model)
    @test isnan(w.value[14 - 2, 15 - 1]) && count(isnan, w.value) == 1 && w.value[8 - 2, 5 - 1] == g.value[8, 5]
    # under a variance law the censored photosite carries the law's variance at 1 (never 0)
    pg = sensor_fixture(; gain = 2.5, read_noise = (3.0, 3.0, 3.0, 3.0)).model
    g = VGF._bilinear_green(Float64, KA.CPU(), mosaic(raw), pg)
    @test g.photosite[8, 7] ≈ value_variance(linearview(mosaic(raw); sensor_model = pg), 8, 7, 1.0) > 0
    @test g.photosite[8, 5] > 0 && g.photosite[8, 6] == 0              # no defect here; a blue site carries none
end

@testitem "the bilinear green image: its variance is Σw²σ²/(Σw)² of the photosites (Monte Carlo)" setup = [NoisyMosaic] begin
    import KernelAbstractions as KA
    using Random: Xoshiro
    using Statistics: var, mean
    const VGF = VisualGeometryToolkitFeatures
    model = pg_model(2.5, 3.0)
    rng = Xoshiro(3)
    n = 4000
    draws = [VGF._bilinear_green(Float64, KA.CPU(), noisy_mosaic(rng, (i, j) -> 3000.0, 6, 6; electrons_per_dn = 2.5, read_dn = 3.0), model)
             for _ in 1:n]
    # Var Ī = Σw²σ²/(Σw)² = density/(2n) (the white-equivalent density twice the photosites' mean variance under the
    # cross, n its usable greens), the law's at each draw's value, averaged, against the draws' variance at a green site
    # (σ²) and at a red and a blue one (≈ σ²/4); n draws: the sample variance's relative standard error √(2/n)
    varĪ(d, i, j) = d.density[i, j] / (2 * VGF._support_count(d.support[i, j]))
    for (i, j) in ((3, 4), (3, 3), (4, 4))
        empirical = var([d.value[i, j] for d in draws])
        @test abs(empirical / mean(varĪ(d, i, j) for d in draws) - 1) < 4 * sqrt(2 / n)
    end
    @test 3.5 < mean(varĪ(d, 3, 4) / varĪ(d, 3, 3) for d in draws) < 4.5
    @test all(d.photosite[3, 4] > 0 && d.photosite[3, 3] == 0 for d in draws)
end

@testitem "the bilinear green image: G1 and G2 taken alike, an imbalance left as a checkerboard (the known limit)" setup = [NoisyMosaic] begin
    import KernelAbstractions as KA
    using VisualGeometryToolkitCore: RawMosaic, SensorModel
    const VGF = VisualGeometryToolkitFeatures
    # G1 (colour index 1) 1 % brighter than G2 (index 3): the sensor model carries no per-channel response, so nothing
    # equalises them — the image holds the two levels at the green sites and their mean at the red and blue ones
    H, W = 12, 12
    code(i, j) = (c = RGGB[mod1(i, 2), mod1(j, 2)]; c == 0x01 ? 20200 : c == 0x03 ? 20000 : 5000)
    g = VGF._bilinear_green(Float64, KA.CPU(), RawMosaic(UInt16[code(i, j) for i in 1:H, j in 1:W], RGGB, (0.0, 0.0, 0.0, 0.0),
                                                         65535.0, (0, 0), (H, W), (; model = "synthetic")), SensorModel())
    interior = [g.value[i, j] * 65535 for i in 2:(H - 1), j in 2:(W - 1)]
    @test sort(unique(round.(interior; digits = 6))) == [20000.0, 20100.0, 20200.0]
end

@testitem "detect_foerstner on a RawMosaic: the F-test's degrees of freedom under the photosite noise (Monte Carlo)" begin
    import KernelAbstractions as KA
    using VisualGeometryToolkitCore: RawMosaic, SensorModel, Disc, Point2
    using Random: Xoshiro, randn
    using Statistics: mean, var
    const VGF = VisualGeometryToolkitFeatures
    # pure noise in the codes (every photosite white): the bilinear image's windows at ρ = 12 (σ = 3), Ω = Iᵀ Q I of the
    # green photosites; ν = 2E²/Var against `_residual_dof` under that noise, and below the white input pixels' ν
    N = 1200
    raw = round.(UInt16, 8000 .+ 40 .* randn(Xoshiro(5), N, N))
    g = VGF._bilinear_green(Float64, KA.CPU(), RawMosaic(raw, UInt8[0 1; 3 2], (0.0, 0.0, 0.0, 0.0), 65535.0, (0, 0), (N, N), (;)),
                            SensorModel()).value
    doubled = zeros(2N, 2N)
    VGF._launch!(VGF._upsample_doubled!, KA.CPU(), doubled, reshape(g, N, N, 1); ndrange = size(doubled))
    seeds = [Detection(Disc(Point2(x, y), 3.0), (; precision = 1.0)) for x in 50:40:(N - 50) for y in 50:40:(N - 50)]
    f = VGF._foerstner_seeded(KA.CPU(), doubled, seeds; derivative = :gaussian2d, derivative_sigma = 1.0)
    @test f.ρ[1] == 12 && length(seeds) == 784
    ν(x) = 2mean(x)^2 / var(x)
    bilinear = VGF._residual_dof(f.taps, 12, VGF._BilinearGreenStructure())
    white = VGF._residual_dof(f.taps, 12)
    # m = 784 windows (spaced past their reach): ν̂'s relative standard error ≈ √(2/m)·√(1 + 12/ν) ≈ 7 %; the circular
    # sum's ν is line 423's, measured lower as for white noise (the test conservative)
    @test abs(ν(f.estimates[7, :]) / bilinear.corner.ν - 1) < 0.15
    @test ν(f.estimates[14, :]) < bilinear.circular.ν * 1.15
    @test bilinear.corner.ν < 0.9 * white.corner.ν && bilinear.circular.ν < 0.9 * white.circular.ν
    # white noise: the pair (I, I), today's numbers
    @test white.corner.ν ≈ 33.796464446324045
end

@testitem "detect_sfop → detect_foerstner on a RawMosaic: the dots, calibrated (NEES ≈ 2), more precise than the green cells" setup = [NoisyMosaic, BayerDots] begin
    import KernelAbstractions as KA
    using Random: Xoshiro
    using LinearAlgebra: norm
    n = 16
    centres = dot_centres(n; same_phase = true)
    e = dot_scene(30(n + 2), 30(n + 2), centres, 1.5)
    model = pg_model(2.5, 3.0)
    m = noisy_mosaic(Xoshiro(107), (i, j) -> e[i, j], size(e)...; electrons_per_dn = 2.5, read_dn = 3.0)
    raw = dot_study(m, model, centres)
    cells = dot_study(m, model, centres; path = :cells)
    @test raw.found == cells.found == n^2
    # the photosite noise propagated: NEES and coverage within 4 standard errors of the sampling
    @test abs(raw.nees - 2) < 4 * 2 / sqrt(raw.found)
    @test abs(raw.cov95 - 0.95) < 4 * sqrt(0.95 * 0.05 / raw.found)
    @test 0.85 < raw.spread / raw.predicted < 1.15
    # the raw grid's dots: more precise than the green cells', no more biased
    @test raw.spread < 0.8 * cells.spread
    @test norm(raw.bias) < norm(cells.bias) + 0.005
    # the windows centred between four dots are circular points of that symmetry (Förstner's model): the raw grid's
    # finer windows label some; the polarity gate keeps the dots alone
    lit = dot_study(m, model, centres; polarity = :light)
    @test lit.found == n^2 && lit.wrong == 0
end

@testitem "detect_sfop → detect_foerstner on a RawMosaic: a dot with a saturated core found and localised, a defective green in a dot's window harmless" setup = [NoisyMosaic, BayerDots] begin
    import KernelAbstractions as KA
    using VisualGeometryToolkitCore: sensor_fixture, parse_unit
    using Random: Xoshiro
    using LinearAlgebra: norm
    n = 8
    centres = dot_centres(n; seed = 3)
    model = pg_model(2.5, 3.0)
    # cores past the white level (80 000 e⁻ at 2.5 e⁻ per code: the codes clip at 16 383): every dot found, :circular,
    # within a tenth of a pixel (the clipped plateau carries no gradient; the rim localises it)
    e = dot_scene(30(n + 2), 30(n + 2), centres, 2.5; plateau = 80000.0)
    m = noisy_mosaic(Xoshiro(5), (i, j) -> e[i, j], size(e)...; electrons_per_dn = 2.5, read_dn = 3.0)
    @test count(>=(WHITE), parent(m)) > 4n^2                           # the cores saturate
    s = dot_study(m, model, centres)
    @test s.found == n^2 && s.rms < 0.1
    # a defective green photosite within a pixel of each dot's centre (the worst place for one): every dot still found, the
    # error within a few hundredths of a pixel more (the 3×3 fallback reads the diagonal greens, off the dot's peak:
    # measured 0.03 → 0.09 px RMS at radius 1.5)
    e = dot_scene(30(n + 2), 30(n + 2), centres, 1.5)
    m = noisy_mosaic(Xoshiro(6), (i, j) -> e[i, j], size(e)...; electrons_per_dn = 2.5, read_dn = 3.0)
    green(r, c) = RGGB[mod1(r, 2), mod1(c, 2)] in (0x01, 0x03)
    sites = [first(p for p in ((round(Int, y + 0.5), round(Int, x + 0.5)), (round(Int, y + 0.5), round(Int, x + 0.5) + 1)) if green(p...))
             for (x, y) in centres]
    sec = parse_unit("s")
    defective = sensor_fixture(; gain = 2.5, read_noise = (3.0, 3.0, 3.0, 3.0),
                               defects = (x = last.(sites), y = first.(sites), onset = fill(0.001sec, length(sites)), size = size(e))).model
    clean, holed = dot_study(m, model, centres), dot_study(m, defective, centres)
    @test holed.found == clean.found == n^2 && holed.rms < 0.15
end

@testitem "detect_sfop and detect_foerstner on a RawMosaic: the keywords, the sensor model, the image-wide search refused" setup = [NoisyMosaic, BayerDots] begin
    import KernelAbstractions as KA
    using VisualGeometryToolkitCore: SensorModel, sensor_fixture
    import VisualGeometryToolkitCore
    using Statistics: cov
    using Random: Xoshiro
    centres = dot_centres(3)
    e = dot_scene(150, 150, centres, 2.0)
    m = noisy_mosaic(Xoshiro(2), (i, j) -> e[i, j], 150, 150; electrons_per_dn = 2.5, read_dn = 3.0)
    model = pg_model(2.5, 3.0)
    seeds = detect_sfop(m; sensor_model = model, device = KA.CPU())
    rows = detect_foerstner(m, seeds; sensor_model = model, device = KA.CPU())
    @test eltype(cov(first(rows).value)) === Float64
    @test all(d -> d.covariance_status in (:propagated_measured, :unclassified), rows) && count(d -> d.label === :circular, rows) >= 9
    # no gain: the a-posteriori covariance; SFOP's noise_sigma
    for nogain in (SensorModel(), sensor_fixture(; read_noise = (3.0, 3.0, 3.0, 3.0)).model)
        s = detect_sfop(m; sensor_model = nogain, device = KA.CPU())
        @test all(d -> d.covariance_status in (:a_posteriori, :unclassified), detect_foerstner(m, s; sensor_model = nogain, device = KA.CPU()))
        @test_throws ArgumentError detect_foerstner(m, s; sensor_model = nogain, device = KA.CPU(), polarity = :light)
    end
    # the image's own blur and upsampling, and the image-wide search, refused
    @test_throws ArgumentError detect_sfop(m; sensor_model = model, device = KA.CPU(), input_blur = 0.5)
    @test_throws ArgumentError detect_sfop(m; sensor_model = model, device = KA.CPU(), upsample = false)
    @test_throws ArgumentError detect_foerstner(m; sensor_model = model)
    @test_throws ArgumentError detect_foerstner(Float64, m; sensor_model = model)
    @test_throws ArgumentError detect_foerstner(m, seeds; sensor_model = model, device = KA.CPU(), integration_sigma = 2.0)
    # a cell whose greens share a row is no quincunx: refused (the cross, the blur and the F-test assume one)
    rows_green = VisualGeometryToolkitCore.RawMosaic(parent(m), UInt8[1 3; 0 2], m.black, m.white, (0, 0), size(m), (;))
    @test_throws ArgumentError detect_sfop(rows_green; device = KA.CPU())
end

# GPU: the same rows as the CPU, within Float32 rounding. Runs only with a functional CUDA device.
@testitem "detect_sfop and detect_foerstner on a RawMosaic: CUDA matches the CPU" setup = [NoisyMosaic, BayerDots] begin
    import KernelAbstractions as KA
    using LinearAlgebra: norm
    using Statistics: cov
    using Random: Xoshiro
    cuda = try
        c = Base.require(Main, :CUDA)
        (c !== nothing && Base.invokelatest(getproperty(c, :functional))) ? c : nothing
    catch
        nothing
    end
    if cuda === nothing
        @test_skip "no functional CUDA device"
    else
        gpu = Base.invokelatest(getproperty(cuda, :CUDABackend))
        centres = dot_centres(6)
        e = dot_scene(240, 240, centres, 1.5)
        m = noisy_mosaic(Xoshiro(4), (i, j) -> e[i, j], 240, 240; electrons_per_dn = 2.5, read_dn = 3.0)
        model = pg_model(2.5, 3.0)
        on(device) = (s = detect_sfop(m; sensor_model = model, device); (s, detect_foerstner(Float32, m, s; sensor_model = model, device)))
        sc, rc = on(KA.CPU())
        sg, rg = Base.invokelatest(on, gpu)
        @test length(sg) == length(sc) && all(norm(a.center .- b.center) < 1.0e-3 for (a, b) in zip(sg, sc))
        @test length(rg) == length(rc) && all(a.label === b.label for (a, b) in zip(rg, rc))
        for (a, b) in zip(rg, rc)
            a.label === :unknown && continue
            @test norm(a.center .- b.center) < 1.0e-3
            @test isapprox(Matrix(cov(a.value)), Matrix(cov(b.value)); rtol = 1.0e-3)
        end
    end
end
