# Sensor noise (raw/sensor_noise.jl) on SYNTHETIC Poisson–Gaussian RGGB mosaics: the photon-transfer estimator of K and
# σ_r, the dark-frame estimator, the resolver's four cases with their provenance, the body match, and THE black
# subtraction of LinearRaw / linearview with a resolved noise (bit-identical without one, CPU = GPU with a dark map).
#
# Every bound is the estimator's own standard error (or a first-principles count) times Z, the two-sided normal
# quantile at a 0.1% false-failure rate per comparison (Z_N: the same over N photosites at once, Bonferroni). The
# frames are seeded, so a run is deterministic; Z states how unlikely a failure of a correct estimator would be.

@testsetup module NoiseFixtures
    using VisualGeometryToolkitCore: RawMosaic
    using Distributions: Poisson, Normal, quantile
    export RGGB, BLACK, WHITE, K_TRUE, SIGMA_R, SIGMA_EFF, LEVELS, Z, z_over, synth_mosaic, code, ptc_frame
    const RGGB = UInt8[0 1; 3 2]
    const BLACK = 512.0
    const WHITE = 16383.0
    const K_TRUE = 1.6                                  # electrons per code
    const SIGMA_R = 2.5                                 # the Gaussian read noise, codes
    const SIGMA_EFF = sqrt(SIGMA_R^2 + 1 / 12)          # with the code's rounding (uniform, variance 1/12: Sheppard)
    z_over(N; α = 0.001) = quantile(Normal(), 1 - α / (2N))   # two-sided, α over N comparisons
    const Z = z_over(1)
    synth_mosaic(raw; black = (BLACK, BLACK, BLACK, BLACK)) =
        RawMosaic(Matrix{UInt16}(raw), RGGB, black, WHITE, (0, 0), size(raw), (; model = "synthetic"))
    # One code: an offset (the black, or a photosite's own), the photo-electrons Poisson(μK)/K (`pois`, or none) and a
    # Gaussian read noise σ, rounded and clipped to [0, white].
    code(rng, offset, pois, σ) =
        UInt16(clamp(round(offset + (pois === nothing ? 0.0 : rand(rng, pois) / K_TRUE) + σ * randn(rng)), 0, WHITE))
    # 16 flat levels above black, geometric from 2 to 12000 codes: the low ones pin σ_r², the high ones K.
    const LEVELS = [2.0 * 6000.0^((l - 1) / 15) for l in 1:16]
    # A photon-transfer scene: 16 × 32 blocks of the estimator's 8 × 8 cells (16 × 16 photosites), block column bj
    # flat at LEVELS[cld(bj, 2)] — 32 blocks per level, so the estimator's 32 groups of 16 blocks hold one level each.
    # `offset` (a per-photosite black, a DSNU) replaces the constant black.
    function ptc_frame(rng; offset = nothing)
        H, W = 16 * 16, 32 * 16
        pois = [Poisson(μ * K_TRUE) for μ in LEVELS]
        raw = Matrix{UInt16}(undef, H, W)
        for j in 1:W, i in 1:H
            raw[i, j] = code(rng, offset === nothing ? BLACK : offset[i, j], pois[cld(cld(j, 16), 2)], SIGMA_R)
        end
        return synth_mosaic(raw)
    end
end

@testitem "sensor noise: photon_transfer recovers K and σ_r within its standard errors" setup = [NoiseFixtures] begin
    using VisualGeometryToolkitCore: photon_transfer
    using StableRNGs: StableRNG
    using Distributions: Chisq, quantile
    using Statistics: mean, var
    R = 24
    ests = [photon_transfer(ptc_frame(StableRNG(100 + r))) for r in 1:R]
    e = ests[1]
    @test e.blocks == 16 * 32
    # one frame: the truth within Z of its returned standard errors (σ_r's truth with the rounding)
    @test abs(e.electrons_per_dn - K_TRUE) <= Z * e.electrons_per_dn_std_error
    @test abs(e.read_noise - SIGMA_EFF) <= Z * e.read_noise_std_error
    # The standard errors (the WLS line's, from the counts: each group's median of m blocks of n cells, v̂/v ~
    # χ²_{n−1}/(n−1)) against a seeded Monte Carlo of R frames: if they are right, the spread of the R estimates over
    # the mean squared standard error is χ²_{R−1}/(R−1), two-sided at 0.1%; and the mean of the R estimates lies
    # within Z standard errors of the mean (no bias at the Monte Carlo's resolution).
    lo, hi = quantile(Chisq(R - 1), 0.0005) / (R - 1), quantile(Chisq(R - 1), 0.9995) / (R - 1)
    for (val, se, truth) in ((:electrons_per_dn, :electrons_per_dn_std_error, K_TRUE), (:read_noise, :read_noise_std_error, SIGMA_EFF))
        xs = getproperty.(ests, val)
        s2 = mean(getproperty.(ests, se) .^ 2)
        @test lo <= var(xs) / s2 <= hi
        @test abs(mean(xs) - truth) <= Z * sqrt(s2 / R)
    end
    @test_throws ArgumentError photon_transfer(ptc_frame(StableRNG(1)); block = 1)
end

@testitem "sensor noise: master_dark — the dark map, black, read noise and the defects" setup = [NoiseFixtures] begin
    using VisualGeometryToolkitCore: master_dark, DarkFrame, photosite_colors, parse_unit, ustrip
    using StableRNGs: StableRNG
    using Statistics: std
    s = parse_unit("s")
    rng = StableRNG(7)
    H, W, n = 128, 128, 16
    black = (512.0, 500.0, 520.0, 508.0)                # per colour index (0 R, 1 G, 2 B, 3 G2)
    σr = (2.5, 2.3, 2.7, 2.4)
    σeff = sqrt.(σr .^ 2 .+ 1 / 12)
    s_dsnu = 3.0                                        # the fixed-pattern offsets' spread, codes
    col = collect(photosite_colors(synth_mosaic(zeros(UInt16, H, W))))
    offset = [black[col[i, j] + 1] + s_dsnu * randn(rng) for i in 1:H, j in 1:W]
    σ = [σr[col[i, j] + 1] for i in 1:H, j in 1:W]
    planted = falses(H, W)
    sites = unique([CartesianIndex(rand(rng, 1:H), rand(rng, 1:W)) for _ in 1:12])
    for (q, p) in enumerate(sites)
        planted[p] = true
        q <= 8 ? (offset[p] += 300) : (offset[p] = 0.0; σ[p] = 0.0)   # hot, or dead (stuck at code 0)
    end
    frames = [synth_mosaic([code(rng, offset[i, j], nothing, σ[i, j]) for i in 1:H, j in 1:W]; black) for _ in 1:n]
    r = master_dark(frames; exposure_band = (0.5s, 1.0s))
    @test r.dark isa DarkFrame && size(r.dark.offset) == (H, W) && r.dark.frames == n
    @test map(t -> ustrip(s, t), r.dark.exposure_band) == (0.5, 1.0)
    # the defects exactly (hot +300 codes and dead are far past z·σ; no false defect at the 1% expected over the sensor)
    @test r.defects == planted
    good = .!planted
    # the offset map: per photosite the mean of n codes, σ_eff/√n about its offset, at Z_N over the N photosites (plus
    # the Float32 storage's half-ulp at the offset's magnitude)
    N = count(good)
    ok = [abs(r.dark.offset[p] - offset[p]) <= z_over(N) * σeff[col[p] + 1] / sqrt(n) + eps(Float32(offset[p])) / 2
          for p in findall(good)]
    @test all(ok)
    for c in 0:3
        sel = good .& (col .== c)
        Nc = count(sel)
        # the black: the channel's population black (the DSNU drawn about it) within Z of its standard error
        @test abs(r.black[c + 1] - black[c + 1]) <= Z * r.black_std_error[c + 1]
        # that standard error against its first-principles count: the offsets' spread √(s_dsnu² + σ_eff²/n) over √Nc,
        # within the sample standard deviation's own relative error, Z/√(2(Nc − 1))
        se0 = sqrt(s_dsnu^2 + σeff[c + 1]^2 / n) / sqrt(Nc)
        @test abs(r.black_std_error[c + 1] / se0 - 1) <= Z / sqrt(2 * (Nc - 1))
        # the read noise (with the rounding) within Z of its standard error, √(2/((n − 1)Nc))·σ²/(2σ)
        @test abs(r.read_noise[c + 1] - σeff[c + 1]) <= Z * r.read_noise_std_error[c + 1]
        @test r.read_noise_std_error[c + 1] ≈ sqrt(2 / ((n - 1) * Nc)) * r.read_noise[c + 1] / 2
    end
    @test_throws ArgumentError master_dark(frames[1:1]; exposure_band = (0.5s, 1.0s))
    @test_throws ArgumentError master_dark([frames[1], synth_mosaic(zeros(UInt16, 4, 4))]; exposure_band = (0.5s, 1.0s))
    @test_throws ArgumentError master_dark(frames; exposure_band = (1.0s, 0.5s))
end

@testitem "sensor noise: the resolver's four cases, each with its provenance" setup = [NoiseFixtures] begin
    using VisualGeometryToolkitCore
    using VisualGeometryToolkitCore: noise_value, NoiseEstimateFailed, parse_unit, value_of, photon_transfer
    using StableRNGs: StableRNG
    s = parse_unit("s")
    src(f) = f.attributes.source
    body = DeviceIdentity(; make = "SONY", model = "ILME-FX3", serial = "5012345")
    frame(t, iso) = (; exposure = ExposureSettings(t, 8, iso), lens_settings = LensSettings(14mm))
    cap = CaptureSettings(Dict(1 => frame(0.8s, 100), 2 => frame(0.002s, 100), 3 => frame(0.8s, 400)); body)
    rng = StableRNG(11)
    s_dsnu = 3.0
    offset = BLACK .+ s_dsnu .* randn(rng, 256, 512)
    m_dark = ptc_frame(rng; offset)                     # a frame with a fixed-pattern offset (DSNU)
    m = ptc_frame(StableRNG(12))                        # a frame without
    dark = DarkFrame(Float32.(offset), fill(Float32(SIGMA_EFF^2), size(offset)), (0.5s, 1.0s), 16)
    mask = falses(size(offset)); mask[3, 5] = true
    rec_black = noise_value((BLACK, BLACK, BLACK, BLACK), :measured; std_error = (0.01, 0.01, 0.01, 0.01))
    full = SensorNoise(; body, iso = 100, electrons_per_dn = noise_value(1.62, :measured; std_error = 0.01),
                       read_noise = noise_value((2.5, 2.5, 2.5, 2.5), :measured), black = rec_black,
                       dark = noise_value(dark, :measured), defects = noise_value(mask, :measured))

    # 1. a record of this body at this ISO: its values as recorded, all :measured; its dark only inside its band
    r = sensor_noise(cap, 1 => m_dark; record = full)
    @test r.body == body && r.iso == 100 && r.reason === nothing
    @test r.electrons_per_dn === full.electrons_per_dn && r.read_noise === full.read_noise && r.black === full.black
    @test r.dark === full.dark && r.defects === full.defects
    @test all(f -> src(f) === :measured, (r.electrons_per_dn, r.read_noise, r.black, r.dark, r.defects))
    r2 = sensor_noise(cap, 2 => m_dark; record = [full])     # 1/500 s: outside the dark's band
    @test r2.dark === nothing && r2.electrons_per_dn === full.electrons_per_dn

    # 2. a partial record (dark frames, no gain): what it holds, the rest estimated with the dark subtracted
    partial = SensorNoise(; body, iso = 100, black = rec_black, dark = noise_value(dark, :measured), defects = noise_value(mask, :measured))
    r = sensor_noise(cap, 1 => m_dark; record = partial)
    @test src(r.black) === :measured && src(r.dark) === :measured && src(r.defects) === :measured
    @test src(r.electrons_per_dn) === :estimated && src(r.read_noise) === :estimated && r.reason === nothing
    K, σ = value_of(r.electrons_per_dn), value_of(r.read_noise)
    @test abs(K - K_TRUE) <= Z * r.electrons_per_dn.std_error
    @test all(abs.(σ .- SIGMA_EFF) .<= Z .* r.read_noise.std_error)
    # without the dark, the fixed pattern reads as read noise: √(σ_eff² + s_dsnu²), far outside the bound
    e = photon_transfer(m_dark)
    @test e.read_noise - SIGMA_EFF > Z * e.read_noise_std_error

    # 3. no record (or none of this ISO): K and σ_r estimated, the file's black, no dark, no defects
    for (t, rec) in ((1, nothing), (3, full))
        r = sensor_noise(cap, t => m; record = rec)
        @test src(r.electrons_per_dn) === :estimated && src(r.read_noise) === :estimated
        @test r.black == noise_value(m.black, :file) && r.dark === nothing && r.defects === nothing && r.reason === nothing
        @test abs(value_of(r.electrons_per_dn) - K_TRUE) <= Z * r.electrons_per_dn.std_error
        @test r.iso == cap[t].exposure.iso
    end

    # 4. the estimate fails: the noise is nothing, with the reason; the black is still the file's
    for (raw, why) in ((fill(UInt16(WHITE), 64, 64), "too few usable blocks"), (fill(UInt16(1000), 256, 256), "one level"))
        r = sensor_noise(cap, 1 => synth_mosaic(raw))
        @test r.electrons_per_dn === nothing && r.read_noise === nothing
        @test occursin(why, r.reason) && src(r.black) === :file
        @test_throws NoiseEstimateFailed photon_transfer(synth_mosaic(raw))
    end
    @test_throws ArgumentError sensor_noise(cap, 9 => m)

    # the values' guards: a source among NOISE_SOURCES; a reason only with no noise
    @test_throws ArgumentError noise_value(1.0, :guessed)
    @test_throws ArgumentError SensorNoise(; body, iso = 100, electrons_per_dn = noise_value(-1.0, :measured))
    @test_throws ArgumentError SensorNoise(; body, iso = 100, electrons_per_dn = noise_value(1.0, :measured), reason = "no")
    @test_throws ArgumentError SensorNoise(; body, iso = 0)
end

@testitem "sensor noise: a record matches the body by serial, else make and model" setup = [NoiseFixtures] begin
    using VisualGeometryToolkitCore
    using VisualGeometryToolkitCore: noise_value, parse_unit
    using StableRNGs: StableRNG
    s = parse_unit("s")
    m = ptc_frame(StableRNG(3))
    fx3(serial) = DeviceIdentity(; make = "SONY", model = "ILME-FX3", serial)
    rec(b) = SensorNoise(; body = b, iso = 100, electrons_per_dn = noise_value(1.6, :measured),
                         read_noise = noise_value((2.5, 2.5, 2.5, 2.5), :measured))
    cap(b) = CaptureSettings(Dict(1 => (; exposure = ExposureSettings(0.8s, 8, 100), lens_settings = LensSettings(14mm))); body = b)
    measured(b, records) = sensor_noise(cap(b), 1 => m; record = records).electrons_per_dn.attributes.source === :measured
    @test measured(fx3("A"), rec(fx3("A")))                                   # the serial
    @test !measured(fx3("B"), rec(fx3("A")))                                  # another body of the model
    @test measured(fx3(nothing), rec(fx3("A")))                               # no serial stated: make and model
    @test measured(fx3("A"), rec(DeviceIdentity(; serial = "A")))             # a record by serial alone
    @test !measured(DeviceIdentity(; make = "CANON", model = "ILME-FX3", serial = "A"), rec(fx3("A")))   # a serial whose make disagrees
    @test !measured(DeviceIdentity(), rec(DeviceIdentity()))                  # nothing stated: no match
    @test_throws ArgumentError measured(fx3(nothing), [rec(fx3("A")), rec(fx3("B"))])   # two bodies of the model: ambiguous
end

@testitem "sensor noise: LinearRaw's one black subtraction — bit-identical without a dark, the dark per photosite, defects masked" setup = [NoiseFixtures] begin
    using VisualGeometryToolkitCore
    using VisualGeometryToolkitCore: noise_value, parse_unit, channelview, photosite_colors, bayer_channels
    using StableRNGs: StableRNG
    s = parse_unit("s")
    rng = StableRNG(5)
    H, W = 12, 16
    black = (512.0, 500.0, 520.0, 508.0)
    m = synth_mosaic(UInt16[rand(rng, 300:16000) for i in 1:H, j in 1:W]; black)
    parent(m)[5, 6] = UInt16(WHITE)
    col = collect(photosite_colors(m))
    body = DeviceIdentity()
    file_only = SensorNoise(; body, iso = 100, black = noise_value(m.black, :file))
    offset = Float32[black[col[i, j] + 1] + 3 * randn(rng) for i in 1:H, j in 1:W]
    defects = falses(H, W); defects[3, 3] = true; defects[8, 9] = true     # an R photosite (odd, odd), a G (even, odd)
    dark = DarkFrame(offset, fill(1.0f0, H, W), (0.5s, 1.0s), 8)
    with_dark = SensorNoise(; body, iso = 100, black = noise_value(black, :measured), dark = noise_value(dark, :measured),
                            defects = noise_value(defects, :measured))
    for w in (m, view(m, 3:12, 2:15), view(m, 2:11, 1:16)), mode in (:green_cells, :bin2), gain in (1.0, 2.5)
        L = LinearRaw(w; mode, gain)
        # no noise, or a noise holding only the file's black: bit for bit
        for noise in (nothing, file_only)
            Ln = LinearRaw(w; mode, gain, noise)
            @test Ln.image == L.image && Ln.saturated == L.saturated && Ln.geometry == L.geometry
            @test copy(linearview(w; noise)) == copy(linearview(w))
        end
        # the dark: (raw − offset) / (white − black[c]) per photosite, at the window's offset into the visible raw
        o = w.window_origin
        Ld = LinearRaw(w; mode, gain, noise = with_dark)
        lv = linearview(w; noise = with_dark)
        @test all(lv[i, j] == (Float64(w[i, j]) - Float64(offset[i + o[1], j + o[2]])) / (WHITE - black[col[i + o[1], j + o[2]] + 1])
                  for i in axes(w, 1), j in axes(w, 2))
        v = channelview(lv)
        names = bayer_channels(w)
        g = findall(n -> n in (:G1, :G2), collect(names))
        Gp = mode === :bin2 ? 2 : 1
        @test Ld.image[:, :, Gp] == (v[g[1], :, :] .+ v[g[2], :, :]) ./ 2 .* gain
        @test Ld.geometry.green_dn == L.geometry.green_dn                     # the span is the channel's, dark or not
        # the defects join the mask: a cell is masked when a photosite its planes read is defective
        sites = mode === :bin2 ? (1:4) : g
        # (channel k of cell (i, j) is photosite (2i − 1 + di, 2j − 1 + dj), k = di + 2dj + 1, at the window's offset)
        hit(i, j) = any(defects[2i - 1 + (k - 1) % 2 + o[1], 2j - 1 + (k - 1) ÷ 2 + o[2]] for k in sites)
        @test Ld.saturated == L.saturated .| [hit(i, j) for i in axes(L.saturated, 1), j in axes(L.saturated, 2)]
    end
    @test any(LinearRaw(m; mode = :bin2, noise = with_dark).saturated .& .!LinearRaw(m; mode = :bin2).saturated)
    # a map that does not reach the mosaic's photosites
    short = SensorNoise(; body, iso = 100, dark = noise_value(DarkFrame(offset[1:10, :], fill(1.0f0, 10, W), (0.5s, 1.0s), 8), :measured))
    @test_throws DimensionMismatch LinearRaw(m; mode = :green_cells, noise = short)
    @test_throws DimensionMismatch linearview(m; noise = short)
    @test_throws ArgumentError linearview(m; noise = 1.0)
end

@testitem "sensor noise: CPU = GPU — LinearRaw with a dark map, photon_transfer, master_dark" setup = [NoiseFixtures] begin
    using VisualGeometryToolkitCore
    using VisualGeometryToolkitCore: noise_value, parse_unit
    using StableRNGs: StableRNG
    using Adapt: adapt
    using CUDA: CUDA
    s = parse_unit("s")
    if CUDA.functional()
        rng = StableRNG(9)
        H, W = 12, 16
        m = synth_mosaic(UInt16[rand(rng, 300:16000) for i in 1:H, j in 1:W])
        offset = Float32.(BLACK .+ 3 .* randn(rng, H, W))
        defects = falses(H, W); defects[4, 7] = true
        noise = SensorNoise(; body = DeviceIdentity(), iso = 100, dark = noise_value(DarkFrame(offset, fill(1.0f0, H, W), (0.5s, 1.0s), 8), :measured),
                            defects = noise_value(defects, :measured))
        d = adapt(CUDA.CuArray, m)
        for (wh, wd) in ((m, d), (view(m, 3:12, 2:15), view(d, 3:12, 2:15))), mode in (:green_cells, :bin2)
            Lh = LinearRaw(wh; mode, noise)
            Ld = LinearRaw(wd; mode, noise)
            @test Ld.image isa CUDA.CuArray && Array(Ld.image) == Lh.image && Array(Ld.saturated) == Lh.saturated
            @test Array(copy(linearview(wd; noise))) == copy(linearview(wh; noise))
        end
        f = ptc_frame(StableRNG(21))
        @test photon_transfer(adapt(CUDA.CuArray, f)) == photon_transfer(f)
        frames = [synth_mosaic([code(rng, BLACK, nothing, SIGMA_R) for i in 1:32, j in 1:32]) for _ in 1:4]
        rh = master_dark(frames; exposure_band = (0.5s, 1.0s))
        rd = master_dark([adapt(CUDA.CuArray, f) for f in frames]; exposure_band = (0.5s, 1.0s))
        @test Array(rd.dark.offset) == rh.dark.offset && Array(rd.dark.variance) == rh.dark.variance
        @test rd.black == rh.black && rd.read_noise == rh.read_noise && rd.defects == rh.defects
    else
        @test_skip CUDA.functional()                         # no functional GPU here: recorded as Broken
    end
end
