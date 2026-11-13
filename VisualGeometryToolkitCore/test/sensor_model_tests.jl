# The sensor model (raw/sensor_model.jl, raw/sensor_estimators.jl) on SYNTHETIC Poisson–Gaussian RGGB mosaics: the
# estimators (photon_transfer, master_dark, defect_onsets), the resolver's cases, the body match, THE black subtraction
# of LinearRaw / linearview under a model (bit-identical with the absent model, CPU = GPU), the uncertain image (D11),
# and the model's identity. Every component comes from Core's one fixture helper, `sensor_fixture` (D13).
#
# Every bound is the estimator's own standard error (or a first-principles count) times Z, the two-sided normal
# quantile at a 0.1% false-failure rate per comparison (Z_N: the same over N photosites at once, Bonferroni). The
# frames are seeded, so a run is deterministic; Z states how unlikely a failure of a correct estimator would be.

@testsetup module SensorFixtures
    using VisualGeometryToolkitCore: RawMosaic
    using Distributions: Poisson, Normal, quantile
    export RGGB, BLACK, WHITE, K_TRUE, SIGMA_R, SIGMA_EFF, LEVELS, Z, z_over, synth_mosaic, code, ptc_frame, scene_frame
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
    # The photon-transfer scene of `ptc_frame` at `light` times its levels, each photosite's expected electrons times its
    # fixed response `response[i, j]` (a G1/G2 gain mismatch, a PRNU: 1 when `nothing`) and `texture[i, j]` (a scene:
    # 1 when `nothing`) — the frames of one exposure share the response and the texture, and differ only in their noise.
    function scene_frame(rng; levels = LEVELS, light = 1.0, response = nothing, texture = nothing, σ = SIGMA_R)
        H, W = 16 * 16, 32 * 16
        raw = Matrix{UInt16}(undef, H, W)
        for j in 1:W, i in 1:H
            μ = light * levels[cld(cld(j, 16), 2)] * (response === nothing ? 1.0 : response[i, j]) *
                (texture === nothing ? 1.0 : texture[i, j])
            raw[i, j] = code(rng, BLACK, Poisson(μ * K_TRUE), σ)
        end
        return synth_mosaic(raw)
    end
end

@testitem "sensor model: photon_transfer — an estimated PoissonGaussian within its standard errors" setup = [SensorFixtures] begin
    using VisualGeometryToolkitCore
    using VisualGeometryToolkitCore: photon_transfer, SensorSource, NoiseEstimateFailed, sensor_fixture, parse_unit
    using StableRNGs: StableRNG
    using Distributions: Chisq, quantile
    using Statistics: mean, var
    R = 24
    ests = [photon_transfer(ptc_frame(StableRNG(100 + r))) for r in 1:R]
    e = ests[1]
    @test e isa PoissonGaussian && e.source === SensorSource.estimated
    @test allequal(e.σ) && allequal(e.std_error.σ)      # the greens' read noise, stated for every colour
    # one frame: the truth within Z of its returned standard errors (σ_r's truth with the rounding)
    @test abs(e.gain - K_TRUE) <= Z * e.std_error.gain
    @test abs(e.σ[1] - SIGMA_EFF) <= Z * e.std_error.σ[1]
    # The standard errors (the WLS line's, from the counts: each group's median of m blocks of n cells, v̂/v ~
    # χ²_{n−1}/(n−1)) against a seeded Monte Carlo of R frames: if they are right, the spread of the R estimates over
    # the mean squared standard error is χ²_{R−1}/(R−1), two-sided at 0.1%; and the mean of the R estimates lies
    # within Z standard errors of the mean (no bias at the Monte Carlo's resolution).
    lo, hi = quantile(Chisq(R - 1), 0.0005) / (R - 1), quantile(Chisq(R - 1), 0.9995) / (R - 1)
    for (val, se, truth) in ((n -> n.gain, n -> n.std_error.gain, K_TRUE), (n -> n.σ[1], n -> n.std_error.σ[1], SIGMA_EFF))
        xs = val.(ests)
        s2 = mean(se.(ests) .^ 2)
        @test lo <= var(xs) / s2 <= hi
        @test abs(mean(xs) - truth) <= Z * sqrt(s2 / R)
    end
    # the model's black slot is what it subtracts: without the fixed pattern the DSNU reads as read noise,
    # √(σ_eff² + s_dsnu²), far outside the bound; with the Dark it is gone
    s = parse_unit("s")
    rng = StableRNG(11)
    offset = BLACK .+ 3.0 .* randn(rng, 256, 512)
    m_dark = ptc_frame(rng; offset)
    e0 = photon_transfer(m_dark)
    @test e0.σ[1] - SIGMA_EFF > Z * e0.std_error.σ[1]
    fx = sensor_fixture(; black = (BLACK, BLACK, BLACK, BLACK), exposure = 0.8s,
                        dark = (; offset = Float32.(offset), variance = fill(Float32(SIGMA_EFF^2), size(offset)),
                                exposure_band = (0.5s, 1.0s), frames = 16))
    @test fx.model.black isa Dark
    ed = photon_transfer(m_dark; sensor_model = fx.model)
    @test abs(ed.gain - K_TRUE) <= Z * ed.std_error.gain && abs(ed.σ[1] - SIGMA_EFF) <= Z * ed.std_error.σ[1]
    # the failures: a saturated frame, a frame of one level
    for (raw, why) in ((fill(UInt16(WHITE), 64, 64), "too few usable blocks"), (fill(UInt16(1000), 256, 256), "one level"))
        err = try photon_transfer(synth_mosaic(raw)); nothing catch x; x end
        @test err isa NoiseEstimateFailed && occursin(why, err.reason)
    end
    @test_throws ArgumentError photon_transfer(ptc_frame(StableRNG(1)); block = 1)
end

# Core 0.10.1: the photon-transfer series (frames at equal exposure: per green photosite the temporal variance, which
# cancels every fixed pattern), its intercept the darks' at each group's own exposure, its range derived from the data
# (Bonferroni searches), returned in the stack's FitResult; and the single-frame estimator's refusal of a μ² term (a
# G1/G2 mismatch and a PRNU on a bright flat). Every bound is Z times the estimate's own standard error (the file's rule).
@testitem "sensor model: photon_transfer(groups, darks) — the darks' intercept per exposure, the derived range, the FitResult" setup = [SensorFixtures] begin
    using VisualGeometryToolkitCore
    using VisualGeometryToolkitCore: photon_transfer, SensorSource, NoiseEstimateFailed, PhotonTransferMetadata, sensor_fixture,
        parse_unit, estimate, retcode, ReturnCode
    using StableRNGs: StableRNG
    refusal(f) = (try f(); nothing catch e; e end)
    s = parse_unit("s")
    # the fixed response: G2 (LibRaw index 3: even row, odd column of RGGB) 2 % above G1, and a 1 % PRNU everywhere
    rr = StableRNG(41)
    response = [(1 + 0.01 * randn(rr)) * (iseven(i) && isodd(j) ? 1.02 : 1.0) for i in 1:256, j in 1:512]
    # two exposures, 4 frames each, one frame of each 0.5 % brighter (flicker); the longer one with a dark-current noise
    # σ_d = 20 codes on top of the read noise: its darks' noise σ₀²(t) is σ_eff² + σ_d²
    σd = 20.0
    t1, t2 = 0.01s, 0.02s
    groups = Dict(t1 => [scene_frame(StableRNG(100 + k); light = k == 2 ? 0.5025 : 0.5, response) for k in 1:4],
                  t2 => [scene_frame(StableRNG(110 + k); light = k == 2 ? 1.005 : 1.0, response, σ = sqrt(SIGMA_R^2 + σd^2)) for k in 1:4])
    rn(σ) = sensor_fixture(; read_noise = (σ, σ, σ, σ)).model.noise
    darks = Dict(t1 => rn(SIGMA_EFF), t2 => rn(sqrt(SIGMA_EFF^2 + σd^2)))
    r = photon_transfer(groups, darks)
    pg = estimate(r)
    @test r isa FitResult && retcode(r) === ReturnCode.Success && r.metadata isa PhotonTransferMetadata
    @test pg isa PoissonGaussian && pg.source === SensorSource.measured
    @test abs(pg.gain - K_TRUE) <= Z * pg.std_error.gain                     # unbiased: the fixed pattern cancels
    # the range, derived from the data, recorded with its Bonferroni searches: m the candidate lower ends, each at α/m
    md = r.metadata
    @test r.n >= 3 && md.points_offered >= r.n && 0 < md.level_fraction[1] < md.level_fraction[2] < 1
    @test md.tests >= 1 && md.test_level == 0.001 / md.tests && md.rollover_tests == md.points_offered - 3
    @test md.rollover_level == 0.001 / md.rollover_tests && abs(md.curvature_t) <= z_over(md.tests)
    @test SIGMA_EFF^2 <= md.intercept <= SIGMA_EFF^2 + σd^2                   # the fitted points' darks (their median)
    # the intercept is each group's OWN exposure's: the longer group's darks taken as the shorter's bias K
    wrong = photon_transfer(groups, Dict(t1 => rn(SIGMA_EFF), t2 => rn(SIGMA_EFF)))
    @test abs(estimate(wrong).gain - K_TRUE) > Z * estimate(wrong).std_error.gain
    # an exposure without darks is refused, never interpolated or substituted
    e = refusal(() -> photon_transfer(groups, Dict(t1 => rn(SIGMA_EFF))))
    @test e isa ArgumentError && occursin("no darks at the exposure", e.msg)
    # pairs are the case of two frames; a larger group gives its first N frames, N the smallest group's
    @test abs(estimate(photon_transfer(Dict(t => g[1:2] for (t, g) in groups), darks)).gain - K_TRUE) <=
          Z * estimate(photon_transfer(Dict(t => g[1:2] for (t, g) in groups), darks)).std_error.gain
    @test photon_transfer(Dict(t1 => groups[t1], t2 => groups[t2][1:3]), darks).estimate ==
          photon_transfer(Dict(t1 => groups[t1][1:3], t2 => groups[t2][1:3]), darks).estimate
    # the guards
    @test_throws ArgumentError photon_transfer(Dict(t1 => groups[t1][1:1], t2 => groups[t2]), darks)   # a group of one frame
    @test_throws ArgumentError photon_transfer(empty(groups), darks)                                     # no group
    @test_throws ArgumentError photon_transfer(groups, darks; significance = 1)
    @test_throws ArgumentError photon_transfer(Dict(t1 => groups[t1], t2 => [synth_mosaic(zeros(UInt16, 64, 64)) for _ in 1:2]), darks)
    # one frame of the bright scene: the single-frame estimator (scheduled for removal) refuses the μ² term, naming the
    # series' method; a dim textured scene, no fixed pattern (≤ 10 % of the white level), is its design case: it accepts
    e = refusal(() -> photon_transfer(scene_frame(StableRNG(42); response)))
    @test e isa NoiseEstimateFailed && occursin("μ² term", e.reason) && occursin("photon_transfer(groups, darks)", e.reason)
    texture = [1 + 0.1 * sinpi(i / 64) * cospi(j / 96) for i in 1:256, j in 1:512]
    dim = [2.0 * (1600.0 / 2)^((l - 1) / 15) for l in 1:16]
    pd = photon_transfer(scene_frame(StableRNG(43); levels = dim, texture))
    @test pd.source === SensorSource.estimated
    @test abs(pd.gain - K_TRUE) <= Z * pd.std_error.gain && abs(pd.σ[1] - SIGMA_EFF) <= Z * pd.std_error.σ[1]
    @test_throws ArgumentError photon_transfer(scene_frame(StableRNG(1)); significance = 0)
end

# The single-frame estimator of 0.10.1 is 0.10.0's, bit for bit, where it accepts: the refactor onto the shared block
# kernel (the cell rule) and the shared fit changed no arithmetic. The numbers are 0.10.0's (release 7b9bec70) on the
# file's seeded `ptc_frame`s.
@testitem "sensor model: photon_transfer of one frame is 0.10.0's, bit for bit" setup = [SensorFixtures] begin
    using VisualGeometryToolkitCore: photon_transfer
    using StableRNGs: StableRNG
    # (seed, K, σ_r, se(K), se(σ_r)) of 0.10.0, printed with repr (round-trip exact)
    for (seed, K, σ, sK, sσ) in (
            (101, 1.5827201366016779, 2.4720165752103225, 0.019932146541470762, 0.03680304426500703),
            (102, 1.6663151369327605, 2.54356250910606, 0.021150720040122115, 0.03680884658603527),
            (103, 1.5628511965896894, 2.535740224174987, 0.01972960712385458, 0.03728417731866704))
        e = photon_transfer(ptc_frame(StableRNG(seed)))
        @test (e.gain, e.σ[1], e.std_error.gain, e.std_error.σ[1]) === (K, σ, sK, sσ)
    end
end

# The contract a per-frame caller of the single-frame estimator relies on (Core never estimates per frame: the
# resolver is a lookup): a refusal is a `NoiseEstimateFailed`, caught by its type, and the caller falls back to the
# characterisation's read noise — on a bright flat with a fixed pattern it falls back, on a dim scene it estimates.
@testitem "sensor model: a refusal is a NoiseEstimateFailed a caller falls back on" setup = [SensorFixtures] begin
    using VisualGeometryToolkitCore
    using VisualGeometryToolkitCore: photon_transfer, NoiseEstimateFailed, sensor_fixture
    using StableRNGs: StableRNG
    recorded = sensor_fixture(; read_noise = (SIGMA_EFF, SIGMA_EFF, SIGMA_EFF, SIGMA_EFF)).model.noise
    noise_of(frame) = try
        (photon_transfer(frame), :estimated)
    catch e
        e isa NoiseEstimateFailed || rethrow()
        (recorded, :fallback)                               # the caller's policy, marked in its record
    end
    rr = StableRNG(41)
    response = [(1 + 0.01 * randn(rr)) * (iseven(i) && isodd(j) ? 1.02 : 1.0) for i in 1:256, j in 1:512]
    @test noise_of(scene_frame(StableRNG(42); response)) == (recorded, :fallback)
    dim = [2.0 * (1600.0 / 2)^((l - 1) / 15) for l in 1:16]
    n, how = noise_of(scene_frame(StableRNG(43); levels = dim))
    @test how === :estimated && n isa PoissonGaussian
end

@testitem "sensor model: master_dark — a Dark, its ChannelBlack, the ReadNoise and the defects; defect_onsets" setup = [SensorFixtures] begin
    using VisualGeometryToolkitCore
    using VisualGeometryToolkitCore: master_dark, defect_onsets, photosite_colors, parse_unit, ustrip, SensorSource
    using StableRNGs: StableRNG
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
    @test r.dark isa Dark && r.dark.black isa ChannelBlack && r.read_noise isa ReadNoise
    @test r.dark.source === r.dark.black.source === r.read_noise.source === SensorSource.measured
    @test size(r.dark.offset) == (H, W) && r.dark.frames == n
    @test map(t -> ustrip(s, t), r.dark.exposure_band) == (0.5, 1.0)
    # the defects exactly (hot +300 codes and dead are far past z·σ; no false defect at the 1% expected over the sensor)
    @test r.defects == planted
    good = .!planted
    # the offset map: per photosite the mean of n codes, σ_eff/√n about its offset, at Z_N over the N photosites (plus
    # the Float32 storage's half-ulp at the offset's magnitude)
    N = count(good)
    @test all(abs(r.dark.offset[p] - offset[p]) <= z_over(N) * σeff[col[p] + 1] / sqrt(n) + eps(Float32(offset[p])) / 2
              for p in findall(good))
    b = r.dark.black
    for c in 0:3
        Nc = count(good .& (col .== c))
        # the black: the channel's population black (the DSNU drawn about it) within Z of its standard error
        @test abs(b.values[c + 1] - black[c + 1]) <= Z * b.std_error[c + 1]
        # that standard error against its first-principles count: the offsets' spread √(s_dsnu² + σ_eff²/n) over √Nc,
        # within the sample standard deviation's own relative error, Z/√(2(Nc − 1))
        se0 = sqrt(s_dsnu^2 + σeff[c + 1]^2 / n) / sqrt(Nc)
        @test abs(b.std_error[c + 1] / se0 - 1) <= Z / sqrt(2 * (Nc - 1))
        # the read noise (with the rounding) within Z of its standard error, √(2/((n − 1)Nc))·σ²/(2σ)
        @test abs(r.read_noise.σ[c + 1] - σeff[c + 1]) <= Z * r.read_noise.std_error[c + 1]
        @test r.read_noise.std_error[c + 1] ≈ sqrt(2 / ((n - 1) * Nc)) * r.read_noise.σ[c + 1] / 2
    end
    @test_throws ArgumentError master_dark(frames[1:1]; exposure_band = (0.5s, 1.0s))
    @test_throws ArgumentError master_dark([frames[1], synth_mosaic(zeros(UInt16, 4, 4))]; exposure_band = (0.5s, 1.0s))
    @test_throws ArgumentError master_dark(frames; exposure_band = (1.0s, 0.5s))

    # the onsets: each site's shortest flagging exposure, the masks in any order
    m1 = falses(4, 5); m1[2, 3] = true
    m2 = falses(4, 5); m2[2, 3] = true; m2[4, 1] = true
    m3 = falses(4, 5); m3[4, 1] = true; m3[1, 5] = true
    d = defect_onsets([0.8s => m3, (1 / 2000)s => m1, (1 / 500)s => m2])
    @test d isa DefectList && d.size == (4, 5) && d.source === SensorSource.measured
    onset(x, y) = ustrip(s, d.onset[only(findall(i -> d.x[i] == x && d.y[i] == y, eachindex(d.x)))])
    @test length(d.x) == 3 && onset(3, 2) == 1 / 2000 && onset(1, 4) == 1 / 500 && onset(5, 1) == 0.8
    @test_throws ArgumentError defect_onsets([1s => m1, 2s => falses(3, 3)])
    @test_throws ArgumentError defect_onsets(Pair{typeof(1.0s), BitMatrix}[])
end

@testitem "sensor model: the resolver — full, partial, absent, unmatched ISO, other body, precedence, onset" setup = [SensorFixtures] begin
    using VisualGeometryToolkitCore
    using VisualGeometryToolkitCore: sensor_fixture, parse_unit, ustrip, SensorSource, Matched, NoCharacterisation, UnmatchedISO, OtherBody
    s = parse_unit("s")
    body = DeviceIdentity(; make = "SONY", model = "ILME-FX3", serial = "5012345")
    frame(t, iso) = (; exposure = ExposureSettings(t, 8, iso), lens_settings = LensSettings(14mm))
    cap = CaptureSettings(Dict(1 => frame(0.8s, 100), 2 => frame(0.002s, 100), 3 => frame(0.8s, 400), 4 => frame(1 / 1000 * s, 100),
                               5 => frame(3.2s, 100)); body)
    H, W = 8, 10
    dark = (; offset = fill(512.0f0, H, W), variance = fill(6.0f0, H, W), exposure_band = (0.5s, 1.0s), frames = 16)
    defects = (; x = [1, 2, 3], y = [1, 2, 3], onset = [(1 / 2000)s, (1 / 500)s, 0.8s], size = (H, W))
    full = sensor_fixture(; body, black = (510.0, 511.0, 512.0, 513.0), black_std_error = (0.1, 0.1, 0.1, 0.1), dark,
                          gain = 1.6, read_noise = (2.5, 2.5, 2.5, 2.5), defects, source = SensorSource.measured).characterisation

    # FULL: every slot selected; the dark inside its band, its channel black outside it
    m = sensor_model(full, cap, 1)
    @test m.reason === Matched && !m.specified_override
    @test m.black isa Dark && m.noise isa PoissonGaussian && m.defects isa DefectList
    @test m.noise == full.entries[100.0].noise
    m2 = sensor_model([full], cap, 2)                    # 1/500 s: outside the dark's band
    @test m2.black isa ChannelBlack && m2.black == full.entries[100.0].black.black
    # ONSET: the sites whose onset is at most the exposure, the boundary included
    sites(t) = sort(collect(sensor_model(full, cap, t).defects.x))
    @test sites(4) == [1]                                # 1/1000 s: the 1/2000 s site
    @test sites(2) == [1, 2]                             # 1/500 s: its own onset included
    @test sites(1) == [1, 2, 3] && sites(5) == [1, 2, 3]   # 0.8 s (its onset), 3.2 s
    @test sensor_model(sensor_fixture(; body, defects = (; defects..., onset = fill(1.0s, 3))).characterisation, cap, 4).defects isa NoDefects

    # PARTIAL: a black and defects only → the noise is NoNoise, typed
    partial = sensor_fixture(; body, black = (510.0, 511.0, 512.0, 513.0), defects).characterisation
    mp = sensor_model(partial, cap, 1)
    @test mp.reason === Matched && mp.black isa ChannelBlack && mp.noise isa NoNoise && mp.defects isa DefectList

    # ABSENT, UNMATCHED ISO, ANOTHER BODY: the absent model, with the reason
    absent = SensorModel()
    for (chars, t, why) in ((nothing, 1, NoCharacterisation), (SensorCharacterisation[], 1, NoCharacterisation), (full, 3, UnmatchedISO),
                            (sensor_fixture(; body = DeviceIdentity(; make = "SONY", model = "ILME-FX3", serial = "9"),
                                            read_noise = (1, 1, 1, 1)).characterisation, 1, OtherBody))
        r = sensor_model(chars, cap, t)
        @test r.reason === why && r.black isa FileBlack && r.noise isa NoNoise && r.defects isa NoDefects
        @test (r == absent) == (why === NoCharacterisation)
    end
    @test absent.reason === NoCharacterisation

    # PRECEDENCE (D5): per slot, specified over measured over estimated; the override noted
    spec = sensor_fixture(; body, gain = 2.0, read_noise = (3, 3, 3, 3)).characterisation        # specified
    est = sensor_fixture(; body, gain = 1.5, read_noise = (2, 2, 2, 2), source = SensorSource.estimated).characterisation
    r = sensor_model([full, spec], cap, 1)
    @test r.noise == spec.entries[100.0].noise && r.specified_override
    @test r.black isa Dark && r.defects isa DefectList     # the other slots still the measured characterisation's
    r = sensor_model([full, est], cap, 1)
    @test r.noise == full.entries[100.0].noise && !r.specified_override
    @test sensor_model([spec, est], cap, 1).noise == spec.entries[100.0].noise
    # two of one source at one slot: refused; a frame the capture lacks: refused
    @test_throws ArgumentError sensor_model([full, full], cap, 1)
    @test_throws ArgumentError sensor_model(full, cap, 9)
    @test_throws ArgumentError sensor_model([1.0], cap, 1)

    # the merged characterisation: two ISOs of one body
    both = sensor_fixture(full, sensor_fixture(; body, iso = 400, read_noise = (1, 1, 1, 1)).characterisation)
    @test sort(collect(keys(both.entries))) == [100.0, 400.0]
    @test sensor_model(both, cap, 3).noise isa ReadNoise && sensor_model(both, cap, 1).noise isa PoissonGaussian
    @test_throws ArgumentError sensor_fixture(full, sensor_fixture(; iso = 400).characterisation)   # another body
    @test_throws ArgumentError sensor_fixture(full, full)                                           # ISO 100 twice
end

@testitem "sensor model: a characterisation matches the body by serial, else make and model" setup = [SensorFixtures] begin
    using VisualGeometryToolkitCore
    using VisualGeometryToolkitCore: sensor_fixture, parse_unit, Matched
    s = parse_unit("s")
    fx3(serial) = DeviceIdentity(; make = "SONY", model = "ILME-FX3", serial)
    rec(b) = sensor_fixture(; body = b, gain = 1.6, read_noise = (2.5, 2.5, 2.5, 2.5)).characterisation
    cap(b) = CaptureSettings(Dict(1 => (; exposure = ExposureSettings(0.8s, 8, 100), lens_settings = LensSettings(14mm))); body = b)
    uses(b, chars) = sensor_model(chars, cap(b), 1).reason === Matched
    @test uses(fx3("A"), rec(fx3("A")))                                   # the serial
    @test !uses(fx3("B"), rec(fx3("A")))                                  # another body of the model
    @test uses(fx3(nothing), rec(fx3("A")))                               # no serial stated: make and model
    @test uses(fx3("A"), rec(DeviceIdentity(; serial = "A")))             # a characterisation by serial alone
    @test !uses(DeviceIdentity(; make = "CANON", model = "ILME-FX3", serial = "A"), rec(fx3("A")))   # a serial whose make disagrees
    @test !uses(DeviceIdentity(), rec(DeviceIdentity()))                  # nothing stated: no match
    @test_throws ArgumentError uses(fx3(nothing), [rec(fx3("A")), rec(fx3("B"))])   # two bodies of the model: ambiguous
end

@testitem "sensor model: the components' guards, isbits, Adapt, equality and content_digest" setup = [SensorFixtures] begin
    using VisualGeometryToolkitCore
    using VisualGeometryToolkitCore: sensor_fixture, content_digest, parse_unit, SensorSource
    using Adapt: adapt
    s = parse_unit("s")
    # the guards, through the one fixture helper
    @test_throws ArgumentError sensor_fixture(; read_noise = (-1, 1, 1, 1))
    @test_throws ArgumentError sensor_fixture(; gain = 0.0, read_noise = (1, 1, 1, 1))
    @test_throws ArgumentError sensor_fixture(; gain = 1.0)                                    # a gain without σ_r
    @test_throws ArgumentError sensor_fixture(; relative_noise = -0.1)
    # a RelativeNoise is specified only (owner ruling on D14): every other source refused, by name
    for src in (SensorSource.estimated, SensorSource.measured, SensorSource.file)
        @test_throws "got source $(string(src))" sensor_fixture(; relative_noise = 0.05, source = src)
    end
    @test sensor_fixture(; relative_noise = 0.05).model.noise.source === SensorSource.specified
    # component_sources: the one provenance view of a model — each slot's source, nothing for an absent one, the reason, the override
    @test VisualGeometryToolkitCore.component_sources(SensorModel()) ==
          (black = SensorSource.file, noise = nothing, defects = nothing, reason = VisualGeometryToolkitCore.NoCharacterisation,
           specified_override = false)
    est = sensor_fixture(; black = (1, 2, 3, 4), gain = 1.5, read_noise = (1, 1, 1, 1),
                         source = (black = SensorSource.measured, noise = SensorSource.estimated)).model
    @test VisualGeometryToolkitCore.component_sources(est) == (black = SensorSource.measured, noise = SensorSource.estimated, defects = nothing,
                                                     reason = VisualGeometryToolkitCore.Matched, specified_override = false)
    @test_throws ArgumentError sensor_fixture(; black = (NaN, 0, 0, 0))
    @test_throws ArgumentError sensor_fixture(; black = (1, 1, 1, 1), source = SensorSource.file)   # the RAW's own is FileBlack
    @test_throws ArgumentError sensor_fixture(; defects = (; x = [11], y = [1], onset = [1s], size = (8, 10)))
    @test_throws ArgumentError sensor_fixture(; defects = (; x = [1, 1], y = [1, 1], onset = [1s, 2s], size = (8, 10)))
    @test_throws ArgumentError sensor_fixture(; defects = (; x = [1], y = [1], onset = [0s], size = (8, 10)))
    @test_throws ArgumentError sensor_fixture(; dark = (; offset = zeros(Float32, 2, 2), variance = zeros(Float32, 2, 2),
                                                        exposure_band = (1s, 0.5s), frames = 4))
    @test_throws DimensionMismatch sensor_fixture(; dark = (; offset = zeros(Float32, 2, 2), variance = zeros(Float32, 3, 2),
                                                           exposure_band = (0.5s, 1s), frames = 4))
    # isbits: a model of isbits components; Adapt moves the maps and lists, the model unchanged
    rn = sensor_fixture(; read_noise = (1, 2, 3, 4)).model
    @test isbits(rn) && isbits(SensorModel()) && isbits(sensor_fixture(; gain = 1, read_noise = (1, 1, 1, 1), black = (1, 1, 1, 1)).model)
    dark = (; offset = rand(Float32, 4, 6), variance = rand(Float32, 4, 6), exposure_band = (0.5s, 2s), frames = 8)
    defects = (; x = [1, 2], y = [3, 4], onset = [0.1s, 0.2s], size = (4, 6))
    full = sensor_fixture(; dark, black = (1, 2, 3, 4), gain = 1.5, read_noise = (1, 1, 1, 1), defects).model
    @test adapt(Array, full) == full && hash(adapt(Array, full)) == hash(full)
    # the identity: equal models one digest; a value, a map's byte, a source or the reason moves it
    again = sensor_fixture(; dark, black = (1, 2, 3, 4), gain = 1.5, read_noise = (1, 1, 1, 1), defects).model
    @test full == again && content_digest(full) == content_digest(again) && occursin(r"^[0-9a-f]{64}$", content_digest(full))
    moved = [sensor_fixture(; dark, black = (1, 2, 3, 4), gain = 1.6, read_noise = (1, 1, 1, 1), defects).model,
             sensor_fixture(; dark = (; dark..., offset = dark.offset .+ 1.0f-3), black = (1, 2, 3, 4), gain = 1.5,
                            read_noise = (1, 1, 1, 1), defects).model,
             sensor_fixture(; dark, black = (1, 2, 3, 4), gain = 1.5, read_noise = (1, 1, 1, 1), defects,
                            source = SensorSource.measured).model,
             SensorModel(), rn]
    @test allunique(content_digest.(vcat([full], moved)))
    empty = sensor_fixture().model                       # an entry with every slot absent: the absent types, Matched
    @test empty.black isa FileBlack && empty.noise isa NoNoise && empty.defects isa NoDefects
    @test content_digest(SensorModel()) != content_digest(empty)   # Matched ≠ NoCharacterisation
end

@testitem "sensor model: LinearRaw's one black subtraction — bit-identical with the absent model, the dark per photosite, defects masked" setup = [SensorFixtures] begin
    using VisualGeometryToolkitCore
    using VisualGeometryToolkitCore: sensor_fixture, parse_unit, channelview, photosite_colors, bayer_channels
    using StableRNGs: StableRNG
    s = parse_unit("s")
    rng = StableRNG(5)
    H, W = 12, 16
    black = (512.0, 500.0, 520.0, 508.0)
    m = synth_mosaic(UInt16[rand(rng, 300:16000) for i in 1:H, j in 1:W]; black)
    parent(m)[5, 6] = UInt16(WHITE)
    col = collect(photosite_colors(m))
    offset = Float32[black[col[i, j] + 1] + 3 * randn(rng) for i in 1:H, j in 1:W]
    # an R photosite (odd, odd) and a G (even, odd), as (x, y) = (column, row)
    defects = (; x = [3, 9], y = [3, 8], onset = [0.1s, 0.1s], size = (H, W))
    mask = falses(H, W); mask[3, 3] = true; mask[8, 9] = true
    same = sensor_fixture(; black = m.black).model                        # a ChannelBlack equal to the file's
    noisy = sensor_fixture(; read_noise = (2, 2, 2, 2)).model              # ReadNoise defines no variance: plain
    with_dark = sensor_fixture(; black, dark = (; offset, variance = fill(1.0f0, H, W), exposure_band = (0.5s, 1.0s), frames = 8),
                               defects, exposure = 0.8s).model
    @test with_dark.black isa Dark && with_dark.defects isa DefectList
    for w in (m, view(m, 3:12, 2:15), view(m, 2:11, 1:16)), mode in (:green_cells, :bin2), gain in (1.0, 2.5)
        L = LinearRaw(w; mode, gain)
        @test L.image isa Array{Float64, 3}
        # the absent model, a black equal to the file's, a read noise alone: bit for bit, a plain image
        for model in (SensorModel(), same, noisy)
            Ln = LinearRaw(w; mode, gain, sensor_model = model)
            @test Ln.image isa Array{Float64, 3} && Ln.image == L.image && Ln.saturated == L.saturated && Ln.geometry == L.geometry
            @test copy(linearview(w; sensor_model = model)) == copy(linearview(w))
        end
        # the dark: (raw − offset) / (white − black[c]) per photosite, at the window's offset into the visible raw
        o = w.window_origin
        Ld = LinearRaw(w; mode, gain, sensor_model = with_dark)
        lv = linearview(w; sensor_model = with_dark)
        @test all(lv[i, j] == (Float64(w[i, j]) - Float64(offset[i + o[1], j + o[2]])) / (WHITE - black[col[i + o[1], j + o[2]] + 1])
                  for i in axes(w, 1), j in axes(w, 2))
        v = channelview(lv)
        g = findall(n -> n in (:G1, :G2), collect(bayer_channels(w)))
        Gp = mode === :bin2 ? 2 : 1
        @test Ld.image[:, :, Gp] == (v[g[1], :, :] .+ v[g[2], :, :]) ./ 2 .* gain
        @test Ld.geometry.plane_sites == L.geometry.plane_sites               # the span is the channel's, dark or not
        # the defects join the mask: a cell is masked when a photosite its planes read is defective
        sites = mode === :bin2 ? (1:4) : g
        # (channel k of cell (i, j) is photosite (2i − 1 + di, 2j − 1 + dj), k = di + 2dj + 1, at the window's offset)
        hit(i, j) = any(mask[2i - 1 + (k - 1) % 2 + o[1], 2j - 1 + (k - 1) ÷ 2 + o[2]] for k in sites)
        @test Ld.saturated == L.saturated .| [hit(i, j) for i in axes(L.saturated, 1), j in axes(L.saturated, 2)]
    end
    @test any(LinearRaw(m; mode = :bin2, sensor_model = with_dark).saturated .& .!LinearRaw(m; mode = :bin2).saturated)
    # a map that does not reach the mosaic's photosites
    short = sensor_fixture(; dark = (; offset = offset[1:10, :], variance = fill(1.0f0, 10, W), exposure_band = (0.5s, 1.0s), frames = 8),
                           exposure = 0.8s).model
    @test_throws DimensionMismatch LinearRaw(m; mode = :green_cells, sensor_model = short)
    @test_throws DimensionMismatch linearview(m; sensor_model = short)
    small = sensor_fixture(; defects = (; x = [1], y = [1], onset = [0.1s], size = (4, 4))).model
    @test_throws DimensionMismatch LinearRaw(m; mode = :green_cells, sensor_model = small)
    @test_throws MethodError linearview(m; sensor_model = 1.0)
end

@testitem "sensor model: the uncertain image (D11) — each value's variance under PoissonGaussian and RelativeNoise" begin
    using Test, StableRNGs, Statistics
    using Distributions: Poisson, Normal
    using VisualGeometryToolkitCore
    using VisualGeometryToolkitCore: RawMosaic, sensor_fixture, parse_unit, Uncertain, value_of, green_plane
    using StructArrays: StructArray, components
    s = parse_unit("s")
    K, σ, black, white = 2.0, (3.0, 2.5, 4.0, 2.5), (510.0, 512.0, 514.0, 513.0), 16383.0
    rng = StableRNG(7)
    H, W, μe = 400, 400, 2000.0                          # a flat field of μe electrons per photosite
    pattern = UInt8[0 1; 3 2]                             # RGGB, LibRaw's indices (G2 = 3)
    col(i, j) = pattern[mod1(i, 2), mod1(j, 2)]
    raw = [UInt16(round(black[col(i, j) + 1] + rand(rng, Poisson(μe)) / K + rand(rng, Normal(0, σ[col(i, j) + 1])))) for i in 1:H, j in 1:W]
    m = RawMosaic(raw, pattern, black, white, (0, 0), (H, W), (;))
    pg = sensor_fixture(; gain = K, read_noise = σ, black).model
    for gain in (1.0, 2.5), mode in (:green_cells, :bin2)
        Lp = LinearRaw(m; mode, gain, sensor_model = sensor_fixture(; black).model)      # the same black, no noise: plain
        L = LinearRaw(m; mode, gain, sensor_model = pg)
        @test L.image isa StructArray && eltype(L.image) <: Uncertain{Float64}
        vals, covs = components(L.image)
        @test vals == Lp.image && L.saturated == Lp.saturated                 # the values bit for bit the plain image's
        @test green_plane(L) isa StructArray && value_of.(green_plane(L)) == green_plane(Lp)
        for p in axes(vals, 3)
            ps = L.geometry.plane_sites[p]; mm = length(ps)
            x = vec(vals[:, :, p])
            vpix = [c[1] for c in vec(covs[:, :, p])]
            # the law, value by value: g·x/(m²K)·Σ 1/d + g²/m²·Σ σ_r[c]²/d² (x > 0 everywhere on this flat)
            α = gain / (mm^2 * K) * sum(q -> 1 / q.dn, ps)
            σ² = gain^2 / mm^2 * sum(q -> σ[q.color + 1]^2 / q.dn^2, ps)
            @test isapprox(VisualGeometryToolkitCore.poisson_slope(L, pg)[p], α; rtol = 16eps())   # the slope the kernel writes with
            # (each side ≤ 8 roundings: rtol 16 ulp)
            @test all(>(0), x) && isapprox(vpix, α .* x .+ σ²; rtol = 16eps())
            # against the simulated flat: the rounding adds 1/12 code² per photosite, which σ_r includes in the model
            # (Sheppard), so the truth here is the law plus the rounding's. The sample variance of n iid values has
            # relative standard error √(2/(n − 1)) (Gaussian, here ≈ Poisson at 2000 e⁻); 3.29 of them: two-sided 0.1%.
            v = α * mean(x) + σ² + gain^2 / mm^2 * sum(q -> (1 / 12) / q.dn^2, ps)
            @test abs(var(x) / v - 1) <= 3.29 * sqrt(2 / (length(x) - 1))
        end
    end
    # RelativeNoise: (σ_rel·x)², no read-noise or dark term
    L = LinearRaw(m; mode = :green_cells, sensor_model = sensor_fixture(; relative_noise = 0.05).model)
    vals, covs = components(L.image)
    @test isapprox([c[1] for c in covs], (0.05 .* vals) .^ 2; rtol = 16eps())
    @test VisualGeometryToolkitCore.poisson_slope(L, sensor_fixture(; relative_noise = 0.05).model) == (0.0,)   # no Poisson term
    # below the black level the Poisson term is 0: a value of 0 has the Gaussian variance alone
    m0 = RawMosaic(fill(UInt16(400), 4, 4), pattern, black, white, (0, 0), (4, 4), (;))
    L0 = LinearRaw(m0; mode = :green_cells, sensor_model = pg)
    ps = L0.geometry.plane_sites[1]
    @test all(c -> isapprox(c[1], sum(q -> σ[q.color + 1]^2 / q.dn^2, ps) / 4; rtol = 16eps()), components(L0.image)[2])
    # a Dark: the photosite's dark variance × (1 + 1/frames) in place of σ_r[c]²
    dv = fill(9.0f0, H, W)
    Ld = LinearRaw(m; mode = :green_cells, sensor_model = sensor_fixture(; gain = K, read_noise = σ, black, exposure = 1s,
        dark = (; offset = Float32.(fill(512.0, H, W)), variance = dv, exposure_band = (0.5s, 2s), frames = 8)).model)
    xd, cd = components(Ld.image)
    ps = Ld.geometry.plane_sites[1]
    @test isapprox([c[1] for c in vec(cd)], (1 / (4K) * sum(q -> 1 / q.dn, ps)) .* max.(vec(xd), 0) .+
                                            1 / 4 * sum(q -> 9 * (1 + 1 / 8) / q.dn^2, ps); rtol = 16eps())
end

# Owner ruling D14: the default uncertain image is the LAZY view — `linearview` under a model with a variance law has
# `Uncertain` elements whose scalar variance is computed on the read, by the one law LinearRaw's kernel writes per
# plane; `copy` materialises it (SoA). Lazy and materialised agree element for element; `var` is the one accessor.
@testitem "sensor model: the lazy uncertain view (D14) — linearview's Uncertain elements, var, the materialised copy" begin
    using Test, StableRNGs, Statistics
    using VisualGeometryToolkitCore
    using VisualGeometryToolkitCore: RawMosaic, LinearRawView, sensor_fixture, parse_unit, Uncertain, value_of, poisson_slope, channelview
    using StructArrays: StructArray, components
    s = parse_unit("s")
    K, σ, black, white = 2.0, (3.0, 2.5, 4.0, 2.5), (510.0, 512.0, 514.0, 513.0), 16383.0
    pattern = UInt8[0 1; 3 2]                             # RGGB, LibRaw's indices (G2 = 3)
    col(i, j) = pattern[mod1(i, 2), mod1(j, 2)]
    rng = StableRNG(14)
    H, W = 10, 12
    m = RawMosaic(UInt16[rand(rng, 300:9000) for i in 1:H, j in 1:W], pattern, black, white, (0, 0), (H, W), (;))
    pg = sensor_fixture(; gain = K, read_noise = σ, black).model
    dark = (; offset = Float32.(512 .+ randn(rng, H, W)), variance = rand(rng, Float32, H, W) .+ 4.0f0, exposure_band = (0.5s, 2s), frames = 8)
    pgd = sensor_fixture(; gain = K, read_noise = σ, black, dark, exposure = 1s).model
    rel = sensor_fixture(; relative_noise = 0.05).model
    # without a variance law the view is plain, as before
    for model in (SensorModel(), sensor_fixture(; black).model, sensor_fixture(; read_noise = (2, 2, 2, 2)).model)
        lv = linearview(m; sensor_model = model)
        @test eltype(lv) === Float64 && lv isa LinearRawView{Float64} && copy(lv) isa Matrix{Float64}
    end
    span(i, j) = white - black[col(i, j) + 1]
    # each model beside the same black slot without a noise law (the RAW's own for RelativeNoise)
    plain = (sensor_fixture(; black).model, sensor_fixture(; black, dark, exposure = 1s).model, SensorModel())
    for (model, same_black) in zip((pg, pgd, rel), plain)
        lv = linearview(m; sensor_model = model)
        @test eltype(lv) <: Uncertain{Float64} && isbits(lv[1, 1])
        @test value_of.(lv) == copy(linearview(m; sensor_model = same_black))   # the values bit for bit the plain view's
        mat = copy(lv)                                         # the materialised form: a value plane and a variance plane
        @test mat isa StructArray && eltype(mat) === eltype(lv)
        @test all(I -> mat[I] === lv[I], CartesianIndices(lv))                   # lazy and materialised agree, element for element
        @test all(I -> var(lv[I]) === Statistics.cov(lv[I])[1, 1], CartesianIndices(lv))   # THE scalar accessor
        @test channelview(lv)[2, 1, 1] === lv[2, 1]            # the cells channel-first, uncertain alike
    end
    # the law per photosite (one site, gain 1): max(v,0)/(K d) + s/d², s = σ_r[c]², or the dark variance × (1 + 1/frames)
    # (each side's expression ≤ 4 roundings, summed in another order: rtol 8 ulp)
    lv, lvd, lvr = linearview(m; sensor_model = pg), linearview(m; sensor_model = pgd), linearview(m; sensor_model = rel)
    for j in 1:W, i in 1:H
        v, d = value_of(lv[i, j]), span(i, j)
        @test isapprox(var(lv[i, j]), max(v, 0) / (K * d) + σ[col(i, j) + 1]^2 / d^2; rtol = 8eps())
        vd = value_of(lvd[i, j])
        @test isapprox(var(lvd[i, j]), max(vd, 0) / (K * d) + dark.variance[i, j] * (1 + 1 / 8) / d^2; rtol = 8eps())
        @test isapprox(var(lvr[i, j]), (0.05 * value_of(lvr[i, j]))^2; rtol = 8eps())
    end
    # the same law as LinearRaw's plane: two greens of one value and one span are the mean of the two, its variance
    # the photosites' sum over 4 (the plane's own expression at m = 2)
    flat = RawMosaic(fill(UInt16(4000), 4, 4), pattern, (512.0, 512.0, 512.0, 512.0), white, (0, 0), (4, 4), (;))
    pf = sensor_fixture(; gain = K, read_noise = σ, black = (512.0, 512.0, 512.0, 512.0)).model
    L, lf = LinearRaw(flat; mode = :green_cells, sensor_model = pf), linearview(flat; sensor_model = pf)
    @test value_of(L.image[1, 1, 1]) == value_of(lf[1, 2])
    @test isapprox(var(L.image[1, 1, 1]), (var(lf[1, 2]) + var(lf[2, 1])) / 4; rtol = 8eps())
    # one plane's Poisson slope, by index or colour
    Lb = LinearRaw(m; mode = :bin2, sensor_model = pg)
    @test poisson_slope(Lb, pg, :G) === poisson_slope(Lb, pg)[2] === poisson_slope(Lb, pg, 2)
    @test poisson_slope(Lb, pg, :R) === poisson_slope(Lb, pg)[1] && poisson_slope(Lb, pg, :B) === poisson_slope(Lb, pg)[3]
    Lg = LinearRaw(m; mode = :green_cells, sensor_model = pg)
    @test poisson_slope(Lg, pg, :G) === only(poisson_slope(Lg, pg))
    @test_throws ArgumentError poisson_slope(Lg, pg, :R)
end

@testitem "sensor model: CPU = GPU — LinearRaw with a ChannelBlack, a Dark, a DefectList and a PoissonGaussian; the estimators" setup = [SensorFixtures] begin
    using VisualGeometryToolkitCore
    using VisualGeometryToolkitCore: sensor_fixture, parse_unit
    using StableRNGs: StableRNG
    using Adapt: adapt
    using StructArrays: components
    using CUDA: CUDA
    s = parse_unit("s")
    if CUDA.functional()
        rng = StableRNG(9)
        H, W = 12, 16
        m = synth_mosaic(UInt16[rand(rng, 300:16000) for i in 1:H, j in 1:W])
        offset = Float32.(BLACK .+ 3 .* randn(rng, H, W))
        defects = (; x = [7, 2], y = [4, 9], onset = [0.1s, 2.0s], size = (H, W))    # the second's onset past the exposure
        dark = (; offset, variance = rand(rng, Float32, H, W) .+ 5.0f0, exposure_band = (0.5s, 1.0s), frames = 8)
        models = [sensor_fixture(; black = (500.0, 505, 510, 515), defects, exposure = 0.8s).model,
                  sensor_fixture(; black = (500.0, 505, 510, 515), dark, defects, gain = 1.6, read_noise = (2.5, 2.4, 2.6, 2.5),
                                 exposure = 0.8s).model,
                  sensor_fixture(; relative_noise = 0.03).model]
        @test models[1].defects isa DefectList && length(models[1].defects.x) == 1
        d = adapt(CUDA.CuArray, m)
        for model in models, (wh, wd) in ((m, d), (view(m, 3:12, 2:15), view(d, 3:12, 2:15))), mode in (:green_cells, :bin2)
            for md in (model, adapt(CUDA.CuArray, model))                  # the model on the host or on the device
                Lh = LinearRaw(wh; mode, sensor_model = model)
                Ld = LinearRaw(wd; mode, sensor_model = md)
                @test adapt(Array, Ld.image) == Lh.image && Array(Ld.saturated) == Lh.saturated
                # the lazy view read on the device (its materialised copy, a broadcast of the view there) = the host's (D14)
                @test adapt(Array, copy(linearview(wd; sensor_model = md))) == copy(linearview(wh; sensor_model = model))
            end
        end
        @test components(LinearRaw(d; mode = :bin2, sensor_model = models[2]).image)[1] isa CUDA.CuArray
        f = ptc_frame(StableRNG(21))
        @test photon_transfer(adapt(CUDA.CuArray, f)) == photon_transfer(f)
        # the series' estimate (0.10.1): the frames on the device, their scales and stacks there
        gs = Dict((l * 0.01)s => [scene_frame(StableRNG(200 + 10l + k); light = l == 1 ? 1.0 : 0.5) for k in 1:3] for l in 1:2)
        gd = Dict(t => [adapt(CUDA.CuArray, f) for f in g] for (t, g) in gs)
        dk = Dict(t => sensor_fixture(; read_noise = (SIGMA_EFF, SIGMA_EFF, SIGMA_EFF, SIGMA_EFF)).model.noise for t in keys(gs))
        rh, rd = photon_transfer(gs, dk; normalize = false), photon_transfer(gd, dk; normalize = false)
        @test rd.estimate == rh.estimate && rd.metadata == rh.metadata   # the same sums, one work item each
        # with the scale, each frame's mean is a reduction whose order the device chooses: the estimate within its own SE
        ph, pd = photon_transfer(gs, dk).estimate, photon_transfer(gd, dk).estimate
        @test abs(pd.gain - ph.gain) <= Z * ph.std_error.gain
        frames = [synth_mosaic([code(rng, BLACK, nothing, SIGMA_R) for i in 1:32, j in 1:32]) for _ in 1:4]
        rh = master_dark(frames; exposure_band = (0.5s, 1.0s))
        rd = master_dark([adapt(CUDA.CuArray, f) for f in frames]; exposure_band = (0.5s, 1.0s))
        @test adapt(Array, rd.dark) == rh.dark && rd.read_noise == rh.read_noise && rd.defects == rh.defects
    else
        @test_skip CUDA.functional()                         # no functional GPU here: recorded as Broken
    end
end
