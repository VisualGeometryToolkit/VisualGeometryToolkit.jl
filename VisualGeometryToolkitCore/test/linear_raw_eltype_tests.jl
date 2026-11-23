# The element type of a RAW image's linear values (Core 0.10.5): `LinearRaw([T], mosaic; mode, …)` and
# `linearview([T], mosaic; …)` store each value as `T` (Float64, today's default; Float32; Float16), the variance plane of an
# uncertain image as `_variance_eltype(T)` (Float32 for Float16: a Float16 variance underflows). Every value is computed in
# Float64 and rounded ONCE on store, so a Float16 value is the Float64 value rounded to nearest, bit for bit. A consumer
# reduces the stored values in Float64 (`sum(Float64, A)`, `mean(Float64, A)`), and the bounds below are derived from that
# one rounding: per value |x̂ − x| ≤ δ(x) = max(eps(Float16)·|x|/2, s/2), s the subnormal spacing 2⁻²⁴ (round to nearest
# within a binade; half the fixed spacing below the smallest normal).

@testsetup module Float16Rounding
    export δ
    # The bound of one rounding to nearest of a Float64 value x to Float16: half a spacing, eps(Float16)·|x|/2 for a
    # normal value, half the subnormal spacing 2⁻²⁴ below the smallest normal.
    δ(x) = max(eps(Float16) * abs(x) / 2, 2.0^-25)                  # (2⁻²⁵ in Float64: in Float16 it rounds to 0)
    # The variance a Float16 store adds to a value x: its rounding, uniform over one spacing (Core's `_storage_variance`).
    export storage_variance
    storage_variance(x) = Float64(eps(Float16(x)))^2 / 12
end

@testitem "linear raw eltype: a Float16 value is the Float64 value rounded once; the variance plane Float32" setup = [SensorFixtures, Float16Rounding] begin
    using VisualGeometryToolkitCore
    using VisualGeometryToolkitCore: sensor_fixture, Uncertain, value_of, green_plane, LinearRawView, linear_quantum
    using Statistics: var
    using StructArrays: StructArray, components
    using StableRNGs: StableRNG
    using Adapt: adapt
    rng = StableRNG(31)
    H, W = 64, 80
    # every code: below the black level, the subnormal range just above it, every binade up to the white level (saturated)
    m = synth_mosaic(UInt16[rand(rng, 0:16383) for i in 1:H, j in 1:W])
    pg = sensor_fixture(; gain = K_TRUE, read_noise = (SIGMA_R, SIGMA_R, SIGMA_R, SIGMA_R)).model
    @test VisualGeometryToolkitCore._LINEAR_RAW_ELTYPE === Float64          # the one default: today's
    for mode in (:green_cells, :bin2), gain in (1.0, 2.5)
        L64 = LinearRaw(m; mode, gain)
        @test L64.image isa Array{Float64, 3}
        @test LinearRaw(Float64, m; mode, gain).image == L64.image           # the default spelled out
        L16 = LinearRaw(Float16, m; mode, gain)
        @test L16.image isa Array{Float16, 3}
        @test L16.saturated == L64.saturated && L16.geometry == L64.geometry && L16.gain == L64.gain
        # the kernel's mean is in Float64 and rounded once on store: bit for bit the Float64 image rounded (a mean of the
        # two greens' Float16 values would round twice)
        @test L16.image == Float16.(L64.image)
        @test all(abs(Float64(a) - b) <= δ(b) for (a, b) in zip(L16.image, L64.image))
        @test LinearRaw(Float32, m; mode, gain).image == Float32.(L64.image)
        # an uncertain image: values of T, the variance from the UNROUNDED value, then rounded once to Float32
        U64 = LinearRaw(m; mode, gain, sensor_model = pg)
        U16 = LinearRaw(Float16, m; mode, gain, sensor_model = pg)
        @test U16.image isa StructArray && eltype(U16.image) <: Uncertain{Float16}
        @test eltype(components(U16.image)[2]) === typeof(CovMat(ScaledIdentity{1}(0.0f0)))
        @test value_of.(U16.image) == Float16.(value_of.(U64.image))
        # the stored value's variance: the law's at the unrounded value plus the storage's rounding, spacing²/12
        @test var.(U16.image) == Float32.(var.(U64.image) .+ storage_variance.(value_of.(U64.image)))
        @test var.(LinearRaw(Float32, m; mode, gain, sensor_model = pg).image) == Float32.(var.(U64.image) .+
                                                                                    Float64.(eps.(Float32.(value_of.(U64.image)))) .^ 2 ./ 12)
        @test green_plane(U16) isa StructArray && eltype(green_plane(U16)) === eltype(U16.image)
        @test eltype(LinearRaw(Float32, m; mode, sensor_model = pg).image) <: Uncertain{Float32}
    end
    # why the variance plane is not Float16: a dark cell's variance, its read noise alone, g²/m²·Σ σ_r²/d² = σ_r²/(2d²)
    # for the two greens (≈ 1.2e-8 here), is below half Float16's smallest subnormal (2⁻²⁴/2 ≈ 3.0e-8): stored as Float16
    # it would be 0, and a whitened row would divide by it
    m0 = synth_mosaic(fill(UInt16(BLACK), 8, 8))
    v0 = var.(LinearRaw(m0; mode = :green_cells, sensor_model = pg).image)
    @test all(v -> Float16(v) == 0, v0)
    @test var.(LinearRaw(Float16, m0; mode = :green_cells, sensor_model = pg).image) == Float32.(v0 .+ storage_variance(0.0)) && all(>(0), Float32.(v0))
    # the lazy view: the same element type, one rounding of the Float64 element
    for model in (SensorModel(), pg)
        lv64, lv16 = linearview(m; sensor_model = model), linearview(Float16, m; sensor_model = model)
        @test copy(linearview(Float64, m; sensor_model = model)) == copy(lv64)
        @test lv16 isa LinearRawView && value_of.(copy(lv16)) == Float16.(value_of.(copy(lv64)))
        @test (model === pg ? eltype(lv16) <: Uncertain{Float16} : eltype(lv16) === Float16) && eltype(copy(lv16)) === eltype(lv16)
        model === pg && @test var.(copy(lv16)) == Float32.(var.(copy(lv64)) .+ storage_variance.(value_of.(copy(lv64))))
        @test eltype(adapt(Array, lv16)) === eltype(lv16)                      # adapt keeps the element type
    end
    # linear_quantum: the resolution of the stored value, the coarser of one code (1/span) and the value's spacing in T
    lv64, lv16 = linearview(m), linearview(Float16, m)
    @test all(linear_quantum(lv64, i, j) == 1 / (WHITE - BLACK) for i in 1:H, j in 1:W)          # Float64: one code, as before
    @test all(linear_quantum(lv16, i, j) == max(1 / (WHITE - BLACK), Float64(eps(lv16[i, j]))) for i in 1:H, j in 1:W)
    # near the white level a Float16 step is span/2048 codes (the spacing 2⁻¹¹ on [½, 1)): 7.75 codes on this sensor
    top = linearview(Float16, synth_mosaic(fill(UInt16(16000), 2, 2)))
    @test linear_quantum(top, 1, 1) == 2.0^-11 && linear_quantum(top, 1, 1) * (WHITE - BLACK) == (WHITE - BLACK) / 2048
    # a value past the type's largest finite value is refused, not stored as Inf: Float16 holds 65504, a code of 65535 at
    # gain g is (65535 − 512)/15871·g ≈ 4.1g
    @test_throws ArgumentError LinearRaw(Float16, m; mode = :green_cells, gain = 2.0e4)
    @test LinearRaw(Float16, m; mode = :green_cells, gain = 1.5e4).image isa Array{Float16, 3}
    @test LinearRaw(m; mode = :green_cells, gain = 2.0e4).image isa Array{Float64, 3}
    # an element type is a float type
    @test_throws MethodError LinearRaw(Int, m; mode = :green_cells)
    @test_throws MethodError linearview(Int, m)
end

@testitem "linear raw eltype: dark and photon-transfer statistics over Float16 storage agree with Float64 to the rounding's bound" setup = [SensorFixtures, Float16Rounding] begin
    using VisualGeometryToolkitCore
    using Statistics: mean, var, std
    using StableRNGs: StableRNG
    using Distributions: Normal, quantile
    # The statistics Core's MasterDark and PhotonTransfer estimate (they read the mosaics' UInt16 codes, so Float16
    # storage cannot move them) computed by a consumer from the STORED green cells, accumulated in Float64: per cell its
    # temporal mean and variance over n frames, the black and σ_r from darks, K from the variance against the mean.
    # The bounds, per cell of n samples x_k stored as x̂_k = x_k + e_k, |e_k| ≤ δ_k = δ(x_k):
    #   mean      |Δμ| ≤ (1/n)·Σ δ_k
    #   variance  s²(x + e) − s²(x) = 2·c(x, e) + s²(e), |c(x, e)| ≤ s(x)·s(e) (Cauchy–Schwarz) and
    #             s(e) ≤ ε = √(Σ δ_k²/(n − 1)), so |Δs²| ≤ 2·s(x)·ε + ε²
    # and in expectation the rounding ADDS its variance, step²/12 per sample (e uniform over one step, uncorrelated with x).
    Z = quantile(Normal(), 1 - 0.001 / 2)
    cells(T, frames) = cat((Float64.(LinearRaw(T, f; mode = :green_cells).image[:, :, 1]) for f in frames)...; dims = 3)
    function moments(X)
        μ = dropdims(mean(X; dims = 3); dims = 3)
        return μ, dropdims(var(X; dims = 3); dims = 3)
    end
    function bounds(X)                                     # X the Float64 values: per cell the mean's and the variance's bound
        n = size(X, 3)
        D = δ.(X)
        bμ = dropdims(sum(D; dims = 3); dims = 3) ./ n
        ε = sqrt.(dropdims(sum(D .^ 2; dims = 3); dims = 3) ./ (n - 1))
        return bμ, 2 .* dropdims(std(X; dims = 3); dims = 3) .* ε .+ ε .^ 2
    end
    span = WHITE - BLACK
    n = 16
    # darks: the black and the read noise (MasterDark's statistics)
    rng = StableRNG(41)
    darks = [synth_mosaic([code(rng, BLACK, nothing, SIGMA_R) for i in 1:64, j in 1:64]) for _ in 1:n]
    X64, X16 = cells(Float64, darks), cells(Float16, darks)
    μ64, v64 = moments(X64); μ16, v16 = moments(X16)
    bμ, bv = bounds(X64)
    @test all(abs.(μ16 .- μ64) .<= bμ) && all(abs.(v16 .- v64) .<= bv)
    black64, black16 = BLACK + span * mean(μ64), BLACK + span * mean(μ16)
    @test abs(black16 - black64) <= span * mean(bμ)
    # σ_r (codes) of a photosite: a green cell's variance is σ_r²/(2 d²), so σ_r = d·√(2·mean v); |√a − √b| ≤ |a − b|/√b
    σr64, σr16 = span * sqrt(2 * mean(v64)), span * sqrt(2 * mean(v16))
    @test abs(σr16 - σr64) <= span * sqrt(2) * mean(bv) / sqrt(mean(v64))
    @test abs(σr64 - SIGMA_EFF) <= 0.05 * SIGMA_EFF                            # (the fixture's own read noise, sanity)
    # flats: photon transfer, the 16 LEVELS of `scene_frame` (one per pair of 16-photosite block columns)
    flats = [scene_frame(StableRNG(400 + k)) for k in 1:n]
    F64, F16 = cells(Float64, flats), cells(Float16, flats)
    m64, w64 = moments(F64); m16, w16 = moments(F16)
    cμ, cv = bounds(F64)
    @test all(abs.(m16 .- m64) .<= cμ) && all(abs.(w16 .- w64) .<= cv)
    level = [level_of_column(2j) for j in axes(F64, 2)]                       # cell column j holds photosite columns 2j − 1, 2j
    pool(A, l) = mean(A[:, level .== l])
    L = length(LEVELS)
    μ_l64, v_l64 = [pool(m64, l) for l in 1:L], [pool(w64, l) for l in 1:L]
    μ_l16, v_l16 = [pool(m16, l) for l in 1:L], [pool(w16, l) for l in 1:L]
    Δμ_l, Δv_l = [pool(cμ, l) for l in 1:L], [pool(cv, l) for l in 1:L]       # each level's bounds (means of the cells')
    # K from the least-squares slope β of v on μ: a green cell's v = σ² + μ/(2 K d), so K = 1/(2 d β). To first order
    # Δβ = Σ w_l Δv_l + Σ ∂β/∂μ_l Δμ_l, w_l = (μ_l − μ̄)/S, ∂β/∂μ_l = (r_l − β(μ_l − μ̄))/S (S = Σ(μ − μ̄)², r the
    # residuals); the second-order terms are below Δμ/range ≤ 1e-3 of the first, inside the factor 2.
    slope(μ, v) = (c = μ .- mean(μ); sum(c .* (v .- mean(v))) / sum(abs2, c))
    β = slope(μ_l64, v_l64)
    c = μ_l64 .- mean(μ_l64); S = sum(abs2, c); r = (v_l64 .- mean(v_l64)) .- β .* c
    Δβ = 2 * (sum(abs.(c ./ S) .* Δv_l) + sum(abs.((r .- β .* c) ./ S) .* Δμ_l))
    K64, K16 = 1 / (2span * β), 1 / (2span * slope(μ_l16, v_l16))
    @test abs(K16 - K64) <= K64 * Δβ / (abs(β) - Δβ)                        # |Δ(1/β)| ≤ Δβ/(|β|(|β| − Δβ))·|β|·K
    @test abs(K64 / K_TRUE - 1) <= 0.05                                        # (the fixture's K, sanity)
    # the added variance in expectation: at the top level (x on [½, 1), one step 2⁻¹¹) the pooled Δv is step²/12, within
    # Z standard errors: per cell 2·c(x, e) has standard deviation 2·s(x)·σ_e/√n, σ_e = step/√12 (s²(e)'s own spread
    # is (σ_e²)·√(2/(n − 1)), below a tenth of it here and added in quadrature), over the level's C cells
    top = level .== L
    spacing = 2.0^-11
    @test all(x -> 0.5 <= x < 1, F64[:, top, :])
    σe² = spacing^2 / 12
    C = count(top) * size(F64, 1)
    se = sqrt((4 * mean(w64[:, top]) * σe² / n + σe²^2 * 2 / (n - 1)) / C)
    @test abs(mean(w16[:, top] .- w64[:, top]) - σe²) <= Z * se
end

@testitem "linear raw eltype: a Float16 image reduced in Float64 — the sum of a million values to its derived bound" setup = [SensorFixtures, Float16Rounding] begin
    using VisualGeometryToolkitCore
    using VisualGeometryToolkitCore: green_plane
    using Statistics: mean
    using StableRNGs: StableRNG
    # 1024 × 1024 green cells near the white level: their sum (≈ 8e5) is past Float16's largest finite value (65504), so
    # a Float16 accumulation cannot hold it; a consumer reduces the stored values in Float64 (`sum(Float64, A)`). The
    # Float64 sum of the stored values differs from the Float64 values' sum by at most Σ δ_k (the storage's roundings)
    # plus each sum's own accumulation error, ≤ (n − 1)·eps(Float64)·Σ|x| for a sum of n terms in any order.
    rng = StableRNG(53)
    m = synth_mosaic(UInt16[rand(rng, 11000:16000) for i in 1:2048, j in 1:2048])
    G64 = green_plane(LinearRaw(m; mode = :green_cells))
    G16 = green_plane(LinearRaw(Float16, m; mode = :green_cells))
    @test eltype(G16) === Float16 && length(G16) == 2^20
    s64 = sum(G64)
    @test s64 > floatmax(Float16)
    n = length(G64)
    bound = sum(δ, G64) + 2 * (n - 1) * eps(Float64) * sum(abs, G64)
    @test abs(sum(Float64, G16) - s64) <= bound
    @test abs(mean(Float64, G16) - mean(G64)) <= bound / n
    # the bound is a half step per value at most; the roundings, centred, cancel far below it
    @test abs(sum(Float64, G16) - s64) <= 0.01 * sum(δ, G64)
end

@testitem "linear raw eltype: CPU = GPU — the Float16 path, plain and uncertain, LinearRaw and linearview" setup = [SensorFixtures] begin
    using VisualGeometryToolkitCore
    using VisualGeometryToolkitCore: sensor_fixture, parse_unit, linear_quantum
    using StableRNGs: StableRNG
    using Adapt: adapt
    using StructArrays: components
    using CUDA: CUDA
    s = parse_unit("s")
    if CUDA.functional()
        rng = StableRNG(61)
        H, W = 24, 32
        mh = synth_mosaic(UInt16[rand(rng, 0:16383) for i in 1:H, j in 1:W])   # (not `m`: Core exports the metre)
        dark = (; offset = Float32.(BLACK .+ 3 .* randn(rng, H, W)), variance = rand(rng, Float32, H, W) .+ 5.0f0,
                exposure_band = (0.5s, 1.0s), frames = 8)
        models = [SensorModel(), sensor_fixture(; gain = K_TRUE, read_noise = (2.5, 2.4, 2.6, 2.5)).model,
                  sensor_fixture(; black = (500.0, 505, 510, 515), dark, gain = 1.6, read_noise = (2.5, 2.4, 2.6, 2.5), exposure = 0.8s).model]
        d = adapt(CUDA.CuArray, mh)
        for T in (Float16, Float32), model in models, (wh, wd) in ((mh, d), (view(mh, 3:20, 2:29), view(d, 3:20, 2:29))),
            mode in (:green_cells, :bin2)
            Lh = LinearRaw(T, wh; mode, gain = 1.5, sensor_model = model)
            Ld = LinearRaw(T, wd; mode, gain = 1.5, sensor_model = adapt(CUDA.CuArray, model))
            @test adapt(Array, Ld.image) == Lh.image && Array(Ld.saturated) == Lh.saturated
            @test (Ld.image isa CUDA.CuArray ? eltype(Ld.image) : eltype(components(Ld.image)[1])) === T
            @test adapt(Array, copy(linearview(T, wd; sensor_model = model))) == copy(linearview(T, wh; sensor_model = model))
        end
        @test linear_quantum(linearview(Float16, d), 3, 4) == linear_quantum(linearview(Float16, mh), 3, 4)
    else
        @test_skip CUDA.functional()                         # no functional GPU here: recorded as Broken
    end
end
