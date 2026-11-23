# Fixtures of Core's noise-model estimators and its device LM: sensor_model_tests' SensorFixtures (copied) and its
# frames, groups and darks; fit_scale_tests' grids; static_lm_tests' problems, pixel_lm_reference_tests' 1000 seeded
# problems (test/fixtures/pixel_lm/problems.jl, copied), lens_inverse_tests' lenses and points (the lens inverse, a
# Core caller of static_lm) and a SplineLens camera (the spline projection, the other caller). A test's noise draw is
# redrawn from StableRNG streams derived from the seed, seeds 1–3, in the test's draw order.
using Test, StableRNGs, StaticArrays, Random, LinearAlgebra
using Distributions: Poisson
using VisualGeometryToolkitCore, VisualGeometryToolkitEstimation
import VisualGeometryToolkitCore as VgtkCore
using VisualGeometryToolkitCore: RawMosaic, sensor_fixture, parse_unit, photosite_colors, IGPrior, CameraModel, KannalaBrandt, px,
    SplineLens, Size2, convert_eltype, Lens, Perspective, Equidistant, Radial, Tangential, ThinPrism, Tilt, Additive

# --- SensorFixtures (sensor_model_tests.jl), verbatim ------------------------------------------------------------

const RGGB = UInt8[0 1; 3 2]
const BLACK = 512.0
const WHITE = 16383.0
const K_TRUE = 1.6
const SIGMA_R = 2.5
const SIGMA_EFF = sqrt(SIGMA_R^2 + 1 / 12)
synth_mosaic(raw; black = (BLACK, BLACK, BLACK, BLACK)) =
    RawMosaic(Matrix{UInt16}(raw), RGGB, black, WHITE, (0, 0), size(raw), (; model = "synthetic"))
code(rng, offset, pois, σ) =
    UInt16(clamp(round(offset + (pois === nothing ? 0.0 : rand(rng, pois) / K_TRUE) + σ * randn(rng)), 0, WHITE))
const LEVELS = [2.0 * 6000.0^((l - 1) / 15) for l in 1:16]
level_of_column(j) = cld(cld(j, 16), 2)
function ptc_frame(rng; offset = nothing)
    H, W = 16 * 16, 32 * 16
    pois = [Poisson(μ * K_TRUE) for μ in LEVELS]
    raw = Matrix{UInt16}(undef, H, W)
    for j in 1:W, i in 1:H
        raw[i, j] = code(rng, offset === nothing ? BLACK : offset[i, j], pois[level_of_column(j)], SIGMA_R)
    end
    return synth_mosaic(raw)
end
function scene_frame(rng; levels = LEVELS, light = 1.0, response = nothing, texture = nothing, σ = SIGMA_R, black = BLACK)
    H, W = 16 * 16, 32 * 16
    raw = Matrix{UInt16}(undef, H, W)
    for j in 1:W, i in 1:H
        μ = light * levels[level_of_column(j)] * (response === nothing ? 1.0 : response[i, j]) *
            (texture === nothing ? 1.0 : texture[i, j])
        raw[i, j] = code(rng, black, Poisson(μ * K_TRUE), σ)
    end
    return synth_mosaic(raw)
end

const SEC = parse_unit("s")
rn(σ) = sensor_fixture(; read_noise = (σ, σ, σ, σ)).model.noise
rse(σ, se) = sensor_fixture(; read_noise = (σ, σ, σ, σ), read_noise_std_error = (se, se, se, se)).model.noise
gain_response(rng) = [(1 + 0.01 * randn(rng)) * (iseven(i) && isodd(j) ? 1.02 : 1.0) for i in 1:256, j in 1:512]
const DIM_LEVELS = [2.0 * (1600.0 / 2)^((l - 1) / 15) for l in 1:16]
const TEXTURE = [1 + 0.1 * sinpi(i / 64) * cospi(j / 96) for i in 1:256, j in 1:512]

# The single frame with a DSNU (a per-photosite offset) and the sensor model whose Dark subtracts it.
function dsnu_frame(rng)
    offset = BLACK .+ 3.0 .* randn(rng, 256, 512)
    m = ptc_frame(rng; offset)
    fx = sensor_fixture(; black = (BLACK, BLACK, BLACK, BLACK), exposure = 0.8SEC,
                        dark = (; offset = Float32.(offset), variance = fill(Float32(SIGMA_EFF^2), size(offset)),
                                exposure_band = (0.5SEC, 1.0SEC), frames = 16))
    return m, fx.model
end
# The 24 whole blocks / 3 points frame of the convergence item.
function three_point_frame(rng)
    lv = LEVELS[[1, 4, 7, 10, 13, 16]]
    return synth_mosaic([code(rng, BLACK, Poisson(lv[cld(j, 16)] * K_TRUE), SIGMA_R) for i in 1:64, j in 1:96])
end

# The photon-transfer series against a ReadNoise Dict: two exposures, 4 frames each, a flicker frame, the longer one
# with dark-current noise σ_d = 20.
function readnoise_series(seed)
    response = gain_response(StableRNG(1000seed + 41))
    σd = 20.0
    t1, t2 = 0.01SEC, 0.02SEC
    groups = Dict(t1 => [scene_frame(StableRNG(1000seed + 100 + k); light = k == 2 ? 0.5025 : 0.5, response) for k in 1:4],
                  t2 => [scene_frame(StableRNG(1000seed + 110 + k); light = k == 2 ? 1.005 : 1.0, response, σ = sqrt(SIGMA_R^2 + σd^2)) for k in 1:4])
    darks = Dict(t1 => rn(SIGMA_EFF), t2 => rn(sqrt(SIGMA_EFF^2 + σd^2)))
    darks_se = Dict(t1 => rse(SIGMA_EFF, 0.01), t2 => rse(sqrt(SIGMA_EFF^2 + σd^2), 0.01))
    wrong = Dict(t1 => rn(SIGMA_EFF), t2 => rn(SIGMA_EFF))
    return (; groups, darks, darks_se, wrong, t1, t2)
end

# The photon-transfer series against a Darks (exposure bands): exact rational exposures, the longer one with a
# dark-current offset Δ and noise σ_d; the Darks from the fixture helper; the defect slots of the test.
function darks_series(seed)
    t1, t2 = (10 // 1000)SEC, (20 // 1000)SEC
    σd, Δ = 20.0, 1000.0
    groups = Dict(t1 => [scene_frame(StableRNG(1000seed + 300 + k); light = k == 2 ? 0.5025 : 0.5) for k in 1:4],
                  t2 => [scene_frame(StableRNG(1000seed + 310 + k); black = BLACK + Δ, σ = sqrt(SIGMA_R^2 + σd^2)) for k in 1:4])
    H, W, nd = 256, 512, 16
    σ0² = (SIGMA_EFF^2, SIGMA_EFF^2 + σd^2)
    band(t, off, v) = (; offset = fill(Float32(off), H, W), variance = fill(Float32(v), H, W), exposure_band = (t, t), frames = nd)
    darks = sensor_fixture(; black = (BLACK, BLACK, BLACK, BLACK), darks = [band(0.01SEC, BLACK, σ0²[1]), band(0.02SEC, BLACK + Δ, σ0²[2])]).characterisation.entries[100.0].black
    rn2(σ²) = sensor_fixture(; read_noise = ntuple(_ -> sqrt(σ²), 4)).model.noise
    float_keyed = Dict(0.01SEC => rn2(σ0²[1]), 0.02SEC => rn2(σ0²[2]))
    one_black = sensor_fixture(; black = (BLACK, BLACK, BLACK, BLACK), dark = band(0.01SEC, BLACK, σ0²[1]), exposure = 0.01SEC).model
    every = (; x = repeat(1:W; inner = H), y = repeat(1:H; outer = W), onset = fill(0.015SEC, H * W), size = (H, W))
    late = sensor_fixture(; defects = (; every..., onset = fill(1.0SEC, H * W))).model.defects
    half = sensor_fixture(; defects = every, exposure = 1.0SEC).model.defects
    small = sensor_fixture(; black = (BLACK, BLACK, BLACK, BLACK),
                           darks = [(; band(0.01SEC, BLACK, 1.0)..., offset = fill(Float32(BLACK), 8, 8), variance = fill(1.0f0, 8, 8))]).characterisation.entries[100.0].black
    return (; groups, darks, float_keyed, one_black, late, half, small, t1, t2)
end

# master_dark's 16 dark frames of 128 × 128: per-colour black, DSNU 3 codes, per-colour read noise, 8 hot and up to 4
# dead planted sites.
function dark_frames(rng)
    H, W, n = 128, 128, 16
    black = (512.0, 500.0, 520.0, 508.0)
    σr = (2.5, 2.3, 2.7, 2.4)
    col = collect(photosite_colors(synth_mosaic(zeros(UInt16, H, W))))
    offset = [black[col[i, j] + 1] + 3.0 * randn(rng) for i in 1:H, j in 1:W]
    σ = [σr[col[i, j] + 1] for i in 1:H, j in 1:W]
    sites = unique([CartesianIndex(rand(rng, 1:H), rand(rng, 1:W)) for _ in 1:12])
    for (q, p) in enumerate(sites)
        q <= 8 ? (offset[p] += 300) : (offset[p] = 0.0; σ[p] = 0.0)
    end
    return [synth_mosaic([code(rng, offset[i, j], nothing, σ[i, j]) for i in 1:H, j in 1:W]; black) for _ in 1:n]
end

# --- fit_scale (fit_scale_tests) ---------------------------------------------------------------------------------

const SCALE_PRIORS = (IGPrior(0.0, 0.0), IGPrior(1.0, 1.0), IGPrior(2.5, 0.125), IGPrior(0.0, 3.0), IGPrior(4.0, 0.0))
fit_scale_counted_grid() = [(prior, 7.5, Float64((k - model_dof) * dg)) for k in (4, 10, 50), model_dof in (4, 7), dg in (1, 2) for prior in SCALE_PRIORS]
fit_scale_jeffreys_grid() = [(IGPrior(0.0, 0.0), rss, dof) for (rss, dof) in ((7.5, 6.0), (0.3, 2.5), (1.0e3, 7.0))]
fit_scale_floor_grid() = vcat(
    [(IGPrior(0.0, 0.0), rss, dof) for (rss, dof) in ((0.0, 0.0), (0.0, 3.0), (1.0e-300, 1.0), (1.0, 1.0e300))],
    [(IGPrior(0.0, 3.0), 0.0, 0.0)],
    [(prior, rss, dof) for prior in SCALE_PRIORS for (rss, dof) in ((0.0, 0.0), (0.0, 1.0), (1.0, 0.0), (1.0e-320, 1.0e300))])
fit_scale_integer_args() = [(IGPrior(1.0, 1.0), 8, 4), (IGPrior(1.0, 1.0), 8.0, 4.0), (IGPrior(2.0, 0.1), 3, 2.5)]
fit_scale_random_grid(rng) = [(SCALE_PRIORS[rand(rng, 1:5)], 10.0^(6rand(rng) - 3), Float64(rand(rng, 0:60))) for _ in 1:200]

# --- static_lm (static_lm_tests) ---------------------------------------------------------------------------------

static_lm_cubic() = (xs = SVector(0.3, -0.2, 0.7), A = SMatrix{3, 3}(2.0, 0.3, -0.1, 0.3, 1.5, 0.2, -0.1, 0.2, 1.0),
    lb = SVector(-1.0, -1.0, -1.0), ub = SVector(1.0, 1.0, 1.0), s = SVector(1.0, 1.0, 1.0))
static_lm_starts(rng) = [SVector(2rand(rng) - 1, 2rand(rng) - 1, 2rand(rng) - 1) for _ in 1:50]

# --- pixel_lm_reference_tests' problems: the suite's own fixture file, included (one copy) ---------------------

Base.include(@__MODULE__, joinpath(@__DIR__, "..", "..", "..", "VisualGeometryToolkitCore", "test", "fixtures", "pixel_lm", "problems.jl"))

# --- the lens inverse (lens_inverse_tests' lenses and points) ----------------------------------------------------

inverse_lenses() = (
    Lens(Perspective(), Tilt(0.02, -0.015) ∘ Additive(Radial((-0.3, 0.09, -0.01)), Tangential(1.0e-3, 5.0e-4))),
    Lens(Equidistant(), Additive(Tangential(4.0e-4, -2.0e-4), ThinPrism((1.0e-3,), (-6.0e-4,))) ∘ Radial((0.03, -0.01, 0.002, -2.0e-4))),
    Lens(Perspective(), Additive(Radial((0.6, -0.4, 0.05), (0.9, -0.2, 0.03)), Tangential(1.5e-3, -8.0e-4))),
)
const INVERSE_POINTS = (SVector(0.0, 0.0), SVector(1.0e-9, -2.0e-9), SVector(1.0e-4, 3.0e-5), SVector(0.3, -0.2), SVector(-0.7, 0.55),
    SVector(0.1, -0.05), SVector(0.45, 0.3), SVector(1.5e-3, 1.5e-3), SVector(-2.0e-4, 1.0e-4))
inverse_points(rng) = [SVector(1.2rand(rng) - 0.6, 1.2rand(rng) - 0.6) for _ in 1:40]

# --- the spline projection (a SplineLens camera of a KB fisheye, its backward spline) ----------------------------

spline_camera() = CameraModel(SplineLens, CameraModel(300.0px, SA[500.0px, 500.0px]; lens = KannalaBrandt(-0.02)), Size2(1000.0, 1000.0);
    grid_size = (16, 16))
# Rays of the source fisheye at uniform pixels (unit directions), the projection solve's data.
function spline_rays(rng)
    src = CameraModel(300.0px, SA[500.0px, 500.0px]; lens = KannalaBrandt(-0.02))
    return [SVector{3, Float64}(VgtkCore.backproject(src, SVector(1000rand(rng), 1000rand(rng)))) for _ in 1:60]
end
