# GOLDEN-SHARED FIXTURE: ci/golden's fixtures include this file, and the recorder runs it against the pre-migration
# checkout (3d8a06da), in that checkout's environment. So: data constructors only, written against the API both
# sides have (the golden side includes it by its path in this tree, not through `pkgdir`, which would resolve to the
# recording checkout); it includes no other shared file (its includer does). Any change must leave every golden
# `#fixture` record unchanged (re-record twice, the records identical and their diff empty).
# The synthetic Poisson–Gaussian RGGB mosaics of sensor_model_tests.jl's SensorFixtures (and of the golden records of
# ci/golden/fixtures/core_sensor_lm.jl): the sensor's constants, one code, the photon-transfer frame and the scene frame.
# Data constructors only; include-once safe (functions and constants).
using Distributions: Poisson
using VisualGeometryToolkitCore: RawMosaic, sensor_fixture

const RGGB = UInt8[0 1; 3 2]
const BLACK = 512.0
const WHITE = 16383.0
const K_TRUE = 1.6                                  # electrons per code
const SIGMA_R = 2.5                                 # the Gaussian read noise, codes
const SIGMA_EFF = sqrt(SIGMA_R^2 + 1 / 12)          # with the code's rounding (uniform, variance 1/12: Sheppard)
synth_mosaic(raw; black = (BLACK, BLACK, BLACK, BLACK)) =
    RawMosaic(Matrix{UInt16}(raw), RGGB, black, WHITE, (0, 0), size(raw), (; model = "synthetic"))
# One code: an offset (the black, or a photosite's own), the photo-electrons Poisson(μK)/K (`pois`, or none) and a
# Gaussian read noise σ, rounded and clipped to [0, white].
code(rng, offset, pois, σ) =
    UInt16(clamp(round(offset + (pois === nothing ? 0.0 : rand(rng, pois) / K_TRUE) + σ * randn(rng)), 0, WHITE))
# 16 flat levels above black, geometric from 2 to 12000 codes: the low ones pin σ_r², the high ones K.
const LEVELS = [2.0 * 6000.0^((l - 1) / 15) for l in 1:16]
# The level of photosite column j in `ptc_frame` and `scene_frame`: LEVELS[level_of_column(j)] (16-photosite blocks, two per level).
level_of_column(j) = cld(cld(j, 16), 2)
# A photon-transfer scene: 16 × 32 blocks of the estimator's 8 × 8 cells (16 × 16 photosites), block column bj
# flat at LEVELS[cld(bj, 2)] — 32 blocks per level, so the estimator's 32 groups of 16 blocks hold one level each.
# `offset` (a per-photosite black, a DSNU) replaces the constant black.
function ptc_frame(rng; offset = nothing)
    H, W = 16 * 16, 32 * 16
    pois = [Poisson(μ * K_TRUE) for μ in LEVELS]
    raw = Matrix{UInt16}(undef, H, W)
    for j in 1:W, i in 1:H
        raw[i, j] = code(rng, offset === nothing ? BLACK : offset[i, j], pois[level_of_column(j)], SIGMA_R)
    end
    return synth_mosaic(raw)
end
# The photon-transfer scene of `ptc_frame` at `light` times its levels, each photosite's expected electrons times its
# fixed response `response[i, j]` (a G1/G2 gain mismatch, a PRNU: 1 when `nothing`) and `texture[i, j]` (a scene:
# 1 when `nothing`) — the frames of one exposure share the response and the texture, and differ only in their noise;
# `black` their offset (a dark current's shift at a long exposure).
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
# A ReadNoise of σ codes in every channel, with its standard error `se` when given (a dark series' read noise).
read_noise_model(σ) = sensor_fixture(; read_noise = (σ, σ, σ, σ)).model.noise
read_noise_model(σ, se) = sensor_fixture(; read_noise = (σ, σ, σ, σ), read_noise_std_error = (se, se, se, se)).model.noise
