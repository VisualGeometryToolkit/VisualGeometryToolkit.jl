# Fixtures of the features_image_circle family: the synthetic frames of VisualGeometryToolkitFeatures' suite
# (test/image_circle_tests.jl's ImageCircleFixtures — `cover`, `rim_frames`, `disc_frames` — and RimFixtures' `rim_synth`,
# and the RAW mosaics and the sensor-model mosaics built from them), each noise draw from `StableRNG(seed)` in place of
# the suite's `Xoshiro(7)` / `Xoshiro(3)`, so a fixture is the same on every Julia. Pure data: images, mosaics, a sensor
# model, an initial ellipse.
using Test, StableRNGs, Random
using Statistics: mean
using VisualGeometryToolkitCore, VisualGeometryToolkitFeatures, VisualGeometryToolkitEstimation
import VisualGeometryToolkitCore as VgtkCore
using VisualGeometryToolkitCore: RawMosaic, sensor_fixture

# `cover`, `rim_frames`, `disc_frames`, `rim_synth` (each draw from the caller's RNG): the suite's fixture file
Base.include(@__MODULE__, joinpath(@__DIR__, "..", "..", "..", "VisualGeometryToolkitFeatures", "test", "fixtures", "image_circle_frames.jl"))
fic_rim_frames(W, H, c, r, seed) = rim_frames(StableRNG(seed), W, H, c, r)
fic_disc_frames(W, H, c, r, n, seed) = disc_frames(StableRNG(seed), W, H, c, r, n)

# One frame of a disc covering three corners (the one-arc item).
function fic_one_arc_frame(W, H, c, r, seed)
    noise = StableRNG(seed)
    return [Float32(0.01 + 0.75 * cover(u, v, c..., r) + 0.01 * randn(noise)) for v in 1:H, u in 1:W]
end

# The RAW mosaics' metadata, and a frame as a 16-bit RGGB mosaic.
const FIC_MD = (; datetime_original = nothing, subsec = nothing, exposure_s = nothing, fnumber = nothing, focal_mm = nothing,
                iso = nothing, model = nothing, daylight_wb = nothing, linear_max = nothing, decoder = "test", libraw_version = "")
fic_mosaic(f) = RawMosaic(round.(UInt16, f .* 65535), UInt8[0 1; 3 2], (0.0, 0.0, 0.0, 0.0), 65535.0, (0, 0), size(f), FIC_MD)
# The second frame on a sensor window 2 rows lower (another raw_frame_map): refused with the first.
fic_shifted_mosaic(f) = RawMosaic(round.(UInt16, f .* 65535), UInt8[0 1; 3 2], (0.0, 0.0, 0.0, 0.0), 65535.0, (0, 0),
                                  (size(f, 1) + 2, size(f, 2)), FIC_MD; window_origin = (2, 0))

# The partial-rim scene (360 × 240, centre (181.3, 118.7), radius 100): frames, mosaics, the shifted mosaic.
const FIC_W, FIC_H, FIC_C = 360, 240, (181.3, 118.7)
const FIC_A = 0.1 * hypot(FIC_W, FIC_H)   # the suite's outlier half-width
function fic_partial_rim(seed)
    frames = fic_rim_frames(FIC_W, FIC_H, FIC_C, 100.0, seed)
    return (; frames, mosaics = fic_mosaic.(frames), shifted = fic_shifted_mosaic(frames[2]))
end

# The sensor-model scene: the ramp on a dark floor photographed by a Poisson-Gaussian sensor (gain 2 e⁻/code, read noise
# 2 codes, black 512, white 16383); the mosaics and the sensor model.
function fic_sensor_scene(seed)
    W, H, c, re, rk = 640, 480, (321.3, 240.7), 150.0, 120.0
    K, σ0, black, white = 2.0, 2.0, 512.0, 16383.0
    rng = StableRNG(100 + seed)
    mosaics = map(rim_synth(StableRNG(seed), W, H, c, re, rk; σ = 0.0, fl = 0.0, sat_at = 95.0, sat_radius = 20)) do f
        e = f .* (white - black) .* K
        codes = black .+ (e .+ sqrt.(e) .* randn(rng, size(e))) ./ K .+ σ0 .* randn(rng, size(e))
        RawMosaic(round.(UInt16, clamp.(codes, 0, white)), UInt8[0 1; 3 2], (black, black, black, black), white, (0, 0), (H, W), FIC_MD)
    end
    model = sensor_fixture(; gain = K, read_noise = (σ0, σ0, σ0, σ0)).model
    return (; mosaics, model)
end

# The centred ramps (800 × 600, centre exactly the image centre): the soft-footed ramps at, inside and beyond the old
# crossover, and the linear ramps at the crossovers.
const FIC_CW, FIC_CH = 800, 600
const FIC_RMAX = hypot(FIC_CW, FIC_CH) / 2
fic_soft_ramp(d, seed) = (re = 2FIC_RMAX / 3 + d;
    rim_synth(StableRNG(seed), FIC_CW, FIC_CH, (FIC_CW / 2, FIC_CH / 2), re, 0.87re; sat_radius = 0, foot = 4.0, σ = 0.001))
fic_linear_ramp(re, seed) = rim_synth(StableRNG(seed), FIC_CW, FIC_CH, (FIC_CW / 2, FIC_CH / 2), re, 0.87re; sat_radius = 0)
fic_centred_initial(off) = VgtkCore.Ellipse(VgtkCore.Point2(FIC_CW / 2 + off[1], FIC_CH / 2 + off[2]), FIC_RMAX, FIC_RMAX, 0.0)
