# Opt-in, real data: Core 0.10.1's multi-frame photon_transfer on the FX3's E4 series (each exposure's first 5 frames one
# group) with the E1_E2 darks of the same exposures (master_dark's ReadNoise per exposure), every RAW through Features'
# load_raw_mosaic. The regression value is the gain of the reference run (dgx01 scratch fx3-sensorchar/e4_series.jl):
# K = 11.330 e⁻/code, Birge-scaled standard error 0.0042, checked at |ΔK| ≤ 3.29·SE (two-sided 0.001). Skipped unless the
# data tier rc_sensor_characterisation_20261113 is under VGTK_DATA_ROOT, or Core is older than 0.10.1 (its photon_transfer
# takes no groups).

@testitem "photon_transfer on the FX3 E4 series (real data, opt-in)" begin
    import VisualGeometryToolkitCore as VgtkCore
    using VisualGeometryToolkitCore: master_dark, photon_transfer, PoissonGaussian, SensorModel, NoNoise, defect_onsets
    using LibRaw_jll: LibRaw_jll                               # the RAW decoder (Features' LibRaw extension)
    using Test
    tier = joinpath(get(ENV, "VGTK_DATA_ROOT", ""), "rc_sensor_characterisation_20261113")
    if !isdir(joinpath(tier, "E4")) || !isdir(joinpath(tier, "E1_E2"))
        @test_skip "the data tier rc_sensor_characterisation_20261113 is not under VGTK_DATA_ROOT ($(repr(get(ENV, "VGTK_DATA_ROOT", ""))))"
    elseif pkgversion(VgtkCore) < v"0.10.1"
        @test_skip "photon_transfer(groups, darks) needs VisualGeometryToolkitCore 0.10.1 (this is $(pkgversion(VgtkCore)))"
    else
        s = VgtkCore.parse_unit("s")
        exposure(name) = occursin('_', name) ? (p = split(name, '_'); parse(Float64, p[1]) / parse(Float64, p[2])) : parse(Float64, rstrip(name, 's'))
        arws(dir) = sort!(filter(endswith(".ARW"), readdir(dir; join = true)))
        t8 = (1 / 8000) * s                                     # the black and its defects: the E1 1/8000 s master dark
        md = master_dark([load_raw_mosaic(f) for f in arws(joinpath(tier, "E1_E2", "1_8000"))]; exposure_band = (t8, t8))
        model = SensorModel(md.dark, NoNoise(), defect_onsets([t8 => md.defects]))
        dirs = sort(filter(d -> isdir(joinpath(tier, "E4", d)), readdir(joinpath(tier, "E4"))); by = exposure)
        darks = Dict(exposure(d) * s => master_dark([load_raw_mosaic(f) for f in arws(joinpath(tier, "E1_E2", d))];
                                                    exposure_band = (exposure(d) * s, exposure(d) * s)).read_noise for d in dirs)
        groups = Dict(exposure(d) * s => [load_raw_mosaic(f) for f in first(arws(joinpath(tier, "E4", d)), 5)] for d in dirs)
        r = photon_transfer(groups, darks; sensor_model = model)
        K = VgtkCore.estimate(r)
        m = r.metadata
        @info "photon_transfer, FX3 E4 (recorded)" K = K.gain SE = K.std_error.gain birge_ratio = m.birge_ratio range_white = m.level_fraction
        @test K isa PoissonGaussian
        @test abs(K.gain - 11.330) <= 3.29 * 0.0042
        @test 0.28 <= m.level_fraction[1] && m.level_fraction[2] <= 0.99
    end
end
