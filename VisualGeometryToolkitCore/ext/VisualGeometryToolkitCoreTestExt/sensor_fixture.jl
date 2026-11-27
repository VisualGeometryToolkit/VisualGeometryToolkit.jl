# =============================================================================
# The ONE builder of sensor-model fixtures for tests (owner ruling D13; stub and docstring in src/quality/test_helpers.jl)
# =============================================================================
#
# Tests never construct a sensor component, characterisation or model themselves (check_duplication's
# `:sensor_component` pattern): they describe one here in plain numbers, and resolve the frame's model through the
# one resolver, as a consumer does.

import VisualGeometryToolkitCore as VgtkCore

# The source of one slot: one source for every slot, or a named tuple by slot (specified where it names none).
_slot_source(s::VgtkCore.SensorSource.T, slot) = s
_slot_source(s::NamedTuple, slot) = get(s, slot, VgtkCore.SensorSource.specified)

function sensor_fixture(; body::VgtkCore.DeviceIdentity = VgtkCore.DeviceIdentity(; make = "SYNTHETIC", model = "SENSOR"),
                        iso::Real = 100, exposure = 1.0 * VgtkCore._SECOND,
                        black = nothing, black_std_error = nothing, dark = nothing, darks = nothing,
                        gain = nothing, gain_std_error = nothing, read_noise = nothing, read_noise_std_error = nothing,
                        relative_noise = nothing, defects = nothing, source = VgtkCore.SensorSource.specified)
    channel = black === nothing ? VgtkCore.FileBlack() : VgtkCore.ChannelBlack(black, black_std_error; source = _slot_source(source, :black))
    (dark !== nothing && darks !== nothing) &&
        throw(ArgumentError("sensor_fixture: one form of the black slot's darks — `dark` (a Dark) or `darks` (a Darks), not both"))
    bs = _slot_source(source, :black)
    one_dark(d) = VgtkCore.Dark(channel, d.offset, d.variance, d.exposure_band, d.frames; source = bs)
    b = dark !== nothing ? one_dark(dark) : darks !== nothing ? VgtkCore.Darks(channel, [one_dark(d) for d in darks]; source = bs) : channel
    gain === nothing || read_noise !== nothing ||
        throw(ArgumentError("sensor_fixture: a gain makes a PoissonGaussian, which needs `read_noise` too"))
    (relative_noise !== nothing && read_noise !== nothing) &&
        throw(ArgumentError("sensor_fixture: one noise component — `relative_noise` or `read_noise` (with a `gain`), not both"))
    ns = _slot_source(source, :noise)
    n = relative_noise !== nothing ? VgtkCore.RelativeNoise(relative_noise; source = ns) :
        read_noise === nothing ? VgtkCore.NoNoise() :
        gain === nothing ? VgtkCore.ReadNoise(read_noise, read_noise_std_error; source = ns) :
        VgtkCore.PoissonGaussian(gain, read_noise, gain_std_error === nothing ? nothing : (gain = gain_std_error, σ = read_noise_std_error);
                             source = ns)
    d = defects === nothing ? VgtkCore.NoDefects() :
        VgtkCore.DefectList(defects.x, defects.y, defects.onset; size = defects.size, source = _slot_source(source, :defects))
    characterization = VgtkCore.SensorCharacterization(body; entries = [iso => (; black = b, noise = n, defects = d)])
    capture = VgtkCore.CaptureSettings(Dict(1 => (; exposure = VgtkCore.ExposureSettings(exposure, 8, iso),
                                              lens_settings = VgtkCore.LensSettings(14 * VgtkCore.mm))); body)
    return (; characterization, capture, model = VgtkCore.sensor_model(characterization, capture, 1))
end

function sensor_fixture(cs::VgtkCore.SensorCharacterization...)
    allequal(c -> c.body, cs) || throw(ArgumentError("sensor_fixture: the characterisations merged are of one body"))
    return VgtkCore.SensorCharacterization(first(cs).body; entries = [iso => e for c in cs for (iso, e) in c.entries])
end
