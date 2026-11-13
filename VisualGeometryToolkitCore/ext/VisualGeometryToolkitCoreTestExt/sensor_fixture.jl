# =============================================================================
# The ONE builder of sensor-model fixtures for tests (owner ruling D13; stub and docstring in src/quality/test_helpers.jl)
# =============================================================================
#
# Tests never construct a sensor component, characterisation or model themselves (check_duplication's
# `:sensor_component` pattern): they describe one here in plain numbers, and resolve the frame's model through the
# one resolver, as a consumer does.

const _VGC = VisualGeometryToolkitCore

# The source of one slot: one source for every slot, or a named tuple by slot (specified where it names none).
_slot_source(s::_VGC.SensorSource.T, slot) = s
_slot_source(s::NamedTuple, slot) = get(s, slot, _VGC.SensorSource.specified)

function sensor_fixture(; body::_VGC.DeviceIdentity = _VGC.DeviceIdentity(; make = "SYNTHETIC", model = "SENSOR"),
                        iso::Real = 100, exposure = 1.0 * _VGC._SECOND,
                        black = nothing, black_std_error = nothing, dark = nothing,
                        gain = nothing, gain_std_error = nothing, read_noise = nothing, read_noise_std_error = nothing,
                        relative_noise = nothing, defects = nothing, source = _VGC.SensorSource.specified)
    channel = black === nothing ? _VGC.FileBlack() : _VGC.ChannelBlack(black, black_std_error; source = _slot_source(source, :black))
    b = dark === nothing ? channel :
        _VGC.Dark(channel, dark.offset, dark.variance, dark.exposure_band, dark.frames; source = _slot_source(source, :black))
    gain === nothing || read_noise !== nothing ||
        throw(ArgumentError("sensor_fixture: a gain makes a PoissonGaussian, which needs `read_noise` too"))
    (relative_noise !== nothing && read_noise !== nothing) &&
        throw(ArgumentError("sensor_fixture: one noise component — `relative_noise` or `read_noise` (with a `gain`), not both"))
    ns = _slot_source(source, :noise)
    n = relative_noise !== nothing ? _VGC.RelativeNoise(relative_noise; source = ns) :
        read_noise === nothing ? _VGC.NoNoise() :
        gain === nothing ? _VGC.ReadNoise(read_noise, read_noise_std_error; source = ns) :
        _VGC.PoissonGaussian(gain, read_noise, gain_std_error === nothing ? nothing : (gain = gain_std_error, σ = read_noise_std_error);
                             source = ns)
    d = defects === nothing ? _VGC.NoDefects() :
        _VGC.DefectList(defects.x, defects.y, defects.onset; size = defects.size, source = _slot_source(source, :defects))
    characterisation = _VGC.SensorCharacterisation(body; entries = [iso => (; black = b, noise = n, defects = d)])
    capture = _VGC.CaptureSettings(Dict(1 => (; exposure = _VGC.ExposureSettings(exposure, 8, iso),
                                              lens_settings = _VGC.LensSettings(14 * _VGC.mm))); body)
    return (; characterisation, capture, model = _VGC.sensor_model(characterisation, capture, 1))
end

function sensor_fixture(cs::_VGC.SensorCharacterisation...)
    allequal(c -> c.body, cs) || throw(ArgumentError("sensor_fixture: the characterisations merged are of one body"))
    return _VGC.SensorCharacterisation(first(cs).body; entries = [iso => e for c in cs for (iso, e) in c.entries])
end
