# DarksAndFlats (raw/sensor_estimators.jl): dark bursts and a photon-transfer series into ONE body's characterisation at
# one ISO, by MasterDark, defect_onsets and PhotonTransfer alone. Its result is bit-identical to the characterisation a
# caller assembles from the three estimators by hand (RadiometricCamera's experiments did, step for step, as below).

@testitem "sensor model: DarksAndFlats — the hand-assembled characterisation, bit for bit; the kind, the context's guards" setup = [SensorFixtures] begin
    import VisualGeometryToolkitCore as VgtkCore
    using VisualGeometryToolkitCore
    using VisualGeometryToolkitCore: parse_unit, SensorSource, Darks, ReturnCode, PhotonTransferMetadata
    using StableRNGs: StableRNG
    s = parse_unit("s")
    t1, t2 = 0.01s, 0.02s
    σd = 20.0                                                    # the longer exposure's dark-current noise
    H, W = 256, 512
    # the dark bursts: per photosite a fixed offset (a DSNU), a hot photosite in both bursts and one at t2 only
    rng = StableRNG(51)
    offset = [BLACK + 3.0 * randn(rng) for _ in 1:H, _ in 1:W]
    offset[40, 70] += 400
    offset2 = copy(offset); offset2[100, 200] += 400
    burst(rng, off, σ) = [synth_mosaic([code(rng, off[i, j], nothing, σ) for i in 1:H, j in 1:W]) for _ in 1:4]
    darks = Dict(t1 => burst(StableRNG(52), offset, SIGMA_R), t2 => burst(StableRNG(53), offset2, sqrt(SIGMA_R^2 + σd^2)))
    flats = Dict(t1 => [scene_frame(StableRNG(60 + k); light = 0.5) for k in 1:4],
                 t2 => [scene_frame(StableRNG(70 + k); σ = sqrt(SIGMA_R^2 + σd^2)) for k in 1:4])
    body = DeviceIdentity(; make = "SONY", model = "ILME-FX3", serial = "5012345")
    iso = 100
    false_defects = 0.01

    # by hand, as RadiometricCamera's characterise_sensor assembled it
    master(frames, t) = (r = solve(EstimationProblem(Dark, frames, (; exposure_band = (t, t))), MasterDark(; false_defects));
                         (; dark = r.u, r.metadata.read_noise, r.metadata.defects))
    md = master(darks[t1], t1)
    masks = Pair[]; rns = Dict{Any, Any}(); dark_maps = VgtkCore.Dark[]
    for t in (t1, t2)
        m = t == t1 ? md : master(darks[t], t)
        push!(masks, t => m.defects); rns[t] = m.read_noise; push!(dark_maps, m.dark)
    end
    black = Darks(md.dark.black, dark_maps; source = SensorSource.measured)
    defects = defect_onsets(identity.(masks))
    model = SensorModel(md.dark, NoNoise(), defect_onsets([t1 => md.defects]))
    fit = solve(EstimationProblem(PoissonGaussian, Dict(k => v for (k, v) in flats),
                                  (; darks = Dict(k => v for (k, v) in rns), sensor_model = model)),
                PhotonTransfer())
    K = fit.u
    noise = PoissonGaussian(K.gain, md.read_noise.σ, (gain = K.std_error.gain, σ = md.read_noise.std_error);
                            source = SensorSource.measured)
    hand = SensorCharacterization(body; entries = [iso => (; black, noise, defects)])

    # the builder
    prob = EstimationProblem(SensorCharacterization, (darks, flats), (; body, iso, reference = t1))
    r = solve(prob, DarksAndFlats(; dark = MasterDark(; false_defects)))
    c = r.u
    @test c isa SensorCharacterization && isequal(c, hand) && c == hand
    @test isequal(c.body, hand.body) && keys(c.entries) == keys(hand.entries)
    e, h = c.entries[100.0], hand.entries[100.0]
    @test isequal(e.black, h.black) && isequal(e.noise, h.noise) && isequal(e.defects, h.defects)
    @test e.noise === h.noise                                    # isbits: identical
    @test all(isequal(a.offset, b.offset) && isequal(a.variance, b.variance) && a.exposure_band == b.exposure_band &&
              a.frames == b.frames && a.black == b.black for (a, b) in zip(e.black.darks, h.black.darks))
    @test e.defects.x == h.defects.x && e.defects.y == h.defects.y && e.defects.onset == h.defects.onset
    @test length(e.defects.x) == 2                               # the two hot photosites (planted), each at its onset
    # the result: the photon-transfer fit's convergence; the metadata the fit and the per-burst master darks
    @test r.retcode === fit.retcode === ReturnCode.Success && r.stop_rule === fit.stop_rule && r.n == fit.n
    @test r.stats == fit.stats && isequal(r.rms, fit.rms)
    @test r.metadata.series.u === fit.u && r.metadata.series.metadata isa PhotonTransferMetadata
    @test isequal(r.metadata.series.metadata, fit.metadata)
    @test sort(collect(keys(r.metadata.darks)); by = t -> VgtkCore.ustrip(s, t)) == [t1, t2]
    @test isequal(r.metadata.darks[t1].metadata.read_noise, md.read_noise)
    @test r.metadata.darks[t1].metadata.defects == md.defects
    # its kind; the problem's default is the builder at its defaults (MasterDark's false_defects is 0.01)
    @test VgtkCore.estimator_kind(DarksAndFlats()) === FitKind()
    @test VgtkCore.default_algorithm(prob) == DarksAndFlats()
    @test isequal(solve(prob).u, c)
    # the context: every key required, none other; the reference one of the bursts' exposures
    @test_throws ArgumentError solve(EstimationProblem(SensorCharacterization, (darks, flats), (; body, iso)), DarksAndFlats())
    @test_throws ArgumentError solve(EstimationProblem(SensorCharacterization, (darks, flats),
                                                       (; body, iso, reference = t1, extra = 1)), DarksAndFlats())
    @test_throws ArgumentError solve(EstimationProblem(SensorCharacterization, (darks, flats),
                                                       (; body, iso, reference = 0.5s)), DarksAndFlats())
    # a group's exposure without a dark burst is PhotonTransfer's refusal, passed through
    @test_throws ArgumentError solve(EstimationProblem(SensorCharacterization, (Dict(t1 => darks[t1]), flats),
                                                       (; body, iso, reference = t1)), DarksAndFlats())
    # the traits gate over Core holds with the builder in it
    @test isempty(VgtkCore.check_algorithm_traits(VisualGeometryToolkitCore))
end
