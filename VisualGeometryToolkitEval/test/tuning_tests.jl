# =============================================================================
# tuning_tests.jl — THE cross-validation specification (src/tuning/*):
# spacings and the grid constructor, the persisted grid specs and `extend`,
# window rules, the `[tuning]` TOML parser
# (every tag, refusals, three-level inheritance, layering), lossless round
# trips and completeness of every spec type against configs/tuning.toml, the
# cross-validation (registered scene splits, every unit sampling, the budget
# draws, stable name-keyed seeds, signature restriction), the protocol hash,
# the tuning-artifact writer/reader (round trip, rebuild from the persisted
# spec alone, old-format refusal) and the run key over resolved specs.
# =============================================================================

@testitem "tuning" setup = [TuningFixtures] begin
    import VisualGeometryToolkitEval as VgtkEval
    using Test, TOML, JLD2, Random
    using StableRNGs: StableRNG
    using VisualGeometryToolkitEval
    using VisualGeometryToolkitEval.Tuning
    using VisualGeometryToolkitEval.Stats
    using VisualGeometryToolkitEval.Tuning: TUNING_DEFAULTS, read_tuning, layer_tuning, resolve_tuning!,
        SPLIT_REGISTRY
    import VisualGeometryToolkitCore
    using VisualGeometryToolkitCore.Pools: PoolPair
    const T = VgtkEval.Tuning
    const VGC = VisualGeometryToolkitCore

    cfg(name, table) = Dict{String,Any}("tuning" => Dict{String,Any}(name => with_cross_validation(table)))
    spec_of(table; name = "t") = fixture_spec(table; name)
    errmsg(f) = try f(); "" catch e; sprint(showerror, e) end

    @testset "spacings: the fencepost, exact endpoints, Base ranges, refusals" begin
        # d decades at p per decade: d·p intervals, d·p + 1 points, step 1/p in log10
        for (lo, hi, p) in ((1.0, 100.0, 5), (0.5, 5.0, 100), (1.0e-3, 1.0, 7), (0.8, 8.0e3, 20))
            g = grid(lo, hi, LogPerDecade(p))
            d = round(Int, log10(hi / lo))
            @test g isa Base.LogRange
            @test length(g) == d * p + 1 == npoints(lo, hi, LogPerDecade(p))
            @test g[1] === lo && g[end] === hi
            @test all(isapprox.(diff(log10.(g)), 1 / p; atol = 1.0e-12))
        end
        # a non-integer decade count rounds the interval count
        g = grid(1.0, 3.3, LogPerDecade(100))         # 0.5185 decades → 52 intervals
        @test length(g) == 53 && g[1] === 1.0 && g[end] === 3.3
        @test grid(1.0, 1000.0, LogPerDecade(1)) ≈ [1.0, 10.0, 100.0, 1000.0]
        # per octave: half-octave ladder
        g8 = grid(0.625, 40.0, LogPerOctave(2))       # 6 octaves → 13 points
        @test length(g8) == 13 && g8[1] === 0.625 && g8[end] === 40.0
        @test g8[3] ≈ 1.25 && all(isapprox.(diff(log2.(g8)), 0.5; atol = 1.0e-12))
        # counts, log and linear
        gl = grid(0.5, 16.0, LogCount(7))
        @test length(gl) == 7 && gl[1] === 0.5 && gl[end] === 16.0
        @test all(isapprox.(diff(log.(gl)), log(32.0) / 6; atol = 1.0e-12))
        gn = grid(1.0, 3.0, LinearCount(5))
        @test gn isa AbstractRange && gn == [1.0, 1.5, 2.0, 2.5, 3.0]
        @test grid(-1.0, 1.0, LinearCount(3)) == [-1.0, 0.0, 1.0]          # linear may cross zero
        # never fewer than two points
        @test grid(1.0, 1.01, LogPerDecade(1)) == [1.0, 1.01]
        @test grid(1.0, 1.01, LogPerOctave(1)) == [1.0, 1.01]
        @test_throws ArgumentError LogPerDecade(0)
        @test_throws ArgumentError LogPerOctave(0)
        @test_throws ArgumentError LogCount(1)
        @test_throws ArgumentError LinearCount(1)
        @test_throws ArgumentError grid(10.0, 1.0, LogCount(5))
        @test_throws ArgumentError grid(0.0, 1.0, LogCount(5))               # log needs lo > 0
        @test_throws ArgumentError grid(1.0, Inf, LinearCount(5))
    end

    @testset "grid specs and extend" begin
        base = SpacedGrid(1.0, 3.3, LogPerDecade(100))
        @test grid(base) == grid(1.0, 3.3, LogPerDecade(100))
        e1, n_lo = extend(base, (:lo, :hi); npts = 4)
        @test n_lo == 4 && e1 == SpacedGrid(1.0, 3.3, LogPerDecade(100), 4, 4)
        g, g1 = grid(base), grid(e1)
        @test length(g1) == length(g) + 8
        @test g1[n_lo .+ eachindex(g)] ≈ g                                  # the old points at +n_lo
        @test all(isapprox.(diff(log10.(g1)), log10(3.3) / 52; atol = 1.0e-12))   # the step continued
        # composition: 4 then 4 is the 8-point extension
        e2, n2 = extend(first(extend(base, (:lo,); npts = 4)), (:lo,); npts = 4)
        @test n2 == 4 && e2 == SpacedGrid(1.0, 3.3, LogPerDecade(100), 8, 0)
        @test grid(e2)[5:end] ≈ grid(first(extend(base, (:lo,); npts = 4)))
        # one side only; linear continuation
        eh, nh = extend(SpacedGrid(1.0, 3.0, LinearCount(5)), (:hi,); npts = 2)
        @test nh == 0 && grid(eh) ≈ [1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0]
        @test_throws ArgumentError extend(base, (:mid,); npts = 1)
        @test_throws ArgumentError extend(base, (:lo,); npts = 0)
        @test_throws ArgumentError extend(ListedGrid([1.0, 2.0]), (:lo,); npts = 1)
        @test_throws ArgumentError ListedGrid([2.0, 1.0])
        @test_throws ArgumentError SpacedGrid(1.0, 3.3, LogCount(5), -1, 0)
        @test grid(ListedGrid([1.0, 2.5])) == [1.0, 2.5]
    end

    @testset "extension_offset compares specs" begin
        base = SpacedGrid(1.0, 3.3, LogPerDecade(100))
        sup = SpacedGrid(1.0, 3.3, LogPerDecade(100), 4, 8)
        @test extension_offset(base, base) == 0
        @test extension_offset(base, sup) == 4                              # extended ⊇ base
        @test extension_offset(sup, base) === nothing                       # not vice versa
        @test extension_offset(base, SpacedGrid(1.0, 3.3, LogPerDecade(50))) === nothing
        @test extension_offset(base, SpacedGrid(1.0, 3.4, LogPerDecade(100))) === nothing
        @test extension_offset(base, ListedGrid([1.0, 2.0])) === nothing
        @test extension_offset(ListedGrid([1.0, 2.0]), ListedGrid([1.0, 2.0])) == 0
    end

    @testset "window rules" begin
        w = ScaleMultipleWindow(1.0, 3.3)
        @test window(FixedWindow(2.0, 5.0)) == (2.0, 5.0)
        @test window(StandardizedWindow(); scale_multiple_window = w) == (1.0, 3.3)
        @test window(CenteredWindow(10.0); center = 2.0) == (0.2, 20.0)
        loss_width_map(m, s) = m * s
        s2 = fill(4.0, 10)                                                  # ŝ = 2 at every quantile
        @test all(window(CoverageWindow(0.02, 0.98, 0.0, 0.0); scale_multiple_window = w, native = loss_width_map, s2) .≈ (2.0, 6.6))
        lo, hi = window(CoverageWindow(0.02, 0.98, 0.1, 1.0); scale_multiple_window = w, native = loss_width_map, s2)
        @test lo ≈ 1.0 * exp(-0.1) && hi ≈ 13.2 * exp(0.1)                 # ζ = 2 widening + margin
        # q_lo == q_hi is the ONE-ANCHOR case: both sides anchor on the same
        # √quantile(σ̂², q), and at q = 0.5 that is `reference_scale(s2)`. So
        # this IS the window-matched grid — the native image of the
        # scale-multiple window and nothing more — which is why a separate rule
        # for it (`StandardizedNativeWindow`) was deleted rather than kept.
        s2v = [1.0, 4.0, 9.0, 16.0, 25.0]                                   # median 9 ⇒ s0 = 3
        @test all(window(CoverageWindow(0.5, 0.5, 0.0, 0.0); scale_multiple_window = w,
                         native = loss_width_map, s2 = s2v) .≈ (3.0, 9.9))
        # and it is not accidentally equal to a spread band on the same σ̂²
        @test !all(window(CoverageWindow(0.02, 0.98, 0.0, 0.0); scale_multiple_window = w,
                          native = loss_width_map, s2 = s2v) .≈ (3.0, 9.9))
        @test occursin("needs `scale_multiple_window`", errmsg(() -> window(StandardizedWindow())))
        @test occursin("needs `s2`", errmsg(() -> window(CoverageWindow(0.1, 0.9, 0.0, 0.0); scale_multiple_window = w, native = loss_width_map)))
        @test occursin("needs `center`", errmsg(() -> window(CenteredWindow(2.0))))
        @test_throws ArgumentError ScaleMultipleWindow(3.3, 1.0)
        @test_throws ArgumentError CoverageWindow(0.9, 0.1, 0.0, 0.0)
        @test_throws ArgumentError CenteredWindow(1.0)
        @test_throws ArgumentError scale_bounds(fill(NaN, 3), (0.1, 0.9))
        @test grid_spec(SpacedAxis(StandardizedWindow(), LogCount(3)); scale_multiple_window = w) ==
              SpacedGrid(1.0, 3.3, LogCount(3))
        @test grid_spec(ListedAxis([1.0, 2.0])) == ListedGrid([1.0, 2.0])
    end

    # One instance of every concrete type of every tagged family.
    wb = spec_of(Dict("type" => "windowed_bootstrap")).scheme
    samples = Dict{String,Vector{Any}}(
        "spacing"   => [LogPerDecade(7), LogPerOctave(3), LogCount(5), LinearCount(4)],
        "window"    => [FixedWindow(1, 2), StandardizedWindow(),
                        CoverageWindow(0.1, 0.9, 0.2, 1.5), CoverageWindow(0.5, 0.5, 0.0, 0.0),
                        CenteredWindow(4.0)],
        "axis"      => [SpacedAxis(FixedWindow(1, 2), LogCount(3)), ListedAxis([1.0, 2.0, 3.0])],
        "grid"      => [SpacedGrid(1.0, 2.0, LogCount(3), 1, 2), ListedGrid([1.0, 2.0])],
        "criterion" => [MedianError(), FiniteMedian(), SceneMeanLowerMedian(), SceneMeanMedian(),
                        MeanAverageAccuracy(3.0)],
        "scheme"    => [wb, spec_of(Dict("type" => "coarse_fine")).scheme,
                        spec_of(Dict("type" => "joint_coarse_fine", "axes" => Dict(
                            "a" => Dict("coarse" => Dict("type" => "spaced", "window" => Dict("type" => "fixed", "lo" => 1.0, "hi" => 2.0),
                                                         "spacing" => Dict("type" => "log_count", "n" => 3)),
                                        "fine" => Dict("type" => "linear_count", "n" => 4), "expensive" => true)))).scheme,
                        spec_of(Dict("type" => "factorial", "axes" => Dict("x" => Dict("type" => "listed", "values" => [1.0])))).scheme],
        "scene_split"   => [RegisteredSceneSplit("x", ["b", "a"], ["c"])],
        "unit_sampling" => [ExhaustiveSampling(), StratifiedSampling(0.5, 3, 11, "StableRNG"),
                            UniformSampling(0.25, 7, 5, "StableRNG")],
    )
    untagged = Any[ScaleMultipleWindow(0.5, 2.0), GridExtension(3, 4),
                   CoarseFineAxis(SpacedAxis(FixedWindow(1, 2), LogCount(3)), GridExtension(2, 1), LinearCount(4), false),
                   Incumbent(Dict("a" => 1.5), 0.9, 200, 3),
                   CrossValidation(RegisteredSceneSplit("x", ["a"], ["c", "b"]), StratifiedSampling(0.5, 3, 11, "StableRNG"),
                                   UniformSampling(0.25, 7, 5, "StableRNG")),
                   BudgetSampling([1, 3], 9, 5, "StableRNG")]
    untagged_type(x) = only(K for K in keys(T._TYPE_KEYS) if x isa K)
    abstract_of = Dict("spacing" => GridSpacing, "window" => WindowRule, "axis" => TuningAxis, "grid" => GridSpec,
                       "criterion" => VgtkEval.Tuning.TuningCriterion,
                       "scheme" => TuningScheme, "scene_split" => SceneSplit, "unit_sampling" => UnitSampling)
    toml_text(x) = (io = IOBuffer(); TOML.print(io, Dict("x" => tuning_table(x)); sorted = true); TOML.parse(String(take!(io)))["x"])

    @testset "every tag parses; lossless round trips" begin
        @test Set(keys(TUNING_TAGS)) == Set(keys(samples))
        for (fam, xs) in samples
            @test Set(T._tag(x) for x in xs) == Set(keys(TUNING_TAGS[fam]))   # every tag covered
            for x in xs
                @test read_tuning(abstract_of[fam], tuning_table(x)) == x       # type → table → type
                @test read_tuning(abstract_of[fam], toml_text(x)) == x          # … through TOML text
                @test tuning_table(read_tuning(abstract_of[fam], toml_text(x))) == tuning_table(x)
            end
        end
        @test Set(untagged_type.(untagged)) == Set(keys(T._TYPE_KEYS))
        for x in untagged
            @test read_tuning(untagged_type(x), tuning_table(x)) == x
            @test read_tuning(untagged_type(x), toml_text(x)) == x
        end
        # every unit-sampling tag parses inside a spec, on either side
        for u in samples["unit_sampling"], side in ("validation", "test")
            cv = Dict("scene_split" => Dict("type" => "registered", "name" => "eth3d"), side => tuning_table(u))
            @test getfield(spec_of(Dict("type" => "coarse_fine", "cross_validation" => cv)).scheme.cross_validation,
                           Symbol(side)) == u
        end
        # every spacing, window and criterion tag parses inside a spec
        for s in samples["spacing"]
            sp = spec_of(Dict("type" => "windowed_bootstrap", "scale_multiple" => Dict("spacing" => tuning_table(s))))
            @test sp.scheme.scale_multiple.spacing == s
        end
        for wr in samples["window"]
            sp = spec_of(Dict("type" => "windowed_bootstrap", "loss_width" => Dict("window" => tuning_table(wr))))
            @test sp.scheme.loss_width.window == wr
        end
        for c in samples["criterion"]
            @test spec_of(Dict("type" => "coarse_fine", "criterion" => tuning_table(c))).scheme.criterion == c
        end
        # a spec round-trips through its own [tuning] table, under its name
        for (i, s) in enumerate(samples["scheme"])
            sp = TuningSpec("s$i", s)
            @test tuning_spec(tuning_toml(sp), "s$i") == sp
            io = IOBuffer(); TOML.print(io, tuning_toml(sp); sorted = true)
            @test tuning_spec(TOML.parse(String(take!(io))), "s$i") == sp
        end
    end

    @testset "refusals name the key path" begin
        m = errmsg(() -> spec_of(Dict("type" => "windowed_bootstrap", "bogus" => 1)))
        @test occursin("[tuning.t]", m) && occursin("bogus", m)
        m = errmsg(() -> spec_of(Dict("type" => "windowed_bootstrap", "scale_multiple" => Dict("spacing" => Dict("per_decad" => 3)))))
        @test occursin("[tuning.t.scale_multiple.spacing]", m) && occursin("per_decad", m)
        m = errmsg(() -> spec_of(Dict("type" => "windowed_bootstrap", "scale_multiple" => Dict("spacing" => Dict("type" => "log_weird")))))
        @test occursin("tuning.t.scale_multiple.spacing", m) && occursin("unknown spacing type", m) && occursin("log_weird", m)
        @test occursin("unknown scheme type", errmsg(() -> spec_of(Dict("type" => "nope"))))
        @test occursin("names no scheme", errmsg(() -> spec_of(Dict("criterion" => Dict()))))
        @test occursin("unknown window type", errmsg(() -> spec_of(Dict("type" => "coarse_fine",
                                                                         "coarse" => Dict("window" => Dict("type" => "wide"))))))
        @test occursin("unknown criterion type", errmsg(() -> spec_of(Dict("type" => "coarse_fine", "criterion" => Dict("type" => "mean")))))
        @test occursin("unknown rng", errmsg(() -> spec_of(Dict("type" => "windowed_bootstrap", "budget_sampling" => Dict("rng" => "Bogus")))))
        # Julia's own generators are refused: their streams change across Julia versions
        @test occursin("unknown rng", errmsg(() -> spec_of(Dict("type" => "windowed_bootstrap", "budget_sampling" => Dict("rng" => "MersenneTwister")))))
        @test occursin("tuning.t.budget_sampling.n_draws",
                       errmsg(() -> spec_of(Dict("type" => "windowed_bootstrap", "budget_sampling" => Dict("n_draws" => "x")))))
        @test occursin("no [tuning.u]", errmsg(() -> tuning_spec(cfg("t", Dict("type" => "coarse_fine")), "u")))
        @test occursin("no [tuning]", errmsg(() -> tuning_spec(Dict{String,Any}(), "t")))
        # a fixed window states its bounds (no tag default)
        @test occursin("missing key", errmsg(() -> spec_of(Dict("type" => "windowed_bootstrap",
                                                                  "loss_width" => Dict("window" => Dict("type" => "fixed", "lo" => 1.0))))))
        # the retired axes are unknown keys
        for retired in ("kernel" * "_width", "thres" * "hold")      # spelled in fragments: retired names
            @test occursin(retired, errmsg(() -> spec_of(Dict("type" => "windowed_bootstrap", retired => Dict()))))
        end
        # a value violating a constructor's invariant is refused with the path
        @test occursin("[tuning.t.scale_multiple_window]", errmsg(() -> spec_of(Dict("type" => "windowed_bootstrap",
                                                                     "scale_multiple_window" => Dict("lo" => 3.0, "hi" => 1.0)))))
        # a persisted (complete) table inherits nothing
        t = tuning_table(LogPerDecade(7)); delete!(t, "per_decade")
        @test occursin("missing key", errmsg(() -> read_tuning(GridSpacing, t)))
    end

    @testset "three-level inheritance; a different tag resets it" begin
        dflt = TUNING_DEFAULTS
        # 1. tag defaults: a bare tag takes its tag's parameters
        sp = spec_of(Dict("type" => "windowed_bootstrap", "scale_multiple" => Dict("spacing" => Dict("type" => "log_count"))))
        @test sp.scheme.scale_multiple.spacing == LogCount(dflt["tags"]["spacing"]["log_count"]["n"])
        # 2. scheme defaults override tag defaults (the probe's quarter decades)
        cf = spec_of(Dict("type" => "coarse_fine")).scheme
        @test cf.coarse.spacing == LogPerDecade(dflt["schemes"]["coarse_fine"]["coarse"]["spacing"]["per_decade"])
        @test cf.coarse.spacing.per_decade != dflt["tags"]["spacing"]["log_per_decade"]["per_decade"]
        @test cf.extension == GridExtension(dflt["types"]["grid_extension"]["npts"],
                                            dflt["schemes"]["coarse_fine"]["extension"]["max_rounds"])
        @test spec_of(Dict("type" => "windowed_bootstrap")).scheme.budget_sampling.n_draws ==
              dflt["schemes"]["windowed_bootstrap"]["budget_sampling"]["n_draws"]
        @test wb.budget_sampling.seed == dflt["types"]["budget_sampling"]["seed"]
        @test wb.cross_validation.validation == ExhaustiveSampling() && wb.cross_validation.test == ExhaustiveSampling()
        sv = spec_of(Dict("type" => "windowed_bootstrap", "cross_validation" => Dict(
            "scene_split" => Dict("type" => "registered", "name" => "heb"),
            "validation" => Dict("type" => "stratified", "fraction" => 0.25)))).scheme.cross_validation.validation
        tg = dflt["tags"]["unit_sampling"]["stratified"]
        @test sv == StratifiedSampling(0.25, tg["cap_per_scene"], tg["seed"], tg["rng"])
        @test wb.scale_multiple_window == ScaleMultipleWindow(dflt["types"]["scale_multiple_window"]["lo"],
                                                              dflt["types"]["scale_multiple_window"]["hi"])
        # 3. the config overrides one key, the rest inherits
        sp3 = spec_of(Dict("type" => "coarse_fine", "coarse" => Dict("spacing" => Dict("per_decade" => 8))))
        @test sp3.scheme.coarse.spacing == LogPerDecade(8) && sp3.scheme.coarse.window == cf.coarse.window
        # a different tag starts from ITS tag's defaults, not the inherited table
        sp4 = spec_of(Dict("type" => "coarse_fine", "coarse" => Dict("window" => Dict("type" => "centered"))))
        @test sp4.scheme.coarse.window == CenteredWindow(dflt["tags"]["window"]["centered"]["factor"])
        sp5 = spec_of(Dict("type" => "coarse_fine", "coarse" => Dict("spacing" => Dict("type" => "linear_count"))))
        @test sp5.scheme.coarse.spacing == LinearCount(dflt["tags"]["spacing"]["linear_count"]["n"])
        # layer_tuning: key-wise merge; a different type replaces
        lower = Dict("type" => "coarse_fine", "coarse" => Dict("spacing" => Dict("type" => "log_count", "n" => 5)), "edge_margin" => 2)
        up1 = layer_tuning(lower, Dict("coarse" => Dict("spacing" => Dict("n" => 7))))
        @test up1["coarse"]["spacing"] == Dict("type" => "log_count", "n" => 7) && up1["edge_margin"] == 2
        up2 = layer_tuning(lower, Dict("coarse" => Dict("spacing" => Dict("type" => "log_per_decade"))))
        @test up2["coarse"]["spacing"] == Dict("type" => "log_per_decade")
        @test spec_of(up2).scheme.coarse.spacing == cf.coarse.spacing    # the scheme default of that tag, not lower's n
        @test lower["coarse"]["spacing"]["n"] == 5                          # inputs untouched
    end

    @testset "completeness against configs/tuning.toml" begin
        concrete = Type[]
        for (_, tags) in TUNING_TAGS, (_, C) in tags
            push!(concrete, C)
        end
        append!(concrete, collect(keys(T._TYPE_KEYS)))
        # every field has its TOML key — the reader's keys ARE the field names
        for C in concrete
            @test Set(VGC.allowed_keys(C)) == Set(String.(fieldnames(C)))
        end
        # every key of the defaults file is a field of the type it defaults (no dead default)
        bad = String[]
        function walk(C, d, path)
            for (k, v) in d
                k == "type" && continue
                Symbol(k) in fieldnames(C) || (push!(bad, "$path.$k"); continue)
                FT = fieldtype(C, Symbol(k))
                fam = T._family(FT)
                if fam !== nothing && v isa AbstractDict
                    haskey(v, "type") || (push!(bad, "$path.$k: no type"); continue)
                    walk(TUNING_TAGS[fam][v["type"]], v, "$path.$k")
                elseif haskey(T._TYPE_KEYS, FT) && v isa AbstractDict
                    walk(FT, v, "$path.$k")
                end
            end
        end
        for (fam, tags) in TUNING_DEFAULTS["tags"], (tag, d) in tags
            haskey(TUNING_TAGS, fam) && haskey(TUNING_TAGS[fam], tag) || (push!(bad, "tags.$fam.$tag"); continue)
            walk(TUNING_TAGS[fam][tag], d, "tags.$fam.$tag")
        end
        tkeys = Dict(v => k for (k, v) in T._TYPE_KEYS)
        for (key, d) in TUNING_DEFAULTS["types"]
            haskey(tkeys, key) || (push!(bad, "types.$key"); continue)
            walk(tkeys[key], d, "types.$key")
        end
        for (tag, d) in TUNING_DEFAULTS["schemes"]
            haskey(TUNING_TAGS["scheme"], tag) || (push!(bad, "schemes.$tag"); continue)
            walk(TUNING_TAGS["scheme"][tag], d, "schemes.$tag")
        end
        @test isempty(bad)
        @test Set(keys(TUNING_DEFAULTS)) == Set(["tags", "types", "schemes"])
        @test Set(keys(TUNING_DEFAULTS["schemes"])) == Set(keys(TUNING_TAGS["scheme"]))
        @test Set(keys(TUNING_DEFAULTS["types"])) == Set(values(T._TYPE_KEYS))
        # every scheme resolves COMPLETELY from the defaults and a named scene
        # split (factorial: its axes are the config's) — and never without one:
        # the data a tuning runs on has no default
        for tag in keys(TUNING_TAGS["scheme"])
            @test spec_of(Dict("type" => tag)).scheme isa TUNING_TAGS["scheme"][tag]
            m = errmsg(() -> tuning_spec(Dict("tuning" => Dict("t" => Dict("type" => tag))), "t"))
            @test occursin("tuning.t.cross_validation.scene_split", m) && occursin("no `type`", m)
        end
    end

    @testset "tuning_hash identifies the protocol" begin
        a = spec_of(Dict("type" => "windowed_bootstrap"); name = "a")
        b = spec_of(Dict("type" => "windowed_bootstrap", "budget_sampling" => Dict("n_draws" => wb.budget_sampling.n_draws)); name = "b")
        @test a.name != b.name && tuning_hash(a) == tuning_hash(b)          # names and spelling do not matter
        @test length(tuning_hash(a)) == 16
        cv(; kw...) = Dict{String,Any}("scene_split" => Dict("type" => "registered", "name" => "phototourism"),
                                       (String(k) => v for (k, v) in kw)...)
        for over in (Dict("budget_sampling" => Dict("seed" => 8)),
                     Dict("scale_multiple" => Dict("spacing" => Dict("n" => 99))),
                     Dict("cross_validation" => cv(validation = Dict("type" => "stratified", "fraction" => 0.5))),
                     Dict("cross_validation" => cv(validation = Dict("type" => "stratified", "seed" => 3))),
                     Dict("cross_validation" => cv(test = Dict("type" => "uniform", "cap" => 9))),
                     Dict("cross_validation" => Dict("scene_split" => Dict("type" => "registered", "name" => "eth3d"))))
            @test tuning_hash(spec_of(merge(Dict{String,Any}("type" => "windowed_bootstrap"), over))) != tuning_hash(a)
        end
        # the RESOLVED scene lists are in the hashed table: the same split name
        # with other lists (a registry edit) is another protocol
        t = tuning_table(a)
        @test t["cross_validation"]["scene_split"]["validation_scenes"] ==
              sort(SPLIT_REGISTRY["splits"]["phototourism"]["validation_scenes"])
        moved = deepcopy(t)
        push!(moved["cross_validation"]["scene_split"]["test_scenes"], "an_added_scene")
        @test tuning_hash(TuningSpec("a", read_tuning(TuningScheme, moved))) != tuning_hash(a)
    end

    @testset "stable_seed: a fixed algorithm, keyed by name" begin
        # pinned against an independent SHA-256 (the first 8 bytes of
        # sha256("20260901\x1fbrandenburg_gate")): a change of the derivation
        # moves every sample of the stack and must fail here
        @test stable_seed(20260901, "brandenburg_gate") === 0x45a700d306efb507
        @test stable_seed(7, "budget", 2, 1) === 0xf5154d7e8f0e4d1c
        @test stable_seed(1, "a") != stable_seed(1, "b") != stable_seed(2, "b")
        @test stable_seed(1, "ab") != stable_seed(1, "a", "b")              # keys are separated
    end

    @testset "registered scene splits" begin
        for name in keys(SPLIT_REGISTRY["splits"])                         # every registered split is valid
            sp = registered_split(name)
            @test sp.name == name && isempty(intersect(sp.validation_scenes, sp.test_scenes))
            @test issorted(sp.validation_scenes) && issorted(sp.test_scenes)
        end
        @test registered_split("phototourism").validation_scenes[1] == "brandenburg_gate"
        @test occursin("no [splits.nope]", errmsg(() -> registered_split("nope")))
        @test occursin("both validation and test", errmsg(() -> RegisteredSceneSplit("x", ["a", "b"], ["b", "c"])))
        @test occursin("no test scenes", errmsg(() -> RegisteredSceneSplit("x", ["a"], String[])))
        @test occursin("duplicate validation", errmsg(() -> RegisteredSceneSplit("x", ["a", "a"], ["b"])))
        @test_throws ArgumentError RegisteredSceneSplit("", ["a"], ["b"])
        # a config names the split; listing scenes is allowed only verbatim
        reg = registered_split("eth3d")
        table(ss) = Dict("type" => "coarse_fine", "cross_validation" => Dict("scene_split" => ss))
        @test spec_of(table(Dict("type" => "registered", "name" => "eth3d"))).scheme.cross_validation.scene_split == reg
        @test spec_of(table(Dict("type" => "registered", "name" => "eth3d",
                                 "test_scenes" => reverse(reg.test_scenes)))).scheme.cross_validation.scene_split == reg
        m = errmsg(() -> spec_of(table(Dict("type" => "registered", "name" => "eth3d", "test_scenes" => ["office"]))))
        @test occursin("tuning.t.cross_validation.scene_split", m) && occursin("differs from [splits.eth3d]", m)
        @test occursin("no [splits.nope]", errmsg(() -> spec_of(table(Dict("type" => "registered", "name" => "nope")))))
        @test occursin("needs a `name`", errmsg(() -> spec_of(table(Dict("type" => "registered")))))
        @test occursin("unknown scene_split type", errmsg(() -> spec_of(table(Dict("type" => "listed")))))
        # unknown keys anywhere in the cross-validation are refused
        cvt(over) = Dict("type" => "coarse_fine", "cross_validation" =>
                         merge(Dict{String,Any}("scene_split" => Dict("type" => "registered", "name" => "heb")), over))
        @test occursin("bogus", errmsg(() -> spec_of(cvt(Dict("bogus" => 1)))))
        @test occursin("cap_per_scene", errmsg(() -> spec_of(cvt(Dict("validation" => Dict("type" => "uniform",
                                                                                              "cap_per_scene" => 2))))))
        @test occursin("unknown unit_sampling type", errmsg(() -> spec_of(cvt(Dict("test" => Dict("type" => "pooled"))))))
        # the retired unit-sampling tags name their successors (the tag rule)
        @test occursin("unit_sampling type \"all\" was renamed \"exhaustive\"",
                       errmsg(() -> spec_of(cvt(Dict("test" => Dict("type" => "all"))))))
        @test occursin("unit_sampling type \"simple_random\" was renamed \"uniform\"",
                       errmsg(() -> spec_of(cvt(Dict("validation" => Dict("type" => "simple_random", "fraction" => 0.5))))))
        for retired in ("val" * "_fraction", "per" * "_scene", "np" * "_val", "np" * "_test")   # spelled in fragments: retired names
            @test occursin(retired, errmsg(() -> spec_of(cvt(Dict("validation" => Dict("type" => "stratified", retired => 1))))))
        end
        @test occursin("unknown key", errmsg(() -> spec_of(Dict("type" => "windowed_bootstrap", "split" => Dict()))))
    end

    @testset "unit samplings: fraction, cap after the shuffle, sort last, determinism, stability" begin
        units = Dict(s => collect(1:40) for s in ("a", "b", "c", "d"))
        @test sample_units(ExhaustiveSampling(), units) == units
        # stratified: the fraction of each scene, from that scene's own name-keyed stream
        st = StratifiedSampling(0.25, 0, 5, "StableRNG")
        k = sample_units(st, units)
        @test all(length(k[s]) == 10 && issorted(k[s]) && allunique(k[s]) && k[s] ⊆ units[s] for s in keys(units))
        @test k == sample_units(st, units)                                    # deterministic
        for s in keys(units)
            @test k[s] == sort(shuffle(StableRNG(stable_seed(5, s)), 1:40)[1:10])
        end
        @test length(sample_units(StratifiedSampling(0.01, 0, 5, "StableRNG"), units)["a"]) == 1   # at least one
        @test sample_units(st, Dict("e" => Int[]))["e"] == Int[]             # an empty scene stays empty
        # the cap applies AFTER the shuffle: never a prefix (the prefix-bias bug)
        cap = sample_units(StratifiedSampling(1.0, 3, 5, "StableRNG"), units)
        @test all(length(cap[s]) == 3 for s in keys(units))
        @test any(cap[s] != [1, 2, 3] for s in keys(units))
        for s in keys(units)
            @test cap[s] == sort(shuffle(StableRNG(stable_seed(5, s)), 1:40)[1:3])
        end
        # fraction then cap: 50 % of 40 = 20, capped at 6 — the first 6 of the same permutation
        fc = sample_units(StratifiedSampling(0.5, 6, 5, "StableRNG"), units)
        @test all(fc[s] == sort(shuffle(StableRNG(stable_seed(5, s)), 1:40)[1:6]) for s in keys(units))
        # adding, removing or reordering OTHER scenes leaves a scene's sample unchanged
        more = merge(units, Dict("aa" => collect(1:40), "zz" => collect(1:7)))
        @test sample_units(st, more)["b"] == k["b"]
        @test sample_units(st, Dict("b" => units["b"]))["b"] == k["b"]
        # units are carried, not just indices
        names = Dict("a" => ["q$i" for i in 1:40])
        @test sample_units(st, names)["a"] == ["q$i" for i in k["a"]]
        # simple random: pooled over scenes, fraction then cap, back in data order
        sr = UniformSampling(0.1, 0, 9, "StableRNG")
        p = sample_units(sr, units)
        @test Set(keys(p)) == Set(keys(units)) && sum(length, values(p)) == 16
        @test all(issorted(p[s]) && p[s] ⊆ units[s] for s in keys(p)) && p == sample_units(sr, units)
        flat = [(s, i) for s in sort(collect(keys(units))) for i in 1:40]
        want = flat[sort(shuffle(StableRNG(stable_seed(9, "simple_random")), 1:160)[1:16])]
        @test sort([(s, i) for (s, v) in p for i in v]) == want
        pc = sample_units(UniformSampling(1.0, 5, 9, "StableRNG"), units)
        @test sum(length, values(pc)) == 5 && length(pc["a"]) != 5           # a pooled cap is no prefix of scene "a"
        # constructor refusals
        @test_throws ArgumentError StratifiedSampling(0.0, 0, 1, "StableRNG")
        @test_throws ArgumentError StratifiedSampling(1.5, 0, 1, "StableRNG")
        @test_throws ArgumentError StratifiedSampling(0.5, -1, 1, "StableRNG")
        @test_throws ArgumentError UniformSampling(0.5, -1, 1, "StableRNG")
        @test occursin("unknown rng", errmsg(() -> UniformSampling(0.5, 0, 1, "Bogus")))
    end

    @testset "validation_units / test_units; the driver's signature restricts the sampling" begin
        split = registered_split("heb")
        units = Dict(s => collect(1:30) for s in vcat(split.validation_scenes, split.test_scenes, ["Extra"]))
        cv = CrossValidation(split, StratifiedSampling(0.5, 0, 1, "StableRNG"), ExhaustiveSampling())
        v, t = validation_units(cv, units), test_units(cv, units)
        @test first.(v) == split.validation_scenes && first.(t) == split.test_scenes   # sorted, extra ignored
        @test all(length(u) == 15 for (_, u) in v) && all(u == collect(1:30) for (_, u) in t)
        @test isempty(intersect(first.(v), first.(t)))
        @test occursin("no units for the validation scenes [\"Alamo\"]",
                       errmsg(() -> validation_units(cv, Dict(s => units[s] for s in keys(units) if s != "Alamo"))))
        # a two-view driver accepts stratified (or all) validation; a pooled one is a MethodError
        two_view(cv::CrossValidation{<:SceneSplit,<:Union{ExhaustiveSampling,StratifiedSampling}}) = :ok
        pooled(cv::CrossValidation{<:SceneSplit,<:Union{ExhaustiveSampling,UniformSampling}}) = :ok
        cvs = CrossValidation(split, UniformSampling(0.5, 0, 1, "StableRNG"), ExhaustiveSampling())
        @test two_view(cv) === :ok && pooled(cvs) === :ok
        @test_throws MethodError two_view(cvs)
        @test_throws MethodError pooled(cv)
        # a data type declares its unit (a scoring pool's is its pair); there is no fallback
        @test sampling_unit(PoolPair) === PairUnit() && sampling_unit(PoolPair{Int}) === PairUnit()
        @test_throws MethodError sampling_unit(Int)
    end

    @testset "budget sampling: draws and their resampling" begin
        b = BudgetSampling([2, 5, 50], 4, 7, "StableRNG")
        d = budget_units(b, 20)
        @test length(d) == 12 && [(n, r) for (n, r, _) in d] == [(n, r) for n in [2, 5, 50] for r in 1:4]
        @test all(length(u) == min(n, 20) && issorted(u) && allunique(u) for (n, _, u) in d)
        @test d == budget_units(b, 20)
        @test d[1][3] == sort(shuffle(StableRNG(stable_seed(7, "budget", 2, 1)), 1:20)[1:2])
        @test rand(resampling_rng(b, 5)) != rand(budget_rng(b, 5, 0))         # the resampler is its own stream
        @test budget_units(b, 0) == [(n, r, Int[]) for n in [2, 5, 50] for r in 1:4]
        @test_throws ArgumentError BudgetSampling(Int[], 4, 7, "StableRNG")
        @test_throws ArgumentError BudgetSampling([0], 4, 7, "StableRNG")
    end

    # A product-grid run's specs from synthetic data: the persisted state. The
    # native maps are the caller's (scale multiple × scale, in the test).
    native(m, s) = m * s
    function run_state(scheme; s2 = collect(range(0.5, 2.0; length = 40)))
        w = scheme.scale_multiple_window
        gs = Dict{Symbol,GridSpec}(
            :outlier_halfwidth => grid_spec(scheme.outlier_halfwidth; scale_multiple_window = w, native, s2),
            :loss_width => grid_spec(scheme.loss_width; scale_multiple_window = w, native, s2),
            :scale_multiple => grid_spec(scheme.scale_multiple; scale_multiple_window = w))
        return gs
    end

    mktempdir() do dir
        spec = spec_of(Dict("type" => "windowed_bootstrap",
                            "cross_validation" => Dict("scene_split" => Dict("type" => "registered", "name" => "heb"),
                                                       "validation" => Dict("type" => "stratified", "fraction" => 0.5)),
                            "budget_sampling" => Dict("n_draws" => 3, "sizes" => [2, 5])); name = "miscal")
        gs = run_state(spec.scheme)
        realized = Dict(k => grid(g) for (k, g) in gs)
        units = Dict(s => collect(1:20) for s in vcat(validation_scenes(spec.scheme.cross_validation),
                                                      test_scenes(spec.scheme.cross_validation)))
        draws0 = budget_units(spec.scheme.budget_sampling, 37)
        val0 = validation_units(spec.scheme.cross_validation, units)

        @testset "artifact round trip" begin
            p = joinpath(dir, "run.jld2")
            save_tuning_artifact(p, spec; grids = gs, unit = PairUnit(), code = (VisualGeometryToolkitCore,),
                                 mode = "miscal", note = "x")
            @test JLD2.load(p, "tuning_format") == VgtkEval.ARTIFACT_FORMATS.tuning.version == 6
            art = load_tuning_artifact(p)
            @test art.spec == spec && art.hash == tuning_hash(spec) && art.unit === PairUnit()
            # the scene lists the run used are persisted with the spec
            @test JLD2.load(p, "tuning_spec")["cross_validation"]["scene_split"]["test_scenes"] ==
                  test_scenes(spec.scheme.cross_validation)
            @test art.grid_specs == gs
            @test all(all(art.grids[k] .=== realized[k]) for k in keys(gs))
            @test art["mode"] == "miscal" && art["note"] == "x" && !haskey(art, "tuning_spec")
            # this package always, plus the caller's code
            @test Set(keys(art.code)) == Set(["VisualGeometryToolkitEval", "VisualGeometryToolkitCore", "julia"])
            @test all(haskey(v, "version") && haskey(v, "git") for (k, v) in art.code if k != "julia")
            # the Julia version and the RNG algorithms the draws used (a bootstrap CI depends on Julia's own stream)
            @test art.code["julia"]["version"] == string(VERSION) && art.code["julia"]["rng"] == ["StableRNG"]
            @test_throws ArgumentError save_tuning_artifact(joinpath(dir, "bad.jld2"), spec; grids = gs, unit = PairUnit(),
                                                            code = (), tuning_hash = "x")
            @test_throws UndefKeywordError save_tuning_artifact(joinpath(dir, "bad.jld2"), spec; grids = gs, unit = PairUnit())
            @test_throws ArgumentError save_tuning_artifact(joinpath(dir, "bad.jld2"), spec; grids = gs, unit = PairUnit(),
                                                            code = ("VisualGeometryToolkitCore",))
        end

        @testset "rebuild a run from its persisted spec alone" begin
            art = load_tuning_artifact(joinpath(dir, "run.jld2"))
            sch = art.spec.scheme
            # the grids: rebuilt from the grid specs bit-exactly, and re-derived from
            # the spec's window rules over the same data to the same specs
            @test all(all(grid(art.grid_specs[k]) .=== realized[k]) for k in keys(gs))
            @test run_state(sch) == gs
            # the units and draws: from the spec's seeds and RNG algorithms alone
            @test budget_units(sch.budget_sampling, 37) == draws0
            @test validation_units(sch.cross_validation, units) == val0
            @test length(draws0) == 2 * 3 && all(length(r) == min(n, 37) for (n, _, r) in draws0)
            @test all(length(u) == 10 for (_, u) in val0) && val0[1][2] != val0[2][2]
            @test budget_rng(sch.budget_sampling, 5, 2) == budget_rng(spec.scheme.budget_sampling, 5, 2)
        end

        @testset "old and newer formats are refused" begin
            old = joinpath(dir, "old.jld2")
            JLD2.jldsave(old; a_grid = [1.0, 2.0], arms = ["x"])               # no stamp: format 1
            m = errmsg(() -> load_tuning_artifact(old))
            @test occursin(old, m) && occursin("tuning format v1", m) && occursin("regenerate it by re-running its tuning", m)
            v1 = joinpath(dir, "v1.jld2")
            JLD2.jldsave(v1; tuning_format = 1)
            @test_throws ArgumentError load_tuning_artifact(v1)
            # a v4 artifact (before the evaluated region) is refused, and the
            # refusal names what the CURRENT version added — a pre-v6 record
            # stores the tuned point as a grid INDEX, meaningless against any
            # other grid, so reading it through would be silently wrong.
            v4 = joinpath(dir, "v4.jld2")
            JLD2.jldsave(v4; tuning_format = 4)
            m4 = errmsg(() -> load_tuning_artifact(v4))
            @test occursin("tuning format v4", m4) && occursin("NATIVE VALUE", m4) &&
                  occursin("regenerate it by re-running its tuning", m4)
            newer = joinpath(dir, "newer.jld2")
            JLD2.jldsave(newer; tuning_format = VgtkEval.ARTIFACT_FORMATS.tuning.version + 1)
            @test occursin("newer than", errmsg(() -> load_tuning_artifact(newer)))
            @test occursin("no such tuning artifact", errmsg(() -> load_tuning_artifact(joinpath(dir, "none.jld2"))))
            # a stored spec whose hash does not match its table is refused
            forged = joinpath(dir, "forged.jld2")
            cp(joinpath(dir, "run.jld2"), forged)
            JLD2.jldopen(forged, "r+") do f
                delete!(f, "tuning_hash"); f["tuning_hash"] = "0000000000000000"
            end
            @test occursin("hashes to", errmsg(() -> load_tuning_artifact(forged)))
        end

    end

    @testset "resolve_tuning! and the run key cover inherited defaults" begin
        RR = VgtkEval.Experiments
        base = Dict{String,Any}("experiment" => Dict{String,Any}("dataset" => "d"),
                                "tuning" => Dict{String,Any}("t" => with_cross_validation(Dict{String,Any}("type" => "coarse_fine"))))
        r = resolve_tuning!(deepcopy(base))
        @test r["tuning"]["t"] == tuning_table(spec_of(Dict("type" => "coarse_fine")))
        @test r["tuning"]["t"]["cross_validation"]["scene_split"]["validation_scenes"] ==
              registered_split("phototourism").validation_scenes             # the lists are in the key
        @test haskey(r["tuning"]["t"]["coarse"]["spacing"], "per_decade")     # the inherited default is explicit
        # stating an inherited value does not move the key; changing it does
        dflt_n = TUNING_DEFAULTS["tags"]["spacing"]["log_count"]["n"]
        same = deepcopy(base); same["tuning"]["t"]["fine"] = Dict{String,Any}("n" => dflt_n)
        changed = deepcopy(base); changed["tuning"]["t"]["fine"] = Dict{String,Any}("n" => dflt_n + 1)
        @test RR.run_key(same) == RR.run_key(base)
        @test RR.run_key(changed) != RR.run_key(base)
        @test base["tuning"]["t"] == with_cross_validation(Dict("type" => "coarse_fine"))   # run_key leaves the config untouched
        @test resolve_tuning!(Dict{String,Any}("x" => 1)) == Dict("x" => 1)
    end
end
