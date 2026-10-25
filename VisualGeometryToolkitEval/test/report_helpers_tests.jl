# =============================================================================
# report_helpers_tests.jl — the helpers UltimateTraveler (UT) and
# LocalOptimizationDoneRight (LODR) each hand-rolled, against their originals.
#
# Provenance of every expected value: the originals' functions are TRANSCRIBED
# verbatim below (named `ut_*` / `lodr_*`) and run on the same recorded inputs;
# neither project's environment is loaded here (UT's needs PythonCall/CondaPkg
# and the campaign pools). Sources:
#   UT   experiments/e2e_welsch_hq.jl `_anytime_replay` (392), curves_superansac_vs_e2e.jl
#        `_replay` (85) / `_f3` (57), common.jl `_q`/`_fmt` (22-23), cross_validation.jl
#        `tdv_tuning_spec` (31), prepare_inputs.jl `write_pixel_inputs` (42-62)
#   LODR accv2026/src/common.jl oracle floor (60-106), scripts/paper_numbers.jl `_read_tsv`,
#        src/superansac.jl `superansac_pixel_inputs!` (200), src/protocol_folds.jl `protocol_tuning` (41)
# =============================================================================

@testitem "report_helpers" begin
    import VisualGeometryToolkitCore as VgtkCore
    using Test, Random, JLD2, Printf, Statistics, Dates, StaticArrays
    using FileWatching.Pidfile: mkpidlock
    using VisualGeometryToolkitEval
    using VisualGeometryToolkitEval.Stats: eligible_maa, finite_median, maa_threshold, quantile_or,
        ImprovementTrace, trace_error_at, anytime_curve
    using VisualGeometryToolkitEval.Render: fmt3, write_tsv, read_tsv
    using VisualGeometryToolkitEval.Tuning: tuning_spec, tuning_table, layer_tuning, TUNING_TAGS
    using VisualGeometryToolkitEval.Experiments: write_oracle_floor, oracle_floor_done, oracle_floor_source,
        oracle_floors_path
    using VisualGeometryToolkitEval.Pools: export_pixel_inputs, ScoringPool, synthetic_e_pool_pair, GT_CONVENTION,
        load_pool
    using VisualGeometryToolkitCore: FundMat, sources, targets
    using VisualGeometryToolkitCore.Pools: PoolPair, pixel_csponds

    # ---- the originals, verbatim ------------------------------------------------
    ut_q(v, p) = isempty(v) ? NaN : quantile(v, p)
    ut_fmt(x) = isfinite(x) ? @sprintf("%.3f", x) : string(x)
    ut_f3(x) = @sprintf("%.3f", x)

    function ut_anytime_replay(d, tgrid)   # `_anytime_replay` with the grid given as an argument
        off, ft, fe = d["trace_offsets"], d["trace_t_ns"], d["trace_err"]
        n = length(off) - 1
        nb = length(tgrid)
        maa10 = fill(NaN, nb); med = fill(NaN, nb); nreach = zeros(Int, nb)
        e_at = Vector{Float64}(undef, n)
        for (b, t) in enumerate(tgrid)
            tns = t * 1e9
            for i in 1:n
                lo, hi = off[i], off[i+1] - 1
                e = Inf
                for k in lo:hi
                    ft[k] <= tns || break
                    e = fe[k]
                end
                e_at[i] = e
                isfinite(e) && (nreach[b] += 1)
            end
            maa10[b] = eligible_maa(e_at)
            med[b]   = ut_q(e_at, 0.5)
        end
        return tgrid, maa10, med, nreach
    end
    function ut_replay(d, i; t_ns = nothing, hyp = nothing)
        off, ft, fh, fe = d["trace_offsets"], d["trace_t_ns"], d["trace_hyp"], d["trace_err"]
        e = Inf
        for k in off[i]:(off[i+1]-1)
            t_ns !== nothing && ft[k] > t_ns && break
            hyp !== nothing && fh[k] > hyp && break
            e = fe[k]
        end
        return e
    end
    function lodr_read_tsv(path)
        lines = filter(l -> !isempty(strip(l)) && !startswith(l, "#"), readlines(path))
        hdr = split(lines[1], '\t')
        return [Dict(String(h) => String(v) for (h, v) in zip(hdr, split(l, '\t'))) for l in lines[2:end]]
    end

    # a recorded trace: 4 pairs; pair 2 never has an event, pair 4 fails late
    d = Dict("trace_offsets" => [1, 4, 4, 6, 9],
             "trace_t_ns"    => [1.0e5, 4.0e5, 9.0e5,  2.0e5, 8.0e5,  3.0e5, 6.0e5, 2.0e6],
             "trace_hyp"     => [1, 3, 8,  2, 5,  1, 4, 9],
             "trace_err"     => [30.0, 12.0, 4.0,  8.0, 2.0,  25.0, 11.0, Inf])
    tr = ImprovementTrace(d["trace_offsets"], d["trace_t_ns"], d["trace_hyp"], d["trace_err"])

    @testset "ImprovementTrace, trace_error_at, anytime_curve" begin
        @test length(tr) == 4
        for t_ns in (0.0, 1e5, 3.5e5, 4e5, 1e6, 5e6), i in 1:4
            @test isequal(trace_error_at(tr, i; t_ns), ut_replay(d, i; t_ns))
        end
        for hyp in (0, 1, 3, 5, 100), i in 1:4
            @test isequal(trace_error_at(tr, i; hyp), ut_replay(d, i; hyp))
            @test isequal(trace_error_at(tr, i; hyp, t_ns = 5e5), ut_replay(d, i; hyp, t_ns = 5e5))
        end
        @test trace_error_at(tr, 2) == Inf == ut_replay(d, 2)
        @test trace_error_at(tr, 1) == 4.0
        tgrid = [1e-4, 2.5e-4, 5e-4, 1e-3, 3e-3]
        _, maa_ref, med_ref, reach_ref = ut_anytime_replay(d, tgrid)
        @test isequal(anytime_curve(tr, tgrid), maa_ref)
        @test isequal(anytime_curve(tr, tgrid; stat = e -> ut_q(e, 0.5)), med_ref)
        @test anytime_curve(tr, tgrid; stat = e -> count(isfinite, e)) == reach_ref
        @test_throws DimensionMismatch ImprovementTrace([1, 3], [1.0], [1], [1.0])
        @test_throws ArgumentError ImprovementTrace([0, 1], [1.0], [1], [1.0])        # 0-based column
        @test_throws ArgumentError ImprovementTrace([1, 3, 2, 3], [1.0, 2.0], [1, 2], [1.0, 2.0])
        kept = anytime_curve(tr, [1e-3, 3e-3]; stat = identity)                         # keeps its argument
        @test kept[1] != kept[2] && kept[1] == [trace_error_at(tr, i; t_ns = 1e6) for i in 1:4]
    end

    @testset "quantile_or, fmt3" begin
        for v in (Float64[], [3.0], [1.0, 2.0, 7.0, 9.0]), p in (0.25, 0.5, 0.75)
            @test isequal(quantile_or(v, p), ut_q(v, p))
        end
        @test quantile_or(Float64[], 0.5; empty = Inf) == Inf
        @test quantile_or([1.0, 2.0], 0.5; empty = Inf) == 1.5
        for x in (0.0, 1.23456, -2.5, 1e6, NaN, Inf, -Inf)
            @test fmt3(x) == ut_fmt(x)                               # UT `_fmt`: string(x)
            isfinite(x) && @test fmt3(x) == ut_f3(x)
        end
        @test fmt3(NaN) == "NaN" && fmt3(Inf) == "Inf" && fmt3(-Inf) == "-Inf"
        @test fmt3(Inf; nonfinite = "NaN") == "NaN" && fmt3(-Inf; nonfinite = "-") == "-"
        @test ut_f3(NaN) == fmt3(NaN) && ut_f3(Inf) == fmt3(Inf)      # UT `_f3` spells Inf as Julia does
        @test fmt3(2) == "2.000"
    end

    @testset "write_tsv / read_tsv" begin
        mktempdir() do dir
            p = joinpath(dir, "sub", "t.tsv")
            @test write_tsv(p, ["arm", "x", "y"], [["a", 1.5, 2], ["b", "", 3]]; comments = ["first", "second note"]) == p
            @test read(p, String) == "# first\n# second note\narm\tx\ty\na\t1.5\t2\nb\t\t3\n"
            @test read_tsv(p) == lodr_read_tsv(p)
            @test read_tsv(p)[1] == Dict("arm" => "a", "x" => "1.5", "y" => "2")
            # a header-and-comments-only file reads as no rows
            q = write_tsv(joinpath(dir, "e.tsv"), ["a"], []; comments = ["c"])
            @test read_tsv(q) == lodr_read_tsv(q) == Dict{String,String}[]
            # UT's report layout: header and rows joined by hand, written with `write`
            txt = join(["# c", join(["metric", "ours"], '\t'), join(["m", ut_fmt(NaN)], '\t')], '\n') * "\n"
            write(joinpath(dir, "ut.tsv"), txt)
            @test read(write_tsv(joinpath(dir, "new.tsv"), ["metric", "ours"], [["m", fmt3(NaN)]]; comments = ["c"]), String) == txt
        end
    end

    @testset "tuning_spec under / over" begin
        cv = Dict{String,Any}("scene_split" => Dict{String,Any}("type" => "registered", "name" => "phototourism"))
        entry = Dict{String,Any}("type" => "windowed_bootstrap")
        cfg = Dict{String,Any}("tuning" => Dict{String,Any}("t" => entry))
        # defaults: unchanged behaviour
        base = tuning_spec(Dict{String,Any}("tuning" => Dict{String,Any}("t" => Dict{String,Any}(entry..., "cross_validation" => cv))), "t")
        full = Dict{String,Any}("tuning" => Dict{String,Any}("t" => Dict{String,Any}(entry..., "cross_validation" => cv)))
        @test tuning_spec(full, "t"; under = Dict{String,Any}(), over = Dict{String,Any}()) == base
        # UT `tdv_tuning_spec`: the campaign cross-validation layered beneath the entry
        ut_spec(name) = begin
            e = layer_tuning(Dict{String,Any}("cross_validation" => cv), cfg["tuning"][name])
            tuning_spec(Dict{String,Any}("tuning" => Dict{String,Any}(name => e)), name)
        end
        @test tuning_spec(cfg, "t"; under = Dict("cross_validation" => cv)) == ut_spec("t") == base
        # LODR `protocol_tuning`: [types] filtered by the scheme's fields, entry, then the cell's over
        types = Dict{String,Any}("cross_validation" => cv, "not_a_scheme_field" => Dict("x" => 1))
        over = Dict{String,Any}("scale_multiple_window" => Dict{String,Any}("lo" => 1.5, "hi" => 2.5))
        function lodr_protocol_tuning(name; over = Dict{String,Any}())
            en = cfg["tuning"][name]
            T = TUNING_TAGS["scheme"][en["type"]]
            ty = Dict{String,Any}(k => v for (k, v) in types if Symbol(k) in fieldnames(T))
            layered = layer_tuning(layer_tuning(ty, en), over)
            return tuning_spec(Dict{String,Any}("tuning" => Dict{String,Any}(name => layered)), name)
        end
        s = tuning_spec(cfg, "t"; under = types, over)
        @test s == lodr_protocol_tuning("t"; over)
        @test s != lodr_protocol_tuning("t")
        @test tuning_table(s)["scale_multiple_window"]["lo"] == 1.5
        @test_throws ArgumentError tuning_spec(cfg, "t"; under = Dict("cross_validation" => cv),
                                               over = Dict("typo_key" => 1))
        # a file path takes the keywords too
        mktempdir() do dir
            f = joinpath(dir, "c.toml")
            open(f, "w") do io
                println(io, "[tuning.t]\ntype = \"windowed_bootstrap\"")
            end
            @test tuning_spec(f, "t"; under = Dict("cross_validation" => cv)) == base
        end
    end

    @testset "oracle floor writer" begin
        # the original, verbatim (LODR common.jl 60-106), with its literal note
        note = "median over test pairs of pool-best model error (<TAG> = floor; " *
            "<TAG>_maa = mean pool-best mAA@10° ceiling); metric = problem-family model_imc; " *
            "selection-only bound (LO may cross); sources[<TAG>] = the pools/scenes of record"
        function lodr_write(path, tag, floor_v, maa; source, npairs, driver)
            mkpath(dirname(path))
            mkpidlock(path * ".lock"; wait = true, stale_age = 600) do
                f       = isfile(path) ? JLD2.load(path) : Dict{String,Any}()
                floors  = get(f, "floors",  Dict{String,Float64}())
                sources = get(f, "sources", Dict{String,Any}())
                floors[tag]          = floor_v
                floors["$(tag)_maa"] = maa
                sources[tag] = merge(source, Dict{String,Any}("npairs" => npairs, "driver" => String(driver),
                                                             "computed" => "FIXED"))
                JLD2.jldsave(path; floors, sources, note)
            end
            return path
        end
        @test maa_threshold() == 10.0
        mktempdir() do dir
            a, b = joinpath(dir, "a", "oracle_floors.jld2"), joinpath(dir, "b", "oracle_floors.jld2")
            src1 = oracle_floor_source("pools/E", :E, ["scene_b", "scene_a"])
            @test src1 == Dict("pool_dir" => "pools/E", "problem" => "E", "scenes" => ["scene_a", "scene_b"])
            src2 = oracle_floor_source("pools/H", "H", ["s"])
            for (p, w) in ((a, write_oracle_floor), (b, lodr_write))
                w(p, "PT_E", 0.75, 0.62; source = src1, npairs = 100, driver = "drv.jl")
                w(p, "PT_H", 1.25, 0.5; source = src2, npairs = 40, driver = :other)
            end
            ours, theirs = JLD2.load(a), JLD2.load(b)
            for k in ("PT_E", "PT_H")      # `computed` is the clock; everything else is identical
                @test delete!(copy(ours["sources"][k]), "computed") == delete!(copy(theirs["sources"][k]), "computed")
                @test ours["sources"][k]["computed"] isa String
            end
            @test ours["floors"] == theirs["floors"] == Dict("PT_E" => 0.75, "PT_E_maa" => 0.62, "PT_H" => 1.25, "PT_H_maa" => 0.5)
            @test ours["note"] == theirs["note"]
            @test Set(keys(ours)) == Set(keys(theirs)) == Set(["floors", "sources", "note"])
            # the done-predicate: the key alone is not enough, the source must match
            @test oracle_floor_done(a, "PT_E"; source = src1)
            @test !oracle_floor_done(a, "PT_E"; source = oracle_floor_source("pools/old", :E, ["scene_a", "scene_b"]))
            @test !oracle_floor_done(a, "PT_X"; source = src1)
            @test !oracle_floor_done(joinpath(dir, "none.jld2"), "PT_E"; source = src1)
            @test oracle_floors_path(dir) == joinpath(dir, "oracle_floors.jld2")
        end
    end

    @testset "export_pixel_inputs" begin
        p1, _ = synthetic_e_pool_pair(MersenneTwister(1))
        p2, _ = synthetic_e_pool_pair(MersenneTwister(2); n = 12, sibling_counts = [3, 1])
        p3, _ = synthetic_e_pool_pair(MersenneTwister(3); n = 14, sibling_counts = [2, 2])
        fields(p) = NamedTuple{fieldnames(typeof(p))}(Tuple(getfield(p, f) for f in fieldnames(typeof(p))))
        withsnn(p, s) = PoolPair{FundMat{Float64}}(; fields(p)..., snn = s)
        ps = [withsnn(p1, rand(MersenneTwister(5), Float32, VgtkCore.Pools.num_points(p1))),
              withsnn(p2, rand(MersenneTwister(6), Float32, VgtkCore.Pools.num_points(p2))),
              withsnn(p3, rand(MersenneTwister(7), Float32, VgtkCore.Pools.num_points(p3)))]
        stamps = (gt_convention = GT_CONVENTION, pool_rng = "LehmerRNG", julia_version = string(VERSION),
                  total_samples = 10, total_solutions = 12, mean_solutions_per_sample = 1.2,
                  provenance = Dict{String,Any}("driver" => "test", "commit" => "abc"))
        pool = ScoringPool(ps, :E, "scene"; scene_names = ["scene", "scene", "scene"], stamps)

        # the original, verbatim (UT prepare_inputs.jl 42-62), `now()` replaced by the fixed stamp
        match_matrix(cs) = [reduce(hcat, collect.(sources(cs)))' reduce(hcat, collect.(targets(cs)))']
        function ut_write_pixel_inputs(path, pool, scene, idx)
            issorted(idx; lt = <=) || throw(ArgumentError("pair indices of $scene must be strictly increasing"))
            ps = pool.pairs[idx]
            mkpath(dirname(path))
            jldsave(path; scene = String(scene), n_pairs = length(ps), pair_indices = collect(Int, idx),
                    pool_key = pool.pool_key,
                    matches = [Matrix{Float64}(match_matrix(pixel_csponds(p))) for p in ps],
                    snn = [Float64.(p.snn) for p in ps],
                    K1 = [Matrix{Float64}(p.K1) for p in ps], K2 = [Matrix{Float64}(p.K2) for p in ps],
                    R_gt = [Matrix{Float64}(p.R_gt) for p in ps], t_gt = [Vector{Float64}(p.t_gt) for p in ps],
                    provenance = "load_pool($(scene)) pool_key $(pool.pool_key); pixel_csponds + snn + K/R/t " *
                                 "as stored; pool pairs $(collect(idx)); julia $(VERSION)",
                    created = "FIXED")
            return path
        end
        mktempdir() do dir
            idx = [1, 3]
            ours = export_pixel_inputs(joinpath(dir, "x", "o.jld2"), pool, idx; created = "FIXED")
            theirs = ut_write_pixel_inputs(joinpath(dir, "t.jld2"), pool, "scene", idx)
            o, t = JLD2.load(ours), JLD2.load(theirs)
            @test o["pool_size"] == 3                                  # LODR's extra key
            @test Set(keys(o)) == Set(keys(t)) ∪ Set(["pool_size"])
            for k in keys(t)
                @test isequal(o[k], t[k])
            end
            @test o["matches"][1] isa Matrix{Float64} && size(o["matches"][1], 2) == 4
            @test_throws ArgumentError export_pixel_inputs(joinpath(dir, "bad.jld2"), pool, [2, 2])
            @test_throws ArgumentError export_pixel_inputs(joinpath(dir, "bad.jld2"), pool, Int[])
            @test !isfile(joinpath(dir, "bad.jld2"))
        end
    end
end
