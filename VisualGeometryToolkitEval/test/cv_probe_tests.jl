# =============================================================================
# cv_probe_tests.jl — THE coarse-fine search (Evaluation cv_probe.jl): its
# scalar forms under a `coarse_fine` spec (coarse expand-on-boundary probe →
# fine free argmin → read-side refusal) and the joint search under a
# `joint_coarse_fine` spec. Every parameter (coarse spacing, extension npts /
# max_rounds, edge margin, fine spacing, criterion, expensive) is the spec's.
# =============================================================================

@testitem "cv_probe" setup = [TuningFixtures] begin
    using Test, Logging
    using VisualGeometryToolkitEval.Evaluation: probe_curve, fine_stage, coarse_fine_cv, enforce_probe_interior
    using VisualGeometryToolkitEval.Tuning: tuning_spec, SpacedGrid, LogCount, LogPerDecade, grid, extend

    cf(over = Dict()) = fixture_spec(merge(Dict{String,Any}("type" => "coarse_fine"), over); name = "p").scheme
    scheme = cf()

    @testset "single CV path (cv_probe)" begin
        quiet(f) = with_logger(f, NullLogger())
        # the objective: the validation errors at x (one here); the spec's
        # criterion (finite median) aggregates them
        obj(x) = [(log10(x) - 0.3)^2]

        @testset "interior argmin — no extension" begin
            coarse = SpacedGrid(0.1, 100.0, LogCount(21))
            rec = quiet(() -> probe_curve(obj, coarse, scheme; method_name = :t))
            @test rec[:boundary_hit] == false
            @test rec[:n_extensions] == 0
            @test rec[:coarse_spec] == coarse
            @test isapprox(log10(rec[:x_star]), 0.3; atol = 0.2)
            @test length(grid(rec[:coarse_spec])) == 21 == length(rec[:curve_ys])
            @test rec[:tuning_criterion] === :median
            @test enforce_probe_interior(rec; context = "t") === nothing
        end

        @testset "expand-on-boundary recovers a low optimum" begin
            lowobj(x) = [(log10(x) + 1.6)^2]          # optimum at 10^-1.6
            coarse = SpacedGrid(0.1, 10.0, LogPerDecade(4))    # 9 points, quarter decades
            rec = quiet(() -> probe_curve(lowobj, coarse, scheme; method_name = :lo))
            @test rec[:n_extensions] >= 1
            @test rec[:boundary_hit] == false
            @test isapprox(log10(rec[:x_star]), -1.6; atol = 0.3)
            # the extension is recorded in the spec and continues the quarter-decade lattice
            s = rec[:coarse_spec]
            @test s.n_lo == rec[:n_extensions] * scheme.extension.npts && s.n_hi == 0
            xs = grid(s)
            @test length(xs) == 9 + rec[:n_extensions] * scheme.extension.npts == length(rec[:curve_ys])
            @test issorted(xs) && all(isapprox.(diff(log10.(xs)), 0.25; atol = 1e-12))
            @test xs[s.n_lo+1:end] ≈ grid(coarse)               # the probed points, by index
        end

        @testset "budget exhaustion flags boundary_hit; readers refuse" begin
            farobj(x) = [log10(x)]                    # monotone — no interior optimum
            rec = quiet(() -> probe_curve(farobj, SpacedGrid(1.0, 100.0, LogCount(9)), scheme; method_name = :far))
            @test rec[:boundary_hit] == true
            @test rec[:n_extensions] == scheme.extension.max_rounds   # low side exhausted
            @test_throws ErrorException enforce_probe_interior(rec; context = "far")
            @test_logs (:warn, r"FORENSIC") enforce_probe_interior(rec; context = "far", allow_edge = true)
            @test enforce_probe_interior(false; context = "flag") === nothing
            @test_throws ErrorException enforce_probe_interior(true; context = "flag")
            # the budget is the spec's: one more round allowed, one more extension made
            more = cf(Dict("extension" => Dict("max_rounds" => 3)))
            rec3 = quiet(() -> probe_curve(farobj, SpacedGrid(1.0, 100.0, LogCount(9)), more; method_name = :far))
            @test rec3[:n_extensions] == 3
        end

        @testset "edge margin is the spec's" begin
            # the argmin one point inside the edge extends at margin 1, not at margin 0
            nearobj(x) = [(log10(x) - log10(grid(SpacedGrid(1.0, 100.0, LogCount(9)))[2]))^2]
            r1 = quiet(() -> probe_curve(nearobj, SpacedGrid(1.0, 100.0, LogCount(9)), scheme; method_name = :m))
            r0 = quiet(() -> probe_curve(nearobj, SpacedGrid(1.0, 100.0, LogCount(9)),
                                         cf(Dict("edge_margin" => 0)); method_name = :m))
            @test r1[:n_extensions] >= 1 && r0[:n_extensions] == 0
        end

        @testset "fine stage + coarse_fine_cv" begin
            coarse = SpacedGrid(0.1, 100.0, LogCount(21))
            rec = quiet(() -> probe_curve(obj, coarse, scheme; method_name = :t))
            fs = fine_stage(obj, rec, scheme)
            @test length(fs.grid) == scheme.fine.n == length(fs.curve)
            @test fs.spec == SpacedGrid(fs.bracket..., scheme.fine) && collect(grid(fs.spec)) == fs.grid
            @test fs.bracket[1] < fs.x_star < fs.bracket[2]
            @test 1 < fs.idx < length(fs.grid)
            @test isapprox(log10(fs.x_star), 0.3; atol = 0.02)
            s50 = cf(Dict("fine" => Dict("n" => 50)))
            cv = quiet(() -> coarse_fine_cv(obj, coarse, s50; method_name = :t))
            @test cv[:x_star_fine] ≈ fine_stage(obj, rec, s50).x_star
            @test length(grid(cv[:fine_spec])) == 50
        end

        @testset "batched form == point-wise, cell for cell" begin
            using VisualGeometryToolkitEval.Evaluation: coarse_fine_cv_batched
            # three cells with different fates: interior, a low optimum that
            # extends, a monotone objective that exhausts its budget
            objs = Dict(:mid => obj, :low => x -> [(log10(x) + 1.6)^2], :far => x -> [log10(x)])
            coarse = Dict(:mid => SpacedGrid(0.1, 100.0, LogCount(21)),
                          :low => SpacedGrid(0.1, 10.0, LogPerDecade(4)),
                          :far => SpacedGrid(1.0, 100.0, LogCount(9)))
            calls = Ref(0); asked = Ref(0)
            curvesfn(xs_of) = (calls[] += 1; asked[] += sum(length, values(xs_of));
                               Dict(k => [objs[k](x) for x in xs] for (k, xs) in xs_of))
            cv = quiet(() -> coarse_fine_cv_batched(curvesfn, coarse, scheme))
            for k in keys(objs)
                ref = quiet(() -> coarse_fine_cv(objs[k], coarse[k], scheme; method_name = k))
                for f in (:coarse_spec, :x_star, :boundary_hit, :n_extensions, :curve_ys,
                          :fine_spec, :fine_curve, :fine_idx, :fine_bracket, :x_star_fine)
                    @test cv[k][f] == ref[f]
                end
            end
            # one call per round (the slowest cell's rounds) plus one fine call,
            # and every point evaluated exactly once
            @test calls[] == 1 + maximum(cv[k][:n_extensions] for k in keys(objs)) + 1
            @test asked[] == sum(length(cv[k][:curve_ys]) + length(cv[k][:fine_curve]) for k in keys(objs))
            # a curvesfn that drops a cell is refused, never silently skipped
            @test_throws ErrorException quiet(() -> coarse_fine_cv_batched(
                xs_of -> Dict{Symbol,Any}(), coarse, scheme))
        end

    end
end

@testitem "joint coarse-fine search" setup = [TuningFixtures] begin
    using Test, Logging
    using VisualGeometryToolkitEval.Evaluation: coarse_search, coarse_fine_search, coarse_fine_cv
    using VisualGeometryToolkitEval.Tuning: SpacedGrid, LogPerDecade, LinearCount, LogCount, grid, grid_spec
    quiet(f) = with_logger(f, NullLogger())

    axis(lo, hi, spacing, fine; expensive = false, max_rounds = 3) = Dict{String,Any}(
        "coarse" => Dict{String,Any}("type" => "spaced", "window" => Dict{String,Any}("type" => "fixed", "lo" => lo, "hi" => hi),
                                     "spacing" => spacing),
        "fine" => fine, "expensive" => expensive, "extension" => Dict{String,Any}("npts" => 2, "max_rounds" => max_rounds))
    joint(axes) = fixture_spec(Dict{String,Any}("type" => "joint_coarse_fine", "axes" => axes); name = "j").scheme
    cells(s) = Dict(:c => NamedTuple{Tuple(Symbol.(sort!(collect(keys(s.axes)))))}(
        Tuple(grid_spec(s.axes[n].coarse) for n in sort!(collect(keys(s.axes))))))
    # a coupled bowl in (log10 a, b): optimum at a = 10^-1.3 (below a's window), b = 0.42
    bowl(p) = [(log10(p.a) + 1.3)^2 + (p.b - 0.42)^2 + 0.5 * (log10(p.a) + 1.3) * (p.b - 0.42)]

    s = joint(Dict("a" => axis(0.1, 10.0, Dict("type" => "log_per_decade", "per_decade" => 4),
                               Dict("type" => "log_count", "n" => 9)),
                   "b" => axis(0.0, 1.0, Dict("type" => "linear_count", "n" => 6),
                               Dict("type" => "linear_count", "n" => 9))))

    @testset "extends the clipped axis only, then a free argmin of the fine product" begin
        r = quiet(() -> coarse_fine_search(ps -> Dict(k => bowl.(v) for (k, v) in ps), cells(s), s))[:c]
        @test r.names == (:a, :b)
        @test r.n_extensions[1, 1] >= 1 && r.n_extensions[2, :] == [0, 0]   # a extended low, b never
        @test r.boundary_hit == (a = false, b = false)
        @test size(r.coarse_objective) == (9 + 2r.n_extensions[1, 1], 6)
        @test size(r.fine_objective) == (9, 9)
        @test isapprox(log10(r.selected.a), -1.3; atol = 0.05) && isapprox(r.selected.b, 0.42; atol = 0.05)
        # the coarse product, every cell once, then the fine product
        @test length(r.evaluated) == length(r.coarse_objective) + length(r.fine_objective)
        @test allunique(first.(r.evaluated)[1:length(r.coarse_objective)])
    end

    @testset "an expensive axis is asked for one value at a time" begin
        se = joint(Dict("a" => axis(0.1, 10.0, Dict("type" => "log_per_decade", "per_decade" => 4),
                                    Dict("type" => "log_count", "n" => 3); expensive = true),
                        "b" => axis(0.0, 1.0, Dict("type" => "linear_count", "n" => 6),
                                    Dict("type" => "linear_count", "n" => 9))))
        seen = Float64[]
        errfn(ps) = (p = only(values(ps)); @test allequal(q.a for q in p); push!(seen, first(p).a);
                     Dict(k => bowl.(v) for (k, v) in ps))
        r = quiet(() -> coarse_fine_search(errfn, cells(se), se))[:c]
        # one call per expensive value per stage; the fine stage's 3 values
        # are the coarse argmin and its neighbours (a caller's run per value
        # is reused across them)
        @test length(seen) == size(r.coarse_objective, 1) + 3
        @test length(unique(round.(seen; sigdigits = 12))) == size(r.coarse_objective, 1)
    end

    @testset "a boundary hit is per axis" begin
        mono(p) = [log10(p.a) + (p.b - 0.42)^2]                             # no interior optimum in a
        r = quiet(() -> coarse_search(ps -> Dict(k => mono.(v) for (k, v) in ps), cells(s), s))[:c]
        @test r.boundary_hit == (a = true, b = false) && r.n_extensions[1, :] == [3, 0]
    end

    @testset "d = 1: the joint search of one axis is the scalar search" begin
        obj(x) = [(log10(x) + 1.6)^2]
        sc = fixture_spec(Dict{String,Any}("type" => "coarse_fine", "coarse" => Dict{String,Any}("type" => "spaced",
            "window" => Dict{String,Any}("type" => "fixed", "lo" => 0.1, "hi" => 10.0),
            "spacing" => Dict{String,Any}("type" => "log_per_decade", "per_decade" => 4)),
            "fine" => Dict{String,Any}("type" => "log_count", "n" => 9)); name = "s").scheme
        sj = joint(Dict("x" => Dict{String,Any}("coarse" => Dict{String,Any}("type" => "spaced",
            "window" => Dict{String,Any}("type" => "fixed", "lo" => 0.1, "hi" => 10.0),
            "spacing" => Dict{String,Any}("type" => "log_per_decade", "per_decade" => 4)),
            "fine" => Dict{String,Any}("type" => "log_count", "n" => 9),
            "extension" => Dict{String,Any}("npts" => sc.extension.npts, "max_rounds" => sc.extension.max_rounds))))
        rs = quiet(() -> coarse_fine_cv(obj, grid_spec(sc.coarse), sc; method_name = :x))
        rj = quiet(() -> coarse_fine_search(ps -> Dict(k => [obj(p.x) for p in v] for (k, v) in ps),
                                            Dict(:c => (x = grid_spec(sj.axes["x"].coarse),)), sj))[:c]
        @test rs[:x_star_fine] == rj.selected.x && rs[:curve_ys] == vec(rj.coarse_objective)
        @test rs[:fine_curve] == vec(rj.fine_objective) && rs[:n_extensions] == sum(rj.n_extensions)
    end

    @testset "the cells must name the spec's axes" begin
        @test_throws ArgumentError coarse_search(ps -> Dict(), Dict(:c => (a = grid_spec(s.axes["a"].coarse),)), s)
    end
end

@testitem "incumbent rule" setup = [TuningFixtures] begin
    using Test, Logging, Random
    using VisualGeometryToolkitEval.Evaluation: coarse_fine_search
    using VisualGeometryToolkitEval.Tuning: grid_spec, tuning_hash, tuning_table
    quiet(f) = with_logger(f, NullLogger())

    ax(lo, hi, sp, fine; expensive = false) = Dict{String,Any}("coarse" => Dict{String,Any}("type" => "spaced",
        "window" => Dict{String,Any}("type" => "fixed", "lo" => lo, "hi" => hi), "spacing" => sp), "fine" => fine,
        "expensive" => expensive)
    table(inc) = merge(Dict{String,Any}("type" => "joint_coarse_fine",
        "criterion" => Dict{String,Any}("type" => "mean_average_accuracy", "max_error" => 10.0),
        "axes" => Dict{String,Any}(
            "a" => ax(0.5, 16.0, Dict{String,Any}("type" => "log_per_decade", "per_decade" => 4), Dict{String,Any}("type" => "log_count", "n" => 5); expensive = true),
            "b" => ax(0.0, 1.0, Dict{String,Any}("type" => "linear_count", "n" => 5), Dict{String,Any}("type" => "linear_count", "n" => 5)))),
        inc === nothing ? Dict{String,Any}() : Dict{String,Any}("incumbent" => Dict{String,Any}("values" => inc)))
    spec(inc) = fixture_spec(table(inc); name = "i").scheme
    # 40 units × 6 items: the error grows away from (a, b) = (2, 0.5), with
    # per-unit noise drawn per cell, so a small objective gap is not resolved
    base = abs.(randn(MersenneTwister(5), 40, 6))
    errs(p) = base .* (1 + 3abs(log(p.a / 2)) + 4abs(p.b - 0.5)) .*
              (1 .+ rand(MersenneTwister(hash((p.a, p.b))), 40))
    cells(s) = Dict(:c => (a = grid_spec(s.axes["a"].coarse), b = grid_spec(s.axes["b"].coarse)))
    run(s; seen = []) = quiet(() -> coarse_fine_search(ps -> (push!(seen, only(values(ps))); Dict(k => errs.(v) for (k, v) in ps)),
                                                     cells(s), s))[:c]

    @testset "a clear gain adopts the selection" begin
        r = run(spec(Dict("a" => 12.0, "b" => 0.95)))
        @test !r.kept_incumbent && r.gain.p_gain >= 0.95 && r.gain.objective > 0
        @test r.selected != r.incumbent && isapprox(r.selected.a, 2.0; rtol = 0.3)
    end

    @testset "a gain within the noise keeps the incumbent" begin
        free = run(spec(nothing))
        inc = Dict("a" => free.selected.a * 1.01, "b" => free.selected.b)
        r = run(spec(inc))
        @test r.kept_incumbent && r.gain.p_gain < 0.95
        @test r.selected == (a = inc["a"], b = inc["b"])
    end

    @testset "the incumbent is one more call, in its own expensive group; no incumbent, no change" begin
        seen = []
        r = run(spec(Dict("a" => 3.0, "b" => 0.25)); seen)
        @test only(last(seen)) == (a = 3.0, b = 0.25)
        @test all(c -> allequal(q.a for q in c), seen)          # one expensive value per call
        @test !haskey(run(spec(nothing)), :incumbent)
        # a spec without the table writes none: its hash is the hash of a table that never had it
        @test !haskey(tuning_table(spec(nothing)), "incumbent")
        @test tuning_hash(spec(nothing)) == tuning_hash(fixture_spec(table(nothing); name = "other").scheme)
    end
end
