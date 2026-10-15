# =============================================================================
# coarse_fine_artifact_tests.jl — the tuning path end to end: the
# rigid-target invariance criterion (Evaluation rigid_invariance.jl), the
# coarse-fine tuning artifact and its gated reader (coarse_fine_artifact.jl),
# the bootstrap of the argmin over unit statistics (Tuning
# bootstrap_argmin.jl) and the curve/surface tables (Render tuning_table.jl).
# =============================================================================

@testitem "rigid_invariance_errors" begin
    using Test, Random, LinearAlgebra, StaticArrays
    using Statistics: median
    using Rotations: RotMatrix, RotXYZ, rotation_angle
    using VisualGeometryEval.Evaluation: rigid_invariance_errors
    using VisualGeometryEval.Stats: eligible_maa

    rng = MersenneTwister(3)
    # Five targets rigidly mounted on one body (mounting poses Y, mm), seen by
    # a camera in 12 units at random body poses A; pose of target m in unit u:
    # A_u ∘ Y_m.
    targets = ["a", "b", "c", "d", "e"]
    Y = Dict(m => (RotMatrix{3, Float64}(RotXYZ(0.3randn(rng), 0.3randn(rng), 0.3randn(rng))),
                   SVector(300randn(rng), 300randn(rng), 20randn(rng))) for m in targets)
    pose(A, m; σ = 0.0) = let (Ra, ta) = A, (Ry, ty) = Y[m]
        (R = SMatrix{3, 3}(Ra * Ry), t = SVector{3}(Ra * ty + ta) + σ * randn(rng, SVector{3}))
    end
    body() = (RotMatrix{3, Float64}(RotXYZ(0.5randn(rng), 0.5randn(rng), 2π * rand(rng))), SVector(0.0, 0.0, 2000.0) + 100randn(rng, SVector{3}))
    units = ["u$(lpad(i, 2, '0'))" for i in 1:12]

    @testset "exact poses: zero error on every observed pair" begin
        poses = Dict(u => (A = body(); Dict(m => pose(A, m) for m in targets)) for u in units)
        r = rigid_invariance_errors(poses; targets)
        @test size(r.translation) == (12, 10) == size(r.rotation)
        @test r.pairs[1] == ("a", "b") && r.units == units
        @test maximum(r.translation) < 1.0e-9 && maximum(r.rotation) < 1.0e-9
        @test eligible_maa(vec(r.translation); T = 10.0) ≈ 1.0
    end

    @testset "missed pairs are Inf: failures under eligible_maa" begin
        poses = Dict(u => (A = body(); Dict(m => pose(A, m; σ = 1.0) for m in targets)) for u in units)
        delete!(poses["u01"], "e")                       # e unseen in one unit
        poses["u02"] = Dict("a" => poses["u02"]["a"])    # a lone target: no pair
        r = rigid_invariance_errors(poses; targets)
        j = findfirst(==(("a", "e")), r.pairs)
        @test isinf(r.translation[1, j])
        @test all(isinf, r.translation[2, :])
        nfin = count(isfinite, r.translation)
        @test nfin == 12 * 10 - 4 - 10
        # Missed pairs count as failures: the all-items mAA is the observed
        # mAA scaled by the observed fraction.
        e = vec(r.translation)
        @test eligible_maa(e; T = 10.0) ≈ eligible_maa(e, isfinite.(e); T = 10.0) * nfin / length(e)
        # The error is translation noise of the relative pose (σ = 1 mm per pose).
        @test 0.5 < median(filter(isfinite, e)) < 3.0
    end

    @testset "fewer than min_obs units: no reference" begin
        poses = Dict(u => (A = body(); Dict(m => pose(A, m) for m in ("a", "b"))) for u in units[1:2])
        @test all(isinf, rigid_invariance_errors(poses; targets = ["a", "b"]).translation)
        @test_throws ArgumentError rigid_invariance_errors(Dict("u" => Dict("z" => pose(body(), "a"))); targets = ["a"])
    end
end

@testitem "coarse_fine_artifact" setup = [TuningFixtures] begin
    using Test, Logging, Random
    using VisualGeometryEval.Evaluation: coarse_fine_search, save_coarse_fine_artifact, deployed_setting
    using VisualGeometryEval.Tuning: bootstrap_argmin, unit_statistics, MeanAverageAccuracy, QueryUnit,
        load_tuning_artifact, grid_spec, save_tuning_artifact, SpacedGrid, LogCount
    using VisualGeometryEval.Stats: eligible_maa
    using VisualGeometryEval.Render: tuning_slice, emit_tuning_table

    ax(lo, hi, sp, fine) = Dict{String,Any}("coarse" => Dict{String,Any}("type" => "spaced",
        "window" => Dict{String,Any}("type" => "fixed", "lo" => lo, "hi" => hi), "spacing" => sp), "fine" => fine)
    spec = fixture_spec(Dict{String,Any}("type" => "joint_coarse_fine",
        "criterion" => Dict{String,Any}("type" => "mean_average_accuracy", "max_error" => 10.0),
        "axes" => Dict{String,Any}(
            "w" => ax(0.5, 16.0, Dict{String,Any}("type" => "log_per_decade", "per_decade" => 4), Dict{String,Any}("type" => "log_count", "n" => 7)),
            "s" => ax(0.0, 1.0, Dict{String,Any}("type" => "linear_count", "n" => 5), Dict{String,Any}("type" => "linear_count", "n" => 5)))); name = "w")
    # 40 units × 6 items; the error grows away from (w, s) = (2, 0.5) and a far w misses items.
    rng = MersenneTwister(5)
    base = abs.(randn(rng, 40, 6))
    errs(p) = (E = base .* (1 + 3abs(log(p.w / 2)) + 4abs(p.s - 0.5)); E[:, 6] .= p.w > 8 ? Inf : E[:, 6]; E)
    cells = Dict(:c => (s = grid_spec(spec.scheme.axes["s"].coarse), w = grid_spec(spec.scheme.axes["w"].coarse)))
    rec = with_logger(NullLogger()) do
        coarse_fine_search(ps -> Dict(k => errs.(v) for (k, v) in ps), cells, spec.scheme)[:c]
    end
    dir = mktempdir()
    path = save_coarse_fine_artifact(joinpath(dir, "w.jld2"), spec, rec; unit = QueryUnit(), code = ())

    @testset "the gated reader" begin
        sel = deployed_setting(path)
        @test sel == rec.selected && keys(sel) == (:s, :w)
        @test isapprox(sel.w, 2.0; rtol = 0.2) && isapprox(sel.s, 0.5; atol = 0.1)
        art = load_tuning_artifact(path)
        @test size(art["cells"]) == (length(rec.evaluated), 2)
        @test length(art["unit_statistics"]) == length(rec.evaluated)
        @test size(first(art["unit_statistics"])) == (40, 2)     # two numbers per unit, never the items
        clipped = merge(rec, (; boundary_hit = (s = false, w = true)))
        bad = save_coarse_fine_artifact(joinpath(dir, "bad.jld2"), spec, clipped; unit = QueryUnit(), code = ())
        @test_throws ErrorException deployed_setting(bad)
        @test_logs (:warn, r"FORENSIC") match_mode = :any deployed_setting(bad; allow_edge = true)
        # the one-scalar layout written before the joint search
        sc = fixture_spec(Dict{String,Any}("type" => "coarse_fine"); name = "old")
        old = save_tuning_artifact(joinpath(dir, "old.jld2"), sc; grids = Dict(:coarse => SpacedGrid(1.0, 2.0, LogCount(3))),
            unit = QueryUnit(), code = (), record = Dict{String,Any}("boundary_hit" => false, "param_name" => :h), x_star_fine = 1.5)
        @test deployed_setting(old) == (h = 1.5,)
    end

    @testset "unit statistics: the mAA and its bootstrap, exactly" begin
        E = errs((w = 3.0, s = 0.25))
        S = unit_statistics(MeanAverageAccuracy(10.0), E)
        @test -sum(S[:, 1]) / sum(S[:, 2]) ≈ -eligible_maa(vec(E); T = 10.0)
        art = load_tuning_artifact(path)
        Es = [errs(p) for p in first.(rec.evaluated)]
        idx = bootstrap_argmin(MeanAverageAccuracy(10.0), art["unit_statistics"], Xoshiro(9); n_resamples = 200)
        rr = Xoshiro(9)
        ref = map(1:200) do _
            rows = rand(rr, 1:40, 40)
            argmax([eligible_maa(vec(E[rows, :]); T = 10.0) for E in Es])
        end
        @test idx == ref
    end

    @testset "curve and surface tables" begin
        c = tuning_slice(path; axes = ["w"])
        @test c.axes == ["w"] && length(c.values) == length(c.xs[1])
        @test maximum(c.values) == c.values[only(c.argmin)]               # the mAA, maximal at the argmin
        sfc = tuning_slice(path; axes = ["s", "w"], stage = :fine)
        @test size(sfc.values) == (5, 7)
        out = emit_tuning_table(path, joinpath(dir, "surface.tex"); axes = ["s", "w"], labels = ["\$s\$", "\$w\$"])
        @test occursin("\\star", read(out, String)) && occursin("mAA", read(out, String))
        out1 = emit_tuning_table(path, joinpath(dir, "curve.tex"); axes = ["w"], labels = ["\$w\$"])
        @test count("\\\\", read(out1, String)) == length(c.xs[1]) + 1
    end
end
