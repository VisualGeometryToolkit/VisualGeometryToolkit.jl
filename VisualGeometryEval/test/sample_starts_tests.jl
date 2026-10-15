# =============================================================================
# sample_starts_tests.jl — THE minimal-sample boundaries (src/pools/
# sample_starts.jl): a repeated draw (identical adjacent column runs) counts as
# its draws, not as one sample (the retired "new sample where the column
# changes" rule merges them); the structural result is certified against the
# `total_samples` stamp, and a duplicated root it cannot resolve is refused;
# the seeded replay resolves it; the sample-starts file round-trips and refuses
# another pool or inadmissible starts.
# =============================================================================

@testitem "sample_starts" begin
    using Test, Random
    using VisualGeometryEval
    using VisualGeometryEval.Pools
    using VisualGeometryEval.Pools: synthetic_e_pool_pair, GT_CONVENTION
    using VisualGeometryCore: FundMat
    using VisualGeometryCore.Pools: PoolPair, num_models

    errmsg(f) = try f(); "" catch e; sprint(showerror, e) end
    fields(p) = NamedTuple{fieldnames(typeof(p))}(Tuple(getfield(p, f) for f in fieldnames(typeof(p))))
    # the record whose model columns are `cols` of `p` (repeats allowed)
    cols_of(p, cols) = PoolPair{FundMat{Float64}}(; fields(p)..., models = p.models[cols],
                                                  sample_indices = p.sample_indices[:, cols],
                                                  errors_r = p.errors_r[cols], errors_t = p.errors_t[cols])
    stamps(total) = (gt_convention = GT_CONVENTION, pool_rng = "LehmerRNG", julia_version = string(VERSION),
                     total_samples = total, total_solutions = total, mean_solutions_per_sample = 1.0,
                     provenance = Dict{String,Any}())
    mkpool(pairs, total) = ScoringPool(pairs, :E, "scene"; scene_names = fill("scene", length(pairs)),
                                       stamps = stamps(total))
    # the retired rule: a new sample wherever the sample_indices column changes
    column_change(p) = [j for j in 1:num_models(p) if j == 1 || p.sample_indices[:, j] != p.sample_indices[:, j-1]]

    # samples: A (2 siblings, columns 1:2), B (1, column 3), C (1, column 4), D (2, columns 5:6)
    p, _ = synthetic_e_pool_pair(MersenneTwister(11); sibling_counts = [2, 1, 1, 2])
    @assert allunique(eachcol(p.sample_indices[:, [1, 3, 4, 5]]))

    @testset "a repeated draw is two samples" begin
        # A drawn twice in a row, then B, then B again, then D: 5 draws
        q = cols_of(p, [1, 2, 1, 2, 3, 3, 5, 6])
        pool = mkpool([q], 5)
        @test sample_starts(pool) == [[1, 3, 5, 6, 7]]
        @test column_change(q) == [1, 5, 7]                  # the old rule: 3 samples, not 5
        @test sample_ids(only(sample_starts(pool)), num_models(q)) == [1, 1, 2, 2, 3, 4, 5, 5]
        # a budget of 2 samples ends before B; the old rule's would run to column 6
        @test findlast(<=(2), sample_ids(only(sample_starts(pool)), num_models(q))) == 4
    end

    @testset "the stamp certifies; a duplicated root is refused" begin
        q = cols_of(p, [3, 3, 4])               # one B draw that returned B's model twice, then C
        @test occursin("total_samples stamp is 2", errmsg(() -> sample_starts(mkpool([q], 2))))
        @test sample_starts(mkpool([q], 3)) == [[1, 2, 3]]   # read as two B draws when the stamp says 3
        @test occursin("total_samples", errmsg(() -> sample_starts(mkpool([q], 4))))
    end

    @testset "the replay resolves what the record cannot" begin
        q = cols_of(p, [3, 3, 4, 1, 2, 1, 2])   # B (duplicated root), C, A, A
        pool = mkpool([q], 4)
        B, C, A = (p.sample_indices[:, j] for j in (3, 4, 1)); X = rand(MersenneTwister(1), 100:200, 5)
        # every trial: rejected draws (X) interleaved; the replayed stream draws B once
        stream_of(draws) = i -> (k = Ref(0); () -> (k[] += 1; draws[min(k[], end)]))
        st = derive_sample_starts(pool, stream_of([X, B, X, C, A, X, A, X]))
        @test st == [[1, 3, 4, 6]]
        # a stream that drew B twice does not fit the stamp
        @test occursin("total_samples", errmsg(() -> derive_sample_starts(pool, stream_of([B, B, C, A, A]))))
        # a stream that never draws the stored sample
        @test occursin("not replayed", errmsg(() -> derive_sample_starts(pool, stream_of([X]); max_trials = 50)))
        # a replayed count the run cannot split into
        r = cols_of(p, [1, 2, 1, 2, 3])
        @test occursin("does not split", errmsg(() -> derive_sample_starts(mkpool([r], 2),
                                                                           stream_of([A, A, A, B]))))
    end

    @testset "the sample-starts file" begin
        q = cols_of(p, [3, 3, 4, 1, 2, 1, 2])
        pool = mkpool([q], 4)
        dir = mktempdir(); path = joinpath(dir, "scene.samples.jld2")
        save_sample_starts(path, pool, [[1, 3, 4, 6]]; derivation = "test stream")
        @test load_sample_starts(path, pool) == [[1, 3, 4, 6]]
        @test sample_starts(pool; derived = path) == [[1, 3, 4, 6]]
        other = mkpool([cols_of(p, [1, 2, 3, 4])], 3)
        @test occursin("pool_key", errmsg(() -> load_sample_starts(path, other)))
        @test occursin("not admissible", errmsg(() -> save_sample_starts(path, pool, [[1, 3, 5, 6]]; derivation = "x")))
        @test occursin("not admissible", errmsg(() -> save_sample_starts(path, pool, [[1, 3, 4, 5]]; derivation = "x")))
        @test occursin("total_samples", errmsg(() -> save_sample_starts(path, pool, [[1, 2, 3, 4, 6]]; derivation = "x")))
        @test occursin("not the sample starts", errmsg(() -> sample_ids([2, 3], 4)))
        @test sample_ids(Int[], 0) == Int[]
    end
    @testset "the one owner: no ad hoc boundaries elsewhere in the package" begin
        adjacent = r"(sample_indices|sidx)[^\n]*\b[jk]\s*[-+]\s*1\b"
        root = pkgdir(VisualGeometryEval)
        offenders = String[]
        for (d, _, fs) in walkdir(joinpath(root, "src")), f in fs
            (endswith(f, ".jl") && f != "sample_starts.jl") || continue
            for (i, line) in enumerate(eachline(joinpath(d, f)))
                occursin(adjacent, first(split(line, '#'; limit = 2))) && push!(offenders, "$f:$i")
            end
        end
        @test isempty(offenders)
    end
end
