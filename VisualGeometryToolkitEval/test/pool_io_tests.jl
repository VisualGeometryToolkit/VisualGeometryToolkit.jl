# =============================================================================
# pool_io_tests.jl — the scoring-pool IO (src/pools): the light file and the
# residuals file round-trip bitwise; `pool_key` binds them and a mismatched key,
# pair count or matrix shape is refused naming the file (and both keys); other
# formats and frames are refused; `write_scene_pool` orchestrates a dummy
# `build`; the PnP pool file round-trips.
# =============================================================================

@testitem "pool_io" begin
    import VisualGeometryToolkitCore as VgtkCore
    using Test, Random, JLD2, StaticArrays
    using VisualGeometryToolkitEval.Pools
    using VisualGeometryToolkitEval.Pools: frame, synthetic_e_pool_pair, GT_CONVENTION
    using VisualGeometryToolkitCore: FundMat, HomographyMat, NormalizedImageFrame, csponds, sources, targets
    using VisualGeometryToolkitCore.Pools: PoolPair, StereoRawPair, num_points, num_models
    using VisualGeometryToolkitEval.Tuning: sampling_unit, PairUnit

    import VisualGeometryToolkitCore
    errmsg(f) = try f(); "" catch e; sprint(showerror, e) end
    # a raw pair for the dummy build: only its index and (for pair 2) a scene label
    struct DummyRaw <: StereoRawPair
        k::Int
    end
    VgtkCore.Pools.pair_scene(r::DummyRaw) = r.k == 2 ? "sub" : nothing
    stamps(; kw...) = merge((gt_convention = GT_CONVENTION, pool_rng = "LehmerRNG", julia_version = string(VERSION),
                             total_samples = 10, total_solutions = 12, mean_solutions_per_sample = 1.2,
                             provenance = Dict{String,Any}("driver" => "test", "commit" => "abc")), kw)
    # two E pairs (one with an SNN vector, one without), each with its residual matrix
    p1, R1 = synthetic_e_pool_pair(MersenneTwister(1))
    p2, R2 = synthetic_e_pool_pair(MersenneTwister(2); n = 12, sibling_counts = [3, 1])
    snn = rand(MersenneTwister(3), Float32, num_points(p1))
    # a record's fields by name: the keyword constructor's arguments
    fields(p) = NamedTuple{fieldnames(typeof(p))}(Tuple(getfield(p, f) for f in fieldnames(typeof(p))))
    p1 = PoolPair{FundMat{Float64}}(; fields(p1)..., snn)
    pool = ScoringPool([p1, p2], :E, "scene"; scene_names = ["scene", "other"], stamps = stamps())
    res = PoolResiduals(pool.pool_key, :tangent_sampson, [R1, R2])
    dir = mktempdir()
    path = joinpath(dir, "scene.jld2")
    save_pool(path, pool); save_pool_residuals(residuals_path(path), res)

    same(a::AbstractArray, b::AbstractArray) = size(a) == size(b) && all(a .=== b)
    same(a, b) = a === b || isequal(a, b)

    @testset "save → load is bitwise" begin
        back = load_pool(path)
        @test back isa ScoringPool{FundMat{Float64}} && frame(back) === NormalizedImageFrame
        @test back.problem === :E && back.scene == "scene" && back.scene_names == ["scene", "other"]
        @test back.pool_key == pool.pool_key && back.stamps == pool.stamps
        for (a, b) in zip(pool.pairs, back.pairs)
            @test same(sources(a.csponds), sources(b.csponds)) && same(targets(a.csponds), targets(b.csponds))
            for f in (:K1, :K2, :R_gt, :t_gt, :sample_indices, :errors_r, :errors_t, :gt_inliers,
                      :gt_variance, :source_map, :target_map)
                @test same(getfield(a, f), getfield(b, f))
            end
            @test all(all(SMatrix{3,3,Float64,9}(m) .=== SMatrix{3,3,Float64,9}(n)) for (m, n) in zip(a.models, b.models))
            @test eltype(b.models) === FundMat{Float64}
        end
        @test back.pairs[1].snn == snn && back.pairs[2].snn === nothing
        r = load_pool_residuals(residuals_path(path), back)
        @test r.kind === :tangent_sampson && r.pool_key == pool.pool_key
        @test same(residual_matrix(r, 1), R1) && same(residual_matrix(r, 2), R2)
        @test sampling_unit(PoolPair{FundMat{Float64}}) === PairUnit()
        # the file holds no stack type: every committed datatype is plain
        JLD2.jldopen(path, "r") do f
            @test f["pool_format"] == 4 && f["problem"] == "E" && f["model_class"] == "FundMat"
            @test f["frame"] == "NormalizedImageFrame" && f["pairs"][1].csponds isa Matrix{Float64}
        end
        @test !occursin("VisualGeometryToolkitCore", read(path, String))
    end

    @testset "pool_key: content, order and the residual indices" begin
        @test pool_key([p1, p2]) == pool.pool_key && length(pool.pool_key) == 64
        @test pool_key([p2, p1]) != pool.pool_key                       # pair order
        q = PoolPair{FundMat{Float64}}(; fields(p2)..., sample_indices = p2.sample_indices[:, [4, 2, 3, 1]])
        @test pool_key([p1, q]) != pool.pool_key                        # sample indices
        # σ̂², errors, the GT and the maps index no residual: they do not move the key
        g = PoolPair{FundMat{Float64}}(; fields(p2)..., errors_r = p2.errors_r .+ 1, gt_variance = 2.0)
        @test pool_key([p1, g]) == pool.pool_key
    end

    @testset "residuals refused: another key, count or shape" begin
        other = ScoringPool([p2], :E, "x"; scene_names = ["x"], stamps = stamps())
        m = errmsg(() -> load_pool_residuals(residuals_path(path), other))
        @test occursin(residuals_path(path), m) && occursin(pool.pool_key, m) && occursin(other.pool_key, m)
        bad = joinpath(dir, "bad.residuals.jld2")
        save_pool_residuals(bad, PoolResiduals(pool.pool_key, :sampson, [R1]))
        @test occursin("1 residual matrices for 2 pairs", errmsg(() -> load_pool_residuals(bad, pool)))
        save_pool_residuals(bad, PoolResiduals(pool.pool_key, :sampson, [R1, R2[:, 1:end-1]]))
        m = errmsg(() -> load_pool_residuals(bad, pool))
        @test occursin("residual matrix 2", m) && occursin("num_points × num_models", m)
        @test_throws ArgumentError PoolResiduals(pool.pool_key, :plane, [R1])
    end

    @testset "residuals read for a subset of pairs" begin
        back = load_pool(path)
        r = load_pool_residuals(residuals_path(path), back; pairs = [2])
        @test r.pairs == [2] && length(r.matrices) == 1
        @test same(residual_matrix(r, 2), R2)                           # still a POOL index
        @test occursin("pair 1 was not read", errmsg(() -> residual_matrix(r, 1)))
        @test occursin("subset", errmsg(() -> save_pool_residuals(joinpath(dir, "sub.residuals.jld2"), r)))
        @test occursin("outside 1:2", errmsg(() -> load_pool_residuals(residuals_path(path), back; pairs = [3])))
        # the key check still applies to a subset read
        other = ScoringPool([p2], :E, "x"; scene_names = ["x"], stamps = stamps())
        @test occursin(other.pool_key, errmsg(() -> load_pool_residuals(residuals_path(path), other; pairs = [1])))
    end

    @testset "light file refusals" begin
        v3 = joinpath(dir, "v3.jld2"); JLD2.jldsave(v3; pool_format = 3, pairs = [1])
        m = errmsg(() -> load_pool(v3))
        @test occursin("pool format v3", m) && occursin("convert_pools_v4", m)
        @test occursin("format v1", errmsg(() -> load_pool(residuals_path(path))))   # not a light file
        # a tampered pair no longer reproduces the stored key
        t = joinpath(dir, "tampered.jld2"); cp(path, t)
        JLD2.jldopen(t, "r+") do f
            ps = f["pairs"]; delete!(f, "pairs")
            ps[1] = merge(ps[1], (sample_indices = ps[1].sample_indices[:, end:-1:1],))
            f["pairs"] = ps
        end
        @test occursin("is not the key of its pairs", errmsg(() -> load_pool(t)))
        # a frame or convention other than the pool's
        for (k, v, msg) in (("frame", "ImageFrame", "NormalizedImageFrame"), ("gt_convention", "inverse", "gt_convention"))
            u = joinpath(dir, "u_$k.jld2"); cp(path, u; force = true)
            JLD2.jldopen(u, "r+") do f; delete!(f, k); f[k] = v; end
            @test occursin(msg, errmsg(() -> load_pool(u)))
        end
        # a model class that is not the problem's
        @test_throws ArgumentError ScoringPool([p1], :H, "s"; scene_names = ["s"], stamps = stamps())
        @test_throws ArgumentError ScoringPool([p1], :E, "s"; scene_names = ["s"], stamps = stamps(gt_convention = "x"))
        @test_throws DimensionMismatch ScoringPool([p1], :E, "s"; scene_names = String[], stamps = stamps())
    end

    @testset "scene_loaders" begin
        ls = scene_loaders(dir, ["scene"])
        @test ls[1]().pool_key == pool.pool_key
        lp, lr = only(scene_loaders(dir, [:scene]; residuals = true))()
        @test lp.pool_key == lr.pool_key && length(lr.matrices) == 2
    end

    @testset "write_scene_pool: a dummy build" begin
        seen = Dict{Int,UInt}()
        lk = ReentrantLock()
        function build(rng, raw)
            lock(lk) do; seen[raw.k] = rand(copy(rng), UInt); end
            raw.k == 3 && return nothing                                     # refused
            p, R = synthetic_e_pool_pair(rng; n = 8 + raw.k, sibling_counts = [1, 2])
            return (; pair = p, residuals = R, samples = 2, solutions = 3)
        end
        raws = [DummyRaw(k) for k in 1:4]
        out = joinpath(dir, "w", "built.jld2")
        @test write_scene_pool(out, raws, build; problem = :E, residual_kind = :tangent_sampson, seed = 42,
                               provenance = Dict("driver" => "dummy")) == out
        wp = load_pool(out)
        @test length(wp) == 3 && wp.scene == "built" && wp.scene_names == ["built", "sub", "built"]
        @test [num_points(p) for p in wp.pairs] == [9, 10, 12]
        @test wp.stamps.total_samples == 6 && wp.stamps.total_solutions == 9 && wp.stamps.mean_solutions_per_sample == 1.5
        @test wp.stamps.pool_rng == "LehmerRNG" && wp.stamps.provenance["driver"] == "dummy"
        @test load_pool_residuals(residuals_path(out), wp).kind === :tangent_sampson
        # each pair's stream is keyed by its index: a rerun reproduces it bitwise
        first_seen = copy(seen)
        again = joinpath(dir, "w", "again.jld2")
        write_scene_pool(again, raws, build; problem = :E, residual_kind = :tangent_sampson, seed = 42)
        @test seen == first_seen && load_pool(again).pool_key == wp.pool_key
        # skip_existing: an existing pair of files is not rebuilt
        empty!(seen)
        @test write_scene_pool(out, raws, build; problem = :E, residual_kind = :tangent_sampson, seed = 42) == out
        @test isempty(seen)
        # every pair refused: nothing written
        none = joinpath(dir, "w", "none.jld2")
        @test write_scene_pool(none, raws, (rng, raw) -> nothing; problem = :E, residual_kind = :sampson,
                               seed = 1) === nothing
        @test !isfile(none)
    end

    @testset "PnP pool file" begin
        p = joinpath(dir, "pnp", "q.jld2")
        pools = [(name = "q1", u = rand(3, 2), X = rand(3, 3), gt_variance = 1.5)]
        save_pnp_pool(p, pools, ["q2"]; score = "snn")
        back = load_pnp_pool(p)
        @test back.pools == pools && back.misses == ["q2"] && back.score == "snn"
        old = joinpath(dir, "pnp", "old.jld2"); JLD2.jldsave(old; pools, misses = String[], score = "none")
        @test occursin("pnp_pool format v1", errmsg(() -> load_pnp_pool(old)))
    end
end
