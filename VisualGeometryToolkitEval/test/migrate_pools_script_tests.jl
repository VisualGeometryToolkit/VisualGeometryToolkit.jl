# =============================================================================
# scripts/migrate_pools.jl — a tree of format-4 light pools migrated in place,
# dry run by default, on a temporary copy of a small pool
# =============================================================================

@testitem "migrate_pools_script" begin
    # scripts/migrate_pools.jl: a tree of format-4 pools, dry run then in place
    using Test, Random, JLD2
    using VisualGeometryToolkitEval.Pools
    using VisualGeometryToolkitEval.Pools: synthetic_e_pool_pair, GT_CONVENTION
    using VisualGeometryToolkitCore: FundMat
    using VisualGeometryToolkitCore.Pools: PoolPair

    script = Module(:MigratePoolsScript)
    Base.include(script, joinpath(pkgdir(VisualGeometryToolkitEval), "scripts", "migrate_pools.jl"))
    migrate_pools = getfield(script, :migrate_pools)

    stamps = (gt_convention = GT_CONVENTION, pool_rng = "LehmerRNG", julia_version = string(VERSION),
              total_samples = 10, total_solutions = 12, mean_solutions_per_sample = 1.2,
              provenance = Dict{String, Any}("driver" => "test", "commit" => "abc"))
    p1, R1 = synthetic_e_pool_pair(MersenneTwister(1))
    p2, R2 = synthetic_e_pool_pair(MersenneTwister(2); n = 12, sibling_counts = [3, 1])
    pool = ScoringPool([p1, p2], :E, "scene"; scene_names = ["scene", "other"], stamps)
    res = PoolResiduals(pool.pool_key, :tangent_sampson, [R1, R2])
    root = mktempdir()
    v5 = joinpath(root, "current.jld2")
    save_pool(v5, pool); save_pool_residuals(residuals_path(v5), res)
    # a format-4 file: format 5's contents with the destination map under its
    # former key, in the same slot (as pool_io's migrate_pool fixture)
    raw = JLD2.load(v5)
    former = (:csponds, :K1, :K2, :R_gt, :t_gt, :models, :sample_indices, :errors_r, :errors_t,
              :gt_inliers, :gt_variance, :source_map, :former_destination_key, :snn)
    v4(path) = (JLD2.jldsave(path; (Symbol(k) => v for (k, v) in raw if k ∉ ("pairs", "pool_format"))...,
                             pairs = [NamedTuple{former}(values(nt)) for nt in raw["pairs"]], pool_format = 4);
                cp(residuals_path(v5), residuals_path(path)); path)
    a = v4(joinpath(root, "a.jld2"))
    b = v4(joinpath(mkpath(joinpath(root, "sub", "deeper")), "b.jld2"))
    bad = joinpath(root, "bad.jld2")
    JLD2.jldsave(bad; (Symbol(k) => v for (k, v) in raw if k ∉ ("pairs", "pool_format"))...,
                 pairs = [Base.structdiff(NamedTuple{former}(values(nt)), NamedTuple{(:snn,)}) for nt in raw["pairs"]], pool_format = 4)
    corrupt = joinpath(root, "sub", "corrupt.jld2")
    write(corrupt, "not a JLD2 file")                           # unreadable: FAILED, never skipped
    before = Dict(p => read(p) for p in (a, b, bad, corrupt))
    snapshot() = sort!([joinpath(r, f) for (r, _, fs) in walkdir(root) for f in fs])
    files = snapshot()

    # the dry run: every format-4 pool verified, nothing written
    io = IOBuffer()
    n = migrate_pools(root; io)
    log = String(take!(io))
    @test n == (migrated = 0, verified = 2, current = 1, skipped = 3, failed = 2)    # skipped: the three residuals files
    @test occursin("verified  $a", log) && occursin("verified  $b", log) && occursin("current   $v5", log)
    @test occursin("FAILED    $bad", log) && occursin("not a format-4 pool pair", log)
    @test occursin("FAILED    $corrupt", log) && occursin("does not open", log)
    @test snapshot() == files && all(read(p) == before[p] for p in keys(before))

    # applied: migrated in place, verified first; the residuals untouched; nothing staged left behind
    n = migrate_pools(root; apply = true, io)
    @test n.migrated == 2 && n.failed == 2 && n.current == 1
    @test snapshot() == files                                   # no staging directory, no copy
    @test read(bad) == before[bad] && read(corrupt) == before[corrupt]   # a failed file is untouched
    for p in (a, b)
        back = load_pool(p)
        @test back.pool_key == pool.pool_key && back.stamps == pool.stamps
        @test all(isequal(x.destination_map, y.destination_map) && isequal(x.source_map, y.source_map) for (x, y) in zip(back.pairs, pool.pairs))
        @test load_pool_residuals(residuals_path(p), back).pool_key == pool.pool_key
        JLD2.jldopen(f -> @test(f["pool_format"] == 5), p, "r")
    end
    # a second run finds every pool current
    n = migrate_pools(root; apply = true, io)
    @test n.migrated == 0 && n.current == 3 && n.failed == 2
end
