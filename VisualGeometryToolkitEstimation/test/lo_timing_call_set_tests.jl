@testitem "lo_timing: the recorded call set" begin
    # THE call set `lo_calls_phototourism_e.bin` and its one reader
    # (test/lo_timing/call_set.jl), which this package's timing bench and
    # LocalOptimizationDoneRight's include from `pkgdir(VisualGeometryToolkitEstimation)`.
    using Test, StaticArrays, LinearAlgebra
    using VisualGeometryToolkitCore, VisualGeometryToolkitEstimation
    dir = joinpath(pkgdir(VisualGeometryToolkitEstimation), "test", "lo_timing")
    include(joinpath(dir, "call_set.jl"))
    pairs, halfwidth = read_fixture(CALL_SET_FIXTURE)
    @test CALL_SET_FIXTURE == joinpath(dir, "lo_calls_phototourism_e.bin")
    @test length(pairs) == 8 && all(pr -> length(pr.seeds) == 3, pairs)
    @test all(pr -> 346 <= pr.n <= 1094, pairs) && halfwidth > 0
    cs = pool_correspondences(first(pairs))
    @test length(cs) == first(pairs).n
    @test problem_of(first(pairs)) isa EssentialProblem
    bad = tempname(); write(bad, "NOTAFIXTURE")
    @test_throws ErrorException read_fixture(bad)
    @test occursin("call_set.jl", read(joinpath(dir, "bench.jl"), String))
    # the arms' runs over the call set: every (pair, seed) once, timed per call, measured as the record reads it
    seen = Tuple{Int, Int}[]
    run_call_set((i, s) -> push!(seen, (i, length(seen))), pairs)
    @test length(seen) == call_set_size(pairs) == 24 && first.(seen) == repeat(1:8; inner = 3)
    @test time_call_set((i, s) -> nothing, pairs) >= 0
    us, bytes = call_set_arm(() -> (i, s) -> nothing, pairs, 3)()
    @test length(us) == 3 && issorted(us) && bytes >= 0
    # THE LO arm (lo_call_set_arm, the binding of every LO timing bench): a strategy's locally_optimize over the call set,
    # each pair with its own scoring and workspace; one rep of a two-update Welsch IRLS
    probs = [problem_of(pr) for pr in pairs]
    scorings = [VisualGeometryToolkitCore.MsacScoring(EssentialProblem; threshold = halfwidth / 20) for _ in probs]
    workspaces = [VisualGeometryToolkitEstimation.RansacWorkspace(probs[i], scorings[i]) for i in eachindex(probs)]
    made = Ref(0)
    arm = lo_call_set_arm(() -> (made[] += 1; Irls{WelschShape}(; max_iter = 2, efficiency_constant = 1.5)), probs, scorings, workspaces, pairs, 1)
    us, bytes = arm()
    @test made[] == 1 && length(us) == 1 && us[1] > 0 && bytes >= 0   # the strategy built once per arm
end
