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
    cs = pool_csponds(first(pairs))
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
end
