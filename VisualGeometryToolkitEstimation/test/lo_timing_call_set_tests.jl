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
end
