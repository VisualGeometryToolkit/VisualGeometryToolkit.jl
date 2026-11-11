@testitem "lo_timing: the bench record's essential scene" begin
    # The seeded 1000-point E scene every LO timing bench times its kernels on
    # (this package's test/lo_timing/bench.jl and LocalOptimizationDoneRight's
    # each built it as `scene()`): one copy, in the bench record both include.
    using Test, Random, LinearAlgebra, StaticArrays
    using VisualGeometryToolkitCore
    include(joinpath(pkgdir(VisualGeometryToolkitCore), "test", "lo_timing", "bench_record.jl"))
    sc = essential_bench_scene(Xoshiro(20261007))
    @test length(sc.u₁) == length(sc.u₂) == 1000
    @test sc.K == SMatrix{3, 3, Float64, 9}(800, 0, 0, 0, 800, 0, 640, 480, 1)
    @test norm(sc.t) ≈ 1 && sc.R' * sc.R ≈ I
    # deterministic in its seed, and the RNG's draws in the recorded order
    @test essential_bench_scene(Xoshiro(20261007)).u₁ == sc.u₁
    @test essential_bench_scene(Xoshiro(1); n = 10).u₁ != sc.u₁[1:10]
    # about 70 % of the pairs satisfy the scene's epipolar constraint (30 % outliers)
    F = FundMat(EssentialMat(sc.R, sc.t), sc.K, sc.K)
    d = [sampson_distance(F, a, b) for (a, b) in zip(sc.u₁, sc.u₂)]
    @test 0.6 < count(<(3.0), d) / 1000 < 0.8
    # the bench reads the scene from the record, not a copy of its own
    @test occursin("essential_bench_scene(", read(joinpath(pkgdir(VisualGeometryToolkitCore), "test", "lo_timing", "bench.jl"), String))
end
