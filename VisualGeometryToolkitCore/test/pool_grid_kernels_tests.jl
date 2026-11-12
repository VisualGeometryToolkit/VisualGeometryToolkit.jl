# The exact grid selection of a residual-quality family (robust/pool_grid_kernels.jl): select_on_grid! on the :pool
# staging scores RANSAC and MSAC by exact_pm_score_kernel! and selects by argmax_models_kernel! (the kernels
# RANSACScoringDoneRight's grid families run on a device), and selects what best_pool_model selects per pair.

@testitem "pool grid kernels: a residual-quality family selects through the exact kernels, as best_pool_model does" begin
    using Test
    using VisualGeometryToolkitCore
    using VisualGeometryToolkitCore.Pools
    using VisualGeometryToolkitCore: CspondNormalization, RansacScoring, MsacScoring, best_pool_model,
        select_on_grid!, selection_staging, stage_selection, has_residual_quality, support_sq_loss_width, sorted_squared_residuals,
        exact_pm_score_kernel!, argmax_models_kernel!, residual_quality, AbstractConstraintProblem
    import VisualGeometryToolkitCore as VgtkCore
    import KernelAbstractions as KA
    using StaticArrays, Random
    using LinearAlgebra: I

    struct _GKProblem <: AbstractConstraintProblem end
    VgtkCore.codimension(::Type{_GKProblem}) = 1
    VgtkCore.minimal_sample_size(::Type{_GKProblem}) = 5

    rng = Xoshiro(11)
    sz = Size2(width = 1000.0, height = 800.0)
    N = pool_normalization(sz, sz)
    K = SMatrix{3, 3, Float64}(900.0, 0, 0, 0, 900.0, 0, 500.0, 400.0, 1.0)
    function pair_of(n, M)
        cs = csponds([SVector(rand(rng), rand(rng)) for _ in 1:n], [SVector(rand(rng), rand(rng)) for _ in 1:n])
        models = FundMat{Float64}[FundMat(SMatrix{3, 3, Float64}(randn(rng, 9)...)) for _ in 1:M]
        idx = M == 0 ? zeros(Int, 5, 0) : reduce(hcat, [sort(randperm(rng, n)[1:5]) for _ in 1:M])
        pair = PoolPair(cs, K, K, SMatrix{3, 3, Float64}(1.0I), SVector(1.0, 0.0, 0.0), models, idx, rand(rng, M), rand(rng, M),
                        BitVector(rand(rng, Bool, n)), 0.25, N, nothing)
        # a mixture of inliers at a model-dependent scale and uniform outliers
        R = Float32[rand(rng) < 0.6 ? 0.3f0 * (1 + j / M) * randn(rng, Float32) : 20 * rand(rng, Float32) for _ in 1:n, j in 1:M]
        return pair, R
    end
    pool = [pair_of(80, 9) for _ in 1:12]
    push!(pool, pair_of(80, 0))                          # a pair without a model selects 0
    pairs, Rs = first.(pool), last.(pool)
    grid = [0.2, 0.5, 1.0, 2.5, 6.0]

    # the traits: Core's RANSAC and MSAC have a per-residual quality, of compact support
    for sc in (RansacScoring(1; threshold = 1.0), MsacScoring(1; threshold = 1.0))
        @test has_residual_quality(sc) && support_sq_loss_width(sc, 3.0) == 9.0
    end
    @test !has_residual_quality(nothing) && support_sq_loss_width(nothing, 3.0) == Inf   # the defaults: no quality, unbounded support

    # the staging the kernels read: squared, the minimal sample at zero, sorted
    p1, R1 = pool[1]
    S = sorted_squared_residuals(p1, R1)
    @test size(S) == (80, 9) && eltype(S) == Float32 && all(issorted, eachcol(S))
    @test S[:, 2] == sort([i in p1.sample_indices[:, 2] ? 0.0f0 : R1[i, 2]^2 for i in 1:80])

    # the score kernel on one pair: per model and width, the sum of the quality
    sc = MsacScoring(1; threshold = 1.0)
    scores = fill(-Inf, 1, 9, length(grid))
    exact_pm_score_kernel!(KA.CPU())(scores, vec(S), Int64[0], Int32[80], Int32[9], grid, Int32(length(grid)), Int32(0), sc,
                                     support_sq_loss_width(sc, maximum(grid)); ndrange = (1, 9))
    @test scores[1, 4, 3] ≈ sum(residual_quality(sc, Float64(d2), grid[3]) for d2 in S[:, 4]) rtol = 1e-12
    errs, best = zeros(1, length(grid)), zeros(Int32, 1, length(grid))
    argmax_models_kernel!(KA.CPU())(errs, best, scores, Int32[9], Float64.(1:9), Int64[0], Int32(0); ndrange = (1, length(grid)))
    @test vec(best) == [argmax(scores[1, :, t]) for t in eachindex(grid)] && vec(errs) == Float64.(vec(best))

    # select_on_grid! on the :pool staging: what best_pool_model selects at every width
    for sc in (RansacScoring(1; threshold = 1.0), MsacScoring(1; threshold = 1.0))
        @test selection_staging(sc) === :pool
        idxs = select_on_grid!(Matrix{Int32}(undef, length(pairs), length(grid)), _GKProblem, sc, grid,
                               stage_selection(:pool, pairs, Rs))
        ref = [best_pool_model(VgtkCore.with_width(sc, VgtkCore.LossWidth(w)), p, R) for (p, R) in zip(pairs, Rs), w in grid]
        @test idxs == ref && all(idxs[end, :] .== 0) && all(idxs[1:(end - 1), :] .> 0)
        # a supplied backend is the one the kernels run on (the host's: the same selections)
        st = stage_selection(:pool, pairs, Rs; backend = KA.CPU())
        @test st.backend === KA.CPU()
        @test select_on_grid!(Matrix{Int32}(undef, length(pairs), length(grid)), _GKProblem, sc, grid, st) == ref
    end

    # GPU-first: the :pool staging's backend is `default_device()` unless the caller supplies one
    old_resolver, old_device = VgtkCore._DEVICE_RESOLVER[], VgtkCore._DEFAULT_DEVICE[]
    calls = Ref(0)
    try
        VgtkCore._DEVICE_RESOLVER[] = () -> (calls[] += 1; KA.CPU())
        VgtkCore.reset_default_device!()
        @test stage_selection(:pool, pairs, Rs).backend === KA.CPU() && calls[] == 1
    finally
        VgtkCore._DEVICE_RESOLVER[] = old_resolver; VgtkCore._DEFAULT_DEVICE[] = old_device
    end
end

@testitem "CSRMatrix defaults to default_device()" begin
    using SparseArrays: sparse
    using LinearAlgebra: mul!
    import VisualGeometryToolkitCore as VgtkCore
    import KernelAbstractions as KA
    A = sparse([1, 2, 3, 3], [1, 2, 1, 3], [2.0, -1.0, 0.5, 4.0], 3, 3)
    old_resolver, old_device = VgtkCore._DEVICE_RESOLVER[], VgtkCore._DEFAULT_DEVICE[]
    calls = Ref(0)
    try
        VgtkCore._DEVICE_RESOLVER[] = () -> (calls[] += 1; KA.CPU())
        VgtkCore.reset_default_device!()
        C = VgtkCore.CSRMatrix(A)
        @test calls[] == 1
        x = [1.0, 2.0, 3.0]
        @test mul!(zeros(3), C, x) == A * x
    finally
        VgtkCore._DEVICE_RESOLVER[] = old_resolver; VgtkCore._DEFAULT_DEVICE[] = old_device
    end
end
