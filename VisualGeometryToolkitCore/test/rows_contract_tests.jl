# =============================================================================
# The chart-rows contract's buffers (robust/rows.jl): RowsWorkspace's layout,
# the per-(task, key) reuse policy and the per-point norms. The providers
# themselves are checked by their owners (VisualGeometryToolkitEstimation's problems).
# =============================================================================
@testitem "rows_contract" begin
    using Test, Random
    using LinearAlgebra: Transpose
    using VisualGeometryToolkitCore: RowsWorkspace, rows_workspace, point_norms!, task_buffers

    @testset "RowsWorkspace{D, C}(n) shapes" begin
        for (D, C, n) in ((5, 1, 12), (8, 2, 7), (6, 2, 1))
            ws = RowsWorkspace{D, C}(n)
            @test ws isa RowsWorkspace{D, C}
            @test length(ws.r) == C * n
            @test size(ws.J) == (C * n, D)
            @test ws.J isa Transpose{Float64, Matrix{Float64}}      # row-contiguous: J is a transposed D × C·n
            @test size(parent(ws.J)) == (D, C * n)
            @test length(ws.rn) == n && length(ws.valid) == n
            @test length(ws.onev) == C * n && all(==(1.0), ws.onev)
        end
    end

    @testset "rows_workspace: one per (task, D, C, n)" begin
        a = rows_workspace(Val(5), Val(2), 10)
        @test a === rows_workspace(Val(5), Val(2), 10)
        @test a !== rows_workspace(Val(5), Val(1), 10)
        @test a !== rows_workspace(Val(5), Val(2), 11)
        @test a !== rows_workspace(Val(6), Val(2), 10)
        @test size(a.J) == (20, 5)
        # task-local: another task builds its own
        @test fetch(Threads.@spawn rows_workspace(Val(5), Val(2), 10)) !== a
    end

    @testset "task_buffers: alloc once per key" begin
        calls = Ref(0)
        alloc() = (calls[] += 1; zeros(3))
        key = (:vgc_rows_contract_test, 1)
        b = task_buffers(alloc, key)
        @test calls[] == 1
        @test task_buffers(alloc, key) === b
        @test calls[] == 1                                     # reused, not rebuilt
        c = task_buffers(alloc, (:vgc_rows_contract_test, 2))
        @test calls[] == 2 && c !== b
        @test task_buffers(alloc, key) === b
    end

    @testset "point_norms!" begin
        rng = MersenneTwister(5)
        n = 9
        for d in (1, 2, 3)
            r = randn(rng, d * n)
            rn = fill(NaN, n)
            @test point_norms!(rn, r, n, d) === rn
            @test rn ≈ [sqrt(sum(abs2, r[(d * (i - 1) + 1):(d * i)])) for i in 1:n] rtol = 1.0e-15
            d == 1 && @test rn == abs.(r)
        end
        # codim 2 is exactly √(x² + y²) in that order of summation
        r = [3.0, 4.0, -1.0, 0.5]
        @test point_norms!(zeros(2), r, 2, 2) == [5.0, sqrt(1.0 + 0.25)]
    end
end
