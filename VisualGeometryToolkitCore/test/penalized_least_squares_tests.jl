@testitem "difference_rows: orders 1–3 along chosen axes, a masked lattice and a texel page; gradient_pairs" begin
    using SparseArrays: sparse, nnz
    using VisualGeometryToolkitCore: difference_rows, gradient_pairs, difference_stencil
    # a full texel page TW × TH at order 1: the horizontal differences first, (j outer, i inner), then the vertical
    TW, TH = 5, 4
    D = difference_rows((TW, TH), 1)
    tid(a, b) = (b - 1) * TW + a
    rows = [[(tid(i, j), -1.0), (tid(i + 1, j), 1.0)] for j in 1:TH for i in 1:(TW - 1)]
    append!(rows, [[(tid(i, j), -1.0), (tid(i, j + 1), 1.0)] for j in 1:(TH - 1) for i in 1:TW])
    R = sparse([k for (k, r) in enumerate(rows) for _ in r], [c for r in rows for (c, _) in r], [v for r in rows for (_, v) in r], length(rows), TW * TH)
    @test D == -R          # Core's stencil (1, −1) is the backward sign of the forward step: DᵀD and every pair's magnitude the same
    # one axis alone: the corresponding block of the two-axis rows
    @test difference_rows((TW, TH), 1; axes = (1,)) == D[1:((TW - 1) * TH), :]
    @test difference_rows((TW, TH), 1; axes = (2,)) == D[((TW - 1) * TH + 1):end, :]
    @test_throws ArgumentError difference_rows((TW, TH), 1; axes = (2, 1))
    @test_throws ArgumentError difference_rows((TW, TH), 1; axes = (3,))
    # orders 2 and 3: each row the stencil on order + 1 consecutive nodes; a masked node drops every row through it
    for order in (2, 3)
        Dk = difference_rows((7, 6), order)
        @test size(Dk, 1) == (7 - order) * 6 + 7 * (6 - order) && all(i -> nnz(Dk[i, :]) == order + 1, axes(Dk, 1))
        @test sort(Vector(Dk[1, :].nzval)) == sort(collect(difference_stencil(order)))
    end
    mask = trues(7, 6); mask[4, 3] = false
    Dm = difference_rows(mask, 3)
    @test iszero(Dm[:, LinearIndices(mask)[4, 3]]) && size(Dm, 1) == 4 * 6 - 4 + 7 * 3 - 3
    # the isotropic pairs: node (i, j)'s rows along each axis, 0 past the last node or across a masked one
    P = gradient_pairs(mask)
    D1 = difference_rows(mask, 1)
    L = LinearIndices(mask)
    for j in 1:6, i in 1:7
        r1, r2 = P[i, j]
        @test r1 == 0 || (D1[r1, L[i, j]] == 1 && D1[r1, L[i + 1, j]] == -1)
        @test r2 == 0 || (D1[r2, L[i, j]] == 1 && D1[r2, L[i, j + 1]] == -1)
        @test (r1 != 0) == (i < 7 && mask[i, j] && mask[i + 1, j]) && (r2 != 0) == (j < 6 && mask[i, j] && mask[i, j + 1])
    end
    @test sort(filter(!=(0), [x for p in P for x in p])) == collect(1:size(D1, 1))     # every row is one node's, once
    @test gradient_pairs((TW, TH)) == gradient_pairs(trues(TW, TH))
end

@testitem "mean_positive_eigenvalue, penalty_scale, row_stencils: the penalty's normalization and its rows as blocks" begin
    using LinearAlgebra: Symmetric, eigvals
    using SparseArrays: sparse
    using VisualGeometryToolkitCore: difference_rows, mean_positive_eigenvalue, penalty_scale, row_stencils
    # m̄ from the operator (‖D‖²_F / rank) is the mean positive eigenvalue of DᵀD, sparse or dense
    D = difference_rows(trues(10, 12), 3)
    λs = eigvals(Symmetric(Matrix(D' * D))); pos = λs[λs .> 1.0e-10 * maximum(λs)]
    @test mean_positive_eigenvalue(D) ≈ sum(pos) / length(pos) rtol = 1.0e-12
    @test mean_positive_eigenvalue(Matrix(D)) ≈ mean_positive_eigenvalue(D) rtol = 1.0e-12
    @test mean_positive_eigenvalue(Symmetric(Matrix(D' * D))) ≈ mean_positive_eigenvalue(D) rtol = 1.0e-12   # the normal-matrix form
    @test penalty_scale(12.0, 3, 2.0) == 2.0
    # the rows as blocks: each row's nodes (ascending) and coefficients, the operator rebuilt from them exactly
    st = row_stencils(D)
    @test length(st) == size(D, 1) && all(s -> issorted(s[1]) && length(s[1]) == 4, st)
    @test sparse([m for (m, s) in enumerate(st) for _ in s[1]], reduce(vcat, first.(st)), reduce(vcat, last.(st)), size(D)...) == D
end

@testitem "cgls: the penalized least squares on sparse operators, a target, a warm start, the stopping rule" begin
    using LinearAlgebra: norm, mul!
    using SparseArrays: sprandn, sparse
    using Random: Xoshiro, randn
    import KernelAbstractions as KA
    using VisualGeometryToolkitCore: difference_rows, PenalizedOperator, cgls, CSRMatrix
    rng = Xoshiro(11)
    TW, TH = 9, 7; n = TW * TH
    A = sprandn(rng, 150, n, 0.08) + sparse(1:n, 1:n, 0.3 .+ rand(rng, n), 150, n)
    D = difference_rows((TW, TH), 1)
    b = randn(rng, 150); h = randn(rng, size(D, 1)); w = 0.5 .+ rand(rng, size(D, 1))
    op = PenalizedOperator(A, D; device = KA.CPU())
    # the CSR form applies the operator as the sparse matrix does
    x = randn(rng, n)
    @test mul!(zeros(150), op.A, x) ≈ A * x && mul!(zeros(n), op.At, b) ≈ A' * b && mul!(zeros(size(D, 1)), CSRMatrix(D), x) == D * x
    # the solution of the normal equations (AᵀA + DᵀWD) x = Aᵀb + DᵀWh, with a target and per-row weights, and with neither
    Wd = sparse(1:length(w), 1:length(w), w)
    exact = Matrix(A' * A + D' * Wd * D) \ (A' * b + D' * (w .* h))
    s = cgls(op, b; w, h, rtol = 1.0e-12, max_iter = 2000)
    @test s.converged && norm(s.x - exact) <= 1.0e-8 * norm(exact)
    plain = Matrix(A' * A + 2.0 * D' * D) \ (A' * b)
    @test norm(cgls(op, b; w = 2.0, rtol = 1.0e-12, max_iter = 2000).x - plain) <= 1.0e-8 * norm(plain)
    # the stopping rule is relative to ‖Aᵀb + DᵀWh‖, not to the start's residual: warm from the solution, 0 iterations;
    # warm from a start near it, fewer iterations than cold, and the same tolerance met
    @test cgls(op, b; w, h, x0 = exact, rtol = 1.0e-6).iterations == 0
    cold = cgls(op, b; w, h, rtol = 1.0e-8, max_iter = 2000)
    warm = cgls(op, b; w, h, x0 = exact .+ 1.0e-4 .* randn(rng, n), rtol = 1.0e-8, max_iter = 2000)
    @test cold.converged && warm.converged && warm.iterations < cold.iterations && warm.relres <= 1.0e-8
    @test cgls(op, b; w, h, rtol = 1.0e-12, max_iter = 3).iterations == 3 && !cgls(op, b; w, h, rtol = 1.0e-12, max_iter = 3).converged
end

@testitem "cgls on CUDA: the CPU solve's result (availability-gated)" begin
    using SparseArrays: sprandn, sparse
    using Random: Xoshiro, randn
    using LinearAlgebra: norm
    import KernelAbstractions as KA
    using VisualGeometryToolkitCore: difference_rows, PenalizedOperator, cgls
    cuda = try
        m = Base.require(Main, :CUDA)
        (m !== nothing && Base.invokelatest(getproperty(m, :functional))) ? m : nothing
    catch
        nothing
    end
    if cuda === nothing
        @test_skip cuda !== nothing      # CUDA is in the test target: no functional GPU here (recorded as Broken)
    else
        rng = Xoshiro(12)
        TW, TH = 40, 30; n = TW * TH
        A = sprandn(rng, 3n, n, 0.002) + sparse(1:n, 1:n, 0.3 .+ rand(rng, n), 3n, n)
        D = difference_rows((TW, TH), 1)
        b = randn(rng, 3n); h = randn(rng, size(D, 1))
        cpu = cgls(PenalizedOperator(A, D; device = KA.CPU()), b; w = 0.7, h, rtol = 1.0e-10, max_iter = 3000)
        dev = Base.invokelatest(getproperty(cuda, :CUDABackend))
        gpu = Base.invokelatest(cgls, Base.invokelatest(PenalizedOperator, A, D; device = dev), b; w = 0.7, h, rtol = 1.0e-10, max_iter = 3000)
        @test gpu.converged && Base.invokelatest(KA.get_backend, gpu.x) == dev
        @test norm(Base.invokelatest(Array, gpu.x) - cpu.x) <= 1.0e-8 * norm(cpu.x)
        @test abs(gpu.iterations - cpu.iterations) <= 2
    end
end

@testitem "normal_cg: CG on an applied normal operator, the solve cgls makes on a stored one; Jacobi; a warm start" begin
    using LinearAlgebra: norm, mul!, Diagonal, diag, Symmetric, cond, qr
    using SparseArrays: sprandn, sparse, SparseMatrixCSC
    using Random: Xoshiro, randn
    import KernelAbstractions as KA
    using VisualGeometryToolkitCore: difference_rows, PenalizedOperator, cgls, normal_cg
    rng = Xoshiro(13)
    TW, TH = 9, 7; n = TW * TH
    A = sprandn(rng, 150, n, 0.08) + sparse(1:n, 1:n, 0.3 .+ rand(rng, n), 150, n)
    D = difference_rows((TW, TH), 1)
    b = randn(rng, 150); h = randn(rng, size(D, 1)); w = 0.5 .+ rand(rng, size(D, 1))
    # the normal operator AᵀA + DᵀWD applied, never stored, and its right-hand side Aᵀb + DᵀWh
    apply!(y, x) = (y .= A' * (A * x) .+ D' * (w .* (D * x)); y)
    g = A' * b .+ D' * (w .* h)
    ref = cgls(PenalizedOperator(A, D; device = KA.CPU()), b; w, h, rtol = 1.0e-12, max_iter = 2000)
    s = normal_cg(apply!, g; rtol = 1.0e-12, max_iter = 2000)
    @test s.converged && norm(s.x - ref.x) <= 1.0e-8 * norm(ref.x) && abs(s.iterations - ref.iterations) <= 2
    # the same solve with the normal operator given as an operator (`mul!`)
    N = SparseMatrixCSC(A' * A + D' * Diagonal(w) * D)
    @test norm(normal_cg(N, g; rtol = 1.0e-12, max_iter = 2000).x - ref.x) <= 1.0e-8 * norm(ref.x)
    # cgls's stopping rule, relative to ‖Aᵀb + DᵀWh‖: warm from the solution, 0 iterations; near it, fewer than cold;
    # the iteration budget counted as cgls counts it
    @test normal_cg(apply!, g; x0 = ref.x, rtol = 1.0e-6).iterations == 0
    cold = normal_cg(apply!, g; rtol = 1.0e-8, max_iter = 2000)
    warm = normal_cg(apply!, g; x0 = ref.x .+ 1.0e-4 .* randn(rng, n), rtol = 1.0e-8, max_iter = 2000)
    @test cold.converged && warm.converged && warm.iterations < cold.iterations && warm.relres <= 1.0e-8
    short = normal_cg(apply!, g; rtol = 1.0e-12, max_iter = 3)
    @test short.iterations == 3 && !short.converged && short.relres > 1.0e-12
    # a diagonal that is not positive is refused
    @test_throws ArgumentError normal_cg(apply!, g; diagonal = zeros(n))
end

@testitem "normal_cg and cgls: a Jacobi preconditioner converges on an ill-conditioned problem" begin
    using LinearAlgebra: norm, Diagonal, diag
    using SparseArrays: sprandn, sparse, SparseMatrixCSC
    using Random: Xoshiro, randn
    import KernelAbstractions as KA
    using VisualGeometryToolkitCore: difference_rows, PenalizedOperator, cgls, normal_cg
    rng = Xoshiro(14)
    TW, TH = 12, 10; n = TW * TH
    # a well-conditioned problem with its columns scaled over eight decades (the unknowns' units: a scene's log radiance
    # beside a profile's controls)
    A0 = sprandn(rng, 3n, n, 0.03) + sparse(1:n, 1:n, 1.0 .+ rand(rng, n), 3n, n)
    S = Diagonal(10.0 .^ range(-4, 4; length = n)[sortperm(rand(rng, n))])
    A = SparseMatrixCSC(A0 * S); D = difference_rows((TW, TH), 2) * S
    b = randn(rng, 3n); w = 1.0e-2
    N = SparseMatrixCSC(A' * A + w * D' * D); g = A' * b
    exact = Matrix(N) \ g
    d = Vector(diag(N))
    plain = normal_cg(N, g; rtol = 1.0e-10, max_iter = 300)
    jacobi = normal_cg(N, g; diagonal = d, rtol = 1.0e-10, max_iter = 300)
    @test !plain.converged
    @test jacobi.converged && jacobi.iterations < 300 && norm(jacobi.x - exact) <= 1.0e-6 * norm(exact)
    # cgls takes the same preconditioner (the stacked operator's column scaling) and makes the same solve
    op = PenalizedOperator(A, D; device = KA.CPU())
    @test !cgls(op, b; w, rtol = 1.0e-10, max_iter = 300).converged
    s = cgls(op, b; w, diagonal = d, rtol = 1.0e-10, max_iter = 300)
    @test s.converged && norm(s.x - exact) <= 1.0e-6 * norm(exact) && abs(s.iterations - jacobi.iterations) <= 3
end

@testitem "normal_cg on CUDA: the CPU solve's result, Jacobi and a warm start (availability-gated)" begin
    using SparseArrays: sprandn, sparse, SparseMatrixCSC
    using Random: Xoshiro, randn
    using LinearAlgebra: norm, Diagonal, diag, mul!
    import KernelAbstractions as KA
    using VisualGeometryToolkitCore: difference_rows, normal_cg, CSRMatrix
    cuda = try
        m = Base.require(Main, :CUDA)
        (m !== nothing && Base.invokelatest(getproperty(m, :functional))) ? m : nothing
    catch
        nothing
    end
    if cuda === nothing
        @test_skip cuda !== nothing      # CUDA is in the test target: no functional GPU here (recorded as Broken)
    else
        rng = Xoshiro(15)
        TW, TH = 40, 30; n = TW * TH
        A = sprandn(rng, 3n, n, 0.002) + sparse(1:n, 1:n, 0.3 .+ rand(rng, n), 3n, n)
        D = difference_rows((TW, TH), 1)
        N = SparseMatrixCSC(A' * A + 0.7 * D' * D); g = randn(rng, n); d = Vector(diag(N))
        x0 = (Matrix(N) \ g) .+ 1.0e-3 .* randn(rng, n)
        cpu = normal_cg(CSRMatrix(N), g; diagonal = d, x0, rtol = 1.0e-10, max_iter = 3000)
        dev = Base.invokelatest(getproperty(cuda, :CUDABackend))
        up(v) = Base.invokelatest(VisualGeometryToolkitCore.on_device, dev, v)
        Nd = Base.invokelatest(CSRMatrix, N; device = dev)
        apply!(y, x) = mul!(y, Nd, x)
        gpu = Base.invokelatest(normal_cg, apply!, up(g); diagonal = up(d), x0 = up(x0), rtol = 1.0e-10, max_iter = 3000)
        @test gpu.converged && Base.invokelatest(KA.get_backend, gpu.x) == dev
        @test norm(Base.invokelatest(Array, gpu.x) - cpu.x) <= 1.0e-8 * norm(cpu.x)
        @test abs(gpu.iterations - cpu.iterations) <= 2
    end
end

@testitem "bending_rows of a thin-plate and a spherical spline system: the second operator kind, its null space without rows" begin
    using LinearAlgebra: norm, normalize, rank
    using Random: Xoshiro, randn
    using StaticArrays: SVector
    using VisualGeometryToolkitCore: ThinPlateSplineSystem, SphericalSplineSystem, grid_nodes, bending_rows, bending_factor,
        bending_energy, mean_positive_eigenvalue
    rng = Xoshiro(21)
    tps = ThinPlateSplineSystem(grid_nodes(5))
    K = length(tps.nodes)
    L = bending_rows(tps)
    # one row per bending direction: the affine values (3) have none; the energy is the form's, and the factor's
    @test size(L) == (K - 3, K) && rank(L) == K - 3
    v = randn(rng, K)
    @test sum(abs2, L * v) ≈ bending_energy(tps, v) rtol = 1.0e-9
    @test L' * L ≈ bending_factor(tps)' * bending_factor(tps) rtol = 1.0e-9
    # m̄ reads the rows as any operator (RadiometricCamera's focus charts divide by it)
    @test mean_positive_eigenvalue(L) ≈ sum(abs2, bending_factor(tps)) / (K - 3) rtol = 1.0e-9
    # the spherical system: its constant values (1) have no row
    dirs = [normalize(SVector(randn(rng), randn(rng), 2 + abs(randn(rng)))) for _ in 1:20]
    sys = SphericalSplineSystem(dirs)
    Ls = bending_rows(sys)
    @test size(Ls) == (19, 20)
    w = randn(rng, 20)
    @test sum(abs2, Ls * w) ≈ bending_energy(sys, w) rtol = 1.0e-9
end
