# _arrow_solve — the Schur elimination of per-board blocks of static size m:
# against a dense solve of the same normal equations at m = 3 (what radial
# homographies need), bit for bit with the division camera's scalar depth loop
# at m = 1, a board whose block carries no information left out, and one whose
# block is symmetric only to rounding kept.
@testitem "arrow_solve" begin
    using Test, LinearAlgebra
    using StableRNGs: StableRNG
    using StaticArrays: SMatrix, SVector
    using VisualGeometryToolkitEstimation: _arrow_solve

    # Random arrow normal equations AᵀA·x = Aᵀy from rows where board i's block
    # columns are nonzero only on its own rows.
    function arrow_system(rng, ns, m, nb; rows = 4m + 2)
        A = zeros(nb * rows, ns + nb * m)
        for i in 1:nb, r in 1:rows
            row = (i - 1) * rows + r
            A[row, 1:ns] .= randn(rng, ns)
            A[row, ns .+ ((i - 1) * m + 1:i * m)] .= randn(rng, m)
        end
        y = randn(rng, size(A, 1))
        return A' * A, A' * y
    end
    blocks_of(M, v, ns, m, nb) = [
        (
                C = SMatrix{ns, m, Float64, ns * m}(M[1:ns, ns .+ ((i - 1) * m + 1:i * m)]),
                D = SMatrix{m, m, Float64, m * m}(M[ns .+ ((i - 1) * m + 1:i * m), ns .+ ((i - 1) * m + 1:i * m)]),
                b = SVector{m, Float64}(v[ns .+ ((i - 1) * m + 1:i * m)]),
            ) for i in 1:nb
    ]

    @testset "m = 3 against the dense solve" begin
        rng = StableRNG(7)
        for (ns, nb) in ((1, 1), (2, 5), (3, 12))
            M, v = arrow_system(rng, ns, 3, nb)
            xs, ts = _arrow_solve(M[1:ns, 1:ns], v[1:ns], blocks_of(M, v, ns, 3, nb))
            x = M \ v
            @test ts isa Vector{SVector{3, Float64}}
            @test norm(vcat(xs, reduce(vcat, ts)) - x) <= 1.0e-10 * norm(x)
        end
    end

    @testset "m = 1 is the division camera's scalar loop, bit for bit" begin
        # the elimination of `_division_camera_solve`, verbatim
        function scalar_arrow(Snn, bsh, Cnb, Dbb, bdep)
            ns, nchosen = size(Cnb)
            Sred = copy(Snn); bred = copy(bsh)
            @inbounds for col in 1:nchosen
                dcol = Dbb[col]
                for p in 1:ns
                    cp = Cnb[p, col] / dcol
                    for q in 1:ns
                        Sred[p, q] -= cp * Cnb[q, col]
                    end
                    bred[p] -= cp * bdep[col]
                end
            end
            xs = Sred \ bred
            ts = map(1:nchosen) do col
                acc = bdep[col]
                for p in 1:ns
                    acc -= Cnb[p, col] * xs[p]
                end
                acc / Dbb[col]
            end
            return xs, ts
        end
        rng = StableRNG(11)
        for (ns, nb) in ((1, 6), (2, 6), (3, 41))
            M, v = arrow_system(rng, ns, 1, nb)
            Cnb = M[1:ns, ns + 1:end]; Dbb = [M[ns + i, ns + i] for i in 1:nb]; bdep = v[ns + 1:end]
            xo, to = scalar_arrow(M[1:ns, 1:ns], v[1:ns], Cnb, Dbb, bdep)
            xs, ts = _arrow_solve(M[1:ns, 1:ns], v[1:ns], blocks_of(M, v, ns, 1, nb))
            @test all(xs .=== xo)
            @test all(first.(ts) .=== to)
        end
    end

    @testset "a block without information is left out, its t NaN" begin
        rng = StableRNG(5)
        ns, m, nb = 2, 3, 4
        M, v = arrow_system(rng, ns, m, nb)
        blocks = blocks_of(M, v, ns, m, nb)
        # board 3 carries nothing: its rows were all weighted to zero
        dead = (C = zero(blocks[3].C), D = zero(blocks[3].D), b = zero(blocks[3].b))
        xs, ts = _arrow_solve(M[1:ns, 1:ns], v[1:ns], [blocks[1:2]; dead; blocks[4:4]])
        keep = [1:ns; ns .+ (1:2m); ns + 3m .+ (1:m)]
        x = M[keep, keep] \ v[keep]
        @test all(isnan, ts[3])
        @test norm(vcat(xs, ts[1], ts[2], ts[4]) - x) <= 1.0e-10 * norm(x)
        # a board whose D is symmetric only to rounding (as an accumulation of
        # w·rrᵀ is: (w·r_a)·r_b against (w·r_b)·r_a) is a board, not dropped
        rng2 = StableRNG(17)
        Ms, vs = arrow_system(rng2, ns, m, nb)
        bs = blocks_of(Ms, vs, ns, m, nb)
        asym = map(bs) do blk
            Dn = Matrix(blk.D)
            Dn[1, 2] = nextfloat(Dn[1, 2]); Dn[3, 2] = prevfloat(Dn[3, 2])  # an ulp off, either side
            (C = blk.C, D = SMatrix{m, m, Float64, m * m}(Dn), b = blk.b)
        end
        @test !any(issymmetric(blk.D) for blk in asym)                      # the case: not bit-symmetric
        @test all(maximum(abs, blk.D - blk.D') <= 1.0e-14 * maximum(abs, blk.D) for blk in asym)
        Ma = copy(Ms)
        for (i, blk) in enumerate(asym)
            idx = ns .+ ((i - 1) * m + 1:i * m)
            Ma[idx, idx] .= Matrix(Symmetric(Matrix(blk.D)))                # the upper triangle, as the solve reads it
        end
        xa, ta = _arrow_solve(Ms[1:ns, 1:ns], vs[1:ns], asym)
        x = Ma \ vs
        @test all(t -> all(isfinite, t), ta)
        @test norm(vcat(xa, reduce(vcat, ta)) - x) <= 1.0e-10 * norm(x)
        # and at m = 1 the camera's rule, d > 0
        M1, v1 = arrow_system(rng, ns, 1, 3)
        b1 = blocks_of(M1, v1, ns, 1, 3)
        _, t1 = _arrow_solve(M1[1:ns, 1:ns], v1[1:ns], [b1[1], (C = zero(b1[2].C), D = zero(b1[2].D), b = zero(b1[2].b)), b1[3]])
        @test isnan(t1[2][1]) && all(isfinite, (t1[1][1], t1[3][1]))
        # a non-finite D is a dead block at every m (a Cholesky of a NaN may report success)
        for bad in (NaN, Inf)
            Dn = Matrix(blocks[3].D); Dn[2, 2] = bad
            nanb = (C = blocks[3].C, D = SMatrix{m, m, Float64, m * m}(Dn), b = blocks[3].b)
            _, tn = _arrow_solve(M[1:ns, 1:ns], v[1:ns], [blocks[1:2]; nanb; blocks[4:4]])
            @test all(isnan, tn[3]) && all(t -> all(isfinite, t), tn[[1, 2, 4]])
        end
        _, t1n = _arrow_solve(M1[1:ns, 1:ns], v1[1:ns], [b1[1], (C = b1[2].C, D = typeof(b1[2].D)(Inf), b = b1[2].b), b1[3]])
        @test isnan(t1n[2][1])
        # any AbstractVector of the blocks is taken (a view, a Memory-backed vector)
        xv, _ = _arrow_solve(M1[1:ns, 1:ns], v1[1:ns], view(b1, 1:3))
        @test xv == _arrow_solve(M1[1:ns, 1:ns], v1[1:ns], b1)[1]
        # the shared block must match the blocks' coupling
        @test_throws DimensionMismatch _arrow_solve(zeros(3, 3), zeros(3), b1)
    end
end
