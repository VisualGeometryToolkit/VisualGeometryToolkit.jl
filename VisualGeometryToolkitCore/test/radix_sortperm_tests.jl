# =============================================================================
# radix_sortperm_tests.jl — output-identity gate for the sweep radix perm-sort
# =============================================================================
#
# `radix_sortperm!` replaces `sortperm!` on the sweep!'s exact-size fast
# path. Base's `sortperm!` documents a stability guarantee, so the radix sort
# must reproduce its permutation BITWISE for every input — ties included
# (stability-on-ties is exactly the risky part, so the adversarial cases here
# are tie-heavy: massive duplicate blocks, all-equal vectors, ±0.0 mixtures,
# NaN clusters, and the underflow-to-zero regimes the RANSAC score vectors
# actually produce).
# =============================================================================

@testitem "radix_sortperm" begin
    using Test
    using Random
    using VisualGeometryToolkitCore
    using VisualGeometryToolkitCore: RadixPermBuffers, radix_sortperm!, radix_key, JeffreysPrior,
        default_scoring_workspace, sweep!

    # The sortperm!-identity contract is a FULL-PRECISION claim, so it is checked
    # at the UInt64 key width. (The default width is UInt32 — see the UInt32
    # testset below for the contract that width actually satisfies.)
    function _check_identity(target::Vector{Float64}, ::Type{K} = UInt64) where {K<:Unsigned}
        n = length(target)
        rb = RadixPermBuffers{K}(n)
        perm = Vector{Int}(undef, n)
        radix_sortperm!(perm, target, rb, n)
        perm == sortperm(target)
    end

    @testset "radix perm-sort output identity" begin
        rng = MersenneTwister(0x5eed)

        @testset "radix_key is isless-exact" begin
            specials = [0.0, -0.0, 1.0, -1.0, Inf, -Inf, NaN, -NaN,
                        reinterpret(Float64, 0x7ff8000000000123),   # NaN payload
                        reinterpret(Float64, 0xfff8000000000456),   # negative NaN payload
                        floatmin(Float64), -floatmin(Float64),
                        5.0e-324, -5.0e-324,                        # subnormals
                        floatmax(Float64), -floatmax(Float64), π, -π]
            for a in specials, b in specials
                ka, kb = radix_key(a), radix_key(b)
                @test (ka < kb) == isless(a, b)
                @test (ka == kb) == (!isless(a, b) && !isless(b, a))
            end
        end

        @testset "random vectors" begin
            for n in (1, 2, 3, 7, 63, 64, 65, 255, 256, 1000, 2500, 4096), rep in 1:5
                @test _check_identity(randn(rng, n))
                @test _check_identity(abs2.(randn(rng, n)))          # score-like (r²)
            end
        end

        @testset "tie-heavy" begin
            for n in (2, 5, 64, 257, 1000, 2500)
                @test _check_identity(zeros(n))                       # all equal
                @test _check_identity(fill(3.75, n))
                @test _check_identity(Float64.(rand(rng, 0:3, n)))    # 4 distinct values
                @test _check_identity(Float64.(rand(rng, 0:1, n)))    # 2 distinct values
                # big zero sea + few distinct (Gumbel-key underflow regime)
                v = zeros(n); v[rand(rng, 1:n, max(1, n ÷ 20))] .= rand(rng, max(1, n ÷ 20))
                @test _check_identity(v)
                # duplicate blocks in shuffled order
                v = repeat(rand(rng, cld(n, 10)), inner = 10)[1:n]
                @test _check_identity(shuffle(rng, v))
            end
        end

        @testset "signed zeros, infinities, NaNs" begin
            for n in (8, 64, 1000)
                v = shuffle(rng, [fill(-0.0, n ÷ 4); fill(0.0, n ÷ 4);
                                  fill(Inf, n ÷ 4); rand(rng, n - 3 * (n ÷ 4))])
                @test _check_identity(v)
                v = shuffle(rng, [fill(NaN, n ÷ 4); fill(-NaN, n ÷ 4);
                                  randn(rng, n - 2 * (n ÷ 4))])
                @test _check_identity(v)
                v = shuffle(rng, [fill(-Inf, 2); fill(Inf, 2); fill(-0.0, 2); fill(0.0, 2);
                                  5.0e-324 .* rand(rng, 0:5, n - 8)])
                @test _check_identity(v)
            end
        end

        @testset "capacity > n (trimmed reuse)" begin
            rb = RadixPermBuffers(100)
            perm = Vector{Int}(undef, 100)
            v100 = randn(rng, 100)
            for n in (100, 37, 5, 1)
                resize!(perm, n)
                v = v100[1:n]
                radix_sortperm!(perm, v, rb, n)
                @test perm == sortperm(v)
            end
        end

        # -------------------------------------------------------------------
        # UInt32 key width (THE DEFAULT — the scoring path's sort precision).
        # Its contract is NOT full-precision sortperm identity: keys within one
        # Float32 ulp collapse, and LSD stability then orders such a group by
        # index. What it DOES guarantee, and what scoring relies on:
        #   (a) exact sortperm identity whenever the data is Float32-representable
        #       — which is the real case, since a pool's residual matrix is a
        #       Matrix{Float32} widened into the Float64 sweep buffer;
        #   (b) a valid ascending order (non-decreasing) for ANY Float64 input;
        #   (c) type stability / concrete inference at both widths.
        # -------------------------------------------------------------------
        @testset "UInt32 key width (default): Float32-exact, always ordered" begin
            rng32 = MersenneTwister(0xf32)
            # (a) Float32-derived data — the production case — must be EXACT.
            for n in (1, 2, 7, 64, 1000)
                v32 = Float64.(Float32.(randn(rng32, n)))
                @test _check_identity(v32, UInt32)
                r32 = Float64.(abs2.(Float32.(randn(rng32, n))))   # score-like r²
                @test _check_identity(r32, UInt32)
                @test _check_identity(Float64.(Float32.(rand(rng32, 0:3, n))), UInt32)
            end
            # ±0.0 / Inf / NaN survive the narrowing.
            for v in ([0.0, -0.0, 1.0, -1.0], [Inf, -Inf, 0.0, 1.0], [NaN, 1.0, -1.0, NaN])
                @test _check_identity(v, UInt32)
            end
            # (b) arbitrary Float64: not necessarily sortperm-identical, but the
            #     emitted order must still be ascending in Float32 precision, and
            #     a permutation of 1:n.
            for n in (2, 33, 517)
                v = randn(rng32, n)
                rb = RadixPermBuffers{UInt32}(n); perm = Vector{Int}(undef, n)
                radix_sortperm!(perm, v, rb, n)
                @test sort(perm) == collect(1:n)
                f = Float32.(v[perm])
                @test issorted(f)
            end
            # (c) default width is UInt32, and both widths infer concretely.
            @test RadixPermBuffers(16) isa RadixPermBuffers{UInt32}
            @test size(RadixPermBuffers{UInt32}(16).counts, 2) == 4
            @test size(RadixPermBuffers{UInt64}(16).counts, 2) == 8
            for K in (UInt32, UInt64)
                rb = RadixPermBuffers{K}(8); perm = Vector{Int}(undef, 8)
                v = randn(MersenneTwister(1), 8)
                @test isconcretetype(typeof(rb))
                @test (@inferred radix_sortperm!(perm, v, rb, 8)) === nothing
            end
            # the scoring workspace carries the width in its type (no abstract field)
            sws32 = default_scoring_workspace(16; dg = 1, scale_prior = JeffreysPrior())
            @test isconcretetype(typeof(sws32))
            @test typeof(sws32).parameters[1] === UInt32
        end

        @testset "sweep! radix path ≡ sortperm! path; the sort allocation-free, the sweep its kernel's column" begin
            for n in (50, 257, 2500)
                sws = default_scoring_workspace(n; scale_prior = JeffreysPrior())
                scores = abs2.(randn(rng, n))
                scores[rand(rng, 1:n, n ÷ 5)] .= 0.0                  # ties in the sort target
                s_radix, k_radix = sweep!(sws.perm, sws.lg_table, 5, log(40.0), 1,
                                          scores, n; scratch = sws.scratch,
                                          radix = sws.radix, scale_prior = JeffreysPrior())
                perm_radix = copy(sws.perm)
                s_cmp, k_cmp = sweep!(sws.perm, sws.lg_table, 5, log(40.0), 1,
                                      scores, n; scratch = sws.scratch, scale_prior = JeffreysPrior())   # no radix ⇒ sortperm!
                @test s_radix === s_cmp
                @test k_radix == k_cmp
                @test perm_radix == sws.perm
                # the radix sort allocates nothing (warm); the sweep is the one prefix-sweep kernel (the owner's
                # one-kernel ruling, twin audit V2): without the workspace's launch buffers it allocates the sorted
                # column and its penalty buffer (16 B a point) and its launch's few hundred bytes, with them (the workspace's own sweep!)
                # nothing
                f(sws, scores, n) = sweep!(sws.perm, sws.lg_table, 5, log(40.0), 1,
                                           scores, n; scratch = sws.scratch,
                                           radix = sws.radix, scale_prior = JeffreysPrior())
                g(sws, scores, n) = sweep!(sws, scores, n; model_dof = 5, log_two_outlier_halfwidth = log(40.0))
                f(sws, scores, n); g(sws, scores, n)
                @test (@allocated VisualGeometryToolkitCore._sort_perm!(sws.perm, scores, n; scratch = sws.scratch, radix = sws.radix)) == 0
                @test (@allocated f(sws, scores, n)) <= 16n + 2048
                @test (@allocated g(sws, scores, n)) == 0 && g(sws, scores, n) === (s_radix, k_radix)
            end
        end
    end
end
