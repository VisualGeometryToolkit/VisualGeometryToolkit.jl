# Native P3P (NativeSolver — port of PoseLib's p3p.cc).
#
# Oracle-free tests: known-pose recovery, solution-set structure, and
# degenerate-input behavior. Solution-for-solution parity against the
# compiled PoseLib backend is validated in RVG's test suite (which loads
# PoseLib) and was established at 1e-8 over 180k random instances.

@testitem "p3p_native" begin
    using Test
    using VisualGeometryToolkitCore
    using VisualGeometryToolkitCore: p3p, NativeSolver
    using StaticArrays, LinearAlgebra, Random
    using Rotations: RotMatrix

    _rand_rotation(rng) = SMatrix{3, 3, Float64}(rand(rng, RotMatrix{3, Float64}))   # Haar-uniform

    @testset "p3p (NativeSolver)" begin
        rng = MersenneTwister(1234)

        @testset "known-pose recovery (10k random)" begin
            worst = 0.0
            n_valid = 0
            for _ in 1:10_000
                R = _rand_rotation(rng)
                t = SVector(randn(rng), randn(rng), 3.0 + 2rand(rng))
                X = [SVector{3, Float64}(2randn(rng, 3)...) for _ in 1:3]
                any((R * Xi + t)[3] <= 0.1 for Xi in X) && continue
                xs = [SVector{3, Float64}(normalize(R * Xi + t)) for Xi in X]
                sols = p3p(NativeSolver(), xs, X)
                @test !isempty(sols)
                isempty(sols) && continue
                n_valid += 1
                err = minimum(
                    norm(SMatrix{3, 3, Float64}(s.R) - R) + norm(s.t - t)
                        for s in sols
                )
                worst = max(worst, err)
            end
            @test n_valid > 8_000
            @test worst < 1.0e-6
        end

        @testset "solutions reproject exactly" begin
            for _ in 1:100
                R = _rand_rotation(rng)
                t = SVector(randn(rng), randn(rng), 4.0)
                X = [SVector{3, Float64}(randn(rng, 3)...) for _ in 1:3]
                any((R * Xi + t)[3] <= 0.1 for Xi in X) && continue
                xs = [SVector{3, Float64}(normalize(R * Xi + t)) for Xi in X]
                for s in p3p(NativeSolver(), xs, X)
                    Rs = SMatrix{3, 3, Float64}(s.R)
                    for (xi, Xi) in zip(xs, X)
                        p = Rs * Xi + s.t
                        # every returned pose maps each 3D point onto its bearing ray
                        @test norm(normalize(p) - xi) < 1.0e-6
                    end
                end
            end
        end

        @testset "bare default routes to NativeSolver" begin
            xs = [
                normalize(SVector(0.1, 0.05, 1.0)), normalize(SVector(-0.2, 0.1, 1.0)),
                normalize(SVector(0.0, -0.15, 1.0)),
            ]
            Xs = [SVector(1.0, 0.0, 5.0), SVector(-1.0, 1.0, 6.0), SVector(0.5, -1.0, 4.0)]
            @test collect(p3p(xs, Xs)) == collect(p3p(NativeSolver(), xs, Xs))
        end

        @testset "degenerate inputs do not throw" begin
            # collinear world points / repeated points: garbage in, no exception out
            p0 = SVector(0.0, 0.0, 0.0); d = SVector(1.0, 0.0, 0.0)
            Xc = [p0, p0 + d, p0 + 2d]
            xs = [normalize(SVector(0.1i, 0.05, 1.0)) for i in 1:3]
            @test p3p(NativeSolver(), xs, Xc) isa Any        # runs without DomainError
            Xr = [p0, p0, p0]
            @test p3p(NativeSolver(), xs, Xr) isa Any
        end

        @testset "solve is allocation-free" begin
            # The solver runs once per RANSAC trial; a heap allocation here is
            # ~40k GC objects per BlobBoards frame. The former MVector{4} scratch
            # escaped LLVM's elision and allocated per call — the NTuple
            # accumulator (2026-08-17) must not regress.
            xs = [
                normalize(SVector(0.1, 0.05, 1.0)), normalize(SVector(-0.2, 0.1, 1.0)),
                normalize(SVector(0.0, -0.15, 1.0)),
            ]
            Xs = [SVector(1.0, 0.0, 5.0), SVector(-1.0, 1.0, 6.0), SVector(0.5, -1.0, 4.0)]
            solve_alloc(xs, Xs) = @allocated p3p(NativeSolver(), xs, Xs)
            solve_alloc(xs, Xs)                    # warm
            @test solve_alloc(xs, Xs) == 0
        end
    end
end
