# Native 5-point essential (NativeSolver — port of PoseLib's relpose_5pt.cc).
#
# Oracle-free: ground-truth recovery + epipolar-constraint residuals.
# Solution-for-solution parity vs the compiled PoseLib backend was validated
# at development time (16,833 hard synthetics: native recovered ground truth
# in more cases than the C++ reference; medians ~3.5e-12).

@testitem "essential_5pt_native" begin
    using Test
    using VisualGeometryToolkitCore
    using VisualGeometryToolkitCore: essential_5pt, NativeSolver, EssentialMat, Vec3
    using StaticArrays, LinearAlgebra, Random
    using Rotations: RotMatrix

    @testset "essential_5pt (NativeSolver)" begin
        rng = MersenneTwister(4321)
        n_ok = 0; n_run = 0
        for _ in 1:2_000
            R = SMatrix{3, 3, Float64, 9}(rand(rng, RotMatrix{3, Float64}))
            t = normalize(SVector{3, Float64}(randn(rng, 3)...))
            X = [SVector{3, Float64}(2randn(rng), 2randn(rng), 6.0 + 2rand(rng)) for _ in 1:5]
            x2p = [R * Xi + t for Xi in X]
            any(p[3] <= 0.2 for p in x2p) && continue
            x1 = ntuple(i -> Vec3{Float64}((X[i] / X[i][3])...), Val(5))
            x2 = ntuple(i -> Vec3{Float64}((x2p[i] / x2p[i][3])...), Val(5))
            sols = essential_5pt(x1, x2)          # bare default = NativeSolver
            n_run += 1
            isempty(sols) && continue

            # every solution satisfies the five epipolar constraints
            for E in sols
                Em = SMatrix{3, 3, Float64, 9}(Tuple(E))
                for i in 1:5
                    @test abs(SVector{3, Float64}(x2[i])' * Em * SVector{3, Float64}(x1[i])) < 1.0e-8
                end
            end

            # ground truth among the solutions (up to sign)
            Egt = (skew(t) * R) / norm(skew(t) * R)
            rec = minimum(
                min(
                    norm(SMatrix{3, 3, Float64, 9}(Tuple(E)) - Egt),
                    norm(SMatrix{3, 3, Float64, 9}(Tuple(E)) + Egt)
                ) for E in sols
            )
            rec < 1.0e-6 && (n_ok += 1)
        end
        # the 5-pt problem has a genuine hard tail; the compiled PoseLib reference
        # achieves ~93.5% recovery on this ensemble and the native port ~95%
        # (full-sweep, n=16.8k). Gate at 92% to absorb subset variance while
        # still catching real regressions.
        # ~2/3 of random configs are skipped by the behind-camera guard, so
        # 2,000 attempts yield ~650 valid trials.
        @test n_run > 500
        @test n_ok / n_run > 0.92
    end

    @testset "plane points are refused by type" begin
        # The wrong implementation this catches: an untyped kernel that reads
        # `SVector{3}(x[i])`, where a 2-vector surfaces as a DimensionMismatch
        # from deep inside the solve — or, worse, a lift that silently appends
        # z = 1 and excludes every ray at or past 90°.
        X = (SVector(0.3, -0.2, 5.0), SVector(-0.4, 0.5, 6.0), SVector(0.7, 0.1, 4.5),
             SVector(-0.2, -0.6, 5.5), SVector(0.1, 0.4, 7.0))
        c, s = cos(0.1), sin(0.1)
        R = SMatrix{3, 3, Float64}(c, 0, -s, 0, 1, 0, s, 0, c)       # 0.1 rad about y
        t = SVector(0.5, 0.05, 0.02)
        f1 = ntuple(i -> normalize(X[i]), Val(5))                  # unit bearings
        f2 = ntuple(i -> normalize(R * X[i] + t), Val(5))
        p = ntuple(i -> PerspectiveMap()(X[i]), Val(5)) # z = 1 plane points
        @test_throws MethodError essential_5pt(NativeSolver(), p, p)
        @test_throws MethodError essential_5pt(NativeSolver(), p, f2)
        @test_throws MethodError essential_5pt(p, p)
        # the accepted shapes still solve, and recover the true pose
        Egt = skew(t) * R
        Egt /= norm(Egt)
        for (a, b) in ((f1, f2), (SVector(f1), SVector(f2)))
            sols = essential_5pt(NativeSolver(), a, b)
            @test !isempty(sols)
            @test minimum(min(norm(SMatrix{3, 3, Float64, 9}(Tuple(E)) / norm(E) - Egt),
                              norm(SMatrix{3, 3, Float64, 9}(Tuple(E)) / norm(E) + Egt)) for E in sols) < 1.0e-8
        end
    end
end
