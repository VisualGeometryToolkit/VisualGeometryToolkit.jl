@testitem "conditioning" begin
    using Test
    using LinearAlgebra
    using Random
    using StaticArrays
    using VisualGeometryToolkitCore
    using VisualGeometryToolkitCore: test_conditioning, HomographyMat

    @testset "test_conditioning — κ ≤ max_cond" begin

        # Well-conditioned identity-like H: κ ≈ 1, should satisfy any reasonable threshold.
        H_id = HomographyMat{Float64}(Tuple(SMatrix{3, 3, Float64}(I)))
        @test test_conditioning(H_id)
        @test test_conditioning(H_id; max_cond = 1.5)

        # Mild scaling: σ = (3, 2, 1), κ = 3. Succeed with max_cond = 10, fail with 2.5.
        H_mild = HomographyMat{Float64}(
            Tuple(
                SMatrix{3, 3, Float64}(
                    3, 0, 0,
                    0, 2, 0,
                    0, 0, 1
                )
            )
        ) # column-major; diag = (3, 2, 1)
        @test test_conditioning(H_mild; max_cond = 10.0)
        @test test_conditioning(H_mild; max_cond = 3.5)
        @test !test_conditioning(H_mild; max_cond = 2.5)

        # Near-rank-deficient: σ ≈ (1, 1, 1e-10), κ ≈ 1e10 → reject at default 100.
        H_bad = HomographyMat{Float64}(
            Tuple(
                SMatrix{3, 3, Float64}(
                    1.0, 0, 0,
                    0, 1.0, 0,
                    0, 0, 1.0e-10
                )
            )
        )
        @test !test_conditioning(H_bad)

        # NaN / Inf entries: must reject without throwing.
        H_nan = HomographyMat{Float64}(
            Tuple(
                SMatrix{3, 3, Float64}(
                    NaN, 0, 0, 0, 1, 0, 0, 0, 1
                )
            )
        )
        @test !test_conditioning(H_nan)

        H_inf = HomographyMat{Float64}(
            Tuple(
                SMatrix{3, 3, Float64}(
                    Inf, 0, 0, 0, 1, 0, 0, 0, 1
                )
            )
        )
        @test !test_conditioning(H_inf)

        # Random well-conditioned H: agree with svdvals-based reference.
        Random.seed!(42)
        for _ in 1:50
            M = SMatrix{3, 3, Float64}(I) + 0.1 * SMatrix{3, 3}(randn(3, 3))
            H = HomographyMat{Float64}(Tuple(M))
            for κ in (5.0, 50.0, 500.0)
                sv = svdvals(SMatrix{3, 3, Float64}(H))
                ref = sv[1] <= κ * (sv[3] + eps(Float64))
                @test test_conditioning(H; max_cond = κ) == ref
            end
        end
    end
end
