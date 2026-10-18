# =============================================================================
# The Gaussian-Uniform boundary, both directions.
#
# `decision_threshold` and its inverse `decision_width(::GauInlier, ·)` are the
# ONE place the MAP rule of the Gaussian-uniform mixture lives, so what needs
# pinning is that they are genuine inverses — across the Gaussian AND the
# scale-marginalised (Student-t) inlier, every codimension, and a label prior
# that is not 1/2.
# =============================================================================

@testitem "gau_inlier_boundary" begin
    using Test
    using VisualGeometryToolkitCore: GauInlier, decision_threshold, decision_width

    @testset "GauInlier boundary: the two directions are inverse" begin
        @testset "round trip outlier half-width -> crossover -> outlier half-width" begin
            for d in (1, 2, 3, 5), ν in (Inf, 1.5, 2.5, 7.0, 40.0),
                s2 in (0.0081, 1.0, 7.3), γ in (0.5, 0.2, 0.8),
                outlier_halfwidth in (0.5, 3.41, 90.0, 1.0e4)

                m = GauInlier(s2; nu = ν, gamma = γ, d_g = d)
                crossover = decision_threshold(m, outlier_halfwidth)
                # 0 means the outlier component already explains everything;
                # there is nothing to invert there, and that guard is tested below.
                crossover > 0 && @test decision_width(m, crossover) ≈ outlier_halfwidth rtol = 1.0e-12
            end
        end

        @testset "round trip crossover -> outlier half-width -> crossover" begin
            for d in (1, 2, 4), ν in (Inf, 3.0), s2 in (0.25, 4.0), crossover in (0.1, 1.0, 12.0)
                m = GauInlier(s2; nu = ν, gamma = 0.5, d_g = d)
                @test decision_threshold(m, decision_width(m, crossover)) ≈ crossover rtol = 1.0e-12
            end
        end

        # The Gaussian is the nu -> Inf limit of the Student-t, not a second model:
        # the two branches must MEET, or one of the closed forms is wrong.
        @testset "Gaussian is the ν → ∞ limit" begin
            for d in (1, 2, 3), s2 in (0.09, 2.0), outlier_halfwidth in (1.0, 20.0)
                g = decision_threshold(GauInlier(s2; d_g = d), outlier_halfwidth)
                t = decision_threshold(GauInlier(s2; nu = 1.0e9, gamma = 0.5, d_g = d), outlier_halfwidth)
                @test t ≈ g rtol = 1.0e-5
            end
        end

        # gamma enters as log(gamma/(1-gamma)) — the term the paper drops at 1/2.
        # A larger inlier prior must ADMIT MORE, i.e. push the boundary out.
        @testset "the label prior moves the boundary monotonically" begin
            m(γ) = GauInlier(1.0; nu = 5.0, gamma = γ, d_g = 2)
            crossovers = [decision_threshold(m(γ), 3.41) for γ in (0.1, 0.3, 0.5, 0.7, 0.9)]
            @test issorted(crossovers)
            @test decision_threshold(m(0.5), 3.41) ≈
                decision_threshold(GauInlier(1.0; nu = 5.0, d_g = 2), 3.41)
        end

        # A half-width small enough that the uniform outlier density beats the
        # inlier peak admits nothing: 0, never a negative root.
        @testset "no admissible set ⇒ 0, not a negative root" begin
            for ν in (Inf, 4.0)
                @test decision_threshold(GauInlier(100.0; nu = ν, d_g = 1), 1.0e-6) == 0.0
            end
        end

        # The retired Welsch native map, ŝ*sqrt(pi/2)*exp(scale_multiple^2/2), IS
        # this pair at codimension 1 — that is why it could be deleted rather than
        # kept. Not bit-identical: the generalised path associates the floats
        # differently, ~1e-14 relative, twelve orders below the ~2.3%/point
        # spacing of the protocol's log grid, so it cannot move a selection.
        @testset "subsumes the retired closed form" begin
            old(scale_multiple, ŝ) = ŝ * sqrt(π / 2) * exp(0.5 * scale_multiple^2)
            for ŝ in (0.05, 0.3, 1.0, 2.7, 11.0), scale_multiple in (0.4, 0.8, 1.5, 3.0, 4.5, 6.0)
                m = GauInlier(ŝ^2; d_g = 1)
                @test decision_width(m, scale_multiple * ŝ) ≈ old(scale_multiple, ŝ) rtol = 1.0e-13
                # …and it inverts to exactly scale_multiple·ŝ, which is what made it a boundary.
                @test decision_threshold(m, decision_width(m, scale_multiple * ŝ)) ≈ scale_multiple * ŝ rtol = 1.0e-12
            end
        end
    end
end
