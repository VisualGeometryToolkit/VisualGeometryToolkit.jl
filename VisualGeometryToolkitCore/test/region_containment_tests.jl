@testitem "region containment: ⊆ for every pair a fit and a support take" begin
    using Test
    using VisualGeometryToolkitCore
    using VisualGeometryToolkitCore: WholePlane, RegionIntersection
    R(x, y, w, h) = Rect(Float64(x), Float64(y), Float64(w), Float64(h))
    E(x, y, a, b, θ = 0.0) = Ellipse(Point2(Float64(x), Float64(y)), Float64(a), Float64(b), Float64(θ))
    big, small = R(0, 0, 100, 100), R(40, 40, 10, 10)
    disc, inner = E(50, 50, 40, 40), E(50, 50, 20, 10, 0.3)
    @test small ⊆ big && !(big ⊆ small)                                  # Rect ⊆ Rect
    @test small ⊆ disc && !(big ⊆ disc)                                  # Rect ⊆ Ellipse
    @test inner ⊆ big && !(E(5, 5, 20, 20) ⊆ big)                        # Ellipse ⊆ Rect
    @test inner ⊆ disc && !(disc ⊆ inner)                                # Ellipse ⊆ Ellipse
    @test !(E(50, 50, 39.9, 10) ⊆ E(50, 50, 40, 10, π / 2))              # a turned ellipse: its ends poke out
    @test E(50, 50, 40, 10) ⊆ E(50, 50, 40, 10)                          # an ellipse in itself (the slack)
    @test small ⊆ (big ∩ disc) && !(big ⊆ (big ∩ disc))                  # ⊆ an intersection: in both parts
    @test inner ⊆ (big ∩ disc)
    @test (big ∩ disc) ⊆ disc && (small ∩ big) ⊆ big && (big ∩ inner) ⊆ disc   # an intersection ⊆ a region: either part
    @test (small ∩ disc) ⊆ (big ∩ disc)                                  # intersection ⊆ intersection
    @test all(r -> r ⊆ WholePlane(), (small, inner, big ∩ disc, WholePlane()))
    @test !(WholePlane() ⊆ big) && !(WholePlane() ⊆ disc) && !(WholePlane() ⊆ (big ∩ disc))
end
