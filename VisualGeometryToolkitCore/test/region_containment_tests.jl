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

@testitem "region containment: an intersection in a region when only the whole of it fits" begin
    using Test
    using VisualGeometryToolkitCore
    using VisualGeometryToolkitCore: check_region_support
    R(x, y, w, h) = Rect(Float64(x), Float64(y), Float64(w), Float64(h))
    E(x, y, a, b, θ = 0.0) = Ellipse(Point2(Float64(x), Float64(y)), Float64(a), Float64(b), Float64(θ))
    # RadiometricCamera's 14 mm housing case: a sensor ∩ its lens coverage inside a grid's support. Neither the sensor
    # nor the ellipse alone fits the support; the sensor clipped to the ellipse's box does, and so the intersection.
    sens = Rect(0, 0, 4256, 2848)
    ell = Ellipse(Point2(2131.8644793065105, 1426.2486754332483), 1718.85, 1716.49, deg2rad(22.48))
    supp = Rect(321.0, -155.294117647059, 3722.0, 3159.5882352941185)
    @test !(sens ⊆ supp) && !(ell ⊆ supp)
    @test (sens ∩ ell) ⊆ supp && (ell ∩ sens) ⊆ supp                                # ⊆ a Rect: the bounding box fits
    @test check_region_support(supp, sens ∩ ell; who = "test") === sens ∩ ell
    @test ((sens ∩ ell) ∩ R(-1000, -1000, 10000, 10000)) ⊆ supp                     # nested, no part alone fits
    @test (sens ∩ ell) ⊆ (supp ∩ E(2128, 1424, 5000, 5000))                         # ⊆ an intersection: both parts
    @test !((sens ∩ ell) ⊆ R(420, -200, 3800, 3300))                                # the ellipse reaches x ≈ 413.4
    # ⊆ an Ellipse: neither the strip nor the disc fits the target, their meet's box does
    strip, disc = R(0, 0, 100, 10), E(50, 5, 10, 10)
    target = E(50, 5, 15, 8)
    @test !(strip ⊆ target) && !(disc ⊆ target)
    @test (strip ∩ disc) ⊆ target && (disc ∩ strip) ⊆ target
    @test !((strip ∩ disc) ⊆ E(50, 5, 9, 8))                                        # the disc reaches x = 40 < 50 − 9
end
