@testitem "iou_below: a proven IoU < t, never a false refusal" begin
    using Test, StableRNGs
    using VisualGeometryToolkitCore
    using VisualGeometryToolkitCore: iou_below
    rng = StableRNG(11)
    proven = 0
    for _ in 1:2000
        e1 = Ellipse(Point2(10rand(rng), 10rand(rng)), 1 + 3rand(rng), 0.5 + rand(rng), π * rand(rng))
        e2 = Ellipse(Point2(10rand(rng), 10rand(rng)), 1 + 3rand(rng), 0.5 + rand(rng), π * rand(rng))
        t = rand(rng)
        if iou_below(e1, e2, t)
            proven += 1
            @test iou(e1, e2) < t                     # sound: never claims below for a pair at or above t
        end
    end
    @test proven > 0                                  # and it proves some
    e = Ellipse(Point2(0.0, 0.0), 2.0, 1.0, 0.0)
    @test !iou_below(e, e, 0.5) && !iou_below(e, e, 0.0)
    @test iou_below(e, Ellipse(Point2(100.0, 0.0), 2.0, 1.0, 0.0), 1.0e-9)   # far apart: disjoint discs
end
