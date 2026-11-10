# grid_golden_minimum — the one 1-D minimizer: a grid scan, then golden-section steps.

@testitem "grid_golden_minimum: an interior minimum refined; an end refined one-sided or returned; a periodic grid" begin
    using VisualGeometryToolkitCore: grid_golden_minimum
    xs = range(-8.0, 4.0; length = 25)                            # half a unit apart
    # interior: refined to the final bracket
    r = grid_golden_minimum(x -> (x + 2.3)^2, xs; iters = 20)
    @test abs(r.x + 2.3) < 1.0e-4 && !r.at_end
    # a minimum inside the first interval, which the scan samples lowest at its end: refined there, interior
    r = grid_golden_minimum(x -> (x + 7.8)^2, xs; iters = 20)
    @test abs(r.x + 7.8) < 1.0e-4 && !r.at_end
    # a minimum past the end: refined to the end itself, reported at the end
    r = grid_golden_minimum(x -> x + 100, xs; iters = 20)
    @test r.at_end && r.x == -8.0 && r.value == -8.0 + 100
    r = grid_golden_minimum(x -> -x, xs; iters = 20)
    @test r.at_end && r.x == 4.0
    # refine_ends = false: an end returned as it is (no evaluation past the scan)
    calls = Ref(0)
    r = grid_golden_minimum(x -> (calls[] += 1; (x + 7.8)^2), xs; iters = 20, refine_ends = false)
    @test r.at_end && r.x == -8.0 && calls[] == length(xs)
    # a descending grid (evaluated top down) finds the same minimum
    r = grid_golden_minimum(x -> (x + 2.3)^2, reverse(xs); iters = 20)
    @test abs(r.x + 2.3) < 1.0e-4
    # periodic: the neighbours wrap, never an end
    r = grid_golden_minimum(ψ -> 1 - cos(ψ - 0.05), range(0, 2π; length = 37)[1:36]; iters = 40, periodic = true)
    @test abs(r.x - 0.05) < 1.0e-6 && !r.at_end
end
