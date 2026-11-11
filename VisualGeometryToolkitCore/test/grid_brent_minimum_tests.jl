# grid_brent_minimum — the one 1-D minimizer: a grid scan, then Brent's method.

@testitem "grid_brent_minimum: an interior minimum to Brent's floor; an end refined one-sided; a periodic grid" begin
    using VisualGeometryToolkitCore: grid_brent_minimum
    xs = range(-8.0, 4.0; length = 25)                            # half a unit apart
    # interior: to Brent's own floor, √eps relative (no tolerance to supply)
    r = grid_brent_minimum(x -> (x + 2.3)^2, xs)
    @test abs(r.x + 2.3) < 1.0e-7 && !r.at_end
    # a quadratic is found by the parabola in a handful of evaluations, far fewer than golden steps to the same floor
    calls = Ref(0)
    r = grid_brent_minimum(x -> (calls[] += 1; (x + 2.3)^2), xs)
    @test calls[] - length(xs) <= 12
    # a non-quadratic minimum (|x + 2.3|^1.5, no parabola fits it) still to the floor's order
    r = grid_brent_minimum(x -> abs(x + 2.3)^1.5, xs)
    @test abs(r.x + 2.3) < 1.0e-6 && !r.at_end
    # a minimum inside the first interval, which the scan samples lowest at its end: refined there, interior
    r = grid_brent_minimum(x -> (x + 7.8)^2, xs)
    @test abs(r.x + 7.8) < 1.0e-7 && !r.at_end
    # a minimum past the end: the end itself, reported at the end
    r = grid_brent_minimum(x -> x + 100, xs)
    @test r.at_end && r.x == -8.0 && r.value == -8.0 + 100
    r = grid_brent_minimum(x -> -x, xs)
    @test r.at_end && r.x == 4.0
    # a descending grid (evaluated top down) finds the same minimum
    r = grid_brent_minimum(x -> (x + 2.3)^2, reverse(xs))
    @test abs(r.x + 2.3) < 1.0e-7
    # Inf and NaN are never the minimum, and a non-finite value beside the basin does not derail the parabola
    r = grid_brent_minimum(x -> x > -1.0 ? Inf : x < -7.0 ? NaN : (x + 2.3)^2, xs)
    @test abs(r.x + 2.3) < 1.0e-7 && isfinite(r.value)
    # periodic: the neighbours wrap, never an end
    r = grid_brent_minimum(ψ -> 1 - cos(ψ - 0.05), range(0, 2π; length = 37)[1:36]; periodic = true)
    @test abs(r.x - 0.05) < 1.0e-7 && !r.at_end
    r = grid_brent_minimum(ψ -> 1 - cos(ψ + 0.05), range(0, 2π; length = 37)[1:36]; periodic = true)   # across 0
    @test abs(mod(r.x + 0.05 + π, 2π) - π) < 1.0e-7
end
