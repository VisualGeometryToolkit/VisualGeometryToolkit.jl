# =============================================================================
# One-dimensional minimization: a grid scan, then golden-section steps
# =============================================================================

"""
    grid_golden_minimum(f, xs::AbstractRange; iters, periodic = false, refine_ends = true)
        -> (; x, value, at_end)

The minimum of `f` over the evenly spaced grid `xs` (evaluated in its order), the least
value refined by `iters` golden-section steps between its neighbours: `(; x, value,
at_end)` of the smallest value evaluated. On a `periodic` grid the neighbours wrap (`f`
periodic). On a linear one a least value at an end of the grid is refined on its one
neighbouring interval when `refine_ends` (else returned as it is), and `at_end` says
the result lies at that end: within the final bracket (`step(xs)·0.618^iters`) of it, or
unrefined there. The stack's one grid-and-golden minimizer: Core's azimuthal minimum
(the view-angle limit, the inscribed radius) and Estimation's GCV weight search.
"""
function grid_golden_minimum(f, xs::AbstractRange; iters::Integer, periodic::Bool = false, refine_ends::Bool = true)
    vs = map(f, xs)
    k = argmin(vs)
    h = abs(step(xs))
    first_end = !periodic && k == firstindex(xs)
    last_end = !periodic && k == lastindex(xs)
    (first_end || last_end) && !refine_ends && return (; x = float(xs[k]), value = vs[k], at_end = true)
    # the bracket: both neighbours, or at an end the one neighbouring interval
    a, b = first_end ? (xs[k], xs[k] + sign(step(xs)) * h) : last_end ? (xs[k] - sign(step(xs)) * h, xs[k]) : (xs[k] - h, xs[k] + h)
    a, b = minmax(a, b)
    g = (sqrt(5) - 1) / 2
    c, d = b - g * (b - a), a + g * (b - a)
    fc, fd = f(c), f(d)
    for _ in 1:iters
        if fc < fd
            b, d, fd = d, c, fc
            c = b - g * (b - a)
            fc = f(c)
        else
            a, c, fc = c, d, fd
            d = a + g * (b - a)
            fd = f(d)
        end
    end
    x, v = fc < fd ? (c, fc) : (d, fd)
    x, v = v < vs[k] ? (x, v) : (xs[k], vs[k])
    at_end = (first_end || last_end) && abs(x - xs[k]) <= h * g^iters
    return (; x = float(x), value = v, at_end)
end
