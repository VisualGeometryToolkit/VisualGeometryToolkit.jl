# =============================================================================
# One-dimensional minimization: a grid scan, then Brent's method
# =============================================================================

"""
    grid_brent_minimum(f, xs::AbstractRange; periodic = false) -> (; x, value, at_end)

The minimum of `f` over the evenly spaced grid `xs` (evaluated in its order), the least
value refined by Brent's method (golden-section steps with parabolic interpolation;
Brent 1973, ch. 5) between its neighbours: `(; x, value, at_end)` of the smallest value
evaluated. Brent stops at its own floor: `x` within `√eps·(|x| + step(xs))` (a smooth
minimum is resolved no better than `√eps` relative), so the result needs no tolerance.
On a `periodic` grid the neighbours wrap (`f` periodic). On a linear one a least value
at an end of the grid is refined on its one neighbouring interval, and `at_end` says
the result lies at that end: Brent's final bracket still has the end as its edge (every
evaluation descended towards it). A value of `Inf` or `NaN` is never the minimum. The
stack's one 1-D minimizer: Core's azimuthal minimum (the view-angle limit, the
inscribed radius) and Estimation's GCV weight search (`gcv_select`).
"""
function grid_brent_minimum(f, xs::AbstractRange; periodic::Bool = false)
    vs = map(f, xs)
    k = argmin(map(v -> isnan(v) ? oftype(v, Inf) : v, vs))
    h = float(abs(step(xs)))
    xk = float(xs[k])
    first_end = !periodic && k == firstindex(xs)
    last_end = !periodic && k == lastindex(xs)
    # the bracket: both neighbours, or at an end the one neighbouring interval
    s = sign(step(xs)) * h
    a, b = minmax((first_end ? (xk, xk + s) : last_end ? (xk - s, xk) : (xk - h, xk + h))...)
    x, v, a, b = _brent(f, a, b, h)
    at_end = (first_end || last_end) && (a == xk || b == xk)
    return v < vs[k] ? (; x, value = v, at_end) : (; x = xk, value = vs[k], at_end = first_end || last_end)
end

# Brent's minimization of `f` on `[a, b]` (Brent 1973, `localmin`): golden-section steps
# with parabolic interpolation, the tolerance `√eps·|x| + √eps·h` (relative, and absolute
# at the scale `h` of the bracket). A non-finite parabola takes the golden step. Returns
# the least point, its value and the final bracket.
function _brent(f, a::Float64, b::Float64, h::Float64)
    ε = sqrt(eps(Float64))
    cg = (3 - sqrt(5)) / 2
    x = w = v = a + cg * (b - a)
    fx = fw = fv = f(x)
    d = e = 0.0
    while true
        xm = (a + b) / 2
        tol1 = ε * abs(x) + ε * h
        tol2 = 2tol1
        abs(x - xm) <= tol2 - (b - a) / 2 && break
        parabolic = false
        if abs(e) > tol1
            r = (x - w) * (fx - fv)
            q = (x - v) * (fx - fw)
            p = (x - v) * q - (x - w) * r
            q = 2 * (q - r)
            q > 0 && (p = -p)
            q = abs(q)
            etemp = e
            e = d
            if isfinite(p) && isfinite(q) && abs(p) < abs(q * etemp / 2) && p > q * (a - x) && p < q * (b - x)
                d = p / q
                u = x + d
                (u - a < tol2 || b - u < tol2) && (d = copysign(tol1, xm - x))
                parabolic = true
            end
        end
        if !parabolic
            e = x >= xm ? a - x : b - x
            d = cg * e
        end
        u = abs(d) >= tol1 ? x + d : x + copysign(tol1, d)
        fu = f(u)
        if fu <= fx
            u >= x ? (a = x) : (b = x)
            v, w, x = w, x, u
            fv, fw, fx = fw, fx, fu
        else
            u < x ? (a = u) : (b = u)
            if fu <= fw || w == x
                v, w = w, u
                fv, fw = fw, fu
            elseif fu <= fv || v == x || v == w
                v, fv = u, fu
            end
        end
    end
    return x, fx, a, b
end
