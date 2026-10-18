# =============================================================================
# Native ellipse-ellipse intersection area — line-by-line port of GTE
# =============================================================================
#
# Source of truth (upstream Geometric Tools Engine, github.com/davideberly/GeometricTools):
#   • GTE/Mathematics/IntrEllipse2Ellipse2.h — FIQuery<double,Ellipse2,Ellipse2>
#     (intersection points via quartic root classification)
#   • GTE/Mathematics/IntrAreaEllipse2Ellipse2.h — AreaEllipse2Ellipse2<double>
#     (area from the intersection configuration)
#   • GTE/Mathematics/RootsPolynomial.h — SolveQuadratic/SolveCubic/SolveQuartic;
#     the double instantiation deduces Rational = double, so the whole root
#     classification runs in plain double arithmetic (no BSNumber/exact types).
# The entry contract of the retired FFI wrapper is mirrored by
# `_native_ellipse_intersection_area`: axis0 = (cosθ, sinθ),
# axis1 = (−sinθ, cosθ), extent = (a, b). Parity vs the compiled GTE oracle
# (200k random pairs + structured tangency/containment suites, max abs diff
# 1.4e-14, zero disagreements > 1e-8) is re-runnable via
# test/gte_parity/ellipse_area_parity.jl, which rebuilds the oracle from an
# upstream GTE checkout.
#
# This is a faithful port: same operations in the same order, same tolerances
# (exact ==/</> tests on computed doubles), same case taxonomy — including the
# GTE quirk in CaseE4NotZero where the divisor==0, λ>0 branch overwrites the
# first of the two candidate points (only the +sqrt point survives).
#
# std::map/std::multimap semantics are reproduced with fixed-size sorted
# arrays: ascending keys, duplicate keys (operator< equivalence, i.e. ==)
# keep the first-inserted multiplicity; multimap ties keep insertion order.
# The arrays are immutable SVectors updated with Base.setindex, so the whole
# pipeline is allocation-free.
#
# All FMA-based helpers (gte::FMA, RobustSOP, RobustDOP) map to Julia `fma`,
# which is the same hardware fused multiply-add std::fma uses.

# C++ float semantics: std::sqrt returns NaN outside its domain (the NaN then
# flows through comparisons, all of which evaluate false), whereas Julia
# throws. This shim reproduces the C++ behavior exactly.
@inline _ea_nan_sqrt(x::Float64) = x < 0.0 ? NaN : sqrt(x)

# The wrapper builds the axes with std::cos/std::sin. Julia's pure-Julia trig
# occasionally rounds the last ulp differently from the C library's, and GTE's
# branch selection (`e3 != 0`, `e4 != 0`) is 1-ulp-sensitive for exact circles
# (a == b), where the ideal e3/e4 are 0 and the computed values are rounding
# noise. Use the same libm the compiled oracle links so the classification
# matches bit-for-bit; fall back to Julia trig where no system libm exists.
@static if Sys.islinux()
    @inline _ea_cos(x::Float64) = ccall((:cos, "libm.so.6"), Cdouble, (Cdouble,), x)
    @inline _ea_sin(x::Float64) = ccall((:sin, "libm.so.6"), Cdouble, (Cdouble,), x)
elseif Sys.isapple()
    @inline _ea_cos(x::Float64) = ccall((:cos, "libSystem.dylib"), Cdouble, (Cdouble,), x)
    @inline _ea_sin(x::Float64) = ccall((:sin, "libSystem.dylib"), Cdouble, (Cdouble,), x)
else
    @inline _ea_cos(x::Float64) = cos(x)
    @inline _ea_sin(x::Float64) = sin(x)
end

# GTE's AreaEllipse2Ellipse2 constants mPi/mTwoPi (double literals).
const _EA_PI = 3.14159265358979323846
const _EA_TWO_PI = 6.28318530717958647692

# --- Functions.h: FMA / RobustSOP / RobustDOP (double versions) --------------

# Robust sum of products u*v + w*z.
@inline function _ea_robust_sop(u::Float64, v::Float64, w::Float64, z::Float64)
    product_wz = w * z
    rounding_error = fma(w, z, -product_wz)
    return fma(u, v, product_wz) + rounding_error
end

# Robust difference of products u*v - w*z.
@inline function _ea_robust_dop(u::Float64, v::Float64, w::Float64, z::Float64)
    product_wz = w * z
    rounding_error = fma(-w, z, product_wz)
    return fma(u, v, -product_wz) + rounding_error
end

# =============================================================================
# RootsPolynomial.h — SolveQuadratic/SolveCubic/SolveQuartic (Rational=double)
# =============================================================================

# std::map<double,int32_t> with at most 4 entries: keys ascending, insert of
# an equivalent key (==) keeps the existing entry. Immutable, isbits.
struct _EaRootMap
    n::Int
    keys::SVector{4, Float64}
    mults::SVector{4, Int}
end

@inline _ea_empty_map() = _EaRootMap(0, SVector(0.0, 0.0, 0.0, 0.0), SVector(0, 0, 0, 0))

@inline function _ea_map_insert(m::_EaRootMap, key::Float64, mult::Int)
    n = m.n
    keys = m.keys
    mults = m.mults
    i = 1
    while i <= n
        k = keys[i]
        if key == k
            # std::map::insert: equivalent key already present, no overwrite.
            return m
        elseif key < k
            break
        end
        i += 1
    end
    j = n
    while j >= i
        keys = Base.setindex(keys, keys[j], j + 1)
        mults = Base.setindex(mults, mults[j], j + 1)
        j -= 1
    end
    keys = Base.setindex(keys, key, i)
    mults = Base.setindex(mults, mult, i)
    return _EaRootMap(n + 1, keys, mults)
end

# rmMap.find(zero): ++multiplicity if a zero root exists, else insert (0, 1).
@inline function _ea_map_bump_zero(m::_EaRootMap)
    for i in 1:m.n
        if m.keys[i] == 0.0
            return _EaRootMap(m.n, m.keys, Base.setindex(m.mults, m.mults[i] + 1, i))
        end
    end
    return _ea_map_insert(m, 0.0, 1)
end

# --- SolveDepressedQuadratic --------------------------------------------------
@inline function _ea_solve_depressed_quadratic(m::_EaRootMap, c0::Float64)
    if c0 < 0.0
        # Two simple roots.
        root1 = sqrt(-c0)
        root0 = -root1
        m = _ea_map_insert(m, root0, 1)
        m = _ea_map_insert(m, root1, 1)
    elseif c0 == 0.0
        # One double root.
        m = _ea_map_insert(m, 0.0, 2)
    end
    # else c0 > 0: complex-conjugate pair, no real roots.
    return m
end

# --- SolveDepressedCubic -------------------------------------------------------
@inline function _ea_solve_depressed_cubic(m::_EaRootMap, c0::Float64, c1::Float64)
    # Special case c0 = 0: reduces to a depressed quadratic.
    if c0 == 0.0
        m = _ea_solve_depressed_quadratic(m, c1)
        m = _ea_map_bump_zero(m)
        return m
    end

    one_third = 1.0 / 3.0
    if c1 == 0.0
        # One simple real root.
        root0 = c0 > 0.0 ? -(c0^one_third) : (-c0)^one_third
        return _ea_map_insert(m, root0, 1)
    end

    # c0 != 0 and c1 != 0.
    delta = -(4.0 * c1 * c1 * c1 + 27.0 * c0 * c0)
    if delta > 0.0
        # Three simple roots.
        delta_div_108 = delta / 108.0
        beta_re = -c0 / 2.0
        beta_im = sqrt(delta_div_108)
        theta = atan(beta_im, beta_re)
        theta_div_3 = theta / 3.0
        cs = cos(theta_div_3)
        sn = sin(theta_div_3)
        rho_sqr = beta_re * beta_re + beta_im * beta_im
        rho_pow_third = rho_sqr^(1.0 / 6.0)
        temp0 = rho_pow_third * cs
        temp1 = rho_pow_third * sn * sqrt(3.0)
        m = _ea_map_insert(m, 2.0 * temp0, 1)
        m = _ea_map_insert(m, -temp0 - temp1, 1)
        m = _ea_map_insert(m, -temp0 + temp1, 1)
    elseif delta < 0.0
        # One simple root.
        delta_div_108 = delta / 108.0
        temp0 = -c0 / 2.0
        temp1 = sqrt(-delta_div_108)
        temp2 = temp0 - temp1
        temp3 = temp0 + temp1
        temp2 = temp2 >= 0.0 ? temp2^one_third : -((-temp2)^one_third)
        temp3 = temp3 >= 0.0 ? temp3^one_third : -((-temp3)^one_third)
        m = _ea_map_insert(m, temp2 + temp3, 1)
    else  # delta = 0
        # One simple root and one double root.
        root0 = -3.0 * c0 / (2.0 * c1)
        root1 = -2.0 * root0
        m = _ea_map_insert(m, root0, 2)
        m = _ea_map_insert(m, root1, 1)
    end
    return m
end

# --- SolveBiquadratic ----------------------------------------------------------
@inline function _ea_solve_biquadratic(m::_EaRootMap, c0::Float64, c2::Float64)
    # Solve x^4 + c2*x^2 + c0 = 0 with c0 != 0 (so x = 0 is not a root).
    u = c2 / -2.0
    v = u * u - c0
    if v > 0.0
        sqrtv = sqrt(v)
        upsqrtv = u + sqrtv
        # u - sqrt(v) = c0 / (u + sqrt(v)) avoids subtractive cancellation.
        umsqrtv = c0 / upsqrtv
        if umsqrtv > 0.0
            # Four real roots: ±sqrt(u−sqrt(v)), ±sqrt(u+sqrt(v)).
            r0 = sqrt(umsqrtv)
            r1 = sqrt(upsqrtv)
            m = _ea_map_insert(m, r0, 1)
            m = _ea_map_insert(m, -r0, 1)
            m = _ea_map_insert(m, r1, 1)
            m = _ea_map_insert(m, -r1, 1)
        elseif upsqrtv < 0.0
            # Two complex-conjugate pairs, no real roots.
        else  # umsqrtv < 0 and upsqrtv > 0
            # Two real roots ±sqrt(u+sqrt(v)).
            r0 = sqrt(upsqrtv)
            m = _ea_map_insert(m, r0, 1)
            m = _ea_map_insert(m, -r0, 1)
        end
    elseif v < 0.0
        # Two complex-conjugate pairs, no real roots.
    else  # v = 0
        if u > 0.0
            # Two real roots, each of multiplicity 2.
            r0 = sqrt(u)
            m = _ea_map_insert(m, r0, 2)
            m = _ea_map_insert(m, -r0, 2)
        end
        # else u < 0: double complex-conjugate pairs.
    end
    return m
end

# --- SolveCubic (used by SolveDepressedQuartic); only the largest root is
# consumed (rmCubicMap.rbegin()->first).
@inline function _ea_cubic_max_root(p0::Float64, p1::Float64, p2::Float64, p3::Float64)
    q0 = p0 / p3
    q1 = p1 / p3
    q2 = p2 / p3
    q2third = q2 / 3.0
    c0 = q0 - q2third * (q1 - 2.0 * q2third * q2third)
    c1 = q1 - q2 * q2third
    m = _ea_solve_depressed_cubic(_ea_empty_map(), c0, c1)
    # Output-map keys are (local root − q2third); the shift is monotone, so
    # the largest output key comes from the largest local key.
    return m.keys[m.n] - q2third
end

# --- SolveDepressedQuartic -----------------------------------------------------
@inline function _ea_solve_depressed_quartic(
        m::_EaRootMap, c0::Float64, c1::Float64,
        c2::Float64
    )
    # Special case c0 = 0: reduces to a depressed cubic.
    if c0 == 0.0
        m = _ea_solve_depressed_cubic(m, c1, c2)
        m = _ea_map_bump_zero(m)
        return m
    end

    # Special case c1 = 0: biquadratic.
    if c1 == 0.0
        return _ea_solve_biquadratic(m, c0, c2)
    end

    # c0 != 0 and c1 != 0.
    c0sqr = c0 * c0
    c1sqr = c1 * c1
    c2sqr = c2 * c2
    delta = c1sqr * (-27.0 * c1sqr + 4.0 * c2 * (36.0 * c0 - c2sqr)) +
        16.0 * c0 * (c2sqr * (c2sqr - 8.0 * c0) + 16.0 * c0sqr)
    a0 = 12.0 * c0 + c2sqr
    a1 = 4.0 * c0 - c2sqr

    if delta > 0.0
        if c2 < 0.0 && a1 < 0.0
            # Four simple real roots.
            t = _ea_cubic_max_root(c1sqr - 4.0 * c0 * c2, 8.0 * c0, 4.0 * c2, -8.0)
            alpha_sqr = 2.0 * t - c2
            alpha = sqrt(max(alpha_sqr, 0.0))
            sgn_c1 = c1 > 0.0 ? 1.0 : -1.0
            arg = t * t - c0
            beta = sgn_c1 * sqrt(max(arg, 0.0))
            dd0 = alpha_sqr - 4.0 * (t + beta)
            sqrt_dd0 = sqrt(max(dd0, 0.0))
            dd1 = alpha_sqr - 4.0 * (t - beta)
            sqrt_dd1 = sqrt(max(dd1, 0.0))
            m = _ea_map_insert(m, (alpha - sqrt_dd0) / 2.0, 1)
            m = _ea_map_insert(m, (alpha + sqrt_dd0) / 2.0, 1)
            m = _ea_map_insert(m, (-alpha - sqrt_dd1) / 2.0, 1)
            m = _ea_map_insert(m, (-alpha + sqrt_dd1) / 2.0, 1)
        end
        # else c2 >= 0 or a1 >= 0: two complex-conjugate pairs, no real roots.
    elseif delta < 0.0
        # Two simple real roots, one complex-conjugate pair.
        t = _ea_cubic_max_root(c1sqr - 4.0 * c0 * c2, 8.0 * c0, 4.0 * c2, -8.0)
        alpha_sqr = 2.0 * t - c2
        alpha = sqrt(max(alpha_sqr, 0.0))
        sgn_c1 = c1 > 0.0 ? 1.0 : -1.0
        arg = t * t - c0
        beta = sgn_c1 * sqrt(max(arg, 0.0))
        if sgn_c1 > 0.0
            dd1 = alpha_sqr - 4.0 * (t - beta)
            sqrt_dd1 = sqrt(max(dd1, 0.0))
            root0 = (-alpha - sqrt_dd1) / 2.0
            root1 = (-alpha + sqrt_dd1) / 2.0
            m = _ea_map_insert(m, root0, 1)
            m = _ea_map_insert(m, root1, 1)
        else
            dd0 = alpha_sqr - 4.0 * (t + beta)
            sqrt_dd0 = sqrt(max(dd0, 0.0))
            root0 = (alpha - sqrt_dd0) / 2.0
            root1 = (alpha + sqrt_dd0) / 2.0
            m = _ea_map_insert(m, root0, 1)
            m = _ea_map_insert(m, root1, 1)
        end
    else  # delta = 0
        if a1 > 0.0 || (c2 > 0.0 && (a1 != 0.0 || c1 != 0.0))
            # One double real root, one complex-conjugate pair.
            root0 = -c1 * a0 / (9.0 * c1sqr - 2.0 * c2 * a1)
            m = _ea_map_insert(m, root0, 2)
        else
            if a0 != 0.0
                # One double real root, two simple real roots.
                root0 = -c1 * a0 / (9.0 * c1sqr - 2.0 * c2 * a1)
                alpha = 2.0 * root0
                beta = c2 + 3.0 * root0 * root0
                discr = alpha * alpha - 4.0 * beta
                temp1 = sqrt(max(discr, 0.0))
                root1 = (-alpha - temp1) / 2.0
                root2 = (-alpha + temp1) / 2.0
                m = _ea_map_insert(m, root0, 2)
                m = _ea_map_insert(m, root1, 1)
                m = _ea_map_insert(m, root2, 1)
            else
                # One triple real root, one simple real root.
                root0 = -3.0 * c1 / (4.0 * c2)
                root1 = -3.0 * root0
                m = _ea_map_insert(m, root0, 3)
                m = _ea_map_insert(m, root1, 1)
            end
        end
    end
    return m
end

# Shift the depressed-solver map into the output map: keys become
# (root − shift), ascending order is preserved (the shift is monotone), and
# std::map::insert dedupe keeps the first (smallest-original) entry when
# rounding collapses two shifted keys.
@inline function _ea_shift_map(m::_EaRootMap, shift::Float64)
    okeys = SVector(0.0, 0.0, 0.0, 0.0)
    omults = SVector(0, 0, 0, 0)
    on = 0
    for i in 1:m.n
        root = m.keys[i] - shift
        if on > 0 && okeys[on] == root
            continue
        end
        on += 1
        okeys = Base.setindex(okeys, root, on)
        omults = Base.setindex(omults, m.mults[i], on)
    end
    return _EaRootMap(on, okeys, omults)
end

# --- SolveQuadratic ------------------------------------------------------------
@inline function _ea_solve_quadratic(p0::Float64, p1::Float64, p2::Float64)
    q0 = p0 / p2
    q1 = p1 / p2
    q1half = q1 / 2.0
    c0 = q0 - q1half * q1half
    m = _ea_solve_depressed_quadratic(_ea_empty_map(), c0)
    return _ea_shift_map(m, q1half)
end

# --- SolveQuartic ----------------------------------------------------------------
@inline function _ea_solve_quartic(
        p0::Float64, p1::Float64, p2::Float64, p3::Float64,
        p4::Float64
    )
    q0 = p0 / p4
    q1 = p1 / p4
    q2 = p2 / p4
    q3 = p3 / p4
    q3fourth = q3 / 4.0
    q3fourth_sqr = q3fourth * q3fourth
    c0 = q0 - q3fourth * (q1 - q3fourth * (q2 - q3fourth_sqr * 3.0))
    c1 = q1 - 2.0 * q3fourth * (q2 - 4.0 * q3fourth_sqr)
    c2 = q2 - 6.0 * q3fourth_sqr
    m = _ea_solve_depressed_quartic(_ea_empty_map(), c0, c1, c2)
    return _ea_shift_map(m, q3fourth)
end

# =============================================================================
# IntrEllipse2Ellipse2.h — FIQuery<double, Ellipse2, Ellipse2>
# =============================================================================

# FIQuery::Result. `numpoints == typemax(Int)` is the "ellipses are equal"
# sentinel (std::numeric_limits<size_t>::max() in the C++).
struct _EaFindResult
    intersect::Bool
    numpoints::Int
    xs::SVector{4, Float64}
    ys::SVector{4, Float64}
    transverse::SVector{4, Bool}
end

@inline _ea_empty_result(numpoints::Int) = _EaFindResult(
    false, numpoints,
    SVector(0.0, 0.0, 0.0, 0.0), SVector(0.0, 0.0, 0.0, 0.0),
    SVector(false, false, false, false)
)

# --- FIQuery::GetStandardForm ----------------------------------------------------
# Returns (Cx, Cy, M00, M01, M11); M is symmetric so M10 == M01 exactly.
@inline function _ea_standard_form(
        cx::Float64, cy::Float64,
        u0::Float64, u1::Float64, v0::Float64, v1::Float64,
        ea::Float64, eb::Float64
    )
    uu00 = u0 * u0
    uu01 = u0 * u1
    uu11 = u1 * u1
    vv00 = v0 * v0
    vv01 = v0 * v1
    vv11 = v1 * v1
    usqrlen = uu00 + uu11  # Trace(UUTrn)
    asqr = ea * ea
    bsqr = eb * eb
    # Matrix operator/= divides by multiplying with the reciprocal.
    inva = 1.0 / asqr
    invb = 1.0 / bsqr
    invu = 1.0 / usqrlen
    m00 = (uu00 * inva + vv00 * invb) * invu
    m01 = (uu01 * inva + vv01 * invb) * invu
    m11 = (uu11 * inva + vv11 * invb) * invu
    return cx, cy, m00, m01, m11
end

# --- Case functions ---------------------------------------------------------------

@inline function _ea_case_e4_zero_e2_zero_e3_not_zero(
        c0x::Float64, c0y::Float64,
        ell::Float64, d0::Float64, d1::Float64, e0::Float64, e1::Float64, e3::Float64
    )
    xs = SVector(0.0, 0.0, 0.0, 0.0)
    ys = SVector(0.0, 0.0, 0.0, 0.0)
    tr = SVector(false, false, false, false)
    np = 0
    isect = false
    rm = _ea_solve_quadratic(e0, e1, e3)
    for i in 1:rm.n
        y0 = rm.keys[i]
        lambda = fma(-d0, y0 * y0, 1.0)
        lambda < 0.0 && continue
        if lambda > 0.0
            y1 = -_ea_nan_sqrt(lambda / d1)
            np += 1
            xs = Base.setindex(xs, fma(-ell, y1, y0) + c0x, np)
            ys = Base.setindex(ys, y1 + c0y, np)
            tr = Base.setindex(tr, rm.mults[i] == 1, np)
            y1 = -y1
            np += 1
            xs = Base.setindex(xs, fma(-ell, y1, y0) + c0x, np)
            ys = Base.setindex(ys, y1 + c0y, np)
            tr = Base.setindex(tr, rm.mults[i] == 1, np)
        else
            np += 1
            xs = Base.setindex(xs, y0 + c0x, np)
            ys = Base.setindex(ys, c0y, np)
            tr = Base.setindex(tr, false, np)
        end
        isect = true
    end
    return _EaFindResult(isect, np, xs, ys, tr)
end

@inline function _ea_case_e4_zero_e2_zero_e3_zero(
        c0x::Float64, c0y::Float64,
        ell::Float64, d0::Float64, d1::Float64, e0::Float64, e1::Float64
    )
    xs = SVector(0.0, 0.0, 0.0, 0.0)
    ys = SVector(0.0, 0.0, 0.0, 0.0)
    tr = SVector(false, false, false, false)
    np = 0
    y0 = -e0 / e1
    lambda = fma(-d0, y0 * y0, 1.0)
    if lambda < 0.0
        return _EaFindResult(false, np, xs, ys, tr)
    end
    if lambda > 0.0
        y1 = -_ea_nan_sqrt(lambda / d1)
        np += 1
        xs = Base.setindex(xs, fma(-ell, y1, y0) + c0x, np)
        ys = Base.setindex(ys, y1 + c0y, np)
        tr = Base.setindex(tr, true, np)
        y1 = -y1
        np += 1
        xs = Base.setindex(xs, fma(-ell, y1, y0) + c0x, np)
        ys = Base.setindex(ys, y1 + c0y, np)
        tr = Base.setindex(tr, true, np)
    else
        np += 1
        xs = Base.setindex(xs, y0 + c0x, np)
        ys = Base.setindex(ys, c0y, np)
        tr = Base.setindex(tr, false, np)
    end
    return _EaFindResult(true, np, xs, ys, tr)
end

@inline function _ea_case_e4_zero_e2_not_zero_e3_zero(
        c0x::Float64, c0y::Float64,
        ell::Float64, d0::Float64, d1::Float64, e0::Float64, e1::Float64, e2::Float64
    )
    xs = SVector(0.0, 0.0, 0.0, 0.0)
    ys = SVector(0.0, 0.0, 0.0, 0.0)
    tr = SVector(false, false, false, false)
    np = 0
    isect = false
    # H = e2^2 * {-1, 0, d0} + (d1 * {e0, e1}) * {e0, e1}  (Polynomial1 ops).
    e2sqr = e2 * e2
    d1e0 = d1 * e0
    d1e1 = d1 * e1
    h0 = e2sqr * -1.0 + d1e0 * e0
    h1 = e2sqr * 0.0 + (d1e0 * e1 + d1e1 * e0)
    h2 = e2sqr * d0 + d1e1 * e1
    rm = _ea_solve_quadratic(h0, h1, h2)
    for i in 1:rm.n
        y0 = rm.keys[i]
        lambda = fma(-d0, y0 * y0, 1.0)
        lambda < 0.0 && continue
        if lambda > 0.0
            # Choose the y1-root with smallest |(e0 + e1*y0) + e2*y1|.
            y1cand0 = -_ea_nan_sqrt(lambda / d1)
            test0 = abs(e0 + _ea_robust_sop(e1, y0, e2, y1cand0))
            y1cand1 = -y1cand0
            test1 = abs(e0 + _ea_robust_sop(e1, y0, e2, y1cand1))
            y1 = test0 <= test1 ? y1cand0 : y1cand1
            np += 1
            xs = Base.setindex(xs, fma(-ell, y1, y0) + c0x, np)
            ys = Base.setindex(ys, y1 + c0y, np)
        else
            np += 1
            xs = Base.setindex(xs, y0 + c0x, np)
            ys = Base.setindex(ys, c0y, np)
        end
        tr = Base.setindex(tr, rm.mults[i] == 1, np)
        isect = true
    end
    return _EaFindResult(isect, np, xs, ys, tr)
end

@inline function _ea_case_e4_zero_e2_not_zero_e3_not_zero(
        c0x::Float64, c0y::Float64,
        ell::Float64, d0::Float64, d1::Float64, e0::Float64, e1::Float64, e2::Float64,
        e3::Float64
    )
    xs = SVector(0.0, 0.0, 0.0, 0.0)
    ys = SVector(0.0, 0.0, 0.0, 0.0)
    tr = SVector(false, false, false, false)
    np = 0
    isect = false
    # H = e2^2 * {-1, 0, d0} + (d1 * {e0, e1, e3}) * {e0, e1, e3}.
    e2sqr = e2 * e2
    d1e0 = d1 * e0
    d1e1 = d1 * e1
    d1e3 = d1 * e3
    h0 = e2sqr * -1.0 + d1e0 * e0
    h1 = e2sqr * 0.0 + (d1e0 * e1 + d1e1 * e0)
    h2 = e2sqr * d0 + ((d1e0 * e3 + d1e1 * e1) + d1e3 * e0)
    h3 = d1e1 * e3 + d1e3 * e1
    h4 = d1e3 * e3
    rm = _ea_solve_quartic(h0, h1, h2, h3, h4)
    for i in 1:rm.n
        y0 = rm.keys[i]
        lambda = fma(-d0, y0 * y0, 1.0)
        lambda < 0.0 && continue
        if lambda > 0.0
            # Choose the y1-root with smallest
            # |(e0 + e1*y0 + e3*y0^2) + e2*y1|.
            term0 = fma(e3, y0, e1)
            term1 = fma(term0, y0, e0)
            y1cand0 = -_ea_nan_sqrt(lambda / d1)
            test0 = abs(fma(e2, y1cand0, term1))
            y1cand1 = -y1cand0
            test1 = abs(fma(e2, y1cand1, term1))
            y1 = test0 < test1 ? y1cand0 : y1cand1
            np += 1
            xs = Base.setindex(xs, fma(-ell, y1, y0) + c0x, np)
            ys = Base.setindex(ys, y1 + c0y, np)
        else
            np += 1
            xs = Base.setindex(xs, y0 + c0x, np)
            ys = Base.setindex(ys, c0y, np)
        end
        tr = Base.setindex(tr, rm.mults[i] == 1, np)
        isect = true
    end
    return _EaFindResult(isect, np, xs, ys, tr)
end

@inline function _ea_case_e4_not_zero(
        c0x::Float64, c0y::Float64,
        ell::Float64, d0::Float64, d1::Float64, e0::Float64, e1::Float64, e2::Float64,
        e3::Float64, e4::Float64
    )
    xs = SVector(0.0, 0.0, 0.0, 0.0)
    ys = SVector(0.0, 0.0, 0.0, 0.0)
    tr = SVector(false, false, false, false)
    np = 0
    isect = false
    # H = ({e2,e4} * {e2,e4}) * {-1, 0, d0} + (d1 * {e0,e1,e3}) * {e0,e1,e3}.
    b0 = e2 * e2
    b1 = e2 * e4 + e4 * e2
    b2 = e4 * e4
    cc0 = b0 * -1.0
    cc1 = b0 * 0.0 + b1 * -1.0
    cc2 = (b0 * d0 + b1 * 0.0) + b2 * -1.0
    cc3 = b1 * d0 + b2 * 0.0
    cc4 = b2 * d0
    d1e0 = d1 * e0
    d1e1 = d1 * e1
    d1e3 = d1 * e3
    h0 = cc0 + d1e0 * e0
    h1 = cc1 + (d1e0 * e1 + d1e1 * e0)
    h2 = cc2 + ((d1e0 * e3 + d1e1 * e1) + d1e3 * e0)
    h3 = cc3 + (d1e1 * e3 + d1e3 * e1)
    h4 = cc4 + d1e3 * e3
    rm = _ea_solve_quartic(h0, h1, h2, h3, h4)
    for i in 1:rm.n
        y0 = rm.keys[i]
        lambda = fma(-d0, y0 * y0, 1.0)
        lambda < 0.0 && continue
        divisor = e2 + e4 * y0
        if divisor != 0.0
            if lambda > 0.0
                # Choose the y1-root with smallest
                # |(e0 + e1*y0 + e3*y0^2) + (e2 + e4*y0)*y1|.
                y1cand0 = -_ea_nan_sqrt(lambda / d1)
                term0 = fma(e3, y0, e1)
                term1 = fma(term0, y0, e0)
                test0 = abs(fma(divisor, y1cand0, term1))
                y1cand1 = -y1cand0
                test1 = abs(fma(divisor, y1cand1, term1))
                y1 = test0 < test1 ? y1cand0 : y1cand1
                np += 1
                xs = Base.setindex(xs, fma(-ell, y1, y0) + c0x, np)
                ys = Base.setindex(ys, y1 + c0y, np)
            else
                np += 1
                xs = Base.setindex(xs, y0 + c0x, np)
                ys = Base.setindex(ys, c0y, np)
            end
            tr = Base.setindex(tr, rm.mults[i] == 1, np)
            isect = true
        else
            # GTE writes two candidate points into the SAME slot when
            # divisor == 0 and lambda > 0 (no increment between the writes),
            # so only the second (+sqrt) point survives. Port that behavior.
            if lambda > 0.0
                y1 = -_ea_nan_sqrt(lambda / d1)
                y1 = -y1
                np += 1
                xs = Base.setindex(xs, fma(-ell, y1, y0) + c0x, np)
                ys = Base.setindex(ys, y1 + c0y, np)
            else
                np += 1
                xs = Base.setindex(xs, y0 + c0x, np)
                ys = Base.setindex(ys, c0y, np)
            end
            tr = Base.setindex(tr, lambda > 0.0, np)
            isect = true
        end
    end
    return _EaFindResult(isect, np, xs, ys, tr)
end

# --- FIQuery::operator()(C0, M0, C1, M1, true) -------------------------------------
function _ea_find_intersections(
        c0x::Float64, c0y::Float64,
        m0_00::Float64, m0_01::Float64, m0_11::Float64,
        c1x::Float64, c1y::Float64,
        m1_00::Float64, m1_01::Float64, m1_11::Float64
    )
    # Same ellipse: infinitely many intersection points.
    if c0x == c1x && c0y == c1y && m0_00 == m1_00 && m0_01 == m1_01 && m0_11 == m1_11
        return _ea_empty_result(typemax(Int))
    end

    # Early exit when the axis-aligned bounding boxes are disjoint
    # (useEarlyExitNoIntersectionTest = true, the wrapper default).
    det0 = m0_00 * m0_11 - m0_01 * m0_01
    w0x = _ea_nan_sqrt(m0_11 / det0)
    w0y = _ea_nan_sqrt(m0_00 / det0)
    det1 = m1_00 * m1_11 - m1_01 * m1_01
    w1x = _ea_nan_sqrt(m1_11 / det1)
    w1y = _ea_nan_sqrt(m1_00 / det1)
    if (c0x + w0x < c1x - w1x) || (c0x - w0x > c1x + w1x) ||
            (c0y + w0y < c1y - w1y) || (c0y - w0y > c1y + w1y)
        return _ea_empty_result(0)
    end

    ell = m0_01 / m0_00
    d0 = m0_00
    d1 = _ea_robust_dop(m0_00, m0_11, m0_01, m0_01) / m0_00
    k0 = c1x - c0x
    k1 = c1y - c0y
    term0 = _ea_robust_sop(k0, m1_00, k1, m1_01)
    term1 = _ea_robust_sop(k0, m1_01, k1, m1_11)
    g0 = _ea_robust_sop(k0, term0, k1, term1) - 1.0
    g1 = -2.0 * term0
    g2 = 2.0 * fma(term0, ell, -term1)
    g3 = m1_00
    g4 = -2.0 * fma(m1_00, ell, -m1_01)
    g5 = fma(-ell, _ea_robust_dop(2.0, m1_01, ell, m1_00), m1_11)
    e0 = fma(d1, g0, g5)
    e1 = d1 * g1
    e2 = d1 * g2
    e3 = _ea_robust_dop(d1, g3, d0, g5)
    e4 = d1 * g4

    if e4 != 0.0
        return _ea_case_e4_not_zero(c0x, c0y, ell, d0, d1, e0, e1, e2, e3, e4)
    else
        if e2 != 0.0
            if e3 != 0.0
                return _ea_case_e4_zero_e2_not_zero_e3_not_zero(
                    c0x, c0y, ell, d0, d1,
                    e0, e1, e2, e3
                )
            else
                return _ea_case_e4_zero_e2_not_zero_e3_zero(
                    c0x, c0y, ell, d0, d1,
                    e0, e1, e2
                )
            end
        else
            if e3 != 0.0
                return _ea_case_e4_zero_e2_zero_e3_not_zero(
                    c0x, c0y, ell, d0, d1,
                    e0, e1, e3
                )
            elseif e1 != 0.0
                return _ea_case_e4_zero_e2_zero_e3_zero(c0x, c0y, ell, d0, d1, e0, e1)
            end
            # else: axis-aligned ellipses with the same center; one strictly
            # inside the other, no intersection.
            return _ea_empty_result(0)
        end
    end
end

# =============================================================================
# IntrAreaEllipse2Ellipse2.h — AreaEllipse2Ellipse2<double>
# =============================================================================

# EllipseInfo (center, axes, and the derived quantities of FinishEllipseInfo).
struct _EaEllipseInfo
    cx::Float64
    cy::Float64
    a0x::Float64  # axis[0]
    a0y::Float64
    a1x::Float64  # axis[1]
    a1y::Float64
    m00::Float64  # E.M (symmetric; m10 == m01 exactly)
    m01::Float64
    m11::Float64
    ab::Float64        # extent[0] * extent[1]
    half_ab::Float64   # AB / 2
    bpa::Float64       # extent[1] + extent[0]
    bma::Float64       # extent[1] - extent[0]
end

@inline function _ea_ellipse_info(
        cx::Float64, cy::Float64,
        u0::Float64, u1::Float64, v0::Float64, v1::Float64,
        ea::Float64, eb::Float64
    )
    # E.M = outer(U,U)/(a^2*Dot(U,U)) + outer(V,V)/(b^2*Dot(V,V)); the matrix
    # scalar division multiplies by the reciprocal.
    dotu = u0 * u0 + u1 * u1
    dotv = v0 * v0 + v1 * v1
    inv0 = 1.0 / (ea * ea * dotu)
    inv1 = 1.0 / (eb * eb * dotv)
    m00 = u0 * u0 * inv0 + v0 * v0 * inv1
    m01 = u0 * u1 * inv0 + v0 * v1 * inv1
    m11 = u1 * u1 * inv0 + v1 * v1 * inv1
    ab = ea * eb
    return _EaEllipseInfo(
        cx, cy, u0, u1, v0, v1, m00, m01, m11,
        ab, ab / 2.0, eb + ea, eb - ea
    )
end

# --- AreaEllipse2Ellipse2::ComputeIntegral -----------------------------------------
@inline function _ea_integral(E::_EaEllipseInfo, theta::Float64)
    two_theta = 2.0 * theta
    sn = sin(two_theta)
    cs = cos(two_theta)
    arg = E.bma * sn / (E.bpa + E.bma * cs)
    return E.half_ab * (theta - atan(arg))
end

# --- AreaEllipse2Ellipse2::ComputeAreaChordRegion ----------------------------------
# P0mC/P1mC are the chord endpoints relative to the ellipse center.
function _ea_chord_region(
        E::_EaEllipseInfo, p0mcx::Float64, p0mcy::Float64,
        p1mcx::Float64, p1mcy::Float64
    )
    # Polar coordinates of P0 and P1 on the ellipse.
    x0 = E.a0x * p0mcx + E.a0y * p0mcy
    y0 = E.a1x * p0mcx + E.a1y * p0mcy
    theta0 = atan(y0, x0)
    x1 = E.a0x * p1mcx + E.a0y * p1mcy
    y1 = E.a1x * p1mcx + E.a1y * p1mcy
    theta1 = atan(y1, x1)

    # The arc straddles the atan2 discontinuity on the negative x-axis; wrap
    # the second angle to be larger than the first.
    if theta1 < theta0
        theta1 += _EA_TWO_PI
    end

    # Area portion of the sector due to the triangle.
    tri_area = abs(p0mcx * p1mcy + p0mcy * -p1mcx) / 2.0  # |DotPerp(P0mC, P1mC)|/2

    dtheta = theta1 - theta0
    if dtheta <= _EA_PI
        # area(theta0,theta1) = F(theta1) - F(theta0) - area(triangle)
        f0 = _ea_integral(E, theta0)
        f1 = _ea_integral(E, theta1)
        sector_area = f1 - f0
        return sector_area - tri_area
    else
        # Sector angle exceeds pi: area = pi*a*b - area(theta1,theta0).
        theta0 += _EA_TWO_PI  # ensure theta0 > theta1
        f0 = _ea_integral(E, theta0)
        f1 = _ea_integral(E, theta1)
        sector_area = f0 - f1
        return _EA_PI * E.ab - (sector_area - tri_area)
    end
end

# --- AreaEllipse2Ellipse2::AreaCS ---------------------------------------------------
function _ea_area_cs(E0::_EaEllipseInfo, E1::_EaEllipseInfo, fr::_EaFindResult)
    if fr.numpoints <= 1
        dx = E0.cx - E1.cx
        dy = E0.cy - E1.cy
        w0x = E0.m00 * dx + E0.m01 * dy
        w0y = E0.m01 * dx + E0.m11 * dy
        qform0 = dx * w0x + dy * w0y
        w1x = E1.m00 * dx + E1.m01 * dy
        w1y = E1.m01 * dx + E1.m11 * dy
        qform1 = dx * w1x + dy * w1y
        if qform0 > 1.0 && qform1 > 1.0
            # Separated, or outside each other and just touching.
            return 0.0
        else
            # One ellipse inside the other; compare areas to pick which.
            return E0.ab < E1.ab ? _EA_PI * E0.ab : _EA_PI * E1.ab
        end
    else
        # Ellipses are equal.
        return _EA_PI * E0.ab
    end
end

# --- AreaEllipse2Ellipse2::Area2 (indices are 1-based here) -------------------------
function _ea_area2(
        E0::_EaEllipseInfo, E1::_EaEllipseInfo, fr::_EaFindResult,
        i0::Int, i1::Int
    )
    p0x = fr.xs[i0]
    p0y = fr.ys[i0]
    p1x = fr.xs[i1]
    p1y = fr.ys[i1]

    p0mc0x = p0x - E0.cx
    p0mc0y = p0y - E0.cy
    p0mc1x = p0x - E1.cx
    p0mc1y = p0y - E1.cy
    p1mc0x = p1x - E0.cx
    p1mc0y = p1y - E0.cy
    p1mc1x = p1x - E1.cx
    p1mc1y = p1y - E1.cy

    # Ellipse normals at P0 determine chord endpoint order.
    n0x = E0.m00 * p0mc0x + E0.m01 * p0mc0y
    n0y = E0.m01 * p0mc0x + E0.m11 * p0mc0y
    n1x = E1.m00 * p0mc1x + E1.m01 * p0mc1y
    n1y = E1.m01 * p0mc1x + E1.m11 * p0mc1y
    dotperp = n1x * n0y + n1y * -n0x  # DotPerp(N1, N0)

    if dotperp > 0.0
        # Chord order for E0 is <P0,P1>, for E1 is <P1,P0>.
        return _ea_chord_region(E0, p0mc0x, p0mc0y, p1mc0x, p1mc0y) +
            _ea_chord_region(E1, p1mc1x, p1mc1y, p0mc1x, p0mc1y)
    else
        # Chord order for E0 is <P1,P0>, for E1 is <P0,P1>.
        return _ea_chord_region(E0, p1mc0x, p1mc0y, p0mc0x, p0mc0y) +
            _ea_chord_region(E1, p0mc1x, p0mc1y, p1mc1x, p1mc1y)
    end
end

# --- AreaEllipse2Ellipse2::Area4 ----------------------------------------------------
function _ea_area4(E0::_EaEllipseInfo, E1::_EaEllipseInfo, fr::_EaFindResult)
    # Counterclockwise ordering via polar coordinates for E0 (std::multimap:
    # ascending keys, ties keep insertion order — a stable insertion sort).
    th = SVector(0.0, 0.0, 0.0, 0.0)
    for i in 1:4
        pmcx = fr.xs[i] - E0.cx
        pmcy = fr.ys[i] - E0.cy
        x = E0.a0x * pmcx + E0.a0y * pmcy
        y = E0.a1x * pmcx + E0.a1y * pmcy
        th = Base.setindex(th, atan(y, x), i)
    end
    permute = SVector(1, 2, 3, 4)
    for i in 2:4
        j = i
        while j > 1 && th[permute[j]] < th[permute[j - 1]]
            pj = permute[j]
            permute = Base.setindex(permute, permute[j - 1], j)
            permute = Base.setindex(permute, pj, j - 1)
            j -= 1
        end
    end

    # Area of the convex quadrilateral.
    d20x = fr.xs[permute[3]] - fr.xs[permute[1]]
    d20y = fr.ys[permute[3]] - fr.ys[permute[1]]
    d31x = fr.xs[permute[4]] - fr.xs[permute[2]]
    d31y = fr.ys[permute[4]] - fr.ys[permute[2]]
    area = abs(d20x * d31y + d20y * -d31x) / 2.0  # |DotPerp(diag20, diag31)|/2

    # Visit consecutive point pairs; pick the ellipse whose tangent is "most
    # counterclockwise" for each chord region.
    i0 = 4
    for i1 in 1:4
        p0x = fr.xs[permute[i0]]
        p0y = fr.ys[permute[i0]]
        p1x = fr.xs[permute[i1]]
        p1y = fr.ys[permute[i1]]

        p0mc0x = p0x - E0.cx
        p0mc0y = p0y - E0.cy
        p0mc1x = p0x - E1.cx
        p0mc1y = p0y - E1.cy
        p1mc0x = p1x - E0.cx
        p1mc0y = p1y - E0.cy
        p1mc1x = p1x - E1.cx
        p1mc1y = p1y - E1.cy

        n0x = E0.m00 * p0mc0x + E0.m01 * p0mc0y
        n0y = E0.m01 * p0mc0x + E0.m11 * p0mc0y
        n1x = E1.m00 * p0mc1x + E1.m01 * p0mc1y
        n1y = E1.m01 * p0mc1x + E1.m11 * p0mc1y
        dotperp = n1x * n0y + n1y * -n0x
        if dotperp > 0.0
            area += _ea_chord_region(E0, p0mc0x, p0mc0y, p1mc0x, p1mc0y)
        else
            area += _ea_chord_region(E1, p0mc1x, p0mc1y, p1mc1x, p1mc1y)
        end
        i0 = i1
    end
    return area
end

# --- AreaEllipse2Ellipse2::AreaDispatch ---------------------------------------------
function _ea_area_dispatch(E0::_EaEllipseInfo, E1::_EaEllipseInfo, fr::_EaFindResult)
    if fr.intersect
        if fr.numpoints == 1
            # Containment or separation.
            return _ea_area_cs(E0, E1, fr)
        elseif fr.numpoints == 2
            if fr.transverse[1]
                return _ea_area2(E0, E1, fr, 1, 2)
            else
                # Both tangential: one ellipse contained in the other.
                return _ea_area_cs(E0, E1, fr)
            end
        elseif fr.numpoints == 3
            # The tangential intersection is irrelevant for the area.
            if !fr.transverse[1]
                return _ea_area2(E0, E1, fr, 2, 3)
            elseif !fr.transverse[2]
                return _ea_area2(E0, E1, fr, 3, 1)
            else  # transverse[3] == false
                return _ea_area2(E0, E1, fr, 1, 2)
            end
        else  # numpoints == 4
            return _ea_area4(E0, E1, fr)
        end
    else
        # Containment, separation, or same ellipse.
        return _ea_area_cs(E0, E1, fr)
    end
end

# =============================================================================
# Entry point — identical contract to the FFI wrapper
# =============================================================================

"""
    _native_ellipse_intersection_area(cx1, cy1, a1, b1, θ1,
                                      cx2, cy2, a2, b2, θ2) -> Float64

Native Julia port of the GTE ellipse-ellipse intersection area computation
(the retired `libellipse_area` FFI entry point). Each ellipse is given by
center `(cx, cy)`, semi-axes `(a, b)` and rotation `θ`, with
`axis0 = (cos θ, sin θ)`, `axis1 = (-sin θ, cos θ)`, `extent = (a, b)` —
exactly the wrapper's construction. Returns the area of the intersection of
the two solid ellipses (0.0 when they are separated).
"""
function _native_ellipse_intersection_area(
        cx1::Float64, cy1::Float64, a1::Float64, b1::Float64, θ1::Float64,
        cx2::Float64, cy2::Float64, a2::Float64, b2::Float64, θ2::Float64
    )
    cs1 = _ea_cos(θ1)
    sn1 = _ea_sin(θ1)
    cs2 = _ea_cos(θ2)
    sn2 = _ea_sin(θ2)

    # AreaEllipse2Ellipse2 ellipse info (FinishEllipseInfo).
    E0 = _ea_ellipse_info(cx1, cy1, cs1, sn1, -sn1, cs1, a1, b1)
    E1 = _ea_ellipse_info(cx2, cy2, cs2, sn2, -sn2, cs2, a2, b2)

    # FIQuery standard forms and intersection points.
    sc0x, sc0y, s0_00, s0_01, s0_11 = _ea_standard_form(cx1, cy1, cs1, sn1, -sn1, cs1, a1, b1)
    sc1x, sc1y, s1_00, s1_01, s1_11 = _ea_standard_form(cx2, cy2, cs2, sn2, -sn2, cs2, a2, b2)
    fr = _ea_find_intersections(
        sc0x, sc0y, s0_00, s0_01, s0_11,
        sc1x, sc1y, s1_00, s1_01, s1_11
    )

    return _ea_area_dispatch(E0, E1, fr)
end

_native_ellipse_intersection_area(
    cx1::Real, cy1::Real, a1::Real, b1::Real, θ1::Real,
    cx2::Real, cy2::Real, a2::Real, b2::Real, θ2::Real
) =
    _native_ellipse_intersection_area(
    Float64(cx1), Float64(cy1), Float64(a1), Float64(b1), Float64(θ1),
    Float64(cx2), Float64(cy2), Float64(a2), Float64(b2), Float64(θ2)
)
