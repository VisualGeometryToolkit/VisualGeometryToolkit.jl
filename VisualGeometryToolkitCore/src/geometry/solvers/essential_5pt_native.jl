# =============================================================================
# Native 5-point essential matrix — port of PoseLib's relpose_5pt.cc (Nistér)
# =============================================================================
#
# Source of truth: PoseLib/solvers/relpose_5pt.cc + misc/sturm.h (Larsson's
# PoseLib, BSD-3). Faithful port: same trace-constraint construction, same
# degree-10 determinant polynomial (machine-transcribed from the C source),
# same Sturm-sequence bracketing with Ridders+Newton polishing.
#
# One deliberate substitution: the 9×5 epipolar nullspace uses Julia's
# Householder QR instead of Eigen's full-pivot variant. Any orthonormal basis
# of the nullspace yields the same essential matrices (the polynomial system
# is basis-covariant, and the final normalization uses row orthonormality,
# which both bases satisfy) — validated against the compiled backend.
#
# Contract mirrors the PoseLib backend: ,  are bearing vectors
# (calibrated homogeneous, z=1 or unit); returns
#  with x2' E x1 = 0.

# ----- misc/sturm.h ----------------------------------------------------------

const _STURM_MAX_DEPTH = 300

function _sturm_build_seq!(svec::MVector{30, Float64}, fvec::MVector{21, Float64})
    N = 10
    f = MVector{30, Float64}(undef)
    @inbounds for i in 1:21
        f[i] = fvec[i]
    end
    o1 = 0; o2 = N + 1; o3 = 2N + 1          # 0-based offsets of f1,f2,f3
    @inbounds for i in 0:(N - 2)
        q1 = f[o1 + N - i + 1] * f[o2 + N - 1 - i + 1]
        q0 = f[o1 + N - 1 - i + 1] * f[o2 + N - 1 - i + 1] - f[o1 + N - i + 1] * f[o2 + N - 2 - i + 1]
        f[o3 + 1] = f[o1 + 1] - q0 * f[o2 + 1]
        for j in 1:(N - 2 - i)
            f[o3 + j + 1] = f[o1 + j + 1] - q1 * f[o2 + j] - q0 * f[o2 + j + 1]
        end
        c = -abs(f[o3 + N - 2 - i + 1])
        ci = 1.0 / c
        for j in 0:(N - 2 - i)
            f[o3 + j + 1] *= ci
        end
        o1, o2, o3 = o2, o3, o1
        svec[3i + 1] = q0
        svec[3i + 2] = q1
        svec[3i + 3] = c
    end
    @inbounds begin
        svec[3N - 2] = f[o1 + 1]
        svec[3N - 1] = f[o1 + 2]
        svec[3N] = f[o2 + 1]
    end
    return svec
end

# polyval over the monic-normalized poly stored in fvec[off+1 .. off+n]
@inline function _sturm_polyval(f::MVector{21, Float64}, off::Int, n::Int, x::Float64)
    @inbounds begin
        fx = x + f[off + n]
        for i in (n - 1):-1:1
            fx = x * fx + f[off + i]
        end
        return fx
    end
end

@inline function _sturm_signchanges(svec::MVector{30, Float64}, x::Float64)
    N = 10
    f = MVector{11, Float64}(undef)
    @inbounds begin
        f[N + 1] = svec[3N]
        f[N] = svec[3N - 2] + x * svec[3N - 1]
        for i in (N - 2):-1:0
            f[i + 1] = (svec[3i + 1] + x * svec[3i + 2]) * f[i + 2] + svec[3i + 3] * f[i + 3]
        end
        count = 0
        neg1 = f[1] < 0
        for i in 1:N
            neg2 = f[i + 1] < 0
            (neg1 ⊻ neg2) && (count += 1)
            neg1 = neg2
        end
        return count
    end
end

function _sturm_ridders_newton!(
        roots::MVector{10, Float64}, n_roots::Int,
        fvec::MVector{21, Float64}, a::Float64, b::Float64, tol::Float64
    )
    N = 10
    fa = _sturm_polyval(fvec, 0, N, a)
    fb = _sturm_polyval(fvec, 0, N, b)
    ((fa < 0) ⊻ (fb < 0)) || return n_roots
    tol_newton = 1.0e-3
    for _ in 1:30
        abs(a - b) < tol_newton && break
        c = (a + b) * 0.5
        fc = _sturm_polyval(fvec, 0, N, c)
        s = sqrt(max(fc * fc - fa * fb, 0.0))
        s == 0.0 && break
        d = fa < fb ? c + (a - c) * fc / s : c + (c - a) * fc / s
        fd = _sturm_polyval(fvec, 0, N, d)
        if (fd >= 0) ? (fc < 0) : (fc > 0)
            a = c; fa = fc; b = d; fb = fd
        elseif (fd >= 0) ? (fa < 0) : (fa > 0)
            b = d; fb = fd
        else
            a = d; fa = fd
        end
    end
    x = (a + b) * 0.5
    for _ in 1:10
        fx = _sturm_polyval(fvec, 0, N, x)
        abs(fx) < tol && break
        fpx = N * _sturm_polyval(fvec, N + 1, N - 1, x)
        dx = fx / fpx
        x -= dx
        abs(dx) < tol && break
    end
    roots[n_roots + 1] = x
    return n_roots + 1
end

function _sturm_isolate!(
        roots::MVector{10, Float64}, n_roots::Int,
        fvec::MVector{21, Float64}, svec::MVector{30, Float64},
        a::Float64, b::Float64, sa::Int, sb::Int, tol::Float64, depth::Int
    )
    depth > _STURM_MAX_DEPTH && return n_roots
    if b - a < tol
        roots[n_roots + 1] = b
        return n_roots + 1
    end
    n_rts = sa - sb
    if n_rts > 1
        c = (a + b) * 0.5
        sc = _sturm_signchanges(svec, c)
        n_roots = _sturm_isolate!(roots, n_roots, fvec, svec, a, c, sa, sc, tol, depth + 1)
        n_roots = _sturm_isolate!(roots, n_roots, fvec, svec, c, b, sc, sb, tol, depth + 1)
    elseif n_rts == 1
        n_roots = _sturm_ridders_newton!(roots, n_roots, fvec, a, b, tol)
    end
    return n_roots
end

"Real roots of the degree-10 poly with coefficients c[1..11] (c[k] · x^(k-1))."
_bisect_sturm10!(roots::MVector{10, Float64}, c::MVector{11, Float64}; tol::Float64 = 1.0e-10) =
    _bisect_sturm10!(roots, c, MVector{21, Float64}(undef), MVector{30, Float64}(undef); tol)
function _bisect_sturm10!(roots::MVector{10, Float64}, c::MVector{11, Float64},
                          fvec::MVector{21, Float64}, svec::MVector{30, Float64}; tol::Float64 = 1.0e-10)
    N = 10
    c[N + 1] == 0.0 && return 0
    c_inv = 1.0 / c[N + 1]
    @inbounds for i in 1:N
        fvec[i] = c[i] * c_inv
    end
    fvec[N + 1] = 1.0
    @inbounds for i in 0:(N - 2)
        fvec[N + 2 + i] = fvec[i + 2] * ((i + 1) / N)
    end
    fvec[2N + 1] = 1.0
    _sturm_build_seq!(svec, fvec)
    r0 = 1.0 + maximum(abs, view(fvec, 1:N))
    a, b = -r0, r0
    sa = _sturm_signchanges(svec, a)
    sb = _sturm_signchanges(svec, b)
    sa - sb == 0 && return 0
    return _sturm_isolate!(roots, 0, fvec, svec, a, b, sa, sb, tol, 0)
end

# ----- degree-1/2 polynomial products (o1/o2 from relpose_5pt.cc) -------------

@inline function _e5_o1!(c, a, b, op::Int)
    # op: 0 = assign, +1 = add, -1 = subtract
    v1 = a[1] * b[1]
    v2 = a[1] * b[2] + a[2] * b[1]
    v3 = a[1] * b[3] + a[3] * b[1]
    v4 = a[1] * b[4] + a[4] * b[1]
    v5 = a[2] * b[2]
    v6 = a[2] * b[3] + a[3] * b[2]
    v7 = a[2] * b[4] + a[4] * b[2]
    v8 = a[3] * b[3]
    v9 = a[3] * b[4] + a[4] * b[3]
    v10 = a[4] * b[4]
    @inbounds if op == 0
        c[1] = v1; c[2] = v2; c[3] = v3; c[4] = v4; c[5] = v5; c[6] = v6; c[7] = v7; c[8] = v8; c[9] = v9; c[10] = v10
    elseif op == 1
        c[1] += v1; c[2] += v2; c[3] += v3; c[4] += v4; c[5] += v5; c[6] += v6; c[7] += v7; c[8] += v8; c[9] += v9; c[10] += v10
    else
        c[1] -= v1; c[2] -= v2; c[3] -= v3; c[4] -= v4; c[5] -= v5; c[6] -= v6; c[7] -= v7; c[8] -= v8; c[9] -= v9; c[10] -= v10
    end
    return c
end

@inline function _e5_o2!(c, a, b, add::Bool)
    t1 = a[1] * b[1]
    t2 = a[5] * b[2]
    t3 = a[1] * b[2] + a[2] * b[1]
    t4 = a[2] * b[2] + a[5] * b[1]
    t5 = a[1] * b[3] + a[3] * b[1]
    t6 = a[1] * b[4] + a[4] * b[1]
    t7 = a[5] * b[3] + a[6] * b[2]
    t8 = a[5] * b[4] + a[7] * b[2]
    t9 = a[2] * b[3] + a[3] * b[2] + a[6] * b[1]
    t10 = a[2] * b[4] + a[4] * b[2] + a[7] * b[1]
    t11 = a[3] * b[3] + a[8] * b[1]
    t12 = a[3] * b[4] + a[4] * b[3] + a[9] * b[1]
    t13 = a[4] * b[4] + a[10] * b[1]
    t14 = a[6] * b[3] + a[8] * b[2]
    t15 = a[6] * b[4] + a[7] * b[3] + a[9] * b[2]
    t16 = a[7] * b[4] + a[10] * b[2]
    t17 = a[8] * b[3]
    t18 = a[8] * b[4] + a[9] * b[3]
    t19 = a[9] * b[4] + a[10] * b[3]
    t20 = a[10] * b[4]
    @inbounds if add
        c[1] += t1; c[2] += t2; c[3] += t3; c[4] += t4; c[5] += t5; c[6] += t6; c[7] += t7; c[8] += t8; c[9] += t9; c[10] += t10
        c[11] += t11; c[12] += t12; c[13] += t13; c[14] += t14; c[15] += t15; c[16] += t16; c[17] += t17; c[18] += t18; c[19] += t19; c[20] += t20
    else
        c[1] = t1; c[2] = t2; c[3] = t3; c[4] = t4; c[5] = t5; c[6] = t6; c[7] = t7; c[8] = t8; c[9] = t9; c[10] = t10
        c[11] = t11; c[12] = t12; c[13] = t13; c[14] = t14; c[15] = t15; c[16] = t16; c[17] = t17; c[18] = t18; c[19] = t19; c[20] = t20
    end
    return c
end

# ----- trace + determinant constraint rows ------------------------------------

# Nm: rows = orthonormal nullspace basis; column 3j+i (0-based) holds the
# 4 coefficients of essential-matrix entry E[i,j] (column-major E).
# Writes the 10x20 constraint matrix into `coeffs` (MMatrix, no allocation).
_e5_trace_constraints!(coeffs::MMatrix{10, 20, Float64, 200}, Nm::SMatrix{4, 9, Float64, 36}) =
    _e5_trace_constraints!(coeffs, Nm, ntuple(_ -> MVector{10, Float64}(undef), Val(6)))
function _e5_trace_constraints!(coeffs::MMatrix{10, 20, Float64, 200}, Nm::SMatrix{4, 9, Float64, 36},
                                ebufs::NTuple{6, MVector{10, Float64}})
    EE(i, j) = SVector{4, Float64}(Nm[1, 3j + i + 1], Nm[2, 3j + i + 1], Nm[3, 3j + i + 1], Nm[4, 3j + i + 1])
    d = MVector{10, Float64}(undef)
    row = MVector{20, Float64}(undef)

    # determinant constraint (row 10)
    _e5_o1!(d, EE(0, 1), EE(1, 2), 0); _e5_o1!(d, EE(0, 2), EE(1, 1), -1)
    _e5_o2!(row, d, EE(2, 0), false)
    _e5_o1!(d, EE(0, 2), EE(1, 0), 0); _e5_o1!(d, EE(0, 0), EE(1, 2), -1)
    _e5_o2!(row, d, EE(2, 1), true)
    _e5_o1!(d, EE(0, 0), EE(1, 1), 0); _e5_o1!(d, EE(0, 1), EE(1, 0), -1)
    _e5_o2!(row, d, EE(2, 2), true)
    @inbounds for k in 1:20
        coeffs[10, k] = row[k]
    end

    # EE^T upper triangle (10-coefficient quadratics) — six fixed buffers
    E00, E01, E02, E11, E12, E22 = ebufs
    EETbuf = (E00, E01, E02, E01, E11, E12, E02, E12, E22)   # 3x3 symmetric, col-major
    for (buf, (i, j)) in (
            (E00, (0, 0)), (E01, (0, 1)), (E02, (0, 2)),
            (E11, (1, 1)), (E12, (1, 2)), (E22, (2, 2)),
        )
        _e5_o1!(buf, EE(i, 0), EE(j, 0), 0)
        _e5_o1!(buf, EE(i, 1), EE(j, 1), 1)
        _e5_o1!(buf, EE(i, 2), EE(j, 2), 1)
    end
    @inbounds for k in 1:10
        t = 0.5 * (E00[k] + E11[k] + E22[k])
        E00[k] -= t; E11[k] -= t; E22[k] -= t
    end
    EET(i, j) = EETbuf[3j + i + 1]                # 0-based symmetric lookup
    cnt = 1
    for i in 0:2, j in 0:2
        _e5_o2!(row, EET(i, 0), EE(0, j), false)
        _e5_o2!(row, EET(i, 1), EE(1, j), true)
        _e5_o2!(row, EET(i, 2), EE(2, j), true)
        @inbounds for k in 1:20
            coeffs[cnt, k] = row[k]
        end
        cnt += 1
    end
    return coeffs
end

# Full Householder QR of a 9x5 matrix, returning the transposed right 4
# columns of Q (an orthonormal nullspace basis) — allocation-free, mirrors
# the structure of the Eigen call in the C++ (any orthonormal basis works).
function _e5_nullspace(M::MMatrix{9, 5, Float64, 45})
    Q = MMatrix{9, 9, Float64, 81}(undef)
    @inbounds for i in 1:9, j in 1:9
        Q[i, j] = (i == j) ? 1.0 : 0.0
    end
    v = MVector{9, Float64}(undef)
    @inbounds for k in 1:5
        # Householder vector for column k
        α = 0.0
        for i in k:9
            α += M[i, k] * M[i, k]
        end
        α = sqrt(α)
        α == 0.0 && continue
        M[k, k] > 0 && (α = -α)
        for i in 1:9
            v[i] = i < k ? 0.0 : M[i, k]
        end
        v[k] -= α
        vnorm2 = 0.0
        for i in k:9
            vnorm2 += v[i] * v[i]
        end
        vnorm2 == 0.0 && continue
        β = 2.0 / vnorm2
        # apply to M (columns k..5)
        for j in k:5
            dotv = 0.0
            for i in k:9
                dotv += v[i] * M[i, j]
            end
            dotv *= β
            for i in k:9
                M[i, j] -= dotv * v[i]
            end
        end
        # accumulate into Q (Q := Q * H)
        for j in 1:9
            dotv = 0.0
            for i in k:9
                dotv += Q[j, i] * v[i]
            end
            dotv *= β
            for i in k:9
                Q[j, i] -= dotv * v[i]
            end
        end
    end
    # rows of the result = columns 6..9 of Q, transposed
    return SMatrix{4, 9, Float64, 36}(
        ntuple(
            idx -> begin
                r = (idx - 1) % 4 + 1; c = (idx - 1) ÷ 4 + 1
                Q[c, 5 + r]
            end, Val(36)
        )
    )
end


# In-place partial-pivot Gaussian elimination: solves LHS * X = RHS for the
# 10x10 blocks of the constraint matrix (ports the Eigen partialPivLu().solve
# call). Zero-allocation on MMatrix — StaticArrays' generic lu falls back to
# a heap path at this size.
function _e5_solve10!(lhs::MMatrix{10, 10, Float64, 100}, rhs::MMatrix{10, 10, Float64, 100})
    @inbounds for k in 1:10
        # partial pivot
        p = k; pv = abs(lhs[k, k])
        for i in (k + 1):10
            a = abs(lhs[i, k])
            a > pv && (pv = a; p = i)
        end
        if p != k
            for j in 1:10
                lhs[k, j], lhs[p, j] = lhs[p, j], lhs[k, j]
                rhs[k, j], rhs[p, j] = rhs[p, j], rhs[k, j]
            end
        end
        d = 1.0 / lhs[k, k]
        for i in (k + 1):10
            f = lhs[i, k] * d
            f == 0.0 && continue
            for j in (k + 1):10
                lhs[i, j] -= f * lhs[k, j]
            end
            for j in 1:10
                rhs[i, j] -= f * rhs[k, j]
            end
            lhs[i, k] = 0.0
        end
    end
    # back substitution
    @inbounds for k in 10:-1:1
        d = 1.0 / lhs[k, k]
        for j in 1:10
            acc = rhs[k, j]
            for i in (k + 1):10
                acc -= lhs[k, i] * rhs[i, j]
            end
            rhs[k, j] = acc * d
        end
    end
    return rhs
end

# Degree-10 determinant polynomial coefficients (machine-transcribed from
# relpose_5pt.cc). Separated behind a function barrier: as one inline block
# the ~1500-term expression tree overwhelms the optimizer.
@noinline function _e5_det_poly!(c::MVector{11, Float64}, A::MMatrix{3, 13, Float64, 39})
    c[1] = A[1, 13] * A[2, 4] * A[3, 8] - A[1, 13] * A[2, 8] * A[3, 4] - A[1, 4] * A[3, 8] * A[2, 13] +
        A[1, 8] * A[3, 4] * A[2, 13] + A[1, 4] * A[2, 8] * A[3, 13] - A[1, 8] * A[2, 4] * A[3, 13]
    c[2] = A[1, 12] * A[2, 4] * A[3, 8] - A[1, 12] * A[2, 8] * A[3, 4] + A[1, 13] * A[2, 3] * A[3, 8] +
        A[1, 13] * A[2, 4] * A[3, 7] - A[1, 13] * A[2, 7] * A[3, 4] - A[1, 13] * A[2, 8] * A[3, 3] -
        A[1, 3] * A[3, 8] * A[2, 13] - A[1, 4] * A[3, 7] * A[2, 13] - A[1, 4] * A[3, 8] * A[2, 12] +
        A[1, 7] * A[3, 4] * A[2, 13] + A[1, 8] * A[3, 3] * A[2, 13] + A[1, 8] * A[3, 4] * A[2, 12] +
        A[1, 3] * A[2, 8] * A[3, 13] + A[1, 4] * A[2, 7] * A[3, 13] + A[1, 4] * A[2, 8] * A[3, 12] -
        A[1, 7] * A[2, 4] * A[3, 13] - A[1, 8] * A[2, 3] * A[3, 13] - A[1, 8] * A[2, 4] * A[3, 12]
    c[3] = A[1, 11] * A[2, 4] * A[3, 8] - A[1, 11] * A[2, 8] * A[3, 4] + A[1, 12] * A[2, 3] * A[3, 8] +
        A[1, 12] * A[2, 4] * A[3, 7] - A[1, 12] * A[2, 7] * A[3, 4] - A[1, 12] * A[2, 8] * A[3, 3] +
        A[2, 2] * A[1, 13] * A[3, 8] + A[1, 13] * A[2, 3] * A[3, 7] + A[1, 13] * A[2, 4] * A[3, 6] -
        A[1, 13] * A[2, 6] * A[3, 4] - A[1, 13] * A[2, 7] * A[3, 3] - A[1, 13] * A[2, 8] * A[3, 2] -
        A[1, 2] * A[3, 8] * A[2, 13] - A[1, 3] * A[3, 7] * A[2, 13] - A[1, 3] * A[3, 8] * A[2, 12] -
        A[1, 4] * A[3, 6] * A[2, 13] - A[1, 4] * A[3, 7] * A[2, 12] - A[1, 4] * A[3, 8] * A[2, 11] +
        A[1, 6] * A[3, 4] * A[2, 13] + A[1, 7] * A[3, 3] * A[2, 13] + A[1, 7] * A[3, 4] * A[2, 12] +
        A[1, 8] * A[3, 2] * A[2, 13] + A[1, 8] * A[3, 3] * A[2, 12] + A[1, 8] * A[3, 4] * A[2, 11] +
        A[1, 2] * A[2, 8] * A[3, 13] + A[1, 3] * A[2, 7] * A[3, 13] + A[1, 3] * A[2, 8] * A[3, 12] +
        A[1, 4] * A[2, 6] * A[3, 13] + A[1, 4] * A[2, 7] * A[3, 12] + A[1, 4] * A[2, 8] * A[3, 11] -
        A[1, 6] * A[2, 4] * A[3, 13] - A[1, 7] * A[2, 3] * A[3, 13] - A[1, 7] * A[2, 4] * A[3, 12] -
        A[1, 8] * A[2, 2] * A[3, 13] - A[1, 8] * A[2, 3] * A[3, 12] - A[1, 8] * A[2, 4] * A[3, 11]
    c[4] = A[1, 4] * A[2, 8] * A[3, 10] - A[1, 4] * A[2, 10] * A[3, 8] - A[1, 8] * A[2, 4] * A[3, 10] +
        A[1, 8] * A[2, 10] * A[3, 4] + A[1, 10] * A[2, 4] * A[3, 8] - A[1, 10] * A[2, 8] * A[3, 4] +
        A[1, 11] * A[2, 3] * A[3, 8] + A[1, 11] * A[2, 4] * A[3, 7] - A[1, 11] * A[2, 7] * A[3, 4] -
        A[1, 11] * A[2, 8] * A[3, 3] + A[2, 1] * A[1, 13] * A[3, 8] + A[1, 12] * A[2, 2] * A[3, 8] +
        A[1, 12] * A[2, 3] * A[3, 7] + A[1, 12] * A[2, 4] * A[3, 6] - A[1, 12] * A[2, 6] * A[3, 4] -
        A[1, 12] * A[2, 7] * A[3, 3] - A[1, 12] * A[2, 8] * A[3, 2] + A[2, 2] * A[1, 13] * A[3, 7] +
        A[1, 13] * A[2, 3] * A[3, 6] + A[1, 13] * A[2, 4] * A[3, 5] - A[1, 13] * A[2, 5] * A[3, 4] -
        A[1, 13] * A[2, 6] * A[3, 3] - A[1, 13] * A[2, 7] * A[3, 2] - A[1, 13] * A[2, 8] * A[3, 1] -
        A[1, 1] * A[3, 8] * A[2, 13] - A[1, 2] * A[3, 7] * A[2, 13] - A[1, 2] * A[3, 8] * A[2, 12] -
        A[1, 3] * A[3, 6] * A[2, 13] - A[1, 3] * A[3, 7] * A[2, 12] - A[1, 3] * A[3, 8] * A[2, 11] -
        A[1, 4] * A[3, 5] * A[2, 13] - A[1, 4] * A[3, 6] * A[2, 12] - A[1, 4] * A[3, 7] * A[2, 11] +
        A[1, 5] * A[3, 4] * A[2, 13] + A[1, 6] * A[3, 3] * A[2, 13] + A[1, 6] * A[3, 4] * A[2, 12] +
        A[1, 7] * A[3, 2] * A[2, 13] + A[1, 7] * A[3, 3] * A[2, 12] + A[1, 7] * A[3, 4] * A[2, 11] +
        A[1, 8] * A[3, 1] * A[2, 13] + A[1, 8] * A[3, 2] * A[2, 12] + A[1, 8] * A[3, 3] * A[2, 11] +
        A[1, 1] * A[2, 8] * A[3, 13] + A[1, 2] * A[2, 7] * A[3, 13] + A[1, 2] * A[2, 8] * A[3, 12] +
        A[1, 3] * A[2, 6] * A[3, 13] + A[1, 3] * A[2, 7] * A[3, 12] + A[1, 3] * A[2, 8] * A[3, 11] +
        A[1, 4] * A[2, 5] * A[3, 13] + A[1, 4] * A[2, 6] * A[3, 12] + A[1, 4] * A[2, 7] * A[3, 11] -
        A[1, 5] * A[2, 4] * A[3, 13] - A[1, 6] * A[2, 3] * A[3, 13] - A[1, 6] * A[2, 4] * A[3, 12] -
        A[1, 7] * A[2, 2] * A[3, 13] - A[1, 7] * A[2, 3] * A[3, 12] - A[1, 7] * A[2, 4] * A[3, 11] -
        A[1, 8] * A[2, 1] * A[3, 13] - A[1, 8] * A[2, 2] * A[3, 12] - A[1, 8] * A[2, 3] * A[3, 11]
    c[5] = A[1, 3] * A[2, 8] * A[3, 10] - A[1, 3] * A[2, 10] * A[3, 8] + A[1, 4] * A[2, 7] * A[3, 10] +
        A[1, 4] * A[2, 8] * A[3, 9] - A[1, 4] * A[2, 9] * A[3, 8] - A[1, 4] * A[2, 10] * A[3, 7] -
        A[1, 7] * A[2, 4] * A[3, 10] + A[1, 7] * A[2, 10] * A[3, 4] - A[1, 8] * A[2, 3] * A[3, 10] -
        A[1, 8] * A[2, 4] * A[3, 9] + A[1, 8] * A[2, 9] * A[3, 4] + A[1, 8] * A[2, 10] * A[3, 3] +
        A[1, 9] * A[2, 4] * A[3, 8] - A[1, 9] * A[2, 8] * A[3, 4] + A[1, 10] * A[2, 3] * A[3, 8] +
        A[1, 10] * A[2, 4] * A[3, 7] - A[1, 10] * A[2, 7] * A[3, 4] - A[1, 10] * A[2, 8] * A[3, 3] +
        A[1, 11] * A[2, 2] * A[3, 8] + A[1, 11] * A[2, 3] * A[3, 7] + A[1, 11] * A[2, 4] * A[3, 6] -
        A[1, 11] * A[2, 6] * A[3, 4] - A[1, 11] * A[2, 7] * A[3, 3] - A[1, 11] * A[2, 8] * A[3, 2] +
        A[2, 1] * A[1, 12] * A[3, 8] + A[2, 1] * A[1, 13] * A[3, 7] + A[1, 12] * A[2, 2] * A[3, 7] +
        A[1, 12] * A[2, 3] * A[3, 6] + A[1, 12] * A[2, 4] * A[3, 5] - A[1, 12] * A[2, 5] * A[3, 4] -
        A[1, 12] * A[2, 6] * A[3, 3] - A[1, 12] * A[2, 7] * A[3, 2] - A[1, 12] * A[2, 8] * A[3, 1] +
        A[2, 2] * A[1, 13] * A[3, 6] + A[1, 13] * A[2, 3] * A[3, 5] - A[1, 13] * A[2, 5] * A[3, 3] -
        A[1, 13] * A[2, 6] * A[3, 2] - A[1, 13] * A[2, 7] * A[3, 1] - A[1, 1] * A[3, 7] * A[2, 13] -
        A[1, 1] * A[3, 8] * A[2, 12] - A[1, 2] * A[3, 6] * A[2, 13] - A[1, 2] * A[3, 7] * A[2, 12] -
        A[1, 2] * A[3, 8] * A[2, 11] - A[1, 3] * A[3, 5] * A[2, 13] - A[1, 3] * A[3, 6] * A[2, 12] -
        A[1, 3] * A[3, 7] * A[2, 11] - A[1, 4] * A[3, 5] * A[2, 12] - A[1, 4] * A[3, 6] * A[2, 11] +
        A[1, 5] * A[3, 3] * A[2, 13] + A[1, 5] * A[3, 4] * A[2, 12] + A[1, 6] * A[3, 2] * A[2, 13] +
        A[1, 6] * A[3, 3] * A[2, 12] + A[1, 6] * A[3, 4] * A[2, 11] + A[1, 7] * A[3, 1] * A[2, 13] +
        A[1, 7] * A[3, 2] * A[2, 12] + A[1, 7] * A[3, 3] * A[2, 11] + A[1, 8] * A[3, 1] * A[2, 12] +
        A[1, 8] * A[3, 2] * A[2, 11] + A[1, 1] * A[2, 7] * A[3, 13] + A[1, 1] * A[2, 8] * A[3, 12] +
        A[1, 2] * A[2, 6] * A[3, 13] + A[1, 2] * A[2, 7] * A[3, 12] + A[1, 2] * A[2, 8] * A[3, 11] +
        A[1, 3] * A[2, 5] * A[3, 13] + A[1, 3] * A[2, 6] * A[3, 12] + A[1, 3] * A[2, 7] * A[3, 11] +
        A[1, 4] * A[2, 5] * A[3, 12] + A[1, 4] * A[2, 6] * A[3, 11] - A[1, 5] * A[2, 3] * A[3, 13] -
        A[1, 5] * A[2, 4] * A[3, 12] - A[1, 6] * A[2, 2] * A[3, 13] - A[1, 6] * A[2, 3] * A[3, 12] -
        A[1, 6] * A[2, 4] * A[3, 11] - A[1, 7] * A[2, 1] * A[3, 13] - A[1, 7] * A[2, 2] * A[3, 12] -
        A[1, 7] * A[2, 3] * A[3, 11] - A[1, 8] * A[2, 1] * A[3, 12] - A[1, 8] * A[2, 2] * A[3, 11]
    c[6] = A[1, 2] * A[2, 8] * A[3, 10] - A[1, 2] * A[2, 10] * A[3, 8] + A[1, 3] * A[2, 7] * A[3, 10] +
        A[1, 3] * A[2, 8] * A[3, 9] - A[1, 3] * A[2, 9] * A[3, 8] - A[1, 3] * A[2, 10] * A[3, 7] +
        A[1, 4] * A[2, 6] * A[3, 10] + A[1, 4] * A[2, 7] * A[3, 9] - A[1, 4] * A[2, 9] * A[3, 7] -
        A[1, 4] * A[2, 10] * A[3, 6] - A[1, 6] * A[2, 4] * A[3, 10] + A[1, 6] * A[2, 10] * A[3, 4] -
        A[1, 7] * A[2, 3] * A[3, 10] - A[1, 7] * A[2, 4] * A[3, 9] + A[1, 7] * A[2, 9] * A[3, 4] +
        A[1, 7] * A[2, 10] * A[3, 3] - A[1, 8] * A[2, 2] * A[3, 10] - A[1, 8] * A[2, 3] * A[3, 9] +
        A[1, 8] * A[2, 9] * A[3, 3] + A[1, 8] * A[2, 10] * A[3, 2] + A[1, 9] * A[2, 3] * A[3, 8] +
        A[1, 9] * A[2, 4] * A[3, 7] - A[1, 9] * A[2, 7] * A[3, 4] - A[1, 9] * A[2, 8] * A[3, 3] +
        A[1, 10] * A[2, 2] * A[3, 8] + A[1, 10] * A[2, 3] * A[3, 7] + A[1, 10] * A[2, 4] * A[3, 6] -
        A[1, 10] * A[2, 6] * A[3, 4] - A[1, 10] * A[2, 7] * A[3, 3] - A[1, 10] * A[2, 8] * A[3, 2] +
        A[1, 11] * A[2, 1] * A[3, 8] + A[1, 11] * A[2, 2] * A[3, 7] + A[1, 11] * A[2, 3] * A[3, 6] +
        A[1, 11] * A[2, 4] * A[3, 5] - A[1, 11] * A[2, 5] * A[3, 4] - A[1, 11] * A[2, 6] * A[3, 3] -
        A[1, 11] * A[2, 7] * A[3, 2] - A[1, 11] * A[2, 8] * A[3, 1] + A[2, 1] * A[1, 12] * A[3, 7] +
        A[2, 1] * A[1, 13] * A[3, 6] + A[1, 12] * A[2, 2] * A[3, 6] + A[1, 12] * A[2, 3] * A[3, 5] -
        A[1, 12] * A[2, 5] * A[3, 3] - A[1, 12] * A[2, 6] * A[3, 2] - A[1, 12] * A[2, 7] * A[3, 1] +
        A[2, 2] * A[1, 13] * A[3, 5] - A[1, 13] * A[2, 5] * A[3, 2] - A[1, 13] * A[2, 6] * A[3, 1] -
        A[1, 1] * A[3, 6] * A[2, 13] - A[1, 1] * A[3, 7] * A[2, 12] - A[1, 1] * A[3, 8] * A[2, 11] -
        A[1, 2] * A[3, 5] * A[2, 13] - A[1, 2] * A[3, 6] * A[2, 12] - A[1, 2] * A[3, 7] * A[2, 11] -
        A[1, 3] * A[3, 5] * A[2, 12] - A[1, 3] * A[3, 6] * A[2, 11] - A[1, 4] * A[3, 5] * A[2, 11] +
        A[1, 5] * A[3, 2] * A[2, 13] + A[1, 5] * A[3, 3] * A[2, 12] + A[1, 5] * A[3, 4] * A[2, 11] +
        A[1, 6] * A[3, 1] * A[2, 13] + A[1, 6] * A[3, 2] * A[2, 12] + A[1, 6] * A[3, 3] * A[2, 11] +
        A[1, 7] * A[3, 1] * A[2, 12] + A[1, 7] * A[3, 2] * A[2, 11] + A[1, 8] * A[3, 1] * A[2, 11] +
        A[1, 1] * A[2, 6] * A[3, 13] + A[1, 1] * A[2, 7] * A[3, 12] + A[1, 1] * A[2, 8] * A[3, 11] +
        A[1, 2] * A[2, 5] * A[3, 13] + A[1, 2] * A[2, 6] * A[3, 12] + A[1, 2] * A[2, 7] * A[3, 11] +
        A[1, 3] * A[2, 5] * A[3, 12] + A[1, 3] * A[2, 6] * A[3, 11] + A[1, 4] * A[2, 5] * A[3, 11] -
        A[1, 5] * A[2, 2] * A[3, 13] - A[1, 5] * A[2, 3] * A[3, 12] - A[1, 5] * A[2, 4] * A[3, 11] -
        A[1, 6] * A[2, 1] * A[3, 13] - A[1, 6] * A[2, 2] * A[3, 12] - A[1, 6] * A[2, 3] * A[3, 11] -
        A[1, 7] * A[2, 1] * A[3, 12] - A[1, 7] * A[2, 2] * A[3, 11] - A[1, 8] * A[2, 1] * A[3, 11]
    c[7] = A[1, 1] * A[2, 8] * A[3, 10] - A[1, 1] * A[2, 10] * A[3, 8] + A[1, 2] * A[2, 7] * A[3, 10] +
        A[1, 2] * A[2, 8] * A[3, 9] - A[1, 2] * A[2, 9] * A[3, 8] - A[1, 2] * A[2, 10] * A[3, 7] +
        A[1, 3] * A[2, 6] * A[3, 10] + A[1, 3] * A[2, 7] * A[3, 9] - A[1, 3] * A[2, 9] * A[3, 7] -
        A[1, 3] * A[2, 10] * A[3, 6] + A[1, 4] * A[2, 5] * A[3, 10] + A[1, 4] * A[2, 6] * A[3, 9] -
        A[1, 4] * A[2, 9] * A[3, 6] - A[1, 4] * A[2, 10] * A[3, 5] - A[1, 5] * A[2, 4] * A[3, 10] +
        A[1, 5] * A[2, 10] * A[3, 4] - A[1, 6] * A[2, 3] * A[3, 10] - A[1, 6] * A[2, 4] * A[3, 9] +
        A[1, 6] * A[2, 9] * A[3, 4] + A[1, 6] * A[2, 10] * A[3, 3] - A[1, 7] * A[2, 2] * A[3, 10] -
        A[1, 7] * A[2, 3] * A[3, 9] + A[1, 7] * A[2, 9] * A[3, 3] + A[1, 7] * A[2, 10] * A[3, 2] -
        A[1, 8] * A[2, 1] * A[3, 10] - A[1, 8] * A[2, 2] * A[3, 9] + A[1, 8] * A[2, 9] * A[3, 2] +
        A[1, 8] * A[2, 10] * A[3, 1] + A[1, 9] * A[2, 2] * A[3, 8] + A[1, 9] * A[2, 3] * A[3, 7] +
        A[1, 9] * A[2, 4] * A[3, 6] - A[1, 9] * A[2, 6] * A[3, 4] - A[1, 9] * A[2, 7] * A[3, 3] -
        A[1, 9] * A[2, 8] * A[3, 2] + A[1, 10] * A[2, 1] * A[3, 8] + A[1, 10] * A[2, 2] * A[3, 7] +
        A[1, 10] * A[2, 3] * A[3, 6] + A[1, 10] * A[2, 4] * A[3, 5] - A[1, 10] * A[2, 5] * A[3, 4] -
        A[1, 10] * A[2, 6] * A[3, 3] - A[1, 10] * A[2, 7] * A[3, 2] - A[1, 10] * A[2, 8] * A[3, 1] +
        A[1, 11] * A[2, 1] * A[3, 7] + A[1, 11] * A[2, 2] * A[3, 6] + A[1, 11] * A[2, 3] * A[3, 5] -
        A[1, 11] * A[2, 5] * A[3, 3] - A[1, 11] * A[2, 6] * A[3, 2] - A[1, 11] * A[2, 7] * A[3, 1] +
        A[2, 1] * A[1, 12] * A[3, 6] + A[2, 1] * A[1, 13] * A[3, 5] + A[1, 12] * A[2, 2] * A[3, 5] -
        A[1, 12] * A[2, 5] * A[3, 2] - A[1, 12] * A[2, 6] * A[3, 1] - A[1, 13] * A[2, 5] * A[3, 1] -
        A[1, 1] * A[3, 5] * A[2, 13] - A[1, 1] * A[3, 6] * A[2, 12] - A[1, 1] * A[3, 7] * A[2, 11] -
        A[1, 2] * A[3, 5] * A[2, 12] - A[1, 2] * A[3, 6] * A[2, 11] - A[1, 3] * A[3, 5] * A[2, 11] +
        A[1, 5] * A[3, 1] * A[2, 13] + A[1, 5] * A[3, 2] * A[2, 12] + A[1, 5] * A[3, 3] * A[2, 11] +
        A[1, 6] * A[3, 1] * A[2, 12] + A[1, 6] * A[3, 2] * A[2, 11] + A[1, 7] * A[3, 1] * A[2, 11] +
        A[1, 1] * A[2, 5] * A[3, 13] + A[1, 1] * A[2, 6] * A[3, 12] + A[1, 1] * A[2, 7] * A[3, 11] +
        A[1, 2] * A[2, 5] * A[3, 12] + A[1, 2] * A[2, 6] * A[3, 11] + A[1, 3] * A[2, 5] * A[3, 11] -
        A[1, 5] * A[2, 1] * A[3, 13] - A[1, 5] * A[2, 2] * A[3, 12] - A[1, 5] * A[2, 3] * A[3, 11] -
        A[1, 6] * A[2, 1] * A[3, 12] - A[1, 6] * A[2, 2] * A[3, 11] - A[1, 7] * A[2, 1] * A[3, 11]
    c[8] = A[1, 1] * A[2, 7] * A[3, 10] + A[1, 1] * A[2, 8] * A[3, 9] - A[1, 1] * A[2, 9] * A[3, 8] -
        A[1, 1] * A[2, 10] * A[3, 7] + A[1, 2] * A[2, 6] * A[3, 10] + A[1, 2] * A[2, 7] * A[3, 9] -
        A[1, 2] * A[2, 9] * A[3, 7] - A[1, 2] * A[2, 10] * A[3, 6] + A[1, 3] * A[2, 5] * A[3, 10] +
        A[1, 3] * A[2, 6] * A[3, 9] - A[1, 3] * A[2, 9] * A[3, 6] - A[1, 3] * A[2, 10] * A[3, 5] +
        A[1, 4] * A[2, 5] * A[3, 9] - A[1, 4] * A[2, 9] * A[3, 5] - A[1, 5] * A[2, 3] * A[3, 10] -
        A[1, 5] * A[2, 4] * A[3, 9] + A[1, 5] * A[2, 9] * A[3, 4] + A[1, 5] * A[2, 10] * A[3, 3] -
        A[1, 6] * A[2, 2] * A[3, 10] - A[1, 6] * A[2, 3] * A[3, 9] + A[1, 6] * A[2, 9] * A[3, 3] +
        A[1, 6] * A[2, 10] * A[3, 2] - A[1, 7] * A[2, 1] * A[3, 10] - A[1, 7] * A[2, 2] * A[3, 9] +
        A[1, 7] * A[2, 9] * A[3, 2] + A[1, 7] * A[2, 10] * A[3, 1] - A[1, 8] * A[2, 1] * A[3, 9] +
        A[1, 8] * A[2, 9] * A[3, 1] + A[1, 9] * A[2, 1] * A[3, 8] + A[1, 9] * A[2, 2] * A[3, 7] +
        A[1, 9] * A[2, 3] * A[3, 6] + A[1, 9] * A[2, 4] * A[3, 5] - A[1, 9] * A[2, 5] * A[3, 4] -
        A[1, 9] * A[2, 6] * A[3, 3] - A[1, 9] * A[2, 7] * A[3, 2] - A[1, 9] * A[2, 8] * A[3, 1] +
        A[1, 10] * A[2, 1] * A[3, 7] + A[1, 10] * A[2, 2] * A[3, 6] + A[1, 10] * A[2, 3] * A[3, 5] -
        A[1, 10] * A[2, 5] * A[3, 3] - A[1, 10] * A[2, 6] * A[3, 2] - A[1, 10] * A[2, 7] * A[3, 1] +
        A[1, 11] * A[2, 1] * A[3, 6] + A[1, 11] * A[2, 2] * A[3, 5] - A[1, 11] * A[2, 5] * A[3, 2] -
        A[1, 11] * A[2, 6] * A[3, 1] + A[2, 1] * A[1, 12] * A[3, 5] - A[1, 12] * A[2, 5] * A[3, 1] -
        A[1, 1] * A[3, 5] * A[2, 12] - A[1, 1] * A[3, 6] * A[2, 11] - A[1, 2] * A[3, 5] * A[2, 11] +
        A[1, 5] * A[3, 1] * A[2, 12] + A[1, 5] * A[3, 2] * A[2, 11] + A[1, 6] * A[3, 1] * A[2, 11] +
        A[1, 1] * A[2, 5] * A[3, 12] + A[1, 1] * A[2, 6] * A[3, 11] + A[1, 2] * A[2, 5] * A[3, 11] -
        A[1, 5] * A[2, 1] * A[3, 12] - A[1, 5] * A[2, 2] * A[3, 11] - A[1, 6] * A[2, 1] * A[3, 11]
    c[9] = A[1, 1] * A[2, 6] * A[3, 10] + A[1, 1] * A[2, 7] * A[3, 9] - A[1, 1] * A[2, 9] * A[3, 7] -
        A[1, 1] * A[2, 10] * A[3, 6] + A[1, 2] * A[2, 5] * A[3, 10] + A[1, 2] * A[2, 6] * A[3, 9] -
        A[1, 2] * A[2, 9] * A[3, 6] - A[1, 2] * A[2, 10] * A[3, 5] + A[1, 3] * A[2, 5] * A[3, 9] -
        A[1, 3] * A[2, 9] * A[3, 5] - A[1, 5] * A[2, 2] * A[3, 10] - A[1, 5] * A[2, 3] * A[3, 9] +
        A[1, 5] * A[2, 9] * A[3, 3] + A[1, 5] * A[2, 10] * A[3, 2] - A[1, 6] * A[2, 1] * A[3, 10] -
        A[1, 6] * A[2, 2] * A[3, 9] + A[1, 6] * A[2, 9] * A[3, 2] + A[1, 6] * A[2, 10] * A[3, 1] -
        A[1, 7] * A[2, 1] * A[3, 9] + A[1, 7] * A[2, 9] * A[3, 1] + A[1, 9] * A[2, 1] * A[3, 7] +
        A[1, 9] * A[2, 2] * A[3, 6] + A[1, 9] * A[2, 3] * A[3, 5] - A[1, 9] * A[2, 5] * A[3, 3] -
        A[1, 9] * A[2, 6] * A[3, 2] - A[1, 9] * A[2, 7] * A[3, 1] + A[1, 10] * A[2, 1] * A[3, 6] +
        A[1, 10] * A[2, 2] * A[3, 5] - A[1, 10] * A[2, 5] * A[3, 2] - A[1, 10] * A[2, 6] * A[3, 1] +
        A[1, 11] * A[2, 1] * A[3, 5] - A[1, 11] * A[2, 5] * A[3, 1] - A[1, 1] * A[3, 5] * A[2, 11] +
        A[1, 5] * A[3, 1] * A[2, 11] + A[1, 1] * A[2, 5] * A[3, 11] - A[1, 5] * A[2, 1] * A[3, 11]
    c[10] = A[1, 1] * A[2, 5] * A[3, 10] + A[1, 1] * A[2, 6] * A[3, 9] - A[1, 1] * A[2, 9] * A[3, 6] -
        A[1, 1] * A[2, 10] * A[3, 5] + A[1, 2] * A[2, 5] * A[3, 9] - A[1, 2] * A[2, 9] * A[3, 5] -
        A[1, 5] * A[2, 1] * A[3, 10] - A[1, 5] * A[2, 2] * A[3, 9] + A[1, 5] * A[2, 9] * A[3, 2] +
        A[1, 5] * A[2, 10] * A[3, 1] - A[1, 6] * A[2, 1] * A[3, 9] + A[1, 6] * A[2, 9] * A[3, 1] +
        A[1, 9] * A[2, 1] * A[3, 6] + A[1, 9] * A[2, 2] * A[3, 5] - A[1, 9] * A[2, 5] * A[3, 2] -
        A[1, 9] * A[2, 6] * A[3, 1] + A[1, 10] * A[2, 1] * A[3, 5] - A[1, 10] * A[2, 5] * A[3, 1]
    c[11] = A[1, 1] * A[2, 5] * A[3, 9] - A[1, 1] * A[2, 9] * A[3, 5] - A[1, 5] * A[2, 1] * A[3, 9] +
        A[1, 5] * A[2, 9] * A[3, 1] + A[1, 9] * A[2, 1] * A[3, 5] - A[1, 9] * A[2, 5] * A[3, 1]

    return c
end

# ----- solver ------------------------------------------------------------------

# Five calibrated bearings, as a tuple or a static vector: the shapes the
# solver layer (`solve(::Essential5pt, …)`) and direct callers supply.
const _Bearings5 = Union{NTuple{5, <:StaticVector{3}}, StaticVector{5, <:StaticVector{3}}}

"""
    essential_5pt(::NativeSolver, x1, x2) -> FixedModels{10, EssentialMat{Float64}}

Nistér 5-point relative pose (essential matrices only), a native Julia port of
PoseLib's `relpose_5pt`. `x1` and `x2` hold five calibrated bearings each, as a
tuple or a static vector of 3-vectors.

Typed to 3-vectors so a 2-vector (a point on the z = 1 plane) is a
`MethodError` rather than a `DimensionMismatch` from inside the kernel, the
same refusal `solve(::Essential5pt, …)` and PoseLib's backend give. The type is
a necessary condition, not a guarantee: a 3-vector still ON the z = 1 plane
is accepted and, the epipolar algebra being scale-invariant per endpoint, even gives
the right answer for rays the chart can represent. Bearings must come from the
camera's back-projection (`backproject_with_jacobian`), where they are built.
"""
# The solver's escaping scratch (the arrays its non-inlined kernels write into),
# built once per task and reused by every solve on it — a minimal solve runs once
# per RANSAC trial, and these arrays were its only heap allocations. Every entry a
# kernel reads it first writes, so reuse cannot leak one solve into the next.
struct _E5Scratch
    M::MMatrix{9, 5, Float64, 45}
    coeffs::MMatrix{10, 20, Float64, 200}
    lhs10::MMatrix{10, 10, Float64, 100}
    rhs10::MMatrix{10, 10, Float64, 100}
    A::MMatrix{3, 13, Float64, 39}
    roots::MVector{10, Float64}
    Es::MVector{10, EssentialMat{Float64}}
    fvec::MVector{21, Float64}
    svec::MVector{30, Float64}
    c::MVector{11, Float64}
    ebufs::NTuple{6, MVector{10, Float64}}
end
_E5Scratch() = _E5Scratch(
    MMatrix{9, 5, Float64, 45}(undef), MMatrix{10, 20, Float64, 200}(undef),
    MMatrix{10, 10, Float64, 100}(undef), MMatrix{10, 10, Float64, 100}(undef),
    MMatrix{3, 13, Float64, 39}(undef), MVector{10, Float64}(undef),
    MVector{10, EssentialMat{Float64}}(undef), MVector{21, Float64}(undef), MVector{30, Float64}(undef),
    MVector{11, Float64}(undef), ntuple(_ -> MVector{10, Float64}(undef), Val(6))
)

function essential_5pt(::NativeSolver, x1::_Bearings5, x2::_Bearings5)
    scratch = task_buffers(_E5Scratch, :vgc_essential_5pt_scratch)::_E5Scratch
    M = scratch.M
    @inbounds for i in 1:5
        a = SVector{3, Float64}(x1[i]); b = SVector{3, Float64}(x2[i])
        M[1, i] = a[1] * b[1]; M[2, i] = a[1] * b[2]; M[3, i] = a[1] * b[3]
        M[4, i] = a[2] * b[1]; M[5, i] = a[2] * b[2]; M[6, i] = a[2] * b[3]
        M[7, i] = a[3] * b[1]; M[8, i] = a[3] * b[2]; M[9, i] = a[3] * b[3]
    end
    Nm = _e5_nullspace(M)                        # 4×9, rows orthonormal

    coeffs = scratch.coeffs
    _e5_trace_constraints!(coeffs, Nm, scratch.ebufs)
    lhs10 = scratch.lhs10
    rhs10 = scratch.rhs10
    @inbounds for j in 1:10, i in 1:10
        lhs10[i, j] = coeffs[i, j]; rhs10[i, j] = coeffs[i, j + 10]
    end
    rhs = _e5_solve10!(lhs10, rhs10)             # in-place 10×10 solve

    # elimination into A (3×13) using rows 5..10 of the reduced system
    A = scratch.A
    @inbounds for i in 0:2
        r1 = 4 + 2i + 1; r2 = 5 + 2i + 1
        A[i + 1, 1] = -rhs[r2, 1]
        A[i + 1, 2] = rhs[r1, 1] - rhs[r2, 2]
        A[i + 1, 3] = rhs[r1, 2] - rhs[r2, 3]
        A[i + 1, 4] = rhs[r1, 3]
        A[i + 1, 5] = -rhs[r2, 4]
        A[i + 1, 6] = rhs[r1, 4] - rhs[r2, 5]
        A[i + 1, 7] = rhs[r1, 5] - rhs[r2, 6]
        A[i + 1, 8] = rhs[r1, 6]
        A[i + 1, 9] = -rhs[r2, 7]
        A[i + 1, 10] = rhs[r1, 7] - rhs[r2, 8]
        A[i + 1, 11] = rhs[r1, 8] - rhs[r2, 9]
        A[i + 1, 12] = rhs[r1, 9] - rhs[r2, 10]
        A[i + 1, 13] = rhs[r1, 10]
    end

    c = _e5_det_poly!(scratch.c, A)

    roots = scratch.roots
    n_sols = _bisect_sturm10!(roots, c, scratch.fvec, scratch.svec)

    Es = scratch.Es
    n_out = 0
    @inbounds for i in 1:n_sols
        z = roots[i]
        z2 = z * z; z3 = z2 * z; z4 = z2 * z2
        B = SMatrix{3, 2, Float64}(
            A[1, 1] * z3 + A[1, 2] * z2 + A[1, 3] * z + A[1, 4],
            A[2, 1] * z3 + A[2, 2] * z2 + A[2, 3] * z + A[2, 4],
            A[3, 1] * z3 + A[3, 2] * z2 + A[3, 3] * z + A[3, 4],
            A[1, 5] * z3 + A[1, 6] * z2 + A[1, 7] * z + A[1, 8],
            A[2, 5] * z3 + A[2, 6] * z2 + A[2, 7] * z + A[2, 8],
            A[3, 5] * z3 + A[3, 6] * z2 + A[3, 7] * z + A[3, 8]
        )
        bv = SVector{3, Float64}(
            A[1, 9] * z4 + A[1, 10] * z3 + A[1, 11] * z2 + A[1, 12] * z + A[1, 13],
            A[2, 9] * z4 + A[2, 10] * z3 + A[2, 11] * z2 + A[2, 12] * z + A[2, 13],
            A[3, 9] * z4 + A[3, 10] * z3 + A[3, 11] * z2 + A[3, 12] * z + A[3, 13]
        )
        # top-2×2 solve, fall back to least squares on all three rows
        B2 = SMatrix{2, 2, Float64}(B[1, 1], B[2, 1], B[1, 2], B[2, 2])
        xz = inv(B2) * SVector(bv[1], bv[2])
        if abs(B[3, 1] * xz[1] + B[3, 2] * xz[2] - bv[3]) > 1.0e-6
            xz = SMatrix{3, 2, Float64}(B) \ bv
        end
        x = -xz[1]; y = -xz[2]
        inv_norm = 1.0 / sqrt(x * x + y * y + z * z + 1.0)
        e = MVector{9, Float64}(undef)
        for k in 1:9
            e[k] = (Nm[1, k] * x + Nm[2, k] * y + Nm[3, k] * z + Nm[4, k]) * inv_norm
        end
        n_out += 1
        # e is column-major E (entry (i,j) at 3j+i, 0-based) — same layout the
        # EssentialMat tuple constructor expects
        Es[n_out] = EssentialMat{Float64}(Tuple(e))
    end
    # Inline the FixedModels builder (validity filter + zero-pad). No closure:
    # capturing a reassigned variable Core.Box-es it, and reusing the name `z`
    # (the root variable above) had type-poisoned the whole loop into ::Any.
    n_valid = 0
    @inbounds for i in 1:n_out
        is_valid_solution(Es[i]) || continue
        n_valid += 1
        Es[n_valid] = Es[i]
    end
    zm = zero(EssentialMat{Float64})
    @inbounds for i in (n_valid + 1):10
        Es[i] = zm
    end
    return FixedModels{10, EssentialMat{Float64}}(n_valid, Tuple(Es))
end

# Bare default: native backend (PoseLib's overload is opt-in via `PoseLibSolver()`).
essential_5pt(x1, x2) = essential_5pt(NativeSolver(), x1, x2)
