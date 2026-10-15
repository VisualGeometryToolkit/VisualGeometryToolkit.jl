# =============================================================================
# Native P3P — line-by-line port of PoseLib's p3p.cc (Ding et al. formulation)
# =============================================================================
#
# Source of truth: PoseLib/solvers/p3p.cc + p3p_common.h + misc/univariate.cc
# (Larsson's PoseLib, solver by Yaqing Ding & Mark Shachkov, BSD-3). This is a
# faithful port — same operations in the same order, same tolerances, same
# lenient/degenerate behavior — so it can be validated solution-for-solution
# against the compiled original via `p3p(PoseLibSolver(), …)`.
#
# Contract (identical to the PoseLib backend):
#   • `x` are UNIT bearing vectors (dot products are used as cosines — the
#     caller normalizes, exactly as with the FFI wrapper which forwards them raw).
#   • Returns `FixedModels{4, Extrinsics3}` with x_cam = R·X + t.
#
# `p3p(NativeSolver(), x, X)` selects this backend explicitly; the bare
# `p3p(x, X)` default below routes here (PoseLib's C++ backend is opt-in via
# `p3p(PoseLibSolver(), …)` after `using PoseLib`).

# C++ float semantics: std::sqrt returns NaN outside its domain
# (the NaN flows through and the resulting pose fails FixedModels validity),
# whereas Julia throws. This shim reproduces the C++ behavior exactly.
@inline _nan_sqrt(x::Float64) = x < 0.0 ? NaN : sqrt(x)

# --- univariate.cc::solve_cubic_single_real ---------------------------------
# The monic cubic s³ + c2·s² + c1·s + c0's root Lambda Twist continues from, on
# VGC's one cubic solver (`_real_cubic_roots`, fundmat.jl): the Cardano root when
# it is the only real one (G = true), else the largest of the three (the
# trigonometric k = 0 root, G = false). The same formulas as PoseLib's for the
# monic case — the Cardano branch bit for bit, the trigonometric branch to
# rounding (the port is validated against PoseLib at 1e-8, not bitwise).
@inline function _p3p_cubic_single_real(c2::Float64, c1::Float64, c0::Float64)
    r = _real_cubic_roots(1.0, c2, c1, c0)
    return r[1], isnan(r[2])
end

# --- p3p_common.h::root2real -------------------------------------------------
@inline function _p3p_root2real(b::Float64, c::Float64)
    THRESHOLD = -1.0e-12
    v = b * b - 4.0 * c
    if v < THRESHOLD
        r = -0.5 * b
        return v >= 0.0, r, r
    end
    if v > THRESHOLD && v < 0.0
        return true, -0.5 * b, -2.0
    end
    y = sqrt(v)
    if b < 0.0
        return true, 0.5 * (-b + y), 0.5 * (-b - y)
    else
        return true, 2.0 * c / (-b + y), 2.0 * c / (-b - y)
    end
end

# --- p3p_common.h::compute_pq -------------------------------------------------
# C is symmetric (built that way by the caller). Returns the two planes
# (pq0 = modified C col 0, pq1 = modified C row 0).
@inline function _p3p_compute_pq(C::SMatrix{3, 3, Float64, 9})
    adj00 = C[2, 3] * C[3, 2] - C[2, 2] * C[3, 3]
    adj11 = C[1, 3] * C[3, 1] - C[1, 1] * C[3, 3]
    adj22 = C[1, 2] * C[2, 1] - C[1, 1] * C[2, 2]
    adj01 = C[1, 2] * C[3, 3] - C[1, 3] * C[3, 2]
    adj02 = C[1, 3] * C[2, 2] - C[1, 2] * C[2, 3]
    adj12 = C[1, 1] * C[2, 3] - C[1, 3] * C[2, 1]

    # v = column of the adjugate with the largest diagonal, scaled by
    # 1/sqrt(diag) — adjugate columns: col0=(adj00,adj01,adj02),
    # col1=(adj01,adj11,adj12), col2=(adj02,adj12,adj22) (symmetric).
    local v::SVector{3, Float64}
    if adj00 > adj11
        if adj00 > adj22
            v = SVector(adj00, adj01, adj02) / _nan_sqrt(adj00)
        else
            v = SVector(adj02, adj12, adj22) / _nan_sqrt(adj22)
        end
    elseif adj11 > adj22
        v = SVector(adj01, adj11, adj12) / _nan_sqrt(adj11)
    else
        v = SVector(adj02, adj12, adj22) / _nan_sqrt(adj22)
    end

    # C ± skew(v):  C01 -= v3; C02 += v2; C12 -= v1; C10 += v3; C20 -= v2; C21 += v1
    pq0 = SVector(C[1, 1], C[2, 1] + v[3], C[3, 1] - v[2])   # modified column 0
    pq1 = SVector(C[1, 1], C[1, 2] - v[3], C[1, 3] + v[2])   # modified row 0
    return pq0, pq1
end

# --- p3p_common.h::refine_lambda (5 Newton steps on the three constraints) ---
@inline function _p3p_refine_lambda(
        λ1::Float64, λ2::Float64, λ3::Float64,
        a12::Float64, a13::Float64, a23::Float64,
        b12::Float64, b13::Float64, b23::Float64
    )
    for _ in 1:5
        r1 = λ1 * λ1 - 2.0 * λ1 * λ2 * b12 + λ2 * λ2 - a12
        r2 = λ1 * λ1 - 2.0 * λ1 * λ3 * b13 + λ3 * λ3 - a13
        r3 = λ2 * λ2 - 2.0 * λ2 * λ3 * b23 + λ3 * λ3 - a23
        if abs(r1) + abs(r2) + abs(r3) < 1.0e-10
            return λ1, λ2, λ3
        end
        x11 = λ1 - λ2 * b12
        x12 = λ2 - λ1 * b12
        x21 = λ1 - λ3 * b13
        x23 = λ3 - λ1 * b13
        x32 = λ2 - λ3 * b23
        x33 = λ3 - λ2 * b23
        detJ = 0.5 / (x11 * x23 * x32 + x12 * x21 * x33)
        λ1 += (-x23 * x32 * r1 - x12 * x33 * r2 + x12 * x23 * r3) * detJ
        λ2 += (-x21 * x33 * r1 + x11 * x33 * r2 - x11 * x23 * r3) * detJ
        λ3 += (x21 * x32 * r1 - x11 * x32 * r2 - x12 * x21 * r3) * detJ
    end
    return λ1, λ2, λ3
end

"""
    p3p(::NativeSolver, x, X) -> FixedModels{4, Extrinsics3}

Perspective-3-Point, native Julia (port of PoseLib's `p3p.cc`). `x` must be
three **unit** bearing vectors; `X` the corresponding 3D points. Up to four
poses with `x_cam = R·X + t`.
"""
function p3p(
        ::NativeSolver,
        x_in::AbstractVector{S},
        X_in::AbstractVector{T}
    ) where {
        S <: StaticVector{3, Float64},
        T <: StaticVector{3, Float64},
    }
    (length(x_in) == 3 && length(X_in) == 3) ||
        throw(ArgumentError("p3p requires exactly 3 bearing/point correspondences, got $(length(x_in))/$(length(X_in))"))

    X0 = SVector{3, Float64}(X_in[1]); X1 = SVector{3, Float64}(X_in[2]); X2 = SVector{3, Float64}(X_in[3])
    x0 = SVector{3, Float64}(x_in[1]); x1 = SVector{3, Float64}(x_in[2]); x2 = SVector{3, Float64}(x_in[3])

    X01 = X0 - X1
    X02 = X0 - X2
    X12 = X1 - X2

    a01 = dot(X01, X01)
    a02 = dot(X02, X02)
    a12 = dot(X12, X12)

    # Switch so BC is the largest distance among {X01, X02, X12}
    if a01 > a02
        if a01 > a12
            x0, x2 = x2, x0
            X0, X2 = X2, X0
            a01, a12 = a12, a01
            X01 = -X12
            X02 = -X02
        end
    elseif a02 > a12
        x0, x1 = x1, x0
        X0, X1 = X1, X0
        a02, a12 = a12, a02
        X01 = -X01
        X02 = X12
    end

    a12d = 1.0 / a12
    a = a01 * a12d
    b = a02 * a12d

    m01 = dot(x0, x1)
    m02 = dot(x0, x2)
    m12 = dot(x1, x2)

    m12sq = -m12 * m12 + 1.0
    m02sq = -1.0 + m02 * m02
    m01sq = -1.0 + m01 * m01
    ab = a * b
    bsq = b * b
    asq = a * a
    m013 = -2.0 + 2.0 * m01 * m02 * m12
    bsqm12sq = bsq * m12sq
    asqm12sq = asq * m12sq
    abm12sq = 2.0 * ab * m12sq

    k3_inv = 1.0 / (bsqm12sq + b * m02sq)
    k2 = k3_inv * ((-1.0 + a) * m02sq + abm12sq + bsqm12sq + b * m013)
    k1 = k3_inv * (asqm12sq + abm12sq + a * m013 + (-1.0 + b) * m01sq)
    k0 = k3_inv * (asqm12sq + a * m01sq)

    s, G = _p3p_cubic_single_real(k2, k1, k0)

    C = SMatrix{3, 3, Float64, 9}(
        -a + s * (1.0 - b), -m02 * s, a * m12 + b * m12 * s,   # col 1
        -m02 * s, s + 1.0, -m01,                    # col 2
        a * m12 + b * m12 * s, -m01, -a - b * s + 1.0
    )        # col 3

    pq = _p3p_compute_pq(C)

    XXm = hcat(X01, X02, cross(X01, X02))
    XX = inv(XXm)

    # Immutable NTuple accumulator, updated through `Base.setindex` — NOT an
    # `MVector(undef)` scratch: the MVector escaped LLVM's alloc-elision here
    # and heap-allocated once per solve, i.e. once per RANSAC trial (~15 MB
    # and ~38k GC objects per BlobBoards frame, measured 2026-08-17). Tuple
    # setindex on isbits elements is splat-generated and stays on the stack.
    z = zero(Extrinsics3)
    poses = (z, z, z, z)
    n_sols = 0

    for i in 1:2
        p0 = pq[i][1]
        p1 = pq[i][2]
        p2 = pq[i][3]
        switch_12 = abs(p0) <= abs(p1)

        if switch_12
            w0 = -p0 / p1
            w1 = -p2 / p1
            ca = 1.0 / (w1 * w1 - b)
            cb = 2.0 * (b * m12 - m02 * w1 + w0 * w1) * ca
            cc = (w0 * w0 - 2.0 * m02 * w0 - b + 1.0) * ca
            ok, τa, τb = _p3p_root2real(cb, cc)
            ok || continue
            for τ in (τa, τb)
                τ <= 0.0 && continue
                d2 = _nan_sqrt(a12 / (τ * (τ - 2.0 * m12) + 1.0))
                d1 = τ * d2
                d0 = w0 * d2 + w1 * d1
                d0 < 0.0 && continue
                d0, d1, d2 = _p3p_refine_lambda(d0, d1, d2, a01, a02, a12, m01, m02, m12)
                v1 = d0 * x0 - d1 * x1
                v2 = d0 * x0 - d2 * x2
                R = hcat(v1, v2, cross(v1, v2)) * XX
                n_sols += 1
                n_sols <= 4 && (poses = Base.setindex(poses, Extrinsics3(R, Vec3{Float64}(d0 * x0 - R * X0)), n_sols))
            end
        else
            w0 = -p1 / p0
            w1 = -p2 / p0
            ca = 1.0 / (-a * w1 * w1 + 2.0 * a * m12 * w1 - a + 1.0)
            cb = 2.0 * (a * m12 * w0 - m01 - a * w0 * w1) * ca
            cc = (1.0 - a * w0 * w0) * ca
            ok, τa, τb = _p3p_root2real(cb, cc)
            ok || continue
            for τ in (τa, τb)
                τ <= 0.0 && continue
                d0 = _nan_sqrt(a01 / (τ * (τ - 2.0 * m01) + 1.0))
                d1 = τ * d0
                d2 = w0 * d0 + w1 * d1
                d2 < 0.0 && continue
                d0, d1, d2 = _p3p_refine_lambda(d0, d1, d2, a01, a02, a12, m01, m02, m12)
                v1 = d0 * x0 - d1 * x1
                v2 = d0 * x0 - d2 * x2
                R = hcat(v1, v2, cross(v1, v2)) * XX
                n_sols += 1
                n_sols <= 4 && (poses = Base.setindex(poses, Extrinsics3(R, Vec3{Float64}(d0 * x0 - R * X0)), n_sols))
            end
        end

        (n_sols > 0 && G) && break
    end

    # Inline the FixedModels builder (validity filter + zero-pad) without the
    # closure capture — keeps the RANSAC hot path allocation-free.
    n = min(n_sols, 4)
    n_valid = 0
    data = (z, z, z, z)
    for i in 1:n
        @inbounds p = poses[i]
        is_valid_solution(p) || continue
        n_valid += 1
        data = Base.setindex(data, p, n_valid)
    end
    return FixedModels{4, Extrinsics3}(n_valid, data)
end

# Bare default: native backend (PoseLib 0.3+ no longer provides this forward).
p3p(
    x_in::AbstractVector{<:StaticVector{3, Float64}},
    X_in::AbstractVector{<:StaticVector{3, Float64}}
) = p3p(NativeSolver(), x_in, X_in)
