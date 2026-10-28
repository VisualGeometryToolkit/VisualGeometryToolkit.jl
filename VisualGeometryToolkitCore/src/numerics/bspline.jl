# =============================================================================
# B-spline kernels: the scalar basis (hoisted from RadiometricCamera) and the
# broadcast kernels (0.14; hoisted from RayTracedCalibration src/bspline.jl)
#
# ONLY what BSplineKit cannot give a GPU kernel lives here: the isbits scalar
# basis on a static knot vector and the device-side Float32 surface and 4-D
# evaluations, each tested EQUAL to BSplineKit on the same knots. Everything
# that runs on the CPU — knot vectors (`BSplineBasis`), evaluation and
# derivatives, the Galerkin (Gram) matrices of the bending energy, the natural
# boundary condition (`RecombinedBSplineBasis`) — is BSplineKit's, read
# through bspline_energy.jl.
#
# The scalar Cox–de Boor basis `bspline_basis` (isbits, any element type) and,
# for structure-of-arrays inputs, its broadcast, the tensor-product surface and
# its derivative surfaces, and the 4D tensor-product spline with its
# memory-disciplined GEMM factorization. Backends come from the array type
# via KernelAbstractions/Adapt — no vendor package is referenced (the CUDA
# import of the pre-hoist copy is gone; the kernels always ran through the
# generic broadcast/GEMM path).
#
# Precision: the array kernels are Float32-typed end to end — they are the
# render currency of the downstream GPU calibration pipeline. Their basis is
# the scalar one, broadcast (one Cox–de Boor, exact against the recursion);
# it replaced a separate broadcast recursion that clamped u to
# knots[end] − 1e-8 and read zero left of the knots, which moved values by up
# to 1.2e-7 at Float32.
#
# The ChainRulesCore rrules of `evaluate_bspline_surface` and
# `evaluate_bspline_4d` live in the VisualGeometryToolkitCoreChainRulesCoreExt
# package extension, with the batched-kernel rules.
# =============================================================================

using Adapt: adapt

# ---------------------------------------------------------------------------
# The scalar basis (hoisted from RadiometricCamera src/camera/bspline.jl,
# `N_ip`, its arithmetic unchanged; its knot helper is BSplineKit's): Cox–de Boor
# (Piegl & Tiller 1997) on a static knot vector,
#
#   N_{i,0}(x) = 1 if k_i ≤ x < k_{i+1}, else 0
#   N_{i,p}(x) = (x − k_i)/(k_{i+p} − k_i)·N_{i,p−1}(x)
#              + (k_{i+p+1} − x)/(k_{i+p+1} − k_{i+1})·N_{i+1,p−1}(x)
#
# with 0/0 := 0 and the last non-empty span closed on the right, so the
# partition of unity holds on the closed interval. Fully unrolled for the
# degree: isbits in, an SVector out, no closures or loops over tuples — it
# infers on the CPU and in a GPU kernel alike, in any element type (Float32,
# Float64, ForwardDiff Duals).
# ---------------------------------------------------------------------------

"""
    bspline_basis(knots::SVector{K}, Val(p), x) -> SVector{K − p − 1}
    bspline_basis(knots::SVector{K}, p::Integer, x)

Every degree-`p` B-spline basis function on `knots` at `x`,
`N_{1,p}(x), …, N_{K−p−1,p}(x)` (Cox–de Boor, Piegl & Tiller 1997): `0/0` is
`0`, and the last non-empty knot span is closed on the right, so the values
sum to one on the whole closed interval `[knots[1], knots[K]]`. `x` outside
it is clamped onto it (constant extrapolation).

The basis (its knots and degree) leads, the point follows (`signatures.md`).
The degree is a `Val` so the result's size is static and nothing allocates;
the `Integer` form is the convenience. Scalar, isbits and generic in the
element type (`promote_type` of `x`'s and the knots'): it runs inside a
KernelAbstractions kernel, in `Float32`, and under ForwardDiff. Why it is
Core's and not BSplineKit's: a kernel needs an isbits basis on a static
knot vector with a static result; on the CPU, BSplineKit's `evaluate_all`
gives the same values (the tests check it).
[`bspline_basis_iterative`](@ref) is its broadcast over an array.
"""
@inline bspline_basis(knots::SVector, p::Integer, x) = bspline_basis(knots, Val(Int(p)), x)

@generated function bspline_basis(knots::SVector{K, TK}, ::Val{p}, x) where {p, K, TK}
    # fully unrolled Cox–de Boor (no closures/tuple loops — inferred on CPU and GPU alike)
    n1 = K - p - 1
    ex = Expr[]
    for i in 1:(K - 1)
        push!(ex, :($(Symbol(:N0_, i)) = ((knots[$i] <= xc < knots[$(i + 1)]) || (xc == hi && $i == last_span)) ? one(T) : zero(T)))
    end
    for d in 1:p, i in 1:(K - 1 - d)
        prev1 = Symbol(:N, d - 1, :_, i)
        prev2 = Symbol(:N, d - 1, :_, i + 1)
        push!(
            ex, quote
                den1 = knots[$(i + d)] - knots[$i]
                den2 = knots[$(i + d + 1)] - knots[$(i + 1)]
                $(Symbol(:N, d, :_, i)) = (den1 > 0 ? (xc - knots[$i]) / den1 * $prev1 : zero(T)) +
                    (den2 > 0 ? (knots[$(i + d + 1)] - xc) / den2 * $prev2 : zero(T))
            end
        )
    end
    outs = [Symbol(:N, p, :_, i) for i in 1:n1]
    return quote
        T = promote_type(typeof(x), TK)
        lo = knots[1]; hi = knots[K]
        xc = clamp(x, lo, hi)
        last_span = _last_span(knots)
        $(ex...)
        return SVector{$n1, T}($(outs...))
    end
end

# The index of the last non-empty knot span (for the right-end closure).
@inline function _last_span(knots::SVector{K}) where {K}
    i = K - 1
    while i > 1 && !(knots[i] < knots[i + 1])
        i -= 1
    end
    return i
end

"""
    bspline_basis_iterative(u, degree, knots) -> Vector{<:AbstractArray{Float32}}

For each of the `n_basis = length(knots) − degree − 1` basis functions, its
values at every point of the device array `u`: [`bspline_basis`](@ref)
broadcast over `u` (on `u`'s backend) in `Float32`, split per basis function
— the structure-of-arrays form the tensor-product kernels below consume. `u`
outside the knots is clamped onto them, as by `bspline_basis`.

Kept because the tensor-product kernels below run on the device array's
backend; validated against `BSplineKit.BSplineBasis` on the downstream demo's
actual θ knot vector — see `test/bspline_tests.jl`.
"""
function bspline_basis_iterative(u, degree, knots)
    k = SVector{length(knots)}(Float32.(Array(knots)))
    B = bspline_basis.(Ref(k), Ref(Val(Int(degree))), u)
    n_basis = length(knots) - degree - 1
    N = Vector{AbstractArray{Float32}}(undef, n_basis)
    for i in 1:n_basis
        N[i] = getindex.(B, i)
    end
    return N
end


"""
    evaluate_bspline_surface(u, v, degree_u, degree_v, knots_u, knots_v, control_points)

Tensor-product B-spline surface evaluation on the compute device (kept for
the device: BSplineKit evaluates on the CPU; tested equal to its `Spline`
evaluation on the same knots). Inputs
`u, v` are broadcasted device arrays of equal shape; the surface is
evaluated point-wise and an array of the same shape is returned.
Differentiable wrt `u`, `v`, and the control points via the ChainRulesCore
rrule (package extension). Periodicity in the second parameter direction is
handled by the CALLER tying the wrap control-point columns — no knot
wrapping happens here.
"""
function evaluate_bspline_surface(u::AbstractArray{Float32}, v::AbstractArray{Float32}, degree_u, degree_v, knots_u::AbstractArray{Float32}, knots_v::AbstractArray{Float32}, control_points::AbstractArray{Float32})

    N_u = bspline_basis_iterative(u, degree_u, knots_u)
    n_u = length(N_u)

    N_v = bspline_basis_iterative(v, degree_v, knots_v)

    result = KernelAbstractions.zeros(get_backend(u), Float32, size(u)...)


    for i in 1:n_u
        for j in 1:length(N_v)
            cp_ij = control_points[i, j]

            result .+= N_u[i] .* N_v[j] .* cp_ij
        end
    end

    return result
end


"""
    bspline_surface_partial(u, v, degree_u, degree_v, knots_u, knots_v, control_points, along)

First partial derivative of a B-spline surface along parameter direction
`along` (1 = u, 2 = v), via the derivative-surface trick: difference the
control points, drop one knot at each end, evaluate at degree − 1 along the
differentiated direction. Shared by the `evaluate_bspline_surface` rrule
(package extension) and the downstream spline-lens Jacobians.
"""
function bspline_surface_partial(u::AbstractArray{Float32}, v::AbstractArray{Float32}, degree_u, degree_v, knots_u::AbstractArray{Float32}, knots_v::AbstractArray{Float32}, control_points::AbstractArray{Float32}, along::Integer)
    if along == 1
        knots_u_d = knots_u[2:(end - 1)]
        cp_d = degree_u .* (control_points[2:end, :] .- control_points[1:(end - 1), :]) ./ reshape(knots_u_d[(degree_u + 1):end] .- knots_u_d[1:(end - degree_u)], :, 1)
        return evaluate_bspline_surface(u, v, degree_u - 1, degree_v, knots_u_d, knots_v, cp_d)
    else
        knots_v_d = knots_v[2:(end - 1)]
        cp_d = degree_v .* (control_points[:, 2:end] .- control_points[:, 1:(end - 1)]) ./ reshape(knots_v_d[(degree_v + 1):end] .- knots_v_d[1:(end - degree_v)], 1, :)
        return evaluate_bspline_surface(u, v, degree_u, degree_v - 1, knots_u, knots_v_d, cp_d)
    end
end


# ---------------------------------------------------------------------------
# 4D tensor-product B-spline (finite-aperture lenses: inputs a_x, a_y, d_x, d_y)
#
# Memory discipline: the dense per-sample basis-product tensor (n1·n2·n3·n4, N)
# is NEVER materialized (5⁴ = 625 rows × 5e5 samples ≈ 1.3 GB). All value /
# VJP paths factor the 4D basis into two PAIR matrices
#   A = (N, n1·n2) with column (j−1)·n1 + i  = N1_i · N2_j
#   B = (N, n3·n4) with column (l−1)·n3 + k  = N3_k · N4_l
# and contract through GEMMs on the owning backend. The (j−1)·n1+i column
# order is the column-major flattening of the (n1, n2) index pair, so a
# (n1·n2, n3·n4) GEMM result reshapes DIRECTLY into the canonical
# (n1, n2, n3, n4) control-point array — same vec(cp) layout the 2D rrule
# uses.
# ---------------------------------------------------------------------------

function _basis_pair_matrix(Na::Vector{<:AbstractArray{Float32}}, Nb::Vector{<:AbstractArray{Float32}})
    na = length(Na)
    nb = length(Nb)
    N = length(Na[1])
    M = KernelAbstractions.zeros(get_backend(Na[1]), Float32, N, na * nb)
    for j in 1:nb, i in 1:na
        M[:, (j - 1) * na + i] = vec(Na[i] .* Nb[j])
    end
    return M
end

function _basis_pair_4d(
        u1::AbstractArray{Float32}, u2::AbstractArray{Float32}, u3::AbstractArray{Float32}, u4::AbstractArray{Float32},
        degrees, k1::AbstractArray{Float32}, k2::AbstractArray{Float32}, k3::AbstractArray{Float32}, k4::AbstractArray{Float32}
    )
    N1 = bspline_basis_iterative(vec(u1), degrees[1], k1)
    N2 = bspline_basis_iterative(vec(u2), degrees[2], k2)
    N3 = bspline_basis_iterative(vec(u3), degrees[3], k3)
    N4 = bspline_basis_iterative(vec(u4), degrees[4], k4)
    A = _basis_pair_matrix(N1, N2)
    B = _basis_pair_matrix(N3, N4)
    return A, B, (length(N1), length(N2), length(N3), length(N4))
end


"""
    evaluate_bspline_4d(u1, u2, u3, u4, degrees, k1, k2, k3, k4, control_points)
        -> AbstractArray{Float32}

4D tensor-product B-spline: S(u) = Σ_ijkl N1_i(u1)·N2_j(u2)·N3_k(u3)·N4_l(u4)·cp[i,j,k,l].
The four inputs must share one shape (callers materialize broadcasts first);
the result has that shape. `degrees` is indexable per dim (NTuple or
Vector). Differentiable wrt the inputs and the control points via the
ChainRulesCore rrule (package extension).
"""
function evaluate_bspline_4d(
        u1::AbstractArray{Float32}, u2::AbstractArray{Float32}, u3::AbstractArray{Float32}, u4::AbstractArray{Float32},
        degrees, k1::AbstractArray{Float32}, k2::AbstractArray{Float32}, k3::AbstractArray{Float32}, k4::AbstractArray{Float32},
        control_points::AbstractArray{Float32, 4}
    )
    A, B, dims = _basis_pair_4d(u1, u2, u3, u4, degrees, k1, k2, k3, k4)
    M = reshape(control_points, dims[1] * dims[2], dims[3] * dims[4])
    T = B * transpose(M)                       # (N, n1·n2)
    return reshape(sum(A .* T, dims = 2), size(u1))
end


# Σ_s w_s · (4D basis product at sample s), returned in control-point shape —
# weights are folded to per-sample scalars FIRST, so only one
# (n1·n2, N)×(N, n3·n4) GEMM remains.
function _contract_pair_weighted(A::AbstractArray{Float32}, B::AbstractArray{Float32}, w::AbstractArray{Float32}, dims::NTuple{4, Int})
    V = transpose(A .* w) * B                  # (n1·n2, n3·n4)
    return reshape(V, dims)
end

"""
    contract_basis_4d_weighted(u1, u2, u3, u4, degrees, k1, k2, k3, k4, w)

`Σ_s w_s · (4D basis product at sample s)`, returned in control-point shape —
the single memory-critical primitive behind every 4D control-point gradient
(the 4D spline rrule and the downstream boundary/render accumulations).
Weights are folded to per-sample scalars FIRST, so only one
`(n1·n2, N)×(N, n3·n4)` GEMM remains and the dense per-sample basis tensor
is never materialized.
"""
function contract_basis_4d_weighted(
        u1::AbstractArray{Float32}, u2::AbstractArray{Float32}, u3::AbstractArray{Float32}, u4::AbstractArray{Float32},
        degrees, k1::AbstractArray{Float32}, k2::AbstractArray{Float32}, k3::AbstractArray{Float32}, k4::AbstractArray{Float32},
        w::AbstractArray
    )
    A, B, dims = _basis_pair_4d(u1, u2, u3, u4, degrees, k1, k2, k3, k4)
    w_vec = vec(w)
    backend = get_backend(u1)
    w_dev = (eltype(w_vec) === Float32 && get_backend(w_vec) == backend) ?
        w_vec : adapt(backend, Float32.(w_vec))
    return _contract_pair_weighted(A, B, w_dev, dims)
end


# Derivative-surface control points along `dim`: difference the CPs, drop one
# knot at each end (generalizes the 2D derivative-surface trick). Plain
# slicing/broadcast so AD can differentiate through it wrt `cp` inside a
# tracked chain.
function _derivative_cp_4d(cp::AbstractArray{Float32, 4}, degree, knots::AbstractArray{Float32}, dim::Int)
    p = Int(degree)
    kd = knots[2:(end - 1)]
    denom = kd[(p + 1):end] .- kd[1:(end - p)]
    if dim == 1
        cp_d = p .* (cp[2:end, :, :, :] .- cp[1:(end - 1), :, :, :]) ./ reshape(denom, :, 1, 1, 1)
    elseif dim == 2
        cp_d = p .* (cp[:, 2:end, :, :] .- cp[:, 1:(end - 1), :, :]) ./ reshape(denom, 1, :, 1, 1)
    elseif dim == 3
        cp_d = p .* (cp[:, :, 2:end, :] .- cp[:, :, 1:(end - 1), :]) ./ reshape(denom, 1, 1, :, 1)
    else
        cp_d = p .* (cp[:, :, :, 2:end] .- cp[:, :, :, 1:(end - 1)]) ./ reshape(denom, 1, 1, 1, :)
    end
    return cp_d, kd
end


"""
    bspline_4d_partial(u1, u2, u3, u4, degrees, k1, k2, k3, k4, control_points, dim)

∂S/∂u_dim of the 4D spline at the sample points, via the derivative surface
(degree − 1 along `dim`, differenced CPs, trimmed knots). AD-transparent
wrt `control_points`.
"""
function bspline_4d_partial(
        u1::AbstractArray{Float32}, u2::AbstractArray{Float32}, u3::AbstractArray{Float32}, u4::AbstractArray{Float32},
        degrees, k1::AbstractArray{Float32}, k2::AbstractArray{Float32}, k3::AbstractArray{Float32}, k4::AbstractArray{Float32},
        control_points::AbstractArray{Float32, 4}, dim::Int
    )
    if dim == 1
        cp_d, kd = _derivative_cp_4d(control_points, degrees[1], k1, 1)
        return evaluate_bspline_4d(u1, u2, u3, u4, (degrees[1] - 1, degrees[2], degrees[3], degrees[4]), kd, k2, k3, k4, cp_d)
    elseif dim == 2
        cp_d, kd = _derivative_cp_4d(control_points, degrees[2], k2, 2)
        return evaluate_bspline_4d(u1, u2, u3, u4, (degrees[1], degrees[2] - 1, degrees[3], degrees[4]), k1, kd, k3, k4, cp_d)
    elseif dim == 3
        cp_d, kd = _derivative_cp_4d(control_points, degrees[3], k3, 3)
        return evaluate_bspline_4d(u1, u2, u3, u4, (degrees[1], degrees[2], degrees[3] - 1, degrees[4]), k1, k2, kd, k4, cp_d)
    else
        cp_d, kd = _derivative_cp_4d(control_points, degrees[4], k4, 4)
        return evaluate_bspline_4d(u1, u2, u3, u4, (degrees[1], degrees[2], degrees[3], degrees[4] - 1), k1, k2, k3, kd, cp_d)
    end
end
