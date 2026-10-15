# =============================================================================
# VisualGeometryCoreChainRulesCoreExt — reverse rules for the B-spline
# kernels (numerics/bspline.jl), plus the constant-plane marker of the
# batched ray-plane intersection (primitives/hyperplane.jl).
#
# The rules live in a package extension so VisualGeometryCore carries no AD
# dependency: they attach automatically when ChainRulesCore is loaded (e.g.
# by Zygote in a downstream calibration pipeline). The pullbacks are
# themselves fused kernels / GEMM contractions, so the reverse sweep stays on
# the backend that owns the primal stacks.
# =============================================================================
module VisualGeometryCoreChainRulesCoreExt

using ChainRulesCore: ChainRulesCore, NoTangent, @non_differentiable
using VisualGeometryCore: VisualGeometryCore
using VisualGeometryCore: bspline_basis_iterative, evaluate_bspline_surface,
    bspline_surface_partial, evaluate_bspline_4d, bspline_4d_partial,
    contract_basis_4d_weighted
using KernelAbstractions: KernelAbstractions, get_backend

# -----------------------------------------------------------------------------
# B-spline kernels (numerics/bspline.jl; hoisted from RayTracedCalibration
# with their rules). The 4D control-point cotangent runs through the
# memory-critical weighted GEMM contraction; the input cotangents come from
# the derivative surfaces.
# -----------------------------------------------------------------------------

function ChainRulesCore.rrule(
        ::typeof(evaluate_bspline_4d), u1::AbstractArray{Float32}, u2::AbstractArray{Float32}, u3::AbstractArray{Float32}, u4::AbstractArray{Float32},
        degrees, k1::AbstractArray{Float32}, k2::AbstractArray{Float32}, k3::AbstractArray{Float32}, k4::AbstractArray{Float32},
        control_points::AbstractArray{Float32, 4}
    )
    result = evaluate_bspline_4d(u1, u2, u3, u4, degrees, k1, k2, k3, k4, control_points)

    function evaluate_bspline_4d_pullback(y)
        # Upstream pullbacks (e.g. getindex) may deliver the cotangent as a
        # ChainRulesCore thunk — materialize it before broadcasting/GEMM.
        y_arr = ChainRulesCore.unthunk(y)
        vjp_control_points = contract_basis_4d_weighted(u1, u2, u3, u4, degrees, k1, k2, k3, k4, y_arr)

        vjp_u1 = bspline_4d_partial(u1, u2, u3, u4, degrees, k1, k2, k3, k4, control_points, 1) .* y_arr
        vjp_u2 = bspline_4d_partial(u1, u2, u3, u4, degrees, k1, k2, k3, k4, control_points, 2) .* y_arr
        vjp_u3 = bspline_4d_partial(u1, u2, u3, u4, degrees, k1, k2, k3, k4, control_points, 3) .* y_arr
        vjp_u4 = bspline_4d_partial(u1, u2, u3, u4, degrees, k1, k2, k3, k4, control_points, 4) .* y_arr

        return NoTangent(), vjp_u1, vjp_u2, vjp_u3, vjp_u4, NoTangent(),
            NoTangent(), NoTangent(), NoTangent(), NoTangent(), vjp_control_points
    end

    return result, evaluate_bspline_4d_pullback
end


function ChainRulesCore.rrule(::typeof(evaluate_bspline_surface), u::AbstractArray{Float32}, v::AbstractArray{Float32}, degree_u, degree_v, knots_u::AbstractArray{Float32}, knots_v::AbstractArray{Float32}, control_points::AbstractArray{Float32})

    result = evaluate_bspline_surface(u, v, degree_u, degree_v, knots_u, knots_v, control_points)

    function evaluate_bspline_surface_pullback(y)
        N_u = bspline_basis_iterative(u, degree_u, knots_u)
        N_v = bspline_basis_iterative(v, degree_v, knots_v)

        n_u = size(N_u, 1)
        n_v = size(N_v, 1)

        grads = KernelAbstractions.zeros(get_backend(u), Float32, n_u * n_v, length(u))
        for i in 1:n_u
            for j in 1:n_v
                grads[(j - 1) * n_u + i, :] = N_u[i] .* N_v[j]
            end
        end

        vjp_control_points = reshape(grads * vec(y), n_u, n_v)

        d_spline_d_u = bspline_surface_partial(u, v, degree_u, degree_v, knots_u, knots_v, control_points, 1)
        d_spline_d_v = bspline_surface_partial(u, v, degree_u, degree_v, knots_u, knots_v, control_points, 2)

        vjp_u = d_spline_d_u .* y
        vjp_v = d_spline_d_v .* y

        return NoTangent(), vjp_u, vjp_v, NoTangent(), NoTangent(), NoTangent(), NoTangent(), vjp_control_points
    end

    return result, evaluate_bspline_surface_pullback
end

# -----------------------------------------------------------------------------
# Batched ray-plane intersection: the plane is a constant of the
# computation — its scalar extraction never carries derivatives (see the
# `ray_plane_intersection` docstring).
# -----------------------------------------------------------------------------

@non_differentiable VisualGeometryCore.Primitives._plane_constants(::Any)

end # module VisualGeometryCoreChainRulesCoreExt
