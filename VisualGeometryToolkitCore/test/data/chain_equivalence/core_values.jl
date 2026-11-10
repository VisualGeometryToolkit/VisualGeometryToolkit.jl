# =============================================================================
# Core's chain-equivalence values: what `core_values.txt` pins, computed on the
# code at hand through an adapter (corpus.jl). Included after corpus.jl by the
# generator and by VisualGeometryToolkitCore's equivalence test item.
# =============================================================================
#
# Every value is in the code's own order (a composition's parameters as the
# code stacks them); the reader permutes the goldens, never these.

using StaticArrays: SVector, SMatrix
using Rotations: QuatRotation
using FlexUnits: ustrip
import VisualGeometryToolkitCore as VgtkCore
using VisualGeometryToolkitCore: m, mm, parameter_vector, parameter_names, parameter_bounds, with_parameters, regularizer,
    length_mask, is_planar, is_affine, has_units, with_unit, PlanarTarget, PosedTarget,
    TargetManifold, tangent_size, retract_jacobian, action_jacobian, retract, EuclideanMap, surface, project, backproject,
    distort_normalized, undistort_normalized, monotone_radius, view_angle_limit, convert_eltype, LensManifold, lens_record,
    CameraModel, valid_domain, LogicalIntrinsics, lens, transform_deriv, transform_deriv_params

columns(ps) = isempty(ps) ? zeros(0, 0) : reduce(hcat, [Float64.(collect(p)) for p in ps])
slices(Js) = cat([Float64.(Matrix(J)) for J in Js]...; dims = 3)
plain_metres(p) = map(x -> ustrip(m, x), p)
attempt(f) = try
    f()
catch e
    e isa Union{ArgumentError, MethodError, DomainError} || rethrow()
    sprint(showerror, e)
end
# a record as a canonical text: keys sorted, numbers by repr
canonical(x::AbstractDict) = "{" * join([repr(String(k)) * ":" * canonical(x[k]) for k in sort(collect(keys(x)); by = String)], ",") * "}"
canonical(x::AbstractVector) = "[" * join(canonical.(x), ",") * "]"
canonical(x::Real) = repr(Float64(x))
canonical(x) = repr(x)

# The module a value is printed relative to: the goldens were printed from a
# session's `Main` that had done `using VisualGeometryToolkitCore` (exported
# names short, the rest qualified), whatever module includes this file.
const _REPR_MODULE = Module(:ChainEquivalenceRepr)
Core.eval(_REPR_MODULE, :(using VisualGeometryToolkitCore))

const TARGET_POSE_ROTATION = QuatRotation(0.9128709291752769, 0.18257418583505536, 0.36514837167011072, 0.0, false)

"""
    target_values(P, V) -> Vector{Pair{String, Any}}

Every calibration case's values but the retired cases' (`V.retired`), in the
code's order.
"""
function target_values(P, V)
    out = Pair{String, Any}[]
    emit(k, v) = push!(out, k => v)
    pts, ppts = design_points(), physical_points()
    jpts = pts[1:jacobian_count]
    physical = Dict(target_cases(P, V; physical = true))
    for (name, c) in target_cases(P, V)
        name in V.retired && continue
        k = "target/$name"
        cp = physical[name]
        θ = parameter_vector(c)
        p = block_permutation(target_blocks(name, 1), V.ct_order)      # code = golden[p]
        δ = P["param/target/$name/delta"][p]
        emit("$k/names", parameter_names(c))
        V.tag(c) === nothing || emit("$k/tag", V.tag(c))
        emit("$k/length_mask", collect(length_mask(c)))
        emit("$k/is_planar", is_planar(c))
        emit("$k/is_affine", is_affine(c))
        emit("$k/theta", collect(θ))
        lo, hi = parameter_bounds(c)
        emit("$k/bounds_lo", collect(lo))
        emit("$k/bounds_hi", collect(hi))
        emit("$k/out", columns(c.(pts)))
        emit("$k/out_physical", columns(plain_metres.(cp.(ppts[1:50]))))
        emit("$k/jacobian_theta", slices([V.jacobian_theta(c, x) for x in jpts]))
        emit("$k/jacobian_x", slices([V.jacobian_x(c, x) for x in jpts]))
        emit("$k/regularizer", collect(regularizer(c)))
        c2 = with_parameters(c, θ + δ)
        emit("$k/with_parameters/theta", collect(parameter_vector(c2)))
        emit("$k/with_parameters/out", columns(c2.(pts[1:50])))
        emit("$k/has_units_physical", has_units(cp))
        emit("$k/with_unit_mm/theta", collect(parameter_vector(with_unit(c, mm))))
        emit("$k/ustrip_mm/theta", collect(parameter_vector(ustrip(mm, cp))))
        t = PlanarTarget(pts; calibration = c)
        ch = TargetManifold(t)
        B = Matrix(retract_jacobian(ch))
        emit("$k/manifold/size", Float64(tangent_size(ch)))
        emit("$k/manifold/action", slices([Matrix(action_jacobian(ch, x)) * B' for x in pts[1:3]]))
        emit("$k/manifold/retract", collect(parameter_vector(retract(ch, B' * δ))))
        X = EuclideanMap(TARGET_POSE_ROTATION, SVector(0.1, -0.05, 0.9))
        s = attempt(() -> surface(PosedTarget(t, X)))
        if s isa AbstractString
            emit("$k/surface/error", s)
        else
            q = s isa VgtkCore.Cameras.Quad ? s : s.quad
            emit("$k/surface/type", string(nameof(typeof(s))))
            emit("$k/surface/A", Matrix(q.A))
            emit("$k/surface/b", collect(q.b))
        end
    end
    return out
end

"""
    lens_values(P, V) -> Vector{Pair{String, Any}}

Every lens case's values, in the code's order.
"""
function lens_values(P, V)
    out = Pair{String, Any}[]
    emit(k, v) = push!(out, k => v)
    K = SMatrix{3, 3}(800.0, 0.0, 0.0, 0.0, 810.0, 0.0, 640.0, 480.0, 1.0)
    for (name, l) in lens_cases(P, V)
        k = "lens/$name"
        fisheye = startswith(name, "fisheye")
        Xs = rays(fisheye ? deg2rad(95) : deg2rad(40))
        xs = image_points(fisheye ? 1.6 : 0.8)
        θ = parameter_vector(l)
        blocks = lens_blocks(name)
        p = blocks === nothing ? collect(eachindex(θ)) : block_permutation(blocks, V.lens_ct_order)
        δ = P["param/lens/$name/delta"][p]
        emit("$k/names", parameter_names(l))
        emit("$k/theta", collect(θ))
        lo, hi = parameter_bounds(l)
        emit("$k/bounds_lo", collect(lo))
        emit("$k/bounds_hi", collect(hi))
        emit("$k/record", canonical(V.legacy_lens_record(lens_record(l))))
        emit("$k/project", columns([project(l, X) for X in Xs]))
        emit("$k/backproject", columns([backproject(l, x) for x in xs]))
        emit("$k/distort_normalized", columns([distort_normalized(l, x) for x in xs]))
        emit("$k/undistort_normalized", columns([undistort_normalized(l, x) for x in xs]))
        emit("$k/jacobian_x", slices([transform_deriv(l, X) for X in Xs[1:jacobian_count]]))
        emit("$k/jacobian_theta", slices([transform_deriv_params(l, X) for X in Xs[1:jacobian_count]]))
        emit("$k/monotone_radius", monotone_radius(l))
        emit("$k/view_angle_limit", view_angle_limit(l))
        l32 = convert_eltype(Float32, l)
        emit("$k/float32/theta", Float64.(collect(parameter_vector(l32))))
        emit("$k/float32/project", columns([project(l32, SVector{3, Float32}(X)) for X in Xs[1:50]]))
        l2 = with_parameters(l, θ + δ)
        emit("$k/with_parameters/theta", collect(parameter_vector(l2)))
        emit("$k/with_parameters/project", columns([project(l2, X) for X in Xs[1:50]]))
        emit("$k/manifold/retract", collect(parameter_vector(retract(LensManifold(l), δ))))
        cm = CameraModel(K; lens = l, pixel_frame = :colmap)
        us = [project(cm, X) for X in Xs[1:50]]
        emit("$k/camera/project", columns(us))
        emit("$k/camera/backproject", columns([backproject(cm, u) for u in us]))
        li = CameraModel(LogicalIntrinsics(VgtkCore.CameraCalibrationMat(K)); lens = l)
        r = attempt(() -> CameraModel(li, VgtkCore.CameraCalibrationMat(SMatrix{3, 3}(400.0, 0.0, 0.0, 0.0, 405.0, 0.0, 640.0, 480.0, 1.0))))
        r isa AbstractString ? emit("$k/camera/rescaled/error", r) : emit("$k/camera/rescaled/theta", collect(parameter_vector(lens(r))))
        vd = attempt(() -> valid_domain(cm))
        emit("$k/camera/valid_domain", vd isa AbstractString ? vd : repr(vd; context = :module => _REPR_MODULE))
    end
    return out
end
