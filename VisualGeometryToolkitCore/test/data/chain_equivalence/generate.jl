# =============================================================================
# The chain-equivalence goldens — written ONCE, on the code before
# `CalibrationChain` and the lens `Chain` were replaced by
# CoordinateTransformations' `∘` (VisualGeometryToolkitCore 0.4.1, main 5b31549).
# =============================================================================
#
# Kept as the record of how the goldens were made. Never rerun it to make a
# golden match: a golden that moves is a change to stop at and explain. It
# spells the compositions with the two retired types, so it runs only on that
# code. From the repository root:
#
#   JULIA_LOAD_PATH="@:$PWD/VisualGeometryToolkitCore/test:@stdlib" \
#       julia --project=VisualGeometryToolkitIO VisualGeometryToolkitCore/test/data/chain_equivalence/generate.jl
#
# (Core's test environment on the load path for StableRNGs.) It writes, next
# to itself:
#
#   corpus_params.txt   the corpus's parameters (StableRNG(20261103)): the
#                       inputs every reader rebuilds the corpus from
#   core_values.txt     Core's outputs: per calibration — parameter vector,
#                       names, bounds, gauge names and directions, length
#                       mask, planarity, the map on 200 plain and 50 physical
#                       points, the parameter and point Jacobians at 10, the
#                       regularizer rows, a with_parameters rebuild, the unit
#                       moves, the target manifold's size, action and
#                       retraction, the surface chart's affine; per lens —
#                       parameter vector, names, bounds, record, project,
#                       backproject, distort and undistort on 200 points, the
#                       input and parameter Jacobians at 10, the monotone
#                       radius and view-angle limit, Float32 projection, a
#                       with_parameters rebuild, the lens manifold's
#                       retraction, a camera's project and backproject, its
#                       rescaled lens and its valid domain
#   io_values.txt       VisualGeometryToolkitIO's: OpenCV coefficient vectors,
#                       COLMAP models and parameters, every rig's content_id
#   records/            the files VisualGeometryToolkitIO writes from them:
#                       vgio/target/v2 (plain and Uncertain calibration),
#                       vgio/rig/v5, vgio/cameras/v3, the OpenCV calibration
#                       export and a COLMAP text model

include(joinpath(@__DIR__, "corpus.jl"))
using .ChainEquivalenceCorpus
using StableRNGs: StableRNG
using StaticArrays: SVector, SMatrix
using LinearAlgebra: Diagonal, I
using Rotations: QuatRotation
using CoordinateTransformations: transform_deriv, transform_deriv_params
import VisualGeometryToolkitCore as VgtkCore
import VisualGeometryToolkitIO as VgtkIO
using VisualGeometryToolkitCore: CalibrationChain, Chain, m, mm, parameter_vector, parameter_names, parameter_bounds,
    with_parameters, regularizer, calibration_jacobian, gauge_names, gauge_directions, length_mask, is_planar, is_affine,
    calibration_tag, has_units, with_unit, PlanarTarget, PosedTarget, TargetManifold, tangent_size, retract_jacobian,
    action_jacobian, retract, Uncertain, CovMat, EuclideanMap, Rig, TargetFrame, surface, project, backproject,
    distort_normalized, undistort_normalized, monotone_radius, view_angle_limit, convert_eltype, LensManifold,
    lens_record, CameraModel, BoundedCamera, Size2, valid_domain, LogicalIntrinsics, lens
using FlexUnits: ustrip

const DIR = @__DIR__
const RECORDS = joinpath(DIR, "records")

compose_calibrations(members::Tuple) = CalibrationChain(members)
compose_distortions(stages::Tuple) = Chain(stages...)

# -----------------------------------------------------------------------------
# The parameters
# -----------------------------------------------------------------------------

function draw_params()
    rng = StableRNG(20261103)
    r(n...) = randn(rng, n...)
    P = Dict{String, Any}()
    P["param/affine/A"] = [1.0 0.0; 0.0 1.0] .+ 0.01 .* r(2, 2)
    P["param/affine/b"] = 0.002 .* r(2)
    P["param/thin_plate_spline/fx"] = 2.0e-4 .* r(9)
    P["param/thin_plate_spline/fy"] = 2.0e-4 .* r(9)
    P["param/height_field/h"] = 5.0e-4 .* r(9)
    # OpenCV's (k1, k2, p1, p2, k3, k4, k5, k6, s1, s2, s3, s4, τx, τy); a
    # shorter model is a prefix
    cv = [-0.2 + 0.05r(), 0.05 + 0.02r(), 1.0e-3r(), 1.0e-3r(), -0.01 + 0.005r(), 0.02r(), 0.01r(), 0.005r(),
          1.0e-3r(), 1.0e-3r(), 1.0e-3r(), 1.0e-3r(), 0.01r(), 0.01r()]
    for n in (4, 5, 8, 12, 14)
        P["param/opencv$n"] = cv[1:n]
    end
    # COLMAP's RAD_TAN_THIN_PRISM_FISHEYE extras (k0…k5, p0, p1, s0…s3)
    P["param/fisheye624"] = [[-0.02, 0.005, -0.001, 1.0e-4, -1.0e-5, 1.0e-6] .* (1 .+ 0.1 .* r(6)); 1.0e-4 .* r(6)]
    # the perturbation of a with_parameters rebuild, per case
    for (name, c) in target_cases(P, compose_calibrations)
        P["param/target/$name/delta"] = 1.0e-5 .* r(length(parameter_vector(c)))
    end
    for (name, l) in lens_cases(P, compose_distortions)
        P["param/lens/$name/delta"] = 1.0e-4 .* r(length(parameter_vector(l)))
    end
    return P
end

# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------

columns(ps) = isempty(ps) ? zeros(0, 0) : reduce(hcat, [Float64.(collect(p)) for p in ps])
slices(Js) = cat([Float64.(Matrix(J)) for J in Js]...; dims = 3)
plain_metres(p) = map(x -> ustrip(m, x), p)
attempt(f) = try
    f()
catch e
    e isa Union{ArgumentError, MethodError, DomainError} || rethrow()
    sprint(showerror, e)
end
emit(io, key, x) = println(io, golden_line(key, x))
# a record as a canonical text: keys sorted, numbers by repr
canonical(x::AbstractDict) = "{" * join([repr(String(k)) * ":" * canonical(x[k]) for k in sort(collect(keys(x)); by = String)], ",") * "}"
canonical(x::AbstractVector) = "[" * join(canonical.(x), ",") * "]"
canonical(x::Real) = repr(Float64(x))
canonical(x) = repr(x)

# -----------------------------------------------------------------------------
# Core's values
# -----------------------------------------------------------------------------

function target_values(io, P)
    pts, ppts = design_points(), physical_points()
    jpts = pts[1:jacobian_count]
    physical = Dict(target_cases(P, compose_calibrations; physical = true))
    for (name, c) in target_cases(P, compose_calibrations)
        k = "target/$name"
        cp = physical[name]
        θ = parameter_vector(c)
        emit(io, "$k/names", parameter_names(c))
        emit(io, "$k/gauge_names", gauge_names(c))
        emit(io, "$k/tag", calibration_tag(typeof(c)))
        emit(io, "$k/length_mask", collect(length_mask(c)))
        emit(io, "$k/is_planar", is_planar(c))
        emit(io, "$k/is_affine", is_affine(c))
        emit(io, "$k/theta", collect(θ))
        lo, hi = parameter_bounds(c)
        emit(io, "$k/bounds_lo", collect(lo))
        emit(io, "$k/bounds_hi", collect(hi))
        emit(io, "$k/out", columns(c.(pts)))
        emit(io, "$k/out_physical", columns(plain_metres.(cp.(ppts[1:50]))))
        emit(io, "$k/jacobian_theta", slices([calibration_jacobian(c, x) for x in jpts]))
        emit(io, "$k/jacobian_x", slices([VgtkCore.Cameras._point_jacobian(c, x) for x in jpts]))
        emit(io, "$k/regularizer", collect(regularizer(c)))
        emit(io, "$k/gauge_directions", gauge_directions(c))
        δ = P["param/target/$name/delta"]
        c2 = with_parameters(c, θ + δ)
        emit(io, "$k/with_parameters/theta", collect(parameter_vector(c2)))
        emit(io, "$k/with_parameters/out", columns(c2.(pts[1:50])))
        emit(io, "$k/has_units_physical", has_units(cp))
        emit(io, "$k/with_unit_mm/theta", collect(parameter_vector(with_unit(c, mm))))
        emit(io, "$k/ustrip_mm/theta", collect(parameter_vector(ustrip(mm, cp))))
        t = PlanarTarget(pts; calibration = c)
        ch = TargetManifold(t)
        B = Matrix(retract_jacobian(ch))
        emit(io, "$k/manifold/size", Float64(tangent_size(ch)))
        emit(io, "$k/manifold/action", slices([Matrix(action_jacobian(ch, x)) * B' for x in pts[1:3]]))
        emit(io, "$k/manifold/retract", collect(parameter_vector(retract(ch, B' * δ))))
        X = EuclideanMap(QuatRotation(0.9128709291752769, 0.18257418583505536, 0.36514837167011072, 0.0, false), SVector(0.1, -0.05, 0.9))
        s = attempt(() -> surface(PosedTarget(t, X)))
        if s isa AbstractString
            emit(io, "$k/surface/error", s)
        else
            q = s isa VgtkCore.Cameras.Quad ? s : s.quad
            emit(io, "$k/surface/type", string(nameof(typeof(s))))
            emit(io, "$k/surface/A", Matrix(q.A))
            emit(io, "$k/surface/b", collect(q.b))
        end
    end
    return nothing
end

function lens_values(io, P)
    K = SMatrix{3, 3}(800.0, 0.0, 0.0, 0.0, 810.0, 0.0, 640.0, 480.0, 1.0)
    for (name, l) in lens_cases(P, compose_distortions)
        k = "lens/$name"
        fisheye = startswith(name, "fisheye")
        Xs = rays(fisheye ? deg2rad(95) : deg2rad(40))
        xs = image_points(fisheye ? 1.6 : 0.8)
        θ = parameter_vector(l)
        emit(io, "$k/names", parameter_names(l))
        emit(io, "$k/theta", collect(θ))
        lo, hi = parameter_bounds(l)
        emit(io, "$k/bounds_lo", collect(lo))
        emit(io, "$k/bounds_hi", collect(hi))
        emit(io, "$k/record", canonical(lens_record(l)))
        emit(io, "$k/project", columns([project(l, X) for X in Xs]))
        emit(io, "$k/backproject", columns([backproject(l, x) for x in xs]))
        emit(io, "$k/distort_normalized", columns([distort_normalized(l, x) for x in xs]))
        emit(io, "$k/undistort_normalized", columns([undistort_normalized(l, x) for x in xs]))
        emit(io, "$k/jacobian_x", slices([transform_deriv(l, X) for X in Xs[1:jacobian_count]]))
        emit(io, "$k/jacobian_theta", slices([transform_deriv_params(l, X) for X in Xs[1:jacobian_count]]))
        emit(io, "$k/monotone_radius", monotone_radius(l))
        emit(io, "$k/view_angle_limit", view_angle_limit(l))
        l32 = convert_eltype(Float32, l)
        emit(io, "$k/float32/theta", Float64.(collect(parameter_vector(l32))))
        emit(io, "$k/float32/project", columns([project(l32, SVector{3, Float32}(X)) for X in Xs[1:50]]))
        δ = P["param/lens/$name/delta"]
        l2 = with_parameters(l, θ + δ)
        emit(io, "$k/with_parameters/theta", collect(parameter_vector(l2)))
        emit(io, "$k/with_parameters/project", columns([project(l2, X) for X in Xs[1:50]]))
        emit(io, "$k/manifold/retract", collect(parameter_vector(retract(LensManifold(l), δ))))
        cm = CameraModel(K; lens = l, image_origin = :colmap)
        us = [project(cm, X) for X in Xs[1:50]]
        emit(io, "$k/camera/project", columns(us))
        emit(io, "$k/camera/backproject", columns([backproject(cm, u) for u in us]))
        li = CameraModel(LogicalIntrinsics(VgtkCore.CameraCalibrationMat(K)); lens = l)
        r = attempt(() -> CameraModel(li, VgtkCore.CameraCalibrationMat(SMatrix{3, 3}(400.0, 0.0, 0.0, 0.0, 405.0, 0.0, 640.0, 480.0, 1.0))))
        r isa AbstractString ? emit(io, "$k/camera/rescaled/error", r) : emit(io, "$k/camera/rescaled/theta", collect(parameter_vector(lens(r))))
        vd = attempt(() -> valid_domain(cm))
        emit(io, "$k/camera/valid_domain", vd isa AbstractString ? vd : repr(vd))
    end
    return nothing
end

# -----------------------------------------------------------------------------
# VisualGeometryToolkitIO's records
# -----------------------------------------------------------------------------

const POSE_ROTATION = QuatRotation(0.9128709291752769, 0.18257418583505536, 0.36514837167011072, 0.0, false)

function io_values(io, P)
    mkpath(RECORDS)
    ppts = physical_points()[1:20]   # a record pins the calibration, not the points
    pose = EuclideanMap(POSE_ROTATION, SVector(0.1, -0.05, 0.9) .* m)
    # a rig's frame is one target's: the gauge, a rigid target at the identity
    gauge = PosedTarget(PlanarTarget(ppts), EuclideanMap(QuatRotation(1.0, 0.0, 0.0, 0.0, false), SVector(0.0, 0.0, 0.0) .* m))
    every = Dict{Symbol, PosedTarget}(:gauge => gauge)
    for (name, c) in target_cases(P, compose_calibrations; physical = true)
        t = PlanarTarget(ppts; calibration = c)
        VgtkIO.save(joinpath(RECORDS, "target_$name.json"), t)
        n = length(parameter_vector(c))
        if n > 0
            u = Uncertain(c, CovMat(SMatrix{n, n, Float64}(Diagonal([1.0e-8 * i for i in 1:n]))))
            VgtkIO.save(joinpath(RECORDS, "target_$name.uncertain.json"), PlanarTarget(ppts; calibration = u))
        end
        id = Symbol("t_", name)
        rig = Rig{TargetFrame{:gauge}}(Dict(:gauge => gauge, id => PosedTarget(t, pose)))
        VgtkIO.save(joinpath(RECORDS, "rig_$name.json"), rig)
        emit(io, "content_id/$name", VgtkIO.content_id(rig))
        every[id] = PosedTarget(t, pose)
    end
    rig = Rig{TargetFrame{:gauge}}(every)
    VgtkIO.save(joinpath(RECORDS, "rig_every.json"), rig)
    emit(io, "content_id/every", VgtkIO.content_id(rig))
    K = SMatrix{3, 3}(800.0, 0.0, 0.0, 0.0, 810.0, 0.0, 640.0, 480.0, 1.0)
    cams = Dict{Symbol, BoundedCamera}()
    for (name, l) in lens_cases(P, compose_distortions)
        cm = CameraModel(K; lens = l, image_origin = :colmap)
        cams[Symbol(name)] = BoundedCamera(cm, Size2(1280, 960))
        VgtkIO.save(joinpath(RECORDS, "cameras_$name.json"), Dict(:cam => cams[Symbol(name)]))
        model = VgtkIO.colmap_model(l)
        emit(io, "colmap/$name/model", model === nothing ? "nothing" : String(model))
        model === nothing || emit(io, "colmap/$name/params", VgtkIO.colmap_params(l, 800.0, 810.0, 640.0, 480.0))
        if startswith(name, "opencv")
            emit(io, "opencv/$name/coefficients", VgtkIO.opencv_coefficients(l))
            VgtkIO.save_opencv_calibration(joinpath(RECORDS, "opencv_$name.json"), cm, [pose]; rms = 0.25, to_unit = mm, images = ["img0.png"])
        end
    end
    VgtkIO.save(joinpath(RECORDS, "cameras_every.json"), cams)
    colmap = filter(p -> VgtkIO.colmap_model(VgtkCore.lens(model_of(last(p)))) !== nothing, cams)
    ccams, _ = VgtkIO.colmap_cameras(colmap)
    dir = joinpath(RECORDS, "colmap")
    mkpath(dir)
    VgtkIO.save(dir, VgtkIO.ColmapModel(ccams, VgtkIO.ColmapImage[], VgtkIO.ColmapPoint3D[], VgtkIO.MM_PX); format = :text)
    return nothing
end
model_of(c) = VgtkCore.model(c)

# -----------------------------------------------------------------------------

function main()
    P = draw_params()
    open(joinpath(DIR, "corpus_params.txt"), "w") do io
        foreach(k -> emit(io, k, P[k]), sort(collect(keys(P))))
    end
    # the readers rebuild the corpus from the stored parameters: so does this
    P = read_goldens(joinpath(DIR, "corpus_params.txt"))
    open(io -> (target_values(io, P); lens_values(io, P)), joinpath(DIR, "core_values.txt"), "w")
    open(io -> io_values(io, P), joinpath(DIR, "io_values.txt"), "w")
    return nothing
end

main()
