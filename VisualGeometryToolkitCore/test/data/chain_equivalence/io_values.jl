# =============================================================================
# VisualGeometryToolkitIO's chain-equivalence values and records: what
# `io_values.txt` and `records/` pin, written by the code at hand through an
# adapter (corpus.jl). Included after corpus.jl by the generator and by
# VisualGeometryToolkitIO's equivalence test item.
# =============================================================================

using StaticArrays: SVector, SMatrix
using LinearAlgebra: Diagonal
using Rotations: QuatRotation
import VisualGeometryToolkitCore as VgtkCore
import VisualGeometryToolkitIO as VgtkIO
using VisualGeometryToolkitCore: m, mm, parameter_vector, PlanarTarget, PosedTarget, Uncertain, CovMat, EuclideanMap, Rig,
    TargetFrame, CameraModel, BoundedCamera, Size2

const RECORD_POSE_ROTATION = QuatRotation(0.9128709291752769, 0.18257418583505536, 0.36514837167011072, 0.0, false)
const RECORD_K = SMatrix{3, 3}(800.0, 0.0, 0.0, 0.0, 810.0, 0.0, 640.0, 480.0, 1.0)

"""
    io_values(P, V, dir) -> Vector{Pair{String, Any}}

Write every record into `dir` (the retired target cases left out, and the rig
of every target with them) and return the values `io_values.txt` pins.
"""
function io_values(P, V, dir::AbstractString)
    out = Pair{String, Any}[]
    emit(k, v) = push!(out, k => v)
    mkpath(dir)
    ppts = physical_points()[1:20]   # a record pins the calibration, not the points
    pose = EuclideanMap(RECORD_POSE_ROTATION, SVector(0.1, -0.05, 0.9) .* m)
    # a rig's frame is one target's: the gauge, a rigid target at the identity
    gauge = PosedTarget(PlanarTarget(ppts), EuclideanMap(QuatRotation(1.0, 0.0, 0.0, 0.0, false), SVector(0.0, 0.0, 0.0) .* m))
    every = Dict{Symbol, PosedTarget}(:gauge => gauge)
    for (name, c) in target_cases(P, V; physical = true)
        name in V.retired && continue
        t = PlanarTarget(ppts; calibration = c)
        VgtkIO.save(joinpath(dir, "target_$name.json"), t)
        n = length(parameter_vector(c))
        if n > 0
            # the covariance over the golden (application) order, permuted to the code's
            σ = [1.0e-8 * i for i in 1:n][block_permutation(target_blocks(name, 1), V.ct_order)]
            u = Uncertain(c, CovMat(SMatrix{n, n, Float64}(Diagonal(σ))))
            VgtkIO.save(joinpath(dir, "target_$name.uncertain.json"), PlanarTarget(ppts; calibration = u))
        end
        id = Symbol("t_", name)
        rig = Rig{TargetFrame{:gauge}}(Dict(:gauge => gauge, id => PosedTarget(t, pose)))
        VgtkIO.save(joinpath(dir, "rig_$name.json"), rig)
        emit("content_id/$name", VgtkIO.content_id(rig))
        every[id] = PosedTarget(t, pose)
    end
    if isempty(V.retired)
        rig = Rig{TargetFrame{:gauge}}(every)
        VgtkIO.save(joinpath(dir, "rig_every.json"), rig)
        emit("content_id/every", VgtkIO.content_id(rig))
    end
    cams = Dict{Symbol, BoundedCamera}()
    for (name, l) in lens_cases(P, V)
        cm = CameraModel(RECORD_K; lens = l, pixel_frame = :colmap)
        cams[Symbol(name)] = BoundedCamera(cm, Size2(1280, 960))
        VgtkIO.save(joinpath(dir, "cameras_$name.json"), Dict(:cam => cams[Symbol(name)]))
        model = VgtkIO.colmap_model(l)
        emit("colmap/$name/model", model === nothing ? "nothing" : String(model))
        model === nothing || emit("colmap/$name/params", VgtkIO.colmap_params(l, 800.0, 810.0, 640.0, 480.0))
        if startswith(name, "opencv")
            emit("opencv/$name/coefficients", VgtkIO.opencv_coefficients(l))
            VgtkIO.save_opencv_calibration(joinpath(dir, "opencv_$name.json"), cm, [pose]; rms = 0.25, to_unit = mm, images = ["img0.png"])
        end
    end
    VgtkIO.save(joinpath(dir, "cameras_every.json"), cams)
    colmap = filter(p -> VgtkIO.colmap_model(VgtkCore.lens(VgtkCore.model(last(p)))) !== nothing, cams)
    ccams, _ = VgtkIO.colmap_cameras(colmap)
    cdir = joinpath(dir, "colmap")
    mkpath(cdir)
    VgtkIO.save(cdir, VgtkIO.ColmapModel(ccams, VgtkIO.ColmapImage[], VgtkIO.ColmapPoint3D[], VgtkIO.MM_PX); format = :text)
    return out
end
