# End-to-end save and load of one 1000-pose trajectory file, the same data
# against either checkout, back to back on one machine:
#   old (ddc1bbe, before the units layering: plain mm translations, no length_unit),
#       BenchmarkTools, Rotations, StaticArrays stacked from a scratch env:
#     JULIA_LOAD_PATH="@:<scratch env>:@stdlib" julia --project=<ddc1bbe checkout> trajectory_end_to_end.jl old
#   new (this tree: mm quantities, length_unit = mm):
#     julia --project=VisualGeometryToolkitIO/benchmark VisualGeometryToolkitIO/benchmark/trajectory_end_to_end.jl new
# Then supply the old medians/minima to lengths_overhead.jl as E2E_BASELINE_*.
# The two files have the same header and records; only the last digits of some
# computed (full-precision) mm numbers differ, 286 of 3000 translations here:
# `x .* mm` rounds x to the nearest metre double, and that value is written back
# decimal-exactly (63.099999999999994 becomes 63.09999999999999).
using BenchmarkTools, Rotations, StaticArrays, SHA
import VisualGeometryToolkitIO as VgtkIO
import VisualGeometryToolkitCore as VgtkCore
using VisualGeometryToolkitCore: EuclideanMap, Trajectory, BodyFrame, WorldFrame

const NEW = ARGS[1] == "new"
tvec(i) = SVector(62.8 + 0.1i, 0.5i + 0.01, 1200.0 - 0.7i)    # the file's mm numbers
pose(i) = EuclideanMap(QuatRotation(RotXYZ(0.001i, 0.002i, 0.003i)), NEW ? tvec(i) .* VgtkCore.mm : tvec(i))
const tr = Trajectory(
    BodyFrame{:board_rig}, WorldFrame, Dict(i => pose(i) for i in 1:1000);
    times = Dict(i => 0.04i for i in 1:1000), time_origin = "2026-05-27T00:00:00",
)
save_file(p, tr) = NEW ? VgtkIO.save(p, tr; length_unit = VgtkCore.mm) : VgtkIO.save(p, tr)

const path = tempname() * ".json"
save_file(path, tr)
const digest = bytes2hex(sha256(read(path)))
const loaded = VgtkIO.load(Trajectory, path)
@assert length(loaded) == 1000
# the loaded translations, as mm numbers, for a cross-checkout comparison
mmvals(X) = NEW ? VgtkCore.ustrip.(VgtkCore.mm, X.t) : X.t
const tsum = sum(sum(mmvals(loaded[i])) for i in 1:1000)

b_save = @benchmark save_file($path, $tr)
b_load = @benchmark VgtkIO.load(Trajectory, $path)
println(
    ARGS[1], " | save median ns ", median(b_save).time, " | save min ns ", minimum(b_save).time,
    " | load median ns ", median(b_load).time, " | load min ns ", minimum(b_load).time,
    " | file bytes ", filesize(path), " | sha256 ", digest, " | sum of loaded mm ", repr(tsum),
)
rm(path; force = true)
