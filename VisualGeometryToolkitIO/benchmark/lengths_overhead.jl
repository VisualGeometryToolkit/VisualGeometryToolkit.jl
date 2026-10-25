# The cost of the file-length pair (read_length / write_length) against the
# multiply path it replaced, on a 1000-pose trajectory. "Today" is a local
# reference reproducing the old conversion per number
# (`ustrip(u, quantity(x, u))`-style multiply by the unit scale), run in this
# process on the same poses; the new path is the package's own pose_json /
# pose_from_json (the records' encode and decode, which is all the conversion
# touches); save and load of one trajectory file, end to end, against the code before the units layering (trajectory_end_to_end.jl).
# Gate (owner-approved): read and write each within 1.5x today.
#   julia --project=VisualGeometryToolkitIO/benchmark VisualGeometryToolkitIO/benchmark/lengths_overhead.jl
#
# Machine: Linux, Julia 1.13.2, `uptime` 03:34:31 before the run: load average 8.51, 8.74, 7.41
#   (shared desktop, other work running; 6.17, 8.01, 7.24 at 03:35:32 after).
# Top CPU processes before timing (ps -eo pcpu,args --sort=-pcpu | head -6): a julia --startup-file=no
#   15.5 %, the Kaimon julia REPL 13.2 %, cosmic-term 9.7 %, cosmic-comp 9.2 %, chrome zygote 7.2 %.
# BenchmarkTools @benchmark, default parameters; ns per call.
# RESULTS (after write_length's one-read-back path, task 11b), conversion only:
#   operation | today median | new median | ratio | today min | new min | ratio | gate
#   read  (1000 records)  | 58151078 | 50480136 |  0.87x | 57457310 | 50050356 |  0.87x | holds
#   write (1000 poses)    |    43896 |  1666857 | 37.97x |    40842 |  1617089 | 39.59x | fails (> 1.5x)
# End to end, one 1000-pose trajectory file, baseline = trajectory_end_to_end.jl old against ddc1bbe,
# run back to back (03:33:27, load 12.41 before / 9.43 after) and supplied as E2E_BASELINE_*:
#   save | 3746064 | 8093121 | 2.16x | 3649578 | 7812282 | 2.14x
#   load | 54471086 | 54627200 | 1.00x | 53947985 | 53910650 | 1.00x
# Before 11b (task 11, Julia 1.12): write 42167 | 3836652 | 90.99x; save (new only) 10157169.
# After 11c (one serialization per save, non-finite numbers found by a direct scan), 03:42:46,
#   load 8.35 / 8.67 / 7.72 before, 5.17 after; baseline rerun at 03:42:16 (load 11.12):
#   write (1000 poses) | 43911 | 1664134 | 37.90x | 41101 | 1612070 | 39.22x | fails (> 1.5x)
#   save | 3777415 | 5426504 | 1.44x | 3667759 | 5188082 | 1.41x
#   load | 54486912 | 54564542 | 1.00x | 53889564 | 53917184 | 1.00x
# The read reference includes parse_unit of the file's unit name per record, as the reader does.

using BenchmarkTools, FlexUnits, Rotations, StaticArrays
using Rotations: QuatRotation
import VisualGeometryToolkitIO as VgtkIo
import VisualGeometryToolkitCore as VgtkCore
using VisualGeometryToolkitCore: EuclideanMap, m, mm

# The old conversion, one number at a time: the file number is a quantity in
# the file's unit and the metre value comes from the unit's scale.
old_read(x::Real, u) = ustrip(m, quantity(x, u))
old_write(q, u) = Float64(ustrip(u, q))

old_translation_from_json(v, u) = SVector{3}(ntuple(i -> old_read(v[i]::Real, u) * m, Val(3)))
old_translation_json(t, u) = Float64[old_write(q, u) for q in t]

const unit = mm
const unit_name = "mm"

# 1000 labeled physical poses; translations are the file's mm numbers times mm.
const poses = [
    VgtkCore.relabel(
            VgtkCore.TargetFrame{:board}, VgtkCore.CameraFrame,
            EuclideanMap(
                QuatRotation(RotXYZ(0.001i, 0.002i, 0.003i)),
                SVector(62.8 + 0.1i, 0.5i + 0.01, 1200.0 - 0.7i) .* mm
            )
        ) for i in 1:1000
]

# The records as a parsed file holds them (Vector{Any} numbers).
const records = [
    (
            rotation = Vector{Any}(r.rotation), translation = Vector{Any}(r.translation),
        ) for r in (VgtkIo.pose_json(X, unit) for X in poses)
]

# Today's decode and encode of one record: the translation through the multiply.
const QMap = EuclideanMap{3, Float64, QuatRotation{Float64}}
function old_pose_from_json(p)
    q = SVector{4, Float64}(Float64.(p.rotation::Vector{Any})...)
    t = old_translation_from_json(p.translation::Vector{Any}, VgtkCore.parse_unit(unit_name))
    return EuclideanMap{3, Float64, QuatRotation{Float64}, typeof(t)}(
        QuatRotation(q[1], q[2], q[3], q[4], false), t,
    )
end
old_pose_json(X) = (rotation = Float64[Rotations.params(QuatRotation(X.R))...], translation = old_translation_json(X.t, unit))

new_decode(recs) = [VgtkIo.pose_from_json(r, unit_name) for r in recs]
old_decode(recs) = [old_pose_from_json(r) for r in recs]
new_encode(ps) = [VgtkIo.pose_json(X, unit) for X in ps]
old_encode(ps) = [old_pose_json(X) for X in ps]

# Both paths agree to a few ulp (the old multiply is the inexact one).
let a = new_decode(records), b = old_decode(records)
    @assert all(i -> maximum(abs, ustrip.(m, a[i].t) - ustrip.(m, b[i].t)) <= 1.0e-12, eachindex(a))
end

const CASES = [
    ("read (decode 1000 records)", @benchmark(old_decode($records)), @benchmark(new_decode($records))),
    ("write (encode 1000 poses)", @benchmark(old_encode($poses)), @benchmark(new_encode($poses))),
]

# ---------------------------------------------------------------------------
# End to end: save and load of ONE 1000-pose trajectory file. The trajectory is
# built exactly as trajectory_end_to_end.jl builds it, which also runs against a
# checkout of the code before the units layering (ddc1bbe, plain mm
# translations) for the baseline: that code cannot load in this process, so its
# numbers come in through the environment, E2E_BASELINE_SAVE_NS and
# E2E_BASELINE_LOAD_NS (medians), and E2E_BASELINE_SAVE_MIN_NS /
# E2E_BASELINE_LOAD_MIN_NS (minima); without them only the new numbers print.
# ---------------------------------------------------------------------------
using VisualGeometryToolkitCore: Trajectory, BodyFrame, WorldFrame
e2e_tvec(i) = SVector(62.8 + 0.1i, 0.5i + 0.01, 1200.0 - 0.7i)    # the file's mm numbers
const trajectory = Trajectory(
    BodyFrame{:board_rig}, WorldFrame,
    Dict(i => EuclideanMap(QuatRotation(RotXYZ(0.001i, 0.002i, 0.003i)), e2e_tvec(i) .* mm) for i in 1:1000);
    times = Dict(i => 0.04i for i in 1:1000), time_origin = "2026-05-27T00:00:00",
)
const path = tempname() * ".json"
VgtkIo.save(path, trajectory; length_unit = unit)
@assert length(VgtkIo.load(Trajectory, path)) == 1000

const FILE_CASES = [
    ("save trajectory file", "E2E_BASELINE_SAVE", @benchmark(VgtkIo.save($path, $trajectory; length_unit = $unit))),
    ("load trajectory file", "E2E_BASELINE_LOAD", @benchmark(VgtkIo.load(Trajectory, $path))),
]

const GATE = 1.5
ns(b, f) = f(b).time
println("conversion only")
println("operation | today median ns | new median ns | ratio | today min ns | new min ns | ratio | gate <= $(GATE)x")
for (name, today, new) in CASES
    r_med = ns(new, median) / ns(today, median)
    r_min = ns(new, minimum) / ns(today, minimum)
    println(
        name, " | ", round(ns(today, median); digits = 1), " | ", round(ns(new, median); digits = 1), " | ",
        round(r_med; digits = 2), " | ", round(ns(today, minimum); digits = 1), " | ", round(ns(new, minimum); digits = 1),
        " | ", round(r_min; digits = 2), " | ", r_med <= GATE ? "holds" : "fails"
    )
end
println("end to end (baseline: trajectory_end_to_end.jl old, run against ddc1bbe)")
println("operation | baseline median ns | new median ns | ratio | baseline min ns | new min ns | ratio")
for (name, key, b) in FILE_CASES
    med, mn = ns(b, median), ns(b, minimum)
    base_med, base_min = get(ENV, key * "_NS", nothing), get(ENV, key * "_MIN_NS", nothing)
    if base_med === nothing || base_min === nothing
        println(name, " | - | ", round(med; digits = 1), " | - | - | ", round(mn; digits = 1), " | -")
    else
        bm, bn = parse(Float64, base_med), parse(Float64, base_min)
        println(
            name, " | ", bm, " | ", round(med; digits = 1), " | ", round(med / bm; digits = 2), " | ",
            bn, " | ", round(mn; digits = 1), " | ", round(mn / bn; digits = 2)
        )
    end
end
rm(path; force = true)
