# The cost of the file-length pair (read_length / write_length) against the
# multiply path it replaced, on a 1000-pose trajectory. "Today" is a local
# reference reproducing the old conversion per number
# (`ustrip(u, quantity(x, u))`-style multiply by the unit scale), run in this
# process on the same poses; the new path is the package's own pose_json /
# pose_from_json (the records' encode and decode, which is all the conversion
# touches), and save / load of the whole file are reported for context.
# Gate (owner-approved): read and write each within 1.5x today.
#   julia --project=VisualGeometryToolkitIO/benchmark VisualGeometryToolkitIO/benchmark/lengths_overhead.jl
#
# Machine: Linux, `uptime` at 03:11:05 and 03:12:15 (before/after the runs):
#   load average 6.79, 8.01, 8.98 then 4.10, 7.02, 8.57 (shared desktop, other work running).
# Top CPU processes before timing (ps -eo pcpu,args --sort=-pcpu | head -5): Kaimon julia 15.6 %,
#   cosmic-term 9.7 %, cosmic-comp 9.3 %, chrome zygote 7.7 %. The known leftover `bfs`
#   search was not running at that moment (pgrep found none); nothing was killed.
# Julia 1.12 (workspace), BenchmarkTools @benchmark, default parameters; ns per call.
# RESULTS: operation | today median | new median | ratio | today min | new min | ratio | gate
#   read  (1000 records)  | 58256934 | 51831594 |  0.89x | 57531370 | 51184214 |  0.89x | holds
#   write (1000 poses)    |    42167 |  3836652 | 90.99x |    39408 |  3757508 | 95.35x | fails (> 1.5x)
#   context, new path only: load file 55225620 median / 54581916 min; save file 10157169 / 9669872.
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

const path = tempname() * ".json"
VgtkIo.save(path, poses; length_unit = unit)
const loaded = VgtkIo.load(Vector{EuclideanMap}, path)
@assert length(loaded) == 1000

const CASES = [
    ("read (decode 1000 records)", @benchmark(old_decode($records)), @benchmark(new_decode($records))),
    ("write (encode 1000 poses)", @benchmark(old_encode($poses)), @benchmark(new_encode($poses))),
]
const FILE_CASES = [
    ("load file (new, context)", @benchmark(VgtkIo.load(Vector{EuclideanMap}, $path))),
    ("save file (new, context)", @benchmark(VgtkIo.save($path, $poses; length_unit = $unit))),
]

const GATE = 1.5
ns(b, f) = f(b).time
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
for (name, b) in FILE_CASES
    println(name, " | median ns ", round(ns(b, median); digits = 1), " | min ns ", round(ns(b, minimum); digits = 1))
end
rm(path; force = true)
