# The cost of the units layering on the public rigid-map path: compose, invert
# and apply to 1000 points, a plain translation against a quantity (metres)
# translation. Gate (owner-approved): each quantity operation within 10x plain.
#   julia --project=VisualGeometryToolkitCore/benchmark VisualGeometryToolkitCore/benchmark/units_overhead.jl
#
# Machine: Linux, Julia 1.13.2, `uptime` 03:34:21 before the run: load average 9.16, 8.87, 7.43
#   (shared desktop, other work running; 8.51, 8.74, 7.41 at 03:34:31 after).
# Top CPU processes before timing (ps -eo pcpu,args --sort=-pcpu | head -6): a julia --startup-file=no
#   15.5 %, the Kaimon julia REPL 13.2 %, cosmic-term 9.7 %, cosmic-comp 9.2 %, chrome zygote 7.2 %.
# BenchmarkTools @benchmark, default parameters; ns per call.
# RESULTS (after the static apply fast path, task 11b):
#   operation | plain median | quantity median | ratio | plain min | quantity min | ratio | gate
#   compose              |  6.2 |  20.1 |  3.23x |   6.2 |   20.0 |  3.23x | holds
#   invert               |  1.8 |  10.5 |  5.69x |   1.8 |   10.2 |  5.58x | holds
#   apply to 1000 points | 631.4 | 7420.2 | 11.75x | 599.6 | 7122.2 | 11.88x | fails (> 10x)
# Review fix (03:51:38, load 8.33 / 7.42 / 7.18 before, 8.27 after; top CPU as above): the fast path
#   is RotMatrix only, a QuatRotation map (every IO-loaded pose) keeps its own product, generic:
#   compose | 6.2 | 20.1 | 3.23x; invert | 1.8 | 10.1 | 5.52x
#   apply (RotMatrix)    |  608.1 |  7425.0 | 12.21x |  598.2 |  7183.5 | 12.01x | fails (> 10x)
#   apply (QuatRotation) | 2497.9 | 30985.0 | 12.40x | 2456.1 | 30493.0 | 12.42x | fails (> 10x)
# Before it (task 11, Julia 1.12, load 6.79): apply 612.4 | 25547.0 | 41.72x. The apply was already
# static and allocation-free; the cost was FlexUnits' dimension check inside every muladd and +.

using BenchmarkTools, FlexUnits, Rotations, StaticArrays
using Rotations: QuatRotation
using VisualGeometryToolkitCore: EuclideanMap, m

const R1 = RotMatrix{3}(RotXYZ(0.10, 0.20, 0.30))
const R2 = RotMatrix{3}(RotXYZ(-0.30, 0.15, 0.05))
const t1 = SVector(0.10, 0.20, 0.30)
const t2 = SVector(-0.40, 0.05, 1.20)

const plain1 = EuclideanMap(R1, t1)
const plain2 = EuclideanMap(R2, t2)
const physical1 = EuclideanMap(R1, t1 .* m)
const physical2 = EuclideanMap(R2, t2 .* m)

# A quaternion map, as IO builds every loaded pose (the explicit type constructor).
quat_map(R, t) = EuclideanMap{3, Float64, QuatRotation{Float64}, typeof(t)}(QuatRotation(R), t)
const plain_quat1 = quat_map(R1, t1)
const physical_quat1 = quat_map(R1, t1 .* m)

const points = [SVector(sin(i), cos(2i), i / 1000) for i in 1:1000]
const physical_points = [p .* m for p in points]

# (name, plain benchmark, quantity benchmark)
const CASES = [
    ("compose", @benchmark($plain1 ∘ $plain2), @benchmark($physical1 ∘ $physical2)),
    ("invert", @benchmark(inv($plain1)), @benchmark(inv($physical1))),
    ("apply to 1000 points (RotMatrix)", @benchmark($plain1.($points)), @benchmark($physical1.($physical_points))),
    ("apply to 1000 points (QuatRotation)", @benchmark($plain_quat1.($points)), @benchmark($physical_quat1.($physical_points))),
]

const GATE = 10.0
ns(b, f) = f(b).time
println("operation | plain median ns | quantity median ns | ratio | plain min ns | quantity min ns | ratio | gate <= $(GATE)x")
for (name, plain, physical) in CASES
    r_med = ns(physical, median) / ns(plain, median)
    r_min = ns(physical, minimum) / ns(plain, minimum)
    println(
        name, " | ", round(ns(plain, median); digits = 1), " | ", round(ns(physical, median); digits = 1), " | ",
        round(r_med; digits = 2), " | ", round(ns(plain, minimum); digits = 1), " | ", round(ns(physical, minimum); digits = 1),
        " | ", round(r_min; digits = 2), " | ", r_med <= GATE ? "holds" : "fails"
    )
end
