# The cost of the units layering on the public rigid-map path: compose, invert
# and apply to 1000 points, a plain translation against a quantity (metres)
# translation. Gate (owner-approved): each quantity operation within 10x plain.
#   julia --project=VisualGeometryToolkitCore/benchmark VisualGeometryToolkitCore/benchmark/units_overhead.jl
#
# Machine: Linux, `uptime` at 03:11:05 and 03:12:15 (before/after the runs):
#   load average 6.79, 8.01, 8.98 then 4.10, 7.02, 8.57 (shared desktop, other work running).
# Top CPU processes before timing (ps -eo pcpu,args --sort=-pcpu | head -5): Kaimon julia 15.6 %,
#   cosmic-term 9.7 %, cosmic-comp 9.3 %, chrome zygote 7.7 %. The known leftover `bfs`
#   search was not running at that moment (pgrep found none); nothing was killed.
# Julia 1.12 (workspace), BenchmarkTools @benchmark, default parameters; ns per call.
# RESULTS: operation | plain median | quantity median | ratio | plain min | quantity min | ratio | gate
#   compose              |    6.0 |    19.3 |  3.20x |   6.0 |   19.3 |  3.21x | holds
#   invert               |    1.9 |    10.5 |  5.63x |   1.8 |   10.2 |  5.58x | holds
#   apply to 1000 points |  612.4 | 25547.0 | 41.72x | 598.8 | 25308.0 | 42.26x | fails (> 10x)

using BenchmarkTools, FlexUnits, Rotations, StaticArrays
using VisualGeometryToolkitCore: EuclideanMap, m

const R1 = RotMatrix{3}(RotXYZ(0.10, 0.20, 0.30))
const R2 = RotMatrix{3}(RotXYZ(-0.30, 0.15, 0.05))
const t1 = SVector(0.10, 0.20, 0.30)
const t2 = SVector(-0.40, 0.05, 1.20)

const plain1 = EuclideanMap(R1, t1)
const plain2 = EuclideanMap(R2, t2)
const physical1 = EuclideanMap(R1, t1 .* m)
const physical2 = EuclideanMap(R2, t2 .* m)

const points = [SVector(sin(i), cos(2i), i / 1000) for i in 1:1000]
const physical_points = [p .* m for p in points]

# (name, plain benchmark, quantity benchmark)
const CASES = [
    ("compose", @benchmark($plain1 ∘ $plain2), @benchmark($physical1 ∘ $physical2)),
    ("invert", @benchmark(inv($plain1)), @benchmark(inv($physical1))),
    ("apply to 1000 points", @benchmark($plain1.($points)), @benchmark($physical1.($physical_points))),
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
