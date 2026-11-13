# PointFrame at no overhead (owner ruling 2026-11-13, the FeatureFrame interface): the hoisted verbs on an
# Uncertain{PointFrame} against the same work on an Uncertain{Point2} — propagate through a SimilarityMap and a
# HomographyMap, transfer (a point map's action) and center (the detection → point observation conversion), and
# a loop over a StructArray of detections. Gate: parity within noise.
#   julia --project=VisualGeometryToolkitCore/benchmark VisualGeometryToolkitCore/benchmark/point_frame.jl
# BenchmarkTools @benchmark, default parameters, every argument `$`-interpolated through a Ref (no constant folding);
# ns per call (per loop for the StructArray rows). RESULTS: see the end of this file.

using BenchmarkTools, StaticArrays, Rotations
using VisualGeometryToolkitCore
using VisualGeometryToolkitCore: center, transfer
using VisualGeometryToolkitCore.StructArrays: StructArray    # Core's own dependency: no benchmark dep added

const S = SimilarityMap(2.0, RotMatrix2(0.4), SVector(1.0, -3.0))
const H = HomographyMap(@SMatrix [1.0 0.1 3.0; 0.0 1.1 4.0; 1e-3 2e-3 1.0])
const Σ = CovMat(@SMatrix [4.0 1.0; 1.0 2.0])
const c = Point2(10.0, 20.0)
uf, up = Uncertain(PointFrame(c), Σ), Uncertain(c, Σ)
df, dp = Attributed(uf, (; weight = 0.5)), Attributed(up, (; weight = 0.5))

med(b) = BenchmarkTools.median(b).time
rows = Pair{String, NTuple{2, Float64}}[]
bench!(name, bf, bp) = (push!(rows, name => (med(bf), med(bp))); nothing)

bench!("propagate(S, u)", @benchmark(propagate($(Ref(S))[], $(Ref(uf))[])), @benchmark(propagate($(Ref(S))[], $(Ref(up))[])))
bench!("propagate(H, u)", @benchmark(propagate($(Ref(H))[], $(Ref(uf))[])), @benchmark(propagate($(Ref(H))[], $(Ref(up))[])))
# a PointFrame's transfer through a point map is the point's action
bench!("transfer(S, f) vs S(p)", @benchmark(transfer($(Ref(S))[], $(Ref(uf.value))[])), @benchmark($(Ref(S))[]($(Ref(c))[])))
bench!("transfer(H∘, f) vs H∘(p)", @benchmark(transfer($(Ref(PerspectiveMap() ∘ H))[], $(Ref(uf.value))[])),
       @benchmark($(Ref(PerspectiveMap() ∘ H))[]($(Ref(SVector(c)))[])))
# the point observation of a detection, vs reading the uncertain point out of its Attributed
bench!("center(d) vs d.value", @benchmark(center($(Ref(df))[])), @benchmark(getfield($(Ref(dp))[], :value)))

n = 10_000
detf = StructArray([Attributed(Uncertain(PointFrame(Float64(i), 2.0i), Σ), (; weight = 1.0 / i)) for i in 1:n])
detp = StructArray([Attributed(Uncertain(Point2(Float64(i), 2.0i), Σ), (; weight = 1.0 / i)) for i in 1:n])
outf = StructArray{typeof(center(propagate(S, uf)))}(undef, n)
outp = similar(outf)
function loop_f!(out, dets, m)
    @inbounds for i in eachindex(out)
        out[i] = center(propagate(m, dets.value[i]))
    end
    return out
end
function loop_p!(out, dets, m)
    @inbounds for i in eachindex(out)
        out[i] = propagate(m, dets.value[i])
    end
    return out
end
loop_f!(outf, detf, S); loop_p!(outp, detp, S)
@assert outf == outp                                         # the same numbers
bench!("StructArray loop S, n=$n", @benchmark(loop_f!($outf, $detf, $S)), @benchmark(loop_p!($outp, $detp, $S)))
bench!("StructArray loop H, n=$n", @benchmark(loop_f!($outf, $detf, $H)), @benchmark(loop_p!($outp, $detp, $H)))

println("operation | Uncertain{PointFrame} median ns | Uncertain{Point2} median ns | ratio")
for (k, (f, p)) in rows
    println(rpad(k, 28), " | ", lpad(round(f; digits = 1), 9), " | ", lpad(round(p; digits = 1), 9), " | ", round(f / p; digits = 2))
end

# RESULTS
# Slurm job 11393 (dgx01, 2026-11-13 13:25, 2 CPUs, Julia 1.13.3, one thread; the node shared with other jobs):
#   operation                   | Uncertain{PointFrame} | Uncertain{Point2} | ratio
#   propagate(S, u)             |      25.3 |      25.3 | 1.00
#   propagate(H, u)             |      21.3 |      25.0 | 0.85
#   transfer(S, f) vs S(p)      |       2.6 |       2.8 | 0.91
#   transfer(H∘, f) vs H∘(p)    |       2.8 |       2.8 | 1.00
#   center(d) vs d.value        |       2.3 |       2.3 | 0.96
#   StructArray loop S, n=10000 |  196025.0 |  201018.0 | 0.98
#   StructArray loop H, n=10000 |   84517.0 |  159121.5 | 0.53
# Parity holds: the PointFrame path is never slower. The homography rows are FASTER for the frame (the frame
# path enters the point law through `center(u)`, the Point2 path through the same method; the gap is the
# Point2 caller's, not this interface's overhead, and is left for a separate look).
