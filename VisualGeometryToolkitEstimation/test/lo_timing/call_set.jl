# =============================================================================
# lo_timing/call_set.jl — THE recorded call set `lo_calls_phototourism_e.bin`
# (its format: bench.jl's header) and its one reader: the pairs, their
# pool-frame correspondences and their E problems. Included by every LO timing
# bench that times on it — this package's test/lo_timing/bench.jl and
# LocalOptimizationDoneRight's (from `pkgdir(VisualGeometryToolkitEstimation)`),
# like VisualGeometryToolkitCore's bench record. The includer has loaded
# VisualGeometryToolkitCore and StaticArrays.
# =============================================================================
using VisualGeometryToolkitCore: CameraModel, Correspondences, EssentialMat, Point2
using VisualGeometryToolkitEstimation: EssentialProblem
using StaticArrays: SMatrix

"The recorded call set, beside this file."
const CALL_SET_FIXTURE = joinpath(@__DIR__, "lo_calls_phototourism_e.bin")

# The pairs (`n`, the pool-frame cameras `Kc1`/`Kc2`, the points `x1`/`x2` as
# 2 × n, the recorded LO seeds) and the selection's outlier half-width.
function read_fixture(path)
    open(path) do io
        String(read(io, 8)) == "LOFX0001" || error("$path: not a LOFX0001 fixture")
        npairs = read(io, Int64); nseeds = read(io, Int64); halfwidth = read(io, Float64)
        rd(k) = [read(io, Float64) for _ in 1:k]
        pairs = map(1:npairs) do _
            n = Int(read(io, Int64))
            Kc1 = SMatrix{3, 3, Float64, 9}(rd(9)); Kc2 = SMatrix{3, 3, Float64, 9}(rd(9))
            x1 = reshape(rd(2n), 2, n); x2 = reshape(rd(2n), 2, n)
            seeds = [EssentialMat(Tuple(rd(9))) for _ in 1:nseeds]
            (; n, Kc1, Kc2, x1, x2, seeds)
        end
        eof(io) || error("$path: trailing bytes")
        return pairs, halfwidth
    end
end

# The pair's pool-frame correspondences, and its E problem (their bearings
# through the pool-frame cameras).
function pool_correspondences(pr)
    P2 = Point2{Float64}
    return Correspondences([P2(pr.x1[1, i], pr.x1[2, i]) for i in 1:pr.n], [P2(pr.x2[1, i], pr.x2[2, i]) for i in 1:pr.n])
end
problem_of(pr) = EssentialProblem(CameraModel(pr.Kc1), CameraModel(pr.Kc2), pool_correspondences(pr))

# The number of recorded calls: every pair's seeds.
call_set_size(pairs) = sum(length(pr.seeds) for pr in pairs)

# Every recorded call once: `call(i, seed)` for each pair `i` and each of its seeds.
function run_call_set(call, pairs)
    for i in eachindex(pairs), s in pairs[i].seeds
        call(i, s)
    end
    return nothing
end

# The µs per call of one run over the call set, each call timed on its own (`time_ns`).
function time_call_set(call, pairs)
    t = UInt64(0)
    for i in eachindex(pairs), s in pairs[i].seeds
        t0 = time_ns(); call(i, s); t += time_ns() - t0
    end
    return t / call_set_size(pairs) / 1.0e3
end

# A bench arm as VisualGeometryToolkitCore's record measures it (bench_record.jl's
# `write_bench_record`): `make()` builds the arm's call `(i, seed) -> …`, its
# workspaces outside the timing; two runs compile and warm it, then its bytes
# per call and `reps` timed runs, sorted.
call_set_arm(make, pairs, reps) = function ()
    call = make()
    run_call_set(call, pairs); run_call_set(call, pairs)
    bytes = @allocated run_call_set(call, pairs)
    return sort!([time_call_set(call, pairs) for _ in 1:reps]), bytes / call_set_size(pairs)
end
