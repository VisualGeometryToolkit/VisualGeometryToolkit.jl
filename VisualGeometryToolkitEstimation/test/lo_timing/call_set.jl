# =============================================================================
# lo_timing/call_set.jl — THE recorded call set `lo_calls_phototourism_e.bin`
# (its format: bench.jl's header) and its one reader: the pairs, their
# pool-frame correspondences and their E problems. Included by every LO timing
# bench that times on it — this package's test/lo_timing/bench.jl and
# LocalOptimizationDoneRight's (from `pkgdir(VisualGeometryToolkitEstimation)`),
# like VisualGeometryToolkitCore's bench record. The includer has loaded
# VisualGeometryToolkitCore and StaticArrays.
# =============================================================================
using VisualGeometryToolkitCore: CameraModel, Csponds, EssentialMat, Point2
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
function pool_csponds(pr)
    P2 = Point2{Float64}
    return Csponds([P2(pr.x1[1, i], pr.x1[2, i]) for i in 1:pr.n], [P2(pr.x2[1, i], pr.x2[2, i]) for i in 1:pr.n])
end
problem_of(pr) = EssentialProblem(CameraModel(pr.Kc1), CameraModel(pr.Kc2), pool_csponds(pr))
