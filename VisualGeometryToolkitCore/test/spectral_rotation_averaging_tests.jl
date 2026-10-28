# =============================================================================
# chordal_averaging (geometry/transforms/rotation_fits.jl): exact on consistent
# relative rotations, the gauge, the weights, noise, and the refusals.
# =============================================================================

@testitem "chordal_averaging: exact, gauged, weighted" begin
    using Test, StaticArrays, LinearAlgebra, Random, Rotations
    using VisualGeometryToolkitCore: chordal_averaging

    rng = Xoshiro(9)
    R = Dict(k => (k == 1 ? one(RotMatrix{3, Float64}) : RotMatrix(QuatRotation(normalize(randn(rng, 4))))) for k in 1:6)
    rel(i, j) = R[j]' * R[i]                                   # f_j = R_ij f_i with d = R_k f_k
    pairs = Dict((i, j) => rel(i, j) for i in 1:6 for j in (i + 1):6)
    est = chordal_averaging(pairs)
    @test est[1] == one(RotMatrix{3, Float64})
    @test all(k -> norm(est[k] - R[k]) < 1.0e-12, 1:6)
    # a chain only (consecutive pairs) is enough, and the gauge is the smallest key
    chain = Dict((k, k + 1) => rel(k, k + 1) for k in 1:5)
    @test all(k -> norm(chordal_averaging(chain)[k] - R[k]) < 1.0e-12, 1:6)
    # noise: the average of many pairs beats a single noisy chain
    noisy = Dict(p => RotMatrix(RotationVec((0.01 .* randn(rng, 3))...)) * Rp for (p, Rp) in pairs)
    e = chordal_averaging(noisy)
    @test maximum(k -> rotation_angle(RotMatrix{3}(e[k]' * R[k])), 1:6) < 0.02
    # a zero weight drops a corrupted pair
    bad = copy(pairs); bad[(1, 2)] = RotMatrix(RotationVec(1.0, 0.0, 0.0)) * pairs[(1, 2)]
    w = Dict(p => (p == (1, 2) ? 0.0 : 1.0) for p in keys(bad))
    @test all(k -> norm(chordal_averaging(bad, w)[k] - R[k]) < 1.0e-12, 1:6)
    @test_throws ArgumentError chordal_averaging(Dict((1, 2) => rel(1, 2), (3, 4) => rel(3, 4)))
    @test_throws ArgumentError chordal_averaging(Dict{Tuple{Int, Int}, RotMatrix{3, Float64, 9}}())
end
