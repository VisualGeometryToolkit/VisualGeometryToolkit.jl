# =============================================================================
# spectral_rotation_averaging (geometry/transforms/rotation_fits.jl): exact on
# consistent relative rotations (complete graph and chain), the gauge, noise,
# a zero weight, the refusals.
# =============================================================================

@testitem "spectral_rotation_averaging: exact, gauged, weighted" begin
    using Test, StaticArrays, LinearAlgebra, Random, Rotations
    using VisualGeometryToolkitCore: spectral_rotation_averaging

    rng = Xoshiro(9)
    R = Dict(k => (k == 1 ? one(RotMatrix{3, Float64}) : RotMatrix(QuatRotation(normalize(randn(rng, 4))))) for k in 1:6)
    rel(i, j) = R[j]' * R[i]                                   # f_j = R_ij f_i with d = R_k f_k
    pairs = Dict((i, j) => rel(i, j) for i in 1:6 for j in (i + 1):6)
    err(e) = maximum(k -> rotation_angle(RotMatrix{3}(e[k]' * R[k])), 1:6)
    est = spectral_rotation_averaging(pairs)
    @test est[1] ≈ one(RotMatrix{3, Float64}) atol = 1.0e-12
    @test err(est) < 1.0e-10
    # a chain only (consecutive pairs) is enough, and the gauge is the smallest key
    @test err(spectral_rotation_averaging(Dict((k, k + 1) => rel(k, k + 1) for k in 1:5))) < 1.0e-10
    # noise: close to the truth
    noisy = Dict(p => RotMatrix(RotationVec((0.01 .* randn(rng, 3))...)) * Rp for (p, Rp) in pairs)
    @test err(spectral_rotation_averaging(noisy)) < 0.02
    # a zero weight drops a corrupted pair
    bad = copy(pairs); bad[(1, 2)] = RotMatrix(RotationVec(1.0, 0.0, 0.0)) * pairs[(1, 2)]
    w = Dict(p => (p == (1, 2) ? 0.0 : 1.0) for p in keys(bad))
    @test err(spectral_rotation_averaging(bad, w)) < 1.0e-10
    @test_throws ArgumentError spectral_rotation_averaging(Dict((1, 2) => rel(1, 2), (3, 4) => rel(3, 4)))
    @test_throws ArgumentError spectral_rotation_averaging(Dict{Tuple{Int, Int}, RotMatrix{3, Float64, 9}}())
    # a zero-weight bridge splits the graph: refused, not a degenerate eigenspace
    # (the chain (1,2),(2,3),(3,4) with w(2,3) = 0 once came back 180° off)
    chain = Dict((k, k + 1) => rel(k, k + 1) for k in 1:3)
    bridge = Dict(p => (p == (2, 3) ? 0.0 : 1.0) for p in keys(chain))
    @test_throws ArgumentError spectral_rotation_averaging(chain, bridge)
    @test_throws ArgumentError spectral_rotation_averaging(chain, Dict(p => -1.0 for p in keys(chain)))
end
