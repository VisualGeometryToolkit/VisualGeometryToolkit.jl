# =============================================================================
# SpectralAveraging (geometry/solvers/closed_forms.jl): exact on
# consistent relative rotations (complete graph and chain), the gauge, noise,
# a zero weight (an Attributed edge), the refusals.
# =============================================================================

@testitem "SpectralAveraging: exact, gauged, weighted" begin
    using Test, StaticArrays, LinearAlgebra, Random, Rotations
    using VisualGeometryToolkitCore: solve, EstimationProblem, AbsoluteRotations, SpectralAveraging, Attributed

    rng = Xoshiro(9)
    R = Dict(k => (k == 1 ? one(RotMatrix{3, Float64}) : RotMatrix(QuatRotation(normalize(randn(rng, 4))))) for k in 1:6)
    rel(i, j) = R[j]' * R[i]                                   # f_j = R_ij f_i with d = R_k f_k
    pairs = Dict((i, j) => rel(i, j) for i in 1:6 for j in (i + 1):6)
    err(e) = maximum(k -> rotation_angle(RotMatrix{3}(e[k]' * R[k])), 1:6)
    est = solve(EstimationProblem(AbsoluteRotations, pairs), SpectralAveraging()).u
    @test est[1] ≈ one(RotMatrix{3, Float64}) atol = 1.0e-12
    @test err(est) < 1.0e-10
    # a chain only (consecutive pairs) is enough, and the gauge is the smallest key
    @test err(solve(EstimationProblem(AbsoluteRotations, Dict((k, k + 1) => rel(k, k + 1) for k in 1:5)), SpectralAveraging()).u) < 1.0e-10
    # noise: close to the truth
    noisy = Dict(p => RotMatrix(RotationVec((0.01 .* randn(rng, 3))...)) * Rp for (p, Rp) in pairs)
    @test err(solve(EstimationProblem(AbsoluteRotations, noisy), SpectralAveraging()).u) < 0.02
    # a zero weight drops a corrupted pair
    bad = copy(pairs); bad[(1, 2)] = RotMatrix(RotationVec(1.0, 0.0, 0.0)) * pairs[(1, 2)]
    weighted = Dict(p => Attributed(Rp, (; weight = (p == (1, 2) ? 0 : 1))) for (p, Rp) in bad)   # integer weights
    @test err(solve(EstimationProblem(AbsoluteRotations, weighted), SpectralAveraging()).u) < 1.0e-10
    # Attributed and bare edges mix; a bare edge weighs 1
    mixed = Dict{Tuple{Int, Int}, Any}(p => (p == (1, 2) ? Attributed(Rp, (; weight = 0.0)) : Rp) for (p, Rp) in bad)
    @test err(solve(EstimationProblem(AbsoluteRotations, mixed), SpectralAveraging()).u) < 1.0e-10
    @test_throws MethodError solve(EstimationProblem(AbsoluteRotations, bad), SpectralAveraging();
                                   weights = Dict(p => 1.0 for p in keys(bad)))   # no weights dict
    @test_throws ArgumentError solve(EstimationProblem(AbsoluteRotations, Dict{Tuple{Int, Int}, RotMatrix{3, Float64, 9}}()), SpectralAveraging())
    # a split graph (two components, or a zero-weight bridge) is the caller's to refuse:
    # Estimation's RotationAveraging checks its ViewGraph (fit_rotation_averaging_tests.jl)
    chain = Dict((k, k + 1) => rel(k, k + 1) for k in 1:3)
    @test_throws ArgumentError solve(EstimationProblem(AbsoluteRotations, Dict(p => Attributed(Rp, (; weight = -1.0)) for (p, Rp) in chain)), SpectralAveraging())
end
