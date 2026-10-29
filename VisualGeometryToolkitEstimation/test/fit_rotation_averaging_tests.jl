# fit_rotation_averaging — robust rotation averaging on a view graph: exact on
# consistent rotations, an outlier edge down-weighted, noise, prior weights, the
# gauge, a disconnected graph refused.

@testitem "fit_rotation_averaging: exact, robust, gauged" begin
    using Test, Random, LinearAlgebra, StaticArrays, Rotations
    using VisualGeometryToolkitEstimation: fit_rotation_averaging, _averaging_columns
    using VisualGeometryToolkitCore: estimate, converged, rms, spectral_rotation_averaging, CauchyShape, TLSShape

    rng = Xoshiro(17)
    n = 8
    R = Dict(k => (k == 1 ? one(RotMatrix{3, Float64}) : RotMatrix(QuatRotation(normalize(randn(rng, 4))))) for k in 1:n)
    rel(i, j) = R[j]' * R[i]
    err(e) = maximum(k -> rotation_angle(RotMatrix{3}(SMatrix{3, 3}(e[k])' * R[k])), 1:n)
    edges = [(i, j) for i in 1:n for j in (i + 1):min(n, i + 3)]
    @testset "consistent edges: exact" begin
        res = fit_rotation_averaging(Dict(e => rel(e...) for e in edges))
        @test converged(res)
        @test err(estimate(res)) < 1.0e-10
        @test estimate(res)[1] ≈ one(RotMatrix{3, Float64}) atol = 1.0e-12
    end
    @testset "noise and one outlier edge: down-weighted" begin
        noisy = Dict(e => RotMatrix(RotationVec((2.0e-3 .* randn(rng, 3))...)) * rel(e...) for e in edges)
        bad = (2, 4); noisy[bad] = RotMatrix(RotationVec(0.5, -0.3, 0.2)) * noisy[bad]
        res = fit_rotation_averaging(noisy; loss = CauchyShape())
        @test err(estimate(res)) < 5.0e-3
        @test res.weights[bad] < 0.01
        @test res.residuals[bad] > 0.3
        # a zero prior weight drops the edge outright
        p = Dict(e => (e == bad ? 0.0 : 1.0) for e in keys(noisy))
        @test err(estimate(fit_rotation_averaging(noisy, p))) < 5.0e-3
    end
    @testset "zero prior: rms and n over the live edges, bridges and all-zero refused" begin
        noisy = Dict(e => RotMatrix(RotationVec((2.0e-3 .* randn(rng, 3))...)) * rel(e...) for e in edges)
        bad = (2, 4); noisy[bad] = RotMatrix(RotationVec(0.5, -0.3, 0.2)) * noisy[bad]
        p = Dict(e => (e == bad ? 0.0 : 1.0) for e in keys(noisy))
        res = fit_rotation_averaging(noisy, p)
        live = [e for e in keys(noisy) if e != bad]
        @test res.n == length(live)
        @test rms(res) ≈ sqrt(sum(e -> res.residuals[e]^2, live) / length(live))
        @test rms(res) < 0.01                                          # the dropped edge's 0.6 rad is not in it
        chain = Dict((k, k + 1) => rel(k, k + 1) for k in 1:3)
        @test_throws ArgumentError fit_rotation_averaging(chain, Dict(e => (e == (2, 3) ? 0.0 : 1.0) for e in keys(chain)))
        @test_throws ArgumentError fit_rotation_averaging(chain, Dict(e => 0.0 for e in keys(chain)))
    end
    @testset "a frame whose every edge the loss drops is held, not singular" begin
        # frame 9 hangs on two mutually inconsistent edges: the seed splits the
        # difference, both residuals sit far past TLS's cutoff, so its weighted
        # degree is 0 from the first iteration — held at its seed rotation (a
        # small prior on its edges keeps the seed of frames 1..8 clean; the TLS
        # weight is what zeroes them). This was a singular solve.
        R[9] = RotMatrix(QuatRotation(normalize(randn(rng, 4))))
        noisy = Dict(e => RotMatrix(RotationVec((2.0e-3 .* randn(rng, 3))...)) * rel(e...) for e in edges)
        noisy[(7, 9)] = RotMatrix(RotationVec(0.6, 0.0, 0.0)) * rel(7, 9)
        noisy[(8, 9)] = RotMatrix(RotationVec(-0.6, 0.0, 0.0)) * rel(8, 9)
        p = Dict(e => (9 in e ? 1.0e-3 : 1.0) for e in keys(noisy))
        res = fit_rotation_averaging(noisy, p; loss = TLSShape(), scale_multiple = 3.0)
        seed = spectral_rotation_averaging(noisy, p)
        @test res.weights[(7, 9)] == 0 && res.weights[(8, 9)] == 0
        @test all(isfinite, estimate(res)[9])
        @test SMatrix{3, 3}(estimate(res)[9]) ≈ SMatrix{3, 3}(seed[9]) atol = 1.0e-9
        @test err(estimate(res)) < 1.0e-2                              # frames 1..8, at the noise level
        delete!(R, 9)
        # the unknowns: a leaf whose only edge weighs 0 is its own anchor, no columns
        col, nunk = _averaging_columns([1, 2, 3, 4], [(1, 2), (2, 3), (3, 4)], [1.0, 1.0, 0.0])
        @test col[1] == -1 && col[4] == -1 && nunk == 6
        # a zero bridge: the far part moves with its first frame held
        col, nunk = _averaging_columns([1, 2, 3, 4], [(1, 2), (2, 3), (3, 4)], [1.0, 0.0, 1.0])
        @test col[1] == -1 && col[3] == -1 && col[2] >= 0 && col[4] >= 0 && nunk == 6
    end
    @test_throws ArgumentError fit_rotation_averaging(Dict((1, 2) => rel(1, 2), (3, 4) => rel(3, 4)))
    @test_throws ArgumentError fit_rotation_averaging(Dict((1, 2) => rel(1, 2)); loss = TLSShape())
end
