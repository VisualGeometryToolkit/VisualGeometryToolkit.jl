# The harness's second-order propagated moments against Monte Carlo: the point a
# minimal homography transfers, extrapolated far outside its four points, where
# the output is nonlinear enough for the second-order terms to matter.
@testitem "second_order_mc" begin
    using Test
    using LinearAlgebra
    using StaticArrays
    using Random
    using Statistics
    using VisualGeometryCore
    using VisualGeometryCore: retract, constraint_with_jacobians, homography_4pt, minimal_manifold
    const CHar = VisualGeometryCore.Synthetic

    # the H transfer of a query point: the implicit chart solution v(x) of the 4-point
    # constraints and the output π(H(v)·q̃); noise σ on all 16 + 2 measurements
    function transfer_moments(seed, q₀, σ; N = 20000)
        rng = MersenneTwister(seed)
        H, cs = CHar.sample_homography_scene(rng)
        ch = minimal_manifold(CHar._unit(H))
        x₀ = reduce(vcat, [vcat(Vector(c.first), Vector(c.second)) for c in cs])
        pairs(x) = [SVector(x[4i - 3], x[4i - 2]) => SVector(x[4i - 1], x[4i]) for i in 1:4]
        res(z) = (Hv = retract(ch, SVector{8}(z[1:8]));
            reduce(vcat, [Vector(constraint_with_jacobians(Hv, SVector{2}(p.first), SVector{2}(p.second))[1]) for p in pairs(z[9:end])]))
        out(w) = Vector(PerspectiveMap()(SMatrix{3, 3}(retract(ch, SVector{8}(w[1:8]))) * SVector(w[9], w[10], one(eltype(w)))))
        Σδ = σ^2 * Matrix(1.0I, 18, 18)
        Σ1, μ2, Σ2 = CHar.second_order_moments(res, out, 8, x₀, q₀, Σδ)
        y₀ = out(vcat(zeros(8), q₀))
        ys = Vector{Vector{Float64}}()
        for _ in 1:N
            δ = σ * randn(rng, 18)
            c = pairs(x₀ + δ[1:16])
            Ĥ = homography_4pt(c[1].first, c[2].first, c[3].first, c[4].first, c[1].second, c[2].second, c[3].second, c[4].second)
            Ĥ === nothing && continue
            h = SMatrix{3, 3}(Ĥ) * SVector(q₀[1] + δ[17], q₀[2] + δ[18], 1.0)
            push!(ys, Vector(PerspectiveMap()(h)) - y₀)
        end
        Y = reduce(hcat, ys)
        return (; Σ1, μ2, Σ2, mean = vec(mean(Y, dims = 2)), C = cov(Y, dims = 2), n = size(Y, 2))
    end

    # measured 2026-09-22: μ₂ = (−8.28, −0.078), MC mean (−8.68 ± 0.56, 0.062 ± 0.092);
    # diag Σ₁ = (5473, 163), Σ₂ = (5632, 164), MC (6313, 170)
    m = transfer_moments(123, [-900.0, 1600.0], 5.0)
    se = sqrt.(diag(m.C) / m.n)
    @testset "the mean shift is μ₂ (the first-order zero mean is rejected)" begin
        @test all(abs.(m.mean - m.μ2) .< 4 .* se)
        @test abs(m.mean[1]) > 4 * se[1]
    end
    @testset "Σ₂ is closer than Σ₁; the excess left is higher order (pinned)" begin
        @test all(abs.(diag(m.C) - diag(m.Σ2)) .< abs.(diag(m.C) - diag(m.Σ1)))
        @test all(1.0 .<= diag(m.C) ./ diag(m.Σ2) .<= 1.2)
    end
end
