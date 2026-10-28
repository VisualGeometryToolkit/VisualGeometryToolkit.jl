# =============================================================================
# ControlPatchManifold (geometry/cameras/generic_models.jl): a block of a generic
# grid model's controls as a manifold — the reference's direction chart, the norm
# kept, points additive; restrict, the NLLSsolver variable.
# =============================================================================

@testitem "ControlPatchManifold: a block of grid controls on the direction chart" begin
    using Test, StaticArrays, LinearAlgebra, Random
    using ForwardDiff: ForwardDiff
    import NLLSsolver
    using VisualGeometryToolkitCore
    using VisualGeometryToolkitCore: ControlPatchManifold, direction_tangents, retract, recenter, retract_jacobian, local_coordinates,
        tangent_size, model, restrict, tangent_part

    rng = Xoshiro(20261028)
    dirs = SMatrix{2, 3}(ntuple(_ -> (1.0 + 0.2 * rand(rng)) * normalize(SVector(0.3 * randn(rng), 0.3 * randn(rng), 1.0)), 6))   # not unit
    pts = SMatrix{2, 3}(ntuple(_ -> SVector(1.0e-3 * randn(rng), 1.0e-3 * randn(rng), 0.0), 6))
    c = ControlPatchManifold(dirs)
    n = ControlPatchManifold(dirs, pts)

    @testset "sizes and the model" begin
        @test tangent_size(c) == 12 && tangent_size(n) == 30
        @test model(c).directions === dirs && model(c).points === nothing && model(n).points === pts
    end

    @testset "a direction steps on its sphere by the reference's chart, its norm kept; a point additively" begin
        v = SVector{30}(1.0e-2 .* randn(rng, 30))
        m = retract(n, v)
        for k in 1:6
            d = dirs[k]; t1, t2 = direction_tangents(normalize(d))
            @test m.directions[k] ≈ norm(d) * normalize(d + v[5k - 4] * t1 + v[5k - 3] * t2) rtol = 1.0e-15
            @test norm(m.directions[k]) ≈ norm(d) rtol = 1.0e-14
            @test m.points[k] == pts[k] + SVector(v[5k - 2], v[5k - 1], v[5k])
        end
        @test model(recenter(n, v)) == m && recenter(n, v) isa typeof(n)
        @test local_coordinates(n, m) ≈ v rtol = 1.0e-10
        w = SVector{12}(1.0e-2 .* randn(rng, 12))
        @test local_coordinates(c, retract(c, w)) ≈ w rtol = 1.0e-10
        @test all(k -> retract(c, zero(w)).directions[k] ≈ dirs[k], 1:6)
    end

    @testset "the step's derivative at zero is (t₁, t₂) per direction, the identity per point" begin
        flat(x) = vcat([vcat(x.directions[k], x.points === nothing ? SVector{0, eltype(x.directions[k])}() : x.points[k]) for k in 1:6]...)
        for ch in (c, n)
            J = ForwardDiff.jacobian(u -> flat(retract(ch, u)), zeros(tangent_size(ch)))
            @test J ≈ retract_jacobian(ch) rtol = 1.0e-12 atol = 1.0e-15
        end
        t1, t2 = direction_tangents(normalize(dirs[1]))
        @test retract_jacobian(c)[1:3, 1:2] ≈ hcat(t1, t2)
    end

    @testset "an NLLSsolver variable; restricted by mask (a pupil-plane model: Δz held)" begin
        u = randn(rng, 40) .* 1.0e-2
        @test NLLSsolver.update(n, u, 5) == recenter(n, tangent_part(n, u, 4))
        @test Int(NLLSsolver.nvars(n)) == 30
        mask = SVector{30}(ntuple(i -> i % 5 != 0, 30))
        r = restrict(n, mask)
        @test tangent_size(r) == 24
        s = NLLSsolver.update(r, u, 1)
        @test all(k -> model(s).points[k][3] == pts[k][3], 1:6)                                  # z exactly where it was
        @test model(s).points[2] == pts[2] + SVector(u[7], u[8], 0.0)                           # (δ₁, δ₂, Δx, Δy) per control
        @test (@inferred NLLSsolver.update(c, u, 1)) isa typeof(c)
    end
end
