# The division-model projection's derivatives and the radial transfer LM rows:
# one edge rule (project's relative field test) for project, its derivatives
# and the rows — against ForwardDiff and central differences through the sl(3)
# chart ⊕ k.
@testitem "radial_rows" begin
    using Test
    using LinearAlgebra
    using StaticArrays
    using Random
    using ForwardDiff: ForwardDiff
    using FiniteDiff: FiniteDiff
    using VisualGeometryCore
    using VisualGeometryCore: RadialHomography, radial_transfer, radial_transfer_rows!,
        radial_transfer_jacobian_rows!, retract, retract_jacobian, project, backproject

    rng = Xoshiro(5)
    H = HomographyMat(SMatrix{3, 3, Float64}(1.0, 0.01, 1.0e-5, -0.02, 0.98, -2.0e-5, 30.0, -20.0, 1.0))
    c = Point2(640.0, 480.0)

    @testset "transform_deriv(_params) = ForwardDiff / central differences; NaN out of field" begin
        for k in (-2.0e-7, 0.0, 1.5e-7)
            m = RadialHomography{Float64}(H, k, c)
            for _ in 1:20
                ray = SVector(400randn(rng), 400randn(rng), 1.0 + 0.2randn(rng))
                d = project(m, ray)
                d === nothing && continue
                @test transform_deriv(m, ray) ≈ ForwardDiff.jacobian(r -> SVector(project(m, r)...), ray) rtol = 1.0e-10
                # ∂/∂k through a Dual-typed model (project is generic in the element type)
                dk = ForwardDiff.derivative(k) do kk
                    D = typeof(kk)
                    SVector(project(RadialHomography{D}(HomographyMat{D}(Tuple(H)), kk, Point2{D}(c)), ray)...)
                end
                @test vec(transform_deriv_params(m, ray)) ≈ dk rtol = 1.0e-10
            end
        end
        mp = RadialHomography{Float64}(H, 0.5, c)
        out = SVector(5.0, 0.0, 0.1)                                   # z² − 4kρ² < 0
        @test project(mp, out) === nothing
        @test all(isnan, transform_deriv(mp, out))
        @test all(isnan, transform_deriv_params(mp, out))
        # on the axis the projection is the centre and the derivative its limit μ·I
        m0 = RadialHomography{Float64}(H, -1.0e-7, c)
        @test project(m0, SVector(0.0, 0.0, 2.0)) == c
        @test transform_deriv(m0, SVector(0.0, 0.0, 2.0)) ≈ SMatrix{2, 3}(0.5, 0.0, 0.0, 0.5, 0.0, 0.0)
    end

    @testset "rows: residuals, chart ⊕ k Jacobian (central differences), sentinel, kernels" begin
        k = -1.1e-7
        m = RadialHomography{Float64}(H, k, c)
        n = 50
        u₁ = [SVector(1200rand(rng), 900rand(rng)) for _ in 1:n]
        u₂ = [radial_transfer(m, s) + 0.5 * SVector(randn(rng), randn(rng)) for s in u₁]
        sqrtw = sqrt.(rand(rng, 2n))
        ch = HomographyManifold8(SMatrix{3, 3}(H))
        v = SVector(1.0e-3, -2.0e-3, 5.0e-4, 1.0e-3, -1.0e-4, 2.0e-4, 1.0e-6, -2.0e-6)
        dH = VisualGeometryCore.retract_directions(ch, v)
        mv = RadialHomography{Float64}(retract(ch, v), k, c)
        r = zeros(2n); J = zeros(2n, 9)
        radial_transfer_rows!(r, mv, u₁, u₂, sqrtw)
        radial_transfer_jacobian_rows!(J, mv, dH, u₁, sqrtw)
        for i in (1, 23)
            d̂ = radial_transfer(mv, u₁[i])
            @test r[(2i - 1):(2i)] ≈ sqrtw[(2i - 1):(2i)] .* (d̂ - u₂[i]) rtol = 1.0e-14
        end
        rv(w) = (out = zeros(2n); radial_transfer_rows!(out, RadialHomography{Float64}(retract(ch, w[SOneTo(8)]), k + w[9], c), u₁, u₂, sqrtw); out)
        w₀ = vcat(v, 0.0)
        # k's coordinate is ~1e-7 against sl(3)'s ~1e-3, and FiniteDiff's step is one
        # scalar: differentiate in k/|k| so that step fits every coordinate
        S = SVector(ntuple(q -> q == 9 ? abs(k) : 1.0, 9))
        fd = FiniteDiff.finite_difference_jacobian(z -> rv(S .* z), w₀ ./ S, Val(:central)) ./ transpose(S)
        @test J ≈ fd rtol = 1.0e-5 atol = 1.0e-6 * maximum(abs, J)

        # a ray outside the field: a constant residual and a zero Jacobian row
        mp = RadialHomography{Float64}(H, 1.0e-6, c)                    # k > 0: far points leave the field
        far = [SVector(3000.0, 2500.0), SVector(700.0, 500.0)]
        @test radial_transfer(mp, far[1]) === nothing
        @test radial_transfer(mp, far[2]) !== nothing
        rp = zeros(4); Jp = ones(4, 9)
        radial_transfer_rows!(rp, mp, far, [c, c], ones(4))
        radial_transfer_jacobian_rows!(Jp, mp, dH, far, ones(4))
        @test rp[1] == rp[2] == VisualGeometryCore.OUT_OF_FIELD_RESIDUAL
        @test all(iszero, Jp[1:2, :]) && !any(iszero, Jp[3:4, 1:2])

        # radial_in_front: in field AND the lift's third component positive
        Hb = HomographyMat(SMatrix{3, 3, Float64}(1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, -1.0))   # z = −1: a back ray
        z0 = Point2(0.0, 0.0)
        mb = RadialHomography{Float64}(Hb, -0.1, z0)
        @test radial_transfer(mb, SVector(1.0, 0.5)) !== nothing           # k < 0: project is total over back rays …
        @test !VisualGeometryCore.radial_in_front(mb, SVector(1.0, 0.5))    # … but the point is behind
        @test VisualGeometryCore.radial_in_front(RadialHomography{Float64}(H, -0.1 * 1.0e-6, c), SVector(1.0, 0.5))
        @test VisualGeometryCore.radial_in_front(RadialHomography{Float64}(Hb, 0.1, z0), SVector(0.5, 0.5))   # k > 0: the lift is always in front
        @test !VisualGeometryCore.radial_in_front(mp, far[1])               # out of field

        # the device kernels on KA.CPU(): the same row functions, the same rows
        KA = VisualGeometryCore.KA
        rk = zeros(2n); Jk = zeros(2n, 9)
        VisualGeometryCore._row_kernel!(KA.CPU())(VisualGeometryCore._radial_rows!, rk, mv, u₁, u₂, sqrtw; ndrange = n)
        VisualGeometryCore._row_kernel!(KA.CPU())(VisualGeometryCore._radial_jacobian_rows!, Jk, mv, dH, u₁, sqrtw; ndrange = n)
        KA.synchronize(KA.CPU())
        @test rk ≈ r rtol = 1.0e-14
        @test Jk ≈ J rtol = 1.0e-14
    end
end
