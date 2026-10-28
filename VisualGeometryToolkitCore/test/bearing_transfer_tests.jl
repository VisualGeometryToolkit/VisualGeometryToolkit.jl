# =============================================================================
# The transfer constraint on unit bearings (geometry/forms.jl,
# geometry/solvers/homography_solvers.jl): zero on exact rotations at any field
# angle, its Jacobians against ForwardDiff, the two-sided Sampson covariance,
# and the bearing rows against the constraint and ForwardDiff.
# =============================================================================

@testitem "bearing transfer: constraint, Jacobians, rows" begin
    using Test, StaticArrays, LinearAlgebra, Random, Rotations
    using VisualGeometryToolkitCore: TransferForm, HomographyMat, constraint_with_jacobians, s2_tangent_basis,
        transfer_whitening!, transfer_rows!, transfer_jacobian_rows!, whitening_factor
    import ForwardDiff

    rng = Xoshiro(7)
    tangent_M(f) = (b = s2_tangent_basis(f); hcat(b[1], b[2]) * (SMatrix{2, 2}(randn(rng, 2, 2)) .* 1.0e-3))   # tangent, ~1 mrad/px
    H(R) = HomographyMat{Float64}(Tuple(SMatrix{3, 3}(R)))
    for _ in 1:50
        R = RotMatrix(QuatRotation(normalize(randn(rng, 4))))
        f₁ = normalize(randn(rng, SVector{3, Float64}))          # the whole sphere: past 90° of every axis
        f₂ = R * f₁
        M₁ = tangent_M(f₁); M₂ = tangent_M(f₂)
        g, G_x, G_θ = constraint_with_jacobians(TransferForm(H(R)), f₁, f₂; M₁, M₂)
        @test norm(g) < 1.0e-14
        # G_θ: ∂g/∂vec(H) by ForwardDiff
        gH(h) = first(constraint_with_jacobians(TransferForm(HomographyMat(Tuple(h))), f₁, f₂; M₁, M₂))
        @test G_θ ≈ ForwardDiff.jacobian(gH, SVector{9}(Tuple(SMatrix{3, 3}(R)))) rtol = 1.0e-10
        # G_x: the f₁ block along f₁(δ) = normalize(f₁ + M₁δ), the f₂ block at ĥ = f₂ (exact there)
        g1(δ) = first(constraint_with_jacobians(TransferForm(H(R)), normalize(f₁ + M₁ * δ), f₂; M₁, M₂))
        g2(δ) = first(constraint_with_jacobians(TransferForm(H(R)), f₁, normalize(f₂ + M₂ * δ); M₁, M₂))
        @test G_x[:, 1:2] ≈ ForwardDiff.jacobian(g1, zeros(SVector{2})) rtol = 1.0e-9
        @test G_x[:, 3:4] ≈ ForwardDiff.jacobian(g2, zeros(SVector{2})) rtol = 1.0e-9
        # the two-sided Sampson covariance P(R M₁M₁ᵀRᵀ + M₂M₂ᵀ)Pᵀ
        b = s2_tangent_basis(f₂); P = vcat(b[1]', b[2]')
        @test G_x * G_x' ≈ P * (R * M₁ * M₁' * R' + M₂ * M₂') * P' rtol = 1.0e-10
        # an antipodal transfer is not cheiral: g is small but the parts say so
        @test !VisualGeometryToolkitCore._bearing_transfer_parts(H(R), f₁, -f₂).cheiral
    end
    @testset "the bearing rows: W·g, and their Jacobian along a rotation chart by ForwardDiff" begin
        n = 40
        R0 = RotMatrix(QuatRotation(normalize(randn(rng, 4))))
        f₁ = [normalize(randn(rng, SVector{3, Float64})) for _ in 1:n]
        f₂ = [normalize(R0 * f + 1.0e-3 * randn(rng, SVector{3, Float64})) for f in f₁]
        M₁ = tangent_M.(f₁); M₂ = tangent_M.(f₂)
        W = Vector{SMatrix{2, 2, Float64, 4}}(undef, n)
        transfer_whitening!(W, H(R0), f₁, f₂; M₁, M₂)
        sqrtw = rand(rng, 2n)
        r = zeros(2n)
        transfer_rows!(r, H(R0), W, f₁, f₂, sqrtw)
        for i in 1:n
            g, G_x, _ = constraint_with_jacobians(TransferForm(H(R0)), f₁[i], f₂[i]; M₁ = M₁[i], M₂ = M₂[i])
            @test W[i] ≈ whitening_factor(G_x)
            @test r[2i-1:2i] ≈ sqrtw[2i-1:2i] .* (W[i] * g)
        end
        Rof(ω) = exp_skew(ω) * R0
        exp_skew(ω) = SMatrix{3, 3}(VisualGeometryToolkitCore.exp_so3(SVector{3}(ω[1], ω[2], ω[3])))
        function rows(ω)
            out = zeros(eltype(ω), 2n)
            Rω = Rof(ω)
            Hω = HomographyMat{eltype(ω)}(Tuple(SMatrix{3, 3}(Rω)))
            transfer_rows!(out, Hω, W, f₁, f₂, sqrtw)
            return out
        end
        dH = ntuple(q -> (e = SVector{3}(ntuple(k -> Float64(k == q), 3)); SMatrix{3, 3}(0, e[3], -e[2], -e[3], 0, e[1], e[2], -e[1], 0) * SMatrix{3, 3}(R0)), 3)
        J = zeros(2n, 3)
        transfer_jacobian_rows!(J, H(R0), dH, W, f₁, f₂, sqrtw)
        @test J ≈ ForwardDiff.jacobian(rows, zeros(3)) rtol = 1.0e-9
    end
end
