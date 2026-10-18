# SO(3)/SE(3) Lie helpers: exp side is explicit Rodrigues (AD-safe, hoisted
# from RayTracedCalibration); log side a Shepperd quaternion + atan, generic
# over Real (ForwardDiff Duals, at the identity too).
# Cross-checks against Rotations.jl validate the exp side without inheriting
# its bit patterns in production paths (downstream pins bit-exact behavior).

@testitem "lie" begin
    using Test
    using StaticArrays
    using LinearAlgebra: I, det, norm
    using Rotations: RotMatrix, RotationVec
    using VisualGeometryToolkitCore: exp_so3, log_so3, left_jacobian, exp_se3, log_se3, SO3,
        EuclideanMap, skew

    @testset "Lie helpers (SO(3), SE(3))" begin
        for T in (Float32, Float64)
            tol = sqrt(eps(T))
            ωs = [
                SVector{3, T}(0.3, -0.2, 0.5),
                SVector{3, T}(-1.2, 0.7, 2.0),
                SVector{3, T}(1.0e-3, 2.0e-3, -1.0e-3),
                SVector{3, T}(1.0e-6, 0, 1.0e-6),      # small-angle Taylor branch
                zero(SVector{3, T}),
            ]

            @testset "exp_so3 vs Rotations ($T)" begin
                for ω in ωs
                    R = exp_so3(ω)
                    R_ref = SMatrix{3, 3, T, 9}(RotMatrix(RotationVec(ω...)))
                    @test maximum(abs.(R .- R_ref)) ≤ 20 * tol
                    @test abs(det(R) - 1) ≤ 20 * tol            # proper rotation
                    @test maximum(abs.(R' * R .- SMatrix{3, 3, T, 9}(I))) ≤ 20 * tol
                end
            end

            @testset "log ∘ exp roundtrip ($T)" begin
                for ω in ωs
                    norm(ω) < T(π) || continue
                    @test norm(log_so3(exp_so3(ω)) - ω) ≤ 50 * tol * max(1, norm(ω))
                end
            end

            @testset "log_so3 vs Rotations; near π ($T)" begin
                for ω in (ωs..., SVector{3, T}(0, 0, T(π) - T(1.0e-3)), SVector{3, T}(1.8, -1.9, 1.1))
                    R = exp_so3(ω)
                    rv = RotationVec(RotMatrix{3, T}(R))
                    @test norm(log_so3(R) - SVector{3, T}(rv.sx, rv.sy, rv.sz)) ≤ 20 * eps(T) * max(1, norm(ω))
                    @test log_so3(RotMatrix{3, T}(R)) === log_so3(SMatrix{3, 3, T, 9}(R))   # any static 3×3
                end
            end

            @testset "left_jacobian(SO3, ω) ($T)" begin
                # FD check: J(ω)·ρ ≈ translation part of d/ds exp_se3(s·ω-path)…
                # simpler identity: for small ω, J ≈ I + ½[ω]×
                ω = SVector{3, T}(1.0e-4, -2.0e-4, 1.0e-4)
                J = left_jacobian(SO3, ω)
                @test maximum(abs.(J .- (SMatrix{3, 3, T, 9}(I) + T(0.5) * skew(ω)))) ≤ 5.0e-7
                # column-scaling: J(0) = I
                @test left_jacobian(SO3, zero(SVector{3, T})) ≈ SMatrix{3, 3, T, 9}(I)
            end

            @testset "exp_se3 / log_se3 roundtrip ($T)" begin
                for ω in ωs
                    norm(ω) < T(π) || continue
                    ρ = SVector{3, T}(0.1, -0.4, 0.25)
                    E = exp_se3(ω, ρ)
                    ω2, ρ2 = log_se3(E)
                    @test norm(ω2 - ω) ≤ 100 * tol * max(1, norm(ω))
                    @test norm(ρ2 - ρ) ≤ 100 * tol * max(1, norm(ρ))
                end
            end
        end

        @testset "exp_se3 composes with EuclideanMap ops" begin
            ω = SVector{3, Float64}(0.2, -0.1, 0.4)
            ρ = SVector{3, Float64}(1.0, 2.0, -0.5)
            E = exp_se3(ω, ρ)
            Ei = inv(E) ∘ E
            @test maximum(abs.(SMatrix{3, 3, Float64, 9}(Ei.R) .- SMatrix{3, 3, Float64, 9}(I))) ≤ 1.0e-12
            @test norm(SVector{3, Float64}(Ei.t)) ≤ 1.0e-12
        end
    end

    @testset "log_so3 is Dual-generic, exact at the identity" begin
        using ForwardDiff: ForwardDiff
        # d log(exp(ω))/dω = I — at a general ω and AT ω = 0, where the RotationVec
        # path differentiates to NaN
        for ω in (SVector(0.3, -0.2, 0.5), zero(SVector{3, Float64}), SVector(0.0, 0.0, 3.0))
            J = ForwardDiff.jacobian(w -> log_so3(exp_so3(w)), ω)
            @test all(isfinite, J)
            @test J ≈ SMatrix{3, 3, Float64}(I) atol = 1.0e-12
        end
    end
end
