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

# left_jacobian's small-angle series: accurate where the closed form cancels
# (θ below 1/4), and the closed form untouched above it.
@testitem "left_jacobian small angles" begin
    using Test
    using StaticArrays
    using Random
    using LinearAlgebra: I, norm, normalize
    using ForwardDiff
    using VisualGeometryToolkitCore: left_jacobian, SO3, skew

    # The closed form as it was before the series covered θ < 1/4 (small-angle switch at θ² < eps).
    function left_jacobian_closed(ω::SVector{3, T}) where {T}
        θ² = ω[1] * ω[1] + ω[2] * ω[2] + ω[3] * ω[3]
        K = skew(ω)
        if θ² < eps(T)
            a = T(0.5) - θ² / T(24)
            b = T(1) / T(6) - θ² / T(120)
        else
            θ = sqrt(θ²)
            a = (one(T) - cos(θ)) / θ²
            b = (θ - sin(θ)) / (θ² * θ)
        end
        return SMatrix{3, 3, T, 9}(I) + a * K + b * (K * K)
    end

    function reference(ω::SVector{3, Float64})
        return setprecision(BigFloat, 300) do
            w = BigFloat.(ω)
            θ = sqrt(sum(abs2, w))
            K = [0 -w[3] w[2]; w[3] 0 -w[1]; -w[2] w[1] 0]
            a = (1 - cos(θ)) / θ^2
            b = (θ - sin(θ)) / θ^3
            Matrix(I, 3, 3) + a * K + b * K * K
        end
    end

    @testset "bit-identical to the closed form for θ in [1/4, π]" begin
        rng = Random.Xoshiro(7)
        for _ in 1:10^4
            u = normalize(randn(rng, SVector{3, Float64}))
            ω = (0.25 + (π - 0.25) * rand(rng)) * u
            @test left_jacobian(SO3, ω) == left_jacobian_closed(ω)
        end
    end

    @testset "accuracy against a 300-bit reference, θ in [1e-12, 1]" begin
        worst = 0.0
        for θ in exp10.(range(-12, 0; length = 121))
            for u in (normalize(SVector(0.3, -0.5, 0.8)), SVector(1.0, 0, 0), SVector(0, 0, -1.0))
                ω = θ * u
                J = left_jacobian(SO3, ω)
                worst = max(worst, maximum(abs.(Float64.(reference(ω)) .- J)) / norm(J) / eps())
            end
        end
        @test worst ≤ 4
        # the closed form alone was off by thousands of eps here
        ω = 1.0e-5 * normalize(SVector(0.3, -0.5, 0.8))
        @test maximum(abs.(Float64.(reference(ω)) .- left_jacobian_closed(ω))) > 100 * eps()
    end

    @testset "series and closed form agree at the switch" begin
        ω = (0.25 - 1.0e-12) * normalize(SVector(0.3, -0.5, 0.8))
        @test maximum(abs.(left_jacobian(SO3, ω) .- left_jacobian_closed(ω))) ≤ 2 * eps()
    end

    @testset "ForwardDiff through the series branch" begin
        ω0 = SVector(1.0e-3, -2.0e-3, 0.5e-3)
        f(x) = vec(Matrix(left_jacobian(SO3, SVector{3}(x))))
        Jd = ForwardDiff.jacobian(f, collect(ω0))
        h = 1.0e-6
        Jfd = reduce(hcat, [(f(collect(ω0) .+ h .* e) .- f(collect(ω0) .- h .* e)) ./ 2h for e in eachcol(Matrix(I, 3, 3))])
        @test all(isfinite, Jd)
        @test maximum(abs.(Jd .- Jfd)) ≤ 1.0e-8
        # at the origin dJ/dω = ½ d[ω]×: finite and exact
        J0 = ForwardDiff.jacobian(f, zeros(3))
        @test all(isfinite, J0)
    end

    @testset "Float32 series" begin
        ω = SVector{3, Float32}(1.0f-3, 2.0f-3, -1.0f-3)
        J = left_jacobian(SO3, ω)
        @test maximum(abs.(Float64.(J) .- Float64.(reference(Float64.(ω))))) ≤ 4 * eps(Float32)
    end
end

@testitem "exp_so3 and log_so3 bits" begin
    using Test
    using StaticArrays
    using Random
    using LinearAlgebra: I, normalize
    using VisualGeometryToolkitCore: exp_so3, log_so3, skew

    # exp_so3 and log_so3 as they were before the Rodrigues coefficients became
    # shared helpers (2026-10): RadiometricCamera's goldens pin exp_so3's bits,
    # so a refactor of either must reproduce these exactly.
    function exp_so3_frozen(ω::SVector{3, T}) where {T}
        θ² = ω[1] * ω[1] + ω[2] * ω[2] + ω[3] * ω[3]
        K = skew(ω)
        if θ² < eps(T)
            a = one(T) - θ² / T(6)
            b = T(0.5) - θ² / T(24)
        else
            θ = sqrt(θ²)
            a = sin(θ) / θ
            b = (one(T) - cos(θ)) / θ²
        end
        return SMatrix{3, 3, T, 9}(I) + a * K + b * (K * K)
    end
    function log_so3_frozen(R::SMatrix{3, 3, T, 9}) where {T}
        r11, r21, r31, r12, r22, r32, r13, r23, r33 = Tuple(R)
        tr = r11 + r22 + r33
        if tr >= r11 && tr >= r22 && tr >= r33
            w = sqrt(1 + tr) / 2; d = 4w
            x = (r32 - r23) / d; y = (r13 - r31) / d; z = (r21 - r12) / d
        elseif r11 >= r22 && r11 >= r33
            x = sqrt(1 + r11 - r22 - r33) / 2; d = 4x
            w = (r32 - r23) / d; y = (r12 + r21) / d; z = (r13 + r31) / d
        elseif r22 >= r33
            y = sqrt(1 - r11 + r22 - r33) / 2; d = 4y
            w = (r13 - r31) / d; x = (r12 + r21) / d; z = (r23 + r32) / d
        else
            z = sqrt(1 - r11 - r22 + r33) / 2; d = 4z
            w = (r21 - r12) / d; x = (r13 + r31) / d; y = (r23 + r32) / d
        end
        if w < 0
            w, x, y, z = -w, -x, -y, -z
        end
        s² = x * x + y * y + z * z
        f = s² < eps(T) ? 2 / w * (1 - s² / (3 * w * w)) : 2 * atan(sqrt(s²), w) / sqrt(s²)
        return SVector{3, T}(f * x, f * y, f * z)
    end

    for T in (Float64, Float32)
        rng = Random.Xoshiro(20261027)
        same_exp = same_log = 0
        n = 10^4
        for _ in 1:n
            θ = exp(log(1.0e-12) + (log(Float64(π)) - log(1.0e-12)) * rand(rng))   # log-uniform on [1e-12, π]
            ω = SVector{3, T}(θ * normalize(randn(rng, SVector{3, Float64})))
            R = exp_so3(ω)
            same_exp += R === exp_so3_frozen(ω)
            same_log += log_so3(R) === log_so3_frozen(R)
        end
        @test same_exp == n
        @test same_log == n
    end
end

# exp_se3 on Real element types: ForwardDiff Duals through the twist (a pose
# retraction differentiated by an NLLS solver), against central differences;
# mixed element types promote; Float64 is unchanged.
@testitem "exp_se3 under ForwardDiff" begin
    using Test, StaticArrays, LinearAlgebra
    using ForwardDiff: ForwardDiff
    using FiniteDiff: FiniteDiff
    using VisualGeometryToolkitCore
    using VisualGeometryToolkitCore: exp_se3, exp_so3, left_jacobian, SO3, rotation, translation

    flat(E) = vcat(vec(SMatrix{3, 3}(rotation(E))), SVector{3}(translation(E)))
    f(v) = flat(exp_se3(SVector(v[1], v[2], v[3]), SVector(v[4], v[5], v[6])))
    for v0 in (SVector(0.3, -0.2, 0.1, 0.4, -0.5, 0.2), SVector(1.0e-9, 2.0e-9, -1.0e-9, 0.1, 0.2, 0.3), SVector(2.5, 0.4, -0.3, 1.0, 0.0, -2.0))
        J = ForwardDiff.jacobian(f, v0)
        Jfd = FiniteDiff.finite_difference_jacobian(w -> Vector(f(SVector{6}(w))), Vector(v0), Val(:central))
        @test J ≈ Jfd rtol = 1.0e-7 atol = 1.0e-9
        @test f(v0) == flat(exp_se3(SVector{3}(v0[1:3]), SVector{3}(v0[4:6])))         # the Float64 values unchanged by the Duals' path
    end
    # the retraction exp_se3(ω, ρ) ∘ T of a camera-to-world pose, differentiated (RadiometricCamera's `retract`)
    T0 = EuclideanMap(VisualGeometryToolkitCore.RotMatrix{3}(exp_so3(SVector(0.3, -0.2, 0.1))), SVector(0.1, 0.2, 1.0))
    g(v) = flat(exp_se3(SVector(v[1], v[2], v[3]), SVector(v[4], v[5], v[6])) ∘ T0)
    Jg = ForwardDiff.jacobian(g, zeros(SVector{6}))
    @test Jg ≈ FiniteDiff.finite_difference_jacobian(w -> Vector(g(SVector{6}(w))), zeros(6), Val(:central)) rtol = 1.0e-7 atol = 1.0e-9
    # mixed element types promote; Float32 stays Float32
    @test eltype(translation(exp_se3(SVector(0.1, 0.2, 0.3), SVector(1, 2, 3)))) === Float64
    @test eltype(translation(exp_se3(SVector(0.1f0, 0.2f0, 0.3f0), SVector(1.0f0, 2.0f0, 3.0f0)))) === Float32
    E = exp_se3(SVector(0.1, 0.2, 0.3), SVector(1.0, 2.0, 3.0))
    @test SVector{3}(translation(E)) == left_jacobian(SO3, SVector(0.1, 0.2, 0.3)) * SVector(1.0, 2.0, 3.0)
end
