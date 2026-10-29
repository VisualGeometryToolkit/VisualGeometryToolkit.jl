# The signed-Sampson LM rows through a chart: the scalar row, the residual and
# Jacobian kernels, and the E-manifold anchors — against finite differences of
# the distance along the chart's own retraction.
@testitem "sampson_rows" begin
    import VisualGeometryToolkitCore as VgtkCore
    using Test
    using LinearAlgebra
    using StaticArrays
    using Random
    using Rotations
    using FiniteDiff: FiniteDiff
    using VisualGeometryToolkitCore
    using VisualGeometryToolkitCore: EssentialManifold6, retract, retract_jacobian, signed_sampson_row,
        signed_sampson_jacobian, sampson_rows!, sampson_jacobian_rows!, signed_sampson_distance,
        epipolar_rounding

    rng = Xoshiro(11)
    R = RotMatrix(RotXYZ(0.2, -0.1, 0.3)); t = normalize(SVector(1.0, 0.2, -0.3))
    E = EssentialMat(Tuple(skew(t) * SMatrix{3, 3}(R)))
    n = 60
    X = [SVector(randn(rng), randn(rng), 4 + rand(rng)) for _ in 1:n]
    u₁ = [PerspectiveMap()(x) + 1.0e-3 * SVector(randn(rng), randn(rng)) for x in X]
    u₂ = [PerspectiveMap()(SMatrix{3, 3}(R) * x + t) + 1.0e-3 * SVector(randn(rng), randn(rng)) for x in X]
    sqrtw = sqrt.(rand(rng, n))

    # dM from the chart differential at v, reshaped per coordinate
    dMof(ch, v) = (A = retract_jacobian(ch, v); ntuple(q -> SMatrix{3, 3}(A[:, q]), size(A, 2)))
    dist(ch, v, i) = signed_sampson_distance(retract(ch, v), u₁[i], u₂[i])

    @testset "anchors: EssentialManifold5/6(E) reproduce E up to scale and sign" begin
        for ch in (EssentialManifold5(E), EssentialManifold6(E))
            M = SMatrix{3, 3}(retract(ch, zero(SVector{VgtkCore.tangent_size(ch), Float64})))
            s = norm(M) / norm(SMatrix{3, 3}(E))
            @test M ≈ s * SMatrix{3, 3}(E) || M ≈ -s * SMatrix{3, 3}(E)
        end
    end

    @testset "row = the distance and its chart derivative (central differences)" begin
        for (ch, v) in (
                (EssentialManifold6(E), SVector(0.01, -0.02, 0.005, 0.03, -0.01, 0.02)),
                (FundMatManifold7(SMatrix{3, 3}(E)), SVector(0.01, -0.02, 0.005, 0.03, -0.01, 0.02, 0.01)),
            )
            M = retract(ch, v); dM = dMof(ch, v)
            for i in (1, 7, 30)
                d, ∂d = signed_sampson_row(M, dM, u₁[i], u₂[i])
                # Not `==`: both are `p.g / sqrt(p.J²)` from the SAME
                # `_epipolar_parts`, but both are `@inline` and the row also
                # forms the `D·x₁` / `Dᵀ·x₂` products, so the backend fuses the
                # shared part differently at the two inline sites. WHICH
                # correspondences differ is codegen-dependent. Measured over 120
                # comparisons (60 pairs × both charts): 26 differ, worst 35 ulps
                # = 7.0e-15 relative = 31.6 eps. rtol is ~14× that, not 1.4×.
                @test d ≈ signed_sampson_distance(M, u₁[i], u₂[i]) rtol = 1.0e-13
                fd = FiniteDiff.finite_difference_gradient(w -> dist(ch, SVector{length(v)}(w), i), Vector(v))   # retract takes an SVector
                @test ∂d ≈ fd rtol = 1.0e-5 atol = 1.0e-9
            end
        end
    end

    # The rows CONTRACT `signed_sampson_jacobian` (one N×9 product, fundmat.jl
    # `_contracted_sampson_terms`); the uncontracted formula, with ∂G_x built
    # direction by direction, must agree to rounding.
    @testset "contracted row = signed_sampson_jacobian with the explicit ∂G_x" begin
        for (ch, v) in (
                (EssentialManifold6(E), SVector(0.01, -0.02, 0.005, 0.03, -0.01, 0.02)),
                (FundMatManifold7(SMatrix{3, 3}(E)), SVector(0.01, -0.02, 0.005, 0.03, -0.01, 0.02, 0.01)),
            )
            M = SMatrix{3, 3}(retract(ch, v)); dM = dMof(ch, v)
            for i in 1:n
                x1 = SVector(u₁[i][1], u₁[i][2], 1.0); x2 = SVector(u₂[i][1], u₂[i][2], 1.0)
                Fx1 = M * x1; Fᵀx2 = M' * x2
                G_x = SVector(Fx1[1], Fx1[2], Fᵀx2[1], Fᵀx2[2])
                G_θ = SVector(map(D -> dot(x2, D * x1), dM))
                ∂G_x = reduce(hcat, map(D -> (a = D * x1; b = D' * x2; SVector(a[1], a[2], b[1], b[2])), dM))
                _, ∂d = signed_sampson_row(M, dM, u₁[i], u₂[i])
                @test ∂d ≈ signed_sampson_jacobian(dot(x2, Fx1), G_x, G_θ, ∂G_x) rtol = 1.0e-12 atol = 1.0e-14
            end
        end
    end

    @testset "kernels: √w-weighted rows, one per correspondence" begin
        ch = EssentialManifold6(E); v = SVector(0.01, -0.02, 0.005, 0.03, -0.01, 0.02)
        M = retract(ch, v); dM = dMof(ch, v)
        r = zeros(n); J = zeros(n, 6)
        sampson_rows!(r, M, u₁, u₂, sqrtw)
        sampson_jacobian_rows!(J, M, dM, u₁, u₂, sqrtw)
        for i in 1:n
            d, ∂d = signed_sampson_row(M, dM, u₁[i], u₂[i])
            # Same fma/inlining reason as the row-vs-distance check above: the
            # kernel weights inside its own row function, the right-hand side
            # weights outside it. Measured over 60: 14 differ, worst 40 ulps =
            # 4.7e-15 relative = 21.2 eps. A row near zero is a cancellation, so
            # its relative error is unbounded: CI's runner measured 9.4e-18 absolute
            # on a 4.8e-6 row (2e-12 relative). Hence the absolute floor.
            @test r[i] ≈ sqrtw[i] * d rtol = 1.0e-13 atol = 1.0e-15
            @test J[i, :] ≈ sqrtw[i] * ∂d rtol = 1.0e-14      # the kernel compiles its own fma choices
        end
        # the device kernels themselves, launched on KA.CPU() (the public functions take
        # the CPU fast path there): the same per-row function, the same rows
        KA = VgtkCore.KA
        rk = zeros(n); Jk = zeros(n, 6)
        VgtkCore._row_kernel!(KA.CPU())(VgtkCore._sampson_residual_row!, rk, M, u₁, u₂, sqrtw; ndrange = n)
        VgtkCore._row_kernel!(KA.CPU())(VgtkCore._sampson_jacobian_row!, Jk, M,
            VgtkCore._direction_matrix(dM), u₁, u₂, sqrtw; ndrange = n)
        KA.synchronize(KA.CPU())
        # Both sides run the same per-row scalar function. Exact until
        # 2026-10-07 (0 ulps, 2026-09-22); since the residual row selects its
        # degenerate zero by `ifelse` rather than a branch, the CPU loop
        # vectorizes across rows and its fma contraction differs from the
        # kernel's: measured over the 60 rows, 17 differ, worst 6.9e-17 absolute
        # = 2.0e-15 relative (~9 ulps). The bound is 5× that.
        @test rk ≈ r rtol = 1.0e-14
        @test Jk ≈ J rtol = 1.0e-14
        # Float32 storage and points: the same kernel, single precision
        r32 = zeros(Float32, n)
        sampson_rows!(r32, SMatrix{3, 3, Float32}(M), [Float32.(u) for u in u₁], [Float32.(u) for u in u₂], Float32.(sqrtw))
        @test r32 ≈ Float32.(r) rtol = 1.0f-3
    end

    @testset "the one degeneracy rule: a zero row, never a floor" begin
        F = FundMat(SMatrix{3, 3, Float64}(0, 1, 0, -1, 0, 0, 0, 0, 1))      # both epipolar lines at infinity at the origin
        o = SVector(0.0, 0.0)
        dM = (SMatrix{3, 3, Float64}(I), SMatrix{3, 3, Float64}(1, 0, 0, 0, 0, 0, 0, 0, 0))
        d, ∂d = signed_sampson_row(F, dM, o, o)
        @test d == 0 && all(iszero, ∂d)
        @test signed_sampson_distance(F, o, o) == Inf                             # the distance keeps its Inf
        r = [1.0]; J = ones(1, 2)
        sampson_rows!(r, F, [o], [o], [1.0]); sampson_jacobian_rows!(J, F, dM, [o], [o], [1.0])
        @test r == [0.0] && J == zeros(1, 2)
        # the generic scalar: a vanishing gradient is a zero row
        @test signed_sampson_jacobian(0.3, zero(SVector{4, Float64}), SVector(1.0, 2.0), zeros(SMatrix{4, 2})) == SVector(0.0, 0.0)
    end
end

# The residual sum of squares of a model's Sampson distances: the per-pair
# squares summed (masked), the device kernel equal to the CPU loop.
@testitem "sampson_rss" begin
    import VisualGeometryToolkitCore as VgtkCore
    using Test
    using LinearAlgebra
    using StaticArrays
    using Random
    using VisualGeometryToolkitCore
    using VisualGeometryToolkitCore: sampson_rss, sampson_distance

    rng = Xoshiro(21)
    F = FundMat(SMatrix{3, 3, Float64}(randn(rng, 3, 3)))
    H = HomographyMat(SMatrix{3, 3, Float64}(1.1, 0.02, 1.0e-4, -0.03, 0.95, 2.0e-4, 12.0, -7.0, 1.0))
    u₁ = [SVector(400rand(rng), 400rand(rng)) for _ in 1:50]
    u₂ = [SVector(400rand(rng), 400rand(rng)) for _ in 1:50]
    mask = rand(rng, Bool, 50)
    for M in (F, H)
        @test sampson_rss(M, u₁, u₂) ≈ sum(sampson_distance(M, u₁[i], u₂[i])^2 for i in 1:50) rtol = 1.0e-14
        @test sampson_rss(M, u₁, u₂, mask) ≈ sum(sampson_distance(M, u₁[i], u₂[i])^2 for i in 1:50 if mask[i]) rtol = 1.0e-14
        # the device kernel on KA.CPU(): the same per-pair function
        KA = VgtkCore.KA
        d² = zeros(50)
        VgtkCore._row_kernel!(KA.CPU())(VgtkCore._sampson_sq_row!, d², M, u₁, u₂, mask; ndrange = 50)
        KA.synchronize(KA.CPU())
        @test sum(d²) ≈ sampson_rss(M, u₁, u₂, mask) rtol = 1.0e-14
    end
end
