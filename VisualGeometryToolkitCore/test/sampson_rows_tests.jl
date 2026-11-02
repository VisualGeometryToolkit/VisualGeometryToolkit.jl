# The Sampson LM rows through a chart: `sampson_residual_with_jacobian` (the one
# path's residual and exact model Jacobian), the residual and Jacobian kernels,
# the epipolar models' `measurement_gradient_jacobian` and their bilinear
# contraction of it, and the E-manifold anchors — against finite differences of
# the residual along the chart's own retraction.
@testitem "sampson_rows" begin
    import VisualGeometryToolkitCore as VgtkCore
    using Test
    using LinearAlgebra
    using StaticArrays
    using Random
    using Rotations
    using FiniteDiff: FiniteDiff
    using VisualGeometryToolkitCore
    using VisualGeometryToolkitCore: EssentialManifold6, retract, retract_jacobian, retract_directions,
        sampson_residual_with_jacobian, measurement_gradient_jacobian, sampson_rows!, sampson_jacobian_rows!,
        sampson_residual, constraint_with_jacobians

    rng = Xoshiro(11)
    R = RotMatrix(RotXYZ(0.2, -0.1, 0.3)); t = normalize(SVector(1.0, 0.2, -0.3))
    E = EssentialMat(Tuple(skew(t) * SMatrix{3, 3}(R)))
    n = 60
    X = [SVector(randn(rng), randn(rng), 4 + rand(rng)) for _ in 1:n]
    u₁ = [PerspectiveMap()(x) + 1.0e-3 * SVector(randn(rng), randn(rng)) for x in X]
    u₂ = [PerspectiveMap()(SMatrix{3, 3}(R) * x + t) + 1.0e-3 * SVector(randn(rng), randn(rng)) for x in X]
    sqrtw = sqrt.(rand(rng, n))
    charts = (
        (EssentialManifold6(E), SVector(0.01, -0.02, 0.005, 0.03, -0.01, 0.02)),
        (FundMatManifold7(SMatrix{3, 3}(E)), SVector(0.01, -0.02, 0.005, 0.03, -0.01, 0.02, 0.01)),
    )
    dist(ch, v, i) = only(sampson_residual(retract(ch, v), u₁[i], u₂[i]))

    @testset "anchors: EssentialManifold5/6(E) reproduce E up to scale and sign" begin
        for ch in (EssentialManifold5(E), EssentialManifold6(E))
            M = SMatrix{3, 3}(retract(ch, zero(SVector{VgtkCore.tangent_size(ch), Float64})))
            s = norm(M) / norm(SMatrix{3, 3}(E))
            @test M ≈ s * SMatrix{3, 3}(E) || M ≈ -s * SMatrix{3, 3}(E)
        end
    end

    @testset "the residual and its chart derivative (central differences)" begin
        for (ch, v) in charts
            M = retract(ch, v); A = retract_jacobian(ch, v)
            for i in (1, 7, 30)
                s, ∂s = sampson_residual_with_jacobian(M, u₁[i], u₂[i]; directions = A)
                @test s == sampson_residual(M, u₁[i], u₂[i])          # the one path's residual
                fd = FiniteDiff.finite_difference_gradient(w -> dist(ch, SVector{length(v)}(w), i), Vector(v))   # retract takes an SVector
                @test vec(∂s) ≈ fd rtol = 1.0e-5 atol = 1.0e-9
                # the directions as the models they move to (`retract_directions`): the same row
                @test sampson_residual_with_jacobian(M, u₁[i], u₂[i]; directions = A)[2] ≈
                    VgtkCore._along(sampson_residual_with_jacobian(M, u₁[i], u₂[i])[2][1, :], VgtkCore._direction_rows(retract_directions(ch, v)))' rtol = 1.0e-13
            end
        end
    end

    # The epipolar models compute (∂G_x/∂θ)ᵀG_x in the bilinear form's structure
    # (fundmat.jl `_normal_rate`); it must equal the dense product with
    # `measurement_gradient_jacobian`, which must equal finite differences of G_x.
    @testset "measurement_gradient_jacobian and the bilinear contraction" begin
        F = SMatrix{3, 3}(retract(charts[2]...))
        Fm = FundMat(F)
        for i in (1, 7, 30, 60)
            G_x = constraint_with_jacobians(Fm, u₁[i], u₂[i])[2]
            Gfd = FiniteDiff.finite_difference_jacobian(θ -> Vector(constraint_with_jacobians(FundMat(SMatrix{3, 3}(θ)), u₁[i], u₂[i])[2]), Vector(vec(F)))
            @test measurement_gradient_jacobian(Fm, u₁[i], u₂[i]) ≈ Gfd rtol = 1.0e-7 atol = 1.0e-9
            @test VgtkCore._normal_rate(Fm, G_x, u₁[i], u₂[i]) ≈
                transpose(measurement_gradient_jacobian(Fm, u₁[i], u₂[i])) * G_x rtol = 1.0e-14 atol = 1.0e-16
        end
    end

    @testset "kernels: √w-weighted rows, one per correspondence" begin
        ch, v = charts[1]
        M = retract(ch, v); dM = retract_directions(ch, v); A = retract_jacobian(ch, v)
        r = zeros(n); J = zeros(n, 6)
        sampson_rows!(r, M, u₁, u₂, sqrtw)
        sampson_jacobian_rows!(J, M, dM, u₁, u₂, sqrtw)
        J2 = zeros(n, 6); sampson_jacobian_rows!(J2, M, A, u₁, u₂, sqrtw)   # directions as the nθ × N matrix
        @test J2 ≈ J rtol = 1.0e-14
        for i in 1:n
            s, ∂s = sampson_residual_with_jacobian(M, u₁[i], u₂[i]; directions = A)
            # The kernel weights inside its own row function, the right-hand side
            # outside it (fma contraction differs). A row near zero is a
            # cancellation, so its relative error is unbounded: hence the absolute floor.
            @test r[i] ≈ sqrtw[i] * only(s) rtol = 1.0e-13 atol = 1.0e-15
            @test J[i, :] ≈ sqrtw[i] * vec(∂s) rtol = 1.0e-13 atol = 1.0e-15
        end
        # the device kernels themselves, launched on KA.CPU() (the public functions take
        # the CPU fast path there): the same per-row function, the same rows
        KA = VgtkCore.KA
        rk = zeros(n); Jk = zeros(n, 6)
        VgtkCore._row_kernel!(KA.CPU())(VgtkCore._sampson_residual_row!, rk, M, u₁, u₂, sqrtw, NamedTuple(); ndrange = n)
        VgtkCore._row_kernel!(KA.CPU())(VgtkCore._sampson_jacobian_row!, Jk, M,
            VgtkCore._direction_rows(dM), u₁, u₂, sqrtw, NamedTuple(); ndrange = n)
        KA.synchronize(KA.CPU())
        # The CPU loop vectorizes across rows; its fma contraction differs from the kernel's.
        @test rk ≈ r rtol = 1.0e-14
        @test Jk ≈ J rtol = 1.0e-14
        # Float32 storage and points: the same kernel, single precision
        r32 = zeros(Float32, n)
        sampson_rows!(r32, FundMat(SMatrix{3, 3, Float32}(M)), [Float32.(u) for u in u₁], [Float32.(u) for u in u₂], Float32.(sqrtw))
        @test r32 ≈ Float32.(r) rtol = 1.0f-3
        # no allocation in the rows or the per-correspondence Jacobian
        rows_allocs(r, M, u, w, s) = (sampson_rows!(r, M, u, w, s); @allocated sampson_rows!(r, M, u, w, s))
        jac_allocs(J, M, dM, u, w, s) = (sampson_jacobian_rows!(J, M, dM, u, w, s); @allocated sampson_jacobian_rows!(J, M, dM, u, w, s))
        @test rows_allocs(r, M, u₁, u₂, sqrtw) == 0
        @test jac_allocs(J, M, dM, u₁, u₂, sqrtw) == 0
        f(M, u, w, A) = sampson_residual_with_jacobian(M, u, w; directions = A)
        allocs(M, u, w, A) = (f(M, u, w, A); @allocated f(M, u, w, A))
        @test allocs(M, u₁[1], u₂[1], A) == 0
    end

    @testset "the one degeneracy rule: a zero row, never a floor" begin
        F = FundMat(SMatrix{3, 3, Float64}(0, 1, 0, -1, 0, 0, 0, 0, 1))      # both epipolar lines at infinity at the origin
        o = SVector(0.0, 0.0)
        dM = (SMatrix{3, 3, Float64}(I), SMatrix{3, 3, Float64}(1, 0, 0, 0, 0, 0, 0, 0, 0))
        s, ∂s = sampson_residual_with_jacobian(F, o, o; directions = SMatrix{9, 2}(transpose(VgtkCore._direction_rows(dM))))
        @test only(s) == Inf && all(iszero, ∂s)                                   # the residual keeps its Inf, the row is zero
        @test only(sampson_residual(F, o, o)) == Inf
        r = [1.0]; J = ones(1, 2)
        sampson_rows!(r, F, [o], [o], [1.0]); sampson_jacobian_rows!(J, F, dM, [o], [o], [1.0])
        @test r == [0.0] && J == zeros(1, 2)
        # NaN propagates through the residual and the row
        sn, ∂n = sampson_residual_with_jacobian(F, SVector(NaN, 0.0), o; directions = SMatrix{9, 2}(transpose(VgtkCore._direction_rows(dM))))
        @test isnan(only(sn)) && all(isnan, ∂n)
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
