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
        sampson_residual_with_jacobian, measurement_gradient_jacobian, sampson_residuals!, sampson_jacobian_rows!,
        sampson_residual, constraint_with_jacobians, Csponds, PointSet, withattrs

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
                # the one path's residual, to rounding: the constraint's StaticArrays
                # products are `muladd`s, which LLVM may contract differently once inlined
                # beside the Jacobian, and g = dᵀFs cancels (measured 1.9e-15 relative)
                @test s ≈ sampson_residual(M, u₁[i], u₂[i]) rtol = 1.0e-13
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

    @testset "kernels: √w-weighted rows over a measurement set" begin
        # The √w rows are the LM's, `weighted_whitened_*`; on a set without noise
        # columns they are the geometric rows times √w. The geometric rows
        # themselves are unweighted, as the whitened ones are.
        wres!, wjac! = VgtkCore.weighted_whitened_residuals!, VgtkCore.weighted_whitened_jacobian_rows!
        ch, v = charts[1]
        M = retract(ch, v); dM = retract_directions(ch, v); A = retract_jacobian(ch, v)
        xs = withattrs(Csponds(u₁, u₂); sqrt_weights = sqrtw)
        r = zeros(n); J = zeros(n, 6)
        wres!(r, M, xs)
        wjac!(J, M, dM, xs)
        J2 = zeros(n, 6); wjac!(J2, M, A, xs)   # directions as the nθ × N matrix
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
        VgtkCore._row_kernel!(KA.CPU())(VgtkCore._sampson_residual_row!, rk, M, (u₁, u₂), NamedTuple(), nothing, sqrtw; ndrange = n)
        VgtkCore._row_kernel!(KA.CPU())(VgtkCore._sampson_jacobian_row!, Jk, M,
            VgtkCore._direction_rows(dM), (u₁, u₂), NamedTuple(), nothing, sqrtw; ndrange = n)
        KA.synchronize(KA.CPU())
        # The CPU loop vectorizes across rows; its fma contraction differs from the kernel's.
        @test rk ≈ r rtol = 1.0e-14
        @test Jk ≈ J rtol = 1.0e-14
        # Float32 storage and points: the same kernel, single precision
        r32 = zeros(Float32, n)
        wres!(r32, FundMat(SMatrix{3, 3, Float32}(M)),
            withattrs(Csponds([Float32.(u) for u in u₁], [Float32.(u) for u in u₂]); sqrt_weights = Float32.(sqrtw)))
        @test r32 ≈ Float32.(r) rtol = 1.0f-3
        # without a weights column every √w is 1
        r1 = zeros(n); sampson_residuals!(r1, M, Csponds(u₁, u₂))
        @test r1 ≈ r ./ sqrtw rtol = 1.0e-13
        r1w = zeros(n); wres!(r1w, M, Csponds(u₁, u₂))
        @test r1w == r1
        # the geometric rows do not read sqrt_weights: one currency with the whitened rows
        rs = zeros(n); sampson_residuals!(rs, M, xs)
        @test rs == r1
        J1 = zeros(n, 6); sampson_jacobian_rows!(J1, M, dM, Csponds(u₁, u₂))
        Js = zeros(n, 6); sampson_jacobian_rows!(Js, M, dM, xs)
        @test Js == J1
        @test J ≈ sqrtw .* J1 rtol = 1.0e-13
        # no allocation per row: the one kernel launch allocates KernelAbstractions' launch
        # objects (a few hundred bytes, whatever the row count) and nothing more. A kernel
        # compiled for a widened signature (`args...` as `Vararg{Any}`) boxes every row
        # (≈150 bytes a row, 2026-11-11), which this catches as growth with n.
        rows_allocs(f!, r, M, xs) = (f!(r, M, xs); @allocated f!(r, M, xs))
        jac_allocs(f!, J, M, dM, xs) = (f!(J, M, dM, xs); @allocated f!(J, M, dM, xs))
        k = 10; xk = withattrs(Csponds(u₁[1:k], u₂[1:k]); sqrt_weights = sqrtw[1:k]); rk1 = zeros(k); Jk1 = zeros(k, 6)
        for f! in (sampson_residuals!, wres!)
            @test rows_allocs(f!, r, M, xs) == rows_allocs(f!, rk1, M, xk)
        end
        for f! in (sampson_jacobian_rows!, wjac!)
            @test jac_allocs(f!, J, M, dM, xs) == jac_allocs(f!, Jk1, M, dM, xk)
        end
        f(M, u, w, A) = sampson_residual_with_jacobian(M, u, w; directions = A)
        allocs(M, u, w, A) = (f(M, u, w, A); @allocated f(M, u, w, A))
        @test allocs(M, u₁[1], u₂[1], A) == 0
    end

    @testset "one measurement per row: a conic's rows" begin
        C = VgtkCore.Conic(SMatrix{3, 3}(0.002, 0.0003, -0.5, 0.0003, 0.003, -0.4, -0.5, -0.4, 150.0))
        xs = [SVector(210.0 + 7k, 180.0 + 11k) for k in 1:12]; w = collect(range(0.5, 1.5; length = 12))
        D = SMatrix{6, 2}(1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0)
        r1 = zeros(12); J1 = zeros(12, 2)
        set = withattrs(xs; sqrt_weights = w)
        @test set isa PointSet && set[3] == xs[3]
        VgtkCore.weighted_whitened_residuals!(r1, C, set); VgtkCore.weighted_whitened_jacobian_rows!(J1, C, D, set)
        r0 = zeros(12); J0 = zeros(12, 2)
        sampson_residuals!(r0, C, set); sampson_jacobian_rows!(J0, C, D, set)   # unweighted
        for i in eachindex(xs)
            s, ∂s = sampson_residual_with_jacobian(C, xs[i]; directions = D)
            @test r1[i] ≈ w[i] * only(s) rtol = 1.0e-14
            @test J1[i, :] ≈ w[i] * vec(∂s) rtol = 1.0e-14
            @test r0[i] ≈ only(s) rtol = 1.0e-14
            @test J0[i, :] ≈ vec(∂s) rtol = 1.0e-14
        end
    end

    @testset "the one degeneracy rule: a zero row, never a floor" begin
        F = FundMat(SMatrix{3, 3, Float64}(0, 1, 0, -1, 0, 0, 0, 0, 1))      # both epipolar lines at infinity at the origin
        o = SVector(0.0, 0.0)
        dM = (SMatrix{3, 3, Float64}(I), SMatrix{3, 3, Float64}(1, 0, 0, 0, 0, 0, 0, 0, 0))
        s, ∂s = sampson_residual_with_jacobian(F, o, o; directions = SMatrix{9, 2}(transpose(VgtkCore._direction_rows(dM))))
        @test only(s) == Inf && all(iszero, ∂s)                                   # the residual keeps its Inf, the row is zero
        @test only(sampson_residual(F, o, o)) == Inf
        r = [1.0]; J = ones(1, 2)
        sampson_residuals!(r, F, Csponds([o], [o])); sampson_jacobian_rows!(J, F, dM, Csponds([o], [o]))
        @test r == [0.0] && J == zeros(1, 2)
        # NaN propagates through the residual and the row
        sn, ∂n = sampson_residual_with_jacobian(F, SVector(NaN, 0.0), o; directions = SMatrix{9, 2}(transpose(VgtkCore._direction_rows(dM))))
        @test isnan(only(sn)) && all(isnan, ∂n)
    end
end

# A vector constraint (the homography's two rows): the exact Sampson rows —
# the Cholesky factor of G_x·G_xᵀ differentiated with the model — against
# ForwardDiff through the residuals, plain and under measurement noise (an
# exact source: the whitened rows of the transfer error); the pool matrix.
@testitem "sampson_rows: vector constraints and noise" begin
    import VisualGeometryToolkitCore as VgtkCore
    using Test
    using LinearAlgebra
    using StaticArrays
    using Random
    using ForwardDiff: ForwardDiff
    using VisualGeometryToolkitCore
    using VisualGeometryToolkitCore: HomographyManifold8, retract, retract_jacobian, sampson_residuals!, whitened_residuals!,
        sampson_jacobian_rows!, whitened_jacobian_rows!, sampson_residual_with_jacobian, sampson_residual,
        measurement_gradient_jacobian, constraint_with_jacobians, sampson_distances!, sampson_distance, Csponds, withattrs, exact,
        noise_columns, mahalanobis_distance, Uncertain, CovMat, TransferForm

    rng = Xoshiro(31)
    H = HomographyMat(SMatrix{3, 3, Float64}(1.1, 0.02, 1.0e-4, -0.03, 0.95, 2.0e-4, 12.0, -7.0, 1.0))
    n = 40
    u₁ = [SVector(400rand(rng), 300rand(rng)) for _ in 1:n]
    u₂ = [SVector{2}(Tuple(PerspectiveMap()(SMatrix{3, 3}(H) * SVector(u[1], u[2], 1.0)))) + 0.5 * randn(rng, SVector{2, Float64}) for u in u₁]
    sqrtw = sqrt.(rand(rng, n)) .+ 0.5
    ch = HomographyManifold8(H); a = SVector(0.01, -0.02, 0.005, 0.003, -0.01, 0.02, 0.001, -0.004)

    @testset "the homography's measurement_gradient_jacobian (ForwardDiff of G_x)" begin
        for i in (1, 17, 40)
            G(h) = vec(constraint_with_jacobians(HomographyMat(Tuple(h)), u₁[i], u₂[i])[2])
            @test measurement_gradient_jacobian(H, u₁[i], u₂[i]) ≈ ForwardDiff.jacobian(G, SVector{9}(Tuple(SMatrix{3, 3}(H)))) rtol = 1.0e-14
        end
    end

    @testset "closed forms pinned to their definitions" begin
        # the homography's contraction of ∂G_x/∂θ with Σ·G_xᵀ, against the dense product
        for i in (2, 19), ΣGt in (nothing, SMatrix{4, 4}(Diagonal(SVector(2.0, 0.5, 1.0, 3.0))))
            Gx = constraint_with_jacobians(H, u₁[i], u₂[i])[2]
            St = ΣGt === nothing ? SMatrix{4, 2}(transpose(Gx)) : ΣGt * transpose(Gx)
            dG = measurement_gradient_jacobian(H, u₁[i], u₂[i])
            Kref = [sum(dG[r % 2 + 1 + 2(k - 1), t] * St[k, r ÷ 2 + 1] for k in 1:4) for r in 0:3, t in 1:9]
            @test VgtkCore._noise_rate(H, St, u₁[i], u₂[i]) ≈ Kref rtol = 1.0e-14
        end
        # the Cholesky factor's derivative term: the dg = 1, 2 closed forms against Φ
        for dg in (1, 2)
            L⁻¹ = dg == 1 ? SMatrix{1, 1}(0.7) : SMatrix{2, 2}(0.7, -0.3, 0.0, 1.9)
            K = SMatrix{dg * dg, 9}(randn(rng, dg * dg, 9)); r = SVector{dg}(randn(rng, dg))
            Φr = hcat([(S = SMatrix{dg, dg}(K[:, t]); VgtkCore._lower_half(L⁻¹ * (S + S') * L⁻¹') * r) for t in 1:9]...)
            @test VgtkCore._factor_rate(L⁻¹, K, r) ≈ Φr rtol = 1.0e-13
        end
        # the bearing transfer form's ∂vec(G_x)/∂vec(H), against ForwardDiff
        f = [normalize(SVector(u[1] - 200, u[2] - 150, 300.0)) for u in u₁]
        Mj = [SMatrix{3, 2}(1.0e-3 .* randn(rng, 3, 2)) for _ in 1:n]
        R = SMatrix{3, 3}(1.0, 0.0, 0.0, 0.0, cos(0.2), sin(0.2), 0.0, -sin(0.2), cos(0.2))
        for i in (1, 9)
            Gb(h) = vec(constraint_with_jacobians(TransferForm(HomographyMat(Tuple(h))), f[i], R * f[i]; M₁ = Mj[i], M₂ = Mj[i])[2])
            @test measurement_gradient_jacobian(TransferForm(HomographyMat(Tuple(R))), f[i], R * f[i]; M₁ = Mj[i], M₂ = Mj[i]) ≈
                ForwardDiff.jacobian(Gb, SVector{9}(Tuple(R))) rtol = 1.0e-12 atol = 1.0e-16
        end
    end

    rows(xs, f!) = A -> (r = zeros(eltype(A), 2n); f!(r, retract(ch, A), xs); r)
    # the cases over the correspondences `I` (all of them, and a prefix for the allocation check)
    cases(I) = (
            ("plain, geometric", withattrs(Csponds(u₁[I], u₂[I]); sqrt_weights = sqrtw[I]), sampson_residuals!, sampson_jacobian_rows!),
            ("exact source, whitened", withattrs(Csponds(u₁[I], u₂[I]); noise_columns(exact.(u₁[I]), u₂[I])..., sqrt_weights = sqrtw[I]),
                whitened_residuals!, whitened_jacobian_rows!),
            ("exact source, whitened, weighted", withattrs(Csponds(u₁[I], u₂[I]); noise_columns(exact.(u₁[I]), u₂[I])..., sqrt_weights = sqrtw[I]),
                VgtkCore.weighted_whitened_residuals!, VgtkCore.weighted_whitened_jacobian_rows!),
            ("uncertain points, whitened",
                withattrs(Csponds(u₁[I], u₂[I]); noise_columns([Uncertain(u, CovMat(SMatrix{2, 2}(2.0, 0.3, 0.3, 0.5))) for u in u₁[I]], u₂[I])..., sqrt_weights = sqrtw[I]),
                whitened_residuals!, whitened_jacobian_rows!),
            ("uncertain points, whitened, weighted",
                withattrs(Csponds(u₁[I], u₂[I]); noise_columns([Uncertain(u, CovMat(SMatrix{2, 2}(2.0, 0.3, 0.3, 0.5))) for u in u₁[I]], u₂[I])..., sqrt_weights = sqrtw[I]),
                VgtkCore.weighted_whitened_residuals!, VgtkCore.weighted_whitened_jacobian_rows!))
    for ((name, xs, f!, J!), (_, xk, _, _)) in zip(cases(1:n), cases(1:5))
        @testset "$name: exact rows (ForwardDiff through the residuals)" begin
            J = zeros(2n, 8); J!(J, retract(ch, a), retract_jacobian(ch, a), xs)
            @test J ≈ ForwardDiff.jacobian(rows(xs, f!), a) rtol = 1.0e-10 atol = 1.0e-12
            # nothing allocated per row: the kernel launch's own objects only (the same at 5 rows as at n)
            allocs(J, M, D, xs) = (J!(J, M, D, xs); @allocated J!(J, M, D, xs))
            M, D = retract(ch, a), retract_jacobian(ch, a)
            @test allocs(J, M, D, xs) == allocs(zeros(10, 8), M, D, xk)
        end
    end
    @testset "whitened rows are unweighted; the weighted rows are √w times them" begin
        xs = withattrs(Csponds(u₁, u₂); noise_columns([Uncertain(u, CovMat(SMatrix{2, 2}(2.0, 0.3, 0.3, 0.5))) for u in u₁], u₂)..., sqrt_weights = sqrtw)
        x0 = withattrs(Csponds(u₁, u₂); noise_columns([Uncertain(u, CovMat(SMatrix{2, 2}(2.0, 0.3, 0.3, 0.5))) for u in u₁], u₂)...)
        M = retract(ch, a)
        r, r0, rw = zeros(2n), zeros(2n), zeros(2n)
        whitened_residuals!(r, M, xs); whitened_residuals!(r0, M, x0); VgtkCore.weighted_whitened_residuals!(rw, M, xs)
        @test r == r0                                   # the sqrt_weights column is not read
        @test rw == repeat(sqrtw; inner = 2) .* r      # bit for bit: √w·s, one product
        J, J0 = zeros(2n, 8), zeros(2n, 8)
        whitened_jacobian_rows!(J, M, retract_jacobian(ch, a), xs); whitened_jacobian_rows!(J0, M, retract_jacobian(ch, a), x0)
        @test J == J0
    end
    @testset "the 1 × 1 and 2 × 2 whitening factors are StaticArrays' Cholesky (IEEE sqrt)" begin
        for _ in 1:200
            A = randn(rng, SMatrix{2, 2, Float64}); P = A * A' + 1.0e-3I
            C, _, ok = VgtkCore._whitening_cholesky(P)
            @test ok && SMatrix{2, 2}(C.L) == SMatrix{2, 2}(cholesky(Symmetric(P)).L)
            p1 = SMatrix{1, 1}(abs(randn(rng)) + 0.1)
            @test SMatrix{1, 1}(VgtkCore._whitening_cholesky(p1)[1].L) == SMatrix{1, 1}(cholesky(Symmetric(p1)).L)
        end
        @test !VgtkCore._whitening_cholesky(SMatrix{2, 2}(1.0, 2.0, 2.0, 1.0))[3]      # indefinite: refused, no throw
        @test !VgtkCore._whitening_cholesky(SMatrix{1, 1}(-1.0))[3]
    end

    @testset "sampson_distances! reads the constraint keywords (E on bearings)" begin
        E = EssentialMat(SMatrix{3, 3}(VgtkCore.skew(normalize(SVector(1.0, 0.2, -0.1)))))
        f₁ = [normalize(randn(rng, SVector{3, Float64}) + SVector(0, 0, 3.0)) for _ in 1:20]
        f₂ = [normalize(randn(rng, SVector{3, Float64}) + SVector(0, 0, 3.0)) for _ in 1:20]
        M₁ = [SMatrix{3, 2}(randn(rng, 3, 2)) for _ in 1:20]; M₂ = [SMatrix{3, 2}(randn(rng, 3, 2)) for _ in 1:20]
        R = zeros(20, 1); sampson_distances!(R, [E], withattrs(Csponds(f₁, f₂); M₁, M₂))
        @test R[:, 1] ≈ [sampson_distance(E, f₁[i], f₂[i]; M₁ = M₁[i], M₂ = M₂[i]) for i in 1:20] rtol = 4eps()   # inlined: muladd fusion may differ by an ulp
        @test_throws ArgumentError sampson_distances!(R, [E], Csponds(f₁, f₂))
    end

    @testset "the residual rows are the one-element verbs" begin
        xs = Csponds(u₁, u₂); r = zeros(2n); sampson_residuals!(r, H, xs)
        @test r ≈ reduce(vcat, [sampson_residual(H, u₁[i], u₂[i]) for i in 1:n]) rtol = 1.0e-14
        xe = withattrs(xs; noise_columns(exact.(u₁), u₂)...); whitened_residuals!(r, H, xe)
        @test [norm(r[2i-1:2i]) for i in 1:n] ≈ [mahalanobis_distance(H, exact(u₁[i]), u₂[i]) for i in 1:n] rtol = 1.0e-13
        # the exact source: the transfer error in target pixels
        tr = [norm(PerspectiveMap()(SMatrix{3, 3}(H) * SVector(u[1], u[2], 1.0)) - v) for (u, v) in zip(u₁, u₂)]
        @test [norm(r[2i-1:2i]) for i in 1:n] ≈ tr rtol = 1.0e-10
        s, ∂s = sampson_residual_with_jacobian(H, u₁[3], u₂[3]; directions = retract_jacobian(ch, zero(a)))
        @test s ≈ sampson_residual(H, u₁[3], u₂[3]) rtol = 1.0e-14
        @test size(∂s) == (2, 8)
    end
    @testset "sampson_distances!: the pool matrix, models by elements" begin
        xs = Csponds(u₁, u₂); F = FundMat(SMatrix{3, 3, Float64}(randn(rng, 3, 3)))
        R = sampson_distances!(Matrix{Float32}(undef, n, 2), [H, F], xs)
        @test R[:, 1] == Float32[sampson_distance(H, u₁[i], u₂[i]) for i in 1:n]
        @test R[:, 2] == Float32[sampson_distance(F, u₁[i], u₂[i]) for i in 1:n]
        @test_throws DimensionMismatch sampson_distances!(Matrix{Float32}(undef, n, 1), [H, F], xs)
        @test sum(abs2, (r = zeros(2n); sampson_residuals!(r, H, xs); r)) ≈ sum(sampson_distance(H, u₁[i], u₂[i])^2 for i in 1:n) rtol = 1.0e-13   # the RSS composes
    end
    @testset "a missing keyword column is refused" begin
        f = [normalize(SVector(u[1], u[2], 300.0)) for u in u₁]
        E = EssentialMat(SMatrix{3, 3}(skew(SVector(1.0, 0.0, 0.0))))
        @test_throws ArgumentError sampson_residuals!(zeros(n), E, Csponds(f, f))
    end
end
