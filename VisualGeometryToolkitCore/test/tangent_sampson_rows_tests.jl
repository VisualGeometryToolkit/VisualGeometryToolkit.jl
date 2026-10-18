# The TANGENT Sampson LM rows: E on unit bearings, batched.
#
# WHY THESE EXIST, and why the separation is a TYPE rather than a convention.
# The image-plane rows read `u[1], u[2]` and lift to (x, y, 1), so a 3-vector
# bearing handed to them has its third component DISCARDED and replaced by 1 —
# a different function, not a scaled one. VisualGeometryToolkitEstimation measured the two
# disagreeing by up to 446× per point after its scoring path moved to bearings
# while `_EssentialLm` did not: the optimizer descended something unrelated to
# what it scored, failed to reduce its own cost (0.006378 → 0.006416), and
# reported `:converged` while a pose went from a 3.4° seed to 26°. It would
# have shipped as "the LO converged".
#
# ONE IMPLEMENTATION is the design constraint, because that defect is what two
# routes to one quantity costs. The constraint and its gradients come from the
# scalar `constraint_with_jacobians(E, f₁, f₂; M₁, M₂)` that the scoring path
# uses, and the WHITENING DERIVATIVE from `signed_sampson_jacobian`, the same
# family-agnostic function the image-plane rows use. Only `∂G_x/∂θ` is new, and
# it is finite-differenced below rather than derived and checked against itself.
#
# TWO DIFFERENT FOCAL LENGTHS throughout (1200 and 600), and that is load
# bearing rather than incidental — see the swap canary at the bottom: with one
# focal a per-view Jacobian swap is nearly invisible.
@testitem "tangent_sampson_rows" begin
    import VisualGeometryToolkitCore as VgtkCore
    using Test
    using LinearAlgebra
    using StaticArrays
    using Random
    using Rotations
    using FiniteDiff: FiniteDiff
    using VisualGeometryToolkitCore
    using VisualGeometryToolkitCore: EssentialMat, EssentialManifold6, retract, retract_jacobian,
        CameraModel, LogicalIntrinsics, CameraCalibrationMat, Pinhole,
        backproject, project, backproject_with_jacobian, sampson_distance,
        sampson_rows!, sampson_jacobian_rows!, signed_tangent_sampson_row

    cam(f, c) = CameraModel(
        LogicalIntrinsics(CameraCalibrationMat(SMatrix{3, 3, Float64}(f, 0, 0, 0, f, 0, c[1], c[2], 1))),
        Pinhole(),
    )
    C1 = cam(1200.0, (640.0, 480.0))
    C2 = cam(600.0, (640.0, 480.0))          # DIFFERENT focal — see the swap canary
    R = RotMatrix(RotXYZ(0.05, -0.08, 0.03))
    t = normalize(SVector(1.0, 0.2, -0.15))
    E = EssentialMat(SMatrix{3, 3}(R), t)

    rng = Xoshiro(3)
    n = 12
    u₁ = [SVector(640 + 300randn(rng) / 3, 480 + 220randn(rng) / 3) for _ in 1:n]
    X₂ = [SMatrix{3, 3}(R) * (backproject(C1, u) * 3.0) + t for u in u₁]
    u₂ = [SVector(project(C2, X)[1], project(C2, X)[2]) for X in X₂]
    # noisy, so g is genuinely non-zero and the comparisons have content
    bm1 = [backproject_with_jacobian(C1, u .+ 0.4 .* SVector(randn(rng), randn(rng))) for u in u₁]
    bm2 = [backproject_with_jacobian(C2, u .+ 0.4 .* SVector(randn(rng), randn(rng))) for u in u₂]
    f₁ = [b[1] for b in bm1]; M₁ = [b[2] for b in bm1]
    f₂ = [b[1] for b in bm2]; M₂ = [b[2] for b in bm2]
    sqrtw = sqrt.(rand(rng, n)) .+ 0.5

    ch = EssentialManifold6(E)
    v₀ = SVector(0.01, -0.02, 0.005, 0.03, -0.01, 0.02)
    Eᵥ = retract(ch, v₀)                     # an EssentialMat
    dEof(c, v) = VgtkCore.retract_directions(c, v)
    dE = dEof(ch, v₀)

    @testset "the rows ARE the scalar residual — the canary against a second route" begin
        # Even with a shared core the batched and scalar paths assemble inputs
        # differently, which is exactly how the 446× divergence began.
        r = zeros(n)
        sampson_rows!(r, E, f₁, f₂, ones(n); M₁, M₂)
        for i in 1:n
            @test abs(r[i]) ≈ sampson_distance(E, f₁[i], f₂[i]; M₁ = M₁[i], M₂ = M₂[i]) rtol = 1.0e-12
        end
        # measured max relative disagreement over the set: 2.9e-14
        scal = [sampson_distance(E, f₁[i], f₂[i]; M₁ = M₁[i], M₂ = M₂[i]) for i in 1:n]
        @test maximum(abs.(abs.(r) .- scal) ./ scal) < 1.0e-12
        # the signed bearing distance: the rows' sign, the unsigned one's magnitude
        sgn = [signed_sampson_distance(E, f₁[i], f₂[i]; M₁ = M₁[i], M₂ = M₂[i]) for i in 1:n]
        @test abs.(sgn) == scal
        @test all(sign.(sgn) .== sign.(r))
    end

    @testset "√w weighting, and the per-row form agrees with the batch" begin
        r1 = zeros(n); r2 = zeros(n)
        sampson_rows!(r1, Eᵥ, f₁, f₂, ones(n); M₁, M₂)
        sampson_rows!(r2, Eᵥ, f₁, f₂, 2 .* ones(n); M₁, M₂)
        @test r2 ≈ 2 .* r1 rtol = 1.0e-12
        for i in (1, 5, 11)
            d, _ = signed_tangent_sampson_row(Eᵥ, dE, f₁[i], f₂[i], M₁[i], M₂[i])
            @test d ≈ r1[i] rtol = 1.0e-12
        end
    end

    @testset "the Jacobian rows against CENTRAL DIFFERENCES of the residual rows" begin
        # Differentiated numerically through the chart's own retraction, not
        # checked against a re-derivation of the same formula. Measured max
        # relative error 1.0e-9 at FiniteDiff's default step (6e-6 here; 1.9e-7
        # at a 1e-7 step, where rounding dominates), not a modelling gap.
        J = zeros(n, 6)
        sampson_jacobian_rows!(J, Eᵥ, dE, f₁, f₂, sqrtw; M₁, M₂)
        rows(v) = sampson_rows!(zeros(n), retract(ch, v), f₁, f₂, sqrtw; M₁, M₂)
        fd = FiniteDiff.finite_difference_jacobian(rows, v₀, Val(:central))
        @test maximum(abs.(J .- fd) ./ max.(abs.(fd), 1.0e-6)) < 1.0e-5
    end

    @testset "a bearing cannot reach the image-plane rows, nor a 2-vector these" begin
        # The same discipline as the deleted 2-vector lift: refused by TYPE, so
        # no caller's convention can put the wrong data in the wrong method.
        r = zeros(n); J = zeros(n, 6)
        @test_throws MethodError sampson_rows!(r, SMatrix{3, 3, Float64}(Eᵥ), f₁, f₂, sqrtw)
        @test_throws MethodError sampson_jacobian_rows!(J, SMatrix{3, 3, Float64}(Eᵥ), dE, f₁, f₂, sqrtw)
        @test_throws MethodError sampson_rows!(r, Eᵥ, u₁, u₂, sqrtw; M₁, M₂)
        # …and the bearing signature cannot silently fall through to the
        # image-plane one by omitting the Jacobians.
        @test_throws UndefKeywordError sampson_rows!(r, Eᵥ, f₁, f₂, sqrtw)
        @test_throws UndefKeywordError sampson_jacobian_rows!(J, Eᵥ, dE, f₁, f₂, sqrtw)
    end

    @testset "a per-view Jacobian SWAP is caught — and needs two focals to be" begin
        # This is why the fixture uses f = 1200 and f = 600 and must keep doing
        # so. Measured here: swapping M₁/M₂ moves the RESIDUAL rows by only
        # 1.8% but the JACOBIAN rows by 341% — the descent DIRECTION is what a
        # swap destroys, not the cost. That is precisely RVG's reported
        # signature: a cost that failed to improve while the optimiser reported
        # convergence.
        r0 = zeros(n); rs = zeros(n)
        sampson_rows!(r0, Eᵥ, f₁, f₂, sqrtw; M₁, M₂)
        sampson_rows!(rs, Eᵥ, f₁, f₂, sqrtw; M₁ = M₂, M₂ = M₁)
        J0 = zeros(n, 6); Js = zeros(n, 6)
        sampson_jacobian_rows!(J0, Eᵥ, dE, f₁, f₂, sqrtw; M₁, M₂)
        sampson_jacobian_rows!(Js, Eᵥ, dE, f₁, f₂, sqrtw; M₁ = M₂, M₂ = M₁)
        @test maximum(abs.(Js .- J0) ./ max.(abs.(J0), 1.0e-6)) > 0.5     # measured 3.41
        @test maximum(abs.(rs .- r0) ./ abs.(r0)) > 5.0e-3                # measured 1.8e-2

        # The control, and the reason the two focals are not decoration: with
        # EQUAL focals the same swap moves the residuals by 0.19%, an order of
        # magnitude less, and would sit inside anything one might call noise.
        Ce = cam(1200.0, (640.0, 480.0))
        u₂e = [
            (X = SMatrix{3, 3}(R) * (backproject(Ce, u) * 3.0) + t;
                SVector(project(Ce, X)[1], project(Ce, X)[2])) for u in u₁
        ]
        be1 = [backproject_with_jacobian(Ce, u) for u in u₁]
        be2 = [backproject_with_jacobian(Ce, u) for u in u₂e]
        fe₁ = [b[1] for b in be1]; Me₁ = [b[2] for b in be1]
        fe₂ = [b[1] for b in be2]; Me₂ = [b[2] for b in be2]
        re0 = zeros(n); res = zeros(n)
        sampson_rows!(re0, Eᵥ, fe₁, fe₂, sqrtw; M₁ = Me₁, M₂ = Me₂)
        sampson_rows!(res, Eᵥ, fe₁, fe₂, sqrtw; M₁ = Me₂, M₂ = Me₁)
        @test maximum(abs.(res .- re0) ./ abs.(re0)) < 5.0e-3             # measured 1.9e-3
    end

    @testset "a vanishing gradient is a zero row, not an Inf" begin
        # The image-plane rows' rule, in the tangent metric: a least-squares row
        # cannot carry the distance's ±Inf.
        Z = [zero(SMatrix{3, 2, Float64}) for _ in 1:n]
        r = ones(n); J = ones(n, 6)
        sampson_rows!(r, Eᵥ, f₁, f₂, sqrtw; M₁ = Z, M₂ = Z)
        sampson_jacobian_rows!(J, Eᵥ, dE, f₁, f₂, sqrtw; M₁ = Z, M₂ = Z)
        @test all(iszero, r)
        @test all(iszero, J)
    end
end
