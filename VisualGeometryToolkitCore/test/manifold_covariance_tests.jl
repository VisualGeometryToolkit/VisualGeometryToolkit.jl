# =============================================================================
# Implicit-function hypothesis covariance + MC calibration harness
#
# Layers, in dependency order:
#   1. local_coordinates — the inverse retraction (harness primitive);
#   2. implicit_covariance — Jacobian pins against finite differences
#      of the LOCAL CONSTRAINED SOLUTION: for H through the unique 4-point
#      solution, for F / E through the chart-Newton tracked solution
#      (`track_manifold_solution` — the same continuation the MC protocol uses);
#   3. harness self-tests — a known-LINEAR problem must be EXACTLY calibrated,
#      and an intentionally wrong Σ must be caught (both directions);
#   4. MC calibration smoke for H / F / E at realistic noise (the full
#      ladders, output-domain containment cells, and near-degenerate regimes
#      run in VisualGeometryToolkitEstimation eval/validation/mc_calibration.jl).
# =============================================================================

@testitem "manifold_covariance" begin
    import VisualGeometryToolkitCore as VgtkCore
    using Test
    using LinearAlgebra
    using Random
    using StaticArrays
    using Rotations
    using FiniteDiff: FiniteDiff
    using VisualGeometryToolkitCore
    using VisualGeometryToolkitCore: local_coordinates, retract, tangent_size   # public, not exported
    using VisualGeometryToolkitCore: model
    using VisualGeometryToolkitCore: homography_4pt   # kernel: public, not exported

    # the MC calibration harness: public in Synthetic (formerly a test-only file)
    const CHar = VgtkCore.Synthetic

    @testset "local_coordinates — inverse retraction" begin
        rng = Random.MersenneTwister(41)
        Rr = RotMatrix(Rotations.AngleAxis(0.6, normalize(randn(rng, 3))...))
        tr_ = Vec3{Float64}(normalize(randn(rng, 3)))
        S0 = svd(randn(rng, 3, 3))
        manifolds = [
            EssentialManifold5(Rr, tr_),
            FundMatManifold7(SMatrix{3, 3, Float64, 9}(S0.U), SMatrix{3, 3, Float64, 9}(S0.V), 0.6),
            HomographyManifold8(
                SMatrix{3, 3, Float64, 9}(RotMatrix(Rotations.AngleAxis(0.3, 0, 0, 1))) +
                    SMatrix{3, 3, Float64, 9}(0.1randn(rng, 3, 3))
            ),
        ]
        for ch in manifolds
            D = tangent_size(ch)
            v0 = 5.0e-2 * SVector{D}(randn(rng, D))
            M = SMatrix{3, 3, Float64, 9}(retract(ch, v0))
            # exact recovery, and invariance to the target's projective scale/sign
            @test local_coordinates(ch, M) ≈ v0 atol = 1.0e-9
            @test local_coordinates(ch, -2.5 * M) ≈ v0 atol = 1.0e-9
            # anchor is the zero of the manifold log
            @test norm(local_coordinates(ch, SMatrix{3, 3, Float64, 9}(model(ch)))) < 1.0e-10
        end
    end

    # Central-difference sensitivity of the local constrained solution in the
    # chart anchored at the base solution: perturb one measurement coordinate,
    # re-obtain the solution single-valuedly, read its chart coordinates.
    _flat_pairs(cs) = collect(
        Iterators.flatten(
            ((c.first[1], c.first[2], c.second[1], c.second[2]) for c in cs)
        )
    )

    _pairs_of(x, k) = [
        Point2{Float64}(x[4i - 3], x[4i - 2]) => Point2{Float64}(x[4i - 1], x[4i])
            for i in 1:k
    ]

    @testset "implicit provider ≡ FD through the unique 4-point solution (H)" begin
        rng = Random.MersenneTwister(7)
        for _ in 1:5
            H, cs = CHar.sample_homography_scene(rng)
            x0 = _flat_pairs(cs)
            Ĥ0 = homography_4pt((c.first for c in cs)..., (c.second for c in cs)...)
            ch0 = minimal_manifold(Ĥ0)
            function slog(x)
                c = _pairs_of(x, 4)
                Ĥ = homography_4pt((p.first for p in c)..., (p.second for p in c)...)
                return Vector(local_coordinates(ch0, SMatrix{3, 3, Float64, 9}(Ĥ)))
            end
            Jfd = FiniteDiff.finite_difference_jacobian(slog, x0, Val(:central))
            Σ = implicit_covariance(ch0, cs)
            @test Σ ≈ Jfd * Jfd' rtol = 1.0e-5
            # anisotropic Σx path: J·blkdiag(Σx)·Jᵀ (independent per-endpoint
            # blocks — both images noisy, covariances differing across images)
            shapes = CHar.make_point_covs(rng, 4; aniso = true)
            Σa = implicit_covariance(ch0, cs; Σx = shapes)
            blk = zeros(16, 16)
            for i in 1:4
                blk[(4i - 3):4i, (4i - 3):4i] = Matrix(shapes[i])
            end
            @test Σa ≈ Jfd * blk * Jfd' rtol = 1.0e-5
        end
    end

    @testset "implicit provider ≡ FD of the tracked local solution (F, mid-gauge)" begin
        rng = Random.MersenneTwister(11)
        # Pin in the mid-gauge calibration frame (fundmat_norm_xform): in raw
        # pixel coordinates the Bartoli–Sturm chart is near-gauge (σ₂/σ₁ ~ 1e-4)
        # and central differences of the near-gauge directions are meaningless at
        # any usable step size (a property of the chart frame, not of any
        # algorithm — see the harness F_NORM_SCALE docstring).
        Tn = CHar.fundmat_norm_xform()
        Tit = inv(Tn)'
        nok = 0
        for _ in 1:6
            _, _, F_px, _, cs_px, _ = CHar.sample_twoview_scene(rng, 7)
            F = FundMat{Float64}(Tuple(Tit * SMatrix{3, 3, Float64, 9}(F_px) * inv(Tn)))
            cs = [
                Point2{Float64}(VgtkCore.to_euclidean(Tn * VgtkCore.to_affine(c.first))) =>
                    Point2{Float64}(VgtkCore.to_euclidean(Tn * VgtkCore.to_affine(c.second)))
                    for c in cs_px
            ]
            ch0 = minimal_manifold(F)
            # near-critical configurations (astronomical sensitivity) are exact
            # for the implicit J but outside central-difference reach — skip them
            # here (the MC ladders characterize them statistically instead)
            Σpre = implicit_covariance(ch0, cs)
            (Σpre === nothing || maximum(abs, Σpre) > 1.0e5) && continue
            x0 = _flat_pairs(cs)
            function slog(x)
                v = CHar.track_manifold_solution(ch0, cs, _pairs_of(x, 7))
                v === nothing && error("tracking diverged inside the FD pin")
                return Vector(v)
            end
            # truncation-bound near criticality: FiniteDiff's relative step on these
            # ~10-unit coordinates (6e-5) misses the worst scene by 6.5e-4, a 1e-7
            # relative step by 1.8e-7
            Jfd = FiniteDiff.finite_difference_jacobian(slog, x0, Val(:central); relstep = 1.0e-7, absstep = 1.0e-7)
            @test Σpre ≈ Jfd * Jfd' rtol = 1.0e-4
            nok += 1
        end
        @test nok ≥ 3   # FD pin must actually have run
    end

    @testset "implicit provider ≡ FD of the tracked local solution (E)" begin
        rng = Random.MersenneTwister(13)
        nok = 0
        for _ in 1:6
            _, _, _, E, _, cs_nm = CHar.sample_twoview_scene(rng, 5)
            ch0 = minimal_manifold(E)
            Σpre = implicit_covariance(ch0, cs_nm)
            (Σpre === nothing || maximum(abs, Σpre) > 1.0e6) && continue
            x0 = _flat_pairs(cs_nm)
            function slog(x)
                v = CHar.track_manifold_solution(ch0, cs_nm, _pairs_of(x, 5))
                v === nothing && error("tracking diverged inside the FD pin")
                return Vector(v)
            end
            Jfd = FiniteDiff.finite_difference_jacobian(slog, x0, Val(:central))
            @test Σpre ≈ Jfd * Jfd' rtol = 1.0e-3
            nok += 1
        end
        @test nok ≥ 3
    end

    # --- Harness self-tests -------------------------------------------------------

    # Exactly linear 2-point line fit: θ = (a, b) from y = a·x + b at x = (0, 1);
    # only y is noisy → the propagated Σ is EXACT at every noise level.
    function _linear_trial(σ::Float64; scale::Float64 = 1.0)
        J = [-1.0 1.0; 1.0 0.0]                # θ̂ − θ = J·(δy₁, δy₂)
        Σ = scale * σ^2 .* (J * J')
        return function (rng)
            δ = σ * randn(rng, 2)
            return J * δ, Σ
        end
    end

    @testset "implicit_covariance: the square solve, the sandwich, blocks; pose correspondences" begin
        using VisualGeometryToolkitCore: implicit_covariance, constraint_jacobians, minimal_manifold,
            retract_jacobian, LeftRigidManifold6, constraint_with_jacobians
        rng = Random.Xoshiro(1)
        rel(a, b) = maximum(abs, a - b) / maximum(abs, b)
        # the blocks of stacked rows: the SANDWICH form, whatever the row count
        blocks(A, B, dg, nx) = i -> (A[(dg * (i - 1) + 1):(dg * i), :], SMatrix{dg, nx}(B[(dg * (i - 1) + 1):(dg * i), (nx * (i - 1) + 1):(nx * i)]))
        # a minimal sample: the sandwich IS the square implicit-function solve
        H, csH = CHar.sample_homography_scene(rng)
        chH = minimal_manifold(CHar._unit(H)); shH = CHar.make_point_covs(rng, 4; aniso = true)
        AH, BH = constraint_jacobians(chH, zero(SVector{8, Float64}), csH)
        ΣH = implicit_covariance(chH, csH; Σx = shH)
        JH = -(AH \ BH)                                                        # the implicit-function solve by hand
        @test rel(sum(JH[:, (4i - 3):(4i)] * shH[i] * JH[:, (4i - 3):(4i)]' for i in 1:4), ΣH) < 1.0e-12
        @test rel(implicit_covariance(i -> (blocks(AH, BH, 2, 4)(i)..., shH[i]), 1:4), ΣH) < 1.0e-10
        _, _, _, E, _, csE = CHar.sample_twoview_scene(rng, 5)
        chE = minimal_manifold(E); shE = CHar.make_point_covs(rng, 5; aniso = true)
        AE, BE = constraint_jacobians(chE, zero(SVector{5, Float64}), csE)
        @test rel(implicit_covariance(i -> (blocks(AE, BE, 1, 4)(i)..., shE[i]), 1:5), implicit_covariance(chE, csE; Σx = shE)) < 1.0e-8
        # overdetermined, isotropic: (AᵀW₀A)⁻¹ with W₀ = diag(1/‖Bᵢ‖²) for the epipolar rows
        _, _, _, E2, _, cs12 = CHar.sample_twoview_scene(rng, 12)
        ch2 = minimal_manifold(E2)
        A, B = constraint_jacobians(ch2, zero(SVector{5, Float64}), cs12)
        Σhand = inv(Symmetric(A' * Diagonal([1 / sum(abs2, B[i, :]) for i in 1:12]) * A))
        @test rel(implicit_covariance(ch2, cs12), Σhand) < 1.0e-10
        @test implicit_covariance(blocks(A, B, 1, 4), 1:12) ≈ implicit_covariance(ch2, cs12) rtol = 1.0e-12   # chart form = stacked rows
        @test_throws ArgumentError implicit_covariance(ch2, cs12[1:4])       # fewer constraints than dof
        @test implicit_covariance(zeros(2, 2), ones(2, 4)) === nothing   # singular square A (the chart's minimal solve)
        # the dense square form: A⁻¹·B·Σx·Bᵀ·A⁻ᵀ with one Σx shape per column block of B
        As = [2.0 0.5; -0.3 1.5]; Bs = [1.0 0.2 -0.4 0.3; 0.1 -0.7 0.6 1.1]
        Σs = [SMatrix{2, 2}(1.0, 0.2, 0.2, 0.5), SMatrix{2, 2}(0.3, 0.0, 0.0, 2.0)]
        Σfull = [Σs[1] zeros(2, 2); zeros(2, 2) Σs[2]]
        @test implicit_covariance(As, Bs; Σx = Σs) ≈ (As \ Bs) * Σfull * (As \ Bs)' rtol = 1.0e-12
        @test implicit_covariance(As, Bs) ≈ (As \ Bs) * (As \ Bs)' rtol = 1.0e-12
        @test_throws DimensionMismatch implicit_covariance(ones(2, 3), ones(2, 4))
        # the block form from a function, never the block-diagonal B — static rows (an
        # SVector is one row) against the stacked rows
        T2 = retract_jacobian(ch2)
        eblock(i) = (cd = VgtkCore.constraint_with_jacobians(model(ch2), cs12[i].first, cs12[i].second);
            (transpose(T2) * cd[3], cd[2]))
        @test implicit_covariance(eblock, 1:12) ≈ implicit_covariance(blocks(A, B, 1, 4), 1:12) rtol = 1.0e-12
        sh12 = CHar.make_point_covs(rng, 12; aniso = true)
        # static rows vs dense row slices round differently, and the normal matrix
        # here has cond ≈ 2.4e5: measured 1.4e-12 (norm-relative), so the rtol is
        # the neighbouring comparisons' 1e-10, not 1e-12
        @test implicit_covariance(i -> (eblock(i)..., sh12[i]), 1:12) ≈ implicit_covariance(i -> (blocks(A, B, 1, 4)(i)..., sh12[i]), 1:12) rtol = 1.0e-10
        # noise_weighted: the (BᵢΣxᵢBᵢᵀ)⁻¹-weighted estimator, (AᵀW₀A)⁻¹ — Σxᵢ folded into Bᵢ
        foldblock(i) = (eblock(i)[1], SMatrix{1, 4}(Tuple(eblock(i)[2])) * cholesky(Symmetric(SMatrix{4, 4}(sh12[i]))).L)
        Σnw = implicit_covariance(i -> (eblock(i)..., SMatrix{4, 4}(sh12[i])), 1:12; noise_weighted = true)
        @test Σnw == implicit_covariance(foldblock, 1:12)
        Wn = Diagonal([(b = B[i, (4i - 3):(4i)]; 1 / (b' * sh12[i] * b)) for i in 1:12])
        @test rel(Σnw, inv(Symmetric(A' * Wn * A))) < 1.0e-8
        @test implicit_covariance(eblock, 1:12; noise_weighted = true) == implicit_covariance(eblock, 1:12)   # no Σx: the same
        @test implicit_covariance(eblock, 1:0) === nothing                    # no blocks
        # a degenerate block (zero measurement gradient) drops out, as from the estimator
        withzero(i) = i == 13 ? (SVector{5}(ones(5)), zero(SVector{4, Float64})) : eblock(i)
        @test implicit_covariance(withzero, 1:13) ≈ implicit_covariance(eblock, 1:12) rtol = 1.0e-14
        # pose correspondences X => u (calibrated) in the generic machinery: the ambient
        # gradient through the chart differential is the left-perturbation G_θ
        R = RotMatrix(RotXYZ(0.3, -0.2, 0.5)); t = SVector(0.2, -0.1, 5.0)
        pose = EuclideanMap(R, t); chP = LeftRigidManifold6(pose)
        Xs = [SVector(0.3, -0.4, 1.0), SVector(-0.5, 0.2, 0.5), SVector(0.1, 0.6, -0.3)]
        csP = [X => (Xc = SMatrix{3, 3}(R) * X + t; Point2(PerspectiveMap()(Xc))) for X in Xs]
        AP, BP = constraint_jacobians(chP, zero(SVector{6, Float64}), csP)
        _, _, Gθ = constraint_with_jacobians(pose, Xs[2], SVector(csP[2].second[1], csP[2].second[2], 1.0))
        @test AP[3:4, :] ≈ Gθ atol = 1.0e-15
        @test BP == kron(Matrix(1.0I, 3, 3), -Matrix(1.0I, 2, 2))            # the image point is the measurement
        shP = [SMatrix{2, 2}(1.0, 0.2, 0.2, 0.7) for _ in 1:3]
        @test rel(implicit_covariance(i -> (blocks(AP, BP, 2, 2)(i)..., shP[i]), 1:3), implicit_covariance(chP, csP; Σx = shP)) < 1.0e-10
    end

    @testset "harness self-test — exact calibration on a linear problem" begin
        rng = Random.MersenneTwister(3)
        for σ in (0.1, 5.0, 50.0)              # exactness is noise-level independent
            r = CHar.run_calibration(_linear_trial(σ), rng, 4000, 2)
            @test CHar.is_calibrated(r)
            @test abs(r.nees_mean - 2) < 4 * r.nees_se
        end
    end

    @testset "harness self-test — wrong Σ is caught" begin
        rng = Random.MersenneTwister(4)
        over = CHar.run_calibration(_linear_trial(1.0; scale = 2.0), rng, 4000, 2)
        under = CHar.run_calibration(_linear_trial(1.0; scale = 0.5), rng, 4000, 2)
        @test !CHar.is_calibrated(over)             # overconfident-by-2 Σ⁻¹ halves the NEES
        @test !CHar.is_calibrated(under)
        @test over.nees_mean < 1.5             # ≈ dof/2
        @test under.nees_mean > 3.0            # ≈ 2·dof
    end

    @testset "MC calibration smoke — chart-domain trials at realistic noise" begin
        rng = Random.MersenneTwister(5)
        # H: the unique minimal solution vs the implicit covariance at θ_gt —
        # calibrated, isotropic and anisotropic per-endpoint covariances.
        rH = CHar.run_calibration(CHar.homography_trial(σ = 0.5), rng, 1200, 8)
        @test CHar.is_calibrated(rH)
        rHa = CHar.run_calibration(CHar.homography_trial(σ = 0.5, aniso = true), rng, 1200, 8)
        @test CHar.is_calibrated(rHa)
        # E / F: the tracked local solution vs the implicit covariance at θ_gt.
        # Full-dof chart NEES is HEAVY-TAILED at realistic noise even with exact
        # single-branch tracking — constraint-variety curvature leaking into the
        # stiff predicted directions, a property of the minimal PROBLEMS (bulk
        # median-ratio stays O(1); the ladders in eval/validation quantify it,
        # and the OUTPUT-domain containment is near-nominal — the headline).
        # Pin the breakdown so a silent change that flips it fires a test.
        # F is measured in the ambient tangent (error = :ambient): the Bartoli–Sturm
        # chart's own nonlinearity along its weak directions puts a tail on the CHART
        # coordinates that is not the covariance's (measured 2026-09-22, this seed:
        # chart NEES 12299 / 7 with median ratio 6.0; ambient 15.9 / 7, median 1.07).
        # The minimal problem's own curvature tail remains — pinned, bulk calibrated.
        rF = CHar.run_calibration(CHar.fundmat_trial(σ = 0.5, error = :ambient), rng, 1200, 7; cond_cap = 1.0e8)
        @test !CHar.is_calibrated(rF)
        @test 0.8 <= rF.nees_median_ratio <= 1.25          # bulk calibrated, tails heavy
        rE = CHar.run_calibration(CHar.essential_trial(σ_px = 0.5), rng, 1200, 5; cond_cap = 1.0e8)
        @test !CHar.is_calibrated(rE)
        @test 0.5 <= rE.nees_median_ratio <= 10.0
    end

    @testset "output-domain smoke — containment at realistic noise (Criminisi 4.1)" begin
        rng = Random.MersenneTwister(6)
        rH = CHar.run_containment(CHar.h_transfer_trial(σ = 0.5), rng, 2000)
        @test CHar.containment_nominal(rH, 1)
        rF = CHar.run_containment(CHar.epipolar_band_trial(:F; σ_px = 0.5), rng, 2000)
        @test CHar.containment_nominal(rF, 1)          # pointwise χ²₁
        @test rF.rates[2] > 0.9                        # whole-line χ²₂ ≥ nominal-ish
        rE = CHar.run_containment(CHar.epipolar_band_trial(:E; σ_px = 0.5), rng, 2000)
        @test CHar.containment_nominal(rE, 1)
    end
end

# The overdetermined (whitened Gauss–Newton vs `implicit_covariance`'s sandwich) and the
# absolute-pose cells of the Synthetic harness — a separate item so it runs in
# parallel with the minimal cells above.
@testitem "manifold_covariance: overdetermined and pose cells" begin
    import VisualGeometryToolkitCore as VgtkCore
    using Test
    using LinearAlgebra
    using Random
    using StableRNGs
    using StaticArrays
    const CHar = VgtkCore.Synthetic

    @testset "overdetermined two-view: the Sampson GN estimate vs its sandwich" begin
        @test CHar.is_calibrated(CHar.run_calibration(CHar.homography_trial(σ = 0.5, npts = 30), MersenneTwister(1), 400, 8))
        @test CHar.is_calibrated(CHar.run_calibration(CHar.essential_trial(σ_px = 0.5, npts = 30), MersenneTwister(2), 400, 5; cond_cap = 1.0e8))
        # F at n = 30, measured in the ambient tangent: calibrated (2026-09-22: NEES
        # 6.66 / 7 at σ = 0.1 px, 7.02 / 7 at 0.5 px). In the chart's own coordinates
        # the same estimates read 10.5 and 150 — the Bartoli–Sturm chart's
        # nonlinearity along its weak directions, not the covariance's error.
        for (σ, seed) in ((0.1, 3), (0.5, 31))
            rF = CHar.run_calibration(CHar.fundmat_trial(σ = σ, npts = 30, error = :ambient), MersenneTwister(seed), 400, 7; cond_cap = 1.0e8)
            @test CHar.is_calibrated(rF)
        end
    end

    @testset "absolute pose: PnP calibrated at every field of view; minimal P3P heavy-tailed" begin
        for (f, seed) in ((600.0, 8), (2000.0, 9), (150.0, 10))       # standard, narrow (14°), wide (118°)
            r = CHar.run_calibration(CHar.pnp_trial(σ_px = 0.5, npts = 20, f = f), MersenneTwister(seed), 400, 6)
            @test CHar.is_calibrated(r)
            @test r.nfail == 0
        end
        # P3P: the tracked minimal solution against the implicit covariance. The bulk
        # is calibrated (median ratio ≈ 0.9–1.1, coverage near nominal); the mean is
        # not reliably — near-degenerate P3P configurations give heavy tails (measured
        # 2026-09-22: NEES 10.8 / 8.2 / 26 over 6 dof at f = 600 / 2000 / 150). Whether
        # 400 draws contain such a configuration depends on the draws, so the mean is
        # not asserted either way. StableRNG, not MersenneTwister: Julia 1.13 changed
        # MersenneTwister's stream, and these bounds are properties of the draws.
        for (f, seed) in ((600.0, 5), (2000.0, 6), (150.0, 7))
            r = CHar.run_calibration(CHar.p3p_trial(σ_px = 0.5, f = f), StableRNG(seed), 400, 6)
            @test 0.8 <= r.nees_median_ratio <= 1.25
            @test 0.85 <= r.coverage[2] <= 0.95
            @test r.nfail <= 0.05 * 400
        end
    end

    @testset "pose output domain: held-out reprojection containment (Criminisi 4.1)" begin
        @test CHar.containment_nominal(CHar.run_containment(CHar.reprojection_trial(σ_px = 0.5), MersenneTwister(11), 1000), 1)
        @test CHar.containment_nominal(CHar.run_containment(CHar.reprojection_trial(σ_px = 0.5, npts = 20), MersenneTwister(12), 1000), 1)
    end

    @testset ":forward minimal E — characterisation, not a gate" begin
        # Forward motion puts the epipole in the image; the minimal 5-point problem
        # there is badly conditioned, and its tracked solution diverges for a large
        # fraction of stratified samples even at 0.5 px. Measured 2026-09-22 (400
        # trials, this seed): 114 divergences, NEES 194 over 5 dof, median ratio 1.22
        # — the bulk calibrated, the tail not. Pinned so a change to the regime or the
        # tracker that moves it is seen.
        r = CHar.run_calibration(CHar.essential_trial(σ_px = 0.5, regime = :forward), MersenneTwister(4), 400, 5; cond_cap = 1.0e8)
        @test 0.15 * 400 <= r.nfail <= 0.45 * 400
        @test r.nees_mean > 20.0
        @test 0.8 <= r.nees_median_ratio <= 2.0
    end

    @testset ":forward two-view regime: epipoles inside, no point without parallax" begin
        rng = MersenneTwister(13)
        f, c = 600.0, 250.0
        for _ in 1:20
            R, t, _, _, cs_px, cs_nm = CHar.sample_twoview_scene(rng, 12; regime = :forward, f = f, c = c)
            @test abs(t[3]) > 0.95                                         # along the optical axis
            e2 = f * PerspectiveMap()(t) .+ c
            C2 = -(SMatrix{3, 3}(R)' * SVector{3}(t)); e1 = f * PerspectiveMap()(C2) .+ c
            @test all(p -> 0 < p < 2c, e1) && all(p -> 0 < p < 2c, e2)
            for (px, nm) in zip(cs_px, cs_nm)
                w = SMatrix{3, 3}(R) * SVector(nm.first[1], nm.first[2], 1.0)
                @test norm(f * PerspectiveMap()(w) .+ c - px.second) >= 1.0
            end
        end
    end
end
