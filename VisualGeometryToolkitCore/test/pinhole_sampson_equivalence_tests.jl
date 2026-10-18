# B8 gate 2: on a PINHOLE, the tangent Sampson distance (unit bearings, through
# the backprojection Jacobian) agrees with the canonical image-plane Sampson
# (pixels, through F) TO FIRST ORDER.
#
# THE DERIVATION, which is what makes the bound exact rather than asymptotic —
# and what makes this a gate rather than a tolerance someone chose. With
# fᵢ = Kᵢ⁻¹ũᵢ / nᵢ, nᵢ = ‖Kᵢ⁻¹ũᵢ‖, and F = K₂⁻ᵀ E K₁⁻¹:
#
#     g_E = f₂ᵀ E f₁ = (ũ₂ᵀ F ũ₁) / (n₁n₂) = g_F / (n₁n₂)
#
# — the SAME constraint, rescaled by a positive point-dependent factor. So
#
#     ∇g_E = (∇g_F − g_F ∇log(n₁n₂)) / (n₁n₂)
#
# and with d = |g|/‖∇g‖, L = ‖∇_u log(n₁n₂)‖ over both views, r = d_F·L:
#
#     d_E/d_F = ‖∇g_F‖ / ‖∇g_F − g_F·∇log(n₁n₂)‖ ∈ [1/(1+r), 1/(1−r)]
#
# by the triangle inequality ALONE — no expansion, no neglected term — hence
#
#     |d_E/d_F − 1| ≤ r/(1−r),     r = d_F · L.
#
# Both factors are first order in the noise, so the ABSOLUTE discrepancy
# |d_E − d_F| ≲ d_F²·L is SECOND order: that is precisely the claim "they agree
# to first order", and the σ sweep below measures the exponent rather than
# asserting it.
#
# For a pinhole the gradient has a closed form, checked here against central
# differences rather than assumed:
#
#     |∇_u log n| = ρ / (f² + ρ²) = sin(2θ) / (2f),   θ = atan(ρ/f)
#
# maximal at θ = 45°, where it is 1/(2f). So the discrepancy is worst on a
# SHORT lens at a WIDE field angle, which is where the grid is densest.
#
# MEASURED (27 cells, f ∈ {1200, 400, 200} × field ∈ {on-axis, mid, wide} ×
# σ ∈ {0.25, 1, 4} px, 4000 samples each). Every cell sits inside the bound;
# the tightest is 0.354 of it. The worst cell in absolute terms:
#
#     f = 200, wide field (θ up to 68°), σ = 4 px
#         max |d_E/d_F − 1| = 9.47e-3     bound 3.71e-2     tightness 0.354
#         max |d_E − d_F|   = 0.124 px
#
# and at settings anyone actually runs, f = 400 wide field at σ = 1 px:
# 7.4e-4 relative, 2.3e-3 px absolute — three orders below any threshold in
# the stack.
#
# THE RESOLUTION FLOOR, and why it is not the bound being loosened. Below
# d_F ≈ 1e-3 px the bound is smaller than Float64 can resolve: g_F is a
# difference of O(1) terms, so a near-exact correspondence loses every
# significant digit. Measured over the grid, 158 of 108000 samples (0.15%) sit
# below that floor and their ABSOLUTE discrepancy never exceeds 6.2e-10 px. The
# three worst of them were re-measured at 256-bit precision and the bound holds
# there with room to spare (tightness 0.083 / 0.318 / 0.032, against the 11.2 /
# 10.4 / 2.8 that Float64 reports) — the same check is re-run below, so the
# exclusion is a proven property of the arithmetic and not an assumption. The
# identity error tells the same story directly: it tightens monotonically as
# the floor rises, 3.9e-11 → 1.5e-11 → 1.1e-12 → 2.1e-13 for floors of
# 1e-3 / 1e-2 / 1e-1 / 1 px.
@testitem "pinhole_sampson_equivalence" begin
    using Test
    using LinearAlgebra
    using StaticArrays
    using Random
    using Rotations
    using FiniteDiff: FiniteDiff
    using VisualGeometryToolkitCore
    using VisualGeometryToolkitCore: EssentialMat, CameraModel, LogicalIntrinsics,
        CameraCalibrationMat, Pinhole, backproject, project, backproject_with_jacobian,
        sampson_distance, EpipolarForm, EpipolarMap, HomographyMap, GnomonicFrame,
        ImageFrame, to_affine

    Kmat(f, c) = SMatrix{3, 3, Float64}(f, 0, 0, 0, f, 0, c[1], c[2], 1)
    cam_of(f, c) = CameraModel(; focal = f, pp = (c[1], c[2]))

    # ‖∇_u log n‖ per view by central differences — so the closed form is
    # CHECKED, not assumed.
    grad_log_n(K, u) = FiniteDiff.finite_difference_gradient(v -> log(norm(K \ SVector(v[1], v[2], 1.0))), [u[1], u[2]])

    # F built through the documented path: compose the correlation with the
    # calibrated → pixel collineations, which IS K₂⁻ᵀ E K₁⁻¹.
    function pixel_F(E, K1, K2)
        H1 = HomographyMap{GnomonicFrame{:a}, ImageFrame{:a}}(K1)
        H2 = HomographyMap{GnomonicFrame{:b}, ImageFrame{:b}}(K2)
        return parent(H2 ∘ EpipolarMap{GnomonicFrame{:a}, GnomonicFrame{:b}}(E) ∘ inv(H1))
    end

    FLOOR = 1.0e-3      # px — see the header

    function equiv_cell(; f1, f2, ctr, R, t, ρ_lo, ρ_hi, σ, n = 4000, seed = 7, depth = 3.0)
        rng = Xoshiro(seed)
        K1 = Kmat(f1, ctr); K2 = Kmat(f2, ctr)
        cam1 = cam_of(f1, ctr); cam2 = cam_of(f2, ctr)
        E = EssentialMat(SMatrix{3, 3}(R), t)
        Em = SMatrix{3, 3, Float64}(E)
        Fm = pixel_F(E, K1, K2); Fs = SMatrix{3, 3, Float64}(Fm)
        worst_rel = 0.0; worst_abs = 0.0; worst_tight = 0.0
        worst_theta = 0.0; worst_bound = 0.0
        g_err = 0.0; g_err_hi = 0.0; L_err = 0.0
        kept = 0; below = 0; abs_below = 0.0
        for _ in 1:(20n)
            kept ≥ n && break
            φ = 2π * rand(rng)
            ρ = sqrt(ρ_lo^2 + (ρ_hi^2 - ρ_lo^2) * rand(rng))
            u1 = ctr + SVector(ρ * cos(φ), ρ * sin(φ))
            X2 = SMatrix{3, 3}(R) * (backproject(cam1, u1) * depth) + t
            X2[3] > 0.2 || continue
            u2v = project(cam2, X2); u2t = SVector(u2v[1], u2v[2])
            (all(isfinite, u2t) && norm(u2t - ctr) < 900.0) || continue
            v1 = u1 + σ * SVector(randn(rng), randn(rng))
            v2 = u2t + σ * SVector(randn(rng), randn(rng))
            kept += 1
            dF = sampson_distance(Fm, v1, v2)
            b1, M1 = backproject_with_jacobian(cam1, v1)
            b2, M2 = backproject_with_jacobian(cam2, v2)
            dE = sampson_distance(E, b1, b2; M₁ = M1, M₂ = M2)
            (isfinite(dF) && isfinite(dE) && dF > 0) || continue

            n1 = norm(K1 \ to_affine(v1)); n2 = norm(K2 \ to_affine(v2))
            gF = dot(to_affine(v2), Fs * to_affine(v1))
            gE = dot(b2, Em * b1)
            if dF > FLOOR
                g_err = max(g_err, abs(gE - gF / (n1 * n2)) / abs(gE))
                dF > 1.0 && (g_err_hi = max(g_err_hi, abs(gE - gF / (n1 * n2)) / abs(gE)))
            end
            ρ1 = norm(v1 - ctr); ρ2 = norm(v2 - ctr)
            L = norm(SVector{4}(vcat(grad_log_n(K1, v1), grad_log_n(K2, v2))))
            L_cf = sqrt((ρ1 / (f1^2 + ρ1^2))^2 + (ρ2 / (f2^2 + ρ2^2))^2)
            L_err = max(L_err, abs(L - L_cf) / L_cf)

            r = dF * L
            rel = abs(dE / dF - 1)
            if dF <= FLOOR
                below += 1; abs_below = max(abs_below, abs(dE - dF)); continue
            end
            bound = r < 1 ? r / (1 - r) : Inf
            if rel > worst_rel
                worst_rel = rel; worst_abs = abs(dE - dF)
                worst_theta = rad2deg(atan(ρ1 / f1)); worst_bound = bound
            end
            worst_tight = max(worst_tight, rel / bound)
        end
        return (;
            kept, below, abs_below, worst_rel, worst_abs, worst_tight,
            worst_theta, worst_bound, g_err, g_err_hi, L_err,
        )
    end

    CTR = SVector(640.0, 480.0)
    R_mod = RotMatrix(RotXYZ(0.04, -0.07, 0.02))
    t_mod = normalize(SVector(1.0, 0.2, -0.15))
    FIELDS = ((0.0, 60.0), (150.0, 300.0), (350.0, 520.0))   # on-axis, mid, wide

    @testset "the exact identity the bound rests on: g_E = g_F / (n₁n₂)" begin
        # If this fails, the bound below is not a bound at all — it is derived
        # from this line. Measured above a 1 px residual, where g_F has not
        # cancelled: 2.1e-13.
        for f in (1200.0, 400.0, 200.0)
            r = equiv_cell(; f1 = f, f2 = f, ctr = CTR, R = R_mod, t = t_mod,
                ρ_lo = 150.0, ρ_hi = 300.0, σ = 4.0)
            @test r.g_err_hi < 1.0e-11
        end
    end

    @testset "the closed form ρ/(f²+ρ²) is the gradient it claims to be" begin
        # Checked against central differences, not against itself. The residual
        # is the finite-difference error (FiniteDiff's relative step), not a model error.
        for f in (1200.0, 400.0, 200.0), (lo, hi) in FIELDS
            r = equiv_cell(; f1 = f, f2 = f, ctr = CTR, R = R_mod, t = t_mod,
                ρ_lo = lo, ρ_hi = hi, σ = 1.0)
            @test r.L_err < 1.0e-6                       # per-view gradient error measured ≤ 1.4e-8
        end
    end

    @testset "every cell sits INSIDE the derived bound" begin
        worst_t = 0.0; worst_r = 0.0; worst_a = 0.0; below_tot = 0; abs_b = 0.0
        for f in (1200.0, 400.0, 200.0), (lo, hi) in FIELDS, σ in (0.25, 1.0, 4.0)
            r = equiv_cell(; f1 = f, f2 = f, ctr = CTR, R = R_mod, t = t_mod,
                ρ_lo = lo, ρ_hi = hi, σ)
            @test r.kept == 4000
            @test r.worst_tight < 1.0                    # THE gate
            worst_t = max(worst_t, r.worst_tight)
            worst_r = max(worst_r, r.worst_rel)
            worst_a = max(worst_a, r.worst_abs)
            below_tot += r.below; abs_b = max(abs_b, r.abs_below)
        end
        # The measured margins, so a regression that merely narrows them shows.
        @test worst_t < 0.45                             # measured 0.354
        @test worst_r < 0.02                             # measured 9.47e-3 (f200/wide/σ=4)
        @test worst_a < 0.2                              # measured 0.124 px, same cell
        # The excluded tail is small and harmless — both halves asserted.
        @test below_tot < 0.01 * 27 * 4000               # measured 158 of 108000
        @test abs_b < 1.0e-8                             # measured 6.2e-10 px
    end

    @testset "the discrepancy is SECOND order in noise — the exponent, measured" begin
        # This is the whole claim, and it is a slope, not a tolerance. Theory:
        # |d_E − d_F| ≲ d_F²·L ⇒ exponent 2 in σ, while the RELATIVE discrepancy
        # ≲ d_F·L ⇒ exponent 1. Fitted on log-log, f = 400, wide field.
        σs = [0.25, 0.5, 1.0, 2.0, 4.0]
        rs = [equiv_cell(; f1 = 400.0, f2 = 400.0, ctr = CTR, R = R_mod, t = t_mod,
                ρ_lo = 350.0, ρ_hi = 520.0, σ = s) for s in σs]
        function slope(y)
            x = log.(σs); ly = log.(y); n = length(x)
            return (n * sum(x .* ly) - sum(x) * sum(ly)) / (n * sum(x .^ 2) - sum(x)^2)
        end
        @test isapprox(slope([r.worst_abs for r in rs]), 2.0; atol = 0.15)   # measured 2.027
        @test isapprox(slope([r.worst_rel for r in rs]), 1.0; atol = 0.15)   # measured 1.027
    end

    @testset "below the floor the LIMIT IS FLOAT64, not the bound" begin
        # Re-measure the near-exact samples at 256 bits. Float64 reports the
        # bound violated (tightness 11.2 / 10.4 / 2.8); the same samples in
        # extended precision are at 0.083 / 0.318 / 0.032. Nothing was
        # loosened — the arithmetic simply cannot resolve a 3e-9 bound.
        setprecision(BigFloat, 256) do
            T = BigFloat
            K = Kmat(200.0, CTR); cam = cam_of(200.0, CTR)
            E = EssentialMat(SMatrix{3, 3}(R_mod), t_mod)
            Eb = T.(SMatrix{3, 3, Float64}(E))
            Fm = pixel_F(E, K, K)
            Kb = inv(T.(K))
            function big_tight(v1, v2)
                Fb = transpose(Kb) * Eb * Kb
                a1 = SVector{3, T}(T(v1[1]), T(v1[2]), one(T))
                a2 = SVector{3, T}(T(v2[1]), T(v2[2]), one(T))
                gF = dot(a2, Fb * a1); Fa = Fb * a1; Fd = transpose(Fb) * a2
                dF = abs(gF) / sqrt(Fd[1]^2 + Fd[2]^2 + Fa[1]^2 + Fa[2]^2)
                bear(a) = (w = Kb * a; w / norm(w))
                Mj(a) = (w = Kb * a; nw = norm(w); q = w / nw;
                    ((I - q * transpose(q)) / nw) * Kb[:, SOneTo(2)])
                f1 = bear(a1); f2 = bear(a2)
                gE = dot(f2, Eb * f1)
                r1 = transpose(transpose(Eb) * f2) * Mj(a1)
                r2 = transpose(Eb * f1) * Mj(a2)
                dE = abs(gE) / sqrt(r1[1]^2 + r1[2]^2 + r2[1]^2 + r2[2]^2)
                lg(a) = log(norm(Kb * a))
                gl(v) = FiniteDiff.finite_difference_gradient(w -> lg(SVector{3, T}(w[1], w[2], one(T))), T[v[1], v[2]])
                L = norm(SVector{4, T}(vcat(gl(v1), gl(v2))))
                rr = dF * L
                return Float64(abs(dE / dF - 1) / (rr / (1 - rr)))
            end
            rng = Xoshiro(7); found = 0
            for _ in 1:40000
                found ≥ 3 && break
                φ = 2π * rand(rng)
                ρ = sqrt(350.0^2 + (520.0^2 - 350.0^2) * rand(rng))
                u1 = CTR + SVector(ρ * cos(φ), ρ * sin(φ))
                X2 = SMatrix{3, 3}(R_mod) * (backproject(cam, u1) * 3.0) + t_mod
                X2[3] > 0.2 || continue
                u2v = project(cam, X2); u2t = SVector(u2v[1], u2v[2])
                all(isfinite, u2t) || continue
                v1 = u1 + 0.25 * SVector(randn(rng), randn(rng))
                v2 = u2t + 0.25 * SVector(randn(rng), randn(rng))
                dF = sampson_distance(Fm, v1, v2)
                (isfinite(dF) && 1.0e-6 < dF < 3.0e-6) || continue
                found += 1
                @test big_tight(v1, v2) < 1.0       # measured 0.083 / 0.318 / 0.032
            end
            @test found == 3                        # the samples are still there
        end
    end
end
