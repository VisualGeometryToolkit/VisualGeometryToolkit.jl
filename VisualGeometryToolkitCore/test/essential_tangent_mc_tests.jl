# B8 gate 1: is the CALIBRATED E residual correctly calibrated, in the Monte
# Carlo sense?
#
# THE CLAIM. With pixel noise ε ~ N(0, Σ) in both views, the tangent Sampson
# residual of an EXACT correspondence, whitened by the pixel covariance
# propagated through the backprojection Jacobian, is a standard normal:
#
#     z = g / √(G_x Σ G_xᵀ),   g = f₂ᵀ E f₁,   G_x = (f₂ᵀE·M₁, f₁ᵀEᵀ·M₂)
#
# so NEES = E[z²] = 1, and a threshold quoted in PIXEL σ means what it says
# (|z| has the half-normal quantiles 1.960 at 95%, 2.576 at 99%). `M` arrives
# from `backproject_with_jacobian`, which is what puts G_x in pixels and keeps
# the threshold in the unit the detector actually guarantees — for any lens,
# including one whose rays the z = 1 chart cannot represent at all.
#
# WHY THIS FILE IS BUILT TO FAIL. A whitening bug does not announce itself; it
# looks like a small constant offset in a number that is near 1 anyway. So
# every cell computes, from the same samples, three DELIBERATELY WRONG
# residuals alongside the right one, and the test asserts both that the right
# one is inside the band and that the wrong one is outside it. Which cell
# catches which error is the substance:
#
#   W1  isotropic-scalar whitening (uses tr(Σ)/4 · ‖G_x‖², ignoring Σ's shape)
#       → INVISIBLE in every isotropic cell (it is the same arithmetic there),
#         and invisible on average even under anisotropic Σ when the geometry
#         is unaligned, because G_x's direction averages out. Caught only by
#         S1, where a horizontal baseline makes ∂g/∂u systematically VERTICAL
#         and Σ is 16× tighter in y: NEES 0.118 against 1.002, ~88 SE.
#   W2  a CONSTANT on-axis M instead of the per-point Jacobian
#       → invisible at the centre of a narrow lens (S5: 0.983, where M barely
#         varies), caught hard off-axis: S3 fisheye periphery 3.556 (~256 SE),
#         S4 wide pinhole full field 0.628 (~37 SE).
#   W3  the pixel Jacobian DROPPED (gradient left in radians)
#       → caught by every cell by five orders of magnitude (1e-5 to 1e-7).
#         This is the error the whole `(f, M)` interface exists to prevent.
#   W4  the two views' Jacobians SWAPPED
#       → NOT CAUGHT BY ANY NEES CELL, and that is measured, not assumed: the
#         swap is nearly norm-preserving because |Eᵀf₂|/|Ef₁| has median 1.018
#         and 5–95% range [0.972, 1.066], so the denominator barely moves (best
#         cell 0.970, 3 SE — inside a 4 SE band). What catches it is the
#         TANGENCY CONTRACT, asserted separately below: fᵀM = 4.8e-19 with the
#         right M against 1.0e-3 with the swapped one, a factor of 2e15.
#         A gate that only averaged would have accepted a swapped-Jacobian port.
#
# TWO CONTROLS, because a NEES near 1 proves nothing on its own:
#
#   σ IS WIRED IN. The cells are PAIRED across σ (same seed, so the same noise
#   directions), which makes any movement purely Sampson's second-order term.
#   Measured on S4: 1.0159 / 1.0159 / 1.0164 / 1.0239 / 1.0841 at σ = 0.5 / 2 /
#   8 / 30 / 80 px. Flat where it should be — second order is negligible below
#   ~10 px on this geometry — and departing at absurd noise, which is what
#   proves the flatness is a result and not an unwired parameter.
#
#   THE OFFSETS ARE SAMPLING NOISE. Single-seed cells land ~1.6% from 1, which
#   is the size a real whitening bias would be. Ten seeds at σ = 1 px, n =
#   20000: mean 0.9985, std 0.0093, against a PREDICTED SE of √(2/n) = 0.0100.
#   Predicting the standard error and comparing is what tells the two apart.
@testitem "essential_tangent_mc" begin
    using Test
    using LinearAlgebra
    using StaticArrays
    using Random
    using Rotations
    using Statistics
    using VisualGeometryToolkitCore
    using VisualGeometryToolkitCore: EssentialMat, CameraModel, LogicalIntrinsics,
        CameraCalibrationMat, Pinhole, backproject, project,
        backproject_with_jacobian, constraint_with_jacobians, sampson_distance

    cam_of(fx, fy, cx, cy, lens = Pinhole()) = CameraModel(
        LogicalIntrinsics(
            CameraCalibrationMat(SMatrix{3, 3, Float64}(fx, 0, 0, 0, fy, 0, cx, cy, 1))
        ), lens,
    )
    axis_M(cam, cx, cy) = backproject_with_jacobian(cam, SVector(cx, cy))[2]

    # An orthonormal tangent basis — the gradient a dropped-Jacobian bug leaves
    # behind, in radians rather than pixels (W3).
    function tangent_basis(fv)
        a = abs(fv[1]) < 0.9 ? SVector(1.0, 0.0, 0.0) : SVector(0.0, 1.0, 0.0)
        e1 = normalize(a - dot(a, fv) * fv)
        return hcat(e1, cross(SVector{3}(fv), e1))
    end

    function cell(;
            cam1, cam2, ctr, R, t, radius, depth = 3.0,
            Σ1 = SMatrix{2, 2, Float64}(I), Σ2 = SMatrix{2, 2, Float64}(I),
            σ = 1.0, n = 20000, seed = 1,
        )
        rng = Xoshiro(seed)
        Em = SMatrix{3, 3, Float64}(EssentialMat(SMatrix{3, 3}(R), t))
        L1 = SMatrix{2, 2, Float64}(σ * cholesky(Symmetric(Matrix(Σ1))).L)
        L2 = SMatrix{2, 2, Float64}(σ * cholesky(Symmetric(Matrix(Σ2))).L)
        Σ4 = σ^2 * SMatrix{4, 4, Float64}([Σ1 zeros(2, 2); zeros(2, 2) Σ2])
        σ̄² = (Σ4[1, 1] + Σ4[2, 2] + Σ4[3, 3] + Σ4[4, 4]) / 4
        A1 = axis_M(cam1, ctr[1], ctr[2]); A2 = axis_M(cam2, ctr[1], ctr[2])
        zs = Float64[]; w1 = Float64[]; w2 = Float64[]; w3 = Float64[]; w4 = Float64[]
        tries = 0
        while length(zs) < n && tries < 50n
            tries += 1
            θ = 2π * rand(rng); r = radius * sqrt(rand(rng))
            u1 = ctr + SVector(r * cos(θ), r * sin(θ))
            X2 = SMatrix{3, 3}(R) * (backproject(cam1, u1) * depth) + t
            X2[3] > 0.2 || continue
            u2v = project(cam2, X2)
            u2 = SVector(u2v[1], u2v[2])
            (all(isfinite, u2) && norm(u2 - ctr) < 3 * radius) || continue
            b1, M1 = backproject_with_jacobian(cam1, u1 + L1 * SVector(randn(rng), randn(rng)))
            b2, M2 = backproject_with_jacobian(cam2, u2 + L2 * SVector(randn(rng), randn(rng)))
            (all(isfinite, M1) && all(isfinite, M2)) || continue
            g = dot(b2, Em * b1)
            a = Em' * b2; b = Em * b1
            gx(Ma, Mb) = SVector{4, Float64}(
                (a' * Ma)[1], (a' * Ma)[2], (b' * Mb)[1], (b' * Mb)[2]
            )
            Gx = gx(M1, M2)
            push!(zs, g / sqrt(dot(Gx, Σ4 * Gx)))
            push!(w1, g / sqrt(σ̄² * sum(abs2, Gx)))
            G2 = gx(A1, A2); push!(w2, g / sqrt(dot(G2, Σ4 * G2)))
            G3 = gx(tangent_basis(b1), tangent_basis(b2)); push!(w3, g / sqrt(dot(G3, Σ4 * G3)))
            G4 = gx(M2, M1); push!(w4, g / sqrt(dot(G4, Σ4 * G4)))
        end
        nees(v) = mean(abs2, v)
        return (
            n = length(zs), nees = nees(zs), se = sqrt(2 / length(zs)),
            q95 = quantile(abs.(zs), 0.95), q99 = quantile(abs.(zs), 0.99),
            W1 = nees(w1), W2 = nees(w2), W3 = nees(w3), W4 = nees(w4),
        )
    end

    PIN_N = cam_of(1200.0, 1200.0, 640.0, 480.0)
    PIN_W = cam_of(400.0, 400.0, 640.0, 480.0)
    MIX1 = cam_of(1500.0, 1500.0, 640.0, 480.0)
    MIX2 = cam_of(260.0, 260.0, 640.0, 480.0, Lens(Equidistant()))
    FISH = cam_of(300.0, 300.0, 640.0, 480.0, Lens(Equidistant()))
    CTR = SVector(640.0, 480.0)
    R_mod = RotMatrix(RotXYZ(0.04, -0.07, 0.02))
    R_id = RotMatrix(RotXYZ(0.01, 0.015, -0.005))
    t_mod = normalize(SVector(1.0, 0.2, -0.15))
    t_horiz = SVector(1.0, 0.0, 0.0)
    ANISO = SMatrix{2, 2, Float64}(1.0, 0, 0, 0.0625)      # 16:1 in variance

    # (S1, …, S5) — the grid. `catches` names the wrong variant this cell is
    # here to expose; `blind` names one it provably cannot, so the table cannot
    # be read as "every cell catches everything".
    grid = (
        (name = "S1 horiz baseline, Σ ⊥ epipolar",
            kw = (; cam1 = PIN_W, cam2 = PIN_W, ctr = CTR, R = R_id, t = t_horiz,
                radius = 400.0, Σ1 = ANISO, Σ2 = ANISO, seed = 201),
            catches = :W1, blind = :W4),
        (name = "S2 mixed rig pinhole ↔ fisheye",
            kw = (; cam1 = MIX1, cam2 = MIX2, ctr = CTR, R = R_mod, t = t_mod,
                radius = 380.0, seed = 202),
            catches = :W3, blind = :W4),
        (name = "S3 fisheye f300, periphery",
            kw = (; cam1 = FISH, cam2 = FISH, ctr = CTR, R = R_mod, t = t_mod,
                radius = 420.0, seed = 203),
            catches = :W2, blind = :W1),
        (name = "S4 pinhole f400, full field",
            kw = (; cam1 = PIN_W, cam2 = PIN_W, ctr = CTR, R = R_mod, t = t_mod,
                radius = 450.0, seed = 204),
            catches = :W2, blind = :W1),
        (name = "S5 pinhole f1200, centre",
            kw = (; cam1 = PIN_N, cam2 = PIN_N, ctr = CTR, R = R_mod, t = t_mod,
                radius = 120.0, seed = 205),
            catches = :W3, blind = :W2),
    )

    @testset "NEES = 1 across the grid, and the thresholds are in pixel σ" begin
        for g in grid, σ in (0.5, 2.0)
            r = cell(; g.kw..., σ)
            @test r.n == 20000
            # 4 SE = 0.04. Measured worst over the grid: 1.0163 (S2).
            @test abs(r.nees - 1) < 4 * r.se
            # |z| is half-normal, so a pixel-σ threshold means what it says.
            # Measured: q95 1.958–1.980, q99 2.522–2.608.
            @test isapprox(r.q95, 1.960; atol = 0.05)
            @test isapprox(r.q99, 2.576; atol = 0.10)
        end
    end

    @testset "the cells have teeth — each wrong variant fails where it should" begin
        for g in grid
            r = cell(; g.kw..., σ = 1.0)
            band = 4 * r.se
            @test abs(getfield(r, g.catches) - 1) > band      # this cell catches it
            @test abs(getfield(r, g.blind) - 1) < band        # and provably misses that one
            # W3 — the dropped pixel Jacobian — is caught EVERYWHERE, by five
            # orders of magnitude. Measured 6.8e-7 (S5) to 1.0e-5 (S3).
            @test r.W3 < 1.0e-4
        end
        # And the specific measured margins, so a regression that merely
        # narrows the gap is visible rather than silently tolerated.
        s1 = cell(; grid[1].kw..., σ = 1.0)
        @test s1.W1 < 0.2                                     # measured 0.118 vs 1.0024
        s3 = cell(; grid[3].kw..., σ = 1.0)
        @test s3.W2 > 3.0                                     # measured 3.556
        s4 = cell(; grid[4].kw..., σ = 1.0)
        @test s4.W2 < 0.7                                     # measured 0.628
        s5 = cell(; grid[5].kw..., σ = 1.0)
        @test abs(s5.W2 - 1) < 0.05                           # 0.983: a narrow lens at
        #   the centre CANNOT see a constant-M bug. Stated so the grid is not
        #   read as "any cell would have caught it".
    end

    @testset "σ is wired in: the second-order term appears at absurd noise" begin
        # Flat below ~10 px is a RESULT (Sampson's second order is negligible
        # there); flat at σ = 80 px would mean σ never reached the residual.
        ns = [cell(; grid[4].kw..., σ = s).nees for s in (0.5, 2.0, 8.0, 30.0, 80.0)]
        @test all(abs.(ns[1:3] .- 1) .< 0.04)          # 1.0159 / 1.0159 / 1.0164
        @test ns[4] > ns[3]                            # 1.0239
        @test ns[5] > 1.05                             # 1.0841 — the term is real
    end

    @testset "the offsets are sampling noise, against a PREDICTED standard error" begin
        # A single cell landing 1.6% from 1 is the size a real whitening bias
        # would be. What separates them is predicting √(2/n) and comparing.
        vals = [cell(; grid[4].kw..., σ = 1.0, seed = s).nees for s in 301:310]
        @test isapprox(mean(vals), 1.0; atol = 0.01)   # measured 0.9985
        @test 0.5 < std(vals) / sqrt(2 / 20000) < 2.0  # measured 0.0093 / 0.0100
    end

    @testset "the tangency contract is what catches a swapped Jacobian" begin
        # The NEES cannot (see the header): the swap is norm-preserving to ~7%.
        # fᵀM = 0 is not an observation but a contract, and it breaks loudly.
        rng = Xoshiro(9)
        own = 0.0; swapped = 0.0; ratios = Float64[]
        Em = SMatrix{3, 3, Float64}(EssentialMat(SMatrix{3, 3}(R_mod), t_mod))
        for _ in 1:2000
            u1 = CTR + SVector(127.0 * randn(rng), 127.0 * randn(rng))
            X2 = SMatrix{3, 3}(R_mod) * (backproject(MIX1, u1) * 3.0) + t_mod
            X2[3] > 0.2 || continue
            u2v = project(MIX2, X2); u2 = SVector(u2v[1], u2v[2])
            all(isfinite, u2) || continue
            b1, M1 = backproject_with_jacobian(MIX1, u1)
            b2, M2 = backproject_with_jacobian(MIX2, u2)
            own = max(own, maximum(abs.(b1' * M1)), maximum(abs.(b2' * M2)))
            swapped = max(swapped, maximum(abs.(b1' * M2)), maximum(abs.(b2' * M1)))
            push!(ratios, norm(Em' * b2) / norm(Em * b1))
        end
        @test own < 1.0e-15                       # measured 4.8e-19
        @test swapped > 1.0e-5                    # measured 1.0e-3
        # …and the reason the average is blind: the two halves of G_x are the
        # same size to within a few percent, so swapping barely moves ‖G_x‖.
        @test 0.9 < median(ratios) < 1.1          # measured 1.018
        @test quantile(ratios, 0.95) < 1.2        # measured 1.066
    end
end
