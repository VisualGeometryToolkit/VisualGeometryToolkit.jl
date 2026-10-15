# A TransferForm's frame change — `∘` on its HomographyMap, the homography's
# transformation law.
#
# WHY THIS FILE EXISTS AS ITS OWN GATE. RobustVisualGeometry solves a
# homography problem through measurement maps and must move the solved model
# back to the data frame. Written inline there, the transform would be a second
# expression of one piece of algebra in a second repo; written here, it is one.
# What makes it worth PINNING rather than merely testing is that the failure
# mode is silent: hand the maps in the opposite direction and the call still
# typechecks, still returns a homography, and still produces plausible
# residuals — it is simply the wrong homography. So the direction itself is
# asserted (`wrong direction fails loudly`), not just the law.
#
# WHAT IS AND IS NOT INVARIANT. A homography residual is EQUIVARIANT, not
# invariant: `g = π(H x̃₁) − x₂` is a displacement measured in the destination
# image, so a frame change moves it by that frame's local Jacobian. Asserting
# `constraint(D₂ ∘ f ∘ inv(D₁), x₁, x₂) == constraint(f, C₁x₁, C₂x₂)` would
# therefore fail on a CORRECT implementation. The three laws below are the ones
# that do hold, each measured over the same grid:
#
#   transferred point        π(H′ b̃₁) = D₂(π(H ã₁))                 exact
#   algebraic constraint     c′ = (λ₁λ₂ det D₂)·D₂⁻ᵀ c               exact
#   transfer residual, affine dst   g′ = A₂ g                        exact
#
# The third is exact rather than first-order precisely because an affine map
# has a CONSTANT Jacobian — and that is the case that ships, since an
# estimator's measurement map is `Kᵢ⁻¹`.
#
# Measured maxima over 2000 draws with κ(H), κ(D₁), κ(D₂) ≤ 100 (seed pinned,
# so these are reproducible, not typical): 8.21e-14, 3.48e-14, 3.18e-13, each
# relative to `max(|value|, 1)`. The assertions below sit ~10× above those, so
# they have teeth: a dropped inverse or a transposed factor moves these by
# twelve orders of magnitude, not by a factor of two.
@testitem "transfer_composition" begin
    using Test
    using LinearAlgebra
    using StaticArrays
    using Random
    using VisualGeometryCore
    using VisualGeometryCore: TransferForm, constraint_with_jacobians,
        HomographyMat, HomographyMap, to_affine, ImageFrame, GnomonicFrame,
        PixelFrame, UnlabeledFrame

    π2 = PerspectiveMap()
    relerr(a, b) = maximum(abs.(a .- b)) / max(maximum(abs.(b)), 1.0)

    # "Non-degenerate" made operational. Without a conditioning bound the grid
    # could report a flattering error simply because a near-singular draw sent
    # both sides to zero.
    function rand_proj(rng, κmax = 100.0)
        while true
            M = SMatrix{3, 3, Float64}(randn(rng, 3, 3))
            cond(Matrix(M)) ≤ κmax && return M
        end
    end
    function rand_affine(rng, κmax = 100.0)
        while true
            A = SMatrix{2, 2, Float64}(randn(rng, 2, 2))
            cond(Matrix(A)) ≤ κmax || continue
            t = 10 .* randn(rng, 2)
            return SMatrix{3, 3, Float64}(
                A[1, 1], A[2, 1], 0.0, A[1, 2], A[2, 2], 0.0, t[1], t[2], 1.0
            )
        end
    end

    @testset "the law, stated as matrices: H′ = D₂ H D₁⁻¹" begin
        rng = MersenneTwister(11)
        H = rand_proj(rng); D1 = rand_proj(rng); D2 = rand_proj(rng)
        f = TransferForm(HomographyMat{Float64}(Tuple(H)))
        fn = TransferForm(HomographyMap(D2) ∘ HomographyMap(f) ∘ inv(HomographyMap(D1)))
        @test SMatrix{3, 3, Float64}(parent(fn)) ≈ D2 * H * inv(D1) rtol = 1.0e-12
        # One-sided: the untouched endpoint keeps its coordinates.
        fs = TransferForm(HomographyMap(f) ∘ inv(HomographyMap(D1)))
        fd = TransferForm(HomographyMap(D2) ∘ HomographyMap(f))
        @test SMatrix{3, 3, Float64}(parent(fs)) ≈ H * inv(D1) rtol = 1.0e-12
        @test SMatrix{3, 3, Float64}(parent(fd)) ≈ D2 * H rtol = 1.0e-12
        # Composing nothing is the identity: the form's own map, same matrix.
        @test parent(TransferForm(HomographyMap(f))) == parent(f)
    end

    @testset "the estimator idiom: measurement maps run DATA → MODEL" begin
        # The direction the consumers hold. `Cᵢ` carries pixels into the model
        # frame, so the result `inv(C₂) ∘ H ∘ C₁`
        # is `H_data = C₂⁻¹ H C₁` — main's algebra, verified rather than taken.
        rng = MersenneTwister(12)
        K1 = SMatrix{3, 3, Float64}(900.0, 0, 0, 0, 900.0, 0, 640.0, 480.0, 1.0)
        K2 = SMatrix{3, 3, Float64}(1200.0, 0, 0, 0, 1200.0, 0, 640.0, 480.0, 1.0)
        C1 = inv(K1); C2 = inv(K2)
        Hm = rand_proj(rng)
        f = TransferForm(HomographyMat{Float64}(Tuple(Hm)))
        fd = TransferForm(inv(HomographyMap(C2)) ∘ HomographyMap(f) ∘ HomographyMap(C1))
        Hd = SMatrix{3, 3, Float64}(parent(fd))
        want = C2 \ (Hm * C1)
        nd = Hd / norm(Hd); nw = want / norm(want)
        dot(nd, nw) < 0 && (nw = -nw)
        @test norm(nd - nw) < 1.0e-13                      # measured 3.1e-15 over 500 draws
    end

    @testset "an exact model-frame correspondence is exact in pixels" begin
        # The zero set is what a frame change preserves. Built in the MODEL
        # frame and pushed to pixels through K — building it from `H_data`
        # instead would be tautological (g = 0 by construction).
        rng = MersenneTwister(4242)
        worst = 0.0
        for _ in 1:500
            f1 = 400 + 2000 * rand(rng); f2 = 400 + 2000 * rand(rng)
            K1 = SMatrix{3, 3, Float64}(f1, 0, 0, 0, f1, 0, 640.0, 480.0, 1.0)
            K2 = SMatrix{3, 3, Float64}(f2, 0, 0, 0, f2, 0, 640.0, 480.0, 1.0)
            C1 = inv(K1)
            Hm = rand_proj(rng)
            f = TransferForm(HomographyMat{Float64}(Tuple(Hm)))
            fd = TransferForm(HomographyMap(K2) ∘ HomographyMap(f) ∘ inv(HomographyMap(K1)))
            u1 = SVector(1280 * rand(rng), 960 * rand(rng))
            x1 = π2(C1 * to_affine(u1))                    # px → model
            x2 = π2(Hm * to_affine(x1))                    # exact, model frame
            u2 = π2(K2 * to_affine(x2))                    # model → px
            worst = max(worst, maximum(abs.(constraint_with_jacobians(fd, u1, u2)[1])))
        end
        @test worst < 1.0e-6                               # measured 4.04e-8 px
    end

    @testset "the three transformation laws, over a conditioned random grid" begin
        rng = MersenneTwister(20260923)
        e_transfer = 0.0; e_alg = 0.0; e_affine = 0.0; wrong_min = Inf
        for _ in 1:2000
            H = rand_proj(rng); D1 = rand_proj(rng); D2 = rand_proj(rng)
            b1 = SVector(randn(rng), randn(rng))
            b2 = SVector(randn(rng), randn(rng))
            f = TransferForm(HomographyMat{Float64}(Tuple(H)))
            m1 = HomographyMap(D1); m2 = HomographyMap(D2)
            Hn = SMatrix{3, 3, Float64}(parent(TransferForm(m2 ∘ HomographyMap(f) ∘ inv(m1))))
            a1 = π2(inv(D1) * to_affine(b1))
            a2 = π2(inv(D2) * to_affine(b2))

            # (1) the transferred point commutes with the frame change
            t_new = π2(Hn * to_affine(b1))
            t_old = π2(D2 * to_affine(π2(H * to_affine(a1))))
            e_transfer = max(e_transfer, relerr(t_new, t_old))

            # (2) the algebraic constraint's exact projective law
            λ1 = 1 / dot(inv(D1)[3, :], to_affine(b1))
            λ2 = 1 / dot(inv(D2)[3, :], to_affine(b2))
            c_new = cross(to_affine(b2), Hn * to_affine(b1))
            c_old = cross(to_affine(a2), H * to_affine(a1))
            e_alg = max(
                e_alg,
                relerr(c_new, (1 / (λ1 * λ2 * det(inv(D2)))) * transpose(inv(D2)) * c_old)
            )

            # (3) affine dst ⟹ the transfer residual scales by its linear part,
            # EXACTLY. This is the shipping case: an estimator's Kᵢ⁻¹ is affine.
            A2 = rand_affine(rng)
            fa = TransferForm(HomographyMap(A2) ∘ HomographyMap(f) ∘ inv(m1))
            g_new = constraint_with_jacobians(fa, b1, b2)[1]
            g_old = constraint_with_jacobians(f, a1, π2(inv(A2) * to_affine(b2)))[1]
            Alin = SMatrix{2, 2, Float64}(A2[1, 1], A2[2, 1], A2[1, 2], A2[2, 2])
            e_affine = max(e_affine, relerr(g_new, Alin * g_old))

            # (4) THE DIRECTION ITSELF. Handing the measurement maps C = D⁻¹
            # straight in — the mistake this gate exists for — must break the
            # transfer identity by a margin no rounding could produce.
            fw = TransferForm(HomographyMap(inv(D2)) ∘ HomographyMap(f) ∘ inv(HomographyMap(inv(D1))))
            tw = π2(SMatrix{3, 3, Float64}(parent(fw)) * to_affine(b1))
            wrong_min = min(wrong_min, relerr(tw, t_old))
        end
        @test e_transfer < 1.0e-12       # measured 8.21e-14
        @test e_alg < 1.0e-12            # measured 3.48e-14
        @test e_affine < 5.0e-12         # measured 3.18e-13
        # The wrong direction is never within 1e-3 of right, over 2000 draws;
        # the measured worst case (i.e. the most flattering) is 4.5e-2, eleven
        # orders of magnitude above the correct call's error.
        @test wrong_min > 1.0e-3
    end

    @testset "frame junctions are checked; wildcards are accepted" begin
        rng = MersenneTwister(13)
        Hm = HomographyMat{Float64}(Tuple(rand_proj(rng)))
        D = rand_proj(rng)
        A1 = ImageFrame{:a}; A2 = ImageFrame{:b}
        B1 = GnomonicFrame{:a}; B2 = GnomonicFrame{:b}
        f = TransferForm{A1, A2}(Hm)
        ok = TransferForm(HomographyMap{A2, B2}(D) ∘ HomographyMap(f) ∘ inv(HomographyMap{A1, B1}(D)))
        @test ok isa TransferForm{B1, B2}
        # A labeled mismatch at either junction is a MethodError, as for the
        # EpipolarForm slots.
        @test_throws MethodError HomographyMap(f) ∘ inv(HomographyMap{A2, B1}(D))
        @test_throws MethodError HomographyMap{PixelFrame{:colmap}, B2}(D) ∘ HomographyMap(f)
        # An UnlabeledFrame on either side of a junction is a wildcard.
        @test TransferForm(HomographyMap(f) ∘ inv(HomographyMap(D))) isa
            TransferForm{UnlabeledFrame{2}, A2}
        fu = TransferForm(Hm)
        @test TransferForm(HomographyMap(fu) ∘ inv(HomographyMap{A1, B1}(D))) isa TransferForm{B1, UnlabeledFrame{2}}
    end
end

@testitem "conjugation_jacobian" begin
    using Test, LinearAlgebra, StaticArrays, Random
    using VisualGeometryCore: _conjugation_jacobian

    # M ↦ A·M·B is linear: vec(A·M·B) = J·vec(M), and the estimators' two
    # pullbacks are its transposes (epipolar: kron(C₁, C₂) = Jᵀ at A = C₂ᵀ,
    # B = C₁; homography: J at A = C₂⁻¹, B = C₁).
    rng = Xoshiro(3)
    vecm(M) = SVector{9}(Tuple(M))
    rmat() = SMatrix{3, 3, Float64}(I + 0.3 * randn(rng, 3, 3))
    for _ in 1:20
        A, B, M = rmat(), rmat(), SMatrix{3, 3, Float64}(randn(rng, 3, 3))
        @test _conjugation_jacobian(A, B) * vecm(M) ≈ vecm(A * M * B) rtol = 1.0e-12
        C₁, C₂ = rmat(), rmat()
        @test transpose(_conjugation_jacobian(transpose(C₂), C₁)) == kron(C₁, C₂)
        @test _conjugation_jacobian(inv(C₂), C₁) == kron(transpose(C₁), inv(C₂))
    end
end
