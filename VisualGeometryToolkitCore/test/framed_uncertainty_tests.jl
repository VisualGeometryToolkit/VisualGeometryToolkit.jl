@testitem "framed_uncertainty" begin
    using Test
    using VisualGeometryToolkitCore
    using FiniteDiff: FiniteDiff
    using StaticArrays
    using LinearAlgebra: dot, norm, I
    using Rotations: RotMatrix, RotXYZ
    using Statistics: cov

    # =============================================================================
    # Framed statistical forms (0.9) — CovMat/PrecMat frame labels + checked
    # propagation through framed maps
    # =============================================================================
    #
    # Covers: the gradual-typing default (every pre-0.9 construction compiles
    # unchanged, frame = UnlabeledFrame{N}), relabel/unlabel, frame preservation
    # through inv / cov / prec (representation swaps, frame does not), BOTH
    # transformation laws verified numerically against direct conjugation
    # (covariance pushforward JΣJᵀ; precision pullback J⁻ᵀΛJ⁻¹ — the direction
    # pair the labels exist to check), the frame-junction wildcard truth table
    # with labeled-mismatch MethodErrors, Uncertain end-to-end transport through
    # framed EuclideanMaps and HomographyMaps (point-dependent Jacobian, pinned
    # against finite differences), and the checked Sampson delegation.

    @testset "Framed statistical forms (0.9)" begin
        Va = ImageFrame{:a}
        Vb = ImageFrame{:b}
        U2 = UnlabeledFrame{2}

        @testset "gradual defaults — pre-0.9 constructions compile unchanged" begin
            Σ = CovMat{2, Float64}(0.1, 0.02, 0.02, 0.2)
            @test Σ isa CovMat{2, Float64, 4, U2}
            Λf = PrecMat{2, Float32}(2.0f0, 0.1f0, 0.1f0, 3.0f0)
            @test Λf isa PrecMat{2, Float32, 4, UnlabeledFrame{2}}
            # historical 3-parameter surface, SMatrix lift, AbstractMatrix path
            @test CovMat{2, Float64, 4}((0.1, 0.0, 0.0, 0.2)) isa CovMat{2, Float64, 4, U2}
            @test CovMat(@SMatrix [0.1 0.0; 0.0 0.2]) isa CovMat{2, Float64, 4, U2}
            @test PrecMat{2, Float64}([1.0 0.0; 0.0 1.0]) isa PrecMat{2, Float64, 4, U2}
            # Uncertain takes a stated form only; a CovMat of a bare SMatrix is unlabeled
            u = Uncertain(Point2(100.0, 200.0), CovMat(@SMatrix [0.1 0.0; 0.0 0.2]))
            @test_throws MethodError Uncertain(Point2(100.0, 200.0), @SMatrix [0.1 0.0; 0.0 0.2])
            @test getfield(u, :attributes) isa CovMat{2, Float64, 4, U2}
            @test u.matrix === getfield(u, :attributes)   # `.matrix` reads the form
            # dof-generic: an 8×8 parameter covariance defaults to UnlabeledFrame{8}
            @test CovMat(SMatrix{8, 8, Float64}(I)) isa CovMat{8, Float64, 64, UnlabeledFrame{8}}
            # StaticArrays algebra still lands where it used to
            @test (2.0 * Σ)[1, 1] ≈ 0.2
            @test Σ * SVector(1.0, 0.0) ≈ SVector(0.1, 0.02)
            # Mahalanobis hot path unchanged
            r = SVector(0.3, -0.1)
            @test dot(r, PrecMat{2, Float64}(1.0, 0.0, 0.0, 1.0) * r) ≈ dot(r, r)
            # storage-boundary convert (the EuclideanMap idiom): containers typed
            # against the historical 3-parameter surface keep accepting values —
            # the downstream RoMa-cspond pattern (VGF dense_match.jl).
            T = Float32
            v = Vector{Uncertain{Point2{T}, PrecMat{2, T, 4}}}(undef, 1)
            v[1] = Uncertain(Point2{T}(1, 2), PrecMat{2, T}(SMatrix{2, 2, T}(I)))
            @test geom(v[1]) == Point2{T}(1, 2)
            # convert may LABEL (unlabeled → stated frame)…
            Σl = convert(CovMat{2, Float64, 4, ImageFrame{:a}}, CovMat{2, Float64}(0.1, 0.0, 0.0, 0.2))
            @test Σl isa CovMat{2, Float64, 4, ImageFrame{:a}}
            # …but never silently relabels a labeled frame to a different one
            @test_throws MethodError convert(
                CovMat{2, Float64, 4, ImageFrame{:b}}, Σl
            )
        end

        @testset "relabel / unlabel; frame survives inv, cov, prec" begin
            Σ = relabel(Va, CovMat{2, Float64}(0.1, 0.02, 0.02, 0.2))
            @test Σ isa CovMat{2, Float64, 4, Va}
            @test Tuple(Σ) == (0.1, 0.02, 0.02, 0.2)
            @test unlabel(Σ) isa CovMat{2, Float64, 4, U2}
            @test relabel(Va, unlabel(Σ)) === Σ
            # a 3D frame cannot label a 2D covariance
            @test_throws MethodError relabel(WorldFrame, Σ)
            # inversion swaps the REPRESENTATION (which law applies), not the frame
            Λ = inv(Σ)
            @test Λ isa PrecMat{2, Float64, 4, Va}
            @test inv(Λ) isa CovMat{2, Float64, 4, Va}
            @test SMatrix{2, 2}(inv(Λ)) ≈ SMatrix{2, 2}(Σ) rtol = 1.0e-12
            # Uncertain accessors preserve the frame in both storage forms
            uc = Uncertain(Point2(1.0, 2.0), Σ)
            @test cov(uc) isa CovMat{2, Float64, 4, Va}
            @test prec(uc) isa PrecMat{2, Float64, 4, Va}
            up = Uncertain(Point2(1.0, 2.0), Λ)
            @test prec(up) isa PrecMat{2, Float64, 4, Va}
            @test cov(up) isa CovMat{2, Float64, 4, Va}
            # scalar/broadcast algebra escapes the wrapper (an arbitrary matrix
            # expression is not a covariance) — same behavior as the other
            # @smatrix_wrapper types; relabel a rewrapped result deliberately.
            @test (2.0 * Σ) isa SMatrix{2, 2, Float64, 4}
            @test relabel(Va, CovMat(2.0 * Σ)) isa CovMat{2, Float64, 4, Va}
        end

        # A labeled 2D rigid map for the law tests: J = R is the (constant)
        # Jacobian of the point action.
        Rm = RotMatrix{2, Float64}(0.7)
        m = relabel(Va, Vb, EuclideanMap(Rm, SVector(3.0, -1.0)))
        J = SMatrix{2, 2, Float64}(Rm)
        Σ0 = @SMatrix [0.1 0.02; 0.02 0.2]
        Λ0 = inv(Σ0)

        @testset "covariance law — pushforward Σ ↦ JΣJᵀ vs direct conjugation" begin
            Σ = relabel(Va, CovMat(Σ0))
            Σ′ = propagate(m, Σ)
            @test Σ′ isa CovMat{2, Float64, 4, Vb}
            @test SMatrix{2, 2}(Σ′) ≈ J * Σ0 * J' rtol = 1.0e-14
            # 3D framed map (ViewFrame labels)
            R3 = RotMatrix{3, Float64}(RotXYZ(0.2, -0.4, 0.9))
            m3 = relabel(ViewFrame{:a}, ViewFrame{:b}, EuclideanMap(R3, SVector(1.0, 2.0, 3.0)))
            Σ3m = @SMatrix [0.3 0.05 0.0; 0.05 0.2 0.01; 0.0 0.01 0.4]
            Σ3′ = propagate(m3, relabel(ViewFrame{:a}, CovMat(Σ3m)))
            @test Σ3′ isa CovMat{3, Float64, 9, ViewFrame{:b}}
            J3 = SMatrix{3, 3, Float64}(R3)
            @test SMatrix{3, 3}(Σ3′) ≈ J3 * Σ3m * J3' rtol = 1.0e-13
        end

        @testset "precision law — pullback Λ ↦ J⁻ᵀΛJ⁻¹ vs direct conjugation" begin
            Λ = relabel(Va, PrecMat(Λ0))
            Λ′ = propagate(m, Λ)
            @test Λ′ isa PrecMat{2, Float64, 4, Vb}
            Jinv = inv(J)
            @test SMatrix{2, 2}(Λ′) ≈ Jinv' * Λ0 * Jinv rtol = 1.0e-12
            # the law pair is coherent: propagate(m, inv(Σ)) == inv(propagate(m, Σ))
            Σ = relabel(Va, CovMat(Σ0))
            @test SMatrix{2, 2}(propagate(m, inv(Σ))) ≈
                SMatrix{2, 2}(inv(propagate(m, Σ))) rtol = 1.0e-10
        end

        @testset "frame junction — labeled mismatch throws; wildcard rows are accepted" begin
            Σa = relabel(Va, CovMat(Σ0))
            # labeled-labeled mismatch (either statistical form) is a MethodError
            @test_throws MethodError propagate(m, relabel(ImageFrame{:c}, CovMat(Σ0)))
            @test_throws MethodError propagate(m, relabel(ImageFrame{:c}, PrecMat(Λ0)))
            # convention-style near-miss: PixelFrame is not ImageFrame
            @test_throws MethodError propagate(m, relabel(PixelFrame{:colmap}, CovMat(Σ0)))
            # wildcard: unlabeled Σ through a labeled map — unchecked, result Dst
            @test propagate(m, CovMat(Σ0)) isa CovMat{2, Float64, 4, Vb}
            # wildcard: labeled Σ through an unlabeled map — unchecked, result
            # takes the map's (unlabeled) Dst as-is, per the composition table
            mu = EuclideanMap(Rm, SVector(3.0, -1.0))
            @test propagate(mu, Σa) isa CovMat{2, Float64, 4, U2}
            # unlabeled-unlabeled never throws
            @test propagate(mu, CovMat(Σ0)) isa CovMat{2, Float64, 4, U2}
            # the raw-Jacobian layer stays frame-free: no check, no label, same math
            @test propagate(J, Σa) ≈ J * Σ0 * J'
        end

        @testset "Uncertain end-to-end through a framed EuclideanMap" begin
            u = Uncertain(Point2(1.0, 2.0), relabel(Va, CovMat(Σ0)))
            u′ = propagate(m, u)
            @test geom(u′) ≈ Rm * SVector(1.0, 2.0) + SVector(3.0, -1.0)
            @test cov(u′) isa CovMat{2, Float64, 4, Vb}
            @test SMatrix{2, 2}(cov(u′)) ≈ J * Σ0 * J' rtol = 1.0e-14
            # precision-side measurement: same one checked call, pullback law
            up′ = propagate(m, Uncertain(Point2(1.0, 2.0), relabel(Va, PrecMat(Λ0))))
            @test prec(up′) isa PrecMat{2, Float64, 4, Vb}
            Jinv = inv(J)
            @test SMatrix{2, 2}(prec(up′)) ≈ Jinv' * Λ0 * Jinv rtol = 1.0e-12
            # a mismatched frame inside the wrapper throws
            @test_throws MethodError propagate(
                m, Uncertain(Point2(1.0, 2.0), relabel(ImageFrame{:c}, CovMat(Σ0)))
            )
        end

        @testset "HomographyMap — point-dependent Jacobian, both laws" begin
            Hm = @SMatrix [1.1 0.2 5.0; -0.3 0.9 -3.0; 1.0e-4 2.0e-4 1.0]
            h = HomographyMap{Va, Vb}(Hm)
            x = Point2(120.0, 80.0)
            f = y -> PerspectiveMap()(Hm * SVector(y[1], y[2], 1.0))
            u′ = propagate(h, Uncertain(x, relabel(Va, CovMat(Σ0))))
            @test geom(u′) ≈ f(x) rtol = 1.0e-12
            @test cov(u′) isa CovMat{2, Float64, 4, Vb}
            # covariance pinned against a finite-difference Jacobian
            Jfd = FiniteDiff.finite_difference_jacobian(f, SVector(x[1], x[2]), Val(:central))
            @test SMatrix{2, 2}(cov(u′)) ≈ Jfd * Σ0 * Jfd' rtol = 1.0e-5
            # precision pulls back with the same J: the two transports are inverses
            upre′ = propagate(h, Uncertain(x, relabel(Va, PrecMat(Λ0))))
            @test prec(upre′) isa PrecMat{2, Float64, 4, Vb}
            @test SMatrix{2, 2}(inv(prec(upre′))) ≈ SMatrix{2, 2}(cov(u′)) rtol = 1.0e-8
            # mismatch throws; unlabeled measurement is the wildcard and lands on Dst
            @test_throws MethodError propagate(h, Uncertain(x, relabel(ImageFrame{:c}, CovMat(Σ0))))
            @test cov(propagate(h, Uncertain(x, CovMat(Σ0)))) isa CovMat{2, Float64, 4, Vb}
        end

        @testset "checked Sampson delegation — slot meets measurement frame" begin
            K1 = @SMatrix [600.0 0.0 320.0; 0.0 600.0 240.0; 0.0 0.0 1.0]
            K2 = @SMatrix [700.0 0.0 330.0; 0.0 700.0 250.0; 0.0 0.0 1.0]
            R = RotMatrix{3, Float64}(RotXYZ(0.08, -0.15, 0.05))
            t = SVector(0.4, -0.15, 0.1)
            F = FundMat(EssentialMat(R, t), K1, K2)
            form = EpipolarForm{Va, Vb}(F)
            x₁ = Point2(300.0, 220.0)
            x₂ = Point2(310.0, 230.0)
            Λ = PrecMat{2, Float64}(2.0, 0.1, 0.1, 3.0)
            u₁ = Uncertain(x₁, relabel(Va, Λ))
            u₂ = Uncertain(x₂, relabel(Vb, Λ))
            # matching frames: same value as the unchecked Mat machinery
            @test sampson_distance(form, u₁, u₂) == sampson_distance(F, u₁, u₂)
            @test sampson_distance(form, u₁ => u₂) == sampson_distance(F, u₁, u₂)
            # labeled mismatch on either slot throws
            u_bad = Uncertain(x₂, relabel(ImageFrame{:c}, Λ))
            @test_throws MethodError sampson_distance(form, u₁, u_bad)
            @test_throws MethodError sampson_distance(form, u_bad, u₂)
            # wildcards: plain points and unlabeled precision are accepted unchecked
            @test sampson_distance(form, x₁, x₂) == sampson_distance(F, x₁, x₂)
            @test sampson_distance(form, Uncertain(x₁, Λ), Uncertain(x₂, Λ)) ==
                sampson_distance(F, Uncertain(x₁, Λ), Uncertain(x₂, Λ))
            # unlabeled form accepts labeled measurements (slot side is the wildcard)
            @test sampson_distance(EpipolarForm(F), u₁, u₂) == sampson_distance(F, u₁, u₂)
        end
    end
end
