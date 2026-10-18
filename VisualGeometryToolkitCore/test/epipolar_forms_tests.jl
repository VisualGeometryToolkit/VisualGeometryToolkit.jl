@testitem "epipolar_forms" begin
    import VisualGeometryToolkitCore as VgtkCore
    using Test
    using VisualGeometryToolkitCore
    using StaticArrays
    using LinearAlgebra: dot, norm
    using Rotations: RotMatrix, RotXYZ

    # =============================================================================
    # Slot-labeled epipolar forms (0.8) — EpipolarForm + view-identity frames
    # =============================================================================
    #
    # Covers: the labeled/unlabeled Mat → form lift and the pinned slot
    # convention (x₂ᵀ M x₁ = 0, slot 1 = right argument), transpose slot swap +
    # round-trip, the DELIBERATE absence of inv, the transformation law by ∘
    # verified numerically against direct conjugation H₂⁻ᵀ M H₁⁻¹ (including the
    # K-normalization bookkeeping E → F), slot-mismatch MethodErrors + the
    # wildcard truth-table rows, epipolar_map correctness on a synthetic
    # two-view setup (incidence, kernel, and pencil properties), the
    # depth_transfer rename (old behavior pinned, old name's Camera methods
    # gone), and ViewFrame/ImageFrame open-hierarchy + StereoRig id minting.

    # A downstream-style slot frame: the hierarchy stays open for forms too.
    # (GnomonicFrame itself was adopted into VGC in 0.9 as the per-view calibrated
    # frame; this local tag keeps pinning the OPEN-hierarchy property.)
    struct LocalFrame{id} <: Frame{2} end

    # Dehomogenize a homogeneous 3-vector/Point3 to Point2.
    dehom(v) = Point2(PerspectiveMap()(v))

    @testset "Slot-labeled epipolar forms (0.8)" begin
        # --- Synthetic two-view geometry -----------------------------------------
        K1 = @SMatrix [600.0 0.0 320.0; 0.0 600.0 240.0; 0.0 0.0 1.0]
        K2 = @SMatrix [700.0 0.0 330.0; 0.0 700.0 250.0; 0.0 0.0 1.0]
        R = RotMatrix{3, Float64}(RotXYZ(0.08, -0.15, 0.05))
        t = SVector(0.4, -0.15, 0.1)

        E = EssentialMat(R, t)          # x̂₂ᵀ E x̂₁ = 0
        F = FundMat(E, K1, K2)          # u₂ᵀ F u₁ = 0

        X1s = (
            SVector(0.3, -0.2, 4.0), SVector(-0.5, 0.4, 5.0), SVector(0.8, 0.1, 6.0),
            SVector(-0.2, -0.6, 4.5), SVector(0.1, 0.3, 5.5),
        )
        u1s = map(X -> dehom(K1 * X), X1s)
        u2s = map(X -> dehom(K2 * (R * X + t)), X1s)

        Va = ImageFrame{:a}
        Vb = ImageFrame{:b}
        form = EpipolarForm{Va, Vb}(F)          # labeled lift
        form_u = EpipolarForm(F)                # unlabeled lift

        @testset "Mat → form lift + pinned slot convention" begin
            @test form isa EpipolarForm{Va, Vb, FundMat{Float64}}
            @test form_u isa EpipolarForm{UnlabeledFrame{2}, UnlabeledFrame{2}, FundMat{Float64}}
            @test parent(form) === F
            # the Mat stays frame-free algebra; the wrapper does NOT enforce rank
            @test EpipolarForm(FundMat{Float64}(Tuple(K1))) isa EpipolarForm  # rank-3 accepted
            # two-slot evaluation IS the pinned convention: form(x₁, x₂) = x̃₂ᵀ M x̃₁
            for (u1, u2) in zip(u1s, u2s)
                # same algebra, two summation orders — equal up to round-off (an
                # exact `==` here tripped at the 1e-17 level under a dep update)
                @test isapprox(form(u1, u2), dot(to_affine(u2), SMatrix{3, 3}(F) * to_affine(u1)); atol = 1.0e-12)
                @test abs(form(u1, u2)) < 1.0e-8          # exact correspondence
            end
            # a scrambled pair violates the constraint
            @test abs(form(u1s[1], u2s[2])) > 1.0e-3
            # EssentialMat lifts too, wrapper type preserved
            fe = EpipolarForm{GnomonicFrame{:a}, GnomonicFrame{:b}}(E)
            @test fe.mat isa EssentialMat{Float64}
        end

        @testset "transpose swaps slots; round-trip; adjoint" begin
            tf = transpose(form)
            @test tf isa EpipolarForm{Vb, Va, FundMat{Float64}}
            @test SMatrix{3, 3}(tf.mat) == transpose(SMatrix{3, 3}(F))
            # swapped slots take the arguments in the other order
            for (u1, u2) in zip(u1s, u2s)
                @test abs(tf(u2, u1)) < 1.0e-8
            end
            # involution: exact round-trip in type and value
            rt = transpose(tf)
            @test rt isa typeof(form)
            @test rt.mat == form.mat
            # adjoint is the same operation (real coefficients)
            @test (form').mat == tf.mat && form' isa typeof(tf)
            # transpose preserves the wrapped Mat type (Eᵀ stays essential)
            @test transpose(EpipolarForm(E)).mat isa EssentialMat{Float64}
        end

        @testset "inv is deliberately absent (rank-2 correlation)" begin
            @test_throws MethodError inv(form)
            @test_throws MethodError inv(form_u)
        end

        @testset "relabel / unlabel escape hatches" begin
            u = unlabel(form)
            @test u isa EpipolarForm{UnlabeledFrame{2}, UnlabeledFrame{2}, FundMat{Float64}}
            @test u.mat === form.mat
            r = relabel(Va, Vb, u)
            @test r isa typeof(form) && r.mat === form.mat
            # a 3D frame cannot label a 2D slot
            @test_throws MethodError relabel(WorldFrame, Vb, form)
        end

        @testset "epipolar_map — incidence, kernel, pencil" begin
            m = epipolar_map(form)
            @test m isa EpipolarMap{Va, Vb, FundMat{Float64}}
            e₁, e₂ = epipoles(form)
            # kernels: F·e₁ ≈ 0 and Fᵀ·e₂ ≈ 0 (unit-norm epipoles from the SVD)
            @test norm(SMatrix{3, 3}(F) * e₁) < 1.0e-12
            @test norm(transpose(SMatrix{3, 3}(F)) * e₂) < 1.0e-12
            for (u1, u2) in zip(u1s, u2s)
                l = m(u1)
                @test l isa Line2D
                # incidence: the corresponding point lies on the epipolar line
                c = SVector{3}(l.coeffs)
                @test abs(dot(c, to_affine(u2))) / hypot(c[1], c[2]) < 1.0e-6  # px distance
                # pencil: every mapped line goes through the other epipole e₂
                @test abs(dot(c, e₂)) / (norm(c) * norm(e₂)) < 1.0e-10
            end
            # transpose direction: lines in view :a through the matching u1 and e₁
            mᵀ = epipolar_map(transpose(form))
            @test mᵀ isa EpipolarMap{Vb, Va}
            for (u1, u2) in zip(u1s, u2s)
                l = mᵀ(u2)
                c = SVector{3}(l.coeffs)
                @test abs(dot(c, to_affine(u1))) / hypot(c[1], c[2]) < 1.0e-6
                @test abs(dot(c, e₁)) / (norm(c) * norm(e₁)) < 1.0e-10
            end
        end

        @testset "composition — law verified against direct conjugation" begin
            H1m = @SMatrix [1.1 0.2 5.0; -0.3 0.9 -3.0; 1.0e-4 2.0e-4 1.0]
            H2m = @SMatrix [0.8 -0.1 -2.0; 0.4 1.2 7.0; -2.0e-4 1.0e-4 1.0]
            H1 = HomographyMap{Va, ImageFrame{:a2}}(H1m)
            H2 = HomographyMap{Vb, ImageFrame{:b2}}(H2m)

            g = EpipolarForm(H2 ∘ epipolar_map(form) ∘ inv(H1))
            @test g isa EpipolarForm{ImageFrame{:a2}, ImageFrame{:b2}, FundMat{Float64}}
            # NUMERICAL check against the written-out law M′ = H₂⁻ᵀ M H₁⁻¹
            M_direct = transpose(inv(H2m)) * SMatrix{3, 3}(F) * inv(H1m)
            @test SMatrix{3, 3}(g.mat) ≈ M_direct rtol = 1.0e-10
            # invariance: the composed form vanishes on the mapped correspondences
            for (u1, u2) in zip(u1s, u2s)
                u1′ = dehom(H1(u1))
                u2′ = dehom(H2(u2))
                @test abs(g(u1′, u2′)) < 1.0e-8
            end

            # one side only: the untouched slot keeps its frame and coordinates
            g1 = EpipolarForm(epipolar_map(form) ∘ inv(H1))
            @test g1 isa EpipolarForm{ImageFrame{:a2}, Vb, FundMat{Float64}}
            @test SMatrix{3, 3}(g1.mat) ≈ SMatrix{3, 3}(F) * inv(H1m) rtol = 1.0e-10
            g2 = EpipolarForm(H2 ∘ epipolar_map(form))
            @test g2 isa EpipolarForm{Va, ImageFrame{:b2}, FundMat{Float64}}
            @test SMatrix{3, 3}(g2.mat) ≈ transpose(inv(H2m)) * SMatrix{3, 3}(F) rtol = 1.0e-10
            # no composition: the form's own correlation, the same matrix
            @test EpipolarForm(epipolar_map(form)).mat === form.mat
            # the derivative of the composition in the model (compose_jacobian)
            vecm(M) = SVector{9}(Tuple(M))
            J = VgtkCore.compose_jacobian(H2, epipolar_map(form), inv(H1))
            @test J * vecm(SMatrix{3, 3}(F)) ≈ vecm(SMatrix{3, 3}(g.mat)) rtol = 1.0e-12
            @test_throws MethodError VgtkCore.compose_jacobian(H2, epipolar_map(form), H1)   # H1's Dst meets slot 1 (:a)

            # K-normalization bookkeeping: composing E with the calibrated → pixel
            # collineations computes exactly F = K₂⁻ᵀ E K₁⁻¹, frames checked.
            fe = EpipolarForm{GnomonicFrame{:a}, GnomonicFrame{:b}}(E)
            Ka = HomographyMap{GnomonicFrame{:a}, Va}(K1)
            Kb = HomographyMap{GnomonicFrame{:b}, Vb}(K2)
            gF = EpipolarForm(Kb ∘ epipolar_map(fe) ∘ inv(Ka))
            @test gF isa EpipolarForm{Va, Vb, FundMat{Float64}}
            @test SMatrix{3, 3}(gF.mat) ≈ SMatrix{3, 3}(F) rtol = 1.0e-8
            # the result is FundMat-backed even from an EssentialMat input: a
            # general frame change preserves rank, not the essential manifold
            @test !(gF.mat isa EssentialMat)
        end

        @testset "composition — slot mismatch throws; wildcard rows are accepted" begin
            H1m = @SMatrix [1.1 0.2 5.0; -0.3 0.9 -3.0; 1.0e-4 2.0e-4 1.0]
            # labeled-labeled mismatch on either slot is a MethodError
            H_wrong = HomographyMap{Vb, ImageFrame{:c}}(H1m)   # Src = :b meets slot :a
            @test_throws MethodError epipolar_map(form) ∘ inv(H_wrong)
            H_wrong2 = HomographyMap{Va, ImageFrame{:c}}(H1m)  # Src = :a meets slot :b
            @test_throws MethodError H_wrong2 ∘ epipolar_map(form)
            # origin-style near-miss: PixelFrame is not ImageFrame
            @test_throws MethodError epipolar_map(form) ∘ inv(HomographyMap{PixelFrame{:colmap}, Va}(H1m))

            # wildcard: unlabeled FORM accepts labeled maps; slots take Dst as-is
            gw = EpipolarForm(epipolar_map(form_u) ∘ inv(HomographyMap{Va, ImageFrame{:a2}}(H1m)))
            @test gw isa EpipolarForm{ImageFrame{:a2}, UnlabeledFrame{2}, FundMat{Float64}}
            # wildcard: labeled form accepts an UNLABELED map (Src is the wildcard)
            gu = EpipolarForm(HomographyMap(H1m) ∘ epipolar_map(form))
            @test gu isa EpipolarForm{Va, UnlabeledFrame{2}, FundMat{Float64}}
            # unlabeled-unlabeled junction never throws
            @test EpipolarForm(epipolar_map(form_u) ∘ inv(HomographyMap(H1m))) isa EpipolarForm
        end

        @testset "sampson_distance delegation" begin
            u1n = Point2(u1s[1][1] + 0.8, u1s[1][2] - 0.5)   # noisy pair
            u2n = Point2(u2s[1][1] - 0.3, u2s[1][2] + 0.6)
            @test sampson_distance(form, u1n, u2n) == sampson_distance(F, u1n, u2n)
            @test sampson_distance(form, u1n => u2n) == sampson_distance(F, u1n, u2n)
            @test sampson_distance(form, u1n, u2n) > 0
        end

        @testset "depth_transfer rename — behavior pinned, old name retired" begin
            rig = StereoRig(K1, K2, SMatrix{3, 3, Float64}(R), t)
            f = depth_transfer(rig)
            u = Point2(300.0, 220.0)
            # pin the closure's semantics: backproject (unit depth) ∘ extrinsics ∘ project
            expected = project(target(rig).model, extrinsics(rig)(backproject(source(rig).model, u)))
            @test f(u) ≈ expected atol = 1.0e-12
            # three-arg primitive form agrees
            g = depth_transfer(source(rig), target(rig), extrinsics(rig))
            @test g(u) ≈ expected atol = 1.0e-12
            # `epipolar_map` no longer accepts cameras/rigs — it is the form action
            @test_throws MethodError epipolar_map(source(rig), target(rig), extrinsics(rig))
            @test_throws MethodError epipolar_map(rig)
        end

        @testset "ViewFrame / ImageFrame — open hierarchy + StereoRig id minting" begin
            # symbol-parametric instance identity
            @test ViewFrame{:left} <: Frame{3}
            @test ImageFrame{:left} <: Frame{2}
            @test ViewFrame{:left} !== ViewFrame{:right}
            @test ImageFrame{:left} !== ImageFrame{:right}
            # ImageFrame is NOT a PixelFrame: view identity ⟂ origin convention
            @test !(ImageFrame{:colmap} <: PixelFrame)

            # frames participate in the standard typed algebra
            Ra = RotMatrix{3, Float64}(RotXYZ(0.3, -0.4, 0.7))
            m = EuclideanMap(Ra, SVector(0.4, -1.2, 2.5))
            X_ba = relabel(ViewFrame{:a}, ViewFrame{:b}, m)
            @test inv(X_ba) isa EuclideanMap{3, <:Any, <:Any, <:Any, ViewFrame{:b}, ViewFrame{:a}}
            # two different views' frames are a labeled junction mismatch
            X_ca = relabel(ViewFrame{:a}, ViewFrame{:c}, m)
            @test_throws MethodError X_ba ∘ X_ca            # junction: :c meets :a
            @test (relabel(ViewFrame{:c}, ViewFrame{:a}, m) ∘ X_ca) isa
                EuclideanMap{3, <:Any, <:Any, <:Any, ViewFrame{:a}, ViewFrame{:a}}
            # ImageFrame labels the homography lift like any 2D frame
            Hm = @SMatrix [1.1 0.2 5.0; -0.3 0.9 -3.0; 1.0e-4 2.0e-4 1.0]
            @test HomographyMap{ImageFrame{:a}, ImageFrame{:b}}(Hm) isa
                HomographyMap{3, Float64, <:Any, ImageFrame{:a}, ImageFrame{:b}}

            # the hierarchy stays open for downstream slot frames too
            @test LocalFrame{:x} <: Frame{2}
            @test EpipolarForm{LocalFrame{:x}, LocalFrame{:y}}(F) isa
                EpipolarForm{LocalFrame{:x}, LocalFrame{:y}, FundMat{Float64}}

            # StereoRig mints ids — canonical defaults. Since 0.9 `ids` is an
            # ACCESSOR reading the labeled extrinsics' frame parameters.
            rig = StereoRig(K1, K2, SMatrix{3, 3, Float64}(R), t)
            @test ids(rig) == (:cam1, :cam2)
            @test view_frames(rig) == (ViewFrame{:cam1}, ViewFrame{:cam2})
            @test image_frames(rig) == (ImageFrame{:cam1}, ImageFrame{:cam2})
            # custom ids thread through every constructor surface
            rig_lr = StereoRig(K1, K2, SMatrix{3, 3, Float64}(R), t; ids = (:left, :right))
            @test view_frames(rig_lr) == (ViewFrame{:left}, ViewFrame{:right})
            rig2 = StereoRig(source(rig), target(rig); ids = (:src, :tgt))
            @test image_frames(rig2) == (ImageFrame{:src}, ImageFrame{:tgt})
            rig3 = StereoRig(source(rig), target(rig), unlabel(extrinsics(rig)); ids = (:x, :y))
            @test ids(rig3) == (:x, :y)
            # minted frames are ready-made slot labels for the form
            Ca, Cb = image_frames(rig_lr)
            @test EpipolarForm{Ca, Cb}(F) isa EpipolarForm{ImageFrame{:left}, ImageFrame{:right}}
            # gradual: nothing forces identities — rig fields/behavior unchanged
            @test extrinsics(rig) isa EuclideanMap
        end

        @testset "StereoRig frames-native (0.9)" begin
            rig = StereoRig(K1, K2, SMatrix{3, 3, Float64}(R), t; ids = (:left, :right))

            # The stored relative transform is LABELED: frame parameters match the
            # minted ids (monogram X_{V₂V₁} — consumes source-view coordinates,
            # produces target-view coordinates), and `ids` reads them back.
            @test extrinsics(rig) isa
                EuclideanMap{3, Float64, <:Any, <:Any, ViewFrame{:left}, ViewFrame{:right}}
            @test ids(rig) == (:left, :right)
            # direction pinned numerically: X₂ = R·X₁ + t
            for X1 in X1s
                @test extrinsics(rig)(X1) ≈ R * X1 + t atol = 1.0e-12
            end

            # pose(rig): on-demand inverse with SWAPPED frames; the cached twin
            # field and the stored id tuple are gone (single source of truth).
            P = pose(rig)
            @test P isa
                EuclideanMap{3, Float64, <:Any, <:Any, ViewFrame{:right}, ViewFrame{:left}}
            for X1 in X1s
                @test P(extrinsics(rig)(X1)) ≈ X1 atol = 1.0e-9
            end
            @test fieldnames(typeof(rig)) == (:members,)

            # 3-arg constructor consistency check: a transform that disagrees with
            # extrinsics(target) ∘ pose(source) from the stored cameras is refused…
            @test StereoRig(source(rig), target(rig), extrinsics(rig)) isa StereoRig
            bad_t = EuclideanMap(RotMatrix{3, Float64}(R), t + SVector(0.05, 0.0, 0.0))
            @test_throws ArgumentError StereoRig(source(rig), target(rig), bad_t)
            bad_R = EuclideanMap(RotMatrix{3, Float64}(RotXYZ(0.09, -0.15, 0.05)), t)
            @test_throws ArgumentError StereoRig(source(rig), target(rig), bad_R)
            # …and check=false admits it (documented escape for synthetic rigs
            # whose cameras are placeholders).
            loose = StereoRig(source(rig), target(rig), bad_t; check = false, ids = (:a, :b))
            @test extrinsics(loose).t ≈ bad_t.t
            @test ids(loose) == (:a, :b)

            # essential_form: EssentialMat-backed, GnomonicFrame slots (ImageFrame
            # stays the pixel frame), E = [t]×R with slot 1 = source view.
            fe = essential_form(rig)
            @test fe isa
                EpipolarForm{GnomonicFrame{:left}, GnomonicFrame{:right}, EssentialMat{Float64}}
            @test SMatrix{3, 3}(fe.mat) ≈ skew(t) * SMatrix{3, 3}(R) rtol = 1.0e-14
            for X1 in X1s
                x̂₁ = dehom(X1)                # bearing coordinates, view 1
                x̂₂ = dehom(R * X1 + t)        # bearing coordinates, view 2
                @test abs(fe(x̂₁, x̂₂)) < 1.0e-10
            end

            # EpipolarForm(rig): the one-call pixel-frame mint equals the
            # hand-built composition bit-for-bit (identical inputs), and matches
            # FundMat(E, K₁, K₂) numerically.
            fF = EpipolarForm(rig)
            @test fF isa EpipolarForm{ImageFrame{:left}, ImageFrame{:right}, FundMat{Float64}}
            Ka = HomographyMap{GnomonicFrame{:left}, ImageFrame{:left}}(calibration_matrix(source(rig)))
            Kb = HomographyMap{GnomonicFrame{:right}, ImageFrame{:right}}(calibration_matrix(target(rig)))
            hand = EpipolarForm(Kb ∘ epipolar_map(fe) ∘ inv(Ka))
            @test SMatrix{3, 3}(fF.mat) == SMatrix{3, 3}(hand.mat)
            @test SMatrix{3, 3}(fF.mat) ≈ SMatrix{3, 3}(FundMat(E, K1, K2)) rtol = 1.0e-8

            # The minted form's correlation satisfies the incidence / epipole /
            # pencil checks on the synthetic correspondences.
            m = epipolar_map(fF)
            @test m isa EpipolarMap{ImageFrame{:left}, ImageFrame{:right}, FundMat{Float64}}
            e₁, e₂ = epipoles(fF)
            @test norm(SMatrix{3, 3}(fF.mat) * e₁) < 1.0e-10
            @test norm(transpose(SMatrix{3, 3}(fF.mat)) * e₂) < 1.0e-10
            for (u1, u2) in zip(u1s, u2s)
                l = m(u1)
                c = SVector{3}(l.coeffs)
                @test abs(dot(c, to_affine(u2))) / hypot(c[1], c[2]) < 1.0e-6
                @test abs(dot(c, e₂)) / (norm(c) * norm(e₂)) < 1.0e-9
            end

            # depth_transfer and bidirectional transfer survive the new layout
            # (pose is now an accessor).
            u = Point2(300.0, 220.0)
            @test depth_transfer(rig)(u) ≈
                project(target(rig).model, extrinsics(rig)(backproject(source(rig).model, u))) atol = 1.0e-12
            fwd = transfer(rig, Attributed(Point2(u1s[1][1], u1s[1][2]), 4.0))
            @test fwd isa Attributed
        end

        @testset "EssentialMat(::EuclideanMap) shim (0.10 regression)" begin
            # The 0.5-era shim read `.linear`/`.translation` — CT AffineMap's
            # field names, which EuclideanMap never had — so this constructor
            # always threw. Pinned here: E from a rigid map == [t]×R.
            X = EuclideanMap(R, t)                        # unlabeled rigid map
            Em = EssentialMat(X)
            @test Em isa EssentialMat{Float64}
            # ≈, not ==: a different evaluation order (FMA on aarch64) moves the last ulp
            @test SMatrix{3, 3}(Em) ≈ skew(SVector{3, Float64}(t)) * SMatrix{3, 3, Float64, 9}(R) rtol = 8eps()
            @test SMatrix{3, 3}(Em) == SMatrix{3, 3}(E)   # matches EssentialMat(R, t)
            # labeled maps are welcome — frames are irrelevant to the algebra
            Ex = CameraExtrinsics(EuclideanMap(R, t))
            @test SMatrix{3, 3}(EssentialMat(Ex)) == SMatrix{3, 3}(Em)
            Xv = relabel(ViewFrame{:a}, ViewFrame{:b}, EuclideanMap(R, t))
            @test SMatrix{3, 3}(EssentialMat(Xv)) == SMatrix{3, 3}(Em)
            # the duck-typed (R, t) tuple path is unchanged
            @test SMatrix{3, 3}(EssentialMat((R, t))) == SMatrix{3, 3}(Em)
        end
    end
end
