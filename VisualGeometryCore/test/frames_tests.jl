@testitem "frames" begin
    using Test
    using VisualGeometryCore
    using StaticArrays
    using GeometryBasics: GeometryBasics
    using Rotations: RotMatrix, RotXYZ
    using LinearAlgebra: I, normalize

    # =============================================================================
    # Parametric coordinate frames on EuclideanMap (0.6) + HomographyMap (0.7)
    # =============================================================================
    #
    # Covers: frame-checked composition (every row of the wildcard truth table,
    # for BOTH matrix-backed Maps), typed inv round-trip, unlabel/relabel escape
    # hatches, dimension mismatch as a type error, CameraPose/CameraExtrinsics
    # alias behavior (both Camera construction routes), lookat/lookat_pose
    # labeled returns, the labeled Extrinsics3 solver currency (p3p end-to-end),
    # the labeled Mat → Map homography lift, and the PixelFrame origin-conversion
    # frame maps that replaced the ImageOrigin value API in 0.7.

    # A downstream-style frame: the hierarchy is open, no VGC changes needed.
    struct RigFrame <: Frame{3} end

    @testset "Parametric frames on EuclideanMap" begin
        # Asymmetric, non-identity maps so a direction mix-up cannot hide.
        Ra = RotMatrix{3, Float64}(RotXYZ(0.3, -0.4, 0.7))
        Rb = RotMatrix{3, Float64}(RotXYZ(-0.2, 0.5, 0.1))
        ta = SVector(0.4, -1.2, 2.5)
        tb = SVector(-0.7, 0.3, 1.1)
        m_a = EuclideanMap(Ra, ta)   # unlabeled
        m_b = EuclideanMap(Rb, tb)   # unlabeled

        # Labeled versions
        X_CW = relabel(WorldFrame, CameraFrame, m_a)    # extrinsics-shaped
        X_WC = relabel(CameraFrame, WorldFrame, m_b)    # pose-shaped
        X_RC = relabel(CameraFrame, RigFrame, m_b)      # downstream frame

        @testset "constructors default to UnlabeledFrame" begin
            @test m_a isa EuclideanMap{3, Float64, <:Any, <:Any, UnlabeledFrame{3}, UnlabeledFrame{3}}
            # every historical constructor surface stays unlabeled
            @test EuclideanMap(Ra, (1.0, 2.0, 3.0)) isa
                EuclideanMap{3, <:Any, <:Any, <:Any, UnlabeledFrame{3}, UnlabeledFrame{3}}
            @test EuclideanMap(1.0, 0.0, 0.0, 0.0, 1.5, 2.5, 3.0) isa
                EuclideanMap{3, <:Any, <:Any, <:Any, UnlabeledFrame{3}, UnlabeledFrame{3}}
            m2 = EuclideanMap(RotMatrix{2}(pi / 4), SVector(2.0, 1.0))
            @test m2 isa EuclideanMap{2, <:Any, <:Any, <:Any, UnlabeledFrame{2}, UnlabeledFrame{2}}
        end

        @testset "frame-checked composition — labeled/labeled" begin
            # matched junction: pose ∘ extrinsics = World -> World
            r = X_WC ∘ X_CW
            @test r isa EuclideanMap{3, <:Any, <:Any, <:Any, WorldFrame, WorldFrame}
            # numeric result identical to the unlabeled composition
            u = m_b ∘ m_a
            @test r.R == u.R && r.t == u.t
            # matched junction, downstream frame participates
            r2 = X_RC ∘ X_CW   # (World -> Camera) then (Camera -> Rig)
            @test r2 isa EuclideanMap{3, <:Any, <:Any, <:Any, WorldFrame, RigFrame}
            # MISMATCHED labeled junction throws MethodError (the frame check)
            @test_throws MethodError X_CW ∘ X_CW     # Camera meets World
            @test_throws MethodError X_CW ∘ X_RC     # Rig meets World
        end

        @testset "frame-checked composition — wildcard rows" begin
            Unl = UnlabeledFrame{3}
            # row: B.Dst labeled, A.Src unlabeled — unchecked, endpoints kept
            r = m_a ∘ X_CW
            @test r isa EuclideanMap{3, <:Any, <:Any, <:Any, WorldFrame, Unl}
            # row: B.Dst unlabeled, A.Src labeled — unchecked, endpoints kept
            r = X_CW ∘ m_a
            @test r isa EuclideanMap{3, <:Any, <:Any, <:Any, Unl, CameraFrame}
            # row: junction unlabeled on both sides
            r = m_a ∘ m_b
            @test r isa EuclideanMap{3, <:Any, <:Any, <:Any, Unl, Unl}
            # fully-labeled A, fully-unlabeled B (design example):
            # EM{A,B} ∘ EM{Unl,Unl} :: EM{Unl,B}
            r = X_WC ∘ m_a
            @test r isa EuclideanMap{3, <:Any, <:Any, <:Any, Unl, WorldFrame}
            # unlabeled never throws, in any orientation
            @test (m_a ∘ X_WC) isa EuclideanMap
            @test (X_RC ∘ m_a) isa EuclideanMap
        end

        @testset "typed inv round-trip" begin
            inv_X_CW = inv(X_CW)
            @test inv_X_CW isa EuclideanMap{3, <:Any, <:Any, <:Any, CameraFrame, WorldFrame}
            rt = inv(inv_X_CW)
            @test rt isa typeof(X_CW)
            @test rt.R ≈ X_CW.R atol = 1.0e-12
            @test rt.t ≈ X_CW.t atol = 1.0e-12
            # inv ∘ original composes to World -> World (junction checks out)
            idish = X_CW ∘ inv_X_CW
            @test idish isa EuclideanMap{3, <:Any, <:Any, <:Any, CameraFrame, CameraFrame}
            @test idish.t ≈ zero(SVector{3, Float64}) atol = 1.0e-12
            # unlabeled inv stays unlabeled
            @test inv(m_a) isa EuclideanMap{3, <:Any, <:Any, <:Any, UnlabeledFrame{3}, UnlabeledFrame{3}}
        end

        @testset "unlabel / relabel escape hatches" begin
            u = unlabel(X_CW)
            @test u isa EuclideanMap{3, <:Any, <:Any, <:Any, UnlabeledFrame{3}, UnlabeledFrame{3}}
            @test u.R == X_CW.R && u.t == X_CW.t
            # relabel overrides existing labels (deliberate, unchecked)
            r = relabel(RigFrame, WorldFrame, X_CW)
            @test r isa EuclideanMap{3, <:Any, <:Any, <:Any, RigFrame, WorldFrame}
            @test r.R == X_CW.R && r.t == X_CW.t
            # round trip through unlabel/relabel is type-exact
            @test relabel(WorldFrame, CameraFrame, unlabel(X_CW)) === X_CW
        end

        @testset "dimension mismatch is a type error" begin
            # attaching a 2D frame to an N = 3 map: no method
            @test_throws MethodError relabel(TargetPlaneFrame, WorldFrame, m_a)
            @test_throws MethodError relabel(WorldFrame, PixelFrame{:colmap}, m_a)
            # and a 3D frame on an N = 2 map
            m2 = EuclideanMap(RotMatrix{2}(pi / 4), SVector(2.0, 1.0))
            @test_throws MethodError relabel(WorldFrame, WorldFrame, m2)
            # 2D frames attach to 2D maps
            @test relabel(TargetPlaneFrame, PixelFrame{:julia}, m2) isa
                EuclideanMap{2, <:Any, <:Any, <:Any, TargetPlaneFrame, PixelFrame{:julia}}
        end

        @testset "CameraPose / CameraExtrinsics aliases" begin
            e = CameraExtrinsics(m_a)
            p = CameraPose(m_b)
            @test e isa CameraExtrinsics && e isa EuclideanMap
            @test p isa CameraPose && p isa EuclideanMap
            @test !(e isa CameraPose) && !(p isa CameraExtrinsics)
            # typed inv flips pose <-> extrinsics automatically
            @test inv(e) isa CameraPose
            @test inv(p) isa CameraExtrinsics
            # identity fast path: labeling an already-labeled map is a no-op
            @test CameraExtrinsics(e) === e
            @test CameraPose(p) === p
            # direct field access (no wrapper, no .map)
            @test e.R === m_a.R && e.t === m_a.t
            # point application
            x = SVector(0.1, 0.2, 0.3)
            @test e(x) == m_a(x)
        end

        @testset "Camera construction — both labeled routes" begin
            K = SMatrix{3, 3, Float64}(600.0, 0, 0, 0, 600.0, 0, 320.0, 240.0, 1.0)
            model = CameraModel(CameraCalibrationMat(K))
            e = CameraExtrinsics(m_a)
            p = inv(e)
            cam_e = Camera(model, e)        # stores directly
            cam_p = Camera(model, p)        # inverts internally
            @test cam_e.extrinsics === e
            @test extrinsics(cam_e) === e
            @test extrinsics(cam_p).R ≈ e.R atol = 1.0e-12
            @test extrinsics(cam_p).t ≈ e.t atol = 1.0e-12
            # accessors return labeled maps even for unlabeled construction
            cam_u = Camera(model, m_a)      # today's behavior: unlabeled = extrinsics
            @test cam_u.extrinsics === m_a
            @test extrinsics(cam_u) isa CameraExtrinsics
            @test pose(cam_u) isa CameraPose
            @test pose(cam_u).t ≈ inv(m_a).t atol = 1.0e-12
        end

        @testset "lookat / lookat_pose return labeled maps" begin
            eye = SVector(0.3, -0.2, 0.6)
            target = SVector(0.1, 0.05, -1.0)
            up = SVector(0.0, -1.0, 0.0)
            E = lookat(eye, target, up)
            P = lookat_pose(eye, target, up)
            @test E isa CameraExtrinsics
            @test P isa CameraPose
            # frame parity: the two constructions are inverses
            @test P.R ≈ inv(E).R atol = 1.0e-12
            @test P.t ≈ inv(E).t atol = 1.0e-12
            # labeled composition across the pair type-checks
            @test (E ∘ P) isa EuclideanMap{3, <:Any, <:Any, <:Any, CameraFrame, CameraFrame}
            # unit-enforced lookat surface is labeled too
            E_mm = lookat(
                SVector(0.0mm, 0.0mm, 0.0mm), SVector(0.0mm, 0.0mm, 1000.0mm),
                SVector(0.0, -1.0, 0.0)
            )
            @test E_mm isa CameraExtrinsics
        end

        @testset "Extrinsics3 labeled solver currency" begin
            # the concrete alias IS a labeled CameraExtrinsics (Pose3 is gone)
            @test Extrinsics3 <: EuclideanMap
            @test Extrinsics3 <: CameraExtrinsics
            @test isbitstype(Extrinsics3)
            @test !(@isdefined Pose3)
            z = zero(Extrinsics3)
            @test z isa Extrinsics3
            @test z isa EuclideanMap{3, Float64, <:Any, <:Any, WorldFrame, CameraFrame}
            # direct construction through the concrete alias (solver hot path)
            pz = Extrinsics3(Ra, GeometryBasics.Vec{3, Float64}(ta))
            @test pz isa Extrinsics3
            @test pz.R == Ra
            # typed algebra applies: inv of the solver model is a CameraPose,
            # and a matched labeled junction composes
            @test inv(pz) isa CameraPose
            @test (pz ∘ CameraPose(m_b)) isa
                EuclideanMap{3, <:Any, <:Any, <:Any, CameraFrame, CameraFrame}
            # a mismatched labeled junction throws (extrinsics ∘ extrinsics)
            @test_throws MethodError CameraExtrinsics(m_a) ∘ pz
            # p3p returns FixedModels{4, Extrinsics3} — labeled end-to-end — and
            # the ground-truth model reprojects through the labeled type
            X = (
                SVector(0.0, 0.0, 0.0), SVector(1.0, 0.0, 0.0), SVector(0.0, 1.0, 0.0),
            )
            gt = Extrinsics3(
                RotMatrix{3, Float64}(RotXYZ(0.1, -0.2, 0.15)),
                GeometryBasics.Vec{3, Float64}(0.1, -0.2, 4.0)
            )
            rays = map(Xi -> SVector{3, Float64}(normalize(gt(Xi))), X)
            sols = p3p(SVector(rays...), SVector(map(SVector{3, Float64}, X)...))
            @test sols isa FixedModels{4, Extrinsics3}
            @test length(sols) >= 1
            @test any(sols) do s
                s isa Extrinsics3 || return false
                all(zip(rays, X)) do (r, Xi)
                    v = s(SVector{3, Float64}(Xi))
                    isapprox(normalize(v), r; atol = 1.0e-6)
                end
            end
        end
    end

    # A downstream-style 2D frame: the hierarchy is open for HomographyMap too.
    struct FacadeFrame <: Frame{2} end

    @testset "Parametric frames on HomographyMap (0.7)" begin
        # Asymmetric projective matrices so a direction mix-up cannot hide.
        Ha = HomographyMat{Float64}(
            SMatrix{3, 3, Float64}(1.1, 0.2, 1.0e-4, -0.3, 0.9, 2.0e-4, 5.0, -3.0, 1.0)
        )
        Hb = HomographyMat{Float64}(
            SMatrix{3, 3, Float64}(0.8, -0.1, -2.0e-4, 0.4, 1.2, 1.0e-4, -2.0, 7.0, 1.0)
        )
        h_a = HomographyMap(Ha)   # unlabeled lift
        h_b = HomographyMap(Hb)   # unlabeled lift

        # Labeled lifts (frames declared at the Mat → Map boundary)
        H_PB = HomographyMap{TargetPlaneFrame, PixelFrame{:colmap}}(Ha)   # board → pixels
        H_FP = HomographyMap{PixelFrame{:colmap}, FacadeFrame}(Hb)  # pixels → facade

        @testset "constructors default to UnlabeledFrame{2}" begin
            @test h_a isa HomographyMap{3, Float64, <:Any, UnlabeledFrame{2}, UnlabeledFrame{2}}
            # raw-SMatrix lift stays unlabeled too
            @test HomographyMap(SMatrix{3, 3, Float64}(Ha)) isa
                HomographyMap{3, Float64, <:Any, UnlabeledFrame{2}, UnlabeledFrame{2}}
            # matrix-type preservation is unchanged by the frame parameters
            @test h_a isa HomographyMap{3, Float64, HomographyMat{Float64}}
        end

        @testset "labeled Mat -> Map lift" begin
            @test H_PB isa HomographyMap{3, Float64, <:Any, TargetPlaneFrame, PixelFrame{:colmap}}
            @test H_PB isa HomographyMap{3, Float64, HomographyMat{Float64}}  # M preserved
            @test H_PB.forward == h_a.forward && H_PB.inverse == h_a.inverse
            # numeric behavior identical to the unlabeled lift
            p = Point2(0.3, -0.7)
            @test H_PB(p) == h_a(p)
            # element-type conversion keeps the labels and the stored inverse
            H32 = HomographyMap{3, Float32}(H_PB)
            @test H32 isa HomographyMap{3, Float32, SMatrix{3, 3, Float32, 9}, TargetPlaneFrame, PixelFrame{:colmap}}
            @test H32.forward == SMatrix{3, 3, Float32}(H_PB.forward) && H32.inverse == SMatrix{3, 3, Float32}(H_PB.inverse)
            @test convert(HomographyMap{3, Float32}, H_PB) == H32
            # a 3D frame cannot label a 3x3 (2D-frame) map
            @test_throws Exception HomographyMap{WorldFrame, CameraFrame}(Ha)
        end

        @testset "frame-checked composition — labeled/labeled" begin
            # matched junction: (board → pixels) then (pixels → facade)
            r = H_FP ∘ H_PB
            @test r isa HomographyMap{3, <:Any, <:Any, TargetPlaneFrame, FacadeFrame}
            # numeric result identical to the unlabeled composition
            u = h_b ∘ h_a
            @test r.forward == u.forward && r.inverse == u.inverse
            # MISMATCHED labeled junction throws MethodError (the frame check)
            @test_throws MethodError H_PB ∘ H_PB       # pixels meet board
            @test_throws MethodError H_PB ∘ H_FP       # facade meets board
            # mismatched origin TAGS on the same frame family are a mismatch too
            H_j = HomographyMap{PixelFrame{:julia}, FacadeFrame}(Hb)
            @test_throws MethodError H_j ∘ H_PB
        end

        @testset "frame-checked composition — wildcard rows" begin
            Unl = UnlabeledFrame{2}
            # row: B.Dst labeled, A.Src unlabeled — unchecked, endpoints kept
            r = h_b ∘ H_PB
            @test r isa HomographyMap{3, <:Any, <:Any, TargetPlaneFrame, Unl}
            # row: B.Dst unlabeled, A.Src labeled — unchecked, endpoints kept
            r = H_FP ∘ h_a
            @test r isa HomographyMap{3, <:Any, <:Any, Unl, FacadeFrame}
            # row: junction unlabeled on both sides
            r = h_a ∘ h_b
            @test r isa HomographyMap{3, <:Any, <:Any, Unl, Unl}
            # unlabeled never throws, in any orientation
            @test (h_a ∘ H_FP) isa HomographyMap
            @test (H_PB ∘ h_b) isa HomographyMap
        end

        @testset "typed inv round-trip" begin
            ih = inv(H_PB)
            @test ih isa HomographyMap{3, <:Any, <:Any, PixelFrame{:colmap}, TargetPlaneFrame}
            @test inv(ih) === H_PB          # O(1) swap is exactly involutive
            # inv ∘ original composes to board → board (junction checks out)
            idish = ih ∘ H_PB
            @test idish isa HomographyMap{3, <:Any, <:Any, TargetPlaneFrame, TargetPlaneFrame}
            @test SMatrix{3, 3}(idish.forward) * SMatrix{3, 3}(idish.inverse) ≈
                SMatrix{3, 3, Float64}(I) atol = 1.0e-10
            # unlabeled inv stays unlabeled
            @test inv(h_a) isa HomographyMap{3, <:Any, <:Any, UnlabeledFrame{2}, UnlabeledFrame{2}}
        end

        @testset "unlabel / relabel escape hatches" begin
            u = unlabel(H_PB)
            @test u isa HomographyMap{3, <:Any, <:Any, UnlabeledFrame{2}, UnlabeledFrame{2}}
            @test u.forward == H_PB.forward && u.inverse == H_PB.inverse
            r = relabel(FacadeFrame, PixelFrame{:julia}, H_PB)
            @test r isa HomographyMap{3, <:Any, <:Any, FacadeFrame, PixelFrame{:julia}}
            # round trip through unlabel/relabel is type-exact
            @test relabel(TargetPlaneFrame, PixelFrame{:colmap}, unlabel(H_PB)) === H_PB
            # dimension guard: 3D frames cannot label a 3x3 map
            @test_throws Exception relabel(WorldFrame, CameraFrame, h_a)
        end

        @testset "labeled EuclideanMap{2} rides the composition bridge" begin
            T = EuclideanMap(PixelFrame{:colmap}, PixelFrame{:julia})
            # matched junction across the bridge: pixels(:colmap) meet pixels(:colmap)
            r = T ∘ H_PB
            @test r isa HomographyMap{3, <:Any, <:Any, TargetPlaneFrame, PixelFrame{:julia}}
            # mismatched junction across the bridge throws
            @test_throws MethodError EuclideanMap(PixelFrame{:julia}, PixelFrame{:colmap}) ∘ H_PB
            # unlabeled EuclideanMap still bridges as a wildcard
            e2 = EuclideanMap(RotMatrix{2}(pi / 6), SVector(0.5, -0.25))
            @test (e2 ∘ H_PB) isa HomographyMap{3, <:Any, <:Any, TargetPlaneFrame, UnlabeledFrame{2}}
        end
    end

    @testset "PixelFrame origin-conversion frame maps (0.7)" begin
        T_cj = EuclideanMap(PixelFrame{:colmap}, PixelFrame{:julia})
        T_jc = EuclideanMap(PixelFrame{:julia}, PixelFrame{:colmap})

        @testset "both directions + round trip" begin
            @test T_cj isa EuclideanMap{2, <:Any, <:Any, <:Any, PixelFrame{:colmap}, PixelFrame{:julia}}
            @test T_jc isa EuclideanMap{2, <:Any, <:Any, <:Any, PixelFrame{:julia}, PixelFrame{:colmap}}
            @test T_cj.t == SVector(0.5, 0.5)
            @test T_jc.t == SVector(-0.5, -0.5)
            p = Point2(3.25, -1.5)
            @test T_cj(p) == Point2(3.75, -1.0)
            @test T_jc(T_cj(p)) == p
            # typed inverse IS the reverse frame map
            @test inv(T_cj) isa typeof(T_jc)
            @test inv(T_cj).t == T_jc.t
            # frame-checked round-trip composition: julia → julia
            rt = T_cj ∘ T_jc
            @test rt isa EuclideanMap{2, <:Any, <:Any, <:Any, PixelFrame{:julia}, PixelFrame{:julia}}
            @test rt.t == SVector(0.0, 0.0)
            # the wrong pairing is a frame mismatch
            @test_throws MethodError T_cj ∘ T_cj
        end

        @testset "identity, :opencv, and unknown origins" begin
            @test EuclideanMap(PixelFrame{:julia}, PixelFrame{:julia}).t == SVector(0.0, 0.0)
            @test EuclideanMap(PixelFrame{:opencv}, PixelFrame{:colmap}).t == SVector(0.5, 0.5)
            @test_throws Exception EuclideanMap(PixelFrame{:vlfeat}, PixelFrame{:julia})
            # the ImageOrigin value API is gone
            @test !(@isdefined COLMAP_ORIGIN)
            @test !(@isdefined JULIA_ORIGIN)
        end
    end

    @testset "CameraModel image_origin regression (0.7.0 leftover)" begin
        # 0.7.0 removed Translation(::Symbol, ::Symbol) but CameraModel's
        # image_origin path still spelled it — silently matching CT's generic
        # Translation(x, y) and building a Translation{SVector{2,Symbol}} that
        # exploded on first use. Pin the fixed path: eager AffineMap fusion and
        # numerically correct principal-point shifts for both non-colmap origins.
        K = CameraCalibrationMat(@SMatrix [800.0 0.0 320.0; 0.0 800.0 240.0; 0.0 0.0 1.0])
        for origin in (:opencv, :julia)
            model = CameraModel(K; image_origin = origin)
            @test model.intrinsics isa LogicalIntrinsics   # fused to a plain affine — no lazy Symbol carrier
            u = project(model, SVector(0.1, -0.2, 2.0))
            @test all(isfinite, u)
        end
        # The shift must equal the frame-map translation, not a Symbol pair.
        shift = EuclideanMap(PixelFrame{:opencv}, PixelFrame{:colmap}).t
        @test shift isa AbstractVector{Float64}
    end

    @testset "IsotropicScale trait arms regression (0.14.0)" begin
        # The trait refactor (3dd507e) shipped characteristic_scale/aspect_ratio
        # dispatching on frame_scale(f) with ONLY the ShapeMatrixScale arm, so
        # both threw MethodError for Disc/OrientedDisc. σ IS the isotropic
        # characteristic scale; the aspect ratio of a disc is 1 by definition.
        d = Disc(GeometryBasics.Point2(3.0, 4.0), 2.5)
        od = OrientedDisc(GeometryBasics.Point2(3.0, 4.0), 2.5, π / 6)
        @test characteristic_scale(d) == 2.5
        @test characteristic_scale(od) == 2.5
        @test aspect_ratio(d) == 1.0
        @test aspect_ratio(od) == 1.0
        # Consistency: an isotropic frame lifted to the shape-matrix
        # representation reports the same scale through the other trait arm.
        ef = EllipticFrame(AffineFrame(d, 0.0))
        @test characteristic_scale(ef) ≈ 2.5
    end
end

# One symmetric square root for every frame constructor (2026-09-17):
# eigenvalues clamped at zero, exactly symmetric.
@testitem "frames: symmetric square root" begin
    using Test
    using VisualGeometryCore
    using VisualGeometryCore: EllipticFrame, Ellipse, Point2
    using StaticArrays: SMatrix

    Σ = SMatrix{2, 2}(4.0, 1.0, 1.0, 2.0)
    S = VisualGeometryCore.Primitives._sym_sqrt(Σ)
    @test S * S ≈ Σ atol = 1.0e-12
    @test S[1, 2] == S[2, 1]
    # a rank-one PSD matrix that rounds a hair negative: clamped, not NaN
    v = SMatrix{2, 1}(0.6, 0.8)
    Σ1 = SMatrix{2, 2}(v * v') - 1.0e-17 * SMatrix{2, 2}(0.8^2, -0.48, -0.48, 0.6^2)
    S1 = VisualGeometryCore.Primitives._sym_sqrt(Σ1)
    @test all(isfinite, S1)
    @test S1 * S1 ≈ Σ1 atol = 1.0e-8
    # every constructor goes through it: an ellipse's frame reproduces its moment matrix
    e = Ellipse(Point2(1.0, 2.0), 3.0, 1.5, 0.4)
    f = EllipticFrame(e)
    @test f.S * f.S ≈ VisualGeometryCore.moment_matrix(e) atol = 1.0e-12
    @test EllipticFrame(Point2(1.0, 2.0), Σ, Val(:moment_matrix)).S == S
    # genuinely indefinite (a paraboloid that is not a peak): refused, not clamped
    @test_throws ArgumentError VisualGeometryCore.Primitives._sym_sqrt(SMatrix{2, 2}(-1.0, 0.0, 0.0, 1.0))
end
