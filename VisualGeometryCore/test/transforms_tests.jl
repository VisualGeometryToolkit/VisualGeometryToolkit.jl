#!/usr/bin/env julia

@testitem "transforms" begin
    using Test
    using VisualGeometryCore
    using StaticArrays
    using CoordinateTransformations
    using Rotations
    using LinearAlgebra

    @testset "EuclideanMap(...; check = true) checks the rotation in every dimension" begin
        # N = 2 used to skip the check entirely
        R2bad = RotMatrix{2}(SMatrix{2, 2}(2.0, 0.0, 0.0, 2.0))
        @test_throws ArgumentError EuclideanMap(R2bad, SVector(0.0, 0.0))
        @test EuclideanMap(R2bad, SVector(0.0, 0.0); check = false) isa EuclideanMap{2}
        @test EuclideanMap(RotMatrix{2}(0.3), SVector(1.0, 2.0)) isa EuclideanMap{2}
        R3bad = RotMatrix{3}(SMatrix{3, 3}(2.0, 0, 0, 0, 1, 0, 0, 0, 1))
        @test_throws ArgumentError EuclideanMap(R3bad, SVector(0.0, 0.0, 0.0))
    end

    @testset "transform_deriv of a HomographyMap; π ∘ H through the chain rule" begin
        using ForwardDiff: ForwardDiff
        C = SMatrix{3, 3, Float64}(0.9, 0.1, 1.0e-3, -0.2, 1.1, -2.0e-3, 5.0, -3.0, 1.0)
        h = HomographyMap(C)
        u = SVector(12.0, -7.5)
        @test transform_deriv(h, u) == C[:, SOneTo(2)]                  # the action on ũ is linear
        @test transform_deriv(h, SVector(12.0, -7.5, 1.0)) == C
        m = PerspectiveMap() ∘ h
        @test m(u) ≈ SVector((C * SVector(u[1], u[2], 1.0))[1:2]...) / (C * SVector(u[1], u[2], 1.0))[3]
        @test transform_deriv(m, u) ≈ ForwardDiff.jacobian(x -> m(x), u) rtol = 1.0e-13
    end

    @testset "Transform Operations" begin

        @testset "Transform operator syntax" begin
            # Create test ellipse (HomographyMap transforms primitives, not matrices)
            ellipse = Ellipse(Point2(0.0, 0.0), 2.0, 1.0, 0.0)

            # Test individual transforms
            scale = to_homogeneous(LinearMap(@SMatrix [2.0 0.0; 0.0 1.5]))
            rotation = to_homogeneous(RotMatrix{2}(π / 6))
            translation = to_homogeneous(Translation(SVector(1.0, 2.0)))

            # Wrap in HomographyMap for callable syntax (preserves typed matrices from to_homogeneous)
            H_scale = HomographyMap(scale)
            H_rotation = HomographyMap(rotation)
            H_translation = HomographyMap(translation)

            # Test transform application on ellipse primitives
            # Anisotropic scale returns conic matrix; convert back to Ellipse between steps
            Q_scaled = H_scale(ellipse)
            @test Q_scaled isa HomEllipseMat{Float64}
            ellipse_scaled = Ellipse(Q_scaled)

            Q_rotated = H_rotation(ellipse_scaled)
            ellipse_rotated = Ellipse(Q_rotated)

            Q_translated = H_translation(ellipse_rotated)
            ellipse_translated = Ellipse(Q_translated)

            # Test composition via HomographyMap ∘
            H_composed = H_translation ∘ H_rotation ∘ H_scale
            Q_composed = H_composed(ellipse)
            ellipse_composed = Ellipse(Q_composed)

            # Verify step-by-step and composed results match
            ellipse_sequential = ellipse_translated

            @test ellipse_sequential.center ≈ ellipse_composed.center atol = 1.0e-12
            @test ellipse_sequential.a ≈ ellipse_composed.a atol = 1.0e-12
            @test ellipse_sequential.b ≈ ellipse_composed.b atol = 1.0e-12
            @test ellipse_sequential.θ ≈ ellipse_composed.θ atol = 1.0e-12
        end

        @testset "Transform matrix construction" begin
            # Test rotation matrix
            θ = π / 4
            rot = RotMatrix{2}(θ)
            hom_rot = to_homogeneous(rot)
            @test hom_rot isa HomRotationMat{Float64}
            @test size(hom_rot) == (3, 3)
            @test hom_rot[3, 3] == 1.0
            @test hom_rot[1:2, 1:2] ≈ [cos(θ) -sin(θ); sin(θ) cos(θ)]

            # Test translation matrix
            t = SVector(3.0, -2.0)
            trans = Translation(t)
            hom_trans = to_homogeneous(trans)
            @test hom_trans isa HomTranslationMat{Float64}
            @test size(hom_trans) == (3, 3)
            @test hom_trans[1:2, 1:2] ≈ I(2)
            @test hom_trans[1:2, 3] ≈ t

            # Test scale matrix
            s = SVector(2.0, 0.5)
            scale = LinearMap(Diagonal(s))
            hom_scale = to_homogeneous(scale)
            @test hom_scale isa HomScalingMat{Float64}
            @test size(hom_scale) == (3, 3)
            @test hom_scale[1:2, 1:2] ≈ Diagonal(s)

            # Test isotropic scale
            s_iso = 1.5
            scale_iso = LinearMap(s_iso * I(2))
            hom_scale_iso = to_homogeneous(scale_iso)
            @test hom_scale_iso isa HomUniformScalingMat{Float64}
        end

        @testset "Transform composition and materialization" begin
            # Create transforms
            rot = to_homogeneous(RotMatrix{2}(π / 3))
            trans = to_homogeneous(Translation(SVector(1.0, -1.0)))
            scale = to_homogeneous(LinearMap(Diagonal(SVector(2.0, 0.8))))

            # Test multiplication
            euclidean = rot * trans
            @test euclidean isa HomEuclideanMat{Float64}

            affine = scale * euclidean
            @test affine isa HomAffineMat{Float64}

            # Test to_homogeneous (materialization)
            mat_euclidean = to_homogeneous(euclidean)
            @test mat_euclidean isa HomEuclideanMat{Float64}
            @test size(mat_euclidean) == (3, 3)

            mat_affine = to_homogeneous(affine)
            @test mat_affine isa HomAffineMat{Float64}
            @test size(mat_affine) == (3, 3)

            # Verify homogeneous matrices work correctly
            test_point = [1.0, 1.0, 1.0]
            result1 = mat_euclidean * test_point
            # Note: Transform objects only work on conics, not points directly
            # So we'll just verify the matrix is correct size and type
        end

        @testset "Result type system" begin
            # Composition result types come from `Base.promote_type` over the
            # AbstractHomMat hierarchy.  Bare-class membership (`<:`) checks the
            # category; concrete type equality also works for the Float64 case.
            rot = to_homogeneous(RotMatrix{2}(π / 4))
            trans = to_homogeneous(Translation(SVector(1.0, 1.0)))
            scale_iso = to_homogeneous(LinearMap(2.0 * I(2)))
            scale_aniso = to_homogeneous(LinearMap(Diagonal(SVector(2.0, 0.5))))

            # Euclidean combinations
            @test promote_type(typeof(rot), typeof(trans)) <: HomEuclideanMat
            @test promote_type(typeof(trans), typeof(rot)) <: HomEuclideanMat

            # Similarity combinations (uniform scaling preserves similarity)
            @test promote_type(typeof(scale_iso), typeof(rot)) <: HomSimilarityMat
            @test promote_type(typeof(rot), typeof(scale_iso)) <: HomSimilarityMat
            @test promote_type(typeof(scale_iso), typeof(trans)) <: HomSimilarityMat
            @test promote_type(typeof(trans), typeof(scale_iso)) <: HomSimilarityMat

            # Anisotropic scaling breaks similarity → Affine
            @test promote_type(typeof(scale_aniso), typeof(rot)) <: HomAffineMat
            @test promote_type(typeof(scale_aniso), typeof(trans)) <: HomAffineMat
            @test promote_type(typeof(rot), typeof(scale_aniso)) <: HomAffineMat
            @test promote_type(typeof(trans), typeof(scale_aniso)) <: HomAffineMat

            # Test actual multiplication results
            @test (rot * trans) isa HomEuclideanMat{Float64}
            @test (trans * rot) isa HomEuclideanMat{Float64}
            @test (scale_iso * rot) isa HomSimilarityMat{Float64}
            @test (rot * scale_iso) isa HomSimilarityMat{Float64}
            @test (scale_iso * trans) isa HomSimilarityMat{Float64}
            @test (trans * scale_iso) isa HomSimilarityMat{Float64}
            @test (scale_aniso * rot) isa HomAffineMat{Float64}
            @test (scale_aniso * trans) isa HomAffineMat{Float64}
        end

        @testset "Circle transformations" begin
            # Test circle to conic conversion and transformation
            circle = Circle(Point2(1.0, 1.0), 1.5)
            Q_circle = HomCircleMat(circle)
            @test Q_circle isa HomCircleMat{Float64}
            @test is_ellipse(Q_circle)

            # Apply diagonal (non-uniform) scaling - will transform circle to ellipse
            scale = to_homogeneous(LinearMap(Diagonal(SVector(2.0, 1.0))))
            H_scale = HomographyMap(scale)

            # Transform the circle primitive directly
            # Anisotropic scaling returns HomEllipseMat (conic matrix), convert to Ellipse
            conic_result = H_scale(circle)
            @test conic_result isa HomEllipseMat{Float64}
            ellipse_result = Ellipse(conic_result)

            # Verify the transformation worked correctly
            @test ellipse_result.center ≈ Point2(2.0, 1.0) atol = 1.0e-12  # x-coordinate scaled by 2
            @test ellipse_result.a ≈ 3.0 atol = 1.0e-12  # radius 1.5 scaled by 2
            @test ellipse_result.b ≈ 1.5 atol = 1.0e-12  # radius 1.5 unchanged in y

            # Test uniform scaling - circle should remain a circle (with HomSimilarityMat)
            iso_scale_mat = HomSimilarityMat{Float64}(Tuple(@SMatrix [2.0 0.0 0.0; 0.0 2.0 0.0; 0.0 0.0 1.0]))
            H_iso_scale = HomographyMap(iso_scale_mat)
            circle_scaled = H_iso_scale(circle)
            @test circle_scaled isa Circle{Float64}  # Similarity preserves circles
            @test circle_scaled.center ≈ Point2(2.0, 2.0) atol = 1.0e-12  # center scaled by 2
            @test circle_scaled.r ≈ 3.0 atol = 1.0e-12  # radius 1.5 scaled by 2
        end

        @testset "CT type contravariant transforms" begin
            # Test primitives
            circle = Circle(Point2(3.0, 4.0), 2.0)
            ellipse = Ellipse(Point2(1.0, 2.0), 5.0, 3.0, π / 6)
            line = Line2D(1.0, -1.0, 2.0)

            @testset "Translation callable" begin
                t = Translation(SVector(1.0, -2.0))

                # Circle
                c′ = t(circle)
                @test c′ isa Circle
                @test centroid(c′) ≈ Point2(4.0, 2.0) atol = 1.0e-12
                @test c′.r ≈ 2.0 atol = 1.0e-12

                # Ellipse
                e′ = t(ellipse)
                @test e′ isa Ellipse
                @test e′.center ≈ Point2(2.0, 0.0) atol = 1.0e-12
                @test e′.a ≈ ellipse.a atol = 1.0e-12
                @test e′.b ≈ ellipse.b atol = 1.0e-12
                @test e′.θ ≈ ellipse.θ atol = 1.0e-12

                # Line2D — verify via HomographyMap reference
                l′ = t(line)
                @test l′ isa Line2D
                H_ref = HomographyMap(to_homogeneous(t))
                l′_ref = H_ref(line)
                @test l′.coeffs ≈ l′_ref.coeffs atol = 1.0e-12
            end

            @testset "+ operator (translation shorthand)" begin
                v = SVector(1.0, -2.0)

                # Circle translation goes through the explicit map (the + sugar
                # on GeometryBasics.Circle was piracy and was removed)
                c1 = Translation(v)(circle)
                @test centroid(c1) ≈ centroid(circle) + v atol = 1.0e-12
                @test c1.r ≈ circle.r atol = 1.0e-12

                # Ellipse + vector
                e1 = ellipse + v
                e2 = v + ellipse
                @test e1.center ≈ e2.center atol = 1.0e-12
                @test e1.a ≈ ellipse.a atol = 1.0e-12

                # Line2D + vector
                l1 = line + v
                l2 = v + line
                @test l1.coeffs ≈ l2.coeffs atol = 1.0e-12
            end

            @testset "LinearMap on Circle" begin
                # Uniform scaling → Circle
                lm_iso = LinearMap(UniformScaling(2.0))
                c′ = lm_iso(circle)
                @test c′ isa Circle
                @test centroid(c′) ≈ Point2(6.0, 8.0) atol = 1.0e-12
                @test c′.r ≈ 4.0 atol = 1.0e-12

                # Rotation → Circle
                lm_rot = LinearMap(RotMatrix{2}(π / 4))
                c′_rot = lm_rot(circle)
                @test c′_rot isa Circle
                @test c′_rot.r ≈ 2.0 atol = 1.0e-12

                # Anisotropic scaling → HomEllipseMat
                lm_aniso = LinearMap(Diagonal(SVector(2.0, 0.5)))
                c′_aniso = lm_aniso(circle)
                @test c′_aniso isa HomEllipseMat

                # Verify against HomographyMap
                H_ref = HomographyMap(to_homogeneous(lm_aniso))
                c′_ref = H_ref(circle)
                @test SMatrix{3, 3}(c′_aniso) ≈ SMatrix{3, 3}(c′_ref) atol = 1.0e-12
            end

            @testset "LinearMap on Ellipse" begin
                lm = LinearMap(Diagonal(SVector(2.0, 0.5)))
                e′ = lm(ellipse)
                @test e′ isa HomEllipseMat

                # Verify against HomographyMap
                H_ref = HomographyMap(to_homogeneous(lm))
                e′_ref = H_ref(ellipse)
                @test SMatrix{3, 3}(e′) ≈ SMatrix{3, 3}(e′_ref) atol = 1.0e-12
            end

            @testset "AffineMap on Circle and Ellipse" begin
                M = @SMatrix [2.0 0.5; -0.3 1.5]
                t = SVector(1.0, -1.0)
                am = AffineMap(M, t)

                # Circle
                c′ = am(circle)
                @test c′ isa HomEllipseMat
                H_ref = HomographyMap(to_homogeneous(am))
                c′_ref = H_ref(circle)
                @test SMatrix{3, 3}(c′) ≈ SMatrix{3, 3}(c′_ref) atol = 1.0e-12

                # Ellipse
                e′ = am(ellipse)
                @test e′ isa HomEllipseMat
                e′_ref = H_ref(ellipse)
                @test SMatrix{3, 3}(e′) ≈ SMatrix{3, 3}(e′_ref) atol = 1.0e-12
            end

            @testset "LinearMap and AffineMap on Line2D" begin
                # LinearMap
                lm = LinearMap(Diagonal(SVector(2.0, 0.5)))
                l′ = lm(line)
                @test l′ isa Line2D
                H_ref = HomographyMap(to_homogeneous(lm))
                l′_ref = H_ref(line)
                @test l′.coeffs ≈ l′_ref.coeffs atol = 1.0e-12

                # AffineMap
                M = @SMatrix [2.0 0.5; -0.3 1.5]
                t = SVector(1.0, -1.0)
                am = AffineMap(M, t)
                l′_am = am(line)
                @test l′_am isa Line2D
                H_ref_am = HomographyMap(to_homogeneous(am))
                l′_ref_am = H_ref_am(line)
                @test l′_am.coeffs ≈ l′_ref_am.coeffs atol = 1.0e-12
            end

            @testset "circle preservation under CT linear maps (was transform_trait)" begin
                c = Circle(Point2(1.0, 2.0), 3.0)
                # Circle-preserving: rotation and uniform scale stay Circle
                @test LinearMap(RotMatrix{2}(0.5))(c) isa Circle
                @test LinearMap(UniformScaling(2.0))(c) isa Circle
                # Circle-breaking: anisotropic/general linear demote to a conic
                @test !(LinearMap(Diagonal(SVector(2.0, 0.5)))(c) isa Circle)
                @test !(LinearMap(SMatrix{2, 2}(1.0, 0.0, 0.5, 1.0))(c) isa Circle)
            end
        end
    end
end

@testitem "EuclideanMap inv keeps the rotation's representation" begin
    using VisualGeometryCore, StaticArrays, Rotations
    q = QuatRotation(0.9, 0.1, -0.3, 0.2)            # normalized by the constructor
    t = SVector(1.0, -2.0, 3.0)
    E = relabel(CameraFrame, WorldFrame, EuclideanMap{3, Float64, typeof(q), typeof(t)}(q, t))
    Ei = inv(E)
    @test Ei.R isa QuatRotation{Float64}
    @test Ei.R == inv(q)
    @test Ei isa EuclideanMap{3, Float64, <:QuatRotation, <:Any, WorldFrame, CameraFrame}
    @test Ei.t == -(inv(q) * t)
    @test inv(Ei).R == q                     # a conjugate twice is the quaternion
    # a RotMatrix map inverts exactly as before: the transpose, its type kept
    M = EuclideanMap(RotMatrix(q), t)
    Mi = inv(M)
    @test typeof(Mi) == typeof(M)
    @test Mi.R == RotMatrix(q)'
    @test Mi.t == -(RotMatrix(q)' * t)
    x = SVector(0.3, 0.4, -0.5)
    @test Ei(E(x)) ≈ x atol = 1.0e-12
    @test @inferred(inv(E)) isa EuclideanMap
end
