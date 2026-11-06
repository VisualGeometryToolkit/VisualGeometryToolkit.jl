#!/usr/bin/env julia

@testitem "transforms" begin
    using Test
    using VisualGeometryToolkitCore
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

            # Lift each map to a HomographyMap: projective in meaning, a Conic out
            H_scale = HomographyMap(LinearMap(@SMatrix [2.0 0.0; 0.0 1.5]))
            H_rotation = HomographyMap(LinearMap(RotMatrix{2}(π / 6)))
            H_translation = HomographyMap(Translation(SVector(1.0, 2.0)))

            # Test transform application on ellipse primitives; convert back to Ellipse between steps
            Q_scaled = H_scale(ellipse)
            @test Q_scaled isa Conic{Float64}
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
            hom_rot = to_homogeneous(LinearMap(rot))
            @test hom_rot isa SMatrix{3, 3, Float64, 9}
            @test hom_rot == to_homogeneous(rot)
            @test hom_rot[3, 3] == 1.0
            @test hom_rot[1:2, 1:2] ≈ [cos(θ) -sin(θ); sin(θ) cos(θ)]

            # Test translation matrix
            t = SVector(3.0, -2.0)
            trans = Translation(t)
            hom_trans = to_homogeneous(trans)
            @test hom_trans isa SMatrix{3, 3, Float64, 9}
            @test hom_trans[1:2, 1:2] ≈ I(2)
            @test hom_trans[1:2, 3] ≈ t

            # Test scale matrix
            s = SVector(2.0, 0.5)
            scale = LinearMap(Diagonal(s))
            hom_scale = to_homogeneous(scale)
            @test hom_scale isa SMatrix{3, 3, Float64, 9}
            @test hom_scale[1:2, 1:2] ≈ Diagonal(s)

            # Test isotropic scale, and the 3-D maps
            hom_scale_iso = to_homogeneous(LinearMap(1.5 * I))
            @test hom_scale_iso == SMatrix{3, 3}(1.5, 0, 0, 0, 1.5, 0, 0, 0, 1)
            E3 = EuclideanMap(RotXYZ(0.1, 0.2, 0.3), SVector(1.0, 2.0, 3.0))
            @test to_homogeneous(E3) isa SMatrix{4, 4, Float64, 16}
            @test to_homogeneous(E3)[1:3, 4] == E3.t && to_homogeneous(E3)[4, :] == SVector(0.0, 0.0, 0.0, 1.0)
        end

        @testset "Transform composition: the class is the map's" begin
            rot = LinearMap(RotMatrix{2}(π / 3))
            trans = Translation(SVector(1.0, -1.0))
            scale = LinearMap(Diagonal(SVector(2.0, 0.8)))

            # the matrix of a composition is the product of the matrices
            @test to_homogeneous(rot ∘ trans) ≈ to_homogeneous(rot) * to_homogeneous(trans)
            @test to_homogeneous(scale ∘ rot ∘ trans) ≈ to_homogeneous(scale) * to_homogeneous(rot) * to_homogeneous(trans)
            # rigid ∘ rigid is rigid, similarity ∘ rigid a similarity
            E = EuclideanMap(RotMatrix{2}(π / 3), SVector(0.0, 0.0)) ∘ EuclideanMap(one(RotMatrix{2, Float64}), SVector(1.0, -1.0))
            @test E isa EuclideanMap{2}
            @test to_homogeneous(E) ≈ to_homogeneous(rot) * to_homogeneous(trans)
            @test SimilarityMap(LinearMap(2.0I)) ∘ E isa SimilarityMap{2}
            # a homography after a structured map is a homography, SMatrix-backed
            @test HomographyMap(HomographyMat(SMatrix{3, 3}(1.0, 0.1, 1.0e-3, 0.0, 1.0, 0.0, 2.0, 1.0, 1.0))) ∘ E isa
                HomographyMap{3, Float64, SMatrix{3, 3, Float64, 9}}
        end

        @testset "Circle transformations" begin
            # Test circle to conic conversion and transformation
            circle = Circle(Point2(1.0, 1.0), 1.5)
            Q_circle = HomCircleMat(circle)
            @test Q_circle isa HomCircleMat{Float64}
            @test is_ellipse(Q_circle)

            # Apply diagonal (non-uniform) scaling - will transform circle to ellipse
            lm = LinearMap(Diagonal(SVector(2.0, 1.0)))
            conic_result = lm(circle)                     # the affine map itself: an ellipse's matrix
            @test conic_result isa HomEllipseMat{Float64}
            @test HomographyMap(lm)(circle) isa Conic{Float64}   # its homography lift: projective in meaning
            ellipse_result = Ellipse(conic_result)

            # Verify the transformation worked correctly
            @test ellipse_result.center ≈ Point2(2.0, 1.0) atol = 1.0e-12  # x-coordinate scaled by 2
            @test ellipse_result.a ≈ 3.0 atol = 1.0e-12  # radius 1.5 scaled by 2
            @test ellipse_result.b ≈ 1.5 atol = 1.0e-12  # radius 1.5 unchanged in y

            # Uniform scaling - a similarity keeps a circle a circle
            circle_scaled = SimilarityMap(LinearMap(2.0I))(circle)
            @test circle_scaled isa Circle{Float64}
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
    using VisualGeometryToolkitCore, StaticArrays, Rotations
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
