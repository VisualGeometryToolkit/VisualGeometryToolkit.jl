# =============================================================================
# shape_transfer_tests.jl — the warping paradigm's two new pieces:
#   * the full-camera reprojection constraint in RAW pixels for any lens
#     (`constraint_with_jacobians(model, pose, X, u)`), built on the chain rule
#     through `imaging_transform` = `AffineMap(K) ∘ lens` (`transform_deriv(lens, X)`
#     on the ray is its leaf; `transform_deriv(::PerspectiveMap)` is kept for
#     homography chains);
#   * the ellipse-PRESERVING transfer of a plane conic through any map
#     (`transfer_ellipse(::Transformation, _)`: exact for a homography, the
#     local affinity elsewhere, stage by stage through compositions) and its
#     camera form (`transfer_ellipse(::PosedCamera, _)`).
# =============================================================================
@testitem "shape_transfer" begin
    import VisualGeometryToolkitCore as VgtkCore
    using Test
    using ForwardDiff: ForwardDiff
    using FiniteDiff: FiniteDiff
    using LinearAlgebra: norm, Symmetric, I
    using StaticArrays: SVector, SMatrix
    using Rotations: RotXYZ, RotMatrix, RotationVec
    using CoordinateTransformations: PerspectiveMap, AffineMap, Translation
    using VisualGeometryToolkitCore
    using VisualGeometryToolkitCore: reprojection_error, transform_deriv, transfer_ellipse, constraint_with_jacobians, imaging_transform,
        distortion_map, pinhole, moment_matrix, boundary_points, backproject, project, skew

    const _K = CameraCalibrationMat(950.0px, [2120.0px, 1416.0px])
    const _CAMS = (
        "pinhole" => CameraModel(LogicalIntrinsics(_K), Pinhole()),
        "division2" => CameraModel(LogicalIntrinsics(_K), Division(-0.3, -0.08)),
        "brown_conrady" => CameraModel(LogicalIntrinsics(_K), Lens(Perspective(), Additive(Radial((-0.2, 0.05, -0.01), (0.03, -0.004, 0.001)), Tangential(0.002, -0.001)))),
        "kannala_brandt" => CameraModel(LogicalIntrinsics(_K), KannalaBrandt(-0.02, 0.003, -0.0004, 0.0001)),
    )
    const _R = RotMatrix(RotXYZ(0.3, -0.2, 0.15))
    const _t = SVector(40.0, -30.0, 600.0)
    const _POSE = EuclideanMap(_R, _t)

    @testset "shape transfer + full-camera reprojection" begin
        @testset "transform_deriv(PerspectiveMap) is the perspective Jacobian" begin
            X = SVector(0.3, -0.2, 2.5)
            J = transform_deriv(PerspectiveMap(), X)
            Jn = ForwardDiff.jacobian(PerspectiveMap(), X)
            @test size(J) == (2, 3) && maximum(abs.(J - Jn)) < 1.0e-8
        end

        @testset "chain rule through imaging_transform — $name" for (name, cam) in _CAMS
            proj = imaging_transform(cam)
            X = SVector(-0.35, 0.22, 1.0) * 600.0                              # ~20° off-axis
            J = transform_deriv(proj, X)
            Jn = FiniteDiff.finite_difference_jacobian(x -> (p = proj(x); SVector(Float64(p[1]), Float64(p[2]))), X, Val(:central))
            @test size(J) == (2, 3) && maximum(abs.(J - Jn)) < 1.0e-5 * max(1.0, maximum(abs.(Jn)))
        end

        @testset "constraint_with_jacobians(model, pose, X, u) — $name" for (name, cam) in _CAMS
            X = SVector(60.0, -45.0, 0.0)
            u_true = SVector{2, Float64}(project(cam, _R * X + _t))
            u = u_true + SVector(0.7, -0.4)                                     # an observation 0.8 px off
            cd = constraint_with_jacobians(cam, _POSE, X, u)
            # the triple every constraint_with_jacobians returns — a 2-way destructuring
            # `e, Gθ = …` would silently bind G_x to Gθ
            @test cd isa Tuple{SVector{2, Float64}, SMatrix{2, 2, Float64, 4}, SMatrix{2, 6, Float64, 12}}
            e, G_x, Gθ = cd
            @test G_x == -SMatrix{2, 2, Float64}(I)                              # ∂e/∂u: the pixel is the measurement
            @test e ≈ -SVector(0.7, -0.4) atol = 1.0e-9                         # e = projection − observation
            # left perturbation: (R, t) ← (ΔR·R, ΔR·t + δt), ΔR = exp([δω]×)
            function e_of(δ)
                ΔR = RotMatrix(RotationVec(δ[1], δ[2], δ[3]))
                p = EuclideanMap(ΔR * _R, ΔR * _t + SVector(δ[4], δ[5], δ[6]))
                return reprojection_error(cam, p, X, u)
            end
            Gn = FiniteDiff.finite_difference_jacobian(e_of, zeros(SVector{6, Float64}), Val(:central))
            @test size(Gθ) == (2, 6) && maximum(abs.(Gθ - Gn)) < 1.0e-4 * max(1.0, maximum(abs.(Gn)))
            if cam.lens isa Pinhole
                # exactly the calibrated form scaled by the pixel block A = K[1:2,1:2]
                g, _, G = constraint_with_jacobians(_POSE, X, SVector{3, Float64}(backproject(cam, u)))
                A = SMatrix{2, 2, Float64, 4}(950.0, 0.0, 0.0, 950.0)
                @test A * g ≈ e atol = 1.0e-9
                @test A * G ≈ Gθ atol = 1.0e-9
            end
        end

        @testset "transfer_ellipse: affine is exact, homography stays exact, compositions chain" begin
            c = Circle(Point2(10.0, -5.0), 3.0)
            e0 = Ellipse(Point2(10.0, -5.0), 4.0, 2.0, 0.4)
            A = AffineMap(SMatrix{2, 2, Float64, 4}(1.2, 0.3, -0.1, 0.8), SVector(5.0, 7.0))
            for shape in (c, e0)
                out = transfer_ellipse(A, shape)
                @test out isa Ellipse
                ctr, Σ = VgtkCore._ellipse_moments(shape)
                @test SVector{2, Float64}(out.center) ≈ A(ctr)
                @test moment_matrix(out) ≈ A.linear * Σ * A.linear' rtol = 1.0e-10
            end
            H = HomographyMap(HomographyMat{Float64}((1.0, 0.1, 1.0e-4, -0.05, 1.0, 2.0e-4, 20.0, -10.0, 1.0)))
            @test transfer_ellipse(H, e0) isa Ellipse
            @test transfer_ellipse(PerspectiveMap() ∘ H, e0) == transfer_ellipse(H, e0)      # the projective composite IS H on conics
            T = Translation(SVector(3.0, 4.0)) ∘ A
            out = transfer_ellipse(T, c)
            @test SVector{2, Float64}(out.center) ≈ T(SVector(10.0, -5.0))
            @test moment_matrix(out) ≈ A.linear * (9.0 * I) * A.linear' rtol = 1.0e-10
        end

        @testset "transfer_ellipse(camera): the conic through the lens stays an ellipse — $name" for (name, cam) in _CAMS
            camera = PosedCamera(cam, _POSE)
            for shape in (Circle(Point2(120.0, -80.0), 2.5), Ellipse(Point2(-140.0, 95.0), 3.0, 1.5, 0.7))
                out = transfer_ellipse(camera, shape)
                @test out isa Ellipse
                # the sampled true boundary lies on the transferred ellipse: a small
                # blob's image under the lens is affine to well within its own size
                Σ = moment_matrix(out); Σi = inv(Σ)
                ctr = SVector{2, Float64}(out.center)
                worst = 0.0
                for p in boundary_points(shape, 64)
                    q = SVector{2, Float64}(project(camera, Point3{Float64}(p[1], p[2], 0.0)))
                    d = q - ctr
                    worst = max(worst, abs(sqrt(d' * Σi * d) - 1.0))
                end
                @test worst < 0.02
                if cam.lens isa Pinhole
                    @test out == transfer_ellipse(HomographyMap(camera), shape)
                else
                    Hp = HomographyMap(PosedCamera(pinhole(cam), _POSE))
                    stages = transfer_ellipse(Hp, shape)
                    for T in (AffineMap(SMatrix{3, 3, Float64}(cam.intrinsics.K_inv)), cam.lens, VgtkCore.Cameras._pixel_affine(cam.intrinsics))
                        stages = transfer_ellipse(T, stages)
                    end
                    @test out == stages                       # exactly distortion_map's stages on the transferred conic
                end
            end
        end

        @testset "transfer_ellipse(camera) refuses a conic behind the camera" begin
            cam = CameraModel(LogicalIntrinsics(_K), Division(-0.3, -0.08))
            behind = PosedCamera(cam, EuclideanMap(RotMatrix{3, Float64}(one(SMatrix{3, 3, Float64})), SVector(0.0, 0.0, -100.0)))
            @test transfer_ellipse(behind, Circle(Point2(12.0, 0.0), 3.0)) === nothing
            @test transfer_ellipse(PosedCamera(pinhole(cam), behind.extrinsics), Circle(Point2(12.0, 0.0), 3.0)) === nothing
            # the division lens is finite all the way to 90° (its pole is the image circle), so 85° off-axis still transfers
            far = PosedCamera(cam, EuclideanMap(RotMatrix{3, Float64}(one(SMatrix{3, 3, Float64})), SVector(0.0, 0.0, 100.0)))
            @test transfer_ellipse(far, Circle(Point2(1200.0, 0.0), 3.0)) isa Ellipse
        end
    end
end
