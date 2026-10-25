@testitem "camera_pose_construction" begin
    using Test
    using VisualGeometryToolkitCore
    using StaticArrays
    using Rotations: Rotations
    using LinearAlgebra: I, norm

    # Frame-pair aliases: PosedCamera(model, ::CameraPose) must invert into the
    # extrinsics slot (so pose(camera) round-trips the labeled map), and
    # lookat_pose must be the direct camera-to-world companion of lookat
    # (bit-exact eye translation, per-convention frame parity).

    @testset "CameraPose / CameraExtrinsics + lookat_pose" begin
        K = SMatrix{3, 3, Float64}(600.0, 0, 0, 0, 600.0, 0, 320.0, 240.0, 1.0)
        model = CameraModel(CameraCalibrationMat(K))

        # Asymmetric viewpoint: inv(P) != P, so a stored-frame mix-up cannot hide.
        eye = SVector(0.3, -0.2, 0.6)
        target = SVector(0.1, 0.05, -1.0)
        up = SVector(0.0, 1.0, 0.0)

        @testset "alias round-trips and direct field access" begin
            P = lookat_pose(eye, target, up; convention = :gfx)
            @test P isa CameraPose
            @test P isa EuclideanMap        # alias, not a wrapper
            E = inv(P)
            @test E isa CameraExtrinsics
            @test inv(E) isa CameraPose
            # field access is direct — a CameraPose IS an EuclideanMap
            @test P.R isa Rotations.RotMatrix{3}
            @test P.t isa SVector{3, Float64}
            # callable: applying the pose maps camera-frame origin to the eye
            @test P(SVector(0.0, 0.0, 0.0)) == P.t
        end

        @testset "pose(PosedCamera(model, P::CameraPose)) round-trips P" begin
            P = lookat_pose(eye, target, up; convention = :gfx)
            cam = PosedCamera(model, P)
            @test SMatrix{3, 3, Float64, 9}(pose(cam).R) ≈ SMatrix{3, 3, Float64, 9}(P.R) atol = 1.0e-12
            @test pose(cam).t ≈ P.t atol = 1.0e-12
            # and the stored extrinsics really is the inverse map
            E = extrinsics(cam)
            @test SMatrix{3, 3, Float64, 9}(E.R) ≈ SMatrix{3, 3, Float64, 9}(P.R)' atol = 1.0e-12
            @test E.t ≈ -SMatrix{3, 3, Float64, 9}(P.R)' * P.t atol = 1.0e-12
            # the two typed routes agree exactly: CameraExtrinsics unwraps, CameraPose inverts
            cam_e = PosedCamera(model, inv(P))
            @test extrinsics(cam_e).t == extrinsics(cam).t
            @test SMatrix{3, 3, Float64, 9}(extrinsics(cam_e).R) ==
                SMatrix{3, 3, Float64, 9}(extrinsics(cam).R)
        end

        @testset "lookat_pose translation is the eye, bit-exactly" begin
            for convention in (:cv, :gfx)
                P = lookat_pose(eye, target, up; convention)
                @test P.t == eye
            end
        end

        @testset "lookat(:gfx) == inv(lookat_pose(:gfx))" begin
            E = lookat(eye, target, up; convention = :gfx)
            Pinv = inv(lookat_pose(eye, target, up; convention = :gfx))
            @test Pinv isa CameraExtrinsics
            @test SMatrix{3, 3, Float64, 9}(E.R) ≈ SMatrix{3, 3, Float64, 9}(Pinv.R) atol = 1.0e-12
            @test E.t ≈ Pinv.t atol = 1.0e-12
        end

        @testset ":cv default is unchanged" begin
            E_default = lookat(eye, target, up)
            E_cv = lookat(eye, target, up; convention = :cv)
            @test E_default isa EuclideanMap          # still an EuclideanMap...
            @test E_default isa CameraExtrinsics      # ...labeled at source
            @test SMatrix{3, 3, Float64, 9}(E_default.R) == SMatrix{3, 3, Float64, 9}(E_cv.R)
            @test E_default.t == E_cv.t
            # :cv pose agrees with inv of the :cv extrinsics
            Pinv = inv(lookat_pose(eye, target, up; convention = :cv))
            @test SMatrix{3, 3, Float64, 9}(E_cv.R) ≈ SMatrix{3, 3, Float64, 9}(Pinv.R) atol = 1.0e-12
        end

        @testset ":gfx reproduces the Mitsuba default-scene frame" begin
            # eye (0,0,0.5) → origin, up +y: the axis-aligned pose every entry of
            # which is exactly representable, so equality is exact.
            P = lookat_pose(SVector(0.0, 0.0, 0.5), SVector(0.0, 0.0, 0.0), up; convention = :gfx)
            @test SMatrix{3, 3, Float64, 9}(P.R) ==
                SMatrix{3, 3, Float64, 9}(-1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, -1.0)
            @test P.t == SVector(0.0, 0.0, 0.5)
        end

        @testset "unknown convention throws" begin
            @test_throws ArgumentError lookat(eye, target, up; convention = :opengl)
            @test_throws ArgumentError lookat_pose(eye, target, up; convention = :opengl)
        end
    end

    # The camera leads (model first), so BoundedCamera carries it through the
    # same @forward as project; the point-first order is gone, not aliased.
    @testset "is_in_front: camera first, BoundedCamera forwards" begin
        K = SMatrix{3, 3, Float64}(600.0, 0, 0, 0, 600.0, 0, 320.0, 240.0, 1.0)
        P = lookat_pose(SVector(0.3, -0.2, 0.6), SVector(0.1, 0.05, -1.0), SVector(0.0, 1.0, 0.0); convention = :gfx)
        cam = PosedCamera(CameraModel(CameraCalibrationMat(K)), P)
        front = Point3(pose(cam)(SVector(0.1, -0.1, 2.0))...)   # camera-frame z = +2
        back = Point3(pose(cam)(SVector(0.1, -0.1, -2.0))...)   # camera-frame z = -2
        for c in (cam, BoundedCamera(cam, Size2(640, 480)))
            @test is_in_front(c, front)
            @test !is_in_front(c, back)
            @test is_in_front.(Ref(c), [front, back]) == [true, false]
            @test is_in_front(c, [front[1], back[1]], [front[2], back[2]], [front[3], back[3]]) == [true, false]
        end
        @test_throws MethodError is_in_front(front, cam)
        @test_throws MethodError focal_length_for_fov(deg2rad(40.0), Size2(width = 640px, height = 480px))
    end

    @testset "CameraModel(K): the pinhole camera of a static 3×3 K" begin
        K = SMatrix{3, 3, Float64}(600.0, 0, 0, 0, 610.0, 0, 320.0, 240.0, 1.0)
        @test CameraModel(K) == CameraModel(CameraCalibrationMat(K))
        @test calibration_matrix(CameraModel(K)) == K
        @test lens(CameraModel(K)) isa Pinhole
        @test CameraModel(SMatrix{3, 3, Int}(600, 0, 0, 0, 600, 0, 320, 240, 1)) ==
            CameraModel(CameraCalibrationMat(SMatrix{3, 3, Float64}(600, 0, 0, 0, 600, 0, 320, 240, 1)))
        @test_throws ArgumentError CameraModel(SMatrix{3, 3, Float64}(K'))   # validated: the transpose is named
    end
end
