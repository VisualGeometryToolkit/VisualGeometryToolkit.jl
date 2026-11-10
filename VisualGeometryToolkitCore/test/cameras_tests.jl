@testitem "camera hierarchy" begin
    # =============================================================================
    # One camera hierarchy: every camera — a model, a posed camera, a bounded one —
    # is an `AbstractCamera`, and `is_posed` says whether it carries a pose.
    # =============================================================================
    #
    # Covers: the trait on each camera form (and forwarded through a
    # BoundedCamera); the refusals of the forms the single hierarchy would
    # otherwise admit (a posed camera posed again, a bounded model posed from
    # inside the bound, a rig or stereo rig of unposed members); the units verbs
    # on an unposed BoundedCamera (the hole the two-root hierarchy left: its
    # methods matched only a posed inner camera); and that the package has no
    # method ambiguities or unbound type parameters.
    import VisualGeometryToolkitCore as VgtkCore
    using Test
    using VisualGeometryToolkitCore
    using VisualGeometryToolkitCore: is_posed, has_units, with_unit, model
    using StaticArrays
    using Rotations: RotMatrix, RotXYZ

    K = @SMatrix [600.0 0.0 320.0; 0.0 600.0 240.0; 0.0 0.0 1.0]
    cm = CameraModel(CameraCalibrationMat(K))
    E = EuclideanMap(RotMatrix{3, Float64}(RotXYZ(0.1, -0.2, 0.05)), SVector(0.3, -0.1, 2.0))
    cam = PosedCamera(cm, E)
    sz = Size2(640, 480)
    gen = NoncentralSplineModel(cm, (640, 480); grid_size = (6, 5))   # an unposed camera that is not a CameraModel

    @testset "one root, one trait" begin
        for c in (cm, gen, cam, BoundedCamera(cm, sz), BoundedCamera(cam, sz))
            @test c isa AbstractCamera
        end
        @test is_posed(cm) == false
        @test is_posed(gen) == false
        @test is_posed(cam) == true
        @test is_posed(BoundedCamera(cm, sz)) == false
        @test is_posed(BoundedCamera(cam, sz)) == true
        @test :is_posed in names(VisualGeometryToolkitCore)   # public
        @test any(f -> f === is_posed, VgtkCore.CONTRACT_GENERICS)
        # the layouts downstream dispatches on are unchanged
        @test cam isa PosedCamera{<:CameraModel{<:AbstractIntrinsics, <:Any}, <:EuclideanMap}
        @test cam isa VgtkCore.PosedPinholeCamera
        @test BoundedCamera(cm, sz) isa BoundedCamera{<:CameraModel}
        @test model(BoundedCamera(cm, sz)) === cm && model(cam) === cm
    end

    @testset "posing a posed camera is refused" begin
        @test_throws ArgumentError PosedCamera(cam, E)
        @test_throws ArgumentError PosedCamera(BoundedCamera(cam, sz), E)
        @test_throws ArgumentError PosedCamera(cam, inv(CameraExtrinsics(E)))   # the CameraPose form
        # a bound goes outside the pose: BoundedCamera(PosedCamera(model, E), extent)
        @test_throws ArgumentError PosedCamera(BoundedCamera(cm, sz), E)
        # with_model takes a model, not a camera
        @test_throws ArgumentError with_model(cam, cam)
        @test_throws ArgumentError with_model(cm, cam)
        @test with_model(cm, gen) === gen
    end

    @testset "a rig member is posed" begin
        rcm = CameraModel(CameraCalibrationMat(K))
        @test_throws ArgumentError Rig((rcm,))
        @test_throws ArgumentError Rig{WorldFrame}((rcm,))
        @test_throws ArgumentError Rig((BoundedCamera(rcm, sz),))
        @test_throws ArgumentError StereoRig(rcm, cam)
        @test_throws ArgumentError StereoRig(cam, BoundedCamera(rcm, sz))
        @test_throws ArgumentError StereoRig(rcm, rcm, E)   # check = true reads the poses
    end

    @testset "units of an unposed BoundedCamera" begin
        bc = BoundedCamera(cm, sz)
        @test !has_units(bc) && !has_units(cm)
        @test with_unit(bc, mm) == bc                    # a model holds no length: nothing to attach
        @test ustrip(mm, bc) == bc
        # the posed forms are unchanged
        bu = BoundedCamera(with_unit(cam, mm), sz)
        @test has_units(bu) && ustrip(mm, bu) isa BoundedCamera && !has_units(ustrip(mm, bu))
        @test with_unit(ustrip(mm, bu), mm) isa BoundedCamera
    end

    @testset "no ambiguities, no unbound parameters" begin
        # Only the package's own methods: another item in this worker may have
        # extended a package function for its fixtures.
        own(meth) = Base.moduleroot(meth.module) === VisualGeometryToolkitCore
        amb = Test.detect_ambiguities(VisualGeometryToolkitCore; recursive = true)
        @test isempty(filter(p -> own(p[1]) && own(p[2]), amb))
        @test isempty(filter(own, Test.detect_unbound_args(VisualGeometryToolkitCore; recursive = true)))
    end

    @testset "the retired names are deleted, not aliased" begin
        # `Camera` and `PinholeCamera` became `PosedCamera` and `PosedPinholeCamera`;
        # a leftover use is an UndefVarError, not a deprecation.
        @test !isdefined(VgtkCore, :Camera)
        @test !isdefined(VgtkCore, :PinholeCamera)
        @test !isdefined(VgtkCore.Cameras, :Camera)
        @test !isdefined(VgtkCore.Cameras, :PinholeCamera)
        @test isdefined(VgtkCore, :PosedCamera) && isdefined(VgtkCore, :PosedPinholeCamera)
    end
end

@testitem "cameras under ForwardDiff" begin
    # =============================================================================
    # A camera differentiates with ForwardDiff Duals: a Dual calibration matrix
    # keeps its element type, intrinsics rebuilt from Dual parameters are Dual,
    # and `backproject` takes Dual pixels. Exact-number input (integer, rational)
    # still promotes to Float64, value for value.
    # =============================================================================
    using Test
    using VisualGeometryToolkitCore
    using StaticArrays
    using LinearAlgebra: I
    using ForwardDiff: ForwardDiff
    using FiniteDiff: FiniteDiff

    @testset "CameraCalibrationMat element types" begin
        Kf = SMatrix{3, 3, Float64}(800, 0, 0, 0, 810, 0, 320, 240, 1)
        for M in ([800 0 320; 0 810 240; 0 0 1], [800 0 320; 0 810 240; 0 0 1] .// 1, SMatrix{3, 3}(800, 0, 0, 0, 810, 0, 320, 240, 1))
            K = CameraCalibrationMat(M)
            @test K isa CameraCalibrationMat{Float64} && SMatrix(K) == Kf
        end
        @test CameraCalibrationMat(Float32.(Kf)) isa CameraCalibrationMat{Float32}
        f = ForwardDiff.Dual{Nothing}(800.0, 1.0)
        D = typeof(f)
        Kd = CameraCalibrationMat([f 0 320; 0 f 240; 0 0 1])
        @test Kd isa CameraCalibrationMat{D} && ForwardDiff.value(Kd[1, 1]) == 800.0
        @test CameraCalibrationMat(SMatrix{3, 3, D}(Kd)) isa CameraCalibrationMat{D}
        @test_throws ArgumentError CameraCalibrationMat(SMatrix{3, 3, D}(Kd)')     # the transpose check reads Duals
    end

    @testset "with_parameters(::LogicalIntrinsics) differentiates" begin
        intr = LogicalIntrinsics(CameraCalibrationMat(SMatrix{3, 3, Float64}(800.0, 0, 0, 1.5, 1000.0, 0, 320.0, 240.0, 1)))
        θ0 = collect(parameter_vector(intr))
        @test ForwardDiff.jacobian(θ -> parameter_vector(with_parameters(intr, θ)), θ0) ≈ I
        d = ForwardDiff.Dual{Nothing}.(θ0, 1.0)
        @test with_parameters(intr, d) isa LogicalIntrinsics{eltype(d)}
    end

    @testset "backproject of Dual pixels" begin
        K = CameraCalibrationMat(SMatrix{3, 3, Float64}(300.0, 0, 0, 0, 300.0, 0, 320.0, 240.0, 1))
        u0 = [400.3, 287.1]
        for lens in (KannalaBrandt(-0.04, 0.01), Lens(DoubleSphere(-0.2, 0.6)))
            cm = CameraModel(K; lens)
            ray(u) = Vector(backproject(cm, SVector{2}(u)))
            J = ForwardDiff.jacobian(ray, u0)
            Jfd = FiniteDiff.finite_difference_jacobian(ray, u0, Val(:central))
            @test isapprox(J, Jfd; rtol = 1.0e-6)
        end
        cm = CameraModel(K; lens = KannalaBrandt(-0.04, 0.01))
        # end to end in the intrinsics: θ → Dual intrinsics → camera → bearing / pixel
        intr = LogicalIntrinsics(CameraCalibrationMat(SMatrix{3, 3, Float64}(300.0, 0, 0, 1.5, 310.0, 0, 320.0, 240.0, 1)))
        θ0 = collect(parameter_vector(intr))
        for lens in (KannalaBrandt(-0.04, 0.01), Lens(DoubleSphere(-0.2, 0.6)))
            bearing(θ) = Vector(backproject(CameraModel(with_parameters(intr, θ); lens), SVector(400.3, 287.1)))
            pixel(θ) = Vector(project(CameraModel(with_parameters(intr, θ); lens), SVector(0.1, 0.2, 1.0)))
            for h in (bearing, pixel)
                @test isapprox(ForwardDiff.jacobian(h, θ0), FiniteDiff.finite_difference_jacobian(h, θ0, Val(:central)); rtol = 1.0e-6)
            end
        end
        # integer pixels: the bearing of their floating-point value
        @test backproject(cm, SVector(400, 300)) == backproject(cm, SVector(400.0, 300.0))
        @test backproject(cm, [SVector(400, 300)]) == [backproject(cm, SVector(400.0, 300.0))]
    end

    @testset "a Dual camera's backproject inlines through BoundedCamera" begin
        # RadiometricCamera's device Jacobian backprojects through a Dual{Float32, 15}
        # camera. With `@forward`'s forwarder and a non-`@inline` CameraModel method
        # the inliner kept both as calls, and the kernel handed the camera over by
        # pointer through local memory (3008 B a thread at KB; 1160 B after). The bearing is
        # the same bits as the pixel-frame method's.
        D = ForwardDiff.Dual{Nothing, Float32, 15}
        seed(x, i) = D(x, ForwardDiff.Partials(ntuple(j -> Float32(j == i), 15)))
        K = CameraCalibrationMat(SMatrix{3, 3, D}(seed(300.0f0, 1), 0, 0, 0, seed(300.0f0, 2), 0, seed(320.0f0, 3), seed(240.0f0, 4), 1))
        lens0 = KannalaBrandt(-0.04f0, 0.01f0)
        θ = parameter_vector(lens0)
        cam = BoundedCamera(CameraModel(K; lens = with_parameters(lens0, [seed(θ[i], 4 + i) for i in eachindex(θ)])), Size2(640, 480))
        u = SVector(400.3f0, 287.1f0)
        @test backproject(cam, u) === Vec3(backproject(cam.camera.intrinsics, cam.camera.lens, u))
        ci = only(code_typed(backproject, Tuple{typeof(cam), typeof(u)}))[1]
        callee(st) = (m = st.args[1]; m isa Core.CodeInstance ? m.def : m)
        called = [callee(st).def.name for st in ci.code if st isa Expr && st.head === :invoke]
        @test :backproject ∉ called
        @test :sin ∉ called && :cos ∉ called
    end
end
