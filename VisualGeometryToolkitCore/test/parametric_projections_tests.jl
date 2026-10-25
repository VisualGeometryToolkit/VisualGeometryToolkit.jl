# =============================================================================
# The parametric projections — double sphere, unified, extended unified:
# closed-form project / backproject with honest validity, their perspective
# views, the identities between them, a batch through pointwise!, and the
# framed-surface fused stage.
# =============================================================================

@testitem "parametric_projections" begin
    import VisualGeometryToolkitCore as VgtkCore
    using Test
    using VisualGeometryToolkitCore: imaging_chain, FramedStage, FramedChain, distort_stage,
        undistort_stage, FramedLensProjection, ray_map, DistortedFrame, Serial, Threaded
    using VisualGeometryToolkitCore
    using VisualGeometryToolkitCore: _precision
    using VisualGeometryToolkitCore.Cameras: _ds_w2
    using StaticArrays: SVector
    using FiniteDiff: FiniteDiff
    using LinearAlgebra: I, norm, normalize

    # Calibration-plausible parameter sets: one on each side of the α = 1/2
    # unprojection-domain split, plus a negative-ξ set.
    DS_A = Lens(DoubleSphere(-0.2, 0.55))
    DS_B = Lens(DoubleSphere(0.3, 0.4))
    DS_C = Lens(DoubleSphere(0.1, 0.7))
    UC = (Lens(Unified(0.6)), Lens(Unified(1.3)))
    EU = (Lens(ExtendedUnified(0.6, 1.1)), Lens(ExtendedUnified(0.4, 0.8)))
    ray(θ; ψ = 0.6) = SVector(sin(θ) * cos(ψ), sin(θ) * sin(ψ), cos(θ))

    @testset "construction, traits, precision hook" begin
        @test DS_A isa Lens{DoubleSphere{Float64}, Nothing}
        @test parameters(DS_A) == (xi = -0.2, alpha = 0.55)
        @test Lens(DoubleSphere(0.1f0, 0.2f0)) isa Lens{DoubleSphere{Float32}}
        @test has_distortion(DS_A)
        @test distortion_frame(DS_A) === nothing
        lens32 = _precision(Float32, DS_A)
        @test lens32 isa Lens{DoubleSphere{Float32}}
        @test lens32.projection.xi === Float32(-0.2)
    end

    @testset "identities between the models" begin
        # ξ = α = 0 is the pinhole; the extended unified model with β = 1 is the
        # unified model scaled by 1 − α, and the double sphere with ξ = 0 is it
        for u in (-0.4, 0.0, 0.7), v in (-0.3, 0.2)
            x = SVector(u, v)
            @test Lens(DoubleSphere(0.0, 0.0))(x) ≈ x atol = 1.0e-15
            @test undistort_normalized(Lens(DoubleSphere(0.0, 0.0)), x) ≈ x atol = 1.0e-12
        end
        for θ in (0.0, 0.4, 1.2, 1.9), α in (0.3, 0.6)
            X = ray(θ)
            @test project(Lens(ExtendedUnified(α, 1.0)), X) ≈ project(Lens(Unified(α / (1 - α))), X) / (1 - α) atol = 1.0e-12
            @test project(Lens(DoubleSphere(0.0, α)), X) ≈ project(Lens(ExtendedUnified(α, 1.0)), X) atol = 1.0e-12
        end
    end

    @testset "closed-form round trip on rays, past 90°" begin
        for lens in (DS_A, DS_B, DS_C, UC..., EU...), θd in (0.0, 10.0, 45.0, 80.0, 95.0, 110.0)
            X = ray(deg2rad(θd))
            u = project(lens, 3.0 * X)
            all(isfinite, u) || continue          # past this lens's field of view
            @test backproject(lens, u) ≈ X atol = 1.0e-10
            @test transform_deriv(lens, X) ≈ FiniteDiff.finite_difference_jacobian(v -> project(lens, v), X, Val(:central)) rtol = 1.0e-5 atol = 1.0e-7
            @test transform_deriv_params(lens, X) ≈
                FiniteDiff.finite_difference_jacobian(q -> project(with_parameters(lens, q), X), parameter_vector(lens), Val(:central)) rtol = 1.0e-5 atol = 1.0e-7
        end
        # every one of them images past 90°
        for lens in (DS_A, DS_B, UC..., EU[1])
            @test all(isfinite, project(lens, ray(deg2rad(100.0))))
        end
    end

    @testset "closed-form round trip at both precisions" begin
        for lens in (DS_A, DS_B, DS_C, UC[1], EU[1])
            for T in (Float64, Float32)
                lens_T = _precision(T, lens)
                worst = zero(T)
                for u in range(T(-0.5), T(0.5); length = 21), v in range(T(-0.5), T(0.5); length = 21)
                    x = SVector{2, T}(u, v)
                    fx = lens_T(x)
                    xu = undistort_normalized(lens_T, SVector{2, T}(fx[1], fx[2]))
                    @test eltype(xu) === T
                    worst = max(worst, abs(xu[1] - x[1]), abs(xu[2] - x[2]))
                end
                @test worst <= (T === Float64 ? 1.0e-12 : 1.0f-5)
            end
        end
    end

    @testset "honest validity: domain edges flag, never throw" begin
        # α > 1/2: unprojection radius bound r² ≤ 1/(2α − 1).
        lens = Lens(DoubleSphere(0.1, 0.7))   # r ≤ √2.5 ≈ 1.58
        @test all(isnan, backproject(lens, SVector(2.0, 0.0)))
        @test all(isnan, undistort_normalized(lens, SVector(2.0, 0.0)))
        # Inside the radius bound but past the image of the perspective plane:
        # the ray exists (Z < 0), its perspective point does not.
        lens_b = Lens(DoubleSphere(0.5, 0.3))
        b = backproject(lens_b, SVector(2.0, 0.0))
        @test all(isfinite, b) && b[3] < 0
        @test all(isnan, undistort_normalized(lens_b, SVector(2.0, 0.0)))
        # w₂ matches the paper's split at α = 1/2
        @test _ds_w2(DoubleSphere(0.0, 0.5)) ≈ 1.0 atol = 1.0e-12
        @test undistort_normalized(lens, SVector(2.0, 0.0)) isa Vec2
        # the unified model with ξ > 1 images a bounded disc; past it, NaN
        @test all(isnan, backproject(Lens(Unified(1.3)), SVector(3.0, 0.0)))
        @test all(isnan, project(Lens(Unified(0.6)), SVector(0.0, 0.0, -1.0)))
    end

    @testset "a batch through pointwise!: every policy the same" begin
        xs = [SVector(u, v) for u in range(-0.4, 0.4; length = 9) for v in range(-0.3, 0.3; length = 9)]
        push!(xs, SVector(2.0, 0.0))                      # not imaged by DS_C
        for lens in (DS_A, DS_C, UC[1], EU[1])
            ref = [undistort_normalized(lens, x) for x in xs]
            for exec in (Serial(), Threaded(), VgtkCore.Auto())
                out = similar(xs)
                undistort_normalized!(out, lens, xs; exec)
                @test isequal(out, ref)                    # NaN where not imaged, identical otherwise
            end
        end
    end

    @testset "frame split refused with the ModelDefined reason" begin
        @test_throws ArgumentError distort_stage(DS_A)
        @test_throws ArgumentError undistort_stage(DS_A)
        @test_throws ArgumentError ray_map(DS_A)
        err = try
            distort_stage(DS_A)
        catch e
            e
        end
        @test occursin("ModelDefined", err.msg)
        @test occursin("FramedLensProjection", err.msg)
    end

    @testset "fused model-carrying projection + imaging_chain" begin
        for lens in (DS_A, UC[1], Lens(Unified(0.7), Radial((-0.1, 0.01))))
            proj = FramedLensProjection(lens; id = :cam)
            @test proj isa FramedStage{ViewFrame{:cam}, DistortedFrame{:cam}}
            # scale-invariant on ray representatives; the lens functor's view on the forward hemisphere
            for (u, v) in ((0.0, 0.0), (0.3, -0.2), (-0.45, 0.4))
                expected = lens(SVector(u, v))
                for s in (1.0, 0.37, 12.0)
                    @test proj(SVector(s * u, s * v, s)) ≈ expected atol = 1.0e-12
                end
            end
            @test_throws MethodError inv(proj)
            intr = LogicalIntrinsics(CameraCalibrationMat(1.0 * I))
            chain = imaging_chain(intr, lens; id = :cam)
            @test chain isa FramedChain && length(chain.stages) == 2
            @test chain.stages[1] isa FramedLensProjection
            X = SVector(0.2, -0.1, 1.0)
            legacy = imaging_transform(intr, lens)(X)
            @test chain(X) ≈ SVector(legacy[1], legacy[2]) atol = 1.0e-12
            @test_throws MethodError inv(chain)
        end
        # past the gnomonic frame's reach and refused beyond the w₂ bound
        proj = FramedLensProjection(DS_A)
        w2 = _ds_w2(DS_A.projection)
        @test isfinite(proj(SVector(1.0, 0.0, -0.5 * w2))[1])
        bad = SVector(1.0, 0.0, -2.0)
        @test bad[3] / norm(bad) < -w2
        @test all(isnan, proj(bad))
    end
end
