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
    using ForwardDiff: ForwardDiff
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

    @testset "double sphere: backproject without the FOV bound (fov_bound = false)" begin
        # The closed-form lift (eq. 46-51), clamped radicands, no domain test at all.
        function ds_lift(ξ, α, m)
            r2 = m[1]^2 + m[2]^2
            mz = (1 - α^2 * r2) / (α * sqrt(max(1 - (2α - 1) * r2, 0)) + 1 - α)
            s = (mz * ξ + sqrt(max(mz^2 + (1 - ξ^2) * r2, 0))) / (mz^2 + r2)
            return SVector(s * m[1], s * m[2], s * mz - ξ)
        end
        # The algebraic domain alone: the unprojection radius and the second radicand.
        function ds_algebraic(ξ, α, m)
            r2 = m[1]^2 + m[2]^2
            disc = 1 - (2α - 1) * r2
            mz = (1 - α^2 * r2) / (α * sqrt(max(disc, 0)) + 1 - α)
            return (α <= 0.5 || disc >= 0) && mz^2 + (1 - ξ^2) * r2 >= 0
        end
        grid = [SVector(u, v) for u in range(-2.3, 2.3; length = 47) for v in range(-2.3, 2.3; length = 47)]
        lenses = (
            DS_A, DS_B, DS_C, Lens(DoubleSphere(-0.6, 0.6)), Lens(DoubleSphere(0.6, 0.3)),
            Lens(DoubleSphere(-0.2, 0.55), Radial((-0.05, 0.01))),
        )
        generic = Tuple{Lens, SVector{2, Float64}}
        for lens in lenses, x in grid
            # the default is the lens contract's answer, bit for bit (the generic Lens method)
            ref = invoke(backproject, generic, lens, x)
            @test isequal(backproject(lens, x), ref)
            @test isequal(backproject(lens, x; fov_bound = true), ref)
            # where the bound holds the unchecked form is that same answer
            all(isfinite, ref) && @test isequal(backproject(lens, x; fov_bound = false), ref)
        end

        # ξ = −0.6, α = 0.6: past the paper's bound (≈ 95.1°) yet inside the algebraic
        # domain (r ≤ √5) — points a caller with only the algebraic domain accepts.
        ξ, α = -0.6, 0.6
        lens = Lens(DoubleSphere(ξ, α))
        past = [x for x in grid if ds_algebraic(ξ, α, x) && all(isnan, backproject(lens, x))]
        radial_past = [SVector(ρ * cos(ψ), ρ * sin(ψ)) for ρ in (2.21, 2.22, 2.23, 2.236) for ψ in (0.0, 0.7, 2.0, -2.9)]
        @test !isempty(past)
        for x in vcat(past, radial_past)
            @test ds_algebraic(ξ, α, x)
            @test all(isnan, backproject(lens, x))
            b = backproject(lens, x; fov_bound = false)
            @test b ≈ ds_lift(ξ, α, x) atol = 1.0e-12
            @test norm(b) ≈ 1 atol = 1.0e-12
            @test all(isnan, project(lens, b))            # project keeps the bound
        end
        # the algebraic domain still yields NaN (r > √5)
        @test all(isnan, backproject(lens, SVector(2.3, 0.0); fov_bound = false))
        # inside the bound, unchecked == checked == the closed form
        for x in (SVector(0.3, -0.2), SVector(1.5, 0.9), SVector(-2.0, 0.5))
            @test isequal(backproject(lens, x; fov_bound = false), backproject(lens, x))
            @test backproject(lens, x) ≈ ds_lift(ξ, α, x) atol = 1.0e-12
        end

        # ForwardDiff through the unchecked form, past the bound: w.r.t. the point
        # and w.r.t. (ξ, α), against central differences. The point sits near the
        # algebraic edge (√(1 − (2α − 1)r²) ≈ 0.12), where a Float64 difference's
        # step error reaches 1e-6, so the differences run in 256-bit BigFloat.
        x0 = SVector(2.22 * cos(0.4), 2.22 * sin(0.4))
        @test all(isnan, backproject(lens, x0))
        fxp(v) = Vector(backproject(Lens(DoubleSphere(v[3], v[4])), SVector(v[1], v[2]); fov_bound = false))
        v0 = [x0[1], x0[2], ξ, α]
        J = ForwardDiff.jacobian(fxp, v0)
        @test all(isfinite, J)
        Jref = setprecision(BigFloat, 256) do
            h = big"1e-25"
            reduce(hcat, [(fxp(big.(v0) .+ h .* (1:4 .== j)) .- fxp(big.(v0) .- h .* (1:4 .== j))) ./ (2h) for j in 1:4])
        end
        @test J ≈ Float64.(Jref) rtol = 1.0e-10

        # Float32: same answer at Float32 precision, Float32 out, no allocation
        lens32 = Lens(DoubleSphere(Float32(ξ), Float32(α)))
        x32 = SVector{2, Float32}(x0)
        b32 = backproject(lens32, x32; fov_bound = false)
        @test eltype(b32) === Float32
        @test b32 ≈ backproject(lens, x0; fov_bound = false) atol = 1.0f-5
        @test all(isnan, backproject(lens32, x32))
        bp(l, x) = backproject(l, x; fov_bound = false)
        bp(lens32, x32)
        @test (@allocated bp(lens32, x32)) == 0
        @test isbitstype(typeof(lens32))
    end

    @testset "double sphere: project without the FOV bound (fov_bound = false)" begin
        # The closed-form forward map (eq. 40-45), no domain test at all.
        function ds_project(ξ, α, X)
            d1 = sqrt(X[1]^2 + X[2]^2 + X[3]^2)
            zs = ξ * d1 + X[3]
            den = α * sqrt(X[1]^2 + X[2]^2 + zs^2) + (1 - α) * zs
            return SVector(X[1] / den, X[2] / den)
        end
        rays = [ray(θ; ψ) for θ in range(0, π; length = 61) for ψ in range(-π, π; length = 13)]
        push!(rays, SVector(0.0, 0.0, 0.0), SVector(0.0, 0.0, -1.0), 3.7 * ray(1.9))
        lenses = (
            DS_A, DS_B, DS_C, Lens(DoubleSphere(-0.6, 0.6)), Lens(DoubleSphere(0.6, 0.3)),
            Lens(DoubleSphere(-0.2, 0.55), Radial((-0.05, 0.01))),
        )
        # A structural reference, not the compiled generic method: the paper's bound
        # (eq. 34-36), the closed form, and the one distortion here, Radial((−0.05,
        # 0.01)), whose profile never folds. To 4 eps: the reference orders its
        # operations differently (x/den, not inv(den)·x).
        function ds_reference(lens, X)
            ξ, α = lens.projection.xi, lens.projection.alpha
            d1 = norm(X)
            inside = (X[3] > 0 && ξ >= 0 && 0 <= α <= 1) || X[3] > -_ds_w2(lens.projection) * d1
            inside || return SVector(NaN, NaN)
            x = ds_project(ξ, α, X)
            lens.distortion === nothing && return x
            s = x[1]^2 + x[2]^2
            return (1 - 0.05s + 0.01s^2) * x
        end
        same(a, b) = isequal(isnan.(a), isnan.(b)) && isapprox(a, b; rtol = 4 * eps(), nans = true)
        for lens in lenses, X in rays
            iszero(X) && continue
            # the default is the lens contract's answer
            @test same(project(lens, X), ds_reference(lens, X))
            @test isequal(project(lens, X; fov_bound = true), project(lens, X))
            # where the bound holds the unchecked form is that same answer
            all(isfinite, project(lens, X)) && @test isequal(project(lens, X; fov_bound = false), project(lens, X))
        end
        # the zero vector is still no ray
        @test all(isnan, project(DS_A, SVector(0.0, 0.0, 0.0); fov_bound = false))

        # where the check fails, the unchecked form is the closed form
        nfailed = 0
        for (ξ, α) in ((-0.2, 0.55), (0.3, 0.4), (0.1, 0.7), (-0.6, 0.6), (0.6, 0.3)), X in rays
            lens = Lens(DoubleSphere(ξ, α))
            (iszero(X) || all(isfinite, project(lens, X))) && continue
            ref = ds_project(ξ, α, X)
            all(isfinite, ref) || continue
            nfailed += 1
            @test isapprox(project(lens, X; fov_bound = false), ref; rtol = 1.0e-12)
        end
        @test nfailed > 100

        # ξ = −0.6, α = 0.6: between the paper's bound (≈ 95.1°) and the algebraic edge
        # (≈ 105.3°) the map is still one-to-one, so the two unchecked forms round-trip
        lens = Lens(DoubleSphere(-0.6, 0.6))
        for θ in (1.67, 1.72, 1.78, 1.83), ψ in (0.0, 1.1, -2.5)
            X = ray(θ; ψ)
            @test all(isnan, project(lens, X))
            u = project(lens, X; fov_bound = false)
            @test backproject(lens, u; fov_bound = false) ≈ X atol = 1.0e-9
        end

        # ForwardDiff through the unchecked form past the bound, w.r.t. the ray and
        # (ξ, α), against 256-bit BigFloat central differences
        fxp(v) = Vector(project(Lens(DoubleSphere(v[4], v[5])), SVector(v[1], v[2], v[3]); fov_bound = false))
        X0 = ray(1.75; ψ = 0.4)
        v0 = [X0[1], X0[2], X0[3], -0.6, 0.6]
        J = ForwardDiff.jacobian(fxp, v0)
        @test all(isfinite, J)
        Jref = setprecision(BigFloat, 256) do
            h = big"1e-25"
            reduce(hcat, [(fxp(big.(v0) .+ h .* (1:5 .== j)) .- fxp(big.(v0) .- h .* (1:5 .== j))) ./ (2h) for j in 1:5])
        end
        @test J ≈ Float64.(Jref) rtol = 1.0e-10

        # Float32: same answer at Float32 precision, Float32 out, no allocation
        lens32 = Lens(DoubleSphere(-0.6f0, 0.6f0))
        X32 = SVector{3, Float32}(X0)
        u32 = project(lens32, X32; fov_bound = false)
        @test eltype(u32) === Float32
        @test u32 ≈ project(lens, X0; fov_bound = false) rtol = 1.0f-5
        @test all(isnan, project(lens32, X32))
        pj(l, X) = project(l, X; fov_bound = false)
        pj(lens32, X32)
        @test (@allocated pj(lens32, X32)) == 0
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
