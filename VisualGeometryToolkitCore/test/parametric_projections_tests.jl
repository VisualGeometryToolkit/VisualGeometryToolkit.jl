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
    using VisualGeometryToolkitCore.Cameras: _ds_end_cos
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
        # at α = 1/2 the profile's end is its pole at θ = π
        @test first(_ds_end_cos(DoubleSphere(0.0, 0.5))) == -1.0
        @test undistort_normalized(lens, SVector(2.0, 0.0)) isa Vec2
        # the unified model with ξ > 1 images a bounded disc; past it, NaN
        @test all(isnan, backproject(Lens(Unified(1.3)), SVector(3.0, 0.0)))
        @test all(isnan, project(Lens(Unified(0.6)), SVector(0.0, 0.0, -1.0)))
    end

    @testset "double sphere: the domain is the fold of its radial profile" begin
        # Independent reference: the closed-form forward map (eq. 40-45), no domain
        # test, and its radius along a view-angle scan — the first angle where the
        # radius stops increasing (a fold) or turns non-finite or negative (a pole).
        function ds_project(ξ, α, X)
            d1 = sqrt(X[1]^2 + X[2]^2 + X[3]^2)
            zs = ξ * d1 + X[3]
            den = α * sqrt(X[1]^2 + X[2]^2 + zs^2) + (1 - α) * zs
            return SVector(X[1] / den, X[2] / den)
        end
        function scanned_end(ξ, α)
            θs = range(1.0e-4, π - 1.0e-9; length = 200001)
            r = [norm(ds_project(ξ, α, ray(θ))) * sign(ds_project(ξ, α, ray(θ))[1]) for θ in θs]
            k = findfirst(i -> !(isfinite(r[i + 1]) && r[i + 1] > r[i] && r[i] > 0), 1:(length(θs) - 1))
            return k === nothing ? Float64(π) : θs[k], maximum(r[1:something(k, length(r))])
        end
        params = ((-0.6, 0.6), (0.5, 0.6), (0.6, 0.3), (-0.2, 0.55), (0.1, 0.7), (0.3, 0.4), (1.3, 0.6), (1.3, 0.3), (0.9, 0.6), (-0.9, 0.2))
        for (ξ, α) in params
            lens = Lens(DoubleSphere(ξ, α))
            θ_end, r_end = scanned_end(ξ, α)
            # the end of the profile and view_angle_limit (1e-6 inside it), to the scan's step (1.6e-5 rad)
            @test acos(first(_ds_end_cos(lens.projection))) ≈ θ_end atol = 2.0e-5
            @test view_angle_limit(lens) ≈ θ_end atol = 2.0e-5
            # the imaged disc: the fold radius (or unbounded at a pole)
            r_dom = VgtkCore.Cameras._imaging_domain_radius(lens)
            r_end < 1.0e4 ? (@test r_dom ≈ r_end rtol = 1.0e-6) : (@test r_dom == Inf)
            # project is finite exactly inside the fold, and backproject inverts it there — the whole domain
            for θ in range(0.0, π; length = 721), ψ in (0.3, 2.0, -2.6)
                abs(θ - θ_end) < 1.0e-4 && continue
                X = ray(θ; ψ)
                u = project(lens, X)
                @test all(isfinite, u) == (θ < θ_end)
                θ < θ_end || continue
                @test u ≈ ds_project(ξ, α, X) rtol = 1.0e-12
                @test backproject(lens, u) ≈ X atol = (θ_end - θ < 1.0e-2 ? 1.0e-6 : 1.0e-10)
            end
            # past the fold radius no point is imaged
            isfinite(r_dom) && @test all(isnan, backproject(lens, SVector(1.001 * r_dom, 0.0)))
        end
        # ξ = 1: both ends are the antipode, where the forward map is 0/0; the radius is its limit 1/α (the scan's
        # supremum), not the second stage's fold radius 1/√(2α − 1) (α > ½) nor Inf (α ≤ ½)
        for α in (0.7, 0.5, 0.3)
            _, r_end = scanned_end(1.0, α)
            r_dom = VgtkCore.Cameras._imaging_domain_radius(Lens(DoubleSphere(1.0, α)))
            @test r_dom == 1 / α
            @test r_dom ≈ r_end rtol = 1.0e-4
        end
        # the probe's case: the paper's bound (95.1°) cut the injective profile, which reaches 105.2°
        lens = Lens(DoubleSphere(-0.6, 0.6))
        X = ray(deg2rad(104.0))
        @test all(isfinite, project(lens, X))
        @test backproject(lens, project(lens, X)) ≈ X atol = 1.0e-9
        @test rad2deg(view_angle_limit(lens)) ≈ 105.245 atol = 1.0e-3
        @test all(isnan, project(lens, ray(deg2rad(106.0))))           # past the fold: two rays would share a point
        # a distortion in the frame: the round trip still holds inside the fold
        ld = Lens(DoubleSphere(-0.2, 0.55), Radial((-0.05, 0.01)))
        for θ in (0.2, 0.9, 1.6, 2.1)
            X = ray(θ)
            u = project(ld, X)
            all(isfinite, u) && @test backproject(ld, u) ≈ X atol = 1.0e-9
        end

        # ForwardDiff through project and backproject past the paper's bound, w.r.t.
        # the ray / point and (ξ, α), against 256-bit BigFloat central differences
        fxp(v) = Vector(project(Lens(DoubleSphere(v[4], v[5])), SVector(v[1], v[2], v[3])))
        X0 = ray(1.75; ψ = 0.4)
        v0 = [X0[1], X0[2], X0[3], -0.6, 0.6]
        J = ForwardDiff.jacobian(fxp, v0)
        @test all(isfinite, J)
        Jref = setprecision(BigFloat, 256) do
            h = big"1e-25"
            reduce(hcat, [(fxp(big.(v0) .+ h .* (1:5 .== j)) .- fxp(big.(v0) .- h .* (1:5 .== j))) ./ (2h) for j in 1:5])
        end
        @test J ≈ Float64.(Jref) rtol = 1.0e-10
        bxp(v) = Vector(backproject(Lens(DoubleSphere(v[3], v[4])), SVector(v[1], v[2])))
        x0 = project(lens, X0)
        w0 = [x0[1], x0[2], -0.6, 0.6]
        Jb = ForwardDiff.jacobian(bxp, w0)
        Jbref = setprecision(BigFloat, 256) do
            h = big"1e-25"
            reduce(hcat, [(bxp(big.(w0) .+ h .* (1:4 .== j)) .- bxp(big.(w0) .- h .* (1:4 .== j))) ./ (2h) for j in 1:4])
        end
        @test Jb ≈ Float64.(Jbref) rtol = 1.0e-9

        # Float32: same answer at Float32 precision, Float32 out, no allocation
        lens32 = Lens(DoubleSphere(-0.6f0, 0.6f0))
        X32 = SVector{3, Float32}(X0)
        u32 = project(lens32, X32)
        @test eltype(u32) === Float32
        @test u32 ≈ project(lens, X0) rtol = 1.0f-5
        b32 = backproject(lens32, u32)
        @test eltype(b32) === Float32
        @test b32 ≈ X0 atol = 1.0f-4
        project(lens32, X32); backproject(lens32, u32)
        @test (@allocated project(lens32, X32)) == 0
        @test (@allocated backproject(lens32, u32)) == 0
        @test isbitstype(typeof(lens32))
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
        # past the gnomonic frame's reach and refused beyond the fold
        proj = FramedLensProjection(DS_A)
        c_end = first(_ds_end_cos(DS_A.projection))
        @test isfinite(proj(SVector(1.0, 0.0, 0.5 * c_end / sqrt(1 - c_end^2)))[1])
        bad = SVector(1.0, 0.0, -2.0)
        @test bad[3] / norm(bad) < c_end
        @test all(isnan, proj(bad))
    end
end
