# =============================================================================
# Batched SoA ray-plane intersection (primitives/hyperplane.jl, 0.14; hoisted
# from RayTracedCalibration) + the constant-plane rule of the extension.
# =============================================================================

@testitem "ray_plane" begin
    using Test
    using VisualGeometryToolkitCore
    using VisualGeometryToolkitCore: ray_plane_intersection   # public, not exported
    using VisualGeometryToolkitCore: Primitives
    using StaticArrays: SVector
    using LinearAlgebra: dot
    import ChainRulesCore
    import ForwardDiff
    import KernelAbstractions as KA

    KA.@kernel function _dual_plane_kernel!(out, @Const(c), x)
        i = KA.@index(Global)
        d = ForwardDiff.Dual{Nothing}(x[i], 1.0f0)
        out[i] = ForwardDiff.value(Plane3D{typeof(d)}(c).coeffs[4] * d)
    end

    @testset "shared-origin form: hits on the plane, correct parameter" begin
        plane = Plane3D(Point3(0.0, 0.0, 2.0), Vec3(0.0, 0.0, 1.0))   # z = 2
        origin = Point3(0.0, 0.0, 0.0)
        dir_x = [0.0, 0.1, -0.2]
        dir_y = [0.0, 0.2, 0.1]
        dir_z = [1.0, 1.0, 0.5]
        x, y, z, t = ray_plane_intersection(plane, origin, dir_x, dir_y, dir_z)
        @test z ≈ [2.0, 2.0, 2.0] atol = 1.0e-12
        @test t ≈ [2.0, 2.0, 4.0] atol = 1.0e-12
        @test x ≈ dir_x .* t atol = 1.0e-12
        @test y ≈ dir_y .* t atol = 1.0e-12
    end

    @testset "random plane: n·hit + d = 0 and hit = o + t·dir (both forms)" begin
        n = SVector(1.0, -2.0, 0.5)
        plane = Plane3D(Point3(0.3, -0.1, 1.2), Vec3(n...))
        n1, n2, n3, d = Primitives._plane_constants(plane)
        N = 32
        o_x = randn(N); o_y = randn(N); o_z = randn(N)
        dir_x = randn(N); dir_y = randn(N); dir_z = randn(N) .+ 2.0

        # per-ray-origin form
        x, y, z, t = ray_plane_intersection(plane, o_x, o_y, o_z, dir_x, dir_y, dir_z)
        @test maximum(abs.(n1 .* x .+ n2 .* y .+ n3 .* z .+ d)) < 1.0e-10
        @test x ≈ o_x .+ dir_x .* t atol = 1.0e-12

        # shared-origin form agrees with the per-ray form at a replicated origin
        origin = SVector(0.4, -0.7, 0.1)
        xs, ys, zs, ts = ray_plane_intersection(plane, origin, dir_x, dir_y, dir_z)
        xp, yp, zp, tp = ray_plane_intersection(
            plane, fill(origin[1], N), fill(origin[2], N), fill(origin[3], N),
            dir_x, dir_y, dir_z
        )
        @test xs ≈ xp atol = 1.0e-12
        @test ts ≈ tp atol = 1.0e-12
    end

    @testset "Float32 SoA: eltype preserved; parallel rays poison, never throw" begin
        plane = Plane3D(Point3(0.0f0, 0.0f0, 1.0f0), Vec3(0.0f0, 0.0f0, 1.0f0))
        origin = Point3(0.0f0, 0.0f0, 0.0f0)
        dir_x = Float32[0.0, 1.0]
        dir_y = Float32[0.0, 0.0]
        dir_z = Float32[1.0, 0.0]     # second ray parallel to the plane
        x, y, z, t = ray_plane_intersection(plane, origin, dir_x, dir_y, dir_z)
        @test eltype(t) === Float32
        @test t[1] ≈ 1.0f0
        @test !isfinite(t[2])         # parallel ray: ±Inf/NaN, masked by callers
    end

    @testset "the plane is a non-differentiable constant (extension rule)" begin
        plane = Plane3D(Point3(0.0, 0.0, 2.0), Vec3(0.0, 0.0, 1.0))
        out = ChainRulesCore.rrule(Primitives._plane_constants, plane)
        @test out !== nothing
        val, pb = out
        @test val == Primitives._plane_constants(plane)
        @test pb(one.(val))[2] isa ChainRulesCore.NoTangent
    end
    @testset "a Dual plane from Float32 coefficients builds on the device (availability-gated)" begin
        # `v[1:M]` on an SVector was a heap Vector whose conversion carries an interpolated message; Julia 1.12
        # left those string-runtime calls in any GPU kernel that builds a plane (RadiometricCamera promotes a
        # Float32 Quad to Dual on the device): InvalidIRError, an external C call. 1.13 compiled it.
        c = SVector(0.0f0, 3.0f0, 4.0f0, -10.0f0)
        @test Plane3D{Float64}(c).coeffs == SVector{4, Float64}(c) / 5
        cuda = try
            m = Base.require(Main, :CUDA)
            (m !== nothing && Base.invokelatest(getproperty(m, :functional))) ? m : nothing
        catch
            nothing
        end
        if cuda === nothing
            @test_skip cuda !== nothing      # CUDA is in the test target: no functional GPU here
        else
            x = Base.invokelatest(getproperty(cuda, :CuArray), Float32[1, 2, 3])
            out = Base.invokelatest(similar, x)
            Base.invokelatest(_dual_plane_kernel!(Base.invokelatest(getproperty(cuda, :CUDABackend)), 3), out, c, x; ndrange = 3)
            @test Base.invokelatest(Array, out) ≈ Float32[-2, -4, -6]
        end
    end
end
