@testitem "hoisted_helpers" begin
    import VisualGeometryToolkitCore as VgtkCore
    using Test
    using LinearAlgebra: norm, I
    using StaticArrays: SVector, SMatrix, MMatrix, MVector
    using Random: Xoshiro, randn
    using Colors: Gray, RGB
    using FixedPointNumbers: N0f8
    using VisualGeometryToolkitCore
    using VisualGeometryToolkitCore: EuclideanMap, PixelFrame, Circle, Ellipse, Line2D, Disc, OrientedDisc,
        EllipticFrame, AffineFrame, Point2, centroid, radius, radial_outline, saturation_bounds,
        best_iou_match, iou, lgamma_lanczos, digamma_stirling, trigamma_stirling, lu_solve!,
        bilinear_sample_clamped, put_pixel!, draw_ellipse!, draw_projected_ellipse!, draw_polyline!,
        RadialHomography, radial_transfer, HomographyMat
    using VisualGeometryToolkitCore.Synthetic: normalize_map, perspective_augmentation
    using GeometryBasics: Rect2
    using VisualGeometryToolkitCore: Csponds, withattrs, weights_column, check_weights, CspondNormalization,
        RadialCamera, radial_rows, center, source_map, destination_map

    @testset "EuclideanMap{2} actions on primitives and frames" begin
        E = EuclideanMap(PixelFrame{:colmap}, PixelFrame{:julia})   # pure translation by (0.5, 0.5)
        t = E.t
        @test E(Circle(Point2(1.0, 2.0), 3.0)) == Circle(Point2(1.0 + t[1], 2.0 + t[2]), 3.0)
        e = E(Ellipse(Point2(1.0, 2.0), 3.0, 2.0, 0.3))
        @test e.center == Point2(1.0 + t[1], 2.0 + t[2]) && e.a == 3.0 && e.b == 2.0 && e.θ ≈ 0.3
        d = E(Disc(Point2(1.0, 2.0), 1.5))
        @test d isa Disc && d.center == Point2(1.0 + t[1], 2.0 + t[2]) && d.σ == 1.5
        od = E(OrientedDisc(Point2(1.0, 2.0), 1.5, 0.7))
        @test od isa OrientedDisc && od.θ ≈ 0.7
        S = SMatrix{2, 2}(2.0, 0.5, 0.5, 1.0)
        ef = E(EllipticFrame(Point2(1.0, 2.0), S))
        @test ef.S ≈ S
        af = E(AffineFrame(Point2(1.0, 2.0), S))
        @test af.A ≈ S
        # broadcast over a detection list is the use case
        blobs = [Disc(Point2(0.0, 0.0), 1.0), Disc(Point2(5.0, 5.0), 2.0)]
        @test all(b -> b isa Disc, E.(blobs))
        # a rotating map conjugates the shape and rotates orientations
        R = EuclideanMap(VgtkCore.RotMatrix{2}(π / 2), SVector(0.0, 0.0))
        ef2 = R(EllipticFrame(Point2(0.0, 0.0), SMatrix{2, 2}(2.0, 0.0, 0.0, 1.0)))
        @test ef2.S ≈ SMatrix{2, 2}(1.0, 0.0, 0.0, 2.0) atol = 1.0e-12
        @test R(OrientedDisc(Point2(0.0, 0.0), 1.0, 0.0)).θ ≈ π / 2
        @test R(Ellipse(Point2(0.0, 0.0), 2.0, 1.0, 0.0)).θ ≈ π / 2
    end

    @testset "saturation_bounds / best_iou_match" begin
        @test saturation_bounds(Float64) == (1.0e-6, 1.0 - 1.0e-6)
        @test saturation_bounds(Gray{Float32}) == (1.0e-6, 1.0 - 1.0e-6)
        lo, hi = saturation_bounds(UInt8)
        @test lo == 1.0 && hi == 254.0
        lo8, hi8 = saturation_bounds(N0f8)
        @test lo8 ≈ 1 / 255 && hi8 ≈ 1 - 1 / 255
        @test saturation_bounds(zeros(Gray{N0f8}, 2, 2)) == saturation_bounds(N0f8)
        c = Circle(Point2(0.0, 0.0), 2.0)
        cands = [Circle(Point2(10.0, 0.0), 2.0), Circle(Point2(0.5, 0.0), 2.0), Circle(Point2(3.0, 0.0), 2.0)]
        idx, v = best_iou_match(c, cands)
        @test idx == 2 && v ≈ iou(c, cands[2])
        @test best_iou_match(c, Circle[]) == (0, 0.0)
        @test best_iou_match(c, [Circle(Point2(100.0, 0.0), 1.0)]) == (0, 0.0)
    end

    @testset "special functions — device-safe series" begin
        # References without SpecialFunctions: Γ(n) = (n-1)!, Γ(n + ½) in closed form,
        # ψ(n) = -γ + H_{n-1}, ψ′(n) = π²/6 - Σ_{k<n} k⁻².
        γ_em = 0.5772156649015329
        loggamma_int(n) = sum(log, 1:(n - 1); init = 0.0)
        loggamma_half(n) = Float64(log(factorial(big(2n))) - n * log(big(4)) - log(factorial(big(n))) + log(sqrt(big(π))))
        for (x, ref) in ((1.0, 0.0), (2.5, loggamma_half(2)), (7.0, loggamma_int(7)), (20.5, loggamma_half(20)), (31.0, loggamma_int(31)))
            @test lgamma_lanczos(x) ≈ ref atol = 1.0e-11 rtol = 1.0e-12
            @test lgamma_lanczos(Float32(x)) ≈ Float32(ref) atol = 1.0e-4 rtol = 2.0e-6
        end
        digamma_int(n) = -γ_em + sum(k -> 1 / k, 1:(n - 1))
        trigamma_int(n) = π^2 / 6 - sum(k -> 1 / k^2, 1:(n - 1))
        for n in (5, 12, 40)   # Stirling: asymptotic, good away from 0
            @test digamma_stirling(Float64(n)) ≈ digamma_int(n) rtol = 1.0e-6
            @test trigamma_stirling(Float64(n)) ≈ trigamma_int(n) rtol = 1.0e-3
        end
    end

    @testset "lu_solve! — in-register partial-pivot LU" begin
        rng = Xoshiro(7)
        for N in (3, 6, 10)
            A0 = rand(rng, N, N) + N * I
            b0 = rand(rng, N)
            A = MMatrix{N, N, Float64}(A0); b = MVector{N, Float64}(b0)
            @test lu_solve!(A, b) == 0
            @test norm(A0 * b - b0) < 1.0e-10
        end
        # an exactly singular pivot reports its index, LAPACK-style
        A = MMatrix{3, 3, Float64}(0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 2.0, 3.0, 4.0)
        b = MVector{3, Float64}(1.0, 2.0, 3.0)
        @test lu_solve!(A, b) == 1
        # matches Julia's dense solve
        A0 = [4.0 1.0; 2.0 3.0]; b0 = [1.0, 2.0]
        A = MMatrix{2, 2}(A0); b = MVector{2}(b0)
        lu_solve!(A, b)
        @test b ≈ A0 \ b0
    end

    @testset "bilinear_sample_clamped" begin
        img = Float32[1 2 3; 4 5 6; 7 8 9]
        H, W = Int32(3), Int32(3)
        @test bilinear_sample_clamped(Float32, img, 2.0, 2.0, H, W) == (true, 5.0f0)      # exact at integers
        @test bilinear_sample_clamped(Float32, img, 1.5, 1.5, H, W) == (true, 3.0f0)      # (1+2+4+5)/4
        @test bilinear_sample_clamped(Float32, img, 3.5, 3.5, H, W) == (true, 9.0f0)      # clamped neighbour
        @test bilinear_sample_clamped(Float32, img, 0.5, 1.0, H, W) == (false, 0.0f0)     # outside
        @test bilinear_sample_clamped(Float64, img, 1.0, 4.0, H, W) == (false, 0.0)
    end

    @testset "bilinear_sample: the half-texel chart, extended at the edges" begin
        using VisualGeometryToolkitCore: bilinear_sample
        using ForwardDiff: ForwardDiff
        using StaticArrays: SVector
        a, b, c, d = 1.0, 2.0, 4.0, 8.0
        img = [a b; c d]                                   # row 1 = (a, b): the texels at v = 1/4
        @test bilinear_sample(img, SVector(0.25, 0.25)) == a   # texel centres
        @test bilinear_sample(img, SVector(0.75, 0.25)) == b
        @test bilinear_sample(img, SVector(0.25, 0.75)) == c
        @test bilinear_sample(img, SVector(0.75, 0.75)) == d
        @test bilinear_sample(img, SVector(0.5, 0.5)) == (a + b + c + d) / 4
        @test bilinear_sample(img, SVector(0.5, 0.25)) == (a + b) / 2
        # off the centres' rectangle: the edge extends
        @test bilinear_sample(img, SVector(0.0, 0.0)) == a && bilinear_sample(img, SVector(-1.0, 2.0)) == c
        @test bilinear_sample(img, SVector(1.0, 0.5)) == (b + d) / 2
        @test bilinear_sample(img, SVector(NaN, 0.25)) == a                       # a non-finite position: the corner texel
        # hand-computed weights and their derivative
        uv = SVector(0.4, 0.6)
        fx, fy = 0.4 * 2 - 0.5, 0.6 * 2 - 0.5
        @test bilinear_sample(img, uv) ≈ (1 - fx) * (1 - fy) * a + fx * (1 - fy) * b + (1 - fx) * fy * c + fx * fy * d
        g = ForwardDiff.gradient(q -> bilinear_sample(img, q), uv)
        @test g ≈ SVector(2 * ((1 - fy) * (b - a) + fy * (d - c)), 2 * ((1 - fx) * (c - a) + fx * (d - b))) rtol = 1.0e-14
        @test ForwardDiff.gradient(q -> bilinear_sample(img, q), SVector(-0.3, 0.6))[1] == 0      # extended: flat across the edge
        # several channels: each channel is the one-channel lookup
        rgb = cat(img, 2 .* img, img .+ 1; dims = 3)
        v = bilinear_sample(rgb, Val(3), uv)
        @test v isa SVector{3, Float64}
        @test v ≈ SVector(bilinear_sample(img, uv), bilinear_sample(2 .* img, uv), bilinear_sample(img .+ 1, uv))
        @test ForwardDiff.jacobian(q -> bilinear_sample(rgb, Val(3), q), uv)[2, :] ≈ 2 .* g
        # a matrix of SVector texels is several channels too; Float32 stays Float32; no allocation
        @test bilinear_sample(map(x -> SVector(x, 2x), img), uv) ≈ SVector(1, 2) .* bilinear_sample(img, uv)
        @test bilinear_sample(Float32.(img), SVector(0.4f0, 0.6f0)) isa Float32
        alloc(im, q) = @allocated bilinear_sample(im, Val(3), q)
        alloc(rgb, uv)
        @test alloc(rgb, uv) == 0
    end

    @testset "raster overlays" begin
        img = zeros(Int, 20, 20)
        put_pixel!(img, 5, 7, 1; w = 1)         # x = col 5, y = row 7
        @test img[7, 5] == 1 && img[6, 4] == 1 && img[8, 6] == 1 && img[9, 5] == 0
        put_pixel!(img, 25, 25, 1; w = 2)      # out of bounds: no-op, no throw
        fill!(img, 0)
        draw_ellipse!(img, SVector(10.0, 10.0), SMatrix{2, 2}(3.0, 0.0, 0.0, 3.0), 1.0, 1; w = 0)
        @test img[10, 13] == 1 && img[13, 10] == 1 && img[10, 10] == 0
        fill!(img, 0)
        draw_polyline!(img, [(2.0, 2.0), (18.0, 2.0), (18.0, 18.0)], 1; w = 0)
        @test img[2, 10] == 1 && img[10, 18] == 1 && img[10, 10] == 1   # closed: the diagonal returns to (2, 2)
        @test img[5, 15] == 0                                                 # off every edge
        fill!(img, 0)
        Hm = SMatrix{3, 3, Float64}(I)
        draw_projected_ellipse!(img, Hm, SVector(10.0, 10.0), SMatrix{2, 2}(3.0, 0.0, 0.0, 3.0), 1.0, 1; w = 0)
        @test_throws MethodError draw_projected_ellipse!(img, SVector(10.0, 10.0), SMatrix{2, 2}(3.0, 0.0, 0.0, 3.0), 1.0, Hm, 1)   # map before shape
        @test img[10, 13] == 1 && img[10, 10] == 0
    end

    @testset "perspective_augmentation" begin
        rng = Xoshiro(3)
        H_wi = SMatrix{3, 3, Float64}(2.0, 0.0, 1.0e-4, 0.0, 2.0, 1.0e-4, 100.0, 100.0, 1.0)
        extent = (0.0, 0.0, 50.0, 50.0)
        H = perspective_augmentation(rng, H_wi, extent)
        @test H[3, 3] == 1.0 && all(isfinite, H)
        @test H != SMatrix{3, 3, Float64}(I)
        # zero magnitudes: exactly the identity up to normalisation
        H0 = perspective_augmentation(rng, H_wi, extent; strengthen_max = 0.0, jitter_max = 0.0)
        @test H0 ≈ SMatrix{3, 3, Float64}(I) atol = 1.0e-10
        # singular H_wi → identity (no augmentation), never a throw
        @test perspective_augmentation(rng, zeros(SMatrix{3, 3, Float64}), extent) == SMatrix{3, 3, Float64}(I)
        S = normalize_map(10.0, 20.0, 5.0, 0.0)
        @test S * SVector(10.0, 20.0, 1.0) ≈ SVector(0.0, 0.0, 1.0)
        @test all(isfinite, S)
    end

    @testset "radial_outline" begin
        # an undistorted radial model (λ = 0) reduces to the plain homography transfer
        Hm = SMatrix{3, 3, Float64}(3.0, 0.0, 0.0, 0.0, 3.0, 0.0, 10.0, 20.0, 1.0)
        m = RadialHomography(HomographyMat{Float64}(Tuple(Hm)), 0.0)
        r = Rect2(0.0, 0.0, 4.0, 2.0)
        pts = radial_outline(m, r; samples = 5)
        @test length(pts) == 20
        @test pts[1] ≈ Point2(10.0, 20.0)                  # origin corner
        @test pts[6] ≈ Point2(10.0 + 12.0, 20.0)           # (4, 0) corner
        d = radial_transfer(m, SVector(4.0, 2.0))
        @test any(p -> p ≈ Point2(d[1], d[2]), pts)
    end

    @testset "weights column: one reader, one contract" begin
        u = [SVector(1.0, 2.0), SVector(3.0, 4.0), SVector(5.0, 7.0)]
        cs = Csponds(u, u)
        @test weights_column(cs) === nothing && check_weights(weights_column(cs)) === nothing
        w = [1.0, 0.0, 2.5]
        @test check_weights(weights_column(withattrs(cs; weights = w))) === w   # zero is admissible
        for bad in (-0.5, NaN, Inf)
            @test_throws ArgumentError check_weights([1.0, bad, 2.0])
        end
    end

    @testset "radial camera frame change about the caller's centre" begin
        rng = Xoshiro(3)
        c = Point2(812.3, 604.1)
        X = [SVector(randn(rng), randn(rng)) for _ in 1:12]
        x = [SVector(c[1] + 300 * randn(rng), c[2] + 300 * randn(rng)) for _ in 1:12]
        N = CspondNormalization(RadialCamera, Csponds(X, x); center = SVector(c[1], c[2]))
        Pn = RadialCamera(SMatrix{2, 3}(0.3, -0.2, 0.9, 0.4, 0.1, -0.5), Point2(0.0, 0.0))
        Pd = inv(HomographyMap(destination_map(N))) ∘ Pn ∘ source_map(N)   # back to pixels by composition
        P = RadialCamera(radial_rows(Pd), c)                      # the rows about a datum centre
        @test center(P) === c                                     # exact: the datum
        @test radial_rows(P) == radial_rows(Pd)
        @test center(Pd) ≈ c
    end
end

@testitem "lm_chart_fallback" begin
    import VisualGeometryToolkitCore as VgtkCore
    using Test, LinearAlgebra, StaticArrays
    using VisualGeometryToolkitCore
    using VisualGeometryToolkitCore: ManifoldLmWorkspace, lmsolve_chart!, lm_retcode, check_rows_jacobian
    using FiniteDiff: FiniteDiff

    # A linearization state with rows only: the Jacobian is the ForwardDiff default.
    struct _CircleLm
        pts::Vector{SVector{2, Float64}}
    end
    function VgtkCore.residual_rows!(f, lm::_CircleLm, x)
        c = SVector(1.0 + x[1], -2.0 + x[2]); ρ = 3.0 + x[3]
        for (i, p) in enumerate(lm.pts)
            f[i] = norm(p - c) - ρ
        end
        return f
    end
    pts = [SVector(1.5 + 2.5cos(θ), -1.8 + 2.5sin(θ)) for θ in range(0, 2π; length = 13)[1:12]]
    lm = _CircleLm(pts)
    # the default is ForwardDiff, so the reference is AD-independent: central differences
    @test check_rows_jacobian(lm, SVector(0.1, 0.0, -0.2), 12; rtol = 1.0e-7, reference = (f, θ) -> FiniteDiff.finite_difference_jacobian(f, θ, Val(:central)))
    stationarity_tol = 0.05       # the caller's configuration owns it (VisualGeometryToolkitEstimation ships 0.05, [chart_lm])
    x, F, conv, iterations = lmsolve_chart!(ManifoldLmWorkspace(12, 3), lm; stationarity_tol)
    @test x ≈ SVector(0.5, 0.2, -0.5) atol = 1.0e-8
    @test F < 1.0e-20 && lm_retcode(conv) === :converged && 0 < iterations < 50
    x1, _, conv1, it1 = lmsolve_chart!(ManifoldLmWorkspace(12, 3), lm; maxit = 1, stationarity_tol)
    @test lm_retcode(conv1) === :max_iterations && it1 == 1

    # The stall postcondition: FastLevenbergMarquardt 0.1.0 reports its xtol
    # stop (2) when the damping grew until no step reduced the cost. Both
    # reproductions of FastLevenbergMarquardt.jl issue #4 are `:stalled`, not
    # `:converged`: f(x) = x − 3 with a wrong-sign Jacobian (no step descends;
    # the run ends at the start), and with the true Jacobian but a NaN wall at
    # x > 0.5 (the minimizer x = 3 is behind it; the run ends at the wall).
    struct _WrongSignLm end
    VgtkCore.residual_rows!(f, ::_WrongSignLm, x) = (f[1] = x[1] - 3.0; f)
    VgtkCore.residual_jacobian_rows!(J, ::_WrongSignLm, x) = (J[1, 1] = -1.0; J)
    struct _NanWallLm end
    VgtkCore.residual_rows!(f, ::_NanWallLm, x) = (f[1] = x[1] > 0.5 ? NaN : x[1] - 3.0; f)
    VgtkCore.residual_jacobian_rows!(J, ::_NanWallLm, x) = (J[1, 1] = 1.0; J)
    xw, Fw, cw, _ = lmsolve_chart!(ManifoldLmWorkspace(1, 1), _WrongSignLm(); stationarity_tol)
    @test cw == VgtkCore._LM_STALLED     # FLM's own code is 2 (upstream: 4); the postcondition rewrites it
    @test xw == SVector(0.0) && Fw == 9.0 && lm_retcode(cw) === :stalled
    xn, Fn, cn, _ = lmsolve_chart!(ManifoldLmWorkspace(1, 1), _NanWallLm(); stationarity_tol)
    @test 0.0 < xn[1] <= 0.5 && Fn > 6.0 && lm_retcode(cn) === :stalled
    # the same runs WITHOUT the stall are what FLM itself reports: `:converged`
    @test lm_retcode(2) === :converged && lm_retcode(4) === :converged && lm_retcode(-1) === :max_iterations
end
