# The public pieces VisualGeometryToolkitFeatures' continuous DoG refinement reads (Core 0.9.5): the device LM at static
# N and its termination record, the cubic B-spline weight derivatives, the 4 × 4 window sweep with second partials, the
# truncated interpolation prefilter, and Sylvester's positive-definiteness test at static N.

@testitem "static_lm: N = 3 on a bounded smooth problem; its termination record" begin
    using Test, StaticArrays, LinearAlgebra
    import VisualGeometryToolkitCore as VgtkCore
    # r(x) = A·(x − x*) + 0.1·(x − x*).^3: a root at x*, inside the box
    xs = SVector(0.3, -0.2, 0.7)
    A = SMatrix{3, 3}(2.0, 0.3, -0.1, 0.3, 1.5, 0.2, -0.1, 0.2, 1.0)
    fj = x -> (d = x - xs; (A * d + 0.1 .* d .^ 3, A + Diagonal(0.3 .* d .^ 2), :extra))
    lb = SVector(-1.0, -1.0, -1.0); ub = SVector(1.0, 1.0, 1.0); s = SVector(1.0, 1.0, 1.0)
    x, ok, J, e, rule = VgtkCore.static_lm(fj, zero(SVector{3, Float64}), lb, ub, s, 50, 1.0e-20)
    @test ok && maximum(abs, x - xs) < 1.0e-9 && e[3] === :extra && rule isa VgtkCore.StopRule
    @test VgtkCore.static_lm_termination(ok, rule) == (VgtkCore.ReturnCode.Success, rule)
    # the root outside the box: the solve ends on the bound, unconverged — MaxIters at the cap, Stalled otherwise
    x2, ok2, _, _, rule2 = VgtkCore.static_lm(fj, zero(SVector{3, Float64}), lb, SVector(1.0, 1.0, 0.5), s, 50, 1.0e-20)
    @test !ok2 && x2[3] <= 0.5
    code2, _ = VgtkCore.static_lm_termination(ok2, rule2)
    @test !VgtkCore.successful_retcode(code2)
    @test VgtkCore.static_lm_termination(false, VgtkCore.IterationCap)[1] == VgtkCore.ReturnCode.MaxIters
    @test VgtkCore.static_lm_termination(false, VgtkCore.IncrementSize)[1] == VgtkCore.ReturnCode.Stalled
    # Float32 runs in Float32; its step tolerance is the reference's in ulps of Float32 (1e-10·eps(Float32)/eps(Float64),
    # 0.05 relative), so it stops coarser than Float64
    A32 = SMatrix{3, 3, Float32}(A); xs32 = SVector{3, Float32}(xs)
    x32, _ = VgtkCore.static_lm(x -> (d = x - xs32; (A32 * d, A32)), zero(SVector{3, Float32}),
        SVector{3, Float32}(lb), SVector{3, Float32}(ub), SVector{3, Float32}(s), 50, 1.0f-10)
    @test x32 isa SVector{3, Float32} && maximum(abs, x32 - xs) < 0.05
end

@testitem "cubic_bspline_weight_derivatives, spline_window_sums: the derivatives of the weights and of the window" begin
    using Test, StaticArrays
    import VisualGeometryToolkitCore as VgtkCore
    w(u) = VgtkCore.cubic_bspline_weights(u)
    h = 1.0e-5
    for u in (0.0, 0.17, 0.5, 0.93)
        d1, d2 = VgtkCore.cubic_bspline_weight_derivatives(u)
        @test all(isapprox.(d1, (w(u + h) .- w(u - h)) ./ 2h; atol = 1.0e-8))
        @test all(isapprox.(d2, (w(u + h) .- 2 .* w(u) .+ w(u - h)) ./ h^2; atol = 1.0e-4))
        @test sum(d1) ≈ 0 atol = 1.0e-14      # a partition of unity's derivatives sum to 0
        @test sum(d2) ≈ 0 atol = 1.0e-14
    end
    @test VgtkCore.cubic_bspline_weight_derivatives(0.5f0) isa NTuple{2, NTuple{4, Float32}}
    # the window sweep: the value and partials of a cubic polynomial surface reproduced by its controls
    ctrl = [sin(0.3r) + 0.2c * r - 0.1c^2 for r in 1:4, c in 1:4]
    S(ux, uy) = VgtkCore.spline_window_sums((r, c) -> ctrl[r, c], 0.0, w(ux), w(uy), first(VgtkCore.cubic_bspline_weight_derivatives(ux)),
        first(VgtkCore.cubic_bspline_weight_derivatives(uy)))[1]
    ux, uy = 0.37, 0.61
    d1x, d2x = VgtkCore.cubic_bspline_weight_derivatives(ux); d1y, d2y = VgtkCore.cubic_bspline_weight_derivatives(uy)
    v = VgtkCore.spline_window_sums((r, c) -> ctrl[r, c], 0.0, w(ux), w(uy), d1x, d1y, d2x, d2y)
    @test length(v) == 6 && v[1] == S(ux, uy)
    @test v[2] ≈ (S(ux + h, uy) - S(ux - h, uy)) / 2h atol = 1.0e-7
    @test v[3] ≈ (S(ux, uy + h) - S(ux, uy - h)) / 2h atol = 1.0e-7
    @test v[4] ≈ (S(ux + h, uy) - 2S(ux, uy) + S(ux - h, uy)) / h^2 atol = 1.0e-3
    @test v[5] ≈ (S(ux + h, uy + h) - S(ux + h, uy - h) - S(ux - h, uy + h) + S(ux - h, uy - h)) / 4h^2 atol = 1.0e-3
    @test v[6] ≈ (S(ux, uy + h) - 2S(ux, uy) + S(ux, uy - h)) / h^2 atol = 1.0e-3
    # without the second-derivative weights: the first three, the same bits
    @test VgtkCore.spline_window_sums((r, c) -> ctrl[r, c], 0.0, w(ux), w(uy), d1x, d1y) === v[1:3]
end

@testitem "cubic_bspline_prefilter_taps: the controls interpolate the samples" begin
    using Test
    import VisualGeometryToolkitCore as VgtkCore
    K = 10
    h = VgtkCore.cubic_bspline_prefilter_taps(K)
    @test length(h) == K + 1
    @test h[1] + 2sum(h[2:end]) ≈ 1 atol = 1.0e-15
    f = [exp(-((j - 40) / 6)^2) + 0.1sin(0.4j) for j in 1:80]
    c = [sum(h[abs(k) + 1] * f[clamp(j + k, 1, 80)] for k in -K:K) for j in 1:80]
    # at an interior node the spline reads (c[j−1] + 4c[j] + c[j+1])/6
    @test maximum(abs((c[j - 1] + 4c[j] + c[j + 1]) / 6 - f[j]) for j in 20:60) < 1.0e-5
    @test_throws ArgumentError VgtkCore.cubic_bspline_prefilter_taps(-1)
end

@testitem "is_positive_definite: Sylvester's criterion at static N; the 2×2 expression unchanged" begin
    using Test, StaticArrays, LinearAlgebra
    import VisualGeometryToolkitCore as VgtkCore
    M2 = SMatrix{2, 2}(2.0, 0.5, 0.5, 1.0)
    @test VgtkCore.is_positive_definite(M2) === (M2[1, 1] > 0 && M2[1, 1] * M2[2, 2] - M2[1, 2] * M2[2, 1] > 0)
    @test VgtkCore.is_positive_definite(Matrix(M2)) && !VgtkCore.is_positive_definite(SMatrix{2, 2}(1.0, 2.0, 2.0, 1.0))
    for N in (1, 3, 4, 5), seed in 1:20
        A = SMatrix{N, N}(sin.(seed .* (1:N^2)) .+ 0.1)
        M = A * A' + 0.1I
        @test VgtkCore.is_positive_definite(M) == isposdef(Matrix(M)) == true
        Mi = M - (maximum(eigvals(Symmetric(Matrix(M)))) + 0.1) * I       # negative definite
        @test !VgtkCore.is_positive_definite(Mi)
        λ = minimum(eigvals(Symmetric(Matrix(M))))
        @test VgtkCore.is_positive_definite(M - 0.5λ * I) && !VgtkCore.is_positive_definite(M - 1.5λ * I)   # λ_min > c ⇔ M − cI ≻ 0
    end
end
