# backproject_with_jacobian of a SplineLens camera and of a NoncentralSplineModel: the
# bearing of backproject, its Jacobian against central differences, tangency, NaN
# outside the area; and the analytic Jacobians the LMs read against ForwardDiff of
# their residuals, inside the area and on the lower edge's rounding slack.

@testitem "SplineLens: backproject_with_jacobian" begin
    using Test, StaticArrays, LinearAlgebra, Random
    using VisualGeometryToolkitCore: SplineLens, CameraModel, KannalaBrandt, px, backproject, backproject_with_jacobian

    src = CameraModel(300.0px, SA[320.0px, 240.0px]; lens = KannalaBrandt(-0.02))
    m = CameraModel(SplineLens, src, (640, 480); grid_size = (20, 16))
    rng = Xoshiro(2)
    for _ in 1:200
        u = SVector(40 + 560rand(rng), 40 + 400rand(rng))
        f, M = backproject_with_jacobian(m, u)
        @test f ≈ backproject(m, u) rtol = 1.0e-14
        h = 1.0e-5
        fd = hcat((backproject(m, u + SVector(h, 0)) - backproject(m, u - SVector(h, 0))) / 2h,
                  (backproject(m, u + SVector(0, h)) - backproject(m, u - SVector(0, h))) / 2h)
        @test M ≈ fd rtol = 1.0e-6
        @test norm(f' * M) < 1.0e-15
    end
    f, M = backproject_with_jacobian(m, SVector(-5.0, 10.0))
    @test all(isnan, f) && all(isnan, M)
end

@testitem "NoncentralSplineModel: backproject_with_jacobian" begin
    using Test, StaticArrays, LinearAlgebra, Random
    using VisualGeometryToolkitCore: SplineLens, NoncentralSplineModel, CameraModel, KannalaBrandt, px, with_points,
        backproject, backproject_with_jacobian

    src = CameraModel(300.0px, SA[320.0px, 240.0px]; lens = KannalaBrandt(-0.02))
    c = CameraModel(SplineLens, src, (640, 480); grid_size = (20, 16))
    gh, gw = size(c.lens.backward.controls)
    # a smooth non-central point field: the direction's Jacobian must not depend on it
    points = [SVector(1.0e-3 * sin(j / 3), 1.0e-3 * cos(i / 4), 2.0e-3 * (i + j) / (gh + gw)) for i in 1:gh, j in 1:gw]
    m = with_points(NoncentralSplineModel(src, (640, 480); grid_size = (20, 16)), points)
    d(u) = SVector{3, Float64}(backproject(m, u).direction)
    rng = Xoshiro(3)
    for _ in 1:200
        u = SVector(40 + 560rand(rng), 40 + 400rand(rng))
        f, M = backproject_with_jacobian(m, u)
        @test f ≈ d(u) rtol = 1.0e-14
        h = 1.0e-5
        fd = hcat((d(u + SVector(h, 0)) - d(u - SVector(h, 0))) / 2h, (d(u + SVector(0, h)) - d(u - SVector(0, h))) / 2h)
        @test M ≈ fd rtol = 1.0e-6
        @test norm(f' * M) < 1.0e-15
        # the central model's on the same direction grid
        @test M ≈ backproject_with_jacobian(c, u)[2] rtol = 1.0e-10
    end
    f, M = backproject_with_jacobian(m, SVector(700.0, 10.0))
    @test all(isnan, f) && all(isnan, M)
end

@testitem "spline LM Jacobians: analytic against ForwardDiff, the lower-edge slack included" begin
    using Test, StaticArrays, LinearAlgebra, Random
    using ForwardDiff: ForwardDiff
    using GeometryBasics: origin, widths
    using VisualGeometryToolkitCore: SplineLens, CameraModel, KannalaBrandt, px, spline_value
    using VisualGeometryToolkitCore.Cameras: _spline_direction_and_jacobian, _spline_direction_differential, _noncentral_residual,
        _noncentral_residual_and_jacobian, _in_backward, _lattice_coords

    # The analytic Jacobians of the two LMs' residuals (the central one's — the
    # normalized spline value, differentiated as `spline_value`, since the residual
    # itself is Float64 by design — and `_noncentral_residual`'s, itself) against ForwardDiff, at
    # 1e-10 relative: at random points of the area, and at points on the lower
    # edge's rounding slack (lattice coordinate in [1 − τ, 1), which `_in_backward`
    # images), where the residuals and the Jacobians must read the same window. The
    # residual each LM reads from its one evaluation per point is its definition's,
    # bit for bit: `normalize` of the spline value, and `_noncentral_residual`.
    src = CameraModel(300.0px, SA[320.0px, 240.0px]; lens = KannalaBrandt(-0.02))
    cam = CameraModel(SplineLens, src, (640, 480); grid_size = (20, 16))
    b = cam.lens.backward
    dctrl, area = b.controls, b.area
    rng = Xoshiro(17)
    gh, gw = size(dctrl)
    pctrl = [SVector(0.02 * randn(rng), 0.02 * randn(rng), 0.01 * randn(rng)) for _ in 1:gh, _ in 1:gw]
    o, w = origin(area), widths(area)
    interior = [SVector(o[1] + w[1] * rand(rng), o[2] + w[2] * rand(rng)) for _ in 1:200]
    # the slack band: a few ulps below the lower edge in x, in y, and in both
    slack = SVector{2, Float64}[]
    for k in (1, 2, 3), t in range(0.05, 0.95; length = 7)
        push!(slack, SVector(prevfloat(o[1], k), o[2] + t * w[2]), SVector(o[1] + t * w[1], prevfloat(o[2], k)),
              SVector(prevfloat(o[1], k), prevfloat(o[2], k)))
    end
    @test all(x -> _in_backward(cam.lens, x), slack)                      # the lens images the band
    @test all(x -> any(<(1), _lattice_coords(area, size(dctrl), x)), slack)  # …which lies below the lattice's first cell
    rel(A, B) = norm(A - B) / norm(B)
    Xs = (SVector(0.3, -0.2, 1.5), SVector(-0.6, 0.4, 0.9))
    worst_d, worst_r = 0.0, 0.0
    for x in vcat(interior, slack)
        d, J = _spline_direction_and_jacobian(dctrl, area, x)
        d2, u, J2 = _spline_direction_differential(dctrl, area, x)
        @test (d2, J2) == (d, J)
        @test u == normalize(spline_value(dctrl, area, x))
        @test d ≈ normalize(spline_value(dctrl, area, x)) rtol = 1.0e-15
        worst_d = max(worst_d, rel(J, ForwardDiff.jacobian(p -> normalize(spline_value(dctrl, area, p)), x)))
        for X in Xs
            r, Jr = _noncentral_residual_and_jacobian(dctrl, pctrl, area, x, X)
            @test r == _noncentral_residual(dctrl, pctrl, area, x, X)
            worst_r = max(worst_r, rel(Jr, ForwardDiff.jacobian(p -> _noncentral_residual(dctrl, pctrl, area, p, X), x)))
        end
    end
    @test worst_d <= 1.0e-10
    @test worst_r <= 1.0e-10
end
