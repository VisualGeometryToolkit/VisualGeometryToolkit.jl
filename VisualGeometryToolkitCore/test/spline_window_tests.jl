# A spline lens as a refinement consumes it (cameras/spline_lens.jl): controls kept as given, the raw backward spline
# continued past its area, a window onto a block of controls that is a camera of its own and evaluates as the camera,
# the lattice as a primitive for any control type, and the 4th-order quasi-interpolation initialisation with its
# border rule.
@testitem "spline windows and quasi-interpolation" begin
    using Test
    using VisualGeometryToolkitCore
    using VisualGeometryToolkitCore: KannalaBrandt, Pinhole, SplineLens, ChartSpline, NoncentralSplineModel, window,
        spline_window, spline_value, quasi_interpolant, convert_eltype, LogicalIntrinsics, CameraCalibrationMat, calibration_matrix,
        window_offset, directions, grid_size
    using VisualGeometryToolkitCore: normalized_point
    using StaticArrays
    using LinearAlgebra: normalize, norm, dot
    using Statistics: median
    using Random: Xoshiro
    import Adapt

    cam(f, l) = CameraModel(; focal = f, pp = (320.0, 240.0), lens = l)
    rng = Xoshiro(20261027)
    W, H = 640, 480
    gw, gh = 14, 11
    K = LogicalIntrinsics(CameraCalibrationMat(SMatrix{3, 3}(300.0, 0, 0, 0, 300.0, 0, 320.0, 240.0, 1.0)))
    area = Rect2{Float64}(-320 / 300, -240 / 300, W / 300, H / 300)          # K⁻¹ of the pixel rectangle [0, W] × [0, H]
    # a seeded grid of controls OFF the sphere (norms 0.8 … 1.2), as an optimizer in the tangent planes leaves them
    G = [SVector{3, Float64}((normalize(SVector(0.1 * (gx - gw / 2), 0.1 * (gy - gh / 2), 1.0)) .* (0.8 + 0.4 * rand(rng)))...) for gy in 1:gh, gx in 1:gw]
    P = [SVector{3, Float64}(1e-3 * randn(rng), 1e-3 * randn(rng), 0.0) for gy in 1:gh, gx in 1:gw]
    inside = [SVector(W * rand(rng), H * rand(rng)) for _ in 1:2000]
    lensed(G) = CameraModel(K, SplineLens(ChartSpline(area, G)))
    raw(c, p) = c.lens.backward(normalized_point(c.intrinsics, p))           # the backward spline, continued
    imaged(c, p) = all(isfinite, backproject(c, p))
    imaged(m::NoncentralSplineModel, p) = all(isfinite, backproject(m, p).direction)

    @testset "controls: kept as given; unit from a source by default" begin
        r = lensed(G)
        @test r.lens.backward.controls == G
        nc = NoncentralSplineModel(K, ChartSpline(area, G), ChartSpline(area, P))
        @test nc.directions.controls == G && nc.points.controls == P
        @test all(v -> abs(norm(v) - 1) < 1e-12, CameraModel(SplineLens, cam(300.0, Pinhole()), Size2(W, H); grid_size = (gw, gh)).lens.backward.controls)
        # inside the area the raw spline normalized IS backproject
        @test all(p -> (d = raw(r, p); d / norm(d) == backproject(r, p)), inside)
        # a conversion converts the controls; it does not re-normalize them
        r32 = convert_eltype(Float32, r)
        @test r32.lens.backward.controls == SVector{3, Float32}.(G)
        @test raw(r32, SVector{2, Float32}(100.5, 200.25)) isa SVector{3, Float32}     # Float32 stays Float32
    end

    @testset "a window is the camera of its block, and evaluates as the camera" begin
        m = lensed(G)
        nc = NoncentralSplineModel(K, ChartSpline(area, G), ChartSpline(area, P))
        rows, cols = 3:9, 4:12
        w = window(m, rows, cols)
        wn = window(nc, rows, cols)
        @test w isa CameraModel{<:Any, <:SplineLens} && wn isa NoncentralSplineModel
        @test w.intrinsics === m.intrinsics && w.lens.backward.controls == G[rows, cols]
        # the window's corner in the full grid, read off the two lattices; its own grid size
        @test window_offset(w, m) == (2, 3) && window_offset(wn, nc) == (2, 3) && window_offset(m, m) == (0, 0)
        @test directions(w) == directions(m)[rows, cols] && grid_size(w) == (length(cols), length(rows)) && grid_size(m) == (gw, gh)
        @test_throws ArgumentError window_offset(m, w)
        n = 0
        for p in inside
            x = normalized_point(m.intrinsics, p)
            ix, iy, _, _ = spline_window(area, size(G), x)
            (first(rows) <= iy - 2 && iy + 1 <= last(rows) && first(cols) <= ix - 2 && ix + 1 <= last(cols)) || continue
            n += 1
            @test imaged(w, p) && imaged(wn, p)
            @test backproject(w, p) ≈ backproject(m, p) atol = 1.0e-12
            @test raw(w, p) ≈ raw(m, p) atol = 1.0e-12
            @test backproject(wn, p).origin ≈ backproject(nc, p).origin atol = 1.0e-12
            @test backproject(wn, p).direction ≈ backproject(nc, p).direction atol = 1.0e-12
            # the window's lattice is the block's: the same window, its index shifted by the block's corner
            jx, jy, _, _ = spline_window(w.lens.backward.area, size(w.lens.backward.controls), x)
            @test (jx, jy) == (ix - (first(cols) - 1), iy - (first(rows) - 1))
        end
        @test n > 200
        # replaced controls: the window is the camera holding them
        D = [SVector{3, Float64}((G[r, c] .* 1.01)...) for r in rows, c in cols]
        G2 = copy(G); G2[rows, cols] .= D
        m2 = lensed(G2)
        w2 = window(m, rows, cols; directions = D)
        for p in inside
            ix, iy, _, _ = spline_window(area, size(G), normalized_point(m.intrinsics, p))
            (first(rows) <= iy - 2 && iy + 1 <= last(rows) && first(cols) <= ix - 2 && ix + 1 <= last(cols)) || continue
            @test raw(w2, p) ≈ raw(m2, p) atol = 1.0e-12
        end
        @test_throws ArgumentError window(m, 0:3, 1:4)
        @test_throws ArgumentError window(m, 1:3, 1:4)                     # a block of fewer than 4 × 4 controls
        @test_throws DimensionMismatch window(m, 1:4, 1:4; directions = G[1:3, 1:4])
        @test_throws MethodError window(m, 1:4, 1:4; points = P[1:4, 1:4])   # points belong to a non-central model's window
        @test convert_eltype(Float32, wn).points.controls == SVector{3, Float32}.(P[rows, cols])
        @test Adapt.adapt(Array, w).lens.backward.controls == w.lens.backward.controls
        @test isbits(window(m, rows, cols; directions = SMatrix{length(rows), length(cols)}(G[rows, cols])))
    end

    @testset "a pixel whose support leaves the block is not imaged by the window" begin
        # a 20 × 16 grid, the 4 × 4 block 5:8 × 5:8 (one window index), a pixel far off it: not imaged — never a
        # read past the block
        Kp = LogicalIntrinsics(CameraCalibrationMat(SMatrix{3, 3}(300.0, 0, 0, 0, 300.0, 0, 320.0, 240.0, 1.0)))
        m = CameraModel(SplineLens, CameraModel(Kp), Size2(640, 480); grid_size = (20, 16))
        nc = NoncentralSplineModel(CameraModel(Kp), Size2(640, 480); grid_size = (20, 16))
        far = SVector(600.0, 400.0)
        for w in (window(m, 5:8, 5:8; directions = fill(SVector(0.0, 0.0, 1.0), 4, 4)), window(m, 5:8, 5:8))
            @test all(isnan, backproject(w, far))
        end
        @test all(isnan, backproject(window(nc, 5:8, 5:8), far).direction)
        # the one pixel cell the block covers still evaluates as the camera
        w = window(m, 5:8, 5:8)
        hx, hy = 640 / 17, 480 / 13                                        # cell size: (W, H) over (gw − 3, gh − 3) cells
        inblock = SVector(4.5 * hx, 4.5 * hy)                              # cell (4.5, 4.5): the window (ix, iy) = (7, 7) of controls 5:8
        @test backproject(w, inblock) ≈ backproject(m, inblock) atol = 1.0e-12
        @test imaged(m, far)                                               # the camera itself images the pixel
    end

    @testset "the lattice as a primitive; the spline continued past the area" begin
        parea = Rect2{Float64}(0, 0, W, H)
        # node j (1-based) sits at cell coordinate j − 2; a LINEAR field's samples are its controls and the spline reproduces it
        # everywhere — inside, and outside the area where the outermost cells' pieces continue
        lin(x, y) = 0.3 + 0.002 * x - 0.001 * y
        hx, hy = W / (gw - 3), H / (gh - 3)
        C = [lin((gx - 2) * hx, (gy - 2) * hy) for gy in 1:gh, gx in 1:gw]
        for p in (SVector(10.0, 20.0), SVector(633.3, 471.9), SVector(-25.0, 300.0), SVector(700.0, -40.0), SVector(W, H))
            @test spline_value(C, parea, p) ≈ lin(p...) atol = 1e-12
        end
        # an integer pixel reads as the same float pixel
        @test spline_value(C, Rect2(0.5, 0.5, 100, 100), SVector(10, 10)) == spline_value(C, Rect2(0.5, 0.5, 100, 100), SVector(10.0, 10.0))
        m = lensed(G)
        @test all(isnan, backproject(m, SVector(-1.0, 10.0))) && all(isfinite, raw(m, SVector(-1.0, 10.0)))
        @test all(isnan, raw(m, SVector(NaN, 10.0)))                                              # never an error
    end

    @testset "quasi_interpolant: the 4th-order stencil" begin
        cub(t) = 0.2 + 0.1t - 0.03t^2 + 0.004t^3
        f = [cub(Float64(j)) for i in 1:1, j in 1:12]
        c = quasi_interpolant(f)
        # interior controls reproduce the cubic at the nodes: (c₋ + 4c + c₊) / 6 = f
        @test all(abs((c[1, j - 1] + 4c[1, j] + c[1, j + 1]) / 6 - f[1, j]) < 1e-13 for j in 3:10)
        @test c[1, 1] == (-f[1, 1] + 8f[1, 1] - f[1, 2]) / 6                                     # the border sample replicated
    end

    @testset "init = :quasi_interpolate and its border rule" begin
        pin = cam(300.0, Pinhole())
        q = CameraModel(SplineLens, pin, Size2(W, H); grid_size = (gw, gh), init = :quasi_interpolate)
        Q = q.lens.backward.controls
        @test Q != map(normalize, Q)                                               # controls kept off the sphere by default
        @test calibration_matrix(q) == calibration_matrix(pin)                    # a camera source keeps its K
        @test maximum(p -> acos(min(1.0, dot(backproject(q, p), backproject(pin, p)))), inside) < 1e-3
        # the outermost nodes sample the source ONE CELL OUTSIDE the area: the same grid as quasi-interpolating the samples
        # of the extended lattice t = −2 … C + 1 by hand
        hx, hy = W / (gw - 3), H / (gh - 3)
        S = [normalize(SVector{3}(backproject(pin, SVector(tx * hx, ty * hy)))) for ty in -2:(gh - 1), tx in -2:(gw - 1)]
        @test Q == SVector{3, Float64}.(quasi_interpolant(S)[2:(end - 1), 2:(end - 1)])
        # a source that images nothing outside the area: per axis, the linear extrapolation of the two samples inward
        inarea(p) = 0 <= p[1] <= W && 0 <= p[2] <= H
        f(p) = inarea(p) ? backproject(pin, p) : nothing
        qf = CameraModel(SplineLens, f, Size2(W, H); grid_size = (gw, gh), init = :quasi_interpolate)
        @test all(v -> all(isfinite, v), qf.lens.backward.controls)
        dir(tx, ty) = normalize(SVector{3}(backproject(pin, SVector(tx * hx, ty * hy))))
        Se = copy(S)
        for (a, ty) in enumerate(-2:(gh - 1)), (b, tx) in enumerate(-2:(gw - 1))
            inarea(SVector(tx * hx, ty * hy)) && continue
            cx, cy = clamp(tx, 0, gw - 3), clamp(ty, 0, gh - 3)
            ix = tx < 0 ? 1 : (tx > gw - 3 ? -1 : 0); iy = ty < 0 ? 1 : (ty > gh - 3 ? -1 : 0)
            f0 = dir(cx, cy)
            Se[a, b] = normalize(f0 + abs(tx - cx) * (f0 - dir(cx + ix, cy)) + abs(ty - cy) * (f0 - dir(cx, cy + iy)))
        end
        @test qf.lens.backward.controls == SVector{3, Float64}.(quasi_interpolant(Se)[2:(end - 1), 2:(end - 1)])
        @test maximum(p -> acos(min(1.0, dot(backproject(qf, p), backproject(pin, p)))), inside) < 2e-3
        # a lens whose domain ends INSIDE the area (a 180° fisheye's fold): every control finite (the centre-ward extrapolation
        # fills the nodes past the fold), and inside the fold the camera follows the lens. Where light falls (the image
        # circle) is the bound's, supplied, not the lens's
        fish = cam(150.0, KannalaBrandt(-4 / (3π^2)))
        circ = Ellipse(Point2(320.0, 240.0), 150π / 3, 150π / 3, 0.0)                  # its image circle, supplied
        qk = CameraModel(SplineLens, fish, Size2(W, H); grid_size = (20, 16), init = :quasi_interpolate)
        @test all(v -> all(isfinite, v), qk.lens.backward.controls)
        lit = [p for p in inside if norm(p - SVector(320.0, 240.0)) < 0.7 * circ.a]
        @test length(lit) > 200
        errs = [acos(min(1.0, dot(backproject(qk, p), backproject(fish, p)))) for p in lit]
        @test maximum(errs) < 2e-2 && median(errs) < 2e-3                             # measured 1.6e-2 / 9.4e-4
        @test 0 < length(free_nodes(qk, imaged_region(BoundedCamera(qk, Size2(W, H); lens_coverage = circ)))) < 20 * 16
        # a source that images nothing on a node's way to the centre: an ArgumentError naming it
        @test_throws ArgumentError CameraModel(SplineLens, p -> nothing, Size2(W, H); grid_size = (gw, gh), init = :quasi_interpolate)
        # the parametric-only modes refuse a function source
        @test_throws ArgumentError CameraModel(SplineLens, f, Size2(W, H); grid_size = (gw, gh), init = :interpolate)
        # non-central: directions as the central camera's, points zero
        nq = NoncentralSplineModel(pin, Size2(W, H); grid_size = (gw, gh), init = :quasi_interpolate)
        @test nq.directions.controls == Q && all(iszero, nq.points.controls)
    end
end
