# The generic models as a refinement consumes them (cameras/generic_models.jl): controls kept as given, the
# raw splines continued past the calibrated area, a window onto a block of controls that evaluates bit for
# bit as the model, the lattice as a primitive for any control type, the 4th-order quasi-interpolation
# initialisation with its border rule, and the nominal pinhole / domain a model is built with.
@testitem "generic model windows and quasi-interpolation" begin
    using Test
    using VisualGeometryToolkitCore
    using VisualGeometryToolkitCore: KannalaBrandt, Pinhole, WholePlane, RegionIntersection, GenericModelWindow, window,
        interpolated_direction, interpolated_point, nominal_intrinsics, spline_window, spline_value, spline_support,
        spline_weights, quasi_interpolant, convert_eltype, LogicalIntrinsics
    using StaticArrays
    using LinearAlgebra: normalize, norm, dot
    using Statistics: median
    using Random: Xoshiro
    import Adapt

    cam(f, l) = CameraModel(; focal = f, pp = (320.0, 240.0), lens = l)
    rng = Xoshiro(20261027)
    W, H = 640, 480
    gw, gh = 14, 11
    # a seeded grid of controls OFF the sphere (norms 0.8 … 1.2), as an optimizer in the tangent planes leaves them
    G = [Vec3{Float64}((normalize(SVector(0.1 * (gx - gw / 2), 0.1 * (gy - gh / 2), 1.0)) .* (0.8 + 0.4 * rand(rng)))...) for gy in 1:gh, gx in 1:gw]
    P = [Vec3{Float64}(1e-3 * randn(rng), 1e-3 * randn(rng), 0.0) for gy in 1:gh, gx in 1:gw]
    inside = [SVector(W * rand(rng), H * rand(rng)) for _ in 1:2000]

    @testset "controls: re-normalized by default, kept as given on request" begin
        m = CentralGenericModel(W, H, 0, 0, W - 1, H - 1, G)
        @test all(v -> abs(norm(v) - 1) < 1e-12, m.grid)
        r = CentralGenericModel(W, H, 0, 0, W - 1, H - 1, G; unit_controls = false)
        @test r.grid == G
        nc = NoncentralGenericModel(W, H, 0, 0, W - 1, H - 1, G, P; unit_controls = false)
        @test nc.directions == G && nc.points == P
        # inside the area the raw spline normalized IS backproject
        @test all(p -> (d = SVector{3}(interpolated_direction(r, p)); d / norm(d) == backproject(r, p)), inside)
        # a conversion converts the controls; it does not re-normalize them
        r32 = convert_eltype(Float32, r)
        @test r32.grid == Vec3{Float32}.(G)
        @test interpolated_direction(r32, SVector{2, Float32}(100.5, 200.25)) isa Vec3{Float32}     # Float32 stays Float32
    end

    @testset "a window evaluates bit for bit as the model" begin
        m = CentralGenericModel(W, H, 0, 0, W - 1, H - 1, G; unit_controls = false)
        nc = NoncentralGenericModel(W, H, 0, 0, W - 1, H - 1, G, P; unit_controls = false)
        rows, cols = 3:9, 4:12
        w = window(m, rows, cols)
        wn = window(nc, rows, cols)
        @test w isa GenericModelWindow && w.points === nothing && wn.points !== nothing
        n = 0
        for p in inside
            ix, iy, _, _ = spline_window(m, p)
            (first(rows) <= iy - 2 && iy + 1 <= last(rows) && first(cols) <= ix - 2 && ix + 1 <= last(cols)) || continue
            n += 1
            @test interpolated_direction(w, p) === interpolated_direction(m, p)
            @test interpolated_direction(wn, p) === interpolated_direction(nc, p)
            @test interpolated_point(wn, p) === interpolated_point(nc, p)
            @test spline_support(w, p) == spline_support(m, p) && spline_window(w, p) === spline_window(m, p)
            @test spline_weights(w, ix, iy, p) === spline_weights(m, ix, iy, p)
        end
        @test n > 200
        # replaced controls: the window is the model holding them
        D = [Vec3{Float64}((G[r, c] .* 1.01)...) for r in rows, c in cols]
        G2 = copy(G); G2[rows, cols] .= D
        m2 = CentralGenericModel(W, H, 0, 0, W - 1, H - 1, G2; unit_controls = false)
        w2 = window(m, rows, cols; directions = D)
        for p in inside
            ix, iy, _, _ = spline_window(m, p)
            (first(rows) <= iy - 2 && iy + 1 <= last(rows) && first(cols) <= ix - 2 && ix + 1 <= last(cols)) || continue
            @test interpolated_direction(w2, p) === interpolated_direction(m2, p)
        end
        @test_throws ArgumentError window(m, 0:3, 1:4)
        @test_throws DimensionMismatch window(m, 1:4, 1:4; directions = G[1:3, 1:4])
        @test_throws ArgumentError window(m, 1:4, 1:4; points = P[1:4, 1:4])
        @test_throws ArgumentError interpolated_point(w, inside[1])
        @test convert_eltype(Float32, wn).points == Vec3{Float32}.(wn.points)
        @test Adapt.adapt(Array, w).directions == w.directions && Adapt.adapt(Array, m).grid == m.grid
        @test isbits(window(m, rows, cols; directions = SMatrix{length(rows), length(cols)}(G[rows, cols])))
    end

    @testset "the lattice as a primitive; the spline continued past the area" begin
        area = Rect2{Float64}(0, 0, W, H)
        # node j (1-based) sits at cell coordinate j − 2; a LINEAR field's samples are its controls and the spline reproduces it
        # everywhere — inside, and outside the area where the outermost cells' pieces continue
        lin(x, y) = 0.3 + 0.002 * x - 0.001 * y
        hx, hy = W / (gw - 3), H / (gh - 3)
        C = [lin((gx - 2) * hx, (gy - 2) * hy) for gy in 1:gh, gx in 1:gw]
        for p in (SVector(10.0, 20.0), SVector(633.3, 471.9), SVector(-25.0, 300.0), SVector(700.0, -40.0), SVector(W, H))
            @test spline_value(C, area, p) ≈ lin(p...) atol = 1e-12
        end
        m = CentralGenericModel(W, H, 0, 0, W - 1, H - 1, G; unit_controls = false)
        for p in inside[1:50]
            ix, iy, wx, wy = spline_window(m, p)
            w, _, _ = spline_weights(m, ix, iy, p)
            @test w == SMatrix{4, 4}([wy[r] * wx[c] for r in 1:4, c in 1:4])
        end
        @test backproject(m, SVector(-1.0, 10.0)) === nothing && all(isfinite, interpolated_direction(m, SVector(-1.0, 10.0)))
        @test all(isnan, interpolated_direction(m, SVector(NaN, 10.0)))                          # never an error
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
        q = CentralGenericModel(pin, (W, H); grid_size = (gw, gh), init = :quasi_interpolate)
        @test q.grid != map(normalize, q.grid)                                     # controls kept off the sphere by default
        @test nominal_intrinsics(q) isa LogicalIntrinsics && q.domain === WholePlane()
        @test maximum(p -> acos(min(1.0, dot(backproject(q, p), backproject(pin, p)))), inside) < 1e-3
        # the outermost nodes sample the source ONE CELL OUTSIDE the area: the same grid as quasi-interpolating the samples
        # of the extended lattice t = −2 … C + 1 by hand
        hx, hy = W / (gw - 3), H / (gh - 3)
        S = [normalize(SVector{3}(backproject(pin, SVector(tx * hx, ty * hy)))) for ty in -2:(gh - 1), tx in -2:(gw - 1)]
        @test q.grid == Vec3{Float64}.(quasi_interpolant(S)[2:(end - 1), 2:(end - 1)])
        # a source that images nothing outside the area: per axis, the linear extrapolation of the two samples inward
        inarea(p) = 0 <= p[1] <= W && 0 <= p[2] <= H
        f(p) = inarea(p) ? backproject(pin, p) : nothing
        qf = CentralGenericModel(f, (W, H); grid_size = (gw, gh), init = :quasi_interpolate)
        @test all(v -> all(isfinite, v), qf.grid)
        dir(tx, ty) = normalize(SVector{3}(backproject(pin, SVector(tx * hx, ty * hy))))
        Se = copy(S)
        for (a, ty) in enumerate(-2:(gh - 1)), (b, tx) in enumerate(-2:(gw - 1))
            inarea(SVector(tx * hx, ty * hy)) && continue
            cx, cy = clamp(tx, 0, gw - 3), clamp(ty, 0, gh - 3)
            ix = tx < 0 ? 1 : (tx > gw - 3 ? -1 : 0); iy = ty < 0 ? 1 : (ty > gh - 3 ? -1 : 0)
            f0 = dir(cx, cy)
            Se[a, b] = normalize(f0 + abs(tx - cx) * (f0 - dir(cx + ix, cy)) + abs(ty - cy) * (f0 - dir(cx, cy + iy)))
        end
        @test qf.grid == Vec3{Float64}.(quasi_interpolant(Se)[2:(end - 1), 2:(end - 1)])
        @test maximum(p -> acos(min(1.0, dot(backproject(qf, p), backproject(pin, p)))), inside) < 2e-3
        # a lens whose domain ends INSIDE the area (a 180° fisheye's fold): every control finite (the centre-ward extrapolation
        # fills the nodes past the fold), the domain is the lens's image circle, and inside it the model follows the lens
        fish = cam(150.0, KannalaBrandt(-4 / (3π^2)))
        qk = CentralGenericModel(fish, (W, H); grid_size = (20, 16), init = :quasi_interpolate)
        @test all(v -> all(isfinite, v), qk.grid)
        @test qk.domain == image_circle(fish) && imaged_region(qk) isa RegionIntersection
        circ = image_circle(fish)
        lit = [p for p in inside if norm(p - SVector(320.0, 240.0)) < 0.7 * circ.a]
        @test length(lit) > 200
        errs = [acos(min(1.0, dot(backproject(qk, p), backproject(fish, p)))) for p in lit]
        @test maximum(errs) < 2e-2 && median(errs) < 2e-3                             # measured 1.6e-2 / 9.4e-4
        @test 0 < length(free_nodes(qk, imaged_region(qk))) < 20 * 16
        # a source that images nothing on a node's way to the centre: an ArgumentError naming it
        @test_throws ArgumentError CentralGenericModel(p -> nothing, (W, H); grid_size = (gw, gh), init = :quasi_interpolate)
        # the parametric-only modes refuse a function source
        @test_throws ArgumentError CentralGenericModel(f, (W, H); grid_size = (gw, gh), init = :interpolate)
        # non-central: directions as the central model's, points zero
        nq = NoncentralGenericModel(pin, (W, H); grid_size = (gw, gh), init = :quasi_interpolate)
        @test nq.directions == q.grid && all(iszero, nq.points)
    end
end
