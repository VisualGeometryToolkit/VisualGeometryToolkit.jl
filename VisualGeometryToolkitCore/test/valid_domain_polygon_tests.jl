# valid_domain of a lens that is not radially symmetric (cameras/image_circle.jl): a star polygon through each
# azimuth's own fold, not the disc at the smallest fold (Core 0.10.10). A decentred lens (Brown–Conrady, OpenCV:
# tangential or thin-prism terms) folds at a radius that depends on the azimuth; the disc at the minimum excluded the
# pixels between it and their own azimuth's fold, which the model maps injectively (the fisheye 10 cm chrome set: 714
# dots for BC5, 31 for OCV14).

@testsnippet FoldReference begin
    using VisualGeometryToolkitCore
    using VisualGeometryToolkitCore: PolygonRegion, RegionIntersection, view_angle_limit
    using GeometryBasics: Point2, Rect2
    using LinearAlgebra: norm
    using StaticArrays

    mk(fx, fy, cx, cy, l) = CameraModel(CameraCalibrationMat(@SMatrix([fx 0.0 cx; 0.0 fy cy; 0.0 0.0 1.0])); lens = l)
    ray(θ, ψ) = SVector(sin(θ) * cos(ψ), sin(θ) * sin(ψ), cos(θ))
    # An independent reference for one ray azimuth ψ: the first fold of |project(lens, ray(θ, ψ))| on a dense θ scan
    # (20 000 samples to the perspective edge), refined by golden section on its bracket — the normalized fold point.
    function reference_fold(l, ψ; edge = π / 2 - 1.0e-3)
        r(θ) = (p = project(l, ray(θ, ψ)); all(isfinite, p) ? norm(p) : -Inf)   # an unimaged ray ends the branch
        θs = range(1.0e-4, edge; length = 20_000)
        k = findfirst(i -> !(r(θs[i + 1]) > r(θs[i])), 1:(length(θs) - 1))
        k === nothing && return SVector{2, Float64}(project(l, ray(edge, ψ)))
        a, b = θs[max(k - 1, 1)], θs[k + 1]
        g = (sqrt(5) - 1) / 2
        for _ in 1:200
            c, d = b - g * (b - a), a + g * (b - a)
            rc, rd = r(c), r(d)
            rc >= rd ? (b = d) : (a = c)
        end
        return SVector{2, Float64}(project(l, ray(a, ψ)))
    end
    pixel(cam, x) = (K = calibration_matrix(cam); Point2{Float64}(K[1, 1] * x[1] + K[1, 2] * x[2] + K[1, 3], K[2, 2] * x[2] + K[2, 3]))
    # A Brown–Conrady lens (radial k1, k2 and tangential p1, p2) whose fold radius varies by 17 % with the azimuth.
    bc = Lens(Perspective(), Additive(Radial((-0.3, 0.04)), Tangential(1.0e-2, -5.0e-3)))
    # An OpenCV-style lens: radial, tangential and thin prism.
    ocv = Lens(Perspective(), Additive(Radial((-0.35, 0.03)), Tangential(-4.0e-3, 6.0e-3), ThinPrism((3.0e-3, 1.0e-3), (-2.0e-3, 1.0e-3))))
end

@testitem "valid_domain: a decentred lens's domain reaches each azimuth's own fold" setup = [FoldReference] begin
    using Test
    for l in (bc, ocv)
        cam = mk(800.0, 800.0, 640.0, 480.0, l)
        q = valid_domain(cam)
        @test q isa PolygonRegion
        ψs = range(0, 2π; length = 3601)[1:3600]
        folds = [reference_fold(l, ψ) for ψ in ψs]
        ρ = norm.(folds)
        @test maximum(ρ) > 1.05 * minimum(ρ)                      # premise: the fold radius depends on the azimuth
        # the azimuth of the widest fold: a pixel inside it, past the smallest fold, is IN; past its own fold OUT
        k = argmax(ρ)
        x_in = 0.97 * folds[k]
        @test norm(x_in) > 1.05 * minimum(ρ)                      # premise: outside the old inscribed disc
        @test pixel(cam, x_in) ∈ q
        @test pixel(cam, 1.01 * folds[k]) ∉ q
        # on every azimuth: inside by 10⁻⁴ of the radius is in (the sampling tolerance, far below a pixel: 0.07 px
        # here), outside by 10⁻⁷ is out (the polygon never claims past the fold)
        @test all(x -> pixel(cam, (1 - 1.0e-4) * x) ∈ q, folds)
        @test all(x -> pixel(cam, (1 + 1.0e-7) * x) ∉ q, folds)
        # through the wrappers, and into the imaged region of a bounded camera
        b = BoundedCamera(cam, Size2(1400, 1400))
        @test valid_domain(b) == q
        r = imaged_region(b)
        @test r isa RegionIntersection
        @test pixel(cam, x_in) ∈ r
    end
end

@testitem "valid_domain polygon: membership, bounding box, pixel mask and containment agree" setup = [FoldReference] begin
    using Test
    using VisualGeometryToolkitCore: farthest_radius, intersects, check_region_support, dilate, Ellipse, pixel_mask
    using GeometryBasics: coordinates, Rect
    cam = mk(800.0, 800.0, 700.0, 700.0, bc)
    q = valid_domain(cam)
    W, H = 1400, 1400                                               # the polygon (radius 587–687 px) inside the sensor
    # the mask is membership at the pixel centres, and the prefilter (the inscribed and circumscribed discs) answers as
    # the crossing-number test of the bare polygon does
    m = pixel_mask(q, W, H)
    @test m == BitMatrix([Point2{Float64}(u - 0.5, v - 0.5) ∈ q.polygon for v in 1:H, u in 1:W])
    @test 0 < count(m) < W * H
    # the bounding box is the vertices' extremes, and holds every masked pixel; the mask reaches it within a pixel
    vs = coordinates(q.polygon)
    box = Rect2(q)
    @test minimum(box) ≈ Point2(minimum(first, vs), minimum(last, vs))
    @test maximum(box) ≈ Point2(maximum(first, vs), maximum(last, vs))
    us = [u - 0.5 for v in 1:H, u in 1:W][m]
    ws = [v - 0.5 for v in 1:H, u in 1:W][m]
    @test minimum(box)[1] <= minimum(us) <= minimum(box)[1] + 1 && maximum(box)[1] - 1 <= maximum(us) <= maximum(box)[1]
    @test minimum(box)[2] <= minimum(ws) <= minimum(box)[2] + 1 && maximum(box)[2] - 1 <= maximum(ws) <= maximum(box)[2]
    # the imaged region: the sensor ∩ the polygon, one mask
    b = BoundedCamera(cam, Size2(W, H))
    r = imaged_region(b)
    @test pixel_mask(r, W, H) == m                                  # the polygon lies inside the sensor here
    @test Rect2(r) == Rect2(q)
    # containment and meeting: rectangles, ellipses, the region itself
    c = Point2(700.0, 700.0)
    @test Rect(660, 660, 80, 80) ∈ q && Rect(660, 660, 80, 80) ⊆ q
    @test Rect(0, 0, W, H) ∉ q                                       # the lens folds inside the extent
    straddle = Rect(Float64(maximum(box)[1]) - 20, 690.0, 40.0, 20.0)
    @test !(straddle ⊆ q) && intersects(q, straddle) && intersects(straddle, q)
    @test !intersects(q, Rect(maximum(box)[1] + 5, 690.0, 10.0, 10.0))
    @test intersects(r, Rect(660, 660, 10, 10)) && !intersects(r, Rect(1, 1, 5, 5))
    @test Ellipse(c, 100.0, 50.0, 0.3) ⊆ q && !(Ellipse(c, 900.0, 50.0, 0.3) ⊆ q)
    @test q ⊆ q && r ⊆ q && q ⊆ Rect(-1.0e4, -1.0e4, 2.0e4, 2.0e4) && !(q ⊆ Rect(660, 660, 80, 80))
    @test q ⊆ Ellipse(c, 2000.0, 2000.0, 0.0) && !(q ⊆ Ellipse(c, 300.0, 300.0, 0.0))
    @test check_region_support(cam, r; who = "test") === r            # a fit over the camera's own imaged region
    @test check_region_support(cam, Rect(660, 660, 80, 80); who = "test") isa Rect
    @test_throws VisualGeometryToolkitCore.RegionBeyondSupport check_region_support(cam, Rect(0, 0, W, H); who = "test")
    # how far it reaches from the principal point: its farthest vertex
    @test farthest_radius(r, c) ≈ maximum(v -> norm(v - c), vs) rtol = 1.0e-6
    # dilated about its centre
    d = dilate(q, 1.5)
    @test d isa PolygonRegion && Rect2(d) ⊇ Rect2(q)
    v1 = vs[1]
    @test c + 1.4 * (v1 - c) ∈ d && c + 1.4 * (v1 - c) ∉ q
    # equal domains compare equal (two calls build two polygons)
    @test valid_domain(cam) == q && imaged_region(b) == r
    # a polygon that is not star-shaped about its centre is refused
    notched = [Point2(0.0, 0.0), Point2(4.0, 0.0), Point2(4.0, 4.0), Point2(2.0, 2.0), Point2(0.0, 4.0)]
    @test PolygonRegion(notched, Point2(1.0, 0.5)) isa PolygonRegion          # star-shaped about a point it sees whole
    @test_throws ArgumentError PolygonRegion(notched, Point2(3.5, 1.0))       # the notch hides an edge from it
    @test_throws ArgumentError PolygonRegion(notched, Point2(9.0, 9.0))       # outside
end

@testitem "valid_domain polygon: a radially symmetric lens keeps its exact disc, which the polygon reproduces" setup = [FoldReference] begin
    using Test
    using VisualGeometryToolkitCore: Ellipse, KannalaBrandt, shape_matrix
    for l in (KannalaBrandt(-0.1), Lens(Perspective(), Radial((-0.2,))))
        cam = mk(900.0, 600.0, 640.0, 480.0, l)
        e = valid_domain(cam)
        @test e isa Ellipse{Float64}                                # exactly the disc as before (bit for bit)
        p = VisualGeometryToolkitCore.Cameras._fold_polygon(cam)     # the polygon the asymmetric rule would build
        @test p isa PolygonRegion
        # within the sampling tolerance: every point off the ellipse's boundary by more than 10⁻⁴ of its radius is
        # classified alike
        M = inv(SMatrix{2, 2, Float64}(shape_matrix(e)))
        form(x) = (d = SVector{2, Float64}(x[1] - 640.0, x[2] - 480.0); sqrt(d' * M * d))
        pts = [Point2(640.0 + 1200 * s * cos(t), 480.0 + 800 * s * sin(t)) for s in range(0.5, 1.5; length = 401) for t in range(0, 2π; length = 181)]
        far = filter(x -> abs(form(x) - 1) > 1.0e-4, pts)
        @test all(x -> (x ∈ p) == (x ∈ e), far)
        @test all(x -> x ∉ p, filter(x -> form(x) > 1, pts))         # never past the disc
    end
end

@testitem "valid_domain polygon: a footprint for view_overlap" setup = [FoldReference] begin
    using Test
    using Rotations: RotY
    # the imaged region of a decentred lens holds the star polygon; view_overlap rings it (the polygon as the clip's
    # subject, the sensor rect as its convex clip)
    cam = BoundedCamera(mk(800.0, 800.0, 700.0, 700.0, bc), Size2(1400, 1200))   # the polygon cut by the sensor's rows
    oriented(R) = BoundedCamera(PosedCamera(model(cam), relabel(CameraFrame, WorldFrame, RotationMap(R))), extent(cam))
    same = view_overlap(oriented(RotY(0.0)), oriented(RotY(0.0)); boundary_points = 64)
    @test same.fraction_i ≈ 1 atol = 1.0e-9
    @test same.fraction_j ≈ 1 atol = 1.0e-9
    turned = view_overlap(oriented(RotY(0.0)), oriented(RotY(0.3)); boundary_points = 64)
    @test 0 < turned.fraction_i < 1
    @test turned.fraction_i ≈ turned.fraction_j rtol = 1.0e-9                  # two frames of one camera
end
