# valid_domain of a lens that is not radially symmetric (cameras/image_circle.jl): a star polygon through each
# azimuth's own fold, not the disc at the smallest fold (Core 0.10.10). A decentred lens (Brown–Conrady, OpenCV:
# tangential or thin-prism terms) folds at a radius that depends on the azimuth; the disc at the minimum excluded the
# pixels between it and their own azimuth's fold, which the model maps injectively (the fisheye 10 cm chrome set: 714
# dots for BC5, 31 for OCV14).

@testsetup module FoldReference
    using VisualGeometryToolkitCore
    using GeometryBasics: Point2
    using LinearAlgebra: norm
    using StaticArrays
    export mk, ray, reference_fold, pixel, bc, ocv

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
    using VisualGeometryToolkitCore: PolygonRegion, RegionIntersection
    using GeometryBasics: Point2, Rect2
    using LinearAlgebra: norm
    using StaticArrays
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
    using VisualGeometryToolkitCore: PolygonRegion, RegionIntersection
    using GeometryBasics: Point2, Rect2
    using LinearAlgebra: norm
    using StaticArrays
    using VisualGeometryToolkitCore: farthest_radius, intersects, check_region_support, dilate, Ellipse, pixel_mask
    using GeometryBasics: Rect, Polygon
    cam = mk(800.0, 800.0, 700.0, 700.0, bc)
    q = valid_domain(cam)
    W, H = 1400, 1400                                               # the polygon (radius 587–687 px) inside the sensor
    # the mask is membership at the pixel centres, and the prefilter (the inscribed and circumscribed discs) answers as
    # the crossing-number test of the bare polygon does
    m = pixel_mask(q, W, H)
    @test m == BitMatrix([Point2{Float64}(u - 0.5, v - 0.5) ∈ Polygon(q) for v in 1:H, u in 1:W])
    @test 0 < count(m) < W * H
    # the bounding box is the vertices' extremes, and holds every masked pixel; the mask reaches it within a pixel
    vs = q.vertices
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
    using VisualGeometryToolkitCore: PolygonRegion, RegionIntersection
    using GeometryBasics: Point2, Rect2
    using LinearAlgebra: norm
    using StaticArrays
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
    using VisualGeometryToolkitCore: PolygonRegion, RegionIntersection
    using GeometryBasics: Point2, Rect2
    using LinearAlgebra: norm
    using StaticArrays
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

@testitem "valid_domain polygon: built once, membership is cheap (timing guard)" setup = [FoldReference] begin
    using Test
    using GeometryBasics: Point2
    using StaticArrays
    # The polygon is built once per model (about 10 ms: 720 azimuth folds) and reused; a caller holds it (a fit's
    # support, a bundle's camera variable), never rebuilds it per point. Membership is the crossing number on the stored
    # ring behind the inscribed/circumscribed discs: about 0.1–0.5 µs a point (measured 4.5 ms for 10⁴ points all in the
    # annulus between the discs, the worst case). The bounds are 10× generous; a regression to a per-point rebuild or to
    # an unfiltered test fails them.
    # Wall-clock bounds hold only uninstrumented: under coverage (CI's `Pkg.test(; coverage = true)`) every line of the
    # scan is counted, the guard records a skip.
    if Base.JLOptions().code_coverage != 0
        @test_skip "timing guard: not under coverage"
    else
        cam = mk(1500.0, 1500.0, 2128.0, 1424.0, bc)
        valid_domain(cam)                                                 # compiled
        t_build = minimum(@elapsed(valid_domain(cam)) for _ in 1:3)
        @test t_build < 0.5
        q = valid_domain(cam)
        c = q.centre
        ann = [Point2{Float64}(c + (q.inner + (q.outer - q.inner) * ((k * 0.6180339887) % 1)) * SVector(cos(k), sin(k))) for k in 1:10_000]
        count(p -> p ∈ q, ann)
        t_member = minimum(@elapsed(count(p -> p ∈ q, ann)) for _ in 1:3)
        @test t_member < 0.05                                             # 10⁴ points in the annulus, under 50 ms
        @test 0 < count(p -> p ∈ q, ann) < length(ann)
    end
end

@testitem "valid_domain polygon: a lattice operation takes it as its region" setup = [FoldReference] begin
    using Test
    using VisualGeometryToolkitCore: PolygonRegion, extend_lattice, folded_pixels
    using GeometryBasics: Rect2
    # a lattice operation reads a region by its bounding box (`extend_lattice`) and by membership (`folded_pixels`): a
    # star polygon (the valid domain of a decentred source) is such a region, as its Ellipse was until Core 0.10.9
    W, H = 1280, 960
    fish = CameraModel(; focal = 250.0, pp = (640.0, 480.0), lens = KannalaBrandt(-4 / (3π^2)))
    m = CameraModel(SplineLens, fish, Size2(W, H); grid_size = (24, 19), init = :quasi_interpolate)
    q = valid_domain(mk(400.0, 400.0, 640.0, 480.0, ocv))
    @test q isa PolygonRegion
    @test extend_lattice(m, q).offset == extend_lattice(m, Rect2(q)).offset          # the box sets the nodes
    f = folded_pixels(m, q; spacing = 16.0)
    @test f.samples > 0 && f.samples < folded_pixels(m, Rect2(q); spacing = 16.0).samples   # the samples inside it
end

@testitem "valid_domain polygon: a pixel-frame map moves it, membership through the map's inverse" setup = [FoldReference] begin
    using Test
    using VisualGeometryToolkitCore: PolygonRegion, RegionIntersection
    using Rotations: RotMatrix2
    using StaticArrays
    # Core's action of a similarity on every region (VisualGeometryToolkitIO's region body moves a region to and from the
    # file's pixel convention through it): a star polygon maps vertex by vertex, its centre with it, and stays one;
    # p ∈ S(q) ⟺ S⁻¹(p) ∈ q on a grid off its boundary
    cam = mk(800.0, 800.0, 640.0, 480.0, ocv)
    q = valid_domain(cam)
    r = imaged_region(BoundedCamera(cam, Size2(1280, 960)))
    @test q isa PolygonRegion && r isa RegionIntersection
    ps = vec([SVector(x, y) for x in -150.25:6.5:1430.0, y in -150.25:6.5:1110.0])
    for S in (SimilarityMap(EuclideanMap(PixelFrame{:colmap}, PixelFrame{:opencv})), SimilarityMap(EuclideanMap(PixelFrame{:colmap}, PixelFrame{:julia})),
              SimilarityMap(2.0, RotMatrix2(0.4), SVector(1.0, -3.0)), SimilarityMap(0.5, RotMatrix2(π / 2), SVector(7.0, 2.0)))
        Sq = S(q)
        @test Sq isa PolygonRegion && length(Sq.vertices) == length(q.vertices) && Sq.centre ≈ S(q.centre)
        Si = inv(S)
        @test [p ∈ Sq for p in ps] == [Si(p) ∈ q for p in ps]
        @test Si(Sq).vertices ≈ q.vertices
    end
    # the imaged region (a Rect ∩ the polygon) maps part by part under a pixel-frame map (a translation)
    S = SimilarityMap(EuclideanMap(PixelFrame{:colmap}, PixelFrame{:opencv}))
    Sr = S(r)
    @test Sr isa RegionIntersection && Sr.b isa PolygonRegion
    @test [p ∈ Sr for p in ps] == [inv(S)(p) ∈ r for p in ps]
end

@testitem "valid_domain polygon: a fold radius that jumps between azimuths stays a star polygon, inside every fold" begin
    using Test
    using VisualGeometryToolkitCore
    using VisualGeometryToolkitCore: PolygonRegion
    using StaticArrays
    using LinearAlgebra: norm
    # a lens type without the RadialSymmetry trait (only `project`): r = θ(1 − a θ²), a = 0.1 on the half plane
    # cos ψ > 0 (a fold at θ = √(10/3), r = 1.217) and a = 0.02 elsewhere (no fold before the sphere's edge, r = 2.52
    # there): the fold radius doubles across ψ = ±π/2. The bulge bound beside the jump exceeds the radius; the vertex
    # radius is clamped at the disc of the smallest fold (inside every fold), so the ring stays star-shaped about the
    # centre and the domain never claims past an azimuth's own fold.
    struct JumpLens <: AbstractLensModel end
    a(ψ) = cos(ψ) > 0 ? 0.1 : 0.02
    function VisualGeometryToolkitCore.project(::JumpLens, X::StaticVector{3})
        θ = atan(hypot(X[1], X[2]), X[3]); ψ = atan(X[2], X[1]); ρ = θ * (1 - a(ψ) * θ^2)
        return SVector(ρ * cos(ψ), ρ * sin(ψ))
    end
    f = 400.0
    cam = CameraModel(CameraCalibrationMat(@SMatrix([f 0.0 640.0; 0.0 f 480.0; 0.0 0.0 1.0])); lens = JumpLens())
    q = valid_domain(cam)
    @test q isa PolygonRegion
    pixel(ρ, ψ) = SVector(640.0 + f * ρ * cos(ψ), 480.0 + f * ρ * sin(ψ))
    θf = sqrt(10 / 3); ρf = θf * (1 - 0.1θf^2)                              # the folding half's fold radius
    θe = π - 1.0e-3; ρe = θe * (1 - 0.02θe^2)                              # the other half's: the sphere's edge
    ψs = range(0, 2π; length = 2001)[1:2000]
    fold(ψ) = cos(ψ) > 0 ? ρf : ρe
    @test all(ψ -> pixel((1 - 1.0e-4) * ρf, ψ) ∈ q, ψs)                    # the disc of the smallest fold is inside
    @test all(ψ -> pixel((1 + 1.0e-7) * fold(ψ), ψ) ∉ q, ψs)               # never past an azimuth's own fold
    far = filter(ψ -> cos(ψ) < -0.2, ψs)                                    # the non-folding half, away from the jump
    @test all(ψ -> pixel(0.99ρe, ψ) ∈ q, far)                              # reaches its own (far) edge there
end

@testitem "valid_domain polygon: the decentred OpenCV lenses' polygons are bit for bit Core 0.10.10's first build" begin
    import VisualGeometryToolkitCore as VgtkCore
    using Test
    using VisualGeometryToolkitCore
    using StaticArrays
    # the chain-equivalence corpus's OpenCV 5-, 8-, 12- and 14-coefficient lenses (`opencv_lens`): smooth fold
    # curves, which neither the disc floor nor the jump cap (`_FOLD_JUMP`) touches. Their polygons' vertices and centre,
    # FNV-1a over the bit patterns, are the ones the build without either gave (recorded at cc4a30f3).
    CV = [-0.21419017050355535, 0.05885963864959478, 0.0017309703615326448, -0.0016899562175219018, -0.013228653203705452,
          -0.008422246136598874, 0.0009299940131988482, -0.004808424838594422, 0.0009362676014743785, 0.002042933538859527,
          -0.0005042701635291866, -0.00011669406371264847, 0.006487455083713491, 0.005372244699677389]
    function fingerprint(q)
        h = 0xcbf29ce484222325
        for x in Iterators.flatten((Iterators.flatten((v[1], v[2]) for v in q.vertices), (q.centre[1], q.centre[2])))
            h = (h ⊻ reinterpret(UInt64, Float64(x))) * 0x100000001b3
        end
        return h
    end
    recorded = Dict(5 => 0x00751efdd7f475b5, 8 => 0xa8f41a1670485883, 12 => 0xaf8d1273cafbbf2b, 14 => 0xba739373b9a580bc)
    K = CameraCalibrationMat(@SMatrix([1200.0 0.0 2000.0; 0.0 1190.0 1500.0; 0.0 0.0 1.0]))
    for n in (5, 8, 12, 14)
        @test fingerprint(valid_domain(CameraModel(K; lens = VgtkCore.opencv_lens(CV[1:n])))) == recorded[n]
    end
end
