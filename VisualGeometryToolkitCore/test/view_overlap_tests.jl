# view_overlap (cameras/view_overlap.jl): the solid angle every pair of a
# panorama's oriented frames shares, against closed forms (spherical caps), an
# exact spherical clip (perspective quadrilaterals) and pixel counting through
# image_warp (a Kannala–Brandt fisheye, a distorted perspective camera); and the
# clipper it stands on (primitives/operators.jl).

@testsetup module ViewOverlapSetup
    using StaticArrays, LinearAlgebra
    using VisualGeometryToolkitCore
    export camera, oriented, frames, footprint, cap, cap_overlap, quad, spherical_clip, fan

    camera(f, cx, cy, lens, w, h) =
        BoundedCamera(CameraModel(CameraCalibrationMat(@SMatrix [f 0.0 cx; 0.0 f cy; 0.0 0.0 1.0]); lens), Size2(w, h))
    # a frame of the camera, oriented by its camera → world rotation R
    oriented(cam, R) = BoundedCamera(PosedCamera(model(cam), relabel(CameraFrame, WorldFrame, RotationMap(R))), extent(cam))
    frames(cam, Rs) = [oriented(cam, R) for R in Rs]
    # a row's frame-i footprint: the shared angle over its share
    footprint(row) = row.solid_angle / row.fraction_i

    # the solid angle of a cap of half-angle α, and of two caps' intersection
    # (half-angles α, β, centres γ apart): two segment areas from arccos terms;
    # two caps past a hemisphere by their complements, smaller caps
    cap(α) = 2π * (1 - cos(α))
    function cap_overlap(α, β, γ)
        α > π / 2 && β > π / 2 && return 4π - cap(π - α) - cap(π - β) + cap_overlap(π - α, π - β, γ)
        γ >= α + β && return 0.0
        γ <= abs(α - β) && return cap(min(α, β))
        return 2 * (
            π - acos((cos(γ) - cos(α) * cos(β)) / (sin(α) * sin(β))) -
                acos((cos(β) - cos(γ) * cos(α)) / (sin(γ) * sin(α))) * cos(α) -
                acos((cos(α) - cos(γ) * cos(β)) / (sin(γ) * sin(β))) * cos(β)
        )
    end

    # a perspective camera's footprint: the great-circle quadrilateral of its corners' bearings
    function quad(cam, R)
        w, h = Float64.(Tuple(extent(cam)))
        return [SVector{3, Float64}(R * normalize(backproject(model(cam), SVector(x, y)))) for (x, y) in ((0.0, 0.0), (w, 0.0), (w, h), (0.0, h))]
    end
    # two convex spherical polygons (geodesic edges, counter-clockwise from outside)
    # intersected by their edges' hemispheres: a crossing on a geodesic is where
    # the hemisphere's linear sign changes along the chord, exactly
    function spherical_clip(P, Q)
        poly = copy(P)
        for k in eachindex(Q)
            nrm = cross(Q[k], Q[mod1(k + 1, length(Q))])
            out = SVector{3, Float64}[]
            for m in eachindex(poly)
                p, q = poly[m], poly[mod1(m + 1, length(poly))]
                sp, sq = dot(nrm, p), dot(nrm, q)
                sp >= 0 && push!(out, p)
                (sp >= 0) != (sq >= 0) && push!(out, normalize(p + (sp / (sp - sq)) * (q - p)))
            end
            poly = out
            isempty(poly) && return poly
        end
        return poly
    end
    # a convex spherical polygon's area: the fan of Van Oosterom–Strackee triangles from its mean
    function fan(ring)
        isempty(ring) && return 0.0
        a = normalize(sum(ring))
        return sum(eachindex(ring)) do k
            b, c = ring[k], ring[mod1(k + 1, length(ring))]
            2 * atan(dot(a, cross(b, c)), 1 + dot(a, b) + dot(b, c) + dot(c, a))
        end
    end
end

@testitem "view_overlap: Sutherland–Hodgman keeps a non-convex subject's winding" begin
    using Test
    using VisualGeometryToolkitCore: Point2
    using VisualGeometryToolkitCore.Primitives: _convex_clip!, _shoelace_signed_area
    P(x, y) = Point2{Float64}(x, y)
    # a U (area 7) whose two arms a band across them cuts into two squares (area 2)
    U = [P(0, 0), P(3, 0), P(3, 3), P(2, 3), P(2, 1), P(1, 1), P(1, 3), P(0, 3)]
    band = [P(-1, 2), P(4, 2), P(4, 4), P(-1, 4)]
    @test _shoelace_signed_area(_convex_clip!(copy(U), copy(band))) ≈ 2 atol = 1.0e-12
    # the subject's orientation is kept: clockwise is negative
    @test _shoelace_signed_area(_convex_clip!(reverse(U), copy(band))) ≈ -2 atol = 1.0e-12
    # a subject winding twice round the clip counts it twice
    sq = [P(-5, -5), P(5, -5), P(5, 5), P(-5, 5)]
    @test _shoelace_signed_area(_convex_clip!(vcat(sq, sq), copy(band))) ≈ 2 * 10 atol = 1.0e-9
end

@testitem "view_overlap: two spherical caps" setup = [ViewOverlapSetup] begin
    using Test, LinearAlgebra
    using VisualGeometryToolkitCore
    using VisualGeometryToolkitCore: Lens, Equidistant
    using Rotations: RotY, RotMatrix
    # an undistorted equidistant camera images a centred circle of radius f·α as a cap of half-angle α
    eq = camera(500.0, 1000.0, 1000.0, Lens(Equidistant()), 2000, 2000)
    α, γ = 0.5, 0.6
    disc = Circle(Point2(1000.0, 1000.0), 500α)
    a, b = oriented(eq, one(RotMatrix{3, Float64})), oriented(eq, RotMatrix(RotY(γ)))
    errors = map((64, 128, 256, 512, 1024)) do n
        row = view_overlap(a, b, disc, disc; boundary_points = n)
        @test (row.i, row.j) == (1, 2)
        @test row.fraction_i ≈ row.fraction_j        # equal footprints
        (footprint(row) - cap(α), row.solid_angle - cap_overlap(α, α, γ))
    end
    # the chords' sag: 1.6e-5 sr of the 0.77 sr cap and 1.0e-5 of the 0.40 sr overlap at 512 …
    @test abs(errors[4][1]) < 2.0e-5 && abs(errors[4][2]) < 1.2e-5
    # … converging as O(1/n²): a quarter per doubling
    for e in (first, last), k in 1:4
        @test 3.9 < e(errors[k]) / e(errors[k + 1]) < 4.1
    end
    # a 190° fisheye: footprints past a hemisphere, side by side and back to back
    wide = camera(300.0, 550.0, 550.0, Lens(Equidistant()), 1100, 1100)
    θ = deg2rad(95)
    rows = view_overlap(frames(wide, [RotMatrix(RotY(ψ)) for ψ in (0.0, π / 2, π)]), Circle(Point2(550.0, 550.0), 300θ))
    @test length(rows) == 3 && rows.i == [1, 1, 2] && rows.j == [2, 3, 3]
    @test footprint(rows[1]) ≈ cap(θ) rtol = 2.0e-6          # 6.83 sr, more than 2π
    @test rows[1].solid_angle ≈ cap_overlap(θ, θ, π / 2) rtol = 2.0e-6
    @test rows[2].solid_angle ≈ cap_overlap(θ, θ, π) rtol = 2.0e-5   # a 10° band round the equator: 1.09 sr
end

@testitem "view_overlap: perspective footprints are great-circle quadrilaterals" setup = [ViewOverlapSetup] begin
    using Test, LinearAlgebra, Random
    using VisualGeometryToolkitCore
    using VisualGeometryToolkitCore: Pinhole
    using Rotations: RotMatrix, AngleAxis
    # a perspective camera maps geodesics to lines: the rings' corners are exact
    # and every chord is the footprint's edge, so the overlap is exact at any n
    # — also where frame j's ring runs behind frame i (a relative turn up to 2.2 rad)
    cam = camera(700.0, 900.0, 500.0, Pinhole(), 1600, 1100)    # off-centre principal point
    rng = MersenneTwister(1)
    pairs = map(1:200) do _
        Ra = rand(rng, RotMatrix{3, Float64})
        Rb = Ra * RotMatrix(AngleAxis(2.2 * rand(rng), normalize(randn(rng, 3))...))
        row = view_overlap(oriented(cam, Ra), oriented(cam, Rb); boundary_points = 64)
        exact = max(fan(spherical_clip(quad(cam, Rb), quad(cam, Ra))), 0.0)
        (row, fan(quad(cam, Ra)), exact)
    end
    @test maximum(((row, Ωa, exact),) -> abs(row.solid_angle - exact), pairs) < 1.0e-12
    overlapping = filter(((row, Ωa, exact),) -> exact > 1.0e-3, pairs)
    @test maximum(((row, Ωa, exact),) -> abs(footprint(row) - Ωa) / Ωa, overlapping) < 1.0e-10
    @test length(overlapping) > 150
end

@testitem "view_overlap: against pixel counting through image_warp" setup = [ViewOverlapSetup] begin
    using Test, LinearAlgebra, StaticArrays
    using VisualGeometryToolkitCore
    using VisualGeometryToolkitCore: KannalaBrandt, Lens, Perspective, Equidistant, Pinhole, Additive, Radial, Tangential,
        Backprojection, WholePlane, image_warp, image_warp!, backproject_with_jacobian, pixel_mask
    using CoordinateTransformations: CoordinateTransformations, LinearMap
    using Rotations: RotMatrix, RotXYZ

    # frame j's camera as a direction → pixel map, for the warp's chain
    struct ToPixel{M} <: CoordinateTransformations.Transformation
        model::M
    end
    (t::ToPixel)(X) = (u = project(t.model, X); SVector(u[1], u[2]))

    # frame j's valid pixels warped into frame i (bilinear: an edge pixel is its
    # covered fraction), each of frame i's pixels weighted by its solid angle
    # |∂f/∂u × ∂f/∂v| from backproject_with_jacobian
    function counted(cam, circle, Ri, Rj)
        W, H = Tuple(extent(cam))
        region = imaged_region(cam, circle) ∩ valid_domain(cam)
        mask = Float64.(pixel_mask(region, W, H))
        math = ToPixel(model(cam)) ∘ LinearMap(SMatrix{3, 3, Float64}(Rj' * Ri)) ∘ Backprojection(model(cam))
        seen = image_warp!(zeros(H, W), image_warp(math, Rect(0, 0, W, H)), mask; fillvalue = 0.0)
        total = shared = 0.0
        for v in 1:H, u in 1:W
            mask[v, u] > 0 || continue
            f, M = backproject_with_jacobian(model(cam), SVector(u - 0.5, v - 0.5))
            dΩ = norm(cross(M[:, 1], M[:, 2]))
            total += dΩ
            shared += dΩ * (isfinite(seen[v, u]) ? seen[v, u] : 0.0)
        end
        return total, shared
    end

    R0 = one(RotMatrix{3, Float64})
    # a Kannala–Brandt fisheye with an image circle of 71°
    kb = camera(320.0, 400.0, 400.0, KannalaBrandt(-0.02), 800, 800)
    disc = Circle(Point2(400.0, 400.0), 380.0)
    R = RotMatrix(RotXYZ(0.3, 1.1, 0.2))
    row = view_overlap(oriented(kb, R0), oriented(kb, R), disc, disc)
    total, shared = counted(kb, disc, R0, R)
    @test footprint(row) ≈ total rtol = 2.0e-4
    @test row.solid_angle ≈ shared rtol = 2.0e-3
    # a distorted perspective camera (radial and tangential terms), its sides bent
    dl = Lens(Perspective(), Additive(Radial((-0.159, 0.048)), Tangential(0.006, -0.001)))
    dc = camera(600.0, 410.0, 300.0, dl, 800, 600)
    R = RotMatrix(RotXYZ(0.2, 0.5, -0.4))
    row = view_overlap(oriented(dc, R0), oriented(dc, R))
    total, shared = counted(dc, WholePlane(), R0, R)
    @test footprint(row) ≈ total rtol = 2.0e-4
    @test row.solid_angle ≈ shared rtol = 2.0e-3
    # a fisheye that sees behind a narrow perspective frame (its footprint holds
    # that frame's backward axis): different cameras, in either order
    pin = camera(300.0, 320.0, 240.0, Pinhole(), 640, 480)
    wide = camera(300.0, 550.0, 550.0, Lens(Equidistant()), 1100, 1100)
    c190 = Circle(Point2(550.0, 550.0), 300 * deg2rad(95))
    Rp = RotMatrix(RotXYZ(0.0, 2.0, 0.0))
    o = view_overlap(oriented(pin, Rp), oriented(wide, R0), WholePlane(), c190)
    p = view_overlap(oriented(wide, R0), oriented(pin, Rp), c190, WholePlane())
    @test o.solid_angle ≈ p.solid_angle rtol = 1.0e-7
    @test o.fraction_i ≈ p.fraction_j rtol = 1.0e-7
    total, _ = counted(pin, WholePlane(), Rp, Rp)
    @test footprint(o) ≈ total rtol = 2.0e-4
    @test 0 < o.fraction_i < 1 && o.fraction_j < o.fraction_i
end

@testitem "view_overlap: coincident, disjoint, the widest fields and refusals" setup = [ViewOverlapSetup] begin
    using Test, LinearAlgebra, StaticArrays
    using VisualGeometryToolkitCore
    using VisualGeometryToolkitCore: Pinhole, Lens, Equidistant, KannalaBrandt
    using Rotations: RotMatrix, RotY
    R(ψ) = RotMatrix(RotY(ψ))
    pin = camera(1000.0, 960.0, 540.0, Pinhole(), 1920, 1080)
    # two frames with the same rotation: each footprint clipped by itself, exactly
    rows = view_overlap(frames(pin, [R(0.3), R(0.3), R(0.3 + π / 2)]))
    @test rows[1].fraction_i ≈ 1 atol = 1.0e-12
    @test rows[1].fraction_j ≈ 1 atol = 1.0e-12
    @test rows[1].solid_angle ≈ 4asin(960 * 540 / sqrt((960^2 + 1000^2) * (540^2 + 1000^2))) rtol = 1.0e-14
    # 90° apart, 87.6° of view: no overlap
    @test rows[2].solid_angle == 0 && rows[2].fraction_j == 0
    # one frame: no pairs
    @test isempty(view_overlap(frames(pin, [R(0.0)])))
    kb = camera(320.0, 400.0, 400.0, KannalaBrandt(-0.02), 800, 800)
    disc = Circle(Point2(400.0, 400.0), 380.0)
    @test view_overlap(oriented(kb, R(1.0)), oriented(kb, R(1.0)), disc, disc).fraction_i ≈ 1 atol = 1.0e-12
    # a 352° field: measured (the fan's apex is the optical axis, not a hemisphere's centre)
    eq = camera(300.0, 1000.0, 1000.0, Lens(Equidistant()), 2000, 2000)
    θ = deg2rad(176)
    c352 = Circle(Point2(1000.0, 1000.0), 300θ)
    row = view_overlap(oriented(eq, R(0.0)), oriented(eq, R(1.0)), c352, c352)
    @test footprint(row) ≈ cap(θ) rtol = 1.0e-6
    @test row.solid_angle ≈ cap_overlap(θ, θ, 1.0) rtol = 1.0e-6
    # refused by name: a field reaching the backward axis, a camera with a
    # position, an unposed camera, mismatched lengths, too few samples
    c359 = Circle(Point2(1000.0, 1000.0), 300 * (π - 0.01))
    @test_throws "backward axis" view_overlap(oriented(eq, R(0.0)), oriented(eq, R(1.0)), c359, c359)
    posed = BoundedCamera(PosedCamera(model(pin), EuclideanMap(one(RotMatrix{3, Float64}), SVector(0.0, 0.0, 0.0))), Size2(1920, 1080))
    @test_throws "posed by an EuclideanMap" view_overlap(posed, oriented(pin, R(0.0)))
    @test_throws "unposed" view_overlap(pin, oriented(pin, R(0.0)))
    @test_throws "2 image circles for 3 cameras" view_overlap(frames(pin, [R(0.0), R(1.0), R(2.0)]), [disc, disc])
    @test_throws "at least 16" view_overlap(frames(pin, [R(0.0), R(1.0)]); boundary_points = 8)
end

@testitem "view_overlap: eleven frames of a panorama" setup = [ViewOverlapSetup] begin
    using Test, LinearAlgebra
    using VisualGeometryToolkitCore
    using VisualGeometryToolkitCore: KannalaBrandt
    using Rotations: RotMatrix, RotY
    kb = camera(500.0, 960.0, 600.0, KannalaBrandt(-0.02), 1920, 1200)
    rows = view_overlap(frames(kb, [RotMatrix(RotY(2π * k / 11)) for k in 0:10]), Circle(Point2(960.0, 600.0), 700.0))
    @test length(rows) == 55
    @test all(r -> r.i < r.j, rows)
    @test all(0 .<= rows.fraction_i .<= 1) && rows.fraction_i ≈ rows.fraction_j   # one camera: equal footprints
    # a ring of equal turns: every pair the same number of frames apart shares the same angle
    gap(r) = min(r.j - r.i, 11 - (r.j - r.i))
    for s in 1:5
        shared = [r.solid_angle for r in rows if gap(r) == s]
        @test maximum(shared) - minimum(shared) < 1.0e-9
    end
    angle(s) = first(r.solid_angle for r in rows if gap(r) == s)
    @test angle(1) > angle(2) > angle(3)
end

@testitem "RotationMap and OrientedCamera" begin
    using Test, LinearAlgebra, StaticArrays
    using VisualGeometryToolkitCore
    using VisualGeometryToolkitCore: Pinhole, KannalaBrandt, convert_eltype, to_homogeneous, rotation, transfer_ellipse,
        has_units, with_unit, mm, m, RadialCamera, Rig, HomographyMat, HomographyMap
    using CoordinateTransformations: transform_deriv
    using Rotations: RotMatrix, RotY, RotX, QuatRotation
    R1, R2 = RotMatrix(RotY(0.3)), RotMatrix(RotX(-0.2))
    M = RotationMap(R1)
    x = SVector(1.0, 2.0, 3.0)
    # a rotation about the origin: no translation, no length
    @test M(x) ≈ R1 * x
    @test fieldnames(typeof(M)) == (:R,)
    @test !has_units(M) && ustrip(mm, M) === M && with_unit(M, mm) === M
    @test ustrip.(mm, M(x .* mm)) ≈ R1 * x       # a physical point stays physical
    @test rotation(M) === M.R && transform_deriv(M, x) == SMatrix{3, 3}(R1)
    @test to_homogeneous(M) == [SMatrix{3, 3}(R1) zeros(3); 0 0 0 1]
    # the stored rotation: EuclideanMap's rule
    @test RotationMap(QuatRotation(R1)).R isa QuatRotation
    @test RotationMap(RotY(0.3)).R isa RotMatrix
    @test_throws ArgumentError RotationMap(2 * one(SMatrix{3, 3, Float64}))
    # frames: inv swaps them, ∘ checks the junction
    P = relabel(CameraFrame, WorldFrame, M)
    @test inv(P) isa RotationMap{3, Float64, <:Any, WorldFrame, CameraFrame}
    @test (inv(P) ∘ P)(x) ≈ x
    @test_throws MethodError P ∘ P
    @test (RotationMap(R2) ∘ M) isa RotationMap && (RotationMap(R2) ∘ M)(x) ≈ R2 * R1 * x
    E = EuclideanMap(R2, SVector(1.0, 0.0, -2.0))
    @test (M ∘ E) isa EuclideanMap && (M ∘ E)(x) ≈ M(E(x))
    @test (E ∘ M) isa EuclideanMap && (E ∘ M)(x) ≈ E(M(x))
    @test_throws MethodError relabel(CameraFrame, WorldFrame, E) ∘ P     # WorldFrame meets CameraFrame
    @test (relabel(WorldFrame, CameraFrame, E) ∘ P) isa EuclideanMap{3, Float64, <:Any, <:Any, CameraFrame, CameraFrame}
    # equality, eltype, show
    @test RotationMap(R1) == RotationMap(QuatRotation(R1)) || RotationMap(R1) ≈ RotationMap(QuatRotation(R1))
    @test RotationMap(R1) ≈ RotationMap(QuatRotation(R1))
    @test P != M && !(P ≈ M)
    F = convert_eltype(Float32, P)
    @test F isa RotationMap{3, Float32, <:Any, CameraFrame, WorldFrame} && F ≈ relabel(CameraFrame, WorldFrame, RotationMap(RotMatrix{3, Float32}(R1)))
    @test occursin("CameraFrame → WorldFrame", sprint(show, P))
    @test isbitstype(typeof(M))

    # an oriented camera: a PosedCamera of a RotationMap
    K = @SMatrix [500.0 0 320; 0 500.0 240; 0 0 1]
    cm = CameraModel(CameraCalibrationMat(K); lens = Pinhole())
    c = PosedCamera(cm, P)                       # a camera → world orientation, stored as its inverse
    @test c isa OrientedCamera && !(PosedCamera(cm, E) isa OrientedCamera)
    @test pose(c) == P && extrinsics(c) == inv(P)
    @test project(c, R1 * x) ≈ project(cm, x)                # a direction in the world, imaged about the origin
    @test backproject(c, SVector(100.0, 50.0)) ≈ R1 * normalize(backproject(cm, SVector(100.0, 50.0)))
    @test unproject(c, SVector(100.0, 50.0), 2.0) ≈ R1 * unproject(cm, SVector(100.0, 50.0), 2.0)
    @test is_in_front(c, Point3(R1 * x)) && !is_in_front(c, Point3(-(R1 * x)))
    @test is_in_front(c, [(R1 * x)[1]], [(R1 * x)[2]], [(R1 * x)[3]]) == [true]
    @test !has_units(c) && ustrip(m, c) === c
    @test convert_eltype(Float32, c).extrinsics isa RotationMap{3, Float32}
    # what needs a position refuses it, by name
    @test_throws "OrientedCamera" HomographyMat(c)
    @test_throws "OrientedCamera" HomographyMap(c)
    @test_throws "OrientedCamera" project(c, Circle(Point2(0.0, 0.0), 1.0))
    @test_throws "OrientedCamera" project(c, Rect(0.0, 0.0, 1.0, 1.0))
    @test_throws "OrientedCamera" transfer_ellipse(c, Circle(Point2(0.0, 0.0), 1.0))
    @test_throws "OrientedCamera" RadialCamera(c)
    @test_throws "rig member" Rig((BoundedCamera(c, Size2(640, 480)),))
end
