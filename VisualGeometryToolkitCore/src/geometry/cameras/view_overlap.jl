# =============================================================================
# The solid angle two views of a rotating camera share
# =============================================================================
#
# `view_overlap(cameras)`: for every pair of frames of a camera turning about
# its centre (a panorama's oriented cameras), the solid angle of the directions
# both frames image, and that angle as a fraction of each frame's footprint.
#
# It is boundary-sampled, as `overlap_area`'s `BoundarySampled` method is: each
# footprint is the ring of its imaged region's boundary, and the pair is clipped
# in frame i's image by Core's Sutherland–Hodgman (`_convex_clip!`,
# primitives/operators.jl) against the convex ring of Ω_i. Only the area differs
# from `overlap_area`: it is a solid angle on the sphere
# (`_spherical_signed_area`, beside `signed_area`), not a pixel area.

using ..VisualGeometryToolkitCore: angle_between   # the well-conditioned atan(‖u × v‖, u · v) (transforms/euclidean_map.jl)
using ..Primitives: RegionIntersection, _convex_clip!, _spherical_signed_area, signed_area

# A direction is imaged by a lens when its pixel brings it back within this angle (rad).
const _ROUND_TRIP_ANGLE = 1.0e-6
# A footprint stays at least this far (rad) from its own backward axis, the
# antipode of the fan's apex, where the triangle formula has no denominator.
const _ANTIPODE_MARGIN = 0.05
# The largest chord angle (rad) of the far circle a footprint's ring follows where frame i does not image it.
const _FAR_ARC_CHORD = deg2rad(10)
# At most this many pieces per edge of a footprint's ring when it is drawn into another frame.
const _MAX_EDGE_PIECES = 64

"""
    view_overlap(cameras; boundary_points = 512) -> StructVector
    view_overlap(a, b; boundary_points = 512) -> NamedTuple

The directions two frames of a camera turning about its centre both image,
for every pair `i < j` of `cameras`: one row
`(; i, j, solid_angle, fraction_i, fraction_j)` per pair, `K(K − 1)/2` rows
in the order `(1, 2), (1, 3), …, (2, 3), …` as a `StructVector`, so
`rows.solid_angle` is a column. `solid_angle` is the solid angle (sr) of the
intersection of the two frames' footprints on the sphere; `fraction_i` is it
over frame `i`'s own footprint, and `fraction_j` over frame `j`'s. A pair of
cameras `a`, `b` gives the one row, `i = 1`, `j = 2`. For a panorama's pair
selection: a pair worth matching shares a fraction of its view.

Each camera is a [`BoundedCamera`](@ref) of an [`OrientedCamera`](@ref), a
frame posed by its orientation alone,
`BoundedCamera(PosedCamera(model, relabel(CameraFrame, WorldFrame, RotationMap(R_k))), extent)`
for a camera → world rotation `R_k`, `d_world = R_k · d_camera`, whose model
is a [`CameraModel`](@ref) (another model, a grid one, is an `ArgumentError`).
Frames may have different cameras. A camera posed by an [`EuclideanMap`](@ref) has a
position, and an unposed one no orientation: both are an `ArgumentError`
naming the fix. Frame `k`'s footprint is the set of directions
`R_k · backproject(model, u)` over its region Ω_k,
[`imaged_region`](@ref)`(camera)`: the sensor ∩ the camera's lens coverage (its
image circle, [`lens_coverage`](@ref)) ∩ where the lens model is valid
([`valid_domain`](@ref)). Each frame carries its own lens coverage; a camera
whose whole sensor is lit gives the extent where the lens model holds, which is
not where light falls for a lens whose image circle lies inside the sensor.

The pair `(i, j)` is computed once, in frame `i`'s image:

1. Ω_i is a ring of `boundary_points` pixels along its boundary: the extent's
   edges (every corner exact), the circle's samples, and their convex
   intersection by Core's Sutherland–Hodgman.
2. P_j is frame `j`'s ring drawn into image `i`: each pixel back-projected by
   `j`, rotated by `R_iᵀ R_j` and projected by `i`, each edge split into
   geodesic pieces no longer in image `i` than Ω_i's own edges. A direction
   frame `i` does not image (behind a perspective camera, past a lens's
   fold, NaN, or one whose pixel does not back-project to it within 10⁻⁶ rad)
   has no pixel; the ring is cut there and continued along a far circle about
   `i`'s principal point, four times Ω_i's radius, at the direction's
   azimuth, so it stays a closed curve that runs outside Ω_i wherever `i`
   cannot draw it.
3. Sutherland–Hodgman clips P_j by the convex Ω_i. A curve of any shape keeps
   its winding number inside the clip, so the clipped ring's area is the
   integral of P_j's winding number over Ω_i; pieces a non-convex P_j splits
   into come back joined by edges along Ω_i's boundary, traversed once each
   way, which add nothing.
4. That area is a solid angle: the clipped ring's vertices are back-projected
   by `i` to unit bearings and joined by geodesics, and their fan of signed
   spherical triangles from `i`'s optical axis is summed (Van Oosterom &
   Strackee, `tan(E/2) = a·(b × c) / (1 + a·b + b·c + c·a)`). Along Ω_i's
   boundary the polygon follows the footprint's own geodesic edges: a vertex
   there is moved onto its edge's great circle, and a clipped edge along a
   sensor side gets the ring's vertices between its ends (a lens bends the side).
5. The detours of item 2 are where the winding can be wrong, and only by an
   integer `n` constant over Ω_i: a detour outside Ω_i can only wind around all
   of it. One interior point of Ω_i, the candidate farthest from P_j, fixes
   `n`: its winding number against whether frame `j` images its direction
   (project by `j`, in Ω_j, back-projected within 10⁻⁶ rad). The overlap is
   the area less `n` times frame `i`'s footprint. The same correction covers a
   footprint `j` that wraps frame `i`'s backward direction (a wide fisheye
   behind a narrow lens), whose ring winds the other way round Ω_i.

Each footprint's own solid angle is the same fan over its ring from its own
optical axis, which needs no hemisphere: the sum is the enclosed area for any
apex whose antipode is outside the region, and a footprint never reaches its
camera's backward axis. So a lens past 180° (an 8 mm fisheye) is handled; a
footprint whose ring comes within 0.05 rad (≈ 3°) of its camera's backward axis
is an `ArgumentError`, as is a region whose boundary the lens cannot
back-project and a principal point outside its region.

A perspective camera maps geodesics to lines, so its footprints are exact
great-circle polygons and their overlap is exact at any `boundary_points`. A
curved boundary (an image circle, a distorted side) is a polygon of
`boundary_points` vertices, so the error is the chords' sag,
`O(1/boundary_points²)`: 1.6 × 10⁻⁵ sr of a 0.77 sr cap at 512 (2 × 10⁻⁵
relative), a quarter of it at 1024. A footprint that coincides with another (the
same camera and rotation) clips to itself: fraction 1. A pair whose footprints'
bounding caps about the optical axes are disjoint is 0 without clipping.

```julia
cam = CameraModel(K; lens = KannalaBrandt(-0.135))
bare(R) = BoundedCamera(PosedCamera(cam, relabel(CameraFrame, WorldFrame, RotationMap(R))), sensor("Sony", "IMX250"))
frame(R) = (b = bare(R); with_lens_coverage(b, image_circle(b, 6.0mm)))   # its lens coverage: the datasheet's 6 mm circle
frames = [frame(RotY(2π * k / 11)) for k in 0:10]   # 11 frames about the vertical axis
rows = view_overlap(frames)
rows[1].fraction_i                                # the share of frame 1's view frame 2 sees
```
"""
function view_overlap(cameras::AbstractVector{<:BoundedCamera}; boundary_points::Integer = 512)
    K = length(cameras)
    n = Int(boundary_points)
    n >= 16 || throw(ArgumentError("view_overlap: boundary_points = $n; at least 16 samples a boundary"))
    footprints = [_footprint(cameras[k], n) for k in 1:K]
    rows = [_overlap_row(footprints[i], footprints[j], i, j) for i in 1:K for j in (i + 1):K]
    return StructArray(isempty(rows) ? _OverlapRow[] : rows)
end
view_overlap(a::BoundedCamera, b::BoundedCamera; boundary_points::Integer = 512) =
    only(view_overlap([a, b]; boundary_points))

const _OverlapRow = @NamedTuple{i::Int, j::Int, solid_angle::Float64, fraction_i::Float64, fraction_j::Float64}

function _overlap_row(fi, fj, i::Int, j::Int)
    Ω = _shared_solid_angle(fi, fj)
    return _OverlapRow((i, j, Ω, Ω / fi.solid_angle, Ω / fj.solid_angle))
end

# -----------------------------------------------------------------------------
# One frame's footprint
# -----------------------------------------------------------------------------

# A frame's footprint: its lens model and region, its rotation, the ring of Ω's
# boundary (counter-clockwise pixels) with its camera-frame bearings, its solid
# angle, the radius of its bounding cap about the optical axis, the ring's
# longest edge (px), the principal point and K's pixel affine (the far circle's
# centre and azimuth map) and the far circle's radius.
struct _Footprint{M, G}
    model::M
    region::G
    R::SMatrix{3, 3, Float64, 9}
    ring::Vector{Point2{Float64}}
    azimuths::Vector{Float64}
    bearings::Vector{SVector{3, Float64}}
    solid_angle::Float64
    reach::Float64
    spacing::Float64
    centre::SVector{2, Float64}
    affine::SMatrix{2, 2, Float64, 4}
    far::Float64
end

# A SplineLens camera's principal point and pixel affine are its gauge K's (the pinhole at
# the pixel imaging ẑ), so the footprint is about its optical axis as for a parametric lens.
function _footprint(camera::BoundedCamera, n::Int)
    R = _orientation(camera.camera)
    m = model(camera)
    m isa CameraModel || throw(
        ArgumentError(
            "view_overlap: a frame's model is a CameraModel (its principal point, pixel affine and valid domain bound the " *
                "footprint); got $(nameof(typeof(m))) — resample a grid model's parametric source, or pose a CameraModel"
        )
    )
    _check_lens_axis(m.lens, "view_overlap")
    region = imaged_region(camera)
    ring = _region_ring(region, n)
    length(ring) >= 3 || throw(ArgumentError("view_overlap: the camera's imaged region is empty"))
    c = SVector{2, Float64}(_principal_point(m))
    Point2{Float64}(c[1], c[2]) ∈ region || throw(
        ArgumentError("view_overlap: the principal point $(Tuple(c)) lies outside the camera's imaged region")
    )
    spacing = maximum(k -> norm(ring[k] - ring[mod1(k + 1, length(ring))]), eachindex(ring))
    bearings = [_ring_bearing(m, u) for u in ring]
    θ = maximum(b -> angle_between(b, _OPTICAL_AXIS), bearings)
    θ > π - _ANTIPODE_MARGIN && throw(
        ArgumentError(
            "view_overlap: the footprint reaches $(rad2deg(θ))° from the optical axis, within " *
                "$(rad2deg(_ANTIPODE_MARGIN))° of the camera's backward axis; a field of view that wide is not measured"
        )
    )
    edge = maximum(k -> angle_between(bearings[k], bearings[mod1(k + 1, length(ring))]), eachindex(ring))
    Ω = _spherical_signed_area(_OPTICAL_AXIS, bearings)
    far = 4 * maximum(u -> norm(u - c), ring)
    affine = SMatrix{2, 2, Float64, 4}(_pixel_affine(m.intrinsics).linear)
    return _Footprint(m, region, R, ring, _ring_azimuths(ring, c), bearings, Ω, θ + edge, spacing, c, affine, far)
end

# A frame's camera → world rotation: an oriented camera's pose. A camera with a
# position, or with no pose, is not a frame of a camera turning about its centre.
_orientation(c::OrientedCamera) = SMatrix{3, 3, Float64}(pose(c).R)
_orientation(::PosedCamera) = throw(
    ArgumentError(
        "view_overlap: the camera is posed by an EuclideanMap, which gives it a position; the frames of a camera " *
            "turning about its centre are oriented cameras — `PosedCamera(model, relabel(CameraFrame, WorldFrame, RotationMap(R)))`"
    )
)
_orientation(::AbstractCamera) = throw(
    ArgumentError(
        "view_overlap: the camera is unposed; orient each frame by its camera → world rotation, " *
            "`BoundedCamera(PosedCamera(model, relabel(CameraFrame, WorldFrame, RotationMap(R))), extent)`"
    )
)

# The counter-clockwise ring of a convex region's boundary, about `n` pixels:
# a Rect's edges by the one per-edge rect walk (every corner a vertex), an
# Ellipse's `boundary_points`, an intersection clipped by Sutherland–Hodgman.
_region_ring(r::Rect{2}, n::Int) = _rect_edge_samples(identity, r, cld(n, 4))
function _region_ring(e::Ellipse, n::Int)
    ring = Point2{Float64}[Point2{Float64}(p[1], p[2]) for p in boundary_points(e, n)]
    signed_area(ring) < 0 && reverse!(ring)
    return ring
end
_region_ring(s::RegionIntersection, n::Int) = _deduplicated!(_convex_clip!(_region_ring(s.a, n), _region_ring(s.b, n)))
_region_ring(::WholePlane, ::Int) = throw(ArgumentError("view_overlap: the whole plane is not a footprint; bound the camera"))
_region_ring(r, ::Int) = throw(
    ArgumentError("view_overlap: a footprint region is a Rect, an Ellipse or their ∩ (imaged_region); got $(typeof(r))")
)

# Consecutive vertices closer than 10⁻⁹ px merged (a clip leaves a vertex on a clip edge twice).
function _deduplicated!(ring::Vector{Point2{Float64}})
    k = 1
    while length(ring) > 1 && k <= length(ring)
        if norm(ring[k] - ring[mod1(k + 1, length(ring))]) < 1.0e-9
            deleteat!(ring, mod1(k + 1, length(ring)))
        else
            k += 1
        end
    end
    return ring
end

# The unit bearing of a pixel of a region's boundary, or an ArgumentError.
function _ring_bearing(m, u)
    b = backproject(m, SVector{2, Float64}(u[1], u[2]))
    all(isfinite, b) || throw(
        ArgumentError(
            "view_overlap: the lens does not back-project the region's boundary pixel $(Tuple(u)); " *
                "bound the region by valid_domain or the image circle"
        )
    )
    return SVector{3, Float64}(normalize(b))
end

const _OPTICAL_AXIS = SVector(0.0, 0.0, 1.0)

# The signed solid angle of a ring clipped into Ω_i: a geodesic polygon of its
# vertices' back-projected bearings. A vertex on an edge of Ω_i's ring is moved
# to the nearest point of that edge's great circle (through the ring's two
# bearings), so the clipped polygon runs along the footprint's own geodesic
# edges and a footprint clipped by itself has the footprint's area exactly (a
# perspective camera's bearing is on it already).
#
# A clipped edge that runs along Ω_i's boundary across several of its ring's
# edges (the ring's vertices on a sensor side are collinear in pixels, so the
# clip leaves one long edge there) gets the ring's vertices between its ends,
# the shorter way round: for a lens that bends the side, the side's image on the
# sphere is the ring's geodesics through them, not one geodesic.
function _clipped_solid_angle(fi::_Footprint, clipped::AbstractVector{Point2{Float64}})
    isempty(clipped) && return 0.0
    n, N = length(fi.ring), length(clipped)
    edges = [_ring_edge(fi, p) for p in clipped]
    bearings = SVector{3, Float64}[]
    for a in 1:N
        p, ka = clipped[a], edges[a]
        b = _ring_bearing(fi.model, p)
        if ka !== nothing
            nrm = normalize(cross(fi.bearings[ka], fi.bearings[mod1(ka + 1, n)]))
            b = SVector{3, Float64}(normalize(b - dot(b, nrm) * nrm))    # onto the edge's great circle
        end
        push!(bearings, b)
        kb = edges[a == N ? 1 : a + 1]
        (ka === nothing || kb === nothing || ka == kb) && continue
        q = clipped[a == N ? 1 : a + 1]
        _ring_edge(fi, (p + q) / 2) === nothing && continue               # a chord through Ω_i, not along its side
        ahead, back = mod(kb - ka, n), mod(ka - kb, n)
        if ahead <= back
            foreach(s -> push!(bearings, fi.bearings[mod1(ka + s, n)]), 1:ahead)
        else
            foreach(s -> push!(bearings, fi.bearings[mod1(ka - s + 1, n)]), 1:back)
        end
    end
    return _spherical_signed_area(_OPTICAL_AXIS, bearings)
end

# The edge `k` of Ω_i's ring that pixel `p` lies on, or nothing: the edge whose
# azimuth about the principal point spans p's (the ring is convex about it), p
# within 10⁻⁹ of the far radius of its chord.
function _ring_edge(fi::_Footprint, p)
    ring, az = fi.ring, fi.azimuths
    n = length(ring)
    φ = atan(p[2] - fi.centre[2], p[1] - fi.centre[1])
    k = clamp(searchsortedlast(az, az[1] + mod(φ - az[1], 2π)), 1, n)
    a, b = ring[k], ring[k == n ? 1 : k + 1]
    e = b - a
    L = norm(e)
    L > 0 || return nothing
    abs(e[1] * (p[2] - a[2]) - e[2] * (p[1] - a[1])) / L <= 1.0e-9 * fi.far || return nothing
    return k
end

# The ring's vertex azimuths about the principal point, unwrapped: increasing from the first vertex's.
function _ring_azimuths(ring, c)
    az = [atan(v[2] - c[2], v[1] - c[1]) for v in ring]
    for k in 2:length(az)
        az[k] = az[k - 1] + mod(az[k] - az[k - 1], 2π)
    end
    return az
end

# -----------------------------------------------------------------------------
# One pair, in frame i's image
# -----------------------------------------------------------------------------

# The pixel of direction `d` (camera frame) when the lens images it, else nothing.
function _imaged_pixel(m, d::SVector{3, Float64})
    u = project(m, d)
    (isfinite(u[1]) && isfinite(u[2])) || return nothing
    b = backproject(m, SVector{2, Float64}(u[1], u[2]))
    all(isfinite, b) || return nothing
    dot(b, d) >= cos(_ROUND_TRIP_ANGLE) * norm(b) * norm(d) || return nothing
    return SVector{2, Float64}(u[1], u[2])
end

# Does frame `f` image direction `d` (its camera frame): a pixel, in its region?
_images(f::_Footprint, d::SVector{3, Float64}) =
    (u = _imaged_pixel(f.model, d); u !== nothing && Point2{Float64}(u[1], u[2]) ∈ f.region)

function _shared_solid_angle(fi::_Footprint, fj::_Footprint)
    zi, zj = fi.R[:, 3], fj.R[:, 3]
    angle_between(zi, zj) >= fi.reach + fj.reach && return 0.0
    Rij = fi.R' * fj.R                       # frame j's camera frame → frame i's
    subject = _drawn_ring(fi, fj, Rij)
    x0 = _probe_point(fi, subject)
    w = round(Int, _winding_number(subject, x0))
    d0 = SVector{3, Float64}(normalize(backproject(fi.model, x0)))
    shift = w - _images(fj, Rij' * d0)
    clipped = _convex_clip!(subject, fi.ring)     # consumes the subject; reads the clip ring only
    S = _clipped_solid_angle(fi, clipped)
    return max(S - shift * fi.solid_angle, 0.0)
end

# Frame j's ring drawn into frame i's image (item 2 of `view_overlap`'s docstring): a closed
# pixel curve, on the far circle wherever frame i does not image the direction.
function _drawn_ring(fi::_Footprint, fj::_Footprint, Rij)
    ds = [Rij * b for b in fj.bearings]
    us = [_imaged_pixel(fi.model, d) for d in ds]
    n = length(ds)
    lo = reduce((p, q) -> min.(p, q), fi.ring) .- fi.spacing    # Ω_i's bounding box, a spacing wider
    hi = reduce((p, q) -> max.(p, q), fi.ring) .+ fi.spacing
    k0 = something(findfirst(!isnothing, us), 1)
    out = Point2{Float64}[]
    far = _FarCircle(false, SVector(1.0, 0.0))
    for offset in 0:(n - 1)
        k = mod1(k0 + offset, n)
        k1 = mod1(k + 1, n)
        _draw!(out, far, fi, ds[k], us[k])
        a, b = us[k], us[k1]
        (a === nothing || b === nothing) && continue
        len = norm(b - a)
        (len > fi.spacing && all(min.(a, b) .<= hi) && all(max.(a, b) .>= lo)) || continue
        s = min(ceil(Int, len / fi.spacing), _MAX_EDGE_PIECES)
        for t in 1:(s - 1)
            d = SVector{3, Float64}(normalize((1 - t / s) * ds[k] + (t / s) * ds[k1]))
            _draw!(out, far, fi, d, _imaged_pixel(fi.model, d))
        end
    end
    # close: back from the far circle to the first vertex when the ring left it
    u0 = us[k0]
    (u0 !== nothing && far.on) && _leave_far!(out, far, fi, u0)
    return out
end

# Whether the drawn ring is on the far circle, and its direction there (the
# azimuth of the last direction frame i could not image).
mutable struct _FarCircle
    on::Bool
    dir::SVector{2, Float64}
end

# One direction onto the drawn ring: its pixel, or the far circle at its azimuth.
function _draw!(out, far::_FarCircle, fi::_Footprint, d, u)
    if u !== nothing
        far.on && _leave_far!(out, far, fi, u)
        push!(out, Point2{Float64}(u[1], u[2]))
    else
        v = fi.affine * SVector(d[1], d[2])
        dir = norm(v) > 1.0e-12 ? v / norm(v) : far.dir
        if !far.on      # leaving the image: radially out from the last pixel
            last = isempty(out) ? fi.centre + dir : SVector{2, Float64}(out[end][1], out[end][2])
            r = last - fi.centre
            far.on, far.dir = true, norm(r) > 0 ? r / norm(r) : dir
            push!(out, _far_point(fi, far.dir))
        end
        _far_arc!(out, far, fi, dir)
    end
    return out
end

# Back from the far circle to pixel `u`: along the circle to u's azimuth, then radially in.
function _leave_far!(out, far::_FarCircle, fi::_Footprint, u)
    r = u - fi.centre
    norm(r) > 0 && _far_arc!(out, far, fi, r / norm(r))
    far.on = false
    return out
end

# Along the far circle, the shorter way, from its current direction to `dir`.
function _far_arc!(out, far::_FarCircle, fi::_Footprint, dir)
    a = far.dir
    φ = atan(a[1] * dir[2] - a[2] * dir[1], dot(a, dir))
    s = max(1, ceil(Int, abs(φ) / _FAR_ARC_CHORD))
    for t in 1:s
        ψ = φ * t / s
        e = SVector(cos(ψ) * a[1] - sin(ψ) * a[2], sin(ψ) * a[1] + cos(ψ) * a[2])
        push!(out, _far_point(fi, e))
    end
    far.dir = dir
    return out
end

_far_point(fi::_Footprint, e) = (p = fi.centre + fi.far * e; Point2{Float64}(p[1], p[2]))

# An interior point of Ω_i far from the drawn ring: the principal point or one
# of eight points halfway to the boundary, whichever is farthest from it.
function _probe_point(fi::_Footprint, subject)
    c = fi.centre
    n = length(fi.ring)
    candidates = SVector{2, Float64}[c]
    for k in 0:7
        push!(candidates, c + 0.5 * (SVector{2, Float64}(fi.ring[1 + (k * n) ÷ 8]...) - c))
    end
    return argmax(p -> _distance_to_ring(subject, p), candidates)
end

function _distance_to_ring(ring, p)
    n = length(ring)
    n == 0 && return Inf
    return minimum(1:n) do k
        a = SVector{2, Float64}(ring[k]...)
        b = SVector{2, Float64}(ring[k == n ? 1 : k + 1]...)
        e = b - a
        t = clamp(dot(p - a, e) / max(dot(e, e), eps()), 0.0, 1.0)
        norm(a + t * e - p)
    end
end

# The winding number of a closed pixel ring about `p`.
function _winding_number(ring, p)
    n = length(ring)
    total = 0.0
    for k in 1:n
        a = SVector{2, Float64}(ring[k]...) - p
        b = SVector{2, Float64}(ring[k == n ? 1 : k + 1]...) - p
        total += atan(a[1] * b[2] - a[2] * b[1], dot(a, b))
    end
    return total / 2π
end
