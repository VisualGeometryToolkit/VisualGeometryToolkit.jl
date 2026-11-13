# ==============================================================================
# Camera Utilities - Helper functions for focal length and camera positioning
# ==============================================================================

"""
    focal_length(resolution::Size2, θ::Real; dimension=:horizontal)

The focal length an angle of view implies: in pixels from the FULL angle of view (plain radians —
angles carry no dimension in the VGC unit system) and resolution. Its inverse, for a
camera, is [`angle_of_view`](@ref). The other forms: `focal_length(sensor::Sensor, θ)` (a
length) and the accessor `focal_length(intrinsics)` (`(fx, fy)` in pixels).

# Arguments
- `resolution::Size2`: Image resolution (width, height) in pixels: plain numbers or
  dimensionless quantities (`px`); a length is a `DimensionError`
- `θ::Real`: the full angle of view in radians
- `dimension::Symbol`: Which dimension the angle of view applies to (`:horizontal` or `:vertical`)

# Returns
- Focal length in pixels (same logical unit as resolution)

# Examples
```julia
focal_length(Size2(width=1280px, height=960px), 0.698)
focal_length(Size2(width=1280px, height=960px), deg2rad(40.0))
```
"""
function focal_length(resolution::Size2, θ::Real; dimension = :horizontal)
    if dimension === :horizontal
        f = 0.5 * _check_pixels(resolution.width)
    elseif dimension === :vertical
        f = 0.5 * _check_pixels(resolution.height)
    else
        throw(ArgumentError("dimension must be :horizontal or :vertical"))
    end

    return f / tan(θ / 2)
end

"""
    focal_length(sensor::Sensor, θ::Real; dimension=:horizontal)

The physical focal length from the FULL field of view angle and the sensor:
`focal_length(sensor.resolution, θ)` pixels times the pitch of that axis.

# Arguments
- `sensor::Sensor`: the sensor (its `resolution` and `pitch`)
- `θ::Real`: the full angle of view in radians
- `dimension::Symbol`: Which dimension the angle of view applies to (`:horizontal` or `:vertical`)

# Returns
- Physical focal length (a length quantity)

# Examples
```julia
s = sensor("Sony", "IMX174")
f_mm = focal_length(s, deg2rad(40.0))
```
"""
function focal_length(sensor::Sensor, θ::Real; dimension = :horizontal)
    # Get focal length in pixels
    f_px = focal_length(sensor.resolution, θ; dimension = dimension)

    # Convert to physical length using pixel pitch
    # Use width pitch for horizontal FOV, height pitch for vertical FOV
    pitch = (dimension === :horizontal) ? sensor.pitch.width : sensor.pitch.height

    # f = f_px × pitch (px × length/px = a length)
    return f_px * pitch
end

"""
    angle_of_view(camera::BoundedCamera; dimension = :horizontal) -> θ

The angle of view of a bounded camera: the FULL angle, in radians, of the scene it
IMAGES in `dimension` — its pixel extent, bounded by its [`lens_coverage`](@ref)
(where the lens covers the sensor) — read through the camera's own
[`backproject`](@ref) (`K⁻¹`, then the lens), so a distorted or fisheye lens
answers its true angle and an anisotropic `K` uses the focal length of that axis.
For a pinhole with a centred principal point it is the inverse of
[`focal_length`](@ref)`(resolution, θ; dimension)`, which takes the same full
angle (`focal_length(sensor, θ)` is its metric form). (The lens's own limit,
[`view_angle_limit`](@ref), is a HALF angle from the optical axis.) The size this
angle spans at a distance is [`field_of_view`](@ref).

The sample points, each moved in along the line from the principal point to where
the lens coverage ends when it lies past it (a circular fisheye's dark corners,
the 14 mm disc cut by a wide sensor):

- `:horizontal`: the two edge points on the principal point's row, `(0, c_y)`
  and `(width, c_y)`;
- `:vertical`: the two on its column, `(c_x, 0)` and `(c_x, height)`;
- `:diagonal`: the four corners of the extent.

The answer is twice the SMALLEST of those points' view angles — the nearer
edge when the principal point is off-centre, so the cone of that full angle
about the optical axis is imaged in that dimension. A camera whose whole sensor
is covered (the default coverage) reads the sensor's edges. A principal point
outside the coverage images nothing about the axis: an `ArgumentError`. For a
lens that is not radially symmetric the [`valid_domain`](@ref) is inscribed, so
a point between it and the true limit counts as the lens's limit: the answer can
then overstate the angle slightly.

A [`SplineLens`](@ref) camera reads about its gauge principal point, the
pixel imaging ẑ, so the angles are about the optical axis.

Clipping by the model: a point outside the lens model's [`valid_domain`](@ref)
(or one the lens does not image) counts as the lens's own limit, so when the
valid domain clips the covered area in that dimension the answer is
`2 · view_angle_limit(lens)`.

```julia
cam = BoundedCamera(CameraModel(K; lens = Lens(Equidistant())), sensor("Sony", "IMX250"))
angle_of_view(cam)                          # horizontal, full angle in radians
angle_of_view(cam; dimension = :diagonal)
```
"""
function angle_of_view(camera::BoundedCamera; dimension = :horizontal)
    m = model(camera)
    # an angle of view needs a metric focal: none for a projective camera
    m isa CameraModel{<:ProjectiveIntrinsics} && throw(
        ArgumentError(
            "an angle of view needs a metric camera; " *
                "a ProjectiveIntrinsics camera has no focal length, so no angle of view"
        )
    )
    l = lens(m)
    _check_lens_axis(l, "angle_of_view")
    w, h = Float64(extent(camera).width), Float64(extent(camera).height)
    cx, cy = _principal_point(m)
    pts = if dimension === :horizontal
        (SVector(0.0, cy), SVector(w, cy))
    elseif dimension === :vertical
        (SVector(cx, 0.0), SVector(cx, h))
    elseif dimension === :diagonal
        (SVector(0.0, 0.0), SVector(w, 0.0), SVector(0.0, h), SVector(w, h))
    else
        throw(ArgumentError("angle_of_view: dimension must be :horizontal, :vertical or :diagonal, got $(repr(dimension))"))
    end
    c = SVector{2, Float64}(cx, cy)
    cov = lens_coverage(camera)
    _covers(cov, c) || throw(
        ArgumentError("angle_of_view: the principal point $(Tuple(c)) lies outside the camera's lens coverage — it images nothing about its axis")
    )
    θ_lim = Float64(view_angle_limit(l))
    q = valid_domain(m)
    θ = minimum(pts) do u0
        u = _last_covered(cov, c, u0)          # the edge point, or where the coverage ends on the way to it
        Point2(u) ∈ q || return θ_lim          # outside the valid domain: past the lens's limit (or on its folded branch)
        b = backproject(m, u)                  # the camera's own pixel → bearing
        θu = atan(hypot(b[1], b[2]), b[3])
        isfinite(θu) ? min(θu, θ_lim) : θ_lim
    end
    return 2θ
end

# Whether the lens coverage holds a point, closed (a sensor edge point lies on the default coverage's boundary): a Rect by
# its bounds, a Circle as its round Ellipse (as `∩` meets it), an Ellipse by its own membership.
_covers(r::Rect2, p) = _in_rect(r, SVector{2}(p[1], p[2]))
_covers(c::Circle, p) = Point2(p) ∈ Ellipse(c)
_covers(e::Ellipse, p) = Point2(p) ∈ e
# The last point of the segment c → u0 the coverage holds (c held): u0 itself when held, else by bisection — the coverage
# is convex, so the held points of the segment are one piece from c.
function _last_covered(cov, c::SVector{2, Float64}, u0::SVector{2, Float64})
    _covers(cov, u0) && return u0
    lo, hi = 0.0, 1.0
    for _ in 1:60
        t = (lo + hi) / 2
        _covers(cov, c + t * (u0 - c)) ? (lo = t) : (hi = t)
    end
    return c + lo * (u0 - c)
end

"""
    field_of_view(camera::BoundedCamera, distance; dimension = :horizontal) -> size

The field of view of a bounded camera at `distance`: the SIZE of the scene it
images there, on a plane perpendicular to its optical axis — `2·distance·tan(θ/2)`
for its [`angle_of_view`](@ref) `θ` in `dimension`, in the unit of `distance` (a
length such as `2.0m`, or a plain number). An angle of view of 180° or more spans
no finite size on a plane, and a distance must be positive: both an
`ArgumentError`. (Not the field-of-view LENS model [`FieldOfView`](@ref), Devernay's
distortion — a different thing that shares the words.)

```julia
field_of_view(cam, 3.0m)                         # the width of the scene imaged 3 m away
field_of_view(cam, 3.0m; dimension = :vertical)
```
"""
function field_of_view(camera::BoundedCamera, distance; dimension = :horizontal)
    distance > zero(distance) || throw(ArgumentError("field_of_view: the distance is positive, got $(distance)"))
    θ = angle_of_view(camera; dimension)
    θ < π || throw(
        ArgumentError("field_of_view: an angle of view of $(rad2deg(θ))° (≥ 180°) spans no finite size on a plane at a distance")
    )
    return 2 * distance * tan(θ / 2)
end

"""
    lookat(eye::StaticVector{3, <:Quantity}, target::StaticVector{3, <:Quantity}, up::StaticVector{3, <:Real};
           convention::Symbol = :cv)

Create a camera extrinsics transform that looks from `eye` position toward `target`.

A PHYSICAL pose: `eye` and `target` are lengths, and the extrinsics' translation
is a length in metres — the plain [`lookat`](@ref) of the two points in metres,
read back in metres (a plain point beside a length is a `DimensionError`).

Uses **Computer Vision convention**:
- +Z axis points FORWARD (toward target)
- +X axis points RIGHT
- +Y axis points DOWN
- Camera -Y aligns with world up direction

# Arguments
- `eye::StaticVector{3, <:Quantity}`: Camera position in world coordinates (physical lengths)
- `target::StaticVector{3, <:Quantity}`: Point to look at in world coordinates (physical lengths)
- `up::StaticVector{3, <:Real}`: World up direction, unitless (typically [0, -1, 0] for CV convention where -Y is up)

# Returns
- `CameraExtrinsics`: World-to-camera transformation (frame-labeled `EuclideanMap`, physical)

# Example
```julia
# Camera at (0, 0, 0) looking toward (0, 0, 1000mm), with world -Y pointing up
extrinsics = lookat(SVector(0.0mm, 0.0mm, 0.0mm), SVector(0.0mm, 0.0mm, 1000.0mm), SVector(0.0, -1.0, 0.0))
```
"""
lookat(eye::StaticVector{3}, target::StaticVector{3}, up::StaticVector{3, <:Real}; convention::Symbol = :cv) =
    _attach_unit(lookat(Point3{Float64}(strip_metres(eye)), Point3{Float64}(strip_metres(target)), up; convention), m)

"""
    lookat(eye::StaticVector{3, <:Real}, target::StaticVector{3, <:Real}, up::StaticVector{3, <:Real};
           convention::Symbol = :cv)

Create a camera extrinsics transform that looks from `eye` position toward `target`.

Unitless version for internal use and backwards compatibility.

`convention` selects the camera-frame convention (same symbol-selector
precedent as `Rect(sensor, :julia)`):

- `:cv` (default) — **Computer Vision convention**:
  - +Z axis points FORWARD (toward target)
  - +X axis points RIGHT
  - +Y axis points DOWN
  - Camera -Y aligns with world up direction (supply `up` as [0, -1, 0])
- `:gfx` — **graphics convention** (Mitsuba/OpenGL-style `look_at` frame):
  camera-to-world columns `z = normalize(target − eye)`,
  `x = normalize(cross(up, z))`, `y = cross(z, x)`; supply `up` as the world
  up direction (typically [0, 1, 0]). The returned map is still the
  world-to-camera extrinsics — only the frame construction differs.

# Arguments
- `eye::StaticVector{3}`: Camera position in world coordinates (plain numbers, in the caller's unit)
- `target::StaticVector{3}`: Point to look at in world coordinates (same unit)
- `up::StaticVector{3}`: World up direction (see per-convention notes above)

# Returns
- `CameraExtrinsics`: World-to-camera transformation (frame-labeled `EuclideanMap`)

# Example
```julia
# Camera at (0, 0, 0) looking toward (0, 0, 1000), with world -Y pointing up
extrinsics = lookat(SVector(0.0, 0.0, 0.0), SVector(0.0, 0.0, 1000.0), SVector(0.0, -1.0, 0.0))
```

See also [`lookat_pose`](@ref) for the camera-to-world form (bit-exact eye
translation, no inversion round-trip).
"""
function lookat(
        eye::StaticVector{3, <:Real}, target::StaticVector{3, <:Real},
        up::StaticVector{3, <:Real}; convention::Symbol = :cv
    )
    xaxis, yaxis, zaxis = _lookat_frame(eye, target, up, convention)

    # Rotation part (camera basis vectors as columns)
    rotation = hcat(xaxis, yaxis, zaxis)

    # Translation: move the world so that the eye is at the origin
    eye_vec = SVector{3}(eye[1], eye[2], eye[3])
    translation = -rotation' * eye_vec

    # Label at source: lookat produces world-to-camera extrinsics by
    # construction, so return the frame-labeled map.
    return CameraExtrinsics(EuclideanMap(Rotations.RotMatrix{3}(rotation'), translation))
end

# Camera-frame basis columns (camera-to-world) for a lookat specification.
# :cv keeps the historical expression tree exactly (normalize on the y axis
# included) so existing callers see bit-identical extrinsics.
function _lookat_frame(
        eye::StaticVector{3, <:Real}, target::StaticVector{3, <:Real},
        up::StaticVector{3, <:Real}, convention::Symbol
    )
    if convention === :cv
        # Computer vision convention: camera looks down +Z axis
        # Right-handed system: +Z forward, +X right, +Y down
        # The "up" parameter is the world up direction (typically [0, -1, 0])
        # Camera -Y aligns with world up (since camera +Y is down)
        zaxis = normalize(target - eye)           # forward (+Z)
        xaxis = normalize(cross(up, zaxis))       # right (+X)
        yaxis = normalize(cross(zaxis, xaxis))    # CV: +Y points away from world `up`
        return xaxis, yaxis, zaxis
    elseif convention === :gfx
        # Graphics (Mitsuba/OpenGL look_at) frame: y = cross(z, x) exactly —
        # already unit for unit z ⟂ x, and skipping the normalize reproduces
        # renderer to_world matrices bit-for-bit.
        zaxis = normalize(target - eye)
        xaxis = normalize(cross(up, zaxis))
        yaxis = cross(zaxis, xaxis)
        return xaxis, yaxis, zaxis
    else
        throw(
            ArgumentError(
                "unknown lookat convention $(repr(convention)); valid options are :cv and :gfx"
            )
        )
    end
end

"""
    lookat_pose(eye::StaticVector{3, <:Real}, target::StaticVector{3, <:Real}, up::StaticVector{3, <:Real};
                convention::Symbol = :cv) -> CameraPose

Camera POSE (camera-to-world transform) that looks from `eye` toward
`target`, as a typed [`CameraPose`](@ref) — the pose-side companion of
[`lookat`](@ref), sharing its `convention` selector (`:cv` / `:gfx`).

Built directly from the frame columns (rotation = camera basis as
camera-to-world columns, translation = `eye` **bit-exactly**) rather than as
`inv(lookat(...))`, so no inversion round-trip touches the translation.
Feed the result straight to the frame-typed camera constructor
`PosedCamera(model, lookat_pose(...))`, which inverts internally into the
extrinsics slot.
"""
function lookat_pose(
        eye::StaticVector{3, <:Real}, target::StaticVector{3, <:Real},
        up::StaticVector{3, <:Real}; convention::Symbol = :cv
    )
    xaxis, yaxis, zaxis = _lookat_frame(eye, target, up, convention)
    rotation = hcat(xaxis, yaxis, zaxis)
    eye_vec = SVector{3}(eye[1], eye[2], eye[3])
    return CameraPose(EuclideanMap(Rotations.RotMatrix{3}(rotation), eye_vec))
end
# A physical pose: `eye` and `target` lengths, computed in metres, the pose's
# translation read back in metres.
lookat_pose(eye::StaticVector{3}, target::StaticVector{3}, up::StaticVector{3, <:Real}; convention::Symbol = :cv) =
    _attach_unit(lookat_pose(Point3{Float64}(strip_metres(eye)), Point3{Float64}(strip_metres(target)), up; convention), m)

# =============================================================================
# P3P Pose Sampling - Moved to random.jl
# =============================================================================
# Note: sample_p3p functions have been moved to cameras/random.jl for better organization

# =============================================================================
# P3P Solver with Correspondences
# =============================================================================

"""
    p3p(model::CameraModel, correspondences) -> FixedModels{4, Extrinsics3}

Solve the Perspective-3-Point (P3P) problem using correspondences.

This is a thin wrapper around the core `p3p(rays, X)` solver that accepts
correspondences directly. It extracts the 3D world points and 2D image points
from the correspondences, backprojects the image points to rays, and calls
the core solver.

# Arguments
- `model::CameraModel`: Camera model for backprojecting image points to rays
- `correspondences`: Correspondence container with exactly 3 point pairs
  (3D world → 2D image); any container satisfying the `sources`/`targets`
  protocol (`Correspondences`, StructArray of Pairs, plain Pair vector)

# Returns
- `FixedModels{4, Extrinsics3}`: Stack-allocated container of 0–4 rigid poses.
  Empty (`length == 0`) if no valid solution exists. Iterate with
  `for ext in result ... end` or `collect(result)`.

# Examples
```julia
# Setup camera
s = sensor("Sony", "IMX174")
sensor_bounds = Rect(s)
f = focal_length(s, deg2rad(40.0); dimension=:horizontal)
pp = centroid(sensor_bounds) .* px
model = CameraModel(f, s.pitch, pp)

# Create correspondences (3D world points → 2D image points)
X3 = [Point3(100.0, 100.0, 0.0), Point3(200.0, 150.0, 0.0), Point3(150.0, 200.0, 0.0)]
u3 = [Point2(640.0, 480.0), Point2(700.0, 500.0), Point2(680.0, 550.0)]
cs = correspondences(X3, u3)

# Solve P3P — always a FixedModels; empty (length 0) when no valid pose
result = p3p(model, cs)
cameras = PosedCamera.(Ref(model), collect(result))
```

# See Also
- `p3p(rays, X)`: Core P3P solver that works with rays directly
- `sample_p3p`: Sample valid camera poses by randomly sampling correspondences
"""
function p3p(model::CameraModel, correspondences)
    length(correspondences) == 3 ||
        throw(ArgumentError("P3P requires exactly 3 correspondences, got $(length(correspondences))"))

    # Extract 3D world points and 2D image points via the protocol
    X = sources(correspondences)
    u = targets(correspondences)

    # Backproject 2D points to normalized rays
    rays = backproject.(Ref(model), u)

    # Call original P3P solver
    return p3p(rays, X)
end
