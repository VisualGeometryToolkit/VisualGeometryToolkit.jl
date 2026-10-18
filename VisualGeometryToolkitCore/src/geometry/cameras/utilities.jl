# ==============================================================================
# Camera Utilities - Helper functions for focal length and camera positioning
# ==============================================================================

"""
    focal_length_for_fov(resolution::Size2, θ::Real; dimension=:horizontal)

The focal length a field of view implies: in pixels from field of view angle (plain radians —
angles carry no dimension in the VGC unit system) and resolution.

# Arguments
- `resolution::Size2`: Image resolution (width, height) in pixels
- `θ::Real`: Field of view angle in radians
- `dimension::Symbol`: Which dimension the FOV applies to (`:horizontal` or `:vertical`)

# Returns
- Focal length in pixels (same logical unit as resolution)

# Examples
```julia
focal_length_for_fov(Size2(width=1280px, height=960px), 0.698)
focal_length_for_fov(Size2(width=1280px, height=960px), deg2rad(40.0))
```
"""
function focal_length_for_fov(resolution::Size2, θ::Real; dimension = :horizontal)
    if dimension === :horizontal
        f = 0.5 * resolution.width
    elseif dimension === :vertical
        f = 0.5 * resolution.height
    else
        throw(ArgumentError("dimension must be :horizontal or :vertical"))
    end

    return f / tan(θ / 2)
end

"""
    focal_length_for_fov(sensor, θ::Real; dimension=:horizontal)

The physical focal length from field of view angle and sensor specifications.

# Arguments
- `sensor`: Sensor with `resolution` and `pitch` fields
- `θ::Real`: Field of view angle in radians
- `dimension::Symbol`: Which dimension the FOV applies to (`:horizontal` or `:vertical`)

# Returns
- Physical focal length (a length quantity)

# Examples
```julia
s = sensor("Sony", "IMX174")
f_mm = focal_length_for_fov(s, deg2rad(40.0))
```
"""
function focal_length_for_fov(sensor, θ::Real; dimension = :horizontal)
    # Get focal length in pixels
    f_px = focal_length_for_fov(sensor.resolution, θ; dimension = dimension)

    # Convert to physical length using pixel pitch
    # Use width pitch for horizontal FOV, height pitch for vertical FOV
    pitch = (dimension === :horizontal) ? sensor.pitch.width : sensor.pitch.height

    # f = f_px × pitch (px × length/px = a length)
    return f_px * pitch
end

"""
    lookat(eye::StaticVector{3, <:Quantity}, target::StaticVector{3, <:Quantity}, up::StaticVector{3, <:Real})

Create a camera extrinsics transform that looks from `eye` position toward `target`.

Units are enforced at construction but stored internally as unitless Float64 in mm.

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
- `CameraExtrinsics`: World-to-camera transformation (frame-labeled `EuclideanMap`)

# Example
```julia
# Camera at (0, 0, 0) looking toward (0, 0, 1000mm), with world -Y pointing up
extrinsics = lookat(SVector(0.0mm, 0.0mm, 0.0mm), SVector(0.0mm, 0.0mm, 1000.0mm), SVector(0.0, -1.0, 0.0))
```
"""
function lookat(eye::StaticVector{3, <:Quantity}, target::StaticVector{3, <:Quantity}, up::StaticVector{3, <:Real})
    # Strip in mm (a ConversionError unless every component is a length)
    eye_mm = ustrip.(mm, eye)
    target_mm = ustrip.(mm, target)

    # Call the unitless version (eye + target are positions → Point3)
    return lookat(Point3{Float64}(eye_mm), Point3{Float64}(target_mm), up)
end

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
- `eye::StaticVector{3}`: Camera position in world coordinates (unitless, assumed mm)
- `target::StaticVector{3}`: Point to look at in world coordinates (unitless, assumed mm)
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
`Camera(model, lookat_pose(...))`, which inverts internally into the
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

# =============================================================================
# P3P Pose Sampling - Moved to random.jl
# =============================================================================
# Note: sample_p3p functions have been moved to cameras/random.jl for better organization

# =============================================================================
# P3P Solver with Correspondences
# =============================================================================

"""
    p3p(model::CameraModel, csponds) -> FixedModels{4, Extrinsics3}

Solve the Perspective-3-Point (P3P) problem using correspondences.

This is a thin wrapper around the core `p3p(rays, X)` solver that accepts
correspondences directly. It extracts the 3D world points and 2D image points
from the correspondences, backprojects the image points to rays, and calls
the core solver.

# Arguments
- `model::CameraModel`: Camera model for backprojecting image points to rays
- `csponds`: Correspondence container with exactly 3 point pairs
  (3D world → 2D image); any container satisfying the `sources`/`targets`
  protocol (`Csponds`, StructArray of Pairs, plain Pair vector)

# Returns
- `FixedModels{4, Extrinsics3}`: Stack-allocated container of 0–4 rigid poses.
  Empty (`length == 0`) if no valid solution exists. Iterate with
  `for ext in result ... end` or `collect(result)`.

# Examples
```julia
# Setup camera
s = sensor("Sony", "IMX174")
sensor_bounds = Rect(s)
f = focal_length_for_fov(s, deg2rad(40.0); dimension=:horizontal)
pp = centroid(sensor_bounds) .* px
model = CameraModel(f, s.pitch, pp)

# Create correspondences (3D world points → 2D image points)
X3 = [Point3(100.0, 100.0, 0.0), Point3(200.0, 150.0, 0.0), Point3(150.0, 200.0, 0.0)]
u3 = [Point2(640.0, 480.0), Point2(700.0, 500.0), Point2(680.0, 550.0)]
cs = csponds(X3, u3)

# Solve P3P — always a FixedModels; empty (length 0) when no valid pose
result = p3p(model, cs)
cameras = Camera.(Ref(model), collect(result))
```

# See Also
- `p3p(rays, X)`: Core P3P solver that works with rays directly
- `sample_p3p`: Sample valid camera poses by randomly sampling correspondences
"""
function p3p(model::CameraModel, csponds)
    length(csponds) == 3 ||
        throw(ArgumentError("P3P requires exactly 3 correspondences, got $(length(csponds))"))

    # Extract 3D world points and 2D image points via the protocol
    X = sources(csponds)
    u = targets(csponds)

    # Backproject 2D points to normalized rays
    rays = backproject.(Ref(model), u)

    # Call original P3P solver
    return p3p(rays, X)
end
