# =============================================================================
# Random Sampling Functions for Camera Geometry
# =============================================================================

# A uniform image point in `r`. GeometryBasics defines no sampler for a
# rectangle, so `rand(rng, r)` was a MethodError on the first draw of every
# `sample_p3p` below, which therefore never ran; a private helper, not a
# `rand` method on the foreign type.
_rand_point(rng::Random.AbstractRNG, r::Rect) =
    Point2{Float64}(GeometryBasics.origin(r) .+ rand(rng, SVector{2, Float64}) .* GeometryBasics.widths(r))

# Attempts before a sampler gives up on finding a P3P configuration.
const _P3P_MAX_RETRIES = 1000

# Method with default RNG
"""
    sample_p3p(model, X, sensor_bounds; max_retries=1000)
    sample_p3p(rng, model, X, sensor_bounds; max_retries=1000)

Sample a random P3P configuration by sampling from world points X and random image points.

Creates 3D-2D correspondences by:
1. Sampling 3 random points from X (world coordinates)
2. Sampling 3 random 2D image points in sensor bounds
3. Pairing them as correspondences and solving P3P

Takes a logical camera (`CameraModel{<:LogicalIntrinsics}`): `X` and the
sensor bounds are unitless (the construction boundary stripped units).

# Arguments
- `rng=Random.default_rng()`: Random number generator (optional first positional argument)
- `model::CameraModel`: Camera model
- `X::Vector{<:StaticVector{3,Float64}}`: 3D world points, unitless
- `sensor_bounds::Rect`: Sensor bounds for sampling image points (unitless, in pixels)

# Keyword Arguments
- `max_retries=1000`: Maximum number of sampling attempts

# Returns
- `(extrinsics, csponds)`: Tuple containing:
  - `extrinsics::Vector{Extrinsics3}`: Camera extrinsics (world-to-camera transforms) for valid P3P solutions
  - `csponds::StructArray{Pair{Point3, Point2}, 1, ...}`: The 3 correspondences used for P3P, backed by `SVector{3}` columns

Throws an error if no valid configuration is found after max_retries attempts.

# Examples
```julia
# Setup camera
s = sensor("Sony", "IMX174")
sensor_bounds = Rect(s)
f = focal_length_for_fov(s, deg2rad(40.0); dimension=:horizontal)
pp = centroid(sensor_bounds) .* px
model = CameraModel(f, s.pitch, pp)

# 3D points (planar points at z=0)
X = [Point3(100.0, 100.0, 0.0), Point3(200.0, 150.0, 0.0), ...]

# With default RNG
extrinsics, csponds = sample_p3p(model, X, sensor_bounds)

# With explicit RNG for reproducibility
rng = Random.MersenneTwister(12345)
extrinsics, csponds = sample_p3p(rng, model, X, sensor_bounds)

# Access correspondence data (StructArray interface)
X_used = csponds.first   # All 3D points
u_used = csponds.second  # All 2D image points
# Or index individual correspondences: csponds[1].first, csponds[1].second

# Use best extrinsics (e.g., based on reprojection error)
cameras = PosedCamera.(Ref(model), extrinsics)
best_camera = cameras[1]  # Or select based on reprojection error
```
"""
function sample_p3p(
        model::CameraModel,
        X::Vector{<:StaticVector{3, Float64}},
        sensor_bounds::Rect;
        max_retries::Int = _P3P_MAX_RETRIES
    )
    return sample_p3p(Random.default_rng(), model, X, sensor_bounds; max_retries = max_retries)
end

# World-unit agnostic: X and sensor_bounds are unitless Float64 (the
# construction boundary stripped units); the same body served both the
# "physical" and "logical" camera flavors verbatim before the
# PhysicalIntrinsics demotion.
function sample_p3p(
        rng::Random.AbstractRNG,
        model::CameraModel{<:LogicalIntrinsics, P},
        X::Vector{<:StaticVector{3, Float64}},  # Logical units (unitless) as Float64
        sensor_bounds::Rect;
        max_retries::Int = _P3P_MAX_RETRIES
    ) where {P}

    length(X) >= 3 || error("Need at least 3 point correspondences for P3P")
    return _sample_p3p(
        rng, model, r -> _draw_world_image(r, X, sensor_bounds), max_retries,
        "Failed to find valid P3P configuration after $max_retries attempts"
    ) do sols, X3, u3
        isempty(sols) ? nothing : (collect(sols), csponds(X3, u3))
    end
end

# THE P3P sampling loop every `sample_p3p` runs: per attempt `draw(rng)` gives 3
# world points and 3 image points, the image points are back-projected through
# `model`, and `accept(p3p(rays, X3), X3, u3)` returns the result or `nothing`
# (try again). P3P never throws on data: a degenerate sample has no solution.
function _sample_p3p(accept::A, rng, model, draw::D, max_retries::Int, msg) where {A, D}
    for _ in 1:max_retries
        X3, u3 = draw(rng)
        r = accept(p3p(backproject.(Ref(model), u3), X3), X3, u3)
        r === nothing || return r
    end
    return error(msg)
end

# 3 image points drawn uniformly in the sensor bounds.
_rand_image_points(rng, sensor_bounds) =
    SVector{3}(_rand_point(rng, sensor_bounds), _rand_point(rng, sensor_bounds), _rand_point(rng, sensor_bounds))

# 3 distinct world points of `X` at random, then 3 image points.
function _draw_world_image(rng, X, sensor_bounds)
    sampled_idx = randperm(rng, length(X))[1:3]
    X3 = SVector{3}(X[sampled_idx[1]], X[sampled_idx[2]], X[sampled_idx[3]])
    return X3, _rand_image_points(rng, sensor_bounds)
end

# Accept the first pose whose camera centre lies on the positive side of `plane`.
function _in_front_of(model, plane)
    return function (sols, X3, u3)
        for ext in sols
            cam = PosedCamera(model, ext)
            distance(plane, Point3(pose(cam).t...)) > 0 && return (cam, StructArray(X3 .=> u3))
        end
        return nothing
    end
end

# =============================================================================
# Chirality-filtered P3P sampling
# =============================================================================

"""
    sample_p3p(model, X, sensor_bounds, plane::Plane3D; max_retries=1000)
    sample_p3p(rng, model, X, sensor_bounds, plane::Plane3D; max_retries=1000)

Sample a random P3P configuration with chirality filtering.

Like `sample_p3p(model, X, sensor_bounds)` but additionally filters solutions
to only those where the camera is on the positive side of the given plane
(i.e., `distance(plane, camera_center) > 0`).

This is useful for planar calibration targets where you want to ensure the
camera is "in front of" the target plane.

# Arguments
- `rng=Random.default_rng()`: Random number generator (optional first positional argument)
- `model::CameraModel`: Camera model
- `X::Vector{<:StaticVector{3,Float64}}`: 3D world points on or near the plane
- `sensor_bounds::Rect`: Sensor bounds for sampling image points (unitless, in pixels)
- `plane::Plane3D`: Reference plane for chirality test

# Keyword Arguments
- `max_retries=1000`: Maximum number of sampling attempts

# Returns
- `(camera, csponds)`: Tuple containing:
  - `camera::PosedCamera`: The camera with valid pose (in front of plane)
  - `csponds::StructArray{Pair{Point3, Point2}, 1, ...}`: The 3 correspondences used

# Example
```julia
# World coordinate system: origin at board center, z=0 plane, z-axis pointing out
board_plane = Plane3D()  # z=0, normal = (0,0,1)

# Sample camera that is in front of the board (z > 0)
camera, csponds = sample_p3p(model, X, sensor_bounds, board_plane)
```
"""
function sample_p3p(
        model::CameraModel,
        X::Vector{<:StaticVector{3, Float64}},
        sensor_bounds::Rect,
        plane::Plane3D;
        max_retries::Int = _P3P_MAX_RETRIES
    )
    return sample_p3p(Random.default_rng(), model, X, sensor_bounds, plane; max_retries = max_retries)
end

function sample_p3p(
        rng::Random.AbstractRNG,
        model::CameraModel,
        X::Vector{<:StaticVector{3, Float64}},
        sensor_bounds::Rect,
        plane::Plane3D;
        max_retries::Int = _P3P_MAX_RETRIES
    )

    length(X) >= 3 || error("Need at least 3 point correspondences for P3P")
    return _sample_p3p(
        _in_front_of(model, plane), rng, model, r -> _draw_world_image(r, X, sensor_bounds), max_retries,
        "Failed to find valid P3P configuration (camera in front of plane) after $max_retries attempts"
    )
end

# =============================================================================
# Winding-consistent P3P sampling (deterministic scale)
# =============================================================================

"""
    sample_p3p(model, X3::SVector{3}, sensor_bounds, plane::Plane3D; max_retries=1000)

Sample a P3P configuration using 3 fixed world points with winding-consistent image sampling.

Given 3 world points (e.g., board corners), samples 3 random image points and sorts
them with consistent winding (world CCW, image CW due to y-axis flip). This ensures:
- **Deterministic scale**: World point distances define the scale
- **Consistent orientation**: Matching winding prevents reflected/flipped solutions

Points can be supplied in any order - the function sorts world points to CCW and
image points to CW (matching winding due to y-axis inversion between coordinate systems).

# Arguments
- `model::CameraModel`: Camera model
- `X3::SVector{3,Point3}`: 3 world points (any order, will be sorted to CCW)
- `sensor_bounds::Rect`: Sensor bounds for sampling image points
- `plane::Plane3D`: Reference plane for chirality test

# Returns
- `(camera, csponds)`: Camera and the 3 correspondences used (in CCW order)

# Example
```julia
# Board corners in world coords (centered, y-up, z=0)
board_rect = Rect(bundle, :world)
corners = coordinates(board_rect)  # 4 corners
X3 = SVector{3}(Point3.(vcat.(corners[[1,2,3]], 0.0)))  # any 3 corners

camera, csponds = sample_p3p(model, X3, sensor_bounds, Plane3D())
```
"""
function sample_p3p(
        model::CameraModel,
        X3::StaticVector{3, <:Point3},
        sensor_bounds::Rect,
        plane::Plane3D;
        max_retries::Int = _P3P_MAX_RETRIES
    )
    return sample_p3p(Random.default_rng(), model, X3, sensor_bounds, plane; max_retries = max_retries)
end

function sample_p3p(
        rng::Random.AbstractRNG,
        model::CameraModel,
        X3::StaticVector{3, <:Point3},
        sensor_bounds::Rect,
        plane::Plane3D;
        max_retries::Int = _P3P_MAX_RETRIES
    )

    # World points to CCW winding (y-up convention); each draw's image points
    # to CW (image y-down, so CW matches world CCW)
    X3_sorted = _sort_ccw_world(X3)
    return _sample_p3p(
        _in_front_of(model, plane), rng, model, r -> (X3_sorted, _sort_cw_image(_rand_image_points(r, sensor_bounds))),
        max_retries, "Failed to find valid P3P configuration after $max_retries attempts"
    )
end

# Sort 3 points to CCW winding around their centroid (for world coords, y-up)
function _sort_ccw_world(pts::StaticVector{3, <:Point3})
    c = (pts[1] + pts[2] + pts[3]) / 3
    angles = SVector{3}(
        atan(pts[1][2] - c[2], pts[1][1] - c[1]),
        atan(pts[2][2] - c[2], pts[2][1] - c[1]),
        atan(pts[3][2] - c[2], pts[3][1] - c[1])
    )
    order = sortperm(angles)
    return SVector{3}(pts[order[1]], pts[order[2]], pts[order[3]])
end

# Sort 3 image points to CW winding (image y-down, so CW matches CCW in world)
function _sort_cw_image(pts::SVector{3, <:Point2})
    c = (pts[1] + pts[2] + pts[3]) / 3
    angles = SVector{3}(
        atan(pts[1][2] - c[2], pts[1][1] - c[1]),
        atan(pts[2][2] - c[2], pts[2][1] - c[1]),
        atan(pts[3][2] - c[2], pts[3][1] - c[1])
    )
    order = sortperm(angles; rev = true)  # Reverse for CW
    return SVector{3}(pts[order[1]], pts[order[2]], pts[order[3]])
end
