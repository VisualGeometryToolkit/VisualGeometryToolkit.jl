# =============================================================================
# Shared Geometry Computation for Plotting
# =============================================================================
#
# Pure functions that compute geometry for visualization.
# Used by both SpecApi functions (Frame, BoardMesh, Frustum) and
# Makie recipes (board!, board3d!, matches!, frameplot!, cameraplot!).
#
# This is the single source of truth for geometry computation.
# =============================================================================

"""
    frame_segments(; origin, rotation, scale) -> (segments, colors)

Compute NaN-separated line segments for a coordinate frame.

Returns segments suitable for a single Lines plot with per-vertex colors.
"""
function frame_segments(;
        origin::Point3 = Point3f(0, 0, 0),
        rotation::AbstractMatrix = SMatrix{3, 3, Float64}(I),
        scale::Real = 1.0
    )
    # Compute axis endpoints
    basis = canonical_basis(SVector{3, Float64})
    scaled_basis = basis .* scale
    axis_ends = Point3f.(Ref(origin) .+ Ref(rotation) .* scaled_basis)
    origin_pt = Point3f(origin)

    # Build NaN-separated segments: [origin, end_x, NaN, origin, end_y, NaN, origin, end_z]
    nan_pt = Point3f(NaN, NaN, NaN)
    segments = [
        origin_pt, axis_ends[1], nan_pt,
        origin_pt, axis_ends[2], nan_pt,
        origin_pt, axis_ends[3],
    ]

    # Colors: 2 points per axis + NaN separator
    colors = [:red, :red, :red, :green, :green, :green, :blue, :blue]

    return segments, colors, axis_ends
end

# Plotting draws plain numbers: a physical pose or camera (translation in
# metres) is shown in millimetres, the scene unit of the plots; a plain one as it is.
_display_mm(x) = has_units(x) ? ustrip(mm, x) : x

"""
    frame_segments(pose::EuclideanMap; scale) -> (segments, colors, axis_ends)

Convenience method for EuclideanMap poses (a physical pose drawn in mm).
"""
function frame_segments(pose::EuclideanMap; scale::Real = 1.0)
    pose = _display_mm(pose)
    return frame_segments(;
        origin = Point3f(pose.t),
        rotation = SMatrix{3, 3, Float64}(pose.R),
        scale = scale
    )
end


"""
    frustum_segments(camera, sensor_bounds, depth) -> (segments, corners_world, cam_center)

Compute NaN-separated line segments for a camera frustum wireframe.

Returns segments suitable for a single Lines plot.
"""
function frustum_segments(camera::PosedCamera, sensor_bounds, depth::Real)
    camera = _display_mm(camera)
    # Get camera pose
    cam_pose = pose(camera)
    cam_center = Point3f(cam_pose.t)

    # Unproject sensor corners to 3D world coordinates
    corners_2d = Point2{Float64}.(coordinates(sensor_bounds))
    corners_cam = unproject.(Ref(camera.model), corners_2d, depth)
    corners_world = Point3f.(cam_pose.(corners_cam))

    # Build frustum wireframe as NaN-separated line segments
    # 4 edges from center to corners + 4 edges around far plane
    nan_pt = Point3f(NaN, NaN, NaN)

    segments = [
        # Edges from center to corners
        cam_center, corners_world[1], nan_pt,
        cam_center, corners_world[2], nan_pt,
        cam_center, corners_world[3], nan_pt,
        cam_center, corners_world[4], nan_pt,
        # Far plane rectangle
        corners_world[1], corners_world[2], nan_pt,
        corners_world[2], corners_world[3], nan_pt,
        corners_world[3], corners_world[4], nan_pt,
        corners_world[4], corners_world[1],
    ]

    return segments, corners_world, cam_center
end

"""
    effective_depth(camera, depth_attr) -> Float64

Compute effective frustum depth: `automatic` falls back to
`default_frustum_depth` (the fx scale from K).
"""
function effective_depth(camera::PosedCamera, depth_attr)
    return depth_attr === automatic ? default_frustum_depth(camera) : Float64(depth_attr)
end
