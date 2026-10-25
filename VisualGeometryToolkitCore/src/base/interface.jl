# =============================================================================
# Interface Definitions for VisualGeometryToolkitCore
# =============================================================================
#
# This file loads and re-exports all interface components:
# 1. Holy Traits for geometric primitives (behavioral contracts)
# 2. Function stubs for shared interface functions
# 3. Canonical geometry constants
# 4. Abstract types and interfaces for Detection and Camera systems
#
# These are loaded early so all submodules can extend them consistently.
#
# =============================================================================
# INCLUDE ORDER
# =============================================================================
#
# traits.jl            - Holy Traits definitions (ConicTrait, HasBoundaryTrait, BoundsTrait, …)
# interface_functions.jl - Function stubs (centroid, project, distance, etc.)
#
# Detection/Camera interfaces are defined inline below since they are
# small and tightly coupled to specific submodules.
#
# =============================================================================

include("traits.jl")
include("interface_functions.jl")

# =============================================================================
# Detection Interface
# =============================================================================
#
# Interface function for working with detections without depending on the
# concrete Detection type from Features submodule.
#
# =============================================================================

"""
    feature_frame(d) -> FeatureFrame

Return the feature frame (the detected local frame: `Disc`, `EllipticFrame`,
…) of a detection or a bare `FeatureFrame`. Named for what it returns, so it
cannot be read as "which coordinate frame".

Unified accessor: works on both `FeatureFrame` (returns itself) and any
[`AbstractDetection`](@ref) (returns the wrapped frame; for a detection of an
`Uncertain` frame, the bare frame without its covariance).
"""
function feature_frame end

# =============================================================================
# Camera/Projection Interface
# =============================================================================
#
# Abstract types and interface functions for cameras. Concrete implementations
# are in geometry/cameras/. This allows code to work with cameras without
# depending on the full Cameras submodule.
#
# =============================================================================

"""
    AbstractCamera

The one camera type: every camera — an unposed model ([`CameraModel`](@ref),
the generic grid models), a posed camera ([`Camera`](@ref)) and a bounded one
([`BoundedCamera`](@ref)) — subtypes it. Whether a camera carries a pose is a
trait, [`has_extrinsics`](@ref), not a second type root.

Every camera implements, in the COLMAP pixel frame:
- `project(camera, X)` — a 3D point (camera frame for a model, world frame for
  a posed camera) to its pixel
- `backproject(camera, u)` — a pixel to its ray

A posed camera (`has_extrinsics(camera) == true`) also implements `pose(camera)`
and `extrinsics(camera)`; on an unposed one they are an `ArgumentError`.

# Example
```julia
# Code that works with any camera type:
function process(cam::AbstractCamera, points::Vector{Point3})
    pixels = [project(cam, p) for p in points]
    # ...
end
```
"""
abstract type AbstractCamera end

"""
    has_extrinsics(camera::AbstractCamera) -> Bool

Whether `camera` carries a pose: `true` for a [`Camera`](@ref) (a model and its
extrinsics), `false` for an unposed model ([`CameraModel`](@ref),
[`CentralGenericModel`](@ref), [`NoncentralGenericModel`](@ref)), and the
wrapped camera's answer for a [`BoundedCamera`](@ref). It is decided by the
camera's type, so a branch on it compiles away.

The posed-only verbs — `pose`, `extrinsics`, rig membership, the
[`StereoRig`](@ref) constructors that read poses — refuse an unposed camera
with an `ArgumentError`, and `Camera(model, X)` refuses a posed `model`. A
camera type of another package that carries a pose adds a method returning
`true`.
"""
has_extrinsics(::AbstractCamera) = false

"""
    pose(camera::AbstractCamera) -> EuclideanMap

Return the pose (position and orientation) of a camera in world coordinates.

The pose is the inverse of the extrinsics: it transforms from camera frame to world frame.
"""
function pose end

"""
    extrinsics(camera::AbstractCamera) -> EuclideanMap

Return the extrinsics (world-to-camera transform) of a camera.
"""
function extrinsics end

"""
    extent(camera) -> Size2{Int}

Return the pixel-space image extent (width × height) of a camera.

Trait-dispatched via `bounds_trait(camera)`: errors at compile time for cameras
that don't carry bounds (default `NoBounds()`).  Cameras that do carry bounds
(e.g. `BoundedCamera`) opt in via `bounds_trait(::Type{...}) = HasBounds()`.
"""
function extent end

"""
    backproject(camera_or_model, pixel::StaticVector{2}) -> SVector{3}

Backproject a 2D pixel coordinate to a 3D ray direction. The return is
a 1-parameter family of 3D points (depth unknown) represented as a
direction vector.
"""
function backproject end

"""
    unproject(camera_or_model, pixel, depth) -> SVector{3}

Unproject a 2D pixel coordinate at the given depth to a 3D point. Unlike
`backproject` (which returns a ray direction), `unproject` collapses the
ray to a specific point using the supplied depth.
"""
function unproject end

"""
    transfer(operator, observation)

Transfer an observation from one view to another via a known multi-view
geometry operator (e.g. a `StereoRig`, a homography, or an explicit
`(src_model, extrinsics, tgt_model)` triple).

The operator comes first, as in the multi-view analogs `project` /
`unproject`. Concrete dispatches live with the operator they consume.
"""
function transfer end

"""
    focal_length(camera_or_model)

Return the focal length of a camera or camera model.

For physical cameras, returns a length with units (e.g., mm).
For logical cameras, returns focal length in pixels.
"""
function focal_length end

"""
    intrinsics(camera_or_model)

Return the intrinsics block (e.g. `LogicalIntrinsics`, `PhysicalIntrinsics`).
"""
function intrinsics end

"""
    lens(camera_or_model)

Return the lens model (e.g. `Pinhole`, a `Lens`, or any user-defined
`<: AbstractLensModel`).  Use this for dispatch in downstream packages.
"""
function lens end
