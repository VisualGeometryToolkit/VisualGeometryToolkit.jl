# =============================================================================
# Frame — dimension-typed coordinate-frame tags
# =============================================================================
#
# Frames are type-level TAGS carried by EuclideanMap/HomographyMap as trailing
# Src/Dst parameters (and, since 0.9, by CovMat/PrecMat as a trailing frame
# parameter). They record WHICH frame a map consumes and produces — or which
# frame a statistical form's coordinates live in — so the pose-vs-extrinsics
# direction (and the covariance-vs-precision transformation law) becomes a
# dispatch fact instead of a docstring convention. Zero runtime cost: frames
# are singleton types with no fields.
#
# This file is loaded BEFORE base/types.jl so the statistical forms can
# constrain their frame parameter; the maps that carry Src/Dst live in
# geometry/transforms/euclidean_map.jl and homography_map.jl.
#
# Drake monogram bridge (stated once, used throughout): the monogram symbol
# `X_AB` — "pose of frame B expressed in frame A", which maps B-coordinates
# to A-coordinates — corresponds to `EuclideanMap{..., Src = B, Dst = A}`.

"""
    Frame{N}

Abstract supertype for coordinate-frame tags of dimension `N`.

A frame tag names a coordinate frame (3D, or a 2D image frame) at the type level;
`EuclideanMap` carries a source and destination frame as trailing type
parameters so that composition and inversion are frame-checked by dispatch.
In Drake monogram notation, `X_AB` (pose of B in A, maps B-coordinates to
A-coordinates) corresponds to `EuclideanMap{..., Src = B, Dst = A}`.

The hierarchy is **open**: downstream packages declare their own frames
(`struct LidarFrame <: Frame{3} end`) and participate in frame-checked
composition with zero changes to VisualGeometryToolkitCore.

See also [`UnlabeledFrame`](@ref), [`WorldFrame`](@ref),
[`CameraFrame`](@ref), [`TargetPlaneFrame`](@ref), [`PixelFrame`](@ref),
[`unlabel`](@ref), [`relabel`](@ref).
"""
abstract type Frame{N} end

"""
    UnlabeledFrame{N} <: Frame{N}

The gradual-typing default frame: "no frame recorded".

Every `EuclideanMap` constructor that does not state frames produces
`Src = Dst = UnlabeledFrame{N}`, so pre-frame code compiles and behaves
exactly as before. `UnlabeledFrame` is the composition **wildcard**: it
composes with anything, the junction is unchecked, and known endpoints are
kept as-is (the truth table at `Base.:∘` for `EuclideanMap`; the rule is
`_junction_ok`, below). This
anti-virality rule means frame labels never force themselves onto code that
does not opt in.
"""
struct UnlabeledFrame{N} <: Frame{N} end

# THE junction rule of every frame-checked operation — a map composition `∘`
# (the junction is `B.Dst` meeting `A.Src`), a form's slot meeting a map's
# `Src` (`∘` on an `EpipolarMap`), a measurement meeting a slot, a statistical form meeting a
# map (`propagate`): two frames meet iff they are the same tag, or either is the
# unlabeled wildcard of the same dimension (gradual typing — labels never force
# themselves onto unlabeled code). Constant-folded for concrete tags, so a
# checked operation pays nothing at run time.
@inline _junction_ok(::Type{A}, ::Type{B}) where {N, A <: Frame{N}, B <: Frame{N}} =
    A === B || A <: UnlabeledFrame || B <: UnlabeledFrame
@inline _junction_ok(::Type{<:Frame}, ::Type{<:Frame}) = false   # different dimensions

# The rule as a check: a mismatch throws the operation's own `MethodError` — an
# explicit throw rather than an absent method, because every checked operation
# has a generic fallback (CoordinateTransformations' lazy `∘`, the Jacobian
# `propagate`) that would otherwise silently apply the wrong semantics.
@inline function _check_junction(::Type{A}, ::Type{B}, f, args) where {A, B}
    _junction_ok(A, B) || throw(MethodError(f, args))
    return nothing
end

"""
    WorldFrame <: Frame{3}

The world (scene) coordinate frame. Destination frame of a camera POSE
(`X_WC ≙ EuclideanMap{..., Src = CameraFrame, Dst = WorldFrame}`), source
frame of camera EXTRINSICS.
"""
struct WorldFrame <: Frame{3} end

"""
    CameraFrame <: Frame{3}

The camera-centered coordinate frame (origin at the optical center, +Z along
the optical axis in the CV convention). Source frame of a camera POSE,
destination frame of camera EXTRINSICS
(`X_CW ≙ EuclideanMap{..., Src = WorldFrame, Dst = CameraFrame}`).
"""
struct CameraFrame <: Frame{3} end
"""
    ViewFrame{id} <: Frame{3}

The 3D camera-centered frame of a SPECIFIC view, tagged by an instance
identity: `id::Symbol` names which camera (`ViewFrame{:left} ≠
ViewFrame{:right}`). This extends the `PixelFrame{:colmap}` idiom from
convention-identity to instance-identity — the symbol parameter
distinguishes coexisting instances of the same kind of frame, so that a
labeled junction between two different cameras' frames is a `MethodError`
instead of a silent [`CameraFrame`](@ref) collision.

Axis conventions are those of `CameraFrame` (origin at the optical center,
+Z along the optical axis); the tag adds WHICH camera, not a new
convention. In Drake monogram terms a two-view relative pose is
`X_{V_b V_a} ≙ EuclideanMap{..., Src = ViewFrame{:a}, Dst = ViewFrame{:b}}`.

Gradual as always: nothing forces view identities — `UnlabeledFrame` (and
the singleton `CameraFrame`) remain the defaults everywhere. `StereoRig`
mints per-view identities on request (see [`view_frames`](@ref)).

See also [`ImageFrame`](@ref) (the 2D image frame of a view),
[`CameraFrame`](@ref), [`Frame`](@ref).
"""
struct ViewFrame{id} <: Frame{3} end
"""
    TargetFrame{id} <: Frame{3}

The 3D body frame of a SPECIFIC calibration target (board) instance, tagged
by an instance identity: `id::Symbol` names which target — the
[`ViewFrame`](@ref) instance-identity idiom applied to the target side of a
calibration. Target-local geometry lives in this frame (for planar boards,
the pattern plane is `z = 0`); the target's POSE in a common frame `F` is
`EuclideanMap{..., Src = TargetFrame{id}, Dst = F}` (target-local
coordinates into `F`).

For content-addressed targets (e.g. BlobBoards board uids) the symbol is
the content hash, so the frame tag doubles as the geometry reference.
Minted from a target rig's keys by `target_frames`.

See also [`ViewFrame`](@ref), [`TargetPlaneFrame`](@ref)
(the 2D frame of the board plane), [`Frame`](@ref).
"""
struct TargetFrame{id} <: Frame{3} end
"""
    BodyFrame{id} <: Frame{3}

The 3D frame of a SPECIFIC rigid body, tagged by an instance identity:
`id::Symbol` names which body — the [`ViewFrame`](@ref) instance-identity idiom
applied to things that are tracked or mounted as a unit (a motion-capture rigid
body, a camera rig, a board rig carrying several targets). A body is one frame,
and a trajectory is its pose over time: `EuclideanMap{..., Src = BodyFrame{id},
Dst = WorldFrame}` per instant, never a new frame per instant.

What is mounted on a body is posed in it: a target on a board rig is
`EuclideanMap{..., Src = TargetFrame{m}, Dst = BodyFrame{:board_rig}}`, a camera
on a camera rig `EuclideanMap{..., Src = BodyFrame{:camera_rig}, Dst =
ViewFrame{id}}` (extrinsics: rig coordinates into the camera's), so a hand-eye
chain is junction-checked by dispatch.

See also [`TargetFrame`](@ref), [`ViewFrame`](@ref), [`WorldFrame`](@ref),
[`Frame`](@ref).
"""
struct BodyFrame{id} <: Frame{3} end
