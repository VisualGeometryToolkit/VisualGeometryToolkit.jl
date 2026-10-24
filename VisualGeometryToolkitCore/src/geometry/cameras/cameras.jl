"""
    Cameras

Camera system submodule for VisualGeometryToolkitCore.

Provides complete camera modeling including:
- **Sensors**: Physical CMOS sensor specifications
- **Intrinsics**: Calibration matrices and intrinsic parameters (physical/logical)
- **Lens Models**: forward lenses (`Lens`: a projection + a distortion in its frame) and lifts (`LiftLens`)
- **Camera Models**: Composable camera models (intrinsics + lens)
- **Cameras**: Complete cameras in 3D space (model + pose)
- **Stereo**: Stereo camera rigs and epipolar geometry
- **Utilities**: Helper functions (focal_length, lookat, P3P solvers)

# Main Types
- `Camera`, `PinholeCamera`
- `CameraModel`
- `LogicalIntrinsics`
- `Lens`, `LiftLens`, `Pinhole`, `KannalaBrandt`, `Division`
- `Sensor`, `CMOS_SENSORS`
- `StereoRig`

# Main Functions
- `project`, `backproject`, `unproject`
- `focal_length`, `lookat`, `pose`
- `p3p`, `sample_p3p`
- `default_frustum_depth`

# Example
```julia
using VisualGeometryToolkitCore
using VisualGeometryToolkitCore.Cameras

s = sensor("Sony", "IMX174")
f = focal_length_for_fov(s, deg2rad(40.0))
model = CameraModel(f, s.pitch, centroid(Rect(s)) .* px)
camera = Camera(model, lookat(...))
```
"""
module Cameras

public pixel_intrinsics   # the pixel intrinsics of a camera (the stack's fits read them)
public coefficient_names  # a lens leaf type's coefficient names (the stack's lens fits read them)

# Import general Julia packages directly (not through parent)
using StaticArrays: StaticArrays, SVector, SMatrix, StaticVector, StaticMatrix
using LinearAlgebra: LinearAlgebra, normalize, cross, norm, det, svd
using Statistics: Statistics   # cov of an Uncertain point (targets.jl, captures.jl)
using LogExpFunctions: logit, logistic   # DoubleSphere's α chart (lens_parameters.jl)
using Random: Random, randperm
import FlexUnits
using FlexUnits: ustrip, dimension
using StructArrays: StructArrays, StructArray
using IntervalSets: IntervalSets
using CoordinateTransformations: CoordinateTransformations, AffineMap, Translation
import CoordinateTransformations: transform_deriv, transform_deriv_params
using Rotations: Rotations, RotMatrix
using GeometryBasics: GeometryBasics
using MacroTools: MacroTools
using KernelAbstractions: KernelAbstractions, @kernel, @index, @Const, get_backend
using Adapt: adapt   # device upload of the grid and rays (generic_models.jl)

# Import from parent module (VisualGeometryToolkitCore-specific types only)
import KernelAbstractions as KA   # batched generic-model projection kernel
import FastLevenbergMarquardt as FLM   # the generic models' pixel-space LM (generic_models.jl `_pixel_lm`)
using ..VisualGeometryToolkitCore: _bilinear_sample, _store, _sample_type   # the one bilinear sampler (warp.jl): the grid remap's dense kernel
using ..VisualGeometryToolkitCore: _pixel_extent   # a size in pixels, plain or `px` (homography_solvers.jl)
using ..VisualGeometryToolkitCore: _pushforward, _pullback   # the constant-Jacobian transport laws (propagate_framed.jl)
using ..VisualGeometryToolkitCore: Point2, Point3, Vec2, Vec3
using ..VisualGeometryToolkitCore: _rect_edge_samples   # the one per-edge rect walk (primitives/rectangles.jl)
using ..Primitives: rect_edges, EdgeGrid, Closed   # its edges on a closed grid: the frame walk (camera_models.jl)
using ..VisualGeometryToolkitCore: block_order   # grouped data: the sorted-key block order (base/cspond_set.jl)
using ..VisualGeometryToolkitCore: Circle, Ellipse, Circle3D, is_ellipse, boundary_points
import ..VisualGeometryToolkitCore: HomographyMap  # Import (not using) to extend the labeled camera lift

using ..VisualGeometryToolkitCore: Plane3D, distance  # For chirality-filtered P3P
using ..VisualGeometryToolkitCore: Line2D   # RadialCamera projects a plane point TO A LINE
using ..VisualGeometryToolkitCore: AbstractRadialModel   # RadialCamera joins the plane→pencil family
using ..VisualGeometryToolkitCore: RadialHomography      # the family's other member (conversion in calibration.jl)
import ..VisualGeometryToolkitCore: center               # ONE `center` generic (base/types.jl)
import ..VisualGeometryToolkitCore: direction            # the radial family's algebraic accessor (public, not exported)
import ..VisualGeometryToolkitCore: Rect  # Import (not using) to extend the constructor
import ..VisualGeometryToolkitCore: project  # Import to extend (shared with Primitives)
import ..VisualGeometryToolkitCore: pose, extrinsics, backproject, unproject, focal_length  # Camera interface functions
import ..VisualGeometryToolkitCore: RadialHomography  # division-model camera bridge (CameraModel(::RadialHomography))
import ..VisualGeometryToolkitCore: transfer  # Multi-view verb (extended on StereoRig in stereo.jl)
# Camera component accessors. `model` MUST extend the parent generic (which
# base/manifold.jl also extends for Manifold): defining a fresh `Cameras.model`
# would not merge into the already-bound `VisualGeometryToolkitCore.model` at
# `using .Cameras`, so the top-level export would silently lack the Camera
# method (latent since 0.8; exposed by the 0.10 imaging-chain tests).
import ..VisualGeometryToolkitCore: intrinsics, lens, model
import ..VisualGeometryToolkitCore: Manifold, SubsetManifold, retract, recenter, retract_jacobian, action_jacobian,
    local_coordinates, minimal_manifold, restrict, tangent_size   # the lens manifold (lens_parameters.jl) extends the contract
using ..VisualGeometryToolkitCore: SphereManifold   # RadialCamera's minimal manifold (calibration.jl)
import ..VisualGeometryToolkitCore: constraint_dim, _measurement_dim, _ambient_coordinates   # … and its place in the covariance machinery
import ..VisualGeometryToolkitCore: _precision, _has_division  # image_warp chain hooks: lens precision, a lens consumes a ray
import ..VisualGeometryToolkitCore: toml_value   # extended on lenses (lens_parameters.jl): a lens writes as its record
using ..VisualGeometryToolkitCore: pointwise!, ExecutionPolicy, Auto   # a lens over a batch of points (lens_rays.jl)
import ..VisualGeometryToolkitCore: aspect_ratio  # Extend interface stub (shared with Primitives)
using ..VisualGeometryToolkitCore: Attributed, csponds  # composition primitives (depth-attributed points, correspondence sets)
using ..VisualGeometryToolkitCore: sources, targets, centroid  # correspondence access protocol (#21)
using ..VisualGeometryToolkitCore: AbstractCamera  # Abstract camera type from interface
using ..VisualGeometryToolkitCore: sampson_distance, skew, FundMat  # GT-pose Sampson filter in validate_gt_metric; skew in the pose Jacobian
using ..VisualGeometryToolkitCore: HasBounds, NoBounds  # Bounds trait
import ..VisualGeometryToolkitCore: bounds_trait, extent           # extend the trait + accessor
using Base: ImmutableDict

# Import types and macros defined in core and geometry
using ..VisualGeometryToolkitCore: Size2, Quantity, px, mm, μm, m
using ..VisualGeometryToolkitCore: _check_pixels, _strip_pixels   # pixel-sized values: refuse a length (base/units.jl)
using FlexUnits: Dimensions, FixRat32   # FlexUnits' base-form quantity, `Quantity{Float64, Dimensions{FixRat32}}`
using ..VisualGeometryToolkitCore: EuclideanMap, CameraPose, CameraExtrinsics, to_affine, to_homogeneous, image_warp, image_warp!
import ..VisualGeometryToolkitCore: has_units, with_unit   # extended on the carriers: targets, mounts, rigs, cameras, captures
using ..VisualGeometryToolkitCore: translation, UnitfulEuclideanMap, _attach_unit, _ustrip_point, _attach_point, _stored_point,
    _check_one_form, _strip_metres, _translation_metres   # the one strip/attach of maps and points (euclidean_map.jl)
using FlexUnits: Units, DimensionError
using ..VisualGeometryToolkitCore: TargetPlaneFrame, PixelFrame, UnlabeledFrame, relabel, unlabel, Frame
using ..VisualGeometryToolkitCore: FramedImage   # rect-form undistort return currency (warp layer)
using ..VisualGeometryToolkitCore: HomographyMap   # rectify.jl — the per-view rect→view collineation
using ..VisualGeometryToolkitCore: ViewFrame, ImageFrame, TargetFrame  # per-view identity tags (minted by StereoRig)
using ..VisualGeometryToolkitCore: NormalizedImageFrame, DiagonalNormalization  # the normalization map's factory (imaging_chain.jl)
using ..VisualGeometryToolkitCore: GnomonicFrame  # per-view CALIBRATED frame (essential_form slots)
using ..VisualGeometryToolkitCore: s2_tangent_basis  # T_f S² basis: shared with EssentialManifold5 (backproject_with_jacobian)
using ..VisualGeometryToolkitCore: Frame, EquidistantFrame, DistortedFrame  # framed imaging chain (0.10)
using ..VisualGeometryToolkitCore: RetinalFrame  # RayMap's defining-frame parameter (0.11)
# The remaining retinal frames (0.13): the frame tags of the pure-projection
# fisheye stages.
using ..VisualGeometryToolkitCore: StereographicFrame, EquisolidFrame, OrthographicFrame
# FrameDependence (0.14): frame_transition's ModelDefined refusal and the
# sphere-section guards dispatch on the trait. `FrameDependence` itself is
# a docstring-scope import (used only by @ref cross-references, like
# epipolar_map above — same allowlist entry).
using ..VisualGeometryToolkitCore: FrameDependence, ModelDefined, frame_dependence
# Docstring-scope import: `epipolar_map` (referenced from stereo.jl) is used
# only by `@ref` cross-references in this module's docstrings — the import
# makes those references resolvable without a docs-build shim (listed in the
# ExplicitImports stale-import allowlist for the same reason).
using ..VisualGeometryToolkitCore: epipolar_map
using ..VisualGeometryToolkitCore: EssentialMat  # rig → labeled epipolar-form mints (stereo.jl)
import ..VisualGeometryToolkitCore: points         # extended: a target rig's world points (rig.jl)
import ..VisualGeometryToolkitCore: EpipolarForm  # Import (not using) to extend with the rig mint
using ..VisualGeometryToolkitCore: @smatrix_wrapper, AbstractCalibrationMat
import ..VisualGeometryToolkitCore: HomographyMat  # Import to extend constructor
import ..VisualGeometryToolkitCore: Extrinsics3   # Import to extend constructor: the planar-target pose from a homography (homography.jl)
using ..VisualGeometryToolkitCore: _cross2d   # the 2-D cross product (base/coordinates.jl): radial_camera_5pt
using ..VisualGeometryToolkitCore: _CTLeafTransformation   # the CT affines RadialCamera's ∘ accepts (calibration.jl)
using ..VisualGeometryToolkitCore: rescale_radial_coeff   # the one radial-coefficient scaling law (radial_models.jl; _rescale_lens)
using ..VisualGeometryToolkitCore: check_weights
import ..VisualGeometryToolkitCore: propagate   # extended for FramedAffine (imaging_chain.jl)
using ..VisualGeometryToolkitCore: CovMat, PrecMat, Uncertain, _junction_ok, _check_junction   # the checked statistical forms and the one frame-junction rule (base/frames.jl)
import ..VisualGeometryToolkitCore: constraint_with_jacobians   # extended for the radial camera and the division lift (calibration.jl), the full camera (camera_models.jl)
import ..VisualGeometryToolkitCore: transfer_ellipse          # extended for the lens camera (ellipse-preserving image of a plane conic)
import ..VisualGeometryToolkitCore: PerspectiveMap  # From CoordinateTransformations (imported to extend)
import ..VisualGeometryToolkitCore: p3p  # P3P solver from solvers.jl (imported to extend with correspondence wrapper)
import ..VisualGeometryToolkitCore: solve, sample_size, max_solutions   # the minimal-solver protocol, extended by RadialCamera5pt (calibration.jl)
using ..VisualGeometryToolkitCore: MinimalSolver, FixedModels


# Include sensor specifications
include("sensors.jl")

# Include modular camera components
include("intrinsics.jl")
include("lenses.jl")         # the lens families: Lens/LiftLens, projections, terms, combinators
include("projections.jl")
include("camera_models.jl")
include("lens_rays.jl")       # THE lens contract: project/backproject on rays + the derived Jacobians
include("lens_parameters.jl") # parameter chart, ∂lens/∂θ, TOML lens registry (0.22.1)
include("generic_models.jl") # Schöps et al. central/non-central generic grid models (0.16)
include("imaging_chain.jl")   # framed stages between image frames + imaging_chain mint (0.10)

# ==============================================================================
# Camera - Complete camera in 3D space (model + pose)
# ==============================================================================
# NOTE: AbstractCamera is defined in base/interface.jl to break circular
# dependencies. This allows other modules to work with cameras without
# depending on the full Cameras submodule.
# ==============================================================================

"""
    Camera{M, Rt} <: AbstractCamera

A camera in 3D space, consisting of a camera model (intrinsics + lens) and extrinsics (pose).

Dispatch on the lens through the model parameter:
`Camera{<:CameraModel{<:Any, <:MyLens}}` (the [`PinholeCamera`](@ref) alias
is that form for `Pinhole`).

Use `project(camera, X)` to map a 3D world point to 2D pixel coordinates,
and `backproject(camera, u)` / `unproject(camera, u, depth)` for the
inverse directions. The transform is computed on demand and inlined.

# Type parameters
- `M <: AbstractCameraModel`: camera model type ([`CameraModel`](@ref), or a
  grid model)
- `Rt <: EuclideanMap`: extrinsics type

# Fields
- `model::M`: Camera model (intrinsics + lens)
- `extrinsics::Rt`: Camera extrinsics (world-to-camera transform)

# Examples
```julia
# Create camera with logical intrinsics
model = CameraModel(800.0px, [320.0px, 240.0px])
extrinsics = EuclideanMap(RotMatrix(I), [0.0, 0.0, 0.0])
camera = Camera(model, extrinsics)

# Dispatch on lens (via the model parameter; PinholeCamera is the
# ready-made alias for the Pinhole case)
foo(::PinholeCamera) = "pinhole path"
foo(::Camera{<:CameraModel{<:AbstractIntrinsics, <:KannalaBrandt}}) = "fisheye path"

# Project a world point
X_world = SVector(1.0mm, 2.0mm, 10.0mm)
u = project(camera, X_world)
```
"""
struct Camera{
        M <: AbstractCameraModel,   # any unposed model — parametric OR generic (0.16)
        Rt <: EuclideanMap,
    } <: AbstractCamera
    model::M
    extrinsics::Rt
end

"""
    PinholeCamera

Cameras whose lens is exactly `Pinhole` — the alias for lens dispatch now
that `Camera` carries only `{M, Rt}` (the former `{I, L, Rt, M}` repeated
what `M` already determines).  Consumed by the induced-homography
constructors in geometry/cameras/homography.jl.
"""
const PinholeCamera = Camera{<:CameraModel{<:AbstractIntrinsics, Pinhole}}

# THE camera shape projection (a z = 0 plane circle/ellipse, a 3D circle): a
# pinhole images the conic exactly through the plane homography — an `Ellipse`
# when it stays one; otherwise (across the line at infinity), and for every
# distorting lens, `n` boundary samples projected into a `Polygon`.
function _camera_project_conic(camera::Camera, shape, n::Int = 64)
    Q = _plane_conic(camera, shape, lens(camera))
    Q !== nothing && is_ellipse(Q) && return Ellipse(Q)
    return _camera_project_world_points(camera, _world_boundary(shape, n))
end
# The exact image conic through the plane-induced homography — pinhole only.
_plane_conic(camera::Camera, shape::Union{Circle, Ellipse}, ::Pinhole) = HomographyMap(HomographyMat(camera))(shape)
_plane_conic(camera::Camera, c::Circle3D, ::Pinhole) =
    HomographyMap(HomographyMat(camera, c.center, c.u, c.v))(Circle(Point2(0.0, 0.0), c.radius))   # the circle's local (u, v) frame
_plane_conic(::Camera, _, ::AbstractLensModel) = nothing
# The world boundary samples: a plane shape on z = 0, a 3D circle as is.
_world_boundary(shape::Union{Circle, Ellipse}, n::Int) = (Point3{Float64}(p[1], p[2], 0.0) for p in boundary_points(shape, n))
_world_boundary(c::Circle3D, n::Int) = boundary_points(c, n)

"""
    transfer_ellipse(camera::Camera, shape::Union{Circle, Ellipse}) -> Ellipse | nothing

The ellipse-preserving image of a `z = 0` plane circle/ellipse under
`camera`: the exact conic through the plane-induced homography of the
pinhole twin, then — for a distorting lens — the lens's local affinity at
the transferred centre — `distortion_map`'s stages applied to the transferred conic. `project(camera, shape)` samples the
boundary into a polygon instead; this is the form the guided association
of blob boards consumes (an ellipse to overlap, a centre to search at) and
it stays an ellipse for every lens. `nothing` behind the camera, across the
line at infinity, or where the lens map is not finite at the centre.
"""
function transfer_ellipse(camera::Camera, shape::Union{Circle, Ellipse})
    m = camera.model
    # cheirality at the centre: a plane conic behind the camera has no image
    # (a homography maps it to a perfectly good conic, and a lens's forward
    # map is finite on both sides of z = 0 — neither would report it)
    c = centroid(shape)
    Xc = camera.extrinsics.R * SVector{3, Float64}(c[1], c[2], 0.0) + camera.extrinsics.t
    Xc[3] > 0 || return nothing
    H = HomographyMap(Camera(pinhole(m), camera.extrinsics))
    return _lens_ellipse(m.lens, m, transfer_ellipse(H, shape))
end

# The lens's part of the transfer, by dispatch on the lens: nothing to move behind
# a failed pinhole transfer, no stage for a pinhole, and otherwise the lens chain,
# COMPOSED, not unrolled: `distortion_map` is
# `AffineMap(K) ∘ lens ∘ AffineMap(K⁻¹)`, and `∘` on CoordinateTransformations
# `Transformation`s yields a `ComposedTransformation` — which `transfer_ellipse`
# already walks, propagating `nothing`. Each stage's local affinity is the
# transform's own `transform_deriv` at the transferred centre, so this is the
# same arithmetic the three unrolled statements did.
#
# (Those statements carried a note that `∘` on "an abstractly-typed lens"
# would infer through `Base.ComposedFunction`. Neither holds: the lens is the
# type parameter `L` of `CameraModel{I, L}`, hence concrete, and composing
# `Transformation`s gives a `ComposedTransformation`, never a
# `Base.ComposedFunction` — the distinction `region_maps.jl` also relies on.)
_lens_ellipse(lens, m, ::Nothing) = nothing
_lens_ellipse(::Pinhole, m, ep::Ellipse) = ep
_lens_ellipse(lens, m, ep::Ellipse) = transfer_ellipse(distortion_map(m), ep)

# Rect: tessellate edges, lift to z=0, project → Polygon
_camera_project_rect(camera::Camera, rect::GeometryBasics.HyperRectangle{2}, n::Int) =
    _camera_project_points(camera, _rect_edge_samples(identity, rect, n))

# Shared: lift 2D → 3D at z=0, project, return Polygon
_camera_project_points(camera::Camera, pts) =
    _camera_project_world_points(camera, (Point3{Float64}(p[1], p[2], 0.0) for p in pts))

# Shared: project 3D world points, return Polygon
function _camera_project_world_points(camera::Camera, pts)
    return GeometryBasics.Polygon([Point2(project(camera, Point3{Float64}(p...))) for p in pts])
end

# For non-EuclideanMap extrinsics, convert first. The model bound keeps a
# wrong first argument a MethodError instead of infinite recursion through
# EuclideanMap(::EuclideanMap).
Camera(model::AbstractCameraModel, extrinsics) = Camera(model, EuclideanMap(extrinsics))

"""
    Camera(model::CameraModel, e::CameraExtrinsics) -> Camera
    Camera(model::CameraModel, p::CameraPose) -> Camera

Construct a `Camera` from a frame-labeled map, so the caller states which
frame it holds instead of relying on the positional convention.

The bare positional constructor `Camera(model, extrinsics::EuclideanMap)`
with an UNLABELED map takes the world-to-camera map (documented convention,
unchecked); a caller holding a camera-to-world map (a `lookat`-style pose, a
renderer's `sensor.to_world`) who supplies it positionally stores the WRONG
map without any error — every accessor then silently answers for the
inverse camera. With frame labels that mistake is unrepresentable:
`CameraExtrinsics` (being the extrinsics-labeled `EuclideanMap`) stores
directly into the extrinsics slot via the default constructor, `CameraPose`
inverts internally, and `pose(Camera(model, P::CameraPose)) ≈ P`.
"""
Camera(model::AbstractCameraModel, p::CameraPose) = Camera(model, inv(p))

"""
    Camera(K::StaticMatrix{3,3}, R::StaticMatrix{3,3}, t::StaticVector{3};
           check=true, project=false) -> Camera

Boundary constructor taking raw matrix data (typical for SfM dataset loaders).
`R` is wrapped via the `EuclideanMap(::SMatrix, ...)` path which validates
(and optionally snaps via `project=true`) before promoting to `RotMatrix`.
"""
function Camera(
        K::StaticMatrix{3, 3, T},
        R::StaticMatrix{3, 3, T},
        t::StaticVector{3, T};
        check::Bool = true,
        project::Bool = false
    ) where {T}
    K_calib = CameraCalibrationMat(K; check = check)
    model = CameraModel(K_calib)
    return Camera(model, EuclideanMap(SMatrix{3, 3, T, 9}(R), t; check = check, project = project))
end

# ============================================================================
# Camera Type Aliases
# ============================================================================

# NOTE (2026-07, #15): the PhysicalCamera/LogicalCamera aliases are GONE —
# PhysicalIntrinsics was demoted (every intrinsics is a K matrix), so the
# split no longer denotes anything.

# ============================================================================
# BoundedCamera{C} — wrap any AbstractCamera with pixel-space image bounds
# ============================================================================

"""
    BoundedCamera{C<:AbstractCamera} <: AbstractCamera

A wrapper around any `AbstractCamera` that adds pixel-space image bounds
(width × height) without disturbing the underlying camera's intrinsics or
units.  Used by SfM/MVS dataset readers (ETH3D, ScanNet, COLMAP exports)
where the camera is described by its calibration plus an explicit image
extent — information the bare `Camera` type doesn't retain.

The wrapped camera may be UNPOSED — a bare `AbstractCameraModel` — which is
COLMAP's "camera" row (intrinsics, lens and image size, no pose) and the
element of a stored camera collection (VisualGeometryToolkitIO's `cameras` kind).
`BoundedCamera(Camera(model(bc), pose), bc.extent)` is its posed counterpart;
the pose accessors of an unposed one are a `MethodError`.

# Field access
- `bc.camera`, `bc.extent` — direct fields.

# Method delegation (via `MacroTools.@forward`)
The standard Camera accessors (`intrinsics`, `lens`, `pose`, `extrinsics`,
`focal_length`, `model`, `calibration_matrix`) and the
projection verbs are forwarded to the wrapped `camera`, so
`intrinsics(bc) === intrinsics(bc.camera)` etc.; `is_in_front(bc, p)` too.

# Trait
`bounds_trait(::BoundedCamera) = HasBounds()` — methods that need image
extent dispatch on `BoundsTrait`.  See `extent(::BoundedCamera) = bc.extent`.

# Example
```julia
inner = Camera(CameraModel(K; lens=Pinhole()), pose)
bc    = BoundedCamera(inner, Size2(6048, 4032))
intrinsics(bc)     # forwarded → CameraCalibrationMat
bc.extent          # Size2(6048, 4032) — direct field
extent(bc)         # same, via trait-dispatched accessor
```
"""
struct BoundedCamera{C <: Union{AbstractCamera, AbstractCameraModel}} <: AbstractCamera
    camera::C
    extent::Size2{Int}
end

# Trait opt-in — instance form (matches VGC convention: see has_boundary(::Ellipse) etc.)
bounds_trait(::BoundedCamera) = HasBounds()

# Method delegation to the wrapped camera (`bc.camera`).  Uses MacroTools.@forward
# to generate the per-method overloads so we don't hand-roll one line per accessor.
# (Method dispatch is not the same as property forwarding; this is the canonical
# Julia idiom for "method on wrapper falls through to inner".)
MacroTools.@forward BoundedCamera.camera (
    model, intrinsics, lens, focal_length, calibration_matrix,
    pose, extrinsics,
    project, backproject, unproject, is_in_front,
)

# ----------------------------------------------------------------------------
# Units — a physical camera is one whose extrinsics are (translation in metres)
# ----------------------------------------------------------------------------
#
# The model's pixel quantities are not lengths of the scene; only the mount
# moves. The camera is rebuilt with its type parameters stated, so a stripped
# map is stored as it is (the bare constructor would invert a `CameraPose`).

# A camera posed by a physical map (bounds stated, so dispatch ranks it above `Camera`).
const _UnitfulCamera = Camera{<:AbstractCameraModel, <:UnitfulEuclideanMap{3}}

has_units(c::Camera) = has_units(c.extrinsics)
has_units(c::BoundedCamera) = has_units(c.camera)
has_units(::AbstractCameraModel) = false   # an unposed camera holds no length
_rebuilt(c::Camera, X::EuclideanMap) = Camera{typeof(c.model), typeof(X)}(c.model, X)

"""
    ustrip(u::Units, camera::Camera | BoundedCamera) -> same type of camera

A physical camera's plain form: its extrinsics stripped in the length unit
`u` (`ustrip(u, E)`), the model and any extent unchanged. The inverse is
[`with_unit`](@ref)`(camera, u)`. An `ArgumentError` on a camera already in
the produced form.
"""
FlexUnits.ustrip(u::Units, c::Camera) = _rebuilt(c, ustrip(u, c.extrinsics))
FlexUnits.ustrip(u::Units, c::BoundedCamera{<:AbstractCamera}) = BoundedCamera(ustrip(u, c.camera), c.extent)
with_unit(c::Camera, u::Units) = _rebuilt(c, _attach_unit(c.extrinsics, u))
with_unit(c::BoundedCamera{<:AbstractCamera}, u::Units) = BoundedCamera(with_unit(c.camera, u), c.extent)

"""
    extent(cam) -> Size2{Int}

Pixel-space image extent (width × height).  Trait-dispatched: errors at
compile time for cameras that don't carry bounds.
"""
extent(cam) = extent(bounds_trait(cam), cam)
extent(::HasBounds, bc::BoundedCamera) = bc.extent
extent(::NoBounds, cam) =
    error("camera lacks image extent — wrap with BoundedCamera if you have it")

# Method-based accessors

"""
    pose(camera) -> CameraPose

Camera pose (camera-to-world transform), inverse of extrinsics. Returns a
frame-labeled map (`CameraFrame → WorldFrame`).

```julia
camera = Camera(model, extrinsics)
P = pose(camera)       # Camera-to-world
P.t                    # Camera position in world coordinates
```
"""
pose(c::Camera) = inv(extrinsics(c))

"""
    extrinsics(camera) -> CameraExtrinsics

Camera extrinsics (world-to-camera transform). Returns a frame-labeled map
(`WorldFrame → CameraFrame`): the extrinsics slot is world-to-camera by
contract, so the accessor labels it even when the camera was constructed
from an unlabeled map (identity when already labeled).

```julia
camera = Camera(model, ext)
E = extrinsics(camera)  # World-to-camera
E(world_point)          # Transform point to camera coordinates
```
"""
extrinsics(c::Camera) = CameraExtrinsics(c.extrinsics)

"""
    is_in_front(camera, point::Point3) -> Bool

Chirality test: returns true if point is in front of the camera (z > 0 in camera coords).

Broadcasts over points:
```julia
mask = is_in_front.(Ref(camera), world_points)
```
"""
is_in_front(c::Camera, p::Point3) = extrinsics(c)(p)[3] > 0
# a physical camera: the test in metres, the point a length (a plain one is a DimensionError)
is_in_front(c::_UnitfulCamera, p::Point3) = is_in_front(ustrip(m, c), Point3(_strip_metres(p)))

"""
    is_in_front(camera, x, y, z) -> AbstractArray{Bool}

Broadcasted component-array (SoA) form of the chirality test: world-frame
point components in, camera-frame `z > 0` mask out. Per-element semantics
match `is_in_front(camera, Point3(xᵢ, yᵢ, zᵢ))`; the component layout keeps
the whole test one fused broadcast on GPU arrays (hoisted from the
RayTracedCalibration GPU pipeline).
"""
function is_in_front(
        c::Camera, x::AbstractArray, y::AbstractArray, z::AbstractArray
    )
    ext = extrinsics(c)
    R = ext.R
    t = ext.t
    z_cam = R[3, 1] .* x .+ R[3, 2] .* y .+ R[3, 3] .* z .+ t[3]
    return z_cam .> zero(eltype(z_cam))
end
is_in_front(c::_UnitfulCamera, x::AbstractArray, y::AbstractArray, z::AbstractArray) =
    is_in_front(ustrip(m, c), _strip_metres(x), _strip_metres(y), _strip_metres(z))

"""
    focal_length(camera::Camera)

Get focal length from camera's intrinsics: the `(fx, fy)` tuple in pixels.

# Examples
```julia
K = CameraCalibrationMat(800.0px, [320.0px, 240.0px])
model = CameraModel(LogicalIntrinsics(K), Pinhole())
camera = Camera(model, extrinsics)
fx, fy = focal_length(camera)  # Returns (800.0, 800.0)
```
"""
focal_length(camera::Camera) = focal_length(intrinsics(camera))

"""
    calibration_matrix(camera::Camera) -> SMatrix{3,3,Float64,9}

Return the 3×3 calibration matrix K as a plain `SMatrix`.

# Example
```julia
camera = Camera(CameraModel(; focal=800.0, pp=(320.0, 240.0)), ext)
K = calibration_matrix(camera)  # [800 0 320; 0 800 240; 0 0 1]
```
"""
calibration_matrix(camera::Camera) = calibration_matrix(camera.model)

"""
    model(camera::Camera) -> CameraModel

Return the camera's underlying `CameraModel` (intrinsics + lens), the
single object the lower-level `project`/`backproject`/`transfer` kernels
take.  Method form so wrappers (e.g. `BoundedCamera`) can delegate via
`@forward`.
"""
model(c::Camera) = c.model
model(m::AbstractCameraModel) = m   # an unposed camera (a BoundedCamera of a model) is its own model

"""
    intrinsics(camera::Camera)
    intrinsics(model::CameraModel)

Return the camera's intrinsics block (e.g. `LogicalIntrinsics`,
`PhysicalIntrinsics`).  Equivalent to `camera.model.intrinsics` /
`model.intrinsics` but lets downstream code stop reaching into the field.
"""
intrinsics(camera::Camera) = intrinsics(camera.model)
intrinsics(model::CameraModel) = model.intrinsics

"""
    lens(camera::Camera)
    lens(model::CameraModel)

Return the camera's lens model (e.g. `Pinhole`, a `Lens`, or any
user-defined `<: AbstractLensModel`).  Use this in dispatch when a downstream
package wants to specialize on lens type without reaching into
`camera.model.lens`.
"""
lens(camera::Camera) = lens(camera.model)
lens(model::CameraModel) = model.lens

"""
    default_frustum_depth(camera::Camera) -> Float64

Default frustum depth for visualization: fx from K, a natural depth scale
in the camera's working units.
"""
default_frustum_depth(camera::Camera) = focal_length(camera)[1]

# ============================================================================
# Camera Projection (World → Pixel)
# ============================================================================

"""
    project(camera::Camera, X_world::AbstractVector) -> Point2

Project 3D point from world coordinates to pixel coordinates.

# Arguments
- `camera::Camera`: Camera with model and extrinsics
- `X_world::AbstractVector`: 3D point in world coordinates (with units)

# Returns
- Point2 with pixel coordinates (px units)

# Example
```julia
model = CameraModel(16.0mm, Size2(width=5.86μm/px, height=5.86μm/px), [320.0px, 240.0px])
extrinsics = lookat(SVector(0.0, 0.0, 10.0), SVector(0.0, 0.0, 0.0), SVector(0.0, 1.0, 0.0))
camera = Camera(model, extrinsics)
X_world = SVector(1.0mm, 2.0mm, 3.0mm)
u = project(camera, X_world)
```
"""
@inline project(camera::Camera, X::AbstractVector) =
    project(camera.model, camera.extrinsics(X))
# A physical camera strips at entry, in metres: exactly the plain camera's call
# on the point in metres (a plain point is a `DimensionError`).
@inline project(camera::_UnitfulCamera, X::AbstractVector) =
    project(camera.model, ustrip(m, camera.extrinsics)(_strip_metres(X)))

"""
    project(camera::Camera, shape::Union{Circle, Ellipse}) -> Ellipse or Polygon

Project a 2D circle or ellipse that lies on the world z=0 plane through
the camera. For pinhole cameras, the analytic conic projection yields an
`Ellipse`; partially-behind-camera cases or distorting lens models fall
back to boundary-point sampling and return a `Polygon`.
"""
@inline project(camera::Camera, shape::Union{Circle, Ellipse}) =
    _camera_project_conic(camera, shape)

"""
    project(camera::Camera, c::Circle3D) -> Ellipse or Polygon

Project an arbitrarily-placed 3D circle through the camera. For pinhole
cameras, the plane homography induced by the circle's local (u, v) frame
gives the analytic image conic; other cases fall back to boundary
sampling and return a `Polygon`.
"""
@inline project(camera::Camera, c::Circle3D) =
    _camera_project_conic(camera, c)

"""
    project(camera::Camera, rect::HyperRectangle{2}, n::Int=20) -> Polygon

Project a 2D rectangle on the world z=0 plane, tessellating each edge
into `n` samples.
"""
@inline project(camera::Camera, rect::GeometryBasics.HyperRectangle{2}, n::Int = 20) =
    _camera_project_rect(camera, rect, n)

"""
    project(camera::Camera, origin::Point3, dir::Vec3) -> (Point2, Vec2)

Project a 3D origin and displacement through the camera and return the
2D arrow endpoints — the origin's image and the vector from the origin's
image to `(origin + dir)`'s image. Use to plot a 3D normal or direction
as an image-space arrow:

```julia
origin_2d, vec_2d = project(cam, c.center, c.radius * normal(c))
arrows2d!(ax, origin_2d, vec_2d; color=:magenta)
```
"""
@inline function project(camera::Camera, origin::GeometryBasics.Point3, dir::GeometryBasics.Vec3)
    o_img = project(camera, Point3{Float64}(origin...))
    tip_img = project(camera, Point3{Float64}((origin .+ dir)...))
    return (Point2(o_img...), Vec2((tip_img .- o_img)...))
end

"""
    backproject(camera::Camera, u::StaticVector{2, Float64}) -> Vec3

Backproject a 2D pixel to a 3D ray direction in **world** coordinates
(normalized).  Returned as a `Vec3` because it represents a direction —
not translatable under affine maps.  The ray's origin is the camera
center (`pose(camera).t`); points along it are
`pose(camera).t + s · dir` for `s > 0`.
"""
@inline backproject(camera::Camera, u::StaticVector{2, Float64}) =
    Vec3(pose(camera).R * backproject(camera.model, u))

@inline function backproject(camera::Camera, points::AbstractVector{<:StaticVector{2, Float64}})
    R = pose(camera).R      # hoisted: one extrinsics inversion for the batch
    m = camera.model
    return [Vec3(R * backproject(m, u)) for u in points]
end

"""
    unproject(camera::Camera, u::AbstractVector{Float64}, depth::Float64) -> Point3

Unproject a 2D pixel at a given camera-space depth (Z along the camera's
forward axis) to a 3D **point** in world coordinates.  Returned as a
`Point3` because it represents a position — translatable under affine
maps.  Works for both `PhysicalIntrinsics` (metric units; `depth` in the
chosen metric unit, e.g. mm) and `LogicalIntrinsics` (unitless pixels;
`depth` carries whatever unit the caller picks for their scene).
"""
@inline unproject(
    camera::Camera,
    u::AbstractVector{Float64}, depth::Float64
) =
    Point3(pose(camera)(unproject(camera.model, u, depth)))
# A physical camera takes a depth that is a length and returns the point in
# metres: the plain call in metres, its result read back in metres.
unproject(camera::_UnitfulCamera, u::AbstractVector{Float64}, depth::Quantity) =
    _attach_point(m, unproject(ustrip(m, camera), u, ustrip(m, depth)))

# ============================================================================
# Camera Accessor Functions
# ============================================================================

# Include remaining camera system components
include("targets.jl")  # AbstractTarget, PointTarget, PlanarTarget{W} (+ the warp and calibratable-target protocols)
include("rig.jl")     # Rig (sensors: tuple storage; targets: dict storage; both: the two side by side; gauge in the frame) + Mounted — before the stereo alias
include("captures.jl")  # Trajectory, TargetPoses, TargetPoints2D/3D: per-frame values of a capture (+ convert_frame)
include("stereo.jl")
include("rectify.jl")    # stereo rectification on the rig types (0.16)
include("utilities.jl")
include("random.jl")

# Include planar homography (depends on Camera)
include("homography.jl")   # camera-derived HomographyMat (moved out of geometry/ 0.29)
include("calibration.jl")  # zhang_intrinsics — Zhang IAC regression from plane homographies

# =============================================================================
# EXPORTS - Organized by category
# =============================================================================

# -----------------------------------------------------------------------------
# 1. SENSORS - Physical CMOS specifications
# -----------------------------------------------------------------------------
export Sensor, sensor                  # Sensor type and lookup
export CMOS_SENSORS                    # Database of common sensors

# -----------------------------------------------------------------------------
# 2. INTRINSICS - Calibration parameters
# -----------------------------------------------------------------------------
export AbstractIntrinsics
export LogicalIntrinsics               # Pixel-based (fx, fy, cx, cy)
export CameraCalibrationMat                        # 3x3 pixel-space calibration matrix
export zhang_intrinsics                # Zhang IAC regression from plane homographies

# -----------------------------------------------------------------------------
# 3. LENS MODELS - Projection + distortion
# -----------------------------------------------------------------------------
export AbstractLensModel
export LensManifold                       # the lens family's Manifold: lens_parameters' coordinates as offsets from an anchor
export Pinhole                         # Ideal perspective (no distortion)
export Lens, LiftLens, Division, KannalaBrandt, radial_ratio   # the two lens kinds and the named models
export Scaramuzza                      # Scaramuzza's omnidirectional polynomial lift (MATLAB fisheyeIntrinsics)
export AbstractProjection, Perspective, Equidistant, Stereographic, Equisolid, Orthographic
export DoubleSphere, Unified, ExtendedUnified   # the parametric projections
export AbstractDistortionTerm, Radial, Tangential, ThinPrism, Tilt, FieldOfView, Additive, Chain, Even, Full
export radial, isdelta, projection_frame, parameters, lens_record, lens_from_record
export division_bounded, undistorted_rect, valid_rect, horizon_radius, fov_radius, DEFAULT_MAX_FOV   # whole-extent undistortion
export undistort, undistort!           # Remove lens distortion from images
export radial_camera_5pt, RadialCamera5pt   # the radial camera's 5-point minimal kernel and its solver
export radial_rows                     # RadialCamera's rows accessor (public, not exported, at the top level)
export RadialCamera                    # the 1-D radial camera model (constraint_with_jacobians/whiten)
export distortion_map                  # Undistorted→distorted COLMAP pixel map
export imaging_transform, pinhole
export undistort_normalized, InverseLens
export lens_parameters, with_parameters, lens_parameter_names
export project_with_jacobians          # the image point and both lens Jacobians in one evaluation (lens_rays.jl)
export reprojection_error              # project(camera, pose(X)) − u, Jacobian-free (camera_models.jl)
export has_distortion
# The lens contract's derived perspective view and radial profile.
export monotone_radius
export distort_normalized
export view_angle, projected_radius   # the radial projection function r(θ) and its inverse
# Framed imaging chain (0.10): typed frame-to-frame stages with checked
# junctions. `imaging_chain` is the fully-typed mint; the legacy
# `imaging_transform` shape is unchanged.
export FramedStage, FramedPerspective, FramedEquidistantProjection
export FramedDistort, FramedUndistort, FramedAffine, FramedChain
export imaging_chain, distort_stage, undistort_stage, calibration_stage
export similarity_scale              # a framed similarity's length ratio (thresholds through a normalization)
export distortion_frame
# The remaining retinal-frame projections (0.13): stage-1 family members
# for the stereographic/equisolid/orthographic frames, plus the framed
# frame-to-frame transition mint (one implementation through the sphere).
export FramedStereographicProjection, FramedEquisolidProjection
export FramedOrthographicProjection
export FrameTransition, frame_transition
# Sphere self-map (0.11): the lens's ray re-orientation, free of any image frame,
# D: S² → S², framed {ViewFrame{id} → ViewFrame{id}}.
export RayMap, InverseRayMap, ray_map
# A lens over a batch of points (pointwise! of the per-point maps).
export project!, backproject!, distort_normalized!, undistort_normalized!
# Backward compatibility aliases

# -----------------------------------------------------------------------------
# 4. CAMERA TYPES - Complete camera specifications
# -----------------------------------------------------------------------------
export CameraModel                     # Intrinsics + lens
export AbstractCamera, Camera          # Camera with extrinsics
export PinholeCamera                   # Lens-dispatch alias
export BoundedCamera, extent           # Wrapper that adds image bounds + accessor
export model                            # CameraModel accessor on Camera/AbstractCamera

# -----------------------------------------------------------------------------
# 5. STEREO - Multi-camera geometry
# -----------------------------------------------------------------------------
export Rig, Mounted, source, target, target_frames  # one Rig for sensors and targets: (model, mount) members; StereoRig = 2-tuple alias
export sensor_rig, target_rig             # the two halves of a rig carrying both
export AbstractTarget, PointTarget, PlanarTarget, resolve, is_resolved   # the target as a model (targets.jl)
export AbstractWarp, Rigid, warp_parameters, warp_parameter_names, warp_jacobian   # a planar target's warp
export target_parameters, target_parameter_names, regularizer, TargetManifold      # the calibratable-target protocol
export Trajectory, TargetPoses, TargetPoints2D, TargetPoints3D   # per-frame values of a capture (captures.jl)
export TargetPoint2D, TargetPoint3D, UncertainPixel, UncertainPoint3
export convert_frame                      # the one way to change the pixel frame of pixel data
export rectify                            # stereo rectification: rectified pinhole rig + FramedImage pair
export view_angle_limit                   # honest θ-domains
export radial_part                         # a lens's radially symmetric part
export AbstractCameraModel, CentralGenericModel, NoncentralGenericModel, Ray3  # generic grid models (Schöps et al.)
export spline_support, spline_weights, window_controls, window_indices, direction_tangents, default_generic_grid  # the grid model's window interface (0.22.2)
export with_model                         # same camera/rig, new model — rebuilds through the validating constructor
export reference_frame                    # rig/target-rig reference-frame type (submodule-level only:
#   the top module deliberately does not re-export it — import explicitly)
export StereoRig, depth_transfer          # `epipolar_map` (0.8): renamed — the old
#   unit-depth closure is `depth_transfer`; `epipolar_map` now names ONLY the
#   EpipolarForm point → line action (top module, geometry/forms.jl).
export view_frames, image_frames           # ViewFrame/ImageFrame tags minted from ids(rig)
export ids                                # view-identity symbols, read off the labeled extrinsics (0.9)
export essential_form                     # rig → EpipolarForm{GnomonicFrame…}(E) mint (0.9);
#   EpipolarForm(rig) is the pixel-frame FundMat mint (method on the parent's type)
export validate_gt_inliers, validate_gt_metric

# -----------------------------------------------------------------------------
# 5b. LENS MODELS - composable distortion layers
# -----------------------------------------------------------------------------
# Pure-projection fisheye family completed (0.13): each the identity in its
# defining retinal frame (the frame IS the model).
# Double-sphere model (0.14): the Usenko–Demmel–Cremers central lens with
# the paper's closed-form unprojection. Its defining frame family is
# ξ-parameterized (ModelDefined), so the framed chain serves it through
# the fused model-carrying projection stage.
export FramedLensProjection

# -----------------------------------------------------------------------------
# 6. PROJECTION FUNCTIONS - World ↔ Pixel transforms
# -----------------------------------------------------------------------------
export project, backproject, unproject, transfer
export backproject_with_jacobian   # the bearing AND its pixel Jacobian: calibrated geometry entry point

# -----------------------------------------------------------------------------
# 7. CAMERA ACCESSORS - Extract camera properties
# -----------------------------------------------------------------------------
export focal_length, focal_length_for_fov, calibration_matrix
export pose, extrinsics, is_in_front, default_frustum_depth

# -----------------------------------------------------------------------------
# 8. CAMERA CONSTRUCTION
# -----------------------------------------------------------------------------
export lookat                          # Construct extrinsics from (eye, target, up)
export lookat_pose                     # Camera-to-world pose form of lookat (pairs with Camera(model; pose=...))

# -----------------------------------------------------------------------------
# 9. POSE ESTIMATION - P3P and calibration
# -----------------------------------------------------------------------------
export p3p, sample_p3p

# -----------------------------------------------------------------------------
# 10. HOMOGRAPHY - Projective transformations
# -----------------------------------------------------------------------------
export HomographyMat                   # 3x3 homography matrix

end # module Cameras
