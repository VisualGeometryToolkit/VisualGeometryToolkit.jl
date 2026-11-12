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
- `PosedCamera`, `PosedPinholeCamera`
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
f = focal_length(s, deg2rad(40.0))
model = CameraModel(f, s.pitch, centroid(Rect(s)) .* px)
camera = PosedCamera(model, lookat(...))
```
"""
module Cameras

public pixel_intrinsics   # the pixel intrinsics of a camera (the stack's fits read them)
public coefficient_names  # a lens leaf type's coefficient names (the stack's lens fits read them)
public reframed_calibration   # the pixel-frame change of a K (VisualGeometryToolkitIO's stored K reads and writes through it)
public RadialSymmetry, RadiallySymmetric, RadiallyAsymmetric   # whether a lens is radially symmetric about the principal point (a Holy trait; RadialCamera and Estimation's fit_camera read it)

# Import general Julia packages directly (not through parent)
using StaticArrays: StaticArrays, SVector, SMatrix, MMatrix, StaticVector, StaticMatrix
using ForwardDiff: ForwardDiff   # a radial profile's centre gradient, a two-partial Dual (vignetting.jl)
using LinearAlgebra: LinearAlgebra, normalize, cross, norm, det, svd, dot, diag
using Statistics: Statistics   # cov of an Uncertain point (targets.jl, captures.jl)
using LogExpFunctions: logit, logistic   # DoubleSphere's α chart (lens_chart.jl)
using Random: Random, randperm
import FlexUnits
using FlexUnits: ustrip, dimension
using StructArrays: StructArrays, StructArray
using IntervalSets: IntervalSets
using CoordinateTransformations: CoordinateTransformations, AffineMap, ComposedTransformation   # ∘: a lens's distortion stages, a target calibration's height field after its in-plane kind
import CoordinateTransformations: transform_deriv, transform_deriv_params
using Rotations: Rotations, Rotation, RotMatrix, Angle2d, rotation_angle
using GeometryBasics: GeometryBasics
using MacroTools: MacroTools
using KernelAbstractions: KernelAbstractions, @kernel, @index, @Const, get_backend
using Adapt: Adapt, adapt   # device upload of the controls and rays; the spline models' `adapt_structure` (spline_lens.jl, noncentral_spline_model.jl)

# Import from parent module (VisualGeometryToolkitCore-specific types only)
import KernelAbstractions as KA   # the spline models' batched kernels
using ..VisualGeometryToolkitCore: RetiredKeyword, refuse_retired_keyword   # LensSpec refuses `free` (base/utils.jl)
using ..VisualGeometryToolkitCore: _pixel_extent   # a size in pixels, plain or `px` (homography_solvers.jl)
using ..VisualGeometryToolkitCore: _pushforward, _pullback   # the constant-Jacobian transport laws (propagate_framed.jl)
using ..VisualGeometryToolkitCore: Point2, Point3, Vec2, Vec3
using ..VisualGeometryToolkitCore: affine_part   # a calibration matrix's pixel affine (transforms/conversions.jl)
using ..VisualGeometryToolkitCore: _rect_edge_samples   # the one per-edge rect walk (primitives/rectangles.jl)
using ..Primitives: rect_edges, EdgeGrid, Closed   # its edges on a closed grid: the frame walk (camera_models.jl)
using ..Primitives: _closed_meet   # Core's one rect-meet convention, closed (primitives/regions.jl): a node's support clip (support.jl)
using ..VisualGeometryToolkitCore: block_order   # grouped data: the sorted-key block order (base/cspond_set.jl)
using ..VisualGeometryToolkitCore: natural_extrapolation   # the natural boundary condition of a uniform cubic lattice (numerics/bspline_energy.jl, BSplineKit)
using ..VisualGeometryToolkitCore: difference_stencil      # the forward-difference coefficients (numerics/bspline_energy.jl): Δ³c = 0's extrapolation
using ..VisualGeometryToolkitCore: grid_brent_minimum   # the one grid-and-Brent minimizer (numerics/minimize.jl): the azimuthal minimum
using ..VisualGeometryToolkitCore: bspline_basis, clamped_knots   # the Cox–de Boor basis on clamped knots: the vignetting fields (vignetting.jl)
using ..VisualGeometryToolkitCore: TagRegistry, tag_of   # the target calibration kinds' tags (quality/tag_registry.jl)
using ..VisualGeometryToolkitCore: Circle, Ellipse, Circle3D, _ellipse_or_nothing, boundary_points
using ..VisualGeometryToolkitCore: _Unchecked   # the unchecked inner constructor's sentinel (CameraCalibrationMat, base/types.jl)
import ..VisualGeometryToolkitCore: _padding_value   # FixedModels' padding of a type with no valid zero (CameraCalibrationMat)
using ..VisualGeometryToolkitCore: intersects   # a node's support against the imaged region (support.jl)
using ..VisualGeometryToolkitCore: HomEllipseMat, WholePlane   # the conic matrix a valid domain is built from, and the whole plane (image_circle.jl)
import ..VisualGeometryToolkitCore: HomographyMap  # Import (not using) to extend the labeled camera lift
import ..VisualGeometryToolkitCore: homography   # its TargetHomographies method, as pose(tp, t, id) (captures.jl)

using ..VisualGeometryToolkitCore: Plane3D, signed_distance  # For chirality-filtered P3P
using ..VisualGeometryToolkitCore: Line2D   # RadialCamera projects a plane point TO A LINE
using ..VisualGeometryToolkitCore: AbstractRadialModel   # RadialCamera joins the plane→pencil family
using ..VisualGeometryToolkitCore: RadialHomography      # the family's other member (conversion in calibration.jl)
import ..VisualGeometryToolkitCore: center               # ONE `center` generic (base/types.jl)
using ..VisualGeometryToolkitCore: ScaledIdentity, _FormStorage, _frame   # λ·I, the three storages a statistical form holds and its frame (base/types.jl): a blur's covariance (blur.jl)
import ..VisualGeometryToolkitCore: direction            # the radial family's algebraic accessor (public, not exported)
import ..VisualGeometryToolkitCore: Rect  # Import (not using) to extend the constructor
import ..VisualGeometryToolkitCore: project  # Import to extend (shared with Primitives)
import ..VisualGeometryToolkitCore: pose, extrinsics, backproject, unproject, focal_length  # Camera interface functions
import ..VisualGeometryToolkitCore: RadialHomography  # division-model camera bridge (CameraModel(::RadialHomography))
import ..VisualGeometryToolkitCore: transfer  # Multi-view verb (extended on StereoRig in stereo.jl)
# Camera component accessors. `model` MUST extend the parent generic (which
# base/manifold.jl also extends for Manifold): defining a fresh `Cameras.model`
# would not merge into the already-bound `VisualGeometryToolkitCore.model` at
# `using .Cameras`, so the top-level export would silently lack the PosedCamera
# method (latent since 0.8; exposed by the 0.10 imaging-chain tests).
import ..VisualGeometryToolkitCore: intrinsics, lens, model
import ..VisualGeometryToolkitCore: Manifold, SubsetManifold, retract, recenter, retract_jacobian, action_jacobian,
    local_coordinates, minimal_manifold, restrict, tangent_size   # the lens manifold (lens_chart.jl) extends the contract
using ..VisualGeometryToolkitCore: SphereManifold   # RadialCamera's minimal manifold (calibration.jl)
using ..VisualGeometryToolkitCore: _lift            # a restriction's coordinates lifted (lens_chart.jl's boxed recenter)
import ..VisualGeometryToolkitCore: _primal         # every Dual layer stripped (base/duals.jl); lens_rays.jl adds the lens method
using ..VisualGeometryToolkitCore: _is_dual_type, primal_float   # (base/duals.jl)
import ..VisualGeometryToolkitCore: constraint_dim, _measurement_dim, ambient_coordinates   # … and its place in the covariance machinery
import ..VisualGeometryToolkitCore: _precision, _has_division  # image_warp chain hooks: lens precision, a lens consumes a ray
import ..VisualGeometryToolkitCore: toml_value   # extended on lenses (lens_chart.jl): a lens writes as its record
using ..VisualGeometryToolkitCore: pointwise!, OnBackend   # a lens over a batch of points (lens_rays.jl)
using ..VisualGeometryToolkitCore: default_device, on_device   # the family's one backend resolver and upload: the warps' and batched kernels' `device` (base/device.jl)
import ..VisualGeometryToolkitCore: aspect_ratio  # Extend interface stub (shared with Primitives)
import ..VisualGeometryToolkitCore: surface, intersection, chart, normal, surface_point, bounds   # a posed target's primitive (target_surfaces.jl); the MappedSurface protocol (mapped_surface.jl)
using ..VisualGeometryToolkitCore: _plane_hit, _plane_coordinates, _masked, _primal, primal_float, _with_value, _lift_dual, _outer_value, _map_numbers, _frame_quad   # MappedSurface's root (mapped_surface.jl)
using ..VisualGeometryToolkitCore: Quad, Rect2, rotation, _in_rect   # _in_rect: the surface containment test, the board check's too
using ..VisualGeometryToolkitCore: _check_length   # a length, or a DimensionError (euclidean_map.jl)
using ..VisualGeometryToolkitCore: Attributed, csponds, Csponds, withattrs  # composition primitives (depth-attributed points, correspondence sets)
using ..VisualGeometryToolkitCore: sources, targets, centroid  # correspondence access protocol (#21)
using ..VisualGeometryToolkitCore: AbstractCamera  # Abstract camera type from interface
import ..VisualGeometryToolkitCore: is_posed  # the pose trait, answered per camera type
using ..VisualGeometryToolkitCore: sampson_distance, skew, FundMat  # GT-pose Sampson filter in validate_gt_metric; skew in the pose Jacobian
using ..VisualGeometryToolkitCore: HasBounds, NoBounds  # Bounds trait
import ..VisualGeometryToolkitCore: bounds_trait, extent           # extend the trait + accessor
using Base: ImmutableDict

# Import types and macros defined in core and geometry
using ..VisualGeometryToolkitCore: Size2, Quantity, BaseQuantity, base_quantity, strip_or_bare, px, mm, μm, m
using ..VisualGeometryToolkitCore: _check_pixels, _strip_pixels   # pixel-sized values: refuse a length (base/units.jl)
using ..VisualGeometryToolkitCore: SimilarityMap   # similarity_scale of a similarity (imaging_chain.jl)
using ..VisualGeometryToolkitCore: EuclideanMap, CameraPose, CameraExtrinsics, to_affine, to_homogeneous, image_warp, image_warp!
using ..VisualGeometryToolkitCore: RotationMap, WorldFrame, CameraFrame, _mapped_origin   # an oriented camera's pose (transforms/rotation_map.jl)
import ..VisualGeometryToolkitCore: has_units, with_unit   # extended on the carriers: targets, mounts, rigs, cameras, captures
import ..VisualGeometryToolkitCore: _unit_form, _form_sample   # the one-form rule's view of each carrier (euclidean_map.jl `_check_one_form`)
using ..VisualGeometryToolkitCore: translation, UnitfulEuclideanMap, _attach_unit, _ustrip_point, _attach_point, _stored_point,
    _check_one_form, strip_metres, _translation_metres,   # the one strip/attach of maps and points, the one-form rule (euclidean_map.jl)
    _strip_length, _attach_length, _metre_factor, _scale_length_form   # one length's strip/attach, a unit's metre factor (a calibration's lengths)
using ..VisualGeometryToolkitCore: ThinPlateSplineSystem, tps_coefficients, tps_pure_bending, tps_evaluate, tps_gradient, nodal_weights, _pure_bending_weights, bending_factor   # the thin-plate calibration kinds (target_calibration.jl)
using FlexUnits: Units, DimensionError
using ..VisualGeometryToolkitCore: TargetPlaneFrame, PixelFrame, UnlabeledFrame, relabel, unlabel, map_frames, Frame
using ..VisualGeometryToolkitCore: _field_tuple_expr   # the generated field-tuple body (transforms/pointwise.jl)
using ..VisualGeometryToolkitCore: FramedImage   # rect-form undistort return currency (warp layer)
using ..VisualGeometryToolkitCore: HomographyMap   # rectify.jl — the per-view rect→view collineation
using ..VisualGeometryToolkitCore: ViewFrame, ImageFrame, TargetFrame  # per-view identity tags (minted by StereoRig)
using ..VisualGeometryToolkitCore: NormalizedImageFrame, DiagonalNormalization  # the normalization map's factory (imaging_chain.jl)
using ..VisualGeometryToolkitCore: GnomonicFrame  # per-view CALIBRATED frame (essential_form slots)
using ..VisualGeometryToolkitCore: s2_tangent_basis, s2_tangent_basis_matrix  # T_f S² basis: shared with EssentialManifold5 (backproject_with_jacobian)
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
import ..VisualGeometryToolkitCore: HomographyMat  # Import to extend constructor
using ..VisualGeometryToolkitCore: sign_normalize   # a homography's canonical parameters (captures.jl)
import ..VisualGeometryToolkitCore: Extrinsics3   # Import to extend constructor: the planar-target pose from a homography (homography.jl)
using ..VisualGeometryToolkitCore: _cross2d   # the 2-D cross product (base/coordinates.jl): radial_camera_5pt
using ..VisualGeometryToolkitCore: _AffinePlanarMap   # the affine-class maps RadialCamera's ∘ accepts (calibration.jl)
using ..VisualGeometryToolkitCore: rescale_radial_coeff   # the one radial-coefficient scaling law (radial_models.jl; _rescale_lens)
using ..VisualGeometryToolkitCore: check_weights, weight_of, value_of   # a homography's weight and value (zhang_intrinsics)
import ..VisualGeometryToolkitCore: propagate   # extended for FramedAffine (imaging_chain.jl)
using ..VisualGeometryToolkitCore: CovMat, PrecMat, Uncertain, _junction_ok, _check_junction   # the checked statistical forms and the one frame-junction rule (base/frames.jl)
import ..VisualGeometryToolkitCore: constraint_with_jacobians   # extended for the radial camera and the division lift (calibration.jl), the full camera (camera_models.jl)
import ..VisualGeometryToolkitCore: measurement_gradient_jacobian   # extended for the radial camera (calibration.jl): its Sampson row
import ..VisualGeometryToolkitCore: transfer_ellipse          # extended for the lens camera (ellipse-preserving image of a plane conic)
import ..VisualGeometryToolkitCore: PerspectiveMap  # From CoordinateTransformations (imported to extend)
import ..VisualGeometryToolkitCore: p3p  # P3P solver from solvers.jl (imported to extend with correspondence wrapper)
import ..VisualGeometryToolkitCore: solve, sample_size, max_solutions   # the minimal-solver protocol, extended by RadialCamera5pt (calibration.jl)
using ..VisualGeometryToolkitCore: MinimalSolver, FixedModels


# Include sensor specifications
include("sensors.jl")
include("boards.jl")           # Board: the board a planar target's pattern is printed on (the sensor's analogue)

# Include modular camera components
include("intrinsics.jl")
include("projective_intrinsics.jl")  # ProjectiveIntrinsics: the principal point alone, no focal length
include("lenses.jl")         # the lens families: Lens/LiftLens, projections, terms, combinators
include("projections.jl")
include("camera_models.jl")
include("lens_rays.jl")       # THE lens contract: project/backproject on rays + the derived Jacobians
include("equirectangular.jl") # the latitude/longitude projection of directions (panoramas, sky maps)
include("lens_chart.jl") # parameter chart, ∂lens/∂θ, TOML lens registry (0.22.1)
include("lens_spec.jl")  # LensSpec: a lens family by name and term counts
include("chart_spline.jl")    # ChartSpline: a uniform cubic B-spline on a rectangle of a 2-D chart; the lattice primitives (weights, window)
include("spline_lens.jl")     # SplineLens: the central grid camera as a CameraModel lens; the spline lattice's shared helpers
include("noncentral_spline_model.jl")  # NoncentralSplineModel: the non-central grid on ChartSpline (the SplineLens's directions plus a point spline)
include("support.jl")         # supports on the sensor: which parameters a camera can see (the spline models' lattice)
include("vignetting.jl")      # TensorProductVignetting, RadialVignetting: a camera's vignetting (uses the support protocol, bspline_basis)
include("lighting.jl")        # EnvironmentLight, SphericalHarmonicMap, sh_basis: the scene's light (extends vignetting.jl's coefficient_jacobian)
include("blur.jl")            # GaussianBlur: one blur of an imaging model, the density N(0, Σ) by its covariance (extends Statistics' cov/var/std, Core's propagate)
include("imaging_chain.jl")   # framed stages between image frames + imaging_chain mint (0.10)

# ==============================================================================
# PosedCamera - Complete camera in 3D space (model + pose)
# ==============================================================================
# NOTE: AbstractCamera is defined in base/interface.jl to break circular
# dependencies. This allows other modules to work with cameras without
# depending on the full Cameras submodule.
# ==============================================================================

"""
    PosedCamera{M, Rt} <: AbstractCamera

A camera in 3D space, consisting of a camera model (intrinsics + lens) and extrinsics (pose).

Dispatch on the lens through the model parameter:
`PosedCamera{<:CameraModel{<:Any, <:MyLens}}` (the [`PosedPinholeCamera`](@ref) alias
is that form for `Pinhole`).

Use `project(camera, X)` to map a 3D world point to 2D pixel coordinates,
and `backproject(camera, u)` / `unproject(camera, u, depth)` for the
inverse directions. The transform is computed on demand and inlined.

# Type parameters
- `M <: AbstractCamera`: an UNPOSED camera model ([`CameraModel`](@ref), or a
  grid model); `PosedCamera(model, X)` refuses a posed one (`is_posed(model)`)
  and a [`BoundedCamera`](@ref) — the bound goes outside the pose,
  `BoundedCamera(PosedCamera(model, X), extent)`
- `Rt <: Union{EuclideanMap, RotationMap{3}}`: extrinsics type. A
  [`RotationMap`](@ref) poses a camera that only turns about its centre, an
  [`OrientedCamera`](@ref); a camera turns in three dimensions, so a
  `RotationMap{2}` is a `MethodError` at construction.

# Fields
- `model::M`: Camera model (intrinsics + lens)
- `extrinsics::Rt`: Camera extrinsics (world-to-camera transform)

# Examples
```julia
# Create camera with logical intrinsics
model = CameraModel(800.0px, [320.0px, 240.0px])
extrinsics = EuclideanMap(RotMatrix(I), [0.0, 0.0, 0.0])
camera = PosedCamera(model, extrinsics)

# Dispatch on lens (via the model parameter; PosedPinholeCamera is the
# ready-made alias for the Pinhole case)
foo(::PosedPinholeCamera) = "pinhole path"
foo(::PosedCamera{<:CameraModel{<:AbstractIntrinsics, <:KannalaBrandt}}) = "fisheye path"

# Project a world point
X_world = SVector(1.0mm, 2.0mm, 10.0mm)
u = project(camera, X_world)
```
"""
struct PosedCamera{
        M <: AbstractCamera,   # an unposed model — parametric OR generic (checked below)
        Rt <: Union{EuclideanMap, RotationMap{3}},
    } <: AbstractCamera
    model::M
    extrinsics::Rt
    # The one hierarchy admits a camera as the model type; the guard refuses the
    # two forms that are not a model. It is decided by the model's type
    # (`is_posed` is a type-level answer), so it compiles away.
    function PosedCamera{M, Rt}(model, extrinsics) where {M <: AbstractCamera, Rt <: Union{EuclideanMap, RotationMap{3}}}
        _check_camera_model(model, "PosedCamera")
        return new{M, Rt}(model, extrinsics)
    end
end
PosedCamera(model::M, extrinsics::Rt) where {M <: AbstractCamera, Rt <: Union{EuclideanMap, RotationMap{3}}} =
    PosedCamera{M, Rt}(model, extrinsics)

is_posed(::PosedCamera) = true

"""
    OrientedCamera{M}

A camera that only turns about its centre: a [`PosedCamera`](@ref) whose
extrinsics are a [`RotationMap`](@ref)`{3}`, `PosedCamera{M, <:RotationMap{3}}`
(an alias, not a type of its own). It has an orientation and no position:
`extrinsics(c)` is the world → camera rotation, `pose(c)` its inverse, and
`project`, `backproject` and `unproject` act about the origin. Build it from
the rotation of either direction:
`PosedCamera(model, relabel(CameraFrame, WorldFrame, RotationMap(R)))` takes
a camera → world orientation and inverts it, an unlabeled or world → camera
`RotationMap` is stored as the extrinsics. What needs a position refuses one
by name: a rig member, the plane-induced homography and the verbs on plane
coordinates (`project` of a plane conic or rectangle, `transfer_ellipse`,
`HomographyMat`, `RadialCamera`), and a non-central grid model's pose. The
frames of a panorama are oriented cameras, as [`view_overlap`](@ref) takes
them.
"""
const OrientedCamera{M <: AbstractCamera} = PosedCamera{M, <:RotationMap{3}}   # M bounded as PosedCamera's, so a method on it is more specific than one on PosedCamera

# A camera → world orientation (labeled so) is stored as its inverse, as a `CameraPose` is.
PosedCamera(model::AbstractCamera, p::RotationMap{3, T, R, CameraFrame, WorldFrame}) where {T, R} = PosedCamera(model, inv(p))

"""
    PosedPinholeCamera

Cameras whose lens is exactly `Pinhole` — the alias for lens dispatch now
that `PosedCamera` carries only `{M, Rt}` (the former `{I, L, Rt, M}` repeated
what `M` already determines).  Consumed by the induced-homography
constructors in geometry/cameras/homography.jl.
"""
const PosedPinholeCamera = PosedCamera{<:CameraModel{<:AbstractIntrinsics, Pinhole}}

# THE camera shape projection (a z = 0 plane circle/ellipse, a 3D circle): a
# pinhole images the conic exactly through the plane homography — an `Ellipse`
# when it stays one; otherwise (across the line at infinity), and for every
# distorting lens, `n` boundary samples projected into a `Polygon`.
function _camera_project_conic(camera::PosedCamera, shape, n::Int = 64)
    Q = _plane_conic(camera, shape, lens(camera))
    e = Q === nothing ? nothing : _ellipse_or_nothing(Q)
    e === nothing || return e
    return _camera_project_world_points(camera, _world_boundary(shape, n))
end
# The exact image conic through the plane-induced homography — pinhole only.
_plane_conic(camera::PosedCamera, shape::Union{Circle, Ellipse}, ::Pinhole) = HomographyMap(HomographyMat(camera))(shape)
_plane_conic(camera::PosedCamera, c::Circle3D, ::Pinhole) =
    HomographyMap(HomographyMat(camera, c.center, c.u, c.v))(Circle(Point2(0.0, 0.0), c.radius))   # the circle's local (u, v) frame
_plane_conic(::PosedCamera, _, ::AbstractLensModel) = nothing
# The world boundary samples: a plane shape on z = 0, a 3D circle as is.
_world_boundary(shape::Union{Circle, Ellipse}, n::Int) = (Point3{Float64}(p[1], p[2], 0.0) for p in boundary_points(shape, n))
_world_boundary(c::Circle3D, n::Int) = boundary_points(c, n)

"""
    transfer_ellipse(camera::PosedCamera, shape::Union{Circle, Ellipse}) -> Ellipse | nothing

The ellipse-preserving image of a `z = 0` plane circle/ellipse under
`camera`: the exact conic through the plane-induced homography of the
pinhole twin, then — for a distorting lens — the lens's local affinity at
the transferred centre — `distortion_map`'s stages applied to the transferred conic. `project(camera, shape)` samples the
boundary into a polygon instead; this is the form the guided association
of blob boards consumes (an ellipse to overlap, a centre to search at) and
it stays an ellipse for every lens. `nothing` behind the camera, across the
line at infinity, or where the lens map is not finite at the centre.
"""
function transfer_ellipse(camera::PosedCamera, shape::Union{Circle, Ellipse})
    m = camera.model
    # cheirality at the centre: a plane conic behind the camera has no image
    # (a homography maps it to a perfectly good conic, and a lens's forward
    # map is finite on both sides of z = 0 — neither would report it)
    c = centroid(shape)
    Xc = camera.extrinsics.R * SVector{3, Float64}(c[1], c[2], 0.0) + camera.extrinsics.t
    Xc[3] > 0 || return nothing
    H = HomographyMap(PosedCamera(pinhole(m), camera.extrinsics))
    return _lens_ellipse(m.lens, m, transfer_ellipse(H, shape))
end

# The lens's part of the transfer: `distortion_map(m)` is a `ComposedTransformation`
# (`affine_part(K) ∘ lens ∘ affine_part(K⁻¹)`), which `transfer_ellipse` walks stage by stage.
_lens_ellipse(lens, m, ::Nothing) = nothing
_lens_ellipse(::Pinhole, m, ep::Ellipse) = ep
_lens_ellipse(lens, m, ep::Ellipse) = transfer_ellipse(distortion_map(m), ep)

# Rect: tessellate edges, lift to z=0, project → Polygon
_camera_project_rect(camera::PosedCamera, rect::GeometryBasics.HyperRectangle{2}, n::Int) =
    _camera_project_points(camera, _rect_edge_samples(identity, rect, n))

# Shared: lift 2D → 3D at z=0, project, return Polygon
_camera_project_points(camera::PosedCamera, pts) =
    _camera_project_world_points(camera, (Point3{Float64}(p[1], p[2], 0.0) for p in pts))

# Shared: project 3D world points, return Polygon
function _camera_project_world_points(camera::PosedCamera, pts)
    return GeometryBasics.Polygon([Point2(project(camera, Point3{Float64}(p...))) for p in pts])
end

# For non-EuclideanMap extrinsics, convert first. The camera bound keeps a
# wrong first argument a MethodError instead of infinite recursion through
# EuclideanMap(::EuclideanMap).
PosedCamera(model::AbstractCamera, extrinsics) = PosedCamera(model, EuclideanMap(extrinsics))

"""
    PosedCamera(model::CameraModel, e::CameraExtrinsics) -> PosedCamera
    PosedCamera(model::CameraModel, p::CameraPose) -> PosedCamera

Construct a `PosedCamera` from a frame-labeled map, so the caller states which
frame it holds instead of relying on the positional convention.

The bare positional constructor `PosedCamera(model, extrinsics::EuclideanMap)`
with an UNLABELED map takes the world-to-camera map (documented convention,
unchecked); a caller holding a camera-to-world map (a `lookat`-style pose, a
renderer's `sensor.to_world`) who supplies it positionally stores the WRONG
map without any error — every accessor then silently answers for the
inverse camera. With frame labels that mistake is unrepresentable:
`CameraExtrinsics` (being the extrinsics-labeled `EuclideanMap`) stores
directly into the extrinsics slot via the default constructor, `CameraPose`
inverts internally, and `pose(PosedCamera(model, P::CameraPose)) ≈ P`.
"""
PosedCamera(model::AbstractCamera, p::CameraPose) = PosedCamera(model, inv(p))

"""
    PosedCamera(K::StaticMatrix{3,3}, R::StaticMatrix{3,3}, t::StaticVector{3};
           check=true, project=false) -> PosedCamera

Boundary constructor taking raw matrix data (typical for SfM dataset loaders).
`R` is wrapped via the `EuclideanMap(::SMatrix, ...)` path which validates
(and optionally snaps via `project=true`) before promoting to `RotMatrix`.
`t` is plain numbers of `K`'s element type, or lengths (`t .* m`), which give
a physical camera.
"""
function PosedCamera(
        K::StaticMatrix{3, 3, T},
        R::StaticMatrix{3, 3, T},
        t::Union{StaticVector{3, T}, StaticVector{3, <:Quantity}};
        check::Bool = true,
        project::Bool = false
    ) where {T}
    K_calib = CameraCalibrationMat(K; check = check)
    model = CameraModel(K_calib)
    return PosedCamera(model, EuclideanMap(SMatrix{3, 3, T, 9}(R), t; check = check, project = project))
end

# ============================================================================
# Camera Type Aliases
# ============================================================================

# NOTE (2026-07, #15): the PhysicalCamera/LogicalCamera aliases are GONE —
# PhysicalIntrinsics was demoted (every intrinsics is a K matrix), so the
# split no longer denotes anything.

# ============================================================================
# BoundedCamera{C, B, R} — wrap any AbstractCamera with its bound (extent or Sensor) and its lens coverage
# ============================================================================

"""
    BoundedCamera{C<:AbstractCamera, B, R} <: AbstractCamera

A wrapper around any `AbstractCamera` that adds pixel-space image bounds
(width × height) without disturbing the underlying camera's intrinsics or
units.  Used by SfM/MVS dataset readers (ETH3D, ScanNet, COLMAP exports)
where the camera is described by its calibration plus an explicit image
extent — information the bare `PosedCamera` type doesn't retain.

The wrapped camera may be UNPOSED — a bare model, `is_posed(bc) == false`
— which is COLMAP's "camera" row (intrinsics, lens and image size, no pose) and
the element of a stored camera collection (VisualGeometryToolkitIO's `cameras`
kind). `BoundedCamera(PosedCamera(model(bc), pose), bc.bound; lens_coverage = lens_coverage(bc))` is its posed
counterpart; the pose accessors of an unposed one are an `ArgumentError`.
[`is_posed`](@ref) is the wrapped camera's.

# The bound
`BoundedCamera(camera, extent::Size2)` bounds by a pixel extent;
`BoundedCamera(camera, sensor::Sensor)` by the physical sensor, whose
resolution is the extent and whose pitch (a length per pixel) carries a
lens datasheet's millimetres into pixels ([`image_circle`](@ref)). Either way
[`extent`](@ref)`(bc)` is the `Size2{Int}` pixel extent; [`sensor`](@ref)`(bc)`
is the `Sensor`, or `nothing` for the extent form. The bound is ONE field,
`bc.bound`: the extent, or the sensor whose resolution is the extent.

# The lens coverage
`BoundedCamera(camera, bound; lens_coverage = region)` — where light reaches the sensor, in
pixels: a `Circle` or an `Ellipse` (the lens's image circle, measured with a
detector or from a datasheet), or a `Rect2`; any real element type, converted
to `Float64`, and kept as the kind given. Omitted, it is the sensor's own `Rect`: the whole
sensor is lit. [`lens_coverage`](@ref)`(bc)` reads it, [`with_lens_coverage`](@ref)
replaces it. What the camera images is [`imaged_region`](@ref)`(bc)`: the
sensor ∩ the lens coverage ∩ where the lens model is valid.

# Field access
- `bc.camera`, `bc.bound`, `bc.lens_coverage` — direct fields; read them through
  `model`, `extent`/`sensor` and `lens_coverage`.

# Method delegation
A BoundedCamera is THE type for pixel operations: every camera verb a
`CameraModel` answers is forwarded to the wrapped `camera` (`MacroTools.@forward`,
and `@inline` `project`/`backproject` for device kernels) — the maps and their Jacobians (`project`,
`backproject`, `backproject_with_jacobian`, `unproject`, `is_in_front`,
`constraint_with_jacobians`), the accessors (`intrinsics`, `lens`, `pose`,
`extrinsics`, `focal_length`, `model`, `calibration_matrix`), the imaging and
its domain (`imaging_transform`, `imaging_chain`, `distortion_map`,
`valid_domain`, `valid_rect`, `undistort`, …) and the camera-derived solvers
(`p3p`, `transfer`, `radial_homography`); so `backproject_with_jacobian(bc, u)
== backproject_with_jacobian(bc.camera, u)`. The bound applies to none of them
(the extent is `image_circle`, `imaged_region` and `lens_coverage`). The
model constructors (`pinhole`, `window`, `with_parameters`) and the parameter
protocol are the model's: call them on `model(bc)`, and rebound with
[`with_model`](@ref).

# Trait
`bounds_trait(::BoundedCamera) = HasBounds()` — methods that need image
extent dispatch on `BoundsTrait`: `extent(bc)` is the bound's pixel extent.

# A device kernel's camera
The bound's kind and the region's are the type parameters `B` (`Size2{Int}`,
or the stored `Sensor` type) and `R` (`Rect2{Float64}`, `Circle{Float64}` or
`Ellipse{Float64}`), so a bounded camera is concrete, and `isbits` whenever the
wrapped camera is (a `CameraModel` of an isbits lens): it travels into a GPU
kernel as it is, its `backproject`/`project` the wrapped camera's.
[`convert_eltype`](@ref)`(Float32, bc)` converts the wrapped camera and keeps
the bound and the region.

# Example
```julia
inner = PosedCamera(CameraModel(K; lens=Pinhole()), pose)
bc    = BoundedCamera(inner, Size2(6048, 4032))
intrinsics(bc)     # forwarded → CameraCalibrationMat
extent(bc)         # Size2(6048, 4032)
lens_coverage(bc)     # Rect2(0, 0, 6048, 4032): the whole sensor is lit
```
"""
struct BoundedCamera{C <: AbstractCamera, B <: Union{Size2{Int}, Sensor{BaseQuantity, BaseQuantity}},
                     R <: Union{Rect2{Float64}, Circle{Float64}, Ellipse{Float64}}} <: AbstractCamera
    camera::C
    bound::B                   # the pixel extent, or the Sensor's stored form (sensors.jl) whose resolution IS the extent
    lens_coverage::R           # where the lens covers the sensor: where light reaches it (pixels); the sensor's own Rect when the whole of it is lit
end
# Both are type parameters, so a bounded camera is concrete and isbits with its camera, whichever bound and region it holds.

# The bound: a pixel extent, or the physical sensor whose resolution is that extent (its stored form).
_bound(e::Size2) = convert(Size2{Int}, e)
_bound(s::Sensor) = Sensor(s.resolution, s.pitch)
_extent(e::Size2{Int}) = e
_extent(s::Sensor) = (r = Rect(s); Size2(Int(r.widths[1]), Int(r.widths[2])))
_sensor(::Size2{Int}) = nothing
_sensor(s::Sensor) = s

# The lens coverage: a geometric primitive in Float64 pixels, kept as the kind given (a Circle stays a Circle: `∩` meets it as
# its round Ellipse); `nothing` is the whole sensor.
_lens_coverage(::Nothing, e::Size2{Int}) = Rect2{Float64}(0, 0, e.width, e.height)
_lens_coverage(r::Rect2, _) = _covering(Rect2{Float64}(r), (r.origin..., r.widths...), r.widths)
_lens_coverage(c::Circle, _) = _covering(Circle{Float64}(Point2{Float64}(c.center), Float64(c.r)), (c.center..., c.r), (c.r,))
_lens_coverage(e::Ellipse, _) =
    _covering(Ellipse{Float64}(Point2{Float64}(e.center), Float64(e.a), Float64(e.b), Float64(e.θ)), (e.center..., e.a, e.b, e.θ), (e.a, e.b))
# A coverage is a real region: every number finite, every extent (a Rect's widths, a radius, the semi-axes) positive —
# a degenerate one would be written by a store whose reader refuses it.
_covering(x, numbers, extents) =
    all(isfinite, numbers) && all(>(0), extents) ? x :
    throw(ArgumentError("a camera's lens coverage is a region with finite numbers and a positive size; got $(x)"))
_lens_coverage(r, _) = throw(ArgumentError("a camera's lens coverage is a Rect2, a Circle or an Ellipse (pixels), got a $(nameof(typeof(r)))"))

BoundedCamera(camera::AbstractCamera, bound::Union{Size2, Sensor}; lens_coverage = nothing) =
    (b = _bound(bound); BoundedCamera(camera, b, _lens_coverage(lens_coverage, _extent(b))))

"""
    sensor(camera::BoundedCamera) -> Union{Sensor, Nothing}

The physical [`Sensor`](@ref) a camera is bounded by — its resolution is the
camera's [`extent`](@ref), its pitch the length per pixel — or `nothing` for a
camera bounded by a bare pixel extent. The one-argument accessor beside the
two-argument database lookup `sensor(manufacturer, model)`: both answer a
`Sensor`.
"""
sensor(bc::BoundedCamera) = _sensor(bc.bound)

"""
    lens_coverage(camera::BoundedCamera) -> Union{Rect2{Float64}, Circle{Float64}, Ellipse{Float64}}

Where light reaches the camera's sensor, in its pixels: the lens's image circle
(a `Circle` or an `Ellipse`, measured or from a datasheet), or the sensor's own
`Rect` when the whole sensor is lit — the kind it was given, in `Float64`. Not
where the lens MODEL is valid — that is
[`valid_domain`](@ref); what the camera images is both, inside the sensor:
[`imaged_region`](@ref)`(camera)`.
"""
lens_coverage(bc::BoundedCamera) = bc.lens_coverage

"""
    lens_coverage(camera::BoundedCamera, rim_margin::Real) -> Union{Rect2{Float64}, Circle{Float64}, Ellipse{Float64}, Nothing}

The camera's [`lens_coverage`](@ref) less `rim_margin` pixels at the LENS's
edge (an image circle's soft rim), or `nothing` when nothing is left: a
`Circle` of radius `r − rim_margin`, an `Ellipse` with both semi-axes less
`rim_margin`, a `Rect` inset by `rim_margin` on each side that lies inside the
sensor — a side on or past the sensor's edge is the sensor's, not the lens's,
and stays, so a camera whose whole sensor is lit loses nothing. The kind it was
given; a margin of `0` returns the coverage. Where a consumer counts a pixel as
lit away from the rim: `imaged_region(with_lens_coverage(camera, c))` is what
such a camera images (with [`with_lens_coverage`](@ref)), and a `Circle`'s
`c.r` the radius it reaches from the circle's centre.
"""
function lens_coverage(bc::BoundedCamera, rim_margin::Real)
    e = extent(bc)
    return _less_rim(lens_coverage(bc), rim_margin, Rect2{Float64}(0, 0, e.width, e.height))
end
_less_rim(c::Circle, m, _) = c.r > m ? Circle(c.center, c.r - m) : nothing
_less_rim(e::Ellipse, m, _) = min(e.a, e.b) > m ? Ellipse(e.center, e.a - m, e.b - m, e.θ) : nothing
function _less_rim(r::Rect, m, sensor::Rect)
    lo, hi = r.origin, r.origin .+ r.widths; slo, shi = sensor.origin, sensor.origin .+ sensor.widths
    x0, y0 = ntuple(i -> lo[i] > slo[i] ? lo[i] + m : lo[i], 2)
    x1, y1 = ntuple(i -> hi[i] < shi[i] ? hi[i] - m : hi[i], 2)
    return x1 > x0 && y1 > y0 ? Rect2{Float64}(x0, y0, x1 - x0, y1 - y0) : nothing
end

"""
    with_lens_coverage(camera::BoundedCamera, region) -> BoundedCamera

The same camera and bound with the lens coverage `region` (a `Rect2`, `Circle` or
`Ellipse` of any real element type, converted to `Float64`; `nothing` for the
whole sensor).
"""
with_lens_coverage(bc::BoundedCamera, region) = BoundedCamera(bc.camera, bc.bound; lens_coverage = region)

# The same bound and lens coverage around another camera: what a rebuild of the wrapped camera
# (new extrinsics, new model, a unit change) keeps.
_rebound(bc::BoundedCamera, camera::AbstractCamera) = BoundedCamera(camera, bc.bound, bc.lens_coverage)
# a bounded camera in another element type: the wrapped camera converted, the bound kept (a device kernel's Float32 camera)
convert_eltype(::Type{T}, bc::BoundedCamera) where {T <: Real} = _rebound(bc, convert_eltype(T, bc.camera))
# The device form (`Adapt.adapt(backend, bc)`): the wrapped camera's arrays move (a spline lens's controls and mask), the
# bound and the lens coverage (isbits) travel as they are. A posed camera's pose stays, its model moves.
Adapt.adapt_structure(to, bc::BoundedCamera) = _rebound(bc, Adapt.adapt(to, bc.camera))
Adapt.adapt_structure(to, c::PosedCamera) = _device_posed(Adapt.adapt(to, c.model), c.extrinsics)
_device_posed(m::M, X::Rt) where {M, Rt} = PosedCamera{M, Rt}(m, X)   # the stored extrinsics as they are, no re-check

"""
    with_present(camera::BoundedCamera, present::AbstractMatrix{Bool}) -> BoundedCamera

A bounded spline camera (a [`SplineLens`](@ref) camera or a
[`NoncentralSplineModel`](@ref), posed or not) holding the nodes `present`: its
model's `with_present`, the bound, the lens coverage and a pose kept.
"""
with_present(bc::BoundedCamera, present::AbstractMatrix{Bool}) = with_model(bc, with_present(model(bc), present))

"""
    with_directions(camera::BoundedCamera, D::AbstractMatrix{<:StaticVector{3}}) -> BoundedCamera

A bounded spline camera (a [`SplineLens`](@ref) camera or a
[`NoncentralSplineModel`](@ref), posed or not) with the direction controls `D`:
its model's `with_directions`, the bound, the lens coverage and a pose kept.
"""
with_directions(bc::BoundedCamera, D::AbstractMatrix) = with_model(bc, with_directions(model(bc), D))

# `axis_gauged` (spline_lens.jl) of a posed camera — its model and its extrinsics turned together — and of a bounded
# one: its model's, the bound and the lens coverage kept (pixels do not move).
axis_gauged(c::PosedCamera, rotation::Rotation{3}, K′::AbstractMatrix) =
    PosedCamera(axis_gauged(c.model, rotation, K′), axis_gauged(c.extrinsics, rotation))
axis_gauged(bc::BoundedCamera, rotation::Rotation{3}, K′::AbstractMatrix) = with_model(bc, axis_gauged(model(bc), rotation, K′))

# Trait opt-in — instance form (matches VGC convention: see has_boundary(::Ellipse) etc.)
bounds_trait(::BoundedCamera) = HasBounds()
is_posed(b::BoundedCamera) = is_posed(b.camera)

# What `PosedCamera(model, X)` (and `with_model`) accepts as the model: an unposed
# camera that is not a bound. A posed one would be posed twice; a bounded model
# would put the bound inside the pose, a second spelling of
# `BoundedCamera(PosedCamera(model, X), extent)`.
@noinline _refuse_posed_model(what, c) = throw(
    ArgumentError(
        "$(what): the model is a posed camera (a $(nameof(typeof(c)))) — a camera is posed once; " *
            "supply its model: `$(what)(model(camera), …)`"
    )
)
@noinline _refuse_bounded_model(what) = throw(
    ArgumentError(
        "$(what): the model is a BoundedCamera — the bound goes outside the pose: " *
            "`BoundedCamera(PosedCamera(model(bc), X), bc.bound; lens_coverage = lens_coverage(bc))`"
    )
)
_check_camera_model(c::AbstractCamera, what) = is_posed(c) ? _refuse_posed_model(what, c) : nothing
_check_camera_model(::BoundedCamera, what) = _refuse_bounded_model(what)

# The posed-only verbs on an unposed camera or target: refused by name, with
# the way to pose it (`_pose_hint`, one method per kind; the target's is in
# targets.jl). A posed type answers them with its own methods, so on one that
# lacks them this is the MethodError it would have been.
@noinline _refuse_unposed(what, x) = throw(
    ArgumentError("$(what): the $(_pose_hint(x)[1]) is unposed (a $(nameof(typeof(x))), `is_posed` is false) — pose it: $(_pose_hint(x)[2])")
)
_pose_hint(::AbstractCamera) = ("camera", "`PosedCamera(model, X)`, or `BoundedCamera(PosedCamera(model(bc), X), bc.bound; lens_coverage = lens_coverage(bc))`")
_unposed_only(f, what, x, args...) = is_posed(x) ? throw(MethodError(f, (x, args...))) : _refuse_unposed(what, x)

# Method delegation to the wrapped camera (`bc.camera`): the `MacroTools.@forward` list at
# the end of this module (after every file that defines a verb it forwards).
# The two maps a device kernel calls are written out `@inline`, which
# `@forward` does not emit: with a Dual camera the inliner declines the
# forwarder and the kernel hands the wrapped camera over by pointer through
# local memory. The keywords stay for the batched generic-model `project`.
@inline project(bc::BoundedCamera, args...; kwargs...) = project(bc.camera, args...; kwargs...)
@inline backproject(bc::BoundedCamera, args...; kwargs...) = backproject(bc.camera, args...; kwargs...)

# ----------------------------------------------------------------------------
# Units — a physical camera is one whose extrinsics are (translation in metres)
# ----------------------------------------------------------------------------
#
# The model's pixel quantities are not lengths of the scene; only the pose
# moves. The camera is rebuilt with its type parameters stated, so a stripped
# map is stored as it is (the bare constructor would invert a `CameraPose`).

# A camera posed by a physical map (bounds stated, so dispatch ranks it above `PosedCamera`).
const _UnitfulPosedCamera = PosedCamera{<:AbstractCamera, <:UnitfulEuclideanMap{3}}

# The plane-coordinate verbs (a z = 0 conic or rectangle, the plane-induced
# homography) take plain plane coordinates with no unit of their own, so a
# physical camera cannot be applied to them without naming that unit: refused
# with an `ArgumentError` naming the fix — the same error the stack's other
# wrong-form refusals (`ustrip` of a plain value, `with_unit` of a physical one)
# raise, since the camera is not of a wrong dimension but in the wrong form.
@noinline _refuse_physical_camera(what) = throw(
    ArgumentError(
        "$(what): the camera is physical (its extrinsics in metres) but the plane coordinates are plain " *
            "numbers with no unit — strip the camera in the unit of the plane coordinates: `ustrip(mm, camera)`"
    )
)


has_units(c::PosedCamera) = has_units(c.extrinsics)
has_units(c::BoundedCamera) = has_units(c.camera)
# An unposed camera holds no length (a non-central grid model's points are plain
# metres by its type, noncentral_spline_model.jl); a posed camera type answers itself.
has_units(c::AbstractCamera) = is_posed(c) ? throw(MethodError(has_units, (c,))) : false
_rebuilt(c::PosedCamera, X::Union{EuclideanMap, RotationMap{3}}) = PosedCamera{typeof(c.model), typeof(X)}(c.model, X)

"""
    ustrip(u::Units, camera::PosedCamera | BoundedCamera) -> same type of camera

A physical camera's plain form: its extrinsics stripped in the length unit
`u` (`ustrip(u, E)`), the model and any extent unchanged. The inverse is
[`with_unit`](@ref)`(camera, u)`. An `ArgumentError` on a camera already in
the produced form. An unposed camera (a model, or a `BoundedCamera` of one)
holds no length, so both verbs return it unchanged.
"""
FlexUnits.ustrip(u::Units, c::PosedCamera) = _rebuilt(c, ustrip(u, c.extrinsics))
FlexUnits.ustrip(u::Units, c::BoundedCamera) = _rebound(c, ustrip(u, c.camera))
with_unit(c::PosedCamera, u::Units) = _rebuilt(c, _attach_unit(c.extrinsics, u))
# an unposed camera holds no length: nothing to strip or attach
FlexUnits.ustrip(u::Units, c::AbstractCamera) = is_posed(c) ? throw(MethodError(ustrip, (u, c))) : c
with_unit(c::AbstractCamera, u::Units) = is_posed(c) ? throw(MethodError(with_unit, (c, u))) : c

# A non-central grid model's point spline is plain metres — the one length of
# the stack its type cannot carry (noncentral_spline_model.jl). A plain extrinsics is
# "in the caller's unit", so a camera of the two would compare the grid's
# metres with scene points in whatever unit the caller meant (mm: offsets
# 1000× too small, a wrong pixel and no error). The ambiguous pair is refused
# where it is built, naming the fix; the plain-metres camera is reached only by
# stating the unit, `ustrip(m, camera)` of a physical one (and `with_unit(·, m)`
# back), so a strip or attach in another unit is refused too.
@noinline _refuse_plain_noncentral(what) = throw(
    ArgumentError(
        "$(what): a non-central grid model's point grid is in metres but the extrinsics are plain numbers " *
            "with no unit — attach the unit to the extrinsics: `PosedCamera(model, EuclideanMap(R, t .* m))`, or " *
            "`PosedCamera(model, with_unit(extrinsics, mm))` for a map in mm"
    )
)
PosedCamera(model::NoncentralSplineModel, X::EuclideanMap{3}) =
    has_units(X) ? PosedCamera{typeof(model), typeof(X)}(model, X) : _refuse_plain_noncentral("PosedCamera")
PosedCamera(model::NoncentralSplineModel, p::CameraPose) = PosedCamera(model, inv(p))
@noinline _refuse_noncentral_unit(what, u) = throw(
    ArgumentError(
        "$(what)($(u), camera): a non-central grid model's point grid is in metres, so its camera's plain form " *
            "is in metres too — use `m`, and the same unit for the scene points"
    )
)
FlexUnits.ustrip(u::Units, c::PosedCamera{<:NoncentralSplineModel}) =
    u == m ? _rebuilt(c, ustrip(u, c.extrinsics)) : _refuse_noncentral_unit("ustrip", u)
with_unit(c::PosedCamera{<:NoncentralSplineModel}, u::Units) =
    u == m ? _rebuilt(c, _attach_unit(c.extrinsics, u)) : _refuse_noncentral_unit("with_unit", u)
with_unit(c::BoundedCamera, u::Units) = _rebound(c, with_unit(c.camera, u))

"""
    extent(cam) -> Size2{Int}

Pixel-space image extent (width × height).  Trait-dispatched: errors at
compile time for cameras that don't carry bounds.
"""
extent(cam) = extent(bounds_trait(cam), cam)
extent(::HasBounds, bc::BoundedCamera) = _extent(bc.bound)
extent(::NoBounds, cam) =
    error("camera lacks image extent — wrap with BoundedCamera if you have it")

# Method-based accessors

"""
    pose(camera) -> CameraPose

Camera pose (camera-to-world transform), inverse of extrinsics. Returns a
frame-labeled map (`CameraFrame → WorldFrame`).

```julia
camera = PosedCamera(model, extrinsics)
P = pose(camera)       # Camera-to-world
P.t                    # Camera position in world coordinates
```
"""
pose(c::PosedCamera) = inv(extrinsics(c))
pose(c::AbstractCamera) = _unposed_only(pose, "pose", c)

"""
    extrinsics(camera) -> CameraExtrinsics

Camera extrinsics (world-to-camera transform). Returns a frame-labeled map
(`WorldFrame → CameraFrame`): the extrinsics slot is world-to-camera by
contract, so the accessor labels it even when the camera was constructed
from an unlabeled map (identity when already labeled).

```julia
camera = PosedCamera(model, ext)
E = extrinsics(camera)  # World-to-camera
E(world_point)          # Transform point to camera coordinates
```
"""
extrinsics(c::PosedCamera) = _labeled_extrinsics(c.extrinsics)
_labeled_extrinsics(X::EuclideanMap) = CameraExtrinsics(X)
_labeled_extrinsics(X::RotationMap) = relabel(WorldFrame, CameraFrame, X)   # an oriented camera's: a RotationMap, labeled the same way
extrinsics(c::AbstractCamera) = _unposed_only(extrinsics, "extrinsics", c)

"""
    is_in_front(camera, point::Point3) -> Bool

Chirality test: returns true if point is in front of the camera (z > 0 in camera coords).

Broadcasts over points:
```julia
mask = is_in_front.(Ref(camera), world_points)
```
"""
is_in_front(c::PosedCamera, p::Point3) = extrinsics(c)(p)[3] > 0
# a physical camera: the test in metres, the point a length (a plain one is a DimensionError)
is_in_front(c::_UnitfulPosedCamera, p::Point3) = is_in_front(ustrip(m, c), Point3(strip_metres(p)))

"""
    is_in_front(camera, x, y, z) -> AbstractArray{Bool}

Broadcasted component-array (SoA) form of the chirality test: world-frame
point components in, camera-frame `z > 0` mask out. Per-element semantics
match `is_in_front(camera, Point3(xᵢ, yᵢ, zᵢ))`; the component layout keeps
the whole test one fused broadcast on GPU arrays (hoisted from the
RayTracedCalibration GPU pipeline).
"""
function is_in_front(
        c::PosedCamera, x::AbstractArray, y::AbstractArray, z::AbstractArray
    )
    ext = extrinsics(c)
    R = ext.R
    t = _mapped_origin(ext)    # the translation; the origin for an oriented camera
    z_cam = R[3, 1] .* x .+ R[3, 2] .* y .+ R[3, 3] .* z .+ t[3]
    return z_cam .> zero(eltype(z_cam))
end
is_in_front(c::_UnitfulPosedCamera, x::AbstractArray, y::AbstractArray, z::AbstractArray) =
    is_in_front(ustrip(m, c), strip_metres(x), strip_metres(y), strip_metres(z))

"""
    focal_length(camera::PosedCamera)

Get focal length from camera's intrinsics: the `(fx, fy)` tuple in pixels.

# Examples
```julia
K = CameraCalibrationMat(800.0px, [320.0px, 240.0px])
model = CameraModel(LogicalIntrinsics(K), Pinhole())
camera = PosedCamera(model, extrinsics)
fx, fy = focal_length(camera)  # Returns (800.0, 800.0)
```
"""
focal_length(camera::PosedCamera) = focal_length(intrinsics(camera))

"""
    calibration_matrix(camera::PosedCamera) -> SMatrix{3,3,Float64,9}

Return the 3×3 calibration matrix K as a plain `SMatrix`. For a
[`SplineLens`](@ref) camera K is the lens's gauge — the pinhole at the pixel
imaging ẑ (skew 0) — kept as it is by every fit.

# Example
```julia
camera = PosedCamera(CameraModel(; focal=800.0, pp=(320.0, 240.0)), ext)
K = calibration_matrix(camera)  # [800 0 320; 0 800 240; 0 0 1]
```
"""
calibration_matrix(camera::PosedCamera) = calibration_matrix(camera.model)

"""
    model(camera::PosedCamera) -> CameraModel

Return the camera's underlying `CameraModel` (intrinsics + lens), the
single object the lower-level `project`/`backproject`/`transfer` kernels
take.  Method form so wrappers (e.g. `BoundedCamera`) can delegate via
`@forward`. A rig's other member kind answers the same accessor:
`model(::PosedTarget)` is its target.
"""
model(c::PosedCamera) = c.model
# an unposed camera (a model, the inside of a BoundedCamera of one) is its own model
model(c::AbstractCamera) = is_posed(c) ? throw(MethodError(model, (c,))) : c

"""
    intrinsics(camera::PosedCamera)
    intrinsics(model::CameraModel)

Return the camera's intrinsics block (e.g. `LogicalIntrinsics`,
`PhysicalIntrinsics`).  Equivalent to `camera.model.intrinsics` /
`model.intrinsics` but lets downstream code stop reaching into the field.
For a [`SplineLens`](@ref) camera they are the lens's gauge — the pinhole at the
pixel imaging ẑ (skew 0) — kept as they are by every fit.
"""
intrinsics(camera::PosedCamera) = intrinsics(camera.model)
intrinsics(model::CameraModel) = model.intrinsics

"""
    lens(camera::PosedCamera)
    lens(model::CameraModel)

Return the camera's lens model (e.g. `Pinhole`, a `Lens`, or any
user-defined `<: AbstractLensModel`).  Use this in dispatch when a downstream
package wants to specialize on lens type without reaching into
`camera.model.lens`.
"""
lens(camera::PosedCamera) = lens(camera.model)
lens(model::CameraModel) = model.lens

"""
    default_frustum_depth(camera::PosedCamera) -> Float64

Default frustum depth for visualization: fx from K, a natural depth scale
in the camera's working units.
"""
default_frustum_depth(camera::PosedCamera) = focal_length(camera)[1]

# ============================================================================
# Camera Projection (World → Pixel)
# ============================================================================

"""
    project(camera::PosedCamera, X_world::AbstractVector) -> Point2

Project 3D point from world coordinates to pixel coordinates.

# Arguments
- `camera::PosedCamera`: Camera with model and extrinsics
- `X_world::AbstractVector`: 3D point in world coordinates (with units)

# Returns
- Point2 with pixel coordinates (px units)

# Example
```julia
model = CameraModel(16.0mm, Size2(width=5.86μm/px, height=5.86μm/px), [320.0px, 240.0px])
extrinsics = lookat(SVector(0.0, 0.0, 10.0), SVector(0.0, 0.0, 0.0), SVector(0.0, 1.0, 0.0))
camera = PosedCamera(model, extrinsics)
X_world = SVector(1.0mm, 2.0mm, 3.0mm)
u = project(camera, X_world)
```
"""
@inline project(camera::PosedCamera, X::AbstractVector) =
    project(camera.model, camera.extrinsics(X))
# A physical camera strips at entry, in metres: exactly the plain camera's call
# on the point in metres (a plain point is a `DimensionError`).
@inline project(camera::_UnitfulPosedCamera, X::AbstractVector) =
    project(camera.model, ustrip(m, camera.extrinsics)(strip_metres(X)))

"""
    project(camera::PosedCamera, shape::Union{Circle, Ellipse}) -> Ellipse or Polygon

Project a 2D circle or ellipse that lies on the world z=0 plane through
the camera. For pinhole cameras, the analytic conic projection yields an
`Ellipse`; partially-behind-camera cases or distorting lens models fall
back to boundary-point sampling and return a `Polygon`.
"""
project(::_UnitfulPosedCamera, ::Circle3D) = _refuse_physical_camera("project(camera, ::Circle3D)")
@inline project(camera::PosedCamera, shape::Union{Circle, Ellipse}) =
    _camera_project_conic(camera, shape)

"""
    project(camera::PosedCamera, c::Circle3D) -> Ellipse or Polygon

Project an arbitrarily-placed 3D circle through the camera. For pinhole
cameras, the plane homography induced by the circle's local (u, v) frame
gives the analytic image conic; other cases fall back to boundary
sampling and return a `Polygon`.
"""
@inline project(camera::PosedCamera, c::Circle3D) =
    _camera_project_conic(camera, c)

"""
    project(camera::PosedCamera, rect::HyperRectangle{2}, n::Int=20) -> Polygon

Project a 2D rectangle on the world z=0 plane, tessellating each edge
into `n` samples.
"""
@inline project(camera::PosedCamera, rect::GeometryBasics.HyperRectangle{2}, n::Int = 20) =
    _camera_project_rect(camera, rect, n)

"""
    project(camera::PosedCamera, origin::Point3, dir::Vec3) -> (Point2, Vec2)

Project a 3D origin and displacement through the camera and return the
2D arrow endpoints — the origin's image and the vector from the origin's
image to `(origin + dir)`'s image. Use to plot a 3D normal or direction
as an image-space arrow:

```julia
origin_2d, vec_2d = project(cam, c.center, c.radius * normal(c))
arrows2d!(ax, origin_2d, vec_2d; color=:magenta)
```
"""
@inline function project(camera::PosedCamera, origin::GeometryBasics.Point3, dir::GeometryBasics.Vec3)
    o_img = project(camera, Point3{Float64}(origin...))
    tip_img = project(camera, Point3{Float64}((origin .+ dir)...))
    return (Point2(o_img...), Vec2((tip_img .- o_img)...))
end
# A physical camera: the origin and the displacement are lengths, stripped in
# metres with the camera (a plain one is a `DimensionError`).
project(camera::_UnitfulPosedCamera, origin::GeometryBasics.Point3, dir::GeometryBasics.Vec3) =
    project(ustrip(m, camera), Point3{Float64}(strip_metres(origin)), Vec3{Float64}(strip_metres(dir)))

"""
    backproject(camera::PosedCamera, u::StaticVector{2, Float64}) -> Vec3

Backproject a 2D pixel to a 3D ray direction in **world** coordinates
(normalized).  Returned as a `Vec3` because it represents a direction —
not translatable under affine maps.  The ray's origin is the camera
center (`pose(camera).t`); points along it are
`pose(camera).t + s · dir` for `s > 0`.
"""
@inline backproject(camera::PosedCamera, u::StaticVector{2, Float64}) =
    Vec3(pose(camera).R * backproject(camera.model, u))

@inline function backproject(camera::PosedCamera, points::AbstractVector{<:StaticVector{2, Float64}})
    R = pose(camera).R      # hoisted: one extrinsics inversion for the batch
    m = camera.model
    return [Vec3(R * backproject(m, u)) for u in points]
end

"""
    unproject(camera::PosedCamera, u::AbstractVector{Float64}, depth::Float64) -> Point3

Unproject a 2D pixel at a given camera-space depth (Z along the camera's
forward axis) to a 3D **point** in world coordinates.  Returned as a
`Point3` because it represents a position — translatable under affine
maps.  Works for both `PhysicalIntrinsics` (metric units; `depth` in the
chosen metric unit, e.g. mm) and `LogicalIntrinsics` (unitless pixels;
`depth` carries whatever unit the caller picks for their scene).
"""
@inline unproject(
    camera::PosedCamera,
    u::AbstractVector{Float64}, depth::Float64
) =
    Point3(pose(camera)(unproject(camera.model, u, depth)))
# A physical camera takes a depth that is a length and returns the point in
# metres: the plain call in metres, its result read back in metres.
unproject(camera::_UnitfulPosedCamera, u::AbstractVector{Float64}, depth::Quantity) =
    _attach_point(m, unproject(ustrip(m, camera), u, strip_metres(depth)))

# ============================================================================
# Camera Accessor Functions
# ============================================================================

# Include remaining camera system components
include("target_calibration.jl")  # TargetCalibration: the kinds of a planar target's calibration and their protocol
include("targets.jl")  # AbstractTarget, PointTarget, PlanarTarget{C} (+ the calibratable-target protocol)
include("rig.jl")     # Rig (sensors: tuple storage; targets: dict storage; both: the two side by side; gauge in the frame) + PosedTarget — before the stereo alias
include("mapped_surface.jl")   # MappedSurface: the surface of a chart map, its ray root (after spline_lens.jl: its LM)
include("target_surfaces.jl")  # surface(::PosedTarget): the Quad or MappedSurface a ray meets
include("captures.jl")  # Trajectory, TargetPoses, TargetPoints2D/3D: per-frame values of a capture (+ convert_frame)
include("stereo.jl")
include("rectify.jl")    # stereo rectification on the rig types (0.16)
include("utilities.jl")
include("image_circle.jl")  # the lens model's valid domain, and the image circle from a datasheet through the pitch
include("view_overlap.jl")  # view_overlap: the solid angle every pair of a rotating camera's frames share (uses imaged_region, valid_domain)
include("random.jl")

# Include planar homography (depends on PosedCamera)
include("homography.jl")   # camera-derived HomographyMat (moved out of geometry/ 0.29)
include("calibration.jl")  # zhang_intrinsics — Zhang IAC regression from plane homographies
include("oriented_camera.jl")  # what an OrientedCamera refuses (plane verbs, rig membership), its convert_eltype (after calibration.jl: RadialCamera)

# =============================================================================
# EXPORTS - Organized by category
# =============================================================================

# -----------------------------------------------------------------------------
# 1. SENSORS - Physical CMOS specifications
# -----------------------------------------------------------------------------
export Sensor, sensor                  # Sensor type and lookup
export Board                            # the board a planar target's pattern is printed on (boards.jl)
export Canvas, CanvasTexture             # the printed page a board lies on, and the image printed on it (boards.jl)
export board                            # a planar target's board (targets.jl; public, not exported, at the top level)
export canvas                           # a planar target's canvas (targets.jl; public, not exported, at the top level)
export CMOS_SENSORS                    # Database of common sensors

# -----------------------------------------------------------------------------
# 2. INTRINSICS - Calibration parameters
# -----------------------------------------------------------------------------
export AbstractIntrinsics
export LogicalIntrinsics               # Pixel-based (fx, fy, cx, cy)
export ProjectiveIntrinsics            # the principal point alone: a projective ray frame, no focal length
export principal_point                 # a projective camera's principal point
export CameraCalibrationMat                        # 3x3 pixel-space calibration matrix
export zhang_intrinsics                # Zhang IAC regression from plane homographies

# -----------------------------------------------------------------------------
# 3. LENS MODELS - Projection + distortion
# -----------------------------------------------------------------------------
export AbstractLensModel
export ParameterManifold, LensManifold    # a parametric model's Manifold (inside its parameter_bounds); its name for a lens
export parameter_box                      # its solver search box (public, not exported, at the top level)
export coordinate_scale                   # its coordinate scales (public, not exported, at the top level)
export Pinhole                         # Ideal perspective (no distortion)
export Lens, LiftLens, Division, KannalaBrandt, radial_ratio   # the two lens kinds and the named models
export Scaramuzza                      # Scaramuzza's omnidirectional polynomial lift (MATLAB fisheyeIntrinsics)
export AbstractProjection, Perspective, Equidistant, Stereographic, Equisolid, Orthographic
export DoubleSphere, Unified, ExtendedUnified   # the parametric projections
export Equirectangular   # the latitude/longitude projection of directions (equirectangular.jl)
export Cylindrical, CylindricalEqualArea, Mercator   # the rest of the cylindrical family (equirectangular.jl)
export Backprojection    # a back-projection as a Transformation (equirectangular.jl)
export AbstractPanoramaProjection   # the family's supertype (public, not exported, at the top level)
export AbstractDistortionTerm, Radial, Tangential, ThinPrism, Tilt, FieldOfView, Additive, Even, Full
export radial, isdelta, projection_frame, parameters, lens_record, lens_from_record, lens_label
export LensSpec, free_names             # a lens family by name and term counts (lens_spec.jl; free_names public, not exported, at the top level)
export CameraModelSpec, SplineLensSpec, intrinsics_kind   # what a calibration is for (lens_spec.jl; SplineLensSpec, intrinsics_kind public, not exported, at the top level)
export division_bounded, undistorted_rect, valid_rect, horizon_radius, fov_radius, DEFAULT_MAX_FOV   # whole-extent undistortion
export radial_homography   # a target homography seen through a projective division camera
export undistort, undistort!           # Remove lens distortion from images
export radial_camera_5pt, RadialCamera5pt   # the radial camera's 5-point minimal kernel and its solver
export radial_rows                     # RadialCamera's rows accessor (public, not exported, at the top level)
export RadialCamera                    # the 1-D radial camera model (constraint_with_jacobians/whiten)
export distortion_map                  # Undistorted→distorted COLMAP pixel map
export imaging_transform, pinhole
export undistort_normalized, InverseLens
export parameter_vector, parameter_names, parameter_bounds, with_parameters
export free_parameter_names   # reaches the top level, public there (not exported)
export convert_eltype                  # a camera, lens or intrinsics in another float type (public, not exported, at the top level)
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
export FramedStage, FramedProjection, FramedPerspective, FramedEquidistantProjection
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
export AbstractCamera, PosedCamera          # PosedCamera with extrinsics
export PosedPinholeCamera                   # Lens-dispatch alias
export BoundedCamera, extent           # Wrapper that adds image bounds + accessor
export lens_coverage, with_lens_coverage     # where light reaches a bounded camera's sensor (its image circle, or the whole sensor)
export model                            # CameraModel accessor on PosedCamera/AbstractCamera

# -----------------------------------------------------------------------------
# 5. STEREO - Multi-camera geometry
# -----------------------------------------------------------------------------
export Rig, PosedTarget, source, destination, target_frames  # one Rig for sensors and targets: (model, pose) members; StereoRig = 2-tuple alias
export sensor_rig, target_rig             # the two halves of a rig carrying both
export AbstractTarget, PointTarget, PlanarTarget, resolve, is_resolved   # the target as a model (targets.jl)
export AprilGrid, point_index                 # Kalibr's AprilGrid layout as a PlanarTarget (public, not exported, at the top level)
export SchoepsPattern                         # Schöps et al.'s tagged star pattern as a PlanarTarget (public, not exported, at the top level)
export AbstractTargetLayout, layout, TARGET_LAYOUT_TAGS   # a target's parametric design, recorded by PlanarTarget(layout), and the layouts' tags (public, not exported, at the top level)
export TargetCalibration, Rigid, ThinPlateSpline, HeightField   # a planar target's calibration kinds (with the affine factors and their ∘)
export Aspect, Skew, Scale                    # the affine print error's gauge-free Iwasawa factors (with a planar EuclideanMap)
export placement                              # a calibration's dimensionless chart in the target's frame, a SimilarityMap{3} (targets.jl; public, not exported, at the top level)
export MappedSurface                          # the surface of a chart map: a deformed posed target's (mapped_surface.jl)
export calibration, design, reference, is_planar   # design, reference: public, not exported, at the top level
export is_affine                         # the surface chart's trait (public, not exported, at the top level)
export PlanarCalibration                 # the planar kinds as one type (public, not exported, at the top level)
export TARGET_CALIBRATION_TAGS, PLANAR_MAP_TAGS, calibration_tag, length_mask   # the kinds' tag registry and a kind's tag (public, not exported, at the top level)
export TARGET_CALIBRATION_ORDER, planar_kind, members, is_pure_bending   # the composition's structure (public, not exported, at the top level)
export iwasawa_parameters, iwasawa_parameters_with_jacobian   # an affine's Iwasawa factors' parameters (public, not exported, at the top level)
export regularizer, TargetManifold      # the calibratable-target protocol
export Trajectory, TargetPoses, TargetHomographies, TargetPoints2D, TargetPoints3D   # per-frame values of a capture (captures.jl)
export TargetDetections, FrameStamps      # per frame, per target, index => detection; a capture's per-frame stamps (captures.jl)
export TargetPoint2D, TargetPoint3D, UncertainPixel, UncertainPoint3
export DeviceIdentity, ExposureSettings, LensSettings, FrameSettings, CaptureSettings, is_constant   # a capture's camera settings per frame, its hardware's identity (captures.jl; is_constant public, not exported, at the top level)
export convert_frame                      # the one way to change the pixel frame of pixel data
export rectify                            # stereo rectification: rectified pinhole rig + FramedImage pair
export view_angle_limit                   # honest θ-domains
export valid_domain                       # where the lens MODEL is valid, not where light falls
export image_circle                       # the lens coverage of the sensor, from a datasheet through the pitch
export imaged_region                      # Rect(extent) ∩ the camera's lens coverage ∩ its valid domain: what a bounded camera images
export view_overlap                       # the solid angle every pair of a rotating camera's frames share (view_overlap.jl)
export OrientedCamera                     # a PosedCamera posed by a RotationMap: a camera turning about its centre
export pixel_mask                         # the pixels whose centres lie in a region (COLMAP centres, Julia layout)
export pixel_centres                      # the centres of a sub-lattice of pixels that lie in a region
export SupportTrait, GlobalSupport, LocalSupport, support_trait, nodes, support, node_position   # supports on the sensor (support.jl)
export free_nodes, node_mask, cubic_bspline_footprint
export BoundaryRule, extend_controls, boundary_rule_jacobian   # the dependent nodes of a lattice outside an imaged region (support.jl)
export AbstractVignetting, TensorProductVignetting, RadialVignetting, LogVignetting   # a camera's vignetting fields (vignetting.jl)
export VIGNETTING_TAGS                    # the fields' tag registry (public, not exported, at the top level)
export log_spline                         # a LogVignetting's spline of log V (public, not exported, at the top level)
export EnvironmentLight, SphericalHarmonicMap   # the scene's light: SH radiance on a sphere about a centre (lighting.jl)
export AbstractBlur, GaussianBlur, convolve   # one blur of an imaging model by its covariance (blur.jl; `convolve` public, not exported, at the top level)
export radiance, exit_direction, radiance_map, coefficients, with_coefficients, max_degree, sh_basis, sh_count, sh_index   # (public, not exported, at the top level)
export coefficient_jacobian, active_coefficients, control_points, with_control_points, n_control_points, knot_vectors, spline_degree,
       field_radius, radial_coordinate, pole_regular_basis   # (public, not exported, at the top level)
export RADIAL_CHARTS, projective_chart, projective_knots, radial_chart, with_center, center_gradient   # a radial profile's chart and centre (public, not exported, at the top level)
export radial_part                         # a lens's radially symmetric part
export SplineLens, NoncentralSplineModel, spline_lattice        # the central grid camera as a lens of a CameraModel; the non-central grid; a lattice from a camera, unsampled
export Ray3                              # a non-central pixel's observation line
export present_nodes, window, lattice_spacing   # the nodes a spline model holds; the camera of a block of its controls; its cell size in px (public, not exported, at the top level)
export directions, grid_size, window_offset, spline_weights   # its control grid, (gx, gy), a window's corner in its model's grid; a point's window weights and their partials (likewise)
export direction_tangents, default_spline_grid   # a direction's tangent basis; the reference's control grid of an area
export ControlPatchManifold              # a block of a spline lattice's controls as a manifold (public, not exported, at the top level)
export spline_window, spline_value, quasi_interpolant   # the uniform cubic lattice as a primitive (any control type); the 4th-order quasi-interpolant
export cubic_bspline_weights             # a uniform cubic B-spline span's four weights (public, not exported, at the top level)
export with_present, with_directions, with_points, central_model   # a spline model with one part replaced; a non-central model's central part (likewise)
export axis_gauged                      # a SplineLens camera (and its poses) in the frame turned so ẑ is its optical axis (spline_lens.jl)
export ChartSpline                       # a uniform cubic B-spline on a 2-D chart
export pixel_centre                      # an image array index's pixel centre, COLMAP frame, in T (public, not exported, at the top level)
export equidistant_chart                 # the equidistant direction chart about an axis (public, not exported, at the top level)
export forward_chart                     # the chart of every SplineLens forward spline (public, not exported, at the top level)
export has_spline_gauge                  # whether a SplineLens pixel lattice gives its own K (public, not exported, at the top level)
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
export normalized_point   # a pixel's normalized image point K⁻¹𝐩, the pixel-frame half of backproject (public, not exported, at the top level)
export bearing_csponds   # pixel pairs → bearing csponds with their M₁/M₂ pixel Jacobians

# -----------------------------------------------------------------------------
# 7. CAMERA ACCESSORS - Extract camera properties
# -----------------------------------------------------------------------------
export focal_length, calibration_matrix
export angle_of_view, field_of_view       # the angle of view a bounded camera images, and the size it spans at a distance (utilities.jl)
export pose, extrinsics, is_in_front, default_frustum_depth

# -----------------------------------------------------------------------------
# 8. CAMERA CONSTRUCTION
# -----------------------------------------------------------------------------
export lookat                          # Construct extrinsics from (eye, target, up)
export lookat_pose                     # Camera-to-world pose form of lookat (pairs with PosedCamera(model; pose=...))

# -----------------------------------------------------------------------------
# 9. POSE ESTIMATION - P3P and calibration
# -----------------------------------------------------------------------------
export p3p, sample_p3p

# -----------------------------------------------------------------------------
# 10. HOMOGRAPHY - Projective transformations
# -----------------------------------------------------------------------------
export HomographyMat                   # 3x3 homography matrix

# =============================================================================
# BoundedCamera: the camera verbs forwarded to its camera
# =============================================================================
#
# A BoundedCamera is THE type for pixel operations (the owner's rule): every camera
# verb a CameraModel answers reaches the wrapped camera (`bc.camera`), by the Julia
# idiom for "a method on the wrapper falls through to the inner value",
# MacroTools.@forward — here, after every file that defines one of them; `project`
# and `backproject` are the two `@inline` maps above. As those two never did, no
# forwarded verb applies the bound (the image extent is `image_circle`,
# `imaged_region` and `lens_coverage`, which a BoundedCamera answers itself). The
# model constructors (`pinhole`, `window`, `with_parameters`) and the parameter
# protocol are the model's, not forwarded (as is the synthetic-scene generator
# `sample_p3p`, which takes its own sensor bounds): call them on `model(bc)` and rebound with
# `with_model`. A verb with its own BoundedCamera method is not in the list (an
# ambiguity otherwise): `valid_domain` (image_circle.jl, through `model`),
# `with_directions`, `with_present`, `axis_gauged`, `image_circle`.
MacroTools.@forward BoundedCamera.camera (
    model, intrinsics, lens, focal_length, calibration_matrix,
    pose, extrinsics,
    unproject, is_in_front, backproject_with_jacobian, constraint_with_jacobians,
    imaging_transform, imaging_chain, distortion_map, has_distortion, fov_radius, horizon_radius,
    valid_rect, undistorted_rect, division_bounded, undistort,
    p3p, transfer, radial_homography,
)
# A mutating verb takes its destination first and the camera after it (Julia's order), which `@forward` cannot reach:
# `undistort!` (and its typed and rect forms) reaches the wrapped camera by hand.
undistort!(dst::AbstractMatrix, bc::BoundedCamera, src::AbstractMatrix, args...; kwargs...) =
    undistort!(dst, bc.camera, src, args...; kwargs...)
undistort!(dst::AbstractMatrix, ::Type{T}, bc::BoundedCamera, src::AbstractMatrix, args...; kwargs...) where {T <: AbstractFloat} =
    undistort!(dst, T, bc.camera, src, args...; kwargs...)

end # module Cameras
