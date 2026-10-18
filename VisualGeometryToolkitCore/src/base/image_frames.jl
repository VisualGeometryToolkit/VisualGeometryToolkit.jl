# =============================================================================
# Image frames — 2-D coordinate systems on the viewing sphere and the sensor
# =============================================================================
#
# These frames are charts in the DIFFERENTIAL-GEOMETRY sense: local coordinate
# systems on S^2 / P^2, each with its own domain of validity, related by
# `FrameTransition` (an atlas' transition maps). `GnomonicFrame` is the
# GNOMONIC chart — the AFFINE chart of P^2, the
# complement of the line at infinity — which is why K acts affinely on it and
# only on it, and why it cannot represent a direction at 90 degrees.
#
# NOT to be confused with `Manifold` (base/manifold.jl): that is a
# parameterization (a retraction) of a MODEL manifold for the optimizer, whose
# points are cameras and poses. These frames' points are directions and pixels.
# Split out of base/frames.jl, which carried both concepts.
#
# Loaded AFTER base/frames.jl (the `Frame{N}` supertype these extend) and
# BEFORE base/types.jl (the statistical forms constrain their frame parameter).
# =============================================================================

"""
    TargetPlaneFrame <: Frame{2}

2D frame of the calibration-board plane (board-local `(x, y)` coordinates on
the plane `z = 0` of the board's 3D frame).

Source frame of the plane-induced homography of a pinhole camera:
`HomographyMap(camera::PinholeCamera)` is labeled
`Src = TargetPlaneFrame, Dst = PixelFrame{:colmap}` (monogram `H_PB`). Adoption by
the F/E actions is deferred to 0.8 (pending view-identity symbol frames).
"""
struct TargetPlaneFrame <: Frame{2} end

"""
    PixelFrame{Origin} <: Frame{2}

2D pixel frame of an image, tagged by its origin convention:
`Origin::Symbol` names the pixel-origin placement. The registered origins:

| frame                 | first-pixel center | corner offset  |
|-----------------------|--------------------|----------------|
| `PixelFrame{:opencv}` | `(0, 0)`           | `(-0.5, -0.5)` |
| `PixelFrame{:colmap}` | `(0.5, 0.5)`       | `(0, 0)`       |
| `PixelFrame{:julia}`  | `(1, 1)`           | `(0.5, 0.5)`   |

(x right / y down throughout; `:julia` is additionally read row-major
`(row, col)` at the array boundary — that permutation is `image_warp`'s
`RCSwap`, not part of the frame.)

Origin conversion is a labeled rigid map between two pixel frames
(identity rotation, constant translation), built by the
constructor-as-factory on the frame TYPES:

```julia
T = EuclideanMap(PixelFrame{:colmap}, PixelFrame{:julia})
T isa EuclideanMap{2, …, PixelFrame{:colmap}, PixelFrame{:julia}}  # true
T.t == Vec2(0.5, 0.5)                                              # true
pt_julia = T(pt_colmap)
```

The map composes (frame-checked) with other labeled 2D maps and fuses in
`image_warp` chains like any `EuclideanMap{2}`. This is the ONLY origin
conversion API: the 0.6-era `ImageOrigin` value surface
(`COLMAP_ORIGIN`, `JULIA_ORIGIN`, `Translation(::ImageOrigin, ...)`,
`Translation(::Symbol, ::Symbol)`) is removed in 0.7.
"""
struct PixelFrame{Origin} <: Frame{2} end
"""
    ImageFrame{id} <: Frame{2}

The 2D image frame of view `id` — the per-view pixel frame used to label the
argument slots of epipolar forms (see [`EpipolarForm`](@ref)) and the
endpoints of inter-view 2D maps.

Deliberate division of labor with [`PixelFrame`](@ref): `PixelFrame{Origin}`
tags a pixel-origin CONVENTION (`:colmap`/`:julia`/`:opencv`) — a property
of how coordinates are written; `ImageFrame{id}` tags WHICH VIEW an image
point belongs to — a property of what the coordinates refer to. The two are
orthogonal, and `ImageFrame` does not encode an origin: convert origin
conventions FIRST (via the `EuclideanMap(PixelFrame{A}, PixelFrame{B})`
frame maps, landing on the canonical `:colmap` convention), then label the
points' frame with their view. An `ImageFrame{id}` point is therefore always
read in the canonical `:colmap` convention.

See also [`ViewFrame`](@ref) (the 3D frame of the same view),
[`GnomonicFrame`](@ref) (the calibrated frame of the same view),
[`image_frames`](@ref), [`Frame`](@ref).
"""
struct ImageFrame{id} <: Frame{2} end

"""
    NormalizedImageFrame{id} <: Frame{2}

The diagonal-normalized 2D image frame of view `id`: [`ImageFrame`](@ref)
coordinates moved to the image centre and divided by the half-diagonal
`s = hypot(width, height) / 2`, so the image corners sit at unit radius.
Pixel-scale constants are sensor-independent here: a focal length is the
dimensionless `f / s` (1 for a 90° diagonal field of view) and a pixel
threshold `τ` becomes `τ / s`. Mint the map with the factory
`FramedAffine(ImageFrame{id}, NormalizedImageFrame{id}, sz)` (or from a
[`PixelFrame`](@ref)`{Origin}` source); its `inv` de-normalizes.

Not to be confused with [`GnomonicFrame`](@ref), the CALIBRATED frame
`K⁻¹·pixel`: this frame needs no intrinsics, only the image size.
"""
struct NormalizedImageFrame{id} <: Frame{2} end
"""
    RetinalFrame{id} <: Frame{2}

Abstract supertype of the canonical 2D ray frames of view `id` — the
coordinate systems on the view's SPHERICAL RETINA, the sphere S² of rays
through the center of projection (the manifold-level object of the
Geyer–Daniilidis unified imaging model). Every central camera has a
spherical retina; what differs between camera models is the FRAME its
distortion is expressed in, so frames — not retinas — are the type
currency of the imaging chain. The "planar retina" of the perspective
camera is not a different manifold but the gnomonic frame on this one
([`GnomonicFrame`](@ref)); the equidistant frame
([`EquidistantFrame`](@ref)) covers the fields of view the gnomonic frame
cannot.

Subtypes share the azimuth and differ in the radial profile — the five
classical azimuthal frames of the fisheye-projection family, each a
function of the polar angle `θ` with its own honest domain:

| frame                        | radial profile   | domain               |
|------------------------------|------------------|----------------------|
| [`GnomonicFrame`](@ref)       | `r = tan θ`      | `θ ∈ [0, π/2)`       |
| [`EquidistantFrame`](@ref)   | `r = θ`          | `θ ∈ [0, π)`         |
| [`StereographicFrame`](@ref) | `r = 2·tan(θ/2)` | `θ ∈ [0, π)`         |
| [`EquisolidFrame`](@ref)     | `r = 2·sin(θ/2)` | `θ ∈ [0, π]`         |
| [`OrthographicFrame`](@ref)  | `r = sin θ`      | `θ ∈ [0, π/2]` ONLY  |

(the orthographic frame FOLDS at the equator — see its docstring). The
transition maps between them are pure radial reparameterizations, minted
as first-class framed stages by `frame_transition`. A point of the retina
written in no 2D frame needs no 2D tag — it is a unit 3-vector in
[`ViewFrame`](@ref)`{id}`.

`imaging_chain` stage 1 (the ray-frame projection family) maps
`ViewFrame{id} → RetinalFrame{id}`; `distortion_frame(lens)` names which
subtype a lens model's distortion is defined in. [`DistortedFrame`](@ref)
is deliberately NOT a subtype: it is defined by the lens map's image, not
as a canonical frame of the retina.

See also [`GnomonicFrame`](@ref), [`EquidistantFrame`](@ref),
[`StereographicFrame`](@ref), [`EquisolidFrame`](@ref),
[`OrthographicFrame`](@ref), [`DistortedFrame`](@ref), [`Frame`](@ref).
"""
abstract type RetinalFrame{id} <: Frame{2} end

"""
    GnomonicFrame{id} <: RetinalFrame{id}

The calibrated (normalized) 2D image frame of view `id`: the frame whose
points are gnomonic coordinates `x̂ = π(K⁻¹ũ)` — pixel coordinates with the
intrinsics removed, i.e. the perspective projection of the view's unit-focal
image plane. `EssentialMat`-backed epipolar forms are honest on this frame
(`x̂₂ᵀ E x̂₁ = 0`); [`ImageFrame`](@ref) remains the PIXEL frame (canonical
`:colmap` convention) and is never overloaded for gnomonic coordinates.

Formally, this is the GNOMONIC chart of the view's ray sphere S² — the
central projection of directions onto the unit-focal plane. Like every
gnomonic chart it covers only the forward hemisphere: it is degenerate at
90° from the optical axis and cannot represent rays with `Z ≤ 0`. It is the
defining frame of the perspective projection's distortions (OpenCV's
radial-tangential-thin-prism models); omnidirectional models define their
distortion on the [`EquidistantFrame`](@ref) instead — see the frame
table at [`DistortedFrame`](@ref). The framed projection onto this frame is
`FramedPerspective` (`imaging_chain` stage 1 for perspective-plane lenses).

The calibrated → pixel collineation of view `id` is the labeled lift of the
view's intrinsics, `HomographyMap{GnomonicFrame{id}, ImageFrame{id}}(K)` —
the slot maps whose composition with the correlation computes `F = K₂⁻ᵀ E K₁⁻¹` with the
frame labels verifying that each `K` met the slot it was meant for. Minted
by `essential_form(rig)` / `EpipolarForm(rig)` on a `StereoRig`.

See also [`EquidistantFrame`](@ref), [`DistortedFrame`](@ref),
[`ImageFrame`](@ref), [`ViewFrame`](@ref), [`Frame`](@ref).
"""
struct GnomonicFrame{id} <: RetinalFrame{id} end

"""
    EquidistantFrame{id} <: RetinalFrame{id}

The equidistant-azimuthal 2D ray frame of view `id`: the frame on the
viewing sphere S² whose point for a ray at polar angle `θ` (from the optical
axis) and azimuth `ψ` is `θ·(cos ψ, sin ψ)` — radial coordinate equal to the
ANGLE, azimuth preserved. Well-defined for `θ ∈ [0, π)` (every ray except
the exact backward direction), so it represents fields of view past 90°
where the gnomonic [`GnomonicFrame`](@ref) is degenerate.

"Equidistant" (radial coordinate `r = θ`) names this frame unambiguously
within the fisheye family, whose classical frames are all functions of the
polar angle — equidistant `r = θ`, stereographic `r = 2·tan(θ/2)`,
equisolid `r = 2·sin(θ/2)`, orthographic `r = sin θ` — so "the angle
frame" would not single one out. The other radial profiles slot under
[`RetinalFrame`](@ref) beside this one (0.13):
[`StereographicFrame`](@ref), [`EquisolidFrame`](@ref),
[`OrthographicFrame`](@ref).

This is the DEFINING frame of the omnidirectional distortion models:
`KannalaBrandt` is the radial polynomial `θ_d = θ·(1 + k₁θ² + …)` on this
frame, the pure `Lens(Equidistant())` is the IDENTITY on it (the "distortion"
of a pure equidistant fisheye is entirely the frame choice), and COLMAP's
`THIN_PRISM_FISHEYE` applies its additive terms to this frame's coordinates.
The framed projection onto this frame is `FramedEquidistantProjection`
(`imaging_chain` stage 1 for equidistant-frame lenses); see the frame table
at [`DistortedFrame`](@ref) for how it slots into the imaging chain.

Not to be confused with [`Manifold`](@ref): that machinery parameterizes
MODEL manifolds (essential matrices, homographies) for estimation;
`EquidistantFrame` is a coordinate frame on the viewing sphere of one
camera.

See also [`GnomonicFrame`](@ref), [`DistortedFrame`](@ref),
[`ImageFrame`](@ref), [`Frame`](@ref).
"""
struct EquidistantFrame{id} <: RetinalFrame{id} end

"""
    StereographicFrame{id} <: RetinalFrame{id}

The stereographic 2D ray frame of view `id`: the frame on the viewing
sphere S² whose point for a ray at polar angle `θ` (from the optical axis)
and azimuth `ψ` is `2·tan(θ/2)·(cos ψ, sin ψ)` — the classical
stereographic projection of the sphere from its backward pole onto the
tangent plane at the forward pole. Well-defined for `θ ∈ [0, π)` (every
ray except the exact antipode), and CONFORMAL — the only angle-preserving
member of the fisheye-projection family, which is why stereographic
("fisheye of least visual distortion") lenses are prized in panorama
work. Radius is unbounded as `θ → π`.

The defining frame of the pure stereographic fisheye projection:
`Lens(Stereographic())` is the IDENTITY on it, and the framed
projection onto this frame is `FramedStereographicProjection`
(`imaging_chain` stage 1). Transitions to the other retinal frames are
minted by `frame_transition`; see the family table at
[`RetinalFrame`](@ref) and the chain table at [`DistortedFrame`](@ref).

See also [`EquidistantFrame`](@ref), [`EquisolidFrame`](@ref),
[`OrthographicFrame`](@ref), [`GnomonicFrame`](@ref), [`Frame`](@ref).
"""
struct StereographicFrame{id} <: RetinalFrame{id} end

"""
    EquisolidFrame{id} <: RetinalFrame{id}

The equisolid-angle 2D ray frame of view `id`: the frame on the viewing
sphere S² whose point for a ray at polar angle `θ` and azimuth `ψ` is
`2·sin(θ/2)·(cos ψ, sin ψ)` — the Lambert azimuthal equal-area projection.
Well-defined on the WHOLE sphere `θ ∈ [0, π]` (the only retinal frame that
reaches the antipode), with bounded radius `r ∈ [0, 2]`; equal areas on
the sphere map to equal areas in the frame, which is why equisolid
("equal-area fisheye") lenses are the standard for sky photometry and
illuminance measurement. The antipode `θ = π` maps to radius 2 under the
pole azimuth convention `ψ = 0`.

The defining frame of the pure equisolid fisheye projection:
`Lens(Equisolid())` is the IDENTITY on it, and the framed
projection onto this frame is `FramedEquisolidProjection`
(`imaging_chain` stage 1). Transitions to the other retinal frames are
minted by `frame_transition`; see the family table at
[`RetinalFrame`](@ref) and the chain table at [`DistortedFrame`](@ref).

See also [`StereographicFrame`](@ref), [`OrthographicFrame`](@ref),
[`EquidistantFrame`](@ref), [`GnomonicFrame`](@ref), [`Frame`](@ref).
"""
struct EquisolidFrame{id} <: RetinalFrame{id} end

"""
    OrthographicFrame{id} <: RetinalFrame{id}

The orthographic 2D ray frame of view `id`: the frame on the viewing
sphere S² whose point for a ray at polar angle `θ` and azimuth `ψ` is
`sin θ·(cos ψ, sin ψ)` — parallel projection of the sphere straight down
onto the equatorial plane, the image formed by an orthographic ("sine
law") fisheye lens.

**Honest domain: `θ ∈ [0, π/2]` ONLY — the frame FOLDS at the equator.**
`r = sin θ` increases to 1 at `θ = π/2` and then DECREASES again: past 90°
every radius value is taken twice, so the projection is non-injective on
any domain crossing the equator and the frame is restricted to the closed
forward hemisphere. This is the poster child for the honest-domain
discipline of the retinal frames — a ray with `Z < 0` is not "a large
radius" in this frame, it is OUTSIDE the frame, and the section/lift
machinery throws a `DomainError` rather than silently aliasing it onto a
forward ray (the lift always selects the forward sheet `cos θ =
+√(1 − r²)`). Radius is bounded, `r ∈ [0, 1]`.

The defining frame of the pure orthographic fisheye projection:
`Lens(Orthographic())` is the IDENTITY on it, and the framed
projection onto this frame is `FramedOrthographicProjection`
(`imaging_chain` stage 1). Transitions to the other retinal frames are
minted by `frame_transition`; see the family table at
[`RetinalFrame`](@ref) and the chain table at [`DistortedFrame`](@ref).

See also [`StereographicFrame`](@ref), [`EquisolidFrame`](@ref),
[`EquidistantFrame`](@ref), [`GnomonicFrame`](@ref), [`Frame`](@ref).
"""
struct OrthographicFrame{id} <: RetinalFrame{id} end

"""
    DistortedFrame{id} <: Frame{2}

The post-lens, pre-calibration 2D frame of view `id`: the frame a lens
model's forward distortion lands on, one framed K-affine short of pixels.
Completes the imaging-frame vocabulary:

| frame                | meaning                                                        |
|----------------------|----------------------------------------------------------------|
| [`GnomonicFrame`](@ref)`{id}`     | gnomonic ray frame (calibrated, undistorted): `x̂ = π(K⁻¹ũ)` |
| [`EquidistantFrame`](@ref)`{id}` | equidistant ray frame (radius `θ`): the omni models' home |
| [`StereographicFrame`](@ref)`{id}` | stereographic ray frame (radius `2·tan(θ/2)`, conformal) |
| [`EquisolidFrame`](@ref)`{id}`   | equisolid ray frame (radius `2·sin(θ/2)`, equal-area)     |
| [`OrthographicFrame`](@ref)`{id}` | orthographic ray frame (radius `sin θ`; folds at 90°)    |
| `DistortedFrame{id}` | post-lens, pre-`K`: where forward distortion lands             |
| [`ImageFrame`](@ref)`{id}`      | canonical pixel frame of the view (`:colmap` convention)   |
| [`PixelFrame`](@ref)`{Origin}` | pixel ORIGIN conventions — orthogonal to view identity    |

The ray-frame rows are all frames on the same viewing sphere S² (the
[`RetinalFrame`](@ref) family): the gnomonic frame (perspective plane) is
ONE frame on S², degenerate at 90° and blind to `Z ≤ 0` rays; the
equidistant and stereographic frames cover `θ ∈ [0, π)`, the equisolid
frame the whole sphere, and the orthographic frame only the closed forward
hemisphere (it folds at the equator).
"Perspective precedes lens" is therefore a property of perspective-plane
MODELS, not a law of the imaging chain — each lens model's distortion is
labeled with the frame it is actually defined in (`distortion_frame`), and the
typed chain of `imaging_chain` reads

    ViewFrame{id} ─ ray-frame projection → Gnomonic/EquidistantFrame{id}
                  ─ lens distortion      → DistortedFrame{id}
                  ─ framed K-affine      → ImageFrame{id}

with every junction frame-checked. When a lens's distortion is the identity
in its defining frame (`has_distortion == false`, e.g. `Pinhole`; also the
pure `Lens(Equidistant())` on its equidistant frame), the distortion stage is
omitted and the ray frame meets `K` directly — the two frames COLLAPSE
rather than being separated by a manufactured identity stage.

**Anonymous view id.** The id axis is optional: the reserved symbol `:_` is
the designated ANONYMOUS view identity, and every stage mint defaults to it.
Frame KIND then still does the junction checking — a lone `CameraModel`
with no view identity gets `GnomonicFrame{:_} → DistortedFrame{:_}` stages
whose distort-vs-undistort direction is checked by kind — while rig-derived
chains mint real ids (`imaging_chain(model; id = ids(rig)[1])`). The
wildcard rules for [`UnlabeledFrame`](@ref) are unchanged; `:_` is a real
label, not a wildcard.

See also [`GnomonicFrame`](@ref), [`EquidistantFrame`](@ref),
[`ImageFrame`](@ref), [`Frame`](@ref).
"""
struct DistortedFrame{id} <: Frame{2} end

# =============================================================================
# FrameDependence — what pins a frame tag down (0.14)
# =============================================================================

"""
    FrameDependence

Holy trait classifying a 2D frame TAG by what DEFINES the frame it names —
the retina's geometry alone, or a model's continuous parameters:

  - [`GeometryDefined`](@ref)`()`: the frame is fixed by geometry and
    conventions — every [`RetinalFrame`](@ref) kind (gnomonic /
    equidistant / stereographic / equisolid / orthographic),
    [`ImageFrame`](@ref), [`NormalizedImageFrame`](@ref) (fixed by the image
    size), [`PixelFrame`](@ref)`{Origin}`, [`TargetPlaneFrame`](@ref).
  - [`ModelDefined`](@ref)`()`: the frame is pinned down by a model's
    continuous parameters — [`DistortedFrame`](@ref) (which frame it IS
    depends on the view's lens coefficients).

The trait encodes a COMMENSURABILITY rule: points in `GeometryDefined`
frames of the same kind and view are always commensurable — the tag alone
determines the frame, so a model-free [`frame_transition`](@ref) between
two such frames exists and the sphere sections/lifts behind it are defined.
Points in `ModelDefined` frames are commensurable only under the
one-lens-per-view invariant that makes `DistortedFrame{id}` honest, and no
model-free transition into or out of them exists — the model-carrying maps
(`distort_stage`/`undistort_stage`, or a fused model-carrying projection
stage) are the only bridges.

Deliberate boundary: the trait classifies the KIND of a frame tag; it
cannot and does not verify parameter equality between two `ModelDefined`
uses — parameter values are a runtime question, exactly as rank lives on
the manifold rather than in the matrix type. Query with
[`frame_dependence`](@ref).
"""
abstract type FrameDependence end

"Frame fixed by the retina's geometry/conventions alone. See [`FrameDependence`](@ref)."
struct GeometryDefined <: FrameDependence end

"Frame pinned down by a model's continuous parameters. See [`FrameDependence`](@ref)."
struct ModelDefined <: FrameDependence end

"""
    frame_dependence(C::Type{<:Frame{2}}) -> FrameDependence

What defines the frame named by tag `C`: [`GeometryDefined`](@ref) for the
[`RetinalFrame`](@ref) family, [`ImageFrame`](@ref),
[`NormalizedImageFrame`](@ref), [`PixelFrame`](@ref), and
[`TargetPlaneFrame`](@ref);
[`ModelDefined`](@ref) for [`DistortedFrame`](@ref). Downstream packages
minting their own frame tags add a method; there is no default — an
unclassified frame tag is a `MethodError`, not a silent guess. See
[`FrameDependence`](@ref) for the commensurability rule the trait encodes.
"""
function frame_dependence end

frame_dependence(::Type{<:RetinalFrame}) = GeometryDefined()
frame_dependence(::Type{<:ImageFrame}) = GeometryDefined()
frame_dependence(::Type{<:NormalizedImageFrame}) = GeometryDefined()
frame_dependence(::Type{<:PixelFrame}) = GeometryDefined()
frame_dependence(::Type{TargetPlaneFrame}) = GeometryDefined()
frame_dependence(::Type{<:DistortedFrame}) = ModelDefined()
