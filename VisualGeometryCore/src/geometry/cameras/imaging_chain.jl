# =============================================================================
# Framed imaging chain — typed frame-to-frame stages (0.10)
# =============================================================================
#
# The imaging chain of a camera, as FRAMED stages with checked junctions:
#
#   ViewFrame{id} ─ ray-frame projection → Gnomonic/EquidistantFrame{id}
#                 ─ lens distortion      → DistortedFrame{id}
#                 ─ framed K-affine      → ImageFrame{id}
#
# Design (see the frame table at `DistortedFrame`):
#
#   * The perspective plane is only the GNOMONIC chart of the viewing sphere
#     S² — degenerate at 90°, blind to Z ≤ 0 rays. "Perspective precedes
#     lens" is a property of perspective-plane MODELS, not a chain law, so
#     stage 1 is a ray-frame projection FAMILY: `FramedPerspective`
#     (gnomonic: the lens contract's Gnomonic frame projection) for
#     perspective-frame lenses, `FramedEquidistantProjection` (equidistant
#     azimuthal) for the omnidirectional models whose distortion is a radial
#     map in θ. `distortion_frame(lens)` names each model's defining frame.
#
#   * The SCALAR math is untouched: framed stages wrap the lens contract's
#     maps (`lens_rays.jl`), never reimplement them. The frame-free
#     `imaging_transform` (`AffineMap(K) ∘ lens`, the lens on the ray) is the
#     same map without frames; `imaging_chain` is the fully-typed mint.
#
#   * Junctions follow the standing wildcard truth table (see `Base.:∘` at
#     `EuclideanMap`), stated ONCE here on the abstract `FramedStage` so
#     every stage pair shares the same five methods. All mints label their
#     stages (the anonymous id `:_` is a real label), so in practice every
#     junction is checked; `UnlabeledFrame` endpoints — reachable only
#     through explicit type-parameter construction — remain wildcards.

"""
    FramedStage{Src <: Frame, Dst <: Frame} <: Transformation

Abstract supertype of the framed imaging-chain stages: a map consuming
`Src`-frame coordinates and producing `Dst`-frame coordinates (same tag
semantics as `EuclideanMap`/`HomographyMap`). Concrete stages:
[`FramedPerspective`](@ref), [`FramedEquidistantProjection`](@ref),
[`FramedStereographicProjection`](@ref), [`FramedEquisolidProjection`](@ref),
[`FramedOrthographicProjection`](@ref) (3D → 2D ray projections),
[`FramedLensProjection`](@ref) (the fused model-carrying 3D → 2D
projection of a lens with a parametric projection),
[`FramedDistort`](@ref), [`FramedUndistort`](@ref), [`FramedAffine`](@ref),
[`FrameTransition`](@ref) (2D → 2D), the sphere self-maps
[`RayMap`](@ref)/[`InverseRayMap`](@ref) (3D → 3D), and the composite
[`FramedChain`](@ref).

`∘` between framed stages is junction-checked under the standing wildcard
truth table (see `Base.:∘` at `EuclideanMap`) and produces a flat
[`FramedChain`](@ref); a labeled junction mismatch is a `MethodError`.
Composing with a foreign (unframed) `Transformation` falls through to
CoordinateTransformations' lazy composition — gradual, as always.
"""
abstract type FramedStage{Src <: Frame, Dst <: Frame} <:
CoordinateTransformations.Transformation end

"The `Src` frame tag of a framed stage (unexported; qualified access)."
src_frame(::FramedStage{Src, Dst}) where {Src, Dst} = Src
"The `Dst` frame tag of a framed stage (unexported; qualified access)."
dst_frame(::FramedStage{Src, Dst}) where {Src, Dst} = Dst

# -----------------------------------------------------------------------------
# Ray-frame projections (3D → 2D): the stage-1 family
# -----------------------------------------------------------------------------

"""
    FramedPerspective{Src <: Frame{3}, Dst <: Frame{2}}
    FramedPerspective(; id::Symbol = :_)

The framed GNOMONIC ray-frame projection: perspective division
`(X, Y, Z) ↦ (X/Z, Y/Z)` from the 3D view frame onto the view's
[`GnomonicFrame`](@ref). The math is the lens contract's Gnomonic frame
projection (`lens_rays.jl`) — one definition of the gnomonic map; this stage
wears the frames. The keyword
constructor mints `FramedPerspective{ViewFrame{id}, GnomonicFrame{id}}()`
(anonymous view `:_` by default — see [`DistortedFrame`](@ref)).

# No `inv` — deliberate

Perspective projection collapses each ray (a 1-parameter fiber) to a point:
a 3D → 2D fiber collapse has no inverse, so `inv` is absent by design (a
`MethodError`, pinned by test). The right-inverses — the SECTIONS of the
projection, each selecting one point per fiber via a depth — are
[`unproject`](@ref) (pixel + depth → 3D point) and [`backproject`](@ref)
(pixel → unit ray, the norm-one section); use those to go back.

Serves the lenses whose distortion is defined on the perspective plane
(`distortion_frame(lens) === GnomonicFrame`). Omnidirectional models use
[`FramedEquidistantProjection`](@ref) instead.
"""
struct FramedPerspective{Src <: Frame{3}, Dst <: Frame{2}} <: FramedStage{Src, Dst} end

FramedPerspective(; id::Symbol = :_) =
    FramedPerspective{ViewFrame{id}, GnomonicFrame{id}}()

@inline (::FramedPerspective)(X::StaticVector{3}) = _sphere_section(GnomonicFrame, X)

# `inv` absent by design (fiber collapse) — pinned as the house MethodError
# idiom. The explicit throw is needed because CT defines a generic
# `inv(::Transformation)` fallback (an ErrorException), which would
# misreport the ABSENCE as a missing definition rather than a non-existent
# operation. Same for FramedEquidistantProjection below.
Base.inv(p::FramedPerspective) = throw(MethodError(inv, (p,)))

"""
    FramedEquidistantProjection{Src <: Frame{3}, Dst <: Frame{2}}
    FramedEquidistantProjection(; id::Symbol = :_)

The framed EQUIDISTANT-AZIMUTHAL ray-frame projection: a 3D direction at
polar angle `θ = atan(√(X²+Y²), Z)` and azimuth `ψ` maps to `θ·(cos ψ,
sin ψ)` on the view's [`EquidistantFrame`](@ref). Well-defined for
`θ ∈ [0, π)` — including `Z ≤ 0` rays, where the gnomonic frame fails —
with the azimuth convention `ψ = 0` at the on-axis poles (`X = Y = 0`).
The keyword constructor mints
`FramedEquidistantProjection{ViewFrame{id}, EquidistantFrame{id}}()`.

Serves the omnidirectional lenses
(`distortion_frame(lens) === EquidistantFrame`): `KannalaBrandt`,
the pure equidistant lens, COLMAP's thin-prism fisheye. Like [`FramedPerspective`](@ref)
it is a fiber collapse with no `inv`; its sections are
`unproject`/`backproject` on the owning camera model.
"""
struct FramedEquidistantProjection{Src <: Frame{3}, Dst <: Frame{2}} <:
    FramedStage{Src, Dst} end

FramedEquidistantProjection(; id::Symbol = :_) =
    FramedEquidistantProjection{ViewFrame{id}, EquidistantFrame{id}}()

@inline (::FramedEquidistantProjection)(X::StaticVector{3}) = _sphere_section(EquidistantFrame, X)

# Fiber collapse — no inverse (see the FramedPerspective inv note).
Base.inv(p::FramedEquidistantProjection) = throw(MethodError(inv, (p,)))

"""
    FramedStereographicProjection{Src <: Frame{3}, Dst <: Frame{2}}
    FramedStereographicProjection(; id::Symbol = :_)

The framed STEREOGRAPHIC ray-frame projection: a 3D direction at polar
angle `θ` and azimuth `ψ` maps to `2·tan(θ/2)·(cos ψ, sin ψ)` on the
view's [`StereographicFrame`](@ref) — projection of the ray sphere from
its backward pole, the conformal member of the fisheye family.
Well-defined for `θ ∈ [0, π)`; the exact antipode is outside the frame,
where the stage answers NaN, as the lens contract does (the stereographic
radius diverges there). The
keyword constructor mints
`FramedStereographicProjection{ViewFrame{id}, StereographicFrame{id}}()`.

Serves `Lens(Stereographic())`
(`distortion_frame(lens) === StereographicFrame`). Like
[`FramedPerspective`](@ref) it is a fiber collapse with no `inv`.
"""
struct FramedStereographicProjection{Src <: Frame{3}, Dst <: Frame{2}} <:
    FramedStage{Src, Dst} end

FramedStereographicProjection(; id::Symbol = :_) =
    FramedStereographicProjection{ViewFrame{id}, StereographicFrame{id}}()

@inline (::FramedStereographicProjection)(X::StaticVector{3}) =
    _sphere_section(StereographicFrame, X)

# Fiber collapse — no inverse (see the FramedPerspective inv note).
Base.inv(p::FramedStereographicProjection) = throw(MethodError(inv, (p,)))

"""
    FramedEquisolidProjection{Src <: Frame{3}, Dst <: Frame{2}}
    FramedEquisolidProjection(; id::Symbol = :_)

The framed EQUISOLID (Lambert equal-area) ray-frame projection: a 3D
direction at polar angle `θ` and azimuth `ψ` maps to
`2·sin(θ/2)·(cos ψ, sin ψ)` on the view's [`EquisolidFrame`](@ref).
Well-defined on the WHOLE sphere `θ ∈ [0, π]` — the only member of the
projection family with no excluded ray; the antipode maps to radius 2
under the pole azimuth convention `ψ = 0`. The keyword constructor mints
`FramedEquisolidProjection{ViewFrame{id}, EquisolidFrame{id}}()`.

Serves `Lens(Equisolid())`
(`distortion_frame(lens) === EquisolidFrame`). Like
[`FramedPerspective`](@ref) it is a fiber collapse with no `inv`.
"""
struct FramedEquisolidProjection{Src <: Frame{3}, Dst <: Frame{2}} <:
    FramedStage{Src, Dst} end

FramedEquisolidProjection(; id::Symbol = :_) =
    FramedEquisolidProjection{ViewFrame{id}, EquisolidFrame{id}}()

@inline (::FramedEquisolidProjection)(X::StaticVector{3}) =
    _sphere_section(EquisolidFrame, X)

# Fiber collapse — no inverse (see the FramedPerspective inv note).
Base.inv(p::FramedEquisolidProjection) = throw(MethodError(inv, (p,)))

"""
    FramedOrthographicProjection{Src <: Frame{3}, Dst <: Frame{2}}
    FramedOrthographicProjection(; id::Symbol = :_)

The framed ORTHOGRAPHIC (sine-law) ray-frame projection: a 3D direction
at polar angle `θ` and azimuth `ψ` maps to `sin θ·(cos ψ, sin ψ)` on the
view's [`OrthographicFrame`](@ref) — parallel projection of the ray
sphere onto the equatorial plane.

**Honest domain: `θ ∈ [0, π/2]` only.** The frame FOLDS at the equator
(`r = sin θ` is non-injective past 90°), so backward rays (`Z < 0`) are
OUTSIDE the frame and map to NaN — they are not "large radii",
and silently aliasing them onto forward rays is exactly the bug the
honest-domain discipline exists to prevent (see
[`OrthographicFrame`](@ref)). The keyword constructor mints
`FramedOrthographicProjection{ViewFrame{id}, OrthographicFrame{id}}()`.

Serves `Lens(Orthographic())`
(`distortion_frame(lens) === OrthographicFrame`). Like
[`FramedPerspective`](@ref) it is a fiber collapse with no `inv`.
"""
struct FramedOrthographicProjection{Src <: Frame{3}, Dst <: Frame{2}} <:
    FramedStage{Src, Dst} end

FramedOrthographicProjection(; id::Symbol = :_) =
    FramedOrthographicProjection{ViewFrame{id}, OrthographicFrame{id}}()

@inline (::FramedOrthographicProjection)(X::StaticVector{3}) =
    _sphere_section(OrthographicFrame, X)

# Fiber collapse — no inverse (see the FramedPerspective inv note).
Base.inv(p::FramedOrthographicProjection) = throw(MethodError(inv, (p,)))

# -----------------------------------------------------------------------------
# distortion_frame — the frame a lens model's distortion is defined in
# -----------------------------------------------------------------------------

"""
    distortion_frame(lens::AbstractLensModel) -> Type{<:RetinalFrame} | nothing

The ray frame a lens model's distortion is DEFINED in — the `Src` frame
kind of its framed distortion stage, and the frame the stage-1 ray
projection of [`imaging_chain`](@ref) must land on. For a [`Lens`](@ref) it
is its projection's frame ([`projection_frame`](@ref)): `GnomonicFrame` for
the perspective projection, `EquidistantFrame` for the equidistant one
(Kannala–Brandt's `θ_d = θ·(1 + k₁θ² + …)` is a radial factor there), and so
on. A [`LiftLens`](@ref) is read on the gnomonic frame. `nothing` for a
parametric projection, whose frame its coefficients define — `ModelDefined`
in the [`FrameDependence`](@ref) vocabulary, which no Symbol tag can carry —
so there is no frame + distortion SPLIT; [`imaging_chain`](@ref) serves such
a lens through the fused model-carrying [`FramedLensProjection`](@ref) onto
`DistortedFrame{id}`.

Relation to the Bool capability traits: `has_distortion(lens) == false`
means the distortion is the identity in this frame, and the chain collapses
the ray frame onto [`DistortedFrame`](@ref) (no distortion stage) — see the
collapse rule there. All these frames assume a single center of projection.
"""
distortion_frame(l::Lens) = projection_frame(l.projection)
distortion_frame(::LiftLens) = GnomonicFrame

# A parametric projection (the double sphere, the unified models) lands on a
# frame its coefficients define — ModelDefined in the FrameDependence
# vocabulary, which no Symbol-only frame tag can name — so such a lens has no
# frame + distortion SPLIT. The framed chain serves it through ONE fused,
# model-carrying projection stage (`FramedLensProjection`) that lands directly
# on the view's DistortedFrame{id}; the mints that presuppose a split
# (distort_stage, the batched stage mints, ray_map) refuse with that reason.
const _MODEL_FRAME_MSG =
    "its projection's frame is defined by the projection's coefficients — " *
    "ModelDefined in the FrameDependence vocabulary, with no Symbol-only frame " *
    "tag to name it — so the lens has no frame + distortion split; use " *
    "imaging_chain's fused model-carrying projection (FramedLensProjection → " *
    "DistortedFrame), the lens callable, or undistort_normalized/backproject."

"""
    FramedLensProjection{Src <: Frame{3}, Dst <: Frame{2}, L <: AbstractLensModel}
    FramedLensProjection(lens; id::Symbol = :_)

The fused MODEL-CARRYING ray projection of a lens whose projection has no
fixed frame (a [`DoubleSphere`](@ref), [`Unified`](@ref) or
[`ExtendedUnified`](@ref) projection): `{ViewFrame{id} → DistortedFrame{id}}`
in ONE stage, `project(lens, X)` on ray representatives — the lens's whole
valid field of view, `Z ≤ 0` rays included, NaN outside it.
[`imaging_chain`](@ref) composes it with the framed K-affine:

    ViewFrame{id} ─ FramedLensProjection(lens) → DistortedFrame{id}
                  ─ calibration_stage(K)       → ImageFrame{id}

Like the other ray projections it is a fiber collapse — no `inv` (a
`MethodError` by design); the closed-form inverse is [`backproject`](@ref).
"""
struct FramedLensProjection{Src <: Frame{3}, Dst <: Frame{2}, L <: AbstractLensModel} <: FramedStage{Src, Dst}
    lens::L
end

FramedLensProjection(lens::AbstractLensModel; id::Symbol = :_) =
    FramedLensProjection{ViewFrame{id}, DistortedFrame{id}, typeof(lens)}(lens)

@inline (p::FramedLensProjection)(X::StaticVector{3}) = project(p.lens, X)

# Fiber collapse — no inverse (see the FramedPerspective inv note).
Base.inv(p::FramedLensProjection) = throw(MethodError(inv, (p,)))

# Is the lens's distortion the identity IN ITS DEFINING FRAME? (Drives the
# frame-collapse rule.) A forward lens with no distortion is the identity in
# its projection's frame — `Pinhole` and the pure fisheye projections alike,
# even though their gnomonic-frame callables (and hence `has_distortion`) are
# nontrivial; a lift never is.
_identity_in_frame(l::Lens) = l.distortion === nothing
_identity_in_frame(::LiftLens) = false

# -----------------------------------------------------------------------------
# Lens distortion stages (2D → 2D): FramedDistort / FramedUndistort
# -----------------------------------------------------------------------------

"""
    FramedDistort{Src <: Frame{2}, Dst <: Frame{2}, L <: AbstractLensModel}

Framed forward-distortion stage `{ray frame → DistortedFrame}` wrapping a
lens model. The math dispatches on the `Src` frame KIND:

  - `Src <: GnomonicFrame`: the wrapped lens callable itself — the lens
    contract already IS a map on the gnomonic frame.
  - `Src <: EquidistantFrame`: the model's radial map in θ
    (`KannalaBrandt`'s polynomial; COLMAP's thin-prism fisheye's additive terms
    evaluated on equidistant-frame coordinates, COLMAP's convention).

Mint with [`distort_stage`](@ref) (which tags the honest frame via
[`distortion_frame`](@ref)); `inv` yields the matching
[`FramedUndistort`](@ref) with the frames swapped. Direct type-parameter
construction is an unchecked expert surface, like `relabel`.
"""
struct FramedDistort{Src <: Frame{2}, Dst <: Frame{2}, L <: AbstractLensModel} <:
    FramedStage{Src, Dst}
    lens::L
end

"""
    FramedUndistort{Src <: Frame{2}, Dst <: Frame{2}, L <: AbstractLensModel}

Framed inverse-distortion stage `{DistortedFrame → ray frame}` — the
frames-swapped inverse of [`FramedDistort`](@ref), minted by
[`undistort_stage`](@ref) or `inv`. Math dispatches on the `Dst` frame
kind: `undistort_normalized(lens, ·)` toward a [`GnomonicFrame`](@ref); the
closed-form/Newton θ-inverse toward an [`EquidistantFrame`](@ref).

Feeding stages the wrong way around — e.g. composing an undistort stage
directly after another undistort stage, or after the ray projection (which
lands on the ray frame, not the distorted frame) — is a junction mismatch
and throws a `MethodError`: distort-vs-undistort DIRECTION is what the
frame tags check, view identity or not.
"""
struct FramedUndistort{Src <: Frame{2}, Dst <: Frame{2}, L <: AbstractLensModel} <:
    FramedStage{Src, Dst}
    lens::L
end

Base.inv(d::FramedDistort{Src, Dst, L}) where {Src, Dst, L} =
    FramedUndistort{Dst, Src, L}(d.lens)
Base.inv(u::FramedUndistort{Src, Dst, L}) where {Src, Dst, L} =
    FramedDistort{Dst, Src, L}(u.lens)

# A framed distortion stage's Jacobian is derived like every lens map's.
@inline transform_deriv(d::FramedDistort, x::StaticVector{2}) = _pointwise_jacobian(d, x)

# Forward math — dispatch on the Src frame kind (see FramedDistort).
@inline (d::FramedDistort{<:GnomonicFrame})(x::StaticVector{2}) = d.lens(x)
@inline (d::FramedDistort{<:EquidistantFrame})(x::StaticVector{2}) =
    _distort_in_frame(d.lens, x)

# Inverse math — dispatch on the Dst frame kind (see FramedUndistort).
@inline (u::FramedUndistort{<:DistortedFrame, <:GnomonicFrame})(xd::StaticVector{2}) =
    undistort_normalized(u.lens, xd)
@inline (u::FramedUndistort{<:DistortedFrame, <:EquidistantFrame})(xd::StaticVector{2}) =
    _undistort_in_frame(u.lens, xd)

"""
    distort_stage(lens::AbstractLensModel; id::Symbol = :_) -> FramedDistort

Mint the framed forward-distortion stage of `lens`,
`{distortion_frame(lens){id} → DistortedFrame{id}}`, with the honest frame
tag read from [`distortion_frame`](@ref). For COLMAP's thin-prism fisheye the stage's
math is its additive bundle, read in the equidistant frame (the equidistant
half is the stage-1 ray projection, not distortion). Errors for lenses with no distortion stage:
identity-in-frame models (`Pinhole`, a pure `Lens(Equidistant())` — the chain
collapses instead, see [`DistortedFrame`](@ref)) and scoped-out models
(a parametric projection, which has no frame split).
"""
function distort_stage(lens::AbstractLensModel; id::Symbol = :_)
    C = distortion_frame(lens)
    C === nothing && throw(
        ArgumentError(
            "distort_stage: " * _MODEL_FRAME_MSG
        )
    )
    _identity_in_frame(lens) && throw(
        ArgumentError(
            "distort_stage: $(typeof(lens)) is the identity in its defining " *
                "frame — there is no distortion stage; the imaging chain " *
                "collapses the ray frame onto DistortedFrame (see the collapse " *
                "rule at DistortedFrame)."
        )
    )
    return FramedDistort{C{id}, DistortedFrame{id}, typeof(lens)}(lens)
end

"""
    undistort_stage(lens::AbstractLensModel; id::Symbol = :_) -> FramedUndistort

Mint the framed inverse-distortion stage
`{DistortedFrame{id} → distortion_frame(lens){id}}` — `inv` of
[`distort_stage`](@ref). The bare [`undistort_normalized`](@ref) surface is
unchanged; this is its framed, direction-checked wrapper.
"""
undistort_stage(lens::AbstractLensModel; id::Symbol = :_) = inv(distort_stage(lens; id))

# -----------------------------------------------------------------------------
# RayMap — the lens's sphere self-map (3D → 3D, distortion free of any image frame)
# -----------------------------------------------------------------------------
#
# A central lens does not "distort 2D coordinates" — it RE-ORIENTS rays.
# The distortion polynomial of a lens model is the CONJUGATION section ∘ D ∘ section⁻¹
# of a self-map D: S² → S² of the view's spherical retina, expressed in the
# model's defining frame. RayMap exposes D itself: project a ray onto the
# defining frame (restricted to the sphere, where the projection is a
# SECTION and hence invertible), apply the forward distortion there, and
# lift back to the unit bearing vector. Both endpoints are the view's 3D
# frame — a RayMap is the frame-typed `{ViewFrame{id} → ViewFrame{id}}`
# stage, and unlike the fiber-collapsing ray projections it has an honest
# `inv` (the undistort conjugation).

# The frame ↔ sphere bridges of the framed surface: `_sphere_section` is the
# projection of a ray onto the frame (`_frame_project`, lens_rays.jl) on its
# domain `_sectionable`, and `_sphere_lift` its inverse (`_frame_backproject`) on
# `_liftable` — the lens contract's own domains. Outside them the framed stages
# answer NaN, exactly as `project`/`backproject` do: one "not imaged" encoding,
# never an alias onto an in-frame point (`_nan_vec2`/`_nan_vec3`, lens_rays.jl).

const _GeometryFrame = Union{GnomonicFrame, EquidistantFrame, StereographicFrame, EquisolidFrame, OrthographicFrame}

@inline _sphere_section(C::Type{<:_GeometryFrame}, v::StaticVector{3}) =
    _sectionable(C, v) ? _frame_project(C, v) : _nan_vec2(v)
# The equisolid frame images the whole sphere: its antipode, outside the lens
# contract's domain (a whole circle there), maps to radius 2 under the pole
# azimuth convention ψ = 0 (matching the equidistant projection's).
@inline function _sphere_section(C::Type{<:EquisolidFrame}, v::StaticVector{3})
    v[3] < 0 && !_off_axis(v) && return Vec2(2 * one(float(v[3])), zero(float(v[3])))
    return _sectionable(C, v) ? _frame_project(C, v) : _nan_vec2(v)
end

@inline _sphere_lift(C::Type{<:_GeometryFrame}, p::StaticVector{2}) =
    _liftable(C, p) ? _frame_backproject(C, p) : _nan_vec3(p)

# FrameDependence dispatch (0.14): sphere sections/lifts exist only for
# GeometryDefined frames — the tag alone determines the frame, so the
# radial profile is a fixed function. A ModelDefined frame (DistortedFrame)
# has no model-free relation to the sphere; the refusal names the fix.
_sphere_section(::Type{<:DistortedFrame}, v::StaticVector{3}) =
    throw(
    ArgumentError(
        "sphere section: DistortedFrame is ModelDefined — which frame it IS " *
            "depends on the view's lens parameters (see FrameDependence), so " *
            "no model-free sphere section exists; bring the lens " *
            "(ray_map / distort_stage are the model-carrying maps)."
    )
)
_sphere_lift(::Type{<:DistortedFrame}, p::StaticVector{2}) =
    throw(
    ArgumentError(
        "sphere lift: DistortedFrame is ModelDefined — no model-free sphere " *
            "lift exists (see FrameDependence); bring the lens " *
            "(ray_map / undistort_stage are the model-carrying maps)."
    )
)

"""
    RayMap{F <: Frame{3}, C <: RetinalFrame, L <: AbstractLensModel}

The lens's SPHERE SELF-MAP `D: S² → S²` as a framed stage
`{ViewFrame{id} → ViewFrame{id}}`: the lens's forward distortion free of
any image frame, computed as the conjugation

    D = lift ∘ distort ∘ section

where `section` is the projection onto the lens's defining frame
([`distortion_frame`](@ref)) restricted to the unit sphere and `lift` is its inverse. `C` records the
defining frame the conjugation runs through; `D` itself needs no 2D frame
tag — its input and output are rays of the view's 3D frame.

Acts on RAYS: any 3D representative may be supplied (the input norm is
irrelevant — both frame projections are scale-invariant on ray representatives), and
the output is always the UNIT bearing vector of the re-oriented ray.

Restricted to the sphere the frame projections are sections — invertible,
unlike the fiber-collapsing full projections `FramedPerspective` /
`FramedEquidistantProjection` — which is why `RayMap` has an honest `inv`
(the [`InverseRayMap`](@ref) undistort conjugation) while the projection
stages deliberately have none.

Domain: the conjugation is defined exactly where the defining frame is —
the forward hemisphere `Z > 0` for gnomonic-frame lenses, `θ ∈ [0, π)`
(all but the exact backward ray) for equidistant-frame lenses. Rays
outside the frame's domain map to NaN.

Mint with [`ray_map`](@ref); direct type-parameter construction is an
unchecked expert surface, like `relabel`.
"""
struct RayMap{F <: Frame{3}, C <: RetinalFrame, L <: AbstractLensModel} <:
    FramedStage{F, F}
    lens::L
end

"""
    InverseRayMap{F <: Frame{3}, C <: RetinalFrame, L <: AbstractLensModel}

The inverse sphere self-map `D⁻¹: S² → S²` — the undistort conjugation
`lift ∘ undistort ∘ section`, minted by `inv` of a [`RayMap`](@ref) (and
inverting back). Same frame typing, ray semantics, unit-norm output, and
frame-domain behavior as `RayMap`.
"""
struct InverseRayMap{F <: Frame{3}, C <: RetinalFrame, L <: AbstractLensModel} <:
    FramedStage{F, F}
    lens::L
end

Base.inv(m::RayMap{F, C, L}) where {F, C, L} = InverseRayMap{F, C, L}(m.lens)
Base.inv(m::InverseRayMap{F, C, L}) where {F, C, L} = RayMap{F, C, L}(m.lens)

# The identity-in-frame lenses' D is the identity on the sphere: return the
# unit representative directly (exact — no frame round-trip round-off), but
# only AFTER the section has enforced the frame's domain (a NaN section), part of
# the conjugation's honest definition. `_identity_in_frame` is a function
# of the lens TYPE, so the branch folds at compile time. Otherwise D is
# the lens contract read in the frame — lift ∘ project, and D⁻¹ =
# backproject ∘ section — so it cannot disagree with `project`.
@inline function (m::RayMap{F, C})(v::StaticVector{3}) where {F, C}
    isnan(_sphere_section(C, v)[1]) && return _nan_vec3(v)
    _identity_in_frame(m.lens) && return normalize(Vec3(v[1], v[2], v[3]))
    return _sphere_lift(C, project(m.lens, v))
end

@inline function (m::InverseRayMap{F, C})(v::StaticVector{3}) where {F, C}
    x = _sphere_section(C, v)
    isnan(x[1]) && return _nan_vec3(v)
    _identity_in_frame(m.lens) && return normalize(Vec3(v[1], v[2], v[3]))
    return backproject(m.lens, x)
end

"""
    ray_map(lens::AbstractLensModel; id::Symbol = :_) -> RayMap

Mint the lens's sphere self-map `D: S² → S²` as the framed stage
`{ViewFrame{id} → ViewFrame{id}}` — what a central lens actually does,
stated with no image frame: it RE-ORIENTS rays. The familiar 2D distortion
polynomial is the conjugation `section ∘ D ∘ section⁻¹` in the lens's defining
frame ([`distortion_frame`](@ref)); `ray_map` returns `D` itself.

Input is any 3D ray representative (norm irrelevant); output is the unit
bearing vector of the re-oriented ray. `inv` yields the undistort
conjugation ([`InverseRayMap`](@ref)) — honest, because on the sphere the
frame projections are invertible sections. Rays outside the defining
frame's domain (gnomonic: `Z ≤ 0`; equidistant: the exact backward ray)
map to NaN — see [`RayMap`](@ref).

For lenses whose distortion is the identity in their defining frame
(`Pinhole`, a pure `Lens(Equidistant())`), `D` is the identity map — still
framed, still returning the unit representative. For COLMAP's thin-prism fisheye
the conjugation runs through its equidistant frame, where its distortion is
the additive bundle.
A lens with a parametric projection has no frame split and is scoped out
(`ArgumentError`), exactly like `distort_stage`.

`D` exists only for CENTRAL lenses (single center of projection, which
every VisualGeometryCore lens model satisfies): a finite or thick pupil has
no sphere self-map, because its rays do not share a center — that regime is
ray tracing, downstream.
"""
function ray_map(lens::AbstractLensModel; id::Symbol = :_)
    C = distortion_frame(lens)
    C === nothing && throw(
        ArgumentError(
            "ray_map: " * _MODEL_FRAME_MSG
        )
    )
    return RayMap{ViewFrame{id}, C{id}, typeof(lens)}(lens)
end

# -----------------------------------------------------------------------------
# frame_transition — the framed transition map between retinal frames (0.13)
# -----------------------------------------------------------------------------

"""
    FrameTransition{Src <: RetinalFrame, Dst <: RetinalFrame}

The framed transition map between two retinal frames of the SAME view,
`{Src → Dst}` — a pure radial reparameterization of the viewing sphere,
computed once through the sphere as the conjugation

    section(Dst) ∘ lift(Src)

so every pair among the five frames ([`GnomonicFrame`](@ref),
[`EquidistantFrame`](@ref), [`StereographicFrame`](@ref),
[`EquisolidFrame`](@ref), [`OrthographicFrame`](@ref)) is served by ONE
implementation, with no per-pair closed forms to keep consistent. Azimuth
is preserved; the radial profile map is `profile_Dst(profile_Src⁻¹(r))`.

Mint with [`frame_transition`](@ref); `inv` is the reverse pair (an honest
inverse — on the sphere the frame projections are invertible sections).
As a [`FramedStage`](@ref) it composes junction-checked with the rest of
the framed surface, and being isbits it composes into a per-point map that
[`pointwise!`](@ref) runs over a batch.

Domain: the intersection of the two frames' honest domains, enforced —
points outside `Src`'s frame radius (equisolid `r > 2`, orthographic
`r > 1`) and rays outside `Dst`'s domain (gnomonic `Z ≤ 0`, orthographic
`Z < 0`, the stereographic/equidistant antipode) map to NaN at
the edge rather than aliasing.
"""
struct FrameTransition{Src <: RetinalFrame, Dst <: RetinalFrame} <:
    FramedStage{Src, Dst} end

@inline (t::FrameTransition{Src, Dst})(x::StaticVector{2}) where {Src, Dst} =
    _sphere_section(Dst, _sphere_lift(Src, x))

Base.inv(::FrameTransition{Src, Dst}) where {Src, Dst} = FrameTransition{Dst, Src}()

"""
    frame_transition(C1::Type{<:RetinalFrame{id}}, C2::Type{<:RetinalFrame{id}}) -> FrameTransition

Mint the framed transition map `{C1 → C2}` between two retinal frames of
view `id` (the id must MATCH — a transition between frames of different
views is a `MethodError`, the labeled-junction rule):

```julia
t = frame_transition(EquidistantFrame{:_}, GnomonicFrame{:_})
t(Vec2(θ, 0.0))            # == Vec2(tan(θ), 0.0)
inv(t)                     # the reverse transition {Gnomonic → Equidistant}
```

This makes a first-class named object of what production code usually
hides in an inline expression: OpenCV's `cv::fisheye::undistortPoints`
ends with the line `scale = tan(θ)/θ_d` — that line IS
`frame_transition(EquidistantFrame{id}, GnomonicFrame{id})`, the
equidistant → gnomonic frame change applied after the polynomial
θ-inverse, unnamed and unchecked in the C++. Here the same
reparameterization is a framed stage with checked junctions and honest
domain edges (see [`FrameTransition`](@ref)).
"""
frame_transition(
    ::Type{C1}, ::Type{C2}
) where {id, C1 <: RetinalFrame{id}, C2 <: RetinalFrame{id}} =
    FrameTransition{C1, C2}()

# FrameDependence dispatch (0.14): the refusal reasons, first-class. A
# ModelDefined endpoint (DistortedFrame) admits NO model-free transition —
# which frame it IS depends on the view's lens parameters, so the refusal
# names the fix (bring the lens) instead of the former implicit
# "not a retinal frame" MethodError. GeometryDefined pairs that reach this
# fallback failed the same-view rule — those keep the labeled-junction
# MethodError.
function frame_transition(::Type{C1}, ::Type{C2}) where {C1 <: Frame{2}, C2 <: Frame{2}}
    if frame_dependence(C1) isa ModelDefined || frame_dependence(C2) isa ModelDefined
        throw(
            ArgumentError(
                "frame_transition: no model-free transition exists through a " *
                    "ModelDefined frame — which frame a DistortedFrame{id} IS " *
                    "depends on the view's lens parameters (see FrameDependence); " *
                    "bring the lens: undistort_stage/distort_stage are the " *
                    "model-carrying maps."
            )
        )
    end
    throw(MethodError(frame_transition, (C1, C2)))
end

# -----------------------------------------------------------------------------
# Framed K-affine (2D → 2D): the calibration stage
# -----------------------------------------------------------------------------

"""
    FramedAffine{Src <: Frame{2}, Dst <: Frame{2}, T}

Framed 2D affine stage `x ↦ A·x + t` — the calibration stage of the imaging
chain, `{DistortedFrame{id} → ImageFrame{id}}` (or `{ray frame →
ImageFrame{id}}` under the no-distortion frame collapse; the identity-`K`
logical case is just `A = I, t = 0`). Same arithmetic as the `AffineMap` in
the legacy `imaging_transform`; the frames are the addition. `inv` swaps
the frames around the inverse affine. Mint with
[`calibration_stage`](@ref).
"""
struct FramedAffine{Src <: Frame{2}, Dst <: Frame{2}, T} <: FramedStage{Src, Dst}
    A::SMatrix{2, 2, T, 4}
    t::SVector{2, T}
end

@inline (m::FramedAffine)(x::StaticVector{2}) = m.A * x + m.t

function Base.inv(m::FramedAffine{Src, Dst, T}) where {Src, Dst, T}
    Ai = inv(m.A)
    return FramedAffine{Dst, Src, T}(Ai, -(Ai * m.t))
end

# The framed affine as a framed collineation: the same two frames, so a model
# map composes with it under the HomographyMap junction rule
# (`HomographyMap(m) ∘ HomographyMap(H)`, `HomographyMap(m) ∘ P` for a radial camera).
HomographyMap(m::FramedAffine{Src, Dst}) where {Src, Dst} = relabel(Src, Dst, HomographyMap(AffineMap(m.A, m.t)))

"""
    calibration_stage(intrinsics::AbstractIntrinsics; id::Symbol = :_) -> FramedAffine
    calibration_stage(Src::Type{<:Frame{2}}, Dst::Type{<:Frame{2}}, intrinsics)

Mint the framed calibration stage of `intrinsics`: the affine action of `K`
(`A = K[1:2, 1:2]`, `t = K[1:2, 3]` — bit-identical to the legacy
`imaging_transform`'s `AffineMap`), labeled
`{DistortedFrame{id} → ImageFrame{id}}`. The 3-argument form states the
frames explicitly — used by [`imaging_chain`](@ref) for the no-distortion
collapse, where `K` consumes the ray frame directly.
"""
function calibration_stage(
        ::Type{Src}, ::Type{Dst}, intrinsics::AbstractIntrinsics
    ) where {Src <: Frame{2}, Dst <: Frame{2}}
    a = _pixel_affine(intrinsics)
    T = eltype(a.linear)
    return FramedAffine{Src, Dst, T}(a.linear, a.translation)
end

calibration_stage(intrinsics::AbstractIntrinsics; id::Symbol = :_) =
    calibration_stage(DistortedFrame{id}, ImageFrame{id}, intrinsics)

"""
    FramedAffine(ImageFrame{id}, NormalizedImageFrame{id}, sz::Size2) -> FramedAffine
    FramedAffine(PixelFrame{O}, NormalizedImageFrame{id}, sz::Size2) -> FramedAffine

The normalization map of an image of size `sz` (pixels, plain or `px`):
`x ↦ (x − c) / s`, `c` the image centre and `s = hypot(width, height) / 2`
the half-diagonal — [`DiagonalNormalization`](@ref), framed — so the image
corners land at unit radius. The storage frame of the benchmark pools, where
one noise level and one threshold must mean the same across resolutions;
estimation normalizes by [`HartleyNormalization`](@ref VisualGeometryCore.HartleyNormalization). The centre is written in the source's
convention: `(width/2, height/2)` for an `ImageFrame` (always read
`:colmap`), moved by `EuclideanMap(PixelFrame{:colmap}, PixelFrame{O})` for
a `PixelFrame{O}`, so every convention lands on the same normalized point.
`inv` de-normalizes. The view ids must agree (a `MethodError` otherwise).
"""
FramedAffine(::Type{ImageFrame{id}}, ::Type{NormalizedImageFrame{id}}, sz::Size2) where {id} =
    _framed_diagonal(ImageFrame{id}, NormalizedImageFrame{id}, sz, SVector(0.0, 0.0))
FramedAffine(::Type{PixelFrame{O}}, ::Type{NormalizedImageFrame{id}}, sz::Size2) where {O, id} =
    _framed_diagonal(
    PixelFrame{O}, NormalizedImageFrame{id}, sz,
    EuclideanMap(PixelFrame{:colmap}, PixelFrame{O}).t
)


function _framed_diagonal(::Type{Src}, ::Type{Dst}, sz::Size2, offset) where {Src, Dst}
    w = _pixel_extent(sz.width); h = _pixel_extent(sz.height)
    c = SVector(w / 2 + offset[1], h / 2 + offset[2])
    a = AffineMap(DiagonalNormalization((w, h); center = c))
    return FramedAffine{Src, Dst, Float64}(
        SMatrix{2, 2, Float64, 4}(a.linear), SVector{2, Float64}(a.translation)
    )
end

"""
    CameraCalibrationMat(m::FramedAffine) -> CameraCalibrationMat

The intrinsics whose calibration stage is `m`, `K = [A t; 0 0 1]` — the
inverse of [`calibration_stage`](@ref). A `K` solved in the
[`NormalizedImageFrame`](@ref) comes back in pixels as
`CameraCalibrationMat(inv(norm) ∘ calibration_stage(DistortedFrame{id}, NormalizedImageFrame{id}, LogicalIntrinsics(K_norm)))`
(framed affines fuse under `∘`, the junction checked). The lens coefficients need no conversion:
they live in `K⁻¹·pixel`, which the normalization leaves where it was.
Checked as any `K` is (positive focals, upper-triangular): a rotated stage
is an `ArgumentError`.
"""
CameraCalibrationMat(m::FramedAffine) =
    CameraCalibrationMat(SMatrix{3, 3}(m.A[1, 1], m.A[2, 1], 0, m.A[1, 2], m.A[2, 2], 0, m.t[1], m.t[2], 1))


"""
    propagate(m::FramedAffine, Σ::CovMat{2})  -> CovMat{2, ..., Dst}
    propagate(m::FramedAffine, Λ::PrecMat{2}) -> PrecMat{2, ..., Dst}
    propagate(m::FramedAffine, u::Uncertain)  -> Uncertain

Checked, frame-labeled propagation through a framed 2D affine stage — the
contract of the `EuclideanMap` methods: the Jacobian is the constant `A`,
the form's frame must meet the map's `Src` under the wildcard truth table,
and the result carries `Dst`. Covariance pushes forward, `Σ ↦ A Σ Aᵀ`;
precision pulls back, `Λ ↦ A⁻ᵀ Λ A⁻¹`. Exact, the map being affine. An
`Uncertain` moves its value by `m` and its form by its law.
"""
function propagate(m::FramedAffine{Src, Dst, T}, Σ::CovMat{2, S, L, C}) where {Src, Dst, T, S, L, C}
    _check_junction(C, Src, propagate, (m, Σ))
    return _pushforward(SMatrix{2, 2, promote_type(T, S)}(m.A), Σ, Dst)
end
function propagate(m::FramedAffine{Src, Dst, T}, Λ::PrecMat{2, S, L, C}) where {Src, Dst, T, S, L, C}
    _check_junction(C, Src, propagate, (m, Λ))
    return _pullback(inv(SMatrix{2, 2, promote_type(T, S)}(m.A)), Λ, Dst)
end
propagate(m::FramedAffine, u::Uncertain{V, <:Union{CovMat{2}, PrecMat{2}}}) where {V} =
    Uncertain(m(getfield(u, :value)), propagate(m, getfield(u, :attributes)))

"""
    similarity_scale(m::FramedAffine) -> Real
    similarity_scale(a::AffineMap) -> Real

The isotropic scale `σ` of a (framed) similarity `x ↦ σ R x + t` — a Hartley
normalization's side included: lengths in
`Src` are `σ` times as long in `Dst`, so a pixel threshold `τ` is
`σ τ` after the map (and `τ / σ` back through `inv(m)`). For the
[`NormalizedImageFrame`](@ref) map `σ = 1/s`, the inverse half-diagonal.
Throws an `ArgumentError` unless `m` is a similarity (`AᵀA = σ² I` to
rounding): a sheared or anisotropic map has no single scale.
"""
similarity_scale(m::FramedAffine) = _similarity_scale(m.A)
similarity_scale(a::AffineMap) = _similarity_scale(SMatrix{2, 2}(a.linear))
function _similarity_scale(A::StaticMatrix{2, 2})
    G = A' * A
    σ² = (G[1, 1] + G[2, 2]) / 2
    tol = 64 * eps(typeof(σ²)) * σ²
    (abs(G[1, 1] - G[2, 2]) <= tol && abs(G[1, 2]) <= tol) ||
        throw(ArgumentError("similarity_scale: the map is not a similarity (AᵀA = $(G))"))
    return sqrt(σ²)
end

# -----------------------------------------------------------------------------
# FramedChain — the checked composite
# -----------------------------------------------------------------------------

"""
    FramedChain{Src <: Frame, Dst <: Frame, T <: Tuple} <: FramedStage{Src, Dst}
    FramedChain(stages::FramedStage...)

Composite of framed stages, applied FIRST-TO-LAST (`stages[1]` innermost):
a `FramedChain` consumes `src_frame(stages[1])` coordinates and produces
`dst_frame(stages[end])` coordinates, and is itself a [`FramedStage`](@ref)
— so chains compose with stages and with each other under the same checked
`∘`, flattening (no nested chains from composition).

The vararg constructor checks EVERY inner junction under the standing
wildcard truth table and throws an `ArgumentError` naming the first
mismatched junction; `∘` between stages performs the same check as a
`MethodError` (the house composition idiom). The callable unrolls at
compile time. `inv` inverts and reverses the stages (frames swap) — and is
therefore a `MethodError` for chains containing a ray projection, whose
fiber collapse has no inverse.
"""
struct FramedChain{Src <: Frame, Dst <: Frame, T <: Tuple{Vararg{FramedStage}}} <:
    FramedStage{Src, Dst}
    stages::T
end

@inline _assert_chain_junctions(::FramedStage) = nothing
@inline function _assert_chain_junctions(a::FramedStage, b::FramedStage, rest::Vararg{FramedStage})
    _junction_ok(dst_frame(a), src_frame(b)) || throw(
        ArgumentError(
            "FramedChain: junction mismatch — $(typeof(a)) produces " *
                "$(dst_frame(a)) but $(typeof(b)) consumes $(src_frame(b))."
        )
    )
    return _assert_chain_junctions(b, rest...)
end

function FramedChain(stages::Vararg{FramedStage, N}) where {N}
    N >= 1 || throw(ArgumentError("FramedChain needs at least one stage"))
    _assert_chain_junctions(stages...)
    return FramedChain{src_frame(stages[1]), dst_frame(stages[N]), typeof(stages)}(stages)
end

@generated function (c::FramedChain{Src, Dst, T})(x) where {Src, Dst, T}
    body = :x
    for i in 1:fieldcount(T)
        body = :(c.stages[$i]($body))
    end
    return body
end

function Base.inv(c::FramedChain{Src, Dst, T}) where {Src, Dst, T}
    st = reverse(map(inv, c.stages))
    return FramedChain{Dst, Src, typeof(st)}(st)
end

# -----------------------------------------------------------------------------
# Frame-checked composition — the one junction rule (`_junction_ok`), stated
# on the abstract FramedStage (chains included: FramedChain <: FramedStage).
# -----------------------------------------------------------------------------
# `A ∘ B` applies B first; the junction is B.Dst meeting A.Src; the result
# keeps (B.Src, A.Dst). The mismatch is a deliberate explicit throw:
# FramedStage <: CT.Transformation, so an absent method would silently fall
# through to CT's lazy ComposedTransformation and defeat the check.

@inline _stages(s::FramedStage) = (s,)
@inline _stages(c::FramedChain) = c.stages

@inline function _compose_framed(
        a::FramedStage, b::FramedStage, ::Type{Src}, ::Type{Dst}
    ) where {Src <: Frame, Dst <: Frame}
    st = (_stages(b)..., _stages(a)...)
    return FramedChain{Src, Dst, typeof(st)}(st)
end

# Two framed affines FUSE into one: `∘`'s truth table has checked the junction
# (every rule routes here), and the fused map keeps (b.Src, a.Dst). So
# `inv(norm) ∘ calibration_stage(K_norm)` is again a calibration stage, and
# `CameraCalibrationMat` reads K back off it.
@inline function _compose_framed(
        a::FramedAffine, b::FramedAffine, ::Type{Src}, ::Type{Dst}
    ) where {Src <: Frame, Dst <: Frame}
    T = promote_type(eltype(a.A), eltype(b.A))
    A = SMatrix{2, 2, T}(a.A)
    return FramedAffine{Src, Dst, T}(A * SMatrix{2, 2, T}(b.A), A * SVector{2, T}(b.t) + SVector{2, T}(a.t))
end

# The junction rule is `_junction_ok` (base/frames.jl); a mismatch is its
# explicit `MethodError`.
function Base.:∘(
        a::FramedStage{SrcA, DstA}, b::FramedStage{SrcB, DstB}
    ) where {SrcA <: Frame, DstA <: Frame, SrcB <: Frame, DstB <: Frame}
    _check_junction(SrcA, DstB, ∘, (a, b))
    return _compose_framed(a, b, SrcB, DstA)
end

# -----------------------------------------------------------------------------
# imaging_chain — the fully-typed composite mint
# -----------------------------------------------------------------------------

@inline _ray_projection(::Type{GnomonicFrame}, id::Symbol) = FramedPerspective(; id)
@inline _ray_projection(::Type{EquidistantFrame}, id::Symbol) =
    FramedEquidistantProjection(; id)
@inline _ray_projection(::Type{StereographicFrame}, id::Symbol) =
    FramedStereographicProjection(; id)
@inline _ray_projection(::Type{EquisolidFrame}, id::Symbol) =
    FramedEquisolidProjection(; id)
@inline _ray_projection(::Type{OrthographicFrame}, id::Symbol) =
    FramedOrthographicProjection(; id)

"""
    imaging_chain(intrinsics, lens; id::Symbol = :_) -> FramedChain
    imaging_chain(model::CameraModel; id::Symbol = :_)

Mint the fully-typed framed imaging composite
`{ViewFrame{id} → ImageFrame{id}}` — the frame-checked counterpart of
[`imaging_transform`](@ref)'s frame-free `AffineMap(K) ∘ lens`. Every inner
junction is checked at construction:

    ViewFrame{id} ─ ray-frame projection → distortion_frame(lens){id}
                  ─ distort_stage(lens)  → DistortedFrame{id}
                  ─ calibration_stage(K) → ImageFrame{id}

Stage 1 is [`FramedPerspective`](@ref) for perspective-plane lenses and
[`FramedEquidistantProjection`](@ref) for equidistant-frame
(omnidirectional) lenses — see [`distortion_frame`](@ref). When the lens is the identity in its
defining frame (`Pinhole`; a pure `Lens(Equidistant())`), the distortion stage
is omitted and `K` consumes the ray frame directly (frame collapse — see
[`DistortedFrame`](@ref)). A lens with a parametric projection is served by
one fused [`FramedLensProjection`](@ref) onto `DistortedFrame{id}`.

The `id` keyword threads the view identity; the default is the reserved
anonymous view `:_` (junctions then check by frame KIND — the
distort-vs-undistort direction check needs no view identity). Rig-derived
chains label from the rig's minted ids:

```julia
a, b = ids(rig)
chain₁ = imaging_chain(model(source(rig)); id = a)   # {ViewFrame{a} → ImageFrame{a}}
```

Numerically the chain matches `imaging_transform`: bit-identical for
perspective-plane lenses (identical operations in identical order);
equivalent to floating-point round-off for equidistant-frame lenses, whose framed
route works in the model's defining frame instead of hopping through the
gnomonic plane (`atan`/`tan` cancellation — the legacy callable's route).
"""
function imaging_chain(
        intrinsics::AbstractIntrinsics, lens::AbstractLensModel; id::Symbol = :_
    )
    C = distortion_frame(lens)
    C === nothing && return FramedChain(FramedLensProjection(lens; id), calibration_stage(intrinsics; id))
    proj = _ray_projection(C, id)
    if _identity_in_frame(lens)
        # Frame collapse: no distortion stage; K consumes the ray frame.
        return FramedChain(proj, calibration_stage(C{id}, ImageFrame{id}, intrinsics))
    end
    return FramedChain(proj, distort_stage(lens; id), calibration_stage(intrinsics; id))
end

imaging_chain(model::CameraModel; id::Symbol = :_) =
    imaging_chain(model.intrinsics, model.lens; id)
