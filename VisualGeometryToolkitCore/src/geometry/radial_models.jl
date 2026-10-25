# =============================================================================
# Radially symmetric plane→image models — the family, and what it does NOT
# promise
# =============================================================================
#
# WHERE THIS SITS IN THE HIERARCHY.  VisualGeometryToolkitCore keeps three layers
# apart so that composition stays checkable:
#
#   1. ALGEBRAIC QUANTITIES — `AbstractHomMat <: StaticMatrix{3,3}` and the
#      affine/similarity/euclidean/calibration chain under it.  Matrices.  They
#      compose by multiplication, and the subtype chain IS the composition
#      algebra.  No direction, no frame, no cached inverse.
#   2. MAPPINGS — `HomographyMap` and the lenses: `Transformation`s carrying a
#      forward AND an inverse, plus `Src`/`Dst` frame parameters, so a bad
#      composition under `∘` is a type error.
#   3. CAMERA MODELS — unposed `AbstractCamera`s; `CameraModel` = intrinsics + lens;
#      `Camera` = model + extrinsics.  Domain is a 3D point.
#
# The radial models belong to NONE of those, and forcing them into one would be
# the mistake.  They are 2D→2D relations between a PLANE and an IMAGE whose
# image side is organized as a pencil of rays about a known centre:
#
#   RadialHomography  H (3×3) + k + center   plane point → PIXEL, invertible;
#                                            `project`/`backproject` are the
#                                            IMAGE-side division lens, `m.H` is
#                                            the plane part.
#   RadialCamera      P (2×3) + center       plane point → LINE through the
#                                            centre.  Many-to-one: the radius is
#                                            deliberately quotiented out, which
#                                            is what makes it invariant to f and
#                                            to ANY radially symmetric lens.
#
# So `project` is NOT part of the family contract — the two members disagree on
# its arity, and that disagreement is the whole point of having both.  What the
# family DOES promise is the one thing that makes either of them a locus rather
# than a bare direction:
#
#   center(m) -> Point2      the centre of radial symmetry
#
# plus the frame law: a member is a MAP (plane point → pixel, plane point →
# line), so it changes frame by `∘` with a plane-side and an image-side map,
# like every map: with x̂ = S·x and ŷ = D·y the model written in the new
# coordinates is D ∘ m ∘ S⁻¹ (for the plane part Ĥ = D·H·S⁻¹) — the same
# relation in other coordinates, not a different one. Each member states
# which image-side maps keep it in the family (a similarity for the division
# model, an affine for the pencil).

"""
    AbstractRadialModel

Plane→image models whose image side is a pencil of rays about a known centre:
[`RadialHomography`](@ref) (plane point → pixel, the division lens carried
explicitly) and [`RadialCamera`](@ref) (plane point → the image LINE the pixel
lies on, invariant to focal length and to any radially symmetric distortion).

The family contract is deliberately narrow — [`center`](@ref), and movement
between coordinate frames by composition (`D ∘ m ∘ inv(S)` with an image-side
`D` and a plane-side `S`). `project` is NOT part of it: the members differ in
arity there, which is exactly why both exist.
"""
abstract type AbstractRadialModel end

"""
    center(m::AbstractRadialModel) -> Point2

The model's centre of radial symmetry — the pencil's vertex, and the principal
point when the model is a calibration one. A radial model's matrix predicts a
DIRECTION; the centre is what turns that direction into a locus in the image,
which is why every member carries one.
"""
center(m::AbstractRadialModel) = m.center

"""
    direction(m, X) -> Vec2

The bare algebraic prediction of a radial model: the DIRECTION of the image
ray a plane point maps to, before the centre turns it into a locus. Defined
for [`RadialCamera`](@ref) in cameras/calibration.jl, where it is `P·X̃`.

Declared here so the family owns the name; the method lives with the model.
"""
function direction end

# The ONE scaling law of a radial coefficient: under the image rescaling whose
# lens-side factor is `s` (radius r ↦ r/s), the coefficient of r^(2j) becomes
# `c·s^(2j)`. The division lens's `k′ = k/s²` (a pixel→normalized map of scale
# `s` supplies `1/s`) and `_rescale_lens`' radial coefficients both read it.
@inline rescale_radial_coeff(c, s, j::Integer) = c * s^(2j)
