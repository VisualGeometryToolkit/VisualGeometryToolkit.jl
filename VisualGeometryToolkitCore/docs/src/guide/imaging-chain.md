# The Framed Imaging Chain

```@meta
CurrentModule = VisualGeometryToolkitCore
```

The imaging chain of a central camera, as framed stages with checked
junctions (since 0.10):

```
ViewFrame{id} ─ ray-frame projection → ray frame (Gnomonic/Equidistant/…)
              ─ lens distortion      → DistortedFrame{id}
              ─ framed K-affine      → ImageFrame{id}
```

[`imaging_chain`](@ref) mints the fully-typed composite
`{ViewFrame{id} → ImageFrame{id}}`; the frame-free
[`imaging_transform`](@ref) is the same map
without frames, `AffineMap(K) ∘ lens` with the lens applied to the ray. The
scalar lens math is never reimplemented: the framed stages wrap the lens
contract's maps.

## One retina, many frames

Every central camera has a *spherical retina* — the sphere S² of rays
through the center of projection (the manifold-level object of the
Geyer–Daniilidis unified imaging model). What differs between camera models
is the **frame** their distortion is expressed in — a 2D coordinate system
on the retina, in differential-geometry terms a chart of the sphere — so
frames, not retinas, are the type currency of the chain. The perspective
plane is not a different manifold: it is the *gnomonic* chart of the
sphere, degenerate at 90° and blind to `Z ≤ 0` rays.

The [`RetinalFrame`](@ref)`{id}` family (complete since 0.13) shares the
azimuth and differs in the radial profile, each with its own honest domain:

| frame | radial profile | domain | note |
|-------|----------------|--------|------|
| [`GnomonicFrame`](@ref) | `r = tan θ` | `θ ∈ [0, π/2)` | gnomonic; calibrated coordinates `x̂ = π(K⁻¹ũ)` |
| [`EquidistantFrame`](@ref) | `r = θ` | `θ ∈ [0, π)` | the omnidirectional models' defining frame |
| [`StereographicFrame`](@ref) | `r = 2·tan(θ/2)` | `θ ∈ [0, π)` | conformal — the only angle-preserving member |
| [`EquisolidFrame`](@ref) | `r = 2·sin(θ/2)` | `θ ∈ [0, π]` | equal-area; the only frame reaching the antipode |
| [`OrthographicFrame`](@ref) | `r = sin θ` | `θ ∈ [0, π/2]` **only** | folds at the equator |

Two more frames complete the vocabulary: [`DistortedFrame`](@ref)`{id}` —
the post-lens, pre-`K` frame forward distortion lands on (defined by the
lens map's image, deliberately *not* a `RetinalFrame`) — and
[`ImageFrame`](@ref)`{id}`, the canonical pixel frame.

## Stage 1: the ray-frame projection family

"Perspective precedes lens" is a property of perspective-plane *models*,
not a law of the chain. Stage 1 is therefore a family — one framed
projection per ray frame, all one type, [`FramedProjection`](@ref)`{Src, Dst}`,
named by their `Dst` family: [`FramedPerspective`](@ref),
[`FramedEquidistantProjection`](@ref),
[`FramedStereographicProjection`](@ref),
[`FramedEquisolidProjection`](@ref),
[`FramedOrthographicProjection`](@ref). Each is a fiber collapse (3D ray →
2D ray-frame point) and deliberately has **no `inv`** — the right-inverses are
the *sections* [`unproject`](@ref) and [`backproject`](@ref), which select
one point per fiber via a depth.

[`distortion_frame`](@ref)`(lens)` names which frame a lens model's
distortion is defined in, and hence which projection stage 1 must be: for a
[`Lens`](@ref) it is its projection's frame ([`projection_frame`](@ref)) —
`GnomonicFrame` for the perspective projection, `EquidistantFrame` for the
equidistant one (Kannala–Brandt's radial factor, COLMAP's thin-prism
fisheye), and so on; a [`LiftLens`](@ref) is read on the gnomonic frame. A
parametric projection ([`DoubleSphere`](@ref), [`Unified`](@ref),
[`ExtendedUnified`](@ref)) lands on a frame its coefficients define, which no
tag can name: `imaging_chain` serves such a lens through one fused
[`FramedLensProjection`](@ref), and the split mints refuse it
(`ArgumentError`).

## Stages 2–3: distortion and calibration

[`distort_stage`](@ref)`(lens)` mints the framed forward distortion
`{distortion_frame(lens){id} → DistortedFrame{id}}` as a
[`FramedDistort`](@ref); [`undistort_stage`](@ref) is its frames-swapped
`inv` ([`FramedUndistort`](@ref)). [`calibration_stage`](@ref)`(K)` mints
the framed affine action of the intrinsics
(`{DistortedFrame{id} → ImageFrame{id}}`) as a [`FramedAffine`](@ref).

Direction checking is the point: composing an undistort stage after
another undistort stage, or after the ray projection, is a junction
mismatch and throws a `MethodError` — distort-vs-undistort *direction* is
what the frame tags check, view identity or not.

## The collapse rule

When a lens's distortion is the identity in its defining frame (`Pinhole`
on the gnomonic `GnomonicFrame`; the pure-projection fisheyes on theirs),
there is no distortion stage: the chain *collapses* the ray frame onto the
calibration stage — `K` consumes the ray frame directly — rather than
inserting a manufactured identity stage. `distort_stage` errors for such
lenses; the collapse is applied by `imaging_chain` automatically.

## The anonymous view id

The id axis is optional. The reserved symbol `:_` is the designated
anonymous view identity and every stage mint defaults to it — frame *kind*
then still does the junction checking, so a lone `CameraModel` with no view
identity gets direction-checked stages. Rig-derived chains thread real
identities:

```julia
a, b = ids(rig)
chain₁ = imaging_chain(model(source(rig)); id = a)   # {ViewFrame{a} → ImageFrame{a}}
```

`:_` is a real label, not a wildcard — the [`UnlabeledFrame`](@ref) rules
are unchanged.

## Composition and `FramedChain`

`∘` between framed stages is junction-checked under the standing wildcard
truth table (same five-method statement as `EuclideanMap`) and produces a
flat [`FramedChain`](@ref), whose callable unrolls at compile time. The
vararg `FramedChain(stages...)` constructor checks every inner junction and
names the first mismatch. `inv` inverts and reverses the stages — and is
therefore a `MethodError` for chains containing a ray projection.

Numerically, `imaging_chain` matches `imaging_transform`: bit-identical for
perspective-plane lenses, equal to round-off for the other frames. Both work
in the model's defining frame; neither hops through the gnomonic plane.

## See also

- [Ray Maps and Frame Transitions](sphere.md) — the sphere self-map behind the in-frame distortion polynomial
- [Back-projection and Undistortion](undistortion.md) — the lens contract and inverting the distortion stage
- [Coordinate Frames](frames.md) — the tag system
- [Imaging Chain and Stereo API](../api/imaging.md)
