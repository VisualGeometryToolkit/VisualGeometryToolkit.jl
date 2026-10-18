# Coordinate Frames

```@meta
CurrentModule = VisualGeometryToolkitCore
```

VisualGeometryToolkitCore records *which* coordinate frame a map consumes and
produces at the type level (since 0.7). A frame tag is a singleton type —
zero fields, zero runtime cost — carried by `EuclideanMap` and
`HomographyMap` as trailing `Src`/`Dst` type parameters. Composition and
inversion are then frame-checked by dispatch: the pose-vs-extrinsics
direction becomes a dispatch fact instead of a docstring convention.

## The `Frame{N}` hierarchy

[`Frame{N}`](@ref Frame) is the abstract supertype of frame tags of
dimension `N`. The hierarchy is **open**: downstream packages declare their
own frames

```julia
struct LidarFrame <: Frame{3} end
```

and participate in frame-checked composition with no changes to
VisualGeometryToolkitCore. The tags shipped here:

| tag | dim | meaning |
|-----|-----|---------|
| [`UnlabeledFrame`](@ref)`{N}` | N | "no frame recorded" — the gradual-typing wildcard |
| [`WorldFrame`](@ref) | 3 | the world (scene) frame |
| [`CameraFrame`](@ref) | 3 | the camera-centered frame (+Z along the optical axis) |
| [`ViewFrame`](@ref)`{id}` | 3 | the camera frame of a *specific* view (since 0.8) |
| [`TargetFrame`](@ref)`{id}` | 3 | the body frame of a *specific* calibration target |
| [`BodyFrame`](@ref)`{id}` | 3 | the frame of a *specific* rigid body (a tracked or mounted rig) |
| [`TargetPlaneFrame`](@ref) | 2 | calibration-target plane frame |
| [`PixelFrame`](@ref)`{Origin}` | 2 | pixel frame tagged by origin convention (`:colmap`, `:julia`, `:opencv`) |
| [`ImageFrame`](@ref)`{id}` | 2 | canonical pixel frame of view `id` |
| [`RetinalFrame`](@ref)`{id}` and subtypes | 2 | ray frames of the viewing sphere (see [The Framed Imaging Chain](imaging-chain.md)) |
| [`DistortedFrame`](@ref)`{id}` | 2 | post-lens, pre-calibration frame |

## The monogram bridge

The Drake monogram symbol `X_AB` — "pose of frame B expressed in frame A",
which maps B-coordinates to A-coordinates — corresponds to
`EuclideanMap{..., Src = B, Dst = A}`. This bridge is stated once and used
throughout: a camera pose is `X_WC` (`Src = CameraFrame`,
`Dst = WorldFrame`), camera extrinsics are `X_CW`, and a two-view relative
transform is `X_{V₂V₁}` (`Src = ViewFrame{id₁}`, `Dst = ViewFrame{id₂}`).

The typed algebra follows: `inv` swaps the frame tags (`X_AB⁻¹ = X_BA`),
so `inv(::CameraPose)` *is* a `CameraExtrinsics` — the frame-pair aliases
[`CameraPose`](@ref), [`CameraExtrinsics`](@ref), and the concrete solver
currency [`Extrinsics3`](@ref) are plain `EuclideanMap`s under labeled
frames, not wrappers.

## The wildcard truth table

`A ∘ B` applies `B` first; the *junction* is `B`'s `Dst` meeting `A`'s
`Src`. The result always keeps `B`'s `Src` and `A`'s `Dst`.
[`UnlabeledFrame`](@ref) is the composition wildcard:

| `B.Dst` | `A.Src` | junction | result |
|---------|---------|----------|--------|
| `F` (labeled) | `F` (same) | checked | composes |
| `F` (labeled) | `G ≠ F` | rejected | `MethodError` |
| any | `UnlabeledFrame` | unchecked | composes |
| `UnlabeledFrame` | any | unchecked | composes |

The same table governs every frame-checked junction in the package:
`EuclideanMap`/`HomographyMap` composition, the imaging-chain stages
([`FramedStage`](@ref)),
the slots of an [`EpipolarMap`](@ref) under `∘`, and frame-labeled [`propagate`](@ref). A
labeled-labeled mismatch is always an explicit `MethodError` — never a
silent fall-through to an unchecked composition.

This rule is deliberately *anti-viral*: every constructor that does not
state frames produces `Src = Dst = UnlabeledFrame{N}`, so pre-frame code
compiles and behaves unchanged, and labels never force themselves onto code
that does not opt in.

## Escape hatches: `relabel` and `unlabel`

[`relabel`](@ref) replaces the frame tags of a map (zero-cost, unchecked —
a deliberate override for trust boundaries such as dataset loaders and
solver outputs); [`unlabel`](@ref) erases them, making the map compose with
anything under the wildcard rule. Both verbs are house-wide: they apply to
maps, epipolar forms (slot tags), statistical forms (`CovMat`/`PrecMat`
frame tags, one tag rather than two), and batched stages.

```julia
E = EuclideanMap(R, t)                          # UnlabeledFrame both ends
X_CW = relabel(WorldFrame, CameraFrame, E)      # now checked as extrinsics
X_WC = inv(X_CW)                                # frames swap: a pose
unlabel(X_WC)                                   # back to the wildcard
```

## View-identity symbol frames

Since 0.8, `Symbol` type parameters distinguish *instances*, not just
conventions: `ViewFrame{:left}` is a different frame than
`ViewFrame{:right}`, so a labeled junction between two cameras' frames is a
`MethodError` instead of a silent `CameraFrame` collision. The 2D
counterpart [`ImageFrame`](@ref)`{id}` tags which view an image point
belongs to; origin conventions remain the orthogonal
[`PixelFrame`](@ref)`{Origin}` axis (convert origins first, then label the
view — an `ImageFrame{id}` point is always read in the canonical `:colmap`
convention).

[`StereoRig`](@ref) mints per-view identities on request (see
[Stereo Rigs and Uncertainty](stereo.md)); the reserved symbol `:_` is the
designated *anonymous* view identity used by the imaging-chain mints when
no rig is in play — a real label (junctions still check by frame kind), not
a wildcard.

## Pixel-origin conversion

Origin conversion is a labeled rigid map between two pixel frames, built by
the constructor-as-factory on the frame types:

```julia
T = EuclideanMap(PixelFrame{:colmap}, PixelFrame{:julia})
T.t == Vec2(0.5, 0.5)          # true
pt_julia = T(pt_colmap)
```

This is the only origin-conversion API; it composes (frame-checked) with
other labeled 2D maps and fuses in `image_warp` chains like any
`EuclideanMap{2}`.

## See also

- [Epipolar Forms](forms.md) — frame tags on the *slots* of a bilinear form
- [Stereo Rigs and Uncertainty](stereo.md) — frames-native rigs, frame-labeled covariance
- [The Framed Imaging Chain](imaging-chain.md) — the frame-to-frame stages of one camera
- [Frames API](../api/frames.md)
