# Stereo Rigs and Uncertainty

Two 0.9 changes share this page because they share a mechanism: the
[`StereoRig`](@ref) became frames-native, and the statistical forms
(`CovMat`/`PrecMat`) became frame-labeled — both so that a direction (or a
transformation law) that used to live in a docstring is now checked by
dispatch.

## The frames-native `StereoRig`

A [`StereoRig`](@ref) is the 2-sensor case of the general [`Rig`](@ref)
— a frame with posed members, whose frame is the gauge member's own. A
sensor rig stores ONLY a tuple of cameras, each expressed rig-relative
(reference frame = view 1, held at the identity); a rig of calibration
targets is the same type with dict storage, [`PosedTarget`](@ref) members (a target
model — [`PlanarTarget`](@ref), [`PointTarget`](@ref) — and its pose) gauged on
a board, and a camera rig carrying its marker constellation holds both. The
pair's two cameras are the method-form accessors [`source`](@ref) and
[`destination`](@ref) (no stored fields), and the relative rigid transform is
*derived* from the stored per-camera labeled maps:

```
extrinsics(rig) :: EuclideanMap{3, …, ViewFrame{id₁}, ViewFrame{id₂}}
```

(monogram `X_{V₂V₁}`: consumes source-view coordinates, produces
destination-view coordinates, i.e. `X₂ = R·X₁ + t`). There is no cached inverse
and no stored id tuple — the labels are the single source of truth:

- [`pose`](@ref pose(::StereoRig))`(rig)` is the on-demand
  `inv(extrinsics(rig))` (the typed algebra swaps the frames to `X_{V₁V₂}`).
- [`ids`](@ref)`(rig)` reads the view-identity symbols off the stored map's
  frame parameters — an accessor, not a field, resolved at compile time.
- [`view_frames`](@ref)`(rig)` / [`image_frames`](@ref)`(rig)` expose the
  minted per-view `ViewFrame{id}` / `ImageFrame{id}` tags, ready to attach
  at a lift.

```julia
rig = StereoRig(left_cam, right_cam; ids = (:left, :right))
ids(rig)                      # (:left, :right)
pose(rig)                     # ViewFrame{:right} → ViewFrame{:left}
```

The 3-argument constructor validates a caller-supplied transform against
`extrinsics(destination) ∘ pose(source)` computed from the given cameras
(`check = false` for synthetic rigs whose cameras are placeholders).

## Minting labeled epipolar forms

The rig mints the slot-labeled forms of the [forms layer](forms.md) in one
call:

- [`essential_form`](@ref)`(rig)` — the calibrated form `E = [t]×R`, slots
  labeled by the per-view [`GnomonicFrame`](@ref)`{id}` tags (the frame on
  which `EssentialMat`-backed forms are honest: gnomonic coordinates
  `x̂ = π(K⁻¹ũ)`).
- [`EpipolarForm`](@ref)`(rig)` — the pixel-frame fundamental form:
  `essential_form(rig)`'s correlation composed with the
  calibrated → pixel lifts
  `HomographyMap{GnomonicFrame{id}, ImageFrame{id}}(Kᵢ)`, computing
  `F = K₂⁻ᵀ E K₁⁻¹` with every junction frame-checked.

Depth-driven utilities complete the two-view toolkit:
[`depth_transfer`](@ref) (unit-depth point transfer — a transfer, not a
correlation), [`transfer`](@ref) with depth-attributed points, and the
ground-truth machinery [`validate_gt_metric`](@ref), [`validate_gt_inliers`](@ref) that validates
correspondences against a rig plus per-view depth maps with a metric
(physical-length) tolerance.

## Frame-labeled statistical forms

[`CovMat`](@ref)`{N, T, L, F}` and [`PrecMat`](@ref)`{N, T, L, F}`
carry a trailing `F <: Frame{N}` parameter naming the frame their
coordinates live in — defaulting to `UnlabeledFrame{N}`, gradual as always.
The point of the labels is the **law pair**:

- covariance is a (2,0) tensor and *pushes forward*: `Σ ↦ J Σ Jᵀ`
- precision is a (0,2) bilinear form and *pulls back*: `Λ ↦ J⁻ᵀ Λ J⁻¹`

Applying the covariance law to a precision matrix (or vice versa) produces
a plausible-looking, silently wrong matrix — this pair is exactly the bug
surface the frame labels exist to check. With the types in place, the law
is dispatched, not remembered.

## Checked propagation through framed maps

The raw-Jacobian layer [`propagate`](@ref)`(J, Σ) = J Σ Jᵀ` stays
frame-free (a bare Jacobian carries no frames). The checked layer sits on
top: pushing a statistical form through a *framed* map checks the form's
frame against the map's `Src` (standing wildcard truth table) and stamps
the result with the map's `Dst`:

```julia
propagate(m::EuclideanMap, Σ::CovMat)     # J = R, pushforward, result labeled Dst
propagate(m::EuclideanMap, Λ::PrecMat)    # J = R, pullback, result labeled Dst
propagate(m::EuclideanMap, u::Uncertain)  # value and uncertainty in one call
propagate(h::HomographyMap, u::Uncertain) # point-dependent Jacobian at u.value
```

For a rigid map the two conjugations coincide numerically (`R` is
orthogonal); the code follows the honest law per type so
`propagate(m, inv(Σ)) == inv(propagate(m, Σ))` holds by construction. The
`HomographyMap` overloads take the measurement (`Uncertain`) rather than a
bare form because the projective point action is nonlinear — the Jacobian
must be evaluated at the point.

`inv`, `cov`, and `prec` preserve the frame (inversion swaps the
*representation*, not the frame); [`relabel`](@ref)/[`unlabel`](@ref) are
the escape hatches, one tag rather than two.

A batched counterpart pushes frame-labeled `CovMat` *stacks* through a
stage's per-sample Jacobian stack — see
[Batched Maps](batched.md#Batched-propagation).

## See also

- [Epipolar Forms](forms.md) — the slot-labeled forms the rig mints
- [Coordinate Frames](frames.md) — view-identity frames
- [Imaging Chain and Stereo API](../api/imaging.md)
