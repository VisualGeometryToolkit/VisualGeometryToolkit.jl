# Epipolar Forms

*Transformations have endpoints, forms have slots, rank lives on the
manifold.* The forms layer (since 0.8) applies the frame system to the
epipolar matrix — an object that is not a map at all.

## Maps vs. forms

A `HomographyMap{Src, Dst}` is a collineation: it consumes `Src`-frame
points and produces `Dst`-frame points, so its frame tags name the two
*endpoints* of a map. The epipolar matrix is a different citizen — a
bilinear form with two contravariant point arguments,

```
x₂ᵀ M x₁ = 0
```

It maps neither argument anywhere; it *constrains* the pair. Its frame tags
therefore label the form's two argument **slots** (in the tensor sense: the
tag names which frame's points an argument position accepts, not an
endpoint of a map).

## `EpipolarForm`

[`EpipolarForm`](@ref)`{S1, S2}` wraps a frame-free [`FundMat`](@ref) or
[`EssentialMat`](@ref) and adds only the slot labels — rank and manifold
constraints stay with the wrapped matrix type and the solvers that produce
it. The slot convention, pinned at the struct and deferred to by every
other method: slot 1 takes points of frame `S1` (view 1) and is the *right*
argument of the matrix, matching the `u₂ᵀ F u₁` convention of the solver
layer.

```julia
form = EpipolarForm{ImageFrame{:a}, ImageFrame{:b}}(F)   # labeled lift
form = EpipolarForm(F)                                 # unlabeled (gradual)
form(x₁, x₂)                                           # x̃₂ᵀ M x̃₁ — 0 for an exact cspond
```

An `EpipolarForm` is not a `Transformation`: it does not compose with `∘`,
and it has **no `inv`** (deliberately — below). Its point ↦ line action is
[`epipolar_map`](@ref), an [`EpipolarMap`](@ref) — and a frame change is
composition on that map.

## Composition — the transformation law

Frame changes act on forms per slot. With slot maps
`slot1 :: HomographyMap{S1, S1′}` and `slot2 :: HomographyMap{S2, S2′}`,
demanding invariance of the constraint gives

```
M′ = H₂⁻ᵀ · M · H₁⁻¹
```

`EpipolarForm(slot2 ∘ epipolar_map(f) ∘ inv(slot1))` is exactly this: a
collineation on the right moves the points fed in, one on the left moves the
returned line by the line law `ℓ′ = H⁻ᵀℓ`. Each junction — a slot frame
meeting a map — is checked under the standing wildcard truth table (see
[Coordinate Frames](frames.md#The-wildcard-truth-table)); a slot not composed
keeps its frame and coordinates.

This is where K-normalization bookkeeping becomes checkable: with the
calibrated → pixel collineations
`Hᵢ = HomographyMap{GnomonicFrame{idᵢ}, ImageFrame{idᵢ}}(Kᵢ)`, composing an
`EssentialMat`-backed correlation computes exactly `F = K₂⁻ᵀ E K₁⁻¹`, with the
frame labels verifying that each `K` met the slot it was meant for. The
result always wraps a `FundMat` when a slot moves: an arbitrary
collineation preserves rank 2 but not the essential-manifold constraint, so
keeping an `EssentialMat` wrapper through a general frame change would be a
silent lie. Rewrap via `EssentialMat(F, K₁, K₂)` at a trusted boundary.

## `transpose`, and the deliberate absence of `inv`

The epipolar correlation is rank 2: its kernel is the epipole `e₁`
(`M e₁ = 0`), every point on one epipolar line maps to the *same* line in
the other view, and the image is not all lines but the one-parameter pencil
through the other epipole. A 2-parameter point domain collapsing onto a
1-parameter family of lines has no inverse — `inv(::EpipolarForm)` is
deliberately left undefined (a `MethodError`, pinned by test).

[`transpose`](@ref transpose(::EpipolarForm)) (swap the slots; wraps `Mᵀ`)
is the only involution the form supports; `adjoint` (`f'`) is the same
operation. The degenerate directions are exposed by [`epipoles`](@ref).

## `epipolar_map` — the honest correlation

[`epipolar_map`](@ref)`(f)` returns the typed callable
[`EpipolarMap`](@ref): point `x₁` in frame `S1` to its epipolar *line*
`ℓ₂ = M x̃₁` in frame `S2`, as a `Line2D` — a correlation lands on a line
(itself a one-slot form on `S2`-points), not on a point.
`epipolar_map(transpose(f))` goes the other way. There is no `transfer`
overload for a form: `transfer` is the point ↦ point depth-transport verb;
the depth-assumed two-view closure is [`depth_transfer`](@ref) (which
`epipolar_map` was renamed *from* in 0.8 — the old name promised a
correlation and delivered a transfer).

## Checked measurement frames

[`sampson_distance`](@ref)`(f, x₁, x₂)` delegates to the Sampson machinery
on the wrapped matrix, and since 0.9 the delegation is *checked* where a
frame is available: an `Uncertain` measurement whose `CovMat`/`PrecMat`
carries a frame label must meet the slot it feeds — same wildcard table,
`MethodError` on a labeled mismatch.

## See also

- [Stereo Rigs and Uncertainty](stereo.md) — `essential_form(rig)` / `EpipolarForm(rig)` mint the labeled forms
- [Coordinate Frames](frames.md) — the tag system the slots reuse
- [Solvers API](../api/solvers.md), [Epipolar Forms API](../api/epipolar_forms.md)
