# Back-projection and Undistortion

```@meta
CurrentModule = VisualGeometryToolkitCore
```

Every lens answers two questions, and only two: which image point a ray lands
on, and which ray is behind an image point. [`project`](@ref)`(lens, X)` and
[`backproject`](@ref)`(lens, x_d)` are that contract, on rays — so a fisheye
that images 190° answers for every ray it images, past 90° included.
Everything else is derived from the pair, for every lens, with no per-lens
code.

## The contract

- `project(lens, X::StaticVector{3}) -> Vec2` — the normalized image point of
  the ray `X` (any positive multiple of it).
- `backproject(lens, x_d::StaticVector{2}) -> Vec3` — the UNIT ray behind the
  normalized image point `x_d`.
- Both return **NaN** where the lens does not image (past a fold, outside a
  model's field of view), never throw, never allocate, and are generic in the
  number type: `Float32` for GPU kernels, ForwardDiff `Dual`s for Jacobians in
  the point and in the lens parameters.
- One domain is the caller's choice: a [`DoubleSphere`](@ref) lens takes
  `backproject(lens, x_d; fov_bound = false)`, the closed-form lift without the
  paper's field-of-view bound (NaN only outside the algebraic domain). Past the
  bound the ray need not round-trip — `project` keeps the bound — and need not
  be the point's only preimage; the default keeps the bound.

A [`Lens`](@ref) — a projection plus a distortion in the projection's frame —
gets both maps from one generic body; a [`LiftLens`](@ref) (the division
family, the rational lift, Scaramuzza's polynomial) lifts the image point to
its ray in closed form and projects by a root on the ray. See
[Lens Families](lenses.md).

## Derived for every lens

- **Perspective views.** [`distort_normalized`](@ref)`(lens, x) =
  project(lens, (x, 1))` — also the functor `lens(x)` — and
  [`undistort_normalized`](@ref)`(lens, x_d)`, the perspective point of
  `backproject(lens, x_d)`: NaN where that ray has `Z ≤ 0`, which no
  perspective image contains. `inv(lens)` is an [`InverseLens`](@ref) whose
  call is `undistort_normalized`.
- **Jacobians.** `transform_deriv(lens, X)`, `transform_deriv_params(lens, X)`
  and `project_with_jacobians(lens, X)` on rays; `transform_deriv(lens, x)`
  and `transform_deriv(inv(lens), x_d)` on the perspective views — ForwardDiff
  through the maps, exact at the solved roots.
- **Radial profile.** [`projected_radius`](@ref)`(lens, θ)` and
  [`view_angle`](@ref)`(lens, ρ)`.
- **Cameras.** `project(model, X)`, `backproject(model, u)`,
  [`unproject`](@ref)`(model, u, depth)` (Z-depth: NaN where the ray has
  `Z ≤ 0`; a point at distance `d` along any ray is `d · backproject(model, u)`),
  and `imaging_transform(model) = AffineMap(K) ∘ lens`, the lens applied to
  the ray.

## Inside the maps

The inverse of an in-plane distortion is a solve, matched to the model's
structure and private to the maps:

| distortion | solve |
|------|-------|
| a lone [`Radial`](@ref) factor (Kannala–Brandt, COLMAP `RADIAL`, …), any frame | Newton on the ratio `g = ρ/ρ_d` — no square root, exact on the axis — safeguarded on the factor's branch |
| an [`Additive`](@ref) led by a `Radial` (OpenCV's models, COLMAP's thin-prism fisheye) | OpenCV's fixed point `x ← (x_d − ΣΔ(x))/F(r²)`, finished by trust-region Newton where it stops contracting |
| an `Additive` of deltas, a lone delta | Picard peel `x ← x_d − ΣΔ(x)`, trust-region Newton if it does not certify |
| [`FieldOfView`](@ref), [`Tilt`](@ref) | closed form |
| a [`Chain`](@ref), anything else | trust-region 2-D Newton |
| a [`LiftLens`](@ref) | `project`: the root of `ρ·N·z − r·D` on the ray, nothing divided by `z`; `backproject`: closed form |
| the parametric projections | closed forms both ways, with the papers' validity bounds |

Every solve stops by its convergence order — a Newton step at the solve
tolerance of the iterate's scale, a contraction's update at rounding level —
under a trip cap, and is certified (residual, and the branch where it
matters); a solve that does not certify is NaN. For `Dual` inputs the
derivative comes from one Newton step at the converged primal root (the
implicit function theorem), so first derivatives are exact.

## Batches

A batch of points goes through the same per-point maps by broadcasting:
[`project!`](@ref)`(u, lens, X)`, [`backproject!`](@ref),
[`distort_normalized!`](@ref) and [`undistort_normalized!`](@ref) are
[`pointwise!`](@ref) of `project`, `backproject`, …: a SIMD loop on a host
array (split across threads above a measured length, or as the `exec` policy
says) and one compiled kernel on a GPU array. A camera is a per-point map
too: `project!(u, camera, X)`. A point the lens does not image comes back NaN.
See [Batches](batched.md).

## Adding a lens model

A new term or projection is a leaf: its coefficients as a tuple, how to rebuild
it from one, their names, its record tag, and its map — a term's `_distort` (or
`_delta`, with `isdelta`), a projection's `_project_ray` / `_lift_point`.
Everything above follows. A lens of a new kind implements `project(lens, X)`
and `backproject(lens, x_d)` directly.

## See also

- [Lens Families](lenses.md) — the lens kinds, projections, terms and records
- [Batches](batched.md) — one per-point function, every backend
- [The Framed Imaging Chain](imaging-chain.md) — the framed, direction-checked stage wrappers
- [Lenses API](../api/lenses.md)
