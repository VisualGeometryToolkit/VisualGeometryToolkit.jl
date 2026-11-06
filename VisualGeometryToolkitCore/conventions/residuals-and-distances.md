## Residuals and distances: the prefix is the currency, the noun is the shape

Every name in the residual and distance family states two choices, each in a
fixed place.

**The currency is the prefix.**

- `sampson_` is GEOMETRIC: the first-order distance to the model's variety in
  the data's own units, whatever noise the measurements carry.
- `whitened_` and `mahalanobis_` are divided by the measurement noise: in σ
  units, which for plain points is the pixel ("Measurement noise lives on the
  measurements", `measurement-noise.md`).

The noise comes from the measurements (plain, [`Uncertain`](@ref),
[`exact`](@ref)), never from a type switch on one name and never from a
whitening the caller supplies. One name for two currencies is what
"Derivatives and Jacobians" forbids (`jacobians.md`), and for the same reason:
for plain points the two currencies agree, so a caller who confuses them stays
green on every plain test and is wrong on the first covariance.

**The shape is the noun.** With `d_g` the constraint's codimension:

| Noun | Value | Sign |
|---|---|---|
| residual | the vector of `d_g` components | signed |
| distance | its norm `‖r‖` | `≥ 0` |
| signed distance | the residual when `d_g = 1`: one number | signed |

[`signed_distance`](@ref)`(plane, p)` is the geometric signed distance
(VisualGeometryToolkitCore 0.5.7). `distance(hyperplane, p)` is deprecated
because it returned a signed value under an unsigned name.

**A square is named as one.** `‖r‖²` is the SQUARED distance: `gᵀΣ̃⁻¹g` is the
squared Mahalanobis distance (`~ χ²(d_g)` under H₀), never "the Mahalanobis
distance". A threshold on a distance and a threshold on its square differ by a
square root, so a square under the plain name is a threshold off by a power.

**Precedent.** Statistics: a residual `eᵢ` is signed, the Mahalanobis distance
`√(rᵀΣ⁻¹r)` is `≥ 0` (R's `mahalanobis` returns the square under the plain
name, the confusion this rule avoids). Ceres: residuals are signed vectors and
the cost is `½‖r‖²`. GTSAM: `whitenedError` and `unwhitenedError` are signed
vectors and `error()` is the scalar.

The names, by currency and shape:

| Currency | Residual (signed vector) | Distance (`≥ 0`) | Signed distance (`d_g = 1`) |
|---|---|---|---|
| geometric | [`sampson_residual`](@ref), [`sampson_residuals!`](@ref) | [`sampson_distance`](@ref), [`sampson_distances!`](@ref) | [`signed_distance`](@ref) |
| whitened | [`whitened_residuals!`](@ref) (model level), [`whitened_constraint_residual`](@ref) (problem level) | [`mahalanobis_distance`](@ref) | the whitened residual, `d_g = 1` |

### The one exception: the RANSAC per-point scalar

[`residuals!`](@ref)`(r, problem, model)` and its hook
[`whitened_residual`](@ref)`(problem, ctx, i)` (with the helper
`whitened_distance(r)`) return the signed distance when `d_g = 1` and the
distance otherwise, under the word "residual". The 0.7 wave renames them:
`residuals!` becomes `mahalanobis_distances!(d, problem, model)`, the hook
`mahalanobis_distance(problem, ctx, i)` (`≥ 0`), `whitened_constraint_residual`
becomes `whitened_residual(problem, ctx, i)` (the signed vector), a new
problem-level `whitened_residuals!(r, problem, model)` stacks its rows, the
helper is removed, and `sq_residuals` / `square_residuals!` become
`sq_distances` / `square_distances!`. Scores are bit-for-bit unchanged: every
scorer reads `r²`, which the sign does not move.
