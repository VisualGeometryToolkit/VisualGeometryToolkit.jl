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
(VisualGeometryToolkitCore 0.5.7). `distance(hyperplane, p)` was removed in 0.7
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
| whitened | [`whitened_residuals!`](@ref) (model level `(r, M, xs)` and problem level `(r, problem, model)`), [`whitened_residual`](@ref)`(problem, ctx, i)` (one point) | [`mahalanobis_distance`](@ref) (model level `(M, x…)` and problem level `(problem, ctx, i)`), [`mahalanobis_distances!`](@ref)`(d, problem, model)` | the whitened residual, `d_g = 1` |

### The RANSAC protocol

The constraint-problem protocol (VisualGeometryToolkitCore's `robust/problem.jl`)
follows the same table. What a scorer reads per point is a DISTANCE:
[`mahalanobis_distances!`](@ref)`(d, problem, model)` fills `d[i] ≥ 0` from
the per-point hook [`mahalanobis_distance`](@ref)`(problem, ctx, i)`, `ctx`
the problem's [`residual_context`](@ref)`(problem, model)` (the model by
default). The signed residual is [`whitened_residual`](@ref)`(problem, ctx,
i)`, in the constraint's shape (a number at `d_g = 1`, its `d_g` components
otherwise), and [`whitened_residuals!`](@ref)`(r, problem, model)` stacks its
rows. The squares are [`sq_distances`](@ref) and [`square_distances!`](@ref).
Scores read `d²`, which the residual's sign does not move. Until the 0.7 wave
the per-point scalar was a signed distance at `d_g = 1` and a distance
otherwise, under the word "residual"; a `whitened_residual` override written
then returns a scalar at `d_g ≥ 2`, and
[`test_constraint_problem`](@ref) refuses its shape.
