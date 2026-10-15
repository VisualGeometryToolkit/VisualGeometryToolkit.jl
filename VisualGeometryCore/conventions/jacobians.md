## Derivatives and Jacobians

One word, `jacobian`; the suffix says what comes back. `deriv` survives only
where an upstream package owns the name.

| Form | Meaning | Example |
|---|---|---|
| `transform_deriv(f, x)`, `transform_deriv_params(f, x)` | a map's Jacobian w.r.t. input / parameters | CoordinateTransformations' API, kept |
| `<f>_with_jacobian` | `(value, J)` | `backproject_with_jacobian(camera, u)` |
| `<f>_with_jacobians` | value plus several, documented | `project_with_jacobians(lens, X) -> (u, J_X, J_p)` |
| [`constraint_with_jacobians`](@ref) | `(g, G_x, G_θ)` for an implicit constraint | `constraint_with_jacobians(E, f₁, f₂; M₁, M₂)` |
| [`constraint_jacobians`](@ref) | the Jacobians alone | `constraint_jacobians(M, v, cs)` |
| `<thing>_jacobian` | the Jacobian alone, named after what it differentiates | [`retract_jacobian`](@ref), [`action_jacobian`](@ref) |

Each docstring names its Ceres or GTSAM equivalent, and says so explicitly where
they disagree:

- `retract_jacobian(ch, v)` is `∂vec(retract(ch, v))/∂v` — Ceres'
  `Manifold::PlusJacobian`. An inverse form would be
  `local_coordinates_jacobian` (Ceres' `MinusJacobian`).
- `left_jacobian(SO3, ω)` is Sophus' `SO3::leftJacobian`. **GTSAM's
  `Rot3::ExpmapDerivative` is its transpose**, the right Jacobian, because GTSAM
  retracts by right multiplication — measured here, not assumed (left identity
  residual 5.8e-13; the right identity fails at 1.8e-6).
- [`exp_so3`](@ref)/[`log_so3`](@ref)/[`exp_se3`](@ref)/[`log_se3`](@ref) keep their names: Manifolds.jl's
  `exp(M, p, X)` is the base-point exponential while ours is the Lie-group one
  (`exp_lie(G, X)`), so `exp(SO3, ω)` would read as the base-point form with the
  point dropped.
- [`backproject`](@ref) keeps its name: GTSAM's nearest counterpart,
  `Cal3_S2::calibrate`, collides with our task verb `calibrate` (estimating the
  camera).
- Implicit constraints follow Förstner's `A = ∂g/∂θ`, `B = ∂g/∂x`, since neither
  Ceres nor GTSAM has the concept.

The stack's Levenberg–Marquardt protocol (VisualGeometryCore `lm_core.jl`) is `residual_rows!` and
`residual_jacobian_rows!`, mirroring Ceres' `CostFunction::Evaluate`. They are
deliberately NOT named `residuals!`: the RANSAC protocol's
`residuals!(r, problem, model)` is unweighted and in data units, while LM rows
are √w-weighted in chart coordinates. One name for two currencies is what the
frame rule forbids, so the currency is in the name.
