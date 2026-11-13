# PoseLib changelog

## 0.3.0 (unreleased)

The breaking wave 10, on VisualGeometryToolkitCore 0.10.

### Breaking

- **On VisualGeometryToolkitCore 0.10** (`[compat]` "0.10.0"): the refiners take Core's `Correspondence`/`Correspondences`
  (the terminology ruling: the old pairing abbreviation is retired with every name built on it).
- **Below the minimal sample a refiner returns `InitialFailure`**: fewer correspondences than the model's minimal
  sample (4 for `refine_homography`, 7 for `refine_fundamental`, 5 for `refine_relpose`, 3 for
  `refine_absolute_pose`) return the seed unrefined, `(InitialFailure, NoRule)` after 0 iterations and `n = 0`, as
  VisualGeometryToolkitEstimation's refines do; before, PoseLib was called and the result read `(Success, NoRule)`.
  `n` is documented: the correspondences handed over (PoseLib reports no weights).
- **The retired keywords `loss`, `loss_width` and `local_optimization` are refused by VisualGeometryToolkitCore's
  `refuse_retired_keyword`** (its message: "the keyword `loss` is retired: give `optimization_strategy`"; `loss_width`
  names `width`). A keyword the refiners never had is Julia's `MethodError`, no longer an `ArgumentError`.

## 0.2.1 (2026-11-13)

- **Julia 1.13** (the owner's ruling of 2026-11-13: the whole stack requires 1.13): `[compat] julia = "1.13"` in the
  package and in every environment of its workspace; CI runs 1.13 only.
- **Every `[compat]` floor raised to the gated resolution** (the owner's ruling of 2026-11-13): a stack dependency at
  its registry latest or its in-wave version, a third-party one at the version the release gate resolved (a floor
  only, the semver upper bound kept); the monorepo's Downgrade workflow is deleted.
- **Tests hardened across Julia versions** (since 2026-11-13 CI runs 1.13 only): the refiners match the PoseLib 0.1.9 baselines
  (test/test_contract.jl) to 1e-12 relative instead of bit for bit; the last ulps move with the Julia version and the CPU.
- README: the Eigen link points at https://libeigen.gitlab.io/ (the old host times out).

## 0.2.0 (2026-11-12)

- **`refine_relpose` refuses a non-finite bearing** by name (a pixel past a fisheye's fold back-projects to NaN), as
  VisualGeometryToolkitEstimation's bearing gate does; before, `NaN` got through the unit-norm test. The strategy constant is
  VisualGeometryToolkitCore's `efficiency_constant(lo, d)` (the private copy deleted; the refusal of a truncation's
  missing constant now reads "TLS (TLSShape) has no efficiency constant (it is a truncation): …").

- **Breaking: the refiners record their termination** (VisualGeometryToolkitCore's §14 record, in the FitResult core):
  `(ReturnCode.Success, NoRule)` before the iteration cap (the owner's ruling: PoseLib reports no stop rule, so none is
  claimed; `retcode_symbol` `:converged`), `(ReturnCode.MaxIters, IterationCap)` at it. `retcode(r)` is the
  `ReturnCode.T`; `converged(r)` is unchanged.

- **Breaking: the refiners speak the stack's robust-estimation contract** (`refine_homography`,
  `refine_fundamental`, both `refine_relpose` forms, both `refine_absolute_pose` forms). A renaming: every
  refinement is bit-identical to the one it replaces (test/test_contract.jl pins the stack's PoseLib baselines).
  - The strategy is `optimization_strategy` and is required: VisualGeometryToolkitCore's `Irls{TLSShape}()`,
    `Irls{HuberShape}()`, `Irls{CauchyShape}()`, PoseLib's `LeZach()`, or `nothing` for least squares. The `loss`
    and `loss_width` keywords are retired and refused with an `ArgumentError` naming their replacements.
    `loss = TLSShape(), loss_width = w` is now `optimization_strategy = Irls{TLSShape}(), width = w`;
    `loss = GaussianShape()` is `optimization_strategy = nothing`. The keyword is the stack's refine name,
    `optimization_strategy`; `local_optimization` is refused, naming it.
  - `LeZachTruncation()` is renamed `LeZach()`, a member of the stack's one optimization-strategy family
    (`LeZach <: AbstractOptimizationStrategy`), registered through VisualGeometryToolkitCore's `@lo_registry` in the
    new public `PoseLib.POSELIB_STRATEGY_TAGS` with TOML tag `"le_zach"` (merge it with the core's `STRATEGY_TAGS` to read
    `optimization_strategy = { type = "le_zach" }`).
  - A `Triggs{K}` strategy, any other loss shape and `Irls{GaussianShape}` are refused: PoseLib's
    Levenberg–Marquardt weighs residuals by `ρ′` alone, and it runs only the losses above.
  - The width is exactly one of `width = w` (unchanged from `loss_width`) and `scale = σ` (or an upstream robust
    result): the loss then runs at `c·σ`, `c` the strategy's efficiency constant, else
    `efficiency_constant(shape, d)` at the residual's codimension. TLS and Le–Zach have none, so take `width`.
  - Each refiner returns a `FitResult` (its `estimate` the model returned before): PoseLib's `iterations`, which
    were discarded, and a `retcode` (`:converged`, or `:max_iterations` when the budget was used up), with the
    scale record `PoseLibRefineMetadata` (`scale`, `scale_source`, `width`; `noise_scale(result)`). PoseLib
    reports no residuals or weights: `rms` is `NaN`, `n` the number of correspondences. Read the model with
    `estimate(result)`.

- **Breaking: the unexported alias `PoseLib.bundle_adjust` is removed.** It was a second name for
  `refine_absolute_pose`, under the name of PoseLib's C++ function. Call `refine_absolute_pose(pose, K, cs; …)`,
  with the same arguments. The C wrapper `poselib_bundle_adjust_cam` that the refinement calls is unchanged.

## 0.1.9 (2026-11-09)

- On VisualGeometryToolkitCore 0.8 (compat `"0.8"`); nothing else changes.

## 0.1.8 (2026-11-07)

- **`calibrated_loss_width(K1, K2, loss_width)` (public)**: the bearing-unit width PoseLib's relative-pose refiner
  works at when handed a width in the points' unit and the two intrinsics, `loss_width · ½(1/f₁ + 1/f₂)` with
  `fᵢ = (Kᵢ₁₁ + Kᵢ₂₂)/2` (PoseLib's `robust.cc` rule), so a calibrated refiner deploys at the same operating point
  (LocalOptimizationDoneRight's Lebeda arms held a copy). The `SIMPLE_PINHOLE` mapping of a K is stated once, for it
  and for the `refine_relpose` and `refine_absolute_pose` bindings. No result changes.

## 0.1.7 (2026-11-06)

VisualGeometryToolkitCore 0.6 (compat `"0.6"`).

- `essential_dlt` with fewer than eight correspondences throws an
  `ArgumentError` naming the count it got (`essential_dlt needs at least 8
  correspondences, got 7`), in place of an `@assert`. An `@assert` may be
  disabled at a higher optimization level, and its `AssertionError` is not the
  input error the stack throws everywhere else.
- `refine_fundamental`, `refine_homography` and `refine_absolute_pose` throw an
  `ArgumentError` naming the size they got when the seed `F`, `H` or the
  calibration `K` is not 3×3 (`H must be 3×3, got 2×2`), in place of an
  `@assert`: with the assertion disabled, the matrix was read under
  `@inbounds` past its end.
