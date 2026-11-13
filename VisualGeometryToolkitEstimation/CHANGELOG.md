# VisualGeometryToolkitEstimation changelog

## 0.11.0 (unreleased)

The breaking wave 10, on VisualGeometryToolkitCore 0.10.

### Breaking

- **Every LM runs through Core's one adapter** (the owner's "NO DUPLICATIONS" ruling, 2026-11-13): the chart refits
  (`_chart_refit`: the LO's `refine!`, a refine's least squares and its Triggs), `refine_camera`'s lens regression,
  the camera-and-poses bundles (`_solve_bundle`: every `refine_camera_and_poses` form, `refine_robot_world_hand_eye`)
  and the relative-pose bundle (`refine_relative_pose`, `RelposeBundleProblem`'s refit) call `lmsolve_chart!` with
  Core's `LmStops` (a bundle as an `LmProblem`, the relpose bundle's kept across refits). Their termination record is
  Core's ONE success rule: `ReturnCode.Success` only where the stationarity postcondition holds at the returned point,
  at every stop that is not a failure or a cap — the cost-decrease stop included (until 0.10 a bundle's cost-decrease
  stop was Success unchecked) — the robust bundles' postcondition √ρ′-weighted. A singular system in a bundle is
  `(Failure, NonFinite)` (the relpose bundle: no solution); any other exception propagates.
- **`[chart_lm]` and `[relpose_bundle] stationarity_tol` are retired** from `configs/default.toml`: the chart LMs' stop
  settings and every LM's stall tolerance are Core's `[lm]` (the same values: 1000, 1e-10, 1e-10, 0.05). The private
  `_CHART_LM_*` constants, `_bundle_retcode`, `_bundle_jacobian` and the bundles' stationarity-row copies are deleted.
  `_refine_relative_pose` returns `(; R, t, sol)` (`sol`: Core's `(; code, rule, iterations, cost)`; its `retcode` and
  `stationarity_tol` keywords are gone — the record is always computed).
