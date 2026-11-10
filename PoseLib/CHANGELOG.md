# PoseLib changelog

## Unreleased

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
