# PoseLib changelog

## Unreleased

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
