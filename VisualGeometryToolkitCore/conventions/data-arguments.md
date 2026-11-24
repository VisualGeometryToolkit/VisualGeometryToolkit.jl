## Data arguments: correspondences in, dicts for groups

- An estimator's data is a vector of correspondences,
  `cs::AbstractVector{<:Correspondence}` (`Correspondence = Pair`; `correspondences(sources, targets)`
  builds the columnar `StructArray`), never parallel arrays (`x1, x2`) or
  index tuples.
- Per-correspondence data (prior scores, weights, covariances) are attribute
  COLUMNS (`withattrs(cs; weights = w)`), never keywords; per-correspondence
  outputs come back the same way.
- **Selection is indexing; influence is a weight** (owner, 2026-11-24). No public
  function takes a `mask`: a hard selection is the caller's own indexing, `cs[idx]`
  (an index list or a `BitVector`), and a soft influence is the `weights` column
  (`withattrs(cs; weights = w)`; a zero weight leaves a datum out). Repeated data is
  valid data: a correspondence listed twice counts twice, which in a least squares is
  the same as weight 2. Index lists and masks live only inside the inner kernels
  (`homography_dlt!(A, N, cs; mask)`, the RANSAC gates), which refuse one that asks
  for more rows than their buffer holds (`DimensionMismatch`).
- Grouped data is an `AbstractDict{K, <:AbstractVector{<:Correspondence}}` keyed by
  the identity of the group's unknown (a marker id, a `(frame, board)`):
  `EstimationProblem(RobotWorldHandEyeCalibration, markers)`, `EstimationProblem(CameraModel, (radial_cameras, views))` (a `DivisionCameraFit`).
  Block order is `sort(collect(keys(groups)))`, stored on the result; results
  come back keyed by `K`, and a joint covariance is ordered
  `[shared | per-group in key order]`.
- Given per-group models lead, keyed by the same `K`; the key sets must match
  (`ArgumentError`).
- Nest only an observation with its evidence: `A_i => correspondences(X, u)` (a rig
  pose => that frame's features) in a `RobotWorldHandEyeProblem`'s refine.
- **Keywords are configuration only.**
- A value carried between stages carries its per-datum attributes as
  `Attributed(x, attrs)` (`Attributed(H, (; weight))`, `Attributed(R_ij, (; weight))`):
  a stage reads the weight with `weight_of` (1.0 for a bare value) and the
  value with `value_of`, never from a parallel `weights` vector or keyword.
  `Attributed` is for the data, not for wrapping results.
- An estimator returns a `FitResult`; its extras go in its metadata, read by
  field (`r.metadata.depths`), and a cross-type contract quantity is read by its
  function (`weights(r)`), never by a forwarded property. Several fits
  come back keyed by what was fitted (a `Dict`). Per-datum outputs are
  index-aligned in the metadata, and callers attach them to their data with
  `withattrs`.
- Minimal-solver kernels stay positional internals (public, not exported);
  the public entry is `solve(MinimalProblem(M, cs, idx), solver)`.
