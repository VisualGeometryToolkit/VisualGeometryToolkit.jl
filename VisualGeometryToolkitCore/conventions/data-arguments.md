## Data arguments: correspondences in, dicts for groups

- An estimator's data is a vector of correspondences,
  `cs::AbstractVector{<:Correspondence}` (`Correspondence = Pair`; `correspondences(sources, targets)`
  builds the columnar `StructArray`), never parallel arrays (`x1, x2`) or
  index tuples.
- Per-correspondence data (prior scores, weights, covariances) are attribute
  COLUMNS (`withattrs(cs; weights = w)`), never keywords; per-correspondence
  outputs come back the same way.
- Grouped data is an `AbstractDict{K, <:AbstractVector{<:Correspondence}}` keyed by
  the identity of the group's unknown (a marker id, a `(frame, board)`):
  `fit_robot_world_hand_eye(markers)`, `fit_division_camera(radial_cameras, views)`.
  Block order is `sort(collect(keys(groups)))`, stored on the result; results
  come back keyed by `K`, and a joint covariance is ordered
  `[shared | per-group in key order]`.
- Given per-group models lead, keyed by the same `K`; the key sets must match
  (`ArgumentError`).
- Nest only an observation with its evidence: `A_i => correspondences(X, u)` (a rig
  pose => that frame's features) in `refine_robot_world_hand_eye`.
- **Keywords are configuration only.**
- A value carried between stages carries its per-datum attributes as
  `Attributed(x, attrs)` (`Attributed(H, (; weight))`, `Attributed(R_ij, (; weight))`):
  a stage reads the weight with `weight_of` (1.0 for a bare value) and the
  value with `value_of`, never from a parallel `weights` vector or keyword.
  `Attributed` is for the data, not for wrapping results.
- A fit returns a `FitResult`; its extras go in its metadata. Several fits
  come back keyed by what was fitted (a `Dict`). Per-datum outputs are
  index-aligned in the metadata, and callers attach them to their data with
  `withattrs`.
- Minimal-solver kernels stay positional internals (public, not exported);
  the public entry is `solve(solver, cs, idx)`.
