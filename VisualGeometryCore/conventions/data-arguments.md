## Data arguments: correspondences in, dicts for groups

- An estimator's data is a vector of correspondences,
  `cs::AbstractVector{<:Cspond}` (`Cspond = Pair`; `csponds(sources, targets)`
  builds the columnar `StructArray`), never parallel arrays (`x1, x2`) or
  index tuples.
- Per-correspondence data (prior scores, weights, covariances) are attribute
  COLUMNS (`withattrs(cs; weights = w)`), never keywords; per-correspondence
  outputs come back the same way.
- Grouped data is an `AbstractDict{K, <:AbstractVector{<:Cspond}}` keyed by
  the identity of the group's unknown (a marker id, a `(frame, board)`):
  `fit_robot_world_hand_eye(markers)`, `fit_division_camera(radial_cameras, views)`.
  Block order is `sort(collect(keys(groups)))`, stored on the result; results
  come back keyed by `K`, and a joint covariance is ordered
  `[shared | per-group in key order]`.
- Given per-group models lead, keyed by the same `K`; the key sets must match
  (`ArgumentError`).
- Nest only a measurement with its evidence: `A_i => csponds(X, u)` (a rig
  pose => that frame's features) in `refine_robot_world_hand_eye`.
- **Keywords are configuration only.**
- Minimal-solver kernels stay positional internals (public, not exported);
  the public entry is `solve(solver, cs, idx)`.
