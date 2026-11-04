# VisualGeometryToolkitIO changelog

## 0.5.0 (2026-11-04)

Breaking: the target and rig records bump with VisualGeometryToolkitCore
0.5's composition (CoordinateTransformations' `∘`) and its gauge-free target
calibration.

- **`vgio/target/v3` and `vgio/rig/v6`.** The calibration kinds are VGC 0.5's:
  new leaf records `euclidean` (a planar `EuclideanMap`, `[angle, tx, ty]`,
  the translation in the file's unit; the angle the one VGC's member stores,
  its `Angle2d`'s, written as it is and read back as an `Angle2d`, so value →
  file → value is exact — no search for an angle that rebuilds a matrix),
  `scale` (`[log_scale]`), `aspect` (`[a1]`) and
  `skew` (`[u12]`), the last three dimensionless; `thin_plate_spline` and
  `height_field` hold pure bending, and a record whose values carry an affine
  term is refused (VGC's constructor would move them). A composition — 2 to 6
  kinds in the canonical order `HeightField ∘ ThinPlateSpline ∘ EuclideanMap
  ∘ Scale ∘ Aspect ∘ Skew`, any left out — is written flat by its structure,
  `{"type": "composition", "compose": [outer, …, inner]}`, in CT's order (the
  outer map first), its covariance over the concatenated parameters in that
  order, and read back right-associated; v2/v5's
  `{"type": "calibration_chain", "members": […]}` listed them in application
  order. The key changed with the order, so the retired form in a current
  record is refused by name (an `ArgumentError` naming `vgio_migrate`), never
  read in the new order. The `"affine"` kind (CT's `AffineMap`) is retired:
  refused by name, naming its factors `EuclideanMap ∘ Scale ∘ Aspect ∘ Skew`
  and `vgio_migrate`. A calibration without lengths (the dimensionless
  factors) is written whatever its form; one with lengths must be physical.
- **A planar target's chart placement** (VGC's `ChartPlacement`): an optional
  `"placement": {"origin": [u, v], "side": s}` beside `board` and `canvas`,
  lengths in the file's unit (decimal-exact), refused on a points target. A
  placed target's calibration acts in the board's dimensionless chart, so its
  record — nodes, parameters, covariance — is plain chart numbers, written and
  read bit for bit in any unit. The placement enters the `content_id` tree
  (origin and side in metres). A body without one reads as before.
- **`vgio/cameras/v4`, and `vgio/rig/v6`'s sensors**: a lens's distortion
  stages in sequence (CT's `∘`) are listed in the lens record's `"compose"`
  outer first (the key a composed calibration lists its members under); v3/v5
  listed them under `"chain"` in application order, and a `"chain"` in a
  current record is refused naming `vgio_migrate`. `vgio_migrate` converts a
  v3 store (and a v4 or v5 rig's sensors with its targets). OpenCV's 14-coefficient lens is `Tilt ∘ Additive` and COLMAP's
  `RAD_TAN_THIN_PRISM_FISHEYE` `Additive ∘ Radial` in `opencv_lens` and the
  COLMAP assembly; the exported coefficients and COLMAP parameters are
  unchanged.
- **`vgio_migrate` converts `vgio/target/v2` and `vgio/rig/v5`** (reversing a
  chain's members and permuting its covariance). An affine calibration
  becomes its Iwasawa factors (A = Q·T by QR, det A > 0 required: ϑ =
  atan(Q₂₁, Q₁₁), log s = ½ log(t₁₁ t₂₂), a₁ = ½ log(t₁₁/t₂₂), u₁₂ = t₁₂/t₁₁),
  the same map, its covariance through the factorization's Jacobian; in a
  chain it is replaced by the four in its place. A chain with no current form
  (in-plane kinds out of the canonical order, a Rigid member) is refused, and
  so is a thin-plate spline or height field whose interpolant carries an
  affine term: no exact form exists (the factors compose after the spline,
  and a plane of heights is a shear, not a rigid motion a pose could take),
  so it is re-fitted rather than silently changed. The readers refuse v2/v5,
  naming the migration. A rig keeps its `content_id` unless its value
  changes: the identity tree lists a composition's members and a lens's
  stages in one fixed order, application order (the order v5 held), whatever
  order the record lists them in; a rig whose target held the affine kind
  changes it, and the migration reports v5's id as the old one so
  `vgio_migrate_tree` re-points its measurement files.
- Depends on CoordinateTransformations (the calibration's composition type).
- The calibration record reads the composition's structure from VGC (`members`,
  `TARGET_CALIBRATION_ORDER`, `planar_kind`, `is_pure_bending`) instead of
  restating it, and the v2 → v3 migration factors a retired affine with VGC's
  `iwasawa_parameters_with_jacobian`; records, covariances and messages are
  unchanged (a reflection's refusal is VGC's message, still naming `det A`; a
  chain with no affine member still has its covariance permuted exactly, an
  `Inf` or a `-0.0` entry included, not multiplied through `J Σ Jᵀ`).

## 0.4.0 (2026-11-02)

- The stack moves to the 0.4 series with VisualGeometryToolkitCore 0.4.0; no change to this package's API.

## 0.2.0 (2026-10-29)

- **A `:length` detection's covariance is written in the declared unit
  squared.** Example: a `Disc` detection's `cov` under `kind = :length`. Under
  `MM_PX` it is written in mm², and it is read back the same way. IO 0.1.0's
  `json_value` wrote it raw, in m², with no marker of the change. The same
  JSON read by 0.2.0 under a mm contract therefore comes back 10⁶ times
  smaller. Such a record from 0.1.0 has to be rewritten: multiply each `cov`
  entry by the number of length units per metre, squared. No writer in the
  stack persisted such records.
- **A detection's covariance size is the size of the value.** It is no
  longer tied to the frame type. For example, ASIFT's `EllipticFrame` carries
  a 3×3 covariance. A covariance that is not finite is written as `"cov":
  null` together with its size under `"cov_size"`. A `null` with no
  `cov_size` (IO 0.1.0's record) reads as 3×3, as it did in 0.1.0.
