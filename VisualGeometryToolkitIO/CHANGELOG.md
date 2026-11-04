# VisualGeometryToolkitIO changelog

## 0.5.0 (unreleased)

Breaking: the target and rig records bump with VisualGeometryToolkitCore
0.5's composition (CoordinateTransformations' `∘`).

- **`vgio/target/v3` and `vgio/rig/v6`.** A composed calibration —
  `HeightField ∘ AffineMap` or `HeightField ∘ ThinPlateSpline`, the only two —
  is written by its structure, `{"type": "calibration_chain", "members":
  [height_field, affine | thin_plate_spline]}`, its members in CT's order (the
  outer map first) and its covariance over that order; v2/v5 listed a chain's
  members in application order. Every other record is unchanged but for its
  schema. The affine kind reads back as CT's `AffineMap` (its `b` lengths in
  metres).
- **`vgio/cameras/v4`, and `vgio/rig/v6`'s sensors**: a lens's distortion
  stages in sequence (CT's `∘`) are listed in the lens record's `"chain"` outer
  first; v3/v5 listed them in application order. `vgio_migrate` converts a
  v3 store (and a v4 or v5 rig's sensors with its targets). OpenCV's 14-coefficient lens is `Tilt ∘ Additive` and COLMAP's
  `RAD_TAN_THIN_PRISM_FISHEYE` `Additive ∘ Radial` in `opencv_lens` and the
  COLMAP assembly; the exported coefficients and COLMAP parameters are
  unchanged.
- **`vgio_migrate` converts `vgio/target/v2` and `vgio/rig/v5`** (reversing a
  chain's members and permuting its covariance); a v2/v5 chain of two
  in-plane kinds has no current form and is refused. The readers refuse v2/v5,
  naming the migration. Every rig keeps its `content_id`: the identity tree
  lists a composition's members and a lens's stages in one fixed order,
  application order (the order v5 held), whatever order the record lists them
  in.
- Depends on CoordinateTransformations (the calibration's `AffineMap` and
  composition types).

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
