# VisualGeometryToolkitIO changelog

## 0.2.0 (unreleased)

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
