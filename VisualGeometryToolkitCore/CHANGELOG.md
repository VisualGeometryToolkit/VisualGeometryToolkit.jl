# VisualGeometryToolkitCore changelog

## 0.2.0 (unreleased)

Behaviour changes a caller can observe, stated here because the commits that
introduced them did not state them all.

- **`parameter_bounds` states which endpoints are closed, and a closed
  endpoint is a box in `ParameterManifold`'s chart.** The closed endpoints
  are the values a model takes: the double sphere's `α = 0` and `α = 1`, the
  extended unified model's `α = 0` (the pinhole), `α = 1` and `β = 0`, and
  the field-of-view term's `ω = 0` (the identity). There, the chart is the
  open ends' chart (`t₀ + v` when there are none) clamped at the endpoint,
  with slope 0 past it. Open endpoints keep the log/logit chart and its
  10⁻⁶ anchor margin. As a result:
  - an anchor at `α = 0` or `ω = 0` stays there and moves with slope 1
    (`ω`: slope `π − ω₀`). Before, it was moved to 10⁻⁶ of the interval with
    a slope of about 3·10⁻⁶.
  - the double sphere's `α` chart is now additive. It is no longer the logit
    that 0.1 used, so `retract` values on `α` differ from 0.1's.
  - `β` of the extended unified model is additive and clamped at 0. It is
    no longer `β₀·exp(v)`.
- **A negative field-of-view `ω` is anchored as `|ω|`.** `ω` and `−ω` are one
  map, so this is the same lens. Before, `FieldOfView(−0.3)` was anchored at
  `ω ≈ 3·10⁻⁶`, a different lens.
- **`ω = 0` is a stationary point of the field-of-view model itself.** Its map
  is even in `ω`, so `∂project/∂ω = 0` there whatever the chart is. Start a
  fit of `ω` at a small `ω ≠ 0`.
- **A coordinate with no step keeps its value exactly.** `retract` with a zero
  chart offset returns the anchor's value bit for bit, so a coordinate that
  `restrict` holds fixed never drifts across recenters. Before, a fixed
  coordinate under a log or logit chart moved by an ulp on each recenter:
  `ξ = 0.1` became `0.10000000000000009`.
