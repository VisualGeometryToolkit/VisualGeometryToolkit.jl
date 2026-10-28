# VisualGeometryToolkitCore changelog

## 0.2.0 (unreleased)

Behaviour changes a caller can observe, stated here because the commits that
introduced them did not state them all.

- **`parameter_bounds` states which endpoints are closed, and a closed
  endpoint is a mirror in `ParameterManifold`'s chart.** The closed endpoints
  are the values a model takes: the double sphere's `α = 0` and `α = 1`, the
  extended unified model's `α = 0` (the pinhole), `α = 1` and `β = 0`, and
  the field-of-view term's `ω = 0` (the identity). There, the chart is the
  open ends' chart (`t₀ + v` when there are none) reflected at the endpoint:
  a value past it folds back inside (with two closed ends, with period
  `2(b − a)`), and the slope keeps its magnitude with its sign flipped. A
  clamp would leave a flat residual past the end, at which a fixed-chart fit
  stalls. Open endpoints keep the log/logit chart and its 10⁻⁶ anchor
  margin. As a result:
  - an anchor at `α = 0` or `ω = 0` stays there and moves with slope 1
    (`ω`: slope `π − ω₀`). Before, it was moved to 10⁻⁶ of the interval with
    a slope of about 3·10⁻⁶.
  - the double sphere's `α` chart is now additive. It is no longer the logit
    that 0.1 used, so `retract` values on `α` differ from 0.1's.
  - `β` of the extended unified model is additive and reflected at 0. It is
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
- **`Adapt.adapt` of a generic grid model replaces a domain that is not isbits
  with `WholePlane()` on the device copy.** An isbits domain, such as an
  ellipse or a rectangle, travels with the model. The domain is host-side
  bookkeeping that no device evaluation reads.

### The LM driver: NLLSsolver (2026-10-27)

Every nonlinear least-squares fit now runs on NLLSsolver, through
`lmsolve_chart!` and `nlls_options`. The previous driver is no longer a
dependency. What changed for a caller:

- `max_iter` counts **accepted** iterations. The previous driver also counted
  rejected candidate steps.
- `xtol` is **absolute**: the max-norm of the step in chart units, `max|pᵢ| <
  xtol`. The previous driver's test was relative per component, `|pᵢ| <
  xtol·(xtol + |xᵢ|)`. The same value is now stricter far from the origin and
  looser near it.
- `ftol` is NLLSsolver's relative cost decrease. The previous driver required
  both the actual and the predicted reduction of an accepted step to fall
  below `ftol·F`. A step whose cost is NaN is still rejected (the adapter
  reads it as +Inf).
- There is no gradient tolerance. The stop codes that the stall postcondition
  reads (step-size stop `2`, cost-decrease stop `1`) keep their meaning.

Measured against the previous driver:

- **Chart LM, E refits.** On 317 captured E refits of the PhotoTourism
  replay at `ftol = 1e-10`, the returned chart vector agreed within 2.0e-7
  (max-norm; the median refit's ‖x‖∞ is 1.8e-3) and the cost within 2.0e-11
  relative, with every refit converged. Time to convergence was 0.84× the
  previous driver's when Jᵀr and JᵀJ are accumulated in one sweep, and 0.97×
  when NLLSsolver forms JᵀJ itself.
- **`ftol` for the chart LM.** The previous driver's default, eps, ran every
  refit to its rounding floor. On the PhotoTourism E replay (360 Welsch-IRLS
  LO calls, 2026-10-07) a refit took 28.6 residual and 8.0 Jacobian sweeps,
  about 20 of them rejected steps at the floor. At 1e-10 it took 5.6 and 4.1
  sweeps and was 2.3× faster, with the returned E within 7.8e-7 (unit
  Frobenius, the maximum over the 360).
- **Relative-pose bundle.** Measured on the A/B scenes (800 px, σ = 0.5 px,
  N = 100 / 1000 / 10000):
  - At eps, the previous driver took 21–34 iterations. At 1e-10 it stopped
    in 5, with the cost within 4.2e-14 relative and the pose within 6e-9
    chart units of the eps run.
  - NLLSsolver also stops in 5 at 1e-10. Both drivers reach the same optimum:
    the cost agrees to ≤ 3e-14 relative and the rotation to ≤ 6e-11°.
  - At the stop the agreement is: NLLSsolver cost ≤ 6.2e-14 relative,
    rotation ≤ 2.1e-8°, translation direction ≤ 1.8e-7°; the previous
    driver 4.6e-14, 1.8e-8° and 1.6e-7°.
  - Time relative to the previous driver was 0.88 / 1.04 / 1.00× with the
    cached problem and 1.78 / 1.49 / 1.44× without it.
  - Allocations per call: 19 / 20 / 20 now, against the previous driver's
    53 / 54 / 54.
- **Evaluation-engine golden.** The IRLS errors moved by at most 3.1e-6
  relative (1.1e-5 absolute), and the trial counts did not move.
- **Pixel LM kernel.** The generic models' kernel (`Cameras._pixel_lm`, the
  one device-side LM) keeps the previous driver's static bounded algorithm.
  It is gated at 2e-6 px against a recorded run of that routine
  (`test/fixtures/pixel_lm`).
