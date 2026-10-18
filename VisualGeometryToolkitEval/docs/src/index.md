# VisualGeometryToolkitEval

The evaluation base of the visual-geometry stack: the statistics, the codified
cross-validation, the experiment framework, the figure renderers, the
scoring-pool IO, the raw dataset readers, and the composition evaluator of
scorer + local-optimization arms that the stack's research projects share.

- [The experimental protocol](protocol.md): the one statement of how the
  stack's experiments tune, evaluate and report (cross-validation, the
  scale-multiple fairness axis, miscalibration, the budgets, the metric, the
  statistics, the figure conventions).
- [The `[tuning]` schema](tuning.md): every parameter of a cross-validation,
  its default in `configs/tuning.toml`, and the tuning artifact that persists
  it.
- [API](api.md): every documented name, by module.

```@docs
VisualGeometryToolkitEval
```
