# VisualGeometryToolkitEval changelog

## 0.5.0 (2026-11-06)

VisualGeometryToolkitCore 0.6, VisualGeometryToolkitIO 0.6 and
VisualGeometryToolkitEstimation 0.6.1 (compat `"0.6"`, `"0.6"`, `"0.6.1"`): the
breaking upstream releases; no change to this package's own code.

## 0.4.3 (2026-11-04)

VisualGeometryToolkitCore 0.5.3 (compat `"0.5.3"`).

- `Stats.stable_seed` is VisualGeometryToolkitCore's `stable_seed`, hoisted there
  so that packages outside the evaluation stack can seed by name without
  depending on Eval. `Stats` re-exports Core's function (no copy); the
  derivation and every seed it gives are unchanged.
- To be removed: the re-export goes in the next breaking release, with the
  VisualGeometryToolkitEstimation 0.7 wave. Import `stable_seed` from
  VisualGeometryToolkitCore (`using VisualGeometryToolkitCore: stable_seed`).
