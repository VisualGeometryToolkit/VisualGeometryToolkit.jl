docs-pass: 885fbb4f
landing: VisualGeometryToolkit/visualgeometrytoolkit.github.io 6674c98 (not re-read: no local clone on dgx01; the patch changes no name, signature or example, which is all the landing text lists)

# VisualGeometryToolkitCore 0.11.1 docs review — branch patch-core0111

Released: Core 0.11.1 (a patch: a 0.11.0 regression). Baseline: origin/main ad9c41e8 (Core 0.11.0, Estimation 0.12.0,
Features 0.9.8). The review covers the tree 885fbb4f. After it, only this record changes (`.release/` is outside the
gate). Code changes in VisualGeometryToolkitCore only; one new test in VisualGeometryToolkitEstimation.

## 1. Public API since Core 0.11.0

- No name added or removed, no signature changed. New internal constant `_NLLS_NONFINITE_BITS` (not public). Compat
  unchanged.

## 2. Docstrings and pages brought up to date

- `nlls_stop_code` (its order: bit 10 is no failure; its stop goes to the postcondition), `nlls_stop_reasons`
  (`:cost_disagreement` stops NLLSsolver one iteration later), `nlls_termination`'s table (rows for bit 10 alone and
  with a tolerance bit; the non-finite rows without 10), `lmsolve_chart!`'s success rule, `LmStops`' `stationarity_tol`
  and `residual_resolution` rows (read again by a host verdict), `StopRule`'s `NoRule` (and its per-value doc),
  `configs/default.toml`'s `stationarity_tol` comment, the 0.11.1 CHANGELOG entry.
- Docstring placement: `_NLLS_NONFINITE_BITS` and its comment sit ABOVE `nlls_stop_code`'s docstring, which stays
  attached to the function (the simplify review caught the first placement detaching it).
- `docs/src/api/termination.md` lists `NonFinite` and the record types only; nothing there states the bit decode, so
  it is unchanged. README unchanged (no example touches the decode).

## 3. Gates

- Strict docs build (`checkdocs = :public`, doctests, no `warnonly`) for Core and Estimation, the `docs`, `README`,
  `quality`, `naming` and `agent docs` items: in the release gate at the head carrying this record.
- No local Julia (kaimon down): every check ran on Slurm.
