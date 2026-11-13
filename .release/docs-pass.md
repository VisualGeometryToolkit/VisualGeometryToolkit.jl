docs-pass: ce8acb3c
landing: VisualGeometryToolkit/visualgeometrytoolkit.github.io 6674c98 (no change: the portal names packages and links their docs; it lists no vgio kind or IO function)

# IO 0.9.4 docs pass (julia-docs-pass) — io-0.9.4 (VisualGeometryToolkitIO 0.9.4 alone; every other package unchanged)

Additive: the `vgio/panorama/v1` kind and `PointFrame` in the detection codec. Baseline: 2f994d2b (wave 113, IO 0.9.3).

## 1. Public API since the baseline

- New public, not exported: `save_panorama(path, camera::BoundedCamera, rotations::Trajectory, vignetting::AbstractVignetting,
  gains; length_unit = mm, provenance = (;))`, `load_panorama(path) -> (; camera, rotations, vignetting, gains)`
  (src/panorama_schema.jl).
- New methods of the public `json_value` / `from_json`: `PointFrame`, and a detection of an `Uncertain{PointFrame}`
  (src/core_json.jl). Nothing removed or re-signatured.

## 2. Static findings

| check | result |
|---|---|
| docstrings and pages | `save_panorama`, `load_panorama` on api/native.md (its `@autodocs` Pages gain panorama_schema.jl); `json_value`/`from_json` docstrings name `PointFrame` and its covariance encoding |
| guides | kinds.md (the kinds table: the panorama row), schemas.md (the current versions: panorama v1), serialization.md (the frames table's `PointFrame` row, the point frame's paragraph and a doctest) |
| module header | VisualGeometryToolkitIO.jl: the current versions list and the I/O surface comment name the kind |
| CHANGELOG | VisualGeometryToolkitIO 0.9.4 |
| versions | IO 0.9.4 (Project.toml and its benchmark env's compat on itself); in-repo dependents' compat unchanged (a patch) |
| floors | floor_compat --latest and --raise --check on the gate's resolve: 0 raises (gate prep 11552) |

## 3. Gate

Fresh-clone gate on Julia 1.13.3, ce8acb3c, all green (prep 11552; tests 11553, 11554; docs 11554_5): IO strict 30760 tests (4 broken: the report-mode stack gates and the pycolmap skip, as at 0.9.3), Eval strict 2906, IO docs (checkdocs = :public, doctest, linkcheck). Table: ~/logs/dgx-agent/io094-ps064.md.
