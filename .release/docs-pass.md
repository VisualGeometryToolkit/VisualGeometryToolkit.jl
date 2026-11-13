docs-pass: 3d8203a4
landing: VisualGeometryToolkit/visualgeometrytoolkit.github.io 6674c98 (no change: a compile-time fix)

# Features 0.8.4 docs pass (julia-docs-pass) — patch-features084

`extract_capture_settings` compiled one zip per capture size (a 171-photo set: 518 s, ~90 % LLVM; a 500-frame test
never finished in 1500 s); the identities now merge as columns (StructArrays' `components` of a `StructArray` of the
identities), and Features' other splats of a data-length collection are gone. Baseline: main 252dbca9 (Core 0.9.6,
Estimation 0.10.2, Eval 0.7.3), then main da7e0948 (IO 0.9.5) merged in (9a3c962b; Features' tree unchanged by it).

## 1. Public API since the baseline

None changed. Internal: `_merged`, `_one_identity`, `_score_eltype` (takes a collection), `detect_foerstner`'s
collect, `compare_images`' `linkaxes!(axes)`.

## 2. Static findings

| check | result |
|---|---|
| docstrings and pages | unchanged (no public name moved); `extract_capture_settings`' docstring states the merge rule as before |
| CHANGELOG | Features 0.8.4 |
| versions | Features 0.8.4; every other package as on main |
| in-repo compat | unchanged (Features' Estimation weak compat "0.10.1" admits 0.10.2) |

## 3. Executed on the release candidate (3d8203a4; Features' tree 70e050f55549, the same at the merge 9a3c962b)

Gate (~/logs/dgx-agent/scripts/patch096, `ONLY_PKGS=VisualGeometryToolkitFeatures`; jobs 14059–14062): the full prep
(registration_lint --subsequent 0 errors 0 warnings, agent_docs 0, floor_compat --latest, audit_manifests,
check_duplication 0 on all six, the GPU stacks with cuDNN 1), then Features' groups: CPU 3069 / 9 (6 groups),
GPU 2011 / 0 (2 groups), docs green. Julia 1.13.3. Features loads no IO (not in its package, test or docs
environment), so IO 0.9.5 under it changes none of these.

---

docs-pass: 458f816b
landing: VisualGeometryToolkit/visualgeometrytoolkit.github.io 6674c98 (no change: the portal names packages and links their docs; it lists no vgio kind or IO function)

# IO 0.9.5 docs pass (julia-docs-pass) — io-0.9.5 (VisualGeometryToolkitIO 0.9.5 alone; every other package unchanged)

Additive (RadiometricCamera's IO audit; owner rulings 2026-11-13): `vgio/raster/v1` and the stack's one TIFF codec, the
target-calibration and lattice codecs made public, height-field per-field units. Baseline: eeb82edf (IO 0.9.4, merged in).

## 1. Public API since the baseline

- New public, not exported: `write_tiff(path, image::AbstractMatrix{T}; compression, predictor, rows_per_strip, density,
  software, icc)`, `read_tiff(T, path)` (src/tiff.jl); `save_raster(path, values::AbstractMatrix{Float32}; frame, nan,
  length_unit, compression, provenance)`, `load_raster(path)` (src/raster_schema.jl).
- Made public (were private, renamed, no alias): `target_calibration_json(s::VgioStyle, c; node_unit)` and
  `target_calibration_from_json(s, record)` (src/rig_schema.jl; `_target_calibration_json` / `_target_calibration_from_json`);
  `lattice_json(spline::ChartSpline; present, length_unit)` and `lattice_from_json(record, Val(K); length_unit)`
  (src/style.jl; camera_schema.jl's `_spline_json` / `_spline_from_json`).
- New keyword `node_unit` of `target_calibration_json` (height-field per-field units). Nothing removed or re-signatured.

## 2. Static findings

| check | result |
|---|---|
| docstrings and pages | each new public name has a docstring with its signature as in the code (`signature_mismatches` 0, test/docs_tests.jl); api/native.md's `@autodocs` Pages gain tiff.jl and raster_schema.jl; `target_calibration_json` and `lattice_json` carry doctests (the lattice one without StaticArrays, which the docs env does not hold) |
| guides | kinds.md: the kinds table's raster row, a "Rasters" section with the header and the TIFF profile, the building-blocks section (the calibration and lattice pairs, the TIFF codec); schemas.md: raster v1 in the current versions |
| module header | VisualGeometryToolkitIO.jl: the I/O surface comment (load_raster / save_raster, target_calibration_json / _from_json) |
| CHANGELOG | VisualGeometryToolkitIO 0.9.5 |
| versions | IO 0.9.5 (Project.toml and its benchmark env's compat on itself); in-repo dependents' compat unchanged (a patch) |
| deps | Libtiff_jll added, [compat] "4.7.3" (the resolve; floor_compat --raise: none to raise); allow-list unaffected (a third-party dep); tifffile in the pixi `py` feature |
| README / landing | no change needed: the README links the kinds guide; no install line changes |
| naming | check_naming clean (one test comment reworded: "passed" → "checked") |

## 3. Gate

Fresh-clone gate on Julia 1.13.3, 458f816b, all green (scripts ~/logs/dgx-agent/scripts/io095/gate.sbatch, submit.sh;
prep 12390, IO groups 12713_1–8, Eval 12392/12393, docs 12391_11, bbcheck 12394): IO strict 31296 tests in eight groups
(3 broken: the two report-mode stack gates and one pre-existing `@test_broken`; the pycolmap and tifffile items RAN, the
pixi `py` env installed), Eval strict 2906 CPU / 2907 GPU, IO docs (checkdocs = :public, doctest, linkcheck), BlobBoards'
board TIFFs byte-identical through IO's write_tiff, registration_lint --subsequent, floor --latest and --raise --check,
audit, check_duplication 0. Table: ~/logs/dgx-agent/io095.md.
