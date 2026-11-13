docs-pass: 4209e867
landing: VisualGeometryToolkit/visualgeometrytoolkit.github.io 6674c98 (no change: a bug fix in the code identity of a run)

# Eval 0.7.4 docs pass (julia-docs-pass) — patch-eval074

A run of record whose `gitpath` is a REGISTERED install (PanoramaStitching's example: `cached_result(…; gitpath =
pkgdir(PanoramaStitching))`, run from a registered install, no git checkout) was refused as "no git checkout".
`code_trees` now identifies such a directory by the tree hash the active environment records for it
(`Pkg.dependencies()`: name, version, uuid, git-tree-sha1). Baseline: main 2a5838ae (Core 0.9.7, Estimation 0.10.3,
Eval 0.7.3).

## 1. Public API since the baseline

None changed: `code_trees`, `refuse_uncommitted_code` and `write_provenance` keep their signatures. A registered
`workdir` is recorded under its package name (`"<git-tree-sha1> <name> <version> <uuid>"`), its project path is that
name, and its stack packages are those of the active project's environment. Internal: `_registered_install`.
docs-pass: c835af7d
landing: VisualGeometryToolkit/visualgeometrytoolkit.github.io 6674c98 (no change)

# Core 0.9.8 docs pass (julia-docs-pass) — patch-core098

One device-capacity query for a kernel: how many of its threads the device holds at once, register-aware (the
owner's ruling: no constant, no config key). Baseline: main bfb7f7ed, main 2a5838ae merged in (bf4d9ac3).

## 1. Public API since the baseline

- Core: `device_resident_threads(kernel, args...; ndrange) -> Int`, public, not exported (base/device.jl, beside
  `device_free_bytes`; in the manual through base/device.jl's `@autodocs`). One method per backend of the private
  `_resident_threads`: the host's threads; the CUDA extension's method (the compiled kernel's occupancy); any other
  backend refused. `check_duplication` counts a kernel's resident threads among the device queries
  (`_TRANSFER_NAMES`). Nothing removed.

## 2. Static findings

| check | result |
|---|---|
| docstrings and pages | `code_trees`' docstring names the registered install; docs/src/reproduce.md lists it among the identities and narrows the refusal to code in no git checkout that is no registered install (accedc52) |
| CHANGELOG | Eval 0.7.4 |
| versions | Eval 0.7.4; every other package as on main |
| in-repo compat | unchanged (Eval's Core "0.9.5", Estimation "0.10.1" and IO "0.9.3" admit General's Core 0.9.6 and Estimation 0.10.2: 0.7.4 does not wait on Core 0.9.7 / Estimation 0.10.3) |

## 3. Executed on the release candidate

Gate on 4209e867 (~/logs/dgx-agent/scripts/w10-est-eval, `NICE=0 BR=patch-eval074 GATE_PKGS=VisualGeometryToolkitEval`;
General 665b2de2763; jobs 14587 prep, 14588 CPU, 14589 docs). Julia 1.13.3. Prep: registration_lint 0 errors 0 warnings,
agent_docs 0, floor_compat --latest, audit_manifests, check_duplication 0, check_naming 0, check_splat 0. Eval's CPU groups
2917 / 2917 (6 groups), docs green.

The first candidate, 7271049c (jobs 14538–14540), failed one test: quality: JET, `dirname(::Nothing)` on
`Base.active_project()` behind an `&&`. 4209e867 binds it to a local before the `nothing` test. The new test
(experiments_data: `code_trees(pkgdir(StableRNGs))`, a registered depot install, accepted with its tree hash, and
`write_provenance(…; gitpath = pkgdir(StableRNGs))` recorded) passes, as do the unchanged refusals of a dirty checkout
and of a path in no git checkout, and a clean checkout's git tree.

The release-records commit on top changes only .release/; its docs build is run again on it.
| docstrings and pages | the new name documented; Core's docs build strict (green) |
| CHANGELOG | Core 0.9.8 |
| versions | Core 0.9.8; every other package as on main |
| landing page | no device API named: unchanged |

## 3. Executed on the release candidate (c835af7d)

Gate (jobs 14548–14551), Julia 1.13.3: prep green (lint 0/0, agent_docs 0, floor --latest, audit, GPU stacks,
check_duplication 0 on all six); Core's CPU groups with device_tests.jl and quality_duplication_tests.jl 74704 / 6 and
61054 / 7; the GPU tier, every package's GPU groups: Core 53128 / 0 (4; device_resident_threads on the device, its
register-heavy kernel below half the raw maximum), Features 2011 / 0 (2), Estimation 55 / 0, Eval 359 / 0; Core docs
green. The other trees are 2a5838ae's (gated as Core 0.9.7 / Estimation 0.10.3).

---

docs-pass: 54447cd7
landing: VisualGeometryToolkit/visualgeometrytoolkit.github.io 6674c98 (no change)

# Core 0.9.7 / Estimation 0.10.3 docs pass (julia-docs-pass) — patch-core097-est0103

One launch-and-reduce helper: RadiometricCamera's `_RadiometryLm` and Estimation's `_LensRegressionLm` each launched a
per-row kernel and then reduced; Core 0.9.7 holds the pair once. Baseline: main 252dbca9, then main da7e0948 (IO 0.9.5)
and 0a49da23 (Features 0.8.4) merged in (54447cd7, bfb7f7ed; Core, Estimation and Eval trees unchanged by them).

## 1. Public API since the baseline

- Core: new methods `reduce_normal_equations(row!, Val(N), args...; partials)` and `reduce_cost(row!, args...;
  partials)` (robust/lm_core.jl; docstrings each, in the manual through api/robust_statistics.md's `@autodocs`).
- Estimation: none (the lens regression's launch-then-reduce pair replaced by the helper). `[compat]` Core "0.9.7".

## 2. Static findings

| check | result |
|---|---|
| docstrings and pages | the two methods documented; Core's docs build strict |
| CHANGELOG | Core 0.9.7, Estimation 0.10.3 |
| versions | Core 0.9.7, Estimation 0.10.3; IO 0.9.5, Features 0.8.4, Eval 0.7.3, PoseLib 0.2.1 as on main |

## 3. Executed on the release candidate (54447cd7; trees equal at bfb7f7ed)

Gate (jobs 14112–14116; `ONLY_PKGS` Core, Estimation, Eval), Julia 1.13.3: prep green (lint 0/0, agent_docs 0,
floor --latest, audit, GPU stacks, check_duplication 0 on all six); Core CPU 410201 / 30 (8 groups), GPU 53121 / 0 (4);
Estimation CPU 8834 / 2 (6), GPU 55 / 0; Eval CPU 2913 / 0 (6), GPU 359 / 0; docs of the three green. The docs pass: the two new methods' docstrings (signature
lines checked by Core's `docs` item, green in CPU group 8), Estimation's lens-regression comment, both CHANGELOGs; the
landing page names no LM internals (unchanged).
IO's tree is main's (gated with IO 0.9.5), Features' the 0.8.4 tree gated at 3d8203a4.

The pair, RC's side (check_duplication of RadiometricCamera with the monorepo as its stack path), verbatim:

BEFORE (job 14318; RC 35c9172ba, monorepo 252dbca9):
```
== check_duplication(/raid/data/jpr/rc080-dev-35c9172b): 2 finding(s)
   src/calibration_pipeline/linear_solve.jl:298  [similar] VgtkCore.residual_cost! (76 tokens; Jaccard 0.67)  ⇔  VisualGeometryToolkitEstimation:src/fitting/refine_camera.jl:178 residual_cost! (74 tokens)  — a function body that restates another: keep one
   src/calibration_pipeline/linear_solve.jl:302  [similar] VgtkCore.residual_normal_equations! (94 tokens; Jaccard 0.64)  ⇔  VisualGeometryToolkitEstimation:src/fitting/refine_camera.jl:182 residual_normal_equations! (92 tokens)  — a function body that restates another: keep one
```
The unchanged RC against Estimation 0.10.3 (job 14319; RC 35c9172ba, monorepo bfb7f7ed): Estimation's half of the pair is
Core's call, so the pair is gone from RC's side before RC changes:
```
== check_duplication(/raid/data/jpr/rc080-dev-35c9172b): 0 finding(s)
```
AFTER (job 14322; RC 2d074aefe calling the helper, monorepo bfb7f7ed):
```
== check_duplication(/raid/data/jpr/rc080-after-2d074aef): 0 finding(s)
```

---

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
