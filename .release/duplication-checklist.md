# 0.9 duplication checklist (monorepo)

Every finding of the 0.9 concept-level duplication audit (~/logs/dgx-agent/dup-audit-monorepo.md items 1–15, the
hoist asks of the downstream audits) and the branch that closes it. The owner's ruling: ONE release, no kept copies.
A box is ticked only when its branch is in the final RC and the final gate is green on it.

Final RC: the version bump 1d816a22 plus this file's commit (no code). Final gate: results in docs-pass.md section 3 once it reports.

## Fixed on `dup-audit-monorepo` (2cae3a7c, landed 2f0e6155)

- [ ] The 25 deleted copies of the audit's "Fixed" table: run-key hash, git reader, scratch flag, marker/record names,
  mount_dir join, tier-path join, failed-leg predicate, refine keyword parsing, least-squares refusal, strategy
  constant, TOML default strategy, scale source, recorded scale, Phase-2 LO, edge rotation, LossWidth → TOML, SplineLens
  forward drop, pose finiteness, duplicated import, `lm_retcode`, `_r13_callee`, Irls/Triggs method pairs, PoseLib
  `_strategy_shape`, Features pixel centre, IO file digest.

## Audit items 1–15

| # | concept | branch (head) | in chain as | status |
|---|---|---|---|---|
| 1 | strategy efficiency constant (Estimation, PoseLib, Core predicate) | dup-items-a (c60fdb9b) | 202223b3 | [ ] |
| 2 | FitResult with its estimate mapped (`map_estimate`) | dup-items-a | 202223b3 | [ ] |
| 3 | retired-keyword sentinel (`RetiredKeyword`, Core) | dup-items-a + monorepo-hoists H7 | 202223b3 | [ ] |
| 4 | weighted Kabsch → Core `kabsch(Csponds(…; weights))` | dup-items-b (95ceb6a9) | b00d18db | [ ] |
| 5 | IrlsWeights normalisation (one rule `ρ′/ρ′(0)`; defect) | dup-items-b | b00d18db | [ ] |
| 6 | three fixed-scale robust refine loops → one `Triggs` LM; `Irls` refine refused; the hand-eye LM on the bundle's one solve | dup-item-6 (13e3e09a, 0773a8d1) | ec490c8a, 2b8a1283 | [ ] |
| 7 | L2 chart LM on the pixel rows (`_triggs_state`; defect) | dup-items-b + dup-item-6 13e3e09a | ec490c8a | [ ] |
| 8 | SO(3) chart covariance → Core `transform_covariance` | dup-items-a | 202223b3 | [ ] |
| 9 | `_spline_record` → Core `worst_retcode` | dup-items-b | b00d18db | [ ] |
| 10 | pair norms → Core `point_norms!` | dup-items-a | 202223b3 | [ ] |
| 11 | twin fit metadata records (one `FitMetadata`; the three retired names refused by `check_naming`) | dup-items-11-14 (3a086ce7) | 7a9ebb99 | [ ] |
| 12 | within-file twins (Eval reuse rule, K₀₂/fx₀ + domain cap, Hartley maps, scoring workspaces, ONNX arena growth, R13 TOML walk) | dup-items-a | 202223b3 | [ ] |
| 13 | ci: sbatch option tail, docs.sbatch pieces, `VGTK_DOCS_LOCAL` reader | dup-items-a | 202223b3 | [ ] |
| 14 | Eval partial overlaps M1–M7 (one copy each); the dependency-free runner's unknown-key refusal stays (it cannot load Core's `check_keys`: an owner ruling, recorded in Eval's CHANGELOG) | dup-items-11-14 (3a086ce7) | 7a9ebb99 | [ ] |
| 15 | median seed (recorded), `_irls_record` vs refine Irls record (moot: Irls refines refused, item 6), PoseLib `_assert_unit_bearings` | dup-items-a + dup-item-6 | ec490c8a | [ ] |

## Downstream audits' monorepo asks

| ask | concept | branch (head) | in chain as | status |
|---|---|---|---|---|
| BB 5 | prefix-sweep loop body (`prefix_sweep`) | dup-items-e (35be7a8c) + integrator fix 04796694 | 04796694 | [ ] |
| BB 6 | IRLS termination record (`irls_termination`) | dup-items-e | 166a986f | [ ] |
| BB 7 | renamed-key refusal (`check_keys(; renamed)`) | dup-items-e | 166a986f | [ ] |
| BB 8 | artifact install (`install_artifact`, `huggingface_token` in Core) | dup-items-e | 166a986f | [ ] |
| F1, F7, F8, F10 | `is_stationary(blocks, …)`, `rotation_center_pose`, `pooled_noise_scale(scales, dofs)`, `lens_coverage(camera, rim_margin)` | monorepo-hoists (f12bd8a0) | 47b72b04 | [ ] |
| F9 | CountedLO per-call `call_records` | dup-items-f (591de3d7) | 52b9762a | [ ] |
| H1–H10, H12–H14, H16, H17 | hoists (scoring constructors, TOML reader, device buffers, `fetch_data!`, radix buffers, `reproject_so3`, `RetiredKeyword`, uncertain `inv`, Student-t default, `rms(residuals, n)`, `bearing_csponds`, Features `propagate`, retired names, `reset_posterior!`) | monorepo-hoists | 47b72b04 | [ ] |
| H11 | one launch policy: `OnBackend(backend; workgroupsize)(kernel)` (DUPLICATION.toml `[[backend]]`) | launch-policy (91f620ad) | 440168ef | [ ] |
| H15 | χ² quantile (whitened units) | whitened-units (f9e35470) | 8300fbf5 | [ ] |

## Stopped with no code (searched, nothing to merge)

- H8 (same per-pair concept already one body), F9's first pass (superseded by dup-items-f), "one luma" (no Features
  defect: dup-items-f.md).

## Out of scope by the audit (not duplications)

Features ASIFT singular-covariance drop (F6 ruling), `is_stationary`'s gradient form, `pooled_noise_scale` (F8),
CountedLO's log hook (F9), the lens-circle hoist (F10).
