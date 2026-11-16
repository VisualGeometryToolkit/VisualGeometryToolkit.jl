docs-pass: ad999834
landing: VisualGeometryToolkit/visualgeometrytoolkit.github.io 6674c98 (no change: it names no LM verdict and no version)

# VisualGeometryToolkitCore 0.11.0 and its monorepo cascade docs review (julia-docs-pass) — branch core-0110-term

Released: Core 0.11.0 (breaking), Estimation 0.12.0 (minor, carrying 0.11.9's exact resample), IO 0.10.6, Features
0.9.7, Eval 0.9.2, PoseLib 0.3.1 (patches). Baseline: origin/main 3416bb16 (Core 0.10.11, IO 0.10.5, Estimation 0.11.9,
Features 0.9.6, Eval 0.9.1, PoseLib 0.3.0). The review covers the tree ad999834. After it, only this record changes
(`.release/` is outside the gate).

## 1. Public API since the baseline

- Core: one method added to a public name, `is_stationary(ews::ManifoldLmWorkspace, lm, x, stops = LmStops())` (the
  chart LM's stationarity postcondition at a point). No name added, removed or re-signatured otherwise. The meaning
  of the host LM's termination record changes (owner ruling, 2026-11-15, and the desktop's rulings of 2026-11-16): an
  `ftol`, `xtol` or `xtol_rel` stop at a FINITE step is `(Success, CostDecrease | IncrementSize)` without the
  stationarity postcondition; a step stop whose terminating trial is non-finite (a NaN wall) is `(Stalled,
  NonFinite)`, `converged = -2`. The rejection cap and a stop of no rule are unreachable through NLLSsolver 5.3.4's
  LM; `LmStops`' `stationarity_tol` and `residual_resolution` are read only by `is_stationary`. The device LM keeps
  its rule. `nlls_termination` is the one decider.
- Estimation: no name added or removed. `optical_axis` reads its own validity gate (converged, its regression
  stationary at the estimate, no fold inside the region) in the continuation and the fallback; `refine_camera`'s
  metadata gains `family.stationary`. Its fits on Core's host LM report the new verdict. 0.11.9's exact resample
  (`_exact_valid_domain`, `_exact_resample`) is carried unchanged.
- Every in-repo `[compat]` on Core is `"0.11"` and on Estimation `"0.12"` (package, test, docs, examples, validation
  and benchmark environments, `ci`). IO's fifteen byte-for-byte fixtures name generator 0.10.6.

## 2. Docstrings and pages brought up to date

- Core: `lm_core.jl`'s header, `LmStops` (the closing line; `stationarity_tol`, `residual_resolution`),
  `_LM_STALLED`, `nlls_stop_code`'s list, the `nlls_termination` table and paragraph, `lmsolve_chart!`'s success
  rule, both `is_stationary` docstrings, `static_lm` and `StaticLmStops`, the workspace's `r` field;
  `ReturnCode.Success` / `Stalled`; the estimators guide's "one LM" paragraph; `configs/default.toml` `[lm]`
  comments. CHANGELOG 0.11.0: "Breaking" (the finite-step rule, the migration, the two knobs) and "Added".
- Estimation: `refine_camera` (`family.stationary`), `optical_axis` (the hold rule, the fallback's gate),
  `refine_camera_and_poses`, `refine_robot_world_hand_eye`, the relative-pose bundle's comments, the radial API
  page's `optical_axis` paragraph. CHANGELOG 0.12.0 above 0.11.9's (unchanged).
- IO, Features, Eval, PoseLib: CHANGELOG entries only (no docstring or page states an LM verdict).

## 3. Gates

Every bumped package's docs build strict (no `warnonly`), and every item (`docs`, `README`, `quality`, `naming`,
`agent docs` among them) at depwarn=error and depwarn=yes, in the core0110-term gate at the final head. No README or
landing example changes.

docs-pass: dcd7c649

# VisualGeometryToolkitEstimation 0.11.9 docs review (julia-docs-pass) — branch patch-est0119

Baseline: Estimation 0.11.8, on origin/main 1d86fa8b (Core 0.10.11, IO 0.10.5). The review covers the tree dcd7c649.
After it, only this record changes (`.release/` is outside the gate). No code outside VisualGeometryToolkitEstimation.

## 1. Public API since Estimation 0.11.8

- No name added or removed, no signature changed, no configs/default.toml key. `fit_camera(SplineLens, source,
  region; …)` of a SplineLens source (the exact resample) returns a lattice whose `valid_domain` holds the region by
  Core's own `⊆`, a region on the source's lattice edge included; each edge exact where a float lands on it, else
  rounded outward by one step of K′'s map. New private helpers only (`_exact_valid_domain`, `_anchor_axis`,
  `_extreme_float`, `_low_holds`, `_high_holds`, `_axis_rect`). Core floor 0.10.10 in the package, test and docs
  environments, unchanged.

## 2. Docstrings and pages brought up to date

- `fit_camera`'s SplineLens-source paragraph (the domain statement: containment, the outward step, Core's `⊆` and K′
  unchanged), the radial API page's `fit_camera` paragraph, the comment above `_exact_resample`. CHANGELOG 0.11.9
  (Fixed: the domain; RadiometricCamera 0.11.3's exact start restored). No README change: the README names no
  exact resample or `valid_domain`.

## 3. Gates

The Estimation docs build strict (no `warnonly`), and every Estimation item (`docs`, `README`, `quality`, `naming`,
`agent docs` among them) in the patch-est0119 gate at the final head.

docs-pass: 8f15d3ed

# VisualGeometryToolkitCore 0.10.11, VisualGeometryToolkitIO 0.10.5 and VisualGeometryToolkitEstimation 0.11.8 docs review (julia-docs-pass) — branch patch-w2

Baseline: Core 0.10.10, IO 0.10.4 and Estimation 0.11.7, on origin/main a7cdde09. The review covers the tree 8f15d3ed.
After it, only this record changes (`.release/` is outside the gate).

## 1. Public API since Core 0.10.10 / IO 0.10.4 / Estimation 0.11.7

- Core: no name added or removed. `LmStops` gains the keyword and field `xtol_rel` (Ceres' parameter tolerance, `[lm]
  xtol_rel = 0.0`: none; ‖x‖ over the free variables, a chart LM's √N); `nlls_stop_reasons` names bit 31
  `:relative_increment_size`, and `nlls_stop_code` / `nlls_termination` read it as the step-size stop (a caller's
  callback value is handed on with that flag cleared). `projective_knots(n, p)` keeps its signature with a new rule
  for `p > 1` (`m = n − 2p` spans to the radius, `p − 1` knots past it at the pixel spacing) and refuses `n ≤ 2p`.
- IO: no name added or removed, no schema version moved. A projective vignetting record is read on Core 0.10.11's
  knots; one of `n ≤ 2p` control points is refused by the reader (`ArgumentError` naming the record, its count and
  degree, the minimum and the Core release). Core floor 0.10.11 in the package, test, docs and benchmark environments;
  the fifteen byte-for-byte fixtures name generator 0.10.5.
- Estimation: no name added or removed. `optical_axis` fits by order continuation; `metadata(r).family` gains
  `fitted_terms`; configs/default.toml gains `[optical_axis] start_radial_terms = 1`. Core floor 0.10.10 in the
  package, test and docs environments.

## 2. Docstrings and pages brought up to date

- Core: `LmStops` (the `xtol_rel` row), `nlls_stop_reasons` and `nlls_termination` (bit 31; the callback row excepts
  it), `projective_knots` (the rule and why, the minimum `2p + 1`), `RadialVignetting`'s projective chart and the
  batched API page's projective knots line; configs/default.toml `[lm] xtol_rel`. CHANGELOG 0.10.11: "Changed (stored
  projective records)" states the rule change for `p > 1`, the refusal of `n ≤ 2p` and why.
- IO: `load(AbstractVignetting, path)` (a radial profile's knots are its chart's rule; the refusal of `n ≤ 2p`) and
  `save(path, ::AbstractVignetting)` ("bit for bit" holds for a profile on its chart's own knots: the record holds no
  knots). The JSON Schemas are unchanged (the refusal is the reader's, not the validator's). CHANGELOG 0.10.5.
- Estimation: `optical_axis` (the continuation, the hold rule, `fitted_terms`, the fold refit as the fallback), the
  radial API page's `optical_axis` paragraph, configs/default.toml `start_radial_terms` with its derivation.
  CHANGELOG 0.11.8.

## 3. Gates

Core, IO and Estimation docs builds strict (no `warnonly`), and every item of the three packages (`docs`, `README`,
`quality`, `naming`, `agent docs` among them) in the patch-w2 gate at the final head. No README example changes: no
README names `LmStops`, `projective_knots`, `optical_axis` or a vignetting record.

docs-pass: fdf0418b

# VisualGeometryToolkitCore 0.10.10 and VisualGeometryToolkitIO 0.10.4 docs review (julia-docs-pass) — branch patch-core0110

Baseline: Core 0.10.9 and IO 0.10.3, on origin/main b38dd88b (Estimation 0.11.7). The review covers the tree fdf0418b.
After it, only this record changes (`.release/` is outside the gate). No code outside VisualGeometryToolkitCore and
VisualGeometryToolkitIO.

## 1. Public API since Core 0.10.9 / IO 0.10.3

- Core: one public name added, `PolygonRegion` (`VisualGeometryToolkitCore.PolygonRegion`, exported from `Primitives`),
  parametrized on its vertex ring (`PolygonRegion{V}`, field `vertices`). New methods only, no removal and no
  re-signature: `valid_domain` returns it for a lens that is not radially symmetric (`Union{Ellipse{Float64},
  PolygonRegion, WholePlane}`), and it joins the region algebra (`∈`, `⊆`, `∩`, `intersects`, `Rect2`, `dilate`,
  `pixel_mask`, `farthest_radius`, `view_overlap`, `extend_lattice`, `folded_pixels`), a `SimilarityMap`'s action,
  `Adapt.adapt` (a `RegionIntersection` part by part) and `GeometryBasics.Polygon(s)`.
- IO: no name added or removed. The region body gains the `polygon` kind; `JSON.lower(s, ::PolygonRegion)` and
  `JSON.lift` into it. No kind's schema version moves (no committed kind schema embeds the composite region body;
  `lens_coverage` stays leaf-only). Core floor 0.10.10 in IO's package, test, docs and benchmark environments.

## 2. Docstrings and pages brought up to date

- `PolygonRegion`: the vertex ring and its device form (adapted, membership inside a kernel; the other operations
  host-side).
- `valid_domain` and `_fold_polygon`'s note: beside a jump of the fold radius between two samples (`_FOLD_JUMP`, 10 %
  between neighbours, its derivation beside it) a vertex is capped at the nearer neighbour's fold and kept outside the
  disc of the smallest fold; a discontinuity narrower than one sample's spacing that both neighbours miss cannot be seen.
  The CHANGELOG's Fixed entry says so (an overclaim at a fold-radius jump, and vertices inside the disc there).
- `valid_domain`, `angle_of_view` (the polygon is pulled in by at most its chords' sag, so the answer can understate
  rather than overstate), `check_region_support` and `farthest_radius` (their region lists name the polygon),
  `extend_lattice` and `folded_pixels` (the polygon is a lattice region), `SimilarityMap`'s region action (the polygon,
  vertex by vertex), `RegionBeyondSupport`'s advice (`with_lens_coverage` for a Rect or an Ellipse support only),
  the internal comments of `_valid_domain_radius` and `view_angle_limit` (the inscribed radius now serves the symmetric
  disc and the spline sampler).
- Core guide `lenses.md`: an Ellipse for a radially symmetric lens; the polygon otherwise, moved by a pixel-frame map
  (IO writes it as a `polygon`) and to a device.
- IO: the region body's docstring (`JSON.lower`/`JSON.lift` on `VgioStyle`), the JSON Schema `region` definition, and
  the kinds guide's region row.
- CHANGELOGs: Core 0.10.10 (heading released; Added: the polygon's lattice, similarity and device forms; the IO 0.10.4
  dependency of a record that stores an imaged region), IO 0.10.4 (Added, Fixed, Compat).

## 3. Gates

Core and IO docs builds strict (no `warnonly`; only the size warnings Documenter does not count as errors), each
package's `docs`, `README`, `quality`, `naming` and `agent docs` items: run 19798 at ba59240f (the same tree as
cf6fd13e), Core green, IO one refusal case that still read `polygon` as an unknown kind, fixed in 4cad0dc0; the gate's first
run (4f969046) found the fifteen byte-for-byte fixtures still naming generator 0.10.3, re-recorded in 9aad1a9c ; the
review of finding 1 (the fold-radius jump: red at e5dc4ce2, its green and the bit-for-bit check of the decentred OpenCV
polygons) added the lines above at fdf0418b (the gate reruns every item at the final head). No README example changes: neither README names the valid domain or the region
body.

docs-pass: 1f396e7c

# VisualGeometryToolkitEstimation 0.11.7 docs pass (julia-docs-pass) — branch patch-est0117

Baseline: Estimation 0.11.6 on origin/main 7e81b8e8 (Core 0.10.9 released). The pass covers the tree 1f396e7c; after it only
this record changes (`.release/` is outside the gate). No code outside VisualGeometryToolkitEstimation.

## 1. Public API since 0.11.6

No name added, removed or re-signatured. Behaviour: `optical_axis` refits a stalled refine inside the start's fold when
the region reaches past `_INSIDE_FOLD` (0.9) of the start's fold, not only when the start folds inside the region.
New private constant `_INSIDE_FOLD` (refine_camera.jl). VisualGeometryToolkitCore's floor is 0.10.9 (Project.toml, test/,
docs/).

## 2. Docstrings and pages brought up to date

- `optical_axis`: the start-fold band rule (a start folding past the widest angle but within 1/0.9 of it is refitted
  when its refine stalls; `start_fold` stays `nothing` for it; a stall from a start folding farther out is refused as
  before). The inline comments beside the refit loop and `_inside_fold` say the same.
- CHANGELOG.md: 0.11.7, a summary line and Fixed (the ag10fe case: 0.9·96.385° = 86.75° < 96.359°).
- README, the landing page and docs/src: no claim affected (docs/src/api/radial.md describes what `optical_axis` fits,
  not its refit rule; `fit_camera`'s docstrings name it only as the route to the axis).
- Prose checked for check_naming's banned words (R7, R14) by a grep of the diff: none.

## 3. Gates

The docs build (strict, `checkdocs = :public`, no `warnonly`) and the `docs`, `readme`, `quality` and `naming` items
run in the fresh-clone gate of the head that carries this record (scripts/patch-est0117). No dev runs: kaimon was down,
so all Julia went through that gate.

docs-pass: b798a0c3

# VisualGeometryToolkitCore 0.10.9 docs pass (julia-docs-pass) — branch patch-core0109

Baseline: Core 0.10.8 on origin/main 23c29974. The pass covers the tree b798a0c3. After it only this record changes
(`.release/` is outside the gate). No code outside VisualGeometryToolkitCore.

## 1. Public API since 0.10.8

No name added, removed or re-signatured. New methods of Base functions: `issubset(::RegionIntersection, ::Rect)` and
`(::RegionIntersection, ::Ellipse)` (split from one method), `issubset` with a `Circle` on either side. One private
helper, `_region_bounds` (regions.jl). Behaviour: an intersection lies in a `Rect` when its bounding box does (compared
at the parts' own bounds), in an `Ellipse` when its box or either part does; `Rect ∈ a ∩ b` is `Rect ⊆ a ∩ b`;
`canonical_json` refuses `typemin(Int64)`.

## 2. Docstrings and pages brought up to date

- `RegionIntersection`: a containment paragraph. Never a false `true` (to rounding, and `Ellipse ⊆ Ellipse`'s 8 eps
  slack); into an intersection as exact as its two tests; into a `Rect` or an `Ellipse` sufficient, not exact (the
  meet of the parts' boxes can reach out where the set does not, and an empty intersection makes no claim).
- regions.jl's containment comment block: the same rule, and why the box is not rebuilt as `origin + widths`.
- `canonical_json`: its docstring already says "an `Int64` past ±2^53" is refused, which is now true for −2^63 too.
- `check_region_support`'s docstring ("Core's one region containment"): unchanged, still true.
- CHANGELOG.md: 0.10.9, Fixed (the containment with RadiometricCamera's 14 mm case, the `Circle` part, the
  `typemin(Int64)` bound).
- README, the landing page and docs/src: no claim affected (none states the containment rule or the integer bound).
- Prose checked for check_naming's banned words (R7 and the rest) by a grep of the diff: none.

## 3. Gates

The docs build (strict, `checkdocs = :public`, no `warnonly`) and the `docs`, `readme`, `quality` and `naming` items
run in the fresh-clone gate of this tree. Dev runs (Julia 1.13.3, depwarn=error): red 19161 (at cc613413: 5 failed, 1
errored, all in the new assertions), green 19182 (at b798a0c3: 9345 of 9345, the region, bounded-camera, image-circle
and canonical items).

docs-pass: 744a5758

# VisualGeometryToolkitEval 0.9.1 docs pass addendum — branch patch-eval091

744a5758 rewords one source comment in `produce` (check_naming R7, a banned word). No docstring, guide or code changed.

docs-pass: 5d94948c

# VisualGeometryToolkitEval 0.9.1 docs pass (julia-docs-pass) — branch patch-eval091

Baseline: Eval 0.9.0 on origin/main 5de83b93. The pass covers the tree 5d94948c; after it only this record changes
(`.release/` is outside the gate). Two fixes, no code outside VisualGeometryToolkitEval.

## 1. Public API since 0.9.0

No name added, removed or re-signatured. The new helpers (`_lower_integers`, `_key_path`, src/code_identity.jl) are
private. Behaviour: the content key (`run_key`, the runner's stage key; `tuning_hash` through the same function)
lowers every integer to `Int64` and refuses one past ±2^53 by its key path; `produce` reuses a `produced` product
whatever its output's kind (a directory output was run again on every call).

## 2. Docstrings and pages brought up to date

- `_content_key` (private) and `run_key`: an integer of any type keys as its value (TOML's hex `0x0b10b5eed` as
  `2970312429`); `_content_key` states the ±2^53 refusal by key path.
- `produce`: "reused or produced"; nothing runs while `produced`, for a file and a directory output alike.
  products.jl's header, `Stage`'s docstring (stages.jl) and Experiments.jl's include comment no longer say
  `produce_or_load` loads: `produced` is the reuse rule, `produce_or_load` the producer.
- docs/src/paper-projects.md, the keyed-product bullet: the same correction.
- CHANGELOG.md: 0.9.1, Fixed (both).
- README and the landing page: no claim affected (no example keys a hex config or a directory output).

## 3. Gates

The docs build (strict, `checkdocs = :public`, no `warnonly`) and the `docs`, `readme`, `quality` and `naming` items
run in the fresh-clone gate of this tree. Dev runs (Julia 1.13.3, depwarn=error, a GPU each): red 18909 (integer, at
9f7602a3) and 18921 (directory output, at 093eed87); green 18913 (b8979f4e), 18922 (f80b8a0b), 18936 (5d94948c).

docs-pass: ff3639a0

# VisualGeometryToolkitEstimation 0.11.6 docs pass addendum — branch patch-est0116-on-core

ff3639a0 corrects the subdivision tie rule in the CHANGELOG and in `fit_camera`'s docstring: a near tie goes to the target's cell aspect (square only for one number), as `_resample_factors` does since b37eadec. No code changed.

docs-pass: 20295371

# VisualGeometryToolkitEstimation 0.11.6 docs pass addendum — branch patch-est0116-on-core

The strict docs build of the gate on 0072ad8c (18695) failed: `BearingResidual` and `ReprojectionResidual` shared
`AbstractViewResidual`'s docstring through `@doc (@doc …)`, which no `@docs` block included (2 missing docstrings, 3
unresolved `@ref`s). 20295371 gives each its own docstring and lists both in api/radial.md's `@docs` block; no code
changed. The gate is re-run on 20295371. Every other tier on 0072ad8c was green.

docs-pass: 0072ad8c

# VisualGeometryToolkitEstimation 0.11.6 docs pass (julia-docs-pass) — branch patch-est0116-on-core

Baseline: Estimation 0.11.5 (0005582b's record, docs-pass 3dc35e52), rebased onto patch-core0108 b1d5a020 (Core
0.10.8). The pass covers the gated tree 0072ad8c. After it, only this record changed (`.release/` is outside the gate).

## 1. Public API since 0.11.5 (additive, one keyword's behaviour changed)

- `fit_camera(SplineLens, source, region; lattice_spacing = nothing, support = nothing, …)`: a SplineLens source is
  resampled exactly (Core's `extend_lattice`, `subdivide_lattice`, then the optical-axis turn). `lattice_spacing` is
  the target cell in px. A `grid_size` with a SplineLens source is an `ArgumentError` naming `lattice_spacing` (0.11.5
  refused it too). The region may exceed the source's valid domain; a stated `support` is checked.
- `refine_camera_and_poses(camera, views::RotationCenterViews, correspondences, residual)`: `AbstractViewResidual`,
  `BearingResidual`, `ReprojectionResidual`, and `without_projection` (public).
- The family record of a resampled SplineLens source: `subdivision`, `lattice_spacing`, `extension_offset`,
  `extrapolated`, `extrapolated_fraction`, `folds`. The other results' `max_px`, `max_px_edge` and `rms` are `f·angle`.
- No name was removed.

## 2. Findings and static checks

| check | result |
|---|---|
| docstrings | `fit_camera`'s SplineLens section and its family record rewritten for the exact path; `AbstractViewResidual` (one docstring for the three types), `without_projection`, `optical_axis` (the fold refit, the refusal) |
| `checkdocs = :public`, doctests, cross-references | no new page; the names live in files already in api/radial.md's `@autodocs`. The strict docs build is in the gate |
| guide / API page | api/radial.md names the exact resample |
| README and landing page | unchanged: no example touches the changed API |
| check_naming | one R7 on 95342208 (a banned stem in the CHANGELOG, 18637), reworded in 0072ad8c; the test helper's banned keyword renamed in 8fcfe6e0's /simplify commit |
| CHANGELOG | `## 0.11.6`: Added, Fixed, Known, Changed (the `grid_size` refusal names `lattice_spacing`; a region past a SplineLens source's domain accepted; `support` defaults to `nothing`; the Core floor 0.10.8) |
| versions, floors | Estimation 0.11.6; Core compat "0.10.8" (package, test, docs) |

/simplify and /code-review high were run on the exact-resample delta and their fixes committed (d3f829b9 tests, red 18674;
b37eadec fixes, green 18675). The full release gate on 0072ad8c starts with prep 18691.

docs-pass: fe0b47a1

# VisualGeometryToolkitCore 0.10.8 docs pass addendum — branch patch-core0108

fe0b47a1 corrects one CHANGELOG sentence: the real-source fold check (ag30 over its sensor folds: 196 own + 191 past its domain; the fisheye 30 cm number came from a coverage ellipse the owner ruled out, item 1010). No docstring, guide or code changed.

docs-pass: 8a7d7a0f

# VisualGeometryToolkitCore 0.10.8 docs pass (julia-docs-pass) — branch patch-core0108

Baseline: GitHub main 10e39aaa (Core 0.10.7). The pass is of the gated tree 8a7d7a0f. After it, only this record
changed (`.release/` is outside the gate).

## 1. Public API since 0.10.7 (additive)

- `extend_lattice(camera, region; fold_spacing = 8.0) -> (; camera, extrapolated, offset, folds)` (public, not
  exported). A SplineLens lattice extended on its own spacing. Its extrapolation minimises the bending energy in pixels.
  `folds` is the fold report over the region; the function never refuses.
- `subdivide_lattice(camera, factors; extrapolated) -> (; camera, extrapolated)` and
  `subdivide_lattice(::ChartSpline, factors)` (public): exact integer knot insertion.
- `folded_pixels(camera, region; spacing = 8.0, reference = nothing) -> (; folded, samples, unimaged)` (public; a
  posed or bounded camera answers its model's): the orientation check of the backward map.
- `difference_rows(…; mixed = 0)`: mixed Δ₁Δ₂ rows at a weight, with `axes = (1, 2)` or alone with `axes = ()`. The
  default is bit for bit 0.10.7's.
- No name was removed or re-signatured.

## 2. Findings and static checks

| check | result |
|---|---|
| every public name has a docstring, attached, signatures matching | the three new names' docstrings list every method form and keyword. The `docs` items run in the gate (CPU and depwarn=yes) |
| `checkdocs = :public`, doctests, cross-references | new API section "Exact lattice operations" in api/spline_lens.md (`Pages = ["cameras/lattice_operations.jl"]`). The strict docs build is in the gate |
| README and landing page | unchanged: no example touches the new API |
| guide | lenses.md "Extending and subdividing a lattice" derives the subdivision's exactness (the two-scale relation `m⁻³(1 + … + z^{m−1})⁴`), the presence rule, and the extension's continuity, invariance, null space and uniqueness. It also covers the fold report and the start-values-only role of the penalty |
| docstrings | `extend_lattice`, `subdivide_lattice` (both forms), `folded_pixels`, `difference_rows` (`mixed`) |
| check_naming | one R7 finding on e95757a3 (a banned stem in `folded_pixels`' docstring), reworded in 8a7d7a0f. The 10e39aaa..HEAD diff holds no other banned or retired word |
| CHANGELOG | `## 0.10.8` (Added, Docs) |
| versions, floors | Core 0.10.8. The downstream floors are unchanged (an additive patch) |

## 3. Executed

Julia 1.13.3, fresh `git clone --shared`, nice 0, the per-agent depot `depot-core0108` (General a GitHub clone,
`JULIA_PKG_SERVER=""`). Gate on 8a7d7a0f: prep 18586, CPU 18587, CPU depwarn=yes 18588, GPU 18589, docs 18590. The
tier results are in the release report; the earlier gates d4107021 (18337–18341) and e95757a3 (18419–18423) were
superseded.

docs-pass: 884058c9

# VisualGeometryToolkitEval 0.9.0 docs pass addendum — branch patch-feat096

One docstring paragraph folded after the combined pass (fdbbcb3b): the `Product` docstring states the per-package code rule (a product is keyed by the packages its step uses). Nothing else changed since fdbbcb3b in Eval, Core or Features (`git diff 43a4a9b4 884058c9` = products.jl only). Re-run in a real clone: Eval prep 18173 all 0; docs_tests, agent_docs_tests, quality_naming 18174 17/17; strict docs build 18175 green.

docs-pass: fdbbcb3b

# VisualGeometryToolkitCore 0.10.7, VisualGeometryToolkitFeatures 0.9.6, VisualGeometryToolkitEval 0.9.0 docs pass (julia-docs-pass) — branch patch-feat096

Baseline: GitHub main 29d7492e (Features 0.9.5), main d40a948a and 0005582b merged (Estimation 0.11.4, 0.11.5: no Core,
Features or Eval change). The gated tree is fdbbcb3b; after it only this record and one sentence of Eval's CHANGELOG
(the atomic save is staged in a temporary directory, as the code does) changed; Eval's naming item re-run on that tree.

## 1. Public API since the last releases

- **Core 0.10.7**, additive: `value_variance(v::LinearRawView, i, j, x)` (public). Docs only: `Triggs`' docstring
  (Estimation 0.11.3's LO over the consensus set, a non-finite residual weighing 0, the weights fixed for the one LM).
  `_with_variance` lost a dead argument (internal).
- **Features 0.9.6**, additive: `polarity` on `detect_sfop` and the seeded `detect_foerstner`; `detect_sfop(L::LinearRaw;
  sensor_model)`; `detect_sfop(m::RawMosaic; …)` and `detect_foerstner([T], m::RawMosaic, seeds; …)` (the bilinear green
  image). No name removed.
- **Eval 0.9.0**, breaking: `Product`, `produce`, `Produced`, `produced`, `product_dir`, `product_output`, `load_output`
  added; `keyed_cell_dir`, `keyed_data_dir`, `experiment_dir`, `fresher`, `run_complete`, `cached_result`,
  `legacy_hash_key` and the 4-argument `Stage` refused by name (retired.jl); `results_root()`/`data_root()` DrWatson's
  `exp_pro`/`exp_raw`.

## 2. Findings and static checks

| check | result |
|---|---|
| every public name has a docstring, attached, signatures matching | the `docs` items green in all three suites (gate CPU and depwarn=yes) |
| `checkdocs = :public`, doctests, cross-references | the three strict docs builds green (Core 18069, Features 18084, Eval 18089) |
| README and landing pages | unchanged (no README example touches the new API) |
| guides | Features foerstner.md: the polarity section, duplicates left to the consumer, the seeded rows' claim corrected; Eval paper-projects.md: the keyed product (output hashes, atomic save, the `.jld2` saver), getting-started/protocol/tuning prose restored after a rename |
| docstrings | `detect_sfop`, `detect_foerstner` (seeded, mosaic), `value_variance`, `Triggs`, `Product`, `Produced`, `produce`, `produced`, `load_output`, `write_run_records` (`gitpath`) |
| check_naming | 0 in the three preps |
| CHANGELOGs | Core `## 0.10.7`, Features `## 0.9.6`, Eval `## 0.9.0` (Breaking, Known issues) |
| versions, floors | Core 0.10.7; Features 0.9.6, Core "0.10.7" (package, test, docs); Eval 0.9.0, Core "0.10.7", IO "0.10.3", Estimation "0.11.5" (package, test; its docs project names no stack package) |

## 3. Executed on fdbbcb3b

Julia 1.13.3, fresh `git clone --shared`, nice 0, the per-agent depot `depot-feat096` (General a GitHub clone,
`JULIA_PKG_SERVER=""`); GPU tiers on the B200s.

| package | prep (lint, agent_docs, floor --latest, audit, dup, naming, upstream dup) | CPU strict | CPU depwarn=yes | GPU | docs |
|---|---|---|---|---|---|
| Core 0.10.7 | 18065: all 0 | 18066, 16 groups green | 18067, 16 groups green | 18068, 6 groups green | 18069 green |
| Features 0.9.6 | 18080: all 0 | 18081, 6 groups green (data items run) | 18082, 6 groups green | 18083 green (+ Core warp) | 18084 green |
| Eval 0.9.0 | 18085: all 0 | 18086, 6 groups green | 18087, 6 groups green | 18088 green | 18089 green |

docs-pass: 3dc35e52

# VisualGeometryToolkitEstimation 0.11.5 docs pass (julia-docs-pass) — branch patch-est0115

Baseline: GitHub main d40a948a (Estimation 0.11.4). The owner's task: remove the duplicate `_zero_nonfinite_weights!`
(0.11.2) now that VisualGeometryToolkitCore 0.10.6 (in General) owns the posterior's non-finite rule. Core floor
"0.10.6" in the package, test and docs Project files.

## 1. The duplicate, site by site

| site | Core 0.10.6 covers it? | verdict |
|---|---|---|
| `consensus_posterior` (after `seed_posterior!`) | yes: `seed_posterior!` → `soft_posterior_weights!` → `_posterior_pair` over the same `sq_distances` buffer, every `i ≤ n`; non-finite → (0.0, 0.0), the values the post-pass wrote | duplicate: removed |
| `update_weights!(::PosteriorLikelihood)` (after Core's `update_weights!(pws, n)`) | yes: the same kernel, both robust weight and inlier posterior | duplicate: removed |
| `LossPlainWeights` (`_weighs`: every `Irls{K}`, `is_inlier`) | no: `weight(shape, …)` is not the posterior | kept |
| the Triggs LO's 0/1 weights (`_triggs_weights!`) | no | kept |
| the chart refit's zero-weight rows (`_ZeroWeightRows`) | no | kept |

## 2. Findings and static checks

| check | result |
|---|---|
| docstrings / comments | `consensus_posterior` (the rule is Core's E-step), `update_weights!(::PosteriorLikelihood)` (cites `soft_posterior_weights!`), the stale RSDR boundary note |
| guides, README, landing page | unchanged (they state the rule, not its owner) |
| tests | lo_nonfinite: a new item on the LO's posterior seeding at a NaN residual (green on 0.11.4's code, 17725, and on 0.11.5, 17726: the same 0.0) |
| check_naming | 0 in the prep (17727) |
| CHANGELOG | Estimation `## 0.11.5`: Changed (the floor, the removal; Core 0.10.6's own NaN result changes named) |
| version | Estimation 0.11.5; Core compat "0.10.6" (package, test, docs) |

Review findings recorded, not fixed: the non-posterior guards could move into Core's `weight(shape, …)` (a Core change;
then `_weighs` and `_triggs_weights!` would duplicate it in turn).

## 3. Executed on 3dc35e52

Julia 1.13.3, fresh clones, nice 0, the per-agent depot (`depot-est0115`):

| gate | job | result |
|---|---|---|
| prep | 17727 | lint 0, agent_docs 0, floor --latest 0, audit 0, check_duplication 0, check_naming 0, upstream duplication 0 |
| CPU (--depwarn=error) | 17728 | groups 1–10 green: 8,896 pass, 2 broken, 8,898 total |
| CPU (--depwarn=yes) | 17729 | groups 11–20 green: 8,896 pass, 2 broken, 8,898 total |
| GPU | 17730 | group 21 green: 66 pass |
| docs | 17731 | green |

"Nothing moves" (instrumented scratch clones, every chart refit, posterior-IRLS and Triggs LO call and every weight
refresh that met a non-finite residual; the suite and the LO bench): old d40a948a (17682/17683, again 17684/17685) vs
new 4520cbfa (17686/17687; its src differs from 3dc35e52 in comments only). 40,959 calls identical; the non-finite weight
refreshes the same on both sides (the lo_nonfinite policy items); the only unmatched calls are test group 9's homography
and TLS calls, as nondeterministic on the baseline itself (old vs old: 180 and 66; old vs new 208/78 and 164/64). Bench
bytes per call identical (1,725,206 / 1,765,189); µs within the instrumented runs' spread.

docs-pass: 0ca7ef45

# VisualGeometryToolkitEstimation 0.11.4 docs pass (julia-docs-pass) — branch patch-est0114

Baseline: GitHub main 5f02071c (Estimation 0.11.3), with main 57294039 (Core 0.10.6) merged at 0ca7ef45 (the gated tree)
and main 29d7492e (Features 0.9.5) at 3b189019 (Estimation and Core identical to 0ca7ef45; the whole-repo prep re-run
there). The owner-approved rule: a zero-weight row is exactly 0, in ONE place. Core's chart LM never reads √w
(`ManifoldLmWorkspace.sqrtw`, "filled by Estimation's refits"); every chart refit of a RANSAC problem's model — the LO's
`refine!` (`_refine!(::NonlinLsq)`), a `refine_*`'s least squares and its Triggs, the Triggs LO — runs Estimation's
`_chart_refit`, which fills √w and builds the state. The rule lives there.

## 1. Public API since 0.11.3

- **Added / removed:** nothing. Exports unchanged.
- **Behaviour:** every chart refit runs on `_ZeroWeightRows` (one state type per workspace: Core's kept LM run is never
  rebuilt): rows and Jacobian rows at a zero √w are exactly 0, a row at a nonzero √w the problem's own, bit for bit; a
  refit with no zero √w overwrites nothing (a flag found once per refit; nothing allocated).

## 2. Findings and static checks

| check | result |
|---|---|
| every public name has a docstring, attached, signatures matching | `docs` item green (gate CPU group 2, depwarn=yes group 12) |
| `checkdocs = :public`, doctests, cross-references | Estimation docs 17562: strict build green |
| README and landing page | unchanged |
| guides | ransac.md, M-step dispatch: "A zero-weight row is exactly 0" |
| docstrings / comments | `_chart_refit` (the rule; one state type), `_ZeroWeightRows` (its scope), the Triggs LO's `locally_optimize!` (its `0·Inf` note retired) |
| check_naming | 0 in the preps (17558, 17678) |
| CHANGELOG | Estimation `## 0.11.4`: Fixed (the zero-weight row rule; PnP and bearing pose rows already finite) |
| version | Estimation 0.11.4; Core compat unchanged ("0.10.0") |

Review findings recorded, not fixed (next patches): LocalOptimizationDoneRight's lift rows (`rows_provider`, unit √w,
weighed by LODR) and the LMs with their own states (`refine_camera`, the relative-pose bundle, `refine_camera_and_poses`)
are outside the rule; a state that overrides Core's cost or normal-equation reductions must not run through
`_chart_refit` (none does); `_zero_nonfinite_weights!` (0.11.2) now duplicates Core 0.10.6's `_posterior_pair` NaN
guard — kept, since removing it needs Core floor "0.10.6" and its own instrumented proof; the rows at a zero weight
are still evaluated before they are overwritten (a source-level skip would touch every problem's rows). The bearing
pose caller-weight fold was dropped: its rows are `√w·sw·e`, finite for every input the problem accepts, so the fold
changed nothing.

## 3. Executed on 0ca7ef45

Julia 1.13.3, fresh clones, nice 0, the per-agent depot (`depot-rows0`; General a GitHub clone, `JULIA_PKG_SERVER=""`):

| gate | job | result |
|---|---|---|
| prep | 17558 (3b189019: 17678) | lint 0, agent_docs 0, floor --latest 0, audit 0, check_duplication 0, check_naming 0, upstream duplication 0 |
| CPU (--depwarn=error) | 17559 | groups 1–10 green: 8,893 pass, 2 broken, 8,895 total |
| CPU (--depwarn=yes) | 17560 | groups 11–20 green: 8,893 pass, 2 broken, 8,895 total |
| GPU | 17561 | group 21 green: 66 pass |
| docs | 17562 | green |

Red → green: 17507 / 17538 (the final tests on 0.11.3's code: ellipse, the Triggs LO, hand-eye red; the PnP, bearing pose
and `refine_absolute_pose` guards green, "not reachable as red") → 17508, 17539 (green; 17539 with the allocation items
of ransac_ellipse/circle/rotation and p3p).

"Nothing else moved" (instrumented scratch clones: every chart refit with whether a zero-weight row is non-finite at the
anchor, every posterior-IRLS and Triggs LO call; the suite and the LO bench): old 57294039 (17544/17545, again
17546/17547) vs new 0ca7ef45 (17563/17564). Every chart refit and LO call of the existing suite identical (36,000+;
chart-ls and chart-robust of every problem, Triggs, Irls); the only new outcomes are the new item's calls (the ellipse,
hand-eye and Triggs-LO red cases — non-finite zero-weight rows — and the 3 PnP/bearing guards). Group 9's homography
and TLS calls are not run-to-run deterministic on the baseline (old vs old: 180 and 73 unmatched; old vs new 152/61 and
184/72). The instrumented clones fail the allocation items on both sides (the logger allocates).

LO bench, uninstrumented, pinned to one core, old/new alternated in one job (17661), µs per call over 3 rounds:
Welsch-IRLS old 364.8 / 383.4 / 373.0, new 371.6 / 391.0 / 374.8 (+1.4% mean, within the old rounds' 5% spread);
GM-IRLS old 330.9 / 322.1 / 325.9, new 324.6 / 323.6 / 329.3 (equal). Bytes per call 8,480 both.

docs-pass: b8e1f41c

# VisualGeometryToolkitFeatures 0.9.5 docs pass (julia-docs-pass) — branch patch-feat095c

Baseline: GitHub main 8f28fde7 (Core 0.10.5 as posted; Features 0.9.4). b8e1f41c is patch-feat095c: the element type
and the Float16 audit (b43cf9fb), the model-path image-circle red test (85d8273e) kept as `@test_broken` with the known
issue documented (3bdff6e8), the /code-review high fixes (b8e1f41c), with GitHub main 5f02071c and 57294039 merged (Core
0.10.6 and Estimation 0.11.3 as posted; no Features file changed). Owner ruling 2026-11-14: variant (c) — 0.9.5 ships
WITHOUT a fix of the image circle under a sensor model (rule A, the noise floor max(model, data), rejected; the
selection rule is its own later patch); the failure is documented and its item held `@test_broken`.

## 1. Public API since 0.9.4

- **Added:** the methods `load_raw(T, path; …)` and `load_raw(T, mosaic; …)`, `T` a `Base.IEEEFloat`, leading and
  positional (conventions/signatures.md). Names: unchanged (no export, no new public name).
- **Behaviour:** none at the default (Float64, bit for bit). `load_raw`'s cache key names `T` when it is not Float64.
  `box_downsample` of a Float16 image sums in Float64. `detect_foerstner(T, …)` refuses a `T` other than Float32 or
  Float64. `detect_image_circle` unchanged (its model-path failure documented).

## 2. Findings and static checks

| check | result |
|---|---|
| every public name has a docstring, attached, signatures matching | `docs` items green in gate CPU task 4 and depwarn=yes task 10 (`load_raw([T], path; …)`, `load_raw([T], mosaic; …)`) |
| `checkdocs = :public`, doctests, cross-references, link check | Features docs, gate task 14 (17611_14): strict build green |
| README and landing page | unchanged: neither shows an element type; the readme items green (tasks 4, 10) |
| guides | raw-images.md: "The element type" bullet (`load_raw(T, …)`, the default Core's, the cache key, the detectors' working precisions, reduce in Float64); detection.md: the known issue of the image circle under a sensor model (measure without one) |
| docstrings | `load_raw` (`T`, the cache key's type), `detect_foerstner` (`T` Float32 or Float64), `box_downsample` (Float16 summed in Float64), `detect_image_circle` (the known issue: halo or barrel-reflection light beyond the rim taken as the rim under a sensor model; the default path unaffected) |
| check_naming, check_duplication, upstream duplication | 0, 0, clean (prep 17607_0) |
| CHANGELOG | Features `## 0.9.5`: the element type, the cache key, the Float16 audit (two fixes, the audited sites), the known issue |
| version | Features 0.9.5, Core compat floor "0.10.5" (package, test and docs environments) |

Review (/simplify on the full 0.9.5 delta, /code-review high on it and on the (c) delta) findings recorded, not fixed:
`_DEFAULT_RAW_ELTYPE` read off a 2 × 2 dummy mosaic's `linearview` (Core holds the default privately: a public
accessor in Core is the fix, a wave-11 candidate); the path and mosaic `load_raw` bodies not merged; the demosaic's
gain and overflow checks run after the decode (an error path); the conditional cache key kept (pre-0.9.5 entries are
Float64). The image circle's model path: diagnosis and rule A on patch-feat095 (75787254), not shipped.

docs-pass: e548389b

# VisualGeometryToolkitCore 0.10.6 docs pass (julia-docs-pass) — branch patch-core0106

Baseline: GitHub main dc983c78 (Estimation 0.11.2 records; Core 0.10.4 as registered). e548389b is patch-core0106: the
red tests (d9e6f5d1), the fix (3bde6291), version and docs (10a7d20f), the /simplify (08c51602) and /code-review
(64266595) fixes, and the merge of GitHub main 8f28fde7 (Core 0.10.5 posted: the Float16 LinearRaw; no file of this
patch touched by it but the version, the CHANGELOG and this record). The bug (found reviewing Estimation 0.11.2):
Core's mixture kernel `_posterior_pair` gave NaN for a NaN squared residual, so the `fit_*` seeds built outside
Estimation's `consensus_posterior` and every E-step outside Estimation's guards made the whole posterior state NaN.

## 1. Public API since 0.10.5

- **Added / removed / exports:** nothing.
- **Behaviour:** a non-finite squared residual's inlier posterior and weight are 0 (`inlier_posterior`,
  `posterior_weight`, `soft_posterior_weights!`, `posterior_weights!`, `posterior_inlier_probs!`, and through them the
  state's E-step and seed), at every mixture; `median_seeded_posterior`'s median skips NaN residuals. Inf at a mixture
  with finite constants and every finite residual: unchanged bit for bit.

## 2. Findings and static checks

| check | result |
|---|---|
| docstrings | `inlier_posterior`, `posterior_weight`, `soft_posterior_weights!` (the non-finite rule), `median_seeded_posterior` (the NaN median), `Irls{GaussianShape}` in `Irls{K}`'s strategy list (the consensus refit set, since Estimation 0.11.2) |
| README, landing page, guides | unchanged: none states the mixture's per-residual rule (estimators.md names `is_map_inlier` only) |
| CHANGELOG | Core `## 0.10.6`: Fixed (the kernel's non-finite rule; the median seed), Changed (`Irls{GaussianShape}`'s docstring) |
| version | Core 0.10.6 |

Review findings recorded, not taken: `mixture_nll` and `posterior_weight_derivative` keep NaN for a NaN residual (a
finite NLL would let an NLL-driven incumbent accept an iterate whose residuals went NaN); `mad_scale` itself is
unchanged (its other caller, the sensor estimators, is outside this patch); Estimation's `_zero_nonfinite_weights!`
duplicates the rule once Estimation's Core floor is 0.10.6 (an Estimation change); a residual that turns NaN between an
E-step and the scale update still meets a nonzero weight there (the next E-step zeroes it).

docs-pass: ca3311df

# VisualGeometryToolkitEstimation 0.11.3 docs pass (julia-docs-pass) — branch patch-est0113

Baseline: GitHub main dc983c78 (Estimation 0.11.2, posted). ca3311df is patch-est0113: the red tests (c5d7c582,
reworked in a4778de6/b75d48ea), the fix (f415bb71), version and docs (6c017264), the /simplify (0c9827c6) and
/code-review (a4d5b600, ca3311df) fixes. The record's commit also corrects one test comment (no code). The change:
0.11.2's two Irls weight rules for the Triggs LO — the chart LM lives in Estimation (`_triggs_lm`, `_chart_refit`,
manifold_providers.jl), Core's `lmsolve_chart!` takes its weights unchanged.

Coordinator rulings 2026-11-14: the `0·Inf` row limit (a zero weight on a row non-finite in Float64) goes to the next
batch; Core's stale `Triggs`/`Irls` docstrings (`strategies.jl`) go to Core 0.10.6 if possible.

## 1. Public API since 0.11.2

- **Added / removed:** nothing. Exports unchanged.
- **Behaviour:** the Triggs LO (`locally_optimize!(L, ::Triggs, …)`) weighs a datum whose seed residual is not finite
  0, under every shape, and `Triggs{GaussianShape}` weighs only the scorer's consensus set (`check_fresh`, then
  `consensus_mask!`); the weights are fixed at the hypothesis for the one LM; with neither rule excluding a datum the
  LM is unchanged (unit weight, `weights = nothing`). `refine_*`'s Triggs is unchanged.

## 2. Findings and static checks

| check | result |
|---|---|
| every public name has a docstring, attached, signatures matching | `docs` item green (gate CPU group 2, depwarn=yes group 12) |
| `checkdocs = :public`, doctests, cross-references | Estimation docs 17391: strict build green |
| README and landing page | unchanged: nothing they show moved |
| guides | ransac.md: a `Triggs{K}` section (the two rules, fixed at the hypothesis; `0·Inf`) |
| docstrings | `locally_optimize!(::Triggs)` (the rules, by reference to Irls's; the weights fixed at the hypothesis; `0·Inf`), `_triggs_lm`, `_triggs_weights!`, `_weighs`, `_lo_consensus` |
| check_naming | 0 in the prep (17387); two R7 hits in this delta's prose caught in development (17358) and fixed |
| CHANGELOG | Estimation `## 0.11.3`: Fixed (the non-finite rule for Triggs, the Gaussian Triggs consensus rule, unchanged otherwise) |
| version | Estimation 0.11.3; Core compat unchanged ("0.10.0") |

Review findings recorded, not fixed: a row non-finite in Float64 stays NaN at weight 0 (`0·Inf`, built by each
problem's `sqrtw·e` rows) — a property of every chart refit (Irls's included), whose general fix (a zero-weight row
exactly 0) belongs in `_chart_refit` or Core's row adapter and changes `refine_*` too; `refine_*`'s Triggs keeps unit
weight (no consensus set; its non-finite seeds are a separate decision); Core's `Triggs` docstring
(`strategies.jl:100, 122`: "the plain chart LM over every datum") is stale — routed with 0.11.2's `Irls` docstring to
Core 0.10.5; `_triggs_weights!` writes `posterior_w` (documented; a dedicated buffer is a struct change); the
`nothing` fast path's extra finiteness scan, and two `check_fresh` calls on the Gaussian path (kept: one rule per
strategy); no end-to-end Triggs test reaches a non-finite row (the circle's far points are degenerate Sampson rows):
the rule is tested on the LM's weights.

## 3. Executed on ca3311df

Julia 1.13.3, fresh clones, nice 0, the per-agent depot (`depot-est0113`; General a GitHub clone, `JULIA_PKG_SERVER=""`):

| gate | job | result |
|---|---|---|
| prep | 17387 | lint 0, agent_docs 0, floor --latest 0, audit 0, check_duplication 0, check_naming 0, upstream duplication 0 |
| CPU (--depwarn=error) | 17388 | groups 1–10 green: 8,883 pass, 2 broken, 8,885 total |
| CPU (--depwarn=yes) | 17389 | groups 11–20 green: 8,883 pass, 2 broken, 8,885 total |
| GPU | 17390 | group 21 green: 66 pass |
| docs | 17391 | green |

Red → green: 17266 (c5d7c582 on 0.11.2's code: Gaussian Triggs 0.519/0.528 px off — the outliers fitted) and 17301
(the final tests on 0.11.2's code, 10870d5c: 2 failed, 3 errored) → 17299 (b75d48ea, 24 pass) → 17384 (ca3311df,
165 pass, 2 broken).

"Nothing else moved" (instrumented scratch clones: every posterior-IRLS and every Triggs LO call logged; the suite and
the LO timing bench): old dc983c78 (17268/17269, again 17270/17271) vs new ca3311df (17392/17393). Identical:
Triggs-Welsch 148, Triggs-Cauchy 49, TLS 7,510, Welsch 540, GM 144, Gaussian 2, Huber 2 (8,395 calls). New only:
the new item's 2 Triggs-Gaussian calls (outliers present) and 2 Triggs-Cauchy calls. Test group 9 is not run-to-run
deterministic on the baseline itself (old vs old: 34 TLS calls unmatched; old vs new 24 and 42; none Triggs, none
Gaussian, none with a non-finite seed). The bench: outcomes identical, bytes per call identical; µs per call within
noise (Welsch 639/639/646, GM 566/567/586).

docs-pass: 15156a4b

# VisualGeometryToolkitCore 0.10.5 docs pass (julia-docs-pass) — branch patch-core-f16

Baseline: GitHub main aaf389a5 (Core 0.10.4). 15156a4b is patch-core-f16: the red tests (fe8672c7), the element type
(c65c1845), the /simplify (8dd638f0) and /code-review high (dc54751f) fixes, with GitHub main dc983c78 merged
(Estimation 0.11.2 as posted; no Core file changed: `VisualGeometryToolkitCore/` is byte-identical to the gated dc54751f).
Owner and coordinator rulings 2026-11-14: a purely additive patch, the default element type unchanged (Float64; the
flip to Float16 is a wave-11 row); the variance plane Float32 beside Float16 values (a Float16 variance underflows), and
the stored value's variance (the law's plus the store's spacing²/12; 0 for Float64); a gain that could overflow the type
refused; the type a leading positional per conventions/signatures.md (`LinearRaw{T}` is taken by the mode).

## 1. Public API since 0.10.4

- **Added:** the methods `LinearRaw(T, mosaic; mode, gain, sensor_model)` and `linearview(T, m; sensor_model)`, `T` a
  `Base.IEEEFloat`. Names: unchanged (no export, no new public name; the default is the private `_LINEAR_RAW_ELTYPE`).
  Removed: nothing public (`_ValueCovariance` was internal).
- **Behaviour:** none at the default (Float64 values and variances bit for bit 0.10.4's). `linear_quantum` of a Float16
  view is the coarser of one code and the stored value's spacing (Float32 and Float64: the code's, bit for bit). A
  Float16 or Float32 uncertain image's variance includes spacing²/12. `LinearRaw(T, …)` refuses a gain at which a value
  could exceed `floatmax(T)`.

## 2. Findings and static checks

| check | result |
|---|---|
| every public name has a docstring, attached, signatures matching | `docs` items green in CPU group 9 and depwarn=yes group 25 (`LinearRaw([T = Float64], mosaic; …)`, `linearview([T = Float64], m; …)` as the methods) |
| `checkdocs = :public`, doctests, cross-references, link check | Core docs 17227: strict build green |
| README and landing page | unchanged: neither shows a `LinearRaw` or `linearview` (no element-type example to add); the readme items stand as run |
| guides | raw-images.md: a new section, "The element type: Float16 values", deriving the Float16 budget on the FX3 (step `d/2048 ≈ 7.75` codes against a shot noise of 26.5–37.6 codes; rounding ≤ 0.71% of the shot variance; 3.3 digits in the darks), the variance plane (why not Float16; the storage term with the FX3 evidence: K 11.18 vs 11.30 e⁻/code without it, ≈ 22× the counts SE, measured/expected 0.98–1.03, 11.297 with it), the overflow refusal and the Float64 reduction rule. No other guide states a RAW element type |
| docstrings | `LinearRaw` (the image's element type, "The element type" section with the derivations), `LinearRaw(…)` (the type argument; the green-cell value of a non-Float64 image is the Float64 mean rounded once), `LinearRawView` (`Uncertain{V}`), `linearview` (`T`, `Uncertain{T}`), `linear_quantum` (the stored value's resolution), `green_plane` (the image's element type) |
| check_naming | 0 in the prep (17223); the test's `step` identifier renamed before the first run |
| CHANGELOG | Core `## 0.10.5`: Added (the type argument, the variance plane's type and storage term with the FX3 evidence, the overflow refusal, `linear_quantum`), Unchanged (the estimators read the codes) |
| wave 11 | `.release/wave11-removals.md`: the default's flip to Float16, with the default's other homes (the docstrings' `[T = Float64]`, `raw_provenance`'s element type, Features' `load_raw` cache key) |
| version | Core 0.10.5 |

Review (/simplify, /code-review high) findings recorded, not fixed: testitems 2 and 3 kept (the brief's derived-bound
tests, a consumer's statistics over Float16 storage); `_LINEAR_RAW_ELTYPE === Float64` kept as the guard a flip must
change deliberately; `_store_variances!` keeps its recursion (the planes are a heterogeneous tuple, unrolled for the
GPU); `IsotropicCovMat` not used for the covariance type (the frame parameter would be restated); `raw_provenance` does
not record the element type (a new key could break downstream records: wave 11); `copy(::LinearRawView{<:Attributed})`'s
two broadcasts predate the patch.

## 3. Executed on dc54751f (Core identical at 15156a4b)

Julia 1.13.3, fresh clones, nice 0, the wave stack paths:

| gate | job | result |
|---|---|---|
| Core prep | 17223 | lint 0, agent_docs 0, floor --latest 0, audit 0, check_duplication 0, check_naming 0, upstream duplication 0 (Core, IO, Features) |
| Core CPU (--depwarn=error) | 17224 | groups 1–16 green: 414,164 pass, 34 broken, 414,198 total |
| Core CPU (--depwarn=yes) | 17225 | groups 17–32 green: 414,164 pass, 34 broken, 414,198 total |
| Core GPU | 17226 | groups 33–38 green: 54,985 pass (the new CPU = GPU item ran: group 33, 0 broken) |
| Core docs | 17227 | green |

Development runs (snapshots of the worktree): 17123 (red: 5 errors, MethodError and UndefVarError), 17125 and 17126
(green; the GPU item's `m`, Core's metre, renamed), 17203 and 17204 (after the review, CPU and GPU green); 17145 the
Float16-default configuration (13 existing items fail only their bit-for-bit Float64 assertions; no exception).
Downstream measurement (scratch only, read-only on RC 0.10.0 4fa0f62a): 17142, 17156 (FX3), 17228 (RC observations),
17229, 17230 (Features detectors).

---

docs-pass: e20a176e

# VisualGeometryToolkitEstimation 0.11.2 docs pass (julia-docs-pass) — branch patch-est0112

Baseline: GitHub main aaf389a5 (Core 0.10.4 records; Estimation 0.11.1 as registered). e20a176e is patch-est0112: the
red tests (4160619d), the fix (7002873d), version and docs (9e2161cb), the /simplify (5d43e2ef) and /code-review
(e20a176e) fixes. The bug (RadiometricCamera's board-space index): `Irls{GaussianShape}` weighed every datum 1, so one
non-finite residual made Σwᵢrᵢ² infinite at the seed and every refit and the local optimization returned its seed.

## 1. Public API since 0.11.1

- **Added:** `LossPlainWeights(rws, n, loss, consensus)` (a fourth field, `consensus::Union{Nothing, BitVector}`; the
  3-argument form is `consensus = nothing`). Exports: unchanged. Removed: nothing.
- **Behaviour:** a datum whose squared residual is not finite weighs 0 in `LossPlainWeights`, in `PosteriorLikelihood`'s
  E-step and in the LO's posterior seeding (`consensus_posterior`), and is no inlier (`is_inlier`);
  `Irls{GaussianShape}` refits the scorer's consensus set only (`consensus_mask!`). Robust shapes: only a NaN residual
  changes (its weight was NaN).

## 2. Findings and static checks

| check | result |
|---|---|
| every public name has a docstring, attached, signatures matching | `docs` item green (gate CPU group 2, depwarn=yes group 12); `LossPlainWeights`'s two forms in its docstring |
| `checkdocs = :public`, doctests, cross-references | Estimation docs 17190: strict build green |
| README and landing page | unchanged: nothing they show moved (no Gaussian LO example) |
| guides | ransac.md, Strategy `Irls{K}`: the two weight rules (a non-finite residual weighs 0; the Gaussian shape refits the consensus set) |
| docstrings | `LossPlainWeights` (both rules, the 4-argument form), `locally_optimize!(::Irls)` (both rules), `posterior_irls!` (the non-finite rule as it holds), `consensus_posterior`; `configs/default.toml` `[pipelines.bearing_pose]` comment |
| check_naming | 0 in the prep (17186) |
| CHANGELOG | Estimation `## 0.11.2`: Fixed (the non-finite rule; the Gaussian consensus rule; bearing_pose unchanged at its default scorer); the 0.11.1 heading's "(unreleased)" dropped (registered) |
| version | Estimation 0.11.2; Core compat unchanged ("0.10.0", the registered 0.11.1's) |

Review (/simplify, /code-review high) findings recorded, not fixed (Core's, routed to Core 0.10.5, or outside this
patch): the root of the non-finite rule is Core's mixture kernel (`_posterior_pair` gives NaN for a NaN residual), so
the `fit_*` seeds built outside `consensus_posterior` (`median_seeded_posterior`, `fit_residual_mixture`'s seed) and a
residual that turns NaN between an E-step and the scale update still read it; Core's `Irls{GaussianShape}` docstring
("plain least squares over all points") is now the consensus set's; `Triggs{GaussianShape}` (Estimation's chart LM over
every datum, as Core's docstring states) keeps outliers and non-finite rows — a separate decision; the extra O(n)
pass after each posterior E-step (fused into Core's kernel with the root fix); the seed zeroing that the frozen
policies overwrite; the second (identical) refit `StopWhenNoImprovement(1, 2)` runs under fixed Gaussian weights.

## 3. Executed on e20a176e

Julia 1.13.3, fresh clones, nice 0, the per-agent depot (General a GitHub clone, `JULIA_PKG_SERVER=""`):

| gate | job | result |
|---|---|---|
| prep | 17186 | lint 0, agent_docs 0, floor --latest 0, audit 0, check_duplication 0, check_naming 0, upstream duplication 0 |
| CPU (--depwarn=error) | 17187 | groups 1–10 green: 8,874 pass, 2 broken, 8,876 total |
| CPU (--depwarn=yes) | 17188 | groups 11–20 green: 8,874 pass, 2 broken, 8,876 total |
| GPU | 17189 | group 21 green: 66 pass |
| docs | 17190 | green |

Red → green: 17124 (4160619d, the tests on 0.11.1's code: 6 failed, 1 errored — the Gaussian LO threw on the
out-of-reach points and fitted the outliers 0.0486 off; NaN weights) → 17127 (7002873d, 409 pass) → 17172
(e20a176e, 365 pass, 2 broken).

"Nothing else moved" (instrumented scratch clones, every posterior-IRLS LO call logged; the suite and the LO timing
bench): old aaf389a5 (17128/17129, again 17165/17166) vs new e20a176e (17191/17192). 8,194 calls identical
(TLS 7,510, Welsch 540, GM 144); the GM/Welsch bench call set identical, bytes per call identical. The only new
outcomes are lo_nonfinite's own 3 calls. A TLS group (test group 9: currency_migration, fit_linear, line_fitting,
pool, ransac_homography, refine_robot_world_hand_eye, triggs) is not run-to-run deterministic on the baseline itself
(old vs old: 61 unmatched calls); old vs new there: 53 and 34, none with a non-finite seed or a Gaussian shape.
Outside the LO, no weight refresh met a non-finite residual in the old suite.

docs-pass: eb5efb43

# VisualGeometryToolkitCore 0.10.4 docs pass (julia-docs-pass) — branch patch-core0104

Baseline: GitHub main 300794c3 (Core 0.10.3). eb5efb43 is patch-core0104 (the Core 0.10.4 delta, its /simplify and
/code-review fixes, 0e7406e0) with GitHub main b0655d56 merged (Features 0.9.4 and IO 0.10.3 as posted, byte for byte;
Core's `conventions/units.md` persisted-form paragraph from IO 0.10.3), so the gate runs on the tree that is pushed.
Owner and coordinator rulings 2026-11-14: the exact-fit rule (the final point's rounding) and the accepted RC verdict
change; a non-finite operand scale gives no rounding floor; `lmsolve_chart!(p)`'s default stops take `p.damping`, only
an explicit mismatch refused.

## 1. Public API since 0.10.3

- **Added:** `photon_transfer(groups, darks::Darks; defects, block, normalize, significance)`; `LogPolarToCartesian`,
  `CartesianToLogPolar` public (not exported); `LmProblem`'s property `damping`; `[photon_transfer] significance` in
  `configs/default.toml`. Exports: unchanged. Removed: nothing (`_defect_mask` was internal).
- **Behaviour:** the LM exact-fit threshold is the final point's rounding, (16ε)²·Σsᵢ², floored by
  `residual_resolution`² (a restart at an exact fit is a `Success`; a stall below m·ε·F₀ is `Stalled`); a non-finite
  operand scale certifies nothing; an explicit `LmStops` of another damping than the problem's is an `ArgumentError`
  (the default stops take the problem's); the series `photon_transfer` reports its line's convergence
  (`IncrementSize`/`MaxIters`) and σ_r's standard error; one frame of fewer than 4 points is refused; `image_warp(T, …)`
  warns once per type for a link left in another precision.

## 2. Findings and static checks

| check | result |
|---|---|
| every public name has a docstring, attached, signatures matching | `docs` items green in CPU group 9 and depwarn=yes group 25; `lmsolve_chart!(p::LmProblem, stops = LmStops(; damping = p.damping))` in its docstring as the method |
| `checkdocs = :public`, doctests, cross-references, link check | Core docs 17014: strict build green; the log-polar types reach the manual through the transforms page |
| README and landing page | unchanged: nothing they show moved (no LM stop, damping or photon_transfer example); the readme items stand as run |
| guides | estimators.md: the `LmProblem`'s damping (its own; the default stops take it, an explicit mismatch refused) and the exact-fit rule (the final point's rounding, floored by `residual_resolution`²). raw-images.md, imaging-chain.md: no photon_transfer statement to change |
| docstrings | `LmStops` (`ftol` never the exact-fit test; `residual_resolution` a residual-vector norm), `LmProblem` (`damping`), `lmsolve_chart!` (the success rule), `_lm_stationary`, `StaticLmStops` (`residual_scale` the same floor), `photon_transfer` (both darks forms, the line's convergence, the refusals), `defect_mask` |
| check_naming | 0 in the prep (17010) |
| CHANGELOG | Core `## 0.10.4`: Added (the Darks method, the significance key, the public log-polar types), Fixed (σ_r's standard error, the line's convergence, the non-finite operand scale, the single frame's 4 points), Changed (the series' stop rule, the Dict method kept until wave 11, the damping's owner, the precision warning, the exact-fit rule with the accepted RC change, internal cleanups), and the 0.10.1/0.10.2 behaviour changes recorded |
| version | Core 0.10.4 |

Review (/simplify, /code-review high) findings recorded, not fixed: the Darks intercept is the plain mean over the sites
`defects` leaves (master_dark's ReadNoise drops its flagged sites; the caller supplies `defects`); the operand unit is 1
for a chart or a non-SVector variable (`EuclideanManifold`: the owner's rule, unchanged); the single frame tests μ²
before it reads the line's convergence (no frame that fails to converge was constructible); `_ptc_range` reads
unconverged intermediate lines; the metadata's level fraction uses the first group's black (equal blacks in every
fixture; per-band blacks need a fixture extension); the log-polar exemption covers every precision; the `_precision`
fallback's field walk runs per call; S² is accumulated when a stated `ftol` makes it unused.

## 3. Executed on eb5efb43

Julia 1.13.3, fresh clones, nice 0, the wave stack paths:

| gate | job | result |
|---|---|---|
| Core prep | 17010 | lint 0, agent_docs 0, floor --latest 0, audit 0, check_duplication 0, check_naming 0, upstream duplication 0 (Core, IO, Features) |
| Core CPU (--depwarn=error) | 17011 | groups 1–16 green: 414,074 pass, 33 broken, 414,107 total |
| Core CPU (--depwarn=yes) | 17012 | groups 17–32 green: 414,074 pass, 33 broken, 414,107 total |
| Core GPU | 17013 | groups 33–38 green: 54,822 pass |
| Core docs | 17014 | green |
| RC 0.10.0 (4fa0f62a) on Core eb5efb43, instrumented | 17015–17018 | 17,191 pass, 2 broken, 17,193 total; 398 postconditions, 1 verdict changed: termination_tests.jl:46, a restart AT an exact fit, Stalled → Success (the accepted change) |
| RC 0.10.0 (4fa0f62a) on registered Core 0.10.3 | 16940, 16944–16946 | 17,191 pass, 2 broken, 17,193 total |

Development runs (the worktree): 16870 (after /simplify), 16878 (red: the non-finite Jacobian, the default damping),
16903 (green).

---

docs-pass: f5e77eca

# VisualGeometryToolkitIO 0.10.3 docs pass (julia-docs-pass) — branch patch-io0103

Baseline: GitHub main 300794c3 (Core 0.10.3, IO 0.10.2, Features 0.9.3). f5e77eca is 10d5845e (IO 0.10.3 and Core's
`conventions/units.md` persisted-form paragraph) with the review's cleanups (86bc023e) and fixes (f5e77eca). Core stays
0.10.3 (its paragraph ships with its next release); IO's Core compat floor stays "0.10.3".

## 1. Public API since main

- **IO 0.10.3:** no public name added or removed; no schema version added. Behaviour: `same_value` (and the
  conversions' pose check) accepts a length with the same persisted identity; `vgio_migrate_tree` finds a file of any
  version a writer emits current; `vgio_migrate` of a `vgio/sensor_characterisation/v1.0` file refuses ("nothing to
  migrate"; the v1.0 file form deleted); `write_length` in a non-decimal unit refuses a length no number of the unit
  has the identity of (no length unit); the validator requires a darks body's darks in increasing band.

## 2. Findings and static checks

| check | result |
|---|---|
| every public name has a docstring, attached, signatures matching | `docs` items green (IO cpu group 4, both tiers) |
| `checkdocs = :public`, doctests, cross-references, link check | IO docs 16882: strict build green; kinds.md's `@eval` table rendered, its Written column `v1.0`, `v2.0` for `vgio/sensor_characterisation` |
| README and landing page | unchanged (nothing they show moved); the readme items stand as run |
| guides | IO: conventions.md (points at Core's "What is exact"), kinds.md (Written generated from `_written_versions`), schemas.md (a file of a written version is current). Core: `conventions/units.md` ("What is exact": the persisted form) |
| docstrings | `write_length` (the last resort per unit kind, the refusal), `same_value`, `vgio_migrate`, `vgio_migrate_tree`, `convert_matlab_calibration` (the persisted identity accepted) |
| published JSON Schemas | `sensor_characterisation/v2.0.json`'s rule states the darks' increasing band (description only) |
| check_naming | 0 on IO (prep 16879, check_naming and upstream_duplication) |
| CHANGELOG | IO `## 0.10.3` (Fixed, Docs, Tests) |
| versions | IO 0.10.3 (Core compat "0.10.3"); Core 0.10.3 unchanged |

## 3. Executed on f5e77eca

Julia 1.13.3, fresh clone, nice 0:

| gate | job | result |
|---|---|---|
| IO prep | 16879 | lint 0, agent_docs 0, floor --latest 0, audit 0, check_duplication 0, check_naming 0, upstream duplication 0 |
| IO cpu (--depwarn=error) | 16880 | groups 1–6 green: 73,098 pass, 9 broken, 73,107 total |
| IO cpu (--depwarn=yes) | 16881 | groups 7–12 green: 73,098 pass, 9 broken, 73,107 total |
| IO docs | 16882 | green |
| dev runs on 10d5845e (whole suite, 4 workers) | 16872 / 16873 (--depwarn=yes) | green: 73,099 pass, 9 broken each; 0 deprecation warnings |

---

docs-pass: c3f1d596

# VisualGeometryToolkitFeatures 0.9.4 docs pass (julia-docs-pass) — branch patch-feat094

Baseline: GitHub main 300794c3 (Features 0.9.3, Core 0.10.3; Features' Core compat floor "0.10.3", kept). The candidate:
22643f9e (all of Features 0.9.4: the thorlabs work, debaf972) and the review's commits 209121af, 22e3f9cd, 111b1afa
(tests, red on 22e3f9cd: dev 16875) and c3f1d596. No other package changes.

## 1. Public API since 0.9.3

- **Added (methods):** `detect_sfop(L::LinearRaw; kwargs...)`; the seeded `detect_foerstner([T = Float32], L::LinearRaw,
  seeds; sensor_model, kwargs...)`. Nothing exported or made public; nothing removed or renamed.
- **Changed:** `detect_foerstner`'s F-test at the residual sums' effective degrees of freedom (labels change where F lies
  between the critical values; positions and covariances do not); the seeded call refuses the image-wide search's
  keywords (`ArgumentError`, naming them) and checks each seed (a point seed an `ArgumentError` naming its frame; any
  uncertain scaled frame accepted); `detect_sfop`'s `nb_octaves` default `min(3, sift_nb_octaves(size(img)...; upsample))`;
  `detect_image_circle`'s deprecation text says "deprecated". `detect_image_circle`'s outputs are 0.9.3's bit for bit.
- **Scheduled (wave 11):** `gaussian_gradient`'s `derivative::Symbol` (`.release/wave11-removals.md`).

## 2. Findings and static checks

| check | result |
|---|---|
| docstrings against the code | `detect_foerstner` (seeded): the `[T = Float32]` LinearRaw form, the point seed's `ArgumentError`, every scaled frame also uncertain (fixed in c3f1d596); `detect_sfop`: the `nb_octaves` default and the LinearRaw form; `detect_image_circle` / `RimProfile`: the saturated bins flagged, not skipped (22643f9e) |
| `checkdocs = :public`, cross-references | docs 16918: strict build green (api/detection.md over `size_threshold_warn`, a size warning, as before) |
| README and landing page | unchanged: nothing they show moved (the landing names `detect_sfop` and `detect_image_circle`, both still true) |
| guides | foerstner.md: difference 4 (the F-test's degrees of freedom, 22643f9e) and the seeded form's keywords and LinearRaw forms (22e3f9cd); detection.md's image circle: nothing it states changed |
| check_naming | 0 (prep 16914, Core and Features; upstream duplication 16919) |
| CHANGELOG | `## 0.9.4` |
| version | VisualGeometryToolkitFeatures 0.9.4 (Core compat "0.10.3" in the package, test and docs projects) |
| `detect_image_circle` bit for bit | circle_dump on 22643f9e (16866) and 209121af (16869, the image-circle code's last change) `cmp`-identical to 0.9.3's reference (circle-cd3170c2.txt) |

## 3. Executed on c3f1d596

Julia 1.13.3, a fresh clone, nice 0 (~/logs/dgx-agent/runs/patch-feat094/c3f1d596), all rc 0:

| gate | job | result |
|---|---|---|
| prep | 16914 | lint 0, agent_docs 0, floor --latest 0, audit 0, check_duplication 0 (Core, Features), check_naming 0 |
| upstream duplication | 16919 | Features: 0 findings, check_naming 0 (name-ownership upstream: Core, Estimation) |
| CPU (--depwarn=error) | 16915 | 6/6 green: 3,588 pass, 13 broken, 3,601 total |
| CPU (--depwarn=yes) | 16916 | 6/6 green: 3,591 pass, 13 broken, 3,604 total |
| GPU | 16917 | green: 2,472 pass; Core warp_tests.jl 432 pass |
| docs | 16918 | green |

---

docs-pass: 7b322318

# VisualGeometryToolkitCore 0.10.3, VisualGeometryToolkitIO 0.10.2, VisualGeometryToolkitFeatures 0.9.3 docs pass (julia-docs-pass) — branch patch-core0103

Baseline: GitHub main 2bbd6e72 (Core 0.10.2, IO 0.10.1, Features 0.9.2). 7b322318 is patch-core0103 (from main) with
the desktop's imgcircle-jacobian a544bdad merged (its 4fcb52bb and the test-only a544bdad), so the gate runs on the
tree that is pushed. Owner rulings 2026-11-14: Darks (1), the writer's version follows the record (a).

## 1. Public API since main

- **Core 0.10.3 (additive):** `Darks` (public; a black-slot component, one Dark per disjoint exposure band, a shared
  channel black); `defect_mask(model_or_defects, mosaic; backend)` (public, was `_defect_mask`); `linear_quantum(view,
  i, j)` (public); `transform_deriv(::LogPolarToCartesian, p)`, `inv(::LogPolarToCartesian)` → `CartesianToLogPolar`
  (4fcb52bb); `sensor_fixture(; darks)`. A `SensorModel`'s black is a frame's (`FileBlack`, `ChannelBlack`, `Dark`): a
  `Darks` is refused there (it is a characterisation's). Removed: nothing. Exports: unchanged.
- **IO 0.10.2:** `vgio/sensor_characterisation/v2.0` (the darks body), written only for a record holding a `Darks`;
  every other characterisation is written as v1.0, IO 0.10.1's bytes; `json_schema(kind[, version])`. No public name
  removed.
- **Features 0.9.3:** no public API change; `detect_image_circle`'s rim radii through the log-polar map's differential
  and inverse (results move within their σ; `RimProfile.noise` per pixel). Its CHANGELOG re-recorded after the post of
  Core 0.10.3 / IO 0.10.2: the f/32 set is a real reference check again (a544bdad: the owner-verified fit, 2136.997,
  1439.331, 1333.768, 1315.245), not "marked broken" (CHANGELOG only; Features' manual does not include it).
- **Deferred to Core 0.10.4:** the Float32 `_precision` of the log-polar maps (it moved `detect_image_circle`: a flat
  image fit an ellipse), and `_precision`'s silent catch-all for float-carrying leaves.

## 2. Findings and static checks

| check | result |
|---|---|
| every public name has a docstring, attached, signatures matching | `docs` items green in all three suites (Core CPU group 9, IO cpu, Features cpu 4) |
| `checkdocs = :public`, doctests, cross-references, link check | Core docs 16568, IO docs 16571, Features docs 16575: strict builds green. `Darks`, `defect_mask`, `linear_quantum` reach the manual through raw.md's `@autodocs`; `CartesianToLogPolar` through the transforms page |
| README and landing pages | Core, Features: unchanged (nothing they show moved). IO's landing page states the written-versions rule. No example changed, so the readme items stand as run |
| guides | IO: kinds.md (v2.0, v1.0 written without a `Darks`), json-schemas.md (the published schemas are the written versions, generated from `_written_versions`), schemas.md ("Earlier versions": the one rule), CLAUDE.md (the convention). Core: none (the sensor model has no guide page; its API page is generated) |
| published JSON Schemas | `schemas/vgio/sensor_characterisation/v1.0.json` (regenerated, byte-identical to IO 0.10.1's) and `v2.0.json`; one per written version, the count test reading the same table |
| check_naming | 0 on every package of the tree and on ci/ (dev 16576); Core, IO and Features in their preps |
| CHANGELOGs | Core `## 0.10.3` (Added); IO `## 0.10.2` (Added: v2.0, the writer's rule); Features `## 0.9.3` (Changed; the 0.9.2 heading dated) |
| versions | Core 0.10.3, IO 0.10.2 (Core compat 0.10.3), Features 0.9.3 (Core compat 0.10.3 in the package, test, docs, validation and examples projects) |

## 3. Executed on 7b322318

Julia 1.13.3, fresh clones, nice 0, the wave stack paths:

| gate | job | result |
|---|---|---|
| Core prep | 16565 | lint 0, agent_docs 0, floor --latest 0, audit 0, check_duplication 0, check_naming 0, upstream duplication 0 (Core, IO, Features) |
| Core CPU | 16566 | groups 1–16 green: 414,027 pass, 33 broken, 414,060 total |
| Core GPU | 16567 | groups 17–22 green: 54,792 pass |
| Core docs | 16568 | green |
| IO prep / cpu / docs | 16569 / 16570 / 16571 | 8/8 green: 73,074 pass, 8 broken, 73,082 total |
| Features prep / cpu / gpu / docs | 16572 / 16573 / 16574 / 16575 | 9/9 green: CPU 3,258 pass, 13 broken; GPU 2,155 pass |

---

docs-pass: 19126b52

# VisualGeometryToolkitCore 0.10.2 docs pass (julia-docs-pass) — branch patch-core0102

Baseline: the Core 0.10.1 release b8cc82b6. GitHub main is still b8cc82b6, so no merge was needed and the gate ran on
the tree that is pushed. The release is additive (owner ruling, option (1), and the coordinator's ruling (A)).

## 1. Public API since 0.10.1

- **New names (public, not exported):** `LmDamping`, `MarquardtDamping`, `LevenbergDamping`; the `ModelReason` value
  `DarkOutOfBand`.
- **New keywords:**
  - `LmStops(; damping = MarquardtDamping(), residual_resolution = 0)`; `LmStops` gains the type parameter
    `{D <: LmDamping}` and the fields `damping` and `residual_resolution`;
  - `LmProblem(variables, blocks; fixed, damping = MarquardtDamping())`.
- **Changed behaviour (Fixed):** the success rule's exact-fit test is `F < max(m·ε·F₀, residual_resolution²)`, no
  longer `F < ftol`; `sensor_model` reports `DarkOutOfBand` where it reported `Matched`.
- **Removed:** nothing. **Exports:** unchanged.

## 2. Static findings

| check | result |
|---|---|
| every public name has a docstring, attached, signatures matching | `test_docs` green (CPU group 9: the docs item) |
| `checkdocs = :public`, cross-references | docs build green (16199). api/robust_statistics.html measured 206 KiB, over the 200 KiB limit (16140), so the chart rows and the LM core moved to a new page, api/lm_core.md "Chart LM" |
| check_naming | 0 on every package of the tree and on ci/ (dev 16132; prep 16196 for Core). ci/slurm/test_summarize.sh:3 had a pre-existing R7 word, now reworded |
| CHANGELOG | Core `## 0.10.2`: Fixed (the exact-fit test, measured on RC 0.9's suite); Added (`DarkOutOfBand`, the damping strategy, `residual_resolution`, the damping measurements and the rejected Moré scaling) |
| version | VisualGeometryToolkitCore 0.10.2 |

## 3. Executed on 19126b52

The gate ran Julia 1.13.3 on a fresh clone, at nice 0, with the wave stack paths. It finished 24/24 rc 0:

| job | result |
|---|---|
| prep 16196 | lint 0, agent_docs 0, floor --latest 0, audit 0, check_duplication 0, check_naming 0 |
| CPU 16197 | groups 1–16 green: 413,609 pass, 32 broken, 413,641 total |
| GPU 16198 | groups 17–22 green: 54,372 pass, 54,372 total |
| docs 16199 | green |

---

docs-pass: e5a6a83e

# VisualGeometryToolkitIO 0.10.1 docs pass (julia-docs-pass) — branch patch-io0101

Baseline: the registered IO 0.10.0 (b808396e, tree a66dc7e2). A patch: no public name added, removed or re-signatured; no
schema or version of any kind changes. 343a062d is patch-io0101 merged with GitHub main b8cc82b6 (Core 0.10.1 on Features
0.9.1), so the gate runs on the tree that is pushed.

- API diff since 0.10.0: none. Internal only: the writer's last-resort policies (`_PersistedLastResort`,
  `_NearestLastResort`) and IO 0.10.0's millimetre leaf form for the 0.9 identity reproductions (`_LEGACY_MM`).
- Docstrings: `write_length` states what is written when no file number reads back to the length (a number with its
  identity, its persisted form); lengths.jl's header the same.
- Guides: docs/src/guide/conventions.md ("Lengths cross the file boundary"), kinds.md (`vgio_migrate`'s comparison) and
  content-identity.md (normalisation) state that a full-precision length's file has the length's identity (IO 0.10.1).
- README and landing page: unchanged (nothing they show moved). The Core conventions text (VisualGeometryToolkitCore
  conventions/units.md, "at most 2 ulp on its first rewrite") is Core's: its sentence still holds for every value a file
  number reads back to; the last-resort case is IO's and documented here (no Core change in an IO patch).
- CHANGELOG: 0.10.1 (Fixed; the tests added and the three restated, each by name); 0.10.0 dated.
- Docs build, docs item, quality and naming: run by the gate on the merged sha.
- Re-recorded at 6ae7230c: an internal helper renamed (`_fixed_point` → `_rewrite_number`: the gate's check_duplication
  found `_fixed_point` owned by VisualGeometryToolkitCore.Cameras); no docs change.
- Re-recorded at e5a6a83e: the byte-for-byte fixtures' envelope names the generator 0.10.1 (the gate on 085b34ad: 16
  fixture comparisons differed only in `generator.version`); no docs change.

---

docs-pass: 90debb2a

# Features 0.9.2 docs pass (julia-docs-pass) — branch imgcircle-edge

A patch, additive: `detect_image_circle` measures the lens coverage on log-polar profiles of the frames' maximum (no
MSER, no hull). Baseline: Features 0.9.1 as released (92d07363). The candidate: ac99ec38 (ONE commit on GitHub main
b8cc82b6, Core 0.10.1), its full gate green (~/logs/dgx-agent/runs/feat092/ac99ec38, jobs 16105–16108), then this
pass's commits 677d58cf (docstring and comments) and 90debb2a (guide, landing, README).

## 1. Public API since the baseline

From the module's `export`/`public` lines and the methods (92d07363..90debb2a); nothing removed or renamed:
- `detect_image_circle([rng,] [T = Ellipse,] frames; outlier_halfwidth, initial, rim_band, rim_angles, significance,
  device)`: new keywords; the result gains `covariance`, `center_scale`, `axes_scale`, `rim`;
- public `RimProfile`, `SensorModelNoise`, `EmpiricalNoise` (documented, in api/detection.md's `Pages` through
  `image_circle_rim.jl`);
- deprecated, inert: `min_hole_area`, `smoothing_weight`, `border_margin` (a depwarn; removal listed in
  `.release/wave11-removals.md`);
- internal: the ImageSmooth extension and the DelaunayTriangulation dependency removed.

## 2. Static findings, fixed

| where | finding | fix |
|---|---|---|
| `detect_image_circle` docstring, step 3 | the candidate was "the run's outer end"; the code takes the steepest floored log drop from the run's start to the next edge's, at the span's middle refined to the sub-sample peak (as the CHANGELOG says) | stated as the code does (677d58cf) |
| `detect_image_circle` docstring, step 2 | the `:auto` binning's precision omitted the rms floors (the edges' median uncertainty, the radial grid's spacing over √12) | stated (677d58cf) |
| `detect_image_circle` docstring, `fit` | "on the candidates; `inlier_mask(fit)` indexes them": it indexes the second RANSAC's one-candidate-per-angle points, not `rim.candidates` | stated (677d58cf) |
| image_circle_rim.jl header, Estimation extension header (comments) | "largest significant drops", "through every angle's candidates"; "on its contour (step 6)" | the two one-point-per-angle fits; step 4 (677d58cf) |
| guide/detection.md, the image circle | step 3 "the largest significant drops", step 4 "through every angle's candidates"; "within half a pixel of the stop" (the suite holds < 1 px on the sharp stop of the soft-ramp item, < 0.5 px only on the fixture disc) | every envelope edge at its steepest log drop, the two fits, "within a pixel" (90debb2a) |
| index.md, README | the landing's detector list and the README's Detection link did not name the image circle | named (90debb2a) |

Checked and true: the guide section otherwise describes the new method (no MSER, no hull, no `min_hole_area`); the
`RimProfile`, `SensorModelNoise`, `EmpiricalNoise` docstrings against the code (fields and order, the empirical noise
mad_scale / √2, the profile_scale standard error and its one-frame form √(π/2·v̄/m_eff)); the module docstring and the
export comments; the CHANGELOG 0.9.2 entry. No other page, example (examples/loma.jl) or validation script names the
image circle's old method. README install line and example unchanged. check_naming 0 on 90debb2a.

## 3. Executed

Docs and docstrings only (no code line), so the docs tier on 90debb2a (scripts/feat092/submit_docs.sh, the gate's
gate.sbatch, Julia 1.13.3, nice 0; ~/logs/dgx-agent/runs/feat092/90debb2a), all rc 0:

| job | result |
|---|---|
| prep 16134 | lint 0, agent_docs 0, floor --latest 0, audit 0, check_duplication 0, check_naming 0 |
| items 16135 | agent_docs, docs, quality_naming, readme items: 25/25 |
| docs 16136 | strict build green (api/detection.md's size_threshold_warn, a size warning, as on ac99ec38) |

---

docs-pass: 78fa6599

# VisualGeometryToolkitCore 0.10.1 docs pass (julia-docs-pass) — branch patch-core0101

Baseline: the registered Core 0.10.0 (7b9bec70). This release is purely additive, option (B) of the owner's ruling. 78fa6599 is
patch-core0101 merged with GitHub main 92d07363 (Features 0.9.1, Estimation 0.11.1), so the gate ran on the tree
that is pushed.

## 1. Public API since 0.10.0

- **New methods and names:**
  - `photon_transfer(groups, darks; sensor_model, block, normalize, significance) -> FitResult{PoissonGaussian, PhotonTransferMetadata}`;
  - `PhotonTransferMetadata` (public);
  - `LmProblem(variables, blocks; fixed)`.
- **Changed, additive:** the single-frame `photon_transfer(mosaic; sensor_model, block, significance)` gains the
  `significance` keyword. It refuses a μ² term; its signature is otherwise unchanged, as is its return. It is
  deprecated, to be removed in the next breaking wave.
- **Removed:** nothing. **Exports:** unchanged. `PhotonTransferMetadata` is `public`, not exported.

## 2. Static findings

| check | result |
|---|---|
| every public name has a docstring, attached, signatures matching | `test_docs` green (CPU group 9: docs item) |
| `checkdocs = :public`, cross-references | docs build green (15740) |
| check_naming | 0 (prep 15737) |
| CHANGELOG | Core `## 0.10.1` (New, Added, Fixed, Deprecated, the measured FX3 limitation) |
| version | VisualGeometryToolkitCore 0.10.1 |

## 3. Executed on 78fa6599

The gate ran Julia 1.13.3 on a fresh clone, at nice 0, with the wave stack paths. It finished 24/24 rc 0:

| job | result |
|---|---|
| prep 15737 | lint 0, agent_docs 0, floor --latest 0, audit 0, check_duplication 0, check_naming 0 |
| CPU 15738 | groups 1–16 green |
| GPU 15739 | groups 17–22 green |
| docs 15740 | green |

---

docs-pass: 825a5e64
landing: VisualGeometryToolkit/visualgeometrytoolkit.github.io 6674c98 (no change: the Features card names the package's scope, not its detectors)

# Features 0.9.1 docs pass (julia-docs-pass) — rel-feat091

A patch, additive, on Core 0.10: SFOP scale selection and its parts. Baseline: VisualGeometryToolkitFeatures 0.9.0, the
registered tree (tag `VisualGeometryToolkitFeatures-v0.9.0` = 7b9bec70). The candidate: features-sfop-gate 0d36b9f7 (a
fast-forward of GitHub main b808396e, Features only), then this pass's commits 074d7c65 (CHANGELOG), 181f6ea6
(docstrings), a81ebf67 (guide), 825a5e64 (landing and API page).

## 1. Public API since the baseline

From the module's `export`/`public` lines and the methods (7b9bec70..825a5e64); nothing removed or renamed:
- exported `gaussian_gradient(img, σ[, mask]; derivative = :gaussian2d) -> (gx, gy)`, Förstner's one gradient;
- exported `gaussian_scale_space(img; …) -> GaussianScaleSpace` and the type, ASIFT's pyramid; public
  `level_sigma(space, o, l)`;
- exported `detect_sfop(img; noise_sigma, significance_level, precision_threshold, …)`;
- the method `detect_foerstner([T = Float32], img, seeds)`: the window estimation at given windows;
- `AsiftPlan`'s type parameters (`A1` gone, `SP` new) and the 8 undocumented fields now `plan.space`: not breaking,
  owner ruling 2026-11-14 (CHANGELOG line, 074d7c65);
- behaviour: `detect_foerstner` faster, results bit for bit (the squared gradient on the device, the window centres
  compacted there); the accepted r ≈ 1.8 px limitation of the SFOP-selected window.

## 2. Static findings, fixed

| where | finding | fix |
|---|---|---|
| `detect_foerstner` docstring | the row shape read `(; label, window, F)`, the rows carry `covariance_status` (since 0.9.0) | the shape names it (181f6ea6) |
| `detect_foerstner` docstring, step 2 | the gradient is now `gaussian_gradient` | linked (181f6ea6) |
| `GaussianScaleSpace` docstring | "one per tilted warp inside every `AsiftPlan`": a plan holds ONE space, a slot per resident warp | corrected (181f6ea6) |
| guide/foerstner.md | no word of `detect_sfop`, the seeded form or the limitation; "assembled on the host" no longer says what stays on the device | a section "Windows chosen by scale: SFOP and seeded windows" (the pipeline, bit-for-bit seeded rows, the accepted limit with what the suite holds), the device paragraph (a81ebf67) |
| index.md, api/detection.md | the detector lists stopped at MSER and Förstner | `detect_sfop`, `gaussian_scale_space`, `gaussian_gradient` named (825a5e64) |
| CHANGELOG 0.9.1 | the `AsiftPlan` change unstated | one line, with the ruling (074d7c65) |

Checked and true: the `detect_sfop`, `gaussian_scale_space`, `level_sigma`, seeded `detect_foerstner` and
`gaussian_gradient` docstrings against the code (signatures and defaults; the finest integration scale 3·2^(2/3)·0.5 ≈
2.4 px; the λ₂ floor; the NMS footprint σ/√2 and half-octave bands; the seeded window half-width; the step response
1/(√(2π)σ)); `sfop.jl` and `asift/pyramid.jl` are in api/detection.md's `Pages`; README (install line and example
unchanged, no new name needed); no timing in a docstring or page (the speed-up's numbers are the CHANGELOG's,
measured, not a gate).

## 3. Executed

On the commit carrying this record (records only on top of 825a5e64): prep (lint, agent_docs, floor, audit,
check_duplication, check_naming), Features' CPU and GPU groups (strict flags) and the docs build (strict); results
with job ids in the release report (a src docstring changed, so the full Features gate).

---

# Estimation 0.11.1 docs pass (julia-docs-pass) — patch-est0111-fixture

The owner's ruling (it blocks LocalOptimizationDoneRight 0.7.0): ONE public fixture for the two-camera F scene, and the
LO bench's arm binding in one place. Baseline: GitHub main 7b9bec70 (the registered wave: Estimation 0.11.0).

## 1. Public API since the baseline

None of the package's API. Test support, public to the stack's tests:
- `synthetic_scenes().EpipolarScenes.fundmat_scene(rng; n_inliers = 100, n_outliers = 30, σ = 0.5) -> (; source, target,
  F, n_inliers)`: THE two-camera F scene, on the family's `stereo_pixel_pairs`; docstring in the extension, listed in
  `synthetic_scenes`' docstring (docs/src/api/ransac.md renders it).
- `test/lo_timing/call_set.jl`'s `lo_call_set_arm(make_lo, problems, scorings, workspaces, pairs, reps)`: the LO bench
  arm, beside `call_set_arm` in the file LocalOptimizationDoneRight's bench includes.

## 2. Static findings

| check | result |
|---|---|
| docstrings and pages | `fundmat_scene` documented; `synthetic_scenes` names it; docs build strict green (gate task 10) |
| CHANGELOG | Estimation 0.11.1: Added (the fixture, the arm), Fixed (the bench's essential threshold key, moved into `scoring` in 0.10) |
| versions | Estimation 0.11.1; every other package as on main 7b9bec70 |
| in-repo compat | unchanged (Eval's Estimation "0.11" admits 0.11.1) |

## 3. Executed on the release candidate (b26cfd06)

Gate (~/logs/dgx-agent/scripts/w10-est-eval, NICE=0, `GATE_PKGS=VisualGeometryToolkitEstimation`; General c8aaee84e13;
jobs 15162 prep, 15163 CPU, 15164 docs, 15165 GPU): prep registration_lint 0 errors 0 warnings, agent_docs 0,
floor_compat --latest, audit_manifests, check_duplication 0, check_naming 0, check_splat 0, with the stack paths
bb-leg-060, **w10-lodr** (LocalOptimizationDoneRight's wave branch, 153bc9f), rsdr-09, cb-release, ps-release-062;
Estimation's CPU groups 7977 passed + 2 broken (8 groups; the 2 broken are `@test_broken` marks in task 2's files, none of which this patch touches), GPU 9/9, docs green.
Julia 1.13.3. The fixture's draws: ransac_fundmat's items pass unchanged (dev 15158: 32/32 with lo_timing_call_set).
The bench runs end to end on the change (dev 15161, `julia -t 1`, 2 reps: Welsch-IRLS 374 µs/call, GM-IRLS 330 µs/call,
8480 bytes/call; it failed before the threshold fix).

---

docs-pass: 5fb8889b
landing: VisualGeometryToolkit/visualgeometrytoolkit.github.io 6674c98 (prepared, unpushed, branch wave-113; one open prose item below)

# Wave 10 release docs pass (julia-docs-pass) — w10-release: Core 0.10.0, IO 0.10.0, Features 0.9.0, Estimation 0.11.0, Eval 0.8.0, PoseLib 0.3.0

The release branch merges up-v13 a8f4aa02, core10's final tip 792557a6 and GitHub main 60b13da3 (the patch line: Core
0.9.6–0.9.9, Estimation 0.10.2/0.10.3, Eval 0.7.3/0.7.4, IO 0.9.5, Features 0.8.4), and est-eval's records e06cde7c.
Each leg's own pass stands below (core10 09018e60, IO 80b9fb45, Features f363b258 in feat09-docs-pass.md, Estimation /
Eval / PoseLib 2d48ca86 in docs-pass-est-eval.md). This pass covers what the integration changed against the commit
each leg's pass and gate covered.

## 1. What the integration changed, per package (against the leg's passed commit)

| package | leg commit | changed in the release |
|---|---|---|
| Core | 792557a6 | CHANGELOG only: the released 0.9.9–0.9.6 sections below 0.10.0 |
| IO | c7854fa7 | nothing (main's IO is byte-identical to IO 0.9.5, 5dcb5e7e, already in the wave) |
| Features | f1435702 | `_merged` takes 0.8.4's final form (3d8203a4, `components(StructArray(ids))`) with its test; CHANGELOG 0.8.4 section; the examples/ and validation/ environments' compat on Features "0.9.0" |
| Estimation | 2d48ca86 | CHANGELOG only: 0.10.3/0.10.2 sections |
| Eval | 2d48ca86 | CHANGELOG 0.7.4/0.7.3 sections; docs/src/reproduce.md takes accedc52 (0.7.4's page); one docstring fix (below) |
| PoseLib | 2d48ca86 | nothing |

## 2. Findings and fixes

- **Eval reproduce.md** (accedc52, through the merge of main): the registered-install bullet under the code identity
  ("a project that is itself a **registered install** … by the tree hash the active environment records for it"), and
  the refusal narrowed to "code that is in no git checkout and is no registered install". Checked against the code:
  `_registered_install` (src/code_identity.jl:155) and its use in `code_trees` (:197–:202), and Eval 0.8.0's
  CHANGELOG entry "A run of record of a REGISTERED package's code is no longer refused".
- **Fixed, 5fb8889b**: `write_provenance`'s docstring (src/experiments/provenance.jl:150) still said a run of record
  refuses "code in no git checkout"; it now says "code in no git checkout that is no registered install", as the page
  and `code_trees`' docstring (src/code_identity.jl:179) do. `refuse_uncommitted_code`'s docstring is exact as it
  stands: it refuses by tree, and a registered install's tree is never a "not in git" one.
- **Features**: `_merged` is private; its comment states the new form; CHANGELOG 0.9.0's "merge field by field
  (StructArrays' `components`)" holds for it. No public docstring or page names it.
- **CHANGELOGs**: every released patch section kept below the wave's (Core 0.9.9/0.9.8/0.9.7/0.9.6, Estimation
  0.10.3/0.10.2, Eval 0.7.4/0.7.3, Features 0.8.4, IO 0.9.5); no version heading twice. The wave sections name what
  they carry from the patch line (Core 0.10.0's forward ports; Estimation 0.11.0's launch-and-reduce and region rule;
  Eval 0.8.0's registered install and 0.7.3's keyed upstream; Features 0.9.0's 0.8.4 fix).
- **Open, landing**: index.html's IO card says "target measurements"; the terminology ruling retires "measurement" for
  an associated detection (IO 0.10's kind is `vgio/observations`). Left for the site's owner on the prepared
  wave-113 branch (not this repository); it goes with the publish.

## 3. Executed

The mechanical gates (`test_docs`: every public name documented, `detached_docstrings`, `signature_mismatches`; each
package's docs build with `checkdocs = :public`, doctest, linkcheck) run in the release's final gate on the release
commit, with every package's CPU and GPU groups: release-integration.md, "Final gate".

---

docs-pass: 09018e60

# Wave 10 docs pass (julia-docs-pass) — VisualGeometryToolkitCore 0.10.0, branch w10-core10

Baseline: the wave-113 release, 2f994d2b (Core 0.9.5). The breaking wave of 2026-11-13. The API is listed in full in
Core's CHANGELOG `## 0.10.0`, with its Breaking and New lists. The function list, with file:line and the stack functions
each one considered, is in ~/logs/dgx-agent/wave10/core10.md. Every public interface change was posted in
~/logs/dgx-agent/wave10/interfaces.md before its commit.

## 1. Public API since the baseline

- **Breaking**:
  - the terminology renames, `Cspond`/`Csponds`/`csponds` → `Correspondence`/`Correspondences`/`correspondences`, and the
    rest (no aliases; each in `check_naming`'s retired names);
  - `TargetDetections` → `TargetObservations`, with `TargetPoints2D/3D` and `TargetPoint2D/3D` deleted;
  - one LM: `LmStops`, `StaticLmStops`, `LmProblem`, `lmsolve_chart!(…, stops)`, one success rule, `nlls_options` deleted;
  - the FeatureFrame interface, with every verb once;
  - the sensor model, which replaces wave 10's interim sensor-noise names;
  - `LinearRawView{T, A, D, L}` (D14);
  - `RelativeNoise`, which is specified only;
  - `block_coordinate_result`, which takes a stop rule;
  - the weak floors, restored.
- **New, public**:
  - FeatureFrame: `center(::Uncertain{<:FeatureFrame})`, `frame_dof`, `transfer(m, ::FeatureFrame)`;
  - the GPU test tier: `gpu_tier`/`gpu_problem`/`require_gpu`;
  - LM: `stop_rule`;
  - the quality gates: `check_splat`/`SplatViolation`;
  - the sensor model:
    - `SensorModel`, `SensorCharacterisation`, `sensor_model`, `content_digest`, `component_sources`;
    - its components and `SensorSource`;
    - its estimators: `photon_transfer`, `master_dark`, `defect_onsets`;
    - `poisson_slope(L, model[, plane])`, `green_plane`;
  - `Statistics.var(::Uncertain{<:Real})`;
  - `canonical_json`, hoisted from IO;
  - regions: `check_region_support`/`RegionBeyondSupport` and the one `⊆`;
  - `BinnedRadialProfile`/`radial_coordinate`;
  - `TargetPlaneFrame{id}`;
  - `iou_below`, `page_examples`, `install_lines`, `refine_width`, `assert_unit_bearings`;
  - forward-ported from Core 0.9.6–0.9.9 (they reach the wave through these ports):
    - 0.9.6: `reduce_normal_equations`/`reduce_cost`;
    - 0.9.7: their launch-and-reduce forms;
    - 0.9.8: `device_resident_threads`;
    - 0.9.9: `pointwise_kernel`.

## 2. Static findings

| check | result |
|---|---|
| every public name has a docstring | `test_docs`: 0 undocumented beyond the allow-lists. The `SensorSource` enum's type and four values each have their own docstring (12e1bc48); there is no exception entry. |
| docstrings attached, signatures match | `detached_docstrings` 0; `signature_mismatches` 0 (the docs item in the gate's CPU group) |
| every public docstring in the manual | `checkdocs = :public` green. The new sensor-model, region and LM pages go through `@autodocs` `Pages`, and `SensorSource`'s members through one `@docs` block in api/raw.md. |
| cross-references | the `SensorSource` docstring's references are qualified (Documenter resolves a bare `@ref` inside the enum's module; 4cbd4285) |
| stale mentions | retired names: none outside the CHANGELOG (`check_naming` R9/R14: 0) |
| CHANGELOG | Core `## 0.10.0` (unreleased), with Breaking, Fixed and New |
| version | VisualGeometryToolkitCore 0.10.0 |

## 3. Executed

| run | Core at | result |
|---|---|---|
| docs build (Julia 1.13.3, `--startup-file=no`, docs env), job 14852 | 09018e60 | green: no warning beyond Documenter's HTML-size notices |
| up-v13 integration gate a8f4aa02 (jobs 14786–14790; all six packages, wave stack paths) | 12e1bc48 | 48/48 green. For Core: CPU 413524 pass / 32 broken over 8 groups; GPU 54319 / 0 over 4 groups; docs green; prep lint, agent_docs, floor --latest, audit 0; `check_duplication` 0 on all six packages |
| load and the pointwise items, job 14851 | 09018e60 | 2937 / 1 broken (the patch agent's GPU run of the same items: 2973 / 2973) |

12e1bc48..09018e60 is the Core 0.9.9 forward port (`pointwise_kernel`, 5006841e and 09018e60). It has its own load, test
and docs runs above, and it goes into up-v14.

---

docs-pass: 80b9fb45
landing: VisualGeometryToolkit/visualgeometrytoolkit.github.io 14ee107 -> 6674c98 prepared locally (/raid/data/jpr/vgtk-landing-wave113, branch wave-113: "The packages require Julia 1.13"; not pushed)

# Wave 10 docs pass: VisualGeometryToolkitIO 0.10.0 (julia-docs-pass) — branch w10-io10-docs 80b9fb45 (w10-io10 0d6e90e5 merged)

The IO breaking wave (specs/2026-11-13-io-breaking-wave.md): one envelope (`schema`, `generator`, one `units` object, one
`conventions` object), every kind a new major `v<M>.<m>`, the observation kind, the detection / sensor_noise /
radial_profile / panorama v2 kinds, one camera store, one lattice record, one frame codec, the deliberate re-id.
Baseline: the IO 0.9.3 docs pass (7c550857). Julia 1.13.3, dev jobs on dgx01 (2 CPUs).

## 1. Public API since IO 0.9.3

- Added (public, unexported): `schema_versions`, `with_caller`, `validate_record`, `migrate_envelope`, `migrate_units`,
  `json_schema`; `VgioStyle(record)`, `VgioStyle(; …, density, length_roles)`; the `load`/`save` methods of
  `vgio/detection/v1.0`, `vgio/sensor_noise/v1.0`, `vgio/radial_profile/v1.0`; `TargetObservations` (`vgio/observations/v3.0`).
- Renamed: `migrate_opencv`/`migrate_matlab` → `convert_opencv_calibration`/`convert_matlab_calibration`; ColmapDatabase
  `keypoints` → `detections`; `save_manifest(…; kind)` → `result_kind`.
- Removed: `ScalarOrQuantity`, `pose_declarations`, `json_value`, `from_json`, `read_quantity`, `write_quantity`, the
  `TargetPoints2D`/`TargetPoints3D`/`TargetDetections` `load`/`save` methods (types gone from Core). Every mention deleted from
  README, docs/src and docstrings (grep: none left outside CHANGELOG and the version-step prose that names them as read).

## 2. Static findings, and what was done

| check | result |
|---|---|
| docstring gates (Core's `test_docs`) | undocumented 0, detached 0 (was 1: `frame_from_name`, consts between docstring and definition, fixed), signature mismatches 0; no exceptions, no backlog |
| docstrings | `frame_from_name` (attached, a doctest), `migrate_envelope` (its `density` keyword), `validate` (no "v2 file"), `VgioStyle` (the retired `JSON.lower(style, contract)` claim), `vgio_migrate_tree` (manifest rebinding, the id map's `rewritten`), OpenCV/MATLAB readers (`target → camera`) |
| manual | kinds.md rewritten: the envelope and minor versions, every kind at its 0.10 version, observations, the camera store with non-central cameras and the one lattice record, targets (tags, point vocabulary, placement), the version table GENERATED from `schema_versions()` (`@eval`), IO 0.10's step, the cameras v1 doctest now migrating to `vgio/cameras/v6.0`, the re-id and `vgio_migrate_tree`, the building blocks (the three `VgioStyle` forms, the quantity/Size2/frame codec, blur planes, `write_record` writing the envelope, `migrate_envelope`, `migrate_units`, `validate_record`); schemas.md, conventions.md, getting-started.md (the printed 0.10 file; a DocTestFilter on the generator version), third-party.md (`convert_*`), related-formats.md, serialization.md (its DocTestSetup was missing), index.md and README (the envelope; links to JSON Schemas and Content Identity) |
| API pages | native.md: the deleted noncentral_spline_schema.jl dropped, units_sidecar.jl added; third-party.md: migrate_v010.jl added |
| docs env | docs/Project.toml gains FlexUnits (the serialization doctest names `Quantity`) |
| terminology (R14) | no "measurement", "keypoint", "sighting", `cspond` in docs/src, README or docstrings (COLMAP's own `keypoints` table name is a foreign token) |

## 3. Executed

| run | result |
|---|---|
| `docs/make.jl` strict (checkdocs = :public, doctest, linkcheck, no warnonly) + Core's `test_docs`, job 13003, a detached clone of 80b9fb45 | BUILD OK; undocumented 0, detached 0, mismatches 0 |
| test items README, docs, agent docs, panorama, json_schema (13004, the same clone) | 443/443 |
| quality items (12878, 49bd5f38; not this slice's) | naming: R9 retired names in test/ only (calibration_manifest, covariance_storage, units_roundtrip, v2_validation tests); Aqua: JSON.lift ambiguities (tests slice fixing core_json.jl); ExplicitImports: `Base.unwrap_unionall` (migrate.jl:286, `schema_versions`); JET 10 > 0 |

## 4. Open

- `VgioStyle(record)` accepts an `AbstractDict` but reads `record.units` by property access: a `Dict` record fails (io10).
- Core's conventions/units.md still describes per-file `length_unit` and the `{"world", "density"}` units layout (core10).
- README/install tests resolving from the registry: known pre-registration failure until the wave registers.

---

docs-pass: 70202fe2
landing: VisualGeometryToolkit/visualgeometrytoolkit.github.io 6674c98 (no change)

# Core 0.9.9 docs pass (julia-docs-pass) — patch-core099

The launch `pointwise!` makes, public and unlaunched, so a caller sizes it with `device_resident_threads` without
mirroring Core's private kernel construction. Baseline: main ddbd9910 (Core 0.9.8, Eval 0.7.4).

## 1. Public API since the baseline

- Core: `pointwise_kernel(f, out, xs...; exec) -> (kernel, args, ndrange)`, public, not exported (transforms/pointwise.jl,
  beside `pointwise!`; in the manual through api/batched.md's `@autodocs` of transforms/pointwise.jl). `pointwise!` is
  it followed by the launch: one path. Nothing removed.

## 2. Static findings

| check | result |
|---|---|
| docstrings and pages | `pointwise_kernel` documented beside `pointwise!`; Core's docs build strict (green) |
| naming | the docstring's and test's "passed" replaced by "given" (check_naming R7, gate 4ffb61e4) |
| CHANGELOG | Core 0.9.9 |
| landing page | unchanged |

## 3. Executed on the release candidate (70202fe2)

Gate, Julia 1.13.3. On 4ffb61e4 (jobs 14755–14758): prep green; Core CPU groups 3 and 4 (pointwise_tests.jl,
device_tests.jl); the GPU tier, every package's GPU groups, green but Core's groups 10 (the device-vs-host FMA bit in
the new test) and 12 (R7). On 70202fe2, which changes only that docstring word and that test (jobs 14831, 14835–14837):
prep green (check_duplication 0 on all six, naming), Core CPU group 3 128099 / 2, GPU group 10 7771 / 0 and group 12
3019 / 0, Core docs green.

---

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

---

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

---

docs-pass: 4c910232
landing: VisualGeometryToolkit/visualgeometrytoolkit.github.io 6674c98 (no change: the portal names packages and links their docs; none of these fixes touches what it says)

# Core 0.9.6 / Estimation 0.10.2 / Eval 0.7.3 docs pass (julia-docs-pass) — patch-core096-eval073

Urgent patches from the catch-up review (wave10/review-monorepo.md C1, E1), with the additions the owner asked for:
Core's `reduce_normal_equations`/`reduce_cost` (Estimation's hand unpack moved there), the weak CUDA floor that made
0.9.5 unresolvable beside cuDNN 1, the chart LM's `ftol` in `refine_camera`'s regression, and `optical_axis(camera,
region)` with every bounded-source fit over `region ∩ imaged_region`. Baseline: 2f994d2b (wave 113), main eeb82edf
(IO 0.9.4) merged in. Core 0.9.6 was released from ccf9e2af (its tree as here).

## 1. Public API since the baseline

- Core, public, not exported: `reduce_normal_equations(partials, Val(N))`, `reduce_cost(partials)` (robust/lm_core.jl;
  docstrings each, on api/robust_statistics.md through its `@autodocs` of robust/lm_core.jl). `axis_gauged`'s
  BoundedCamera method: no signature change; its docstring's bounded-camera line now says a posed one's extrinsics turn.
  Weak `[compat] CUDA` "5.11.3" → "5.11 - 5". Nothing removed.
- Estimation: `optical_axis(camera::BoundedCamera, region = imaged_region(camera))` (a new optional argument; docstring
  and docs/src/api/radial.md); `fit_camera`/`refine_camera`/`approximate_camera` of a bounded source fit over
  `region ∩ imaged_region(source)` (docstrings say so; `fit_camera(SplineLens, …)` unchanged: its region is the
  lattice's support). The private hand unpack deleted. `[compat] VisualGeometryToolkitCore = "0.9.6"`.
- Eval: no API change. `PlannedStage`'s docstring (the run key folds the upstream stages' keys) and the reproduce page
  (docs/src/reproduce.md, "Run manifest, reuse") updated.

## 2. Static findings

| check | result |
|---|---|
| docstrings and pages | the two new Core names documented and in the manual (Core docs build strict); signatures checked by the `docs` items |
| stale mentions | none: the deleted Estimation unpack had no docs; the runner's old key rule appears only in CHANGELOGs |
| CHANGELOG | Core 0.9.6, Estimation 0.10.2, Eval 0.7.3 (Eval's names the one-time rerun of manifests from 0.7.2) |
| versions | Core 0.9.6, Estimation 0.10.2, Eval 0.7.3; IO 0.9.4 (main), Features 0.8.3, PoseLib 0.2.1 unchanged |
| in-repo compat | Estimation (package, test, docs) on Core "0.9.6"; every other member as it was |
| README / landing | unchanged (no install line or example moved) |

## 3. Executed on the release candidate (ccf9e2af)

Gate (~/logs/dgx-agent/scripts/patch096/gate.sbatch, submit.sh): a fresh `git clone --shared`, the sha detached, no
Manifest, General a GitHub clone in the patch's depot (JULIA_PKG_SERVER=""), a CPU prep that precompiles every test env
(the GPU code included) with the tasks' flags (--check-bounds=yes --depwarn=error), 2-CPU single-worker test groups,
each GPU group on its own GPU; Julia 1.13.3 only. Three runs; a package's results carry by its tree hash
(`git rev-parse <sha>:<dir>`): Core, IO, PoseLib from ccf9e2af (= 4c910232's trees), Features from 562d197e (= 4c910232's),
Estimation and Eval from 4c910232. Counts are pass / broken.

| | CPU | GPU | docs | run |
|---|---|---|---|---|
| prep: registration_lint --subsequent, agent_docs, floor_compat --latest, audit_manifests | 0 errors, 0 warnings; 0; rc 0; up to date | | | each run |
| check_duplication (all six, stack paths set) | 0 findings each | | | each run |
| GPU stacks: the six beside cuDNN 1; bb-063 and cb-053 with them | resolve (cuDNN 1.4.7, CUDA 5.11.1, CUDA_Runtime_Discovery 1.0.0) | | | each run |
| Core 0.9.6 | 410198 / 30 (8 groups) | 53121 / 0 (4) | 0 | ccf9e2af 12587–12591 |
| IO 0.9.4 | 30756 / 4 (6) | — (no device file) | 0 | ccf9e2af |
| Features 0.8.3 (tests of 562d197e) | 3063 / 9 (6) | 2011 / 0 (2) | 0 | 562d197e 12703–12706 |
| Estimation 0.10.2 | 8830 / 2 (6) | 55 / 0 (1) | 0 | 4c910232 13140–13143 |
| Eval 0.7.3 | 2913 / 0 (6) | 359 / 0 (1) | 0 | 4c910232 |
| PoseLib 0.2.1 | 373 / 1 (suite) | — | 0 | ccf9e2af |

ccf9e2af's Features GPU group 25 failed (47, `vgf_tests.jl` "the one matcher ≡ the reference rules": a GPU matrix
against the CPU reference, once another item had loaded CUDA); the same failure on the released 2f994d2b (job 12586).
562d197e's test-only fix (feat09's) is green.

---

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

---

# Wave 113 docs pass (julia-docs-pass) — wave-113 (Core 0.9.5, IO 0.9.3, Features 0.8.3, Estimation 0.10.1, Eval 0.7.2, PoseLib 0.2.1)

The wave: the whole stack on Julia 1.13 only, every [compat] floor raised to the gated resolution, the Downgrade
workflow deleted; Core 0.9.5 = the SIFT landing plus `farthest_radius`, the masked `bending_rows`, `backproject` of a
non-central model at any number type, `plane_map` for every camera model, `transform_deriv` of the 2-D homography map,
the CUDA default device with `always_inline`; Estimation 0.10.1 = `held_parameters`, `pixel_whitening_defect`; the
private forwards and aliases deleted (the owner's no-shim order). The "cuda" device-tag rename was stopped as breaking
(Features' `execution_provider`), by the coordinator's ruling left as it is. Baseline: sift-landing 8f979b38.

## 1. Public API since the baseline

- Core, public, not exported: `farthest_radius`, `plane_map`; new methods `bending_rows(mask; spacing)`,
  `transform_deriv(PerspectiveMap() ∘ ::HomographyMap{3}, ::StaticVector{2})`, `project(::NoncentralSplineModel, ::StaticVector{3, <:Dual})`;
  `Ray3{T <: Real}` (was `AbstractFloat`); `backproject(::NoncentralSplineModel)` eltype-generic. Nothing removed.
- Estimation, public, not exported: `held_parameters`, `pixel_whitening_defect`. Nothing removed.
- IO, Features, Eval, PoseLib: no API change.
- The function list with file:line, the stack functions considered and the deleted shims: ~/logs/dgx-agent/wave113-functions.md.

## 2. Static findings

| check | result |
|---|---|
| docstrings and pages | Core: `farthest_radius` (cameras/image_circle.jl) and `plane_map` (cameras/cameras.jl) on api/lenses.md and api/cameras.md through their `@autodocs`; `bending_rows(mask)` on api/batched.md (numerics/bspline_energy.jl); the 2-D homography map's derivative is a commented CT method (no docstring on a foreign function, as `transform_deriv(::PerspectiveMap, X)`). Estimation: `pixel_whitening_defect` in api/bearings.md's `@docs`, `held_parameters` in api/radial.md's, both named in the rotation-centre prose. |
| stale mentions | the deleted private names: none outside CHANGELOGs and earlier release records (grep in wave113-functions.md). "Julia 1.12 or later": every README, docs index and getting-started page reads 1.13; CLAUDE.md's workspace line too. |
| CHANGELOG | Core 0.9.5, IO 0.9.3, Features 0.8.3, Estimation 0.10.1, Eval 0.7.2, PoseLib 0.2.1 |
| versions | as above |
| in-repo compat | every member's floors at the gated resolution (wave113-floors.txt, 164 entries; floor_compat `--raise --check` on a copy: 0 more); stack siblings at their in-wave versions |
| README / landing | the landing page's "require Julia 1.12" line prepared as 6674c98 (above) |

## 3. Executed on the final RC (7c550857)

Gate (~/logs/dgx-agent/scripts/wave113/gate.sbatch, submit.sh; jobs 11013–11020): a fresh clone per task of `wave-113`, no
Manifest, General a GitHub clone in the wave's own depot (JULIA_PKG_SERVER=""), prep without a GPU (resolve, precompile,
lint, floor, audit, dup), then the CPU tiers and docs without `--gres` and each GPU tier on its own GPU, `--depwarn=error`,
JULIA_NUM_THREADS = the task's CPUs, Core at 3 ReTestItems workers. Julia 1.13.3 only. Counts are pass / broken.

| | CPU | GPU | docs |
|---|---|---|---|
| prep: registration_lint --subsequent, agent_docs, floor_compat --latest, audit_manifests | 0 errors, 0 warnings; rc 0; rc 0; all up to date | | |
| check_duplication (all six, stack paths set) | 0 findings each | | |
| Core | 410185 / 30 | 411768 / 2 | 0 |
| IO | 30541 / 4 | 30541 / 4 | 0 |
| Features | 3063 / 9 | 3133 / 2 | 0 |
| Estimation | 8813 / 2 | 8823 / 2 | 0 |
| Eval | 2906 / 0 | 2907 / 0 | 0 |
| PoseLib | 373 / 1 | 373 / 1 | 0 |

The duplication item is the same without and with VGTK_STACK_PATHS (the idiom cap counts the repository's own
definitions; job 10988: 1889/1889 both ways).

---

docs-pass: 94a303ff (sift-landing)
landing: VisualGeometryToolkit/visualgeometrytoolkit.github.io 14ee107 (no change: its Features line names ASIFT detection, not a refinement option)

# 0.9.5 / 0.8.3 docs pass (julia-docs-pass) — sift-landing (VisualGeometryToolkitCore 0.9.5, VisualGeometryToolkitFeatures 0.8.3)

The packages released: VisualGeometryToolkitCore 0.9.5 and VisualGeometryToolkitFeatures 0.8.3, both additive patches (the
owner's SIFT landing rulings of 2026-11-13). Estimation, IO, Eval and PoseLib are unchanged. The baseline is main c39b4a2d
(Core's tree its registered 0.9.4's, Features' its 0.8.2's).

## 1. Public API since Core 0.9.4 / Features 0.8.2

- Core, public, not exported: `static_lm` (the former private `_spline_lm`, at static N), `static_lm_termination`,
  `cubic_bspline_weight_derivatives` (the former private `_cubic_bspline_weights_deriv`, with the second derivative),
  `spline_window_sums`, `cubic_bspline_prefilter_taps`; `is_positive_definite` gains its static-N method. A generic
  docstring for `score_sq_residuals!`. Nothing removed from the public API.
- Features, exported: `ContinuousRefinement`; `asift_detect`/`asift_detect!` take the keyword `refinement` (default
  `nothing`: unchanged). Every other new name is private.
- The function list with each new name's file:line and its concept-level duplication check:
  ~/logs/dgx-agent/sift-landing-functions.md (in the owner's hands before posting).

## 2. Static findings

| check | result |
|---|---|
| docstrings and pages | Core: each new public name has a docstring on its page's `@autodocs` (cameras/spline_lens.jl, cameras/chart_spline.jl, primitives/feature_frames.jl, robust/student_t_uniform_scoring.jl); `static_lm`'s docstring sits on the function (the comment block above it, not between). Features: `ContinuousRefinement` in an `@docs` block on api/detection.md (continuous_refine.jl is not in the page's `@autodocs`: its other docstrings are private); the `asift_detect` docstring's "Continuous refinement (opt-in)" paragraph; the guide's section "Continuous refinement (opt-in)" with the measured numbers. No `@ref` from a public docstring to a private name. |
| stale mentions | `_spline_lm`, `_spline_lm_increment`, `_cubic_bspline_weights_deriv`: none outside CHANGELOGs and the historical plan under docs/superpowers. The private Features names (`_DogSplineWindow`, …) appear in no doc page. |
| CHANGELOG | Core 0.9.5, Features 0.8.3 |
| versions | Core 0.9.5, Features 0.8.3 |
| in-repo compat | Features' package, test and docs environments: VisualGeometryToolkitCore "0.9.5". The other packages' "0.9" / "0.9.2" admit it. |
| README / landing | no README or landing text names the refinement: unchanged |

## 3. Executed on the final RC (94a303ff)

Gate (~/logs/dgx-agent/scripts/siftland/gate.sh; jobs 10745–10748): a fresh clone per run of `sift-landing`, no
Manifest, General a GitHub clone in the gate's own depot (holding Core 0.9.4, IO 0.9.2, Features 0.8.2, Estimation 0.10.0,
Eval 0.7.1), CauRegistry from the shared depot; every suite strict (`--depwarn=error`), Core with RETESTITEMS_NWORKERS=3.
Counts are pass / fail / error / broken.

| | 1.13.3 GPU (10745) | 1.13.3 CPU (10746) | 1.12.7 GPU (10747) | 1.12.7 CPU (10748) |
|---|---|---|---|---|
| registration_lint --subsequent | rc 0 | | rc 0 | |
| floor_compat --latest | rc 0 | | rc 0 | |
| check_duplication(Core), (Features) | 0, 0 findings | | 0, 0 findings | |
| Core (--depwarn=error) | 411597 / 0 / 0 / 2 | 410019 / 0 / 0 / 30 | 411594 / 0 / 0 / 2 | 410016 / 0 / 0 / 30 |
| Features (--depwarn=error; JET, Aqua, ExplicitImports, duplication in-suite) | 3133 / 0 / 0 / 2 | 3063 / 0 / 0 / 9 | 3133 / 0 / 0 / 2 | 3063 / 0 / 0 / 9 |
| Estimation (--depwarn=error) | 8817 / 0 / 0 / 2 | 8807 / 0 / 0 / 2 | 8805 / 0 / 0 / 2 | 8795 / 0 / 0 / 2 |
| Eval (--depwarn=error) | 2907 / 0 / 0 / 0 | 2906 / 0 / 0 / 0 | 2907 / 0 / 0 / 0 | 2906 / 0 / 0 / 0 |
| Core docs, Features docs (strict make.jl) | 0, 0 | | 0, 0 | |

Two earlier gates were red and fixed before this one: 12ab92d0 (jobs 10732–10735) on a test of mine (the Float32 static-LM
case built a Float64 residual) and on the SplineLens optimization-level item, whose `M.stored` read in the item's world is
an `UndefVarError` under `--depwarn=error` on Julia ≥ 1.12 — main c39b4a2d's strict Core suite fails it too (job 10744);
94a303ff reads the Core.eval'd functions inside `invokelatest`. 5377c3b5 (job 10729) had two stale DUPLICATION.toml
entries (removed in 12ab92d0) and my test syntax.

---

docs-pass: f6a3e7f2
landing: VisualGeometryToolkit/visualgeometrytoolkit.github.io 14ee107 (no change: it names neither the SplineLens refine nor the forward fit)

# 0.10.0 docs pass (julia-docs-pass) — release-chrome-095 (VisualGeometryToolkitEstimation 0.10.0, Features 0.8.2, Eval 0.7.1)

The packages released: VisualGeometryToolkitEstimation 0.10.0 (breaking), VisualGeometryToolkitFeatures 0.8.2 and
VisualGeometryToolkitEval 0.7.1 (compat-only patches). Core, IO and PoseLib are unchanged. The baseline is the
Estimation 0.9.2 release (edab2655, below).

## 1. Public API since 0.9.2

No name added or removed. Changed behaviour of the exported `refine_camera_and_poses` on a SplineLens camera: it takes
`fixed = (:controls,)` only; the default `fixed = ()` (the image-space refine of the controls) and a `penalty` are
refused by name. `CameraPosesRefineMetadata.spline` loses the retired route's fields (CHANGELOG 0.10.0).

## 2. Static findings

| check | result |
|---|---|
| docstrings and pages | `refine_camera_and_poses`: its SplineLens signature line (`fixed = (:controls,)`, required), the `camera` argument, the `fixed` keyword and the whole "# A SplineLens camera" section rewritten for the held route; `CameraPosesRefineMetadata`'s `spline` and `free` for a SplineLens camera. api/radial.md: the camera-and-poses paragraph (the held route, the refusal, the retired `free = ()` spelling replaced by `fixed`) and the GCV paragraph (no longer names the SplineLens refine). configs/default.toml's `forward_tolerance` comment names `fit_forward_spline`. |
| stale mentions | the retired route's metadata fields, `penalty = :gcv` and the window/rebuild/GCV prose: none left outside CHANGELOGs and the plan notes under docs/superpowers (historical plans, not documentation). Core's spline_lens.jl comment that `_projection_jacobians` serves "a SplineLens camera's bundle refinement" is Core's (unchanged tree; Core is not released here) — noted for Core's next release. |
| CHANGELOG | Estimation 0.10.0 (Breaking, Changed), Features 0.8.2, Eval 0.7.1 |
| versions | Estimation 0.10.0, Features 0.8.2, Eval 0.7.1 |
| in-repo compat | Features (weak dependency) and Eval, and their test environments: Estimation "0.10" |
| README / landing | no README or landing text names the SplineLens refine's route: unchanged |

## 3. Executed on the final RC

The first 0.10.0 gate (66900bd9; jobs 10693–10696) was green on both Julia versions for every gated item; Features' strict
run (recorded, not gated there) errored one item, the L0 smoothing, on a third-party deprecation: ImageSmooth 0.1.1, its
latest release (2021, no newer upstream commit), calls ImageBase's deprecated `fdiff!` inside `L0Smooth`; Features calls
only ImageSmooth's public `smooth!`. f6a3e7f2 runs that one item's check in a child Julia with `--depwarn=yes` when the
suite is strict (the coordinator's option 2, the comment at the site names the cause); every other item stays strict, and
Features' strict run gates. No docs change.

The final gate on a0016dbb (jobs 10706–10709): two fresh clones per Julia version, no Manifest, General cloned from
GitHub (the package server bypassed), CauRegistry from the shared depot. Counts are pass / fail / error / broken.

| | 1.13.3 GPU (10706) | 1.13.3 CPU (10707) | 1.12.7 GPU (10708) | 1.12.7 CPU (10709) |
|---|---|---|---|---|
| registration_lint --subsequent | 0 errors, 0 warnings | | 0 errors, 0 warnings | |
| floor_compat --latest | rc 0 | | rc 0 | |
| check_duplication(Estimation) | 0 findings | | 0 findings | |
| Estimation (--depwarn=error) | 8817 / 0 / 0 / 2 | 8807 / 0 / 0 / 2 | 8805 / 0 / 0 / 2 | 8795 / 0 / 0 / 2 |
| Features (--depwarn=error) | 3071 / 0 / 0 / 2 | | 3071 / 0 / 0 / 2 | |
| Eval (--depwarn=error) | 2907 / 0 / 0 / 0 | | 2907 / 0 / 0 / 0 | |
| Estimation docs (strict make.jl) | 0 | | 0 | |

Estimation's count fell from 20979 (0.9.2) to 8817 with the retired route's items (174 → 171 items: the window
blocks' finite-difference Jacobian item, the smoothness/GCV item, the free-route refine items and the opt-in benchmark
item removed or converted; two refusal / held-camera items added).

---

# 0.9.2 docs pass (julia-docs-pass) — release-chrome-095 (VisualGeometryToolkitEstimation 0.9.2)

The package released: VisualGeometryToolkitEstimation 0.9.2 alone, a patch. The others are unchanged (Core's tree is
its registered 0.9.4's). The baseline is VisualGeometryToolkitEstimation-v0.9.1 (on main 540c2591 its tree differs from
the tag by the CHANGELOG heading only).

## 1. Public API since VisualGeometryToolkitEstimation-v0.9.1

One public name added, not exported: `fit_forward_spline(camera::BoundedCamera; tolerance, region, device)`
(fitting/fit_camera.jl; docstring on api/radial.md in its `@docs` block, beside `fit_camera`, and a paragraph of that
page's prose). Nothing removed, renamed or re-signatured. Changed behaviour, no API change: a held SplineLens camera is
projected whole (`refine_camera_and_poses` with `fixed = (:controls,)`).

## 2. Static findings

| check | result |
|---|---|
| docstrings and pages | `fit_forward_spline`: a docstring (signature, what is kept, how it differs from `fit_camera(SplineLens, …)`, the result, the `ArgumentError`), on api/radial.md. `refine_camera_and_poses`: its "A SplineLens camera" section said the forward is refitted by `fit_camera(SplineLens, …)`; it now names `fit_forward_spline` (72085327). Its held-camera paragraph (no window, the whole camera through `project`) came with the fix. |
| stale mentions | `fit_camera(SplineLens, …)` as a held camera's forward: none left in the monorepo. PanoramaStitching's rotation_bundle.jl docstring and sampling.jl comment named it: fixed on PS's ps-0.6.2. |
| CHANGELOG | an Estimation 0.9.2 section above 0.9.1 |
| versions | Estimation 0.9.2; no other package moves |
| in-repo compat | none needs moving: Features (weak), Eval and their test environments declare Estimation "0.9" |
| README / landing | no README or landing text names the refine's SplineLens route or the forward fit: unchanged |

## 3. Executed on the final RC

The first gate (61d44aa3; jobs 10673–10676, Julia 1.13.3 and 1.12.7) failed one Estimation item on both versions under
`--depwarn=error`: `ransac_weights`' `@test_throws ErrorException r.sigma` (a missing field is a `FieldError` since
Julia 1.12; Test deprecates asserting `ErrorException` for it). d06f1098 asserts `FieldError`; no docs change. The
final gate's results are recorded by its records commit.

---

# 0.9.4 docs pass (julia-docs-pass) — release-0.9.4

The package released: VisualGeometryToolkitCore 0.9.4 alone, a patch (RadiometricCamera 0.8.0 waits for it). The others
are unchanged. The baseline is 1c7581ec (release-0.9.3's head; 8ae2ea67 plus a test-only fix); the earlier passes are below.

## 1. Public API since 1c7581ec

Core adds the one public name `is_inside_texel_centres` (transforms/warp.jl, docstring on api/warping.md through its
`@autodocs`): whether a position lies in the closed rectangle of the texel centres of `bilinear_sample`'s half-texel
chart, exactly where `bilinear_weights` does not extend the edge. And one method of the exported `raw_frame_map`,
`raw_frame_map(::Type{LinearRaw{mode}})` (raw/linear_raw.jl, on api/raw.md through its `@autodocs`): a cell mode's frame
map without an image. And one method of the public `raw_provenance`, `raw_provenance(::Type{LinearRaw{mode}})`
(raw/linear_raw.jl, on api/raw.md): the `(; mode, frame_map)` an image's provenance holds, by mode. No new exported name; nothing removed or renamed in any package. `bilinear_weights` and the cell
images' maps and records are unchanged bit for bit (they now read the private chart `_texel_chart`/`_clamp_to_centres`, the
mode method and `_frame_map_record` respectively).

## 2. Static findings

| check | result |
|---|---|
| docstrings and pages | `is_inside_texel_centres`: a docstring, on api/warping.md, and a line in that page's prose beside `bilinear_weights`. `raw_frame_map(::Type{LinearRaw{mode}})`: a docstring, on api/raw.md, a line in that page's intro and a row in the guide's raw_frame_map table (guide/raw-images.md). `raw_provenance(::Type{LinearRaw{mode}})`: a docstring, on api/raw.md, and the same intro line |
| CHANGELOG | a Core 0.9.4 section above 0.9.3 |
| versions | Core 0.9.4; no other package moves |
| in-repo compat | none needs moving: every Core compat in the repo ("0.9", "0.9.2", "0.9.3") admits 0.9.4, and no in-repo package uses either addition |

## 3. Executed on the final RC

The final RC is 4055b67b (0.9.4 on b7ad736e, the pushed 0.9.3). The gate ran on two fresh clones, one per Julia version: a
`git clone` with no Manifest, its workspace resolved from scratch. Counts are pass / fail / error / broken.

| Package | 1.13.3 GPU (10623) | 1.13.3 CPU (10624) | 1.12.7 GPU (10632) | 1.12.7 CPU (10633) | docs 1.13 / 1.12 |
|---|---|---|---|---|---|
| VisualGeometryToolkitCore | 411320 / 0 / 0 / 2 | 409742 / 0 / 0 / 30 | 411317 / 0 / 0 / 2 | 409739 / 0 / 0 / 30 | 0 / 0 (10625, 10634) |
| VisualGeometryToolkitIO | 30541 / 0 / 0 / 4 | 30541 / 0 / 0 / 4 | 30541 / 0 / 0 / 4 | 30541 / 0 / 0 / 4 | 0 / 0 (10626, 10635) |
| VisualGeometryToolkitEstimation | 20961 / 0 / 0 / 3 | 20951 / 0 / 0 / 3 | 20949 / 0 / 0 / 3 | 20939 / 0 / 0 / 3 | 0 / 0 |
| VisualGeometryToolkitFeatures | 3071 / 0 / 0 / 2 | 3014 / 0 / 0 / 8 | 3071 / 0 / 0 / 2 | 3014 / 0 / 0 / 8 | 0 / 0 |
| VisualGeometryToolkitEval | 2907 / 0 / 0 / 0 | 2906 / 0 / 0 / 0 | 2907 / 0 / 0 / 0 | 2906 / 0 / 0 / 0 | 0 / 0 |
| PoseLib | 373 / 0 / 0 / 1 | 373 / 0 / 0 / 1 | 373 / 0 / 0 / 1 | 373 / 0 / 0 / 1 | 0 / 0 |

No failures or errors anywhere. The previous RC 2133c287 (the provenance method with an untyped map argument) failed Core's JET
gate on two of its four runs (13 reports against the bound 12: `translation(S::Any)`, union-split onto `Disc`/`OrientedDisc`);
4055b67b types the argument. Earlier RCs: eff6ff34 (before the provenance method), all green.

---|---|---|---|---|---|
| VisualGeometryToolkitCore | 410869 / 0 / 0 / 2 | 409291 / 0 / 0 / 30 | 410866 / 0 / 0 / 2 | 409288 / 0 / 0 / 30 | 0 / 0 (10513, 10517) |
| VisualGeometryToolkitIO | 30541 / 0 / 0 / 4 | 30541 / 0 / 0 / 4 | 30541 / 0 / 0 / 4 | 30541 / 0 / 0 / 4 | 0 / 0 (10514, 10518) |
| VisualGeometryToolkitEstimation | 20961 / 0 / 0 / 3 | 20951 / 0 / 0 / 3 | 20949 / 0 / 0 / 3 | 20939 / 0 / 0 / 3 | 0 / 0 |
| VisualGeometryToolkitFeatures | 3071 / 0 / 0 / 2 | 3014 / 0 / 0 / 8 | 3071 / 0 / 0 / 2 | 3014 / 0 / 0 / 8 | 0 / 0 |
| VisualGeometryToolkitEval | 2907 / 0 / 0 / 0 | 2906 / 0 / 0 / 0 | 2907 / 0 / 0 / 0 | 2906 / 0 / 0 / 0 | 0 / 0 |
| PoseLib | 373 / 0 / 0 / 1 | 373 / 0 / 0 / 1 | 373 / 0 / 0 / 1 | 373 / 0 / 0 / 1 | 0 / 0 |

No failures or errors anywhere. The 0.9.4 delta's own Core suites on the worktree, on the pre-rebase base 9762b9b4 (CPU 10425/10429,
GPU 10426/10430), failed only the base's `require_extension` world-age item that 1c7581ec fixes; the new items passed on all four.

---

# 0.9.3 docs pass (julia-docs-pass) — release-0.9.3

The packages released: VisualGeometryToolkitCore 0.9.3 (one public name added), VisualGeometryToolkitIO 0.9.2 and
VisualGeometryToolkitFeatures 0.8.1, all three patches. The others are unchanged. The baseline is 9dc838a1 (the 0.9.2
release); the earlier passes are below.

## 1. Public API since 9dc838a1

Core adds the public `require_extension` (base/utils.jl, docstring on api/core.md through its `base/utils.jl`
`@autodocs`): the stack's one check for a package extension, hoisted from Features' and BlobBoards' private copies.
Nothing else is added, removed or renamed in any package. Core changes: the MappedSurface root computes with explicit fused
products, its bits now the same at every call site (a fix); the private `_chart_affine` is deleted. IO: an embedded old
grid camera body is now read (a fix).

## 2. Static findings

| check | result |
|---|---|
| docstrings and pages | `require_extension`: a docstring with both signatures and an example, on api/core.md (that page is ~177 KiB rendered; the docstring adds ~2 KiB). Changed in place: docstrings changed in place: `camera_from_json` (IO: the old embedded body, read in colmap), the IO kinds page |
| CHANGELOG | Core 0.9.3 and IO 0.9.2 sections; the registered headings dated 2026-11-12 (Core 0.9.2, IO 0.9.1, Estimation 0.9.1, Features 0.8.0) |
| versions | Core 0.9.3, IO 0.9.2, Features 0.8.1 (its Core floor "0.9.3" in package, docs and test environments) |
| in-repo compat | none needs moving: every package's Core compat admits 0.9.3, and Eval's IO compat "0.9" admits 0.9.2 |

## 3. Executed on the final RC

The final RC is 1c7581ec. The gate ran on two fresh clones, one per Julia version: a `git clone` with no Manifest, its
workspace resolved from scratch. Counts are pass / fail / error / broken.

| Package | 1.13.3 GPU (10494) | 1.13.3 CPU (10495) | 1.12.7 GPU (10498) | 1.12.7 CPU (10499) | docs 1.13 / 1.12 |
|---|---|---|---|---|---|
| Core 0.9.3 | 396965 / 0 / 0 / 2 | 395387 / 0 / 0 / 30 | 396962 / 0 / 0 / 2 | 395384 / 0 / 0 / 30 | exit 0 / exit 0 |
| IO 0.9.2 | 30541 / 0 / 0 / 4 | 30541 / 0 / 0 / 4 | 30541 / 0 / 0 / 4 | 30541 / 0 / 0 / 4 | exit 0 / exit 0 |
| Estimation 0.9.1 | 20961 / 0 / 0 / 3 | 20951 / 0 / 0 / 3 | 20949 / 0 / 0 / 3 | 20939 / 0 / 0 / 3 | exit 0 / exit 0 |
| Features 0.8.1 | 3071 / 0 / 0 / 2 | 3014 / 0 / 0 / 8 | 3071 / 0 / 0 / 2 | 3014 / 0 / 0 / 8 | exit 0 / exit 0 |
| Eval 0.7.0 | 2907 / 0 / 0 / 0 | 2906 / 0 / 0 / 0 | 2907 / 0 / 0 / 0 | 2906 / 0 / 0 / 0 | exit 0 / exit 0 |
| PoseLib 0.2.0 | 373 / 0 / 0 / 1 | 373 / 0 / 0 / 1 | 373 / 0 / 0 / 1 | 373 / 0 / 0 / 1 | exit 0 / exit 0 |

There are no failures or errors anywhere, the README items included.

---

# 0.9.2 docs pass (julia-docs-pass) — release-0.9.2

The packages released: VisualGeometryToolkitCore 0.9.2, VisualGeometryToolkitIO 0.9.1, VisualGeometryToolkitFeatures
0.8.0 (Breaking) and VisualGeometryToolkitEstimation 0.9.1. Eval 0.7.0 and PoseLib 0.2.0 are unchanged. The baseline
is d1a5f837 (the registered stack). The 0.9 pass is below.

## 1. Public API since d1a5f837

| Package | added | removed / no longer exported |
|---|---|---|
| Core | exported: `RawMosaic`, `LinearRaw`, `BayerCellFrame`, `raw_frame_map`, `bayer_plane_map`, `green_cell_sites`, `raw_coordinates`, `channelview` methods, `photositeview`, `linearview`, `saturationview`. Public: `bayer_channels`, `LinearRawView`, `SaturationView`, `PhotositeColors`, `raw_provenance`, `cfa_string`, `photosite_colors`, `signed_area` (was `_shoelace_signed_area`), `normalized_point` (was `_normalized_point`), `bilinear_weights` (was `_bilinear_setup`, its arguments reordered), `julia_blocks`. New methods: `with_parameters(::MappedSurface, θ)`, `pixel_mask(region, frame_map, w, h)`, the `SimilarityMap` actions on `Rect` and `RegionIntersection` | none |
| IO | public `metres` (was `_metres`), `with_provenance` (was `_with_provenance`) | none |
| Estimation | public `image_size` (was `_image_size`; in `CONTRACT_GENERICS`) | none |
| Features | exported `detect_image_circle` | no longer exported: the sensor image moved to Core (`RawMosaic`, `LinearRaw`, `BayerCellFrame`, `raw_frame_map`, `bayer_plane_map`, `green_cell_sites`, `raw_coordinates`, `raw_provenance`, `cfa_string`, `photosite_colors`), and the old re-exports of Core's `feature_frame`, `Cspond` and `csponds` |

Every change is in its package's CHANGELOG. Features 0.8.0 lists each dropped name.

## 2. Static findings, and what was done

| check | result |
|---|---|
| Removed or no-longer-exported names in docs, README, examples, configs | No build or example failure. Prose in the Features guides now names Core as the owner of `feature_frame` and `csponds`. The raw-images snippet loads Features for `load_raw`. |
| Added public names: docstring, file covered by `@autodocs`/`@docs` | All covered. The sensor image is on a new Core page, `api/raw.md`. The other additions are on primitives, cameras, warping, lenses and stack_contracts in Core, on native in IO, on radial in Estimation, and on detection in Features. |
| Page sizes (200 KiB hard limit) | `api/raw.md` is new. `api/lenses` gains ~3 KiB (`pixel_mask`), estimated ~185 KiB. The strict build is the check (§3). |
| docs/make.jl gates | `checkdocs = :public`, `doctest = true`, linkcheck, no `warnonly` (Core's inventory-only rebuild excepted, as in 0.9). The new pages are wired. |
| Detached docstrings in changed files | none |
| README install lines | unpinned; no README example uses a dropped name |
| Versions, headings, in-repo compat | Core 0.9.2, IO 0.9.1, Features 0.8.0, Estimation 0.9.1, each with an "(unreleased)" CHANGELOG section. The Core floor is "0.9.2" in Estimation's and Features' package, docs and test environments. Features' examples and validation compat is "0.8". No other package depends on Features. |

## 3. Executed on the final RC

The final RC is 76efbdbb. The gate ran on two fresh clones, one per Julia version: a `git clone` with no Manifest, its
workspace resolved from scratch. Each suite includes its `docs`, README, agent-docs, quality and naming items. Counts
are pass / fail / error / broken.

| Package | 1.13.3 GPU (10211) | 1.13.3 CPU (10212) | 1.12.7 GPU (10215, Features rerun 10279) | 1.12.7 CPU (10216) | docs 1.13 / 1.12 |
|---|---|---|---|---|---|
| Core | 380433 / 0 / 0 / 2 | 378855 / 0 / 0 / 30 | 380430 / 0 / 0 / 2 | 378852 / 0 / 0 / 30 | exit 0 / exit 0 |
| IO | 30531 / 0 / 0 / 3 | 30531 / 0 / 0 / 3 | 30531 / 0 / 0 / 3 | 30531 / 0 / 0 / 3 | exit 0 / exit 0 |
| Estimation | 20961 / 0 / 0 / 3 | 20951 / 0 / 0 / 3 | 20949 / 0 / 0 / 3 | 20939 / 0 / 0 / 3 | exit 0 / exit 0 |
| Features | 3069 / 0 / 0 / 2 | 3012 / 0 / 0 / 8 | 3069 / 0 / 0 / 2 (rerun) | 3012 / 0 / 0 / 8 | exit 0 / exit 0 |
| Eval | 2907 / 0 / 0 / 0 | 2906 / 0 / 0 / 0 | 2907 / 0 / 0 / 0 | 2906 / 0 / 0 / 0 | exit 0 / exit 0 |
| PoseLib | 373 / 0 / 0 / 1 | 373 / 0 / 0 / 1 | 373 / 0 / 0 / 1 | 373 / 0 / 0 / 1 | exit 0 / exit 0 |

The first 1.12.7 GPU run of Features (10215) had 1 failure and 1 error. The single rerun on the same tree (10279) is
green.
- Aqua "persistent tasks" was not a persistent task. Its subprocess could not load the package: another job was
  precompiling it in the same depot ("Being precompiled by another process", then a cache miss for ImageMorphology).
  Aqua's own message says "a precompilation error, not a persistent task".
- "pairwise_matrix: exact for any descriptors" (`vgf_tests.jl:692`, the default device's Float32 L2 matrix `==` the
  CPU's) failed once and passed on the rerun and at the released d1a5f837 (10278). This batch does not touch the test
  or the kernel. It is listed as a post-release follow-up: an intermittent GPU/CPU bit difference.

---

# 0.9 docs pass (julia-docs-pass) — final RC

Packages: VisualGeometryToolkitCore 0.9.0, VisualGeometryToolkitIO 0.9.0, VisualGeometryToolkitEstimation 0.9.0,
VisualGeometryToolkitFeatures 0.7.0, VisualGeometryToolkitEval 0.7.0, PoseLib 0.2.0. Baseline: 3eabba46 (the last
docs-reviewed candidate). Final RC: 9c74e264 (the version bump 1d816a22 and the release records).

The pass has two parts:
- **Static** (done on dd9eeec5, read-only): the API diff, stale mentions, docs coverage, make.jl gates, detached
  docstrings.
- **Executed** (the final gate, in Slurm): the strict docs builds and the suite items that hold the docs true. These
  are `docs` (`undocumented_public`, `signature_mismatches`, `detached_docstrings`), `README`, `agent docs`, `quality`
  and `naming`.

## 1. Public API since 3eabba46

| Package | added | removed / renamed |
|---|---|---|
| Core | the strategy layer, moved from Estimation: `AbstractOptimizationStrategy`, `Irls`, `Triggs`, `EfficiencyConstant`, `LOSS_TAGS`, `STRATEGY_TAGS`, `@lo_registry`, `fixed_width`, `strategy_loss`, `lo_*` |  |
| | the termination record: `ReturnCode`, `StopRule` (18 rules), `stop_rule`, `found_model`, `retcode_symbol`, `retcode_from_symbol`, `successful_retcode`, `worst_retcode`, `nlls_termination`, `test_termination_record` | `nlls_retcode` → `nlls_termination`; `lm_retcode` removed |
| | hoists: `RetiredKeyword`, `refuse_retired_keyword`, `bearing_csponds`, `prefix_sweep`, `install_artifact`, `huggingface_token`, `free_device_buffers!`, `free_parameter_names`, `map_estimate`, `reproject_so3`, `s2_tangent_basis_matrix`, `sample_point_held`, `unit_weights`, `@threshold_scoring_constructors` |  |
| | | `AbstractLocalOptimization` → `AbstractOptimizationStrategy`; `chisq_median`, `ScaleMultiple` and `deployed_scale_multiple` removed |
| IO | none | none |
| Features | `raw_coordinates` | none |
| Estimation | `OPTIMIZATION_STRATEGY_TAGS`, `RefineMetadata`, `RigidAlignmentMetadata`, `RotationCenterRefineMetadata`, `RotationCenterViews`, `rotation_center_pose`, `ViewAngleLimitError`, `approximate_camera`, `bundle_loss`, `default_student_t_uniform_scoring`, `fit_filtered_rotation_averaging`, `fit_rigid_alignment`, `irls_termination`, `lo_resolve_efficiency_constant`, `n_without_bearing`, `without_bearing`, `optimization_strategy_from_toml`, `pooled_noise_scale`, `refine_camera`, `refine_relative_pose`, `uncertain_rotation`, `with_extent` | renamed: `bundle_refine_relpose` → `refine_relative_pose`, `filtered_rotation_averaging` → `fit_filtered_rotation_averaging`, `local_optimization_from_toml` → `optimization_strategy_from_toml`, `lo_resolve_scale_multiple` → `lo_resolve_efficiency_constant`, `LO_TAGS` → `OPTIMIZATION_STRATEGY_TAGS`. Removed: `NoLocalOptimization`, `deployed_shape`, `plain_scale_multiple`, `lo_scale_multiple_shape`. Moved to Core: the strategy layer above |
| Eval | `REPRODUCE_SCRIPT_TEMPLATE`, `UNPUBLISHED_TIER`, `catalog_view`, `catalog_view_mount`, `code_tree_changes`, `code_trees`, `fetch_data!`, `file_sha256`, `inspect_input`, `mount_dir`, `project_relative_path`, `provenance_token`, `refuse_uncommitted_code`, `reproduce_script`, `scratch_record`, `slurm_queue_states`, `stage_markers`, `stage_markers_dir`, `stage_run_state`, `tier_data_root`, `tier_variable`, `tree_sha256`, `write_reproduce_script` | none |
| PoseLib | `LeZach`, `POSELIB_STRATEGY_TAGS`, `PoseLibRefineMetadata` | `LeZachTruncation` → `LeZach` |

Every removal and rename is in its package's CHANGELOG (Breaking).

## 2. Static findings, and what was done

| check | result |
|---|---|
| Removed or renamed names still in docs, README, docstrings, examples or configs | One hit, fixed in fc3bf6dd: PoseLib/CLAUDE.md:62 `LeZachTruncation()` → `LeZach()`. The retired-name lists in check_naming are deliberate. |
| Estimation prose that passes `Irls` to a `refine_*` (refused since dup-item-6) | None. The refine prose says `Triggs` or `nothing` and names the refusal; the remaining `Irls` mentions are `fit_*` or ransac LO. One ambiguous comment (configs/default.toml:88, `[pipelines.bearing_pose]`: "the default refine") now reads "the default local optimization" (fc3bf6dd). |
| Added public names: docstring present, file covered by `@autodocs`/`@docs` | All present and covered by static reading. Confirmed by `checkdocs = :public` in the strict builds. |
| docs/make.jl gates (`checkdocs = :public`, `doctest = true`, linkcheck, no `warnonly`) | All six. Exception: Core make.jl sets `warnonly` only when `VGTK_DOCS_INVENTORY_ONLY=1`. That is the Slurm docs gate's inventory-only rebuild after a failed strict Core build (ci/slurm/docs.sbatch). It is never set for the strict build that gates. Reason at the site (make.jl comment). |
| `linkcheck_ignore` | host-level only (github.com, doi.org, publishers, the local inventory server), each with its reason |
| Detached docstrings (150 changed src files) | none |
| README install lines and pinned versions | none pinned; `julia = "1.12"` everywhere; no README example uses a removed name |
| docs and README test items | all six packages have `docs_tests`, `readme_tests` and agent-docs items |
| In-repo compat for the bumps | Eval and Eval/test on Estimation "0.8" → "0.9", Features examples and validation on Features "0.6" → "0.7": in the version-bump commit 1d816a22 |

## 3. Executed on the final RC

The final RC is 9c74e264. The gate ran on a fresh clone (/home/jpr/src/vgtk-final-9c74e264): a `git clone` with no
Manifest, its workspace resolved from scratch with Julia 1.13.3. Each suite includes its `docs`, README, agent-docs,
quality and naming items. Counts are pass / fail / error / broken.

| Package | strict docs build (9765/9766) | GPU suite (9763) | CPU suite (9764) |
|---|---|---|---|
| Core | exit 0 | 342441 / 0 / 0 / 2 | 342175 / 0 / 0 / 25 |
| IO | exit 0 | 30522 / 0 / 0 / 3 | 30522 / 0 / 0 / 3 |
| Estimation | exit 0 | 20956 / 0 / 0 / 3 | 20946 / 0 / 0 / 3 |
| Features | exit 0 | 15520 / 0 / 0 / 2 | 15146 / 0 / 0 / 9 |
| Eval | exit 0 | 2907 / 0 / 0 / 0 | 2906 / 0 / 0 / 0 |
| PoseLib | exit 0 | 373 / 0 / 0 / 1 | 373 / 0 / 0 / 1 |

There are no failures or errors anywhere, the README items included. The CPU run's extra broken counts are the GPU
items, which are recorded as broken when no device is present. Core's api/estimators page, which had been 201.15 KiB
(over the 200 KiB limit), is now under the 150 KiB warning size since the split in 541e1681.

## 4. Before General (cau-release runs it)

`bash ~/.claude/skills/julia-docs-pass/general_preflight.sh <package dir>`, per package, after the owner's go. Then
read every MANUAL line against general-checklist.md.
