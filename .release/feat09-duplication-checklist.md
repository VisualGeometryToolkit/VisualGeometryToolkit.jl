# Features 0.9.0 duplication checklist (wave10, feat09)

Every new or changed function, the stack concept it touches and what was reused. check_duplication (Features, Core) on the
gated clone: Features 0 findings; Core's findings are core10's (gpu_tier.jl backend branches), not this branch's.

| function (VisualGeometryToolkitFeatures) | concept | stack functions considered / reused | status |
|---|---|---|---|
| `detect_foerstner(...; noise)` src/foerstner.jl:165 | noise-propagated covariance of a detector | Core `propagate(J, Σ)` (base/propagate.jl; J is never formed: Σ accumulates per pixel in the kernel); ASIFT's IFT covariance (asift/uncertainty.jl, another detector's pipeline); the study's `impulse_response`/`linearized_jacobian` (benchmark only, free-space, replaced by the exact adjoints) | [x] |
| `_propagated_covariances!` src/foerstner.jl:392 | IFT δx̂ = N⁻¹Σ A(p)δ∇g(p) per window | `_window_estimate` (same N, same window); one KA kernel CPU/GPU | [x] |
| `_pipeline_bands` src/foerstner.jl:356 | the doubling + correlation's linear response per axis | ASIFT `_correlate_adj`, `_up_adj`, `_Line` (asift/uncertainty.jl:112, 142) reused; `_upsample_sample` the one 2× upsampling | [x] |
| `_correlate_adj` asift/uncertainty.jl:112 | adjoint of a zero-padded correlation | was `_blur_adj` (symmetric only): `_blur_adj` is now its special case, bit for bit | [x] |
| `_up_adj(l, n)` asift/uncertainty.jl:142 | adjoint of the 2× upsampling | one function; the bounded edge only on request (ASIFT unchanged) | [x] |
| `_foerstner_classes` src/foerstner.jl:301 | the F-test labels | split out of `_foerstner_points` (no copy) | [x] |
| `_pixel_variance` src/foerstner.jl:198 | iid noise σn (8-bit grey) → variance on [0, 1] | the reference's `noise_sigma / 255` (same rule) | [x] |
| `propagate(m, d)` src/detection_types.jl:168 | detection → point observation | Core `center(d)` (the ONE conversion); the hand conversion deleted | [x] |
| `_require_decoder` src/raw_image.jl:52 | extension-missing fallback | Core `require_extension`; private `_missing_decoder` deleted (review F1) | [x] |
| `_refine_dog_extremum(...; extent)` asift/continuous_refine.jl:296 | keep a point in the level | the driver's keep test bounds (1 ≤ x ≤ W) (review F4) | [x] |
| `variance_stabilize` src/variance_stabilization.jl:72 | green plane, noise model | WAITING: Core's public green-plane accessor and the greens' colour indices (handoffs.md, review F2), `SensorNoise` argument | [ ] |
