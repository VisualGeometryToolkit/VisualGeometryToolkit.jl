# Known failures at a release (not caused by it), each with its follow-up

## 2026-11-21 — Core 0.12.0, IO 0.11.0, Estimation 0.12.1, Features 0.9.9, Eval 0.9.3, PoseLib 0.3.2 (release commit 0399ceeb)

- **VisualGeometryToolkitFeatures, `test/scale_space_tests.jl:33` and `:39`** ("asift_detect: bit-identical across the
  scale-space factoring (CPU)"): two pinned sha256 hashes of CPU feature output differ on the cluster node `nereus`
  (its CPU's floating-point evaluation order), and pass on `hephaestus` and `poseidon`. The failure is the same at
  `origin/main` before this release (registered Features 0.9.8 on Core 0.11.1; cluster job 6067, the same two hashes),
  and Features' code does not change in 0.9.9 (its compat on Core only). Released with it by the owner's ruling.
  Follow-up, immediately after the cascade: the test states its property on one machine — the two factorings' outputs
  compared with each other, not with a hash pinned across machines — gated on both nodes, in the next Features patch.
