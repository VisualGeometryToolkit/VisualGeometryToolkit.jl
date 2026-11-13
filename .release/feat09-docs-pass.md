docs-pass: (pending re-gate)

# Features 0.9.0 docs pass (wave10, feat09)

- Built: VisualGeometryToolkitFeatures/docs/make.jl on a fresh shared clone at c50c633e, Julia 1.13.3: rc 0 (no deploy).
- Public API changes documented: `detect_foerstner(...; noise)` and the `covariance_source` attribute (docstring, guide/foerstner.md
  "The covariance"); `CorrespondenceSet` (renamed in docstrings and guide/matching.md); `propagate(m, d)` via Core's `center(d)`
  (docstring); a Förstner point's error ellipse `Ellipse(d.value; sigma)` (docstring, guide, test).
- Terminology: "keypoint", "measurement", "cspond" absent from src, ext, docs, test and benchmark names and prose (outside
  CHANGELOG and third-party titles: the ANMS paper, cv::KeyPoint, the shader's ExtractKeypoints); check_naming R14 green.
- CHANGELOG 0.9.0 Breaking / Fixed / Internal written; version 0.9.0; [compat] Core "0.10.0", Estimation "0.11".
- [x] docs build  [x] docstrings  [x] guide pages  [x] CHANGELOG  [ ] floors (floor_compat --raise after the wave resolves from the registry)
