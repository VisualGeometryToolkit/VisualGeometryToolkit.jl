# Conventions

This is the stack's canonical convention text, kept once, in
VisualGeometryToolkitCore's `conventions/` folder. VisualGeometryToolkitCore,
VisualGeometryToolkitIo, VisualGeometryToolkitFeatures, VisualGeometryToolkitEval,
VisualGeometryToolkitEstimation, RansacScoringDoneRight, LocalOptimizationDoneRight,
PoseLib.jl and BlobBoards all follow it; each package's `CLAUDE.md` points here
instead of carrying a copy, and adds only how a rule lands in that package.

Every rule below exists because breaking it produced a defect that a green test
suite did not catch. Where that history matters it is stated, because a
convention whose reason is lost gets "simplified" away by the next cleanup.
