## Coordinate frames and normalization: frame in = frame out

1. A public function (exported or `public`) returns results in the frame and
   units its inputs came in — models, residuals, covariances, inlier sets,
   scales: pixels in, pixels out. No public signature takes or returns
   normalized coordinates or a normalization, and no result or metadata
   carries one.
2. Normalization is private conditioning owned by ONE estimator (`fit_*`,
   `find_*`, `refine_*`, or a one-step algorithm's public form such as
   `homography_dlt(cs)`): data mapped into the normalized frame once at entry,
   models composed back and covariances propagated back once at exit. The
   strategy is chosen in one place, `CspondNormalization(M, cs; center)`, by
   model class.
3. The inner loop never normalizes — minimal kernels, `*_step` updates,
   workspace `!` forms, RANSAC problem internals and their validity gates.
   They take normalized data plus the estimator's
   [`CspondNormalization`](@ref); a `frame = nothing` keyword that falls back
   to normalizing is a bug.
4. Public thresholds and tolerances are in the caller's units, converted
   inside by the normalization's scale ([`similarity_scale`](@ref)).
5. The one exception is the benchmark pool. Its record, VisualGeometryToolkitCore's
   [`Pools.PoolPair`](@ref VisualGeometryToolkitCore.Pools.PoolPair), stores
   correspondences and models in [`NormalizedImageFrame`](@ref) (image
   centre, half diagonal: corners at unit radius) together with the per-image
   maps that put them there. VisualGeometryToolkitEstimation builds a record from a
   pixel-frame problem (`PoolPair(rng, problem, raw)`) and owns the record's
   pool-frame RANSAC problems (`HomographyProblem(pair)`,
   `FundMatProblem(pair)`, `EssentialProblem(pair)`); VisualGeometryToolkitEval reads
   and writes pools; RANSACScoringDoneRight's pool scorers score them in that
   frame. Every API where pool data enters or leaves names that frame
   ([`pool_frame_maps`](@ref VisualGeometryToolkitCore.Pools.pool_frame_maps),
   [`pool_pixel_scale`](@ref VisualGeometryToolkitCore.Pools.pool_pixel_scale),
   [`pixel_csponds`](@ref VisualGeometryToolkitCore.Pools.pixel_csponds)); nothing
   converts silently.

| Layer | Examples | Normalizes |
|---|---|---|
| task verb | `find_boards`, `track_boards` | never; forwards pixels |
| estimator, one-step public form | `find_homography`, `fit_division_camera`, `homography_dlt(cs)` | once in, once out |
| inner loop | [`homography_dlt!`](@ref), `division_camera_step`, `refine!(ws, problem, θ, w)`, [`test_model`](@ref) | never; takes the estimator's |
| minimal kernel | [`homography_4pt`](@ref), [`essential_5pt`](@ref), `solve(solver, cs, idx)` | never |
| pool | [`PoolPair`](@ref VisualGeometryToolkitCore.Pools.PoolPair) (VGC `Pools`), VisualGeometryToolkitEstimation's pool-frame problems, VisualGeometryToolkitEval's pool IO, RANSACScoringDoneRight's pool scorers | stored in [`NormalizedImageFrame`](@ref), declared |

```julia
function homography_dlt(cs)                       # pixels in
    N = CspondNormalization(HomographyMat, cs)    # the one strategy, private
    return homography_dlt!(A, N, cs)              # inner loop: uses N, never builds one
end                                               # H back in pixels
τ_n = similarity_scale(target_map(N)) * τ_px      # a pixel tolerance, inside
```

Names: the strategies `HartleyNormalization(; center)` (centroid, or a known
centre; mean distance √2) and `DiagonalNormalization(size; center)` (the pool
frame), applied as `AffineMap(strategy, points)`; the two-sided value
[`CspondNormalization`](@ref), whose two sides ([`source_map`](@ref),
`target_map`) move a model by COMPOSITION — `D ∘ M ∘ inv(S)` into the
normalized frame and `inv(D) ∘ M ∘ S` back, with `S = source_map(N)`,
`D = target_map(N)`, `M` a map or a bare model matrix (a `HomographyMat` comes
back a `HomographyMat`, an F/E a `FundMat`; an [`EpipolarMap`](@ref) stays
one) — never by a dedicated verb (the retired frame-change verbs are listed in
`check_naming`). **`*Frame` names frame TAGS only** ([`ImageFrame`](@ref),
[`NormalizedImageFrame`](@ref)); a normalization is never called a frame.
Retired: `hartley_normalize`, `diag_normalize`, `rms_normalize`,
`HartleyFrame`, `coord_normalization`, `normalization_map`.

**Strip where you normalize.** Physical units leave a value at the same place
as its frame. The estimator that owns the normalization strips its physical
inputs into its private working unit once at entry, with the whole value's
`ustrip(u, X)`, and reattaches the unit once at exit, where it composes the
model back. Every layer below it sees plain numbers. A task verb forwards
quantities as it forwards pixels, and a minimal kernel never sees a unit. See
"Units" (`units.md`) for the layering and its four enforcement points.

Calibrated geometry (E, relative pose, PnP) is on unit bearings, never the
z = 1 plane, and takes no normalization: see "Calibrated geometry is on
bearings" (`bearings.md`).

Each package states how the rule lands in it in its own CLAUDE.md ("In this
package", "Frames here").

### Why this rule is worth its cost

A tolerance is a number in a frame, and neither the type system nor a test suite
carries the frame with it. One constant in this stack — the coincidence
tolerance that rejects duplicate detections — was read in three frames by three
consumers, none of which names the helper, so grepping its call sites finds none
of them: as metres in P3P's world points (a 10 cm radius), as a
spread-normalized unit (≈17 px at full frame), and in a calibrated chart
(120 px at f = 1200, rejecting 22% of clean minimal samples).

None of those showed up as a failure. An over-strict gate makes RANSAC discard
more samples, so the suites go **quieter, never red**. The question that finds
this class of bug is: *what units is my data in, at every problem I construct?*

### The frames of a rig, its members and a capture

A [`Rig`](@ref) is posed in ONE frame `F` (a body `BodyFrame{:camera_rig}`, the
world, a tracker's room, or a member's own frame — the gauge). Each member is a
(model, pose) pair and names its own frame:

- a sensor's is `ViewFrame{id}`; its pose is its extrinsics, `F → ViewFrame{id}`;
- a target's is `TargetFrame{id}`; its pose ([`PosedTarget`](@ref)) is
  `TargetFrame{id} → F`, stored unlabeled and minted on access. The target's
  geometry — [`points`](@ref)`(target)`, a [`PlanarTarget`](@ref)'s plane
  `z = 0` or a [`PointTarget`](@ref)'s points — is in `TargetFrame{id}` and
  independent of the pose, so one target can sit in several rigs.

A capture's per-frame values keep the same frames: a [`Trajectory`](@ref)
maps one body's frame into one frame over time (`BodyFrame{:board_rig} →
WorldFrame`), [`TargetPoses`](@ref) map `TargetFrame{id} → F` per frame (`F`
the camera's frame), [`TargetPoints2D`](@ref) are pixels in a declared
`PixelFrame`, [`TargetPoints3D`](@ref) points in a declared 3D frame. A
measurement names a target point by (target id, point index), resolved
against the target's definition — the rig member.
