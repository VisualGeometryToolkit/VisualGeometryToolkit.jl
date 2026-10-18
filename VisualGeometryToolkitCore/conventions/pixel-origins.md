## Pixel origins

Three canonical conventions, no aliases:

- `:opencv` — top-left pixel centre at (0, 0)
- `:colmap` — top-left pixel centre at (0.5, 0.5)
- `:julia` — top-left pixel centre at (1, 1)

`:vlfeat`, `:makie` and `:matlab` were removed as redundant aliases; do not
reintroduce them.

Geometric operations default to COLMAP: `Rect(sensor)` has its origin at (0, 0)
with widths equal to the resolution, and the image-display functions
(`imshow`, `CampixelImshow`) and the other geometric APIs follow. Conversion
between conventions is an `EuclideanMap(PixelFrame{A}, PixelFrame{B})` — see
[Coordinate Frames](../docs/src/guide/frames.md). Persisted files declare their
convention; a reader returns typed data rather than a bare array, so the frame
travels with the numbers instead of living in a comment.
