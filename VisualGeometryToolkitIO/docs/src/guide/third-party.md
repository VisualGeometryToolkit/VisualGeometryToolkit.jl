# Third-Party Calibrations

## Migrating an export: `migrate_opencv`, `migrate_matlab`

Both functions convert one third-party calibration export into the same two
native files:

- `<stem>_cameras.json`, a `vgio/cameras/v3` store holding one camera named
  `<stem>`;
- `<stem>_poses.json`, a `vgio/transforms/v3` file with one pose per posed
  image, labeled `TargetFrame{target} → ViewFrame{image}`, in the export's
  `tvec` unit. An
  image that the export left `null` gets no record.

The round trip is verified before anything is written. The original file is
never touched, and an output file that already exists is refused.

## What the export leaves implicit

An export does not state everything a reader needs. Each missing piece is an
argument, and each is a statement about the file:

- `file_frame` is the pixel frame of `cx, cy`. `migrate_opencv` defaults to
  OpenCV's own frame, with the top-left pixel's centre at (0, 0),
  `PixelFrame{:opencv}`. `migrate_matlab` defaults to MATLAB's documented
  (1, 1), `PixelFrame{:julia}`. A fitted principal point cannot tell the two
  apart (two fits of one capture differ by less than a pixel), so override a
  default only for a file you have verified to be in another frame.
- `file_unit` is the unit of `tvec`, which is whatever unit the person who
  typed the object points used for the pattern. It is required unless the file
  declares it (`tvec_unit`, or this package's own `unit` key). A declaration
  that disagrees with the caller is an error, never a silent rescale. On one
  capture, the OpenCV export was in metres and the MATLAB export in
  millimetres.
- `images` and `image_size` are required. `images` gives one unique, non-empty
  name per pose entry, in the export's order; together with `image_size` it
  supplies the positional join and the image extent that an export leaves to
  its reader.

## MATLAB specifics

- MATLAB's `mean_error` is a mean reprojection error, not an RMS, and it is
  recorded as such (`error_statistic = "mean"`).
- MATLAB's board frame has x and y exchanged relative to OpenCV's for the same
  pattern. The poses are kept in MATLAB's own board frame, unchanged, so give
  `target` a name that keeps them apart from an OpenCV calibration's.
- The toolbox's exports declare no `tvec_unit`, so `file_unit` is required.
  This stack's captures were typed in millimetres.
- A fisheye export is read as VGC's `Scaramuzza` model. Its `stretch_matrix`
  is factored as `S = U·Q`. The part `a0·U` goes into `K`, and `Q`, a rotation
  about the optical axis (a fraction of a degree in practice), goes into the
  camera frame. Every returned pose carries that rotation, so the camera and
  the poses stay one consistent model.
- A non-positive `a0` is MATLAB's mirrored solution, with the target behind
  the camera. It is refused rather than read as a camera.

## The readers and the writer

The migrations are built on `load_opencv_calibration` and
`load_matlab_calibration`. They return an `ExternalCalibration`, a
foreign-format adapter holding VGC values: the camera as a `CameraModel`, and
the poses as `EuclideanMap`s labeled `TargetFrame{target} → CameraFrame`. The
migration relabels them `→ ViewFrame{image}`.

`save_opencv_calibration` writes one of our cameras back out in the OpenCV
shape. It is the one place where a pixel frame is a conversion target
(`to_frame`) rather than a statement about a file.

## The command line

`bin/vgio_migrate.jl` converts a third-party calibration from the shell. It is
a dry run unless you give `--apply`: the export is converted into a temporary
directory and verified, and nothing is written.

```
julia --project=<VisualGeometryToolkitIO> bin/vgio_migrate.jl opencv --help
… opencv --images IMGDIR --image-size 6000x4000 --file-unit m calibration_opencv_full.json
… matlab --images IMGLIST --image-size 6000x4000 --file-unit mm --target board_matlab \
    --apply calibration_matlab_fisheye.json
```
