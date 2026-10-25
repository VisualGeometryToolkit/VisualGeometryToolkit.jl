# Conventions on Disk

The stack's conventions (how estimators are named, how signatures order their
arguments, and the rule that results come back in the frame their inputs came
in) are one page, VisualGeometryToolkitCore's
[Conventions](https://visualgeometrytoolkit.github.io/VisualGeometryToolkitCore/stable/guide/conventions/).
This page describes what they mean once the numbers are on disk. Nothing in
this package normalizes data: it holds no estimator, so it owns no
conditioning.

## Labels come from the data's types

Three conventions travel with stored numbers: the pixel frame, the length
unit, and a pose's direction. Each is written from what the data already is,
and none of them is an argument to `save`.

### Pixel frame

Pixel data carries its frame in its type: VGC's `TargetPoints2D{F}`, where
`F` is a VGC `PixelFrame{Origin}`. The producer that knows its pixels'
convention attaches `F`. `save` writes `pixel_frame` from it and converts no
number, so the label cannot disagree with the pixels under it. VGC owns the
set of registered frames, and this package keeps no list of its own.

The kinds that store no producer pixels (the camera store, rigs and the grid
models) declare the canonical `colmap` frame. A stored camera carries no frame
of its own, because its model's intrinsics are canonical by construction. A
source frame is declared where the conversion happens, as in
`CameraModel(K; image_origin = :opencv)`. A foreign file that declares
`opencv` or `julia` is converted on read and saved again in the canonical
frame.

### Length unit

In memory a length is a FlexUnits quantity in metres. In a file it is a number
in the file's declared `length_unit`, the unit that reads naturally for the
quantity. Every writer of a vgio kind writes millimetres unless its
`length_unit` keyword names another unit, and it declares the unit once. The
foreign exporters write the unit their format fixes: `save_tum_trajectory` and
`save_euroc_trajectory` write metres, and `save_opencv_calibration` writes
`tvec` in the unit its required `to_unit` keyword names. A writer refuses plain
numbers where a length is meant, so attach the unit first (`t .* m`). A reader
accepts any declared length unit and returns lengths in metres. Unit names go
through VisualGeometryToolkitCore's `parse_unit` and `unit_string`.

### Lengths cross the file boundary decimal-exactly

Every length is read with `read_length(x, u)` and written with
`write_length(q, u)`, and nothing in this package converts a length by
multiplying by a scale. For a power-of-ten unit (mm, cm, μm, km) the pair
shifts the decimal exponent, so `62.8` mm is the metre value nearest the
decimal 0.0628 and is written back as `62.8`. Converting by the scale and back
(`62.8 * 0.001`, then `/ 0.001`) gives `62.79999999999999`, and a few percent
of typical values move in the same way. For another unit (inch, ft, pt) the pair reads by
the unit's scale and writes the shortest decimal that reads back to the
identical value.

A file this package wrote is rewritten byte for byte when its numbers have at
most 15 significant digits. A full-precision number (16 or 17 digits) may move
by at most 2 ulp on its first rewrite, and it stays fixed after that. This is
inherent to holding metres in memory and millimetres in files.

A length built in code by multiplication has already rounded before it reaches
a writer. `1.3mm` is held as `0.0013000000000000002` m and is written as
`1.3000000000000003`. To make an exact length from a decimal, read it the way
a file is read, `VisualGeometryToolkitIO.read_length(1.3, mm)`.

A covariance follows the same rule. In memory it is in base units (for a pose,
m² in the translation block and m·rad in the cross block). On disk it is in the
declared `length_unit`, converted by the same decimal shift raised to each
entry's power of length.

### Pose direction

A pose's frames are its labels. A transforms record's `from` and `to` are the
frame names of the stored `EuclideanMap{…, S, D}`, and an unlabeled pose is
refused. The file declares `transform_convention = "x_to = R * x_from + t"`
and `rotation_encoding = "quaternion_wxyz_hamilton"`.

## Readers return the labels as types

`load(TargetPoints2D, path)` returns a `TargetPoints2D{F}`, where `F` is the
frame the file declares. The frame travels with the numbers instead of living
in a comment. VGC's `convert_frame(G, x)` is the only way to change it, and it
moves every pixel by VGC's `EuclideanMap(F, G)`.

The one frame stated by hand is a foreign file's. The OpenCV and MATLAB
readers and migrations take `file_frame`, a statement about the file. The
OpenCV-shaped export takes `to_frame`, the convention a third-party consumer
expects.

## A file of unknown origin

To identify a file, read its `schema` first.
`VisualGeometryToolkitIO.validate(path)` checks any vgio file against its
kind's table, the same table the reader and the writer use. A vgio file
answers every question about its own conventions.

Anything else, such as an OpenCV or MATLAB export, states less than a reader
needs. Each missing statement becomes an argument that you supply, as
described in [Third-Party Calibrations](third-party.md). No reader infers a
convention from the numbers. It refuses the file instead of assuming a
default.
