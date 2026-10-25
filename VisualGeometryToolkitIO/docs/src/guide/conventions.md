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

In memory a length is a FlexUnits quantity in metres; in a file it is a
number in the file's declared `length_unit`. Every writer writes millimetres
unless given another `length_unit`, declares it once, and refuses plain
numbers where a length is meant (attach the unit: `t .* m`). A reader accepts
any declared length unit and returns lengths in metres. Every length crosses
the file boundary through `read_length` and `write_length`, which shift the
decimal exponent for a power-of-ten unit, so a file of human-scale numbers is
rewritten byte for byte; unit names go through VisualGeometryToolkitCore's
`parse_unit` and `unit_string`.

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
