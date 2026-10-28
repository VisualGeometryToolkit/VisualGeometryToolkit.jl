# Layers and Design

## Three layers for the foreign formats

1. **Wire** (`ColmapCamera`, `ColmapImage`, `ColmapPoint3D`). These records
   are unit-free and faithful to the format. They are always Float64 on disk.
   This layer is the COLMAP adapter.
2. **Unit contract** (`UnitContract`, `MM_PX`). The checked boundary between
   the magnitudes in a file and quantities in memory.
3. **Assembly** (`camera_model`, `colmap_camera`, `rigid_transform`,
   `colmap_image`). Conversion to and from VisualGeometryToolkitCore's
   `CameraModel` and rigid `EuclideanMap`s. It goes through quantities first,
   so VGC's unit checks at construction apply to all file data.

Every COLMAP 4.0 camera model reads as the VGC lens that computes COLMAP's own
function, and `colmap_model(lens)` names the COLMAP model a lens is written as
(`nothing` when no COLMAP model computes it). The OpenCV conventions
(`opencv_lens`, `opencv_coefficients`) and the Kalibr convention
(`kalibr_lens`) are exact adapters of the same kind; `opencv_camera` and
`opencv_parameters` move OpenCV's whole calibration (its camera matrix in its
own pixel frame and the distortion vector) to a camera in the COLMAP frame and
back, bit-exactly, through VGC's pixel-frame map. `camera_model(Float32, c)`
gives single-precision values in memory; the wire stays Float64.

Pixel data carries its frame in its type, and every stored label is derived
from the data's types rather than supplied to a writer. See
[Conventions on Disk](conventions.md).

## Why the camera store is JSON, not COLMAP

COLMAP's set of camera models is closed and smaller than VGC's lens families.
The double sphere model, for one, has no COLMAP counterpart. Storing cameras
as COLMAP would limit what a configuration can refer to, and the failure would
appear at export time, long after calibration. COLMAP therefore stays an
export for the subset it can express. `colmap_cameras(store)` performs that
export, and it throws on a lens it cannot express rather than approximating
it.
