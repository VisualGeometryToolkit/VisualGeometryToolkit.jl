# =============================================================================
# VGIoFixtures — the prelude every item shares
# =============================================================================
#
# Pulled into an item by `setup = [VGIoFixtures]`: the unexported `load`/`save`
# verbs, the units, the lens and camera types the suite builds fixtures from,
# and one quaternion and one pose. An item sees a setup's EXPORTED names only,
# so everything below is exported; anything an item needs beyond this it
# imports itself.

@testsetup module VGIoFixtures
    import VisualGeometryToolkitCore as VgtkCore
    import VisualGeometryToolkitIO as VgtkIO

using JSON: JSON
using VisualGeometryToolkitIO: VisualGeometryToolkitIO, ColmapModel, ColmapImage, canonical_qvec
using VisualGeometryToolkitIO: load, save   # the two-verb I/O surface (unexported)
using StaticArrays: SVector, SMatrix
using LinearAlgebra: norm
using Rotations: QuatRotation, RotMatrix
using VisualGeometryToolkitCore: mm, m, px, CameraModel, CameraCalibrationMat, Pinhole,
    Lens, LiftLens, Perspective, Equidistant, Radial, Tangential, ThinPrism, Additive, Chain,
    KannalaBrandt, Scaramuzza, Division, EuclideanMap,
    AbstractLensModel, Size2, extent, DoubleSphere, CentralGenericModel, PixelFrame,
    lens_record, lens_from_record, parameters, lens_parameters
import VisualGeometryToolkitCore
using FlexUnits: DimensionError, ConversionError, quantity, ustrip
using VisualGeometryToolkitCore: parse_unit, unit_string
export quantity, ustrip, parse_unit, unit_string

export JSON, parse_json, ColmapModel, ColmapImage, canonical_qvec, load, save
export SVector, SMatrix, norm, QuatRotation, RotMatrix
export mm, m, px, CameraModel, CameraCalibrationMat, Pinhole,
    Lens, LiftLens, Perspective, Equidistant, Radial, Tangential, ThinPrism, Additive, Chain,
    KannalaBrandt, Scaramuzza, Division, EuclideanMap,
    AbstractLensModel, Size2, extent, DoubleSphere, CentralGenericModel, PixelFrame,
    lens_record, lens_from_record, parameters, lens_parameters
export VisualGeometryToolkitCore, DimensionError, ConversionError
export q0, _mkpose, qpose

const q0 = canonical_qvec(
    SVector(
        0.9128709291752769, 0.18257418583505536,
        0.36514837167011072, 0.0
    )
)

# A pose as a vgio reader returns one: its rotation the stored quaternion itself,
# its translation the stack's Vec3.
qpose(q, t) = EuclideanMap{3, Float64, QuatRotation{Float64}, VgtkCore.Vec3{Float64}}(
    QuatRotation(q[1], q[2], q[3], q[4], false), VgtkCore.Vec3{Float64}(t),
)
qpose(R::RotMatrix, t) = qpose(VgtkIO._qvec(R), t)

_mkpose() = qpose(q0, SVector(251.9, -144.4, 904.1))

# A file read the way a test inspects it: Symbol keys and dot access, so a test
# can `merge` a Symbol-keyed edit into it and write it back.
parse_json(s::AbstractString) = JSON.parse(s; dicttype = JSON.Object{Symbol, Any})

end
