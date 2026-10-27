# ==============================================================================
# Camera Intrinsics - Calibration matrices and intrinsics models
# ==============================================================================

abstract type AbstractIntrinsics end

# =============================================================================
# Intrinsics Kind Trait (Holy Trait pattern)
# =============================================================================

# NOTE (2026-07, #15): the IntrinsicsTrait family (IsLogical/IsPhysical/
# intrinsics_trait) is GONE — it had zero dispatch consumers, and
# PhysicalIntrinsics itself was demoted (physical scale lives with the
# Sensor, not the intrinsics; every intrinsics is a K matrix).

# CameraCalibrationMat — pixel-space K, slots into AbstractCalibrationMat
# (which is itself a subtype of AbstractAffineMat → AbstractHomMat).
@smatrix_wrapper CameraCalibrationMat AbstractCalibrationMat

"""
    CameraCalibrationMat(f::Quantity, pp::AbstractVector{<:Quantity}) -> CameraCalibrationMat{Float64}
    CameraCalibrationMat(f::Tuple{T,T}, pp::AbstractVector{<:Quantity}) where T<:Quantity -> CameraCalibrationMat{Float64}

Construct camera intrinsics matrix K from focal length(s) and principal point in pixel units.

Units are enforced at construction but stored internally as unitless Float64 for performance.
A pixel is dimensionless, so `f` and `pp` are dimensionless quantities (`800.0px`); a length
(`16.0mm`, a metric focal length without its pitch) is a `DimensionError`.

# Arguments
- `f`: Focal length in pixels - either scalar for isotropic or tuple `(fx, fy)` for anisotropic
- `pp`: Principal point in pixels as 2D vector [cx, cy]

# Returns
3×3 unitless Float64 camera intrinsics matrix in column-major order:
```
[fx  0  cx]
[ 0 fy  cy]
[ 0  0   1]
```

# Examples
```julia
# Isotropic focal length (fx = fy)
K = CameraCalibrationMat(800.0px, [320.0px, 240.0px])

# Anisotropic focal length (fx ≠ fy)
K = CameraCalibrationMat((800.0px, 805.0px), [320.0px, 240.0px])
```
"""
function CameraCalibrationMat(f::Quantity, pp::AbstractVector{<:Quantity})
    # Strip units and store as Float64 (pixels are dimensionless: a length is a DimensionError)
    f_val = _strip_pixels(f)
    pp_val = _strip_pixels.(pp)
    K = SMatrix{3, 3, Float64}(
        f_val, 0.0, 0.0,
        0.0, f_val, 0.0,
        pp_val[1], pp_val[2], 1.0
    )
    return CameraCalibrationMat{Float64}(Tuple(K))
end

function CameraCalibrationMat(f::Tuple{T, T}, pp::AbstractVector{<:Quantity}) where {T <: Quantity}
    # Strip units and store as Float64 (pixels are dimensionless: a length is a DimensionError)
    fx_val = _strip_pixels(f[1])
    fy_val = _strip_pixels(f[2])
    pp_val = _strip_pixels.(pp)
    K = SMatrix{3, 3, Float64}(
        fx_val, 0.0, 0.0,
        0.0, fy_val, 0.0,
        pp_val[1], pp_val[2], 1.0
    )
    return CameraCalibrationMat{Float64}(Tuple(K))
end

# Validate the structural invariants of a 3×3 pinhole calibration matrix:
#   [ fx   s  cx ]        fx > 0, fy > 0 (positive focals),
#   [  0  fy  cy ]        lower-triangular zeros,
#   [  0   0   1 ]        last row = [0, 0, 1].
# Skew term s = K[1,2] is not constrained (zero in most pipelines but allowed).
# Common pitfall this catches: HEB stores K transposed — users supply it
# directly and get a silently-wrong CameraModel.  With this check, the error
# message points at the transpose as the likely fix.
function _assert_calibration_matrix(
        M::AbstractMatrix{<:Real};
        atol::Real = 1.0e-6
    )
    size(M) == (3, 3) ||
        throw(DimensionMismatch("expected 3×3 matrix, got $(size(M))"))
    (abs(M[3, 1]) < atol && abs(M[3, 2]) < atol && abs(M[3, 3] - 1) < atol) ||
        throw(
        ArgumentError(
            "K's third row must be [0, 0, 1]; got $(Vector(M[3, :])). " *
                "Common pitfall: data source may store K transposed — try `K'`."
        )
    )
    abs(M[2, 1]) < atol ||
        throw(
        ArgumentError(
            "K must be upper-triangular: K[2,1] must be 0, got $(M[2, 1])."
        )
    )
    (M[1, 1] > 0 && M[2, 2] > 0) ||
        throw(
        ArgumentError(
            "K focal lengths must be positive: fx=$(M[1, 1]), fy=$(M[2, 2])."
        )
    )
    return nothing
end

"""
    CameraCalibrationMat(M::AbstractMatrix{<:Real}; check=true) -> CameraCalibrationMat

Construct from a 3×3 matrix (e.g., from OpenCV calibration JSON).

Values are stored unitless (COLMAP convention: pixel centers at 0.5). An
integer, rational or irrational matrix is promoted to `Float64`; any other
concrete real element type is kept, so a matrix of ForwardDiff `Dual`s gives a
`Dual` calibration matrix that automatic differentiation runs through.
`check` validates the pinhole structure (positive focals, upper-triangular,
last row `[0, 0, 1]`) and names the transpose when the rows are swapped.

# Example
```julia
# From OpenCV JSON (row-major nested arrays)
Km = [[800, 0, 320], [0, 800, 240], [0, 0, 1]]
K = CameraCalibrationMat(stack(Km)')
```
"""
function CameraCalibrationMat(M::AbstractMatrix{T}; check::Bool = true) where {T <: Real}
    check && _assert_calibration_matrix(M)
    S = _calibration_eltype(T)
    return CameraCalibrationMat{S}(Tuple(SMatrix{3, 3, S}(M)))
end

# The stored element type of a calibration matrix built from a `T` matrix: an
# exact number (integer, rational, irrational) or an abstract eltype is
# promoted to Float64, as it always was; any other concrete real (a
# ForwardDiff Dual) is kept.
_calibration_eltype(::Type{<:Union{Integer, Rational, AbstractIrrational}}) = Float64
_calibration_eltype(::Type{T}) where {T <: Real} = isconcretetype(T) ? T : Float64

"""
    CameraCalibrationMat(M::AbstractMatrix{T}; check=true) where T<:AbstractFloat
        -> CameraCalibrationMat{T}

Eltype-preserving constructor: when `M` is an `AbstractMatrix` of a concrete
floating-point type (e.g. `Float32`), build a `CameraCalibrationMat{T}`
without promoting to `Float64`.  Use this on hot paths or GPU pipelines where
the calibration matrix should stay in single precision.
"""
function CameraCalibrationMat(
        M::AbstractMatrix{T};
        check::Bool = true
    ) where {T <: AbstractFloat}
    check && _assert_calibration_matrix(M)
    return CameraCalibrationMat{T}(Tuple(SMatrix{3, 3, T}(M)))
end

# Static-matrix inputs: the @smatrix_wrapper converter `CameraCalibrationMat(
# ::StaticArray{…,2})` is more specific than the AbstractMatrix methods above
# and would bypass validation and the exact-number promotion. These two are more
# specific still (bounded T), so an SMatrix/RotMatrix/MMatrix K is checked
# like any other matrix (dataset loaders build K as an SMatrix).
function CameraCalibrationMat(
        M::StaticArrays.StaticArray{<:Tuple, T, 2};
        check::Bool = true
    ) where {T <: AbstractFloat}
    check && _assert_calibration_matrix(M)
    return CameraCalibrationMat{T}(Tuple(SMatrix{3, 3, T}(M)))
end
function CameraCalibrationMat(
        M::StaticArrays.StaticArray{<:Tuple, T, 2};
        check::Bool = true
    ) where {T <: Real}
    check && _assert_calibration_matrix(M)
    S = _calibration_eltype(T)
    return CameraCalibrationMat{S}(Tuple(SMatrix{3, 3, S}(M)))
end

"""
    CameraCalibrationMat(s::LinearAlgebra.UniformScaling)
        -> CameraCalibrationMat

Identity-scaling shortcut: `CameraCalibrationMat(I)` returns
`CameraCalibrationMat{Float64}` with `K = SMatrix{3,3,Float64}(I)`.  Supply a
scaled UniformScaling like `Float32(1)*I` to choose the eltype, or use a
positive scalar `α` to get `α·I` (upper-triangular by construction, so there
is nothing to validate).
"""
function CameraCalibrationMat(s::LinearAlgebra.UniformScaling{T}) where {T <: Real}
    s.λ > 0 || throw(
        ArgumentError(
            "UniformScaling scalar must be positive (got $(s.λ))"
        )
    )
    Tf = T <: AbstractFloat ? T : Float64
    K = SMatrix{3, 3, Tf}(s)
    return CameraCalibrationMat{Tf}(Tuple(K))
end

"""
    CameraCalibrationMat(f::Quantity, pitch::Size2{<:Quantity}, pp::AbstractVector{<:Quantity}) -> CameraCalibrationMat{Float64}

Construct camera intrinsics matrix K from physical focal length, pixel pitch, and principal point.

Units are enforced at construction but stored internally as unitless Float64 for performance.

This constructor converts from metric units to pixel units using the factorization:
```
K = [1  0  cx] [1/px   0   0] [f  0  0]
    [0  1  cy] [0   1/py  0] [0  f  0]
    [0  0   1] [0    0    1] [0  0  1]
```

where f is in mm, px/py are pixel pitch in mm/px, and (cx,cy) are in pixels.

# Arguments
- `f::Quantity`: Focal length in physical units (e.g., mm)
- `pitch::Size2{<:Quantity}`: Pixel pitch (e.g., `Size2(width=5.86μm/px, height=5.86μm/px)`)
- `pp::AbstractVector{<:Quantity}`: Principal point in pixels [cx, cy]

# Returns
3×3 unitless Float64 camera intrinsics matrix with focal length in pixels:
```
[αx  0  cx]
[ 0  αy cy]   where αx = f/px, αy = f/py
[ 0  0   1]
```

# Examples
```julia
# From sensor datasheet and lens specification
f = 16.0mm                                    # Lens focal length
pitch = Size2(width=5.86μm/px, height=5.86μm/px)  # Sensor pixel pitch
pp = [320.0px, 240.0px]                       # Principal point (image center)
K = CameraCalibrationMat(f, pitch, pp)
# K[1,1] ≈ 2730.38 (focal length in pixels)
```
"""
function CameraCalibrationMat(f::Quantity, pitch::Size2{<:Quantity}, pp::AbstractVector{<:Quantity})
    # Convert focal length from physical units to pixels
    # αx = f / pitch.width, αy = f / pitch.height
    # length / (length/px) is a pixel quantity
    fx_px = f / pitch.width
    fy_px = f / pitch.height

    # Now construct matrix using the pixel-based constructor (which strips units)
    # Need to convert to tuple for the anisotropic case
    return CameraCalibrationMat((fx_px, fy_px), pp)
end

# ============================================================================
# Intrinsics Models: LogicalIntrinsics and PhysicalIntrinsics
# ============================================================================

"""
    LogicalIntrinsics{T}

Camera intrinsics containing only the calibration matrix K (dimensionless),
parametric in storage element type `T <: Real`. Defaults to `Float64` for all
existing call sites; supply a Float32 `CameraCalibrationMat` to get
single-precision storage end-to-end (useful for GPU pipelines).

Used when physical parameters cannot be disentangled from K, such as when K is
recovered from autocalibration, Zhang's method, or other calibration techniques
that don't provide metric information about the sensor.

# Fields
- `K::CameraCalibrationMat{T}`: 3×3 calibration matrix
- `K_inv::SMatrix{3,3,T,9}`: Precomputed inverse of K for efficient backprojection

# Examples
```julia
# From autocalibration (defaults to Float64 storage)
K = CameraCalibrationMat(800.0px, [320.0px, 240.0px])
intrinsics = LogicalIntrinsics(K)              # LogicalIntrinsics{Float64}

# Float32 path for GPU pipelines
K32 = CameraCalibrationMat(Float32[1 0 0; 0 1 0; 0 0 1])
intrinsics32 = LogicalIntrinsics(K32)          # LogicalIntrinsics{Float32}
```
"""
struct LogicalIntrinsics{T <: Real} <: AbstractIntrinsics
    K::CameraCalibrationMat{T}
    K_inv::SMatrix{3, 3, T, 9}

    # Inner constructor: derive K_inv in T, no Float64 promotion
    function LogicalIntrinsics{T}(K::CameraCalibrationMat{T}) where {T <: Real}
        K_inv = SMatrix{3, 3, T, 9}(inv(K))   # inv(K) is CameraCalibrationMat, store as SMatrix for the K_inv field
        return new{T}(K, K_inv)
    end
end

# Outer constructor: deduce T from K
LogicalIntrinsics(K::CameraCalibrationMat{T}) where {T <: Real} =
    LogicalIntrinsics{T}(K)

"""
    focal_length(intrinsics::LogicalIntrinsics) -> Tuple{Float64, Float64}

Get focal lengths (fx, fy) from LogicalIntrinsics K matrix (unitless, in pixels).

Returns a tuple (fx, fy) extracted from the calibration matrix K.
For square pixels, fx ≈ fy.

The other forms answer the focal length a FULL field of view `θ` implies:
`focal_length(resolution::Size2, θ; dimension = :horizontal)` in pixels and
`focal_length(sensor::Sensor, θ; dimension = :horizontal)` as a length (see
their docstrings); [`fov`](@ref) is their inverse for a camera.

# Example
```julia
intrinsics = LogicalIntrinsics(K)
fx, fy = focal_length(intrinsics)  # Returns (Float64, Float64) in pixels
```
"""
focal_length(intrinsics::LogicalIntrinsics) = (intrinsics.K[1, 1], intrinsics.K[2, 2])

# ============================================================================
# Convenience Constructors for Intrinsics
# ============================================================================

"""
    LogicalIntrinsics(f::Quantity, pp::AbstractVector{<:Quantity}) -> LogicalIntrinsics

Convenience constructor for LogicalIntrinsics from focal length and principal point (both in pixels).

This constructor builds the calibration matrix K internally. Use this when you have
calibrated intrinsics but no physical sensor parameters.

# Example
```julia
logical = LogicalIntrinsics(800.0px, [320.0px, 240.0px])
```
"""
function LogicalIntrinsics(f::Quantity, pp::AbstractVector{<:Quantity})
    K = CameraCalibrationMat(f, pp)
    return LogicalIntrinsics(K)
end

"""
    LogicalIntrinsics(f::Tuple{T,T}, pp::AbstractVector{<:Quantity}) where T<:Quantity -> LogicalIntrinsics

Convenience constructor for LogicalIntrinsics with anisotropic focal lengths (fx ≠ fy).

# Example
```julia
# Different focal lengths in x and y
logical = LogicalIntrinsics((800.0px, 805.0px), [320.0px, 240.0px])
```
"""
function LogicalIntrinsics(f::Tuple{T, T}, pp::AbstractVector{<:Quantity}) where {T <: Quantity}
    K = CameraCalibrationMat(f, pp)
    return LogicalIntrinsics(K)
end

"""
    LogicalIntrinsics(f::Quantity, pitch::Size2{<:Quantity}, pp::AbstractVector{<:Quantity}) -> LogicalIntrinsics

Physical-spec construction: build the calibration matrix from a metric
focal length and pixel pitch, then store ONLY K.  The physical scale is
construction-boundary metadata — it belongs to the `Sensor` record, not
the intrinsics (the former `PhysicalIntrinsics` carried it redundantly
and doubled the camera dispatch table; demoted 2026-07, audit #15).

# Example
```julia
f = 16.0mm
pitch = Size2(width=5.86μm/px, height=5.86μm/px)
pp = [320.0px, 240.0px]
intrinsics = LogicalIntrinsics(f, pitch, pp)
```
"""
function LogicalIntrinsics(f::Quantity, pitch::Size2{<:Quantity}, pp::AbstractVector{<:Quantity})
    K = CameraCalibrationMat(f, pitch, pp)
    return LogicalIntrinsics(K)
end
