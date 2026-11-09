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

# =============================================================================
# CameraCalibrationMat — pixel-space K, stored as its five parameters
# =============================================================================

"""
    CameraCalibrationMat{T} <: StaticMatrix{3, 3, T}

The pinhole calibration matrix

    [fx  s  cx]
    [ 0 fy  cy]
    [ 0  0   1]

stored as its five parameters `(fx, fy, s, cx, cy)`: indexing returns the full
matrix, its structural entries exactly `0` and `1`. Every constructor from a
3×3 matrix checks that structure (row 3 = `0 0 1` and `K[2, 1] = 0`, within
`atol = 1e-6`; the transpose named as the likely fix) — the storage cannot hold
anything else; `check = false` skips only the positive-focal check. Arithmetic
on it gives an `SMatrix`; `inv(K)` is analytic, an `SMatrix` with exact
structural zeros (`K⁻¹` is not a calibration matrix). Isbits and eltype-generic
(`Float32`, ForwardDiff `Dual`).
"""
struct CameraCalibrationMat{T} <: StaticMatrix{3, 3, T}
    fx::T
    fy::T
    skew::T
    cx::T
    cy::T
    CameraCalibrationMat{T}(fx, fy, s, cx, cy, ::_Unchecked) where {T} = new{T}(fx, fy, s, cx, cy)
end

Base.IndexStyle(::Type{<:CameraCalibrationMat}) = IndexLinear()
@inline function Base.Tuple(K::CameraCalibrationMat{T}) where {T}
    z, o = zero(T), one(T)
    return (K.fx, z, z, K.skew, K.fy, z, K.cx, K.cy, o)
end
# An entry, by its position (a constant index folds to the parameter or the structural 0/1)
@inline function _calibration_entry(K::CameraCalibrationMat{T}, i::Int, j::Int) where {T}
    i == 1 && return j == 1 ? K.fx : j == 2 ? K.skew : K.cx
    i == 2 && return j == 1 ? zero(T) : j == 2 ? K.fy : K.cy
    return j == 3 ? one(T) : zero(T)
end
@inline function Base.getindex(K::CameraCalibrationMat, i::Int, j::Int)
    @boundscheck (1 <= i <= 3 && 1 <= j <= 3) || throw(BoundsError(K, (i, j)))
    return _calibration_entry(K, i, j)
end
@inline function Base.getindex(K::CameraCalibrationMat, k::Int)
    @boundscheck 1 <= k <= 9 || throw(BoundsError(K, k))
    return _calibration_entry(K, (k - 1) % 3 + 1, (k - 1) ÷ 3 + 1)
end

# (K·v is StaticArrays' product through getindex: it keeps the vector's type
# and is bit for bit `SMatrix(K) * v`, so pixels through K, through
# `calibration_matrix` and through `HomographyMap(K)` agree.)

# Immutable, and its structure is its storage: StaticArrays' generic `copy` and
# cross-eltype `convert` rebuild through the 9-entry constructor, whose focal
# check refused a K built with `check = false`. An eltype conversion converts
# the five parameters and checks nothing.
Base.copy(K::CameraCalibrationMat) = K
Base.convert(::Type{C}, K::C) where {C <: CameraCalibrationMat} = K
Base.convert(::Type{CameraCalibrationMat{T}}, K::CameraCalibrationMat) where {T} = CameraCalibrationMat{T}(K)
# FixedModels' padding: the zero parameters, row 3 still `0 0 1` (a calibration
# matrix has no zero; a padding slot is never read as a model)
_padding_value(::Type{CameraCalibrationMat{T}}) where {T} = CameraCalibrationMat{T}(zero(T), zero(T), zero(T), zero(T), zero(T), _Unchecked())

# The calibration matrix of the entries of a 3×3 (any indexable `M`): the
# structure always checked, the focals when `check`.
function _calibration_from_matrix(::Type{T}, M; check::Bool = true) where {T}
    _assert_calibration_structure(M)
    check && _assert_calibration_focals(M)
    return CameraCalibrationMat{T}(M[1, 1], M[2, 2], M[1, 2], M[1, 3], M[2, 3], _Unchecked())
end

# The entry constructors (column-major, as every StaticArray takes them)
CameraCalibrationMat{T}(data::NTuple{9}; check::Bool = true) where {T} =
    _calibration_from_matrix(T, SMatrix{3, 3}(data); check)
CameraCalibrationMat(data::NTuple{9, T}; check::Bool = true) where {T} = CameraCalibrationMat{T}(data; check)
CameraCalibrationMat{T}(args::Vararg{Number, 9}; check::Bool = true) where {T} = CameraCalibrationMat{T}(args; check)
CameraCalibrationMat{T}(M::AbstractMatrix; check::Bool = true) where {T} = (size(M) == (3, 3) ||
    throw(DimensionMismatch("expected 3×3 matrix, got $(size(M))")); _calibration_from_matrix(T, M; check))
CameraCalibrationMat{T}(M::StaticArrays.StaticArray{<:Tuple, <:Any, 2}; check::Bool = true) where {T} =
    _calibration_from_matrix(T, SMatrix{3, 3}(M); check)
# an eltype conversion: the parameters converted, nothing to check
CameraCalibrationMat{T}(K::CameraCalibrationMat) where {T} =
    CameraCalibrationMat{T}(K.fx, K.fy, K.skew, K.cx, K.cy, _Unchecked())

"""
    inv(K::CameraCalibrationMat) -> SMatrix{3, 3, T, 9}

The analytic inverse `[A⁻¹ −A⁻¹c; 0 0 1]`, `A = [fx s; 0 fy]`, its structural
entries exact. An `SMatrix`: `K⁻¹` maps pixels to the normalized plane and is
not a calibration matrix.
"""
function Base.inv(K::CameraCalibrationMat{T}) where {T}
    id = inv(K.fx * K.fy)
    a11, a12, a22 = K.fy * id, -K.skew * id, K.fx * id
    z, o = zero(a11), one(a11)
    return SMatrix{3, 3, typeof(a11), 9}(a11, z, z, a12, a22, z, -(a11 * K.cx + a12 * K.cy), -(a22 * K.cy), o)
end

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
CameraCalibrationMat(f::Quantity, pp::AbstractVector{<:Quantity}) = CameraCalibrationMat((f, f), pp)

function CameraCalibrationMat(f::Tuple{T, T}, pp::AbstractVector{<:Quantity}) where {T <: Quantity}
    # Strip units and store as Float64 (pixels are dimensionless: a length is a DimensionError)
    pp_val = _strip_pixels.(pp)
    return CameraCalibrationMat{Float64}(
        Float64(_strip_pixels(f[1])), Float64(_strip_pixels(f[2])), 0.0,
        Float64(pp_val[1]), Float64(pp_val[2]), _Unchecked()
    )
end

# Validate the structural invariants of a 3×3 pinhole calibration matrix:
#   [ fx   s  cx ]        fx > 0, fy > 0 (positive focals: `_assert_calibration_focals`),
#   [  0  fy  cy ]        lower-triangular zeros,
#   [  0   0   1 ]        last row = [0, 0, 1].
# Skew term s = K[1,2] is not constrained (zero in most pipelines but allowed).
# Common pitfall this catches: HEB stores K transposed — users supply it
# directly and get a silently-wrong CameraModel.  With this check, the error
# message points at the transpose as the likely fix. ALWAYS checked: the
# storage holds the five parameters only, so a structure it cannot hold is
# refused rather than dropped.
function _assert_calibration_structure(M; atol::Real = 1.0e-6)
    size(M) == (3, 3) ||
        throw(DimensionMismatch("expected 3×3 matrix, got $(size(M))"))
    (abs(M[3, 1]) < atol && abs(M[3, 2]) < atol && abs(M[3, 3] - 1) < atol) ||
        throw(
        ArgumentError(
            "K's third row must be [0, 0, 1]; got $(Vector(M[3, :])). " *
                "Common pitfall: data source may store K transposed — try `K'`."
        )
    )
    return _assert_calibration_k21(M[2, 1]; atol)
end
_assert_calibration_k21(k21; atol::Real = 1.0e-6) = abs(k21) < atol ? nothing :
    throw(ArgumentError("K must be upper-triangular: K[2,1] must be 0, got $(k21)."))

# The calibration matrix of an affine pixel map `x ↦ A·x + t` (row 3 is `0 0 1`
# by its class): the rest of the structure rule, `A[2, 1] = 0`, and the focals.
function _calibration_from_affine(::Type{T}, A, t; check::Bool = true) where {T}
    (size(A) == (2, 2) && length(t) == 2) || throw(
        ArgumentError(
            "a calibration matrix is an affine map of the image plane (a 2×2 linear part and a 2-vector " *
                "translation); got a $(size(A)) linear part and a $(length(t))-vector"
        )
    )
    _assert_calibration_k21(A[2, 1])
    check && _assert_calibration_focals(A)
    return CameraCalibrationMat{T}(A[1, 1], A[2, 2], A[1, 2], t[1], t[2], _Unchecked())
end
function _assert_calibration_focals(M)
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
The pinhole structure (upper-triangular, last row `[0, 0, 1]`) is always
checked, and names the transpose when the rows are swapped; `check` adds the
positive focals.

# Example
```julia
# From OpenCV JSON (row-major nested arrays)
Km = [[800, 0, 320], [0, 800, 240], [0, 0, 1]]
K = CameraCalibrationMat(stack(Km)')
```
"""
CameraCalibrationMat(M::AbstractMatrix{T}; check::Bool = true) where {T <: Real} =
    CameraCalibrationMat{_calibration_eltype(T)}(M; check)

# The stored element type of a calibration matrix built from a `T` matrix: an
# exact number (integer, rational, irrational) or an abstract eltype is
# promoted to Float64, as it always was; any other concrete real (a
# ForwardDiff Dual) is kept.
_calibration_eltype(::Type{<:Union{Integer, Rational, AbstractIrrational}}) = Float64
_calibration_eltype(::Type{T}) where {T <: Real} = isconcretetype(T) ? T : Float64

# Static-matrix inputs: StaticArrays' generic converter `(::Type{SA})(::StaticArray)`
# is more specific than the AbstractMatrix method above and would bypass the
# exact-number promotion. This one is more specific still (bounded T), so an
# SMatrix/RotMatrix/MMatrix K is checked like any other matrix (dataset loaders
# build K as an SMatrix).
CameraCalibrationMat(M::StaticArrays.StaticArray{<:Tuple, T, 2}; check::Bool = true) where {T <: Real} =
    CameraCalibrationMat{_calibration_eltype(T)}(M; check)
CameraCalibrationMat(K::CameraCalibrationMat) = K

"""
    CameraCalibrationMat(s::LinearAlgebra.UniformScaling)
        -> CameraCalibrationMat

Identity shortcut: `CameraCalibrationMat(I)` returns
`CameraCalibrationMat{Float64}` with `K = SMatrix{3,3,Float64}(I)`.  Supply
`Float32(1)*I` to choose the eltype. Only the identity: `α·I` with `α ≠ 1` has
`K₃₃ = α`, not a calibration matrix's structure (`ArgumentError`; the
calibration of focal `α` is `CameraCalibrationMat(Diagonal(SVector(α, α, 1)))`).
"""
function CameraCalibrationMat(s::LinearAlgebra.UniformScaling{T}) where {T <: Real}
    s.λ > 0 || throw(
        ArgumentError(
            "UniformScaling scalar must be positive (got $(s.λ))"
        )
    )
    Tf = T <: AbstractFloat ? T : Float64
    return CameraCalibrationMat{Tf}(SMatrix{3, 3, Tf}(s))
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
        K_inv = SMatrix{3, 3, T, 9}(inv(K))   # inv(K): the analytic inverse, an SMatrix with exact structural zeros
        return new{T}(K, K_inv)
    end
end

# Outer constructor: deduce T from K
LogicalIntrinsics(K::CameraCalibrationMat{T}) where {T <: Real} =
    LogicalIntrinsics{T}(K)

# `_K` / `_K_inv` are the seam where the camera reads its pixel matrix and its
# inverse for the projective (centred-pixel) maps: `project` (through
# `_pixel_affine`), `backproject`, `distortion_map` / `undistort` and the
# undistorted-rect frame walk. `LogicalIntrinsics` stores K; an intrinsics block
# whose K is implicit (`ProjectiveIntrinsics`) adds these two methods and shares
# those maps. The metric operations (`calibration_matrix`, `unproject`, `angle_of_view`,
# ...) read the `LogicalIntrinsics` K directly and have no projective method.
@inline _K(intrinsics::LogicalIntrinsics) = intrinsics.K
@inline _K_inv(intrinsics::LogicalIntrinsics) = intrinsics.K_inv

"""
    focal_length(intrinsics::LogicalIntrinsics) -> Tuple{Float64, Float64}

Get focal lengths (fx, fy) from LogicalIntrinsics K matrix (unitless, in pixels).

Returns a tuple (fx, fy) extracted from the calibration matrix K.
For square pixels, fx ≈ fy.

The other forms answer the focal length a FULL field of view `θ` implies:
`focal_length(resolution::Size2, θ; dimension = :horizontal)` in pixels and
`focal_length(sensor::Sensor, θ; dimension = :horizontal)` as a length (see
their docstrings); [`angle_of_view`](@ref) is their inverse for a camera.

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
