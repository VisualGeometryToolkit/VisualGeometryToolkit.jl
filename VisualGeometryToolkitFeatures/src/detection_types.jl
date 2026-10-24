# =============================================================================
# Feature Detection Types
# =============================================================================
#
# Types specific to feature detection results, separate from geometric primitives.
#
# Key design: one parametric Detection{F,M} for every frame type F (a FeatureFrame)
# and metadata M, rather than a type per detector.

# =============================================================================
# Detection — Type alias for Attributed{<:FeatureFrame, M}
# =============================================================================

# A feature frame with its covariance, VisualGeometryToolkitCore's `Uncertain` — what
# `asift_detect(…; uncertainty = FeatureCovariance())` returns as a detection's frame.
const _UncertainFrame = Uncertain{<:FeatureFrame, <:CovMat}

# Every detection, a frame's or an uncertain frame's, is VGC's `AbstractDetection`
# (`Attributed{<:Union{FeatureFrame, Uncertain{<:FeatureFrame}}}`, VGC ≥ 0.31.1).

"""
    Detection{F<:FeatureFrame, M}

Type alias for `Attributed{F, M}` — a detection result composing a geometric
frame with detector-specific metadata.

The frame type `F` determines the geometry:
- `Detection{Disc{T}, M}` - isotropic detection (3 DOF)
- `Detection{OrientedDisc{T}, M}` - oriented isotropic detection (4 DOF)
- `Detection{EllipticFrame{T}, M}` - anisotropic detection (5 DOF)
- `Detection{AffineFrame{T}, M}` - full affine detection (6 DOF)
- `Detection{<:Uncertain{F}, M}` - a frame `F` with its covariance (VisualGeometryToolkitCore's
  `Uncertain(frame, Σ)`), from `asift_detect(…; uncertainty = FeatureCovariance())`:
  `cov(d)` is the covariance, `feature_frame(d)` the bare frame, `geom(d)` the
  detection without it, and the properties below read as on the bare detection

# Type Parameters
- `F<:FeatureFrame`: The geometric frame type
- `M`: Metadata, a `NamedTuple`. ASIFT emits `(response, edge_score,
  laplacian_scale_score, warp_idx)`; `laplacian_scale_score` is the polarity
  field [`is_dark`](@ref)/[`is_light`](@ref) read (> 0 dark, < 0 light).

# Fields (from Attributed)
- `value::F`: The geometric frame (center + shape)
- `attributes::M`: Detector-specific scores and information

# Coordinates
The frame's `center` is `(x, y)` in COLMAP pixel-centre coordinates, the
convention of every detector in this package: x along the image columns, y along the rows,
the top-left pixel's centre at (0.5, 0.5). The pixel `img[r, c]` has its centre
at `(c − 0.5, r − 0.5)`. [`asift_detect`](@ref) documents its measured sub-pixel
accuracy.

# Property Access
Properties are forwarded from frame and metadata:
- `d.frame`, `d.value` → the geometric frame
- `d.metadata`, `d.attributes` → the metadata
- `d.center` → `frame.center`
- `d.σ` → scale from frame (exact for Disc, characteristic_scale for anisotropic)
- `d.Σ` → moment matrix (blob shape, not measurement covariance)
- `d.response`, `d.edge_score`, etc. → forwarded from metadata

# Example
```jldoctest
julia> using VisualGeometryToolkitFeatures, VisualGeometryToolkitCore

julia> det = Detection(Disc(Point2(100.0, 200.0), 5.0), (response = 0.05, laplacian_scale_score = -0.1));

julia> Tuple(det.center), det.σ, det.response   # from the frame and the metadata
((100.0, 200.0), 5.0, 0.05)

julia> is_light(det)
true
```

See also: [`Disc`](@ref), [`EllipticFrame`](@ref), VisualGeometryToolkitCore's `Attributed`
"""
const Detection{F <: Union{FeatureFrame, _UncertainFrame}, M} = Attributed{F, M}

# Primary constructor (type aliases don't auto-generate constructors)
Detection(frame::F, meta::M) where {F <: FeatureFrame, M} = Attributed(frame, meta)
Detection(frame::_UncertainFrame, meta) = Attributed(frame, meta)

# =============================================================================
# feature_frame() accessor — unified for FeatureFrame and Detection
# =============================================================================

# `feature_frame(::AbstractDetection)` (either kind) is VGC's.
feature_frame(f::FeatureFrame) = f

# An uncertain frame's detection: `cov` its covariance, `geom` the bare detection.
# On NamedTuple metadata (a Detection's): an `Uncertain` is itself an
# `Attributed` with a CovMat/PrecMat attribute, and VGC's `cov`/`geom` own those.
cov(d::Attributed{<:_UncertainFrame, <:NamedTuple}) = cov(getfield(d, :value))
geom(d::Attributed{<:_UncertainFrame, <:NamedTuple}) = Attributed(feature_frame(d), getfield(d, :attributes))

# =============================================================================
# Detection Property Forwarding
# =============================================================================
#
# Specialized @generated getproperty for Attributed{V<:FeatureFrame, A}.
# More specific than the generic Attributed{V,A} method — Julia dispatches
# correctly to this method when V <: FeatureFrame.

# Moment matrix accessor (blob shape). `d.σ` is VGC's `characteristic_scale`:
# σ itself for the isotropic frames, √|det S| for the shape-matrix ones.
_get_moment_matrix(f::Union{Disc{T}, OrientedDisc{T}}) where {T} =
    SMatrix{2, 2, T}(f.σ^2, zero(T), zero(T), f.σ^2)
_get_moment_matrix(f::EllipticFrame) = moment_matrix(f)
_get_moment_matrix(f::AffineFrame) = moment_matrix(f)

@generated function Base.getproperty(d::Attributed{V, A}, s::Symbol) where {V <: FeatureFrame, A}
    exprs = Expr[]
    push!(exprs, :(s === :value      && return getfield(d, :value)))
    push!(exprs, :(s === :attributes && return getfield(d, :attributes)))
    push!(exprs, :(s === :frame      && return getfield(d, :value)))       # compat
    push!(exprs, :(s === :metadata   && return getfield(d, :attributes)))  # compat
    push!(exprs, :(s === :center     && return getfield(d, :value).center))
    push!(exprs, :(s === :σ          && return characteristic_scale(getfield(d, :value))))
    push!(exprs, :(s === :Σ          && return _get_moment_matrix(getfield(d, :value))))
    for name in fieldnames(A)
        push!(exprs, :(s === $(QuoteNode(name)) && return getfield(getfield(d, :attributes), $(QuoteNode(name)))))
    end
    push!(exprs, :(return getproperty(getfield(d, :value), s)))
    return Expr(:block, exprs...)
end

@generated function Base.propertynames(d::Attributed{V, A}) where {V <: FeatureFrame, A}
    attr_names = fieldnames(A)
    return :((:value, :attributes, :frame, :metadata, :center, :σ, :Σ, $(QuoteNode.(attr_names)...)))
end

# The same for a detection of an uncertain frame: `value` is the `Uncertain`,
# `frame` and the geometry read through to the bare frame.
@generated function Base.getproperty(d::Attributed{V, A}, s::Symbol) where {V <: _UncertainFrame, A}
    exprs = Expr[]
    push!(exprs, :(s === :value      && return getfield(d, :value)))
    push!(exprs, :(s === :attributes && return getfield(d, :attributes)))
    push!(exprs, :(s === :frame      && return feature_frame(d)))
    push!(exprs, :(s === :metadata   && return getfield(d, :attributes)))
    push!(exprs, :(s === :center     && return feature_frame(d).center))
    push!(exprs, :(s === :σ          && return characteristic_scale(feature_frame(d))))
    push!(exprs, :(s === :Σ          && return _get_moment_matrix(feature_frame(d))))
    for name in fieldnames(A)
        push!(exprs, :(s === $(QuoteNode(name)) && return getfield(getfield(d, :attributes), $(QuoteNode(name)))))
    end
    push!(exprs, :(return getproperty(feature_frame(d), s)))
    return Expr(:block, exprs...)
end

@generated function Base.propertynames(d::Attributed{V, A}) where {V <: _UncertainFrame, A}
    attr_names = fieldnames(A)
    return :((:value, :attributes, :frame, :metadata, :center, :σ, :Σ, $(QuoteNode.(attr_names)...)))
end

# Centroid for generic geometry interface
centroid(d::Attributed{<:FeatureFrame}) = getfield(d, :value).center
centroid(d::Attributed{<:_UncertainFrame}) = feature_frame(d).center

# Region forwarding — converts Detection to its supported k·σ primitive (Circle/Ellipse)
region(d::Attributed{<:FeatureFrame}, args...) = region(getfield(d, :value), args...)
region(d::Attributed{<:_UncertainFrame}, args...) = region(feature_frame(d), args...)

# Moment matrix access (blob shape)
moment_matrix(d::Attributed{<:FeatureFrame}) = _get_moment_matrix(getfield(d, :value))
moment_matrix(d::Attributed{<:_UncertainFrame}) = _get_moment_matrix(feature_frame(d))

# Custom show to print as "Detection(...)" for readability
function Base.show(io::IO, d::Attributed{V, A}) where {V <: FeatureFrame, A}
    print(io, "Detection(")
    show(io, getfield(d, :value))
    print(io, ", ")
    show(io, getfield(d, :attributes))
    return print(io, ")")
end

function Base.show(io::IO, d::Attributed{V, A}) where {V <: _UncertainFrame, A}
    print(io, "Detection(Uncertain(")
    show(io, feature_frame(d))
    print(io, ", ")
    show(io, cov(d))
    print(io, "), ")
    show(io, getfield(d, :attributes))
    return print(io, ")")
end

# =============================================================================
# Detection Constructors
# =============================================================================

# From frame + response with polarity symbol. `:light` is ASIFT's word for
# `:bright`; `:unknown` records no polarity (NaN). Any other symbol is a typo,
# which used to become NaN silently.
function Detection(frame::F, response::Real; polarity::Symbol = :unknown) where {F <: FeatureFrame}
    laplacian = if polarity === :dark
        1.0
    elseif polarity === :bright || polarity === :light
        -1.0
    elseif polarity === :unknown
        NaN
    else
        throw(ArgumentError("polarity must be :dark, :light (or :bright) or :unknown, got :$polarity"))
    end
    meta = (response = Float64(response), laplacian_scale_score = laplacian)
    return Attributed(frame, meta)
end

# =============================================================================
# Polarity Traits (Compile-time dispatch for metadata types)
# =============================================================================

"""
    HasPolarityTrait

Holy trait for metadata types indicating whether they contain polarity
information: [`HasPolarity`](@ref) or [`NoPolarity`](@ref), as
[`polarity_trait`](@ref) returns it. Public, not exported.
"""
abstract type HasPolarityTrait end

"""
    HasPolarity()

The [`HasPolarityTrait`](@ref) of detection metadata that records a polarity:
a `NamedTuple` with a `laplacian_scale_score` field, positive for a dark blob
and negative for a light one. [`is_dark`](@ref) and [`is_light`](@ref) read
that field's sign. Public, not exported.
"""
struct HasPolarity <: HasPolarityTrait end

"""
    NoPolarity()

The [`HasPolarityTrait`](@ref) of detection metadata that records no polarity.
[`is_dark`](@ref) and [`is_light`](@ref) are both `false` for such a detection.
Public, not exported.
"""
struct NoPolarity <: HasPolarityTrait end

"""
    polarity_trait(::Type{M}) -> HasPolarityTrait
    polarity_trait(d::Detection) -> HasPolarityTrait

Whether metadata of type `M`, or the metadata of the detection `d`, records a
polarity: [`HasPolarity`](@ref) for a `NamedTuple` with a
`laplacian_scale_score` field, [`NoPolarity`](@ref) otherwise. A detector whose
metadata is another type opts in by adding a method for its type, and must then
give it a `laplacian_scale_score` property. Public, not exported.

```jldoctest
julia> using VisualGeometryToolkitFeatures, VisualGeometryToolkitCore

julia> import VisualGeometryToolkitFeatures as VgtkFeatures

julia> VgtkFeatures.polarity_trait(Detection(Disc(Point2(1.0, 2.0), 3.0), (response = 1.0,)))
VisualGeometryToolkitFeatures.NoPolarity()

julia> VgtkFeatures.polarity_trait(typeof((response = 1.0, laplacian_scale_score = -1.0)))
VisualGeometryToolkitFeatures.HasPolarity()
```
"""
function polarity_trait end

@generated function polarity_trait(::Type{M}) where {M <: NamedTuple}
    if :laplacian_scale_score in fieldnames(M)
        return :(HasPolarity())
    else
        return :(NoPolarity())
    end
end

polarity_trait(::Type{M}) where {M} = NoPolarity()
polarity_trait(::Type{<:Attributed{F, M}}) where {F <: FeatureFrame, M} = polarity_trait(M)
polarity_trait(::Type{<:Attributed{F, M}}) where {F <: _UncertainFrame, M} = polarity_trait(M)
polarity_trait(d::AbstractDetection) = polarity_trait(typeof(getfield(d, :attributes)))

# =============================================================================
# Polarity Filtering (Light/Dark Classification)
# =============================================================================

@inline _is_light(d, ::HasPolarity) = d.laplacian_scale_score < 0
@inline _is_light(d, ::NoPolarity) = false

@inline _is_dark(d, ::HasPolarity) = d.laplacian_scale_score > 0
@inline _is_dark(d, ::NoPolarity) = false

"""
    is_light(d::Detection) -> Bool

Whether `d` is a light blob on a darker background: its metadata records a
polarity (see [`polarity_trait`](@ref)) and its `laplacian_scale_score` is
negative. `false` for a detection without a recorded polarity. ASIFT
detections record one, from the sign of the DoG response.

```jldoctest
julia> using VisualGeometryToolkitFeatures, VisualGeometryToolkitCore

julia> d = Detection(Disc(Point2(1.0, 2.0), 3.0), 0.5; polarity = :light);

julia> is_light(d), is_dark(d)
(true, false)
```

See also [`is_dark`](@ref).
"""
@inline is_light(d::AbstractDetection) = _is_light(d, polarity_trait(d))

"""
    is_dark(d::Detection) -> Bool

Whether `d` is a dark blob on a lighter background: its metadata records a
polarity (see [`polarity_trait`](@ref)) and its `laplacian_scale_score` is
positive. `false` for a detection without a recorded polarity.

```jldoctest
julia> using VisualGeometryToolkitFeatures, VisualGeometryToolkitCore

julia> is_dark(Detection(Disc(Point2(1.0, 2.0), 3.0), 0.5; polarity = :dark))
true

julia> is_dark(Detection(Disc(Point2(1.0, 2.0), 3.0), 0.5))   # no polarity recorded
false
```

See also [`is_light`](@ref).
"""
@inline is_dark(d::AbstractDetection) = _is_dark(d, polarity_trait(d))

# =============================================================================
# Geometry Conversions
# =============================================================================

"""
    Disc(d::Detection) -> Disc

Extract or convert to a Disc frame (uses characteristic_scale for anisotropic).
"""
Disc(d::Attributed{<:Disc}) = getfield(d, :value)
Disc(d::Attributed{<:FeatureFrame}) = Disc(d.center, d.σ)
Disc(d::Attributed{<:_UncertainFrame}) = Disc(geom(d))

"""
    EllipticFrame(d::Detection) -> EllipticFrame

Extract or convert to an EllipticFrame.
"""
EllipticFrame(d::Attributed{<:EllipticFrame}) = getfield(d, :value)
EllipticFrame(d::Attributed{<:Disc}) = EllipticFrame(getfield(d, :value))
EllipticFrame(d::Attributed{<:OrientedDisc}) = EllipticFrame(Disc(getfield(d, :value)))
EllipticFrame(d::Attributed{<:AffineFrame}) = EllipticFrame(getfield(d, :value))
EllipticFrame(d::Attributed{<:_UncertainFrame}) = EllipticFrame(geom(d))

"""
    Ellipse(d::Detection; cutoff=1.0) -> Ellipse

Convert to an ellipse at the `cutoff`·σ contour (via `region`).
"""
Ellipse(d::Attributed{<:FeatureFrame}; cutoff::Real = 1.0) = Ellipse(region(getfield(d, :value), cutoff))
Ellipse(d::Attributed{<:_UncertainFrame}; cutoff::Real = 1.0) = Ellipse(geom(d); cutoff)

# =============================================================================
# JSON Serialization
# =============================================================================
#
# A detection is written as one JSON object: its frame under `frame`, its
# covariance (if any) under `cov`, and every metadata field under its own name.
# `JSON.json(d)` writes it and `JSON.parse(str, Detection)` reads it back; the
# hooks are JSON.jl's `lower` and `lift` (see serialization.jl). They are
# declared on `Attributed` of a feature frame only, so no other `Attributed`
# (e.g. a robust estimate wrapping a vector) changes how it serializes.

const _DetectionTypes = Attributed{<:Union{FeatureFrame, _UncertainFrame}}

StructUtils.structlike(::Type{<:_DetectionTypes}) = false

function JSON.lower(::JSON.JSONStyle, d::Attributed{V, A}) where {V <: FeatureFrame, A}
    meta = getfield(d, :attributes)
    return (frame = getfield(d, :value), Base.pairs(meta)...)
end

# An uncertain frame writes its covariance as `cov`, the matrix's entries in
# column order — or `null` when it is NaN (an unrefined keypoint's), which JSON
# cannot hold. Read back, `cov` makes the frame `Uncertain` again.
function JSON.lower(::JSON.JSONStyle, d::Attributed{V, A}) where {V <: _UncertainFrame, A}
    Σ = cov(d)
    c = all(isfinite, Σ) ? vec(collect(Float64, Σ)) : nothing
    return (frame = feature_frame(d), cov = c, Base.pairs(getfield(d, :attributes))...)
end

# The metadata fields every detector writes as Float64, in their canonical order.
const _FLOAT_METADATA = (:response, :edge_score, :laplacian_scale_score)

function JSON.lift(::Type{<:_DetectionTypes}, x::_JSONObject)
    frame_data = _getfield(x, "frame")
    # The frame's type is read off its keys.
    fr = if _hasfield(frame_data, "A")
        JSON.lift(AffineFrame, frame_data)
    elseif _hasfield(frame_data, "S")
        JSON.lift(EllipticFrame, frame_data)
    elseif _hasfield(frame_data, "θ")
        JSON.lift(OrientedDisc, frame_data)
    else
        JSON.lift(Disc, frame_data)
    end

    # Every key but `frame` and `cov` is metadata. The documented fields come
    # first, as Float64 (a writer may print 1.0 as 1), then the others sorted by
    # name, so the result does not depend on the key order of the input. Fields
    # outside the documented ones (ASIFT's warp_idx among them) are kept.
    present = Symbol[Symbol(k) for k in _keys(x) if k ∉ ("frame", "cov")]
    names = (filter(in(present), collect(_FLOAT_METADATA))..., sort!(setdiff(present, _FLOAT_METADATA))...)
    vals = Tuple(n in _FLOAT_METADATA ? Float64(_getfield(x, String(n))) : _getfield(x, String(n)) for n in names)

    meta = NamedTuple{names}(vals)
    _hasfield(x, "cov") || return Attributed(fr, meta)
    c = _getfield(x, "cov")
    N = c === nothing ? 3 : isqrt(length(c))
    Σ = c === nothing ? fill(NaN, 3, 3) : reshape(Float64.(collect(c)), N, N)
    return Attributed(Uncertain(fr, CovMat{N, Float64, N * N}(Tuple(Σ))), meta)
end
