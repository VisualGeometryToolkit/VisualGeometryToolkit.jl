# =============================================================================
# DescribedFeature — Detection + fixed-length descriptor, StructArray-friendly
# =============================================================================
#
# Combines a Detection (frame + metadata) with an SVector descriptor.
# Uses @generated getproperty to forward all Detection properties at zero cost.
#
# Designed for StructArray storage:
#   sa.detection  → Vector{Detection}
#   sa.descriptor → Vector{SVector{N,Float32}} (contiguous, matrix-reinterpretable)
#
# =============================================================================

# match, NNMatch, NNStrategy and MutualNN come from nn_match.jl; csponds is
# VisualGeometryCore's (matching.jl imports it, nn_match.jl adds the NNMatch method)

"""
    DescribedFeature{D<:Detection, N}

A feature detection paired with a fixed-length descriptor vector.

Wraps a `Detection` (frame + metadata) with an `SVector{N, Float32}` descriptor.
All detection properties (`.center`, `.σ`, `.frame`, `.metadata`, etc.) are
forwarded transparently via `@generated getproperty`.

Use with `StructArray` for efficient columnar storage and zero-copy
descriptor matrix extraction:

```julia
sa = described_features(detections, desc_matrix)
matches = VisualGeometryFeatures.match(sa_board, sa_query, MutualNN())
scored = csponds(sa_board, sa_query, matches)
```
"""
struct DescribedFeature{D <: AbstractDetection, N}
    detection::D
    descriptor::SVector{N, Float32}
end

# =============================================================================
# Property Forwarding — @generated for zero-cost dispatch
# =============================================================================

@generated function Base.getproperty(df::DescribedFeature{D, N}, s::Symbol) where {D, N}
    exprs = Expr[]
    push!(exprs, :(s === :detection  && return getfield(df, :detection)))
    push!(exprs, :(s === :descriptor && return getfield(df, :descriptor)))
    # Forward everything else to the underlying detection
    push!(exprs, :(return getproperty(getfield(df, :detection), s)))
    return Expr(:block, exprs...)
end

@generated function Base.propertynames(::DescribedFeature{D, N}) where {D, N}
    # D = Attributed{F, M} — extract metadata fieldnames for tab completion
    det_props = if D isa DataType && length(D.parameters) >= 2
        M = D.parameters[2]
        M isa DataType ? fieldnames(M) : ()
    else
        ()
    end
    return :(
        (
            :detection, :descriptor, :frame, :metadata, :center, :σ, :Σ,
            $(QuoteNode.(det_props)...),
        )
    )
end

# =============================================================================
# Geometry forwarding
# =============================================================================

centroid(df::DescribedFeature) = centroid(getfield(df, :detection))
region(df::DescribedFeature, args...) = region(getfield(df, :detection), args...)
feature_frame(df::DescribedFeature) = feature_frame(getfield(df, :detection))
moment_matrix(df::DescribedFeature) = moment_matrix(getfield(df, :detection))

# =============================================================================
# StructArray constructors and accessors
# =============================================================================

"""
    described_features(detections, desc_matrix) -> StructArray{DescribedFeature}

Build a `StructArray` of `DescribedFeature` from a detection vector and an
`N×D` descriptor matrix. The descriptor dimension `D` becomes the static
size parameter of each `SVector{D, Float32}`.
"""
function described_features(
        detections::AbstractVector{<:AbstractDetection},
        desc_matrix::AbstractMatrix{Float32}
    )
    N, D = size(desc_matrix)
    length(detections) == N || throw(
        DimensionMismatch(
            "$(length(detections)) detections but $N descriptor rows"
        )
    )
    S = SVector{D, Float32}
    descs = N > 0 ? [S(view(desc_matrix, i, :)) for i in 1:N] : S[]
    return StructArray{DescribedFeature}((detection = detections, descriptor = descs))
end

"""
    descriptor_matrix(sa::StructArray{<:DescribedFeature}) -> AbstractMatrix{Float32}

Zero-copy `N×D` descriptor matrix view from a StructArray of DescribedFeature.
Suitable for supplying directly to `match`.
"""
function descriptor_matrix(sa::StructArray{<:DescribedFeature})
    return reinterpret(reshape, Float32, sa.descriptor)'
end

# =============================================================================
# Matching integration — match and csponds on StructArray{DescribedFeature}
# =============================================================================

"""
    match(f1::StructArray{<:DescribedFeature}, f2::StructArray{<:DescribedFeature},
          strategy; kwargs...) -> Vector{NNMatch}

Match described features by their descriptors. Extracts descriptor matrices
and delegates to the matrix-based `match`.
"""
function match(
        f1::StructArray{<:DescribedFeature},
        f2::StructArray{<:DescribedFeature},
        strategy::NNStrategy = MutualNN(); kwargs...
    )
    return match(descriptor_matrix(f1), descriptor_matrix(f2), strategy; kwargs...)
end
