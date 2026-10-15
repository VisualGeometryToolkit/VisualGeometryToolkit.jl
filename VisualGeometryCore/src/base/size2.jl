# =============================================================================
# Size2 — a width/height container (plain numbers or quantities)
# =============================================================================

"""
    Size2{T}

Width/height container with (optionally unit-carrying) elements for image
and canvas sizes.

- `width`: columns (x-axis)
- `height`: rows (y-axis)

Constructors:
- `Size2(; width, height)` keyword constructor
- `Size2(matrix)` from a matrix's `(width, height)`

Examples:
```julia
Size2(width=800, height=600)
Size2(width=210mm, height=297mm)
Size2(width=800px, height=600px)
```
"""
struct Size2{T} <: FieldVector{2, T}
    width::T   # columns, x-axis
    height::T  # rows, y-axis
end

Size2(; width, height) = Size2(width, height)
Size2(matrix::AbstractMatrix) = Size2(reverse(size(matrix))...)
Size2(matrix::StaticArrays.StaticArray{<:Tuple, <:Any, 2}) = Size2(reverse(size(matrix))...)   # vs StaticArrays' converters

# Preserve Size2 type under broadcasting operations like ustrip.()
StaticArrays.similar_type(::Type{Size2{T}}, ::Type{T2}, s::StaticArrays.Size{(2,)}) where {T, T2} = Size2{T2}
StaticArrays.similar_type(::Type{Size2{T}}, ::Type{T2}) where {T, T2} = Size2{T2}

# NOTE: aspect_ratio(::Size2) and area(::Size2) accessor methods live in
# base/geometry_utils.jl (the interface stubs load after this file). Size2
# deliberately has NO getproperty override — computed properties made every
# field access dynamically dispatched.

# A Size2 is a FieldVector: FlexUnits broadcasts over it and keeps the type,
# `ustrip.(px, s)` and `ubase.(s)` are Size2s. No unit method of its own.
