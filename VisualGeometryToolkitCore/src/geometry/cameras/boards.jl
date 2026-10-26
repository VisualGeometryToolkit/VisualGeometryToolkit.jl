"""
    Board{T}

The calibration board a planar target's pattern is printed on, the target's
analogue of a [`Sensor`](@ref): its size and where it lies in the design chart.

# Fields
- `size::Size2{T}`: the board proper, width × height (lengths; anything else is a
  `DimensionError`) — BlobBoards' `board_size`, the printed board inside the
  sheet's white margin, frame and corners included; NOT its `canvas_size` (the
  sheet) and not the pattern's extent
- `origin::SVector{2, Float64}`: the board's lower-left corner in the design chart
  (the coordinates of the nominal points), plain metres

Read them as properties, `board.size` and `board.origin`, as a sensor's
`resolution` and `pitch` are.

# Constructors
```julia
Board(Size2(width = 297mm, height = 210mm))                              # corner at the design origin
Board(Size2(width = 297mm, height = 210mm), [-10.0mm, -10.0mm])          # the corner as lengths
Board(Size2(width = 297mm, height = 210mm), SVector(-0.01, -0.01))       # or plain metres
```

`Rect(board)` is the board's domain in the design chart, plain metres — what a
target's [`surface`](@ref) is bounded by, as `Rect(sensor)` is the image's
domain in pixels.

# See Also
- [`aspect_ratio`](@ref)
- [`PlanarTarget`](@ref), [`board`](@ref)
"""
struct Board{T <: Quantity}
    size::Size2{T}                  # a length each (stored in metres, FlexUnits' base form)
    origin::SVector{2, Float64}     # the lower-left corner in the design chart, plain metres

    # Inner constructor — the dims enforcement boundary: the size two
    # non-negative lengths (anything else is a `DimensionError`).
    function Board{T}(size::Size2{T}, origin::SVector{2, Float64}) where {T <: Quantity}
        foreach(_check_length, size)
        all(q -> ustrip(m, q) >= 0, size) || throw(ArgumentError("a board's size is non-negative, got $size"))
        return new{T}(size, origin)
    end
end

# Constructors: any FlexUnits length in, stored in FlexUnits' base form
# `Quantity{Float64, Dimensions{FixRat32}}` (metres).
function Board(size::Size2{<:Quantity}, origin::AbstractVector = SVector(0.0, 0.0))
    Q = Quantity{Float64, Dimensions{FixRat32}}
    return Board{Q}(Size2{Q}(map(q -> _check_length(q), size)), _board_origin(origin))
end
_board_origin(o::AbstractVector{<:Quantity}) = SVector{2, Float64}(map(q -> ustrip(m, _check_length(q)), Tuple(SVector{2}(o))))
_board_origin(o::AbstractVector{<:Real}) = SVector{2, Float64}(o)

Base.:(==)(a::Board, b::Board) = a.size == b.size && a.origin == b.origin
Base.hash(bd::Board, h::UInt) = hash(bd.origin, hash(bd.size, hash(:Board, h)))

# Accessors

"""
    aspect_ratio(board) -> Float64

Width-to-height ratio of the board.

```julia
aspect_ratio(Board(Size2(width = 297mm, height = 210mm)))  # 1.414…
```
"""
aspect_ratio(bd::Board) = ustrip(m, bd.size.width) / ustrip(m, bd.size.height)

"""
    Rect(board::Board) -> Rect2{Float64}

The board's domain in the design chart: origin `board.origin`, widths the board's
size, plain metres. A planar target's [`surface`](@ref) is bounded by it.
"""
Rect(board::Board) = Rect2{Float64}(board.origin[1], board.origin[2], ustrip(m, board.size.width), ustrip(m, board.size.height))
