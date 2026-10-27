"""
    Board{T}

The calibration board a planar target's pattern is printed on, the target's
analogue of a [`Sensor`](@ref): its size and where it lies in the design chart.
It follows the target's unit rule: physical (lengths, stored in metres) for a
physical target, plain numbers for a unitless one.

# Fields
- `size::Size2{T}`: the board proper, width × height — BlobBoards' `board_size`,
  the printed board inside the sheet's white margin, frame and corners
  included; NOT its `canvas_size` (the sheet) and not the pattern's extent.
  Lengths (a quantity of any other dimension is a `DimensionError`) or plain
  numbers; a mix is an `ArgumentError`.
- `origin::SVector{2, Float64}`: the board's lower-left corner in the design chart
  (the coordinates of the nominal points): plain metres for a physical board,
  the design's plain numbers for a unitless one.

Read them as properties, `board.size` and `board.origin`, as a sensor's
`resolution` and `pitch` are.

# Constructors
```julia
Board(Size2(width = 297mm, height = 210mm))                              # corner at the design origin
Board(Size2(width = 297mm, height = 210mm), [-10.0mm, -10.0mm])          # the corner as lengths
Board(Size2(width = 297mm, height = 210mm), SVector(-0.01, -0.01))       # or plain metres
Board(Size2(width = 297.0, height = 210.0), [-10.0, -10.0])              # a unitless board
```

`Rect(board)` is the board's domain in the design chart in the board's own
numbers (plain metres when physical) — what a target's [`surface`](@ref) is
bounded by, as `Rect(sensor)` is the image's domain in pixels. `ustrip(u, board)`
and [`with_unit`](@ref)`(board, u)` move it between the two forms, as they move
a target's points.

# See Also
- [`aspect_ratio`](@ref)
- [`PlanarTarget`](@ref), [`board`](@ref)
"""
struct Board{T <: Number}
    size::Size2{T}                  # lengths (stored in metres, FlexUnits' base form) or plain numbers
    origin::SVector{2, Float64}     # the lower-left corner in the design chart: plain metres, or plain

    # Inner constructor — the dims enforcement boundary: the size two
    # non-negative lengths (anything else is a `DimensionError`) or two
    # non-negative plain numbers.
    function Board{T}(size::Size2{T}, origin::SVector{2, Float64}) where {T <: Number}
        T <: Quantity && foreach(_check_length, size)
        all(x -> _board_number(x) >= 0, size) || throw(ArgumentError("a board's size is non-negative, got $size"))
        return new{T}(size, origin)
    end
end
_board_number(x::Quantity) = ustrip(m, x)
_board_number(x::Real) = Float64(x)

# Constructors: a FlexUnits length in, stored in FlexUnits' base form
# `Quantity{Float64, Dimensions{FixRat32}}` (metres); plain numbers as Float64.
function Board(size::Size2{<:Quantity}, origin::AbstractVector = SVector(0.0, 0.0))
    Q = Quantity{Float64, Dimensions{FixRat32}}
    return Board{Q}(Size2{Q}(map(q -> _check_length(q), size)), _board_origin(origin))
end
function Board(size::Size2{<:Real}, origin::AbstractVector = SVector(0.0, 0.0))
    any(x -> x isa Quantity, origin) && _refuse_board_mix(size, origin)
    return Board{Float64}(Size2{Float64}(size), SVector{2, Float64}(origin))
end
Board(size::Size2, origin::AbstractVector = SVector(0.0, 0.0)) = _refuse_board_mix(size, origin)
@noinline _refuse_board_mix(size, origin) = throw(
    ArgumentError("Board: the size and the corner are all lengths or all plain numbers; got $(Tuple(size)) and $(Tuple(origin))")
)
_board_origin(o::AbstractVector{<:Quantity}) = SVector{2, Float64}(strip_metres(map(_check_length, SVector{2}(o))))
_board_origin(o::AbstractVector{<:Real}) = SVector{2, Float64}(o)
_board_origin(o::AbstractVector) = _refuse_board_mix((), o)

Base.:(==)(a::Board, b::Board) = a.size == b.size && a.origin == b.origin
Base.hash(bd::Board, h::UInt) = hash(bd.origin, hash(bd.size, hash(:Board, h)))

# Units: the target's rule. A physical board's lengths go to the plain unit
# `u` (its corner from metres); a plain one's numbers become lengths in `u`.
has_units(::Board{<:Quantity}) = true
has_units(::Board) = false
_form_sample(bd::Board) = bd.size.width
FlexUnits.ustrip(u::Units, bd::Board{<:Quantity}) =
    Board(Size2(width = ustrip(u, bd.size.width), height = ustrip(u, bd.size.height)), map(x -> _from_metres(x, u), bd.origin))
FlexUnits.ustrip(::Units, ::Board) = throw(ArgumentError("ustrip: the board is plain numbers already; `with_unit` attaches a unit"))
with_unit(::Board{<:Quantity}, ::Units) = throw(ArgumentError("with_unit: the board already has units"))
with_unit(bd::Board, u::Units) =
    Board(Size2(width = bd.size.width * u, height = bd.size.height * u), map(x -> _to_metres(x, u), bd.origin))

# Accessors

"""
    aspect_ratio(board) -> Float64

Width-to-height ratio of the board.

```julia
aspect_ratio(Board(Size2(width = 297mm, height = 210mm)))  # 1.414…
```
"""
aspect_ratio(bd::Board) = _board_number(bd.size.width) / _board_number(bd.size.height)

"""
    Rect(board::Board) -> Rect2{Float64}

The board's domain in the design chart: origin `board.origin`, widths the
board's size, in the board's own numbers — plain metres for a physical board,
its plain numbers for a unitless one. A planar target's [`surface`](@ref) is
bounded by it.
"""
Rect(board::Board) = Rect2{Float64}(board.origin[1], board.origin[2], _board_number(board.size.width), _board_number(board.size.height))
