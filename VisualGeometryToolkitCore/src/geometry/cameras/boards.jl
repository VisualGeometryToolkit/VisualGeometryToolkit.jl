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
  (the coordinates of the nominal points, stated once at [`Canvas`](@ref)):
  plain metres for a physical board, the design's plain numbers for a unitless
  one.

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
bounded by (grown by a 16-eps rounding margin), as `Rect(sensor)` is the
image's domain in pixels. `ustrip(u, board)`
and [`with_unit`](@ref)`(board, u)` move it between the two forms, as they move
a target's points.

# See Also
- [`aspect_ratio`](@ref)
- [`PlanarTarget`](@ref), [`board`](@ref)
- [`Canvas`](@ref): the printed page the board lies on
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
# `BaseQuantity` (metres); plain numbers as Float64. One rule for every
# rectangle of the design chart — the board, the canvas, a canvas texture:
# `_stored_rect` returns the stored size and corner.
Board(size::Size2, origin::AbstractVector = SVector(0.0, 0.0)) = _new_rect(Board, _stored_rect("Board", size, origin)...)
_new_rect(::Type{B}, size::Size2{T}, origin, rest...) where {B, T} = B{T}(size, origin, rest...)
function _stored_rect(who, size::Size2{<:Quantity}, origin::AbstractVector)
    Q = BaseQuantity
    return Size2{Q}(map(q -> _check_length(q), size)), _board_origin(origin)
end
function _stored_rect(who, size::Size2{<:Real}, origin::AbstractVector)
    any(x -> x isa Quantity, origin) && _refuse_board_mix(who, size, origin)
    return Size2{Float64}(size), SVector{2, Float64}(origin)
end
_stored_rect(who, size::Size2, origin::AbstractVector) = _refuse_board_mix(who, size, origin)
@noinline _refuse_board_mix(who, size, origin) = throw(
    ArgumentError("$who: the size and the corner are all lengths or all plain numbers; got $(Tuple(size)) and $(Tuple(origin))")
)
_board_origin(o::AbstractVector{<:Quantity}) = SVector{2, Float64}(strip_metres(map(_check_length, SVector{2}(o))))
_board_origin(o::AbstractVector{<:Real}) = SVector{2, Float64}(o)
_board_origin(o::AbstractVector) = _refuse_board_mix("Board", (), o)

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

# ------------------------------------------------------------------------------
# Canvas: the printed page, and the image printed on it
# ------------------------------------------------------------------------------

"""
    CanvasTexture{T}

The image printed on a [`Canvas`](@ref): a reference to the image file and the
rectangle of the design chart it covers. VGC never reads the image; the
reference is what a reader resolves and checks.

# Fields
- `path::String`: the image file, RELATIVE to the file that names it (a rig or
  board file); an absolute path is an `ArgumentError`.
- `sha256::String`: the SHA-256 of the image file's bytes, 64 lowercase hex
  digits (an `ArgumentError` otherwise) — the reader's integrity check.
- `size::Size2{T}`, `origin::SVector{2, Float64}`: the rectangle the image
  covers, by the board's rule (lengths stored in metres or plain numbers; its
  lower-left corner in the design chart, plain metres when physical). The
  size is positive.

The image's first row lies along the rectangle's top edge (largest y) and
its first column along the left edge (smallest x); its outer pixel edges lie
on the rectangle's edges, so each pixel of a `W × H` image covers
`width/W × height/H` of the chart.

# Constructors
```julia
CanvasTexture("board.png", sha, Size2(width = 297mm, height = 210mm))                    # corner at the design origin
CanvasTexture("board.png", sha, Size2(width = 297mm, height = 210mm), [-10.0mm, -5.0mm])
CanvasTexture("board.png", sha, Size2(width = 297.0, height = 210.0), [-10.0, -5.0])     # a unitless one
```

`Rect(texture)` is the covered rectangle in its own numbers (plain metres
when physical).
"""
struct CanvasTexture{T <: Number}
    path::String
    sha256::String
    size::Size2{T}
    origin::SVector{2, Float64}
    function CanvasTexture{T}(path::AbstractString, sha256::AbstractString, size::Size2{T}, origin::SVector{2, Float64}) where {T <: Number}
        T <: Quantity && foreach(_check_length, size)
        all(x -> _board_number(x) > 0, size) || throw(ArgumentError("a canvas texture's size is positive, got $size"))
        isempty(path) && throw(ArgumentError("CanvasTexture: the image path is empty"))
        isabspath(path) && throw(ArgumentError("CanvasTexture: the image path is relative to the file that names it; got the absolute path $(repr(path))"))
        occursin(r"\A[0-9a-f]{64}\z", sha256) ||
            throw(ArgumentError("CanvasTexture: sha256 is 64 lowercase hex digits; got $(repr(sha256))"))
        return new{T}(String(path), String(sha256), size, origin)
    end
end
function CanvasTexture(path::AbstractString, sha256::AbstractString, size::Size2, origin::AbstractVector = SVector(0.0, 0.0))
    s, o = _stored_rect("CanvasTexture", size, origin)
    return _texture(path, sha256, s, o)
end
_texture(path, sha256, size::Size2{T}, origin) where {T} = CanvasTexture{T}(path, sha256, size, origin)

"""
    Canvas{T}

The page a planar target is printed on: the sheet that holds the [`Board`](@ref)
(the calibrated rectangle) and around it the margin; the pattern is what is
printed on the board. A canvas ⊃ board ⊃ pattern hierarchy, the canvas being
the target's record of what was printed, not of what was calibrated.

# The design chart
The board, the canvas and a canvas texture are axis-aligned rectangles of ONE
chart, the design chart: the plane coordinates of a planar target's nominal
points, x to the right and y up as the printed side is seen from the front,
in metres for a physical target (lengths at the constructors, plain metres
stored) and in the design's plain numbers for a unitless one. Each rectangle
is its lower-left corner (`origin`) and its size; the board's placement on
the canvas is `board.origin − canvas.origin`.

# Fields
- `size::Size2{T}`: the page, width × height, by the board's length rule
  (lengths stored in metres, a quantity of any other dimension a
  `DimensionError`; or plain numbers; non-negative).
- `origin::SVector{2, Float64}`: its lower-left corner in the design chart.
- `texture::Union{Nothing, CanvasTexture{T}}`: the printed image
  ([`CanvasTexture`](@ref)), or `nothing`; in the canvas's form (FlexUnits'
  `DimensionError` otherwise).

# Constructors
```julia
Canvas(Size2(width = 297mm, height = 210mm), [-20.0mm, -15.0mm])
Canvas(Size2(width = 297mm, height = 210mm), [-20.0mm, -15.0mm];
       texture = CanvasTexture("canvas.png", sha, Size2(width = 297mm, height = 210mm), [-20.0mm, -15.0mm]))
Canvas(Size2(width = 297.0, height = 210.0))                       # a unitless canvas
```

`Rect(canvas)` is its rectangle in its own numbers. `ustrip(u, canvas)` and
[`with_unit`](@ref)`(canvas, u)` move it (and its texture) between the two
forms, as they move a board. A [`PlanarTarget`](@ref) carries it beside its
board (`canvas = …`; [`canvas`](@ref)`(t)` reads it).
"""
struct Canvas{T <: Number}
    size::Size2{T}
    origin::SVector{2, Float64}
    texture::Union{Nothing, CanvasTexture{T}}
    function Canvas{T}(size::Size2{T}, origin::SVector{2, Float64}, texture::Union{Nothing, CanvasTexture}) where {T <: Number}
        T <: Quantity && foreach(_check_length, size)
        all(x -> _board_number(x) >= 0, size) || throw(ArgumentError("a canvas's size is non-negative, got $size"))
        texture === nothing || has_units(texture) == (T <: Quantity) ||
            throw(DimensionError(size.width, texture.size.width))
        return new{T}(size, origin, texture)
    end
end
Canvas(size::Size2, origin::AbstractVector = SVector(0.0, 0.0); texture::Union{Nothing, CanvasTexture} = nothing) =
    _new_rect(Canvas, _stored_rect("Canvas", size, origin)..., texture)

for R in (:CanvasTexture, :Canvas)
    @eval Rect(r::$R) = Rect2{Float64}(r.origin[1], r.origin[2], _board_number(r.size.width), _board_number(r.size.height))
    @eval has_units(::$R{<:Quantity}) = true
    @eval has_units(::$R) = false
    @eval _form_sample(r::$R) = r.size.width
end
Base.:(==)(a::CanvasTexture, b::CanvasTexture) = a.path == b.path && a.sha256 == b.sha256 && a.size == b.size && a.origin == b.origin
Base.hash(x::CanvasTexture, h::UInt) = hash(x.origin, hash(x.size, hash(x.sha256, hash(x.path, hash(:CanvasTexture, h)))))
Base.:(==)(a::Canvas, b::Canvas) = a.size == b.size && a.origin == b.origin && a.texture == b.texture
Base.hash(c::Canvas, h::UInt) = hash(c.texture, hash(c.origin, hash(c.size, hash(:Canvas, h))))

# Units: the board's rule, the texture moving with its canvas.
_strip_rect(u, r) = (Size2(width = ustrip(u, r.size.width), height = ustrip(u, r.size.height)), map(x -> _from_metres(x, u), r.origin))
_attach_rect(u, r) = (Size2(width = r.size.width * u, height = r.size.height * u), map(x -> _to_metres(x, u), r.origin))
FlexUnits.ustrip(u::Units, x::CanvasTexture{<:Quantity}) = CanvasTexture(x.path, x.sha256, _strip_rect(u, x)...)
FlexUnits.ustrip(u::Units, c::Canvas{<:Quantity}) =
    Canvas(_strip_rect(u, c)...; texture = c.texture === nothing ? nothing : ustrip(u, c.texture))
with_unit(x::CanvasTexture{<:Real}, u::Units) = CanvasTexture(x.path, x.sha256, _attach_rect(u, x)...)
with_unit(c::Canvas{<:Real}, u::Units) = Canvas(_attach_rect(u, c)...; texture = c.texture === nothing ? nothing : with_unit(c.texture, u))
for R in (:CanvasTexture, :Canvas)
    @eval FlexUnits.ustrip(::Units, ::$R{<:Real}) = throw(ArgumentError(string("ustrip: the ", $(string(R)), " is plain numbers already; `with_unit` attaches a unit")))
    @eval with_unit(::$R{<:Quantity}, ::Units) = throw(ArgumentError(string("with_unit: the ", $(string(R)), " already has units")))
end
