# =============================================================================
# The JSON file boundary — one reader, one writer
# =============================================================================
#
# Every vgio schema file crosses this boundary through these functions, so the
# on-disk shape of a vgio file is decided in one place rather than re-agreed per
# schema, and a reader cannot pick up a variant spelling by being written later
# than its siblings. JSON.jl (1.x) does the encoding and the parsing.
#
# The shape: pretty-printed UTF-8 with a four-space indent, and no trailing
# newline, so every vgio file ends `}`. These files are configuration-adjacent
# and get diffed on recalibration, where a missing final newline shows up as
# `\ No newline at end of file`. Adding one is a one-line change here and
# nowhere else, but it changes the bytes of every vgio file, so it belongs in a
# release that says so.
#
# Numbers are written as JSON.jl writes them: a Float64 keeps its `.0`
# (`1.0`) and an integer is bare. The JSON3-based writer of earlier releases
# printed an integral float as an integer (`1`) and an empty array over two
# lines. Every reader accepts both, and a content identity (canonical.jl) is
# the same for both.
#
# `_parse_json` deliberately does not narrow field types. A parsed object is a
# `JSON.Object{String, Any}` (an ordered `AbstractDict` with dot access, which
# also accepts Symbol keys) and an array is a `Vector{Any}`. The narrowing
# belongs at each use — `d.frame::String`, `d.poses::Vector{Any}` — where the
# expected type is a documented property of that schema field and the assert
# doubles as validation. A blanket narrowing here would only move the `Any`
# one call deeper.

_json(x) = JSON.json(x)

_parse_json(s::AbstractString) = JSON.parse(s)

_write_json(path::AbstractString, body) = open(path, "w") do io
    JSON.json(io, body; pretty = 4)
end

_read_json(path::AbstractString) = _parse_json(read(path, String))
