# =============================================================================
# The JSON file boundary — one reader, one writer
# =============================================================================
#
# Every vgio schema file crossed this boundary through its own inlined copy of
# the same three lines (six writers, ten readers). Owned once here, so the
# on-disk shape of a vgio file is decided in ONE place rather than re-agreed
# per schema, and a reader cannot pick up a variant spelling by being written
# later than its siblings.
#
# THE SHAPE: pretty-printed UTF-8, and NO trailing newline — `JSON3.pretty`
# does not emit one, so every vgio file ends `}`. That is a deliberate
# statement of fact, not an endorsement: these files are configuration-adjacent
# and get diffed on recalibration, where a missing final newline shows up as
# `\ No newline at end of file`. Adding one is a one-line change HERE and
# nowhere else — which is the point of this file — but it changes the bytes of
# every vgio file, so it belongs in a release that says so.
#
# `_read_json` deliberately does NOT narrow field types. JSON3's `Object`
# returns a wide `Union` from every property access, and the narrowing belongs
# at each USE — `d.frame::String`, `d.poses::JSON3.Array` — where the expected
# type is a documented property of that schema field and the assert doubles as
# validation. A blanket narrowing here would only move the Union one call
# deeper.

_write_json(path::AbstractString, body) = open(path, "w") do io
    JSON3.pretty(io, JSON3.write(body))
end

_read_json(path::AbstractString; allow_inf::Bool = false) =
    JSON3.read(read(path, String); allow_inf)
