# =============================================================================
# The stack's one RAW path, checked mechanically — its result type and entry point
# =============================================================================
#
# Every RAW → image step lives in one place: the decode and the one display-only
# demosaic in VisualGeometryToolkitFeatures' raw module (src/raw_image.jl and its
# LibRaw_jll extension), the mosaic, its colour phase, the linear scaling and the
# binning in VisualGeometryToolkitCore's sensor image (src/raw/). `check_raw_decoding` flags a
# decoding library or a colour-filter interpolation used anywhere else. It is
# implemented in the Test extension (ext/VisualGeometryToolkitCoreTestExt/
# check_raw_decoding.jl), beside `check_naming`: every caller is a quality test.
# =============================================================================

"""
    RawDecodingViolation

One finding of [`check_raw_decoding`](@ref): the file (relative to the scanned
directory), the line, its code text, and the use found — `:rawpy`, `:libraw`
(a LibRaw C call), `:dcraw`, `:postprocess` (rawpy's), `:demosaic`,
`:cfa_interpolation` (a Bayer/CFA interpolation by another name) or
`:png_raw_output` (a decoded RAW output re-read as an image, `PNGFiles.load`
or `Gray.(…)`).
"""
struct RawDecodingViolation
    file::String
    line::Int
    text::String
    use::Symbol
end

function Base.show(io::IO, v::RawDecodingViolation)
    return print(
        io, v.file, ":", v.line, "  [", v.use, "] ", v.text, "  — RAW decoding lives in ",
        "VisualGeometryToolkitFeatures' raw module (`load_raw`, `load_raw_mosaic`) and Core's `RawMosaic`/`LinearRaw`: go through them"
    )
end

"""
    check_raw_decoding(dir; allow = ()) -> Vector{RawDecodingViolation}

The stack's one-RAW-path rule, checked over the code under `dir` (the files
[`naming_scope`](@ref) reads, less prose: Julia, Python, shell, TOML and YAML
files, with comments and docstrings left out). Flags any use, outside
VisualGeometryToolkitFeatures' raw module, of rawpy, a LibRaw C call
(`libraw_…`), dcraw, rawpy's `postprocess(`, a demosaic, a Bayer/CFA
interpolation, or `PNGFiles.load`/`Gray.(…)` on a decoded RAW output (a line
naming linear, decoded, prepared, bin2, green or `raw_` images). Line-level
matching: a finding is a line to port, read by a person. The raw module's own
files (VisualGeometryToolkitFeatures' `src/raw_image.jl`, its LibRaw_jll
extension and their tests; VisualGeometryToolkitCore's `src/raw/` sensor image
and its test) are the one allowed place; they are recognized
when `dir` is the owning package's directory, so scan each package of a monorepo
from its own directory (from the repository root they are flagged). The
Features API asked for its display demosaic, `load_raw(…; mode = :demosaic)`
or `LinearRaw(…; mode = :demosaic)`, is the approved display path in every
repository and is not flagged: the mode keyword inside that call's own
parentheses, over every line the call spans; a demosaic by any other route,
on the same line or not, is.

`allow` lists a repository's genuine exceptions, each a path relative to
`dir` or a `Regex` over such paths (one exception may be given bare). They should be rare: a file is allowed
only when it cannot go through the Features raw module (say, a third-party
script kept verbatim), never as a way to keep a second decoding path.

Implemented by the
VisualGeometryToolkitCoreTestExt extension: `using Test` loads it.
"""
function check_raw_decoding end
