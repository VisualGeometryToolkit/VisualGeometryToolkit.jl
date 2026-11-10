# =============================================================================
# The stack's one RAW path, checked mechanically — its result type and entry point
# =============================================================================
#
# Every RAW → image step (the decode, the linear scaling, the binning, the one
# display-only demosaic) lives in VisualGeometryToolkitFeatures' raw module
# (src/raw_image.jl and its LibRaw_jll extension). `check_raw_decoding` flags a
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
        "VisualGeometryToolkitFeatures' raw module (`load_raw`, `load_raw_mosaic`, `LinearRaw`): go through it"
    )
end

"""
    check_raw_decoding(dir) -> Vector{RawDecodingViolation}

The stack's one-RAW-path rule, checked over the code under `dir` (the files
[`naming_scope`](@ref) reads, less prose: Julia, Python, shell and TOML files,
with comments and docstrings left out). Flags any use, outside
VisualGeometryToolkitFeatures' raw module, of rawpy, a LibRaw C call
(`libraw_…`), dcraw, rawpy's `postprocess(`, a demosaic, a Bayer/CFA
interpolation, or `PNGFiles.load`/`Gray.(…)` on a decoded RAW output (a line
naming linear, decoded, prepared, bin2, green or `raw_` images). Line-level
matching: a finding is a line to port, read by a person. The raw module's own
files (VisualGeometryToolkitFeatures' `src/raw_image.jl`, its LibRaw_jll
extension and their tests) are the one allowed place. Implemented by the
VisualGeometryToolkitCoreTestExt extension: `using Test` loads it.
"""
function check_raw_decoding end
