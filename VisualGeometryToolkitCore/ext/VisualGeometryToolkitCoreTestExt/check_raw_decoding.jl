# =============================================================================
# check_raw_decoding — the stack's one RAW path, checked mechanically
# =============================================================================
#
# Every RAW → image step lives in VisualGeometryToolkitFeatures' raw module
# (src/raw_image.jl, ext/VisualGeometryToolkitFeaturesLibRawExt.jl). This scan
# flags a decoding library or a colour-filter interpolation used anywhere else,
# line by line, in code only: Julia files without their comments and
# docstrings (Julia's own tokenizer separates them), Python, shell, TOML and
# YAML files without their `#` comments. Prose (Markdown, comments) names the
# retired paths freely. Like `check_naming`, the patterns are BUILT from
# fragments, so this file spells none of the words it flags.
# =============================================================================

const _RAWPY = "raw" * "py"
const _DEMOSAIC = "demo" * "saic"

# The uses, each `use => pattern`, matched on a line's code text.
const _RAW_DECODING_USES = (
    :rawpy => Regex("(?i)\\b" * _RAWPY * "\\b"),
    # a LibRaw C call, `libraw_x(…)` or ccall's `:libraw_x`: not a name that merely starts so (the
    # mosaic's `libraw_version` metadata field), nor `LibRaw_jll`, the decoder trigger
    :libraw => Regex("\\b" * "lib" * "raw_" * "\\w+\\s*\\(|:" * "lib" * "raw_" * "\\w+"),
    :dcraw => Regex("(?i)" * "dc" * "raw"),
    :postprocess => Regex("\\b" * "post" * "process" * "\\("),
    :demosaic => Regex("(?i)" * _DEMOSAIC),
    :cfa_interpolation => Regex("(?i)(bayer|cfa)\\w*.*interp|interp\\w*.*(bayer|cfa)"),
)
# A decoded RAW output re-read as an image: the call, on a line naming a decoded output.
const _PNG_READ = Regex("PNG" * "Files\\.load\\(|\\bGray\\.\\(")
const _DECODED_OUTPUT = r"(?i)linear|decoded|prepared|bin2|green|raw_|_raw\b"

const _RAW_CODE_EXTENSIONS = (".jl", ".py", ".sh", ".sbatch", ".toml", ".yml", ".yaml")

# The approved display path in every repository: the Features API asked for its display
# mode, `load_raw(…; mode = :<display>)` or `LinearRaw(…; mode = :<display>)`. Inside such a
# call's own parentheses (over as many lines as it spans) that mode keyword is the API's,
# not a decoding path: it is blanked before the lines are matched, its newlines kept.
# Everything else, on the call's lines or outside its parentheses, is still matched.
const _FEATURES_RAW_CALL = r"\b(load_raw|LinearRaw)\s*\("
const _DISPLAY_MODE = Regex("\\bmode\\s*=\\s*:" * _DEMOSAIC * "\\b")

# The byte index of the parenthesis closing the one at `open` (the text's end when unclosed).
function _closing_paren(bytes, open)
    depth = 0
    for i in open:length(bytes)
        bytes[i] == UInt8('(') && (depth += 1)
        bytes[i] == UInt8(')') && (depth -= 1; depth == 0 && return i)
    end
    return length(bytes)
end

function _without_display_calls(lines)
    text = join(lines, '\n')
    bytes = codeunits(text)
    out = IOBuffer()
    pos = 1                                              # the next byte not yet written
    for m in eachmatch(_FEATURES_RAW_CALL, text)
        open = m.offset + ncodeunits(m.match) - 1        # the call's `(`: ASCII, a valid index
        open < pos && continue                           # inside a call already blanked
        close = _closing_paren(bytes, open)
        write(out, view(bytes, pos:(open - 1)))
        print(out, replace(String(bytes[open:close]), _DISPLAY_MODE => d -> replace(d, r"[^\n]" => "")))
        pos = close + 1
    end
    write(out, view(bytes, pos:length(bytes)))
    return split(String(take!(out)), '\n')
end

# A repository's genuine exception: a relative path, or a pattern over relative paths.
_allowed(f, a::AbstractString) = f == a
_allowed(f, a::Regex) = occursin(a, f)

# The one allowed place: the raw module of the package that owns it, and its tests.
const _RAW_MODULE_PACKAGE = "VisualGeometryToolkitFeatures"
const _RAW_MODULE_FILES = ("src/raw_image.jl", "ext/VisualGeometryToolkitFeaturesLibRawExt.jl", "test/raw_image_tests.jl")

function check_raw_decoding(dir::AbstractString; allow = ())
    allow isa Union{AbstractString, Regex} && (allow = (allow,))   # one exception, given bare
    (; files) = naming_scope(dir)
    # (`_package_name` is check_naming's reader of the Project.toml name)
    allowed = _package_name(dir) == _RAW_MODULE_PACKAGE ? Set(_RAW_MODULE_FILES) : Set{String}()
    out = RawDecodingViolation[]
    for f in files
        any(e -> endswith(f, e), _RAW_CODE_EXTENSIONS) || continue
        (f in allowed || any(a -> _allowed(f, a), allow)) && continue
        src = read(joinpath(dir, f), String)
        lines = endswith(f, ".jl") ? _julia_code_lines(src) : _hash_code_lines(src)
        for (n, (code, line)) in enumerate(zip(lines, _without_display_calls(lines)))
            for (use, re) in _RAW_DECODING_USES
                occursin(re, line) && push!(out, RawDecodingViolation(f, n, strip(code), use))
            end
            occursin(_PNG_READ, line) && occursin(_DECODED_OUTPUT, line) &&
                push!(out, RawDecodingViolation(f, n, strip(code), :png_raw_output))
        end
    end
    return out
end
