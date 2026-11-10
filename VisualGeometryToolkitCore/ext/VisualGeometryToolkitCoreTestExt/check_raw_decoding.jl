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
# mode, `load_raw(…; mode = :<display>)` or `LinearRaw(…; mode = :<display>)`. On a line
# that calls either, that mode keyword is the API's, not a decoding path: it is taken out
# before the line is matched (anything else on the line is still matched).
const _FEATURES_RAW_CALL = r"\b(load_raw|LinearRaw)\s*\("
const _DISPLAY_MODE = Regex("\\bmode\\s*=\\s*:" * _DEMOSAIC * "\\b")
_without_display_call(line) = occursin(_FEATURES_RAW_CALL, line) ? replace(line, _DISPLAY_MODE => "") : line

# A repository's genuine exception: a relative path, or a pattern over relative paths.
_allowed(f, a::AbstractString) = f == a
_allowed(f, a::Regex) = occursin(a, f)

# The one allowed place: the raw module of the package that owns it, and its tests.
const _RAW_MODULE_PACKAGE = "VisualGeometryToolkitFeatures"
const _RAW_MODULE_FILES = ("src/raw_image.jl", "ext/VisualGeometryToolkitFeaturesLibRawExt.jl", "test/raw_image_tests.jl")

# The code text of a Julia source, line by line: comments and docstring (triple-quoted
# string) contents blanked, the line structure kept.
const _JS = Base.JuliaSyntax   # Julia's own tokenizer: comments and string pieces as tokens
function _julia_code_lines(src::AbstractString)
    out = IOBuffer()
    in_triple = false
    bytes = codeunits(src)
    for t in _JS.tokenize(src)
        k = _JS.kind(t)
        text = String(bytes[t.range])                   # a token's byte range (its last character may be multi-byte)
        if k == _JS.K"\"\"\"" || k == _JS.K"```"
            in_triple = !in_triple
            print(out, text)
        elseif k == _JS.K"Comment" || (in_triple && k == _JS.K"String")
            print(out, replace(text, r"[^\n]" => ""))   # its newlines kept: the line numbers stand
        else
            print(out, text)
        end
    end
    return split(String(take!(out)), '\n')
end

# The code text of a `#`-commented file (Python, shell, TOML), line by line: a
# comment starts at a `#` outside a quoted run (a line-level approximation).
function _hash_code_lines(src::AbstractString)
    return map(split(src, '\n')) do line
        quote_char = nothing
        for (i, c) in pairs(line)
            if quote_char === nothing
                c == '#' && return line[1:prevind(line, i)]
                (c == '"' || c == '\'') && (quote_char = c)
            elseif c == quote_char
                quote_char = nothing
            end
        end
        line
    end
end

function check_raw_decoding(dir::AbstractString; allow = ())
    (; files) = naming_scope(dir)
    # (`_package_name` is check_naming's reader of the Project.toml name)
    allowed = _package_name(dir) == _RAW_MODULE_PACKAGE ? Set(_RAW_MODULE_FILES) : Set{String}()
    out = RawDecodingViolation[]
    for f in files
        any(e -> endswith(f, e), _RAW_CODE_EXTENSIONS) || continue
        (f in allowed || any(a -> _allowed(f, a), allow)) && continue
        src = read(joinpath(dir, f), String)
        lines = endswith(f, ".jl") ? _julia_code_lines(src) : _hash_code_lines(src)
        for (n, code) in enumerate(lines)
            line = _without_display_call(code)
            for (use, re) in _RAW_DECODING_USES
                occursin(re, line) && push!(out, RawDecodingViolation(f, n, strip(code), use))
            end
            occursin(_PNG_READ, line) && occursin(_DECODED_OUTPUT, line) &&
                push!(out, RawDecodingViolation(f, n, strip(code), :png_raw_output))
        end
    end
    return out
end
