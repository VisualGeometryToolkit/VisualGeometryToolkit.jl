# =============================================================================
# The lexers the checks share: ONE Julia token walk, ONE `#`-language lexer
# =============================================================================
#
# Julia is read through Julia's own tokenizer (`_julia_tokens`): comments,
# string pieces, interpolations and character literals as tokens, so no check
# lexes Julia by hand. Python, shell, TOML and YAML are read through
# `_hash_code`, which finds their `#` comments and quoted runs. Consumers:
# `_julia_code_lines` / `_hash_code_lines` (check_raw_decoding, check_duplication's
# retired patterns), `_code_tokens` (check_duplication's shingles), `_julia_prose`
# (check_naming's prose rules) and `_python_code` (check_naming's Python names).

const _JS = Base.JuliaSyntax

# A Julia source's tokens in order: kind and text (a token's byte range; its last
# character may be multi-byte), and the byte the token starts at.
function _julia_tokens(src::AbstractString)
    bytes = codeunits(src)
    return [(kind = _JS.kind(t), text = String(bytes[t.range]), start = first(t.range)) for t in _JS.tokenize(src)]
end

_is_triple_delim(k) = k == _JS.K"\"\"\"" || k == _JS.K"```"
_is_string_delim(k) = k == _JS.K"\"" || k == _JS.K"`" || _is_triple_delim(k)
_is_string_piece(k) = k == _JS.K"String" || k == _JS.K"CmdString"

# The code text of a Julia source, line by line: comments and docstring (triple-quoted
# string) contents blanked, the line structure kept.
function _julia_code_lines(src::AbstractString)
    out = IOBuffer()
    in_triple = false
    for t in _julia_tokens(src)
        if _is_triple_delim(t.kind)
            in_triple = !in_triple
            print(out, t.text)
        elseif t.kind == _JS.K"Comment" || (in_triple && t.kind == _JS.K"String")
            print(out, replace(t.text, r"[^\n]" => ""))   # its newlines kept: the line numbers stand
        else
            print(out, t.text)
        end
    end
    return split(String(take!(out)), '\n')
end

# A Julia source's code tokens (comments and whitespace dropped): their start bytes,
# raw texts and normalized texts (every identifier `I`, every string piece `S`).
function _code_tokens(src::AbstractString)
    starts = Int[]; raw = String[]; normalized = String[]
    for t in _julia_tokens(src)
        k = t.kind
        (k == _JS.K"Whitespace" || k == _JS.K"NewlineWs" || k == _JS.K"Comment") && continue
        push!(starts, t.start); push!(raw, t.text)
        push!(normalized, k == _JS.K"Identifier" ? "I" : _is_string_piece(k) ? "S" : t.text)
    end
    return (; starts, raw, normalized)
end

# The comment and string text of `src` by line, each line's text one chunk with a
# leading space; with `comments = false`, its plain string literals' text alone (not
# a docstring's or any triple-quoted string's, not a test's name: those are prose).
# A literal's end is a space, so separate literals stay separate words; an
# interpolation `$(…)` is code (its own literals are read), a `$name` is text.
function _julia_prose(src::AbstractString; comments::Bool = true)
    toks = _cmd_interpolations(_julia_tokens(src))
    prose = Dict{Int, String}()
    buf = IOBuffer()
    line = Ref(1)
    function flush_line!()
        s = String(take!(buf))
        isempty(s) || (prose[line[]] = " " * s)
        line[] += 1
    end
    # text kept (or not), a newline flushing its line either way
    function emit(text, keep)
        for (j, piece) in enumerate(split(text, '\n'))
            j > 1 && flush_line!()
            keep && print(buf, piece)
        end
    end
    # a string's frame: its delimiter kind and whether its text is kept; an
    # interpolation's: the paren depth back to its string
    strings = Tuple{_JS.Kind, Bool}[]
    interps = Int[]
    code_depth() = isempty(strings) || (!isempty(interps) && length(interps) == length(strings))
    prev = 0                                   # the last token that is not whitespace on this line
    for (i, t) in enumerate(toks)
        k = t.kind
        if !code_depth()                       # inside a string's text
            delim, keep = strings[end]
            if k == delim
                pop!(strings)
                keep && print(buf, ' ')
            elseif k == _JS.K"$" && i < length(toks) && toks[i + 1].kind == _JS.K"("
                push!(interps, 0)
            else
                emit(t.text, keep)             # a piece, an escape, or a `$name`'s `$` and name
            end
        elseif k == _JS.K"Comment"
            body = t.text[(nextind(t.text, 1, 2)):end]   # `#…` or `#=…=#`, the opening dropped
            if startswith(t.text, "#=")
                _block_comment_prose!(emit, flush_line!, buf, body, comments)
            else
                emit(t.text[nextind(t.text, 1):end], comments)
            end
        elseif _is_string_delim(k)
            as_prose = !comments && (_is_triple_delim(k) || _names_a_test(toks, prev))
            push!(strings, (k, !as_prose))
        elseif !isempty(interps) && length(interps) == length(strings) && k == _JS.K"("
            interps[end] += 1
        elseif !isempty(interps) && length(interps) == length(strings) && k == _JS.K")"
            interps[end] -= 1
            interps[end] == 0 && pop!(interps)
        else
            emit(t.text, false)                # code: only its newlines count
        end
        k == _JS.K"Whitespace" || (prev = i)
    end
    s = String(take!(buf))
    isempty(s) || (prose[line[]] = " " * s)
    return prose
end
# The tokenizer keeps a command literal's text whole, one `CmdString`; an
# interpolating command's (not a command macro's: `` r`…` ``) `$(…)` are split out
# here as a string's are, `$`, `(`, the code's tokens, `)`. A backslash escapes.
function _cmd_interpolations(toks)
    out = similar(toks, 0)
    open = false; interp = false
    for (i, t) in enumerate(toks)
        if t.kind == _JS.K"`" || t.kind == _JS.K"```"
            open = !open
            open && (interp = !(i > 1 && toks[i - 1].kind == _JS.K"Identifier"))
            push!(out, t)
        elseif t.kind == _JS.K"CmdString" && interp
            _split_cmd!(out, t.text)
        else
            push!(out, t)
        end
    end
    return out
end
function _split_cmd!(out, text)
    piece(s) = isempty(s) || push!(out, (kind = _JS.K"CmdString", text = String(s), start = 0))
    from = i = firstindex(text)
    while i <= lastindex(text)
        c = text[i]
        if c == '\\'
            i = nextind(text, nextind(text, i))
        elseif c == '$' && nextind(text, i) <= lastindex(text) && text[nextind(text, i)] == '('
            piece(text[from:prevind(text, i)])
            body = nextind(text, i, 2)                 # past `$(`
            inner = _julia_tokens(text[body:end])
            depth = 1; close = nothing
            for (j, t) in enumerate(inner)
                depth += t.kind == _JS.K"(" ? 1 : t.kind == _JS.K")" ? -1 : 0
                depth == 0 && (close = j; break)
            end
            push!(out, (kind = _JS.K"$", text = "\$", start = 0), (kind = _JS.K"(", text = "(", start = 0))
            close === nothing && (append!(out, inner); return out)
            append!(out, inner[1:close])
            from = i = body + inner[close].start           # past the `)`: its start is 1-based in the slice
        else
            i = nextind(text, i)
        end
    end
    piece(text[from:end])
    return out
end
# A block comment's text after its `#=`: nested `#=` dropped, each `=#` a space.
function _block_comment_prose!(emit, flush_line!, buf, body, comments)
    cs = collect(body)
    i = 1
    while i <= length(cs)
        c = cs[i]
        if c == '\n'
            flush_line!()
            i += 1
        elseif c == '#' && i < length(cs) && cs[i + 1] == '='
            i += 2
        elseif c == '=' && i < length(cs) && cs[i + 1] == '#'
            print(buf, ' ')
            i += 2
        else
            comments && print(buf, c)
            i += 1
        end
    end
end
# Whether the string opening after token `prev` is a test's name: the code before it
# (spaces skipped) is `@testset` or `@testitem`.
_names_a_test(toks, prev) =
    prev >= 2 && toks[prev].kind == _JS.K"Identifier" && toks[prev].text in ("testset", "testitem") && toks[prev - 1].kind == _JS.K"@"

# A `#`-commented source (Python, shell, TOML, YAML) with its comments blanked and,
# with `strings = false`, its quoted runs too (single, double or triple quotes, a
# backslash escaping): newlines and byte offsets kept, so what is left is code.
function _hash_code(src::AbstractString; strings::Bool = true)
    b = Vector{UInt8}(codeunits(src))
    n = length(b)
    blank!(r) = foreach(k -> b[k] == UInt8('\n') || (b[k] = UInt8(' ')), r)
    i = 1
    while i <= n
        c = b[i]
        if c == UInt8('#')
            j = i
            while j <= n && b[j] != UInt8('\n')
                j += 1
            end
            blank!(i:(j - 1))
            i = j
        elseif c == UInt8('"') || c == UInt8('\'')
            q = i + 2 <= n && b[i + 1] == c && b[i + 2] == c ? 3 : 1
            j = i + q
            while j <= n
                if b[j] == UInt8('\\')
                    j += 2
                elseif q == 1 && b[j] == UInt8('\n')
                    break
                elseif b[j] == c && (q == 1 || (j + 2 <= n && b[j + 1] == c && b[j + 2] == c))
                    j += q
                    break
                else
                    j += 1
                end
            end
            strings || blank!(i:min(j - 1, n))
            i = j
        else
            i += 1
        end
    end
    return String(b)
end
# The code text of a `#`-commented file, line by line (its strings kept: a TOML
# value or a shell argument is code there).
_hash_code_lines(src::AbstractString) = split(_hash_code(src), '\n')
# A Python source's code: comments and string literals blanked.
_python_code(src::AbstractString) = _hash_code(src; strings = false)
