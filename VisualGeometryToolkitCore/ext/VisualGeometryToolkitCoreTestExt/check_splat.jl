# =============================================================================
# check_splat — no call splats a collection whose length is data
# =============================================================================
#
# Typed: every method of the package is inferred at its own signature
# (unoptimized, so each call is the one written in that method) and each
# `Core._apply_iterate(iterate, f, args...)` — the compiler's form of every
# splat — is checked argument by argument against the fixed-length types.
# The entry point and its result type are in src/quality/splat.jl; the
# exceptions are DUPLICATION.toml's `[[splat]]` entries (check_duplication.jl
# reads the same file).
# =============================================================================

# A splat of one of these has a length its TYPE fixes: one specialization of the callee.
const _FIXED_LENGTH = Union{Tuple, NamedTuple, StaticArray, Pair}

# Every method whose module is one of `mods` (the one global method table, Julia ≥ 1.12).
function _methods_in(mods)
    out = Method[]
    Base.visit(Core.methodtable) do m
        m isa Method && m.module in mods && push!(out, m)
    end
    return out
end

# The modules a run checks: M's tree, and (a package's) the trees of its loaded extensions.
function _splat_modules(M::Module, extensions::Bool)
    mods = Module[]
    VisualGeometryToolkitCore._module_tree(M, mods)
    extensions && pkgdir(M) !== nothing || return mods
    project = joinpath(pkgdir(M), "Project.toml")
    for name in keys(get(TOML.parsefile(project), "extensions", Dict{String, Any}()))
        ext = Base.get_extension(M, Symbol(name))
        ext === nothing || VisualGeometryToolkitCore._module_tree(ext, mods)
    end
    return mods
end

# A method's function as a finding names it: a closure (`#f##0`) and a keyword body (`#f#3`) by the enclosing `f`.
function _splat_function_name(m::Method)
    s = string(m.name)
    startswith(s, '#') || return s
    head = first(split(s[2:end], '#'))
    return isempty(head) || all(isdigit, head) ? s : String(head)
end

# The inferred (widened) type of an argument of a statement of `ci`.
function _splat_argtype(ci::Core.CodeInfo, x)
    T = if x isa Core.SSAValue
        ci.ssavaluetypes[x.id]
    elseif x isa Core.SlotNumber || x isa Core.Argument
        ci.slottypes[x isa Core.SlotNumber ? x.id : x.n]
    elseif x isa GlobalRef
        isdefined(x.mod, x.name) ? Core.Typeof(getglobal(x.mod, x.name)) : Any
    elseif x isa QuoteNode
        Core.Typeof(x.value)
    else
        Core.Typeof(x)
    end
    return Base.Compiler.widenconst(T)
end

# The callee of a splat, as a finding prints it.
function _splat_callee(ci::Core.CodeInfo, f)
    T = _splat_argtype(ci, f)
    isdefined(T, :instance) && T.instance isa Function && return string(nameof(T.instance))
    T <: Type && T isa DataType && !isempty(T.parameters) && return string(T.parameters[1])
    return string(T)
end

# The call's file and line — the innermost scope in the method's own file (a macro's expansion sits at the
# macro call), else the method's line — and the file of the innermost scope, where the splat is written
# (another package's file: that macro's expansion).
function _splat_location(ci::Core.CodeInfo, m::Method, pc::Int)
    scopes = try
        Base.IRShow.buildLineInfoNode(ci.debuginfo, m, pc)
    catch
        []
    end
    written = isempty(scopes) ? string(m.file) : string(last(scopes).file)
    i = findlast(s -> string(s.file) == string(m.file) && s.line > 0, scopes)
    i === nothing && return (string(m.file), Int(m.line), written)
    return (string(scopes[i].file), Int(scopes[i].line), written)
end

# The method a typed code is of (a signature can match more specific methods too).
function _code_method(ci::Core.CodeInfo)
    def = ci.debuginfo.def
    def isa Core.MethodInstance && return def.def
    return def
end

# A callee that is compiled once whatever the arity: a builtin (`Core._expr`, `Core.apply_type`, …) or the
# `@nospecialize` varargs `Expr` constructor. `Core.tuple` is a builtin too, but the tuple it returns has the
# data's length in its type, so a splat into it is a finding.
_arity_free(F) = F === Type{Expr} || (F <: Core.Builtin && F !== typeof(Core.tuple))

# The splats of one method: `(check, file, line, name, callee, type, written)`, the check `:splat` for a type that is
# not fixed-length, `:unknown` for `Any` (an untyped argument: inference at the method's own signature).
function _method_splats(m::Method)
    out = Tuple{Symbol, String, Int, String, String, String, String}[]
    codes = try
        Base.code_typed_by_type(m.sig; optimize = false)
    catch e
        e isa InterruptException && rethrow()
        return out
    end
    for (ci, _) in codes
        ci isa Core.CodeInfo || continue
        d = _code_method(ci)
        d isa Method && d !== m && continue
        for (pc, stmt) in enumerate(ci.code)
            stmt isa Expr && stmt.head === :(=) && (stmt = stmt.args[2])
            stmt isa Expr && stmt.head === :call && length(stmt.args) >= 4 || continue
            callee = stmt.args[1]
            callee isa GlobalRef && callee.mod === Core && callee.name === :_apply_iterate || continue
            ci.ssavaluetypes isa Vector && ci.ssavaluetypes[pc] === Union{} && continue   # unreachable
            _arity_free(_splat_argtype(ci, stmt.args[3])) && continue
            for a in stmt.args[4:end]
                T = _splat_argtype(ci, a)
                T === Union{} || T <: _FIXED_LENGTH || begin
                    file, line, written = _splat_location(ci, m, pc)
                    push!(out, (T === Any ? :unknown : :splat, file, line, _splat_function_name(m),
                                _splat_callee(ci, stmt.args[3]), string(T), written))
                end
            end
        end
    end
    return out
end

function check_splat(M::Module; extensions::Bool = true, unknown::Bool = false, root = nothing)
    dir = pkgdir(M)
    root === nothing && dir === nothing &&
        throw(ArgumentError("check_splat: $M is not a package; give the repository root of its files as `root`"))
    root = root === nothing ? _repo_root(dir) : String(rstrip(abspath(root), '/'))
    entries = _duplication_exceptions(root)["splat"]
    used = falses(length(entries))
    scanned = Set{String}()
    found = Set{SplatViolation}()
    rel(f) = startswith(f, root * "/") ? relpath(f, root) : f
    for m in _methods_in(_splat_modules(M, extensions))
        push!(scanned, rel(string(m.file)))
        for (check, file, line, name, callee, type, written) in _method_splats(m)
            check === :unknown && !unknown && continue
            # written in another package's macro expansion (Test's @testset, Base's): that macro's, not M's
            written == string(m.file) || written == "none" || startswith(written, root * "/") || continue
            file = rel(file)
            i = findfirst(e -> e["file"] == file && e["function"] == name, entries)
            i === nothing || (used[i] = true; continue)
            push!(found, SplatViolation(check, file, line, name, callee, type))
        end
    end
    # stale: an entry about this package that matched nothing, when its file was scanned (an extension's only when
    # loaded) or is gone
    own = dir === nothing ? "." : relpath(dir, root)
    own = own == "." ? "" : own * "/"
    for (e, u) in zip(entries, used)
        u || !startswith(e["file"], own) || !(e["file"] in scanned || !isfile(joinpath(root, e["file"]))) ||
            push!(found, SplatViolation(:stale_exception, "DUPLICATION.toml", 0, e["function"], "", _entry_text("splat", e)))
    end
    return sort!(collect(found); by = v -> (v.check, v.file, v.line, v.name, v.type))
end
