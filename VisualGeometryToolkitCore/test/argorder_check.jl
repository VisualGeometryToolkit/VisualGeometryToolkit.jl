#!/usr/bin/env julia
# =============================================================================
# argorder_check.jl — mechanical calling-convention check (package-cleanup
# Phase 4a). Flags, for every public (exported or `public`) function a package
# DEFINES:
#
#   TYPE-AFTER-INPUT  non-`!`: a positional type argument after an argument that
#                     is not a function, IO, RNG or type. Julia's style guide puts
#                     a type in slot (4), before every non-mutated input
#                     (`zeros(Float32, n)`, `rand(rng, Float32, dims)`).
#   TYPE-BEFORE-DST   `!`: a type argument before the mutated input(s) — the type
#                     follows them (`undistort!(dst, T, model, src)`).
#   TYPE-KEYWORD      a precision/element type taken as a keyword; Base takes it
#                     positionally (`zeros([T=Float64], dims)`).
#   MUTATES-NO-BANG   a non-`!` method whose body calls a `!` function (or
#                     `setindex!`/`setproperty!`) on a positional argument or a
#                     field of one — "Append `!` to names of functions that modify
#                     their arguments". A heuristic: direct calls in the method
#                     body only (plus the keyword-body function), so review each
#                     hit; mutation through helpers needs the reviewer agent.
#
# Methods added to functions another module owns are skipped (they keep the
# owner's order), as are constructors and `convert`.
#
#     julia --project=<repo> argorder_check.jl <Package> [allowlist]
#
# The allowlist (default `<repo>/.argorder_allow`) has one `name RULE  # why` per
# line; allowed findings are reported but do not fail. Exit 1 on anything else,
# so it gates Phase 6 like the JET ratchet.
# =============================================================================
using Random: AbstractRNG

# An untyped keyword with one of these names is taken to be a type; any keyword
# declared `::Type…` is one regardless of its name.
const TYPE_KEYWORDS = (:precision, :eltype, :element_type, :float_type, :dtype, :numeric_type, :problem_type, :pt)


within(mod::Module, root::Module) = mod === root || (parentmodule(mod) !== mod && within(parentmodule(mod), root))

_ub(p) = p isa TypeVar ? _ub(p.ub) : p isa UnionAll ? Base.unwrap_unionall(p) : p
is_type_arg(p) = (q = _ub(p); q !== Union{} && q isa Type && q <: Type)
function is_lead_ok(p)
    q = _ub(p)
    q === Any && return false
    return q isa Type && (q <: Function || q <: IO || q <: AbstractRNG)
end

function positional_params(m::Method)
    sig = Base.unwrap_unionall(m.sig)
    ps = collect(sig.parameters[2:end])
    return [Base.isvarargtype(p) ? Any : p for p in ps]
end

function public_functions(root::Module)
    out = Dict{Symbol, Tuple{Module, Function}}()
    seen = Set{Module}()
    function walk(mod::Module)
        mod in seen && return
        push!(seen, mod)
        for n in names(mod)
            isdefined(mod, n) || continue
            v = getfield(mod, n)
            if v isa Module
                v !== mod && within(v, root) && walk(v)
            elseif v isa Function && !(v isa Type) && within(parentmodule(v), root)
                out[n] = (mod, v)
            end
        end
        return
    end
    walk(root)
    return out
end

# Lowered code often loads the callee into an SSA value first: `%1 = Mod.f; %1(args)`.
_callee(code, f) = f isa Core.SSAValue && code[f.id] isa GlobalRef ? code[f.id] : f

# The positional-argument slots mutated in a lowered body: a `!` call whose first
# argument is an argument slot, or a getproperty of one.
function mutated_slots(ci, first_slot::Integer, last_slot::Integer)
    hits = Set{Int}()
    code = ci.code
    fieldof(x) = x isa Core.SSAValue && (
            s = code[x.id]; s isa Expr && s.head === :call &&
            length(s.args) >= 2 && s.args[1] isa GlobalRef && s.args[1].name === :getproperty
        ) ?
        code[x.id].args[2] : x
    for st in code
        st isa Expr || continue
        ex = st.head === :(=) ? st.args[2] : st
        ex isa Expr && ex.head === :call && length(ex.args) >= 2 || continue
        f = _callee(code, ex.args[1])
        fname = f isa GlobalRef ? string(f.name) : ""
        endswith(fname, "!") || continue
        a = fieldof(ex.args[2])
        a isa Core.SlotNumber && first_slot <= a.id <= last_slot && push!(hits, a.id - first_slot + 1)
    end
    return hits
end

function body_mutations(m::Method, fname::Symbol)
    npos = m.nargs - 1
    ci = try
        Base.uncompressed_ast(m)
    catch
        return Set{Int}()
    end
    hits = mutated_slots(ci, 2, m.nargs)
    # keyword methods lower to a body function `#fname#N(kws..., #self#, args...)`
    for st in ci.code
        ex = st isa Expr && st.head === :(=) ? st.args[2] : st
        ex isa Expr && ex.head === :call || continue
        g = _callee(ci.code, ex.args[1])
        g isa GlobalRef || continue
        startswith(string(g.name), "#$(fname)#") || continue
        isdefined(g.mod, g.name) || continue
        for bm in methods(getfield(g.mod, g.name))
            bci = try
                Base.uncompressed_ast(bm)
            catch
                continue
            end
            union!(hits, mutated_slots(bci, bm.nargs - npos + 1, bm.nargs))
        end
    end
    return hits
end

# Keywords that carry a type: declared `::Type…` in the keyword-body method
# (`#f#N(kws..., #self#, args...)`, whose leading parameters are the keywords in
# declaration order), or untyped with a precision-like name.
function type_keywords(m::Method, fname::Symbol)
    kws = try
        Base.kwarg_decl(m)
    catch
        Symbol[]
    end
    isempty(kws) && return Symbol[]
    kwtypes = fill(Any, length(kws))
    ci = try
        Base.uncompressed_ast(m)
    catch
        nothing
    end
    if ci !== nothing
        for st in ci.code
            ex = st isa Expr && st.head === :(=) ? st.args[2] : st
            ex isa Expr && ex.head === :call || continue
            g = _callee(ci.code, ex.args[1])
            g isa GlobalRef && startswith(string(g.name), "#$(fname)#") && isdefined(g.mod, g.name) || continue
            for bm in methods(getfield(g.mod, g.name))
                ps = positional_params(bm)
                length(ps) >= length(kws) && (kwtypes = ps[1:length(kws)]; break)
            end
        end
    end
    return [k for (k, T) in zip(kws, kwtypes) if is_type_arg(T) || (_ub(T) === Any && k in TYPE_KEYWORDS)]
end

function check(root::Module)
    findings = Tuple{Symbol, String, String}[]
    for (name, (_, f)) in sort(collect(public_functions(root)); by = first)
        name === :convert && continue
        bang = endswith(string(name), "!")
        for m in methods(f)
            within(m.module, root) || continue
            ps = positional_params(m)
            loc = "$(relpath(string(m.file), pwd())):$(m.line)"
            sigstr = string(name, "(", join((string(_ub(p)) for p in ps), ", "), ")")
            if bang
                k = findfirst(p -> !is_lead_ok(p), ps)
                k !== nothing && is_type_arg(ps[k]) &&
                    push!(findings, (name, "TYPE-BEFORE-DST", "$sigstr  $loc"))
            else
                for (i, p) in enumerate(ps)
                    is_type_arg(p) || continue
                    any(j -> !is_lead_ok(ps[j]) && !is_type_arg(ps[j]), 1:(i - 1)) &&
                        (push!(findings, (name, "TYPE-AFTER-INPUT", "$sigstr  $loc")); break)
                end
                hit = sort(collect(body_mutations(m, name)))
                filter!(i -> i <= length(ps) && !is_lead_ok(ps[i]), hit)
                isempty(hit) ||
                    push!(findings, (name, "MUTATES-NO-BANG", "$sigstr mutates arg $(join(hit, ","))  $loc"))
            end
            bad = type_keywords(m, name)
            isempty(bad) || push!(findings, (name, "TYPE-KEYWORD", "$sigstr ; $(join(bad, ","))  $loc"))
        end
    end
    return unique(findings)
end

function read_allowlist(path)
    allow = Set{Tuple{Symbol, String}}()
    isfile(path) || return allow
    for ln in eachline(path)
        s = strip(first(split(ln, '#')))
        isempty(s) && continue
        w = split(s)
        length(w) >= 2 && push!(allow, (Symbol(w[1]), String(w[2])))
    end
    return allow
end

function main(args)
    isempty(args) && (println(stderr, "usage: argorder_check.jl <Package> [allowlist]"); return 2)
    pkg = Symbol(args[1])
    root = Base.require(Main, pkg)
    allow = read_allowlist(length(args) >= 2 ? args[2] : joinpath(pkgdir(root), ".argorder_allow"))
    findings = check(root)
    nbad = 0
    for (name, rule, detail) in findings
        ok = (name, rule) in allow
        nbad += !ok
        println(rpad(ok ? "allowed" : "FLAG", 8), rpad(rule, 18), detail)
    end
    println("argorder_check: $(length(findings)) finding(s), $nbad not allowlisted")
    return nbad == 0 ? 0 : 1
end

if abspath(PROGRAM_FILE) == @__FILE__
    exit(main(ARGS))
end
