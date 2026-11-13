# =============================================================================
# The stack's splat gate — its result type and entry points
# =============================================================================
#
# `check_splat` flags a call that splats a collection whose LENGTH is data
# (a Vector, a generator, a Dict, an abstract type): the callee is compiled once
# per length (Features 0.8.3's `zip(map(_identity_fields, ids)...)`, one `zip`
# specialization per capture size, 518 s of LLVM at 171 frames). Typed, over
# `code_typed` of every method of the package. Implemented in the Test extension
# (ext/VisualGeometryToolkitCoreTestExt/check_splat.jl), beside `check_duplication`,
# whose `DUPLICATION.toml` holds its exceptions.
# =============================================================================

"""
    SplatViolation

One finding of [`check_splat`](@ref): the check that found it (`:splat`, or
`:stale_exception` for a `[[splat]]` entry of `DUPLICATION.toml` that matches
nothing), the file (relative to the repository root) and line of the call, the
function whose method holds it (closures and keyword bodies by their enclosing
function's name; the name an exception entry gives), the callee, and the
splatted argument's inferred type.
"""
struct SplatViolation
    check::Symbol
    file::String
    line::Int
    name::String
    callee::String
    type::String
end

function Base.show(io::IO, v::SplatViolation)
    v.check === :stale_exception && return print(io, v.file, "  [stale_exception] ", v.type,
        "  — a [[splat]] entry of DUPLICATION.toml that matches nothing: remove it")
    return print(io, v.file, ":", v.line, "  [splat] ", v.name, ": ", v.callee, "(…::", v.type, "...)",
        "  — a splat of a data-length collection compiles the callee once per length: pass the collection (a Vector-taking method, reduce/mapreduce, vcat of collections)")
end

"""
    check_splat(M::Module; extensions = true, root = <the git checkout holding pkgdir(M)>) -> Vector{SplatViolation}

Every call in package `M` (its submodules, and with `extensions` its LOADED
package extensions) that splats an argument whose inferred type is not
`<:Union{Tuple, NamedTuple, StaticArray, Pair}`: a `Vector`, a generator, a
`Dict`, a `ValueIterator`, a `String`, an abstract type. Such a splat compiles
its callee once per LENGTH of the data (`zip(v...)` is a new `zip` for every
`length(v)`), and the length is the data's.

Typed, not syntactic: each method whose `module` is one of `M`'s is inferred
at its own signature (`code_typed(…; optimize = false)`: the calls as written,
nothing inlined) and every `Core._apply_iterate` — the compiler's form of
every `f(x...)`, `[x...]`, `(x...,)` — is checked argument by argument. So a
bare name (`f(rows...)`) is found, and a call returning a tuple
(`f(_pad(k, 2)...)`) is not a finding. A method whose argument is untyped is
inferred at `Any`, so splatting it is a finding: type the argument (a `Tuple`,
an `SVector`) or pass the collection whole.

The fixes: a method that takes the collection (`linkaxes!(axes)`,
`collect(T, rows)`, `convert(Vector{T}, xs)`), `reduce`/`mapreduce`
(`mapreduce(typeof, typejoin, xs)`, `reduce(vcat, parts)`), `vcat` of
collections, `SVector{N}(v)` for a length fixed by contract.

A module that is not a package (a script's) is checked with `root`, the
directory its files' paths and its `DUPLICATION.toml` are relative to.

Exceptions live in the repository root's `DUPLICATION.toml` (the file
[`check_duplication`](@ref) reads), each with its reason; an entry about a file
this run checked that matches nothing is itself a finding:

```toml
[[splat]]                 # every splat in one function of one file
file = "src/a.jl"         # relative to the repository root
function = "f"            # the name the finding prints
reason = "…"
```

Implemented by the VisualGeometryToolkitCoreTestExt extension: `using Test`
loads it.
"""
function check_splat end
