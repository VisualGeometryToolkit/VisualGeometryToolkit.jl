# =============================================================================
# The stack's duplication gate — its result type and entry point
# =============================================================================
#
# One implementation per concept: `check_duplication` flags a name an upstream
# stack package already owns, a function body that restates another (within the
# repository or in another stack repository), a file copied whole, and a retired
# pattern whose owner is named. Implemented in the Test extension
# (ext/VisualGeometryToolkitCoreTestExt/check_duplication.jl), beside
# `check_naming` and `check_raw_decoding`: every caller is a quality test.
# =============================================================================

"""
    DuplicationViolation

One finding of [`check_duplication`](@ref): the check that found it (`:name`,
`:similar`, `:file`, `:pattern` or `:stale_exception`), the file (relative to
the repository root) and line, the text found (a name, a function, a code line
or an exception entry), and `other` — the upstream owner, the matching function
or file elsewhere (`<Repository>:<path>:<line>` outside this repository, a
monorepo's member as `<Member>:<path>:<line>`), or the owner a retired pattern names.
"""
struct DuplicationViolation
    check::Symbol
    file::String
    line::Int
    text::String
    other::String
end

const _DUPLICATION_RULE_TEXT = Dict{Symbol, String}(
    :name => "a name an upstream stack package owns: call the owner's",
    :similar => "a function body that restates another: keep one",
    :file => "a file that is a copy of another: keep one",
    :pattern => "a retired pattern: go through its owner",
    :stale_exception => "a DUPLICATION.toml entry that matches nothing: remove it",
)

function Base.show(io::IO, v::DuplicationViolation)
    return print(
        io, v.file, ":", v.line, "  [", v.check, "] ", v.text, "  ⇔  ", v.other, "  — ",
        get(_DUPLICATION_RULE_TEXT, v.check, "")
    )
end

"""
    check_duplication(M::Module; kwargs...) -> Vector{DuplicationViolation}
    check_duplication(dir; stack = <ENV["VGTK_STACK_PATHS"]>, similarity = ((0.6, 60), (0.85, 25)),
                      shingle = 8, min_file_tokens = 25) -> Vector{DuplicationViolation}

The stack's no-duplication rule, checked over a package (`M`, its directory
`pkgdir(M)`) or a directory `dir`. Four checks:

1. **Name ownership** (the `Module` form only). Every binding `M` (or a
   submodule of it) owns — defined there, not imported, a function or type
   whose `parentmodule` is that module; method extensions of another package's
   function do not count — whose name an upstream stack package also owns,
   private `_names` included (Julia's per-module `eval`, `include` and
   `__init__`, and the per-package `CONTRACT_GENERICS`, are each package's
   own). Upstream is every package in `M`'s dependency
   closure (its loaded `Project.toml` `[deps]`, transitively) that is
   VisualGeometryToolkitCore or depends on it: Core < IO, Features <
   Estimation < Eval < the downstream repositories.
2. **Function-body similarity.** Every function and macro definition in the
   Julia files is tokenized with Julia's own tokenizer (JuliaSyntax), its
   comments and whitespace dropped, each identifier mapped to one symbol and
   each string piece to another, and cut into `shingle`-token shingles. Two
   definitions in different files are a finding when their shingle sets'
   Jaccard index reaches a threshold of `similarity`, each a pair
   `(jaccard, min_tokens)` that both bodies' token counts must reach: by
   default `J ≥ 0.6` with at least 60 tokens, or `J ≥ 0.85` with at least 25.
   (A shingle shared by more than 40 definitions — a common idiom — is not
   counted towards any pair.) Compared within the repository and against
   `stack`.
3. **Whole files.** A Julia file whose token text, comments and whitespace
   dropped (at least `min_file_tokens` tokens), equals another file's.
4. **Retired patterns.** A line of code (comments and docstrings left out) that
   spells a pattern whose owner lives elsewhere: `NLLSOptions(` (Core's
   `nlls_options`), `acos(clamp((tr(…` (Rotations' `rotation_angle`), the sRGB
   transfer constants (Features' `src/linear_image.jl`), and a `\\newcommand`
   writer (Eval's `src/reproduce/number_macros.jl`). The failure names the owner.

The repository is the git checkout holding `dir`. Inside a monorepo the other
packages' files are compared too (the package's own files are the ones checked;
a pair or a file found twice is reported by each package). `stack` lists other
stack repositories' root directories (by default the paths in the environment
variable `VGTK_STACK_PATHS`, separated by `:`, or `;` on Windows);
those absent are skipped with a note, so a CI checkout checks itself only. Files
are the ones [`naming_scope`](@ref) reads (vendored trees out).

Exceptions live in the repository root's `DUPLICATION.toml`, each with its
reason; an entry about this package that matches nothing is itself a finding:

```toml
[[name]]                  # a binding M owns that an upstream package also owns
module = "MyPackage"
name = "foo"
reason = "…"

[[similar]]               # two functions, `<path>::<name>` (another repository's `<Repository>:<path>::<name>`)
functions = ["src/a.jl::f", "test/b.jl::g"]
reason = "…"

[[file]]                  # two files (paths as above)
files = ["test/a.jl", "OtherRepo:test/a.jl"]
reason = "…"

[[pattern]]               # a retired pattern in one file
use = "rotation_angle"
file = "src/a.jl"
reason = "…"
```

Implemented by the VisualGeometryToolkitCoreTestExt extension: `using Test`
loads it.
"""
function check_duplication end
