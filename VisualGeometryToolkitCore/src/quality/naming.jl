# =============================================================================
# The stack's naming checker — its result type and entry points
# =============================================================================
#
# `check_naming` and `naming_scope` are implemented in the package extension
# VisualGeometryToolkitCoreTestExt (ext/VisualGeometryToolkitCoreTestExt/check_naming.jl),
# which loads with the Test standard library: every caller is a quality test,
# so the linter is never compiled into a runtime load. `NamingViolation` is its
# result type and lives here so the public name exists without the extension.
# =============================================================================

"""
    NamingViolation

One finding of [`check_naming`](@ref): the file (relative to the package
directory), the line, the offending name or line of text, and the rule it
breaks (`:R3` … `:R11`, or `:parse` for a file that does not parse).
"""
struct NamingViolation
    file::String
    line::Int
    text::String
    rule::Symbol
end

const _NAMING_RULE_TEXT = Dict{Symbol, String}(
    :R3 => "a name beginning `update_` or `_update_` ends in `!`",
    :R4 => "a name beginning `next_` or `_next_` is pure and does not end in `!`",
    :R5 => "the word `step` is reserved for `<fit>_step` of a `fit_<fit>` defined in the same package",
    :R6 => "a local model's Δ is `<method>_increment`",
    :R7 => "a banned word or phrase: a random hypothesis draw is a trial, never an iteration, and a deterministic cycle is never a trial (the lists live in `check_naming`)",
    :R8 => "a variant of a `block_coordinate_descent!` hook name",
    :R9 => "a retired name (the list lives in `check_naming`)",
    :R10 => "a width is bound by its full name — `scale_multiple`, `loss_width` or `outlier_halfwidth` — never an abbreviation or a letter (the lists live in `check_naming`)",
    :R11 => "unit parsing, unit tables and unit conversion live in VisualGeometryToolkitCore's units module (its registry StackUnits) and FlexUnits: `parse_unit`/`unit_string`, `quantity`, `base_quantity`, `ustrip`, `strip_or_bare` — never `uparse`/`qparse`/`register_unit` or a unit or registry built by hand (`Units`, `AffineTransform`, `PermanentDict`, `register_unit!`, `registry_defaults!`), a table of unit names, `convert(Quantity{…}, …)`, an `isa Quantity ? ustrip` branch, a bare `ubase` in src or a hand-written factor (25.4, 0.0254, 304.8, 0.3048, 127//5000) (the allow-list lives in `check_naming`)",
    :parse => "the file does not parse",
)

function Base.show(io::IO, v::NamingViolation)
    return print(
        io, v.file, ":", v.line, "  [", v.rule, "] ", v.text, "  — ",
        get(_NAMING_RULE_TEXT, v.rule, "")
    )
end

"""
    check_naming(pkgdir) -> Vector{NamingViolation}

The stack's naming rules, checked over a package directory. Implemented by the
VisualGeometryToolkitCoreTestExt extension: `using Test` loads it.
"""
function check_naming end

"""
    naming_scope(dir) -> Vector{String}

The files [`check_naming`](@ref) reads under `dir`. Implemented by the
VisualGeometryToolkitCoreTestExt extension: `using Test` loads it.
"""
function naming_scope end
