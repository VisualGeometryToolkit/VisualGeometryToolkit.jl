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
breaks (`:R3` … `:R14`, `:R21`, or `:parse` for a file that does not parse).
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
    :R5 => "no identifier holds the word `step` (an algorithm's model-block update is its `update_model!`), except Manopt.jl's `step_solver!` and SciMLBase's `nsteps`",
    :R6 => "a local model's Δ is `<method>_increment`",
    :R7 => "a banned word or phrase: a random hypothesis draw is a trial, never an iteration, and a deterministic cycle is never a trial (the lists live in `check_naming`)",
    :R8 => "a variant of a `BlockCoordinateDescent` hook name",
    :R9 => "a retired name (the list lives in `check_naming`)",
    :R10 => "a width is bound by its full name — `efficiency_constant`, `loss_width` or `outlier_halfwidth` — never an abbreviation or a letter (the lists live in `check_naming`)",
    :R11 => "unit parsing, unit tables and unit conversion live in VisualGeometryToolkitCore's units module (its registry StackUnits) and FlexUnits: `parse_unit`/`unit_string`, `quantity`, `base_quantity`, `ustrip`, `strip_or_bare` — never `uparse`/`qparse`/`register_unit` or a unit or registry built by hand (`Units`, `AffineTransform`, `PermanentDict`, `register_unit!`, `registry_defaults!`), a table of unit names, `convert(Quantity{…}, …)`, an `isa Quantity ? ustrip` branch, a bare `ubase` in src or a hand-written factor (25.4, 0.0254, 304.8, 0.3048, 127//5000) (the allow-list lives in `check_naming`)",
    :R12 => "the Student-t + uniform scorer has no threshold: its width is `outlier_halfwidth`; a threshold is a threshold scorer's (the word is flagged within six tokens of the scorer's type name or TOML tag)",
    :R21 => "a FitResult is read by field (`r.u`, `r.retcode`, `r.stats.nsteps`, `r.stop_rule`, `r.rms`, `r.metadata`) and `successful_retcode(r)`, its cross-type quantities only by their functions (`weights(r)`, `dof(r)`, `noise_scale(r)`, `scale_source(r)`, `inlier_mask(r)`): no retired accessor call (`estimate`, `iterations`, `converged`, `retcode`, `metadata`, `stop_rule` on a result, the retired nested-metadata lister) and no `r.metadata.<contract field>` outside the contract method",
    :R22 => "outside Core, user-level code (experiments, scripts, examples, benchmarks, dev, a README or a docs page, a test that is no contract test) calls only `solve`/`init`/`solve!`/`reinit!`: no call of a `SOLVER_HOOKS` name outside a definition of one, and nowhere a reach of a Core `_` name or a `CORE_INTERNAL_DRIVERS` name (a contract test carries `# R22: contract test — <contract>`)",
    :parse => "the file does not parse",
)

# A quality gate's violation as every gate prints it (naming, duplication, raw decoding, splat): `file:line  [tag] `,
# the text's parts, then what to do.
_print_violation(io::IO, file, line, tag, advice, text...) = print(io, file, ":", line, "  [", tag, "] ", text..., "  — ", advice)

Base.show(io::IO, v::NamingViolation) = _print_violation(io, v.file, v.line, v.rule, get(_NAMING_RULE_TEXT, v.rule, ""), v.text)

"""
    check_naming(pkgdir) -> Vector{NamingViolation}
    check_naming(M::Module) -> Vector{NamingViolation}

The stack's naming rules (R3–R21, `conventions/naming.md`), checked over a
package directory: its source, tests, scripts and docs, parsed. The module form
checks the module's package directory and adds the rules read from the loaded
module: R20's run-time check, a module-level `const` whose value is one concrete
instantiation of a stack type under another name, or under the type's own name
when it is not the type itself (a computed one, `typeof(…)`, included; a partial
application, a `UnionAll`, is not one; a type of a package outside the stack is
not checked). Implemented by the VisualGeometryToolkitCoreTestExt extension:
`using Test` loads it.
"""
function check_naming end

"""
    naming_scope(dir) -> Vector{String}

The files [`check_naming`](@ref) reads under `dir`. Implemented by the
VisualGeometryToolkitCoreTestExt extension: `using Test` loads it.
"""
function naming_scope end
