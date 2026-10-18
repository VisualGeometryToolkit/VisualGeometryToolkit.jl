# =============================================================================
# The stack's shared test helpers — entry points
# =============================================================================
#
# One copy of what every stack package's tests used to write for themselves:
# the chart-LM rows checks and the package quality gates. Implemented in the
# package extensions, so nothing here is compiled into a runtime load. (Finite
# differences are FiniteDiff.jl's, a test dependency of each package using them.)
#
#   VisualGeometryToolkitCoreTestExt     (loads with `Test`)            check_rows_jacobian,
#                                 rows_cost, test_naming
#   VisualGeometryToolkitCoreQualityExt  (loads with `Test`, `Aqua`,     test_quality_gates
#                                  `ExplicitImports`, `JET`)
# =============================================================================

"""
    rows_cost(lm, θ, m) -> Real

The chart-LM cost `Σ rᵢ²` of a linearization state `lm` at chart point `θ`, its
`m` rows filled by [`residual_rows!`](@ref). Implemented by
VisualGeometryToolkitCoreTestExt (`using Test` loads it).
"""
function rows_cost end

"""
    check_rows_jacobian(lm, θ, m; atol = 0, rtol = 0, reference = ForwardDiff.jacobian,
                        jacobian = Matrix{Float64}(undef, m, length(θ))) -> Bool

Does the analytic [`residual_jacobian_rows!`](@ref) of `lm` at `θ` (`m` rows)
match a reference Jacobian of [`residual_rows!`](@ref) to
`maximum|J − J_ref| ≤ atol + rtol·max(1, maximum|J_ref|)`? `reference(f, θ)`
returns the Jacobian of `f` at `θ`: ForwardDiff by default; where a Dual cannot
flow through the rows, a finite difference such as
`(f, θ) -> FiniteDiff.finite_difference_jacobian(f, θ, Val(:central))`.
`jacobian` is the buffer the analytic Jacobian is written into — a dense
matrix by default; a state whose method takes another representation (a
fixed-pattern `SparseMatrixCSC`) is checked in that one. On a miss the two errors are logged. Implemented by VisualGeometryToolkitCoreTestExt
(`using Test` loads it).
"""
function check_rows_jacobian end

"""
    test_naming(M::Module)

The stack's naming rules ([`check_naming`](@ref)) over `M`'s package directory,
as a test: every finding printed, and `@test` that there is none. Implemented by
VisualGeometryToolkitCoreTestExt (`using Test` loads it).
"""
function test_naming end

"""
    test_quality_gates(M::Module; jet_bound, treat_as_own = [], stale_ignore = (),
                       qualified_ignore = nothing, persistent_tasks = true,
                       tag_rules = :enforce, tag_tables = :enforce, config_tags = :enforce,
                       config_dirs = nothing, registries = (), stack_dependencies = :enforce,
                       stack_name = string(nameof(M)), stack_extra = nothing,
                       public_names = :enforce, third_party_public = :report,
                       contract_generics = :enforce)

The stack's static quality gates over package `M`, as one test set — RATCHETS:
each bound is the value the package meets today and only ever moves down.

- Aqua: undefined exports, unbound type parameters, stale deps, deps compat,
  project extras, persistent tasks (`persistent_tasks = false` skips it), no
  method ambiguities involving a method of `M` (pairs touching a test item's
  `Main` methods are dropped and logged), and no piracy in `M` or any of its
  submodules beyond `treat_as_own`.
- ExplicitImports: no implicit imports, explicit imports via their owners, no
  stale explicit imports beyond `stale_ignore`; with `qualified_ignore` (a
  tuple of names) also every qualified access public, those names excepted.
- JET: `report_package(M)` counts at most `jet_bound` reports, not counting
  KernelAbstractions LAUNCH reports (a `kwcall(::NamedTuple{(:ndrange,)},
  ::Kernel, …)` no-method report, which JET raises at every launch site because
  the backend is known only at run time). The filter is checked against a real
  launch and two decoys every run.

- Stack contracts, each gate in a mode — `:enforce` (`@test` that it finds
  nothing), `:report` (every finding logged with `@warn` and the gate recorded
  `@test_broken`; a gate that finds nothing is an ordinary test) or `:off`:
  - `tag_rules`: every [`TagRegistry`](@ref) constant of `M` and its
    submodules obeys the tag rule ([`tag_rule_violations`](@ref));
  - `tag_tables`: no tag table outside a `TagRegistry` in `M`'s `src/`
    ([`tag_table_violations`](@ref), a heuristic scan);
  - `config_tags`: every string at a registry's key path in the TOML files
    under `config_dirs` (default: `M`'s `configs/`, when it exists) is a
    current tag of `M`'s registries and of the extra `registries`
    ([`config_tag_violations`](@ref));
  - `stack_dependencies`: `M`'s Project.toml names only the stack packages in
    `stack_name`'s row of the allow-list ([`stack_dependency_violations`](@ref));
    a project kept outside the published stack gives its own row as
    `stack_extra` (the `extra` of [`stack_dependencies`](@ref));
  - `public_names`: no `_`-prefixed name reached from another stack package
    ([`private_name_violations`](@ref) over `src/` and `ext/`);
  - `third_party_public` (default `:report`): ExplicitImports'
    `check_all_explicit_imports_are_public` and
    `check_all_qualified_accesses_are_public` (with `qualified_ignore`) — a
    non-public name of ANY module, the stack's or a third party's (ForwardDiff's
    `Dual`, StaticArraysCore's `StaticArray`);
  - `contract_generics`: every method `M` adds to another stack package's
    function lands on that owner's `CONTRACT_GENERICS`
    ([`contract_generic_violations`](@ref)).

Implemented by VisualGeometryToolkitCoreQualityExt: `using Test, Aqua,
ExplicitImports, JET` loads it.
"""
function test_quality_gates end
