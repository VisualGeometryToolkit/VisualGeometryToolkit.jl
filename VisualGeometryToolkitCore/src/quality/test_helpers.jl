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
#                                 rows_cost, test_naming, test_docs
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
    test_duplication(M::Module; kwargs...)

The stack's no-duplication rule ([`check_duplication`](@ref)) over package
`M`, as a test: every finding printed, and `@test` that there is none. The
keywords go to `check_duplication` (the stack repositories compared come from
`VGTK_STACK_PATHS` by default). Implemented by VisualGeometryToolkitCoreTestExt
(`using Test` loads it).
"""
function test_duplication end

"""
    test_splat(M::Module; mode = :enforce, extensions = true)

The stack's splat rule ([`check_splat`](@ref)) over package `M`, as a test in
a gate mode: `:enforce` (`@test` that there is no finding), `:report` (each
finding logged with `@warn` and the test recorded `@test_broken`; none found,
an ordinary test) or `:off`. `extensions` goes to `check_splat`. A package on
[`test_quality_gates`](@ref) gets the same check as its `splat` gate.
Implemented by VisualGeometryToolkitCoreTestExt (`using Test` loads it).
"""
function test_splat end

# A gate as a test in its mode (`:enforce`, `:report`, `:off`) over the findings `findings()` returns:
# test_splat and every gate of test_quality_gates. Implemented by VisualGeometryToolkitCoreTestExt.
function _test_gate end

"""
    test_recorded(make; nested = RECORDED_NESTED_STATES) -> Int

That a [`Recorded`](@ref) run of [`block_coordinate_descent!`](@ref) IS the bare
run, as tests: `make()` returns a fresh `(state, likelihood, method, problem, θ₀,
max_iter)` each call (fresh workspaces too), whose result has `estimate`,
`retcode` and `iterations`. The recorded run returns the bare run's estimate,
retcode and iteration count and leaves the same state; it records one iterate
per good iteration (a `:solve_failed` run one fewer), with as many costs and
scales; and the bare run truncated after every `k` has the recorded run's `k`-th
cost — iterate for iterate. A state is compared by what a run moves: its
numbers, arrays and best iterates (`Attributed(θ, (; cost))`), and the fields
of the nested state types whose names are in `nested`; anything else it points
into is skipped. Returns the number of recorded iterates. Implemented by
VisualGeometryToolkitCoreTestExt (`using Test` loads it).
"""
function test_recorded end

"""
    RECORDED_NESTED_STATES

The names of the nested state types [`test_recorded`](@ref) compares field by
field by default (the posterior weights, the division camera's IRLS state, the
GNC ladder); every other non-bits field of a state is a workspace it points into.
"""
const RECORDED_NESTED_STATES = (:PosteriorWeightState, :_PositionalDivisionState, :MuLadderState)

"""
    thrown_message(f; shown = false) -> String

The message of the `ArgumentError` that `f()` throws, `""` when it throws
nothing; any other exception is rethrown. With `shown = true`, any exception
is caught and its shown text (`sprint(showerror, e)`, its type included) is
returned. For a test that reads what a refusal
says: `@test occursin("frame", thrown_message(() -> load(T, path)))`.
Implemented by VisualGeometryToolkitCoreTestExt (`using Test` loads it).
"""
function thrown_message end

"""
    test_docs(M::Module; exceptions = Dict{String,String}(), backlog = Set{String}(),
              signature_exceptions = Set{String}(), dirs = ("src",)) -> NamedTuple

The docstring gates the Documenter build cannot run (its `checkdocs = :public`
sees only docstrings that exist), over package `M`, as tests:

- every public name of `M` and its submodules has a docstring, except the
  `exceptions` (`name => reason`: undocumented on purpose) and the `backlog`
  (names still to document; the list only shrinks — a name documented must
  leave it); a stale exception is a failure;
- no docstring under `M`'s `dirs` (relative to its package directory; a missing
  one is skipped) is detached from its definition — a bare string statement
  the parser did not attach, after a blank or comment line;
- every signature line of a docstring is accepted by a method of its name
  (methods a test item defined under `Main` are not the package's), and a
  docstring that shows a sibling's signature leaves the sibling documented on
  its own, except the `signature_exceptions` (stale ones fail).

A name in a submodule is `Sub.name`. The exceptions are logged with `@info`.
Returns the findings, `(; undocumented, detached, mismatches)`. Implemented by
VisualGeometryToolkitCoreTestExt (`using Test` loads it).
"""
function test_docs end

"""
    test_quality_gates(M::Module; jet_bound, treat_as_own = [], stale_ignore = (),
                       qualified_ignore = nothing, persistent_tasks = true,
                       tag_rules = :enforce, tag_tables = :enforce, config_tags = :enforce,
                       config_dirs = nothing, registries = (), stack_dependencies = :enforce,
                       stack_name = string(nameof(M)), stack_extra = nothing,
                       public_names = :enforce, third_party_public = :report,
                       contract_generics = :enforce, splat = :report)

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
    ([`contract_generic_violations`](@ref));
  - `splat` (default `:report`): no call splats a collection whose length is
    data — a `Vector`, a generator, an abstract type — which compiles the
    callee once per length ([`check_splat`](@ref) over `M` and its loaded
    extensions; exceptions are `DUPLICATION.toml`'s `[[splat]]` entries).

Implemented by VisualGeometryToolkitCoreQualityExt: `using Test, Aqua,
ExplicitImports, JET` loads it.
"""
function test_quality_gates end

# -----------------------------------------------------------------------------
# The README's examples, run as a reader runs them (the Test extension's readme.jl)
# -----------------------------------------------------------------------------

"""
    ReadmeExample(page, lang, row, illustrative, code)

One fenced code block of a documentation page ([`readme_examples`](@ref)):
the `page` (relative to the package root), its `lang` (`"julia"`, `"python"`,
`"bash"`), the install `row` its first line names (`# Install row: <row>`;
`""` when it names none), whether it is `illustrative` (its first line
`# Illustrative`, the row on the second: a block that cannot run as written)
and its `code`.
"""
struct ReadmeExample
    page::String
    lang::String
    row::String
    illustrative::Bool
    code::String
end

"""
    readme_examples(root, pages; langs = ("julia", "python", "bash")) -> Vector{ReadmeExample}

The fenced code blocks of `langs` in each of `pages` (paths relative to `root`),
in page order, as [`ReadmeExample`](@ref)s. Implemented by
VisualGeometryToolkitCoreTestExt (`using Test` loads it).
"""
function readme_examples end

"""
    page_examples(examples, page; lang = "julia") -> Vector{String}

The code of the `lang` examples of one page, in page order, from
[`readme_examples`](@ref)'s records: what a README test compares between the
README, the landing page and a guide ("one example, shown on every page").
Implemented by VisualGeometryToolkitCoreTestExt (`using Test` loads it).
"""
function page_examples end

"""
    install_lines(root, page) -> Vector{String}

The packages of every `pkg> add …` line of the page at `root/page` (each line's
text after `add`), in page order: what a README test compares with an install
row's [`install_packages`](@ref). Implemented by VisualGeometryToolkitCoreTestExt.
"""
function install_lines end

"""
    install_table(path, tasks) -> Dict{String, String}

The install table of the page at `path`: each row `| task | entry | … |` whose
task is one of `tasks`, as `task => entry`. Implemented by
VisualGeometryToolkitCoreTestExt (`using Test` loads it).
"""
function install_table end

"""
    install_packages(entry) -> Vector{String}

The packages of the `` `pkg> add …` `` line in an install table's `entry`;
empty when it holds none. Implemented by VisualGeometryToolkitCoreTestExt
(`using Test` loads it).
"""
function install_packages end

"""
    julia_blocks(path) -> Vector{String}

The code of every ```` ```julia ```` fenced block of the page at `path`, in page
order (a block's opening fence exactly ```` ```julia ````, so a `julia-repl`
block — an install line — is not one; the code without its fences). What a
README test compares between the README and the landing page, and runs.
Implemented by VisualGeometryToolkitCoreTestExt (`using Test` loads it).
"""
function julia_blocks end

"""
    loaded_packages(code) -> Set{String}

The packages Julia `code` loads (`using`/`import` lines, a trailing comment
ignored), the standard library left out: what an install line must add for the
code to run. Implemented by VisualGeometryToolkitCoreTestExt (`using Test` loads it).
"""
function loaded_packages end

"""
    readme_environment(root, packages; developed = Dict(), setup = "") -> String

A fresh environment holding exactly `packages`, as a reader's install line
builds it: the package at `root` and those named in `developed` (`name =>
path`, the stack's unregistered checkouts) by path, the rest from the
registries; `setup` runs after the adds (a runtime pin), then the environment
is precompiled. Built in a child Julia ([`run_readme_child`](@ref)); `@test`
that it builds. Returns its directory. Implemented by
VisualGeometryToolkitCoreTestExt (`using Test` loads it).
"""
function readme_environment end

"""
    run_readme_child(env, args...) -> Int

Run Julia with `args` (a script, or `"-e", code`) in the environment `env`, a child Julia with the
test run's flags (`Base.julia_cmd()`: `--depwarn=error`, coverage) and a load
path of the project and the standard library only, so no default environment
supplies a package the install line does not add. Returns its exit code.
Implemented by VisualGeometryToolkitCoreTestExt (`using Test` loads it).
"""
function run_readme_child end

"""
    test_readme_row(env, task, blocks; preamble = "", after = "")

Run each Julia example of `blocks` (code strings) in the environment `env`,
each in a module of its own, so each is shown complete on its own; a block
whose last line ends `# true` must return `true`. `preamble` is top-level code
run first (an `import` the assertions need); `after` is code run in the same
child after the examples, `mods` their modules in order (the row's
assertions). `@test` that the child succeeds. Implemented by
VisualGeometryToolkitCoreTestExt (`using Test` loads it).
"""
function test_readme_row end

"""
    test_readme_example(root; index = "docs/src/index.md", install_pages = (), developed = Dict(), checks = "")

The README's one example, as tests: README.md holds one Julia block and one
`pkg> add` line; the landing page `index` shows the same block, it and each of
`install_pages` the same line; the line adds exactly the packages the block
loads ([`loaded_packages`](@ref)); and the block runs in a fresh environment
built from that line ([`readme_environment`](@ref)), expression by expression,
every expression whose source ends `# true` returning `true`, then `checks` —
code run with `readme` the example's module. Implemented by
VisualGeometryToolkitCoreTestExt (`using Test` loads it).
"""
function test_readme_example end

"""
    sensor_fixture(; body, iso = 100, exposure = 1.0s, black = nothing, black_std_error = nothing, dark = nothing, darks = nothing,
                   gain = nothing, gain_std_error = nothing, read_noise = nothing, read_noise_std_error = nothing,
                   relative_noise = nothing, defects = nothing, source = SensorSource.specified)
        -> (; characterisation, capture, model)
    sensor_fixture(characterisations::SensorCharacterisation...) -> SensorCharacterisation

THE builder of sensor-model fixtures for tests (owner ruling D13): a test never
constructs a sensor component, a [`SensorCharacterisation`](@ref) or a
[`SensorModel`](@ref) itself (check_duplication's `:sensor_component` pattern;
the absent model `SensorModel()` excepted), it describes one here in plain
numbers. The slots, each absent when its keywords are `nothing`:

- black: `black` (a 4-tuple per colour index) → [`ChannelBlack`](@ref) (with
  `black_std_error`); `dark = (; offset, variance, exposure_band, frames)` → a
  [`Dark`](@ref) over that channel black (the [`FileBlack`](@ref) without `black`);
  `darks`, a vector of such named tuples → a [`Darks`](@ref), one Dark per band,
  each over that channel black, which is also the shared fallback (`dark` or
  `darks`, not both);
- noise: `read_noise` → [`ReadNoise`](@ref) (with `read_noise_std_error`); with
  a `gain` → [`PoissonGaussian`](@ref) (`gain_std_error` and
  `read_noise_std_error` its standard errors); `relative_noise` →
  [`RelativeNoise`](@ref);
- defects: `defects = (; x, y, onset, size)` → [`DefectList`](@ref).

`source` is one `SensorSource.T` for every slot or a named tuple by slot
(`(; black = SensorSource.measured)`, specified where it names none). Returns
the characterisation of `body` with that one entry at `iso`, a one-frame
`capture` (frame `1` at `iso` and `exposure`, f/8, 14 mm) and the frame's
`model`, resolved by [`sensor_model`](@ref). The second form merges
characterisations of one body (their ISOs) into one. The components' own
`ArgumentError`s reach the caller. Implemented by VisualGeometryToolkitCoreTestExt
(`using Test` loads it).
"""
function sensor_fixture end
