## Stack contracts: tags, dependencies and package boundaries

Six rules hold across the stack. Each is a function in VisualGeometryToolkitCore
(`src/quality/`) that a paper project's suite can call directly, and a gate of
`VisualGeometryToolkitCore.test_quality_gates`, which every stack package's quality
test runs.

**The tag rule.** A configuration file names a type by its *tag*, and every
tag is `snake_case` of the type's name, with the family's declared suffix
stripped from its end when present. `snake_case` has exactly two rules, then
lowercases: an `_` goes before an uppercase letter that follows a lowercase
letter or a digit, and before the last uppercase letter of an uppercase run
when a lowercase letter follows it. A digit stays with its letters:
`FixedS2` → `fixed_s2`, `WelschHQNewton` → `welsch_hq_newton`,
`StudentTUniform` → `student_t_uniform`, `SimFitPP` → `sim_fit_pp`. A tag that
selects `nothing` is spelled `none`.

**One registry.** `TagRegistry(family, suffix, "tag" => Type, …; retired, paths)`
is the only structure a config loader may use to map a tag to a type:

```julia
const MODEL_UPDATES = TagRegistry(:model_update, "",
    "keep_model" => KeepModel, "weighted_refine" => WeightedRefine, "none" => nothing;
    retired = ("keep" => "keep_model",),
    paths = [r"local_optimization\.model_update"])

MODEL_UPDATES["keep_model"]        # KeepModel
MODEL_UPDATES["keep"]              # ArgumentError: tag "keep" was renamed "keep_model"
tag_of(MODEL_UPDATES, KeepModel)   # "keep_model" (the writer side)
```

An unknown tag is an `ArgumentError` that lists the known ones. `paths` are
the TOML key paths (dotted, an array element `[i]` counted from 1, matched
whole) whose string values are tags of the family. Construction does not check
the rule — a package's load never fails on a spelling; `tag_rule_violations`
does.

**The dependency allow-list.** `configs/stack_dependencies.toml` in
VisualGeometryToolkitCore has one row per project: the stack packages its
Project.toml may name in `[deps]`, `[weakdeps]`, `[extras]` or the test target.
A *stack package* is any key of that table, and a project missing from it is an
error. `stack_dependency_violations(project_toml_path, name)` checks one
project. The table lists the published packages and paper projects only; a
project kept elsewhere supplies its own row through the `extra` keyword (a TOML
file in the same format, or a dictionary), and the quality gate's
`stack_extra`. An extra row adds a project and never replaces a published one.

**Public names only, across a package boundary.** A name imported from, or
accessed qualified in, another package is public there (ExplicitImports'
`check_all_explicit_imports_are_public` and
`check_all_qualified_accesses_are_public`), and no name reached from another
stack package begins with `_` (`private_name_violations`, a source scan).

**Extension points are declared.** A package adds methods to another stack
package's function only when the owner lists it in its `CONTRACT_GENERICS`
(VisualGeometryToolkitCore's: the block-coordinate-descent hooks, the chart-LM row
protocol, the RANSAC/LO interface, the minimal-solver backends, the TOML config
hooks, the accessors other packages implement for their own types, the
parameter protocol, the camera maps `project` and `backproject` and their
`convert_eltype`, the
`Manifold` protocol and the plotting stubs). Any stack package may declare its own;
`contract_generic_violations(M)` lists the methods that land elsewhere.
Functions owned by Base, Core, the standard library or a package outside the
stack are exempt.

**Agent instructions carry nothing private.** CLAUDE.md, AGENTS.md and the
tracked `.claude/` files ship with every package. `agent_doc_violations(root)`
checks them against `AGENT_DOC_FORBIDDEN` (absolute home-directory paths,
private keys and access tokens); every stack package's suite runs it with
`extra = get(ENV, "AGENT_DOC_PATTERNS", nothing)`, a file of further patterns
that release tooling supplies.

**One copy of the conventions.** This text lives once, in VisualGeometryToolkitCore's
`conventions/` folder (`VisualGeometryToolkitCore.conventions_dir()`). A package's
CLAUDE.md points at the folder and carries no copy of it; each package's
agent-docs test checks that its CLAUDE.md names the folder and holds none of
the shared sections.

| Gate keyword | Checks | Function |
|---|---|---|
| `tag_rules` | every `TagRegistry` constant of the package obeys the tag rule | `tag_rule_violations` |
| `tag_tables` | no `Dict{String, <:Type}` table or `"tag" => SomeType` pair outside a `TagRegistry(...)` call in `src/` (a heuristic scan; comments, docstrings and Base/Core converter types such as `"cutoff" => Float32` are skipped) | `tag_table_violations` |
| `config_tags` | every string at a registry's `paths` in the TOML files under `config_dirs` (default: the package's `configs/`) is a current tag | `config_tag_violations` |
| `stack_dependencies` | the Project.toml against the package's row (`stack_name`) | `stack_dependency_violations` |
| `public_names` | the two ExplicitImports public checks and the private-name scan | `private_name_violations` |
| `contract_generics` | methods on other stack packages' functions land on their contracts | `contract_generic_violations` |

Each gate takes a mode: `:enforce` (`@test` that it finds nothing), `:report`
(each finding logged with `@warn` and the gate recorded as broken; a gate that
finds nothing is an ordinary test) or `:off`. Every gate defaults to
`:report` until the stack conforms; a package turns one to `:enforce` by
supplying the keyword.
