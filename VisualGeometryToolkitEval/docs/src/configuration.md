# Defaults and file formats

## Defaults of record

Every default of the package lives in one TOML file under `configs/` and is
read through one function. An unknown key or tag is refused.

| File | Holds | Read by |
|---|---|---|
| `configs/default.toml` | every value that changes a reported statistic, a rendered panel or a raw dataset read: the `[metrics]`, `[render]`, `[pool.<dataset>]`, `[gt.<dataset>.<matcher>]`, `[datasets.<dataset>]`, `[local_optimization]`, `[refit]` and `[selection]` tables | the package's internal default lookup, which throws on a missing table or key |
| `configs/tuning.toml` | every cross-validation parameter: grid spacings, window rules, extension rules, the objective, the cross-validation and the budget draws | [`tuning_spec`](@ref) (see [The `[tuning]` schema](@ref)) |
| `configs/splits.toml` | every scene list of the stack, named by a spec's `cross_validation.scene_split` and never restated | [`registered_split`](@ref) |

[`DEFAULTS_PATH`](@ref VisualGeometryToolkitEval.DEFAULTS_PATH) is the path of
`configs/default.toml`. The file is parsed when the package is precompiled, so
an edit takes effect after Julia restarts.

Every random sample is seeded by [`stable_seed`](@ref) from a spec's seed and
a name, such as a scene name. See [The experimental protocol](@ref).

## Artifact formats

Every persisted artifact carries a format stamp, the key `"<kind>_format"` in
its JLD2 file. One writer stamps each kind and one reader refuses any other
format. An older file is converted or regenerated; it is never read through a
compatibility shim.

[`ARTIFACT_FORMATS`](@ref VisualGeometryToolkitEval.ARTIFACT_FORMATS) lists the
current version of each kind (`:pool`, `:pool_residuals`, `:pool_samples`,
`:pnp_pool`, `:tuning`), the change that the version introduced and the remedy
for an older file.

## The driver command line

A driver reads its options as `--key=value` with
[`cli_option`](@ref VisualGeometryToolkitEval.cli_option), and calls
[`start_output_flush!`](@ref VisualGeometryToolkitEval.start_output_flush!) so
that buffered output survives a batch job that is killed.
