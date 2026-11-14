# JSON Schemas

Every `vgio/<kind>` file has a published JSON Schema (draft 2020-12), so a
tool outside Julia — Python's `jsonschema`, an editor's completion, a CI check
— validates a file against the shape the Julia reader checks. The schemas are
generated from the validator's own tables by [`json_schema`](@ref VisualGeometryToolkitIO.json_schema): there is
no second, hand-kept description of a file.

The schemas of the versions the writers emit — each kind's current version,
and an earlier one a writer still writes for a record that version can hold
(`vgio/sensor_characterisation/v1.0`, for a characterisation without a
`Darks`); the table is generated from the package, as the files are:

```@eval
import VisualGeometryToolkitIO as VgtkIO
using Markdown: Markdown
rows = ["| [`$(VgtkIO._json_schema_path(k, v))`](../schemas/$(VgtkIO._json_schema_path(k, v))) |" for k in sort!(collect(keys(VgtkIO._SCHEMA_VERSION))) for v in VgtkIO._written_versions(k)]
Markdown.parse(join(["| Schema (its `\$id` is this path below `https://visualgeometrytoolkit.github.io/schemas/`) |", "|---|", rows...], "\n"))
```

`vgio/units` is a COLMAP model directory's `vgio_units.json`.

The files are published with this manual, under `schemas/`; the `$id` URLs
name where they are to be served from, which is not yet set up, so a tool
resolves a schema from this manual's copy (or `json_schema`) until it is.

```julia
import VisualGeometryToolkitIO as VgtkIO
schema = VgtkIO.json_schema("rig")          # a JSON.Object; JSON.json(schema; pretty = 4) is the published file
```

```python
import json, jsonschema
schema = json.load(open("rig/v10.0.json"))
jsonschema.Draft202012Validator(schema).validate(json.load(open("my_rig.json")))
```

## What a schema says, and what only the reader checks

Each table combinator becomes its JSON Schema form: an object with exactly its
keys (`additionalProperties: false`), a declaration token as a `const`, a
registered name as the `enum` VisualGeometryToolkitCore holds (the length and
density units, the pixel frames) or the `pattern` the validator matches (frame
names, digests, time origins), a shape told apart by a key as `oneOf` plus
`if`/`then`. The positional primitives a file stores compactly are described
field by field (`prefixItems` with each field's name): a rectangle
`[x, y, width, height]`, a circle `[cx, cy, r]`, an ellipse
`[cx, cy, a, b, theta]` and a rotation `[w, x, y, z]`.

A cross-field check (a lattice's node count against its `present` runs, a
covariance's symmetry, a calibration's parameter count) is stated in the
`description` of the object it constrains, and as `if`/`then` or a bound in an
`allOf` member where JSON Schema can say it. The Julia validator stays the
authority: it may refuse a file the schema accepts for such a check — never
the reverse. The test suite holds both directions: every fixture validates
against its kind's schema in Python, and every document the validator's
deletion tests refuse (each required key deleted in turn, an unknown key) is
refused by the schema too. One looseness is JSON Schema's own: it counts
`1.0` as an `integer`, where the reader asks for an integer literal.

## Versions

A schema is the schema of one version, `v<major>.<minor>`. A reader reads its
major version at any minor and tolerates the keys a higher minor added (they
are optional by the version rule), so a file of a later minor than a schema's
may be refused by that schema: validate a file against the schema of the
version its `schema` string names. Each version a writer emits has a schema
(`_written_versions`: the current one, and an earlier one still written for a
record it can hold); any other earlier version is read through its version
steps, and its current record is what validates.
