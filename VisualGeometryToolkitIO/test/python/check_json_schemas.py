"""Validate vgio documents against the published JSON Schemas, with Python's `jsonschema`.

    python check_json_schemas.py <schemas_dir> <cases.json>

`schemas_dir` holds the generated schemas (docs/src/schemas: vgio/<kind>/v<M>.<m>.json),
each checked first as a draft 2020-12 schema. `cases.json` is a list of
`{"name", "schema_id", "doc"}`: each document is validated against the schema
of that `$id`. Prints a JSON list of `{"name", "valid", "error"}` (the first
error's message, or null) on stdout, in the cases' order; the Julia test
(test/json_schema_tests.jl) compares it with the validator's verdicts.
"""

import json
import pathlib
import sys

from jsonschema import Draft202012Validator


def load_schemas(root):
    schemas = {}
    for path in sorted(pathlib.Path(root).rglob("*.json")):
        schema = json.loads(path.read_text(encoding="utf-8"))
        Draft202012Validator.check_schema(schema)
        if schema.get("$schema") != "https://json-schema.org/draft/2020-12/schema":
            raise ValueError(f"{path}: not a draft 2020-12 schema")
        rel = path.relative_to(root).as_posix()
        if not schema["$id"].endswith("/schemas/" + rel):
            raise ValueError(f"{path}: its $id {schema['$id']} is not its path")
        schemas[schema["$id"]] = schema
    return schemas


def main(root, cases_path):
    schemas = load_schemas(root)
    validators = {i: Draft202012Validator(s) for i, s in schemas.items()}
    with open(cases_path, encoding="utf-8") as f:
        cases = json.load(f)
    out = []
    for case in cases:
        errors = sorted(validators[case["schema_id"]].iter_errors(case["doc"]), key=lambda e: list(e.absolute_path))
        out.append({"name": case["name"], "valid": not errors, "error": errors[0].message if errors else None})
    json.dump(out, sys.stdout)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
