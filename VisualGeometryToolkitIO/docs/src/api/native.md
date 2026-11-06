# Native Kinds

`load`, `save` and `validate` for the `vgio/<kind>` files, the decimal-exact
length pair `read_length` and `write_length`, the quantity JSON format
(`read_quantity`, `write_quantity`, `json_value`, `from_json`), the serialization
style every embedded VGC value is written and read on (`VgioStyle`), the frame grammar,
content identities, and the building blocks other packages reuse for kinds of
their own.

```@meta
CurrentModule = VisualGeometryToolkitIO
```

```@autodocs
Modules = [VisualGeometryToolkitIO]
Pages = [
    "VisualGeometryToolkitIO.jl", "lengths.jl", "core_json.jl", "json.jl", "style.jl", "validate.jl", "frame_names.jl",
    "transform_schema.jl", "camera_schema.jl", "generic_camera_schema.jl",
    "rig_schema.jl", "captures_schema.jl", "capture_settings_schema.jl", "radiometry_schema.jl", "canonical.jl", "jsonl.jl",
]
```
