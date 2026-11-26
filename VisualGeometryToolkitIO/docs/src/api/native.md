# Native Kinds

`load`, `save` and `validate` for the `vgio/<kind>` files, the decimal-exact
length pair `read_length` and `write_length`, the serialization style every
embedded VGC value — a quantity, a `Size2`, a feature frame — is written and
read on (`VgioStyle`, `JSON.lower`/`JSON.lift`), the frame grammar,
content identities, and the building blocks other packages reuse for kinds of
their own.

```@meta
CurrentModule = VisualGeometryToolkitIO
```

```@autodocs
Modules = [VisualGeometryToolkitIO]
Pages = [
    "VisualGeometryToolkitIO.jl", "lengths.jl", "core_json.jl", "json.jl", "style.jl", "validate.jl", "frame_names.jl",
    "units_sidecar.jl", "transform_schema.jl", "camera_schema.jl",
    "rig_schema.jl", "captures_schema.jl", "capture_settings_schema.jl", "blur_schema.jl", "json_schema.jl", "jsonl.jl", "calibration_manifest.jl",
    "detection_schema.jl", "sensor_characterization_schema.jl", "tiff.jl", "raster_schema.jl", "kind_spec.jl", "kind_test_helpers.jl", "retired.jl",
]
```
