# Third-Party Calibrations

The OpenCV and Kalibr lens conventions, the OpenCV and MATLAB calibration
readers, the Schöps camera_calibration readers, the OpenCV-shaped writer, the migrations to native files, and the
migration of an earlier schema version: the version steps every reader runs
(`migrate_schema`, `migrate_record`), the rewrite of a file
(`vgio_migrate`, `vgio_migrate_tree`) and the comparison that verifies one
(`same_value`).

```@meta
CurrentModule = VisualGeometryToolkitIO
```

```@autodocs
Modules = [VisualGeometryToolkitIO]
Pages = ["conventions.jl", "external_calibration.jl", "schoeps.jl", "migrate.jl"]
```
