#!/usr/bin/env julia
# Convert a third-party OpenCV or MATLAB calibration export into the native
# kinds (a vgio/cameras/v3 store and the image-named poses) — a dry run unless
# --apply.
#
#     julia --project=<path to VisualGeometryToolkitIO> bin/vgio_migrate.jl opencv --help
#     julia --project=<path to VisualGeometryToolkitIO> bin/vgio_migrate.jl matlab --help
#
# The logic lives in VisualGeometryToolkitIO (`migrate_opencv`, `migrate_matlab` and
# the command behind this script), so it is tested with the package.
import VisualGeometryToolkitIO as VgtkIO
using VisualGeometryToolkitIO
exit(VgtkIO._migrate_cli(ARGS))
