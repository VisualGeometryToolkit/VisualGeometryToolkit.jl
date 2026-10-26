#!/usr/bin/env julia
# Migrate a file into the native kinds — a dry run unless --apply:
#
#   a third-party OpenCV or MATLAB calibration export (a vgio/cameras/v3 store
#   and the image-named poses),
#   a file of an earlier schema version (vgio/target/v1, vgio/rig/v4, and any
#   schema whose owner added a migration), a tree of them, or a measurement
#   file re-pointed at its migrated rigs.
#
#     julia --project=<path to VisualGeometryToolkitIO> bin/vgio_migrate.jl opencv --help
#     julia --project=<path to VisualGeometryToolkitIO> bin/vgio_migrate.jl matlab --help
#     julia --project=<path to VisualGeometryToolkitIO> bin/vgio_migrate.jl target|rig|file|tree|repoint --help
#
# The logic lives in VisualGeometryToolkitIO (`migrate_opencv`, `migrate_matlab`,
# `vgio_migrate`, `vgio_migrate_tree`, `repoint_targets` and the command behind
# this script), so it is tested with the package.
import VisualGeometryToolkitIO as VgtkIO
using VisualGeometryToolkitIO
exit(VgtkIO._migrate_cli(ARGS))
