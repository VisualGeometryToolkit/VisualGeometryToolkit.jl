#!/usr/bin/env julia
# Migrate a file into the native kinds — a dry run unless --apply:
#
#   a third-party OpenCV or MATLAB calibration export (a camera store and the
#   image-named poses),
#   a file of an earlier schema version (every vgio kind's earlier versions a
#   published file holds, and any schema whose owner added a migration), a
#   tree of them, or an observation file re-pointed at its re-identified rigs.
#
#     julia --project=<path to VisualGeometryToolkitIO> bin/vgio_migrate.jl opencv --help
#     julia --project=<path to VisualGeometryToolkitIO> bin/vgio_migrate.jl matlab --help
#     julia --project=<path to VisualGeometryToolkitIO> bin/vgio_migrate.jl target|rig|file|tree|repoint --help
#
# The logic lives in VisualGeometryToolkitIO (`convert_opencv_calibration`, `convert_matlab_calibration`,
# `vgio_migrate`, `vgio_migrate_tree` and the command behind
# this script), so it is tested with the package.
import VisualGeometryToolkitIO as VgtkIO
using VisualGeometryToolkitIO
exit(VgtkIO._migrate_cli(ARGS))
