# GOLDEN-SHARED FIXTURE: ci/golden's fixtures include this file, and the recorder runs it against the pre-migration
# checkout (3d8a06da), in that checkout's environment. So: data constructors only, written against the API both
# sides have (the golden side includes it by its path in this tree, not through `pkgdir`, which would resolve to the
# recording checkout); it includes no other shared file (its includer does). Any change must leave every golden
# `#fixture` record unchanged (re-record twice, the records identical and their diff empty).
# Small geometry constructors the stack's suites share (VisualGeometryToolkitCore's, VisualGeometryToolkitEstimation's and
# VisualGeometryToolkitIO's tests, and the golden records of ci/golden/fixtures/est_other_fits.jl). Include-once safe
# (functions only).
using StaticArrays: SVector, SMatrix
using Rotations: RotMatrix
using VisualGeometryToolkitCore: SimilarityMap

# column i of a 3-row matrix as an SVector
column3(A, i) = SVector(A[1, i], A[2, i], A[3, i])
# the 3×3 similarity [s 0 tx; 0 s ty; 0 0 1] (column-major storage)
similarity_matrix(s, tx, ty) = SMatrix{3, 3, Float64, 9}(s, 0.0, 0.0, 0.0, s, 0.0, tx, ty, 1.0)
# a planar target's placement: the design chart's origin at `o` (plane z = 0), scaled by `side`, unrotated
planar_placement(o, side) = SimilarityMap(side, one(RotMatrix{3, Float64}), SVector(o[1], o[2], 0.0))
