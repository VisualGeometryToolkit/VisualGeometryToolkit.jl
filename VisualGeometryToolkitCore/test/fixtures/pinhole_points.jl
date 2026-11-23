# GOLDEN-SHARED FIXTURE: ci/golden's fixtures include this file, and the recorder runs it against the pre-migration
# checkout (3d8a06da), in that checkout's environment. So: data constructors only, written against the API both
# sides have (the golden side includes it by its path in this tree, not through `pkgdir`, which would resolve to the
# recording checkout); it includes no other shared file (its includer does). Any change must leave every golden
# `#fixture` record unchanged (re-record twice, the records identical and their diff empty).
# A 3-vector's pinhole point (X₁/X₃, X₂/X₃) and its pixel through a 3×3 K (spline_lens_tests.jl, PoseLib's
# test_refine.jl, and the golden records of ci/golden/fixtures/poselib.jl). Include-once safe (functions only).
using StaticArrays: SVector

pinhole_point(X) = SVector(X[1] / X[3], X[2] / X[3])
pinhole_pixel(K, X) = pinhole_point(K * X)
