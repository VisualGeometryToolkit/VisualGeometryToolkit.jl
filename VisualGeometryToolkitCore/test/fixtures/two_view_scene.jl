# GOLDEN-SHARED FIXTURE: ci/golden's fixtures include this file, and the recorder runs it against the pre-migration
# checkout (3d8a06da), in that checkout's environment. So: data constructors only, written against the API both
# sides have (the golden side includes it by its path in this tree, not through `pkgdir`, which would resolve to the
# recording checkout); it includes no other shared file (its includer does). Any change must leave every golden
# `#fixture` record unchanged (re-record twice, the records identical and their diff empty).
# The two-view epipolar scene of solver_layer_tests.jl (and of the golden records of ci/golden/fixtures/core_minimal.jl
# and core_closed_forms.jl): a Haar rotation R, t = (0.4, −0.2, 0.1), 20 points in front of both views; their bearings
# and their normalized pixels, drawn from `rng`; and a Haar-uniform rotation. Data constructors only; include-once safe (functions only).
using LinearAlgebra: normalize
using StaticArrays: SVector, SMatrix
using Rotations: RotMatrix
using VisualGeometryToolkitCore: Point2, PerspectiveMap

# a Haar-uniform rotation, as a plain 3×3 matrix (solver_layer_tests.jl, p3p_native_tests.jl)
haar_rotation(rng) = SMatrix{3, 3, Float64}(rand(rng, RotMatrix{3, Float64}))
function two_view_scene(rng)
    R = haar_rotation(rng)
    t = SVector(0.4, -0.2, 0.1)
    X3 = [SVector(2randn(rng), 2randn(rng), 6.0 + 2rand(rng)) for _ in 1:20]
    x1b = [normalize(Xi) for Xi in X3]                    # bearings, cam1 at origin
    x2b = [normalize(R * Xi + t) for Xi in X3]            # bearings, cam2
    # the pixels as solver_layer_tests drew them (and the golden records hold their bits): view 1's by
    # CoordinateTransformations' `PerspectiveMap` (the point times the reciprocal of its depth), view 2's by dividing
    # each coordinate of `R * Xi + t`, evaluated per coordinate; `pinhole_point` of either is not bit-identical
    u1 = [Point2{Float64}(PerspectiveMap()(Xi)) for Xi in X3]
    u2 = [Point2{Float64}((R * Xi + t)[1] / (R * Xi + t)[3], (R * Xi + t)[2] / (R * Xi + t)[3]) for Xi in X3]
    return (; R, t, x1b, x2b, u1, u2)
end
