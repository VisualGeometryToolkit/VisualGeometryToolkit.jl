# GOLDEN-SHARED FIXTURE: ci/golden's fixtures include this file, and the recorder runs it against the pre-migration
# checkout (3d8a06da), in that checkout's environment. So: data constructors only, written against the API both
# sides have (the golden side includes it by its path in this tree, not through `pkgdir`, which would resolve to the
# recording checkout); it includes no other shared file (its includer does). Any change must leave every golden
# `#fixture` record unchanged (re-record twice, the records identical and their diff empty).
# The division model's forward distortion of an undistorted point, and a homography's point transfer
# (homography_5pt_radial_tests.jl, and the golden records of ci/golden/fixtures/core_minimal.jl). Data constructors only;
# include-once safe.
using LinearAlgebra: norm
using StaticArrays: SVector, SMatrix
using VisualGeometryToolkitCore: PerspectiveMap, HomographyMap

# the d with d / (1 + k‖d‖²) = u, or `nothing` past the model's fold
function distort_division(u, k)
    r = norm(u)
    (abs(k) < 1.0e-14 || r < 1.0e-14) && return SVector{2, Float64}(u[1], u[2])
    disc = 1 - 4k * r^2
    disc <= 0 && return nothing
    t = (1 - sqrt(disc)) / (2k * r)
    return SVector{2, Float64}(u[1] * t / r, u[2] * t / r)
end

# the predicted (undistorted) target of a source point s through the homography H
homography_transfer(H, s) = (PerspectiveMap() ∘ HomographyMap(SMatrix{3, 3, Float64}(H)))(s)
