# GOLDEN-SHARED FIXTURE: ci/golden's fixtures include this file, and the recorder runs it against the pre-migration
# checkout (3d8a06da), in that checkout's environment. So: data constructors only, written against the API both
# sides have (the golden side includes it by its path in this tree, not through `pkgdir`, which would resolve to the
# recording checkout); it includes no other shared file (its includer does). Any change must leave every golden
# `#fixture` record unchanged (re-record twice, the records identical and their diff empty).
# The thin homography incumbent (ransac_homography_tests.jl, termination_record_tests.jl, and the golden records of
# ci/golden/fixtures/est_conics_ransac.jl): five exact correspondences of a homography among twelve scattered ones, so
# the incumbent's consensus is m + 1 = 5 points with no residual dof. Data constructor only; include-once safe.
using StaticArrays: SVector, @SMatrix
using StableRNGs: StableRNG
using VisualGeometryToolkitCore: correspondences

function thin_homography_correspondences()
    rng = StableRNG(3)
    H = @SMatrix [1.1 0.05 3.0; -0.02 0.95 -2.0; 1.0e-4 2.0e-4 1.0]
    ap(x) = (y = H * SVector(x[1], x[2], 1.0); SVector(y[1] / y[3], y[2] / y[3]))
    src = [SVector(100rand(rng), 100rand(rng)) for _ in 1:5]; dst = ap.(src)
    for _ in 1:12
        push!(src, SVector(100rand(rng), 100rand(rng))); push!(dst, SVector(100rand(rng), 100rand(rng)))
    end
    return correspondences(src, dst)
end
