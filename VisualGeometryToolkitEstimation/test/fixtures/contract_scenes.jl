# GOLDEN-SHARED FIXTURE: ci/golden's fixtures include this file, and the recorder runs it against the pre-migration
# checkout (3d8a06da), in that checkout's environment. So: data constructors only, written against the API both
# sides have (the golden side includes it by its path in this tree, not through `pkgdir`, which would resolve to the
# recording checkout); it includes no other shared file (its includer does). Any change must leave every golden
# `#fixture` record unchanged (re-record twice, the records identical and their diff empty).
# The robust-estimation contract's scenes (fit_contract_tests.jl's FitContractScenes, and the golden records of
# ci/golden/fixtures/est_other_fits.jl), each drawn from the caller's RNG: a linear system, rotated directions, a
# rotation-averaging graph, a rigid point set and a robot-world hand-eye marker set. Data constructors only — no
# estimator is called; include-once safe (functions only).
using LinearAlgebra: normalize
using StaticArrays: SVector, SMatrix
using Rotations: RotMatrix, RotXYZ
using VisualGeometryToolkitCore: Correspondences, EuclideanMap, exp_so3, rotation, translation

function contract_linear(rng)
    A = randn(rng, 60, 3); x0 = randn(rng, 3)
    return A, A * x0 .+ 1.0e-3 .* randn(rng, 60)
end
function contract_rotation(rng)
    R = RotMatrix(RotXYZ(0.2, -0.4, 0.9))
    as = [normalize(randn(rng, SVector{3, Float64})) for _ in 1:40]
    return Correspondences(as, [R * a for a in as])
end
function contract_averaging()
    R = Dict(k => RotMatrix(RotXYZ(0.3k, -0.2k, 0.1k^2)) for k in 1:6)
    return Dict((i, j) => SMatrix{3, 3, Float64, 9}(R[j]' * R[i]) for i in 1:6 for j in (i + 1):6)
end
# the correspondences, the rotation and the translation
function contract_points(rng; n = 60)
    R = RotMatrix(RotXYZ(0.1, 0.5, -0.3)); t = SVector(0.4, -1.0, 2.0)
    as = [SVector{3, Float64}(4 .* randn(rng, 3)) for _ in 1:n]
    return Correspondences(as, [R * a + t for a in as]), R, t
end
# robot-world hand-eye: N frames × M markers, a little pose noise (`noise` rad / length)
function contract_handeye(rng; N = 12, M = 2, noise = 1.0e-4)
    rot(s) = SMatrix{3, 3, Float64, 9}(RotMatrix(RotXYZ((s .* randn(rng, 3))...)))
    E(R, t) = EuclideanMap(R, SVector{3, Float64}(t))
    Y = E(rot(1.0), 0.5 .* randn(rng, 3)); X = [E(rot(0.5), 0.3 .* randn(rng, 3)) for _ in 1:M]
    A = [E(rot(1.0), randn(rng, 3)) for _ in 1:N]
    return Dict(m => [let B = inv(Y) ∘ A[i] ∘ X[m]
                          A[i] => E(SMatrix{3, 3, Float64, 9}(exp_so3(noise .* randn(rng, SVector{3, Float64}))) * rotation(B),
                                    translation(B) + noise .* randn(rng, SVector{3, Float64}))
                      end for i in 1:N] for m in 1:M)
end
