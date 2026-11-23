# GOLDEN-SHARED FIXTURE: ci/golden's fixtures include this file, and the recorder runs it against the pre-migration
# checkout (3d8a06da), in that checkout's environment. So: data constructors only, written against the API both
# sides have (the golden side includes it by its path in this tree, not through `pkgdir`, which would resolve to the
# recording checkout); it includes no other shared file (its includer does). Any change must leave every golden
# `#fixture` record unchanged (re-record twice, the records identical and their diff empty).
# The absolute pose's pixel scenes of refine_absolute_pose_tests.jl's PixelPoseScenes (and of the golden records of
# ci/golden/fixtures/est_pose.jl): a pose, its perturbation, world points spiralling off the optical axis and their
# exact pixels. Data constructors only — no estimator is called; include-once safe (functions only).
using StaticArrays: SVector
using Rotations: RotMatrix, RotXYZ
using VisualGeometryToolkitCore: EuclideanMap, exp_so3, project

true_pose() = EuclideanMap(RotMatrix(RotXYZ(0.3, -0.2, 0.1)), SVector(0.1, -0.05, 2.0))
# the truth moved by 0.08 rad about a skew axis and 0.15 in translation
function perturbed(E)
    ΔR = exp_so3(SVector(0.05, -0.04, 0.05))
    return EuclideanMap(RotMatrix{3, Float64}(ΔR * E.R), E.t + SVector(0.1, -0.08, 0.07))
end
# world points whose camera-frame directions span ±`θmax` off the optical axis
function scene_points(E, θmax; n = 40, depth = 3.0)
    pts = SVector{3, Float64}[]
    for k in 1:n
        θ = θmax * (k / n); φ = 2.399963 * k                     # golden-angle spiral
        d = SVector(sin(θ) * cos(φ), sin(θ) * sin(φ), cos(θ))
        Xc = (depth * (0.7 + 0.6 * ((7k) % 10) / 10)) * d
        push!(pts, E.R' * (Xc - E.t))                            # back into the world
    end
    return pts
end
# `X => u` through the camera, the points it can image
function pixel_pairs(cam, E, Xs)
    keep = [X for X in Xs if all(isfinite, project(cam, E.R * X + E.t))]
    return [X => SVector{2, Float64}(project(cam, E.R * X + E.t)) for X in keep]
end
