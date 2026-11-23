# GOLDEN-SHARED FIXTURE: ci/golden's fixtures include this file, and the recorder runs it against the pre-migration
# checkout (3d8a06da), in that checkout's environment. So: data constructors only, written against the API both
# sides have (the golden side includes it by its path in this tree, not through `pkgdir`, which would resolve to the
# recording checkout); it includes no other shared file (its includer does). Any change must leave every golden
# `#fixture` record unchanged (re-record twice, the records identical and their diff empty).
# The pure rotation's two views of a calibrated fisheye (ransac_rotation_tests.jl's RotationScene, and the golden
# records of ci/golden/fixtures/{core_minimal,est_conics_ransac,est_epipolar}.jl). Data constructors only;
# include-once safe (functions, and a constant of an immutable value).
using Random: randn, rand, randperm
using LinearAlgebra: normalize
using StaticArrays: SA, SVector
using VisualGeometryToolkitCore: CameraModel, KannalaBrandt, px, project, backproject_with_jacobian, Correspondences

# A fisheye whose field reaches 100° off axis (an equidistant-based KB lens, f = 300 px).
const FISHEYE = CameraModel(300.0px, SA[600.0px, 600.0px]; lens = KannalaBrandt(-0.02))

# n correspondences of a pure rotation R between two views of FISHEYE, each ray within
# `θmax` of both optical axes; pixel noise σ (px), a fraction `outl` replaced by random pixels.
# Returns the correspondences, each ray's angle off the first axis, and the replaced indices.
function rotation_scene(rng, R, n; σ = 0.5, outl = 0.0, θmax = deg2rad(100))
    u₁ = SVector{2, Float64}[]; u₂ = SVector{2, Float64}[]; θs = Float64[]
    while length(u₁) < n
        d = normalize(randn(rng, SVector{3, Float64}))
        (acos(clamp(d[3], -1, 1)) < θmax && acos(clamp((R * d)[3], -1, 1)) < θmax) || continue
        a = project(FISHEYE, d); b = project(FISHEYE, R * d)
        (all(isfinite, a) && all(isfinite, b)) || continue
        push!(u₁, a + σ * randn(rng, SVector{2, Float64})); push!(u₂, b + σ * randn(rng, SVector{2, Float64}))
        push!(θs, acos(clamp(d[3], -1, 1)))
    end
    bad = randperm(rng, n)[1:round(Int, outl * n)]
    for i in bad
        while true
            c = SVector(600 + 500 * (2rand(rng) - 1), 600 + 500 * (2rand(rng) - 1))
            all(isfinite, backproject_with_jacobian(FISHEYE, c)[1]) && (u₂[i] = c; break)
        end
    end
    return Correspondences(u₁, u₂), θs, bad
end
