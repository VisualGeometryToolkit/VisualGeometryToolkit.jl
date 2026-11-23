# GOLDEN-SHARED FIXTURE: ci/golden's fixtures include this file, and the recorder runs it against the pre-migration
# checkout (3d8a06da), in that checkout's environment. So: data constructors only, written against the API both
# sides have (the golden side includes it by its path in this tree, not through `pkgdir`, which would resolve to the
# recording checkout); it includes no other shared file (its includer does). Any change must leave every golden
# `#fixture` record unchanged (re-record twice, the records identical and their diff empty).
# The relative-pose bundle's scenes (relpose_bundle_tests.jl, and the golden records of
# ci/golden/fixtures/est_epipolar.jl): two calibrated cameras and their two-view scene, a seed, and the A/B harness
# scene. Data constructors only; include-once safe (functions, and constants of immutable values).
using LinearAlgebra: normalize, cross
using StaticArrays: SVector, SMatrix
using StableRNGs: StableRNG
using Rotations: RotMatrix, RotXYZ, AngleAxis
using VisualGeometryToolkitCore: CameraModel, calibration_matrix, Point3, Vec3
import VisualGeometryToolkitEstimation as VgtkEstimation

# Two calibrated cameras (normalized units) and their K matrices.
const CAM1 = CameraModel(; focal = (1.1, 1.05), pp = (0.5, 0.5))
const CAM2 = CameraModel(; focal = (1.2, 1.15), pp = (0.48, 0.52))
const KC1 = SMatrix{3, 3, Float64, 9}(calibration_matrix(CAM1))
const KC2 = SMatrix{3, 3, Float64, 9}(calibration_matrix(CAM2))

# Synthetic two-view scene, projected through the two cameras.
function relpose_scene(rng, N; σ = 0.0)
    Rgt = RotMatrix(RotXYZ(0.1, -0.15, 0.08)); tgt = normalize(SVector(0.7, 0.2, 0.6))
    u1, u2 = VgtkEstimation.synthetic_scenes().EpipolarScenes.stereo_pixel_pairs(
        rng, CAM1, CAM2, Rgt, tgt; n = N, σ₁ = σ, σ₂ = σ,
        point = rng -> Point3(2randn(rng), 2randn(rng), 5 + 2rand(rng))
    )
    return Rgt, tgt, u1, u2
end

# The seed a refinement starts from: the truth's rotation composed with
# RotXYZ(δ...), its translation direction moved by s·N(0, I₃).
relpose_seed(rng, Rgt, tgt; δ, s) = (
    SMatrix{3, 3, Float64, 9}(Rgt * RotXYZ(δ...)),
    Vec3{Float64}(normalize(tgt + s * SVector(randn(rng), randn(rng), randn(rng)))),
)

# The A/B harness's scene (2026-10-27, nlls-schur): pixel cameras at ~800 px focal,
# σ = 0.5 px, N points in a 8 × 6 × (6…12) box, the seed 1° (rotation) and 2°
# (translation direction) off; the seed structure is the bundle's own ray
# triangulation.
const KP1 = SMatrix{3, 3, Float64, 9}(800.0, 0, 0, 0, 800.0, 0, 640, 360, 1)
const KP2 = SMatrix{3, 3, Float64, 9}(820.0, 0, 0, 0, 815.0, 0, 630, 370, 1)
function harness_scene(N; seed = 1, σ = 0.5)
    rng = StableRNG(seed)
    Rgt = SMatrix{3, 3, Float64, 9}(RotMatrix(AngleAxis(deg2rad(8.0), normalize(SVector(0.1, 1.0, 0.05))...)))
    tgt = normalize(SVector(-1.0, 0.05, 0.1))
    u1 = SVector{2, Float64}[]; u2 = SVector{2, Float64}[]
    while length(u1) < N
        X = SVector(8 * (rand(rng) - 0.5), 6 * (rand(rng) - 0.5), 6 + 6 * rand(rng))
        Y = Rgt * X + tgt
        Y[3] <= 0.5 && continue
        w1 = KP1 * X; w2 = KP2 * Y
        p1 = SVector(w1[1], w1[2]) / w1[3]; p2 = SVector(w2[1], w2[2]) / w2[3]
        (0 <= p1[1] <= 1280 && 0 <= p1[2] <= 720 && 0 <= p2[1] <= 1280 && 0 <= p2[2] <= 720) || continue
        push!(u1, p1 + σ * SVector(randn(rng), randn(rng)))
        push!(u2, p2 + σ * SVector(randn(rng), randn(rng)))
    end
    a = normalize(SVector(randn(rng), randn(rng), randn(rng)))
    R0 = SMatrix{3, 3, Float64, 9}(RotMatrix(AngleAxis(deg2rad(1.0), a...))) * Rgt
    p = normalize(cross(tgt, SVector(randn(rng), randn(rng), randn(rng))))
    t0 = normalize(SVector{3, Float64}(RotMatrix(AngleAxis(deg2rad(2.0), p...)) * tgt))
    return R0, t0, u1, u2
end
